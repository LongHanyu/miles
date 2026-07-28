"""Local OpenAI-compatible endpoint: what the agent CLI inside the sandbox talks to.

One proxy per rollout sample, so one proxy == one set of ledgers == one training
candidate. Each turn runs the same short pipeline::

    render prompt tokens -> route to a ledger -> SGLang /generate
        -> parse the completion into an assistant message -> record -> reply

There are no turn guards. The CLI's own ``--max-session-turns``, the agent
timeout and the sandbox TTL already bound a rollout, and a turn the proxy cut
short would only add tokens the model never chose.

The tool schemas the harness sends are forwarded to *both* the renderer and the
tool-call parser. That is not a detail: the parser types arguments from the
schema, and without it every argument degrades to a string, which the chat
template then renders differently from what the model emitted -- see
``trajectory.render_prompt_ids``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from miles.utils.http_utils import post

from trajectory import Trajectory, render_prompt_ids

MODEL_NAME = "default"


def _tool_call(name: str, arguments: Any, index: int) -> dict[str, Any]:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "type": "function",
        "id": f"call_{uuid.uuid4().hex}_{index}",
        "function": {"name": name, "arguments": arguments},
    }


def parse_completion(
    text: str,
    *,
    tools: list[dict[str, Any]],
    tool_parser: str | None,
    reasoning_parser: str | None,
) -> dict[str, Any]:
    """Raw model text -> OpenAI assistant message (thinking split out, tool calls typed).

    ``reasoning_content`` is returned as its own field because the CLI echoes it
    back on the next request and the chat template renders it again -- that
    round trip is what keeps the ledger prefix-clean for a thinking model.
    """
    reasoning, body = "", text
    if reasoning_parser:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        reasoning, body = ReasoningParser(model_type=reasoning_parser, stream_reasoning=False).parse_non_stream(text)
        reasoning, body = reasoning or "", body or ""

    tool_calls: list[dict[str, Any]] = []
    if tool_parser and tools:
        from sglang.srt.entrypoints.openai.protocol import Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        schemas = [Tool(**tool) for tool in tools if tool.get("type") == "function"]
        parser = FunctionCallParser(tools=schemas, tool_call_parser=tool_parser)
        if parser.has_tool_call(body):
            body, calls = parser.parse_non_stream(body)
            tool_calls = [_tool_call(call.name, call.parameters or "{}", i) for i, call in enumerate(calls)]

    message: dict[str, Any] = {"role": "assistant", "content": body}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


@dataclass
class ModelProxy:
    """Serves one sandbox's agent and records its main trajectory."""

    tokenizer: Any
    loop: asyncio.AbstractEventLoop
    model_url: str  # SGLang /generate endpoint, from miles' router table
    sampling_params: dict[str, Any]
    tool_parser: str | None = None
    reasoning_parser: str | None = None

    trajectory: Trajectory = field(default_factory=Trajectory, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _server: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> ModelProxy:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        server.proxy = self  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server, self._thread = server, thread
        return self

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._server = self._thread = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        """Complete one OpenAI request and record it when it extends the main trajectory."""
        messages, tools = body["messages"], body.get("tools") or []
        prompt_ids = render_prompt_ids(self.tokenizer, messages, tools)
        with self._lock:
            record = self.trajectory.extends(prompt_ids)
            completion = asyncio.run_coroutine_threadsafe(self._generate(prompt_ids), self.loop).result()
            message = parse_completion(
                completion["text"],
                tools=tools,
                tool_parser=self.tool_parser,
                reasoning_parser=self.reasoning_parser,
            )
            if record:
                self.trajectory.append_turn(prompt_ids, completion["token_ids"], completion["log_probs"])

        choice = {
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else completion["finish_reason"],
        }
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body["model"],
            "choices": [choice],
        }

    async def _generate(self, prompt_ids: list[int]) -> dict[str, Any]:
        """One SGLang generation, by token ids so the ledger owns the prompt."""
        sampling_params = {**self.sampling_params, "skip_special_tokens": True}
        payload = {"input_ids": prompt_ids, "sampling_params": sampling_params, "return_logprob": True}
        output = await post(self.model_url, payload)
        # Contract of SGLang /generate with return_logprob: output_token_logprobs
        # is always present, each entry (logprob, token_id, ...). A malformed
        # response must raise -- dropping one token would misalign the whole
        # training sequence against what the model actually emitted.
        meta = output["meta_info"]
        log_probs, token_ids = [], []
        for logprob, token_id, *_ in meta["output_token_logprobs"]:
            log_probs.append(float(logprob))
            token_ids.append(int(token_id))
        return {
            "text": str(output["text"]),
            "token_ids": token_ids,
            "log_probs": log_probs,
            "finish_reason": meta["finish_reason"]["type"],
        }


def _stream_frames(payload: dict[str, Any]) -> list[Any]:
    """Replay a finished completion as chunks; agent CLIs ask for streams by default."""
    choice = payload["choices"][0]

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        return {
            **payload,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    return [chunk(choice["message"]), chunk({}, choice["finish_reason"]), "[DONE]"]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return

    @property
    def _proxy(self) -> ModelProxy:
        return self.server.proxy  # type: ignore[attr-defined,return-value]

    def do_GET(self) -> None:
        if urlsplit(self.path).path.rstrip("/") == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
            return
        self._send_json(404, {"error": {"message": f"unknown path {self.path!r}"}})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length") or "0")
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if urlsplit(self.path).path.rstrip("/") != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": f"unknown path {self.path!r}"}})
                return
            payload = self._proxy.complete(body)
            if body.get("stream"):
                self._send_stream(payload)
            else:
                self._send_json(200, payload)
        except Exception as exc:  # noqa: BLE001 - the agent must see an error, not a hang
            traceback.print_exc()
            self._send_json(500, {"error": {"message": str(exc), "type": exc.__class__.__name__}})

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_stream(self, payload: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        for frame in _stream_frames(payload):
            data = frame if isinstance(frame, str) else json.dumps(frame, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()
        self.close_connection = True
