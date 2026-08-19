"""The agentic rollout: one call == one episode.

    boot a sandbox -> tunnel the proxy into it -> let the CLI work
        -> ask the task for a reward -> hand the trajectory to miles

Nothing here knows the task; swapping the import below trains something else.
Wired in with ``--custom-generate-function-path generate.generate``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import shlex

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.sglang_rollout import get_model_url

if __package__:
    from . import sandbox
    from .proxy import MODEL_NAME, ModelProxy
    from .swe import SweTask
else:  # Loaded as generate.generate through the rollout PYTHONPATH.
    import sandbox
    from proxy import MODEL_NAME, ModelProxy
    from swe import SweTask

SANDBOX_MODEL_PORT = 30001  # where the tunnel surfaces the proxy inside the sandbox
WSTUNNEL_SERVER_PORT = 19090  # sandbox-side ingress port the host client dials
# Paths inside the prebuilt sandbox template; host-side wstunnel comes from PATH.
AGENT_BIN = "/__avaeval_agentic_protocol_v1__/frameworks/qwen_code/bin/qwen"
WSTUNNEL_BIN = "/__avaeval_agentic_protocol_v1__/linux/bin/wstunnel"
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "80"))
AGENT_TIMEOUT = int(os.environ.get("AGENT_TIMEOUT_SECONDS", str(90 * 60)))
MAX_TURNS_EXIT_MESSAGE = "Reached max session turns for this session."


def _is_controlled_agent_stop(exit_code: int, output: str) -> bool:
    """The CLI reports an intentional max-turn stop with a non-zero status."""
    return exit_code != 0 and MAX_TURNS_EXIT_MESSAGE in output


def agent_command(workdir: str, prompt: str) -> str:
    """Non-interactive qwen-code; swapping the agent CLI is this function alone."""
    base_url = f"http://127.0.0.1:{SANDBOX_MODEL_PORT}/v1"
    # qwen-code needs ~/.qwen/tmp/<sha256(cwd)>, so workdir must not be a symlink.
    project_hash = hashlib.sha256(workdir.encode("utf-8")).hexdigest()
    cli = [
        AGENT_BIN,
        "--approval-mode",
        "yolo",
        "--max-session-turns",
        str(MAX_TURNS),
        "--auth-type",
        "openai",
        "--openai-base-url",
        base_url,
        "--openai-api-key",
        "agentic-swe",
        "--model",
        MODEL_NAME,
    ]
    return "\n".join(
        [
            "set -euo pipefail",
            'mkdir -p "${HOME}/.qwen/tmp/' + project_hash + '"',
            f"{shlex.join(cli)} {shlex.quote(prompt)}",
        ]
    )


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args, sample, state = input.args, input.sample, input.state
    task = SweTask.from_sample(sample)
    prompt = sample.prompt[0]["content"]

    proxy = ModelProxy(
        tokenizer=state.tokenizer,
        loop=asyncio.get_running_loop(),
        model_url=get_model_url(args, MODEL_NAME),
        sampling_params=input.sampling_params,
        tool_parser=getattr(args, "sglang_tool_call_parser", None),
        reasoning_parser=getattr(args, "sglang_reasoning_parser", None),
    )
    async with contextlib.AsyncExitStack() as stack:
        proxy.start()
        stack.push_async_callback(asyncio.to_thread, proxy.close)
        runtime = await sandbox.create_sandbox(template=task.template, envs=task.env)
        try:
            await task.setup(runtime)
            async with sandbox.reverse_tunnel(
                runtime,
                proxy_port=proxy.port,
                sandbox_port=SANDBOX_MODEL_PORT,
                server_port=WSTUNNEL_SERVER_PORT,
                wstunnel_bin=WSTUNNEL_BIN,
                user=task.sandbox_user,
            ):
                agent_result = await sandbox.run(
                    runtime,
                    agent_command(task.workdir, prompt),
                    timeout=AGENT_TIMEOUT,
                    user=task.sandbox_user,
                    cwd=task.workdir,
                )
                if agent_result.exit_code != 0 and not _is_controlled_agent_stop(
                    agent_result.exit_code, agent_result.output
                ):
                    raise RuntimeError(
                        f"qwen-code failed (exit_code={agent_result.exit_code}): {agent_result.output[-4000:]}"
                    )
            patch = await task.candidate_patch(runtime)
        finally:
            # Release the agent before creating the clean grading sandbox. At
            # quota-sized concurrency, retaining it here deadlocks every task.
            await sandbox.delete_sandbox(runtime)
        reward = await task.reward_patch(patch)
    result = proxy.trajectory.to_sample(
        sample,
        reward=reward,
    )
    if result is None:
        raise RuntimeError("agent produced no trainable trajectory")
    return GenerateFnOutput(samples=result)
