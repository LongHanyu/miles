"""The agentic rollout: one call == one episode.

    boot a sandbox -> tunnel the proxy into it -> let the CLI work
        -> ask the task for a reward -> hand the main trajectory to miles

Nothing here knows what the task is. ``sandbox`` owns the platform, ``swe`` owns
the task, and the single task-specific line is the import below -- swap it and
the same rollout trains something else.

Wire it up with::

    --custom-generate-function-path generate.generate
and put this directory on PYTHONPATH (see ``run_qwen35_35b_a3b.sh``).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import shlex

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.sglang_rollout import get_model_url

from proxy import MODEL_NAME, ModelProxy
import sandbox
from swe import SweTask

SANDBOX_MODEL_PORT = 30001  # where the tunnel surfaces the proxy inside the sandbox
WSTUNNEL_SERVER_PORT = 19090  # sandbox-side ingress port the host client dials
# Paths inside the prebuilt sandbox template; change them if your image installs
# qwen-code or wstunnel elsewhere. The host-side wstunnel is resolved from PATH.
AGENT_BIN = "/__avaeval_agentic_protocol_v1__/frameworks/qwen_code/bin/qwen"
WSTUNNEL_BIN = "/__avaeval_agentic_protocol_v1__/linux/bin/wstunnel"
MAX_TURNS = 80
AGENT_TIMEOUT = 90 * 60  # one qwen-code episode


def agent_command(workdir: str, prompt: str) -> str:
    """Non-interactive qwen-code, pointed at the tunnelled proxy.

    Swapping the CLI is this function and nothing else: the ledger, the proxy
    and the task do not know which agent is running.
    """
    base_url = f"http://127.0.0.1:{SANDBOX_MODEL_PORT}/v1"
    # qwen-code writes per-project state under ~/.qwen/tmp/<sha256(cwd)> and
    # fails if the directory is missing. cwd is hashed as the CLI sees it, so
    # this assumes the workdir is not reached through a symlink.
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
        stack.push_async_callback(runtime.kill)
        await stack.enter_async_context(
            sandbox.reverse_tunnel(
                runtime,
                proxy_port=proxy.port,
                sandbox_port=SANDBOX_MODEL_PORT,
                server_port=WSTUNNEL_SERVER_PORT,
                wstunnel_bin=WSTUNNEL_BIN,
                user=task.sandbox_user,
            )
        )
        await task.setup(runtime)
        await sandbox.run(
            runtime,
            agent_command(task.workdir, prompt),
            timeout=AGENT_TIMEOUT,
            user=task.sandbox_user,
            cwd=task.workdir,
        )
        reward = await task.reward(runtime)
    result = proxy.trajectory.to_sample(
        sample,
        reward=reward,
    )
    if result is None:
        raise RuntimeError("agent produced no trainable trajectory")
    return GenerateFnOutput(samples=result)
