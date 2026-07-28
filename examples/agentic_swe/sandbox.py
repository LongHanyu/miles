"""Inspire sandboxes: boot one, run commands in it, reach it from the trainer.

Platform code only. Nothing here knows what the agent is doing or how it is
graded, so running the same rollout somewhere else is a matter of replacing this
module.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass

# The SDK sits on PYTHONPATH rather than in site-packages -- it is installed at a
# cluster path; run_qwen35_35b_a3b.sh puts it there.
from inspire_sandbox import AsyncSandbox, CommandExitException

SANDBOX_TTL = 3 * 60 * 60  # platform lifetime of each sandbox


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()


async def create_sandbox(
    *,
    template: str,
    envs: Mapping[str, str],
) -> AsyncSandbox:
    return await AsyncSandbox.create(
        template=template,
        timeout=SANDBOX_TTL,
        envs=dict(envs),
        network={"allow_public_traffic": True},
    )


async def run(
    sandbox: AsyncSandbox,
    script: str,
    *,
    timeout: int,
    user: str | None = None,
    cwd: str | None = None,
) -> ExecResult:
    """Run a shell script in the sandbox and always return a result.

    The SDK raises ``CommandExitException`` on a non-zero exit, but non-zero is
    the normal case here (failing tests, an agent CLI exiting on its turn limit),
    so it is turned back into data.
    """
    try:
        result = await sandbox.commands.run(
            f"/bin/bash -c {shlex.quote(script)}",
            timeout=timeout,
            request_timeout=timeout + 60,
            user=user,
            cwd=cwd,
        )
        return ExecResult(0, result.stdout or "", result.stderr or "")
    except CommandExitException as exc:
        return ExecResult(int(exc.exit_code), exc.stdout or "", exc.stderr or "")


@contextlib.asynccontextmanager
async def reverse_tunnel(
    sandbox: AsyncSandbox,
    *,
    proxy_port: int,
    sandbox_port: int,
    server_port: int,
    wstunnel_bin: str,
    user: str | None = None,
):
    """Expose the trainer-side proxy at ``sandbox_port`` inside the sandbox."""
    handle = await sandbox.commands.run(
        f"{shlex.quote(wstunnel_bin)} server ws://0.0.0.0:{server_port}",
        background=True,
        timeout=0,
        request_timeout=120,
        user=user,
    )
    try:
        process = subprocess.Popen(
            [
                "wstunnel",
                "client",
                # Keep the gateway WebSocket alive between model requests.
                "--websocket-ping-frequency",
                "30s",
                "--connection-min-idle",
                "2",
                "-R",
                f"tcp://127.0.0.1:{sandbox_port}:127.0.0.1:{proxy_port}",
                f"wss://{sandbox.get_host(server_port)}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        try:
            await asyncio.sleep(3)
            if process.poll() is not None:
                raise RuntimeError("host wstunnel client exited before the agent started")
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    await asyncio.to_thread(process.wait, timeout=10)
                if process.poll() is None:
                    process.kill()
    finally:
        with contextlib.suppress(Exception):
            await handle.kill()
