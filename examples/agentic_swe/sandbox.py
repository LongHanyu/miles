"""YiCloud OpenSandbox adapter for Agentic SWE rollouts."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shlex
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from ava_core_yicloud import YiCloudSandboxSession

SANDBOX_TTL = timedelta(minutes=120)
WSTUNNEL_HEADER_REFRESH_SECONDS = 120
WSTUNNEL_STARTUP_SECONDS = 3
EXECD_STREAM_MAX_SECONDS = int(os.environ.get("AVATRAIN_EXECD_STREAM_MAX_SECONDS", "300"))
DETACHED_POLL_SECONDS = float(os.environ.get("AVATRAIN_DETACHED_POLL_SECONDS", "10"))
DETACHED_MAX_INCONCLUSIVE_POLLS = int(os.environ.get("AVATRAIN_DETACHED_MAX_INCONCLUSIVE_POLLS", "6"))
logger = logging.getLogger(__name__)


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
) -> YiCloudSandboxSession:
    attempts = int(os.environ.get("AVATRAIN_SANDBOX_CREATE_ATTEMPTS", "6"))
    if attempts < 1:
        raise ValueError("AVATRAIN_SANDBOX_CREATE_ATTEMPTS must be positive")
    ready_timeout_seconds = int(os.environ.get("AVATRAIN_SANDBOX_READY_TIMEOUT_SECONDS", "300"))
    if ready_timeout_seconds < 1:
        raise ValueError("AVATRAIN_SANDBOX_READY_TIMEOUT_SECONDS must be positive")
    for attempt in range(attempts):
        runtime = YiCloudSandboxSession(
            template,
            environment_id=os.environ["YICLOUD_SANDBOX_ENVIRONMENT_ID"],
            project_name=os.environ.get("YICLOUD_PROJECT_NAME"),
            api_host=os.environ.get("YICLOUD_API_HOST", "https://gate.yicloud.com.cn"),
            envs=dict(envs),
            timeout=SANDBOX_TTL,
            ready_timeout=timedelta(seconds=ready_timeout_seconds),
            exposed_ports=(19090,),
        )
        try:
            await runtime.start()
            return runtime
        except Exception:
            if attempt + 1 == attempts:
                raise
            delay = min(2**attempt, 10)
            logger.warning(
                "YiCloud Sandbox create failed; retrying in %ss (attempt %d/%d)",
                delay,
                attempt + 1,
                attempts,
                exc_info=True,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def delete_sandbox(runtime: YiCloudSandboxSession) -> None:
    try:
        await runtime.delete()
    except Exception:
        # Cleanup is best-effort: a transient control-plane failure must not
        # replace a completed trajectory or reward with a rollout failure.
        logger.warning("failed to delete YiCloud Sandbox after retries", exc_info=True)


async def write(
    runtime: YiCloudSandboxSession,
    path: str,
    data: str | bytes,
    *,
    user: str | None = None,
) -> None:
    payload = data.encode() if isinstance(data, str) else data
    await runtime.write(path, payload)
    if user and user != "root":
        result = await runtime.run(["chown", user, path], user="root")
        if result.return_code:
            raise RuntimeError(f"failed to set owner of {path!r}: {result.stderr}")


async def run(
    runtime: YiCloudSandboxSession,
    script: str,
    *,
    timeout: int,
    user: str | None = None,
    cwd: str | None = None,
) -> ExecResult:
    """Run a shell script and always return a result."""
    if timeout > EXECD_STREAM_MAX_SECONDS:
        return await _run_detached(runtime, script, timeout=timeout, user=user, cwd=cwd)
    return await _run_direct(runtime, script, timeout=timeout, user=user, cwd=cwd)


async def _run_direct(
    runtime: YiCloudSandboxSession,
    script: str,
    *,
    timeout: int,
    user: str | None,
    cwd: str | None,
) -> ExecResult:
    result = await runtime.run(
        ["/bin/bash", "-c", script],
        timeout=timedelta(seconds=timeout),
        user=user,
        cwd=cwd,
    )
    return ExecResult(result.return_code, result.stdout, result.stderr)


async def _run_detached(
    runtime: YiCloudSandboxSession,
    script: str,
    *,
    timeout: int,
    user: str | None,
    cwd: str | None,
) -> ExecResult:
    """Run a long command without holding one gateway SSE response open."""
    command_id = uuid.uuid4().hex
    prefix = f"/tmp/avatrain-command-{command_id}"
    wrapper_path = f"{prefix}.sh"
    stdout_path = f"{prefix}.stdout"
    stderr_path = f"{prefix}.stderr"
    status_path = f"{prefix}.status"
    status_tmp_path = f"{status_path}.tmp"
    wrapper = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set +e",
            "(",
            script,
            ")",
            "rc=$?",
            f"printf '%s\\n' \"$rc\" > {shlex.quote(status_tmp_path)}",
            f"mv -f {shlex.quote(status_tmp_path)} {shlex.quote(status_path)}",
            'exit "$rc"',
        ]
    )
    await write(runtime, wrapper_path, wrapper, user=user)

    launch = await _run_direct(
        runtime,
        "\n".join(
            [
                f"chmod 700 {shlex.quote(wrapper_path)}",
                (
                    f"setsid /bin/bash {shlex.quote(wrapper_path)} "
                    f"> {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)} < /dev/null &"
                ),
                "printf '%s\\n' \"$!\"",
            ]
        ),
        timeout=30,
        user=user,
        cwd=cwd,
    )
    pid = launch.stdout.strip().splitlines()[-1] if launch.stdout.strip() else ""
    if launch.exit_code != 0 or not pid.isdigit():
        raise RuntimeError(f"failed to start detached sandbox command: {launch.output[-2000:]}")

    deadline = asyncio.get_running_loop().time() + timeout
    completed = False
    inconclusive_polls = 0
    try:
        while asyncio.get_running_loop().time() < deadline:
            check = await _run_direct(
                runtime,
                "\n".join(
                    [
                        f"if [[ -f {shlex.quote(status_path)} ]]; then",
                        f"  cat {shlex.quote(status_path)}",
                        f"elif kill -0 {pid} 2>/dev/null; then",
                        "  printf 'RUNNING\\n'",
                        "else",
                        "  printf 'LOST\\n'",
                        "fi",
                    ]
                ),
                timeout=30,
                user=user,
                cwd=cwd,
            )
            state = check.stdout.strip().splitlines()[-1] if check.stdout.strip() else ""
            if state.isdigit():
                exit_code = int(state)
                completed = True
                break
            if state == "RUNNING":
                inconclusive_polls = 0
            else:
                inconclusive_polls += 1
                if inconclusive_polls >= DETACHED_MAX_INCONCLUSIVE_POLLS:
                    raise RuntimeError(
                        "detached sandbox command could not be confirmed after "
                        f"{inconclusive_polls} polls: pid={pid}; state={state!r}; "
                        f"poll_exit_code={check.exit_code}; poll_output={check.output[-1000:]!r}"
                    )
                logger.warning(
                    "detached sandbox command poll was inconclusive: pid=%s state=%r exit_code=%s attempt=%d/%d",
                    pid,
                    state,
                    check.exit_code,
                    inconclusive_polls,
                    DETACHED_MAX_INCONCLUSIVE_POLLS,
                )
            await asyncio.sleep(DETACHED_POLL_SECONDS)
        else:
            raise TimeoutError(f"sandbox command exceeded timeout={timeout}s")

        stdout, stderr = await asyncio.gather(runtime.read(stdout_path), runtime.read(stderr_path))
        return ExecResult(
            exit_code=exit_code,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )
    finally:
        if not completed:
            with contextlib.suppress(Exception):
                await _run_direct(
                    runtime,
                    f"kill -TERM -- -{pid} 2>/dev/null || true; sleep 1; kill -KILL -- -{pid} 2>/dev/null || true",
                    timeout=30,
                    user=user,
                    cwd=cwd,
                )
        with contextlib.suppress(Exception):
            await _run_direct(
                runtime,
                shlex.join(
                    [
                        "rm",
                        "-f",
                        wrapper_path,
                        stdout_path,
                        stderr_path,
                        status_path,
                        status_tmp_path,
                    ]
                ),
                timeout=30,
                user=user,
                cwd=cwd,
            )


def _write_wstunnel_headers(path: Path, headers: Mapping[str, str]) -> None:
    lines: list[str] = []
    for name, value in headers.items():
        if "\n" in name or "\r" in name or "\n" in value or "\r" in value:
            raise ValueError("wstunnel headers must not contain line breaks")
        lines.append(f"{name}: {value}\n")

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.writelines(lines)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


async def _refresh_wstunnel_headers(
    runtime: YiCloudSandboxSession,
    *,
    server_port: int,
    endpoint: str,
    header_path: Path,
) -> None:
    while True:
        await asyncio.sleep(WSTUNNEL_HEADER_REFRESH_SECONDS)
        refreshed_endpoint, headers = runtime.endpoint(server_port)
        if refreshed_endpoint != endpoint:
            raise RuntimeError("YiCloud Sandbox endpoint changed while the reverse tunnel was active")
        _write_wstunnel_headers(header_path, headers)


@contextlib.asynccontextmanager
async def reverse_tunnel(
    runtime: YiCloudSandboxSession,
    *,
    proxy_port: int,
    sandbox_port: int,
    server_port: int,
    wstunnel_bin: str,
    user: str | None = None,
):
    """Expose the trainer-side proxy at ``sandbox_port`` inside the sandbox."""
    start = await run(
        runtime,
        (
            f"nohup {shlex.quote(wstunnel_bin)} server --websocket-mask-frame "
            f"ws://0.0.0.0:{server_port} >/tmp/avatrain-wstunnel.log 2>&1 "
            "<&- & echo $!"
        ),
        timeout=30,
        user=user,
    )
    if start.exit_code != 0 or not start.stdout.strip().splitlines()[-1].isdigit():
        raise RuntimeError(f"failed to start sandbox wstunnel: {start.output[-2000:]}")
    server_pid = start.stdout.strip().splitlines()[-1]
    endpoint, headers = runtime.endpoint(server_port)
    endpoint_url = urlsplit(endpoint)
    endpoint_path = endpoint_url.path.strip("/")
    websocket_url = f"ws{endpoint_url.scheme.removeprefix('http')}://{endpoint_url.netloc}"
    client_options = (
        ["--websocket-mask-frame", "--http-upgrade-path-prefix", f"{endpoint_path}/v1"] if endpoint_path else []
    )
    with tempfile.TemporaryDirectory(prefix="avatrain-wstunnel-") as header_directory:
        header_path = Path(header_directory) / "headers"
        _write_wstunnel_headers(header_path, headers)
        process = subprocess.Popen(
            [
                "wstunnel",
                "client",
                *client_options,
                "--websocket-ping-frequency",
                "30s",
                "--connection-min-idle",
                "2",
                "-R",
                f"tcp://127.0.0.1:{sandbox_port}:127.0.0.1:{proxy_port}",
                "--http-headers-file",
                str(header_path),
                websocket_url,
            ],
            env={key: value for key, value in os.environ.items() if key.upper() != "HTTP_PROXY"},
        )
        refresh_task = asyncio.create_task(
            _refresh_wstunnel_headers(
                runtime,
                server_port=server_port,
                endpoint=endpoint,
                header_path=header_path,
            )
        )
        try:
            await asyncio.sleep(WSTUNNEL_STARTUP_SECONDS)
            if process.poll() is not None:
                log = await run(runtime, "tail -200 /tmp/avatrain-wstunnel.log", timeout=30, user=user)
                raise RuntimeError(f"host wstunnel client exited before the agent started: {log.output[-2000:]}")
            yield
        finally:
            refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresh_task
            if process.poll() is None:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    await asyncio.to_thread(process.wait, timeout=10)
                if process.poll() is None:
                    process.kill()
            with contextlib.suppress(Exception):
                await run(runtime, f"kill {server_pid}", timeout=30, user=user)
