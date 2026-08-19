from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from examples.agentic_swe import sandbox


class FakeProcess:
    def __init__(self, argv: list[str], **_: object) -> None:
        self.argv = argv
        self.return_code: int | None = None

    def poll(self) -> int | None:
        return self.return_code

    def terminate(self) -> None:
        self.return_code = 0

    def wait(self, *, timeout: int) -> int:
        del timeout
        self.return_code = 0
        return 0

    def kill(self) -> None:
        self.return_code = -9


class FakeRuntime:
    def __init__(self) -> None:
        self.endpoint_calls = 0

    def endpoint(self, port: int) -> tuple[str, dict[str, str]]:
        assert port == 19090
        self.endpoint_calls += 1
        return (
            "https://gate.yicloud.com.cn/sandbox-connect/v1/sandboxes/sbx-test/proxy/19090/ping",
            {
                "X-OGW-TICK": str(self.endpoint_calls),
                "X-Sandbox-Access-Token": "secret-token",
            },
        )


class FakeCommandResult:
    def __init__(self, return_code: int, stdout: str = "", stderr: str = "") -> None:
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class FakeDetachedRuntime:
    def __init__(self) -> None:
        self.polls = 0
        self.writes: dict[str, bytes] = {}
        self.commands: list[str] = []

    async def write(self, path: str, data: bytes) -> None:
        self.writes[path] = data

    async def run(self, argv: list[str], **_: object) -> FakeCommandResult:
        command = argv[-1]
        self.commands.append(command)
        if "setsid /bin/bash" in command:
            return FakeCommandResult(0, "12345\n")
        if command.startswith("if [[ -f"):
            self.polls += 1
            return FakeCommandResult(0, "RUNNING\n" if self.polls == 1 else "7\n")
        return FakeCommandResult(0)

    async def read(self, path: str) -> bytes:
        if path.endswith(".stdout"):
            return b"detached-stdout\n"
        if path.endswith(".stderr"):
            return b"detached-stderr\n"
        raise AssertionError(path)


class ReverseTunnelTest(unittest.IsolatedAsyncioTestCase):
    async def test_long_command_uses_detached_polling(self) -> None:
        runtime = FakeDetachedRuntime()
        with (
            mock.patch.object(sandbox, "EXECD_STREAM_MAX_SECONDS", 1),
            mock.patch.object(sandbox, "DETACHED_POLL_SECONDS", 0),
        ):
            result = await sandbox.run(runtime, "exit 7", timeout=2)  # type: ignore[arg-type]

        self.assertEqual(result.exit_code, 7)
        self.assertEqual(result.stdout, "detached-stdout\n")
        self.assertEqual(result.stderr, "detached-stderr\n")
        self.assertEqual(runtime.polls, 2)
        self.assertTrue(any(command.startswith("rm -f") for command in runtime.commands))

    async def test_create_sandbox_retries_transient_failure(self) -> None:
        runtimes = [mock.AsyncMock(), mock.AsyncMock(), mock.AsyncMock()]
        runtimes[0].start.side_effect = RuntimeError("quota release pending")
        runtimes[1].start.side_effect = RuntimeError("service unavailable")

        with (
            mock.patch.object(sandbox, "YiCloudSandboxSession", side_effect=runtimes) as session_cls,
            mock.patch.object(sandbox.asyncio, "sleep", mock.AsyncMock()) as sleep,
            mock.patch.dict(
                sandbox.os.environ,
                {
                    "YICLOUD_SANDBOX_ENVIRONMENT_ID": "env-test",
                    "AVATRAIN_SANDBOX_CREATE_ATTEMPTS": "3",
                    "AVATRAIN_SANDBOX_READY_TIMEOUT_SECONDS": "900",
                },
            ),
        ):
            result = await sandbox.create_sandbox(template="image:tag", envs={})

        self.assertIs(result, runtimes[2])
        self.assertEqual(sleep.await_args_list, [mock.call(1), mock.call(2)])
        self.assertEqual(session_cls.call_args.kwargs["ready_timeout"].total_seconds(), 900)

    async def test_reverse_tunnel_refreshes_headers_from_a_file(self) -> None:
        processes: list[FakeProcess] = []

        async def fake_run(*_: object, **__: object) -> sandbox.ExecResult:
            return sandbox.ExecResult(exit_code=0, stdout="123\n", stderr="")

        def fake_popen(argv: list[str], **kwargs: object) -> FakeProcess:
            process = FakeProcess(argv, **kwargs)
            processes.append(process)
            return process

        runtime = FakeRuntime()
        with (
            mock.patch.object(sandbox, "run", fake_run),
            mock.patch.object(sandbox.subprocess, "Popen", fake_popen),
            mock.patch.object(sandbox, "WSTUNNEL_STARTUP_SECONDS", 0),
            mock.patch.object(sandbox, "WSTUNNEL_HEADER_REFRESH_SECONDS", 0.01),
        ):
            async with sandbox.reverse_tunnel(
                runtime,  # type: ignore[arg-type]
                proxy_port=12345,
                sandbox_port=30001,
                server_port=19090,
                wstunnel_bin="/usr/local/bin/wstunnel",
            ):
                await asyncio.sleep(0.03)
                argv = processes[0].argv
                header_path = Path(argv[argv.index("--http-headers-file") + 1])
                self.assertEqual(header_path.stat().st_mode & 0o777, 0o600)
                self.assertNotIn("X-OGW-TICK: 1", header_path.read_text())
                self.assertNotIn("secret-token", argv)
                self.assertFalse(any(argument.startswith("--http-headers=") for argument in argv))

        self.assertGreaterEqual(runtime.endpoint_calls, 2)
        self.assertFalse(header_path.exists())

    async def test_wstunnel_header_file_rejects_line_breaks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "line breaks"):
                sandbox._write_wstunnel_headers(Path(directory) / "headers", {"Header": "bad\nvalue"})
