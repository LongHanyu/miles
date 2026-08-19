"""Prepare and grade one SWE-bench Verified task."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from swebench.harness.log_parsers import MAP_REPO_TO_PARSER_PY
from unidiff import PatchSet

if TYPE_CHECKING:
    from miles.utils.types import Sample

if __package__:
    from . import sandbox, swe_rebench_v2_log_parsers
else:  # Loaded as swe through the rollout PYTHONPATH.
    import sandbox
    import swe_rebench_v2_log_parsers

# swebench 4.1 ships get_modified_files but not get_new_files, so the test patch
# is split here. unidiff comes with swebench, which parses patches with it too.

CANDIDATE_PATCH_PATH = "/tmp/candidate.patch"
TEST_PATCH_PATH = "/tmp/test.patch"
EVAL_READY_MARKER = "__AVATRAIN_SWE_EVAL_READY__"
EVAL_TIMEOUT = 60 * 60  # tests in the fresh grading sandbox
BOOTSTRAP_TIMEOUT = 60 * 60  # prepare tools, repository, and test environment
WSTUNNEL_ARCHIVE_ENV = "AVATRAIN_SANDBOX_WSTUNNEL_ARCHIVE"
WSTUNNEL_ARCHIVE_PATH = "/tmp/avatrain-wstunnel.tar.gz"
WSTUNNEL_BIN = "/__avaeval_agentic_protocol_v1__/linux/bin/wstunnel"
REPO_ARCHIVE_ENV = "AVATRAIN_SANDBOX_REPO_ARCHIVE"
REPO_ARCHIVE_PATH = "/tmp/avatrain-repo.tar.gz"
_TIMING_NORMALIZE_RES = (
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
    re.compile(
        r"\s*\(\s*\d+(?:\.\d+)?\s*ms\s+and\s+\d+\s+assertions?\s*\)\s*$",
        re.IGNORECASE,
    ),
)


@lru_cache(maxsize=2)
def _archive_from_env(name: str) -> bytes | None:
    path = os.environ.get(name, "").strip()
    if not path:
        return None
    return Path(path).read_bytes()


def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def _test_commands(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise TypeError("install_config.test_cmd must be a string or list of strings")
    commands = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if not commands:
        raise ValueError("install_config.test_cmd must contain at least one command")
    jest_workers = int(os.environ.get("AVATRAIN_JEST_MAX_WORKERS", "2"))
    if jest_workers < 1:
        raise ValueError("AVATRAIN_JEST_MAX_WORKERS must be positive")
    return tuple(
        f"{command} --maxWorkers={jest_workers}"
        if re.search(r"(?:^|\s)npx\s+jest(?:\s|$)", command) and "--maxWorkers" not in command
        else command
        for command in commands
    )


def _apply_patch(path: str, *, three_way: bool = True) -> str:
    args = ["git", "apply", "-v"]
    if three_way:
        args.append("--3way")
    args.extend(["--recount", "--ignore-space-change", "--whitespace=nowarn", path])
    return shlex.join(args)


@dataclass(frozen=True)
class SweTask:
    repo: str
    workdir: str
    template: str
    env: dict[str, str]
    sandbox_bootstrap: str
    sandbox_agent_bootstrap: str
    sandbox_user: str | None
    base_commit: str
    test_patch: str
    test_commands: tuple[str, ...]
    log_parser: str | None
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]

    @classmethod
    def from_sample(cls, sample: Sample) -> SweTask:
        metadata = sample.metadata
        install = metadata["install_config"]
        repo = metadata["repo"]
        return cls(
            repo=repo,
            workdir=metadata.get("repo_workdir") or f"/{repo.rsplit('/', 1)[-1]}",
            template=metadata.get("inspire_template") or metadata["image_name"],
            env=dict(metadata.get("docker_image_env") or {}),
            sandbox_bootstrap=metadata.get("sandbox_bootstrap", ""),
            sandbox_agent_bootstrap=metadata.get("sandbox_agent_bootstrap", ""),
            sandbox_user=metadata.get("docker_image_default_user") or None,
            base_commit=metadata["base_commit"],
            test_patch=metadata.get("test_patch", ""),
            test_commands=_test_commands(install["test_cmd"]),
            log_parser=install.get("log_parser"),
            fail_to_pass=tuple(metadata["FAIL_TO_PASS"]),
            pass_to_pass=tuple(metadata["PASS_TO_PASS"]),
        )

    async def setup(self, runtime: Any) -> None:
        """Pin the repo at the base commit, so the later diff is the agent's work."""
        await self.bootstrap(runtime)
        await self.bootstrap_agent(runtime)
        script = "\n".join(
            [
                "set -euo pipefail",
                f"git reset --hard {shlex.quote(self.base_commit)}",
                "git clean -fd",
            ]
        )
        result = await sandbox.run(runtime, script, timeout=600, user=self.sandbox_user, cwd=self.workdir)
        if result.exit_code != 0:
            raise RuntimeError(f"workspace preparation failed: {result.output[-2000:]}")

    async def bootstrap(self, runtime: Any) -> None:
        """Prepare a generic image when the task has no prebuilt SWE image."""
        archive = _archive_from_env(WSTUNNEL_ARCHIVE_ENV)
        if archive is not None:
            await sandbox.write(runtime, WSTUNNEL_ARCHIVE_PATH, archive, user="root")
            install = await sandbox.run(
                runtime,
                "\n".join(
                    [
                        "set -euo pipefail",
                        f"mkdir -p {shlex.quote(str(Path(WSTUNNEL_BIN).parent))}",
                        f"tar -xzf {shlex.quote(WSTUNNEL_ARCHIVE_PATH)} -C /tmp wstunnel",
                        f"install -m 0755 /tmp/wstunnel {shlex.quote(WSTUNNEL_BIN)}",
                        f"rm -f /tmp/wstunnel {shlex.quote(WSTUNNEL_ARCHIVE_PATH)}",
                    ]
                ),
                timeout=120,
                user="root",
                cwd="/",
            )
            if install.exit_code != 0:
                raise RuntimeError(f"sandbox wstunnel install failed: {install.output[-2000:]}")
        repo_archive = _archive_from_env(REPO_ARCHIVE_ENV)
        if repo_archive is not None:
            await sandbox.write(runtime, REPO_ARCHIVE_PATH, repo_archive, user="root")
        if not self.sandbox_bootstrap.strip():
            return
        result = await sandbox.run(
            runtime,
            self.sandbox_bootstrap,
            timeout=BOOTSTRAP_TIMEOUT,
            user="root",
            cwd="/",
        )
        if result.exit_code != 0:
            raise RuntimeError(f"sandbox bootstrap failed: {result.output[-4000:]}")

    async def bootstrap_agent(self, runtime: Any) -> None:
        if not self.sandbox_agent_bootstrap.strip():
            return
        result = await sandbox.run(
            runtime,
            self.sandbox_agent_bootstrap,
            timeout=BOOTSTRAP_TIMEOUT,
            user="root",
            cwd="/",
        )
        if result.exit_code != 0:
            raise RuntimeError(f"agent runtime bootstrap failed: {result.output[-4000:]}")

    async def reward(self, runtime: Any) -> float:
        """Take the agent's diff, then grade it in a sandbox it never touched."""
        return await self.reward_patch(await self.candidate_patch(runtime))

    async def candidate_patch(self, runtime: Any) -> str:
        return await self._diff(runtime)

    async def reward_patch(self, patch: str) -> float:
        """Grade a captured patch without retaining the agent sandbox."""
        clean = await sandbox.create_sandbox(template=self.template, envs=self.env)
        try:
            await self.bootstrap(clean)
            for path, content in ((CANDIDATE_PATCH_PATH, patch), (TEST_PATCH_PATH, self.test_patch)):
                if content.strip():
                    await sandbox.write(clean, path, content, user=self.sandbox_user)
            result = await sandbox.run(
                clean,
                self._eval_script(patch),
                timeout=EVAL_TIMEOUT,
                user=self.sandbox_user,
                cwd=self.workdir,
            )
            if EVAL_READY_MARKER not in result.stdout:
                raise RuntimeError(
                    "grading workspace preparation failed before tests started: "
                    f"exit_code={result.exit_code}; output={result.output[-4000:]}"
                )
            return self.score(result.output.replace(EVAL_READY_MARKER, ""))
        finally:
            await sandbox.delete_sandbox(clean)

    async def _diff(self, runtime: Any) -> str:
        script = "\n".join(
            [
                "set -euo pipefail",
                "git add -N .",
                "git diff",
            ]
        )
        result = await sandbox.run(runtime, script, timeout=600, user=self.sandbox_user, cwd=self.workdir)
        if result.exit_code != 0:
            # Swallowing this would grade an empty patch and record a false zero.
            raise RuntimeError(f"could not read the candidate diff: {result.output[-2000:]}")
        return result.stdout

    def _eval_script(self, patch: str) -> str:
        lines = [
            "set -e",
            f"git reset --hard {shlex.quote(self.base_commit)}",
            "git clean -fd",
        ]
        if patch.strip():
            # This diff was captured from the same base commit. Direct apply is
            # deterministic and supports new untracked files that --3way may
            # reject because they do not exist in the base index.
            lines.append(_apply_patch(CANDIDATE_PATCH_PATH, three_way=False))
        if self.test_patch.strip():
            # The agent may have edited the tests it is graded on; reset only
            # the files the official patch owns, not the candidate patch.
            files = list(PatchSet(self.test_patch))
            modified = [file.source_file[2:] for file in files if file.source_file.startswith("a/")]
            new = [file.target_file[2:] for file in files if file.source_file == "/dev/null"]
            if modified:
                lines.append(shlex.join(["git", "checkout", self.base_commit, "--", *modified]))
            if new:
                # Candidate application may have added the same test to both the
                # worktree and index. Clear both before applying the held-out copy.
                lines.append(shlex.join(["git", "rm", "-f", "--ignore-unmatch", "--", *new]))
                lines.append(shlex.join(["rm", "-f", "--", *new]))
            lines.append(_apply_patch(TEST_PATCH_PATH))
        return "\n".join([*lines, f"printf '%s\\n' {EVAL_READY_MARKER}", "set +e", *self.test_commands])

    def score(self, test_output: str) -> float:
        """Reward = (fraction of FAIL_TO_PASS fixed) x (fraction of PASS_TO_PASS kept).

        Dense because a binary flag leaves hard instances with no gradient.
        """
        if self.log_parser:
            parser = swe_rebench_v2_log_parsers.NAME_TO_PARSER.get(self.log_parser)
            if parser is None:
                parser = getattr(swe_rebench_v2_log_parsers, self.log_parser, None)
            if parser is None:
                raise ValueError(f"unknown SWE-Rebench-V2 log parser: {self.log_parser}")
            parsed = parser(test_output or "")
        else:
            # The legacy SWE-bench Verified parsers require a test_spec argument.
            parsed = MAP_REPO_TO_PARSER_PY[self.repo](test_output or "", None)  # type: ignore[arg-type]
        parsed = {_normalize_test_name(name): status for name, status in parsed.items()}
        passed = {name for name, status in parsed.items() if status in {"PASSED", "XFAIL"}}
        skipped = {name for name, status in parsed.items() if status == "SKIPPED"}

        f2p_expected = {_normalize_test_name(name) for name in self.fail_to_pass}
        p2p_expected = {_normalize_test_name(name) for name in self.pass_to_pass}
        f2p_fixed = passed & f2p_expected
        f2p_skipped = skipped & f2p_expected
        p2p_skipped = skipped & p2p_expected
        p2p_broken = p2p_expected - passed - p2p_skipped

        f2p_ratio = len(f2p_fixed) / max(len(f2p_expected - f2p_skipped), 1)
        p2p_ratio = 1.0 - len(p2p_broken) / max(len(p2p_expected - p2p_skipped), 1)
        return f2p_ratio * p2p_ratio
