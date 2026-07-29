"""Prepare and grade one SWE-bench Verified task."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from swebench.harness.log_parsers import MAP_REPO_TO_PARSER_PY
from swebench.harness.utils import get_modified_files, get_new_files

import sandbox
from miles.utils.types import Sample

CANDIDATE_PATCH_PATH = "/tmp/candidate.patch"
TEST_PATCH_PATH = "/tmp/test.patch"
EVAL_TIMEOUT = 60 * 60  # tests in the fresh grading sandbox


def _apply_patch(path: str) -> str:
    return shlex.join(
        [
            "git",
            "apply",
            "-v",
            "--3way",
            "--recount",
            "--ignore-space-change",
            "--whitespace=nowarn",
            path,
        ]
    )


@dataclass(frozen=True)
class SweTask:
    repo: str
    workdir: str
    template: str
    env: dict[str, str]
    sandbox_user: str | None
    base_commit: str
    test_patch: str
    test_command: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]

    @classmethod
    def from_sample(cls, sample: Sample) -> SweTask:
        metadata = sample.metadata
        install = metadata["install_config"]
        return cls(
            repo=metadata["repo"],
            workdir=metadata["repo_workdir"],
            template=metadata["inspire_template"],
            env=dict(metadata.get("docker_image_env") or {}),
            sandbox_user=metadata.get("docker_image_default_user") or None,
            base_commit=metadata["base_commit"],
            test_patch=metadata.get("test_patch", ""),
            test_command=install["test_cmd"],
            fail_to_pass=tuple(metadata["FAIL_TO_PASS"]),
            pass_to_pass=tuple(metadata["PASS_TO_PASS"]),
        )

    async def setup(self, runtime: Any) -> None:
        """Pin the repo at the base commit, so the later diff is the agent's work."""
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

    async def reward(self, runtime: Any) -> float:
        """Take the agent's diff, then grade it in a sandbox it never touched."""
        patch = await self._diff(runtime)
        clean = await sandbox.create_sandbox(template=self.template, envs=self.env)
        try:
            for path, content in ((CANDIDATE_PATCH_PATH, patch), (TEST_PATCH_PATH, self.test_patch)):
                if content.strip():
                    await clean.files.write(path, content, user=self.sandbox_user)
            result = await sandbox.run(
                clean,
                self._eval_script(patch),
                timeout=EVAL_TIMEOUT,
                user=self.sandbox_user,
                cwd=self.workdir,
            )
            return self.score(result.output)
        finally:
            await clean.kill()

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
        ]
        if patch.strip():
            lines.append(_apply_patch(CANDIDATE_PATCH_PATH))
        if self.test_patch.strip():
            # The agent may have edited the tests it is graded on; reset only
            # the files the official patch owns, not the candidate patch.
            modified, new = get_modified_files(self.test_patch), get_new_files(self.test_patch)
            if modified:
                lines.append(shlex.join(["git", "checkout", self.base_commit, "--", *modified]))
            if new:
                lines.append(shlex.join(["rm", "-f", "--", *new]))
            lines.append(_apply_patch(TEST_PATCH_PATH))
        return "\n".join([*lines, "set +e", self.test_command])

    def score(self, test_output: str) -> float:
        """Reward = (fraction of FAIL_TO_PASS fixed) x (fraction of PASS_TO_PASS kept).

        Dense because a binary flag leaves hard instances with no gradient.
        """
        parsed = MAP_REPO_TO_PARSER_PY[self.repo](test_output or "", None)  # type: ignore[arg-type]
        parsed = {name.strip(): status for name, status in parsed.items()}
        passed = {name for name, status in parsed.items() if status in {"PASSED", "XFAIL"}}
        skipped = {name for name, status in parsed.items() if status == "SKIPPED"}

        f2p_expected = {name.strip() for name in self.fail_to_pass}
        p2p_expected = {name.strip() for name in self.pass_to_pass}
        f2p_fixed = passed & f2p_expected
        f2p_skipped = skipped & f2p_expected
        p2p_skipped = skipped & p2p_expected
        p2p_broken = p2p_expected - passed - p2p_skipped

        f2p_ratio = len(f2p_fixed) / max(len(f2p_expected - f2p_skipped), 1)
        p2p_ratio = 1.0 - len(p2p_broken) / max(len(p2p_expected - p2p_skipped), 1)
        return f2p_ratio * p2p_ratio
