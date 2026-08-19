from __future__ import annotations

from examples.agentic_swe.swe import (
    CANDIDATE_PATCH_PATH,
    TEST_PATCH_PATH,
    SweTask,
    _normalize_test_name,
    _test_commands,
)


def test_jest_command_uses_bounded_workers(monkeypatch):
    monkeypatch.setenv("AVATRAIN_JEST_MAX_WORKERS", "2")

    assert _test_commands("npx jest --verbose --no-color") == ("npx jest --verbose --no-color --maxWorkers=2",)


def test_jest_command_preserves_explicit_workers(monkeypatch):
    monkeypatch.setenv("AVATRAIN_JEST_MAX_WORKERS", "2")

    assert _test_commands("npx jest --maxWorkers=4") == ("npx jest --maxWorkers=4",)


def test_hapi_test_name_drops_runtime_and_assertion_suffix():
    assert _normalize_test_name("does the expected thing (16 ms and 2 assertions)") == "does the expected thing"


def test_eval_script_directly_applies_candidate_and_clears_new_held_out_test():
    task = SweTask(
        repo="example/repo",
        workdir="/repo",
        template="example",
        env={},
        sandbox_bootstrap="",
        sandbox_agent_bootstrap="",
        sandbox_user=None,
        base_commit="abc123",
        test_patch="""\
diff --git a/tests/test_new.py b/tests/test_new.py
new file mode 100644
--- /dev/null
+++ b/tests/test_new.py
@@ -0,0 +1 @@
+pass
""",
        test_commands=("pytest -q",),
        log_parser="parse_log_pytest",
        fail_to_pass=(),
        pass_to_pass=(),
    )

    script = task._eval_script("candidate")

    assert "git reset --hard abc123\ngit clean -fd\n" in script
    candidate_line = next(line for line in script.splitlines() if CANDIDATE_PATCH_PATH in line)
    test_line = next(line for line in script.splitlines() if TEST_PATCH_PATH in line)
    assert "--3way" not in candidate_line
    assert "git rm -f --ignore-unmatch -- tests/test_new.py" in script
    assert "rm -f -- tests/test_new.py" in script
    assert "--3way" in test_line
