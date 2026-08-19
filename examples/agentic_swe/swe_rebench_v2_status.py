from enum import Enum


class TestStatus(str, Enum):  # noqa: UP042 - keep the vendored upstream API
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
