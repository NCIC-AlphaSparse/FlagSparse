"""FlagGems-compatible pytest benchmark recording hooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

BUILTIN_MARKS = {
    "filterwarnings",
    "parametrize",
    "skip",
    "skipif",
    "timeout",
    "tryfirst",
    "trylast",
    "usefixtures",
    "xfail",
}
REPORT_FILE = "benchmark_result.json"
RECORD_JSON = False
TEST_RESULTS: dict[str, dict[str, object]] = {}


def update_result(op: str, data: dict[str, object]) -> None:
    if not RECORD_JSON:
        return
    TEST_RESULTS.setdefault(op, {})
    TEST_RESULTS[op].setdefault("details", [])
    TEST_RESULTS[op]["details"].append(data)  # type: ignore[index]


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--record",
            action="store",
            default="none",
            choices=["none", "log", "json"],
            help="record benchmark results in log/json files or not",
        )
        parser.addoption(
            "--output", default=REPORT_FILE, help="path to JSON result file"
        )
    except ValueError:
        pass
    try:
        parser.addoption(
            "--collect-marks",
            action="store_true",
            help="collect benchmark tests with marker information without executing them",
        )
    except ValueError:
        pass


def pytest_configure(config):
    global RECORD_JSON
    global REPORT_FILE
    RECORD_JSON = config.getoption("--record") == "json"
    if RECORD_JSON:
        REPORT_FILE = config.getoption("--output") or REPORT_FILE


def _operator_marks(item) -> list[str]:
    return [mark.name for mark in item.iter_markers() if mark.name not in BUILTIN_MARKS]


def _reason_from_report(report) -> str:
    longrepr = getattr(report, "longrepr", None)
    if hasattr(longrepr, "reprcrash"):
        return longrepr.reprcrash.message
    if isinstance(longrepr, tuple):
        return str(longrepr[2])
    return str(longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    marks = _operator_marks(item)
    report.opid = marks[0] if marks else item.nodeid


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    if not RECORD_JSON:
        return

    op = getattr(report, "opid", report.nodeid)
    TEST_RESULTS.setdefault(op, {})
    if report.when == "setup" and report.outcome == "skipped":
        TEST_RESULTS[op]["result"] = "skipped"
        TEST_RESULTS[op]["reason"] = _reason_from_report(report)
        TEST_RESULTS[op]["test_case"] = report.nodeid
    elif report.when == "call":
        TEST_RESULTS[op]["result"] = report.outcome
        TEST_RESULTS[op]["test_case"] = report.nodeid
        if report.outcome in {"failed", "skipped"}:
            TEST_RESULTS[op]["reason"] = _reason_from_report(report)
        else:
            TEST_RESULTS[op]["reason"] = None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not RECORD_JSON:
        return
    path = Path(REPORT_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, object] = {}
    if path.exists() and path.stat().st_size:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            existing = {}
    existing.update(TEST_RESULTS)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


def pytest_collection_modifyitems(session, config, items):
    if not config.getoption("--collect-marks"):
        return
    report = []
    for item in items:
        report.append(
            {
                "file": item.location[0],
                "test_case": item.name,
                "function": item.originalname,
                "marks": _operator_marks(item),
            }
        )
    print(json.dumps(report, indent=2, default=str))
    items.clear()
