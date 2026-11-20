# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Accuracy collection and result reports for the FlagSparse pytest suites."""

import sys
from pathlib import Path

import pytest

from tools.pytest_reporting import operator_marks, report_reason, write_json_report

# Benchmark scripts in tests/ also use script-local imports during collection.
_TESTS_ROOT = str(Path(__file__).resolve().parent)
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)

QUICK_MODE = False
_ACCURACY_DIR = Path(_TESTS_ROOT) / "pytest"


def pytest_addoption(parser):
    parser.addoption("--mode", choices=("normal", "quick"), default="normal")
    parser.addoption(
        "--quick", action="store_true", help="Run the quick accuracy grid."
    )
    parser.addoption("--ref", choices=("cpu",), help="Use the CPU accuracy suite.")


def pytest_configure(config):
    global QUICK_MODE
    QUICK_MODE = config.getoption("--quick") or config.getoption("--mode") == "quick"
    config._flagsparse_accuracy_results = {}


def pytest_ignore_collect(collection_path, config):
    if not config.getoption("--ref") and config.getoption("--collect-marks") is None:
        return None
    if not collection_path.is_relative_to(_ACCURACY_DIR):
        return True
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if item.config.getoption("--record") != "json":
        return
    callspec = getattr(item, "callspec", None)
    result = item.config._flagsparse_accuracy_results.setdefault(
        item.nodeid,
        {
            "params": dict(callspec.params) if callspec else {},
            "result": None,
            "opname": operator_marks(item),
        },
    )
    if report.when == "call" or report.failed or report.skipped:
        result["result"] = report.outcome
        result["reason"] = None if report.passed else report_reason(report)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if config.getoption("--record") == "json":
        path = config.getoption("--output") or "accuracy_result.json"
        write_json_report(path, config._flagsparse_accuracy_results)
