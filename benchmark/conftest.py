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

"""Write benchmark pytest outcomes in the format consumed by optest."""

import pytest

from tools.pytest_reporting import operator_marks, report_reason, write_json_report


def pytest_addoption(parser):
    parser.addoption("--level", choices=("core",), default="core")


def pytest_configure(config):
    config._flagsparse_benchmark_results = {}


@pytest.fixture
def record_benchmark(request):
    def record(details):
        request.node._flagsparse_benchmark_details = details

    return record


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if item.config.getoption("--record") != "json":
        return
    if report.when != "call" and report.passed:
        return

    marks = (
        mark
        for mark in operator_marks(item)
        if mark != "performance" and not mark.endswith("_perf")
    )
    op = next(marks, item.nodeid)
    item.config._flagsparse_benchmark_results[op] = {
        "result": report.outcome,
        "reason": None if report.passed else report_reason(report),
        "test_case": report.nodeid,
        "details": getattr(item, "_flagsparse_benchmark_details", []),
    }


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if config.getoption("--record") == "json":
        path = config.getoption("--output") or "benchmark_result.json"
        write_json_report(path, config._flagsparse_benchmark_results)
