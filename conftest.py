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

"""Pytest options shared by FlagSparse's accuracy and benchmark suites."""

import json
from pathlib import Path

import pytest

from tools.pytest_reporting import operator_marks


def pytest_addoption(parser):
    parser.addoption("--record", choices=("none", "log", "json"), default="none")
    parser.addoption("--output", help="Path to the JSON result file.")
    parser.addoption(
        "--collect-marks",
        nargs="?",
        const="-",
        metavar="FILE",
        help="Collect test markers and write them to FILE or stdout.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    destination = config.getoption("--collect-marks")
    if destination is None:
        return
    marks = [
        {
            "file": item.location[0],
            "test_case": item.name,
            "function": getattr(item, "originalname", None),
            "marks": operator_marks(item),
        }
        for item in items
    ]
    report = json.dumps(marks, indent=2)
    if destination == "-":
        print(report)
    else:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    if not config.option.collectonly:
        items.clear()
