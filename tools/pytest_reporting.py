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

"""Common report fields used by FlagSparse's pytest suites."""

import json
from pathlib import Path

_BUILTIN_MARKS = {
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


def operator_marks(item):
    return [
        mark.name for mark in item.iter_markers() if mark.name not in _BUILTIN_MARKS
    ]


def report_reason(report):
    longrepr = report.longrepr
    if hasattr(longrepr, "reprcrash"):
        return longrepr.reprcrash.message
    if isinstance(longrepr, tuple):
        return str(longrepr[2])
    return str(longrepr)


def write_json_report(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists() and path.stat().st_size:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    existing.update(results)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
