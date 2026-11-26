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

"""Checks for the shared accuracy testing policy."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACCURACY_UTILS = PROJECT_ROOT / "tests" / "pytest" / "accuracy_utils.py"
ACCURACY_CONFTEST = PROJECT_ROOT / "tests" / "conftest.py"


def test_accuracy_policy_helpers_are_defined():
    text = ACCURACY_UTILS.read_text(encoding="utf-8")
    for snippet in [
        "TOLERANCE_BY_DTYPE",
        "def tolerance_for_dtype",
        "def golden_reference_close",
        "def golden_reference_equal",
        "def gems_assert_close",
        "def gems_assert_equal",
        "torch.testing.assert_close",
    ]:
        assert snippet in text


def test_accuracy_policy_documents_goldens():
    text = ACCURACY_UTILS.read_text(encoding="utf-8")
    assert "CPU float64 golden reference" in text
    assert "CPU int32" in text


def test_accuracy_pytest_json_record_plugin_exists():
    shared = (PROJECT_ROOT / "conftest.py").read_text(encoding="utf-8")
    accuracy = ACCURACY_CONFTEST.read_text(encoding="utf-8")
    assert '"--record"' in shared
    assert '"--output"' in shared
    assert "pytest_terminal_summary" in accuracy
    assert "accuracy_result.json" in accuracy
