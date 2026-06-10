"""Checks for the shared accuracy testing policy."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACCURACY_UTILS = PROJECT_ROOT / "tests" / "pytest" / "accuracy_utils.py"
ACCURACY_CONFTEST = PROJECT_ROOT / "tests" / "pytest" / "conftest.py"


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
    text = ACCURACY_CONFTEST.read_text(encoding="utf-8")
    for snippet in [
        '"--record"',
        '"--output"',
        "RECORD_JSON",
        "pytest_terminal_summary",
        "accuracy_result.json",
    ]:
        assert snippet in text
