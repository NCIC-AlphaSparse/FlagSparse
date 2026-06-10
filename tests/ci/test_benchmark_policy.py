"""Checks for the benchmark framework policy."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
BENCHMARK_CONFTEST = BENCHMARK_DIR / "conftest.py"


def _read(path):
    return (BENCHMARK_DIR / path).read_text(encoding="utf-8")


def test_performance_utils_defines_required_metrics_and_flow():
    text = _read("performance_utils.py")
    for snippet in [
        "DEFAULT_METRICS",
        '"latency_base"',
        '"latency"',
        '"speedup"',
        "statistics.median",
        "synchronize",
        "two_level_average_speedup",
        "warmup",
        "iterations",
    ]:
        assert snippet in text


def test_benchmark_shapes_are_centralized():
    assert (BENCHMARK_DIR / "attri_util.py").is_file()
    assert (BENCHMARK_DIR / "core_shapes.yaml").is_file()


def test_summary_for_plot_entrypoint_exists():
    text = _read("summary_for_plot.py")
    assert "two_level_average_speedup" in text
    assert "result_file" in text


def test_benchmark_pytest_json_record_plugin_exists():
    text = BENCHMARK_CONFTEST.read_text(encoding="utf-8")
    for snippet in [
        '"--record"',
        '"--output"',
        "def update_result",
        "RECORD_JSON",
        "pytest_terminal_summary",
        "benchmark_result.json",
    ]:
        assert snippet in text
