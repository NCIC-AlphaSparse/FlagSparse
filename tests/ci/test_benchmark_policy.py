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


def test_spgemm_csv_is_written_incrementally():
    # A rocSPARSE csrgemm page fault on DCU aborts the process with SIGABRT,
    # which no Python handler can intercept.  SpGEMM used to buffer every row
    # until the sweep ended, so the crash left an empty CSV and discarded the
    # matrices that had already succeeded.  Rows must reach disk per matrix.
    source = (PROJECT_ROOT / "tests" / "test_spgemm.py").read_text(encoding="utf-8")
    export = source.split("def run_all_dtypes_export_csv(", 1)[1].split("\ndef ", 1)[0]
    assert "on_entry=_emit" in export, "per-matrix CSV callback is not wired up"
    assert "handle.flush()" in source and "os.fsync(handle.fileno())" in source
    # the batch must not be buffered into a list that is only written at the end
    assert "rows.append(" not in export


def test_spgemm_records_the_item_in_flight_for_crash_attribution():
    source = (PROJECT_ROOT / "tests" / "test_spgemm.py").read_text(encoding="utf-8")
    assert "def _record_spgemm_inflight(" in source
    assert "def _clear_spgemm_inflight(" in source
    export = source.split("def run_all_dtypes_export_csv(", 1)[1].split("\ndef ", 1)[0]
    # written before the work starts, cleared only after a clean finish
    assert "_record_spgemm_inflight(" in export
    assert "_clear_spgemm_inflight(csv_path)" in export
