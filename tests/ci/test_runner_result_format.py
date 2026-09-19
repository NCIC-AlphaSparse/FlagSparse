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

"""Checks for the unified FlagSparse pytest runner result format."""

import csv
import json
import re
from pathlib import Path

import pytest

import run_flagsparse_pytest as runner
from tools.delivery_variants import load_delivery_variants

ROOT = Path(__file__).resolve().parents[2]


def _gather_accuracy_artifact(op_dir: Path) -> Path:
    """A raw pytest artifact shaped like the real gather accuracy run.

    Three dtypes are recorded and the two complex ones are absent, which is
    what the kernel actually supports -- the delivery projection has to slice
    this one file into five gather variants.
    """
    cases = {
        "tests/pytest/test_gather.py::test_gather[int32-half-512-128]": {
            "params": {"index_dtype": "torch.int32", "dtype": "torch.float16"},
            "result": "passed",
            "reason": None,
        },
        "tests/pytest/test_gather.py::test_gather[int32-float-512-128]": {
            "params": {"index_dtype": "torch.int32", "dtype": "torch.float32"},
            "result": "passed",
            "reason": None,
        },
        "tests/pytest/test_gather.py::test_gather[int32-float-1024-256]": {
            "params": {"index_dtype": "torch.int32", "dtype": "torch.float32"},
            "result": "passed",
            "reason": None,
        },
        "tests/pytest/test_gather.py::test_gather[int32-double-512-128]": {
            "params": {"index_dtype": "torch.int32", "dtype": "torch.float64"},
            "result": "skipped",
            "reason": "fp64 is not available on this device",
        },
    }
    path = op_dir / "accuracy_result.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def test_runner_writes_flaggems_style_summary(tmp_path):
    # The runner records one entry per OPERATOR and leaves the raw pytest
    # artifact beside it; write_summary is what projects those onto the fixed
    # delivery registry. So the fixture is operator-shaped, not variant-shaped.
    op_dir = tmp_path / "gather"
    op_dir.mkdir()
    accuracy_artifact = _gather_accuracy_artifact(op_dir)
    results = [
        {
            "operator": "gather",
            "gpu": 0,
            "customized": True,
            "labels": ["flagsparse", "sparse"],
            "accuracy": {
                "operator": "gather",
                "phase": "accuracy",
                "configured": True,
                "status": "PASS",
                "returncode": 0,
                "exit_code": 0,
                "duration_sec": 1.25,
                "duration": 1.25,
                "command": ["python", "-m", "pytest"],
                "log_path": "gather/accuracy_stdout.log",
                "stdout_log_path": "gather/accuracy_stdout.log",
                "stderr_log_path": "gather/accuracy_stderr.log",
                "result_path": str(accuracy_artifact),
                "data_file": "gather/accuracy_result.json",
                "passed": 3,
                "failed": 0,
                "skipped": 1,
                "errors": 0,
                "xfailed": 0,
                "xpassed": 0,
                "total": 4,
            },
            "performance": {
                "operator": "gather",
                "phase": "performance",
                "configured": True,
                "status": "SKIP",
                "returncode": 0,
                "exit_code": 0,
                "duration_sec": 0.5,
                "duration": 0.5,
                "command": ["python", "tests/test_gather.py"],
                "log_path": "gather/performance_stdout.log",
                "stdout_log_path": "gather/performance_stdout.log",
                "stderr_log_path": "gather/performance_stderr.log",
                "data_path": "gather/performance.csv",
                "data_file": "gather/performance_result.json",
                "row_count": 0,
                "test_case": "csv",
                "data": {
                    "float32": {
                        "result": "Skipped",
                        "speedup": 1.5,
                        "details": {
                            "512x128": {"base": 2.0, "gems": 1.0, "speedup": 2.0}
                        },
                    },
                    "float16": {"result": "Skipped", "speedup": 1.0, "details": {}},
                },
            },
        }
    ]
    env = {
        "python": {"version": "3.test"},
        "platform": {
            "system": "Linux",
            "release": "test-release",
            "machine": "x86_64",
        },
        "packages": {
            "torch": {"version": "2.test"},
            "triton": {"version": "3.test"},
            "flagsparse": {"version": "1.test"},
        },
        "cuda": {
            "available": True,
            "device_count": 1,
            "devices": [{"name": "Test GPU"}],
        },
    }

    runner.write_summary(results, tmp_path, env)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert list(summary) == ["timestamp", "env", "result"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", summary["timestamp"])
    assert set(summary["env"]) == {
        "architecture",
        "os_name",
        "os_release",
        "python",
        "torch",
        "flagtree",
        "triton",
        "flag_gems",
    }
    assert summary["env"]["python"] == "3.test"
    assert set(summary["env"]["torch"]) == {
        "version",
        "cuda_available",
        "device_name",
        "device_count",
    }
    assert set(summary["env"]["triton"]) == {"version", "has_config"}
    assert set(summary["env"]["flag_gems"]) == {"version", "vendor", "device"}
    # `result` is keyed by DELIVERY VARIANT, not by operator: the names come from
    # conf/operators.yaml's delivery_variants, which 算子列表注册修改.xlsx spells the
    # same way (gather_f32_int, spmv_csr_f32_int_non, ...). The C API's
    # capi/tools/write_summary.py keys its summary identically, from the same
    # loader, so the two front ends produce comparable files.
    variants = load_delivery_variants(ROOT / "conf" / "operators.yaml")
    assert set(summary["result"]) == {v["id"] for v in variants}

    gather = summary["result"]["gather_f32_int"]
    assert list(gather) == ["customized", "accuracy", "performance", "labels"]
    assert gather["customized"] is True
    # Labels now carry the variant's own axes; "sparse" was the operator-level tag.
    assert gather["labels"] == ["flagsparse", "delivery", "sparse_vector", "f32"]

    accuracy = gather["accuracy"]
    assert set(accuracy) == {
        "total",
        "skipped",
        "failed",
        "passed",
        "details",
        "status",
        "duration",
        "exit_code",
        "data_file",
    }
    # Counts are the f32 SLICE of the artifact (2 of its 4 cases), not the
    # operator-level totals the runner recorded.
    assert accuracy["status"] == "Passed"
    assert accuracy["total"] == 2
    assert accuracy["passed"] == 2
    assert accuracy["skipped"] == 0
    assert accuracy["duration"] == 1.25
    assert accuracy["exit_code"] == 0
    assert accuracy["data_file"] == "gather/accuracy_result.json"

    performance = gather["performance"]
    assert set(performance) == {
        "duration",
        "exit_code",
        "data",
        "status",
        "data_file",
        "test_case",
    }
    assert performance["status"] == "Skipped"
    assert performance["data_file"] == "gather/performance_result.json"
    assert performance["test_case"] == "csv"
    # The benchmark's fp16 rows belong to a different variant.
    assert set(performance["data"]) == {"fp32"}

    # The same artifact read through a different dtype: f64 was skipped, and the
    # two complex variants have no cases at all, so they report NotFound rather
    # than inheriting gather's operator-level PASS.
    assert summary["result"]["gather_f64_int"]["accuracy"]["status"] == "Skipped"
    assert summary["result"]["gather_c32_int"]["accuracy"]["status"] == "NotFound"
    assert summary["result"]["gather_c32_int"]["performance"]["status"] == "NotFound"
    unselected = next(v for v in variants if v["operator"] != "gather")
    assert summary["result"][unselected["id"]]["accuracy"]["status"] == "NotFound"

    compat_summary = json.loads(
        (tmp_path / "summary_flat.json").read_text(encoding="utf-8")
    )
    assert {"timestamp", "env", "result", "totals", "results"} <= set(compat_summary)
    # summary_flat.json is fed the ALREADY projected results; if it projected a
    # second time every variant would be looked up by its own id and the whole
    # registry would come back not configured.
    assert {str(item["operator"]) for item in compat_summary["results"]} == {
        v["id"] for v in variants
    }
    accuracy_totals = compat_summary["totals"]["by_phase"]["accuracy"]
    assert accuracy_totals["Passed"] == 2  # gather_f16_int, gather_f32_int
    assert accuracy_totals["Skipped"] == 1  # gather_f64_int
    assert accuracy_totals["NOT_CONFIGURED"] == len(variants) - 3
    performance_totals = compat_summary["totals"]["by_phase"]["performance"]
    assert performance_totals["SKIP"] == 2  # the fp16 and fp32 benchmark rows
    assert performance_totals["NOT_CONFIGURED"] == len(variants) - 2

    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["operator"] for row in rows} == {v["id"] for v in variants}
    row = next(
        item
        for item in rows
        if item["operator"] == "gather_f32_int" and item["phase"] == "accuracy"
    )
    assert row["duration"] == "1.25"
    assert row["exit_code"] == "0"
    # A variant row keeps the parent's logs: they are how a reader gets from a
    # delivery name back to the pytest run it was sliced from.
    assert row["stdout_log_path"] == "gather/accuracy_stdout.log"
    assert row["stderr_log_path"] == "gather/accuracy_stderr.log"

    html_report = (tmp_path / "result.html").read_text(encoding="utf-8")
    assert "FlagSparse Test Report" in html_report
    assert "<h3>Test Environment</h3>" in html_report
    assert "<h3>Test Result</h3>" in html_report
    assert 'class="table-stats"' in html_report
    assert "AccRes<br>" in html_report
    assert "PerfRes<br>" in html_report
    assert "OPAverageSpeedUp" in html_report
    assert "filterTable()" in html_report
    assert "sortTable(5, 'asc')" in html_report
    assert "gather_f32_int" in html_report
    assert "accuracy_result.json" in html_report


def test_flaggems_summary_schema_does_not_change_on_errors():
    accuracy = runner._flaggems_accuracy_result(
        {
            "status": "CRASH",
            "errors": 2,
            "reason": "worker crashed",
            "exit_code": 17,
        }
    )
    performance = runner._flaggems_performance_result(
        {
            "status": "CRASH",
            "reason": "benchmark crashed",
            "exit_code": 18,
        }
    )

    assert set(accuracy) == {
        "total",
        "skipped",
        "failed",
        "passed",
        "details",
        "status",
        "exit_code",
        "duration",
        "data_file",
    }
    assert set(performance) == {
        "duration",
        "exit_code",
        "data_file",
        "data",
        "status",
        "test_case",
    }
    assert "errors" not in accuracy
    assert "reason" not in performance


def test_flaggems_operator_schema_is_fixed_for_single_phase_runs():
    summary = runner._operator_summary(
        {
            "operator": "gather",
            "accuracy": {"phase": "accuracy", "status": "PASS"},
        }
    )

    assert list(summary) == ["customized", "accuracy", "performance", "labels"]
    assert summary["performance"] == {
        "duration": 0.0,
        "exit_code": 0,
        "data_file": "",
        "data": {},
        "status": "NotFound",
        "test_case": "Unknown",
    }


def test_runner_writes_per_operator_phase_result(tmp_path):
    raw_path = tmp_path / "accuracy_result.json"
    raw_path.write_text('{"raw": {"result": "passed"}}', encoding="utf-8")
    phase_result = {
        "operator": "gather",
        "phase": "accuracy",
        "configured": True,
        "status": "FAIL",
        "returncode": 1,
        "exit_code": 1,
        "duration_sec": 2.0,
        "duration": 2.0,
        "command": ["python", "-m", "pytest"],
        "log_path": "gather/accuracy_stdout.log",
        "stdout_log_path": "gather/accuracy_stdout.log",
        "stderr_log_path": "gather/accuracy_stderr.log",
        "passed": 1,
        "failed": 1,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "total": 2,
        "failures": ["FAILED tests/pytest/test_example.py::test_case"],
        "tests": [
            {
                "nodeid": "tests/pytest/test_example.py::test_case[dtype0]",
                "status": "Failed",
                "status_raw": "FAILED",
            }
        ],
    }

    path = runner.write_phase_result(tmp_path, "accuracy", phase_result)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "accuracy_result.json"
    assert data == {"raw": {"result": "passed"}}

    detail = json.loads((tmp_path / "accuracy_detail.json").read_text(encoding="utf-8"))
    assert {
        "total",
        "skipped",
        "failed",
        "passed",
        "details",
        "status",
        "duration",
        "exit_code",
        "data_file",
    } <= set(detail)
    assert detail["status"] == "Failed"
    assert detail["duration"] == 2.0
    assert detail["exit_code"] == 1
    assert detail["data_file"] == "accuracy_result.json"
    assert detail["details"]["failed"] == [
        "FAILED tests/pytest/test_example.py::test_case"
    ]
    assert detail["status_raw"] == "FAIL"
    assert detail["stdout_log_path"] == "gather/accuracy_stdout.log"
    assert detail["stderr_log_path"] == "gather/accuracy_stderr.log"
    assert detail["summary"]["failed"] == 1
    assert (
        detail["tests"][0]["nodeid"]
        == "tests/pytest/test_example.py::test_case[dtype0]"
    )


def test_runner_parses_pytest_verbose_cases():
    output = """
tests/pytest/test_gather.py::test_gather[float32-32] PASSED
tests/pytest/test_gather.py::test_scatter[float64] SKIPPED (CUDA required)
tests/pytest/test_gather.py::test_bad_case FAILED
FAILED tests/pytest/test_gather.py::test_bad_case - AssertionError
"""

    cases = runner.parse_pytest_cases(output)

    assert [case["status"] for case in cases] == ["Passed", "Skipped", "Failed"]
    assert cases[0]["nodeid"] == "tests/pytest/test_gather.py::test_gather[float32-32]"
    assert cases[0]["parameters"] == {"param_0": "float32", "param_1": "32"}
    assert cases[1]["message"] == "(CUDA required)"


def test_runner_normalizes_performance_csv_rows(tmp_path):
    csv_path = tmp_path / "performance.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dtype",
                "shape",
                "triton_ms",
                "pytorch_ms",
                "triton_speedup_vs_pytorch",
                "ok",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dtype": "float32",
                "shape": "64x64",
                "triton_ms": "0.5",
                "pytorch_ms": "1.0",
                "triton_speedup_vs_pytorch": "2.0",
                "ok": "True",
            }
        )

    summary = runner.summarize_performance_csv(csv_path)

    entry = summary["benchmark"]["float32"]["64x64"][0]
    assert entry["metrics"]["triton_ms"] == 0.5
    assert entry["metrics"]["triton_speedup_vs_pytorch"] == 2.0
    assert entry["metadata"]["ok"] == "True"
    details = summary["data"]["float32"]["details"]
    assert len(details) == 1
    detail = next(iter(details.values()))
    assert detail["base"] == 1.0
    assert detail["gems"] == 0.5
    assert detail["speedup"] == 2.0
    assert (
        summary["records_by_dtype_shape"]["float32"]["64x64"][0]["pytorch_ms"] == "1.0"
    )


def test_spgemm_performance_isolates_dtype_crashes(tmp_path, monkeypatch):
    """A float32 rocSPARSE abort must not prevent the float64 sweep."""

    def fake_run_subprocess(cmd, **_kwargs):
        csv_path = Path(cmd[cmd.index("--csv") + 1])
        dtype = cmd[cmd.index("--dtypes") + 1]
        csv_path.write_text(
            "matrix,value_dtype,triton_ms,cusparse_ms,pytorch_ms,"
            "triton_speedup_vs_cusparse,status\n"
            f"sample.mtx,{dtype},1.0,2.0,3.0,2.0,PASS\n",
            encoding="utf-8",
        )
        return (-6 if dtype == "float32" else 0), "", "", 0.1, False

    monkeypatch.setattr(runner, "run_subprocess", fake_run_subprocess)
    result = runner._run_spgemm_split_dtypes(
        project_root=ROOT,
        op="spgemm_csr",
        gpu_id=0,
        script_device=0,
        template=("tests/test_spgemm.py", "--csv", "{csv}"),
        op_dir=tmp_path,
        benchmark_input=None,
        warmup=1,
        iters=1,
        extra_args=[],
        timeout=10,
    )

    assert result["status"] == "FAIL"
    assert result["failed_dtypes"] == ["float32"]
    assert len(result["commands"]) == 2
    assert [cmd[cmd.index("--dtypes") + 1] for cmd in result["commands"]] == [
        "float32",
        "float64",
    ]
    with (tmp_path / "performance.csv").open(encoding="utf-8", newline="") as handle:
        # A crashed child may lose its in-flight final row, but the independent
        # float64 child must still contribute a reportable measurement.
        assert {row["value_dtype"] for row in csv.DictReader(handle)} == {"float64"}


def test_runner_parses_flaggems_accuracy_json(tmp_path):
    result_path = tmp_path / "accuracy_result.json"
    result_path.write_text(
        json.dumps(
            {
                "tests/pytest/test_gather.py::test_ok[dtype0]": {
                    "params": {"dtype": "float32"},
                    "result": "passed",
                    "opname": ["gather"],
                },
                "tests/pytest/test_gather.py::test_bad[dtype0]": {
                    "params": {"dtype": "float64"},
                    "result": "failed",
                    "reason": "AssertionError",
                    "opname": ["gather"],
                },
            }
        ),
        encoding="utf-8",
    )

    result = runner.parse_accuracy_json(result_path)

    assert result["status"] == "Failed"
    assert result["total"] == 2
    assert result["passed"] == 1
    assert result["failed"] == 1
    assert "AssertionError" in result["details"]["failed"]


def test_runner_writes_and_parses_flaggems_benchmark_json(tmp_path):
    csv_path = tmp_path / "performance.csv"
    csv_path.write_text(
        "\n".join(
            [
                "dtype,shape,triton_ms,pytorch_ms,triton_speedup_vs_pytorch",
                "float32,64x64,0.5,1.0,2.0",
            ]
        ),
        encoding="utf-8",
    )
    json_path = tmp_path / "performance_result.json"

    runner.write_benchmark_json_from_csv("gather", csv_path, json_path)
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    parsed = runner.parse_performance_json("gather", json_path)

    assert raw["gather"]["result"] == "passed"
    assert raw["gather"]["test_case"] == "csv"
    assert raw["gather"]["details"][0]["dtype"] == "float32"
    assert parsed["status"] == "Passed"
    assert parsed["data"]["float32"]["details"]["64x64"]["base"] == 1.0
    assert parsed["data"]["float32"]["details"]["64x64"]["gems"] == 0.5
    assert parsed["data"]["float32"]["speedup"] == 2.0


def test_runner_treats_flaggems_failed_status_as_failure():
    assert runner._should_fail(
        [{"operator": "gather", "accuracy": {"phase": "accuracy", "status": "Failed"}}],
        strict=False,
    )


def test_timeout_status_survives_a_partial_result_artifact():
    # A run killed by --timeout can still have flushed a partial CSV/JSON, and
    # the summary parsed from it only covers the rows that reached disk.
    # Observed on DCU: spmm_bsr and spmm_csc were reported "Passed" with
    # returncode -100 because parse_performance_json overwrote the status.
    assert (
        runner.resolve_status_with_parsed(
            "TIMEOUT", "Passed", returncode=runner.TIMEOUT_RETURN_CODE, timed_out=True
        )
        == "TIMEOUT"
    )


def test_crash_status_survives_a_partial_result_artifact():
    assert (
        runner.resolve_status_with_parsed(
            "FAIL", "Passed", returncode=-6, timed_out=False
        )
        == "FAIL"
    )


def test_parsed_status_still_refines_a_clean_run():
    # The parsed artifact is more specific than the exit code on a clean run,
    # so it must keep winning there -- e.g. a benchmark that ran to completion
    # but reported its own failure.
    assert (
        runner.resolve_status_with_parsed(
            "PASS", "Failed", returncode=0, timed_out=False
        )
        == "Failed"
    )
    assert (
        runner.resolve_status_with_parsed("PASS", None, returncode=0, timed_out=False)
        == "PASS"
    )


def test_missing_measurements_are_null_not_zero(tmp_path):
    # A CSV with no speedup/latency columns carries no measurement at all.
    # Reporting 0.0 made it look like a measured 0.00x speedup -- seen on DCU
    # for spmv_csc / spmm_csc / spmm_bsr, none of which run a vendor baseline.
    csv_path = tmp_path / "performance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["matrix", "dtype", "ms"])
        writer.writeheader()
        writer.writerow({"matrix": "a.mtx", "dtype": "float32", "ms": "1.5"})

    summary = runner.summarize_performance_csv(csv_path)
    detail = next(iter(summary["data"]["float32"]["details"].values()))
    assert detail["base"] is None
    assert detail["gems"] is None
    assert detail["speedup"] is None

    out = tmp_path / "performance_result.json"
    runner.write_benchmark_json_from_csv("spmv_csc", csv_path, out)
    payload = json.loads(out.read_text(encoding="utf-8"))["spmv_csc"]
    entry = payload["details"][0]["result"][0]
    assert entry["latency_base"] is None
    assert entry["latency"] is None
    assert entry["speedup"] is None
    # the raw column is still carried through, so nothing is lost
    assert entry["ms"] == 1.5


def test_runner_excludes_interrupted_matrix_and_invalid_speedups(tmp_path):
    csv_path = tmp_path / "performance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "matrix",
                "dtype",
                "triton_ms",
                "pytorch_ms",
                "triton_speedup_vs_pytorch",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "matrix": "complete.mtx",
                "dtype": "float32",
                "triton_ms": "1.0",
                "pytorch_ms": "2.0",
                "triton_speedup_vs_pytorch": "2.0",
                "status": "PASS",
            }
        )
        writer.writerow(
            {
                "matrix": "interrupted.mtx",
                "dtype": "float32",
                "triton_ms": "1.0",
                "pytorch_ms": "4.0",
                "triton_speedup_vs_pytorch": "4.0",
                "status": "PASS",
            }
        )
        writer.writerow(
            {
                "matrix": "failed.mtx",
                "dtype": "float32",
                "triton_ms": "1.0",
                "pytorch_ms": "8.0",
                "triton_speedup_vs_pytorch": "8.0",
                "status": "FAIL",
            }
        )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    filtered, metadata = runner.filter_interrupted_performance_rows(
        rows,
        output="RUNNING: interrupted.mtx | dtype=float32",
        returncode=-15,
        timed_out=False,
    )
    summary = runner.summarize_performance_csv(csv_path, rows=filtered, **metadata)

    assert summary["raw_row_count"] == 3
    assert summary["row_count"] == 2
    assert summary["excluded_row_count"] == 1
    assert summary["excluded_matrix_keys"] == ["interrupted.mtx"]
    assert summary["speedup_row_count"] == 1
    assert summary["speedup"] == 2.0


def test_parse_op_benchmark_args_keeps_arguments_scoped_to_each_operator():
    parsed = runner.parse_op_benchmark_args(
        ["spmv_bsr=--resume", "spmv_bsr=--dtypes float64,complex64"]
    )

    assert parsed == {"spmv_bsr": ["--resume", "--dtypes", "float64,complex64"]}


@pytest.mark.parametrize("value", ["spmv_bsr", "=--resume", "spmv_bsr="])
def test_parse_op_benchmark_args_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="op-benchmark-args"):
        runner.parse_op_benchmark_args([value])


def test_html_speedups_use_the_variant_data_and_have_an_fp64_column():
    # A delivery-variant row carries its own `data` next to the operator-wide
    # `records`/`speedup`. The HTML used to miss the short dtype keys, fall back
    # to the operator's records, and print the operator's speedup in the first
    # column -- so f32 and f64 rows showed identical, mixed numbers.
    assert "fp64" in [display for display, _ in runner.HTML_SPEEDUP_DTYPES]
    performance = {
        "speedup": 13.0,
        "data": {"fp64": {"speedup": 2.0, "details": {}}},
        "records": [
            {"value_dtype": "float32", "triton_speedup_vs_pytorch": "9.0"},
            {"value_dtype": "complex64", "triton_speedup_vs_pytorch": "7.0"},
        ],
    }
    overall, by_dtype = runner._performance_speedups_for_html(performance)
    assert by_dtype == {"fp64": 2.0}
    assert overall == 2.0


def test_operator_speedup_prefers_the_vendor_column(tmp_path):
    csv_path = tmp_path / "performance.csv"
    csv_path.write_text(
        "matrix,value_dtype,triton_ms,cusparse_ms,pytorch_ms,"
        "triton_speedup_vs_cusparse,triton_speedup_vs_pytorch,status\n"
        "a.mtx,float32,1.0,2.0,9.0,2.0,9.0,PASS\n",
        encoding="utf-8",
    )
    summary = runner.summarize_performance_csv(csv_path)
    assert summary["speedup"] == 2.0


def test_delivery_projection_keeps_only_int32_non_rows_with_a_speedup():
    # A variant named ..._int_non used to average int64 and trans/conj rows in,
    # and rows without a speedup counted as 0 (spsm f32 on CUDA: 8.9x read 3.8x).
    header = "matrix,value_dtype,index_dtype,op,triton_ms,cusparse_ms,"
    header += "triton_speedup_vs_cusparse,status"
    lines = [
        "a.mtx,float32,int32,non,1.0,2.0,2.0,PASS",
        "b.mtx,float32,int32,non,1.0,4.0,4.0,PASS",
        "c.mtx,float32,int32,non,1.0,,,PASS",
        "a.mtx,float32,int64,non,1.0,50.0,50.0,PASS",
        "a.mtx,float32,int32,trans,1.0,90.0,90.0,PASS",
    ]
    rows = list(csv.DictReader([header, *lines]))
    phase = {"records": rows, "data": runner._flaggems_perf_data(rows)}
    projected = runner._delivery_performance_phase(phase, "f32")
    assert projected["delivery_row_count"] == 3
    assert projected["non_delivery_row_count"] == 2
    assert projected["data"]["fp32"]["speedup"] == 3.0


def test_delivery_benchmark_args_name_flags_the_scripts_accept():
    delivery_parents = {
        variant["operator"]
        for variant in load_delivery_variants(ROOT / "conf" / "operators.yaml")
    }
    assert set(runner.DELIVERY_BENCHMARK_ARGS) <= delivery_parents
    for op, args in runner.DELIVERY_BENCHMARK_ARGS.items():
        script = ROOT / runner.OP_TEST_CONFIGS[op].performance_cmd[0]
        source = script.read_text(encoding="utf-8")
        for flag in (arg for arg in args if arg.startswith("--")):
            assert f'"{flag}"' in source, f"{script.name} has no {flag}"


@pytest.mark.parametrize(
    ("backend", "ascend_mask", "child_device"),
    [("ascend", "6", 0), ("xpu", None, 0), ("cuda", None, 6), ("", None, 6)],
)
def test_child_device_matches_the_visible_device_mask(
    monkeypatch, backend, ascend_mask, child_device
):
    # torch_npu ignores CUDA_VISIBLE_DEVICES, so Ascend children landed on NPU 0
    # whatever --gpus said; masked children then see their one card as device 0.
    monkeypatch.setenv("FLAGSPARSE_BACKEND", backend)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    env = runner._base_env(ROOT, 6)
    assert env["CUDA_VISIBLE_DEVICES"] == "6"
    assert env.get("ASCEND_RT_VISIBLE_DEVICES") == ascend_mask
    assert runner._subprocess_device_id(6) == child_device


def test_delivery_projection_keeps_a_timeout_distinct_from_not_run():
    # Ascend, 2026-09-17: an operator killed by --timeout had no dtype rows, and
    # every one of its variants then read NotFound -- the same as never running.
    timed_out = runner._delivery_performance_phase({"status": "TIMEOUT"}, "f32")
    assert timed_out["status"] == "TIMEOUT"
    assert timed_out["data"] == {}
    empty = runner._delivery_performance_phase({"status": "PASS"}, "f32")
    assert empty["status"] == "NOT_CONFIGURED"


def test_ascend_benchmark_commands_pass_the_matrix_input_the_script_accepts():
    # The runner resolved --benchmark-input but never handed it to
    # benchmark_ascend.py, so "30 real matrices" runs measured the synthetic case.
    script = (ROOT / "benchmark" / "benchmark_ascend.py").read_text(encoding="utf-8")
    assert '"--input"' in script
    for op in runner.ASCEND_BASELINE_OPS:
        template = runner.ASCEND_PERFORMANCE_COMMANDS[op]
        assert template[template.index("--input") + 1] == "{input}"
        assert "bfloat16" not in template[template.index("--dtypes") + 1]
