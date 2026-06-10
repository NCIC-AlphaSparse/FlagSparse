"""Checks for the unified FlagSparse pytest runner result format."""

import csv
import json

import run_flagsparse_pytest as runner


def test_runner_writes_flaggems_style_summary(tmp_path):
    results = [
        {
            "operator": "gather",
            "gpu": 0,
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
                "log_path": "gather/accuracy.log",
                "data_file": "gather/accuracy_result.json",
                "passed": 3,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "xfailed": 0,
                "xpassed": 0,
                "total": 3,
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
                "log_path": "gather/performance.log",
                "data_path": "gather/performance.csv",
                "data_file": "gather/performance_result.json",
                "row_count": 0,
            },
        }
    ]
    env = {"python": {"version": "3.test"}}

    runner.write_summary(results, tmp_path, env)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"timestamp", "env", "result"}
    assert summary["env"] == env
    gather = summary["result"]["gather"]
    assert set(gather) == {"customized", "accuracy", "performance"}
    assert gather["customized"] is False

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
    assert accuracy["status"] == "Passed"
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
    }
    assert performance["status"] == "Skipped"
    assert performance["data_file"] == "gather/performance_result.json"

    compat_summary = json.loads(
        (tmp_path / "summary_flat.json").read_text(encoding="utf-8")
    )
    assert {"timestamp", "env", "result", "totals", "results"} <= set(compat_summary)
    assert compat_summary["totals"]["by_phase"]["accuracy"]["PASS"] == 1

    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["operator"] == "gather"
    assert rows[0]["phase"] == "accuracy"
    assert rows[0]["duration"] == "1.25"
    assert rows[0]["exit_code"] == "0"


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
        "log_path": "gather/accuracy.log",
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
