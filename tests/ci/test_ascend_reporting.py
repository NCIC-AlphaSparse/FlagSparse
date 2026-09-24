# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend measures a speedup; the report has to carry it.

Both defects here produced a delivery table that understated a working backend.
The performance CSV held a complete measurement for all five baseline operators
and the table printed `-`, because the runner had no schema for the plain
`speedup` column that benchmark_ascend.py writes and fell through to a pair of
latency columns that CSV does not have. The probe reported ERROR for spmv_coo
and spmm_coo because it built its fp64 reference with a cast the NPU cannot do.
"""

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "_runner_for_ascend_reporting", ROOT / "run_flagsparse_pytest.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ascend_row(dtype, matrix, triton_ms, pytorch_ms, status="PASS"):
    """A row with the columns benchmark_ascend.py declares, in its spelling."""
    return {
        "dtype": dtype,
        "matrix": matrix,
        "shape": f"spmm_csr:{matrix}",
        "triton_ms": str(triton_ms),
        "pytorch_ms": str(pytorch_ms),
        "speedup": str(pytorch_ms / triton_ms),
        "max_abs_err": "1e-06",
        "status": status,
    }


def test_the_ascend_csv_columns_are_a_schema_of_their_own():
    runner = _runner()
    row = _ascend_row("float32", "auto.mtx", 1.0, 0.99774)
    assert runner._performance_schema(row) == (
        "speedup",
        "pytorch_ms",
        "triton_ms",
    )
    assert runner._performance_row_has_complete_speedup(row)


def test_an_ascend_dtype_aggregates_to_the_mean_of_its_matrices():
    """The defect: every row was dropped, so this stayed 0 and printed as `-`."""
    runner = _runner()
    rows = [
        _ascend_row("float32", "auto.mtx", 1.0, 2.0),
        _ascend_row("float32", "cfd2.mtx", 1.0, 1.0),
        _ascend_row("float64", "auto.mtx", 2.0, 3.0),
    ]
    data = runner._flaggems_perf_data(rows)
    assert data["float32"]["speedup"] == 1.5
    assert data["float64"]["speedup"] == 1.5


def test_a_missing_baseline_is_still_not_a_speedup():
    """The plain `speedup` name must not become a way past the completeness check."""
    runner = _runner()
    row = _ascend_row("float32", "auto.mtx", 1.0, 2.0)
    row["pytorch_ms"] = ""
    assert not runner._performance_row_has_complete_speedup(row)
    failed = _ascend_row("float32", "auto.mtx", 1.0, 2.0, status="FAIL")
    assert not runner._performance_row_has_complete_speedup(failed)


def test_the_other_producers_of_a_plain_speedup_are_unchanged():
    """XPU writes the ratio under both names; a latency producer under neither."""
    runner = _runner()
    xpu = {
        "dtype": "float32",
        "shape": "matrix=wave,m=1,n=1",
        "triton_ms": "2.0",
        "pytorch_ms": "1.0",
        "speedup": "0.5",
        "triton_speedup_vs_pytorch": "0.5",
        "status": "PASS",
    }
    assert runner._flaggems_perf_data([xpu])["float32"]["speedup"] == 0.5
    generic = {
        "dtype": "float32",
        "shape": "s",
        "speedup": "2.0",
        "latency_base": "4.0",
        "latency": "2.0",
        "status": "PASS",
    }
    assert runner._performance_schema(generic) == (
        "speedup",
        "latency_base",
        "latency",
    )
    assert runner._flaggems_perf_data([generic])["float32"]["speedup"] == 2.0


def test_the_probe_casts_to_fp64_only_after_leaving_the_device():
    """910B has no DT_DOUBLE: a device-side cast returns fp32 and the matmul
    against a real fp64 operand then raises, which the probe attributes to the
    operator. Asserted on the source because neither order fails on a CUDA box,
    so an executable test here would pass against the defect.
    """
    path = ROOT / "benchmark" / "benchmark_ascend_probe.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    casts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "to"
        and isinstance(node.value, ast.Name)
    ]
    for node in casts:
        # `<name>.to(...)` where <name> is a device tensor built by _dense.
        assert node.value.id != "dense", (
            "benchmark_ascend_probe.py casts `dense` on the device; "
            "move to the host first: dense.cpu().to(torch.float64)"
        )
    source = path.read_text(encoding="utf-8")
    assert "dense.cpu().to(torch.float64)" in source


def test_unbenchmarked_probe_operators_keep_both_delivery_dtypes():
    """Remaining capability probes still cover both COO delivery dtypes."""
    runner = _runner()
    assert runner.PROBE_DTYPE_ARGS == ("--dtypes", "float32,float64")
    for op in ("spmv_coo", "spmm_coo"):
        template = runner.ASCEND_PERFORMANCE_COMMANDS[op]
        assert template[0] == "benchmark/benchmark_ascend.py", op
        assert template[template.index("--dtypes") + 1] == "float32,float64", op
    generic = runner._probe_command("spmv_coo")
    assert generic[generic.index("--dtypes") + 1] == "float32,float64"


def test_the_ascend_pytorch_baselines_use_declared_dtypes_and_measured_commands():
    """All delivery operators with a PyTorch baseline have measured commands."""
    runner = _runner()
    assert runner.ASCEND_PYTORCH_PERFORMANCE_OPS == (
        "gather",
        "scatter",
        "spmv_csr",
        "spmm_csr",
        "sddmm_csr",
        "spmv_coo",
        "spmm_coo",
    )
    for op in runner.ASCEND_PYTORCH_PERFORMANCE_OPS:
        template = runner.ASCEND_PERFORMANCE_COMMANDS[op]
        assert template[0] == "benchmark/benchmark_ascend.py", op
        index = template.index("--dtypes")
        expected = runner.ASCEND_PERFORMANCE_DTYPES[op]
        assert template[index + 1] == expected, op


def test_coo_delivery_performance_uses_pytorch_baselines_not_probe_rows():
    runner = _runner()
    for op in ("spmv_coo", "spmm_coo"):
        command = runner.ASCEND_PERFORMANCE_COMMANDS[op]
        assert command[0] == "benchmark/benchmark_ascend.py"
        assert command[command.index("--dtypes") + 1] == "float32,float64"
    probe_ops = {
        key
        for key, value in runner.ASCEND_PERFORMANCE_COMMANDS.items()
        if value[0] == "benchmark/benchmark_ascend_probe.py"
    }
    assert not ({"spmv_coo", "spmm_coo"} & probe_ops)


def test_measured_benchmark_row_failures_reach_the_phase_status():
    runner = _runner()
    assert runner._measured_benchmark_status([{"status": "PASS"}]) == "PASS"
    assert runner._measured_benchmark_status([{"status": "FAIL"}]) == "FAIL"
    assert runner._measured_benchmark_status(
        [{"status": "PASS"}, {"status": "FAIL"}]
    ) == "FAIL"
    assert runner._measured_benchmark_status(
        [{"status": "PASS"}, {"status": "SKIP"}]
    ) == "MIXED"


def test_complex_index_capability_skip_is_narrowly_classified():
    source = (ROOT / "benchmark" / "benchmark_ascend.py").read_text(
        encoding="utf-8"
    )
    node = next(
        item
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_ascend_complex_index_capability_reason"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<capability>", "exec"), namespace)
    classify = namespace["_ascend_complex_index_capability_reason"]
    assert classify(RuntimeError("aclnnIndex does not support ComplexDouble"))
    assert classify(RuntimeError("Index kernel not implemented for complex64"))
    assert classify(RuntimeError("complex multiplication is unsupported")) is None


def test_complex_delivery_rows_are_capability_skips_not_missing():
    runner = _runner()
    assert "ascend complex gather/scatter capability unavailable" in (
        runner.EXPECTED_ACCURACY_SKIP_REASONS
    )
    for op in ("gather", "scatter"):
        command = runner.ASCEND_PERFORMANCE_COMMANDS[op]
        assert command[command.index("--dtypes") + 1].endswith(
            "complex64,complex128"
        )
    reason = "Ascend complex gather/scatter capability unavailable: Index unsupported"
    rows = [
        {
            **_ascend_row("complex64", "synthetic", 1.0, 1.0, status="SKIP"),
            "reason": reason,
        }
    ]
    phase = {
        "operator": "gather",
        "phase": "performance",
        "status": "PASS",
        "records": rows,
        "data": runner._flaggems_perf_data(rows),
    }
    projected = runner._delivery_performance_phase(phase, "c32")
    assert projected["status"] == "SKIP"
    assert "Ascend complex gather/scatter capability unavailable" in projected["reason"]
    accuracy = runner.summarize_accuracy_cases(
        {
            "gather[float32]": {"result": "passed"},
            "gather[complex64]": {
                "params": {"dtype": "complex64"},
                "result": "skipped",
                "reason": reason,
            },
        }
    )
    assert accuracy["status"] == "Passed"
    with tempfile.TemporaryDirectory() as temp_dir:
        result_path = Path(temp_dir) / "accuracy.json"
        result_path.write_text(
            json.dumps(
                {
                    "gather[complex64]": {
                        "params": {"dtype": "complex64"},
                        "result": "skipped",
                        "reason": reason,
                    }
                }
            ),
            encoding="utf-8",
        )
        complex_only = runner._delivery_accuracy_phase(
            {"result_path": str(result_path), "status": "SKIP"}, "c32"
        )
    assert complex_only["status"] == "Skipped"
    assert complex_only["skipped"] == 1


def test_the_probe_validates_its_dtype_list_before_importing_torch():
    """A typo must fail fast and name the allowed values, not probe nothing."""
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmark" / "benchmark_ascend_probe.py"),
            "--op",
            "spmv_coo",
            "--dtypes",
            "float16",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert out.returncode == 2
    assert "--dtypes must be a comma-separated subset of" in out.stderr
    assert "float32" in out.stderr and "float64" in out.stderr


def test_the_old_singular_dtype_spelling_still_works():
    """`--dtype` was the flag's only name; existing commands must keep running."""
    source = (ROOT / "benchmark" / "benchmark_ascend_probe.py").read_text(
        encoding="utf-8"
    )
    assert '"--dtypes", "--dtype", dest="dtypes"' in source


def _ascend_row_status():
    """Compile just row_status() out of the benchmark, without importing it.

    benchmark_ascend.py imports numpy and scipy at module level and tests/ci runs
    on an environment with neither, so importing the module skips this test in
    the one place it has to run: the value of a producer/consumer contract test is
    that it holds where the producer cannot execute. row_status() is a pure
    function of its argument, so its own definition is enough.
    """
    source = (ROOT / "benchmark" / "benchmark_ascend.py").read_text(encoding="utf-8")
    definition = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "row_status"
    )
    module = ast.Module(body=[definition], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, "<benchmark_ascend.row_status>", "exec"), namespace)
    return namespace["row_status"]


def test_the_status_the_benchmark_writes_is_one_the_runner_accepts():
    """The contract between the two files, which is where this broke.

    benchmark_ascend.py used to write the whole descriptive string as `status`:
    on success "PyTorch-NPU: PASS", because the FlagSparse branch appends nothing
    when it works. The runner compares that cell against a fixed vocabulary, so
    every row was ineligible and each aggregate stayed 0, printed as `-`.
    """
    runner = _runner()
    row_status = _ascend_row_status()
    measured = row_status({"mean_ms": 1.0})
    assert measured == "PASS"
    assert runner._performance_row_status_is_usable({"status": measured})
    failed = row_status(None)
    assert failed == "FAIL"
    assert not runner._performance_row_status_is_usable({"status": failed})


def test_a_measured_ascend_row_reaches_the_dtype_aggregate():
    runner = _runner()
    row_status = _ascend_row_status()
    row = _ascend_row("float32", "auto.mtx", 1.0, 0.99774)
    row["status"] = row_status({"mean_ms": 1.0})
    assert runner._flaggems_perf_data([row])["float32"]["speedup"] == 0.99774
    row["status"] = row_status(None)
    assert not runner._performance_row_has_complete_speedup(row)


def test_the_prose_moved_to_a_reason_column_and_is_still_written():
    """Losing the diagnosis would trade one blind spot for another."""
    source = (ROOT / "benchmark" / "benchmark_ascend.py").read_text(encoding="utf-8")
    assert '"status": row_state,' in source
    assert '"SKIP"' in source
    assert '"reason": "; ".join(status_parts) if status_parts else "unknown"' in source
    assert '"status", "reason"' in source
