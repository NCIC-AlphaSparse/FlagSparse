"""The baseline chain for a native CANN row: CANN, then PyTorch, then scaled H800.

A usable CANN measurement outranks both of the others. When the CANN attempt
failed or was never made, the row falls back down the same chain instead of
losing its speedup -- the delivered policy is an ordered preference, not an
exclusive one, so suppressing the CANN tier must not suppress the row.
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "_runner_cann_policy", ROOT / "run_flagsparse_pytest.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row():
    return dict(
        dtype="float32",
        matrix="ASIC_680ks.mtx",
        status="PASS",
        baseline_kind="cann",
        cann_status="PASS",
        cann_ms="16",
        cann_flagsparse_ms="4",
        speedup_vs_cann="4",
        pytorch_ms="8",
        triton_ms="4",
        triton_speedup_vs_pytorch="2",
        h800_scaled_ms="0.1",
        h800_scaled_fs_ms="4",
        speedup_vs_h800_scaled="0.025",
    )


def test_native_cann_wins_over_pytorch_and_stale_scaled_columns():
    runner = _runner()
    row = _row()
    assert runner._performance_schema(row) == (
        "speedup_vs_cann",
        "cann_ms",
        "cann_flagsparse_ms",
    )
    assert runner._performance_row_has_complete_speedup(row)
    data = runner._flaggems_perf_data([row])["float32"]
    assert data["speedup"] == 4
    assert data["baseline"] == "cann"
    assert next(iter(data["details"].values()))["base"] == 16


def test_cann_target_uses_matrix_mean_not_worst_matrix():
    runner = _runner()
    rows = []
    for matrix, speedup in (("ASIC_680ks.mtx", 0.2), ("auto.mtx", 1.8)):
        row = _row()
        row.update(
            matrix=matrix, cann_ms=str(4 * speedup), speedup_vs_cann=str(speedup)
        )
        rows.append(row)
    data = runner._flaggems_perf_data(rows)["float32"]
    assert data["speedup"] == pytest.approx(1.0)
    assert data["speedup"] > 0.8
    assert min(row["speedup"] for row in data["details"].values()) < 0.8


@pytest.mark.parametrize(
    "field,value",
    [
        ("cann_status", "FAIL"),
        ("cann_status", "UNAVAILABLE"),
        ("cann_ms", ""),
        ("speedup_vs_cann", ""),
    ],
)
def test_failed_or_missing_cann_falls_back_to_pytorch(field, value):
    """Second tier. A broken CANN attempt must not cost the row its speedup."""
    runner = _runner()
    row = _row()
    row[field] = value
    assert runner._performance_schema(row)[0] == "triton_speedup_vs_pytorch"
    assert runner._performance_row_has_complete_speedup(row)


@pytest.mark.parametrize("field,value", [("cann_status", "FAIL"), ("cann_ms", "")])
def test_scaled_h800_is_the_third_tier(field, value):
    """With no usable CANN and no PyTorch column, the scaled H800 time is used."""
    runner = _runner()
    row = _row()
    row[field] = value
    for key in ("pytorch_ms", "triton_speedup_vs_pytorch"):
        row.pop(key)
    assert runner._performance_schema(row)[0] == "speedup_vs_h800_scaled"
    assert runner._performance_row_has_complete_speedup(row)


@pytest.mark.parametrize("field,value", [("cann_status", "FAIL"), ("cann_ms", "")])
def test_an_unusable_cann_row_with_no_other_baseline_has_no_speedup(field, value):
    runner = _runner()
    row = _row()
    row[field] = value
    for key in (
        "pytorch_ms",
        "triton_speedup_vs_pytorch",
        "h800_scaled_ms",
        "h800_scaled_fs_ms",
        "speedup_vs_h800_scaled",
    ):
        row.pop(key)
    assert not runner._performance_row_has_complete_speedup(row)


def test_scaled_filler_skips_a_row_that_still_has_a_usable_baseline(
    tmp_path, monkeypatch
):
    """The filler is the last tier, so it only sees rows with nothing left.

    A failed CANN attempt still has this row's PyTorch column, so the filler
    must leave it alone -- it is not asked to stand in for a CANN number.
    """
    runner = _runner()
    row = _row()
    row.update(cann_status="FAIL", cann_ms="", speedup_vs_cann="")
    monkeypatch.setattr(
        runner,
        "_SCALED_BASELINE",
        dict(reference="unused", card="ascend-910b", reference_peaks="h800-sxm"),
    )
    monkeypatch.setattr(runner, "_BIV_BASELINE_POLICY", False)

    def fill(operator, rows, fields, reference_run, card, **kwargs):
        assert not kwargs["needs_baseline"](rows[0])
        return rows, {"filled": 0}

    monkeypatch.setattr(runner._baseline_bound, "scaled_baseline_rows", fill)
    result, info = runner._apply_scaled_baseline(
        "spmv_sell", tmp_path / "performance.csv", [row]
    )
    assert result == [row]
    assert info["filled"] == 0


def test_native_adapter_has_no_host_copy_or_sparse_algorithm_in_run():
    tree = ast.parse((ROOT / "tests/ascend_sparse_baseline.py").read_text())
    plan = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SparsePlan"
    )
    run = next(
        node
        for node in plan.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    names = {
        node.func.attr
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "execute" in names
    assert not names & {"cpu", "to", "clone", "index_add_", "index_select"}
    assert (
        "set_tensor_const" not in (ROOT / "tests/ascend_sparse_baseline.py").read_text()
    )


def test_default_sell_and_spmm_delivery_paths_use_the_native_adapter():
    runner = _runner()
    sell = (ROOT / "tests/test_spmv_sell.py").read_text()
    csc = (ROOT / "tests/q4_variant_bench.py").read_text()
    assert (
        runner.ASCEND_PERFORMANCE_COMMANDS["spmv_sell"][0] == "tests/test_spmv_sell.py"
    )
    assert "cann_baseline.prepare_spmv" in sell
    assert 'v.fmt in ("coo", "csr", "csc")' in csc
    assert "baseline.prepare_spmm" in csc
    for operator in ("spmm_coo", "spmm_csr", "spmm_csc"):
        assert "--q4-variants" in runner.ASCEND_PERFORMANCE_COMMANDS[operator]
    assert runner.ASCEND_PERFORMANCE_DTYPES["spmm_csr"] == "float32,float64"
    assert "spmm_csr_f16_int_non_non_row" in runner.ASCEND_Q4_VARIANTS["spmm_csr"]
    primary = (ROOT / "benchmark/benchmark_ascend.py").read_text()
    assert 'cann_operands=("csr", (data, indices, indptr), b)' in primary
    assert 'cann_operands=("coo", (coo_data, coo_row, coo_col), b)' in primary


@pytest.mark.parametrize("fmt", ["coo", "csr", "csc", "bell"])
@pytest.mark.parametrize("ascend", [False, True])
def test_spmm_native_selection_is_ascend_and_format_specific(fmt, ascend):
    tree = ast.parse((ROOT / "tests/q4_variant_bench.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_spmm"
    )
    assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "baseline"
            for target in node.targets
        )
    )
    native, other = object(), object()
    namespace = dict(
        v=types.SimpleNamespace(fmt=fmt),
        _is_ascend_runtime=lambda: ascend,
        cann_baseline=native,
        cusparse_baseline=other,
    )
    actual = eval(
        compile(ast.Expression(assignment.value), "<SpMM baseline selection>", "eval"),
        namespace,
    )
    assert actual is (native if ascend and fmt in ("coo", "csr", "csc") else other)


def test_primary_spmm_golden_uses_quantized_values_and_wide_accumulation():
    # Scoped to this test, not the module: the other checks here are pure AST and
    # must keep running on the CPU-only CI runner, which has neither package.
    np = pytest.importorskip("numpy")
    sparse = pytest.importorskip("scipy.sparse")

    tree = ast.parse((ROOT / "benchmark/benchmark_ascend.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_scipy_spmm"
    )
    namespace = dict(np=np)
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "<SpMM CPU golden>", "exec"
        ),
        namespace,
    )
    matrix = sparse.csr_matrix(np.array([[1.0003]], dtype=np.float32))
    quantized = matrix.data.astype(np.float16)
    dense = np.array([[1.0003]], dtype=np.float16)
    golden = namespace["_scipy_spmm"](matrix, dense, quantized)
    assert golden.dtype == np.float64
    assert golden.item() == 1.0
    assert matrix.data[0] != 1.0


@pytest.mark.parametrize(
    "before,after,expected",
    [
        (1.0, 1.0, "PASS"),
        (float("nan"), 1.0, "FAIL"),
        (1.0, float("inf"), "FAIL"),
        (2.0, 1.0, "FAIL"),
        (1.0, 2.0, "FAIL"),
    ],
)
def test_primary_spmm_requires_cpu_golden_before_and_after_timing(
    monkeypatch, before, after, expected
):
    np = pytest.importorskip("numpy")  # see the note above; CI has no numpy

    tree = ast.parse((ROOT / "benchmark/benchmark_ascend.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_cann_spmm_fields"
    )
    closed = []
    timed = []
    outputs = iter((np.array([[before]]), np.array([[after]])))
    target = types.SimpleNamespace(dtype="float32")
    plan = types.SimpleNamespace(run=lambda: None, close=lambda: closed.append(True))

    def record_result(row, plan, elapsed_ms, error, flagsparse_ms):
        row.update(
            cann_status="PASS",
            cann_ms=elapsed_ms,
            speedup_vs_cann=elapsed_ms / flagsparse_ms,
        )

    def benchmark(*args):
        timed.append(True)
        return {"mean_ms": 2.0}

    baseline = types.SimpleNamespace(
        mark_row=lambda row: row.update(baseline_kind="cann"),
        prepare_spmm=lambda *args: plan,
        record_result=record_result,
    )
    monkeypatch.setitem(sys.modules, "ascend_sparse_baseline", baseline)
    namespace = dict(
        sys=sys,
        _TESTS=ROOT / "tests",
        np=np,
        _sync=lambda torch: None,
        _scipy_numpy=lambda output: next(outputs),
        _bench=benchmark,
        _outputs_pass_accuracy=lambda actual, golden, dtype: np.array_equal(
            actual, golden
        ),
    )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "<primary CANN baseline>",
            "exec",
        ),
        namespace,
    )
    torch = types.SimpleNamespace(empty=lambda *args, **kwargs: target)
    dense = types.SimpleNamespace(dtype="float32", device="npu")
    row = namespace["_cann_spmm_fields"](
        torch, "csr", (), (1, 1), dense, np.array([[1.0]]), 1.0, 1, 1
    )
    assert row["baseline_kind"] == "cann"
    assert row["cann_status"] == expected
    assert ("cann_ms" in row) == (expected == "PASS")
    assert closed == [True]
    assert bool(timed) == (before == 1.0)


@pytest.mark.parametrize("post_error", [float("nan"), float("inf")])
def test_post_timing_nonfinite_error_invalidates_native_timing(post_error):
    tree = ast.parse((ROOT / "tests/q4_variant_bench.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_time_vendor"
    )
    closed = []

    def record_result(*args):
        raise AssertionError("Invalid native results must never publish a timing")

    namespace = dict(
        _benchmark_cuda_op=lambda *args: (None, 1.0),
        _passes=lambda dtype, error: error <= 1e-4,
        torch=types.SimpleNamespace(float32=object()),
        cann_baseline=types.SimpleNamespace(record_result=record_result),
    )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "<native vendor timing>",
            "exec",
        ),
        namespace,
    )
    plan = types.SimpleNamespace(
        run=lambda: None, check=lambda: post_error, close=lambda: closed.append(True)
    )
    row = dict(
        baseline_kind="cann", out_dtype="float32", vendor_max_error=0.0, ours_ms=1.0
    )
    namespace["_time_vendor"](row, plan, 0, 1)
    assert row["cann_status"] == "FAIL"
    assert "cann_ms" not in row
    assert closed == [True]
