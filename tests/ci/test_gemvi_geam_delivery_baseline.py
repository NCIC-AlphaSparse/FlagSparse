import ast
import math
import types
from pathlib import Path

import pytest

from tests.ci.test_ascend_cann_baseline_policy import _row, _runner


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("ascend", [False, True])
@pytest.mark.parametrize("kind", ["geam", "spgemm"])
def test_geam_wiring_is_ascend_only(ascend, kind):
    tree = ast.parse((ROOT / "tests/q4_variant_bench.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sparse_result"
    )
    namespace = dict(
        _is_ascend_runtime=lambda: ascend, v=types.SimpleNamespace(kind=kind)
    )
    assignment = function.body[0]
    assert isinstance(assignment, ast.Assign)
    actual = eval(
        compile(ast.Expression(assignment.value), "<geam backend>", "eval"), namespace
    )
    assert actual == (ascend and kind == "geam")


@pytest.mark.parametrize("ascend", [False, True])
def test_gemvi_wiring_is_ascend_only(ascend):
    tree = ast.parse((ROOT / "tests/test_gemvi.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_case"
    )
    branch = next(node for node in function.body if isinstance(node, ast.If))
    marked = []
    namespace = dict(
        _is_ascend_runtime=lambda: ascend,
        row={},
        cann_baseline=types.SimpleNamespace(mark_row=lambda *args: marked.append(args)),
    )
    exec(
        compile(ast.Module(body=[branch], type_ignores=[]), "<gemvi backend>", "exec"),
        namespace,
    )
    assert bool(marked) == ascend


@pytest.mark.parametrize("biv", [False, True])
@pytest.mark.parametrize("status", ["PASS", "FAIL", "UNAVAILABLE", ""])
def test_required_cann_never_selects_other_baselines(monkeypatch, biv, status):
    runner = _runner()
    monkeypatch.setattr(runner, "_BIV_BASELINE_POLICY", biv)
    row = _row()
    row.update(baseline_required="cann", cann_status=status, biv_speedup="99")
    assert runner._performance_schema(row)[0] == "speedup_vs_cann"
    assert runner._performance_row_has_complete_speedup(row) == (status == "PASS")
    if status == "PASS":
        assert runner._flaggems_perf_data([row])["float32"]["baseline"] == "cann"


@pytest.mark.parametrize("field", ["cann_ms", "cann_flagsparse_ms", "speedup_vs_cann"])
def test_required_cann_needs_all_measurements(field):
    runner = _runner()
    row = _row()
    row.update(baseline_required="cann")
    row.pop(field)
    assert not runner._performance_row_has_complete_speedup(row)


def test_missing_required_cann_disqualifies_partial_variant():
    runner = _runner()
    passing = _row()
    passing.update(baseline_required="cann", variant="gemvi_f32_int_non")
    failed = dict(passing, matrix="auto.mtx", cann_status="FAIL")
    result = runner._q4_performance_phase(
        dict(status="PASS", records=[passing, failed]), "gemvi_f32_int_non"
    )
    assert result["status"] == "FAIL"
    assert result["required_baseline_missing"] == ["auto.mtx"]
    assert result["speedup"] == ""
    assert result["data"] == {}


@pytest.mark.parametrize("biv", [False, True])
@pytest.mark.parametrize("status", ["PASS", "FAIL"])
def test_required_cann_is_not_filled_by_scaled_or_biv(
    tmp_path, monkeypatch, biv, status
):
    runner = _runner()
    monkeypatch.setattr(runner, "_BIV_BASELINE_POLICY", biv)
    monkeypatch.setattr(
        runner,
        "_SCALED_BASELINE",
        dict(reference="unused", card="ascend-910b", reference_peaks="h800-sxm"),
    )
    row = _row()
    row.update(baseline_required="cann", cann_status=status)

    def fill(op, rows, fields, reference_run, card, **kwargs):
        assert not kwargs["needs_baseline"](rows[0])
        assert not rows[0].get("biv_speedup")
        return rows, {"filled": 0}

    monkeypatch.setattr(runner._baseline_bound, "scaled_baseline_rows", fill)
    rows, info = runner._apply_scaled_baseline("gemvi", tmp_path / "perf.csv", [row])
    assert info["filled"] == 0
    assert rows[0]["cann_status"] == status


@pytest.mark.parametrize(
    "before,after", [(0, 0), (math.nan, 0), (0, math.inf), (1, 0), (0, 1)]
)
@pytest.mark.parametrize(
    "elapsed,ours", [(2, 1), (math.inf, 1), (2, math.nan), (0, 1), (2, -1)]
)
def test_measure_checks_accuracy_timings_and_cleanup(before, after, elapsed, ours):
    tree = ast.parse((ROOT / "tests/ascend_gemvi_geam_baseline.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "measure"
    )
    closed, timed = [], []
    errors = iter((before, after))
    plan = types.SimpleNamespace(
        metadata={}, prepare_ms=0.5, run=lambda: None, close=lambda: closed.append(True)
    )

    def benchmark(*args):
        timed.append(True)
        return None, elapsed

    dtype = object()
    namespace = dict(
        math=math,
        torch=types.SimpleNamespace(float32=dtype, complex64=object()),
        _benchmark_cuda_op=benchmark,
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<measure>", "exec"),
        namespace,
    )
    row = dict(cann_ms=99, speedup_vs_cann=99)
    namespace["measure"](
        row, lambda: plan, lambda output: next(errors), dtype, ours, 0, 1
    )
    valid = before == after == 0 and elapsed == 2 and ours == 1
    assert row["cann_status"] == ("PASS" if valid else "FAIL")
    assert ("cann_ms" in row) == valid
    assert ("speedup_vs_cann" in row) == valid
    assert closed == [True]
    assert bool(timed) == (before == 0)
