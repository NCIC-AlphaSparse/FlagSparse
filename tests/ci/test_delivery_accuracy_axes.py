# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""A delivery accuracy row is sliced on dtype AND on the delivery axes.

A row such as ``spmv_csr_f32_int_non`` is int32-index and non-transposed. Slicing
on dtype alone dragged the int64/trans/conj cases and the external-matrix suite
(skipped unless FLAGSPARSE_SPMV_CSR_MTX_DIR is set, which the runner never sets)
into every row, and any skip turns the whole row into ``Skipped`` -- spmv_csr could
not report Passed.
"""

import json

import run_flagsparse_pytest as runner

SUITE = "tests/pytest/test_spmv_csr_accuracy.py"


def _case(name, result="passed", reason="", **params):
    item = {"result": result, "params": params}
    if reason:
        item["reason"] = reason
    return f"{SUITE}::{name}[{'-'.join(map(str, params.values()))}]", item


def _project(tmp_path, cases, dtype="f32"):
    artifact = tmp_path / "accuracy_result.json"
    artifact.write_text(json.dumps(dict(cases)), encoding="utf-8")
    return runner._delivery_accuracy_phase({"result_path": str(artifact)}, dtype)


def _surface(**over):
    base = {"dtype": "torch.float32", "op": "non", "col_dtype": "torch.int32"}
    return _case("test_spmv_csr_full_dtype_op_surface", **{**base, **over})


def test_off_axis_index_and_op_cases_are_left_out_of_the_row(tmp_path):
    cases = [
        _surface(),
        _surface(col_dtype="torch.int64"),
        _surface(op="trans"),
        _surface(op="conj"),
    ]
    row = _project(tmp_path, cases)
    assert (row["passed"], row["total"], row["off_axis_excluded"]) == (1, 1, 3)
    assert row["status"] == "Passed"


def test_the_external_matrix_suite_no_longer_turns_the_row_into_skipped(tmp_path):
    external = [
        _case(
            "test_spmv_csr_external_matrix_regressions",
            result="skipped",
            reason="Skipped: external matrix regression directory not configured",
            matrix_name=f"m{i}.mtx",
            op="non",
            dtype="torch.float32",
            alg="legacy_segbin",
        )
        for i in range(5)
    ]
    row = _project(tmp_path, [_surface(), *external])
    assert row["status"] == "Passed"
    assert (row["skipped"], row["off_axis_excluded"]) == (0, 5)


def test_a_failure_on_the_delivery_axes_still_fails_the_row(tmp_path):
    bad = _surface(alg="row_tile")
    bad[1]["result"], bad[1]["reason"] = "failed", "AssertionError"
    row = _project(tmp_path, [_surface(), bad])
    assert row["status"] == "Failed" and row["failed"] == 1


def test_a_skip_on_the_delivery_axes_still_skips_the_row(tmp_path):
    skipped = _surface(alg="x")
    skipped[1]["result"], skipped[1]["reason"] = "skipped", "Skipped: needs a GPU"
    row = _project(tmp_path, [_surface(), skipped])
    assert row["status"] == "Skipped"


def test_a_case_that_does_not_record_an_axis_is_not_constrained_by_it(tmp_path):
    # sddmm/spgemm record only indptr_dtype; spsm records neither axis.
    cases = [_case("test_sddmm", dtype="torch.float32", indptr_dtype="torch.int64")]
    row = _project(tmp_path, cases)
    assert (row["passed"], row["off_axis_excluded"]) == (1, 0)


def test_spsv_trans_modes_are_off_axis_but_non_is_kept(tmp_path):
    cases = [
        _case("test_spsv", dtype="torch.float32", op_mode="TRANS"),
        _case("test_spsv", dtype="torch.float32", op_mode="CONJ"),
        _case("test_spsv", dtype="torch.float32", op_mode="NON"),
        _case("test_spsv", dtype="torch.float32", op_mode="N"),
    ]
    row = _project(tmp_path, cases)
    assert (row["passed"], row["off_axis_excluded"]) == (2, 2)


def test_a_slice_made_only_of_off_axis_cases_is_not_emptied(tmp_path):
    # NotFound would say "did not run"; these ran, just off-axis.
    cases = [_surface(col_dtype="torch.int64"), _surface(op="trans")]
    row = _project(tmp_path, cases)
    assert row["status"] == "Passed" and row["passed"] == 2
    assert row["off_axis_excluded"] == 0


def test_other_dtypes_are_still_sliced_out(tmp_path):
    cases = [_surface(), _surface(dtype="torch.float64")]
    assert _project(tmp_path, cases, "f32")["passed"] == 1
    assert _project(tmp_path, cases, "f64")["passed"] == 1
