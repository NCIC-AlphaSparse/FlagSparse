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


SPSV_ACCURACY_FILES = (
    "tests/pytest/test_spsv_csr_accuracy.py",
    "tests/pytest/test_spsv_coo_accuracy.py",
)


def test_delivery_spsv_accuracy_is_limited_to_int32_non_non_unit_at_run_time():
    for op in ("spsv_csr", "spsv_coo"):
        assert runner._delivery_accuracy_pytest_args(op, True) == [
            "-k",
            "non_trans and int32 and not unit",
        ]
        assert runner._delivery_accuracy_pytest_args(op, False) == []
    # The other operators are not narrowed at run time.
    for op in ("spmv_csr", "spmm_csr", "gather", "spsm_csr"):
        assert runner._delivery_accuracy_pytest_args(op, True) == []


def test_the_spsv_filter_keeps_lower_and_upper_non_and_drops_unit_and_transpose():
    """`-k` is a substring match, so check it against the real test names.

    A keyword that no longer matches after a rename would select zero tests, and a
    keyword that is too long silently drops the upper-triangular NON tests.
    """
    import re
    from pathlib import Path

    keep_keyword, drop_keyword = "non_trans", "unit"
    for path in SPSV_ACCURACY_FILES:
        source = (Path(runner.__file__).parent / path).read_text(encoding="utf-8")
        names = re.findall(r"^def (test_\w+)\(", source, re.M)
        kept = [n for n in names if keep_keyword in n and drop_keyword not in n]
        assert any("upper" not in n for n in kept), f"{path}: no lower NON test kept"
        assert any("upper" in n for n in kept), f"{path}: upper NON tests are dropped"
        unit = [n for n in names if drop_keyword in n]
        assert unit and not set(unit) & set(kept), f"{path}: unit tests not excluded"
        transposed = [n for n in names if "trans_supported" in n and "non_" not in n]
        assert not set(transposed) & set(kept), f"{path}: transpose tests kept"


def test_run_accuracy_puts_the_filter_before_user_pytest_args(monkeypatch, tmp_path):
    seen = {}

    def fake_run_subprocess(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        return 0, "", "", 0.0, False

    monkeypatch.setattr(runner, "run_subprocess", fake_run_subprocess)
    common = dict(
        project_root=tmp_path,
        op="spsv_coo",
        gpu_id=0,
        marker="spsv_coo",
        mode="quick",
        op_dir=tmp_path,
        extra_pytest_args=["--user-flag"],
        timeout=1,
    )

    runner.run_accuracy(**common, delivery_only=True)
    cmd = seen["cmd"]
    assert cmd[cmd.index("-k") + 1] == "non_trans and int32 and not unit"
    assert cmd.index("-k") < cmd.index("--user-flag")

    runner.run_accuracy(**common, delivery_only=False)
    assert "-k" not in seen["cmd"] and "--user-flag" in seen["cmd"]


EXTERNAL_SKIP = "Skipped: external matrix regression directory not configured"
BUCKET_SKIP = "Skipped: legacy bucket requires int32 column indices"
RETIRED_SKIP = "Skipped: new CSR algorithms have no verified profile for this backend"


def _skipped_case(reason, **params):
    return _case("test_spmv_csr_full_dtype_op_surface", "skipped", reason, **params)


def test_capability_skips_do_not_downgrade_a_run_that_passed(tmp_path):
    cases = [
        _surface(alg="legacy_rowpar"),
        _skipped_case(BUCKET_SKIP, alg="legacy_bucket_vector", op="non"),
        _skipped_case(EXTERNAL_SKIP, alg="x", op="non"),
        _skipped_case(EXTERNAL_SKIP, alg="y", op="non"),
    ]
    summary = runner.summarize_accuracy_cases(dict(cases))
    assert summary["status"] == "Passed"
    # Still counted and still listed: only the status stops being downgraded.
    assert (summary["passed"], summary["skipped"], summary["total"]) == (1, 3, 4)
    assert set(summary["details"]["skipped"]) == {EXTERNAL_SKIP, BUCKET_SKIP}


def test_a_delivery_row_with_only_capability_skips_still_gets_a_passed_row(tmp_path):
    def on_axis_skip(reason, alg):
        return _skipped_case(
            reason, alg=alg, op="non", dtype="torch.float32", col_dtype="torch.int32"
        )

    cases = [
        _surface(alg="legacy_segbin"),
        on_axis_skip(BUCKET_SKIP, "legacy_bucket_vector"),
        on_axis_skip(EXTERNAL_SKIP, "legacy_rowpar"),
    ]
    row = _project(tmp_path, cases)
    assert row["status"] == "Passed"
    assert (row["passed"], row["skipped"], row["failed"]) == (1, 2, 0)


def test_any_other_skip_still_makes_the_run_skipped():
    cases = [
        _surface(),
        _skipped_case("Skipped: needs a GPU", alg="x", op="non"),
        _skipped_case(BUCKET_SKIP, alg="legacy_bucket_vector", op="non"),
    ]
    assert runner.summarize_accuracy_cases(dict(cases))["status"] == "Skipped"


def test_the_retired_no_verified_profile_skip_is_no_longer_exempt():
    """The spmv_csr tests stopped emitting it; if it comes back it must be seen."""
    cases = [_surface(), _skipped_case(RETIRED_SKIP, alg="row_tile", op="non")]
    assert runner.summarize_accuracy_cases(dict(cases))["status"] == "Skipped"
    assert not runner._is_expected_accuracy_skip(RETIRED_SKIP)


def test_only_capability_skips_and_nothing_passed_is_still_skipped():
    cases = [_skipped_case(BUCKET_SKIP, alg="legacy_bucket_vector", op="non")]
    assert runner.summarize_accuracy_cases(dict(cases))["status"] == "Skipped"


def test_a_failure_or_an_issue_skip_still_fails_next_to_capability_skips():
    failed = _surface(alg="row_tile")
    failed[1]["result"], failed[1]["reason"] = "failed", "AssertionError"
    cases = [_surface(), failed, _skipped_case(EXTERNAL_SKIP, alg="x", op="non")]
    assert runner.summarize_accuracy_cases(dict(cases))["status"] == "Failed"

    cases = [_surface(), _skipped_case("Skipped: Issue #12", alg="x", op="non")]
    assert runner.summarize_accuracy_cases(dict(cases))["status"] == "Failed"


def test_the_exempt_reasons_are_the_ones_the_tests_actually_emit():
    """Rewording a skip in the tests would silently stop exempting it."""
    from pathlib import Path

    sources = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (Path(runner.__file__).parent / "tests" / "pytest").glob("*.py")
    )
    for reason in runner.EXPECTED_ACCURACY_SKIP_REASONS:
        assert reason in sources, f"no test emits a skip containing {reason!r}"
        assert runner._is_expected_accuracy_skip(f"Skipped: {reason.upper()}")
