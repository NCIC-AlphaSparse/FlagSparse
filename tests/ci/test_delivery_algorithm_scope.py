# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""A delivery speedup is the path a caller gets, not a mean over algorithms.

spmv_csr's benchmark ran with --alg compare, which writes one row per registered
algorithm, and the delivery projection averaged them all: BW1000 reported
spmv_csr_f32_int_non at 0.207x when its production route (auto -> row_tile)
measured 0.738x on the same CSV, and CUDA reported 0.379x for auto ->
legacy_segbin's 0.861x. One failing row (row_tile fp32 on auto.mtx) also
disappeared into that mean and the variant still read Passed.
"""

import run_flagsparse_pytest as runner


def _row(alg, matrix, speedup, status="PASS", dtype="float32"):
    return {
        "matrix": matrix,
        "dtype": dtype,
        "index_dtype": "int32",
        "op": "non",
        "alg": alg,
        "alg_requested": alg,
        "alg_resolved": alg if alg != "auto" else "row_tile",
        "ms": "1.0",
        "vendor_ms": str(speedup),
        "speedup_vs_vendor": str(speedup),
        "status": status,
    }


def _phase(rows):
    return {"status": "PASS", "records": rows}


def _project(rows, dtype="f32"):
    return runner._q4_performance_phase(
        _phase(rows), f"spmv_csr_{dtype}_int_non", dtype
    )


def test_the_spmv_csr_command_measures_the_auto_route():
    cmd = runner.PERFORMANCE_COMMANDS["spmv_csr"]
    assert cmd[cmd.index("--alg") + 1] == "auto"


def test_rows_for_other_algorithms_are_not_delivery_rows():
    auto_rows = [_row("auto", "a", 0.8), _row("auto", "b", 0.7)]
    swept = [_row(alg, m, 0.1) for alg in ("row_vector", "legacy_rowpar") for m in "ab"]
    result = _project(auto_rows + swept)
    assert result["status"] == "PASS"
    assert result["delivery_row_count"] == 2
    assert {r["alg_requested"] for r in result["records"]} == {"auto"}


def test_a_compare_only_run_says_why_it_has_no_delivery_number():
    swept = [_row(alg, "a", 0.5) for alg in ("row_tile", "legacy_segbin")]
    result = _project(swept)
    assert result["status"] == "NOT_CONFIGURED"
    assert "--alg auto" in result["reason"]


def test_rows_without_an_algorithm_column_are_unaffected():
    row = _row("auto", "a", 1.5)
    for key in ("alg", "alg_requested", "alg_resolved"):
        del row[key]
    assert runner._is_delivery_performance_row(row)


def test_one_failed_row_fails_the_variant():
    rows = [_row("auto", "a", 0.8), _row("auto", "auto.mtx", 0.0, status="FAIL")]
    result = _project(rows)
    assert result["status"] == "FAIL"
    assert result["failed_rows"] == ["auto.mtx"]


def test_a_failed_row_of_another_dtype_does_not_fail_this_one():
    rows = [
        _row("auto", "a", 0.8),
        _row("auto", "a", 0.0, status="FAIL", dtype="float64"),
    ]
    assert _project(rows)["status"] == "PASS"
