# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The baseline a row is judged against: CANN, then PyTorch, then scaled H800.

A real CANN operator time outranks the row's own PyTorch column and the scaled
H800 stand-in. It is an ordered-table entry rather than a backend branch, so a
row that has no usable CANN measurement keeps falling through the same chain it
always did.
"""

import pytest

runner = pytest.importorskip("run_flagsparse_pytest")


def _row(**extra):
    row = {
        "matrix": "auto",
        "dtype": "float32",
        "status": "PASS",
        "pytorch_ms": "4.0",
        "triton_ms": "2.0",
        "triton_speedup_vs_pytorch": "2.0",
    }
    row.update(extra)
    return row


def test_cann_wins_over_pytorch():
    row = _row(
        baseline_kind="cann",
        cann_status="PASS",
        cann_ms="100.0",
        cann_flagsparse_ms="2.0",
        speedup_vs_cann="50.0",
    )

    assert runner._performance_schema(row)[0] == "speedup_vs_cann"


def test_pytorch_is_used_when_there_is_no_cann_column():
    assert runner._performance_schema(_row())[0] == "triton_speedup_vs_pytorch"


def test_a_failed_cann_run_falls_back_to_pytorch():
    """Not "no speedup": the row still has a PyTorch baseline to be judged on."""
    row = _row(
        baseline_kind="cann",
        cann_status="FAIL",
        cann_reason="CANN post-timing CPU golden validation failed",
        cann_ms="100.0",
        cann_flagsparse_ms="2.0",
        speedup_vs_cann="50.0",
    )

    assert runner._performance_schema(row)[0] == "triton_speedup_vs_pytorch"
    assert runner._performance_row_has_complete_speedup(row)


def test_scaled_h800_is_last_resort_only():
    row = {
        "matrix": "auto",
        "dtype": "float32",
        "status": "PASS",
        "h800_scaled_ms": "1.0",
        "h800_scaled_fs_ms": "10.0",
        "speedup_vs_h800_scaled": "0.1",
    }

    assert runner._performance_schema(row)[0] == "speedup_vs_h800_scaled"

    row.update(
        baseline_kind="cann",
        cann_status="PASS",
        cann_ms="100.0",
        cann_flagsparse_ms="10.0",
        speedup_vs_cann="10.0",
    )

    assert runner._performance_schema(row)[0] == "speedup_vs_cann"


def test_cann_columns_do_not_affect_a_cuda_row():
    """Other backends never write baseline_kind, so the CANN tier stays off."""
    row = _row(cusparse_ms="3.0", triton_speedup_vs_cusparse="1.5")

    assert runner._performance_schema(row)[0] == "triton_speedup_vs_cusparse"
