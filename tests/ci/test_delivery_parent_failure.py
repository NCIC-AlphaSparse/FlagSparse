# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""A failed parent benchmark process must not fail variants it did not fail.

On MetaX C550 one legacy ``spmm_csc_base`` trans row (err_vs_pytorch 1.397)
made test_spmm_csc.py exit 1, and all three spmm_csc q4 delivery variants read
Failed although each had 10 PASS rows. The parent's FAIL is set aside only
when another variant's failing row fully explains a plain ``exit 1`` -- never
for a timeout, a signal, a traceback or a variant with missing rows.
"""

import run_flagsparse_pytest as runner

VARIANT = "spmm_csc_f32_int_non_non_row"


def _q4_row(matrix, status="PASS", variant=VARIANT):
    return {
        "variant": variant,
        "matrix": matrix,
        "dtype": "float32",
        "status": status,
        "ms": "1.0",
        "pytorch_ms": "2.0",
        "triton_speedup_vs_pytorch": "2.0",
    }


def _legacy_fail_row():
    return {
        "matrix": "amazon0601.mtx",
        "dtype": "float32",
        "op": "trans",
        "alg": "spmm_csc_base",
        "status": "FAIL",
        "reason": "correctness check failed",
    }


def _phase(rows, *, status="FAIL", returncode=1, traceback=False, matrices=2):
    return {
        "status": status,
        "returncode": returncode,
        "exit_code": returncode,
        "stderr_has_traceback": traceback,
        "input_matrix_count": matrices,
        "records": rows,
    }


def _q4_rows():
    return [_q4_row("a.mtx"), _q4_row("b.mtx"), _legacy_fail_row()]


def _project(phase):
    return runner._q4_performance_phase(phase, VARIANT, "f32")


def test_legacy_row_failure_does_not_fail_complete_q4_variant():
    result = _project(_phase(_q4_rows()))

    assert result["status"] == "PASS"
    assert result["delivery_row_count"] == 2
    assert result["parent_status"] == "FAIL"
    assert result["parent_returncode"] == 1
    assert result["parent_failure_rows"] == [
        "amazon0601.mtx/float32/trans/spmm_csc_base"
    ]


def test_timeout_parent_stays_timeout():
    result = _project(_phase(_q4_rows(), status="TIMEOUT", returncode=-9))

    assert result["status"] == "TIMEOUT"
    assert "parent_status" not in result


def test_signal_exit_parent_stays_failed():
    result = _project(_phase(_q4_rows(), returncode=-11))

    assert result["status"] == "FAIL"
    assert "parent_status" not in result


def test_traceback_parent_stays_failed():
    result = _project(_phase(_q4_rows(), traceback=True))

    assert result["status"] == "FAIL"
    assert "parent_status" not in result


def test_incomplete_q4_variant_keeps_parent_failure():
    result = _project(_phase(_q4_rows(), matrices=3))

    assert result["status"] == "FAIL"
    assert "parent_status" not in result


def test_incomplete_q4_variant_detected_without_recorded_matrix_count():
    rows = _q4_rows() + [
        _q4_row(name, variant="spmm_csc_f16_int_non_non_row")
        for name in ("a.mtx", "b.mtx", "c.mtx")
    ]
    phase = _phase(rows)
    del phase["input_matrix_count"]

    assert _project(phase)["status"] == "FAIL"


def test_exit_one_without_a_failing_row_keeps_parent_failure():
    rows = [_q4_row("a.mtx"), _q4_row("b.mtx")]

    assert _project(_phase(rows))["status"] == "FAIL"


def test_own_failing_row_still_fails_the_variant():
    rows = [_q4_row("a.mtx"), _q4_row("b.mtx", status="FAIL"), _legacy_fail_row()]

    assert _project(_phase(rows))["status"] == "FAIL"


def test_traceback_read_from_stderr_log_when_flag_absent(tmp_path):
    log = tmp_path / "performance_stderr.log"
    phase = _phase(_q4_rows())
    del phase["stderr_has_traceback"]

    log.write_text("UserWarning: sparse CSC is beta\n", encoding="utf-8")
    phase["stderr_log_path"] = str(log)
    assert _project(dict(phase))["status"] == "PASS"

    log.write_text("Traceback (most recent call last):\n  ...\n", encoding="utf-8")
    assert _project(dict(phase))["status"] == "FAIL"

    del phase["stderr_log_path"]
    assert _project(dict(phase))["status"] == "FAIL"


def _classic_row(dtype, status="PASS"):
    return {
        "matrix": "a.mtx",
        "dtype": dtype,
        "index_dtype": "int32",
        "op": "non",
        "ms": "1.0",
        "pytorch_ms": "2.0",
        "triton_ms": "1.0",
        "triton_speedup_vs_pytorch": "2.0",
        "status": status,
    }


def test_classic_other_dtype_failure_does_not_fail_this_dtype():
    rows = [_classic_row("float32"), _classic_row("float64", status="FAIL")]

    f32 = runner._delivery_performance_phase(_phase(rows), "f32")
    f64 = runner._delivery_performance_phase(_phase(rows), "f64")

    assert f32["status"] == "PASS"
    assert f32["parent_status"] == "FAIL"
    assert f64["status"] == "FAIL"


def test_classic_counterexamples_keep_parent_status():
    rows = [_classic_row("float32"), _classic_row("float64", status="FAIL")]

    def project(**kwargs):
        return runner._delivery_performance_phase(_phase(rows, **kwargs), "f32")

    assert project(status="TIMEOUT", returncode=-9)["status"] == "TIMEOUT"
    assert project(returncode=-11)["status"] == "FAIL"
    assert project(traceback=True)["status"] == "FAIL"


def test_input_matrix_count_matches_q4_variant_bench(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("a.mtx", "sub/b.mtx", "notes.txt"):
        (tmp_path / name).write_text("", encoding="utf-8")

    assert runner._input_matrix_count(tmp_path) == 2
    assert runner._input_matrix_count(tmp_path / "a.mtx") == 1
    assert runner._input_matrix_count(None) is None
