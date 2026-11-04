# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""C API summary: a variant that ran and passed without a vendor baseline."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _row(dtype, matrix, **extra):
    row = {
        "name": f"gather_spvec_{dtype}_{matrix}",
        "dtype": dtype,
        "matrix": matrix,
        "reporting": "delivery",
        "status": "ok",
        "accuracy": "pass",
        "median_ms": 0.03,
    }
    row.update(extra)
    return row


def test_capi_summary_reports_nobaseline_instead_of_skipped(tmp_path):
    # MUSA, 2026-09-18: muSPARSE has no fp16 gather, so every f16 row ran and
    # passed with baseline_status "failed". That used to surface as "Skipped".
    no_vendor = {
        "baseline_status": "failed",
        "baseline_detail": "muSPARSE: Gather dtype unsupported",
    }
    rows = [
        _row("f16", "a", **no_vendor),
        _row("f16", "b", **no_vendor),
        _row("f32", "a", baseline_status="ok", baseline_ms=0.06, speedup=2.0),
        # Never ran: must stay Skipped, not be promoted to NoBaseline.
        _row("f64", "a", status="skipped: no operand", accuracy="", median_ms=0),
    ]
    (tmp_path / "gather_benchmark.json").write_text(
        json.dumps({"operator": "gather", "env": {"backend": "MUSA"}, "result": rows}),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "capi" / "tools" / "write_summary.py"),
            "--bench-dir",
            str(tmp_path),
            "--out",
            str(tmp_path),
            "--no-html",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )
    result = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))[
        "result"
    ]

    f16 = result["gather_f16_int"]["performance"]
    assert f16["status"] == "NoBaseline"
    assert f16["data"]["fp16"]["result"] == "NoBaseline"
    assert f16["data"]["fp16"]["reason"] == ["muSPARSE: Gather dtype unsupported"]
    assert result["gather_f32_int"]["performance"]["status"] == "Passed"
    assert result["gather_f64_int"]["performance"]["status"] == "Skipped"


def _summarize(tmp_path, rows):
    (tmp_path / "gather_benchmark.json").write_text(
        json.dumps({"operator": "gather", "env": {"backend": "CUDA"}, "result": rows}),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "capi" / "tools" / "write_summary.py"),
            "--bench-dir",
            str(tmp_path),
            "--out",
            str(tmp_path),
            "--no-html",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )
    return json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["result"]


def test_a_failed_row_fails_the_variant_instead_of_dropping_out(tmp_path):
    # CUDA, 2026-11-02: three SpMV transposes failed their accuracy gate on one
    # matrix of ten and still read NoBaseline -- the failed row was filtered out
    # before "every measured row passed" was checked. Same with a baseline.
    no_vendor = {"baseline_status": "failed", "baseline_detail": "not wired"}
    failed = {"status": "failed", "accuracy": "fail", "median_ms": 0}
    rows = [
        _row("f16", "a", **no_vendor),
        _row("f16", "b", **no_vendor, **failed),
        _row("f32", "a", baseline_status="ok", baseline_ms=0.06, speedup=2.0),
        _row("f32", "b", baseline_status="ok", baseline_ms=0.06, **failed),
        _row("f64", "a", baseline_status="ok", baseline_ms=0.06, speedup=1.5),
    ]
    result = _summarize(tmp_path, rows)

    f16 = result["gather_f16_int"]["performance"]
    assert f16["status"] == "Failed"
    assert f16["data"]["fp16"]["failed_rows"] == ["b"]
    f32 = result["gather_f32_int"]["performance"]
    assert f32["status"] == "Failed"
    # The speedup is still over the rows that passed, and says which did not.
    assert f32["data"]["fp32"]["speedup"] == 2.0
    assert f32["data"]["fp32"]["failed_rows"] == ["b"]
    assert result["gather_f64_int"]["performance"]["status"] == "Passed"


def test_baseline_preparation_scope_survives_summary_projection(tmp_path):
    metadata = {
        "baseline_method": "CSC(A) -> CSR(A^H); muSPARSE SpMV",
        "baseline_timing_scope": "operator call; preparation excluded",
        "baseline_prepare_ms": 1.25,
    }
    result = _summarize(
        tmp_path,
        [
            _row(
                "f32",
                "a",
                baseline_status="ok",
                baseline_ms=0.06,
                speedup=2.0,
                **metadata,
            ),
            _row("f32", "b", baseline_status="ok", baseline_ms=0.09, speedup=3.0),
        ],
    )
    details = result["gather_f32_int"]["performance"]["data"]["fp32"]["details"]
    assert all(details["a"][key] == value for key, value in metadata.items())
    assert all(key not in details["b"] for key in metadata)
    # Preparation time is labelled separately, never silently added to the ratio.
    assert details["a"]["base"] == 0.06
    assert details["a"]["speedup"] == 2.0
