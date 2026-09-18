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
