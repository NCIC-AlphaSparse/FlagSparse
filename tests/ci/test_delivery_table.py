# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""tools/delivery_table.py: the one table every backend reports the delivery run with."""

import json
import subprocess
import sys
from pathlib import Path

from tools.delivery_variants import load_delivery_variants

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "delivery_table.py"


def _run(tmp_path, result):
    (tmp_path / "summary.json").write_text(
        json.dumps({"result": result}), encoding="utf-8"
    )
    return subprocess.run(
        [sys.executable, str(TOOL), str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_complete_summary_lists_every_variant_and_exits_zero(tmp_path):
    result = {
        v["id"]: {
            "accuracy": {"status": "Passed"},
            "performance": {"status": "Passed", "data": {"fp32": {"speedup": 2.0}}},
        }
        for v in load_delivery_variants()
    }
    proc = _run(tmp_path, result)
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1 + len(result)
    assert "2.000x" in proc.stdout


def test_missing_variant_is_printed_and_fails(tmp_path):
    ids = [v["id"] for v in load_delivery_variants()]
    result = {i: {"accuracy": {"status": "Passed"}} for i in ids[1:]}
    proc = _run(tmp_path, result)
    assert proc.returncode == 1
    assert f"{ids[0]}" in proc.stdout and "MISSING" in proc.stdout


def test_a_speedup_against_the_scaled_h800_time_is_labelled(tmp_path):
    ids = [v["id"] for v in load_delivery_variants()]
    result = {
        vid: {
            "accuracy": {"status": "Passed"},
            "performance": {"status": "Passed", "data": {"fp32": {"speedup": 2.0}}},
        }
        for vid in ids
    }
    result[ids[0]]["performance"]["data"]["fp32"]["baseline"] = "h800_scaled"
    proc = _run(tmp_path, result)
    assert proc.returncode == 0, proc.stderr
    labelled = [line for line in proc.stdout.splitlines() if "(H800折算)" in line]
    assert len(labelled) == 1 and ids[0] in labelled[0]


def test_a_pytorch_fallback_speedup_says_so(tmp_path):
    ids = [v["id"] for v in load_delivery_variants()]
    result = {
        vid: {
            "accuracy": {"status": "Passed"},
            "performance": {"status": "Passed", "data": {"fp32": {"speedup": 2.0}}},
        }
        for vid in ids
    }
    result[ids[0]]["performance"]["data"]["fp32"]["baseline"] = "h800_pytorch_scaled"
    proc = _run(tmp_path, result)
    line = next(line for line in proc.stdout.splitlines() if ids[0] in line)
    assert "(H800 PyTorch折算)" in line
