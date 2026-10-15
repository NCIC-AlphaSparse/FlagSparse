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


def test_a_scaled_speedup_shows_only_the_number_unless_asked(tmp_path):
    ids = [v["id"] for v in load_delivery_variants()]
    result = {
        vid: {
            "accuracy": {"status": "Passed"},
            "performance": {"status": "Passed", "data": {"fp32": {"speedup": 2.0}}},
        }
        for vid in ids
    }
    result[ids[0]]["performance"]["data"]["fp32"]["baseline"] = "h800_scaled"
    result[ids[1]]["performance"]["data"]["fp32"]["baseline"] = "h800_pytorch_scaled"
    proc = _run(tmp_path, result)
    assert proc.returncode == 0, proc.stderr
    assert "H800" not in proc.stdout and "h800" not in proc.stdout
    line = next(line for line in proc.stdout.splitlines() if ids[0] in line)
    assert line.rstrip().endswith("2.000x")
    # --show-baseline says where each denominator came from
    shown = subprocess.run(
        [sys.executable, str(TOOL), str(tmp_path), "--show-baseline"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.splitlines()
    by_id = {vid: next(line for line in shown if vid in line) for vid in ids[:3]}
    assert by_id[ids[0]].rstrip().endswith("h800_scaled")
    assert by_id[ids[1]].rstrip().endswith("h800_pytorch_scaled")
    assert by_id[ids[2]].rstrip().endswith("-")
