#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Print the delivery report as one table: variant x (accuracy, performance, speedup).

Reads a ``summary.json`` written by ``run_flagsparse_pytest.py`` (or the
``summary_split.json`` of ``run_flagsparse_split_delivery.py``, or the C API's
``capi/tools/write_summary.py``) and lists every registered delivery variant in
``conf/operators.yaml`` order. A variant the summary does not contain is printed
as MISSING rather than skipped, so a short run cannot pass for a complete one.

Exit status is 0 when all registered variants are present, 1 otherwise.

Usage:
    python3 tools/delivery_table.py <results-dir or summary.json> [--markdown]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.delivery_variants import load_delivery_variants  # noqa: E402


def _summary_path(target: Path) -> Path:
    if target.is_file():
        return target
    for name in ("summary_split.json", "summary.json"):
        if (target / name).is_file():
            return target / name
    raise SystemExit(f"no summary_split.json or summary.json under {target}")


def _speedup(performance: dict) -> str:
    data = performance.get("data")
    values = []
    if isinstance(data, dict):
        for entry in data.values():
            if isinstance(entry, dict) and entry.get("speedup"):
                values.append(float(entry["speedup"]))
    return f"{sum(values) / len(values):.3f}x" if values else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=Path, help="results directory or summary json")
    parser.add_argument(
        "--markdown", action="store_true", help="print a Markdown table"
    )
    args = parser.parse_args()

    path = _summary_path(args.target)
    result = json.loads(path.read_text(encoding="utf-8")).get("result", {})
    variants = load_delivery_variants()

    rows = []
    for variant in variants:
        entry = result.get(variant["id"])
        if entry is None:
            rows.append((variant["id"], "MISSING", "MISSING", "-"))
            continue
        accuracy = entry.get("accuracy") or {}
        performance = entry.get("performance") or {}
        rows.append(
            (
                variant["id"],
                str(accuracy.get("status", "-")),
                str(performance.get("status", "-")),
                _speedup(performance),
            )
        )

    header = ("variant", "accuracy", "performance", "speedup")
    if args.markdown:
        print("| # | " + " | ".join(header) + " |")
        print("|---:|---|---|---|---:|")
        for index, row in enumerate(rows, 1):
            print(f"| {index} | `{row[0]}` | " + " | ".join(row[1:]) + " |")
    else:
        width = max(len(row[0]) for row in rows)
        print(f"{'#':>3}  {header[0]:<{width}}  {header[1]:<10}  {header[2]:<12}  {header[3]}")
        for index, row in enumerate(rows, 1):
            print(f"{index:>3}  {row[0]:<{width}}  {row[1]:<10}  {row[2]:<12}  {row[3]}")

    missing = sum(1 for row in rows if row[1] == "MISSING")
    accuracy_counts = Counter(row[1] for row in rows)
    performance_counts = Counter(row[2] for row in rows)
    print(
        f"\n{path}: {len(rows)} registered variants, {missing} missing; "
        f"accuracy {dict(accuracy_counts)}; performance {dict(performance_counts)}",
        file=sys.stderr,
    )
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
