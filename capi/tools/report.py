#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""One table of the operator list's accuracy AND performance, from the ctest JSONs.

The per-operator JSONs already carry both: each benchmark row checks its result
against the host fp64 oracle before timing it, so `accuracy`/`error_ratio` and
`median_ms`/`speedup` describe the same run of the same variant. What was missing
was a single view of the list, which is what this prints.

One row per (operator, format, dtype) -- the variant, not the file. Where a
variant was swept over many matrices, the columns aggregate ACROSS matrices and
say how many fed each number, because a speedup over 3 matrices and one over 30
are not the same claim:

    accuracy    pass/total matrices, and the worst error ratio seen
    speedup     geometric mean over the matrices that passed accuracy, and the
                min/max, because a single geomean hides the skew effect entirely
                (SpMV ranges 0.55x to 2.5x depending on the matrix)

Usage:
    python3 tools/report.py --bench-dir ./bench [--csv out.csv] [--by-matrix]
"""

import argparse
import collections
import csv
import json
import math
import pathlib
import sys


def load_rows(bench_dir):
    rows = []
    files = sorted(bench_dir.glob("*_benchmark.json"))
    if not files:
        sys.exit(f"no *_benchmark.json under {bench_dir}")
    for path in files:
        doc = json.loads(path.read_text())
        op = doc.get("operator", path.stem.replace("_benchmark", ""))
        env = doc.get("env", {})
        for r in doc.get("result", []):
            r["_op"] = op
            r["_backend"] = env.get("backend", "?")
            r["_arch"] = env.get("arch", "?")
            rows.append(r)
    return rows


def geomean(vals):
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench-dir",
        type=pathlib.Path,
        default=pathlib.Path("capi_results"),
        help="directory of *_benchmark.json (default: capi_results/)",
    )
    ap.add_argument("--csv", type=pathlib.Path)
    ap.add_argument(
        "--all",
        action="store_true",
        help="include retained variants; default is the delivery "
        "list only (registered delivery variants, conf/operators.yaml)",
    )
    ap.add_argument(
        "--by-matrix",
        action="store_true",
        help="one row per (variant, matrix) instead of aggregating",
    )
    args = ap.parse_args()

    rows = load_rows(args.bench_dir)
    # Default to the delivery list. Retained variants are measured and stored --
    # they just do not belong in the headline report.
    total_rows = len(rows)
    if not args.all:
        rows = [r for r in rows if r.get("reporting", "delivery") == "delivery"]
    held = total_rows - len(rows)
    backend = rows[0]["_backend"] if rows else "?"
    arch = rows[0]["_arch"] if rows else "?"
    baseline = next((r.get("baseline") for r in rows if r.get("baseline")), "none")

    # Group by variant. The n/k sweeps inside an operator are folded together
    # here: they are the same variant measured at several widths, and separating
    # them would make the list longer than the operator list it is reporting.
    key = (
        (
            lambda r: (
                r["_op"],
                r.get("format", "?"),
                r.get("dtype", "?"),
                r.get("matrix", "?"),
            )
        )
        if args.by_matrix
        else (lambda r: (r["_op"], r.get("format", "?"), r.get("dtype", "?")))
    )

    groups = collections.OrderedDict()
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    scope = "all variants" if args.all else "delivery list"
    print(f"backend {backend} ({arch})   baseline {baseline}   scope: {scope}")
    print(
        f"{len(groups)} variants from {len(rows)} rows"
        + (f"   ({held} rows held back as retained; --all to include)" if held else "")
        + "\n"
    )

    hdr = (
        f"{'operator':<9}{'fmt':<6}{'dtype':<6}{'accuracy':>12}{'worst_err':>11}"
        f"{'speedup':>10}{'range':>17}{'status':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    out_csv = []
    n_pass = n_variant_ok = 0
    all_speedups = []
    for k, rs in groups.items():
        op, fmt, dt = k[0], k[1], k[2]
        checked = [r for r in rs if r.get("accuracy") in ("pass", "fail")]
        passed = [r for r in rs if r.get("accuracy") == "pass"]
        worst = max((r.get("error_ratio") or 0.0) for r in checked) if checked else None
        sp = [r["speedup"] for r in rs if r.get("speedup")]
        gm = geomean(sp)
        if gm:
            all_speedups.extend(sp)
        # A variant counts as ok only if every matrix that was checked passed and
        # nothing errored -- "mostly passing" is not a state worth reporting as ok.
        bad = [r for r in rs if r.get("status") not in ("ok",)]
        ok = bool(checked) and len(passed) == len(checked) and not bad
        n_variant_ok += ok
        n_pass += len(passed)

        acc = f"{len(passed)}/{len(checked)}" if checked else "unchecked"
        rng = f"{min(sp):.2f}–{max(sp):.2f}" if len(sp) > 1 else ""
        status = "ok" if ok else (bad[0].get("status") if bad else "acc-fail")
        print(
            f"{op:<9}{fmt:<6}{dt:<6}{acc:>12}"
            f"{(f'{worst:.3g}' if worst is not None else '-'):>11}"
            f"{(f'{gm:.3f}' if gm else '-'):>10}{rng:>17}{status:>10}"
        )
        out_csv.append(
            {
                "operator": op,
                "format": fmt,
                "dtype": dt,
                "matrices_checked": len(checked),
                "matrices_passed": len(passed),
                "worst_error_ratio": worst,
                "speedup_geomean": gm,
                "speedup_min": min(sp) if sp else None,
                "speedup_max": max(sp) if sp else None,
                "status": status,
            }
        )

    overall = geomean(all_speedups)
    print("-" * len(hdr))
    print(
        f"{n_variant_ok}/{len(groups)} variants fully passing; "
        f"{n_pass} variant-matrix pairs accurate"
    )
    if overall:
        print(
            f"overall speedup geomean {overall:.3f}x vs {baseline} "
            f"over {len(all_speedups)} accurate measurements"
        )
        print(
            "  NOTE: a single geomean across a mixed corpus is the least "
            "informative number here -- read the per-variant range column."
        )

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_csv[0].keys()))
            w.writeheader()
            w.writerows(out_csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
