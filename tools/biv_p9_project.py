#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project each SpMV variant's 10-matrix mean speedup from a P9 run (biv_spmv_probe.py).

For every matrix and dtype the P9b rows give the fastest K-wide SELL-C-sigma kernel. Its
bare-launch time plus the API overhead measured on the same plan (the ``api`` row minus
the ``K=1`` row) estimates what a delivered call would take. Dividing each variant's
existing baseline (``h800_scaled_ms`` -- the denominator of the delivery reports) by
that, and by the variant's current time, gives two arithmetic means per variant:

  now        baseline / current delivered time        (from the results directories)
  projected  baseline / min(current, P9b estimate)    (a SELL / CSR / COO variant whose
             prepare re-encodes the matrix as SELL-C-sigma and runs the P9b kernel)

The projection is an estimate (separate runs, launch overhead approximated), not a
delivery number; it decides whether the K-wide kernel is worth implementing.

  python3 tools/biv_p9_project.py results_biv_p9.csv \\
      --sell results_biv_sell_auto_final_20261025 --delivery results_biv_65_20261024_final
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

# (value dtype, out dtype) as biv_spmv_probe.py writes them -> variant dtype tag
TAGS = {
    ("float32", "float32"): "f32",
    ("float16", "float16"): "f16",
    ("float16", "float32"): "f16f32",
    ("int8", "int32"): "i8i32",
    ("complex64", "complex64"): "c32",
}
VARIANT = re.compile(r"^spmv_(sell|csr|coo)_(f32|f16|f16f32|i8i32|c32)_int_non$")
# Listed with their current mean only: P9b has no kernel for them (mixed f32 -> c32,
# transposed or conjugated A).
OTHER = re.compile(r"^spmv_(sell|csr|coo)_\w+$")


def _stem(name):
    return Path(name).name.removesuffix(".mtx")


def _float(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0 and math.isfinite(value) else None


def _csv(path):
    path = Path(path)
    if path.is_dir():
        hits = sorted(path.rglob("performance.csv"))
        if not hits:
            raise SystemExit(f"no performance.csv under {path}")
        rows = []
        for hit in hits:
            rows += _csv(hit)
        return rows
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_p9(path):
    """{(tag, matrix): estimated delivered ms of the best P9b kernel}."""
    by_key = defaultdict(lambda: {"api": {}, "k1": {}, "best": {}})
    for row in _csv(path):
        if row.get("probe") != "P9b wide" or row.get("status") == "FAIL":
            continue
        tag = TAGS.get((row.get("dtype"), row.get("out_dtype")))
        ms = _float(row.get("ms"))
        if tag is None or ms is None:
            continue
        config = row["config"]
        c = config.split()[0]
        slot = by_key[(tag, _stem(row["matrix"]))]
        if " api " in config:
            slot["api"][c] = ms
        else:
            if " K=1 " in config:
                slot["k1"][c] = ms
            if c not in slot["best"] or ms < slot["best"][c][0]:
                slot["best"][c] = (ms, config)
    out = {}
    for key, slot in by_key.items():
        estimates = []
        for c, (ms, config) in slot["best"].items():
            overhead = 0.0
            if c in slot["api"] and c in slot["k1"]:
                overhead = max(0.0, slot["api"][c] - slot["k1"][c])
            estimates.append((ms + overhead, config))
        if estimates:
            out[key] = min(estimates)
    return out


def load_baselines(paths):
    """{variant: {matrix: (baseline_ms, current_ms)}} for the SELL / CSR / COO SpMV variants."""
    table = defaultdict(dict)
    for path in paths:
        for row in _csv(path):
            variant = row.get("variant", "")
            if not OTHER.match(variant):
                continue
            base = _float(row.get("h800_scaled_ms"))
            # h800_scaled_fs_ms is the numerator of the delivery report's speedup
            now = _float(row.get("h800_scaled_fs_ms")) or _float(row.get("triton_ms"))
            if base and now:
                table[variant][_stem(row["matrix"])] = (base, now)
    return table


def project(p9, baselines):
    lines = []
    for variant in sorted(baselines):
        cells = baselines[variant]
        hit = VARIANT.match(variant)
        if hit is None:
            now = sum(b / c for b, c in cells.values()) / len(cells)
            lines.append((variant, len(cells), now, None, 0, len(cells)))
            continue
        tag = hit.group(2)
        now, proj, used = [], [], 0
        for matrix, (base, current) in sorted(cells.items()):
            now.append(base / current)
            estimate = p9.get((tag, matrix))
            if estimate and estimate[0] < current:
                used += 1
                proj.append(base / estimate[0])
            else:
                proj.append(base / current)
        missing = sum(1 for m in cells if (tag, m) not in p9)
        lines.append((variant, len(cells), sum(now) / len(now), sum(proj) / len(proj), used, missing))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("p9_csv", help="the --csv written by biv_spmv_probe.py with P9b rows")
    parser.add_argument("--sell", default="results_biv_sell_auto_final_20261025",
                        help="SELL results dir or performance.csv (baseline + current time)")
    parser.add_argument("--delivery", default="results_biv_65_20261024_final",
                        help="delivery results dir (spmv_csr / spmv_coo performance.csv)")
    parser.add_argument("--verbose", action="store_true", help="also print the per-matrix choice")
    args = parser.parse_args(argv)

    p9 = load_p9(args.p9_csv)
    if not p9:
        raise SystemExit("no P9b rows in the P9 csv (run biv_spmv_probe.py without --skip P9)")
    paths = [args.sell]
    delivery = Path(args.delivery)
    paths += [delivery / op for op in ("spmv_csr", "spmv_coo") if (delivery / op).is_dir()]
    baselines = load_baselines(paths)
    print(f"{'variant':26s} {'matrices':>8s} {'now':>7s} {'projected':>9s}  P9b used / missing")
    for variant, count, now, proj, used, missing in project(p9, baselines):
        if proj is None:
            print(f"{variant:26s} {count:8d} {now:7.3f} {'-':>9s}  not covered by P9b")
            continue
        flag = "  >= 1.0" if proj >= 1.0 else ""
        print(f"{variant:26s} {count:8d} {now:7.3f} {proj:9.3f}  {used:2d} / {missing:2d}{flag}")
    if args.verbose:
        for (tag, matrix), (ms, config) in sorted(p9.items()):
            print(f"  {tag:7s} {matrix:14s} {ms * 1e3:8.1f} us  {config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
