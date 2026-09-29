#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Project q4 variants (repo-root conf/operators.yaml's ``q4_variants:``) out of
capi's *_benchmark.json, the same files write_summary.py reads for the 20-variant
delivery report.

WHY A SEPARATE SCRIPT, NOT A MODE OF write_summary.py. write_summary.py's
``variant_name(row)`` keys purely on (operator, dtype) because that is unambiguous
for the 20 delivered variants -- each (operator, dtype) pair names exactly one of
them. It is NOT unambiguous for the 42 q4 variants: q4 has, for example, three
`spmm_csr` variants at dtype f32 alone (`_non_non_col`, `_non_trans_row`,
`_trans_non_row`), which differ only in a transpose/layout axis the benchmark rows
do not currently tag. Reusing that function for q4 would either collide silently
or require changing it in a way that risks the delivery report's existing
semantics. So this file owns a SEPARATE, explicit table (CONFIRMED) of exactly
which q4 variant IDs a benchmark row can currently be attributed to, and why --
see the comment on CONFIRMED below before adding to it.

Usage:
    python3 tools/write_summary_q4.py --bench-dir ./bench --out ./bench
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.delivery_variants import load_q4_variants  # noqa: E402

# Every q4 variant this script can currently attribute to a real capi benchmark
# row, keyed by variant id. Each entry names the (operator, dtype) the row must
# carry -- both spelled exactly as capi/tools/gen_variants.py's DTYPES table and
# the ctest row tags already spell them (q4_variants' `dtype: c32` in
# conf/operators.yaml IS the row tag, no alias needed).
#
# A variant only belongs here once its row's ACTUAL measured op/layout has been
# hand-verified (by reading the benchmark .cpp, not assumed from the yaml) to
# match what the variant id claims. Getting this wrong silently mislabels a
# number as something it is not, which is worse than leaving the variant
# NotCoveredYet. See NOT_YET_COVERED below for the specific gap blocking each of
# the other 39 -- consult that before adding a new CONFIRMED entry.
CONFIRMED: dict[str, dict[str, str]] = {
    # spmv's only supported opA is NON_TRANSPOSE (CSR/COO reject transpose:
    # capi/src/ops/spmv.cpp), so every spmv_csr/spmv_coo row IS the "_non"
    # direction already -- no ambiguity to check beyond dtype.
    "spmv_csr_c32_int_non": {"operator": "spmv_csr", "dtype": "c32"},
    "spmv_coo_c32_int_non": {"operator": "spmv_coo", "dtype": "c32"},
    # ctest/benchmark/test_spmm.cpp hardcodes
    # opA=NON_TRANSPOSE, opB=NON_TRANSPOSE, and BOTH dense operands
    # FLAGSPARSE_ORDER_COL (see the flagsparseSpMM call in its measured loop) --
    # that is exactly "_non_non_col". f32 is the only q4 spmm_csr variant at that
    # exact op/layout; the other spmm_csr f32/f16/c32 variants need row-major
    # and/or opA=trans, neither of which this benchmark measures.
    "spmm_csr_f32_int_non_non_col": {"operator": "spmm_csr", "dtype": "f32"},
}

# The rest of the 42, and the specific reason each is not in CONFIRMED yet. Kept
# here (not just in commit history) so re-deriving this classification is not
# needed again before the next slice of work.
NOT_YET_COVERED_REASONS = {
    "no_i8_in_sweep": (
        "capi/ctest/sweep.hpp's upload_as/elem_bytes/read_back and "
        "capi/tools/gen_variants.py's DTYPES table have no int8 case; adding "
        "int8 to operators.yaml today would silently size it as fp32."
    ),
    "no_csc_benchmark_builder": (
        "capi/ctest/benchmark/test_spmv.cpp has no CSC operand builder "
        "(accuracy coverage exists; benchmark rows report "
        "status=not_implemented_in_test)."
    ),
    "no_f16_accuracy_host_type": (
        "capi/ctest has no shared half-precision host type; "
        "ctest/accuracy/test_spmv.cpp's run_spmv_fmt<T> is only instantiated "
        "for float/double/complex<float>/complex<double> today."
    ),
    "spmm_row_or_trans_not_measured": (
        "ctest/benchmark/test_spmm.cpp measures exactly one op/layout "
        "(non/non/col) per dtype; this variant needs row-major C and/or "
        "opA=TRANSPOSE, which capi/src/ops/spmm.cpp does not support yet "
        "(opA transpose) or the benchmark does not sweep (row layout)."
    ),
    "sddmm_op_order_not_measured": (
        "ctest/benchmark/test_sddmm.cpp measures one fixed op/order per "
        "(dtype, k); it has no axis distinguishing this variant's op/order "
        "from the others at the same dtype."
    ),
    "new_capi_operator_group": (
        "this operator (spvv/axpby/spmv_sell/spmm_bell/spmm_bsr/spmm_csc) has "
        "no entry in capi/conf/operators.yaml at all yet."
    ),
    "mixed_precision_kernel_not_wired": (
        "this variant calls a mixed_spmx.py kernel that is not the kernel "
        "capi's existing dispatch for this operator is wired to."
    ),
    "opA_transpose_or_conj_not_supported": (
        "capi/src/ops/spmv.cpp (or spmm.cpp) returns NOT_SUPPORTED for this "
        "operator's opA=TRANSPOSE/CONJUGATE_TRANSPOSE."
    ),
}


def _classify_uncovered(variant: dict[str, str]) -> str:
    vid, op, dtype = variant["id"], variant["operator"], variant["dtype"]
    if dtype == "i8" or dtype.startswith("i8"):
        return "no_i8_in_sweep"
    if op == "spmv_csc":
        return "no_csc_benchmark_builder"
    if dtype == "f16" and op in ("spmv_csr", "spmv_coo", "spmv_csc"):
        return "no_f16_accuracy_host_type"
    if op in ("spmm_csr", "spmm_coo") and vid not in CONFIRMED:
        return "spmm_row_or_trans_not_measured"
    if op == "sddmm_csr":
        return "sddmm_op_order_not_measured"
    if op in ("spvv", "axpby", "spmv_sell", "spmm_bell", "spmm_bsr", "spmm_csc"):
        return "new_capi_operator_group"
    if dtype in ("f16f32", "i8i32", "i8f32", "f32c32"):
        return "mixed_precision_kernel_not_wired"
    if "trans" in vid or "conj" in vid:
        return "opA_transpose_or_conj_not_supported"
    return "unclassified"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=pathlib.Path, default=pathlib.Path("capi_results"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("capi_results"))
    args = ap.parse_args()

    variants = load_q4_variants()
    files = sorted(args.bench_dir.glob("*_benchmark.json"))
    rows_by_op_dtype: dict[tuple[str, str], list[dict]] = {}
    for path in files:
        doc = json.loads(path.read_text())
        for r in doc.get("result", []):
            rows_by_op_dtype.setdefault((r.get("operator", ""), r.get("dtype", "")), []).append(r)

    result = {}
    for variant in variants:
        vid = variant["id"]
        confirmed = CONFIRMED.get(vid)
        if confirmed is None:
            reason_key = _classify_uncovered(variant)
            result[vid] = {
                "status": "NotCoveredYet",
                "reason_key": reason_key,
                "reason": NOT_YET_COVERED_REASONS.get(reason_key, "unclassified gap"),
            }
            continue
        rows = rows_by_op_dtype.get((confirmed["operator"], confirmed["dtype"]), [])
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        with_baseline = [r for r in ok_rows if r.get("baseline_status") == "ok"]
        if not rows:
            result[vid] = {
                "status": "NoRows",
                "reason": (
                    f"no benchmark row for operator={confirmed['operator']} "
                    f"dtype={confirmed['dtype']} in {args.bench_dir} -- rerun the "
                    "capi benchmark for this operator family first"
                ),
            }
            continue
        details = {
            r.get("matrix", r.get("name", "?")): {
                "flagsparse_ms": r.get("median_ms"),
                "vendor_ms": r.get("baseline_ms"),
                "vendor": r.get("baseline"),
                "speedup_vs_vendor": r.get("speedup"),
                "accuracy": r.get("accuracy"),
            }
            for r in ok_rows
        }
        speedups = [r["speedup"] for r in with_baseline if r.get("speedup")]
        result[vid] = {
            "status": "Passed" if speedups else ("Measured" if ok_rows else "Failed"),
            "rows_measured": len(rows),
            "rows_ok": len(ok_rows),
            "rows_with_vendor_baseline": len(with_baseline),
            "vendor": (with_baseline[0].get("baseline") if with_baseline else None),
            "avg_speedup_vs_vendor": (sum(speedups) / len(speedups)) if speedups else None,
            "details": details,
        }

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "summary_q4.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    n_confirmed = sum(1 for v in result.values() if v["status"] in ("Passed", "Measured"))
    n_uncovered = sum(1 for v in result.values() if v["status"] == "NotCoveredYet")
    n_no_rows = sum(1 for v in result.values() if v["status"] == "NoRows")
    print(
        f"wrote {out_path}  ({len(result)} q4 variants: "
        f"{n_confirmed} measured through capi, {n_no_rows} confirmed-but-not-run, "
        f"{n_uncovered} not covered yet)"
    )


if __name__ == "__main__":
    main()
