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
# the rest -- consult that before adding a new CONFIRMED entry.
CONFIRMED: dict[str, dict[str, str]] = {
    # The current benchmark rows use NON_TRANSPOSE. CSR/COO also have narrow
    # q4 atomic transpose/conjugate paths, but no benchmark row records that
    # op axis yet, so these two ordinary-complex rows remain unambiguous.
    "spmv_csr_c32_int_non": {"operator": "spmv_csr", "dtype": "c32"},
    "spmv_coo_c32_int_non": {"operator": "spmv_coo", "dtype": "c32"},
    # ctest/benchmark/test_spmm.cpp hardcodes
    # opA=NON_TRANSPOSE, opB=NON_TRANSPOSE, and BOTH dense operands
    # FLAGSPARSE_ORDER_COL (see the flagsparseSpMM call in its measured loop) --
    # that is exactly "_non_non_col". f32 is the only q4 spmm_csr variant at that
    # exact op/layout; the other spmm_csr f32/f16/c32 variants need row-major
    # and/or opA=trans, neither of which this benchmark measures.
    "spmm_csr_f32_int_non_non_col": {"operator": "spmm_csr", "dtype": "f32"},
    # ctest/benchmark/test_spgemm.cpp hardcodes opA=NON_TRANSPOSE,
    # opB=NON_TRANSPOSE (the two `NT, NT` arguments to every flagsparseSpGEMM_*
    # call in its measured loop) -- exactly "_non_non". spgemm_csr has no other
    # op axis, so this is unambiguous at f32 (part of the original 20-variant
    # delivery set, already implemented and benchmarked against cuSPARSE).
    "spgemm_csr_f32_int_non_non": {"operator": "spgemm_csr", "dtype": "f32"},
    # gather/scatter have no op axis at all (pure data movement); i8 is already
    # in capi/conf/operators.yaml's dtypes list (added alongside the ctest
    # accuracy/benchmark support this session) and gets a real cuSPARSE
    # baseline (verified: geomean 0.70x over 18 matrices).
    "scatter_i8_int": {"operator": "scatter", "dtype": "i8"},
    # ctest/benchmark/test_sddmm.cpp hardcodes opA=NON_TRANSPOSE,
    # opB=NON_TRANSPOSE, and both dense operands FLAGSPARSE_ORDER_ROW -- exactly
    # "_non_non_row". f16 is now dispatch-generic in capi (see
    # capi/src/ops/sddmm.cpp and conf/operators.yaml's sddmm_csr entry);
    # cuSPARSE itself has no fp16 SDDMM (SDDMM_bufferSize returns an error
    # status for it), so this row has no vendor speedup -- that is a real
    # vendor-library gap, not a defect here.
    "sddmm_csr_f16_int_non_non_row": {"operator": "sddmm_csr", "dtype": "f16"},
}

# The rest of the 42, and the specific reason each is not in CONFIRMED yet. Kept
# here (not just in commit history) so re-deriving this classification is not
# needed again before the next slice of work.
NOT_YET_COVERED_REASONS = {
    "no_i8_in_sweep": (
        "plain int8 (gather/scatter) now has an upload_as/elem_bytes/read_back "
        "case in capi/ctest/sweep.hpp and an entry in "
        "capi/tools/gen_variants.py's DTYPES table -- this reason covers "
        "int8 gather/scatter variants that just haven't been wired into "
        "operators.yaml's dtypes list yet, not a sweep.hpp gap."
    ),
    "no_csc_benchmark_builder": (
        "capi/ctest/benchmark/test_spmv.cpp has no CSC operand builder "
        "(accuracy coverage exists; benchmark rows report "
        "status=not_implemented_in_test)."
    ),
    "f16_precision_exceeds_tolerance_on_corpus": (
        "the dispatch/kernel path is wired and verified correct: "
        "ctest/accuracy/test_spmv.cpp's Float16MatchesHostReference and "
        "CooFloat16MatchesHostReference pass (using the shared Half host type "
        "in ctest/common.hpp) on a low-nnz-per-row matrix. But "
        "ctest/benchmark/test_spmv.cpp's real corpus matrices have enough "
        "nonzeros per row that fp16 storage quantization (comparing against "
        "the *unquantized* fp64 reference, the same source of error "
        "documented for sddmm's fp16 case) exceeds "
        "default_tolerance(FLAGSPARSE_R_16F) -- error ratio 4.5x-9.4x "
        "observed, not a wiring bug. cuSPARSE itself also has no fp16 SpMV "
        "(baseline_status=failed), so there would be no vendor speedup even "
        "if this cleared tolerance."
    ),
    "spmm_row_or_trans_not_measured": (
        "ctest/benchmark/test_spmm.cpp measures exactly one op/layout "
        "(non/non/col) per dtype; this variant needs a row-major or transpose "
        "benchmark axis that is not present in the current sweep. If dispatch "
        "is already covered by a focused accuracy test, classify it as "
        "DispatchVerified rather than treating it as an implementation gap."
    ),
    "sddmm_dtype_not_in_capi_registry": (
        "capi/conf/operators.yaml's sddmm_csr entry only lists dtypes: "
        "[f32, f64]; the Python kernel now supports this dtype "
        "(src/flagsparse/sparse_operations/sddmm_csr.py), but capi has no "
        "C++ dispatch path for it yet -- f16 would need widening the dtype "
        "list only (the kernel is dtype-generic), c32 would need a new "
        "complex dispatch branch in capi/src/ops/sddmm.cpp mirroring "
        "_sddmm_csr_complex_kernel's view_as_real convention."
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
        "the Python side now supports this dtype combination "
        "(src/flagsparse/sparse_operations/mixed_spmx.py), but capi has no "
        "widened-output-dtype (out_dtype) concept at all for any operator -- "
        "wiring one variant means designing that concept in capi first, not "
        "just pointing at the existing Python kernel."
    ),
    "mixed_precision_dispatch_verified_no_benchmark_row": (
        "capi/src/ops/spmv.cpp or spmm.cpp's mixed-precision (out_dtype) "
        "dispatch IS wired for this variant and passes a hand-built "
        "ctest accuracy TEST_F on real hardware. It has no benchmark row "
        "yet only because "
        "capi/tools/gen_variants.py's DTYPES table maps one manifest dtype "
        "to one row tag and cannot express 'input dtype != output dtype' -- "
        "that generator/schema gap, not a dispatch gap, is what blocks "
        "CONFIRMED status here."
    ),
    "capi_dispatch_verified_no_benchmark_row": (
        "The C API dispatch and a targeted device accuracy test implement this "
        "q4 variant. It has no matching benchmark row yet, so it cannot be "
        "classified as CONFIRMED/measured without mislabelling another op, "
        "layout, or dtype combination."
    ),
    "implemented_corpus_tolerance_gap": (
        "The C API path is implemented and has focused accuracy coverage, but "
        "the corpus benchmark compares fp16 storage against an unquantized fp64 "
        "oracle. Quantization error exceeds the current global tolerance on "
        "these matrices; this is an experiment/tolerance policy gap, not a "
        "missing operator implementation."
    ),
    "opA_transpose_or_conj_not_supported": (
        "capi/src/ops/spmv.cpp (or spmm.cpp) returns NOT_SUPPORTED for this "
        "operator's opA=TRANSPOSE/CONJUGATE_TRANSPOSE."
    ),
}


_MIXED_DISPATCH_VERIFIED = frozenset({
    "spmv_csr_i8i32_int_non",
    "spmv_csr_i8f32_int_non",
    "spmv_csr_f16f32_int_non",
    "spmv_coo_i8i32_int_non",
    "spmv_coo_f16f32_int_non",
    "spmv_csr_f32c32_int_non",
    "spmm_csr_i8i32_int_non_non_row",
    "spmm_csr_f16f32_int_non_non_row",
    "spmm_coo_i8i32_int_non_non_row",
    "spvv_f16f32_int_non",
    "spvv_i8i32_int_non",
    "spmv_sell_i8i32_int_non",
})

# These are ordinary C API paths (not the widened-output family above).  Keep
# them separate: benchmark attribution lacks the needed op/layout axis, but the
# variants are implemented and have CUDA ctest accuracy coverage.
_CAPI_DISPATCH_VERIFIED = frozenset({
    "axpby_f16_int",
    "spmm_csc_c32_int_non_non_row",
    "spmm_csc_f16_int_non_non_row",
    "spmm_csc_f32_int_non_non_row",
    "spmv_sell_c32_int_non",
    "spmv_sell_f16_int_non",
    "spmv_sell_f32_int_non",
    "spvv_c32_int_conj",
    "spmv_csr_f32_int_trans",
    "spmv_coo_f32_int_trans",
    "spmv_csr_c32_int_conj",
    "spmv_coo_c32_int_conj",
    "sddmm_csr_c32_int_non_non_row",
})

_DISPATCH_VERIFIED_REASONS = frozenset({
    "mixed_precision_dispatch_verified_no_benchmark_row",
    "capi_dispatch_verified_no_benchmark_row",
})


def _classify_uncovered(variant: dict[str, str]) -> str:
    vid, op, dtype = variant["id"], variant["operator"], variant["dtype"]
    # Compound/widened dtypes (mixed_spmx.py territory) before the plain-i8
    # check below -- "i8i32"/"i8f32" starting with "i8" would otherwise match
    # the wrong, now-partly-stale reason.
    if vid in _MIXED_DISPATCH_VERIFIED:
        return "mixed_precision_dispatch_verified_no_benchmark_row"
    if vid in _CAPI_DISPATCH_VERIFIED:
        return "capi_dispatch_verified_no_benchmark_row"
    if vid in {"spmv_csr_f16_int_non", "spmv_coo_f16_int_non"}:
        return "implemented_corpus_tolerance_gap"
    if dtype in ("f16f32", "i8i32", "i8f32", "f32c32"):
        return "mixed_precision_kernel_not_wired"
    if op in ("spvv", "axpby", "spmv_sell", "spmm_bell", "spmm_bsr", "spmm_csc"):
        return "new_capi_operator_group"
    if dtype == "i8":
        return "no_i8_in_sweep"
    if op == "spmv_csc":
        return "no_csc_benchmark_builder"
    if dtype == "f16" and op in ("spmv_csr", "spmv_coo"):
        return "f16_precision_exceeds_tolerance_on_corpus"
    if op == "sddmm_csr" and dtype in ("f16", "c32"):
        return "sddmm_dtype_not_in_capi_registry"
    if op in ("spmm_csr", "spmm_coo") and vid not in CONFIRMED:
        return "spmm_row_or_trans_not_measured"
    if op == "sddmm_csr":
        return "sddmm_op_order_not_measured"
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
    rows_by_q4_variant: dict[str, list[dict]] = {}
    for path in files:
        doc = json.loads(path.read_text())
        for r in doc.get("result", []):
            rows_by_op_dtype.setdefault((r.get("operator", ""), r.get("dtype", "")), []).append(r)
            if r.get("q4_variant"):
                rows_by_q4_variant.setdefault(r["q4_variant"], []).append(r)

    result = {}
    for variant in variants:
        vid = variant["id"]
        exact_rows = rows_by_q4_variant.get(vid)
        if exact_rows:
            ok_rows = [r for r in exact_rows if r.get("status") == "ok"]
            with_baseline = [r for r in ok_rows if r.get("baseline_status") == "ok"]
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
                "rows_measured": len(exact_rows), "rows_ok": len(ok_rows),
                "rows_with_vendor_baseline": len(with_baseline),
                "vendor": with_baseline[0].get("baseline") if with_baseline else None,
                "avg_speedup_vs_vendor": sum(speedups) / len(speedups) if speedups else None,
                "details": details,
            }
            continue
        confirmed = CONFIRMED.get(vid)
        if confirmed is None:
            reason_key = _classify_uncovered(variant)
            result[vid] = {
                "status": (
                    "DispatchVerified" if reason_key in _DISPATCH_VERIFIED_REASONS
                    else "NotCoveredYet"
                ),
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
    n_verified = sum(1 for v in result.values() if v["status"] == "DispatchVerified")
    n_uncovered = sum(1 for v in result.values() if v["status"] == "NotCoveredYet")
    n_no_rows = sum(1 for v in result.values() if v["status"] == "NoRows")
    print(
        f"wrote {out_path}  ({len(result)} q4 variants: "
        f"{n_confirmed} measured through capi, {n_no_rows} confirmed-but-not-run, "
        f"{n_verified} dispatch-verified, {n_uncovered} not covered yet)"
    )


if __name__ == "__main__":
    main()
