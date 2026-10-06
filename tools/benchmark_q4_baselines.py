# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Retest performance targets with the original corpus, values and wall timing."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import torch
import q4_variant_bench as q4
import test_spmv_sell as sell
import test_vector_ops as vector

TARGETS = {
    "spmv_csr_f32_int_trans", "spmv_csr_c32_int_conj",
    "spmv_coo_f32_int_trans", "spmv_coo_c32_int_conj",
    "spmm_csr_f32_int_trans_non_row",
    "spmv_sell_f32_int_non", "spmv_sell_f16_int_non",
    "spmv_sell_c32_int_non", "spmv_sell_i8i32_int_non",
    "spmv_csr_f16_int_non", "spmv_csr_c32_int_non",
    "spmv_coo_f16f32_int_non", "spmv_coo_c32_int_non",
    "spmv_coo_f16_int_non", "spmv_coo_i8i32_int_non",
    "spmv_csc_f32_int_non", "spmv_csc_c32_int_non", "spmv_csc_f16_int_non",
    "spmm_csr_f16_int_non_non_row", "spmm_csr_c32_int_non_non_row",
    "spmm_coo_c32_int_non_non_row", "spvv_c32_int_conj",
}
FAMILIES = ("spmv_csr", "spmv_coo", "spmv_csc", "spmm_csr", "spmm_coo", "spmm_csc", "sddmm_csr")
VECTOR_VARIANTS = {
    "spvv_c32_int_conj": ("spvv", torch.complex64, "conj"),
    "spvv_f16f32_int_non": ("spvv", torch.float16, "non"),
    "spvv_i8i32_int_non": ("spvv", torch.int8, "non"),
    "axpby_f16_int": ("axpby", torch.float16, "non"),
}


def run_scatter(dense, nnz, warmup, iters, gen):
    """Keep the historical scatter CUDA-graph timing, including output reset."""
    import cupy as cp
    from flagsparse.sparse_operations.gather_scatter import _triton_scatter_impl
    from flagsparse.sparse_operations._common import _benchmark_cuda_graph_op
    device = torch.device("cuda")
    idx = torch.randperm(dense, generator=gen)[:nnz].to(device, torch.int32)
    values = torch.randint(-128, 128, (nnz,), generator=gen, dtype=torch.int8).to(device)
    ours_out = torch.empty(dense, dtype=torch.int8, device=device)
    pt_out = torch.empty_like(ours_out)
    cp_out = cp.empty(dense, dtype=cp.int8)
    ic, xc = cp.from_dlpack(idx), cp.from_dlpack(values)

    def ours():
        return _triton_scatter_impl(values, idx, dense, out=ours_out,
                                    reset_output=True, index_fallback_policy="strict")

    def pt_run():
        pt_out.zero_()
        pt_out.index_copy_(0, idx.long(), values)
        return pt_out

    def cp_run():
        # Capture CuPy kernels on PyTorch's capture stream, not the default stream.
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            cp_out.fill(0)
            cp_out[ic] = xc
        return cp_out

    ref = torch.zeros(dense, dtype=torch.int8)
    ref[idx.cpu().long()] = values.cpu()
    ours()
    pt_run()
    cp_run()
    cp.cuda.runtime.deviceSynchronize()
    passed = torch.equal(ours_out.cpu(), ref) and torch.equal(pt_out.cpu(), ref)
    cp_error = float((torch.as_tensor(cp.asnumpy(cp_out)).int() - ref.int()).abs().max())
    passed = passed and cp_error == 0
    times = {base: _benchmark_cuda_graph_op(run, graph_batch=100, warmup=warmup, repeats=iters)
             for base, run in (("ours_ms", ours), ("pytorch_ms", pt_run), ("cupy_ms", cp_run))}
    return {"variant": "scatter_i8_int", "case_id": f"scatter|int8|dense={dense}|nnz={nnz}",
            "dense_size": dense, "nnz": nnz, "index_dtype": "int32", "value_dtype": "int8",
            "reset_output": True, "unique_indices": True, "timed": "CUDA graph; batch=100",
            "cupy_method": "zero fill and unique indexed assignment", "cupy_max_error": cp_error,
            "status": "PASS" if passed else "FAIL", **times,
            "speedup_vs_pytorch": times["pytorch_ms"] / times["ours_ms"],
            "speedup_vs_cupy": times["cupy_ms"] / times["ours_ms"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "tests/data"))
    parser.add_argument("--variants", default=",".join(sorted(TARGETS)))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capi", action="store_true", help="also time the built C API CSR transpose SpMM")
    parser.add_argument("--min-speedup", type=float, default=.8)
    args = parser.parse_args()
    selected = set(args.variants.split(","))
    known = {v.name for family in FAMILIES for v in q4.VARIANTS[family]} | TARGETS | set(VECTOR_VARIANTS) | {"scatter_i8_int"}
    if selected - known:
        parser.error(f"unknown variants: {sorted(selected - known)}")
    paths = [p for p in q4.matrix_paths([args.input]) if Path(p).name != "q4_worker_smoke.mtx"]
    os.environ["FLAGSPARSE_BENCH_CUPY"] = "1"
    if args.capi:
        os.environ["FLAGSPARSE_BENCH_CAPI"] = "1"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    def save():
        summary = {}
        for variant in sorted({r["variant"] for r in rows}):
            cases = [r for r in rows if r["variant"] == variant]
            entry = {"cases": len(cases), "passed": sum(r.get("status") == "PASS" for r in cases)}
            for base in ("pytorch", "cupy"):
                ratios = [r[f"speedup_vs_{base}"] for r in cases if r.get("status") == "PASS" and r.get(f"speedup_vs_{base}") is not None]
                entry[base] = {"valid": len(ratios), "mean": sum(ratios) / len(ratios) if ratios else None,
                               "min": min(ratios) if ratios else None,
                               "below_0_8": sum(v < .8 for v in ratios)}
            summary[variant] = entry
            base = "cupy" if entry["cupy"]["valid"] else "pytorch"
            expected = 4 if variant in VECTOR_VARIANTS or variant == "scatter_i8_int" else len(paths)
            entry["acceptance_baseline"] = base
            entry["accepted"] = (entry["cases"] == expected and entry["passed"] == expected
                                 and entry[base]["valid"] == expected
                                 and entry[base]["mean"] is not None
                                 and entry[base]["mean"] >= args.min_speedup)
            if any("capi_ms" in r for r in cases):
                entry["capi"] = {}
                for base in ("pytorch", "cupy"):
                    ratios = [r[f"capi_speedup_vs_{base}"] for r in cases if f"capi_speedup_vs_{base}" in r]
                    entry["capi"][base] = {"valid": len(ratios), "mean": sum(ratios) / len(ratios),
                                           "min": min(ratios), "below_0_8": sum(v < .8 for v in ratios)}
        report = {"device": torch.cuda.get_device_name(), "warmup": args.warmup, "iters": args.iters,
                  "timing": ("wall clock with synchronization; prepare outside timing; "
                             "scatter uses historical CUDA graph timing (batch=100)"),
                  "min_speedup": args.min_speedup, "baseline_priority": "cupy; pytorch only if cupy unavailable",
                  "matrices": paths, "results": rows, "summary": summary}
        (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if rows:
            fields = sorted(set().union(*(r.keys() for r in rows)))
            with (output / "performance.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fields)
                writer.writeheader()
                writer.writerows(rows)

    for family in FAMILIES:
        variants = [v for v in q4.VARIANTS[family] if v.name in selected]
        if not variants:
            continue
        q4.VARIANTS[family] = variants
        # Persist every matrix, so errors or an interruption cannot hide coverage.
        for path in paths:
            cases = q4.run_variants(family, [path], args.warmup, args.iters)
            for row in cases:
                row["speedup_vs_cupy"] = q4._ratio(row.get("cupy_ms"), row.get("ours_ms"))
            rows.extend(cases)
            save()
    for variant, (op, dtype, operation) in VECTOR_VARIANTS.items():
        if variant not in selected:
            continue
        gen = torch.Generator().manual_seed(0)
        for dense, nnz in vector._parse_cases(vector.DEFAULT_CASES):
            row = vector._run_case(op, dtype, operation, dense, nnz,
                                   args.warmup, args.iters, gen)
            row["ours_ms"] = row["triton_ms"]
            row["speedup_vs_pytorch"] = row.get("triton_speedup_vs_pytorch")
            row["speedup_vs_cupy"] = row.get("triton_speedup_vs_cupy")
            rows.append(row)
            print(f"[q4] {row['case_id']} {row['status']}", flush=True)
            save()
    if "scatter_i8_int" in selected:
        gen = torch.Generator().manual_seed(0)
        for dense, nnz in vector._parse_cases(vector.DEFAULT_CASES):
            row = run_scatter(dense, nnz, args.warmup, args.iters, gen)
            rows.append(row)
            print(f"[q4] {row['case_id']} {row['status']}", flush=True)
            save()
    gen = torch.Generator().manual_seed(0)
    for path in paths if any(v.startswith("spmv_sell") for v in selected) else []:
        matrix = sell.read_scipy_csr(path)
        for dtype in (torch.float32, torch.float16, torch.complex64, torch.int8):
            tag = {torch.float32: "f32", torch.float16: "f16", torch.complex64: "c32", torch.int8: "i8i32"}[dtype]
            if f"spmv_sell_{tag}_int_non" not in selected:
                continue
            out_dtype = torch.int32 if dtype == torch.int8 else dtype
            row = sell._run(path, matrix, dtype, out_dtype, 32, args.warmup, args.iters, gen)
            row["ours_ms"] = row["triton_ms"]
            row["speedup_vs_pytorch"] = row.get("triton_speedup_vs_pytorch")
            row["speedup_vs_cupy"] = row.get("triton_speedup_vs_cupy")
            rows.append(row)
            print(f"[q4] {row['matrix']} {row['variant']} {row['status']} torch_speedup={row['speedup_vs_pytorch']}", flush=True)
            save()
    save()
    report = json.loads((output / "results.json").read_text())
    print(json.dumps(report["summary"], indent=2), flush=True)
    if set(report["summary"]) != selected or not all(s["accepted"] for s in report["summary"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
