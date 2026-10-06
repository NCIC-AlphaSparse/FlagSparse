# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic before/after checks; run with PYTHONPATH=src:. python tools/benchmark_q4_optimization.py.

Timings compare implementation versions on the same device, not vendor speedups.
"""
import importlib
import json
import argparse
from pathlib import Path

import torch
import triton
import triton.language as tl

csr = importlib.import_module("flagsparse.sparse_operations.spmv_csr")
sell = importlib.import_module("flagsparse.sparse_operations.spmv_sell")
from capi.flagsparse_codegen.spmm_transpose import spmm_csr_transpose_atomic_f32


@triton.jit
def old_spmm(V, I, P, B, C, M: tl.constexpr, N: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    pos = tl.load(P + row) + tl.program_id(2) * BK + tl.arange(0, BK)
    valid = pos < tl.load(P + row + 1)
    col = tl.load(I + pos, valid, other=0)
    v = tl.load(V + pos, valid, other=0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    b = tl.load(B + row * N + n, n < N, other=0)
    tl.atomic_add(C + col[:, None] * N + n[None, :],
                  v[:, None] * b[None, :], valid[:, None] & (n[None, :] < N), sem="relaxed")


def measure(before, after):
    # Warm both paths; include output initialization where the caller does it.
    before()
    after()
    torch.cuda.synchronize()
    old = triton.testing.do_bench(before, warmup=100, rep=300)
    new = triton.testing.do_bench(after, warmup=100, rep=300)
    return {"before_ms": old, "after_ms": new, "speedup": old / new}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="append", default=[], help="Matrix Market sparsity pattern; values are randomized")
    parser.add_argument("--real-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20261006)
    results = []
    cases = []
    for skew in (() if args.real_only else (False, True)):
        m, k, n = 8192, 8192, 64
        lengths = torch.full((m,), 16, dtype=torch.int64, device="cuda")
        if skew:
            lengths[::128] = 2048
            lengths[1::128] = 0
        ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(torch.int32)
        nnz = int(ptr[-1])
        col = torch.randint(k, (nnz,), device="cuda", dtype=torch.int32)
        cases.append(("synthetic_skew" if skew else "synthetic_uniform", m, k, ptr, col))
    for path in args.matrix:
        from scipy.io import mmread
        matrix = mmread(path).tocsr()
        ptr = torch.as_tensor(matrix.indptr, dtype=torch.int32, device="cuda")
        col = torch.as_tensor(matrix.indices, dtype=torch.int32, device="cuda")
        cases.append((Path(path).stem, *matrix.shape, ptr, col))
    for name, m, k, ptr, col in cases:
        case_start = len(results)
        n = 64
        lengths = (ptr[1:] - ptr[:-1]).long()
        nnz = col.numel()
        maximum = int(lengths.max())
        skew = maximum > 64 and maximum > 4 * (nnz // max(m, 1) + 1)
        rows = torch.repeat_interleave(torch.arange(m, device="cuda"), lengths)
        for dtype in (torch.float32, torch.complex64):
            v = torch.randn(nnz, device="cuda", dtype=dtype)
            x = torch.randn(m, device="cuda", dtype=dtype)
            op = "conj" if dtype.is_complex else "trans"
            p = csr.prepare_spmv_csr(v, col, ptr, (m, k), op=op)
            ref = torch.zeros(k, device="cuda", dtype=dtype)
            ref.index_add_(0, col.long(), (v.conj() if dtype.is_complex else v) * x[rows])
            actual = csr.flagsparse_spmv_csr_run(p, x)
            torch.testing.assert_close(actual, ref, atol=2e-4, rtol=2e-4)
            enabled = csr._spmv_transpose_scatter_enabled
            def before():
                csr._spmv_transpose_scatter_enabled = lambda p: False
                try:
                    return csr.flagsparse_spmv_csr_run(p, x)
                finally:
                    csr._spmv_transpose_scatter_enabled = enabled
            results.append({"operator": "csr_" + op, "dtype": str(dtype), "skew": skew,
                            **measure(before, lambda: csr.flagsparse_spmv_csr_run(p, x))})
            if not skew:
                sv, sc, so = sell.csr_to_sell(v, col, ptr, m, 32)
                sx = torch.randn(k, device="cuda", dtype=dtype)
                sy = torch.empty(m, device="cuda", dtype=dtype)
                kernel_args = (tuple(torch.view_as_real(t).reshape(-1) for t in (sy, sv))
                        + (sc, so, torch.view_as_real(sx).reshape(-1), m)
                        if dtype.is_complex else (sy, sv, sc, so, sx, m))
                kernel = sell._spmv_sell_complex_kernel if dtype.is_complex else sell._spmv_sell_real_kernel
                kw = {} if dtype.is_complex else {"ACC": tl.float32}
                def launch(group, warps=4):
                    kernel[(triton.cdiv(triton.cdiv(m, 32), group),)](*kernel_args, SLICE=32, BLOCK_R=32, GROUP=group, num_warps=warps, **kw)
                    return sy
                expected = launch(1).clone()
                torch.testing.assert_close(launch(4), expected, atol=0, rtol=0)
                results.append({"operator": "sell", "dtype": str(dtype),
                                **measure(lambda: launch(1), lambda: launch(4))})
                results.append({"operator": "sell_one_warp", "dtype": str(dtype),
                                **measure(lambda: launch(1), lambda: launch(1, 1))})
                if dtype.is_complex:
                    def parallel():
                        sell._spmv_sell_complex_parallel_kernel[(triton.cdiv(m, 32),)](
                            *kernel_args, SLICE=32, BLOCK_R=32, SLOTS=4)
                        return sy
                    torch.testing.assert_close(parallel(), expected, atol=2e-5, rtol=2e-5)
                    results.append({"operator": "sell_parallel_slots", "dtype": str(dtype),
                                    **measure(lambda: launch(1), parallel)})
        v = torch.randn(nnz, device="cuda")
        b = torch.randn(m, n, device="cuda")
        c = torch.empty(k, n, device="cuda")
        segments = triton.cdiv(int(lengths.max()), 32)
        def before_mm():
            c.zero_()
            old_spmm[(m, 1, segments)](v, col, ptr, b, c, m, n, 64, 32)
            return c
        def after_mm():
            c.zero_()
            grid = (triton.cdiv(nnz, 32), 1, 1) if skew else (m, 1, segments)
            spmm_csr_transpose_atomic_f32[grid](
                v, col, ptr, b, c, 1., m, nnz, n, n, 1, n, 1, 64, 32, skew)
            return c
        expected = before_mm().clone()
        torch.testing.assert_close(after_mm(), expected, atol=2e-4, rtol=2e-4)
        results.append({"operator": "capi_spmm_kernel", "skew": skew, **measure(before_mm, after_mm)})
        for result in results[case_start:]:
            result.update(matrix=name, rows=m, cols=k, nnz=nnz, max_row_nnz=maximum)
    print(json.dumps({"device": torch.cuda.get_device_name(), "results": results}, indent=2))


if __name__ == "__main__":
    main()
