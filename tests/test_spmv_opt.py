"""
SpMV optimisation A/B test: compare _impl (baseline) vs _impl_opt (optimised)
side-by-side, together with PyTorch and cuSPARSE baselines.

Usage:
    python tests/test_spmv_opt.py <dir/>                # batch run, default float32
    python tests/test_spmv_opt.py <dir/> --csv opt.csv  # selected dtype, export CSV
"""
import argparse
import csv
import glob
import math
import os
import time

import torch
import flagsparse as fs
import flagsparse.sparse_operations.spmv_csr as spmv_csr_mod

VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPES = [torch.int32]
WARMUP = 10
ITERS = 50


def load_mtx_to_csr_torch(file_path, dtype=torch.float32, device=None):
    import math as _math
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    mm_field = "real"
    mm_symmetry = "general"
    data_lines = []
    header_info = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%%MatrixMarket"):
            tokens = stripped.split()
            if len(tokens) >= 5:
                mm_field = tokens[3].lower()
                mm_symmetry = tokens[4].lower()
            continue
        if stripped.startswith("%"):
            continue
        if not header_info and stripped:
            parts = stripped.split()
            n_rows = int(parts[0])
            n_cols = int(parts[1])
            nnz = int(parts[2]) if len(parts) > 2 else 0
            header_info = (n_rows, n_cols, nnz)
            continue
        if stripped:
            data_lines.append(stripped)
    if header_info is None:
        raise ValueError(f"Cannot parse .mtx header: {file_path}")
    n_rows, n_cols, nnz = header_info
    if nnz == 0:
        data = torch.tensor([], dtype=dtype, device=device)
        indices = torch.tensor([], dtype=torch.int64, device=device)
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
        return data, indices, indptr, (n_rows, n_cols)
    is_pattern = (mm_field == "pattern")
    is_symmetric = mm_symmetry in ("symmetric", "hermitian")
    is_skew = (mm_symmetry == "skew-symmetric")
    row_maps = [dict() for _ in range(n_rows)]
    for line in data_lines[:nnz]:
        parts = line.split()
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        v = 1.0 if is_pattern else float(parts[2])
        if 0 <= r < n_rows and 0 <= c < n_cols:
            row_maps[r][c] = row_maps[r].get(c, 0.0) + v
            if r != c:
                if is_symmetric and 0 <= c < n_rows and 0 <= r < n_cols:
                    row_maps[c][r] = row_maps[c].get(r, 0.0) + v
                elif is_skew and 0 <= c < n_rows and 0 <= r < n_cols:
                    row_maps[c][r] = row_maps[c].get(r, 0.0) - v
    cols_s = []
    vals_s = []
    indptr_list = [0]
    for r in range(n_rows):
        row = row_maps[r]
        for c in sorted(row.keys()):
            cols_s.append(c)
            vals_s.append(row[c])
        indptr_list.append(len(cols_s))
    data = torch.tensor(vals_s, dtype=dtype, device=device)
    indices = torch.tensor(cols_s, dtype=torch.int64, device=device)
    indptr = torch.tensor(indptr_list, dtype=torch.int64, device=device)
    return data, indices, indptr, (n_rows, n_cols)


def _timed_spmv(prepared, x, warmup, iters, use_opt):
    opt_buckets = None
    symbolic_ms = 0.0
    count = max(1, int(iters))
    if use_opt:
        opt_buckets = spmv_csr_mod._build_spmv_opt_runtime_buckets(prepared)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(count):
            opt_buckets = spmv_csr_mod._build_spmv_opt_runtime_buckets(prepared)
        torch.cuda.synchronize()
        symbolic_ms = (time.perf_counter() - t0) * 1000.0 / count

    def op():
        return spmv_csr_mod._run_spmv_prepared_with_fallback(
            prepared, x, use_opt=use_opt, opt_buckets=opt_buckets
        )

    y = op()
    torch.cuda.synchronize()
    for _ in range(warmup):
        y = op()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(count):
        y = op()
    e1.record()
    torch.cuda.synchronize()
    compute_ms = e0.elapsed_time(e1) / count
    return y, symbolic_ms + compute_ms, symbolic_ms, compute_ms


def _timed_pytorch(data, indices, indptr, x, shape, warmup, iters):
    device = data.device
    n_rows = int(shape[0])
    try:
        A = torch.sparse_csr_tensor(
            indptr.to(torch.int64), indices.to(torch.int64), data,
            size=shape, device=device,
        )
        y = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        op = lambda: torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    except Exception:
        row_ind = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        A = torch.sparse_coo_tensor(
            torch.stack([row_ind, indices.to(torch.int64)]),
            data, shape, device=device,
        ).coalesce()
        y = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        op = lambda: torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    torch.cuda.synchronize()
    for _ in range(warmup):
        op()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        op()
    e1.record()
    torch.cuda.synchronize()
    return y, e0.elapsed_time(e1) / iters


def _timed_cusparse(data, indices, indptr, x, shape, warmup, iters):
    import cupy as cp
    import cupyx.scipy.sparse as cpx
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
    ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
    ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
    x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
    A = cpx.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = A @ x_cp
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        _ = A @ x_cp
    e1.record()
    torch.cuda.synchronize()
    y_cp = A @ x_cp
    y = torch.utils.dlpack.from_dlpack(y_cp.toDlpack())
    return y, e0.elapsed_time(e1) / iters


def _fmt(v):
    return "N/A" if v is None else f"{v:.4f}"


def _spd(base, other):
    if base is None or other is None or other <= 0:
        return "N/A"
    return f"{base / other:.2f}x"


def _err(v):
    return "N/A" if v is None else f"{v:.2e}"


HEADER = (
    f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10}  "
    f"{'Base(ms)':>9} {'Opt(ms)':>9} {'Sym(ms)':>9} {'Comp(ms)':>9} "
    f"{'PT(ms)':>9} {'CU(ms)':>9}  "
    f"{'Opt/Base':>8} {'Opt/PT':>8} {'Opt/CU':>8}  "
    f"{'Err(Base)':>10} {'Err(Opt)':>10} {'Status':>6}"
)
SEP = "-" * 170


def run_one_mtx(path, dtype, index_dtype, warmup, iters):
    device = torch.device("cuda")
    data, indices, indptr, shape = load_mtx_to_csr_torch(path, dtype=dtype, device=device)
    indices = indices.to(index_dtype)
    n_rows, n_cols = shape
    nnz = data.numel()
    x = torch.randn(n_cols, dtype=dtype, device=device)
    prepared = fs.prepare_spmv_csr(data, indices, indptr, shape)

    if dtype == torch.float32:
        atol, rtol = 1e-4, 1e-2
    else:
        atol, rtol = 1e-12, 1e-10

    # ── Reference (float64 accumulation via PyTorch) ──
    try:
        ref_dtype = torch.float64 if dtype == torch.float32 else dtype
        data_ref = data.to(ref_dtype)
        x_ref = x.to(ref_dtype)
        try:
            A_ref = torch.sparse_csr_tensor(
                indptr.to(torch.int64), indices.to(torch.int64), data_ref,
                size=shape, device=device,
            )
            y_ref = torch.sparse.mm(A_ref, x_ref.unsqueeze(1)).squeeze(1).to(dtype)
        except Exception:
            row_ind = torch.repeat_interleave(
                torch.arange(n_rows, device=device, dtype=torch.int64),
                indptr[1:] - indptr[:-1],
            )
            A_ref = torch.sparse_coo_tensor(
                torch.stack([row_ind, indices.to(torch.int64)]),
                data_ref, shape, device=device,
            ).coalesce()
            y_ref = torch.sparse.mm(A_ref, x_ref.unsqueeze(1)).squeeze(1).to(dtype)
    except Exception:
        y_ref = None

    # ── Baseline (use_opt=False) ──
    y_base, base_ms, base_symbolic_ms, base_compute_ms = _timed_spmv(
        prepared, x, warmup, iters, use_opt=False
    )

    # ── Optimised (use_opt=True) ──
    y_opt, opt_ms, symbolic_ms, compute_ms = _timed_spmv(
        prepared, x, warmup, iters, use_opt=True
    )

    # ── PyTorch ──
    pt_ms = None
    try:
        _, pt_ms = _timed_pytorch(data, indices, indptr, x, shape, warmup, iters)
    except Exception:
        pass

    # ── cuSPARSE ──
    cu_ms = None
    try:
        _, cu_ms = _timed_cusparse(data, indices, indptr, x, shape, warmup, iters)
    except Exception:
        pass

    # ── Correctness vs reference ──
    err_base = None
    err_opt = None
    if y_ref is not None and n_rows > 0:
        diff_b = torch.abs(y_base - y_ref).to(torch.float64)
        diff_o = torch.abs(y_opt - y_ref).to(torch.float64)
        tol = (atol + rtol * torch.abs(y_ref).to(torch.float64))
        err_base = float(torch.max(diff_b / tol).item())
        err_opt = float(torch.max(diff_o / tol).item())

    base_ok = err_base is not None and (not math.isnan(err_base)) and err_base <= 1.0
    opt_ok = err_opt is not None and (not math.isnan(err_opt)) and err_opt <= 1.0
    status = "PASS" if opt_ok else "FAIL"

    return {
        "path": path, "shape": shape, "nnz": nnz,
        "base_ms": base_ms, "opt_ms": opt_ms,
        "base_symbolic_ms": base_symbolic_ms,
        "base_compute_ms": base_compute_ms,
        "symbolic_ms": symbolic_ms,
        "compute_ms": compute_ms,
        "op_total_ms": opt_ms,
        "pt_ms": pt_ms, "cu_ms": cu_ms,
        "err_base": err_base, "err_opt": err_opt,
        "base_ok": base_ok, "opt_ok": opt_ok,
        "status": status,
    }


def print_row(r):
    name = os.path.basename(r["path"])[:27]
    n_rows, n_cols = r["shape"]
    print(
        f"{name:<28} {n_rows:>7} {n_cols:>7} {r['nnz']:>10}  "
        f"{_fmt(r['base_ms']):>9} {_fmt(r['opt_ms']):>9} "
        f"{_fmt(r['symbolic_ms']):>9} {_fmt(r['compute_ms']):>9} "
        f"{_fmt(r['pt_ms']):>9} {_fmt(r['cu_ms']):>9}  "
        f"{_spd(r['base_ms'], r['op_total_ms']):>8} "
        f"{_spd(r['pt_ms'], r['op_total_ms']):>8} "
        f"{_spd(r['cu_ms'], r['op_total_ms']):>8}  "
        f"{_err(r['err_base']):>10} {_err(r['err_opt']):>10} {r['status']:>6}"
    )


def run_batch(paths, dtype, index_dtype, warmup, iters):
    results = []
    for p in paths:
        try:
            r = run_one_mtx(p, dtype, index_dtype, warmup, iters)
        except Exception as e:
            print(f"  ERROR on {os.path.basename(p)}: {e}")
            continue
        results.append(r)
        print_row(r)
    return results


def run_all_csv(paths, csv_path, warmup, iters, dtype_filter=None):
    all_rows = []
    dtypes = VALUE_DTYPES if dtype_filter is None else [dtype_filter]
    for dtype in dtypes:
        for idx_dtype in INDEX_DTYPES:
            dname = str(dtype).replace("torch.", "")
            iname = str(idx_dtype).replace("torch.", "")
            print("=" * 170)
            print(f"Value dtype: {dname}  |  Index dtype: {iname}")
            print(
                "Base = FlagSparse baseline (fp64-accum for fp32). "
                "Opt = FlagSparse CSR-Vector (fp32/fp64 native accum, wide tiles, few launches). "
                "Opt(ms) = Sym(ms) CPU wall time + Comp(ms) CUDA event time. "
                "Speedup = Base/Opt or Ref/Opt."
            )
            print(SEP)
            print(HEADER)
            print(SEP)
            results = run_batch(paths, dtype, idx_dtype, warmup, iters)
            print(SEP)
            for r in results:
                n_rows, n_cols = r["shape"]
                all_rows.append({
                    "matrix": os.path.basename(r["path"]),
                    "value_dtype": dname,
                    "index_dtype": iname,
                    "n_rows": n_rows, "n_cols": n_cols, "nnz": r["nnz"],
                    "base_ms": r["base_ms"], "opt_ms": r["opt_ms"],
                    "symbolic_ms": r["symbolic_ms"],
                    "compute_ms": r["compute_ms"],
                    "op_total_ms": r["op_total_ms"],
                    "pt_ms": r["pt_ms"], "cu_ms": r["cu_ms"],
                    "opt_vs_base": r["base_ms"] / r["op_total_ms"] if r["op_total_ms"] and r["op_total_ms"] > 0 else None,
                    "opt_vs_pt": r["pt_ms"] / r["op_total_ms"] if r["pt_ms"] and r["op_total_ms"] and r["op_total_ms"] > 0 else None,
                    "opt_vs_cu": r["cu_ms"] / r["op_total_ms"] if r["cu_ms"] and r["op_total_ms"] and r["op_total_ms"] > 0 else None,
                    "triton_speedup_vs_pytorch": r["pt_ms"] / r["op_total_ms"] if r["pt_ms"] and r["op_total_ms"] and r["op_total_ms"] > 0 else None,
                    "triton_speedup_vs_cusparse": r["cu_ms"] / r["op_total_ms"] if r["cu_ms"] and r["op_total_ms"] and r["op_total_ms"] > 0 else None,
                    "err_base": r["err_base"], "err_opt": r["err_opt"],
                    "status": r["status"],
                })
    fields = [
        "matrix", "value_dtype", "index_dtype",
        "n_rows", "n_cols", "nnz",
        "base_ms", "opt_ms", "symbolic_ms", "compute_ms", "op_total_ms", "pt_ms", "cu_ms",
        "opt_vs_base", "opt_vs_pt", "opt_vs_cu",
        "triton_speedup_vs_pytorch", "triton_speedup_vs_cusparse",
        "err_base", "err_opt", "status",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})
    print(f"\nWrote {len(all_rows)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SpMV opt A/B: baseline vs optimised, with PyTorch/cuSPARSE."
    )
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--csv", type=str, default=None, metavar="FILE",
                        help="Export selected dtype(s) to CSV")
    parser.add_argument("--dtype", default="all",
                        choices=["float32", "float64", "all"])
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()

    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    if not paths:
        print("No .mtx files. Usage: python test_spmv_opt.py <dir/> [--csv out.csv]")
        return

    if args.csv:
        print("=" * 80)
        print("FLAGSPARSE SpMV Optimisation A/B Test - export CSV")
        print("=" * 80)
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  CSV: {args.csv}")
        dtype_map = {"float32": torch.float32, "float64": torch.float64}
        dtype_filter = None if args.dtype == "all" else dtype_map[args.dtype]
        run_all_csv(paths, args.csv, args.warmup, args.iters, dtype_filter)
        return

    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    dtypes = VALUE_DTYPES if args.dtype == "all" else [dtype_map[args.dtype]]
    for dtype in dtypes:
        dname = str(dtype).replace("torch.", "")
        print("=" * 170)
        print(f"FLAGSPARSE SpMV Optimisation A/B Test")
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  dtype: {dname}  |  Files: {len(paths)}")
        print(
            "Base = FlagSparse baseline (fp64-accum for fp32). "
            "Opt = FlagSparse CSR-Vector (fp32/fp64 native accum, wide tiles, few launches). "
            "Opt(ms) = Sym(ms) CPU wall time + Comp(ms) CUDA event time. "
            "Speedup = Base/Opt or Ref/Opt."
        )
        print(SEP)
        print(HEADER)
        print(SEP)
        results = run_batch(paths, dtype, torch.int32, args.warmup, args.iters)
        print(SEP)
        passed = sum(1 for r in results if r["status"] == "PASS")
        print(f"Passed: {passed} / {len(results)}")


if __name__ == "__main__":
    main()
