"""
SpMM alg1 test: compare base vs optimised path with PyTorch and cuSPARSE timings.
Alg1 timings report CPU-wall runtime preprocessing plus CUDA-event compute time.

Usage:
    python tests/test_spmm_opt.py <dir/> --dense-cols 32
    python tests/test_spmm_opt.py <dir/> --csv spmm_opt.csv
"""

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs
import flagsparse.sparse_operations.spmm_csr as spmm_csr_mod

VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPES = [torch.int32]
WARMUP = 10
ITERS = 50
DEFAULT_DENSE_COLS = 32
DEFAULT_SEED = None


def load_mtx_to_csr_torch(file_path, dtype=torch.float32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(file_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

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

    is_pattern = mm_field == "pattern"
    is_symmetric = mm_symmetry in ("symmetric", "hermitian")
    is_skew = mm_symmetry == "skew-symmetric"
    row_maps = [dict() for _ in range(n_rows)]
    for line in data_lines[:nnz]:
        parts = line.split()
        row = int(parts[0]) - 1
        col = int(parts[1]) - 1
        value = 1.0 if is_pattern else float(parts[2])
        if 0 <= row < n_rows and 0 <= col < n_cols:
            row_maps[row][col] = row_maps[row].get(col, 0.0) + value
            if row != col:
                if is_symmetric and 0 <= col < n_rows and 0 <= row < n_cols:
                    row_maps[col][row] = row_maps[col].get(row, 0.0) + value
                elif is_skew and 0 <= col < n_rows and 0 <= row < n_cols:
                    row_maps[col][row] = row_maps[col].get(row, 0.0) - value

    cols_sorted = []
    vals_sorted = []
    indptr_list = [0]
    for row in range(n_rows):
        row_map = row_maps[row]
        for col in sorted(row_map.keys()):
            cols_sorted.append(col)
            vals_sorted.append(row_map[col])
        indptr_list.append(len(cols_sorted))

    data = torch.tensor(vals_sorted, dtype=dtype, device=device)
    indices = torch.tensor(cols_sorted, dtype=torch.int64, device=device)
    indptr = torch.tensor(indptr_list, dtype=torch.int64, device=device)
    return data, indices, indptr, (n_rows, n_cols)


def _timed_spmm_base(data, indices, indptr, B, shape, warmup, iters):
    op = lambda: fs.flagsparse_spmm_csr(data, indices, indptr, B, shape)
    out = op()
    torch.cuda.synchronize()
    for _ in range(warmup):
        out = op()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = op()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / iters


def _timed_spmm_alg1_impl(data, indices, indptr, B, shape, warmup, iters):
    prepared = fs.prepare_spmm_csr_opt_alg1(data, indices, indptr, shape)
    count = max(1, int(iters))
    runtime_prepared = spmm_csr_mod._build_spmm_csr_opt_runtime_symbolic_triton(
        prepared
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(count):
        runtime_prepared = spmm_csr_mod._build_spmm_csr_opt_runtime_symbolic_triton(
            prepared
        )
    torch.cuda.synchronize()
    preprocess_ms = (time.perf_counter() - t0) * 1000.0 / count

    def op():
        out, _ = spmm_csr_mod._triton_spmm_csr_impl_opt_prepared(
            runtime_prepared, B
        )
        return out

    out = op()
    torch.cuda.synchronize()
    for _ in range(warmup):
        out = op()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(count):
        out = op()
    end.record()
    torch.cuda.synchronize()
    compute_ms = start.elapsed_time(end) / count
    return out, preprocess_ms + compute_ms, preprocess_ms, compute_ms


def _timed_spmm_opt(data, indices, indptr, B, shape, warmup, iters):
    return _timed_spmm_alg1_impl(data, indices, indptr, B, shape, warmup, iters)


def _timed_spmm_opt_alg1_preprocess(data, indices, indptr, B, shape, warmup, iters):
    return _timed_spmm_alg1_impl(data, indices, indptr, B, shape, warmup, iters)


def _timed_pytorch(data, indices, indptr, B, shape, warmup, iters):
    device = data.device
    try:
        sparse = torch.sparse_csr_tensor(
            indptr.to(torch.int64),
            indices.to(torch.int64),
            data,
            size=shape,
            device=device,
        )
    except Exception:
        n_rows = int(shape[0])
        row_ind = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        sparse = torch.sparse_coo_tensor(
            torch.stack([row_ind, indices.to(torch.int64)]),
            data,
            shape,
            device=device,
        ).coalesce()
    op = lambda: torch.sparse.mm(sparse, B)
    out = op()
    torch.cuda.synchronize()
    for _ in range(warmup):
        out = op()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = op()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / iters


def _timed_cusparse(data, indices, indptr, B, shape, warmup, iters):
    import cupy as cp
    import cupyx.scipy.sparse as cpx

    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
    ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
    ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
    B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B))
    sparse = cpx.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = sparse @ B_cp
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = sparse @ B_cp
    end.record()
    torch.cuda.synchronize()
    out_cp = sparse @ B_cp
    out = torch.utils.dlpack.from_dlpack(out_cp.toDlpack())
    return out, start.elapsed_time(end) / iters


def _build_reference(data, indices, indptr, B, shape, dtype):
    device = data.device
    ref_dtype = torch.float64 if dtype == torch.float32 else dtype
    sparse = torch.sparse_csr_tensor(
        indptr.to(torch.int64),
        indices.to(torch.int64),
        data.to(ref_dtype),
        size=shape,
        device=device,
    )
    return torch.sparse.mm(sparse, B.to(ref_dtype)).to(dtype)


def _error_ratio(candidate, reference, dtype):
    if dtype == torch.float32:
        atol, rtol = 1e-4, 1e-1
    else:
        atol, rtol = 1e-12, 1e-9
    if candidate.numel() == 0:
        return 0.0
    diff = torch.abs(candidate - reference).to(torch.float64)
    denom = (atol + rtol * torch.abs(reference)).to(torch.float64)
    return float(torch.max(diff / denom).item())


def _fmt(v):
    return "N/A" if v is None else f"{v:.4f}"


def _spd(base, other):
    if base is None or other is None or other <= 0:
        return "N/A"
    return f"{base / other:.2f}x"


def _err(v):
    return "N/A" if v is None else f"{v:.2e}"


HEADER = (
    f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'DenseN':>8}  "
    f"{'Base(ms)':>9} {'Alg1(ms)':>9} {'A1Prep':>9} {'A1Comp':>9} "
    f"{'PT(ms)':>9} {'CU(ms)':>9}  "
    f"{'Base/A1':>8} {'PT/A1':>8} {'CU/A1':>8}  "
    f"{'Err(Base)':>10} {'Err(A1)':>10} {'Status':>6}"
)
SEP = "-" * 205


def _seeded_dense_matrix(shape, dtype, device, seed):
    if seed is None:
        return torch.randn(shape, dtype=dtype, device=device)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return torch.randn(shape, dtype=dtype, device=device)


def run_one_mtx(path, dtype, index_dtype, dense_cols, warmup, iters, seed=None):
    device = torch.device("cuda")
    data, indices, indptr, shape = load_mtx_to_csr_torch(path, dtype=dtype, device=device)
    indices = indices.to(index_dtype)
    n_rows, n_cols = shape
    nnz = data.numel()
    B = _seeded_dense_matrix((n_cols, dense_cols), dtype, device, seed)
    ref = _build_reference(data, indices, indptr, B, shape, dtype)

    y_base, base_ms = _timed_spmm_base(data, indices, indptr, B, shape, warmup, iters)
    y_opt, opt_ms, preprocess_ms, compute_ms = _timed_spmm_opt(
        data, indices, indptr, B, shape, warmup, iters
    )

    pt_ms = None
    try:
        _, pt_ms = _timed_pytorch(data, indices, indptr, B, shape, warmup, iters)
    except Exception:
        pass

    cu_ms = None
    try:
        _, cu_ms = _timed_cusparse(data, indices, indptr, B, shape, warmup, iters)
    except Exception:
        pass

    err_base = _error_ratio(y_base, ref, dtype)
    err_opt = _error_ratio(y_opt, ref, dtype)
    base_ok = err_base <= 1.0
    opt_ok = err_opt <= 1.0
    status = "PASS" if opt_ok else "FAIL"
    return {
        "path": path,
        "shape": shape,
        "nnz": nnz,
        "dense_cols": dense_cols,
        "base_ms": base_ms,
        "opt_ms": opt_ms,
        "alg1_ms": opt_ms,
        "alg1_preprocess_ms": preprocess_ms,
        "alg1_compute_ms": compute_ms,
        "pt_ms": pt_ms,
        "cu_ms": cu_ms,
        "err_base": err_base,
        "err_opt": err_opt,
        "err_alg1": err_opt,
        "base_ok": base_ok,
        "opt_ok": opt_ok,
        "status_alg1": "PASS" if opt_ok else "FAIL",
        "seed": seed,
        "status": status,
    }


def print_row(row):
    name = os.path.basename(row["path"])[:27]
    n_rows, n_cols = row["shape"]
    print(
        f"{name:<28} {n_rows:>7} {n_cols:>7} {row['nnz']:>10} {row['dense_cols']:>8}  "
        f"{_fmt(row['base_ms']):>9} {_fmt(row['alg1_ms']):>9} "
        f"{_fmt(row['alg1_preprocess_ms']):>9} {_fmt(row['alg1_compute_ms']):>9} "
        f"{_fmt(row['pt_ms']):>9} {_fmt(row['cu_ms']):>9}  "
        f"{_spd(row['base_ms'], row['alg1_ms']):>8} "
        f"{_spd(row['pt_ms'], row['alg1_ms']):>8} "
        f"{_spd(row['cu_ms'], row['alg1_ms']):>8}  "
        f"{_err(row['err_base']):>10} {_err(row['err_alg1']):>10} {row['status']:>6}"
    )


def run_batch(paths, dtype, index_dtype, dense_cols, warmup, iters, seed=None):
    results = []
    for path in paths:
        try:
            row = run_one_mtx(path, dtype, index_dtype, dense_cols, warmup, iters, seed=seed)
        except Exception as exc:
            print(f"  ERROR on {os.path.basename(path)}: {exc}")
            continue
        results.append(row)
        print_row(row)
    return results


def run_all_csv(paths, csv_path, dense_cols, warmup, iters, seed=None, value_dtypes=None):
    rows = []
    if value_dtypes is None:
        value_dtypes = VALUE_DTYPES
    for dtype in value_dtypes:
        for index_dtype in INDEX_DTYPES:
            dname = str(dtype).replace("torch.", "")
            iname = str(index_dtype).replace("torch.", "")
            print("=" * 182)
            print(f"Value dtype: {dname}  |  Index dtype: {iname}  |  Dense cols: {dense_cols}")
            print(
                "Base = existing CSR SpMM baseline (fp64-accum for fp32). "
                "Alg1 = bucketed CSR SpMM native path with Triton runtime preprocessing. "
                "Alg1(ms) = A1Prep CPU wall time + A1Comp CUDA event time. "
                "Speedup = reference / Alg1."
            )
            print(SEP)
            print(HEADER)
            print(SEP)
            results = run_batch(paths, dtype, index_dtype, dense_cols, warmup, iters, seed=seed)
            print(SEP)
            for row in results:
                n_rows, n_cols = row["shape"]
                rows.append({
                    "matrix": os.path.basename(row["path"]),
                    "value_dtype": dname,
                    "index_dtype": iname,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "nnz": row["nnz"],
                    "dense_cols": row["dense_cols"],
                    "seed": row["seed"],
                    "base_ms": row["base_ms"],
                    "opt_ms": row["opt_ms"],
                    "alg1_ms": row["alg1_ms"],
                    "alg1_preprocess_ms": row["alg1_preprocess_ms"],
                    "alg1_compute_ms": row["alg1_compute_ms"],
                    "pt_ms": row["pt_ms"],
                    "cu_ms": row["cu_ms"],
                    "opt_vs_base": (row["base_ms"] / row["opt_ms"] if row["opt_ms"] and row["opt_ms"] > 0 else None),
                    "opt_vs_pt": (row["pt_ms"] / row["opt_ms"] if row["pt_ms"] and row["opt_ms"] and row["opt_ms"] > 0 else None),
                    "opt_vs_cu": (row["cu_ms"] / row["opt_ms"] if row["cu_ms"] and row["opt_ms"] and row["opt_ms"] > 0 else None),
                    "base_vs_alg1_speedup": (row["base_ms"] / row["alg1_ms"] if row["alg1_ms"] and row["alg1_ms"] > 0 else None),
                    "torch_vs_alg1_speedup": (row["pt_ms"] / row["alg1_ms"] if row["pt_ms"] and row["alg1_ms"] and row["alg1_ms"] > 0 else None),
                    "cusparse_vs_alg1_speedup": (row["cu_ms"] / row["alg1_ms"] if row["cu_ms"] and row["alg1_ms"] and row["alg1_ms"] > 0 else None),
                    "err_base": row["err_base"],
                    "err_opt": row["err_opt"],
                    "err_alg1": row["err_alg1"],
                    "status_alg1": row["status_alg1"],
                    "status": row["status"],
                })
    fields = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "n_rows",
        "n_cols",
        "nnz",
        "dense_cols",
        "seed",
        "base_ms",
        "opt_ms",
        "alg1_ms",
        "alg1_preprocess_ms",
        "alg1_compute_ms",
        "pt_ms",
        "cu_ms",
        "opt_vs_base",
        "opt_vs_pt",
        "opt_vs_cu",
        "base_vs_alg1_speedup",
        "torch_vs_alg1_speedup",
        "cusparse_vs_alg1_speedup",
        "err_base",
        "err_opt",
        "err_alg1",
        "status_alg1",
        "status",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    print(f"\nWrote {len(rows)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="SpMM alg1: baseline vs optimised, with PyTorch/cuSPARSE timings.")
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--csv", type=str, default=None, metavar="FILE", help="Export selected dtype to CSV")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--dense-cols", type=int, default=DEFAULT_DENSE_COLS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Optional fixed seed for reproducible dense RHS generation")
    args = parser.parse_args()

    paths = []
    for path in args.mtx:
        if os.path.isfile(path) and path.endswith(".mtx"):
            paths.append(path)
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))
    if not paths:
        print("No .mtx files. Usage: python test_spmm_opt.py <dir/> [--csv out.csv]")
        return

    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    dtype = dtype_map[args.dtype]
    if args.csv:
        run_all_csv(paths, args.csv, args.dense_cols, args.warmup, args.iters, seed=args.seed, value_dtypes=[dtype])
        return

    print("=" * 182)
    print("FLAGSPARSE SpMM Alg1 Test")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  dtype: {args.dtype}  |  Dense cols: {args.dense_cols}  |  Files: {len(paths)}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(
        "Base = existing CSR SpMM baseline (fp64-accum for fp32). "
        "Alg1 = bucketed CSR SpMM native path with Triton runtime preprocessing. "
        "Alg1(ms) = A1Prep CPU wall time + A1Comp CUDA event time. "
        "Speedup = reference / Alg1."
    )
    print(SEP)
    print(HEADER)
    print(SEP)
    results = run_batch(paths, dtype, torch.int32, args.dense_cols, args.warmup, args.iters, seed=args.seed)
    print(SEP)
    passed = sum(1 for row in results if row["status"] == "PASS")
    print(f"Passed: {passed} / {len(results)}")


if __name__ == "__main__":
    main()
