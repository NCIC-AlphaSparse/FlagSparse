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

"""Native Blocked-ELL (BELL) SpMM benchmark and correctness script."""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs


VALUE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
INDEX_DTYPES = (torch.int32, torch.int64)
OPS = ("non", "trans", "conj")
SUPPORTED_OPS = ("non",)
ALGS = ("auto", "spmm_bell_base", "base", "all")
TEST_SIZES = ((63, 95, 16), (128, 256, 32), (160, 1024, 48))
WARMUP = 10
ITERS = 50
DEFAULT_MAX_BELL_STORAGE_MB = 2048.0

PERF_FIELDS = [
    "matrix",
    "dtype",
    "index_dtype",
    "op",
    "layout",
    "alg",
    "ref",
    "out_rows",
    "n_rows",
    "n_cols",
    "nnzb",
    "ell_width_blocks",
    "stored_values",
    "padding_ratio",
    "estimated_storage_mb",
    "block_dim",
    "dense_cols",
    "b_stride",
    "c_stride",
    "ms",
    "gpu_ms",
    "process_cpu_ms",
    "torch_bell_ms",
    "cupy_bell_ms",
    "err_vs_ref",
    "err_vs_torch_bell",
    "err_vs_cupy_bell",
    "status",
    "reason",
    "torch_bell_reason",
    "cupy_bell_reason",
]
TIMING_FIELDS = ["process_gpu_ms", "compute_ms"]

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
INDEX_DTYPE_MAP = {"int32": torch.int32, "int64": torch.int64}


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _dtype_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


def _reference_dtype(dtype):
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _reference_tolerance(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1.3e-6, 1e-3
    if dtype in (torch.float64, torch.complex128):
        return 1e-7, 1e-5
    return 1e-6, 1e-5


def _error_ratio(actual, expected, dtype):
    if actual is None or expected is None:
        return None
    atol, rtol = _reference_tolerance(dtype)
    if expected.numel() == 0:
        return 0.0
    diff = torch.abs(actual - expected).to(torch.float64)
    denom = (atol + rtol * torch.abs(expected)).to(torch.float64)
    ratios = diff / denom
    return float(torch.max(ratios).item()) if ratios.numel() else 0.0


def _parse_csv_tokens(value, mapping, option_name):
    tokens = [token.strip().lower() for token in str(value).split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{option_name} must not be empty")
    invalid = [token for token in tokens if token not in mapping]
    if invalid:
        raise ValueError(
            f"unsupported {option_name}: {', '.join(invalid)}; allowed: {', '.join(mapping)}"
        )
    return [mapping[token] for token in tokens]


def _parse_ops(value):
    token = str(value).strip().lower()
    if token == "all":
        return list(OPS)
    ops = [item.strip().lower() for item in token.split(",") if item.strip()]
    invalid = [op for op in ops if op not in OPS]
    if not ops or invalid:
        raise ValueError(f"unsupported --ops: {', '.join(invalid or ops)}")
    return ops


def _parse_algs(value):
    token = str(value or "auto").strip().lower().replace("-", "_")
    algs = [item.strip().lower().replace("-", "_") for item in token.split(",") if item.strip()]
    aliases = {"base": "spmm_bell_base"}
    algs = [aliases.get(alg, alg) for alg in algs]
    invalid = [alg for alg in algs if alg not in ALGS and alg != "spmm_bell_base"]
    if not algs or invalid:
        raise ValueError("unsupported --alg; allowed: auto, all, base, spmm_bell_base")
    return algs


def _expand_algs(algs, op, dtype):
    del dtype
    if op not in SUPPORTED_OPS:
        return []
    out = []
    for alg in algs:
        if alg in ("auto", "all"):
            out.append("spmm_bell_base")
        else:
            out.append(alg)
    deduped = []
    for alg in out:
        if alg not in deduped:
            deduped.append(alg)
    return deduped


def _parse_block_dims(value):
    dims = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        dim = int(token)
        if dim <= 0:
            raise ValueError("--block-dims values must be positive")
        dims.append(dim)
    if not dims:
        raise ValueError("--block-dims must not be empty")
    return dims


def _build_dense_B(K, N, dtype, device, layout):
    B = torch.randn((K, N), dtype=_reference_dtype(dtype), device=device).to(dtype) * 0.25
    if dtype.is_complex:
        B = B + 1j * (torch.randn((K, N), dtype=_reference_dtype(dtype), device=device).to(dtype) * 0.25)
    if layout == "col":
        return B.t().contiguous().t()
    return B.contiguous()


def _zero_value(dtype):
    return 0j if dtype.is_complex else 0.0


def _random_values(shape, dtype, device):
    ref_dtype = _reference_dtype(dtype)
    vals = torch.randn(shape, dtype=ref_dtype, device=device)
    if dtype.is_complex:
        vals = vals + 1j * torch.randn(shape, dtype=ref_dtype, device=device)
    return vals.to(dtype)


def _mtx_value_for_dtype(value, dtype):
    if dtype.is_complex:
        return complex(value)
    return float(value.real if isinstance(value, complex) else value)


def _build_bell_plan(entries, shape, block_dim):
    M, K = shape
    mb = (M + block_dim - 1) // block_dim
    blocks = {}
    for (row, col), value in entries.items():
        brow = int(row) // block_dim
        bcol = int(col) // block_dim
        inner_row = int(row) % block_dim
        inner_col = int(col) % block_dim
        block = blocks.setdefault(
            (brow, bcol),
            None,
        )
        if block is None:
            block = {}
            blocks[(brow, bcol)] = block
        block[(inner_row, inner_col)] = block.get((inner_row, inner_col), 0) + value
    row_blocks = [[] for _ in range(mb)]
    for key in sorted(blocks):
        row_blocks[key[0]].append(key)
    ell_width_blocks = max([len(row) for row in row_blocks] or [0])
    ell_width_blocks = max(1, ell_width_blocks)
    return blocks, row_blocks, mb, ell_width_blocks


def _estimate_bell_storage(entries, shape, dtype, index_dtype, block_dim):
    blocks, row_blocks, mb, ell_width_blocks = _build_bell_plan(entries, shape, block_dim)
    nnzb = len(blocks)
    stored_values = mb * ell_width_blocks * block_dim * block_dim
    index_values = mb * ell_width_blocks
    estimated_bytes = stored_values * _dtype_size(dtype) + index_values * _dtype_size(index_dtype)
    padding_ratio = float(stored_values) / max(1, len(entries))
    return {
        "blocks": blocks,
        "row_blocks": row_blocks,
        "mb": mb,
        "ell_width_blocks": ell_width_blocks,
        "nnzb": nnzb,
        "stored_values": stored_values,
        "estimated_storage_mb": estimated_bytes / (1024.0 * 1024.0),
        "padding_ratio": padding_ratio,
    }


def _entries_to_bell(entries, shape, dtype, index_dtype, block_dim, device, plan=None):
    if plan is None:
        plan = _estimate_bell_storage(entries, shape, dtype, index_dtype, block_dim)
    blocks = plan["blocks"]
    row_blocks = plan["row_blocks"]
    mb = int(plan["mb"])
    ell_width_blocks = int(plan["ell_width_blocks"])
    data = torch.zeros(
        (mb, ell_width_blocks, block_dim, block_dim),
        dtype=dtype,
        device=device,
    )
    indices = torch.full(
        (mb, ell_width_blocks),
        -1,
        dtype=index_dtype,
        device=device,
    )
    for brow, keys in enumerate(row_blocks):
        for slot, key in enumerate(keys):
            indices[brow, slot] = key[1]
            for (inner_row, inner_col), value in blocks[key].items():
                data[brow, slot, inner_row, inner_col] = _mtx_value_for_dtype(value, dtype)
    return data, indices


def _dense_to_entries(dense):
    rows, cols = torch.nonzero(dense != 0, as_tuple=True)
    return {
        (int(row.item()), int(col.item())): dense[row, col].item()
        for row, col in zip(rows, cols)
    }


def _read_mtx_entries(path):
    entries = {}
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split()
        if len(header) < 5 or header[0] != "%%MatrixMarket":
            raise ValueError("invalid MatrixMarket header")
        field = header[3].lower()
        symmetry = header[4].lower()
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        n_rows, n_cols, _nnz = [int(v) for v in line.strip().split()[:3]]
        pattern = field == "pattern"
        complex_field = field == "complex"
        for line in fh:
            if not line.strip() or line.startswith("%"):
                continue
            parts = line.strip().split()
            row = int(parts[0]) - 1
            col = int(parts[1]) - 1
            if pattern:
                value = 1.0
            elif complex_field:
                value = complex(float(parts[2]), float(parts[3]))
            else:
                value = float(parts[2])
            entries[(row, col)] = entries.get((row, col), 0) + value
            if symmetry in ("symmetric", "hermitian") and row != col:
                mirror = value.conjugate() if symmetry == "hermitian" else value
                entries[(col, row)] = entries.get((col, row), 0) + mirror
    return entries, (n_rows, n_cols)


def _make_synthetic_case(M, K, dtype, block_dim, device):
    del block_dim
    p = min(0.25, max(0.06, 32.0 / max(M * K, 1)))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    dense = torch.where(
        mask,
        _random_values((M, K), dtype, device) * 0.125,
        torch.zeros((), dtype=dtype, device=device),
    )
    return "synthetic", _dense_to_entries(dense), (M, K)


def _entries_to_torch_coo(entries, shape, dtype, device):
    if not entries:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        values = torch.empty(0, dtype=dtype, device=device)
        return torch.sparse_coo_tensor(
            torch.stack([empty, empty]), values, size=shape, device=device, dtype=dtype
        ).coalesce()
    rows = torch.tensor([key[0] for key in entries], dtype=torch.int64, device=device)
    cols = torch.tensor([key[1] for key in entries], dtype=torch.int64, device=device)
    values = torch.tensor([_mtx_value_for_dtype(value, dtype) for value in entries.values()], dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), values, size=shape, device=device, dtype=dtype
    ).coalesce()


def _torch_spmm_coo_reference_from_original_coo(entries, B, shape, dtype):
    ref_dtype = _reference_dtype(dtype)
    A = _entries_to_torch_coo(entries, shape, dtype, B.device).to(ref_dtype)
    return torch.sparse.mm(A, B.to(ref_dtype)).to(dtype)


def _cuda_event_benchmark(op, warmup, iters):
    out = None
    for _ in range(max(0, int(warmup))):
        out = op()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    count = max(1, int(iters))
    start.record()
    for _ in range(count):
        out = op()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / count


def _time_flagsparse_bell(data, indices, B, shape, block_dim, alg, op, warmup, iters, timing=False):
    prepared = fs.prepare_spmm_bell_route(
        data,
        indices,
        shape,
        block_dim=block_dim,
        alg=alg,
        op=op,
    )
    out, gpu_ms = _cuda_event_benchmark(
        lambda: fs.flagsparse_spmm_bell_run(prepared, B, alg=alg, op=op),
        warmup,
        iters,
    )
    _meta_out, meta = fs.flagsparse_spmm_bell_run(
        prepared,
        B,
        alg=alg,
        op=op,
        return_meta=True,
        timing=bool(timing),
    )
    process_cpu_ms = float(meta.get("process_cpu_ms", 0.0) or 0.0)
    process_gpu_ms = meta.get("process_gpu_ms") if timing else None
    compute_ms = meta.get("compute_ms") if timing else None
    if timing:
        if process_gpu_ms is None:
            process_gpu_ms = 0.0
        if compute_ms is None:
            compute_ms = gpu_ms
    return {
        "out": out,
        "ms": process_cpu_ms + gpu_ms,
        "gpu_ms": gpu_ms,
        "process_cpu_ms": process_cpu_ms,
        "process_gpu_ms": process_gpu_ms,
        "compute_ms": compute_ms,
        "meta": meta,
    }


def _run_case(matrix_name, entries, shape, dtype, index_dtype, block_dim, dense_cols, layout, alg, op, warmup, iters, timing, max_bell_storage_mb):
    M, K = shape
    device = torch.device("cuda")
    plan = _estimate_bell_storage(entries, shape, dtype, index_dtype, block_dim)
    row = {
        "matrix": matrix_name,
        "dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": op,
        "layout": layout,
        "alg": alg,
        "ref": "torch_spmm_coo_from_original_coo",
        "out_rows": M,
        "n_rows": M,
        "n_cols": K,
        "nnzb": int(plan["nnzb"]),
        "ell_width_blocks": int(plan["ell_width_blocks"]),
        "stored_values": int(plan["stored_values"]),
        "padding_ratio": float(plan["padding_ratio"]),
        "estimated_storage_mb": float(plan["estimated_storage_mb"]),
        "block_dim": block_dim,
        "dense_cols": dense_cols,
        "b_stride": None,
        "c_stride": None,
        "ms": None,
        "gpu_ms": None,
        "process_cpu_ms": 0.0,
        "torch_bell_ms": None,
        "cupy_bell_ms": None,
        "err_vs_ref": None,
        "err_vs_torch_bell": None,
        "err_vs_cupy_bell": None,
        "status": "ERROR",
        "reason": "",
        "torch_bell_reason": "PyTorch has no same-format BELL/Blocked-ELL SpMM baseline",
        "cupy_bell_reason": "CuPy has no same-format BELL/Blocked-ELL SpMM baseline",
    }
    if timing:
        row.update({"process_gpu_ms": None, "compute_ms": None})
    if op not in SUPPORTED_OPS:
        row.update(
            {
                "status": "SKIP",
                "reason": "spmm_bell_base supports op='non' only; trans/conj are reserved",
            }
        )
        return row
    if plan["estimated_storage_mb"] > float(max_bell_storage_mb):
        row.update(
            {
                "status": "SKIP",
                "reason": (
                    "BELL padded storage estimate "
                    f"{plan['estimated_storage_mb']:.1f} MiB exceeds guard "
                    f"{float(max_bell_storage_mb):.1f} MiB"
                ),
            }
        )
        return row
    try:
        data, indices = _entries_to_bell(
            entries, shape, dtype, index_dtype, block_dim, device, plan=plan
        )
        B = _build_dense_B(K, dense_cols, dtype, device, layout)
        ref = _torch_spmm_coo_reference_from_original_coo(entries, B, shape, dtype)
        row["b_stride"] = B.stride(0)
        bell = _time_flagsparse_bell(
            data, indices, B, shape, block_dim, alg, op, warmup, iters, timing=timing
        )
        out = bell["out"]
        err = _error_ratio(out, ref, dtype)
        row.update(
            {
                "c_stride": out.stride(0),
                "ms": bell["ms"],
                "gpu_ms": bell["gpu_ms"],
                "process_cpu_ms": bell["process_cpu_ms"],
                "process_gpu_ms": bell["process_gpu_ms"],
                "compute_ms": bell["compute_ms"],
                "err_vs_ref": err,
                "status": "PASS" if err is not None and err <= 1.0 else "FAIL",
                "reason": "" if err is not None and err <= 1.0 else "correctness check failed",
            }
        )
    except Exception as exc:
        row["reason"] = str(exc)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def _print_notes():
    print("FlagSparse BELL uses native Blocked-ELL arrays; empty slots are indices == -1.")
    print("Accuracy reference: Ref=torch_spmm_coo_from_original_coo builds torch sparse COO from the original matrix entries; this is correctness-only.")
    print("PyTorch/vendor BELL baselines: unavailable unless a real same-format Blocked-ELL API is present; no casting or format fallback is used.")
    print("Timing policy: ms = process_cpu_ms + gpu_ms; BELL SpMM v1 has no process phase.")
    print("Memory guard: oversized BELL padded storage is reported as SKIP before tensor allocation.")


def _print_row(row, timing=False):
    extra = (
        f" {_fmt(row.get('process_gpu_ms')):>9} {_fmt(row.get('compute_ms')):>9}"
        if timing
        else ""
    )
    print(
        f"{os.path.basename(str(row['matrix']))[:28]:<28} {row['op']:<5} {row['alg']:<15} "
        f"{row['block_dim']:>4} {row['out_rows']:>8} {row['n_rows']:>8} {row['n_cols']:>8} "
        f"{row['nnzb']:>8} {row['ell_width_blocks']:>6} {row['estimated_storage_mb']:>8.1f} {row['dense_cols']:>6} "
        f"{_fmt(row['ms']):>9} {_fmt(row['gpu_ms']):>9} {_fmt(row['process_cpu_ms']):>9}"
        f"{extra} {_fmt(row['err_vs_ref'], 2):>10} {row['status']:>6}"
    )
    if row.get("reason"):
        print(f"  reason: {row['reason']}")


def main():
    parser = argparse.ArgumentParser(description="Native Blocked-ELL SpMM benchmark")
    parser.add_argument("inputs", nargs="*", help="MatrixMarket files or directories")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--csv-bell")
    parser.add_argument("--dtypes", default="float32,float64,complex64,complex128")
    parser.add_argument("--index-dtypes", default="int32,int64")
    parser.add_argument("--block-dims", default="2")
    parser.add_argument("--ops", default="non")
    parser.add_argument("--alg", default="auto")
    parser.add_argument("--dense-cols", default="32")
    parser.add_argument("--layout", default="row", choices=("row", "col", "all"))
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument(
        "--max-bell-storage-mb",
        type=float,
        default=DEFAULT_MAX_BELL_STORAGE_MB,
        help="Skip cases whose estimated BELL data+index storage exceeds this MiB guard",
    )
    parser.add_argument("--no-cusparse", action="store_true", help="Accepted for CLI compatibility; no BELL vendor baseline is used")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm PyTorch device is required for native BELL SpMM")
    dtypes = _parse_csv_tokens(args.dtypes, DTYPE_MAP, "--dtypes")
    index_dtypes = _parse_csv_tokens(args.index_dtypes, INDEX_DTYPE_MAP, "--index-dtypes")
    block_dims = _parse_block_dims(args.block_dims)
    ops = _parse_ops(args.ops)
    algs = _parse_algs(args.alg)
    dense_cols_values = [int(v.strip()) for v in str(args.dense_cols).split(",") if v.strip()]
    layouts = ("row", "col") if args.layout == "all" else (args.layout,)
    fields = PERF_FIELDS + (TIMING_FIELDS if args.timing else [])
    rows = []
    _print_notes()
    writer = None
    fh = None
    if args.csv_bell:
        fh = open(args.csv_bell, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
    try:
        cases = []
        if args.synthetic:
            for M, K, N in TEST_SIZES:
                cases.append(("synthetic", None, (M, K), N))
        for input_path in args.inputs:
            paths = sorted(glob.glob(os.path.join(input_path, "*.mtx"))) if os.path.isdir(input_path) else [input_path]
            for path in paths:
                cases.append((os.path.basename(path), path, None, None))
        if not cases:
            raise ValueError("provide --synthetic or at least one .mtx input")
        for dtype in dtypes:
            for index_dtype in index_dtypes:
                for op in ops:
                    print("-" * 132)
                    print(f"Value dtype: {_dtype_name(dtype)} | Index dtype: {_dtype_name(index_dtype)} | op: {op}")
                    print("-" * 132)
                    header = (
                        f"{'Matrix':<28} {'Op':<5} {'Alg':<15} {'BDim':>4} {'Out':>8} {'Rows':>8} "
                        f"{'Cols':>8} {'NNZB':>8} {'ELLW':>6} {'MiB':>8} {'DCols':>6} {'ms':>9} {'gpu_ms':>9} "
                        f"{'cpu_ms':>9}"
                        + (f" {'gpu_proc':>9} {'compute':>9}" if args.timing else "")
                        + f" {'Err':>10} {'Status':>6}"
                    )
                    print(header)
                    for block_dim in block_dims:
                        for case_name, path, synthetic_shape, synthetic_dense_cols in cases:
                            if path is None:
                                entries_name, entries, shape = _make_synthetic_case(
                                    synthetic_shape[0], synthetic_shape[1], dtype, block_dim, torch.device("cuda")
                                )
                                matrix_name = entries_name
                                default_dense_cols = synthetic_dense_cols
                            else:
                                entries, shape = _read_mtx_entries(path)
                                matrix_name = case_name
                                default_dense_cols = dense_cols_values[0]
                            for dense_cols in (dense_cols_values if path is not None else [default_dense_cols]):
                                for layout in layouts:
                                    for alg in _expand_algs(algs, op, dtype) or ["spmm_bell_base"]:
                                        row = _run_case(
                                            matrix_name,
                                            entries,
                                            shape,
                                            dtype,
                                            index_dtype,
                                            block_dim,
                                            dense_cols,
                                            layout,
                                            alg,
                                            op,
                                            args.warmup,
                                            args.iters,
                                            args.timing,
                                            args.max_bell_storage_mb,
                                        )
                                        rows.append(row)
                                        _print_row(row, timing=args.timing)
                                        if writer is not None:
                                            writer.writerow({field: row.get(field) for field in fields})
                                        if args.fail_fast and row["status"] in ("FAIL", "ERROR"):
                                            raise SystemExit(1)
    finally:
        if fh is not None:
            fh.close()
    return rows


if __name__ == "__main__":
    main()
