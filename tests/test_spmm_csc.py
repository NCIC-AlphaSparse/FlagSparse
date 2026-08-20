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

"""Native CSC SpMM benchmark and correctness script."""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import torch

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except ImportError:
    cp = None
    cpx_sparse = None

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs


VALUE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
INDEX_DTYPES = (torch.int32, torch.int64)
OPS = ("non", "trans", "conj")
SUPPORTED_OPS = ("non",)
ALGS = ("auto", "spmm_csc_base", "base", "all")
TEST_SIZES = ((64, 96, 16), (160, 1024, 32), (128, 256, 48))
WARMUP = 10
ITERS = 50

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
    "nnz",
    "dense_cols",
    "b_stride",
    "c_stride",
    "ms",
    "gpu_ms",
    "process_cpu_ms",
    "cusparse_ms",
    "cusparse_vs_alg_speedup",
    "err_vs_ref",
    "err_vs_cusparse",
    "status",
    "reason",
    "cusparse_reason",
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


def _ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


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
    aliases = {"base": "spmm_csc_base"}
    algs = [aliases.get(alg, alg) for alg in algs]
    invalid = [alg for alg in algs if alg not in ALGS and alg != "spmm_csc_base"]
    if not algs or invalid:
        raise ValueError("unsupported --alg; allowed: auto, all, base, spmm_csc_base")
    return algs


def _expand_algs(algs, op, dtype):
    del dtype
    if op not in SUPPORTED_OPS:
        return []
    out = []
    for alg in algs:
        if alg in ("auto", "all"):
            out.append("spmm_csc_base")
        else:
            out.append(alg)
    deduped = []
    for alg in out:
        if alg not in deduped:
            deduped.append(alg)
    return deduped


def _normalize_layout_name(layout):
    token = str(layout).strip().lower()
    if token in ("row", "row_major", "row-major", "c", "c_order", "auto", "default"):
        return "row"
    if token in ("col", "column", "col_major", "column-major", "f", "fortran"):
        return "col"
    raise ValueError("layout must be one of: row, col, all")


def _layout_names(value):
    token = str(value).strip().lower()
    if token == "all":
        return ["row", "col"]
    return [_normalize_layout_name(token)]


def _materialize_dense_layout(tensor, layout):
    layout = _normalize_layout_name(layout)
    if layout == "row":
        return tensor.contiguous()
    out = torch.empty_strided(
        tuple(tensor.shape),
        (1, max(1, int(tensor.shape[0]))),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    out.copy_(tensor)
    return out


def _stride_string(tensor):
    return "x".join(str(int(v)) for v in tensor.stride())


def _random_values(shape, dtype, device):
    if dtype in (torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.complex64:
        return torch.complex(
            torch.randn(shape, dtype=torch.float32, device=device),
            torch.randn(shape, dtype=torch.float32, device=device),
        )
    if dtype == torch.complex128:
        return torch.complex(
            torch.randn(shape, dtype=torch.float64, device=device),
            torch.randn(shape, dtype=torch.float64, device=device),
        )
    raise TypeError(f"unsupported dtype: {dtype}")


def _zero_value(dtype):
    return 0j if dtype in (torch.complex64, torch.complex128) else 0.0


def _mtx_value_for_dtype(raw_value, dtype):
    if dtype in (torch.complex64, torch.complex128):
        return complex(raw_value)
    return float(raw_value.real if isinstance(raw_value, complex) else raw_value)


def _entries_to_csc(entries, shape, dtype, index_dtype, device):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    col_entries = [[] for _ in range(n_cols)]
    for (row, col), value in entries.items():
        if value == 0:
            continue
        col_entries[int(col)].append((int(row), _mtx_value_for_dtype(value, dtype)))
    data_values = []
    indices_values = []
    indptr_values = [0]
    for values in col_entries:
        for row, value in sorted(values, key=lambda item: item[0]):
            indices_values.append(row)
            data_values.append(value)
        indptr_values.append(len(indices_values))
    data = torch.tensor(data_values, dtype=dtype, device=device).contiguous()
    indices = torch.tensor(indices_values, dtype=index_dtype, device=device).contiguous()
    indptr = torch.tensor(indptr_values, dtype=index_dtype, device=device).contiguous()
    return data, indices, indptr


def _dense_to_csc(dense, index_dtype):
    rows, cols = torch.nonzero(dense != 0, as_tuple=True)
    entries = {
        (int(row.item()), int(col.item())): dense[row, col].item()
        for row, col in zip(rows, cols)
    }
    return _entries_to_csc(entries, tuple(dense.shape), dense.dtype, index_dtype, dense.device)


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


def _make_synthetic_case(M, K, dtype, index_dtype, device):
    p = min(0.25, max(0.06, 32.0 / max(M * K, 1)))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    dense = torch.where(
        mask,
        _random_values((M, K), dtype, device) * 0.125,
        torch.zeros((), dtype=dtype, device=device),
    )
    data, indices, indptr = _dense_to_csc(dense, index_dtype)
    return "synthetic", data, indices, indptr, (M, K)


def _csc_to_torch_coo(data, indices, indptr, shape):
    n_cols = int(shape[1])
    counts = indptr[1:].to(torch.int64) - indptr[:-1].to(torch.int64)
    cols = torch.repeat_interleave(
        torch.arange(n_cols, dtype=torch.int64, device=data.device),
        counts,
    )
    rows = indices.to(torch.int64)
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        data,
        size=shape,
        device=data.device,
        dtype=data.dtype,
    ).coalesce()


def _torch_spmm_coo_reference(data, indices, indptr, B, shape, dtype):
    ref_dtype = _reference_dtype(dtype)
    A = _csc_to_torch_coo(data.to(ref_dtype), indices, indptr, shape)
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


def _time_flagsparse_csc(data, indices, indptr, B, shape, alg, op, warmup, iters, timing=False):
    prepared = fs.prepare_spmm_csc_route(data, indices, indptr, shape, alg=alg, op=op)
    out, gpu_ms = _cuda_event_benchmark(
        lambda: fs.flagsparse_spmm_csc_run(prepared, B, alg=alg, op=op),
        warmup,
        iters,
    )
    _meta_out, meta = fs.flagsparse_spmm_csc_run(
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
    }


def _cupy_csc_unavailable_reason():
    if cp is None or cpx_sparse is None:
        return "CuPy cupyx.scipy.sparse is not available"
    if not hasattr(cpx_sparse, "csc_matrix"):
        return "CuPy cupyx.scipy.sparse has no csc_matrix baseline"
    return None


def _time_cusparse_csc(data, indices, indptr, B, shape, warmup, iters):
    reason = _cupy_csc_unavailable_reason()
    if reason:
        return None, reason, None
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
    ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices))
    ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
    B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B))
    A = cpx_sparse.csc_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
    for _ in range(max(0, int(warmup))):
        out_cp = A @ B_cp
    cp.cuda.runtime.deviceSynchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    count = max(1, int(iters))
    start.record()
    for _ in range(count):
        out_cp = A @ B_cp
    end.record()
    end.synchronize()
    out = torch.utils.dlpack.from_dlpack(out_cp.toDlpack())
    return cp.cuda.get_elapsed_time(start, end) / count, None, out


def _run_case(
    matrix_name,
    data,
    indices,
    indptr,
    shape,
    dtype,
    index_dtype,
    dense_cols,
    layout,
    alg,
    op,
    warmup,
    iters,
    timing,
    run_cusparse,
):
    B = _materialize_dense_layout(
        _random_values((int(shape[1]), int(dense_cols)), dtype, data.device) * 0.125,
        layout,
    )
    ref = _torch_spmm_coo_reference(data, indices, indptr, B, shape, dtype)
    row = {
        "matrix": matrix_name,
        "dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": op,
        "layout": layout,
        "alg": alg,
        "ref": "torch_spmm_coo",
        "out_rows": int(shape[0]),
        "n_rows": int(shape[0]),
        "n_cols": int(shape[1]),
        "nnz": int(data.numel()),
        "dense_cols": int(dense_cols),
        "b_stride": _stride_string(B),
        "c_stride": "",
        "ms": None,
        "gpu_ms": None,
        "process_cpu_ms": 0.0,
        "cusparse_ms": None,
        "cusparse_vs_alg_speedup": None,
        "err_vs_ref": None,
        "err_vs_cusparse": None,
        "status": "ERROR",
        "reason": "",
        "cusparse_reason": "",
        "process_gpu_ms": None,
        "compute_ms": None,
    }
    try:
        csc = _time_flagsparse_csc(
            data, indices, indptr, B, shape, alg, op, warmup, iters, timing=timing
        )
        row.update(
            {
                "ms": csc["ms"],
                "gpu_ms": csc["gpu_ms"],
                "process_cpu_ms": csc["process_cpu_ms"],
                "process_gpu_ms": csc["process_gpu_ms"],
                "compute_ms": csc["compute_ms"],
                "c_stride": _stride_string(csc["out"]),
                "err_vs_ref": _error_ratio(csc["out"], ref, dtype),
            }
        )
        row["status"] = "PASS" if row["err_vs_ref"] is not None and row["err_vs_ref"] <= 1.0 else "FAIL"
        if row["status"] == "FAIL":
            row["reason"] = "correctness check failed"
    except Exception as exc:
        row["reason"] = str(exc)
    if row["status"] != "ERROR" and run_cusparse:
        try:
            cu_ms, cu_reason, cu_out = _time_cusparse_csc(
                data, indices, indptr, B, shape, warmup, iters
            )
            row["cusparse_ms"] = cu_ms
            row["cusparse_reason"] = cu_reason or ""
            if cu_out is not None:
                row["err_vs_cusparse"] = _error_ratio(cu_out, ref, dtype)
        except Exception as exc:
            row["cusparse_reason"] = str(exc)
    row["cusparse_vs_alg_speedup"] = _ratio(row["cusparse_ms"], row["ms"])
    return row


def _resolve_input_paths(input_paths):
    paths = []
    for path in input_paths:
        if os.path.isfile(path) and path.lower().endswith(".mtx"):
            paths.append(os.path.abspath(path))
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))
    return paths


def _print_notes(run_cusparse):
    print("FlagSparse CSC SpMM v1 supports native op=non only; trans/conj are reserved and reported as SKIP.")
    print("Accuracy reference: Ref=torch_spmm_coo expands the same CSC arrays to COO and runs torch.sparse.mm; this is correctness-only, not the FlagSparse compute path.")
    print("PyTorch CSC SpMM baseline: unavailable; torch.sparse.mm documents CSC @ Dense as unsupported, so no PyTorch CSC fallback is used.")
    if run_cusparse:
        reason = _cupy_csc_unavailable_reason()
        if reason:
            print(f"CuPy CSC baseline: unavailable ({reason}); CU(ms)=N/A.")
        else:
            print("CuPy CSC baseline: CU(ms) uses cupyx.scipy.sparse.csc_matrix @ dense with construction outside timing.")
    else:
        print("CuPy CSC baseline disabled by --no-cusparse; CU(ms)=N/A.")
    print("Timing policy: ms = process_cpu_ms + gpu_ms; CSC SpMM v1 has no process phase.")


def _print_row(row, timing=False):
    print(
        f"{row['matrix']:<28} {row['dtype']:<10} {row['index_dtype']:<5} {row['op']:<5} {row['layout']:<4} {row['alg']:<14} "
        f"{row['n_rows']:>7} {row['n_cols']:>7} {row['nnz']:>9} {row['dense_cols']:>5} "
        f"{_fmt(row['ms']):>9} {_fmt(row['gpu_ms']):>9} {_fmt(row['process_cpu_ms']):>9} "
        f"{_fmt(row['cusparse_ms']):>9} {_fmt(row['cusparse_vs_alg_speedup'], 2):>8} "
        f"{_fmt(row['err_vs_ref'], 2):>10} {_fmt(row['err_vs_cusparse'], 2):>10} {row['status']:>6}"
        + (
            f" {_fmt(row.get('process_gpu_ms')):>9} {_fmt(row.get('compute_ms')):>9}"
            if timing
            else ""
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--csv-csc", type=str, default=None, metavar="FILE")
    parser.add_argument("--dtypes", default="float32,float64,complex64,complex128")
    parser.add_argument("--index-dtypes", default="int32,int64")
    parser.add_argument("--ops", default="non")
    parser.add_argument("--alg", default="auto")
    parser.add_argument("--dense-cols", type=int, default=32)
    parser.add_argument("--layout", default="row")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--no-cusparse", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native CSC SpMM benchmark")
    dtypes = _parse_csv_tokens(args.dtypes, DTYPE_MAP, "--dtypes")
    index_dtypes = _parse_csv_tokens(args.index_dtypes, INDEX_DTYPE_MAP, "--index-dtypes")
    ops = _parse_ops(args.ops)
    algs = _parse_algs(args.alg)
    layouts = _layout_names(args.layout)
    run_cusparse = not args.no_cusparse
    _print_notes(run_cusparse)

    fields = PERF_FIELDS + (TIMING_FIELDS if args.timing else [])
    rows = []
    if args.csv_csc:
        Path(args.csv_csc).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    fh = None
    if args.csv_csc:
        fh = open(args.csv_csc, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

    try:
        print("-" * 150)
        print(
            f"{'Matrix':<28} {'DType':<10} {'Index':<5} {'Op':<5} {'Lay':<4} {'Alg':<14} "
            f"{'Rows':>7} {'Cols':>7} {'NNZ':>9} {'DCols':>5} "
            f"{'MS':>9} {'GPU':>9} {'CPUProc':>9} {'CU':>9} {'CU/Alg':>8} "
            f"{'Err':>10} {'CUErr':>10} {'Status':>6}"
            + (f" {'GPUProc':>9} {'Compute':>9}" if args.timing else "")
        )
        print("-" * 150)
        device = torch.device("cuda")
        cases = []
        if args.synthetic:
            for M, K, dense_cols in TEST_SIZES:
                cases.append(("synthetic", None, (M, K), dense_cols))
        for path in _resolve_input_paths(args.mtx):
            cases.append((os.path.basename(path), path, None, args.dense_cols))
        if not cases:
            raise ValueError("provide --synthetic or at least one .mtx input")
        for dtype in dtypes:
            for index_dtype in index_dtypes:
                for case_name, path, synthetic_shape, dense_cols in cases:
                    if path is None:
                        data, indices, indptr, shape = _make_synthetic_case(
                            synthetic_shape[0], synthetic_shape[1], dtype, index_dtype, device
                        )[1:]
                    else:
                        entries, shape = _read_mtx_entries(path)
                        data, indices, indptr = _entries_to_csc(
                            entries, shape, dtype, index_dtype, device
                        )
                    for op in ops:
                        expanded_algs = _expand_algs(algs, op, dtype)
                        if not expanded_algs:
                            row = {field: None for field in fields}
                            row.update(
                                {
                                    "matrix": case_name,
                                    "dtype": _dtype_name(dtype),
                                    "index_dtype": _dtype_name(index_dtype),
                                    "op": op,
                                    "layout": "",
                                    "alg": "",
                                    "ref": "torch_spmm_coo",
                                    "out_rows": int(shape[0]),
                                    "n_rows": int(shape[0]),
                                    "n_cols": int(shape[1]),
                                    "nnz": int(data.numel()),
                                    "dense_cols": int(dense_cols),
                                    "process_cpu_ms": 0.0,
                                    "cusparse_reason": "",
                                    "status": "SKIP",
                                    "reason": "spmm_csc trans/conj are reserved but not implemented in v1",
                                }
                            )
                            rows.append(row)
                            _print_row(row, timing=args.timing)
                            if writer:
                                writer.writerow({field: row.get(field) for field in fields})
                                fh.flush()
                            continue
                        for layout in layouts:
                            for alg in expanded_algs:
                                row = _run_case(
                                    case_name,
                                    data,
                                    indices,
                                    indptr,
                                    shape,
                                    dtype,
                                    index_dtype,
                                    dense_cols,
                                    layout,
                                    alg,
                                    op,
                                    args.warmup,
                                    args.iters,
                                    args.timing,
                                    run_cusparse,
                                )
                                rows.append(row)
                                _print_row(row, timing=args.timing)
                                if writer:
                                    writer.writerow({field: row.get(field) for field in fields})
                                    fh.flush()
                                if args.fail_fast and row["status"] in ("FAIL", "ERROR"):
                                    raise RuntimeError(row.get("reason") or "case failed")
    finally:
        if fh is not None:
            fh.close()
    failures = sum(1 for row in rows if row.get("status") in ("FAIL", "ERROR"))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
