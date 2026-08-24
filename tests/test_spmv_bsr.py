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

"""Native BSR SpMV benchmark and correctness script."""

import argparse
import csv
import glob
import math
import os
import sys
import warnings
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except ImportError:
    cp = None
    cpx_sparse = None


VALUE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
INDEX_DTYPES = (torch.int32, torch.int64)
OPS = ("non", "trans", "conj")
SUPPORTED_OPS = OPS
ALGS = ("base", "blockrow_reduce", "compare", "auto")
TEST_SIZES = ((64, 96), (160, 1024), (128, 256))
DEFAULT_BLOCK_DIMS = (4,)
WARMUP = 10
ITERS = 50
PYTORCH_BSR_TRANSPOSE_UNSUPPORTED = (
    "PyTorch CUDA BSR transpose/conjugate-transpose matvec is unsupported: "
    "A.transpose(0, 1) produces SparseBsc, and SparseBsc @ Strided is not implemented"
)


def _cupy_bsr_unavailable_reason():
    if cp is None or cpx_sparse is None:
        return "CuPy/cupyx.scipy.sparse is not available"
    if not hasattr(cpx_sparse, "bsr_matrix"):
        return "CuPy cupyx.scipy.sparse has no bsr_matrix baseline"
    return None


def _print_baseline_notes(run_cusparse=True):
    print(
        "FlagSparse BSR follows AlphaSparse/cuSPARSE-style padded block-grid semantics; native output is padded and correctness checks slice back to the logical output length."
    )
    print(
        "Accuracy reference: Ref=spmv-coo expands the same BSR arrays to COO and runs sparse matvec; PyTorch BSR is a baseline, not the reference."
    )
    print(
        "PyTorch baseline: PT(ms) uses torch.sparse_bsr_tensor only when shape is divisible by block_dim; "
        "PTPad(ms) uses padded shape and slices back to the logical output length for diagnostics."
    )
    print(
        "PyTorch BSR trans/conj baseline: unsupported on CUDA because BSR transpose becomes SparseBsc; PT(ms)/PTPad(ms)=N/A for those ops."
    )
    if run_cusparse:
        reason = _cupy_bsr_unavailable_reason()
        if reason:
            print(f"CuPy baseline: unavailable for BSR ({reason}); CU(ms)=N/A.")


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
INDEX_DTYPE_MAP = {"int32": torch.int32, "int64": torch.int64}


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
    token = "non" if value is None else str(value).strip().lower()
    if token == "all":
        return list(OPS)
    ops = [item.strip().lower() for item in token.split(",") if item.strip()]
    invalid = [op for op in ops if op not in OPS]
    if not ops or invalid:
        raise ValueError(f"unsupported --ops: {', '.join(invalid or ops)}")
    return ops


def _parse_algs(value):
    token = "base" if value is None else str(value).strip().lower().replace("-", "_")
    algs = [item.strip().lower().replace("-", "_") for item in token.split(",") if item.strip()]
    invalid = [alg for alg in algs if alg not in ALGS]
    if not algs or invalid:
        raise ValueError(f"unsupported --alg: {', '.join(invalid or algs)}")
    return algs


def _expand_algs(algs, op):
    expanded = []
    for alg in algs:
        if alg == "compare":
            expanded.extend(["base", "blockrow_reduce"] if op == "non" else ["base"])
        elif alg == "auto":
            expanded.append("blockrow_reduce" if op == "non" else "base")
        elif alg == "blockrow_reduce" and op != "non":
            expanded.append("base")
        else:
            expanded.append(alg)
    out = []
    for alg in expanded:
        if alg not in out:
            out.append(alg)
    return out


def _parse_block_dims(value):
    token = str(value or "4").strip().lower()
    if token == "auto":
        return ["auto"]
    dims = []
    for item in token.split(","):
        item = item.strip()
        if not item:
            continue
        dim = int(item)
        if dim <= 1:
            raise ValueError("--block-dims values must be greater than 1")
        dims.append(dim)
    if not dims:
        raise ValueError("--block-dims must not be empty")
    return dims


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


def _mtx_value_for_dtype(raw_value, dtype):
    if dtype in (torch.complex64, torch.complex128):
        return complex(raw_value)
    return float(raw_value.real if isinstance(raw_value, complex) else raw_value)


def _zero_value(dtype):
    return 0j if dtype in (torch.complex64, torch.complex128) else 0.0


def _choose_auto_block_dim(entries, shape):
    nnz = max(1, len(entries))
    best = None
    for block_dim in (16, 8, 4, 2):
        blocks = {
            (int(row) // block_dim, int(col) // block_dim)
            for row, col in entries.keys()
        }
        stored = len(blocks) * block_dim * block_dim
        if stored <= 2.0 * nnz:
            return block_dim
        if best is None or stored < best[0]:
            best = (stored, block_dim)
    return best[1] if best is not None else 2


def _padded_shape(shape, block_dim):
    block_dim = int(block_dim)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    return (
        ((n_rows + block_dim - 1) // block_dim) * block_dim,
        ((n_cols + block_dim - 1) // block_dim) * block_dim,
    )


def _op_transposes(op):
    return str(op).lower() in ("trans", "conj")


def _logical_x_size(shape, op):
    return int(shape[0]) if _op_transposes(op) else int(shape[1])


def _padded_x_size(shape, block_dim, op):
    padded_rows, padded_cols = _padded_shape(shape, block_dim)
    return padded_rows if _op_transposes(op) else padded_cols


def _logical_out_size(shape, op):
    return int(shape[1]) if _op_transposes(op) else int(shape[0])


def _padded_out_size(shape, block_dim, op):
    padded_rows, padded_cols = _padded_shape(shape, block_dim)
    return padded_cols if _op_transposes(op) else padded_rows


def _pad_vector(x, length):
    length = int(length)
    if x.numel() == length:
        return x.contiguous()
    if x.numel() > length:
        raise ValueError(f"cannot pad vector of length {x.numel()} to shorter length {length}")
    out = torch.zeros(length, dtype=x.dtype, device=x.device)
    out[: x.numel()].copy_(x)
    return out


def _entries_to_bsr_torch(entries, shape, dtype, index_dtype, block_dim, device):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    block_dim = int(block_dim)
    blocks = {}
    for (row, col), value in entries.items():
        brow = int(row) // block_dim
        bcol = int(col) // block_dim
        inner_row = int(row) % block_dim
        inner_col = int(col) % block_dim
        block = blocks.setdefault(
            (brow, bcol),
            [_zero_value(dtype) for _ in range(block_dim * block_dim)],
        )
        block[inner_row * block_dim + inner_col] += _mtx_value_for_dtype(value, dtype)
    n_block_rows = (n_rows + block_dim - 1) // block_dim
    rows = [[] for _ in range(n_block_rows)]
    for key in sorted(blocks):
        rows[key[0]].append(key)
    data_values = []
    indices_values = []
    indptr_values = [0]
    for row_blocks in rows:
        for key in row_blocks:
            indices_values.append(key[1])
            data_values.extend(blocks[key])
        indptr_values.append(len(indices_values))
    data = torch.tensor(data_values, dtype=dtype, device=device)
    data = data.reshape(-1, block_dim, block_dim).contiguous()
    indices = torch.tensor(indices_values, dtype=index_dtype, device=device)
    indptr = torch.tensor(indptr_values, dtype=index_dtype, device=device)
    return data, indices.contiguous(), indptr.contiguous()


def _dense_to_bsr(dense, index_dtype, block_dim):
    rows, cols = dense.nonzero(as_tuple=True)
    entries = {
        (int(row.item()), int(col.item())): dense[row, col].item()
        for row, col in zip(rows, cols)
    }
    return _entries_to_bsr_torch(
        entries,
        tuple(dense.shape),
        dense.dtype,
        index_dtype,
        block_dim,
        dense.device,
    )


def _bsr_block_rows(indptr):
    counts = indptr[1:].to(torch.int64) - indptr[:-1].to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(indptr.numel() - 1, dtype=torch.int64, device=indptr.device),
        counts,
    )


def _bsr_to_torch_coo(data, indices, indptr, shape, block_dim):
    block_rows = _bsr_block_rows(indptr)
    nnzb = int(data.shape[0])
    if nnzb == 0:
        empty = torch.empty(0, dtype=torch.int64, device=data.device)
        return torch.sparse_coo_tensor(
            torch.stack([empty, empty]),
            data.reshape(-1),
            size=shape,
            device=data.device,
            dtype=data.dtype,
        ).coalesce()
    local = torch.arange(block_dim * block_dim, dtype=torch.int64, device=data.device)
    inner_rows = local // block_dim
    inner_cols = local % block_dim
    rows = block_rows[:, None] * block_dim + inner_rows[None, :]
    cols = indices.to(torch.int64)[:, None] * block_dim + inner_cols[None, :]
    values = data.reshape(nnzb, block_dim * block_dim)
    mask = (rows < int(shape[0])) & (cols < int(shape[1])) & (values != 0)
    rows = rows[mask]
    cols = cols[mask]
    values = values[mask]
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        values,
        size=shape,
        device=data.device,
        dtype=data.dtype,
    ).coalesce()


def _spmv_coo_reference(data, indices, indptr, x, shape, dtype, block_dim, op):
    ref_dtype = _reference_dtype(dtype)
    A = _bsr_to_torch_coo(
        data.to(ref_dtype),
        indices,
        indptr,
        shape,
        block_dim,
    )
    if op == "trans":
        A = A.transpose(0, 1)
    elif op == "conj":
        A = A.conj().transpose(0, 1)
    return torch.sparse.mm(A, x.to(ref_dtype).unsqueeze(1)).squeeze(1).to(dtype)


def _error_stats(actual, expected, atol, rtol):
    if expected.numel() == 0:
        return {
            "ratio": 0.0,
            "max_abs": 0.0,
            "max_rel": 0.0,
            "index": 0,
            "actual": None,
            "expected": None,
        }
    diff = torch.abs(actual - expected).to(torch.float64)
    denom = atol + rtol * torch.abs(expected).to(torch.float64)
    ratio_values = diff / denom
    flat_ratio = ratio_values.reshape(-1)
    max_pos = int(torch.argmax(flat_ratio).item())
    expected_abs = torch.abs(expected).to(torch.float64).reshape(-1)
    rel = diff.reshape(-1) / torch.clamp(expected_abs, min=1.0e-30)
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    return {
        "ratio": float(flat_ratio[max_pos].item()),
        "max_abs": float(diff.reshape(-1)[max_pos].item()),
        "max_rel": float(rel[max_pos].item()),
        "index": max_pos,
        "actual": actual_flat[max_pos].detach().cpu().item(),
        "expected": expected_flat[max_pos].detach().cpu().item(),
    }


def _allclose_error_ratio(actual, expected, atol, rtol):
    return _error_stats(actual, expected, atol, rtol)["ratio"]


def _cuda_event_benchmark(op, warmup, iters):
    out = None
    count = max(1, int(iters))
    for _ in range(max(0, int(warmup))):
        out = op()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(count):
        out = op()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / count


def _time_flagsparse_bsr(data, indices, indptr, x, shape, block_dim, op, alg, warmup, iters, timing=False):
    use_opt = alg == "blockrow_reduce"
    prepared = fs.prepare_spmv_bsr(data, indices, indptr, shape, block_dim, op=op)
    x_for_bsr = _pad_vector(x, _padded_x_size(shape, block_dim, op))
    out, gpu_ms = _cuda_event_benchmark(
        lambda: fs.flagsparse_spmv_bsr(x=x_for_bsr, prepared=prepared, use_opt=use_opt),
        warmup,
        iters,
    )
    process_gpu_ms = 0.0
    compute_ms = gpu_ms
    if timing:
        try:
            _timed_out, meta = fs.flagsparse_spmv_bsr(
                x=x_for_bsr,
                prepared=prepared,
                use_opt=use_opt,
                return_meta=True,
            )
            process_gpu_ms = meta.get("process_gpu_ms") or 0.0
            compute_ms = meta.get("compute_ms") or gpu_ms
        except Exception:
            process_gpu_ms = 0.0
            compute_ms = gpu_ms
    return {
        "out": out,
        "ms": gpu_ms,
        "gpu_ms": gpu_ms,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": process_gpu_ms if timing else None,
        "compute_ms": compute_ms if timing else None,
    }


def _apply_pytorch_op(A, x, op):
    if op == "trans":
        A = A.transpose(0, 1)
    elif op == "conj":
        A = A.conj().transpose(0, 1)
    return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)


def _time_pytorch(data, indices, indptr, x, shape, block_dim, op, warmup, iters):
    if _op_transposes(op):
        return None, PYTORCH_BSR_TRANSPOSE_UNSUPPORTED, None
    if int(shape[0]) % int(block_dim) != 0 or int(shape[1]) % int(block_dim) != 0:
        return (
            None,
            "PyTorch BSR baseline requires both matrix dimensions to be divisible by block_dim",
            None,
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse BSR tensor support is in beta state.*",
            category=UserWarning,
        )
        A = torch.sparse_bsr_tensor(
            indptr,
            indices,
            data,
            size=shape,
            device=data.device,
            dtype=data.dtype,
        )
    fn = lambda: _apply_pytorch_op(A, x, op)
    out, ms = _cuda_event_benchmark(fn, warmup, iters)
    return ms, None, out


def _time_pytorch_padded(data, indices, indptr, x, shape, block_dim, op, warmup, iters):
    if _op_transposes(op):
        return None, PYTORCH_BSR_TRANSPOSE_UNSUPPORTED, None
    padded_shape = _padded_shape(shape, block_dim)
    padded_x = _pad_vector(x, padded_shape[0] if _op_transposes(op) else padded_shape[1])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse BSR tensor support is in beta state.*",
            category=UserWarning,
        )
        A = torch.sparse_bsr_tensor(
            indptr,
            indices,
            data,
            size=padded_shape,
            device=data.device,
            dtype=data.dtype,
        )
    fn = lambda: _apply_pytorch_op(A, padded_x, op)
    out, ms = _cuda_event_benchmark(fn, warmup, iters)
    return ms, None, out


def _time_cusparse(data, indices, indptr, x, shape, block_dim, op, warmup, iters):
    if cp is None or cpx_sparse is None:
        return None, "CuPy/cupyx.scipy.sparse is not available"
    if not hasattr(cpx_sparse, "bsr_matrix"):
        return None, "CuPy cupyx.scipy.sparse has no bsr_matrix baseline"
    if data.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        return None, f"unsupported cuSPARSE dtype: {_dtype_name(data.dtype)}"

    def run_with_index_dtype(index_dtype, fallback_note=None):
        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
        ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(index_dtype)))
        ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr.to(index_dtype)))
        x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
        A = cpx_sparse.bsr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
        if op == "trans":
            fn = lambda: A.T @ x_cp
        elif op == "conj":
            fn = lambda: A.conj().T @ x_cp
        else:
            fn = lambda: A @ x_cp
        for _ in range(max(0, int(warmup))):
            _ = fn()
        cp.cuda.runtime.deviceSynchronize()
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        count = max(1, int(iters))
        start.record()
        for _ in range(count):
            _ = fn()
        end.record()
        end.synchronize()
        return cp.cuda.get_elapsed_time(start, end) / count, fallback_note

    try:
        return run_with_index_dtype(indices.dtype)
    except Exception as exc:
        if indices.dtype == torch.int32 and indptr.dtype == torch.int32:
            raise
        try:
            return run_with_index_dtype(
                torch.int32,
                f"CuPy BSR failed with {_dtype_name(indices.dtype)} indices; used int32 baseline fallback: {exc}",
            )
        except Exception:
            raise exc


def _fmt(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _spd(base, other):
    if base is None or other is None or other <= 0:
        return "N/A"
    return f"{base / other:.2f}x"


def _status(ok):
    return "PASS" if ok else "FAIL"


def _header(timing=False):
    split = f" {'ProcGPU':>9} {'Compute':>9}" if timing else ""
    return (
        f"{'Matrix':<28} {'Alg':>15} {'Op':>5} {'BDim':>5} {'Ref':>8} {'Out':>7} {'PadOut':>7} {'PadRows':>7} {'Rows':>7} {'Cols':>7} {'NNZB':>9} {'Pad':>7}  "
        f"{'BSR(ms)':>9} {'BSRGPU':>9} {'CPUProc':>9}{split} "
        f"{'PT(ms)':>9} {'PTPad':>9} {'PTPMode':>9} {'CU(ms)':>9}  {'BSR/PT':>8} {'BSR/CU':>8} "
        f"{'BSRErr':>10} {'PTPadErr':>10} {'B/PT':>10} {'B/PTPad':>10} {'Status':>6}"
    )


def _sep(timing=False):
    return "-" * (176 if timing else 156)


def _print_row(row, timing=False):
    name = str(row["matrix"])[:27]
    if len(str(row["matrix"])) > 27:
        name += "..."
    split = (
        f" {_fmt(row.get('process_gpu_ms')):>9} {_fmt(row.get('compute_ms')):>9}"
        if timing
        else ""
    )
    print(
        f"{name:<28} {row.get('algorithm', 'base')[:15]:>15} {row['op']:>5} {row['block_dim']:>5} {row['reference']:>8} {row['out_size']:>7} {row['padded_out_size']:>7} {row['pad_rows']:>7} {row['n_rows']:>7} {row['n_cols']:>7} {row['nnzb']:>9} {row['padding_ratio']:>7}  "
        f"{_fmt(row['bsr_ms']):>9} {_fmt(row['bsr_gpu_ms']):>9} {_fmt(row['process_cpu_ms']):>9}{split} "
        f"{_fmt(row['pytorch_ms']):>9} {_fmt(row['pytorch_padded_ms']):>9} {str(row.get('pytorch_padded_mode') or 'N/A')[:9]:>9} {_fmt(row['cusparse_ms']):>9}  "
        f"{_spd(row['pytorch_ms'], row['bsr_ms']):>8} {_spd(row['cusparse_ms'], row['bsr_ms']):>8} "
        f"{_fmt_err(row['err']):>10} {_fmt_err(row.get('pytorch_padded_err')):>10} "
        f"{_fmt_err(row.get('bsr_vs_pytorch_err')):>10} {_fmt_err(row.get('bsr_vs_pytorch_padded_err')):>10} {row['status']:>6}"
    )
    error = row.get("error")
    if error:
        print(f"  error: {str(error)[:240]}")
    if row.get("pytorch_error"):
        if row["pytorch_error"] != PYTORCH_BSR_TRANSPOSE_UNSUPPORTED:
            print(f"  pt: {str(row['pytorch_error'])[:240]}")
    if row.get("pytorch_padded_error"):
        if row["pytorch_padded_error"] != PYTORCH_BSR_TRANSPOSE_UNSUPPORTED:
            print(f"  pt_padded: {str(row['pytorch_padded_error'])[:240]}")
    if row.get("status") == "FAIL":
        print(
            "  debug: "
            f"max_abs={_fmt_err(row.get('max_abs_err'))}, "
            f"max_rel={_fmt_err(row.get('max_rel_err'))}, "
            f"max_idx={row.get('max_err_index')}, "
            f"actual={row.get('actual_at_max')}, "
            f"expected={row.get('expected_at_max')}, "
            f"pytorch_err={_fmt_err(row.get('pytorch_err'))}, "
            f"pytorch_padded_err={_fmt_err(row.get('pytorch_padded_err'))}, "
            f"bsr_vs_pytorch_err={_fmt_err(row.get('bsr_vs_pytorch_err'))}, "
            f"bsr_vs_pytorch_padded_err={_fmt_err(row.get('bsr_vs_pytorch_padded_err'))}"
        )


def _base_row(
    matrix_name,
    dtype,
    index_dtype,
    op,
    alg,
    shape,
    data,
    block_dim,
    logical_nnz=None,
    status="ERROR",
):
    nnzb = int(data.shape[0])
    stored_nnz = int(data.numel())
    logical_nnz = max(1, int(logical_nnz if logical_nnz is not None else stored_nnz))
    logical_out = _logical_out_size(shape, op) if op in SUPPORTED_OPS else "UNSUP"
    padded_out = _padded_out_size(shape, block_dim, op) if op in SUPPORTED_OPS else "UNSUP"
    return {
        "matrix": matrix_name,
        "value_dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": op,
        "algorithm": alg,
        "reference": "spmv-coo",
        "block_dim": int(block_dim),
        "out_size": logical_out,
        "padded_out_size": padded_out,
        "pad_rows": (max(0, int(padded_out) - int(logical_out)) if op in SUPPORTED_OPS else "UNSUP"),
        "n_rows": int(shape[0]),
        "n_cols": int(shape[1]),
        "nnzb": nnzb,
        "logical_nnz": logical_nnz,
        "stored_nnz": stored_nnz,
        "padding_ratio": f"{stored_nnz / logical_nnz:.2f}",
        "bsr_ms": None,
        "bsr_gpu_ms": None,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": None,
        "compute_ms": None,
        "pytorch_ms": None,
        "pytorch_error": None,
        "pytorch_padded_ms": None,
        "pytorch_padded_error": None,
        "pytorch_padded_err": None,
        "pytorch_padded_mode": None,
        "bsr_vs_pytorch_err": None,
        "bsr_vs_pytorch_padded_err": None,
        "cusparse_ms": None,
        "cusparse_error": None,
        "bsr_err": None,
        "err": None,
        "max_abs_err": None,
        "max_rel_err": None,
        "max_err_index": None,
        "actual_at_max": None,
        "expected_at_max": None,
        "pytorch_err": None,
        "status": status,
        "error": None,
    }


def _skip_row(matrix_name, dtype, index_dtype, op, shape, block_dim, logical_nnz, error):
    try:
        block_dim_value = int(block_dim)
    except (TypeError, ValueError):
        block_dim_value = block_dim
    try:
        logical_out = _logical_out_size(shape, op) if op in SUPPORTED_OPS else "UNSUP"
        padded_out = _padded_out_size(shape, block_dim_value, op) if op in SUPPORTED_OPS else "UNSUP"
        pad_rows = max(0, int(padded_out) - int(logical_out)) if op in SUPPORTED_OPS else "UNSUP"
    except Exception:
        logical_out = "SKIP"
        padded_out = "SKIP"
        pad_rows = "SKIP"
    return {
        "matrix": matrix_name,
        "value_dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": op,
        "algorithm": "base",
        "reference": "spmv-coo",
        "block_dim": block_dim_value,
        "out_size": logical_out,
        "padded_out_size": padded_out,
        "pad_rows": pad_rows,
        "n_rows": int(shape[0]),
        "n_cols": int(shape[1]),
        "nnzb": "SKIP",
        "logical_nnz": max(1, int(logical_nnz)),
        "stored_nnz": "SKIP",
        "padding_ratio": "SKIP",
        "bsr_ms": None,
        "bsr_gpu_ms": None,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": None,
        "compute_ms": None,
        "pytorch_ms": None,
        "pytorch_error": None,
        "pytorch_padded_ms": None,
        "pytorch_padded_error": None,
        "pytorch_padded_err": None,
        "pytorch_padded_mode": None,
        "bsr_vs_pytorch_err": None,
        "bsr_vs_pytorch_padded_err": None,
        "cusparse_ms": None,
        "cusparse_error": None,
        "bsr_err": None,
        "err": None,
        "max_abs_err": None,
        "max_rel_err": None,
        "max_err_index": None,
        "actual_at_max": None,
        "expected_at_max": None,
        "pytorch_err": None,
        "status": "SKIP",
        "error": error,
    }


def _run_one_case(
    data,
    indices,
    indptr,
    shape,
    dtype,
    index_dtype,
    op,
    alg,
    matrix_name,
    block_dim,
    warmup,
    iters,
    timing=False,
    run_cusparse=True,
    logical_nnz=None,
):
    data = data.contiguous()
    indices = indices.to(index_dtype).contiguous()
    indptr = indptr.to(index_dtype).contiguous()
    row = _base_row(
        matrix_name,
        dtype,
        index_dtype,
        op,
        alg,
        shape,
        data,
        block_dim,
        logical_nnz=logical_nnz,
    )
    row["process_gpu_ms"] = 0.0 if timing else None
    if op not in SUPPORTED_OPS:
        row["status"] = "SKIP"
        row["error"] = "unsupported BSR SpMV op"
        return row
    x = _random_values((_logical_x_size(shape, op),), dtype, data.device)
    atol, rtol = _reference_tolerance(dtype)
    try:
        bsr = _time_flagsparse_bsr(data, indices, indptr, x, shape, block_dim, op, alg, warmup, iters, timing=timing)
    except Exception as exc:
        row["error"] = f"flagsparse_spmv_bsr failed: {exc}"
        return row
    row.update(
        {
            "bsr_ms": bsr["ms"],
            "bsr_gpu_ms": bsr["gpu_ms"],
            "process_cpu_ms": bsr["process_cpu_ms"],
            "process_gpu_ms": bsr["process_gpu_ms"],
            "compute_ms": bsr["compute_ms"],
        }
    )
    try:
        y_ref = _spmv_coo_reference(data, indices, indptr, x, shape, dtype, block_dim, op)
        row["padded_out_size"] = int(bsr["out"].numel())
        logical_out = _logical_out_size(shape, op)
        row["out_size"] = logical_out
        row["pad_rows"] = max(0, int(bsr["out"].numel()) - int(logical_out))
        y_bsr = bsr["out"][: logical_out]
        stats = _error_stats(y_bsr, y_ref, atol, rtol)
        err = stats["ratio"]
        row.update(
            {
                "err": err,
                "bsr_err": err,
                "max_abs_err": stats["max_abs"],
                "max_rel_err": stats["max_rel"],
                "max_err_index": stats["index"],
                "actual_at_max": stats["actual"],
                "expected_at_max": stats["expected"],
            }
        )
    except Exception as exc:
        row["error"] = f"reference failed after BSR run: {exc}"
        return row
    pytorch_out = None
    try:
        row["pytorch_ms"], row["pytorch_error"], pytorch_out = _time_pytorch(
            data,
            indices,
            indptr,
            x,
            shape,
            block_dim,
            op,
            warmup,
            iters,
        )
        if pytorch_out is not None:
            row["pytorch_err"] = _allclose_error_ratio(pytorch_out, y_ref, atol, rtol)
            row["bsr_vs_pytorch_err"] = _allclose_error_ratio(
                y_bsr, pytorch_out, atol, rtol
            )
    except Exception as exc:
        row["pytorch_error"] = str(exc)
    padded_shape = _padded_shape(shape, block_dim)
    if (
        pytorch_out is not None
        and int(padded_shape[0]) == int(shape[0])
        and int(padded_shape[1]) == int(shape[1])
    ):
        row["pytorch_padded_ms"] = row["pytorch_ms"]
        row["pytorch_padded_error"] = None
        row["pytorch_padded_err"] = row["pytorch_err"]
        row["bsr_vs_pytorch_padded_err"] = row["bsr_vs_pytorch_err"]
        row["pytorch_padded_mode"] = "same_as_pt"
    else:
        try:
            (
                row["pytorch_padded_ms"],
                row["pytorch_padded_error"],
                pytorch_padded_out,
            ) = _time_pytorch_padded(
                data,
                indices,
                indptr,
                x,
                shape,
                block_dim,
                op,
                warmup,
                iters,
            )
            row["pytorch_padded_mode"] = "padded_shape"
            if pytorch_padded_out is not None:
                pytorch_padded_logical = pytorch_padded_out[: logical_out]
                row["pytorch_padded_err"] = _allclose_error_ratio(
                    pytorch_padded_logical, y_ref, atol, rtol
                )
                row["bsr_vs_pytorch_padded_err"] = _allclose_error_ratio(
                    y_bsr, pytorch_padded_logical, atol, rtol
                )
        except Exception as exc:
            row["pytorch_padded_error"] = str(exc)
            row["pytorch_padded_mode"] = "padded_shape"
    if run_cusparse:
        try:
            row["cusparse_ms"], row["cusparse_error"] = _time_cusparse(
                data,
                indices,
                indptr,
                x,
                shape,
                block_dim,
                op,
                warmup,
                iters,
            )
        except Exception as exc:
            row["cusparse_error"] = str(exc)
    ok = (not math.isnan(err)) and err <= 1.0
    row["status"] = _status(ok)
    row["error"] = None if ok else "correctness check failed"
    return row


def load_mtx_entries(path):
    """{(row, col): value} + shape via the C-accelerated scipy reader
    (see tests/mtx_fast.py); the former pure-Python parser took minutes on
    large SuiteSparse matrices."""
    from mtx_fast import load_entries

    return load_entries(path)


def _resolve_block_dims(block_dims, entries, shape):
    if block_dims == ["auto"]:
        return [_choose_auto_block_dim(entries, shape)]
    return block_dims


def run_synthetic(value_dtypes=None, index_dtypes=None, block_dims=None, ops=None, algs=None, warmup=WARMUP, iters=ITERS, timing=False, run_cusparse=True):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    value_dtypes = VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    block_dims = list(DEFAULT_BLOCK_DIMS) if block_dims is None else block_dims
    ops = SUPPORTED_OPS if ops is None else ops
    algs = ["base"] if algs is None else algs
    print("=" * 140)
    print("FLAGSPARSE SpMV BSR BENCHMARK (native BSR Triton)")
    print("=" * 140)
    print("Timing policy: bsr_ms = process_cpu_ms + bsr_gpu_ms; BSR construction is setup.")
    _print_baseline_notes(run_cusparse=run_cusparse)
    for dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for block_dim in block_dims:
                if block_dim == "auto":
                    block_dim = 4
                for op in ops:
                    op_algs = _expand_algs(algs, op)
                    print(_sep(timing))
                    print(f"dtype: {_dtype_name(dtype)} | index_dtype: {_dtype_name(index_dtype)} | block_dim: {block_dim} | op: {op} | alg: {','.join(op_algs)}")
                    print(_sep(timing))
                    print(_header(timing))
                    print(_sep(timing))
                    for m, n in TEST_SIZES:
                        dense = _random_values((m, n), dtype, device)
                        dense *= (torch.rand(m, n, device=device) < 0.1).to(dtype=dtype)
                        logical_nnz = int(torch.count_nonzero(dense).item())
                        data, indices, indptr = _dense_to_bsr(dense, index_dtype, int(block_dim))
                        for alg in op_algs:
                            row = _run_one_case(
                                data,
                                indices,
                                indptr,
                                (m, n),
                                dtype,
                                index_dtype,
                                op,
                                alg,
                                f"{m}x{n}",
                                int(block_dim),
                                warmup,
                                iters,
                                timing=timing,
                                run_cusparse=run_cusparse,
                                logical_nnz=logical_nnz,
                            )
                            _print_row(row, timing=timing)
                    print(_sep(timing))
                    print()


def run_csv(mtx_paths, csv_path, value_dtypes=None, index_dtypes=None, block_dims=None, ops=None, algs=None, warmup=WARMUP, iters=ITERS, timing=False, run_cusparse=True, fail_fast=False):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    value_dtypes = VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    block_dims = list(DEFAULT_BLOCK_DIMS) if block_dims is None else block_dims
    ops = SUPPORTED_OPS if ops is None else ops
    algs = ["base"] if algs is None else algs
    rows = []
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "op",
        "algorithm",
        "reference",
        "block_dim",
        "out_size",
        "padded_out_size",
        "pad_rows",
        "n_rows",
        "n_cols",
        "nnzb",
        "logical_nnz",
        "stored_nnz",
        "padding_ratio",
        "bsr_ms",
        "bsr_gpu_ms",
        "process_cpu_ms",
        "process_gpu_ms",
        "compute_ms",
        "pytorch_ms",
        "pytorch_error",
        "pytorch_err",
        "pytorch_padded_ms",
        "pytorch_padded_error",
        "pytorch_padded_err",
        "pytorch_padded_mode",
        "bsr_vs_pytorch_err",
        "bsr_vs_pytorch_padded_err",
        "max_abs_err",
        "max_rel_err",
        "max_err_index",
        "actual_at_max",
        "expected_at_max",
        "cusparse_ms",
        "cusparse_error",
        "bsr_err",
        "err",
        "status",
        "error",
    ]
    if not timing:
        fieldnames = [
            field
            for field in fieldnames
            if field not in ("process_gpu_ms", "compute_ms")
        ]
    csv_parent = Path(csv_path).parent
    if str(csv_parent) not in ("", "."):
        csv_parent.mkdir(parents=True, exist_ok=True)
    # Stream each row to disk as it is produced.  The BSR sweeps are the ones
    # that outrun --timeout, and a killed process used to leave an empty CSV
    # because every row was buffered until the very end.
    csv_handle = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_handle, fieldnames=fieldnames, extrasaction="ignore"
    )
    csv_writer.writeheader()
    csv_handle.flush()

    def _emit(row):
        rows.append(row)
        csv_writer.writerow(
            {key: ("" if value is None else value) for key, value in row.items()}
        )
        # flush is enough here: the failure mode is the process being killed,
        # not the node going down, so the OS still owns the buffered bytes.
        csv_handle.flush()

    _print_baseline_notes(run_cusparse=run_cusparse)
    for dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for op in ops:
                op_algs = _expand_algs(algs, op)
                print(_sep(timing))
                print(f"Value dtype: {_dtype_name(dtype)} | Index dtype: {_dtype_name(index_dtype)} | op: {op} | alg: {','.join(op_algs)}")
                print(_sep(timing))
                print(_header(timing))
                print(_sep(timing))
                for path in mtx_paths:
                    try:
                        entries, shape = load_mtx_entries(path)
                        resolved_block_dims = _resolve_block_dims(block_dims, entries, shape)
                        for block_dim in resolved_block_dims:
                            data, indices, indptr = _entries_to_bsr_torch(
                                entries, shape, dtype, index_dtype, int(block_dim), device
                            )
                            for alg in op_algs:
                                row = _run_one_case(
                                    data,
                                    indices,
                                    indptr,
                                    shape,
                                    dtype,
                                    index_dtype,
                                    op,
                                    alg,
                                    os.path.basename(path),
                                    int(block_dim),
                                    warmup,
                                    iters,
                                    timing=timing,
                                    run_cusparse=run_cusparse,
                                    logical_nnz=len(entries),
                                )
                                if fail_fast and row.get("status") == "ERROR":
                                    raise RuntimeError(row.get("error") or "BSR SpMV case failed")
                                _emit(row)
                                _print_row(row, timing=timing)
                    except Exception as exc:
                        if fail_fast:
                            raise
                        row = {
                            "matrix": os.path.basename(path),
                            "value_dtype": _dtype_name(dtype),
                            "index_dtype": _dtype_name(index_dtype),
                            "op": op,
                            "algorithm": "base",
                            "reference": "spmv-coo",
                            "block_dim": "ERR",
                            "out_size": "ERR",
                            "padded_out_size": "ERR",
                            "pad_rows": "ERR",
                            "n_rows": "ERR",
                            "n_cols": "ERR",
                            "nnzb": "ERR",
                            "logical_nnz": "ERR",
                            "stored_nnz": "ERR",
                            "padding_ratio": "ERR",
                            "bsr_ms": None,
                            "bsr_gpu_ms": None,
                            "process_cpu_ms": None,
                            "process_gpu_ms": None,
                            "compute_ms": None,
                            "pytorch_ms": None,
                            "pytorch_error": None,
                            "pytorch_padded_ms": None,
                            "pytorch_padded_error": None,
                            "pytorch_padded_err": None,
                            "pytorch_padded_mode": None,
                            "bsr_vs_pytorch_err": None,
                            "bsr_vs_pytorch_padded_err": None,
                            "cusparse_ms": None,
                            "cusparse_error": None,
                            "bsr_err": None,
                            "err": None,
                            "status": "ERROR",
                            "error": str(exc),
                        }
                        _emit(row)
                        _print_row(row, timing=timing)
                print(_sep(timing))
    csv_handle.close()
    print(f"Wrote {len(rows)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Native BSR SpMV benchmark/test.")
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--csv-bsr", type=str, default=None, metavar="FILE")
    parser.add_argument("--dtypes", default="float32,float64,complex64,complex128")
    parser.add_argument("--index-dtypes", default="int32,int64")
    parser.add_argument("--block-dims", default="4")
    parser.add_argument("--ops", default="non")
    parser.add_argument("--alg", default="base", help="BSR algorithm: base, blockrow_reduce, auto, compare, or comma-separated values")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--no-cusparse", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    try:
        value_dtypes = _parse_csv_tokens(args.dtypes, DTYPE_MAP, "--dtypes")
        index_dtypes = _parse_csv_tokens(args.index_dtypes, INDEX_DTYPE_MAP, "--index-dtypes")
        block_dims = _parse_block_dims(args.block_dims)
        ops = _parse_ops(args.ops)
        algs = _parse_algs(args.alg)
    except ValueError as exc:
        parser.error(str(exc))
    if args.synthetic:
        run_synthetic(
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            block_dims=block_dims,
            ops=ops,
            algs=algs,
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
            run_cusparse=not args.no_cusparse,
        )
        return
    paths = []
    for path in args.mtx:
        if os.path.isfile(path) and path.endswith(".mtx"):
            paths.append(path)
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))
    if args.csv_bsr:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-bsr")
            return
        run_csv(
            paths,
            args.csv_bsr,
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            block_dims=block_dims,
            ops=ops,
            algs=algs,
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
            run_cusparse=not args.no_cusparse,
            fail_fast=args.fail_fast,
        )
        return
    if not paths:
        print("No .mtx files. Use --synthetic or --csv-bsr with inputs.")
        return
    run_csv(
        paths,
        "spmv_bsr_results.csv",
        value_dtypes=value_dtypes,
        index_dtypes=index_dtypes,
        block_dims=block_dims,
        ops=ops,
        algs=algs,
        warmup=args.warmup,
        iters=args.iters,
        timing=args.timing,
        run_cusparse=not args.no_cusparse,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
