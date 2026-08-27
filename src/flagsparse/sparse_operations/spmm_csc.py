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

"""Native CSC SpMM kernels and route helpers."""

from dataclasses import dataclass

from ._common import *

import triton
import triton.language as tl


SUPPORTED_SPMM_CSC_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

SPMM_CSC_OP_NON = 0
SPMM_CSC_OP_TRANS = 1
SPMM_CSC_OP_CONJ_TRANS = 2
SPMM_CSC_OP_NAMES = {
    SPMM_CSC_OP_NON: "non",
    SPMM_CSC_OP_TRANS: "trans",
    SPMM_CSC_OP_CONJ_TRANS: "conj",
}
SPMM_CSC_SUPPORTED_OP_NAMES = ("non", "trans", "conj")
_SPMM_CSC_OP_NAME_TO_CODE = {name: code for code, name in SPMM_CSC_OP_NAMES.items()}

SPMM_CSC_ALG_BASE = "spmm_csc_base"
_SPMM_CSC_RESERVED_OP_MESSAGE = "spmm_csc_base supports op='non', 'trans', and 'conj'"


class SpmmCscAlgorithmUnavailable(RuntimeError):
    """Raised when a requested CSC SpMM route is unavailable."""


@dataclass(frozen=True)
class SpmmCscAlgorithm:
    name: str
    display_name: str
    supported_ops: tuple[str, ...]
    supported_dtypes: tuple
    run: object


def _normalize_spmm_csc_op(op=None, transpose=False):
    if op is None:
        return SPMM_CSC_OP_TRANS if bool(transpose) else SPMM_CSC_OP_NON
    if isinstance(op, str):
        token = op.strip().lower()
        if token not in _SPMM_CSC_OP_NAME_TO_CODE:
            raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
        return _SPMM_CSC_OP_NAME_TO_CODE[token]
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj") from exc
    if op_code not in SPMM_CSC_OP_NAMES:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
    return op_code


def _spmm_csc_op_to_name(op):
    return SPMM_CSC_OP_NAMES[_normalize_spmm_csc_op(op)]


def _spmm_csc_op_transposes(op):
    return _normalize_spmm_csc_op(op) in (
        SPMM_CSC_OP_TRANS,
        SPMM_CSC_OP_CONJ_TRANS,
    )


def _ensure_spmm_csc_supported_op(op_code):
    op_name = _spmm_csc_op_to_name(op_code)
    if op_name not in SPMM_CSC_SUPPORTED_OP_NAMES:
        raise ValueError(_SPMM_CSC_RESERVED_OP_MESSAGE)


def _normalize_spmm_csc_alg(alg):
    token = "auto" if alg is None else str(alg).strip().lower().replace("-", "_")
    if token in ("auto", "base", "csc_base", "spmm_csc_base"):
        return "auto" if token == "auto" else SPMM_CSC_ALG_BASE
    raise ValueError("unsupported CSC SpMM algorithm; supported: auto, spmm_csc_base")


def _normalize_spmm_csc_index_fallback_policy(index_fallback_policy):
    policy = str(index_fallback_policy).lower()
    if policy not in ("auto", "strict"):
        raise ValueError("index_fallback_policy must be 'auto' or 'strict'")
    return policy


class PreparedCscSpmm:
    """Prepared CSC metadata for native SpMM routes."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "nnz",
        "block_n",
        "block_n_override",
        "block_nnz",
        "max_segments",
        "col_lengths",
        "max_col_nnz",
        "op",
        "alg",
        "index_fallback_policy",
        "index_fallback_applied",
        "index_fallback_reason",
    )

    def __init__(
        self,
        *,
        data,
        kernel_indices,
        kernel_indptr,
        shape,
        n_rows,
        n_cols,
        block_n,
        block_nnz,
        max_segments,
        max_col_nnz,
        col_lengths=None,
        op="non",
        alg="auto",
        index_fallback_policy="auto",
        index_fallback_applied=False,
        index_fallback_reason=None,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.nnz = int(data.numel())
        self.block_n = int(block_n) if block_n is not None else 0
        self.block_n_override = block_n is not None
        self.block_nnz = int(block_nnz)
        self.max_segments = int(max_segments)
        if col_lengths is None:
            col_lengths = kernel_indptr[1:] - kernel_indptr[:-1]
        self.col_lengths = col_lengths
        self.max_col_nnz = int(max_col_nnz)
        self.op = _spmm_csc_op_to_name(op)
        self.alg = _normalize_spmm_csc_alg(alg)
        self.index_fallback_policy = str(index_fallback_policy).lower()
        self.index_fallback_applied = bool(index_fallback_applied)
        self.index_fallback_reason = index_fallback_reason


@triton.jit
def _spmm_csc_non_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    n_cols,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    SEG: tl.constexpr,
):
    col = tl.program_id(0)
    pid_n = tl.program_id(1)
    if col >= n_cols:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    offs = start + SEG * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    vals = tl.load(data_ptr + offs, mask=mask, other=0.0)
    b_vals = tl.load(
        b_ptr + col * stride_bk + offs_n * stride_bn,
        mask=mask_n,
        other=0.0,
    )
    tl.atomic_add(
        c_ptr + rows[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        vals[:, None] * b_vals[None, :],
        mask=mask[:, None] & mask_n[None, :],
    )


@triton.jit
def _spmm_csc_non_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    b_ri_ptr,
    c_ri_ptr,
    n_cols,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    stride_cr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    SEG: tl.constexpr,
):
    col = tl.program_id(0)
    pid_n = tl.program_id(1)
    if col >= n_cols:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    offs = start + SEG * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    a_re = tl.load(data_ri_ptr + offs * 2, mask=mask, other=0.0)
    a_im = tl.load(data_ri_ptr + offs * 2 + 1, mask=mask, other=0.0)
    b_re = tl.load(
        b_ri_ptr + col * stride_bk + offs_n * stride_bn,
        mask=mask_n,
        other=0.0,
    )
    b_im = tl.load(
        b_ri_ptr + col * stride_bk + offs_n * stride_bn + stride_br,
        mask=mask_n,
        other=0.0,
    )
    prod_re = a_re[:, None] * b_re[None, :] - a_im[:, None] * b_im[None, :]
    prod_im = a_re[:, None] * b_im[None, :] + a_im[:, None] * b_re[None, :]
    tl.atomic_add(
        c_ri_ptr + rows[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        prod_re,
        mask=mask[:, None] & mask_n[None, :],
    )
    tl.atomic_add(
        c_ri_ptr
        + rows[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
        + stride_cr,
        prod_im,
        mask=mask[:, None] & mask_n[None, :],
    )


@triton.jit
def _spmm_csc_trans_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    n_cols,
    n_dense_cols,
    stride_bm,
    stride_bn,
    stride_ck,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    SEG: tl.constexpr,
):
    col = tl.program_id(0)
    pid_n = tl.program_id(1)
    if col >= n_cols:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    offs = start + SEG * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    vals = tl.load(data_ptr + offs, mask=mask, other=0.0)
    b_vals = tl.load(
        b_ptr + rows[:, None] * stride_bm + offs_n[None, :] * stride_bn,
        mask=mask[:, None] & mask_n[None, :],
        other=0.0,
    )
    acc = tl.sum(vals[:, None] * b_vals, axis=0)
    tl.atomic_add(
        c_ptr + col * stride_ck + offs_n * stride_cn,
        acc,
        mask=mask_n,
    )


@triton.jit
def _spmm_csc_trans_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    b_ri_ptr,
    c_ri_ptr,
    n_cols,
    n_dense_cols,
    stride_bm,
    stride_bn,
    stride_br,
    stride_ck,
    stride_cn,
    stride_cr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    SEG: tl.constexpr,
    CONJ: tl.constexpr,
):
    col = tl.program_id(0)
    pid_n = tl.program_id(1)
    if col >= n_cols:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    offs = start + SEG * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    a_re = tl.load(data_ri_ptr + offs * 2, mask=mask, other=0.0)
    a_im_raw = tl.load(data_ri_ptr + offs * 2 + 1, mask=mask, other=0.0)
    a_im = a_im_raw
    if CONJ:
        a_im = -a_im_raw
    b_re = tl.load(
        b_ri_ptr + rows[:, None] * stride_bm + offs_n[None, :] * stride_bn,
        mask=mask[:, None] & mask_n[None, :],
        other=0.0,
    )
    b_im = tl.load(
        b_ri_ptr + rows[:, None] * stride_bm + offs_n[None, :] * stride_bn + stride_br,
        mask=mask[:, None] & mask_n[None, :],
        other=0.0,
    )
    prod_re = a_re[:, None] * b_re - a_im[:, None] * b_im
    prod_im = a_re[:, None] * b_im + a_im[:, None] * b_re
    acc_re = tl.sum(prod_re, axis=0)
    acc_im = tl.sum(prod_im, axis=0)
    tl.atomic_add(
        c_ri_ptr + col * stride_ck + offs_n * stride_cn,
        acc_re,
        mask=mask_n,
    )
    tl.atomic_add(
        c_ri_ptr + col * stride_ck + offs_n * stride_cn + stride_cr,
        acc_im,
        mask=mask_n,
    )


def _select_block_n(n_dense_cols, dtype, device=None):
    rocm_launch = _spmm_rocm_launch_overrides(
        n_dense_cols=n_dense_cols,
        fmt="csc",
        dtype=dtype,
        device=device,
    )
    if rocm_launch is not None and rocm_launch.get("block_n") is not None:
        return int(rocm_launch["block_n"])
    if dtype in (torch.float64, torch.complex128):
        return 16 if n_dense_cols >= 16 else 8
    return 32 if n_dense_cols >= 32 else 16


def _prepare_spmm_csc_matrix(data, indices, indptr, shape):
    if not all(torch.is_tensor(t) for t in (data, indices, indptr)):
        raise TypeError("data, indices, indptr must all be torch.Tensor")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError("shape dimensions must be non-negative")
    if indptr.numel() != n_cols + 1:
        raise ValueError(
            f"indptr length must be n_cols+1={n_cols + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if not all(t.is_cuda for t in (data, indices, indptr)):
        raise ValueError("data, indices, indptr must be CUDA tensors")
    if not all(t.device == data.device for t in (indices, indptr)):
        raise ValueError("data, indices, indptr must be on the same CUDA device")
    if data.dtype not in SUPPORTED_SPMM_CSC_VALUE_DTYPES:
        raise TypeError("CSC SpMM supports float32, float64, complex64, and complex128")
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if indptr.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")
    data = data.contiguous()
    indices = indices.contiguous()
    indptr = indptr.contiguous()
    if indptr.numel() > 0:
        if int(indptr[0].item()) != 0:
            raise ValueError("indptr must start at zero")
        if int(indptr[-1].item()) != data.numel():
            raise ValueError("indptr[-1] must equal nnz")
        if indptr.numel() > 1 and torch.any(indptr[1:] < indptr[:-1]).item():
            raise ValueError("indptr must be non-decreasing")
    if data.numel() > 0:
        min_index = int(indices.min().item())
        max_index = int(indices.max().item())
        if min_index < 0 or max_index >= n_rows:
            raise IndexError("indices out of range for n_rows")
    col_lengths = indptr[1:] - indptr[:-1]
    max_col_nnz = int(col_lengths.max().item()) if n_cols > 0 else 0
    return data, indices, indptr, n_rows, n_cols, col_lengths, max_col_nnz


def prepare_spmm_csc_route(
    data,
    indices,
    indptr,
    shape,
    *,
    block_n=None,
    block_nnz=256,
    max_segments=None,
    op="non",
    alg="auto",
    index_fallback_policy="auto",
):
    index_fallback_policy = _normalize_spmm_csc_index_fallback_policy(
        index_fallback_policy
    )
    op_code = _normalize_spmm_csc_op(op)
    _ensure_spmm_csc_supported_op(op_code)
    data, indices, indptr, n_rows, n_cols, col_lengths, max_col_nnz = (
        _prepare_spmm_csc_matrix(data, indices, indptr, shape)
    )
    block_nnz_use = int(block_nnz)
    if block_nnz_use <= 0:
        raise ValueError("block_nnz must be positive")
    if max_segments is None:
        max_segments_use = max((max_col_nnz + block_nnz_use - 1) // block_nnz_use, 1)
        while max_segments_use > 2048 and block_nnz_use < 65536:
            block_nnz_use *= 2
            max_segments_use = max(
                (max_col_nnz + block_nnz_use - 1) // block_nnz_use,
                1,
            )
    else:
        max_segments_use = max(1, int(max_segments))
    resolved_alg = _normalize_spmm_csc_alg(alg)
    if resolved_alg != "auto":
        resolve_spmm_csc_algorithm(resolved_alg, _spmm_csc_op_to_name(op_code), data.dtype)
    block_n_use = int(block_n) if block_n is not None else None
    if block_n_use is not None and block_n_use <= 0:
        raise ValueError("block_n must be positive")
    return PreparedCscSpmm(
        data=data,
        kernel_indices=indices,
        kernel_indptr=indptr,
        shape=shape,
        n_rows=n_rows,
        n_cols=n_cols,
        block_n=block_n_use,
        block_nnz=block_nnz_use,
        max_segments=max_segments_use,
        max_col_nnz=max_col_nnz,
        col_lengths=col_lengths,
        op=_spmm_csc_op_to_name(op_code),
        alg=resolved_alg,
        index_fallback_policy=index_fallback_policy,
    )


def _validate_spmm_csc_B(B, prepared, op_code):
    if B is None or not torch.is_tensor(B):
        raise TypeError("B must be a torch.Tensor")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != prepared.data.device:
        raise ValueError("B must be on the same CUDA device as sparse matrix data")
    if B.dtype != prepared.data.dtype:
        raise TypeError("B dtype must match sparse matrix dtype")
    if _spmm_csc_op_transposes(op_code):
        expected = prepared.n_rows
        name = "n_rows"
    else:
        expected = prepared.n_cols
        name = "n_cols"
    if B.shape[0] != expected:
        raise ValueError(f"B.shape[0] must be {name}={expected}, got {B.shape[0]}")
    return B


def _triton_spmm_csc_base_kernel(prepared, B, op_code=None):
    op_code = _normalize_spmm_csc_op(prepared.op if op_code is None else op_code)
    transposes = _spmm_csc_op_transposes(op_code)
    dtype = prepared.data.dtype
    n_dense_cols = int(B.shape[1])
    out_rows = prepared.n_cols if transposes else prepared.n_rows
    C = torch.zeros((out_rows, n_dense_cols), dtype=dtype, device=prepared.data.device)
    if prepared.nnz == 0 or n_dense_cols == 0:
        return C
    block_n = (
        prepared.block_n
        if prepared.block_n_override
        else _select_block_n(n_dense_cols, dtype, prepared.data.device)
    )
    grid = (
        prepared.n_cols,
        triton.cdiv(n_dense_cols, block_n),
    )
    for seg in range(prepared.max_segments):
        if _is_complex_dtype(dtype):
            data_ri = torch.view_as_real(prepared.data).reshape(-1)
            B_ri = torch.view_as_real(B)
            C_ri = torch.view_as_real(C)
            if transposes:
                _spmm_csc_trans_complex_kernel[grid](
                    data_ri,
                    prepared.kernel_indices,
                    prepared.kernel_indptr,
                    B_ri,
                    C_ri,
                    prepared.n_cols,
                    n_dense_cols,
                    B_ri.stride(0),
                    B_ri.stride(1),
                    B_ri.stride(2),
                    C_ri.stride(0),
                    C_ri.stride(1),
                    C_ri.stride(2),
                    BLOCK_N=block_n,
                    BLOCK_NNZ=prepared.block_nnz,
                    SEG=seg,
                    CONJ=op_code == SPMM_CSC_OP_CONJ_TRANS,
                )
            else:
                _spmm_csc_non_complex_kernel[grid](
                    data_ri,
                    prepared.kernel_indices,
                    prepared.kernel_indptr,
                    B_ri,
                    C_ri,
                    prepared.n_cols,
                    n_dense_cols,
                    B_ri.stride(0),
                    B_ri.stride(1),
                    B_ri.stride(2),
                    C_ri.stride(0),
                    C_ri.stride(1),
                    C_ri.stride(2),
                    BLOCK_N=block_n,
                    BLOCK_NNZ=prepared.block_nnz,
                    SEG=seg,
                )
        else:
            if transposes:
                _spmm_csc_trans_real_kernel[grid](
                    prepared.data,
                    prepared.kernel_indices,
                    prepared.kernel_indptr,
                    B,
                    C,
                    prepared.n_cols,
                    n_dense_cols,
                    B.stride(0),
                    B.stride(1),
                    C.stride(0),
                    C.stride(1),
                    BLOCK_N=block_n,
                    BLOCK_NNZ=prepared.block_nnz,
                    SEG=seg,
                )
            else:
                _spmm_csc_non_real_kernel[grid](
                    prepared.data,
                    prepared.kernel_indices,
                    prepared.kernel_indptr,
                    B,
                    C,
                    prepared.n_cols,
                    n_dense_cols,
                    B.stride(0),
                    B.stride(1),
                    C.stride(0),
                    C.stride(1),
                    BLOCK_N=block_n,
                    BLOCK_NNZ=prepared.block_nnz,
                    SEG=seg,
                )
    return C


def _run_spmm_csc_base_route(prepared, B, *, timing=False, diagnostics=False):
    del diagnostics
    compute_ms = None
    block_n = (
        prepared.block_n
        if prepared.block_n_override
        else _select_block_n(int(B.shape[1]), prepared.data.dtype, prepared.data.device)
    )
    backend_info = _get_device_backend_info(prepared.data.device)
    if timing:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    C = _triton_spmm_csc_base_kernel(prepared, B, _normalize_spmm_csc_op(prepared.op))
    if timing:
        end.record()
        torch.cuda.synchronize()
        compute_ms = start.elapsed_time(end)
    return C, {
        "process_cpu_ms": 0.0,
        "process_gpu_ms": 0.0 if timing else None,
        "compute_ms": compute_ms,
        "block_n": int(block_n),
        "block_nnz": int(prepared.block_nnz),
        "launch_backend": backend_info["backend"],
        "device_warp_size": int(backend_info["device_warp_size"]),
    }


SPMM_CSC_ALGORITHMS = {
    SPMM_CSC_ALG_BASE: SpmmCscAlgorithm(
        name=SPMM_CSC_ALG_BASE,
        display_name="CSCBase",
        supported_ops=SPMM_CSC_SUPPORTED_OP_NAMES,
        supported_dtypes=SUPPORTED_SPMM_CSC_VALUE_DTYPES,
        run=_run_spmm_csc_base_route,
    ),
}


def resolve_spmm_csc_algorithm(alg, op, dtype):
    token = _normalize_spmm_csc_alg(alg)
    if token == "auto":
        token = SPMM_CSC_ALG_BASE
    if token not in SPMM_CSC_ALGORITHMS:
        supported = ", ".join(sorted(SPMM_CSC_ALGORITHMS))
        raise ValueError(f"unsupported CSC SpMM algorithm {alg!r}; supported: auto, {supported}")
    algorithm = SPMM_CSC_ALGORITHMS[token]
    op_name = _spmm_csc_op_to_name(op)
    if op_name not in algorithm.supported_ops:
        raise ValueError(_SPMM_CSC_RESERVED_OP_MESSAGE)
    if dtype not in algorithm.supported_dtypes:
        raise TypeError(f"algorithm {token!r} does not support dtype {dtype}")
    return algorithm


def list_spmm_csc_algorithms(op=None, dtype=None):
    op_name = None if op is None else _spmm_csc_op_to_name(op)
    names = []
    for name, algorithm in SPMM_CSC_ALGORITHMS.items():
        if op_name is not None and op_name not in algorithm.supported_ops:
            continue
        if dtype is not None and dtype not in algorithm.supported_dtypes:
            continue
        names.append(name)
    return tuple(names)


def _spmm_csc_uses_int64_indices(prepared):
    return (
        prepared.kernel_indices.dtype == torch.int64
        or prepared.kernel_indptr.dtype == torch.int64
    )


def _spmm_csc_int32_fallback_blocker(prepared):
    if prepared.nnz > _INDEX_LIMIT_INT32:
        return f"nnz {prepared.nnz} cannot fit int32"
    if prepared.kernel_indices.numel() > 0:
        max_row = int(prepared.kernel_indices.max().item())
        if max_row > _INDEX_LIMIT_INT32:
            return f"row index {max_row} cannot fit int32"
    if prepared.kernel_indptr.numel() > 0:
        max_ptr = int(prepared.kernel_indptr[-1].item())
        if max_ptr > _INDEX_LIMIT_INT32:
            return f"indptr offset {max_ptr} cannot fit int32"
    return None


def _spmm_csc_prepared_with_int32_indices(prepared, reason):
    blocker = _spmm_csc_int32_fallback_blocker(prepared)
    if blocker is not None:
        raise RuntimeError(f"int32 fallback is unsafe: {blocker}") from reason
    return PreparedCscSpmm(
        data=prepared.data,
        kernel_indices=prepared.kernel_indices.to(torch.int32).contiguous(),
        kernel_indptr=prepared.kernel_indptr.to(torch.int32).contiguous(),
        shape=prepared.shape,
        n_rows=prepared.n_rows,
        n_cols=prepared.n_cols,
        block_n=prepared.block_n if prepared.block_n_override else None,
        block_nnz=prepared.block_nnz,
        max_segments=prepared.max_segments,
        max_col_nnz=prepared.max_col_nnz,
        col_lengths=prepared.col_lengths,
        op=prepared.op,
        alg=prepared.alg,
        index_fallback_policy=prepared.index_fallback_policy,
        index_fallback_applied=True,
        index_fallback_reason=str(reason),
    )


def _run_spmm_csc_prepared_with_fallback(prepared, B, algorithm, collect_timing=False):
    try:
        return algorithm.run(prepared, B, timing=bool(collect_timing))
    except RuntimeError as exc:
        if (
            prepared.index_fallback_policy != "auto"
            or not _spmm_csc_uses_int64_indices(prepared)
        ):
            raise
        fallback_prepared = _spmm_csc_prepared_with_int32_indices(prepared, exc)
        C, meta = algorithm.run(fallback_prepared, B, timing=bool(collect_timing))
        meta["index_fallback_applied"] = True
        meta["index_fallback_reason"] = str(exc)
        return C, meta


def flagsparse_spmm_csc_run(
    prepared,
    B,
    *,
    alg=None,
    op=None,
    return_time=False,
    return_meta=False,
    timing=False,
    diagnostics=False,
):
    """Run a registered native CSC SpMM route."""
    if not isinstance(prepared, PreparedCscSpmm):
        raise TypeError("prepared must be a PreparedCscSpmm instance")
    op_name = prepared.op if op is None else _spmm_csc_op_to_name(op)
    _ensure_spmm_csc_supported_op(_normalize_spmm_csc_op(op_name))
    if op_name != prepared.op:
        raise ValueError(f"op={op_name} does not match prepared.op={prepared.op}")
    alg_name = prepared.alg if alg is None else _normalize_spmm_csc_alg(alg)
    algorithm = resolve_spmm_csc_algorithm(alg_name, op_name, prepared.data.dtype)
    B = _validate_spmm_csc_B(B, prepared, _normalize_spmm_csc_op(op_name))
    collect_timing = bool(return_time or return_meta)
    if collect_timing:
        torch.cuda.synchronize()
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
    C, route_meta = _run_spmm_csc_prepared_with_fallback(
        prepared,
        B,
        algorithm,
        collect_timing=bool(timing),
    )
    if collect_timing:
        event_end.record()
        torch.cuda.synchronize()
        gpu_ms = event_start.elapsed_time(event_end)
    else:
        gpu_ms = None
    process_cpu_ms = float(route_meta.get("process_cpu_ms", 0.0) or 0.0)
    operator_ms = process_cpu_ms + float(gpu_ms) if gpu_ms is not None else None
    if return_meta:
        meta = {
            "alg": algorithm.name,
            "display_name": algorithm.display_name,
            "op": op_name,
            "logical_shape": prepared.shape,
            "block_n": route_meta.get("block_n", prepared.block_n),
            "block_nnz": route_meta.get("block_nnz", prepared.block_nnz),
            "launch_backend": route_meta.get("launch_backend"),
            "device_warp_size": route_meta.get("device_warp_size"),
            "max_segments": prepared.max_segments,
            "nnz": prepared.nnz,
            "operator_ms": operator_ms,
            "gpu_ms": gpu_ms,
            "process_cpu_ms": process_cpu_ms,
            "symbolic_ms": process_cpu_ms + float(route_meta.get("process_gpu_ms", 0.0) or 0.0)
            if timing
            else None,
            "process_gpu_ms": float(route_meta.get("process_gpu_ms", 0.0) or 0.0)
            if timing
            else None,
            "compute_ms": route_meta.get("compute_ms") if timing else None,
            "index_fallback_applied": bool(route_meta.get("index_fallback_applied", prepared.index_fallback_applied)),
            "index_fallback_reason": route_meta.get("index_fallback_reason", prepared.index_fallback_reason),
        }
        if timing:
            meta["op_total_ms"] = (
                process_cpu_ms + meta["process_gpu_ms"] + float(meta["compute_ms"] or 0.0)
            )
        else:
            meta["op_total_ms"] = operator_ms
        if return_time:
            return C, operator_ms, meta
        return C, meta
    if return_time:
        return C, operator_ms
    return C


def flagsparse_spmm_csc(
    data=None,
    indices=None,
    indptr=None,
    B=None,
    shape=None,
    block_n=None,
    block_nnz=256,
    max_segments=None,
    out=None,
    return_time=False,
    return_meta=False,
    prepared=None,
    transpose=None,
    op=None,
    alg="auto",
    timing=False,
    index_fallback_policy="auto",
):
    """CSC SpMM using native Triton CSC kernels."""
    op_explicit = op is not None
    op_code = _normalize_spmm_csc_op(
        op,
        transpose=False if transpose is None else bool(transpose),
    )
    if (
        op_explicit
        and transpose is not None
        and bool(transpose) != _spmm_csc_op_transposes(op_code)
    ):
        raise ValueError("transpose conflicts with op")
    _ensure_spmm_csc_supported_op(op_code)
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_spmm_csc_route(
            data,
            indices,
            indptr,
            shape,
            block_n=block_n,
            block_nnz=block_nnz,
            max_segments=max_segments,
            op=_spmm_csc_op_to_name(op_code),
            alg=alg,
            index_fallback_policy=index_fallback_policy,
        )
    else:
        if op_explicit and _spmm_csc_op_to_name(op_code) != prepared.op:
            raise ValueError(f"op={_spmm_csc_op_to_name(op_code)} does not match prepared.op={prepared.op}")
        if transpose is not None and bool(transpose) != _spmm_csc_op_transposes(prepared.op):
            raise ValueError(f"transpose={bool(transpose)} does not match prepared.op={prepared.op}")
        if not op_explicit:
            op_code = _normalize_spmm_csc_op(prepared.op)
    C = flagsparse_spmm_csc_run(
        prepared,
        B,
        alg=alg,
        op=_spmm_csc_op_to_name(op_code),
        return_time=return_time,
        return_meta=return_meta,
        timing=timing,
    )
    if out is None:
        return C
    result = C[0] if return_time or return_meta else C
    if not out.is_cuda:
        raise ValueError("out must be a CUDA tensor")
    if out.device != result.device:
        raise ValueError("out must be on the same CUDA device as the result")
    if out.shape != result.shape or out.dtype != result.dtype:
        raise ValueError("out shape/dtype must match result")
    out.copy_(result)
    if return_time and return_meta:
        return out, C[1], C[2]
    if return_time:
        return out, C[1]
    if return_meta:
        return out, C[1]
    return out
