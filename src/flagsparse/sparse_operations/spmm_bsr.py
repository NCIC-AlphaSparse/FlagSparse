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

"""Native BSR SpMM kernels and route helpers."""

from dataclasses import dataclass

from ._common import *

import triton
import triton.language as tl


SUPPORTED_SPMM_BSR_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

SPMM_BSR_OP_NON = 0
SPMM_BSR_OP_TRANS = 1
SPMM_BSR_OP_CONJ_TRANS = 2
SPMM_BSR_OP_NAMES = {
    SPMM_BSR_OP_NON: "non",
    SPMM_BSR_OP_TRANS: "trans",
    SPMM_BSR_OP_CONJ_TRANS: "conj",
}
SPMM_BSR_SUPPORTED_OP_NAMES = ("non", "trans", "conj")
_SPMM_BSR_OP_NAME_TO_CODE = {name: code for code, name in SPMM_BSR_OP_NAMES.items()}

SPMM_BSR_ALG_BASE = "spmm_bsr_base"


class SpmmBsrAlgorithmUnavailable(RuntimeError):
    """Raised when a requested BSR SpMM route is unavailable."""


@dataclass(frozen=True)
class SpmmBsrAlgorithm:
    name: str
    display_name: str
    supported_ops: tuple[str, ...]
    supported_dtypes: tuple
    run: object


def _normalize_spmm_bsr_op(op=None, transpose=False):
    if op is None:
        return SPMM_BSR_OP_TRANS if bool(transpose) else SPMM_BSR_OP_NON
    if isinstance(op, str):
        token = op.strip().lower()
        if token not in _SPMM_BSR_OP_NAME_TO_CODE:
            raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
        return _SPMM_BSR_OP_NAME_TO_CODE[token]
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj") from exc
    if op_code not in SPMM_BSR_OP_NAMES:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
    return op_code


def _spmm_bsr_op_to_name(op):
    return SPMM_BSR_OP_NAMES[_normalize_spmm_bsr_op(op)]


def _spmm_bsr_op_transposes(op):
    return _normalize_spmm_bsr_op(op) in (
        SPMM_BSR_OP_TRANS,
        SPMM_BSR_OP_CONJ_TRANS,
    )


def _ensure_spmm_bsr_supported_op(op_code):
    op_name = _spmm_bsr_op_to_name(op_code)
    if op_name not in SPMM_BSR_SUPPORTED_OP_NAMES:
        supported = ", ".join(SPMM_BSR_SUPPORTED_OP_NAMES)
        raise ValueError(f"spmm_bsr supports ops: {supported}")


def _normalize_spmm_bsr_alg(alg):
    token = "auto" if alg is None else str(alg).strip().lower().replace("-", "_")
    if token in ("auto", "base", "bsr_base", "spmm_bsr_base"):
        return "auto" if token == "auto" else SPMM_BSR_ALG_BASE
    raise ValueError("unsupported BSR SpMM algorithm; supported: auto, spmm_bsr_base")


def _normalize_spmm_bsr_index_fallback_policy(index_fallback_policy):
    policy = str(index_fallback_policy).lower()
    if policy not in ("auto", "strict"):
        raise ValueError("index_fallback_policy must be 'auto' or 'strict'")
    return policy


class PreparedBsrSpmm:
    """Prepared BSR metadata for native SpMM routes."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "padded_n_rows",
        "padded_n_cols",
        "block_dim",
        "n_block_rows",
        "n_block_cols",
        "nnzb",
        "stored_nnz",
        "block_row_lengths",
        "max_block_row_nnz",
        "block_nnz",
        "max_segments",
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
        block_dim,
        n_block_rows,
        n_block_cols,
        block_nnz,
        max_segments,
        max_block_row_nnz,
        block_row_lengths,
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
        self.n_rows = int(shape[0])
        self.n_cols = int(shape[1])
        self.block_dim = int(block_dim)
        self.n_block_rows = int(n_block_rows)
        self.n_block_cols = int(n_block_cols)
        self.padded_n_rows = self.n_block_rows * self.block_dim
        self.padded_n_cols = self.n_block_cols * self.block_dim
        self.nnzb = int(data.shape[0])
        self.stored_nnz = int(data.numel())
        self.block_row_lengths = block_row_lengths
        self.max_block_row_nnz = int(max_block_row_nnz)
        self.block_nnz = int(block_nnz)
        self.max_segments = int(max_segments)
        self.op = _spmm_bsr_op_to_name(op)
        self.alg = _normalize_spmm_bsr_alg(alg)
        self.index_fallback_policy = str(index_fallback_policy).lower()
        self.index_fallback_applied = bool(index_fallback_applied)
        self.index_fallback_reason = index_fallback_reason


@triton.jit
def _spmm_bsr_non_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    n_block_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    SEG: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    brow = tl.program_id(0)
    inner_row = tl.program_id(1)
    pid_n = tl.program_id(2)
    if brow >= n_block_rows:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + brow)
    end = tl.load(indptr_ptr + brow + 1)
    offs = start + SEG * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    bcols = tl.load(indices_ptr + offs, mask=mask, other=0)
    acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    for inner_col in tl.static_range(0, BLOCK_DIM):
        cols = bcols * BLOCK_DIM + inner_col
        vals = tl.load(
            data_ptr + offs * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM + inner_col,
            mask=mask,
            other=0.0,
        )
        b_vals = tl.load(
            b_ptr + cols[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.sum(vals[:, None] * b_vals, axis=0)
    row = brow * BLOCK_DIM + inner_row
    tl.atomic_add(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_bsr_non_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    b_ri_ptr,
    c_ri_ptr,
    n_block_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    stride_cr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    SEG: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    brow = tl.program_id(0)
    inner_row = tl.program_id(1)
    pid_n = tl.program_id(2)
    if brow >= n_block_rows:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + brow)
    end = tl.load(indptr_ptr + brow + 1)
    offs = start + SEG * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    bcols = tl.load(indices_ptr + offs, mask=mask, other=0)
    acc_re = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    acc_im = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    for inner_col in tl.static_range(0, BLOCK_DIM):
        cols = bcols * BLOCK_DIM + inner_col
        elem = offs * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM + inner_col
        a_re = tl.load(data_ri_ptr + elem * 2, mask=mask, other=0.0)
        a_im = tl.load(data_ri_ptr + elem * 2 + 1, mask=mask, other=0.0)
        b_re = tl.load(
            b_ri_ptr + cols[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask[:, None] & mask_n[None, :],
            other=0.0,
        )
        b_im = tl.load(
            b_ri_ptr
            + cols[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
            + stride_br,
            mask=mask[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc_re += tl.sum(a_re[:, None] * b_re - a_im[:, None] * b_im, axis=0)
        acc_im += tl.sum(a_re[:, None] * b_im + a_im[:, None] * b_re, axis=0)
    row = brow * BLOCK_DIM + inner_row
    tl.atomic_add(
        c_ri_ptr + row * stride_cm + offs_n * stride_cn,
        acc_re,
        mask=mask_n,
    )
    tl.atomic_add(
        c_ri_ptr + row * stride_cm + offs_n * stride_cn + stride_cr,
        acc_im,
        mask=mask_n,
    )


@triton.jit
def _spmm_bsr_trans_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    n_block_rows,
    n_dense_cols,
    stride_bm,
    stride_bn,
    stride_ck,
    stride_cn,
    BLOCK_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    SEG: tl.constexpr,
):
    brow = tl.program_id(0)
    inner_row = tl.program_id(1)
    pid_n = tl.program_id(2)
    if brow >= n_block_rows:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    row = brow * BLOCK_DIM + inner_row
    b_vals = tl.load(
        b_ptr + row * stride_bm + offs_n * stride_bn,
        mask=mask_n,
        other=0.0,
    )
    start = tl.load(indptr_ptr + brow)
    end = tl.load(indptr_ptr + brow + 1)
    offs = start + SEG * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    bcols = tl.load(indices_ptr + offs, mask=mask, other=0)
    for inner_col in tl.static_range(0, BLOCK_DIM):
        cols = bcols * BLOCK_DIM + inner_col
        vals = tl.load(
            data_ptr + offs * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM + inner_col,
            mask=mask,
            other=0.0,
        )
        tl.atomic_add(
            c_ptr + cols[:, None] * stride_ck + offs_n[None, :] * stride_cn,
            vals[:, None] * b_vals[None, :],
            mask=mask[:, None] & mask_n[None, :],
        )


@triton.jit
def _spmm_bsr_trans_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    b_ri_ptr,
    c_ri_ptr,
    n_block_rows,
    n_dense_cols,
    stride_bm,
    stride_bn,
    stride_br,
    stride_ck,
    stride_cn,
    stride_cr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    SEG: tl.constexpr,
    CONJ: tl.constexpr,
):
    brow = tl.program_id(0)
    inner_row = tl.program_id(1)
    pid_n = tl.program_id(2)
    if brow >= n_block_rows:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    row = brow * BLOCK_DIM + inner_row
    b_re = tl.load(
        b_ri_ptr + row * stride_bm + offs_n * stride_bn,
        mask=mask_n,
        other=0.0,
    )
    b_im = tl.load(
        b_ri_ptr + row * stride_bm + offs_n * stride_bn + stride_br,
        mask=mask_n,
        other=0.0,
    )
    start = tl.load(indptr_ptr + brow)
    end = tl.load(indptr_ptr + brow + 1)
    offs = start + SEG * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    bcols = tl.load(indices_ptr + offs, mask=mask, other=0)
    for inner_col in tl.static_range(0, BLOCK_DIM):
        cols = bcols * BLOCK_DIM + inner_col
        elem = offs * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM + inner_col
        a_re = tl.load(data_ri_ptr + elem * 2, mask=mask, other=0.0)
        a_im_raw = tl.load(data_ri_ptr + elem * 2 + 1, mask=mask, other=0.0)
        a_im = a_im_raw
        if CONJ:
            a_im = -a_im_raw
        prod_re = a_re[:, None] * b_re[None, :] - a_im[:, None] * b_im[None, :]
        prod_im = a_re[:, None] * b_im[None, :] + a_im[:, None] * b_re[None, :]
        tl.atomic_add(
            c_ri_ptr + cols[:, None] * stride_ck + offs_n[None, :] * stride_cn,
            prod_re,
            mask=mask[:, None] & mask_n[None, :],
        )
        tl.atomic_add(
            c_ri_ptr
            + cols[:, None] * stride_ck
            + offs_n[None, :] * stride_cn
            + stride_cr,
            prod_im,
            mask=mask[:, None] & mask_n[None, :],
        )


def _prepare_spmm_bsr_matrix(data, indices, indptr, shape, block_dim):
    if not torch.is_tensor(data) or not torch.is_tensor(indices) or not torch.is_tensor(indptr):
        raise TypeError("data, indices, and indptr must be torch.Tensor")
    if data.ndim != 3 or data.shape[1] != data.shape[2]:
        raise ValueError("data must have shape (nnzb, block_dim, block_dim)")
    block_dim = int(block_dim)
    if block_dim <= 1:
        raise ValueError("block_dim must be greater than 1")
    if int(data.shape[1]) != block_dim:
        raise ValueError("data block shape must match block_dim")
    if data.dtype not in SUPPORTED_SPMM_BSR_VALUE_DTYPES:
        raise TypeError("BSR SpMM supports float32, float64, complex64, and complex128")
    if indices.dtype not in SUPPORTED_INDEX_DTYPES or indptr.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices and indptr dtype must be torch.int32 or torch.int64")
    if not data.is_cuda or not indices.is_cuda or not indptr.is_cuda:
        raise ValueError("data, indices, and indptr must be CUDA tensors")
    if indices.device != data.device or indptr.device != data.device:
        raise ValueError("indices and indptr must be on the same CUDA device as data")
    if indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("indices and indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError("shape dimensions must be non-negative")
    n_block_rows = (n_rows + block_dim - 1) // block_dim
    n_block_cols = (n_cols + block_dim - 1) // block_dim
    if indptr.numel() != n_block_rows + 1:
        raise ValueError("indptr length must be n_block_rows + 1")
    data = data.contiguous()
    indices = indices.contiguous()
    indptr = indptr.contiguous()
    if int(indptr[0].item()) != 0:
        raise ValueError("indptr must start at zero")
    if int(indptr[-1].item()) != data.shape[0]:
        raise ValueError("indptr[-1] must equal nnzb")
    if indptr.numel() > 1 and torch.any(indptr[1:] < indptr[:-1]).item():
        raise ValueError("indptr must be non-decreasing")
    if indices.numel() > 0:
        min_index = int(indices.min().item())
        max_index = int(indices.max().item())
        if min_index < 0 or max_index >= n_block_cols:
            raise IndexError("indices out of range for n_block_cols")
    block_row_lengths = indptr[1:] - indptr[:-1]
    max_block_row_nnz = int(block_row_lengths.max().item()) if n_block_rows > 0 else 0
    return (
        data,
        indices,
        indptr,
        n_rows,
        n_cols,
        n_block_rows,
        n_block_cols,
        block_row_lengths,
        max_block_row_nnz,
    )


def prepare_spmm_bsr_route(
    data,
    indices,
    indptr,
    shape,
    *,
    block_dim,
    block_nnz=128,
    max_segments=None,
    op="non",
    alg="auto",
    index_fallback_policy="auto",
):
    index_fallback_policy = _normalize_spmm_bsr_index_fallback_policy(index_fallback_policy)
    op_code = _normalize_spmm_bsr_op(op)
    _ensure_spmm_bsr_supported_op(op_code)
    (
        data,
        indices,
        indptr,
        n_rows,
        n_cols,
        n_block_rows,
        n_block_cols,
        block_row_lengths,
        max_block_row_nnz,
    ) = _prepare_spmm_bsr_matrix(data, indices, indptr, shape, block_dim)
    block_nnz_use = int(block_nnz)
    if block_nnz_use <= 0:
        raise ValueError("block_nnz must be positive")
    if max_segments is None:
        max_segments_use = max((max_block_row_nnz + block_nnz_use - 1) // block_nnz_use, 1)
        while max_segments_use > 2048 and block_nnz_use < 65536:
            block_nnz_use *= 2
            max_segments_use = max((max_block_row_nnz + block_nnz_use - 1) // block_nnz_use, 1)
    else:
        max_segments_use = max(1, int(max_segments))
    resolved_alg = _normalize_spmm_bsr_alg(alg)
    if resolved_alg != "auto":
        resolve_spmm_bsr_algorithm(resolved_alg, _spmm_bsr_op_to_name(op_code), data.dtype)
    return PreparedBsrSpmm(
        data=data,
        kernel_indices=indices,
        kernel_indptr=indptr,
        shape=shape,
        block_dim=block_dim,
        n_block_rows=n_block_rows,
        n_block_cols=n_block_cols,
        block_nnz=block_nnz_use,
        max_segments=max_segments_use,
        max_block_row_nnz=max_block_row_nnz,
        block_row_lengths=block_row_lengths,
        op=_spmm_bsr_op_to_name(op_code),
        alg=resolved_alg,
        index_fallback_policy=index_fallback_policy,
    )


def _validate_spmm_bsr_B(B, prepared, op_code):
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
    if _spmm_bsr_op_transposes(op_code):
        logical_rows = prepared.n_rows
        padded_rows = prepared.padded_n_rows
        logical_name = "n_rows"
        padded_name = "padded_n_rows"
    else:
        logical_rows = prepared.n_cols
        padded_rows = prepared.padded_n_cols
        logical_name = "n_cols"
        padded_name = "padded_n_cols"
    if B.shape[0] not in (logical_rows, padded_rows):
        raise ValueError(
            f"B.shape[0] must be {logical_name}={logical_rows} or {padded_name}={padded_rows}, got {B.shape[0]}"
        )
    if B.shape[0] == padded_rows:
        return B
    if B.stride(0) == 1 and B.shape[0] > 1:
        padded = torch.empty_strided(
            (padded_rows, B.shape[1]),
            (1, max(1, padded_rows)),
            dtype=B.dtype,
            device=B.device,
        )
        padded.zero_()
    else:
        padded = torch.zeros(
            (padded_rows, B.shape[1]),
            dtype=B.dtype,
            device=B.device,
        )
    padded[: B.shape[0], :].copy_(B)
    return padded


def _select_block_n(n_dense_cols, dtype):
    if dtype in (torch.float64, torch.complex128):
        return 16 if n_dense_cols >= 16 else 8
    return 32 if n_dense_cols >= 32 else 16


def _triton_spmm_bsr_base_kernel(prepared, B, op_code=None):
    op_code = _normalize_spmm_bsr_op(prepared.op if op_code is None else op_code)
    transposes = _spmm_bsr_op_transposes(op_code)
    dtype = prepared.data.dtype
    n_dense_cols = int(B.shape[1])
    out_rows = prepared.padded_n_cols if transposes else prepared.padded_n_rows
    C = torch.zeros(
        (out_rows, n_dense_cols),
        dtype=dtype,
        device=prepared.data.device,
    )
    if prepared.nnzb == 0 or n_dense_cols == 0:
        return C
    block_n = _select_block_n(n_dense_cols, dtype)
    grid = (
        prepared.n_block_rows,
        prepared.block_dim,
        triton.cdiv(n_dense_cols, block_n),
    )
    for seg in range(prepared.max_segments):
        if _is_complex_dtype(dtype):
            data_ri = torch.view_as_real(prepared.data)
            B_ri = torch.view_as_real(B)
            C_ri = torch.view_as_real(C)
            if transposes:
                _spmm_bsr_trans_complex_kernel[grid](
                    data_ri.reshape(-1),
                    prepared.kernel_indices,
                    prepared.kernel_indptr,
                    B_ri,
                    C_ri,
                    prepared.n_block_rows,
                    n_dense_cols,
                    B_ri.stride(0),
                    B_ri.stride(1),
                    B_ri.stride(2),
                    C_ri.stride(0),
                    C_ri.stride(1),
                    C_ri.stride(2),
                    BLOCK_DIM=prepared.block_dim,
                    BLOCK_N=block_n,
                    BLOCK_NNZ=prepared.block_nnz,
                    SEG=seg,
                    CONJ=op_code == SPMM_BSR_OP_CONJ_TRANS,
                )
            else:
                _spmm_bsr_non_complex_kernel[grid](
                    data_ri.reshape(-1),
                    prepared.kernel_indices,
                    prepared.kernel_indptr,
                    B_ri,
                    C_ri,
                    prepared.n_block_rows,
                    n_dense_cols,
                    B_ri.stride(0),
                    B_ri.stride(1),
                    B_ri.stride(2),
                    C_ri.stride(0),
                    C_ri.stride(1),
                    C_ri.stride(2),
                    BLOCK_DIM=prepared.block_dim,
                    BLOCK_N=block_n,
                    BLOCK_NNZ=prepared.block_nnz,
                    SEG=seg,
                    ACC_DTYPE=tl.float64 if dtype == torch.complex128 else tl.float32,
                )
        else:
            if transposes:
                _spmm_bsr_trans_real_kernel[grid](
                    prepared.data,
                    prepared.kernel_indices,
                    prepared.kernel_indptr,
                    B,
                    C,
                    prepared.n_block_rows,
                    n_dense_cols,
                    B.stride(0),
                    B.stride(1),
                    C.stride(0),
                    C.stride(1),
                    BLOCK_DIM=prepared.block_dim,
                    BLOCK_N=block_n,
                    BLOCK_NNZ=prepared.block_nnz,
                    SEG=seg,
                )
            else:
                _spmm_bsr_non_real_kernel[grid](
                    prepared.data,
                    prepared.kernel_indices,
                    prepared.kernel_indptr,
                    B,
                    C,
                    prepared.n_block_rows,
                    n_dense_cols,
                    B.stride(0),
                    B.stride(1),
                    C.stride(0),
                    C.stride(1),
                    BLOCK_DIM=prepared.block_dim,
                    BLOCK_N=block_n,
                    BLOCK_NNZ=prepared.block_nnz,
                    SEG=seg,
                    ACC_DTYPE=tl.float64 if dtype == torch.float64 else tl.float32,
                )
    return C


def _run_spmm_bsr_base_route(prepared, B, *, timing=False, diagnostics=False):
    del diagnostics
    compute_ms = None
    if timing:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    C = _triton_spmm_bsr_base_kernel(prepared, B, _normalize_spmm_bsr_op(prepared.op))
    if timing:
        end.record()
        torch.cuda.synchronize()
        compute_ms = start.elapsed_time(end)
    return C, {
        "process_cpu_ms": 0.0,
        "process_gpu_ms": 0.0 if timing else None,
        "compute_ms": compute_ms,
    }


SPMM_BSR_ALGORITHMS = {
    SPMM_BSR_ALG_BASE: SpmmBsrAlgorithm(
        name=SPMM_BSR_ALG_BASE,
        display_name="BSRBase",
        supported_ops=SPMM_BSR_SUPPORTED_OP_NAMES,
        supported_dtypes=SUPPORTED_SPMM_BSR_VALUE_DTYPES,
        run=_run_spmm_bsr_base_route,
    ),
}


def resolve_spmm_bsr_algorithm(alg, op, dtype):
    token = _normalize_spmm_bsr_alg(alg)
    if token == "auto":
        token = SPMM_BSR_ALG_BASE
    if token not in SPMM_BSR_ALGORITHMS:
        supported = ", ".join(sorted(SPMM_BSR_ALGORITHMS))
        raise ValueError(f"unsupported BSR SpMM algorithm {alg!r}; supported: auto, {supported}")
    algorithm = SPMM_BSR_ALGORITHMS[token]
    op_name = _spmm_bsr_op_to_name(op)
    if op_name not in algorithm.supported_ops:
        raise ValueError(f"algorithm {token!r} does not support op {op_name!r}")
    if dtype not in algorithm.supported_dtypes:
        raise TypeError(f"algorithm {token!r} does not support dtype {dtype}")
    return algorithm


def list_spmm_bsr_algorithms(op=None, dtype=None):
    op_name = None if op is None else _spmm_bsr_op_to_name(op)
    names = []
    for name, algorithm in SPMM_BSR_ALGORITHMS.items():
        if op_name is not None and op_name not in algorithm.supported_ops:
            continue
        if dtype is not None and dtype not in algorithm.supported_dtypes:
            continue
        names.append(name)
    return tuple(names)


def _spmm_bsr_uses_int64_indices(prepared):
    return (
        prepared.kernel_indices.dtype == torch.int64
        or prepared.kernel_indptr.dtype == torch.int64
    )


def _spmm_bsr_int32_fallback_blocker(prepared):
    if prepared.nnzb > _INDEX_LIMIT_INT32:
        return f"nnzb {prepared.nnzb} cannot fit int32"
    if prepared.kernel_indices.numel() > 0:
        max_col = int(prepared.kernel_indices.max().item())
        if max_col > _INDEX_LIMIT_INT32:
            return f"block column index {max_col} cannot fit int32"
    if prepared.kernel_indptr.numel() > 0:
        max_ptr = int(prepared.kernel_indptr[-1].item())
        if max_ptr > _INDEX_LIMIT_INT32:
            return f"indptr offset {max_ptr} cannot fit int32"
    return None


def _spmm_bsr_prepared_with_int32_indices(prepared, reason):
    blocker = _spmm_bsr_int32_fallback_blocker(prepared)
    if blocker is not None:
        raise RuntimeError(f"int32 fallback is unsafe: {blocker}") from reason
    return PreparedBsrSpmm(
        data=prepared.data,
        kernel_indices=prepared.kernel_indices.to(torch.int32).contiguous(),
        kernel_indptr=prepared.kernel_indptr.to(torch.int32).contiguous(),
        shape=prepared.shape,
        block_dim=prepared.block_dim,
        n_block_rows=prepared.n_block_rows,
        n_block_cols=prepared.n_block_cols,
        block_nnz=prepared.block_nnz,
        max_segments=prepared.max_segments,
        max_block_row_nnz=prepared.max_block_row_nnz,
        block_row_lengths=prepared.block_row_lengths,
        op=prepared.op,
        alg=prepared.alg,
        index_fallback_policy=prepared.index_fallback_policy,
        index_fallback_applied=True,
        index_fallback_reason=str(reason),
    )


def _run_spmm_bsr_prepared_with_fallback(prepared, B, algorithm, collect_timing=False):
    try:
        return algorithm.run(prepared, B, timing=bool(collect_timing))
    except RuntimeError as exc:
        if (
            prepared.index_fallback_policy != "auto"
            or not _spmm_bsr_uses_int64_indices(prepared)
        ):
            raise
        fallback_prepared = _spmm_bsr_prepared_with_int32_indices(prepared, exc)
        C, meta = algorithm.run(fallback_prepared, B, timing=bool(collect_timing))
        meta["index_fallback_applied"] = True
        meta["index_fallback_reason"] = str(exc)
        return C, meta


def flagsparse_spmm_bsr_run(
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
    """Run a registered native BSR SpMM route."""
    if not isinstance(prepared, PreparedBsrSpmm):
        raise TypeError("prepared must be a PreparedBsrSpmm instance")
    op_name = prepared.op if op is None else _spmm_bsr_op_to_name(op)
    _ensure_spmm_bsr_supported_op(_normalize_spmm_bsr_op(op_name))
    if op_name != prepared.op:
        raise ValueError(f"op={op_name} does not match prepared.op={prepared.op}")
    alg_name = prepared.alg if alg is None else _normalize_spmm_bsr_alg(alg)
    algorithm = resolve_spmm_bsr_algorithm(alg_name, op_name, prepared.data.dtype)
    B = _validate_spmm_bsr_B(B, prepared, _normalize_spmm_bsr_op(op_name))
    collect_timing = bool(return_time or return_meta)
    if collect_timing:
        torch.cuda.synchronize()
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
    C, route_meta = _run_spmm_bsr_prepared_with_fallback(
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
            "padded_shape": (prepared.padded_n_rows, prepared.padded_n_cols),
            "block_dim": prepared.block_dim,
            "n_block_rows": prepared.n_block_rows,
            "n_block_cols": prepared.n_block_cols,
            "nnzb": prepared.nnzb,
            "stored_nnz": prepared.stored_nnz,
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


def flagsparse_spmm_bsr(
    data=None,
    indices=None,
    indptr=None,
    B=None,
    shape=None,
    block_dim=None,
    block_nnz=128,
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
    """BSR SpMM using native Triton BSR kernels.

    The native compute layer follows padded block-grid semantics. It returns
    ``padded_rows`` rows for ``op='non'`` and ``padded_cols`` rows for
    ``op='trans'`` or ``op='conj'``.
    """
    op_explicit = op is not None
    op_code = _normalize_spmm_bsr_op(
        op,
        transpose=False if transpose is None else bool(transpose),
    )
    if (
        op_explicit
        and transpose is not None
        and bool(transpose) != _spmm_bsr_op_transposes(op_code)
    ):
        raise ValueError("transpose conflicts with op")
    _ensure_spmm_bsr_supported_op(op_code)
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape, block_dim)):
            raise ValueError(
                "data, indices, indptr, shape, and block_dim are required when prepared is not provided"
            )
        prepared = prepare_spmm_bsr_route(
            data,
            indices,
            indptr,
            shape,
            block_dim=block_dim,
            block_nnz=block_nnz,
            max_segments=max_segments,
            op=_spmm_bsr_op_to_name(op_code),
            alg=alg,
            index_fallback_policy=index_fallback_policy,
        )
    else:
        if op_explicit and _spmm_bsr_op_to_name(op_code) != prepared.op:
            raise ValueError(f"op={_spmm_bsr_op_to_name(op_code)} does not match prepared.op={prepared.op}")
        if transpose is not None and bool(transpose) != _spmm_bsr_op_transposes(prepared.op):
            raise ValueError(f"transpose={bool(transpose)} does not match prepared.op={prepared.op}")
        if not op_explicit:
            op_code = _normalize_spmm_bsr_op(prepared.op)
    C = flagsparse_spmm_bsr_run(
        prepared,
        B,
        alg=alg,
        op=_spmm_bsr_op_to_name(op_code),
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
