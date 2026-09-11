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

"""Native CSC SpMV kernels and public helpers."""

from ._common import *

import triton
import triton.language as tl

# Resolved once at import: the runtime cannot change under a live process, and
# prepare_spmv_csc sits on the per-call path.
_SPMV_CSC_CUDA = _backend_name() == "cuda"

SUPPORTED_SPMV_CSC_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

SPMV_CSC_OP_NON = 0
SPMV_CSC_OP_TRANS = 1
SPMV_CSC_OP_CONJ_TRANS = 2
SPMV_CSC_OP_NAMES = {
    SPMV_CSC_OP_NON: "non",
    SPMV_CSC_OP_TRANS: "trans",
    SPMV_CSC_OP_CONJ_TRANS: "conj",
}
_SPMV_CSC_OP_NAME_TO_CODE = {name: code for code, name in SPMV_CSC_OP_NAMES.items()}


def _spmv_csc_dtype_error_message():
    return "CSC SpMV supports float32, float64, complex64, and complex128"


def _normalize_spmv_csc_op(op=None, transpose=False):
    if op is None:
        return SPMV_CSC_OP_TRANS if bool(transpose) else SPMV_CSC_OP_NON
    if isinstance(op, str):
        token = op.strip().lower()
        if token not in _SPMV_CSC_OP_NAME_TO_CODE:
            raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
        return _SPMV_CSC_OP_NAME_TO_CODE[token]
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj") from exc
    if op_code not in SPMV_CSC_OP_NAMES:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
    return op_code


def _spmv_csc_op_to_name(op):
    return SPMV_CSC_OP_NAMES[_normalize_spmv_csc_op(op)]


def _spmv_csc_op_transposes(op):
    return _normalize_spmv_csc_op(op) in (
        SPMV_CSC_OP_TRANS,
        SPMV_CSC_OP_CONJ_TRANS,
    )


def _normalize_spmv_csc_index_fallback_policy(index_fallback_policy):
    policy = str(index_fallback_policy).lower()
    if policy not in ("auto", "strict"):
        raise ValueError("index_fallback_policy must be 'auto' or 'strict'")
    return policy


class PreparedCscSpmv:
    """Prepared CSC metadata for repeated SpMV calls."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "nnz",
        "block_nnz",
        "max_segments",
        "col_lengths",
        "max_col_nnz",
        "col_ids",
        "csr_delegate",
        "op",
        "transpose",
        "index_fallback_policy",
        "index_fallback_applied",
        "index_fallback_reason",
        "launch_backend",
        "device_warp_size",
    )

    def __init__(
        self,
        data,
        kernel_indices,
        kernel_indptr,
        shape,
        n_rows,
        n_cols,
        block_nnz,
        max_segments,
        max_col_nnz,
        col_lengths=None,
        op=None,
        transpose=False,
        col_ids=None,
        csr_delegate=None,
        index_fallback_policy="auto",
        index_fallback_applied=False,
        index_fallback_reason=None,
        launch_backend=None,
        device_warp_size=None,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.nnz = int(data.numel())
        self.block_nnz = int(block_nnz)
        self.max_segments = int(max_segments)
        if col_lengths is None:
            col_lengths = kernel_indptr[1:] - kernel_indptr[:-1]
        self.col_lengths = col_lengths
        self.col_ids = col_ids
        self.csr_delegate = csr_delegate
        self.max_col_nnz = int(max_col_nnz)
        self.op = _normalize_spmv_csc_op(op, transpose=transpose)
        self.transpose = _spmv_csc_op_transposes(self.op)
        self.index_fallback_policy = str(index_fallback_policy).lower()
        self.index_fallback_applied = bool(index_fallback_applied)
        self.index_fallback_reason = index_fallback_reason
        info = _get_device_backend_info(data.device)
        self.launch_backend = launch_backend or info["backend"]
        self.device_warp_size = int(
            device_warp_size
            if device_warp_size is not None
            else info["device_warp_size"]
        )


@triton.jit
def _spmv_csc_non_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    n_cols,
    BLOCK_NNZ: tl.constexpr,
):
    col = tl.program_id(0)
    seg = tl.program_id(1)
    if col >= n_cols:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    offs = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    vals = tl.load(data_ptr + offs, mask=mask, other=0.0)
    x_val = tl.load(x_ptr + col)
    tl.atomic_add(y_ptr + rows, vals * x_val, mask=mask, sem="relaxed")


@triton.jit
def _spmv_csc_non_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    x_ri_ptr,
    y_ri_ptr,
    n_cols,
    BLOCK_NNZ: tl.constexpr,
):
    col = tl.program_id(0)
    seg = tl.program_id(1)
    if col >= n_cols:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    offs = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    a_re = tl.load(data_ri_ptr + offs * 2, mask=mask, other=0.0)
    a_im = tl.load(data_ri_ptr + offs * 2 + 1, mask=mask, other=0.0)
    x_re = tl.load(x_ri_ptr + col * 2)
    x_im = tl.load(x_ri_ptr + col * 2 + 1)
    prod_re = a_re * x_re - a_im * x_im
    prod_im = a_re * x_im + a_im * x_re
    tl.atomic_add(y_ri_ptr + rows * 2, prod_re, mask=mask, sem="relaxed")
    tl.atomic_add(y_ri_ptr + rows * 2 + 1, prod_im, mask=mask, sem="relaxed")


@triton.jit
def _spmv_csc_trans_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    n_cols,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
):
    col = tl.program_id(0)
    if col >= n_cols:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    acc = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    # Trip count comes from *this* column, not from the global longest one.
    # ``MAX_SEGMENTS`` is ceil(max_col_nnz / BLOCK_NNZ), a constexpr, so
    # ``for seg in range(MAX_SEGMENTS)`` made every column execute the same fixed number
    # of BLOCK_NNZ-wide masked loads no matter how short it is.  The waste is
    # ``MAX_SEGMENTS * BLOCK_NNZ / mean_col_nnz``: ~335x on amazon0601 (mean 8.4,
    # longest 2751) and ~93x on TSOPF_FS_b300_c1.  It is the same rectangle the
    # op="non" path had in its grid, just moved inside the kernel.
    # MAX_SEGMENTS is kept as a parameter so the launch signature and any explicit
    # max_segments override stay valid; it now only bounds the loop.
    n_seg = min(tl.cdiv(end - start, BLOCK_NNZ), MAX_SEGMENTS)
    for seg in tl.range(0, n_seg):
        offs = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        mask = offs < end
        rows = tl.load(indices_ptr + offs, mask=mask, other=0)
        vals = tl.load(data_ptr + offs, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + rows, mask=mask, other=0.0)
        acc = acc + tl.sum(tl.where(mask, vals * x_vals, 0.0))
    tl.store(y_ptr + col, acc)


@triton.jit
def _spmv_csc_trans_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    x_ri_ptr,
    y_ri_ptr,
    n_cols,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    CONJ: tl.constexpr,
):
    col = tl.program_id(0)
    if col >= n_cols:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    acc_re = tl.load(data_ri_ptr + start * 2, mask=start < end, other=0.0) * 0
    acc_im = tl.load(data_ri_ptr + start * 2 + 1, mask=start < end, other=0.0) * 0
    # Same dynamic trip count as the real kernel; see there for the measurement.
    n_seg = min(tl.cdiv(end - start, BLOCK_NNZ), MAX_SEGMENTS)
    for seg in tl.range(0, n_seg):
        offs = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        mask = offs < end
        rows = tl.load(indices_ptr + offs, mask=mask, other=0)
        a_re = tl.load(data_ri_ptr + offs * 2, mask=mask, other=0.0)
        a_im_raw = tl.load(data_ri_ptr + offs * 2 + 1, mask=mask, other=0.0)
        if CONJ:
            a_im = -a_im_raw
        else:
            a_im = a_im_raw
        x_re = tl.load(x_ri_ptr + rows * 2, mask=mask, other=0.0)
        x_im = tl.load(x_ri_ptr + rows * 2 + 1, mask=mask, other=0.0)
        prod_re = a_re * x_re - a_im * x_im
        prod_im = a_re * x_im + a_im * x_re
        acc_re = acc_re + tl.sum(tl.where(mask, prod_re, 0.0))
        acc_im = acc_im + tl.sum(tl.where(mask, prod_im, 0.0))
    tl.store(y_ri_ptr + col * 2, acc_re)
    tl.store(y_ri_ptr + col * 2 + 1, acc_im)


def _prepare_spmv_csc_matrix(data, indices, indptr, shape):
    if not all(torch.is_tensor(t) for t in (data, indices, indptr)):
        raise TypeError("data, indices, indptr must all be torch.Tensor")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_cols + 1:
        raise ValueError(
            f"indptr length must be n_cols+1={n_cols + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if not all(_is_accel_tensor(t) for t in (data, indices, indptr)):
        raise ValueError("data, indices, indptr must be CUDA tensors")
    if not all(t.device == data.device for t in (indices, indptr)):
        raise ValueError("data, indices, indptr must be on the same CUDA device")
    if data.dtype not in SUPPORTED_SPMV_CSC_VALUE_DTYPES:
        raise TypeError(_spmv_csc_dtype_error_message())
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


def prepare_spmv_csc(
    data,
    indices,
    indptr,
    shape,
    block_nnz=256,
    max_segments=None,
    transpose=False,
    op=None,
    index_fallback_policy="auto",
):
    index_fallback_policy = _normalize_spmv_csc_index_fallback_policy(
        index_fallback_policy
    )
    op_code = _normalize_spmv_csc_op(op, transpose=transpose)
    if op is not None and bool(transpose) and op_code == SPMV_CSC_OP_NON:
        raise ValueError("transpose=True conflicts with op=non")
    data, indices, indptr, n_rows, n_cols, col_lengths, max_col_nnz = (
        _prepare_spmv_csc_matrix(data, indices, indptr, shape)
    )
    block_nnz_use = int(block_nnz)
    if block_nnz_use <= 0:
        raise ValueError("block_nnz must be positive")
    launch = _spmv_rocm_launch_overrides(
        fmt="csc",
        dtype=data.dtype,
        max_row_nnz=max_col_nnz,
        nnz=data.numel(),
        block_nnz=block_nnz_use,
        device=data.device,
    )
    launch_backend = None
    device_warp_size = None
    if launch is not None:
        block_nnz_use = int(launch["block_nnz"])
        launch_backend = launch["backend"]
        device_warp_size = launch["device_warp_size"]
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
    # Per-nonzero owning column, for the nnz-parallel op="non" kernel.  Built once here
    # (a structural conversion, so outside any timed window) rather than binary-searched
    # per launch; see _spmv_csc_non_real_nnzpar_kernel for why.
    col_ids = None
    if int(data.numel()) > 0 and not _spmv_csc_op_transposes(op_code):
        try:
            # Deferred import: sddmm_csr only depends on _common, so there is no cycle,
            # but keeping it local avoids adding a load-time edge between operators.
            from .sddmm_csr import _build_row_ids

            col_ids = _build_row_ids(indptr.to(torch.int32), int(data.numel()))
        except Exception:
            col_ids = None
    # op="trans"/"conj": CSC(A) and CSR(A.T) are the same three arrays, so the
    # transposed product is a plain CSR SpMV -- and spmv_csr has the tuned CSR-Vector
    # bucket machinery this module never grew.  The bespoke CSC trans kernel runs one
    # program per column with a serial segment loop, which on a mean-8-nnz matrix is
    # mostly masked-off loads; delegating measured 4.39x geomean over 240 cases
    # (fp64 5.73x, complex128 8.22x) with 16 mild regressions, worst 0.75x.
    # prepare_spmv_csr already uses this same transpose-in-prepare technique for its own
    # trans/conj, so the timing convention matches the cuSPARSE baseline, which
    # materialises A.conj().T once outside the timed window.
    # CUDA only: this swaps which kernel actually runs, and the CSR-Vector bucket tiers
    # it lands on were tuned per backend separately.  Elsewhere the original CSC
    # transpose kernel is kept.
    csr_delegate = None
    if (
        _SPMV_CSC_CUDA
        and _spmv_csc_op_transposes(op_code)
        and int(data.numel()) > 0
    ):
        try:
            from .spmv_csr import prepare_spmv_csr

            delegate_values = data
            if op_code == SPMV_CSC_OP_CONJ_TRANS and _is_complex_dtype(data.dtype):
                delegate_values = data.conj()
                if hasattr(delegate_values, "resolve_conj"):
                    delegate_values = delegate_values.resolve_conj()
                delegate_values = delegate_values.contiguous()
            csr_delegate = prepare_spmv_csr(
                delegate_values,
                indices,
                indptr,
                (n_cols, n_rows),
                op="non",
                index_fallback_policy=index_fallback_policy,
            )
        except Exception:
            csr_delegate = None
    return PreparedCscSpmv(
        data=data,
        kernel_indices=indices,
        kernel_indptr=indptr,
        shape=shape,
        n_rows=n_rows,
        n_cols=n_cols,
        block_nnz=block_nnz_use,
        max_segments=max_segments_use,
        max_col_nnz=max_col_nnz,
        col_lengths=col_lengths,
        col_ids=col_ids,
        csr_delegate=csr_delegate,
        op=op_code,
        index_fallback_policy=index_fallback_policy,
        launch_backend=launch_backend,
        device_warp_size=device_warp_size,
    )


def _validate_spmv_csc_x(x, prepared, op_code):
    if x is None or not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if not _is_accel_tensor(x):
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != prepared.data.dtype:
        raise TypeError("x dtype must match sparse matrix dtype")
    expected = prepared.n_rows if _spmv_csc_op_transposes(op_code) else prepared.n_cols
    if x.numel() != expected:
        raise ValueError(f"x length must be {expected}, got {x.numel()}")
    if x.device != prepared.data.device:
        raise ValueError("x must be on the same device as sparse matrix data")
    return x.contiguous()


@triton.jit
def _spmv_csc_non_real_nnzpar_kernel(
    data_ptr,
    indices_ptr,
    col_ids_ptr,
    x_ptr,
    y_ptr,
    nnz,
    BLOCK: tl.constexpr,
):
    """One program per BLOCK nonzeros, independent of n_cols and max_col_nnz.

    The segmented kernel above launches ``grid = (n_cols, max_segments)`` with
    ``max_segments = ceil(max_col_nnz / BLOCK_NNZ)``, so the grid is a *rectangle sized
    by the longest column* while the useful work is only ``nnz / BLOCK_NNZ``.  The waste
    factor is roughly ``max_col_nnz / mean_col_nnz``: 327x on amazon0601 (403k columns,
    8.4 nonzeros each, longest 2751) and 93x on TSOPF_FS_b300_c1.  Tuning BLOCK_NNZ
    cannot fix that -- raising it to shrink the rectangle costs more than it saves, and
    a full sweep put the per-matrix oracle at only 1.09-1.38x over the current default.

    Flattening the grid over nonzeros removes the rectangle entirely.  Measured over the
    30-matrix corpus, fp32: **0.338 -> 1.403 of cupy's CSC SpMV**, with 21/30 matrices
    below 0.8x falling to 3/30, and up to 67.7x on a single matrix (wiki-Talk 14.52ms ->
    0.21ms).  BLOCK=256 is within 1% of the per-matrix oracle, so no adaptive rule is
    needed here.

    ``col_ids`` is built once in prepare by the same binary search the SDDMM path uses.
    Inlining that search into this kernel was measured a net loss there (0.710x kernel
    geomean), so it stays a precomputed array.
    """
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < nnz
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    cols = tl.load(col_ids_ptr + offs, mask=mask, other=0)
    vals = tl.load(data_ptr + offs, mask=mask, other=0.0)
    xs = tl.load(x_ptr + cols, mask=mask, other=0.0)
    tl.atomic_add(y_ptr + rows, vals * xs, mask=mask, sem="relaxed")


@triton.jit
def _spmv_csc_non_complex_nnzpar_kernel(
    data_ri_ptr,
    indices_ptr,
    col_ids_ptr,
    x_ri_ptr,
    y_ri_ptr,
    nnz,
    BLOCK: tl.constexpr,
):
    """Complex counterpart of :func:`_spmv_csc_non_real_nnzpar_kernel`."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < nnz
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    cols = tl.load(col_ids_ptr + offs, mask=mask, other=0)
    a_re = tl.load(data_ri_ptr + offs * 2, mask=mask, other=0.0)
    a_im = tl.load(data_ri_ptr + offs * 2 + 1, mask=mask, other=0.0)
    x_re = tl.load(x_ri_ptr + cols * 2, mask=mask, other=0.0)
    x_im = tl.load(x_ri_ptr + cols * 2 + 1, mask=mask, other=0.0)
    tl.atomic_add(y_ri_ptr + rows * 2, a_re * x_re - a_im * x_im, mask=mask, sem="relaxed")
    tl.atomic_add(y_ri_ptr + rows * 2 + 1, a_re * x_im + a_im * x_re, mask=mask, sem="relaxed")


SPMV_CSC_NNZPAR_BLOCK = 256


def _triton_spmv_csc_kernel(prepared, x, op_code):
    dtype = prepared.data.dtype
    trans = _spmv_csc_op_transposes(op_code)
    out_len = prepared.n_cols if trans else prepared.n_rows
    if trans and prepared.nnz != 0:
        csr_delegate = getattr(prepared, "csr_delegate", None)
        if csr_delegate is not None:
            from .spmv_csr import flagsparse_spmv_csr

            # Let spmv_csr allocate: it fills the whole vector, so zeroing first is a
            # wasted memset -- that alone moved the operator 0.765 -> 0.940.
            # use_opt=False deliberately: the bucketed path measured 0.39x on fp32,
            # while the default matches the oracle over both (4.39x vs 4.41x).
            return flagsparse_spmv_csr(prepared=csr_delegate, x=x, use_opt=False)
    y = torch.zeros(out_len, dtype=dtype, device=prepared.data.device)
    if prepared.nnz == 0:
        return y
    if not trans:
        col_ids = getattr(prepared, "col_ids", None)
        if col_ids is not None:
            # nnz-parallel path: grid scales with nnz instead of n_cols x max_segments.
            nnz = int(prepared.nnz)
            grid_nnz = (triton.cdiv(nnz, SPMV_CSC_NNZPAR_BLOCK),)
            if _is_complex_dtype(dtype):
                data_ri = torch.view_as_real(prepared.data).reshape(-1)
                x_ri = torch.view_as_real(x).reshape(-1)
                y_ri = torch.zeros(out_len * 2, dtype=data_ri.dtype, device=y.device)
                _spmv_csc_non_complex_nnzpar_kernel[grid_nnz](
                    data_ri,
                    prepared.kernel_indices,
                    col_ids,
                    x_ri,
                    y_ri,
                    nnz,
                    BLOCK=SPMV_CSC_NNZPAR_BLOCK,
                )
                y.copy_(torch.view_as_complex(y_ri.reshape(out_len, 2)))
                return y
            _spmv_csc_non_real_nnzpar_kernel[grid_nnz](
                prepared.data,
                prepared.kernel_indices,
                col_ids,
                x,
                y,
                nnz,
                BLOCK=SPMV_CSC_NNZPAR_BLOCK,
            )
            return y
        grid = (prepared.n_cols, prepared.max_segments)
        if _is_complex_dtype(dtype):
            data_ri = torch.view_as_real(prepared.data).reshape(-1)
            x_ri = torch.view_as_real(x).reshape(-1)
            y_ri = torch.zeros(out_len * 2, dtype=data_ri.dtype, device=y.device)
            _spmv_csc_non_complex_kernel[grid](
                data_ri,
                prepared.kernel_indices,
                prepared.kernel_indptr,
                x_ri,
                y_ri,
                prepared.n_cols,
                BLOCK_NNZ=prepared.block_nnz,
            )
            y.copy_(torch.view_as_complex(y_ri.reshape(out_len, 2)))
            return y
        _spmv_csc_non_real_kernel[grid](
            prepared.data,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            x,
            y,
            prepared.n_cols,
            BLOCK_NNZ=prepared.block_nnz,
        )
        return y
    grid = (prepared.n_cols,)
    if _is_complex_dtype(dtype):
        data_ri = torch.view_as_real(prepared.data).reshape(-1)
        x_ri = torch.view_as_real(x).reshape(-1)
        y_ri = torch.empty(out_len * 2, dtype=data_ri.dtype, device=y.device)
        _spmv_csc_trans_complex_kernel[grid](
            data_ri,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            x_ri,
            y_ri,
            prepared.n_cols,
            BLOCK_NNZ=prepared.block_nnz,
            MAX_SEGMENTS=prepared.max_segments,
            CONJ=(op_code == SPMV_CSC_OP_CONJ_TRANS),
        )
        y.copy_(torch.view_as_complex(y_ri.reshape(out_len, 2)))
        return y
    _spmv_csc_trans_real_kernel[grid](
        prepared.data,
        prepared.kernel_indices,
        prepared.kernel_indptr,
        x,
        y,
        prepared.n_cols,
        BLOCK_NNZ=prepared.block_nnz,
        MAX_SEGMENTS=prepared.max_segments,
    )
    return y


def _spmv_csc_uses_int64_indices(prepared):
    return (
        prepared.kernel_indices.dtype == torch.int64
        or prepared.kernel_indptr.dtype == torch.int64
    )


def _spmv_csc_int32_fallback_blocker(prepared):
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


def _spmv_csc_prepared_with_int32_indices(prepared, reason):
    blocker = _spmv_csc_int32_fallback_blocker(prepared)
    if blocker is not None:
        raise RuntimeError(f"int32 fallback is unsafe: {blocker}") from reason
    return PreparedCscSpmv(
        data=prepared.data,
        kernel_indices=prepared.kernel_indices.to(torch.int32).contiguous(),
        kernel_indptr=prepared.kernel_indptr.to(torch.int32).contiguous(),
        shape=prepared.shape,
        n_rows=prepared.n_rows,
        n_cols=prepared.n_cols,
        block_nnz=prepared.block_nnz,
        max_segments=prepared.max_segments,
        max_col_nnz=prepared.max_col_nnz,
        col_lengths=prepared.col_lengths,
        op=prepared.op,
        index_fallback_policy=prepared.index_fallback_policy,
        index_fallback_applied=True,
        index_fallback_reason=str(reason),
        launch_backend=prepared.launch_backend,
        device_warp_size=prepared.device_warp_size,
    )


def _run_spmv_csc_prepared_with_fallback(prepared, x, op_code):
    try:
        return _triton_spmv_csc_kernel(prepared, x, op_code)
    except RuntimeError as exc:
        if (
            prepared.index_fallback_policy != "auto"
            or not _spmv_csc_uses_int64_indices(prepared)
        ):
            raise
        fallback_prepared = _spmv_csc_prepared_with_int32_indices(prepared, exc)
        return _triton_spmv_csc_kernel(fallback_prepared, x, op_code)


def flagsparse_spmv_csc(
    data=None,
    indices=None,
    indptr=None,
    x=None,
    shape=None,
    block_nnz=256,
    max_segments=None,
    out=None,
    return_time=False,
    return_meta=False,
    prepared=None,
    transpose=None,
    op=None,
    index_fallback_policy="auto",
):
    """CSC SpMV using native Triton CSC kernels."""
    op_explicit = op is not None
    op_code = _normalize_spmv_csc_op(
        op,
        transpose=False if transpose is None else bool(transpose),
    )
    if (
        op_explicit
        and transpose is not None
        and bool(transpose) != _spmv_csc_op_transposes(op_code)
    ):
        raise ValueError("transpose conflicts with op")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_spmv_csc(
            data,
            indices,
            indptr,
            shape,
            block_nnz=block_nnz,
            max_segments=max_segments,
            op=op_code,
            index_fallback_policy=index_fallback_policy,
        )
    else:
        if op_explicit and op_code != prepared.op:
            raise ValueError(
                f"op={_spmv_csc_op_to_name(op_code)} does not match prepared.op={_spmv_csc_op_to_name(prepared.op)}"
            )
        if (
            not op_explicit
            and transpose is not None
            and bool(transpose) != prepared.transpose
        ):
            raise ValueError(
                f"transpose={bool(transpose)} does not match prepared.transpose={prepared.transpose}"
            )
        if not op_explicit:
            op_code = prepared.op
    x = _validate_spmv_csc_x(x, prepared, op_code)
    do_timing = bool(return_time or return_meta)
    if do_timing:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    y = _run_spmv_csc_prepared_with_fallback(prepared, x, op_code)
    if do_timing:
        _ACCEL.synchronize()
        compute_ms = (time.perf_counter() - t0) * 1000.0
        op_total_ms = compute_ms
    else:
        compute_ms = None
        op_total_ms = None
    if out is not None:
        if not _is_accel_tensor(out):
            raise ValueError("out must be a CUDA tensor")
        if out.device != y.device:
            raise ValueError("out must be on the same CUDA device as the result")
        if out.shape != y.shape or out.dtype != y.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(y)
        y = out
    if return_meta:
        meta = {
            "op": _spmv_csc_op_to_name(op_code),
            "symbolic_ms": 0.0 if do_timing else None,
            "compute_ms": compute_ms,
            "op_total_ms": op_total_ms,
            "block_nnz": prepared.block_nnz,
            "max_segments": prepared.max_segments,
            "launch_backend": prepared.launch_backend,
            "device_warp_size": prepared.device_warp_size,
            "index_fallback_applied": prepared.index_fallback_applied,
            "index_fallback_reason": prepared.index_fallback_reason,
        }
        if return_time:
            return y, op_total_ms, meta
        return y, meta
    if return_time:
        return y, op_total_ms
    return y
