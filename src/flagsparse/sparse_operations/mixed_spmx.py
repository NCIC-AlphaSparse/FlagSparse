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

"""Mixed-precision and integer SpMV / SpMM (the cuSPARSE 12.5 type combinations).

The public entry points (``flagsparse_spmv_csr``, ``flagsparse_spmv_coo``,
``flagsparse_spmm_csr``) route here only for combinations their native paths do
not take, so every previously supported call keeps its exact code path:

=========================  ==============================  ===========
matrix (and x / B)         output (``out_dtype``/``out``)  accumulate
=========================  ==============================  ===========
float16 / bfloat16         float32                         float32
int8                       int32 (default) or float32      int32 / f32
float32 A, complex64 x     complex64 (SpMV only)           float32 x2
float16 / bfloat16 (COO)   same half type                  float32
=========================  ==============================  ===========

Backend notes:
- COO accumulates with ``tl.atomic_add`` on a float32 / int32 buffer and casts at
  the end; fp16 and complex atomics are never needed (not every backend has them).
- CSR kernels loop over a row's entries at runtime; nothing is unrolled, so the
  per-thread private memory stays small (MetaX caps it at 4 KB).
- Complex vectors are read as interleaved real/imag planes: no complex indexing,
  which Moore Threads lacks.
- Ascend's Triton lacks the pieces these kernels use, so it takes a torch_npu
  ``index_add_`` path (complex added as two real planes).
"""

from ._common import *

import os
import time

import triton
import triton.language as tl
from ._rocm_sparse_cache import TensorPlanCache
from ._musa_spmv_launch import launch as _launch_musa_spmv

_HALF = (torch.float16, torch.bfloat16)
_TL_ACC = {torch.float32: tl.float32, torch.float64: tl.float64, torch.int32: tl.int32}
_ROCM_CSR_MIXED_CACHE = {}
_ROCM_COO_CSR_CACHE = TensorPlanCache()
_ILUVATAR_COO_SPMM_CSR_CACHE = TensorPlanCache()
_MUSA_COO_CSR_CACHE = TensorPlanCache()
_MUSA_CSC_CSR_CACHE = TensorPlanCache()
_MUSA_CSR_NNZ_CACHE = TensorPlanCache(maxsize=16)

# value dtype -> widened output dtypes (the first is the default for int8).
_WIDE_OUT = {
    torch.float16: (torch.float32,),
    torch.bfloat16: (torch.float32,),
    torch.int8: (torch.int32, torch.float32),
}
# real matrix dtype -> complex vector dtype it may multiply (SpMV only).
_REAL_TIMES_COMPLEX = {torch.float32: torch.complex64, torch.float64: torch.complex128}
# NVIDIA CUDA reuses compiled launchers (_musa_spmv_launch): Triton's JIT dispatch
# alone outlasts these kernels on short matrices. Needs a Triton with knobs.
try:
    from triton import knobs as _knobs  # noqa: F401

    _CUDA_DIRECT_LAUNCH = _backend_name() == "cuda"
except ImportError:
    _CUDA_DIRECT_LAUNCH = False


class PreparedCsrMixedSpmv:
    __slots__ = (
        "data", "indices", "indptr", "shape", "n_rows", "n_cols",
        "result_dtype", "acc_dtype", "rows_per", "block", "num_warps",
        "route", "bucket_plan", "nnz_plan",
    )

    def __init__(self, data, indices, indptr, shape, result_dtype, acc_dtype,
                 rows_per, block, num_warps, route, bucket_plan=None,
                 nnz_plan=None):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows, self.n_cols = self.shape
        self.result_dtype = result_dtype
        self.acc_dtype = acc_dtype
        self.rows_per = rows_per
        self.block = block
        self.num_warps = num_warps
        self.route = route
        self.bucket_plan = bucket_plan
        self.nnz_plan = nnz_plan


def _requested_out_dtype(out, out_dtype):
    if out is not None and out_dtype is not None and out.dtype != out_dtype:
        raise ValueError("out.dtype conflicts with out_dtype")
    return out_dtype if out_dtype is not None else (None if out is None else out.dtype)


def spmv_needs_mixed(
    data_dtype, x_dtype, out, out_dtype, *, coo=False, op=None, transpose=False,
    explicit_route=False,
):
    """Whether an SpMV call is one of the combinations handled in this module.

    ``op``/``transpose``/``explicit_route`` only gate BI-V's plain-f16 opt-in below:
    this module's kernels are op='non' only, and a caller that names an algorithm
    (alg/config/use_opt) asked for the native route.
    """
    requested = _requested_out_dtype(out, out_dtype)
    if (
        _is_iluvatar_runtime()
        and data_dtype == torch.float16
        and x_dtype == data_dtype
        and requested in (None, data_dtype)
        and not explicit_route
        and not transpose
        and str(op or "non").strip().lower() in ("non", "n", "non_trans")
    ):
        return True
    if data_dtype == torch.int8:
        return True
    if coo and data_dtype in _HALF:
        return True  # COO's native path has no half-precision kernel
    if x_dtype is not None and x_dtype != data_dtype and data_dtype in _REAL_TIMES_COMPLEX:
        return True
    return requested is not None and requested != data_dtype and data_dtype in _WIDE_OUT


def spmm_needs_mixed(data_dtype, out, out_dtype):
    requested = _requested_out_dtype(out, out_dtype)
    if data_dtype == torch.int8:
        return True
    return requested is not None and requested != data_dtype and data_dtype in _WIDE_OUT


def _resolve_types(data_dtype, rhs_dtype, out, out_dtype, allow_complex_rhs):
    """-> (output dtype, accumulate dtype); raises for combinations cuSPARSE lacks."""
    requested = _requested_out_dtype(out, out_dtype)
    if rhs_dtype != data_dtype:
        if not (allow_complex_rhs and _REAL_TIMES_COMPLEX.get(data_dtype) == rhs_dtype):
            raise TypeError(
                f"{data_dtype} matrix with {rhs_dtype} operand is not a supported combination"
            )
        if requested not in (None, rhs_dtype):
            raise TypeError(f"real-by-complex SpMV writes {rhs_dtype}")
        return rhs_dtype, data_dtype
    if data_dtype == torch.int8:
        result = requested or torch.int32
        if result not in _WIDE_OUT[torch.int8]:
            raise TypeError("int8 SpMV/SpMM writes int32 or float32")
        return result, (torch.int32 if result == torch.int32 else torch.float32)
    if data_dtype in _HALF:
        result = requested or data_dtype
        if result not in (data_dtype, torch.float32):
            raise TypeError(f"{data_dtype} SpMV/SpMM writes {data_dtype} or float32")
        return result, torch.float32
    if requested not in (None, data_dtype):
        raise TypeError(f"{data_dtype} SpMV/SpMM has no {requested} output")
    return data_dtype, data_dtype


def _normalize_non_op(op, transpose):
    token = "non" if op is None else str(op).strip().lower()
    if token in ("0", "n", "non_trans"):
        token = "non"
    if token != "non" or bool(transpose):
        raise NotImplementedError("mixed-precision/int8 paths support op='non' only")


def _check_1d(*pairs):
    for tensor, name in pairs:
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be a 1D tensor")
        if not _is_accel_tensor(tensor):
            raise ValueError(f"{name} must be an accelerator tensor")


def _index32(t):
    return t.to(torch.int32) if t.dtype == torch.int64 else t


def _csr_row_ids(indptr, n_rows, nnz):
    lengths = (indptr[1:] - indptr[:-1]).to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(n_rows, device=indptr.device, dtype=torch.int64), lengths, output_size=nnz
    )


def _finish(y, out, t0, return_time):
    if out is not None:
        if y is not out:
            out.copy_(y)
        y = out
    if return_time:
        _ACCEL.synchronize()
        return y, (time.perf_counter() - t0) * 1000.0
    return y


def _start(return_time):
    if not return_time:
        return None
    _ACCEL.synchronize()
    return time.perf_counter()


# ---------------------------------------------------------------------------
# CSR SpMV
# ---------------------------------------------------------------------------


# CSR SpMV walks a tile of ROWS rows per program, BLOCK entries of each row at a time.
# One program per row was 3-4x slower (cage12: 63 us vs 15 us, cuSPARSE fp32 21 us):
# 130k programs of ~16 entries each spend their time launching, not loading.
@triton.jit
def _csr_spmv_mixed_kernel(
    y_ptr, data_ptr, cols_ptr, indptr_ptr, x_ptr, n_rows,
    ACC: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    start = tl.load(indptr_ptr + rows, mask=rmask, other=0)
    end = tl.load(indptr_ptr + rows + 1, mask=rmask, other=0)
    max_len = tl.max(end - start, axis=0)
    acc = tl.zeros((ROWS, BLOCK), dtype=ACC)
    for j in range(0, max_len, BLOCK):
        offs = start[:, None] + j + tl.arange(0, BLOCK)[None, :]
        mask = offs < end[:, None]
        col = tl.load(cols_ptr + offs, mask=mask, other=0)
        v = tl.load(data_ptr + offs, mask=mask, other=0).to(ACC)
        xv = tl.load(x_ptr + col, mask=mask, other=0).to(ACC)
        acc += v * xv
    tl.store(y_ptr + rows, tl.sum(acc, axis=1).to(y_ptr.dtype.element_ty), mask=rmask)


@triton.jit
def _csr_spmv_mixed_cached_kernel(
    y_ptr, data_ptr, cols_ptr, indptr_ptr, x_ptr, n_rows,
    ACC: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    start = tl.load(indptr_ptr + rows, mask=rmask, other=0)
    end = tl.load(indptr_ptr + rows + 1, mask=rmask, other=0)
    max_len = tl.max(end - start, axis=0)
    acc = tl.zeros((ROWS, BLOCK), dtype=ACC)
    for j in range(0, max_len, BLOCK):
        offs = start[:, None] + j + tl.arange(0, BLOCK)[None, :]
        mask = offs < end[:, None]
        col = tl.load(cols_ptr + offs, mask=mask, other=0, cache_modifier=".cg")
        v = tl.load(data_ptr + offs, mask=mask, other=0, cache_modifier=".cg").to(ACC)
        xv = tl.load(x_ptr + col, mask=mask, other=0, cache_modifier=".ca").to(ACC)
        acc += v * xv
    tl.store(y_ptr + rows, tl.sum(acc, axis=1).to(y_ptr.dtype.element_ty), mask=rmask)


@triton.jit
def _csr_spmv_real_by_complex_kernel(
    y_ri_ptr, data_ptr, cols_ptr, indptr_ptr, x_ri_ptr, n_rows,
    ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    start = tl.load(indptr_ptr + rows, mask=rmask, other=0)
    end = tl.load(indptr_ptr + rows + 1, mask=rmask, other=0)
    max_len = tl.max(end - start, axis=0)
    acc_re = tl.zeros((ROWS, BLOCK), dtype=y_ri_ptr.dtype.element_ty)
    acc_im = tl.zeros((ROWS, BLOCK), dtype=y_ri_ptr.dtype.element_ty)
    for j in range(0, max_len, BLOCK):
        offs = start[:, None] + j + tl.arange(0, BLOCK)[None, :]
        mask = offs < end[:, None]
        col = tl.load(cols_ptr + offs, mask=mask, other=0)
        v = tl.load(data_ptr + offs, mask=mask, other=0.0)
        acc_re += v * tl.load(x_ri_ptr + 2 * col, mask=mask, other=0.0)
        acc_im += v * tl.load(x_ri_ptr + 2 * col + 1, mask=mask, other=0.0)
    tl.store(y_ri_ptr + 2 * rows, tl.sum(acc_re, axis=1), mask=rmask)
    tl.store(y_ri_ptr + 2 * rows + 1, tl.sum(acc_im, axis=1), mask=rmask)


@triton.jit
def _csr_spmv_real_by_complex_cached_kernel(
    y_ri_ptr, data_ptr, cols_ptr, indptr_ptr, x_ri_ptr, n_rows,
    ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    start = tl.load(indptr_ptr + rows, mask=rmask, other=0)
    end = tl.load(indptr_ptr + rows + 1, mask=rmask, other=0)
    max_len = tl.max(end - start, axis=0)
    acc = tl.zeros((ROWS, BLOCK, 2), dtype=tl.float32)
    components = tl.arange(0, 2)
    for offset in range(0, max_len, BLOCK):
        offs = start[:, None] + offset + tl.arange(0, BLOCK)[None, :]
        mask = (offs < end[:, None]) & rmask[:, None]
        col = tl.load(cols_ptr + offs, mask=mask, other=0, cache_modifier=".cg")
        value = tl.load(data_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg").to(tl.float32)
        vector = tl.load(
            x_ri_ptr + 2 * col[:, :, None] + components[None, None, :],
            mask=mask[:, :, None],
            other=0.0,
            cache_modifier=".ca",
        ).to(tl.float32)
        acc += value[:, :, None] * vector
    result = tl.sum(acc, axis=1)
    tl.store(y_ri_ptr + 2 * rows[:, None] + components[None, :], result, mask=rmask[:, None])


@triton.jit
def _csr_spmv_real_by_complex_bucket_kernel(
    y_ri_ptr, data_ptr, cols_ptr, indptr_ptr, x_ri_ptr, row_ids, n_rows, n_bucket_rows,
    ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    bucket_rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    bucket_mask = bucket_rows < n_bucket_rows
    rows = tl.load(row_ids + bucket_rows, mask=bucket_mask, other=0)
    row_mask = bucket_mask & (rows < n_rows)
    start = tl.load(indptr_ptr + rows, mask=row_mask, other=0)
    end = tl.load(indptr_ptr + rows + 1, mask=row_mask, other=0)
    max_len = tl.max(end - start, axis=0)
    acc = tl.zeros((ROWS, BLOCK, 2), dtype=tl.float32)
    components = tl.arange(0, 2)
    for offset in range(0, max_len, BLOCK):
        offs = start[:, None] + offset + tl.arange(0, BLOCK)[None, :]
        mask = (offs < end[:, None]) & row_mask[:, None]
        col = tl.load(cols_ptr + offs, mask=mask, other=0, cache_modifier=".cg")
        value = tl.load(data_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg")
        vector = tl.load(
            x_ri_ptr + 2 * col[:, :, None] + components[None, None, :],
            mask=mask[:, :, None], other=0.0, cache_modifier=".ca",
        )
        acc += value[:, :, None] * vector
    result = tl.sum(acc, axis=1)
    tl.store(y_ri_ptr + 2 * rows[:, None] + components[None, :], result, mask=row_mask[:, None])


def _iluvatar_csr_complex_bucket_plan(ptr, n_rows, cutoff=16):
    # Built once by prepare_spmv_csr_mixed and held by the prepared plan.
    cutoff = int(os.environ.get("FLAGSPARSE_ILUVATAR_CSR_BUCKET_CUTOFF", cutoff))
    lengths = ptr[1:] - ptr[:-1]
    short_rows = torch.nonzero(lengths <= cutoff, as_tuple=False).reshape(-1).to(torch.int32)
    long_rows = torch.nonzero(lengths > cutoff, as_tuple=False).reshape(-1).to(torch.int32)
    return torch.cat((short_rows, long_rows)), short_rows.numel(), long_rows.numel()


def _iluvatar_csr_complex_bucket_run(y, data, cols, ptr, x, n_rows, plan):
    row_ids, short_count, long_count = plan
    rows_per = max(1, int(os.environ.get("FLAGSPARSE_ILUVATAR_CSR_BUCKET_ROWS", "16")))
    block = max(1, int(os.environ.get("FLAGSPARSE_ILUVATAR_CSR_BUCKET_BLOCK", "16")))
    y_ri = torch.view_as_real(y).reshape(-1)
    x_ri = torch.view_as_real(x)
    count = short_count + long_count
    if count:
        _csr_spmv_real_by_complex_bucket_kernel[(triton.cdiv(count, rows_per),)](
            y_ri, data, cols, ptr, x_ri, row_ids, n_rows, count,
            ROWS=rows_per, BLOCK=block, num_warps=1, num_stages=2,
        )
    return y


def _iluvatar_csr_complex_nnz_plan(ptr, n_rows):
    lengths = ptr[1:] - ptr[:-1]
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=ptr.device, dtype=torch.int32),
        lengths, output_size=int(ptr[-1].item()),
    )
    empty = torch.nonzero(lengths == 0, as_tuple=False).reshape(-1).to(torch.int32)
    return row_ids, empty


def _iluvatar_csr_complex_nnz_run(y, data, cols, ptr, x, plan):
    from ._spmv_csr_nnz import _nnz_kernel

    row_ids, empty = plan
    nnz = data.numel()
    block = max(1, int(os.environ.get("FLAGSPARSE_ILUVATAR_CSR_NNZ_BLOCK", "128")))
    warps = max(1, min(8, int(os.environ.get("FLAGSPARSE_ILUVATAR_CSR_NNZ_WARPS", "1"))))
    grid = (triton.cdiv(max(nnz, empty.numel()), block),)
    if grid[0]:
        # Same launch on CUDA (5090, ASIC_680ks: 11.0 us vs row-tile 27 us,
        # cuSPARSE 15.7 us); MUSA=False, COMPLEX_RHS=True.
        _launch_mixed(
            _nnz_kernel, grid,
            (data, cols, ptr, row_ids, torch.view_as_real(x), torch.view_as_real(y), empty),
            (), (empty.numel(), nnz, block, tl.float32, False, True), warps,
        )
    return y


def _launch_mixed(kernel, grid, tensors, scalars, constants, num_warps, num_stages=None):
    """Positional launch; reuses the compiled launcher on NVIDIA CUDA."""
    if _CUDA_DIRECT_LAUNCH:
        _launch_musa_spmv(kernel, grid, tensors, scalars, constants,
                          num_warps=num_warps, num_stages=num_stages, cuda=True)
        return
    options = {"num_warps": num_warps}
    if num_stages is not None:
        options["num_stages"] = num_stages
    kernel[grid](*tensors, *scalars, *constants, **options)


def _iluvatar_csr_complex_use_plain_loads(ptr, n_rows):
    lengths = (ptr[1:] - ptr[:-1]).to(torch.float32)
    mean = float(lengths.mean().item())
    variance = float(((lengths - mean) ** 2).mean().item())
    coefficient = variance ** 0.5 / max(mean, 1.0)
    return mean <= 4.0 and coefficient >= 0.8


def _csr_mixed_tile(nnz, n_rows, n_cols):
    if _is_iluvatar_runtime():
        return _iluvatar_mixed_spmv_tile(nnz, n_rows, n_cols)
    rows_per, block = _csr_row_tile(nnz, n_rows)
    return rows_per, block, _csr_mixed_num_warps()


def _expected_rhs_dtype(data_dtype, result_dtype):
    if data_dtype == torch.float32 and result_dtype == torch.complex64:
        return torch.complex64
    if data_dtype == torch.float64 and result_dtype == torch.complex128:
        return torch.complex128
    return data_dtype


def _prepared_mixed_uses_call(data_dtype, result_dtype):
    """Backends whose mixed CSR route is not these Triton kernels keep it."""
    return (
        _is_mthreads_runtime()
        or _is_ascend_runtime()
        or (_is_rocm_runtime() and data_dtype == torch.float16 and result_dtype == torch.float16)
    )


def prepare_spmv_csr_mixed(data, indices, indptr, x_dtype, shape, *,
                           out=None, out_dtype=None, route="auto"):
    """Plan a mixed-precision / real-by-complex CSR SpMV once; run it with
    ``spmv_csr_mixed_prepared`` or ``flagsparse_spmv_csr(prepared=...)``.

    Row statistics, the short/long bucket order and the per-nonzero row ids are
    built here (synchronizing) and held by the plan. The plan keeps the CSR
    tensors, not copies: later in-place writes to their values are seen, but a
    changed sparsity structure needs a new plan. MUSA, Ascend and ROCm f16->f16
    keep their own call routes (``route == "call"``).
    """
    _check_1d((data, "data"), (indices, "indices"), (indptr, "indptr"))
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() != indices.numel() or indptr.numel() != n_rows + 1:
        raise ValueError("invalid CSR dimensions")
    result_dtype, acc_dtype = _resolve_types(data.dtype, x_dtype, out, out_dtype, True)
    data, indices, indptr = data.contiguous(), _index32(indices.contiguous()), _index32(indptr.contiguous())
    real_by_complex64 = data.dtype == torch.float32 and x_dtype == torch.complex64
    if _is_iluvatar_runtime() and real_by_complex64:
        rows_per, block, num_warps = _iluvatar_complex_mixed_spmv_tile(
            data.numel(), n_rows, n_cols
        )
    else:
        rows_per, block, num_warps = _csr_mixed_tile(data.numel(), n_rows, n_cols)
    selected = str(route or "auto").lower()
    bucket_plan = None
    nnz_plan = None
    if _prepared_mixed_uses_call(data.dtype, result_dtype):
        if selected not in ("auto", "call"):
            raise ValueError("this backend runs mixed CSR through its call route only")
        selected = "call"
    elif real_by_complex64 and (_is_iluvatar_runtime() or _CUDA_DIRECT_LAUNCH):
        if selected in ("auto", "adaptive"):
            lengths = indptr[1:] - indptr[:-1]
            mean = float(data.numel()) / max(1, n_rows)
            maximum = int(lengths.max().item()) if n_rows else 0
            if mean <= 4.0 and maximum > 128:
                selected = "nnz"
            elif _is_iluvatar_runtime() and _iluvatar_csr_complex_use_plain_loads(indptr, n_rows):
                selected = "plain"
            else:
                selected = "cached"
        if selected == "bucket":
            bucket_plan = _iluvatar_csr_complex_bucket_plan(indptr, n_rows)
        elif selected == "nnz":
            if int(indptr[0].item()) != 0 or int(indptr[-1].item()) != data.numel():
                raise ValueError("indptr must start at 0 and end at nnz")
            nnz_plan = _iluvatar_csr_complex_nnz_plan(indptr, n_rows)
        elif selected not in ("plain", "cached"):
            raise ValueError("route must be auto, cached, plain, bucket, or nnz")
    elif _is_iluvatar_runtime() and data.dtype in (torch.float16, torch.int8):
        nnz_route = os.environ.get(
            "FLAGSPARSE_ILUVATAR_CSR_MIXED_NNZ", "auto"
        ).strip().lower()
        if nnz_route not in ("auto", "0", "1"):
            raise ValueError(
                "FLAGSPARSE_ILUVATAR_CSR_MIXED_NNZ must be auto, 0, or 1"
            )
        if selected in ("auto", "adaptive"):
            lengths = indptr[1:] - indptr[:-1]
            mean = float(data.numel()) / max(1, n_rows)
            maximum = int(lengths.max().item()) if n_rows else 0
            if nnz_route != "0" and (
                nnz_route == "1" or (mean <= 4.0 and maximum > 128)
            ):
                selected = "nnz"
            else:
                selected = "cached"
        if selected == "nnz":
            if int(indptr[0].item()) != 0 or int(indptr[-1].item()) != data.numel():
                raise ValueError("indptr must start at 0 and end at nnz")
            nnz_plan = _iluvatar_csr_complex_nnz_plan(indptr, n_rows)
        elif selected not in ("plain", "cached"):
            raise ValueError("route must be auto, cached, plain, or nnz")
    elif selected not in ("auto", "cached", "plain"):
        raise ValueError("route is unsupported for this mixed CSR dtype combination")
    if selected == "auto":
        selected = "cached"
    return PreparedCsrMixedSpmv(
        data, indices, indptr, shape, result_dtype, acc_dtype,
        rows_per, block, num_warps, selected, bucket_plan, nnz_plan,
    )


def spmv_csr_mixed_prepared(prepared, x, *, out=None, return_time=False):
    if not isinstance(prepared, PreparedCsrMixedSpmv):
        raise TypeError("prepared must be PreparedCsrMixedSpmv")
    if not torch.is_tensor(x) or x.ndim != 1 or x.dtype != _expected_rhs_dtype(
        prepared.data.dtype, prepared.result_dtype
    ):
        raise TypeError("x has an incompatible dtype or shape")
    if x.numel() != prepared.n_cols or x.device != prepared.data.device:
        raise ValueError("x shape/device does not match the prepared CSR matrix")
    if out is not None and (
        out.shape != (prepared.n_rows,) or out.dtype != prepared.result_dtype
        or out.device != x.device
    ):
        raise ValueError("out shape/dtype/device does not match the prepared CSR result")
    if prepared.route == "call":
        return spmv_csr_mixed(
            prepared.data, prepared.indices, prepared.indptr, x, prepared.shape,
            out=out, out_dtype=prepared.result_dtype, return_time=return_time,
        )
    x = x.contiguous()
    # A strided out cannot be viewed as its real components; write a temporary.
    y = out if out is not None and out.is_contiguous() else torch.empty(
        prepared.n_rows, dtype=prepared.result_dtype, device=x.device
    )
    t0 = _start(return_time)
    if prepared.n_rows:
        grid = ((prepared.n_rows + prepared.rows_per - 1) // prepared.rows_per,)
        if prepared.route == "bucket":
            _iluvatar_csr_complex_bucket_run(
                y, prepared.data, prepared.indices, prepared.indptr,
                x, prepared.n_rows, prepared.bucket_plan,
            )
        elif prepared.route == "nnz":
            if _is_complex_dtype(prepared.result_dtype):
                _iluvatar_csr_complex_nnz_run(
                    y, prepared.data, prepared.indices, prepared.indptr,
                    x, prepared.nnz_plan,
                )
            else:
                from ._spmv_csr_nnz import _nnz_kernel

                row_ids, empty = prepared.nnz_plan
                block = max(
                    32,
                    int(os.environ.get("FLAGSPARSE_ILUVATAR_CSR_MIXED_NNZ_BLOCK", "128")),
                )
                warps = max(
                    1,
                    min(
                        8,
                        int(os.environ.get("FLAGSPARSE_ILUVATAR_CSR_MIXED_NNZ_WARPS", "1")),
                    ),
                )
                nnz = prepared.data.numel()
                grid = (triton.cdiv(max(nnz, empty.numel()), block),)
                _nnz_kernel[grid](
                    prepared.data, prepared.indices, prepared.indptr,
                    row_ids, x, y, empty, empty.numel(), nnz, block,
                    _TL_ACC[prepared.acc_dtype], num_warps=warps,
                )
        elif _is_complex_dtype(prepared.result_dtype):
            cached = prepared.route == "cached"
            kernel = (
                _csr_spmv_real_by_complex_cached_kernel if cached
                else _csr_spmv_real_by_complex_kernel
            )
            _launch_mixed(
                kernel, grid,
                (torch.view_as_real(y), prepared.data, prepared.indices,
                 prepared.indptr, torch.view_as_real(x)),
                (prepared.n_rows,), (prepared.rows_per, prepared.block),
                prepared.num_warps, 2 if cached else None,
            )
        else:
            _launch_mixed(
                _csr_spmv_mixed_kernel, grid,
                (y, prepared.data, prepared.indices, prepared.indptr, x),
                (prepared.n_rows,),
                (_TL_ACC[prepared.acc_dtype], prepared.rows_per, prepared.block),
                prepared.num_warps,
            )
    return _finish(y, out, t0, return_time)


@triton.jit
def _musa_csr_real_by_complex_kernel(
    Y, V, C, P, X, M, R: tl.constexpr, K: tl.constexpr, ACC: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    start = tl.load(P + rows, rows < M, other=0)
    end = tl.load(P + rows + 1, rows < M, other=0)
    lanes = tl.arange(0, K)
    components = tl.arange(0, 2)
    acc = tl.zeros((R, K, 2), ACC)
    for offset in range(0, tl.max(end - start, 0), K):
        pos = start[:, None] + offset + lanes[None, :]
        valid = (rows[:, None] < M) & (pos < end[:, None])
        col = tl.load(C + pos, valid, other=0)
        value = tl.load(V + pos, valid, other=0)
        vector = tl.load(X + 2 * col[:, :, None] + components[None, None, :],
                         valid[:, :, None], other=0)
        acc += value[:, :, None] * vector
    tl.store(Y + 2 * rows[:, None] + components[None, :], tl.sum(acc, 1), rows[:, None] < M)


def _csr_row_tile(nnz, n_rows):
    """(ROWS, BLOCK) from the mean row length: one BLOCK covers a typical row.

    Measured on a 5090 over cage12 / GL7d14 / roadNet-TX / amazon0601 / wave; a
    BLOCK much wider than the rows wastes the tile (roadNet-TX, 2.8 nnz/row:
    16x32 took 72 us, 128x4 took 14 us). ROWS * BLOCK stays 512 lanes, which
    four warps cover at warp size 32 or 64.
    """
    mean = nnz / max(1, n_rows)
    if mean <= 4:
        return 128, 4
    if mean <= 12:
        return 64, 8
    if mean <= 24:
        return 32, 16
    return 16, 32


def _musa_row_tile(nnz, n_rows, integer=False):
    override = os.environ.get("FLAGSPARSE_MUSA_CSR_TILE")
    if override:
        try:
            rows, block = (int(v.strip()) for v in override.split(",", 1))
            if rows > 0 and block > 0:
                return rows, block
        except (TypeError, ValueError):
            pass
    mean = nnz / max(1, n_rows)
    if mean <= 3:
        return 256, 2
    if 4 < mean <= 12:
        return (32, 4) if integer or n_rows >= 262144 else (64, 8)
    if 12 < mean <= 24:
        return (128, 8) if n_rows >= 262144 else (32, 4)
    return 16, 8


@triton.jit
def _musa_csr_half_fixed_kernel(
    Y, V, C, P, X, M, ROWS: tl.constexpr, BLOCK: tl.constexpr,
    MAX_LENGTH: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    begin = tl.load(P + rows, rows < M, other=0)
    end = tl.load(P + rows + 1, rows < M, other=0)
    acc = tl.zeros((ROWS, BLOCK), tl.float32)
    # The version-checked row pointers bound every row, so no per-program
    # reduction is needed to discover the loop limit on each invocation.
    for offset in range(0, MAX_LENGTH, BLOCK):
        pos = begin[:, None] + offset + tl.arange(0, BLOCK)[None, :]
        valid = (rows[:, None] < M) & (pos < end[:, None])
        col = tl.load(C + pos, valid, other=0)
        value = tl.load(V + pos, valid, other=0).to(tl.float32)
        rhs = tl.load(X + col, valid, other=0).to(tl.float32)
        acc += value * rhs
    tl.store(Y + rows, tl.sum(acc, 1), rows < M)


def _musa_coo_half_spmv(data, cols, ptr, x, n_rows, plans, out=None):
    """Reuse topology and launch specializations; always read live operands."""
    plan = plans.get("half_f32")
    if plan is None:
        lengths = ptr[1:] - ptr[:-1]
        maximum = int(lengths.max().item())
        mean = data.numel() / n_rows
        nnz_mode = maximum > 128 and mean <= 4
        if nnz_mode:
            # Keep row ownership and the existing cross-block prefix merge.
            from ._spmv_csr_nnz import _nnz_kernel

            row_ids = torch.repeat_interleave(
                torch.arange(n_rows, device=ptr.device, dtype=torch.int32),
                lengths.long(), output_size=data.numel(),
            )
            empty = torch.nonzero(lengths == 0).flatten().to(torch.int32)
            sparse = (data, cols, ptr, row_ids, empty)
            kernel = _nnz_kernel
            grid = ((max(data.numel(), empty.numel()) + 511) // 512,)
            scalars = ()
            constants = (empty.numel(), data.numel(), 512, tl.float32, True, False)
        else:
            sparse = (data, cols, ptr)
            if maximum <= 128 and mean > 4:
                rows, block = (32, 4) if mean <= 12 and n_rows >= 262144 else (16, 8)
                kernel = _musa_csr_half_fixed_kernel
                constants = (rows, block, maximum)
            else:
                rows, block = _musa_row_tile(data.numel(), n_rows)
                kernel = _csr_spmv_mixed_kernel
                constants = (tl.float32, rows, block)
            grid = ((n_rows + rows - 1) // rows,)
            scalars = (n_rows,)
        plan = (kernel, grid, sparse, tuple(t.data_ptr() for t in sparse),
                scalars, constants, nnz_mode, {})
        plans["half_f32"] = plan
    kernel, grid, sparse, sparse_addresses, scalars, constants, nnz_mode, runners = plan
    y = out if out is not None and out.is_contiguous() else torch.empty(
        n_rows, dtype=torch.float32, device=x.device
    )
    yp, xp = y.data_ptr(), x.data_ptr()
    # The sparse cache checks all source versions/storage/metadata. RHS and
    # output may change on every call: preserve alignment/device/alias JIT
    # specializations while resolving their current pointers and stream.
    aliases = tuple((p == yp, p == xp) for p in sparse_addresses) + (yp == xp,)
    key = (x.dtype, x.device, xp % 16, yp % 16, aliases, kernel.debug)
    runner = runners.get(key)
    if nnz_mode:
        tensors = (*sparse[:4], x, y, sparse[4])
        addresses = (*sparse_addresses[:4], xp, yp, sparse_addresses[4])
    else:
        tensors = (y, *sparse, x)
        addresses = (yp, *sparse_addresses, xp)
    if runner is None:
        runner = _launch_musa_spmv(
            kernel, grid, tensors, scalars, constants,
        )
        if runner is not None:
            if len(runners) >= 16:
                runners.pop(next(iter(runners)))
            runners[key] = runner
    else:
        runner(tensors, scalars, constants, addresses)
    return y


def _musa_coo_csr(data, row, col, shape, *, spmv=False):
    """Reuse row ownership, invalidating both topology and values on mutation."""
    sources = (data, row, col)
    options = tuple(shape)
    state = _MUSA_COO_CSR_CACHE.get(sources, options)
    if state is None:
        _check_1d((data, "data"), (row, "row"), (col, "col"))
        if not (data.numel() == row.numel() == col.numel()):
            raise ValueError("data, row and col must have the same length")
        rows, cols = _index32(row.contiguous()), _index32(col.contiguous())
        if rows.numel() > 1 and not bool(torch.all(rows[1:] >= rows[:-1]).item()):
            order = torch.argsort(rows, stable=True)
            rows = rows.index_select(0, order)
            cols = cols.index_select(0, order)
            data = data.index_select(0, order)
        # MUSA bincount can lose updates for repeated sorted row IDs.
        boundaries = torch.arange(int(shape[0]) + 1, device=rows.device, dtype=rows.dtype)
        ptr = torch.searchsorted(rows, boundaries).to(torch.int32)
        state = data.contiguous(), cols.contiguous(), ptr.contiguous(), {}
        _MUSA_COO_CSR_CACHE.put(sources, options, state)
    return state if spmv else state[:3]


def _musa_csr_spmv(data, cols, ptr, x, n_rows, result_dtype, acc_dtype, out=None):
    y = out if out is not None and out.is_contiguous() else torch.empty(
        n_rows, dtype=result_dtype, device=x.device
    )
    if not n_rows:
        return y
    if data.numel() / n_rows <= 4:
        plan = _MUSA_CSR_NNZ_CACHE.get((ptr,), (n_rows,))
        if plan is None:
            lengths = ptr[1:] - ptr[:-1]
            if int(lengths.max().item()) > 128:
                rows = torch.repeat_interleave(
                    torch.arange(n_rows, device=ptr.device, dtype=torch.int32),
                    lengths.long(), output_size=data.numel(),
                )
                empty = torch.nonzero(lengths == 0).flatten().to(torch.int32)
                plan = rows, empty
            else:
                plan = ()
            _MUSA_CSR_NNZ_CACHE.put((ptr,), (n_rows,), plan)
        if plan:
            from ._spmv_csr_nnz import _nnz_kernel
            rows, empty = plan
            complex_rhs = _is_complex_dtype(result_dtype)
            _launch_musa_spmv(
                _nnz_kernel, ((max(data.numel(), empty.numel()) + 511) // 512,),
                (data, cols, ptr, rows,
                 torch.view_as_real(x) if complex_rhs else x,
                 torch.view_as_real(y) if complex_rhs else y, empty),
                (), (empty.numel(), data.numel(), 512, _TL_ACC[acc_dtype], True, complex_rhs),
            )
            return y
    rows_per, block = _musa_row_tile(data.numel(), n_rows, data.dtype == torch.int8)
    if _is_complex_dtype(result_dtype):
        mean = data.numel() / n_rows
        rows_per, block = (128, 2) if mean <= 3 else (32, 4) if mean <= 12 else (16, 8)
        _musa_csr_real_by_complex_kernel[((n_rows + rows_per - 1) // rows_per,)](
            torch.view_as_real(y), data, cols, ptr,
            torch.view_as_real(x), n_rows, rows_per, block, _TL_ACC[acc_dtype],
            num_warps=4, num_stages=1,
        )
    else:
        _launch_musa_spmv(
            _csr_spmv_mixed_kernel, ((n_rows + rows_per - 1) // rows_per,),
            (y, data, cols, ptr, x), (n_rows,), (_TL_ACC[acc_dtype], rows_per, block),
        )
    return y


def _csr_mixed_num_warps():
    """Return the mixed CSR SpMV launch width for ROCm tuning."""
    if _is_rocm_runtime():
        value = os.environ.get("FLAGSPARSE_CSR_MIXED_WARPS")
        if value:
            return max(1, min(8, int(value)))
    return 4


def _iluvatar_mixed_spmv_tile(nnz, n_rows, n_cols):
    mean_row_nnz = nnz / max(1, n_rows)
    if mean_row_nnz <= 3:
        rows_per, block, num_warps = 64, 2, 1
    elif mean_row_nnz <= 12:
        rows_per, block, num_warps = 16, 8, 1
    elif mean_row_nnz <= 24:
        rows_per, block, num_warps = 16, 8, 1
    else:
        rows_per, block, num_warps = 8, 8, 1
    rows_override = os.environ.get("FLAGSPARSE_ILUVATAR_MIXED_ROWS")
    block_override = os.environ.get("FLAGSPARSE_ILUVATAR_MIXED_BLOCK")
    warps_override = os.environ.get("FLAGSPARSE_ILUVATAR_MIXED_WARPS")
    if rows_override:
        rows_per = max(1, int(rows_override))
    if block_override:
        block = max(1, int(block_override))
    if warps_override:
        num_warps = max(1, min(8, int(warps_override)))
    return rows_per, block, num_warps


def _iluvatar_complex_mixed_spmv_tile(nnz, n_rows, n_cols):
    mean_row_nnz = nnz / max(1, n_rows)
    if mean_row_nnz <= 3:
        rows_per, block, num_warps = 64, 2, 1
    elif mean_row_nnz <= 12:
        rows_per, block, num_warps = 16, 8, 1
    else:
        rows_per, block, num_warps = 8, 8, 1
    rows_override = os.environ.get("FLAGSPARSE_ILUVATAR_MIXED_ROWS")
    block_override = os.environ.get("FLAGSPARSE_ILUVATAR_MIXED_BLOCK")
    warps_override = os.environ.get("FLAGSPARSE_ILUVATAR_MIXED_WARPS")
    if rows_override:
        rows_per = max(1, int(rows_override))
    if block_override:
        block = max(1, int(block_override))
    if warps_override:
        num_warps = max(1, min(8, int(warps_override)))
    return rows_per, block, num_warps


def _torch_spmv_rows(data, rows, cols, x, n_out, acc_dtype):
    """Ascend path: gather, multiply in the accumulate type, index_add by output row."""
    cols = cols.to(torch.int64)
    if _is_complex_dtype(x.dtype):
        xv = torch.view_as_real(x).index_select(0, cols)
        d = data.to(xv.dtype).unsqueeze(-1)
        y = torch.zeros(n_out, dtype=x.dtype, device=x.device)
        torch.view_as_real(y).index_add_(0, rows, d * xv)
        return y
    y = torch.zeros(n_out, dtype=acc_dtype, device=x.device)
    y.index_add_(0, rows, data.to(acc_dtype) * x.index_select(0, cols).to(acc_dtype))
    return y


def spmv_csr_mixed(data, indices, indptr, x, shape, *, op=None, transpose=None,
                   out=None, out_dtype=None, return_time=False):
    _normalize_non_op(op, transpose)
    _check_1d((data, "data"), (indices, "indices"), (indptr, "indptr"), (x, "x"))
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() != indices.numel() or indptr.numel() != n_rows + 1:
        raise ValueError("invalid CSR dimensions")
    if x.numel() != n_cols:
        raise ValueError("x shape does not match CSR operation")
    result_dtype, acc_dtype = _resolve_types(data.dtype, x.dtype, out, out_dtype, True)
    if out is not None and (out.shape != (n_rows,) or out.device != x.device):
        raise ValueError("out shape/device must match the CSR SpMV result")
    data, x = data.contiguous(), x.contiguous()
    cols, ptr = _index32(indices.contiguous()), _index32(indptr.contiguous())
    t0 = _start(return_time)
    if _is_mthreads_runtime():
        y = _musa_csr_spmv(data, cols, ptr, x, n_rows, result_dtype, acc_dtype, out)
        return _finish(y, out, t0, return_time)
    if (
        _is_rocm_runtime()
        and data.dtype == torch.float16
        and result_dtype == torch.float16
    ):
        # Reuse the same native CSR descriptor as the PyTorch baseline.  The
        # benchmark calls this route repeatedly with a stable matrix; version
        # tracking keeps prepared callers correct when values are mutated.
        key = (data.data_ptr(), data.numel(), str(data.device), str(indices.dtype), n_rows, n_cols)
        state = _ROCM_CSR_MIXED_CACHE.get(key)
        version = getattr(data, "_version", -1)
        if state is None:
            values = data.to(torch.float32)
            matrix = torch.sparse_csr_tensor(
                ptr.to(torch.int64), cols.to(torch.int64), values,
                size=(n_rows, n_cols), device=data.device,
            )
            state = (matrix, values, version, None, None, None)
            _ROCM_CSR_MIXED_CACHE[key] = state
        elif state[2] != version:
            state[1].copy_(data.to(torch.float32))
            state = (state[0], state[1], version, None, None, None)
            _ROCM_CSR_MIXED_CACHE[key] = state
        x_version = getattr(x, "_version", -1)
        if state[3] != x.data_ptr() or state[4] != x_version:
            state = (state[0], state[1], state[2], x.data_ptr(), x_version, x.to(torch.float32))
            _ROCM_CSR_MIXED_CACHE[key] = state
        y = torch.sparse.mm(state[0], state[5].unsqueeze(1)).squeeze(1)
        return _finish(y.to(result_dtype), out, t0, return_time)
    if _is_ascend_runtime():
        rows = _csr_row_ids(ptr, n_rows, data.numel())
        y = _torch_spmv_rows(data, rows, cols, x, n_rows, acc_dtype).to(result_dtype)
        return _finish(y, out, t0, return_time)
    # A strided out cannot be viewed as its real components; write a temporary.
    y = out if out is not None and out.is_contiguous() else torch.empty(
        n_rows, dtype=result_dtype, device=x.device
    )
    if n_rows and _CUDA_DIRECT_LAUNCH:
        rows_per, block = _csr_row_tile(data.numel(), n_rows)
        grid = ((n_rows + rows_per - 1) // rows_per,)
        if _is_complex_dtype(result_dtype):
            kernel, constants = _csr_spmv_real_by_complex_kernel, (rows_per, block)
            tensors = (torch.view_as_real(y), data, cols, ptr, torch.view_as_real(x))
        else:
            kernel = _csr_spmv_mixed_kernel
            constants = (_TL_ACC[acc_dtype], rows_per, block)
            tensors = (y, data, cols, ptr, x)
        _launch_musa_spmv(kernel, grid, tensors, (n_rows,), constants, num_warps=4, cuda=True)
    elif n_rows:
        rows_per, block = _csr_row_tile(data.numel(), n_rows)
        num_warps = _csr_mixed_num_warps()
        if _is_iluvatar_runtime():
            if data.dtype == torch.float32 and x.dtype == torch.complex64:
                rows_per, block, num_warps = _iluvatar_complex_mixed_spmv_tile(
                    data.numel(), n_rows, n_cols
                )
            else:
                rows_per, block, num_warps = _iluvatar_mixed_spmv_tile(
                    data.numel(), n_rows, n_cols
                )
        grid = (triton.cdiv(n_rows, rows_per),)
        if _is_complex_dtype(result_dtype):
            if (
                _is_iluvatar_runtime()
                and data.dtype == torch.float32
                and x.dtype == torch.complex64
            ):
                grid = (triton.cdiv(n_rows, rows_per),)
                _csr_spmv_real_by_complex_cached_kernel[grid](
                    torch.view_as_real(y).reshape(-1), data, cols, ptr,
                    torch.view_as_real(x), n_rows,
                    ROWS=rows_per, BLOCK=block, num_warps=num_warps, num_stages=2,
                )
            else:
                _csr_spmv_real_by_complex_kernel[grid](
                    torch.view_as_real(y).reshape(-1), data, cols, ptr,
                    torch.view_as_real(x).reshape(-1), n_rows,
                    ROWS=rows_per, BLOCK=block, num_warps=num_warps,
                )
        else:
            kernel = _csr_spmv_mixed_cached_kernel if _is_iluvatar_runtime() else _csr_spmv_mixed_kernel
            kernel[grid](
                y, data, cols, ptr, x, n_rows,
                ACC=_TL_ACC[acc_dtype], ROWS=rows_per, BLOCK=block,
                num_warps=num_warps,
            )
    return _finish(y, out, t0, return_time)


# ---------------------------------------------------------------------------
# COO SpMV
# ---------------------------------------------------------------------------


@triton.jit
def _coo_spmv_atomic_kernel(
    acc_ptr, data_ptr, rows_ptr, cols_ptr, x_ptr, nnz,
    ACC: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz
    row = tl.load(rows_ptr + offs, mask=mask, other=0)
    col = tl.load(cols_ptr + offs, mask=mask, other=0)
    v = tl.load(data_ptr + offs, mask=mask, other=0).to(ACC)
    xv = tl.load(x_ptr + col, mask=mask, other=0).to(ACC)
    tl.atomic_add(acc_ptr + row, v * xv, mask=mask)


def spmv_coo_mixed(data, row, col, x, shape, *, op=None, transpose=None,
                   out=None, out_dtype=None, return_time=False):
    _normalize_non_op(op, transpose)
    musa = _is_mthreads_runtime()
    if musa:
        # Cached sparse operands retain their metadata and mutation signature.
        _check_1d((x, "x"))
    else:
        _check_1d((data, "data"), (row, "row"), (col, "col"), (x, "x"))
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if not musa and not (data.numel() == row.numel() == col.numel()):
        raise ValueError("data, row and col must have the same length")
    if x.numel() != n_cols:
        raise ValueError("x shape does not match COO operation")
    result_dtype, acc_dtype = _resolve_types(data.dtype, x.dtype, out, out_dtype, False)
    if out is not None and (out.shape != (n_rows,) or out.device != x.device):
        raise ValueError("out shape/device must match the COO SpMV result")
    if musa:
        t0 = _start(return_time)
        values, cols, ptr, plans = _musa_coo_csr(data, row, col, shape, spmv=True)
        if data.dtype == torch.float16 and result_dtype == torch.float32 and n_rows:
            y = _musa_coo_half_spmv(values, cols, ptr, x.contiguous(), n_rows, plans, out)
        else:
            y = _musa_csr_spmv(values, cols, ptr, x.contiguous(), n_rows,
                               result_dtype, acc_dtype, out)
        return _finish(y, out, t0, return_time)
    sources = (data, row, col)
    data, x = data.contiguous(), x.contiguous()
    t0 = _start(return_time)
    if _is_ascend_runtime():
        rows, cols = _index32(row.contiguous()), _index32(col.contiguous())
        y = _torch_spmv_rows(data, rows.to(torch.int64), cols, x, n_rows, acc_dtype)
        return _finish(y.to(result_dtype), out, t0, return_time)
    if _is_rocm_runtime() or _is_iluvatar_runtime():
        # Group sorted COO rows into CSR tiles instead of atomically updating
        # every output row once per nonzero.
        options = (n_rows, n_cols)
        state = _ROCM_COO_CSR_CACHE.get(sources, options)
        if state is None:
            rows, cols = _index32(row.contiguous()), _index32(col.contiguous())
            ordered = True
            if rows.numel() > 1:
                ordered = bool(torch.all(rows[1:] >= rows[:-1]).item())
            if ordered:
                use_data, use_cols = data, cols
            else:
                order = torch.argsort(rows.to(torch.int64), stable=True)
                use_data = data.index_select(0, order).contiguous()
                use_cols = cols.index_select(0, order).contiguous()
                rows = rows.index_select(0, order).contiguous()
            counts = torch.bincount(rows.to(torch.int64), minlength=n_rows)
            ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int32).contiguous()
            nnz_plan = None
            nnz_route = os.environ.get(
                "FLAGSPARSE_ILUVATAR_COO_MIXED_NNZ", "auto"
            ).strip().lower()
            if _is_iluvatar_runtime() and nnz_route not in ("0", "false", "off"):
                lengths = ptr[1:] - ptr[:-1]
                if data.numel() / max(1, n_rows) <= 4 and int(lengths.max().item()) > 128:
                    nnz_rows = torch.repeat_interleave(
                        torch.arange(n_rows, device=rows.device, dtype=torch.int32),
                        lengths,
                        output_size=data.numel(),
                    )
                    empty_rows = torch.nonzero(lengths == 0).flatten().to(torch.int32)
                    nnz_plan = (nnz_rows.contiguous(), empty_rows.contiguous())
            state = (use_data, use_cols, ptr, nnz_plan)
            _ROCM_COO_CSR_CACHE.put(sources, options, state)
        if len(state) == 3:
            use_data, use_cols, ptr = state
            nnz_plan = None
        else:
            use_data, use_cols, ptr, nnz_plan = state
        y = torch.empty(n_rows, dtype=result_dtype, device=x.device)
        if n_rows:
            if _is_iluvatar_runtime() and nnz_plan is not None:
                from ._spmv_csr_nnz import _nnz_kernel

                nnz_rows, empty_rows = nnz_plan
                if data.dtype == torch.float16 and x.dtype == torch.float32:
                    block, num_warps = 128, 2
                else:
                    block, num_warps = 256, 2
                grid = (triton.cdiv(max(data.numel(), empty_rows.numel()), block),)
                _nnz_kernel[grid](
                    use_data, use_cols, ptr, nnz_rows, x, y, empty_rows,
                    empty_rows.numel(), data.numel(), block, _TL_ACC[acc_dtype],
                    num_warps=num_warps,
                )
                return _finish(y, out, t0, return_time)
            rows_per, block = _csr_row_tile(data.numel(), n_rows)
            num_warps = _csr_mixed_num_warps()
            if _is_iluvatar_runtime():
                rows_per, block, num_warps = _iluvatar_mixed_spmv_tile(
                    data.numel(), n_rows, n_cols
                )
            kernel = _csr_spmv_mixed_cached_kernel if _is_iluvatar_runtime() else _csr_spmv_mixed_kernel
            kernel[(triton.cdiv(n_rows, rows_per),)](
                y, use_data, use_cols, ptr, x, n_rows,
                ACC=_TL_ACC[acc_dtype], ROWS=rows_per, BLOCK=block,
                num_warps=num_warps,
            )
        return _finish(y, out, t0, return_time)
    rows, cols = _index32(row.contiguous()), _index32(col.contiguous())
    acc = torch.zeros(n_rows, dtype=acc_dtype, device=x.device)
    nnz = data.numel()
    if nnz:
        block = 256
        _coo_spmv_atomic_kernel[(triton.cdiv(nnz, block),)](
            acc, data, rows, cols, x, nnz, ACC=_TL_ACC[acc_dtype], BLOCK=block, num_warps=4,
        )
    y = acc if acc_dtype == result_dtype else acc.to(result_dtype)
    return _finish(y, out, t0, return_time)


# ---------------------------------------------------------------------------
# CSR SpMM (C = A @ B, B/C row- or column-major through their strides)
# ---------------------------------------------------------------------------


@triton.jit
def _csc_spmv_mixed_kernel(A, I, P, X, Y, N, R: tl.constexpr, K: tl.constexpr):
    cols = tl.program_id(0) * R + tl.arange(0, R)
    valid_cols = cols < N
    starts = tl.load(P + cols, valid_cols, other=0)
    ends = tl.load(P + cols + 1, valid_cols, other=0)
    x = tl.load(X + cols, valid_cols, other=0.).to(tl.float32)
    lanes = tl.arange(0, K)
    for base in range(0, tl.max(ends - starts, 0), K):
        pos = starts[:, None] + base + lanes[None, :]
        mask = valid_cols[:, None] & (pos < ends[:, None])
        row = tl.load(I + pos, mask, other=0)
        v = tl.load(A + pos, mask, other=0.).to(tl.float32)
        tl.atomic_add(Y + row, v * x[:, None], mask, sem="relaxed")


def spmv_csc_half(data, indices, indptr, x, shape, *, out=None, out_dtype=None, return_time=False):
    """Read CSC half values directly; avoid expanding column IDs every call."""
    _check_1d((data, "data"), (indices, "indices"), (indptr, "indptr"), (x, "x"))
    m, n = map(int, shape)
    if x.numel() != n:
        raise ValueError("x shape does not match CSC operation")
    result_dtype, acc_dtype = _resolve_types(data.dtype, x.dtype, out, out_dtype, False)
    if acc_dtype != torch.float32:
        raise TypeError("CSC half route requires float32 accumulation")
    if out is not None and (out.shape != (m,) or out.device != x.device):
        raise ValueError("out shape/device must match CSC SpMV result")
    if _is_mthreads_runtime():
        sources = (data, indices, indptr)
        state = _MUSA_CSC_CSR_CACHE.get(sources, (m, n))
        if state is None:
            from . import _spmv_csr_transpose
            cols, ptr, order, *_ = _spmv_csr_transpose.prepare(indices, indptr, (n, m))
            state = data.index_select(0, order.long()).contiguous(), cols, ptr
            _MUSA_CSC_CSR_CACHE.put(sources, (m, n), state)
        values, cols, ptr = state
        return spmv_csr_mixed(
            values, cols, ptr, x, shape, out=out, out_dtype=result_dtype,
            return_time=return_time,
        )
    t0 = _start(return_time)
    acc = torch.zeros(m, dtype=torch.float32, device=x.device)
    if n and data.numel():
        batch, lanes = _csr_row_tile(data.numel(), n)
        # Atomic CSC tiles are sensitive to skewed column lengths: fewer columns
        # per program avoid looping an entire large batch for one long column.
        if data.numel() / n > 4:
            batch, lanes = 16, 32
        _csc_spmv_mixed_kernel[(triton.cdiv(n, batch),)](
            data.contiguous(), indices.contiguous(), indptr.contiguous(), x.contiguous(),
            acc, n, batch, lanes, num_warps=4, num_stages=1)
    return _finish(acc.to(result_dtype), out, t0, return_time)


@triton.jit
def _csr_spmm_mixed_kernel(
    c_ptr, data_ptr, cols_ptr, indptr_ptr, b_ptr, n_dense,
    stride_bk, stride_bn, stride_cm, stride_cn,
    ACC: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    n_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < n_dense
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    acc = tl.zeros((BLOCK_N,), dtype=ACC)
    for p in range(start, end):
        col = tl.load(cols_ptr + p)
        a = tl.load(data_ptr + p).to(ACC)
        b = tl.load(b_ptr + col * stride_bk + n_offs * stride_bn, mask=n_mask, other=0).to(ACC)
        acc += a * b
    tl.store(
        c_ptr + row * stride_cm + n_offs * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=n_mask,
    )


@triton.jit
def _csr_spmm_mixed_rows_kernel(
    c_ptr, data_ptr, cols_ptr, indptr_ptr, b_ptr, n_rows, n_dense,
    stride_bk, stride_bn, stride_cm, stride_cn,
    ACC: tl.constexpr, ROWS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    n_mask = n_offs < n_dense
    starts = tl.load(indptr_ptr + rows, row_mask, other=0)
    ends = tl.load(indptr_ptr + rows + 1, row_mask, other=0)
    acc = tl.zeros((ROWS, BLOCK_N), dtype=ACC)
    max_length = tl.max(ends - starts, 0)
    for offset in range(max_length):
        pos = starts + offset
        valid = row_mask & (pos < ends)
        col = tl.load(cols_ptr + pos, valid, other=0)
        value = tl.load(data_ptr + pos, valid, other=0).to(ACC)
        dense = tl.load(
            b_ptr + col[:, None] * stride_bk + n_offs[None, :] * stride_bn,
            valid[:, None] & n_mask[None, :],
            other=0,
        ).to(ACC)
        acc += value[:, None] * dense
    tl.store(
        c_ptr + rows[:, None] * stride_cm + n_offs[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        row_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _csr_spmm_complex_rows_kernel(
    c_ri_ptr, data_ri_ptr, cols_ptr, indptr_ptr, b_ri_ptr, n_rows, n_dense,
    stride_bk, stride_bn, stride_br, stride_cm, stride_cn, stride_cr,
    ACC: tl.constexpr, ROWS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    n_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < n_rows
    n_mask = n_offs < n_dense
    starts = tl.load(indptr_ptr + rows, row_mask, other=0)
    ends = tl.load(indptr_ptr + rows + 1, row_mask, other=0)
    acc_re = tl.zeros((ROWS, BLOCK_N), dtype=ACC)
    acc_im = tl.zeros((ROWS, BLOCK_N), dtype=ACC)
    max_length = tl.max(ends - starts, 0)
    for offset in range(max_length):
        pos = starts + offset
        valid = row_mask & (pos < ends)
        col = tl.load(cols_ptr + pos, valid, other=0)
        a_re = tl.load(data_ri_ptr + pos * 2, valid, other=0.0).to(ACC)
        a_im = tl.load(data_ri_ptr + pos * 2 + 1, valid, other=0.0).to(ACC)
        b_base = col[:, None] * stride_bk + n_offs[None, :] * stride_bn
        b_re = tl.load(b_ri_ptr + b_base, valid[:, None] & n_mask[None, :], other=0.0).to(ACC)
        b_im = tl.load(
            b_ri_ptr + b_base + stride_br,
            valid[:, None] & n_mask[None, :],
            other=0.0,
        ).to(ACC)
        acc_re += a_re[:, None] * b_re - a_im[:, None] * b_im
        acc_im += a_re[:, None] * b_im + a_im[:, None] * b_re
    out_base = rows[:, None] * stride_cm + n_offs[None, :] * stride_cn
    mask = row_mask[:, None] & n_mask[None, :]
    tl.store(c_ri_ptr + out_base, acc_re, mask=mask)
    tl.store(c_ri_ptr + out_base + stride_cr, acc_im, mask=mask)


def spmm_csr_mixed(data, indices, indptr, B, shape, *, op=None, transpose=None,
                   out=None, out_dtype=None, return_time=False):
    """``B`` may be any 2D strided view (column-major, or a transposed ``op_b``)."""
    _normalize_non_op(op, transpose)
    _check_1d((data, "data"), (indices, "indices"), (indptr, "indptr"))
    if not torch.is_tensor(B) or B.ndim != 2 or not _is_accel_tensor(B):
        raise ValueError("B must be a 2D accelerator tensor")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() != indices.numel() or indptr.numel() != n_rows + 1:
        raise ValueError("invalid CSR dimensions")
    if B.shape[0] != n_cols:
        raise ValueError(f"B has {B.shape[0]} rows, expected {n_cols}")
    result_dtype, acc_dtype = _resolve_types(data.dtype, B.dtype, out, out_dtype, False)
    n_dense = int(B.shape[1])
    if out is not None and (tuple(out.shape) != (n_rows, n_dense) or out.device != B.device):
        raise ValueError("out shape/device must match the CSR SpMM result")
    data = data.contiguous()
    cols, ptr = _index32(indices.contiguous()), _index32(indptr.contiguous())
    t0 = _start(return_time)
    if data.dtype in (torch.complex64, torch.complex128):
        C = out if out is not None else torch.empty(
            n_rows, n_dense, dtype=result_dtype, device=B.device
        )
        if n_rows and n_dense:
            block_n = min(128, max(16, triton.next_power_of_2(n_dense)))
            if _is_iluvatar_runtime():
                block_override = os.environ.get("FLAGSPARSE_ILUVATAR_COMPLEX_BLOCK_N")
                rows_override = os.environ.get("FLAGSPARSE_ILUVATAR_COMPLEX_ROWS")
                warps_override = os.environ.get("FLAGSPARSE_ILUVATAR_COMPLEX_WARPS")
                stages_override = os.environ.get("FLAGSPARSE_ILUVATAR_COMPLEX_STAGES")
                biv_warps_override = os.environ.get(
                    "FLAGSPARSE_ILUVATAR_COMPLEX_BIV_WARPS"
                )
                biv_stages_override = os.environ.get(
                    "FLAGSPARSE_ILUVATAR_COMPLEX_BIV_STAGES"
                )
                if block_override:
                    block_n = min(128, max(16, int(block_override)))
                rows_per = int(rows_override) if rows_override else 2
                complex_warps = int(warps_override) if warps_override else 1
                complex_stages = int(stages_override) if stages_override else 3
                use_biv_rows = (
                    data.dtype == torch.complex64
                    and n_dense == 32
                    and not any(
                        (block_override, rows_override, warps_override, stages_override)
                    )
                )
            else:
                rows_per = 1
                complex_warps = 1
                complex_stages = 3
                use_biv_rows = False
            data_ri = torch.view_as_real(data).contiguous().reshape(-1)
            B_ri = torch.view_as_real(B)
            C_ri = torch.view_as_real(C)
            if use_biv_rows:
                from .spmm_coo import _spmm_coo_biv_complex_rows_kernel

                mean_row_nnz = data.numel() / max(1, n_rows)
                if mean_row_nnz <= 3.5:
                    biv_rows, biv_lanes = 4, 1
                else:
                    biv_rows, biv_lanes = 1, 2
                _spmm_coo_biv_complex_rows_kernel[
                    (triton.cdiv(n_rows, biv_rows),)
                ](
                    data_ri.reshape(-1, 2),
                    cols,
                    ptr,
                    B_ri,
                    C_ri,
                    n_rows,
                    n_dense,
                    *B.stride(),
                    *C.stride(),
                    ROWS=biv_rows,
                    LANES=biv_lanes,
                    num_warps=int(biv_warps_override) if biv_warps_override else 1,
                    num_stages=(
                        int(biv_stages_override) if biv_stages_override else 3
                    ),
                )
            else:
                _csr_spmm_complex_rows_kernel[
                    (triton.cdiv(n_rows, rows_per), triton.cdiv(n_dense, block_n))
                ](
                    C_ri,
                    data_ri,
                    cols,
                    ptr,
                    B_ri,
                    n_rows,
                    n_dense,
                    B_ri.stride(0),
                    B_ri.stride(1),
                    B_ri.stride(2),
                    C_ri.stride(0),
                    C_ri.stride(1),
                    C_ri.stride(2),
                    ACC=tl.float64 if data.dtype == torch.complex128 else tl.float32,
                    ROWS=rows_per,
                    BLOCK_N=block_n,
                    num_warps=complex_warps,
                    num_stages=complex_stages,
                )
        return _finish(C, out, t0, return_time)
    if _is_ascend_runtime():
        rows = _csr_row_ids(ptr, n_rows, data.numel())
        C = torch.zeros(n_rows, n_dense, dtype=acc_dtype, device=B.device)
        C.index_add_(0, rows, data.to(acc_dtype).unsqueeze(1) * B.index_select(0, cols.to(torch.int64)).to(acc_dtype))
        return _finish(C.to(result_dtype), out, t0, return_time)
    C = out if out is not None else torch.empty(n_rows, n_dense, dtype=result_dtype, device=B.device)
    if n_rows and n_dense:
        if _is_mthreads_runtime():
            from . import _spmm_row_gather
            _spmm_row_gather.compute(data, cols, ptr, B, C)
            if return_time:
                _ACCEL.synchronize()
                return C, (time.perf_counter() - t0) * 1000.0
            return C
        block_n = min(128, max(16, triton.next_power_of_2(n_dense)))
        if _is_iluvatar_runtime():
            if data.dtype == torch.float16:
                rows_per, mixed_warps = max(1, 256 // block_n), 1
            elif data.dtype == torch.float32:
                rows_per, mixed_warps = max(1, 128 // block_n), 1
            else:
                rows_per, mixed_warps = max(1, 128 // block_n), 1
            _csr_spmm_mixed_rows_kernel[(triton.cdiv(n_rows, rows_per), triton.cdiv(n_dense, block_n))](
                C, data, cols, ptr, B, n_rows, n_dense,
                B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                ACC=_TL_ACC[acc_dtype], ROWS=rows_per, BLOCK_N=block_n,
                num_warps=mixed_warps,
            )
        else:
            _csr_spmm_mixed_kernel[(n_rows, triton.cdiv(n_dense, block_n))](
                C, data, cols, ptr, B, n_dense,
                B.stride(0), B.stride(1), C.stride(0), C.stride(1),
                ACC=_TL_ACC[acc_dtype], BLOCK_N=block_n, num_warps=2,
            )
    if return_time:
        _ACCEL.synchronize()
        return C, (time.perf_counter() - t0) * 1000.0
    return C


# ---------------------------------------------------------------------------
# COO SpMM (C = A @ B, B/C row- or column-major through their strides)
# ---------------------------------------------------------------------------


@triton.jit
def _coo_spmm_mixed_batched_kernel(
    c_ptr, data_ptr, rows_ptr, cols_ptr, b_ptr, nnz, n_dense,
    stride_bk, stride_bn, stride_cm, stride_cn,
    ACC: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = p < nnz
    row = tl.load(rows_ptr + p, valid, other=0)
    col = tl.load(cols_ptr + p, valid, other=0)
    a = tl.load(data_ptr + p, valid, other=0).to(ACC)
    mask = valid[:, None] & (n[None, :] < n_dense)
    b = tl.load(b_ptr + col[:, None] * stride_bk + n[None, :] * stride_bn,
                mask, other=0).to(ACC)
    tl.atomic_add(c_ptr + row[:, None] * stride_cm + n[None, :] * stride_cn,
                  a[:, None] * b, mask=mask)


@triton.jit
def _coo_spmm_mixed_kernel(
    c_ptr, data_ptr, rows_ptr, cols_ptr, b_ptr, nnz, n_dense,
    stride_bk, stride_bn, stride_cm, stride_cn,
    ACC: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """One program per (nonzero, dense-column tile); every nonzero's contribution
    to its output row is atomic_add'd independently, same shape as
    _coo_spmv_atomic_kernel above with a dense-N axis added (mirroring
    _csr_spmm_mixed_kernel's per-row accumulator, but COO has no contiguous
    per-row range to loop over without a CSR-style indptr)."""
    idx = tl.program_id(0)
    if idx >= nnz:
        return
    n_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < n_dense
    row = tl.load(rows_ptr + idx)
    col = tl.load(cols_ptr + idx)
    a = tl.load(data_ptr + idx).to(ACC)
    b = tl.load(b_ptr + col * stride_bk + n_offs * stride_bn, mask=n_mask, other=0).to(ACC)
    tl.atomic_add(c_ptr + row * stride_cm + n_offs * stride_cn, a * b, mask=n_mask)


def spmm_coo_mixed(data, row, col, B, shape, *, op=None, transpose=None,
                   out=None, out_dtype=None, return_time=False):
    """``B`` may be any 2D strided view (column-major, or a transposed ``op_b``)."""
    _normalize_non_op(op, transpose)
    _check_1d((data, "data"), (row, "row"), (col, "col"))
    if not torch.is_tensor(B) or B.ndim != 2 or not _is_accel_tensor(B):
        raise ValueError("B must be a 2D accelerator tensor")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if not (data.numel() == row.numel() == col.numel()):
        raise ValueError("data, row and col must have the same length")
    if B.shape[0] != n_cols:
        raise ValueError(f"B has {B.shape[0]} rows, expected {n_cols}")
    result_dtype, acc_dtype = _resolve_types(data.dtype, B.dtype, out, out_dtype, False)
    n_dense = int(B.shape[1])
    if out is not None and (tuple(out.shape) != (n_rows, n_dense) or out.device != B.device):
        raise ValueError("out shape/device must match the COO SpMM result")
    if _is_mthreads_runtime():
        values, cols, ptr = _musa_coo_csr(data, row, col, shape)
        return spmm_csr_mixed(
            values, cols, ptr, B, shape, out=out, out_dtype=result_dtype,
            return_time=return_time,
        )
    data = data.contiguous()
    rows, cols = _index32(row.contiguous()), _index32(col.contiguous())
    t0 = _start(return_time)
    if _is_ascend_runtime():
        C = torch.zeros(n_rows, n_dense, dtype=acc_dtype, device=B.device)
        C.index_add_(
            0, rows.to(torch.int64),
            data.to(acc_dtype).unsqueeze(1) * B.index_select(0, cols.to(torch.int64)).to(acc_dtype),
        )
        return _finish(C.to(result_dtype), out, t0, return_time)
    if _is_iluvatar_runtime():
        sources = (data, row, col)
        options = (n_rows, n_cols)
        state = _ILUVATAR_COO_SPMM_CSR_CACHE.get(sources, options)
        if state is None:
            ordered = rows.numel() < 2 or bool(
                torch.all(rows[1:] >= rows[:-1]).item()
            )
            if ordered:
                values, csr_cols = data, cols
            else:
                order = torch.argsort(rows.to(torch.int64), stable=True)
                values = data.index_select(0, order).contiguous()
                csr_cols = cols.index_select(0, order).contiguous()
                rows = rows.index_select(0, order).contiguous()
            counts = torch.bincount(rows.to(torch.int64), minlength=n_rows)
            ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            ptr = ptr.to(torch.int32).contiguous()
            state = (values, csr_cols, ptr)
            _ILUVATAR_COO_SPMM_CSR_CACHE.put(sources, options, state)
        values, csr_cols, ptr = state
        return spmm_csr_mixed(
            values, csr_cols, ptr, B, shape, out=out, out_dtype=result_dtype,
            return_time=return_time,
        )
    acc = torch.zeros(n_rows, n_dense, dtype=acc_dtype, device=B.device)
    nnz = data.numel()
    if nnz and n_dense:
        block_n = min(128, max(16, triton.next_power_of_2(n_dense)))
        if _backend_name() == "cuda" or _is_rocm_runtime() or _is_iluvatar_runtime():
            # Group adjacent COO entries for CUDA, ROCm, and Iluvatar; other
            # backends retain the scalar path.
            _coo_spmm_mixed_batched_kernel[(triton.cdiv(nnz, 16), triton.cdiv(n_dense, block_n))](
                acc, data, rows, cols, B, nnz, n_dense,
                B.stride(0), B.stride(1), acc.stride(0), acc.stride(1),
                ACC=_TL_ACC[acc_dtype], BLOCK_N=block_n, BLOCK_P=16, num_warps=4,
            )
        else:
            _coo_spmm_mixed_kernel[(nnz, triton.cdiv(n_dense, block_n))](
                acc, data, rows, cols, B, nnz, n_dense,
                B.stride(0), B.stride(1), acc.stride(0), acc.stride(1),
                ACC=_TL_ACC[acc_dtype], BLOCK_N=block_n, num_warps=2,
            )
    y = acc if acc_dtype == result_dtype else acc.to(result_dtype)
    return _finish(y, out, t0, return_time)
