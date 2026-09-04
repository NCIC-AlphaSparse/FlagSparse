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

"""CSR SpMV: Triton baseline kernels + optimised CSR-Vector buckets."""

from ._common import *

import os
import time
import triton
import triton.language as tl

SUPPORTED_SPMV_VALUE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

SPMV_OP_NON = 0
SPMV_OP_TRANS = 1
SPMV_OP_CONJ_TRANS = 2
SPMV_OP_NAMES = {
    SPMV_OP_NON: "non",
    SPMV_OP_TRANS: "trans",
    SPMV_OP_CONJ_TRANS: "conj",
}
_SPMV_OP_NAME_TO_CODE = {name: code for code, name in SPMV_OP_NAMES.items()}


def _normalize_spmv_op(op=None, transpose=False):
    if op is None:
        return SPMV_OP_TRANS if bool(transpose) else SPMV_OP_NON
    if isinstance(op, str):
        token = op.strip().lower()
        if token not in _SPMV_OP_NAME_TO_CODE:
            raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
        return _SPMV_OP_NAME_TO_CODE[token]
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj") from exc
    if op_code not in SPMV_OP_NAMES:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
    return op_code


def _spmv_op_to_name(op):
    op_code = _normalize_spmv_op(op)
    return SPMV_OP_NAMES[op_code]


def _spmv_op_transposes(op):
    return _normalize_spmv_op(op) in (SPMV_OP_TRANS, SPMV_OP_CONJ_TRANS)


class PreparedCsrSpmv:
    """Cached CSR metadata for repeated SpMV calls on the same sparse matrix."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "block_nnz",
        "max_segments",
        "opt_max_segments",
        "row_lengths",
        "max_row_nnz",
        "opt_buckets",
        "supports_opt",
        "transpose",
        "op",
        "index_fallback_policy",
        "index_fallback_applied",
        "index_fallback_reason",
        "_baseline_compute_dtype",
        "_baseline_data",
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
        max_row_nnz,
        opt_buckets=None,
        opt_max_segments=None,
        row_lengths=None,
        transpose=False,
        op=None,
        index_fallback_policy="auto",
        index_fallback_applied=False,
        index_fallback_reason=None,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.block_nnz = block_nnz
        self.max_segments = max_segments
        self.opt_max_segments = opt_max_segments
        if row_lengths is None:
            row_lengths = kernel_indptr[1:] - kernel_indptr[:-1]
        self.row_lengths = row_lengths
        self.max_row_nnz = max_row_nnz
        self.opt_buckets = [] if opt_buckets is None else opt_buckets
        self.supports_opt = (
            data.dtype in (torch.float32, torch.float64)
            and kernel_indices.dtype == torch.int32
        )
        self.op = _normalize_spmv_op(op, transpose=transpose)
        self.transpose = _spmv_op_transposes(self.op)
        self.index_fallback_policy = str(index_fallback_policy).lower()
        self.index_fallback_applied = bool(index_fallback_applied)
        self.index_fallback_reason = index_fallback_reason
        if data.dtype in (torch.float16, torch.bfloat16):
            self._baseline_compute_dtype = torch.float32
        elif data.dtype == torch.float32:
            self._baseline_compute_dtype = torch.float64
        else:
            self._baseline_compute_dtype = data.dtype
        self._baseline_data = None


# Performance-first CSR-Vector buckets.  num_warps*32 >= block_size.
# First bucket uses batch_rows>1: one program processes several short rows
# (fewer blocks → better occupancy on graphs with millions of low-degree rows).
_SPMV_OPT_BUCKET_CONFIGS = (
    {
        "max_row_nnz": 64,
        "block_size": 32,
        "num_warps": 1,
        "num_stages": 2,
        "batch_rows": 16,
    },
    {"max_row_nnz": 512, "block_size": 256, "num_warps": 8, "num_stages": 2},
    {"max_row_nnz": 4096, "block_size": 512, "num_warps": 16, "num_stages": 2},
    {"max_row_nnz": None, "block_size": 1024, "num_warps": 32, "num_stages": 3},
)
# fp64: extra row-length tiers + smaller tiles vs f32; batch_rows=4 for short-row kernel.
_SPMV_OPT_BUCKET_CONFIGS_FP64 = (
    {
        "max_row_nnz": 64,
        "block_size": 32,
        "num_warps": 1,
        "num_stages": 2,
        "batch_rows": 4,
    },
    {"max_row_nnz": 256, "block_size": 64, "num_warps": 2, "num_stages": 2},
    {"max_row_nnz": 2048, "block_size": 128, "num_warps": 4, "num_stages": 2},
    {"max_row_nnz": 8192, "block_size": 256, "num_warps": 8, "num_stages": 2},
    {"max_row_nnz": None, "block_size": 512, "num_warps": 16, "num_stages": 1},
)
# DCU/ROCm bucket tiers: smaller tiles and a num_warps ceiling of 8.
_SPMV_OPT_BUCKET_CONFIGS_HIP = (
    {
        "max_row_nnz": 64,
        "block_size": 32,
        "num_warps": 1,
        "num_stages": 2,
        "batch_rows": 16,
    },
    {"max_row_nnz": 512, "block_size": 256, "num_warps": 8, "num_stages": 2},
    {"max_row_nnz": 4096, "block_size": 512, "num_warps": 8, "num_stages": 1},
    {"max_row_nnz": None, "block_size": 512, "num_warps": 8, "num_stages": 1},
)
_SPMV_OPT_BUCKET_CONFIGS_HIP_FP64 = (
    {
        "max_row_nnz": 64,
        "block_size": 32,
        "num_warps": 1,
        "num_stages": 2,
        "batch_rows": 4,
    },
    {"max_row_nnz": 256, "block_size": 64, "num_warps": 2, "num_stages": 2},
    {"max_row_nnz": 2048, "block_size": 256, "num_warps": 4, "num_stages": 2},
    {"max_row_nnz": 8192, "block_size": 512, "num_warps": 8, "num_stages": 1},
    {"max_row_nnz": None, "block_size": 512, "num_warps": 8, "num_stages": 1},
)


_SPMV_OPT_ACC_MODES = ("fast", "mixed", "accurate")


@triton.jit
def _spmv_seg_add(row_a, val_a, row_b, val_b):
    """Associative combine for a segmented (per-row) inclusive sum."""
    return row_b, val_b + tl.where(row_a == row_b, val_a, val_b - val_b)


def _normalize_spmv_opt_device_props(device):
    props = _ACCEL.get_device_properties(device)
    device_name = str(getattr(props, "name", "cuda"))
    name_lower = device_name.lower()
    # Package-wide probe first (so FLAGSPARSE_BACKEND reaches here), then the
    # device-name fallback for ROCm builds that do not set torch.version.hip.
    is_hip = _is_rocm_runtime() or any(
        token in name_lower for token in ("dcu", "hygon", "rocm", "amd")
    )
    warp_size = int(getattr(props, "warp_size", 64 if is_hip else 32) or 32)
    max_threads_per_block = int(getattr(props, "max_threads_per_block", 1024) or 1024)
    max_threads_per_mp = int(
        getattr(props, "max_threads_per_multi_processor", max_threads_per_block)
        or max_threads_per_block
    )
    return {
        # Only "hip" changes bucket tiers today; the other names are reported
        # faithfully so a future backend can branch without another rename.
        "backend": (
            "hip"
            if is_hip
            else (
                "metax"
                if _is_maca_runtime()
                else (
                    "mthreads"
                    if _is_mthreads_runtime()
                    else ("ascend" if _is_ascend_runtime() else "cuda")
                )
            )
        ),
        "device_name": device_name,
        "warp_size": max(1, warp_size),
        "max_threads_per_block": max(32, max_threads_per_block),
        "max_threads_per_mp": max(32, max_threads_per_mp),
    }


def _spmv_opt_bucket_configs(dtype, device_props):
    if device_props["backend"] == "hip":
        return (
            _SPMV_OPT_BUCKET_CONFIGS_HIP_FP64
            if dtype == torch.float64
            else _SPMV_OPT_BUCKET_CONFIGS_HIP
        )
    return (
        _SPMV_OPT_BUCKET_CONFIGS_FP64
        if dtype == torch.float64
        else _SPMV_OPT_BUCKET_CONFIGS
    )


def _clip_spmv_opt_launch_spec(spec, device_props):
    spec = dict(spec)
    warp_size = max(1, int(device_props["warp_size"]))
    max_warps_by_block = max(1, int(device_props["max_threads_per_block"]) // warp_size)
    max_warps_by_mp = max(1, int(device_props["max_threads_per_mp"]) // warp_size)
    max_supported = min(max_warps_by_block, max_warps_by_mp)
    if device_props["backend"] == "hip":
        max_supported = min(max_supported, 8)
        spec["block_size"] = min(int(spec["block_size"]), 512)
    spec["num_warps"] = max(1, min(int(spec["num_warps"]), max_supported))
    spec["num_stages"] = max(1, int(spec["num_stages"]))
    return spec


# ── DCU/ROCm row-parallel SpMV ──────────────────────────────────────
# One program per row with an in-row segment loop. This is the DCU branch's
# kernel; CUDA keeps the nnz-partitioned segbin path below. Selected by
# _spmv_csr_default_backend().

@triton.jit
def _spmv_csr_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    acc = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end
        a = tl.load(data_ptr + offsets, mask=mask, other=0.0)
        col = tl.load(indices_ptr + offsets, mask=mask, other=0)
        x_vals = tl.load(x_ptr + col, mask=mask, other=0.0)
        part = tl.where(mask, a * x_vals, 0.0)
        acc = acc + tl.sum(part)
    tl.store(y_ptr + row, acc)


@triton.jit
def _spmv_csr_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    x_ri_ptr,
    y_ri_ptr,
    n_rows,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    acc_re = tl.load(data_ri_ptr + start * 2, mask=start < end, other=0.0) * 0
    acc_im = tl.load(data_ri_ptr + start * 2 + 1, mask=start < end, other=0.0) * 0
    for seg in range(MAX_SEGMENTS):
        idx = start + seg * BLOCK_NNZ
        offsets = idx + tl.arange(0, BLOCK_NNZ)
        mask = offsets < end
        a_re = tl.load(data_ri_ptr + offsets * 2, mask=mask, other=0.0)
        a_im = tl.load(data_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
        col = tl.load(indices_ptr + offsets, mask=mask, other=0)
        x_re = tl.load(x_ri_ptr + col * 2, mask=mask, other=0.0)
        x_im = tl.load(x_ri_ptr + col * 2 + 1, mask=mask, other=0.0)
        prod_re = tl.where(mask, a_re * x_re - a_im * x_im, 0.0)
        prod_im = tl.where(mask, a_re * x_im + a_im * x_re, 0.0)
        acc_re = acc_re + tl.sum(prod_re)
        acc_im = acc_im + tl.sum(prod_im)
    tl.store(y_ri_ptr + row * 2, acc_re)
    tl.store(y_ri_ptr + row * 2 + 1, acc_im)


# ── Optimised SpMV (CSR-Vector, perf-oriented, no CuPy) ─────────────


@triton.jit
def _spmv_csr_segbin_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    nnz,
    n_rows,
    STEPS: tl.constexpr,
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Preprocessing-free, load-balanced CSR SpMV.

    Each program owns a fixed BLOCK-sized run of nonzeros (balanced regardless of
    the row-length distribution). The row of each nonzero is found in-kernel by an
    upper-bound binary search on indptr (no per-nonzero row-id array). A segmented
    inclusive scan sums products belonging to the same row within the tile, so
    each row-run contributes with a single atomic add — bounding atomic contention
    even for very dense rows. y must be pre-zeroed and typed as the accumulator."""
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    offs = pid * BLOCK + lane
    mask = offs < nnz
    # row = max r such that indptr[r] <= offs (upper-bound binary search).
    lo = tl.zeros((BLOCK,), dtype=tl.int32)
    hi = tl.full((BLOCK,), n_rows, dtype=tl.int32)
    for _ in tl.static_range(STEPS):
        mid = (lo + hi + 1) // 2
        v = tl.load(indptr_ptr + mid, mask=mask, other=0)
        take = v <= offs
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)
    row = lo
    a = tl.load(data_ptr + offs, mask=mask, other=0.0)
    col = tl.load(indices_ptr + offs, mask=mask, other=0)
    xv = tl.load(x_ptr + col, mask=mask, other=0.0)
    prod = a.to(ACC) * xv.to(ACC)
    _, seg = tl.associative_scan((row, prod), axis=0, combine_fn=_spmv_seg_add)
    # Flush a row-run's partial sum at its last nonzero within this tile (either
    # the row genuinely ends here, or the tile ends; a row spanning tiles is
    # summed across tiles by the atomics).
    row_end = tl.load(indptr_ptr + row + 1, mask=mask, other=0) - 1
    is_bnd = mask & ((offs == row_end) | (lane == BLOCK - 1))
    tl.atomic_add(y_ptr + row, tl.where(is_bnd, seg, seg - seg), mask=is_bnd)


@triton.jit
def _spmv_seg_add_complex(row_a, re_a, im_a, row_b, re_b, im_b):
    """Associative combine for a segmented (per-row) inclusive complex sum."""
    same = row_a == row_b
    zero = re_b - re_b
    return row_b, re_b + tl.where(same, re_a, zero), im_b + tl.where(same, im_a, zero)


@triton.jit
def _spmv_csr_complex_segbin_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    x_ri_ptr,
    y_ri_ptr,
    nnz,
    n_rows,
    STEPS: tl.constexpr,
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Complex counterpart of _spmv_csr_segbin_kernel. Values are stored as
    interleaved real/imag pairs; each nonzero's row is found by binary search and
    a segmented inclusive scan over (row, re, im) bounds atomic contention on
    dense rows. y (interleaved) must be pre-zeroed and typed as the accumulator."""
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    offs = pid * BLOCK + lane
    mask = offs < nnz
    lo = tl.zeros((BLOCK,), dtype=tl.int32)
    hi = tl.full((BLOCK,), n_rows, dtype=tl.int32)
    for _ in tl.static_range(STEPS):
        mid = (lo + hi + 1) // 2
        v = tl.load(indptr_ptr + mid, mask=mask, other=0)
        take = v <= offs
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)
    row = lo
    a_re = tl.load(data_ri_ptr + offs * 2, mask=mask, other=0.0).to(ACC)
    a_im = tl.load(data_ri_ptr + offs * 2 + 1, mask=mask, other=0.0).to(ACC)
    col = tl.load(indices_ptr + offs, mask=mask, other=0)
    x_re = tl.load(x_ri_ptr + col * 2, mask=mask, other=0.0).to(ACC)
    x_im = tl.load(x_ri_ptr + col * 2 + 1, mask=mask, other=0.0).to(ACC)
    p_re = a_re * x_re - a_im * x_im
    p_im = a_re * x_im + a_im * x_re
    _, s_re, s_im = tl.associative_scan(
        (row, p_re, p_im), axis=0, combine_fn=_spmv_seg_add_complex
    )
    row_end = tl.load(indptr_ptr + row + 1, mask=mask, other=0) - 1
    is_bnd = mask & ((offs == row_end) | (lane == BLOCK - 1))
    zero = s_re - s_re
    tl.atomic_add(y_ri_ptr + row * 2, tl.where(is_bnd, s_re, zero), mask=is_bnd)
    tl.atomic_add(y_ri_ptr + row * 2 + 1, tl.where(is_bnd, s_im, zero), mask=is_bnd)


# ── Optimised SpMV (CSR-Vector, perf-oriented, no CuPy) ─────────────
# fp32 / fp64 native lane accum.  Batched kernel for many short rows per program.


@triton.jit
def _spmv_csr_batched_short_f32(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    rows_ptr,
    n_bucket_rows,
    BATCH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEGS: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)
    for b in range(BATCH):
        ridx = pid * BATCH + b
        active = ridx < n_bucket_rows
        row = tl.load(rows_ptr + ridx, mask=active, other=0)
        start = tl.load(indptr_ptr + row, mask=active, other=0)
        end = tl.load(indptr_ptr + row + 1, mask=active, other=0)
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for seg in range(MAX_SEGS):
            offs = start + seg * BLOCK_SIZE + lane
            mask = offs < end
            a = tl.load(data_ptr + offs, mask=mask, other=0.0)
            col = tl.load(indices_ptr + offs, mask=mask, other=0)
            xv = tl.load(x_ptr + col, mask=mask, other=0.0)
            acc += tl.where(mask, a * xv, 0.0)
        tl.store(y_ptr + row, tl.sum(acc), mask=active)


@triton.jit
def _spmv_csr_batched_short_f64(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    rows_ptr,
    n_bucket_rows,
    BATCH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEGS: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)
    for b in range(BATCH):
        ridx = pid * BATCH + b
        active = ridx < n_bucket_rows
        row = tl.load(rows_ptr + ridx, mask=active, other=0)
        start = tl.load(indptr_ptr + row, mask=active, other=0)
        end = tl.load(indptr_ptr + row + 1, mask=active, other=0)
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
        for seg in range(MAX_SEGS):
            offs = start + seg * BLOCK_SIZE + lane
            mask = offs < end
            a = tl.load(data_ptr + offs, mask=mask, other=0.0)
            col = tl.load(indices_ptr + offs, mask=mask, other=0)
            xv = tl.load(x_ptr + col, mask=mask, other=0.0)
            acc += tl.where(mask, a * xv, 0.0)
        tl.store(y_ptr + row, tl.sum(acc), mask=active)


@triton.jit
def _spmv_csr_vector_rows_f32(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    rows_ptr,
    n_bucket_rows,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEGS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    lane = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for seg in range(MAX_SEGS):
        offs = start + seg * BLOCK_SIZE + lane
        mask = offs < end
        a = tl.load(data_ptr + offs, mask=mask, other=0.0)
        col = tl.load(indices_ptr + offs, mask=mask, other=0)
        xv = tl.load(x_ptr + col, mask=mask, other=0.0)
        acc = tl.where(mask, acc + a * xv, acc)
    tl.store(y_ptr + row, tl.sum(acc))


@triton.jit
def _spmv_csr_vector_rows_f64(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    rows_ptr,
    n_bucket_rows,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEGS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    lane = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float64)
    for seg in range(MAX_SEGS):
        offs = start + seg * BLOCK_SIZE + lane
        mask = offs < end
        a = tl.load(data_ptr + offs, mask=mask, other=0.0)
        col = tl.load(indices_ptr + offs, mask=mask, other=0)
        xv = tl.load(x_ptr + col, mask=mask, other=0.0)
        acc = tl.where(mask, acc + a * xv, acc)
    tl.store(y_ptr + row, tl.sum(acc))


def _build_spmv_opt_buckets(
    row_lengths,
    max_row_nnz,
    row_index_dtype,
    max_segments=None,
    fp64=False,
    device_props=None,
):
    buckets = []
    lower_bound = 0
    # Bucket tiers and launch shape follow the runtime: HIP gets smaller tiles and
    # a num_warps ceiling of 8. Falls back to the CUDA tiers when device_props is
    # not supplied.
    if device_props is None:
        configs = _SPMV_OPT_BUCKET_CONFIGS_FP64 if fp64 else _SPMV_OPT_BUCKET_CONFIGS
    else:
        configs = _spmv_opt_bucket_configs(
            torch.float64 if fp64 else torch.float32, device_props
        )
    for spec in configs:
        if device_props is not None:
            spec = _clip_spmv_opt_launch_spec(spec, device_props)
        upper_bound = spec["max_row_nnz"]
        if upper_bound is None:
            mask = row_lengths > lower_bound
            bucket_max_row_nnz = max_row_nnz
        elif lower_bound == 0:
            # Include nnz==0 rows in the first bucket (they still need y[i]=0).
            mask = row_lengths <= upper_bound
            bucket_max_row_nnz = upper_bound
        else:
            mask = (row_lengths > lower_bound) & (row_lengths <= upper_bound)
            bucket_max_row_nnz = upper_bound
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        if rows.numel() == 0:
            if upper_bound is not None:
                lower_bound = upper_bound
            continue
        if max_segments is None:
            max_segs = max(
                (bucket_max_row_nnz + spec["block_size"] - 1) // spec["block_size"],
                1,
            )
        else:
            max_segs = max_segments
        buckets.append(
            {
                "rows": rows.to(row_index_dtype),
                "block_size": spec["block_size"],
                "max_segs": max_segs,
                "num_warps": spec["num_warps"],
                "num_stages": spec["num_stages"],
                "batch_rows": int(spec.get("batch_rows", 1)),
            }
        )
        if upper_bound is not None:
            lower_bound = upper_bound
    return buckets


def _build_spmv_opt_runtime_buckets(prepared):
    row_index_dtype = (
        torch.int32 if prepared.n_rows <= _INDEX_LIMIT_INT32 else torch.int64
    )
    return _build_spmv_opt_buckets(
        prepared.row_lengths,
        max_row_nnz=prepared.max_row_nnz,
        row_index_dtype=row_index_dtype,
        max_segments=prepared.opt_max_segments,
        fp64=prepared.data.dtype == torch.float64,
        device_props=_normalize_spmv_opt_device_props(prepared.data.device),
    )


def _triton_spmv_csr_impl_opt_prepared(prepared, x, opt_buckets=None):
    # First bucket includes nnz==0 rows; every row gets exactly one store.
    dtype = prepared.data.dtype
    y = torch.empty(prepared.n_rows, dtype=dtype, device=prepared.data.device)
    if prepared.n_rows == 0:
        return y
    if opt_buckets is None:
        opt_buckets = prepared.opt_buckets
    vec_f32 = _spmv_csr_vector_rows_f32
    vec_f64 = _spmv_csr_vector_rows_f64
    bat_f32 = _spmv_csr_batched_short_f32
    bat_f64 = _spmv_csr_batched_short_f64
    for bucket in opt_buckets:
        rows = bucket["rows"]
        br = max(1, int(bucket.get("batch_rows", 1)))
        n_r = rows.numel()
        if br > 1:
            kernel = bat_f64 if dtype == torch.float64 else bat_f32
            grid = (triton.cdiv(n_r, br),)
            kernel[grid](
                prepared.data,
                prepared.kernel_indices,
                prepared.kernel_indptr,
                x,
                y,
                rows,
                n_bucket_rows=n_r,
                BATCH=br,
                BLOCK_SIZE=bucket["block_size"],
                MAX_SEGS=bucket["max_segs"],
                num_warps=bucket["num_warps"],
                num_stages=bucket["num_stages"],
            )
        else:
            kernel = vec_f64 if dtype == torch.float64 else vec_f32
            grid = (n_r,)
            kernel[grid](
                prepared.data,
                prepared.kernel_indices,
                prepared.kernel_indptr,
                x,
                y,
                rows,
                n_bucket_rows=n_r,
                BLOCK_SIZE=bucket["block_size"],
                MAX_SEGS=bucket["max_segs"],
                num_warps=bucket["num_warps"],
                num_stages=bucket["num_stages"],
            )
    return y


def _normalize_spmv_index_fallback_policy(index_fallback_policy):
    policy = str(index_fallback_policy).lower()
    if policy not in ("auto", "strict"):
        raise ValueError("index_fallback_policy must be 'auto' or 'strict'")
    return policy


def _spmv_dtype_error_message():
    return "data dtype must be one of: " + ", ".join(
        str(dtype).replace("torch.", "") for dtype in SUPPORTED_SPMV_VALUE_DTYPES
    )


def _transpose_csr_for_spmv(data, indices, indptr, shape):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    nnz = data.numel()
    device = data.device
    if nnz == 0:
        out_index_dtype = indices.dtype if n_rows <= _INDEX_LIMIT_INT32 else torch.int64
        out_indptr_dtype = indptr.dtype if nnz <= _INDEX_LIMIT_INT32 else torch.int64
        return (
            data,
            torch.empty(0, dtype=out_index_dtype, device=device),
            torch.zeros(n_cols + 1, dtype=out_indptr_dtype, device=device),
            (n_cols, n_rows),
        )

    row_counts = indptr[1:] - indptr[:-1]
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, dtype=torch.int64, device=device),
        row_counts.to(torch.int64),
    )
    col_ids = indices.to(torch.int64)
    try:
        order = torch.argsort(col_ids, stable=True)
    except TypeError:
        order = torch.argsort(col_ids)
    sorted_cols = col_ids[order]
    sorted_rows = row_ids[order]
    transposed_data = data[order].contiguous()

    nnz_per_transposed_row = torch.bincount(sorted_cols, minlength=n_cols)
    transposed_indptr64 = torch.zeros(n_cols + 1, dtype=torch.int64, device=device)
    transposed_indptr64[1:] = torch.cumsum(nnz_per_transposed_row, dim=0)
    out_index_dtype = indices.dtype if n_rows <= _INDEX_LIMIT_INT32 else torch.int64
    out_indptr_dtype = indptr.dtype if nnz <= _INDEX_LIMIT_INT32 else torch.int64
    return (
        transposed_data,
        sorted_rows.to(out_index_dtype).contiguous(),
        transposed_indptr64.to(out_indptr_dtype).contiguous(),
        (n_cols, n_rows),
    )


def _prepare_spmv_csr_matrix(
    data, indices, indptr, shape, index_fallback_policy="auto"
):
    _normalize_spmv_index_fallback_policy(index_fallback_policy)
    if not all(torch.is_tensor(t) for t in (data, indices, indptr)):
        raise TypeError("data, indices, indptr must all be torch.Tensor")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_rows + 1:
        raise ValueError(
            f"indptr length must be n_rows+1={n_rows + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if not all(_is_accel_tensor(t) for t in (data, indices, indptr)):
        raise ValueError("data, indices, indptr must be CUDA tensors")
    if not all(t.device == data.device for t in (indices, indptr)):
        raise ValueError("data, indices, indptr must be on the same CUDA device")
    if data.dtype not in SUPPORTED_SPMV_VALUE_DTYPES:
        raise TypeError(_spmv_dtype_error_message())
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

    nnz = data.numel()
    if nnz > 0:
        min_index = int(indices.min().item())
        max_index = int(indices.max().item())
        if min_index < 0 or max_index >= n_cols:
            raise IndexError("indices out of range for n_cols")
    kernel_indices = indices
    kernel_indptr = indptr
    row_lengths = kernel_indptr[1:] - kernel_indptr[:-1]
    max_row_nnz = int(row_lengths.max().item()) if n_rows > 0 else 0
    return (
        data,
        kernel_indices,
        kernel_indptr,
        n_rows,
        n_cols,
        row_lengths,
        max_row_nnz,
    )


def _validate_spmv_x(x, prepared):
    if x is None or not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if not _is_accel_tensor(x):
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != prepared.data.dtype:
        raise TypeError("x dtype must match sparse matrix dtype")
    if x.numel() != prepared.n_cols:
        raise ValueError(f"x length must be n_cols={prepared.n_cols}, got {x.numel()}")
    if x.device != prepared.data.device:
        raise ValueError("x must be on the same device as sparse matrix data")
    return x.contiguous()


def prepare_spmv_csr(
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
    index_fallback_policy = _normalize_spmv_index_fallback_policy(index_fallback_policy)
    op_code = _normalize_spmv_op(op, transpose=transpose)
    if op is not None and bool(transpose) and op_code == SPMV_OP_NON:
        raise ValueError("transpose=True conflicts with op=non")
    transpose = _spmv_op_transposes(op_code)
    if transpose:
        data, indices, indptr, *_ = _prepare_spmv_csr_matrix(
            data,
            indices,
            indptr,
            shape,
            index_fallback_policy=index_fallback_policy,
        )
        if op_code == SPMV_OP_CONJ_TRANS and _is_complex_dtype(data.dtype):
            data = data.conj()
            if hasattr(data, "resolve_conj"):
                data = data.resolve_conj()
        data, indices, indptr, shape = _transpose_csr_for_spmv(
            data, indices, indptr, shape
        )
    (
        data,
        kernel_indices,
        kernel_indptr,
        n_rows,
        n_cols,
        row_lengths,
        max_row_nnz,
    ) = _prepare_spmv_csr_matrix(
        data,
        indices,
        indptr,
        shape,
        index_fallback_policy=index_fallback_policy,
    )
    block_nnz_use = block_nnz
    if max_segments is None:
        max_segments_use = max((max_row_nnz + block_nnz_use - 1) // block_nnz_use, 1)
        while max_segments_use > 2048 and block_nnz_use < 65536:
            block_nnz_use *= 2
            max_segments_use = max(
                (max_row_nnz + block_nnz_use - 1) // block_nnz_use,
                1,
            )
    else:
        max_segments_use = max_segments
    return PreparedCsrSpmv(
        data=data,
        kernel_indices=kernel_indices,
        kernel_indptr=kernel_indptr,
        shape=shape,
        n_rows=n_rows,
        n_cols=n_cols,
        block_nnz=block_nnz_use,
        max_segments=max_segments_use,
        opt_max_segments=max_segments,
        row_lengths=row_lengths,
        max_row_nnz=max_row_nnz,
        opt_buckets=None,
        transpose=transpose,
        op=op_code,
        index_fallback_policy=index_fallback_policy,
    )


def _get_spmv_baseline_data(prepared):
    compute_dtype = prepared._baseline_compute_dtype
    if compute_dtype == prepared.data.dtype:
        return compute_dtype, prepared.data
    if (
        prepared._baseline_data is None
        or prepared._baseline_data.dtype != compute_dtype
    ):
        prepared._baseline_data = prepared.data.to(compute_dtype)
    return compute_dtype, prepared._baseline_data


# ── MetaX/MACA tuning profiles ──────────────────────────────────────
# Keyed by MetaX model (see _common._maca_device_model). C550 is FlagTree's
# reference metax part; its entry is seeded from the CUDA path and is the one
# place to change once C550 measurements say otherwise.
_MACA_SPMV_PROFILES = {
    "c550": {
        "csr_kernel": "segbin",  # ROCm: "rowpar"
    },
}
_MACA_SPMV_DEFAULT_PROFILE = "c550"


def _maca_spmv_knob(name):
    """Read one MetaX SpMV tuning knob for the current model."""
    model = _maca_device_model() or _MACA_SPMV_DEFAULT_PROFILE
    profile = _MACA_SPMV_PROFILES.get(model, _MACA_SPMV_PROFILES[_MACA_SPMV_DEFAULT_PROFILE])
    return profile[name]


def _spmv_csr_default_backend():
    """Pick the default (non-opt) CSR SpMV kernel family for this runtime.

    DCU/ROCm uses the row-parallel kernel the DCU branch tuned for gfx936; CUDA
    keeps the nnz-partitioned segbin path. Override with
    FLAGSPARSE_SPMV_CSR_KERNEL=segbin|rowpar for A/B measurement.
    """
    override = os.environ.get("FLAGSPARSE_SPMV_CSR_KERNEL", "").strip().lower()
    if override in ("segbin", "rowpar"):
        return override
    if override:
        raise ValueError(
            "FLAGSPARSE_SPMV_CSR_KERNEL must be 'segbin' or 'rowpar', "
            f"got {override!r}"
        )
    if _is_rocm_runtime():
        return "rowpar"
    if _is_maca_runtime():
        # MetaX/MACA starts from the CUDA kernel; retune once C550 numbers exist.
        return _maca_spmv_knob("csr_kernel")
    if _is_mthreads_runtime() or _is_ascend_runtime():
        # Moore Threads and Ascend also start from the CUDA kernel.
        return "segbin"
    return "segbin"


def _triton_spmv_csr_impl_rowpar(prepared, x, compute_dtype):
    """DCU/ROCm row-parallel SpMV: one program per row, in-row segment loop."""
    device = prepared.data.device
    dtype = prepared.data.dtype
    y = torch.empty(prepared.n_rows, dtype=dtype, device=device)
    _, data_in = _get_spmv_baseline_data(prepared)
    x_in = x if compute_dtype == x.dtype else x.to(compute_dtype)
    grid = (prepared.n_rows,)
    if not _is_complex_dtype(compute_dtype):
        y_out = torch.empty(prepared.n_rows, dtype=compute_dtype, device=device)
        _spmv_csr_real_kernel[grid](
            data_in,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            x_in,
            y_out,
            n_rows=prepared.n_rows,
            BLOCK_NNZ=prepared.block_nnz,
            MAX_SEGMENTS=prepared.max_segments,
        )
        y.copy_(y_out if dtype == compute_dtype else y_out.to(dtype))
        return y
    data_ri = torch.view_as_real(data_in).reshape(-1)
    x_ri = torch.view_as_real(x_in).reshape(-1)
    y_ri = torch.empty(prepared.n_rows * 2, dtype=data_ri.dtype, device=device)
    _spmv_csr_complex_kernel[grid](
        data_ri,
        prepared.kernel_indices,
        prepared.kernel_indptr,
        x_ri,
        y_ri,
        n_rows=prepared.n_rows,
        BLOCK_NNZ=prepared.block_nnz,
        MAX_SEGMENTS=prepared.max_segments,
    )
    y.copy_(torch.view_as_complex(y_ri.reshape(prepared.n_rows, 2)))
    return y


def _triton_spmv_csr_impl_prepared(prepared, x):
    device = prepared.data.device
    dtype = prepared.data.dtype
    if prepared.n_rows == 0:
        return torch.empty(0, dtype=dtype, device=device)
    compute_dtype = prepared._baseline_compute_dtype
    if _spmv_csr_default_backend() == "rowpar":
        return _triton_spmv_csr_impl_rowpar(prepared, x, compute_dtype)
    # Fast path: preprocessing-free, load-balanced segmented nnz-split. Real dtype
    # only; native fp32/fp64 accumulation (fp16/bf16 accumulate in fp32). No
    # per-nonzero row-id or long-row metadata is needed — the row is found by an
    # in-kernel binary search on indptr, so this is fair to compare cold against
    # cuSPARSE (which also needs no separable analysis).
    if not _is_complex_dtype(compute_dtype):
        # fp32 accumulates natively (bandwidth-optimal, cuSPARSE-like); fp64
        # accumulates in fp64; fp16/bf16 accumulate in fp32. Native fp32
        # summation is order-dependent, so results carry standard fp32 SpMV error
        # (not the fp64-then-cast accuracy of the former baseline).
        acc_dtype = dtype if dtype in (torch.float32, torch.float64) else torch.float32
        nnz = int(prepared.data.numel())
        y_out = torch.zeros(prepared.n_rows, dtype=acc_dtype, device=device)
        if nnz > 0:
            acc_tl = tl.float64 if acc_dtype == torch.float64 else tl.float32
            steps = max(1, (prepared.n_rows + 1).bit_length())
            BLOCK = 256
            grid = ((nnz + BLOCK - 1) // BLOCK,)
            _spmv_csr_segbin_kernel[grid](
                prepared.data,
                prepared.kernel_indices,
                prepared.kernel_indptr,
                x,
                y_out,
                nnz,
                prepared.n_rows,
                STEPS=steps,
                ACC=acc_tl,
                BLOCK=BLOCK,
            )
        return y_out if acc_dtype == dtype else y_out.to(dtype)
    # Complex path: same preprocessing-free segmented nnz-split on interleaved
    # real/imag values, accumulating in native component precision (fp32 for
    # complex64, fp64 for complex128). complex64 therefore carries standard fp32
    # SpMV error, like the real fp32 path.
    data_ri = torch.view_as_real(prepared.data).reshape(-1)
    x_ri = torch.view_as_real(x.contiguous()).reshape(-1)
    comp_dtype = data_ri.dtype
    nnz = int(prepared.data.numel())
    y_ri = torch.zeros(prepared.n_rows * 2, dtype=comp_dtype, device=device)
    if nnz > 0:
        acc_tl = tl.float64 if comp_dtype == torch.float64 else tl.float32
        steps = max(1, (prepared.n_rows + 1).bit_length())
        BLOCK = 256
        grid = ((nnz + BLOCK - 1) // BLOCK,)
        _spmv_csr_complex_segbin_kernel[grid](
            data_ri,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            x_ri,
            y_ri,
            nnz,
            prepared.n_rows,
            STEPS=steps,
            ACC=acc_tl,
            BLOCK=BLOCK,
        )
    return torch.view_as_complex(y_ri.reshape(prepared.n_rows, 2))


def _spmv_uses_int64_indices(prepared):
    return (
        prepared.kernel_indices.dtype == torch.int64
        or prepared.kernel_indptr.dtype == torch.int64
    )


def _spmv_int32_fallback_blocker(prepared):
    if (
        prepared.kernel_indices.dtype == torch.int64
        and prepared.kernel_indices.numel() > 0
    ):
        min_index = int(prepared.kernel_indices.min().item())
        max_index = int(prepared.kernel_indices.max().item())
        if min_index < 0 or max_index > _INDEX_LIMIT_INT32:
            return (
                f"column index range [{min_index}, {max_index}] cannot fit int32 "
                f"for shape={prepared.shape}"
            )
    if (
        prepared.kernel_indptr.dtype == torch.int64
        and prepared.kernel_indptr.numel() > 0
    ):
        max_offset = int(prepared.kernel_indptr[-1].item())
        if max_offset > _INDEX_LIMIT_INT32:
            return f"CSR nnz offset {max_offset} cannot fit int32 for shape={prepared.shape}"
    if prepared.n_rows > _INDEX_LIMIT_INT32:
        return f"row count {prepared.n_rows} cannot fit int32 row metadata"
    return None


def _spmv_prepared_with_int32_indices(prepared, reason):
    blocker = _spmv_int32_fallback_blocker(prepared)
    if blocker is not None:
        raise RuntimeError(
            f"native int64 CSR SpMV failed and int32 fallback is unsafe: {blocker}"
        )
    kernel_indices = prepared.kernel_indices.to(torch.int32)
    kernel_indptr = prepared.kernel_indptr.to(torch.int32)
    row_lengths = kernel_indptr[1:] - kernel_indptr[:-1]
    max_row_nnz = int(row_lengths.max().item()) if prepared.n_rows > 0 else 0
    return PreparedCsrSpmv(
        data=prepared.data,
        kernel_indices=kernel_indices,
        kernel_indptr=kernel_indptr,
        shape=prepared.shape,
        n_rows=prepared.n_rows,
        n_cols=prepared.n_cols,
        block_nnz=prepared.block_nnz,
        max_segments=prepared.max_segments,
        opt_max_segments=prepared.opt_max_segments,
        row_lengths=row_lengths,
        max_row_nnz=max_row_nnz,
        opt_buckets=None,
        transpose=prepared.transpose,
        op=prepared.op,
        index_fallback_policy=prepared.index_fallback_policy,
        index_fallback_applied=True,
        index_fallback_reason=str(reason),
    )


def _run_spmv_prepared(prepared, x, use_opt=False, opt_buckets=None):
    if use_opt and prepared.supports_opt:
        return _triton_spmv_csr_impl_opt_prepared(prepared, x, opt_buckets=opt_buckets)
    return _triton_spmv_csr_impl_prepared(prepared, x)


def _run_spmv_prepared_with_fallback(prepared, x, use_opt=False, opt_buckets=None):
    try:
        return _run_spmv_prepared(prepared, x, use_opt=use_opt, opt_buckets=opt_buckets)
    except Exception as exc:
        if prepared.index_fallback_policy != "auto" or not _spmv_uses_int64_indices(
            prepared
        ):
            raise
        fallback_prepared = _spmv_prepared_with_int32_indices(prepared, exc)
        fallback_buckets = None
        if use_opt and fallback_prepared.supports_opt:
            fallback_buckets = _build_spmv_opt_runtime_buckets(fallback_prepared)
        return _run_spmv_prepared(
            fallback_prepared,
            x,
            use_opt=use_opt,
            opt_buckets=fallback_buckets,
        )


def flagsparse_spmv_csr(
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
    use_opt=False,
    prepared=None,
    transpose=None,
    op=None,
    index_fallback_policy="auto",
):
    """
    CSR SpMV using Triton.
    data, indices, indptr: CSR arrays; x: dense vector; shape: (n_rows, n_cols).
    prepared: cached CSR metadata from prepare_spmv_csr for steady-state runs.
    op: 0/'non' for A @ x, 1/'trans' for A.T @ x, 2/'conj' for A.conj().T @ x.
    max_segments: None = auto-compute from indptr so all NNZ per row are covered.
    use_opt: if True, use the faster CSR-Vector bucketed path (fp32/fp64 native accum).
    """
    op_explicit = op is not None
    op_code = _normalize_spmv_op(
        op,
        transpose=False if transpose is None else bool(transpose),
    )
    if (
        op_explicit
        and transpose is not None
        and bool(transpose) != _spmv_op_transposes(op_code)
    ):
        raise ValueError("transpose conflicts with op")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_spmv_csr(
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
                f"op={_spmv_op_to_name(op_code)} does not match prepared.op={_spmv_op_to_name(prepared.op)}"
            )
        if (
            not op_explicit
            and transpose is not None
            and bool(transpose) != prepared.transpose
        ):
            raise ValueError(
                f"transpose={bool(transpose)} does not match prepared.transpose={prepared.transpose}"
            )
    x = _validate_spmv_x(x, prepared)
    do_timing = bool(return_time or return_meta)
    symbolic_ms = 0.0 if do_timing else None
    compute_ms = None
    op_total_ms = None
    opt_buckets = None
    if do_timing:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    if use_opt and prepared.supports_opt:
        opt_buckets = _build_spmv_opt_runtime_buckets(prepared)
    if do_timing:
        _ACCEL.synchronize()
        t1 = time.perf_counter()
        symbolic_ms = (t1 - t0) * 1000.0 if use_opt and prepared.supports_opt else 0.0
    y = _run_spmv_prepared_with_fallback(
        prepared, x, use_opt=use_opt, opt_buckets=opt_buckets
    )
    if do_timing:
        _ACCEL.synchronize()
        t2 = time.perf_counter()
        compute_ms = (t2 - t1) * 1000.0
        op_total_ms = symbolic_ms + compute_ms
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
            "symbolic_ms": symbolic_ms,
            "compute_ms": compute_ms,
            "op_total_ms": op_total_ms,
            "bucket_count": int(len(opt_buckets)) if opt_buckets is not None else 0,
        }
        if return_time:
            return y, op_total_ms, meta
        return y, meta
    if return_time:
        return y, op_total_ms
    return y


def _coo_is_sorted_lex(row_i64, col_i64, n_cols):
    """True iff COO rows are non-decreasing lex order (row, col)."""
    n = row_i64.numel()
    if n <= 1:
        return True
    scale = max(1, int(n_cols))
    key = row_i64 * scale + col_i64
    return bool((key[1:] >= key[:-1]).all().item())


def coo_to_csr_for_spmv(data, row, col, shape, assume_sorted=False):
    """Convert COO to CSR triple (data, csr_col_indices, indptr) for SpMV."""
    n_rows, n_cols = int(shape[0]), int(shape[1])
    row64 = row.to(torch.int64)
    col64 = col.to(torch.int64)
    if row64.numel() == 0:
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
        return data, col64.to(torch.int32), indptr

    if assume_sorted or _coo_is_sorted_lex(row64, col64, n_cols):
        row_s, col_s, data_s = row64, col64, data
    else:
        key = row64 * max(1, n_cols) + col64
        order = torch.argsort(key)
        row_s = row64[order]
        col_s = col64[order]
        data_s = data[order].to(data.dtype)

    indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
    nnz = data_s.numel()
    if nnz > 0:
        nnz_per_row = torch.bincount(row_s, minlength=n_rows)
        indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    indices = col_s.to(torch.int32)
    return data_s, indices, indptr


def prepare_spmv_coo_tocsr(
    data,
    row,
    col,
    shape,
    block_nnz=256,
    max_segments=None,
    assume_sorted=False,
):
    """One-time COO → CSR + bucket metadata; use with ``flagsparse_spmv_coo_tocsr(..., prepared=p)``."""
    if not all(torch.is_tensor(t) for t in (data, row, col)):
        raise TypeError("data, row, col must all be torch.Tensor")
    if not all(_is_accel_tensor(t) for t in (data, row, col)):
        raise ValueError("data, row, col must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must all be 1D tensors")
    if data.dtype not in SUPPORTED_SPMV_VALUE_DTYPES:
        raise TypeError(_spmv_dtype_error_message())
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if row.numel() != col.numel() or data.numel() != row.numel():
        raise ValueError("data, row, col must have the same length")

    data_s, indices, indptr = coo_to_csr_for_spmv(
        data, row, col, shape, assume_sorted=assume_sorted
    )
    return prepare_spmv_csr(
        data_s,
        indices,
        indptr,
        shape,
        block_nnz=block_nnz,
        max_segments=max_segments,
    )


def flagsparse_spmv_coo_tocsr(
    data=None,
    row=None,
    col=None,
    x=None,
    shape=None,
    block_nnz=256,
    max_segments=None,
    out=None,
    return_time=False,
    use_opt=True,
    prepared=None,
    assume_sorted=False,
):
    """COO SpMV via CSR conversion: y = A @ x.

    Default ``use_opt=True`` enables the fast CSR-Vector path for float32/float64.
    If COO is already lex-sorted by (row, col), pass ``assume_sorted=True`` to skip ``argsort``.

    Steady-state: ``p = prepare_spmv_coo_tocsr(data, row, col, shape)`` then call with ``prepared=p``
    (``data``/``row``/``col`` may be omitted).
    """
    if prepared is not None:
        if x is None:
            raise TypeError("x is required")
        if shape is None:
            shape = prepared.shape
        sh = (int(shape[0]), int(shape[1]))
        if sh != prepared.shape:
            raise ValueError(
                f"shape {sh} does not match prepared.shape {prepared.shape}"
            )
        return flagsparse_spmv_csr(
            x=x,
            shape=shape,
            block_nnz=block_nnz,
            max_segments=max_segments,
            out=out,
            return_time=return_time,
            use_opt=use_opt,
            prepared=prepared,
        )

    if not all(torch.is_tensor(t) for t in (data, row, col, x)):
        raise TypeError("data, row, col, x must all be torch.Tensor")
    if not all(_is_accel_tensor(t) for t in (data, row, col, x)):
        raise ValueError("data, row, col, x must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1 or x.ndim != 1:
        raise ValueError("data, row, col, x must all be 1D tensors")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.dtype not in SUPPORTED_SPMV_VALUE_DTYPES:
        raise TypeError(_spmv_dtype_error_message())
    if x.dtype != data.dtype:
        raise TypeError("x dtype must match data dtype")

    data_s, indices, indptr = coo_to_csr_for_spmv(
        data, row, col, shape, assume_sorted=assume_sorted
    )

    return flagsparse_spmv_csr(
        data_s,
        indices,
        indptr,
        x,
        shape,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        return_time=return_time,
        use_opt=use_opt,
    )
