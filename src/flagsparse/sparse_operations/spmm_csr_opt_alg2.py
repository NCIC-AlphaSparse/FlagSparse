"""Hardware-aware CSR SpMM opt-alg2 kernels, helpers, and benchmark entry points."""

from ._common import *

import math

from .spmm_csr import (
    _prepare_spmm_csr_matrix,
    _spmm_coo_reference_tolerance,
    _spmm_validation_metrics,
    flagsparse_spmm_csr,
    flagsparse_spmm_csr_opt,
    prepare_spmm_csr_opt,
)


SUPPORTED_SPMM_OPT_ALG2_DTYPES = (torch.float32, torch.float64)


class PreparedCsrSpmmOptAlg2:
    """Cached CSR metadata for repeated alg2 CSR SpMM calls on the same sparse matrix."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "row_lengths",
        "max_row_nnz",
        "opt_buckets",
        "supports_opt",
    )

    def __init__(
        self,
        data,
        kernel_indices,
        kernel_indptr,
        shape,
        n_rows,
        n_cols,
        row_lengths,
        max_row_nnz,
        opt_buckets,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.row_lengths = row_lengths
        self.max_row_nnz = int(max_row_nnz)
        self.opt_buckets = opt_buckets
        self.supports_opt = data.dtype in SUPPORTED_SPMM_OPT_ALG2_DTYPES


_SPMM_OPT_ALG2_BUCKET_SPECS_F32 = (
    {
        "label": "short_16",
        "kind": "batched2d",
        "min_row_nnz": 0,
        "max_row_nnz": 16,
        "batch_rows": 8,
        "block_k": 16,
        "block_n_cap": 32,
    },
    {
        "label": "short_64",
        "kind": "batched2d",
        "min_row_nnz": 17,
        "max_row_nnz": 64,
        "batch_rows": 4,
        "block_k": 32,
        "block_n_cap": 64,
    },
    {
        "label": "row_256",
        "kind": "row2d",
        "min_row_nnz": 65,
        "max_row_nnz": 256,
        "batch_rows": 1,
        "block_k": 64,
        "block_n_cap": 64,
    },
    {
        "label": "seg_1024",
        "kind": "row2d_segmented",
        "min_row_nnz": 257,
        "max_row_nnz": 1024,
        "batch_rows": 1,
        "block_k": 64,
        "block_n_cap": 128,
        "segments": 4,
    },
    {
        "label": "seg_long",
        "kind": "row2d_segmented",
        "min_row_nnz": 1025,
        "max_row_nnz": None,
        "batch_rows": 1,
        "block_k": 128,
        "block_n_cap": 128,
        "segments": 8,
    },
)

_SPMM_OPT_ALG2_BUCKET_SPECS_F64 = (
    {
        "label": "short_16",
        "kind": "batched2d",
        "min_row_nnz": 0,
        "max_row_nnz": 16,
        "batch_rows": 4,
        "block_k": 16,
        "block_n_cap": 16,
    },
    {
        "label": "short_64",
        "kind": "batched2d",
        "min_row_nnz": 17,
        "max_row_nnz": 64,
        "batch_rows": 2,
        "block_k": 32,
        "block_n_cap": 32,
    },
    {
        "label": "row_256",
        "kind": "row2d",
        "min_row_nnz": 65,
        "max_row_nnz": 256,
        "batch_rows": 1,
        "block_k": 32,
        "block_n_cap": 32,
    },
    {
        "label": "seg_1024",
        "kind": "row2d_segmented",
        "min_row_nnz": 257,
        "max_row_nnz": 1024,
        "batch_rows": 1,
        "block_k": 32,
        "block_n_cap": 64,
        "segments": 4,
    },
    {
        "label": "seg_long",
        "kind": "row2d_segmented",
        "min_row_nnz": 1025,
        "max_row_nnz": None,
        "batch_rows": 1,
        "block_k": 64,
        "block_n_cap": 64,
        "segments": 8,
    },
)


def _spmm_opt_alg2_bucket_specs(dtype):
    return (
        _SPMM_OPT_ALG2_BUCKET_SPECS_F64
        if dtype == torch.float64
        else _SPMM_OPT_ALG2_BUCKET_SPECS_F32
    )


def _round_down_power_of_two(value):
    if value <= 1:
        return 1
    return 1 << (int(value).bit_length() - 1)


def _normalize_spmm_opt_alg2_device_props(device):
    props = torch.cuda.get_device_properties(device)
    warp_size = int(getattr(props, "warp_size", 32) or 32)
    sm_count = int(getattr(props, "multi_processor_count", 0) or 0)
    max_threads_per_mp = int(getattr(props, "max_threads_per_multi_processor", 2048) or 2048)
    max_threads_per_block = int(getattr(props, "max_threads_per_block", 1024) or 1024)
    shared_memory_per_block = int(getattr(props, "shared_memory_per_block", 0) or 0)
    name = str(getattr(props, "name", "cuda"))
    capability = (
        int(getattr(props, "major", 0) or 0),
        int(getattr(props, "minor", 0) or 0),
    )
    return {
        "device_name": name,
        "warp_size": warp_size,
        "sm_count": sm_count,
        "max_threads_per_mp": max_threads_per_mp,
        "max_threads_per_block": max_threads_per_block,
        "shared_memory_per_block": shared_memory_per_block,
        "capability": capability,
    }


def _select_spmm_opt_alg2_block_n(n_dense_cols, block_n_cap):
    if n_dense_cols <= 8:
        block_n = 8
    elif n_dense_cols <= 16:
        block_n = 16
    elif n_dense_cols <= 32:
        block_n = 32
    elif n_dense_cols <= 64:
        block_n = 64
    else:
        block_n = 128
    return max(8, min(int(block_n), int(block_n_cap)))


def _select_spmm_opt_alg2_num_warps(bucket, block_n, device_props, dtype):
    warp_size = max(1, int(device_props["warp_size"]))
    max_threads_per_block = max(32, int(device_props["max_threads_per_block"]))
    max_threads_per_mp = max(max_threads_per_block, int(device_props["max_threads_per_mp"]))
    lane_need = max(1, math.ceil(int(block_n) / warp_size))
    kind = bucket["kind"]
    block_k = int(bucket["block_k"])
    segments = int(bucket.get("segments", 1))
    batch_rows = int(bucket.get("batch_rows", 1))

    if kind == "batched2d":
        desired = max(lane_need, 1 if block_k <= 16 else 2)
        if batch_rows >= 8 and block_n >= 64:
            desired = max(desired, 2)
    elif kind == "row2d":
        desired = max(lane_need, 2 if block_k <= 32 else 4)
    else:
        desired = max(lane_need, 4)
        if segments >= 4:
            desired = max(desired, 8 if block_k >= 64 else 4)
        if segments >= 8:
            desired = max(desired, 16 if dtype == torch.float32 and block_n >= 64 else 8)

    if dtype == torch.float64 and desired > 8:
        desired = 8
    if max_threads_per_mp < 1536 and desired > 8:
        desired = 8

    max_warps_by_block = max(1, max_threads_per_block // warp_size)
    max_warps_by_mp = max(1, max_threads_per_mp // warp_size)
    max_supported = min(16, max_warps_by_block, max_warps_by_mp)
    supported = [value for value in (1, 2, 4, 8, 16) if value <= max_supported]
    if not supported:
        return 1
    clipped = min(desired, supported[-1])
    for value in reversed(supported):
        if value <= clipped:
            return value
    return supported[0]


def _select_spmm_opt_alg2_num_stages(bucket, block_n, num_warps, device_props, dtype):
    kind = bucket["kind"]
    segments = int(bucket.get("segments", 1))
    shared_memory_per_block = int(device_props["shared_memory_per_block"])

    if dtype == torch.float64:
        if kind == "row2d_segmented":
            stages = 1
        else:
            stages = 2 if block_n <= 32 and num_warps <= 4 else 1
    else:
        if kind == "batched2d":
            stages = 2
        elif kind == "row2d":
            stages = 2 if block_n <= 64 else 1
        else:
            stages = 1 if segments >= 8 or num_warps >= 16 else 2

    if shared_memory_per_block and shared_memory_per_block < 65536:
        stages = min(stages, 2)
    if kind == "row2d_segmented" and block_n >= 128:
        stages = min(stages, 2)
    return max(1, min(int(stages), 4))


def _resolve_spmm_opt_alg2_launch(bucket, n_dense_cols, dtype, device_props):
    block_n = _select_spmm_opt_alg2_block_n(n_dense_cols, bucket["block_n_cap"])
    num_warps = _select_spmm_opt_alg2_num_warps(
        bucket,
        block_n,
        device_props,
        dtype,
    )
    num_stages = _select_spmm_opt_alg2_num_stages(
        bucket,
        block_n,
        num_warps,
        device_props,
        dtype,
    )
    launch = {
        "bucket_label": bucket["label"],
        "kind": bucket["kind"],
        "block_k": int(bucket["block_k"]),
        "block_n": int(block_n),
        "num_warps": int(num_warps),
        "num_stages": int(num_stages),
        "batch_rows": int(bucket.get("batch_rows", 1)),
        "segments": int(bucket.get("segments", 1)),
        "row_count": int(bucket["rows"].numel()),
    }
    return launch


def _build_spmm_opt_alg2_buckets(row_lengths, dtype):
    device = row_lengths.device
    row_count = int(row_lengths.numel())
    row_index_dtype = torch.int32 if row_count <= _INDEX_LIMIT_INT32 else torch.int64
    all_rows = torch.arange(row_count, device=device, dtype=row_index_dtype)
    buckets = []
    for spec in _spmm_opt_alg2_bucket_specs(dtype):
        lower = int(spec["min_row_nnz"])
        upper = spec["max_row_nnz"]
        if upper is None:
            mask = row_lengths >= lower
        elif lower <= 0:
            mask = row_lengths <= upper
        else:
            mask = (row_lengths >= lower) & (row_lengths <= upper)
        rows = all_rows[mask]
        if rows.numel() == 0:
            continue
        bucket = {
            "label": spec["label"],
            "kind": spec["kind"],
            "rows": rows,
            "batch_rows": int(spec["batch_rows"]),
            "block_k": int(spec["block_k"]),
            "block_n_cap": int(spec["block_n_cap"]),
            "segments": int(spec.get("segments", 1)),
        }
        buckets.append(bucket)
    return buckets


@triton.jit
def _spmm_opt_alg2_symbolic_count_kernel(
    row_lengths_ptr,
    bucket_counts_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_rows
    lens = tl.load(row_lengths_ptr + offs, mask=mask, other=0)
    bucket = tl.where(lens <= 16, 0, tl.where(lens <= 64, 1, tl.where(lens <= 256, 2, tl.where(lens <= 1024, 3, 4))))
    for bid in tl.static_range(0, 5):
        hits = mask & (bucket == bid)
        count = tl.sum(tl.where(hits, 1, 0), axis=0)
        tl.atomic_add(bucket_counts_ptr + bid, count, sem="relaxed")


@triton.jit
def _spmm_opt_alg2_symbolic_compact_kernel(
    row_lengths_ptr,
    bucket_offsets_ptr,
    bucket_write_counts_ptr,
    rows_flat_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_rows
    lens = tl.load(row_lengths_ptr + offs, mask=mask, other=0)
    bucket = tl.where(lens <= 16, 0, tl.where(lens <= 64, 1, tl.where(lens <= 256, 2, tl.where(lens <= 1024, 3, 4))))
    for bid in tl.static_range(0, 5):
        hits = mask & (bucket == bid)
        ranks = tl.cumsum(tl.where(hits, 1, 0), axis=0) - 1
        local_count = tl.sum(tl.where(hits, 1, 0), axis=0)
        base = tl.atomic_add(bucket_write_counts_ptr + bid, local_count, sem="relaxed")
        offset = tl.load(bucket_offsets_ptr + bid)
        tl.store(rows_flat_ptr + offset + base + ranks, offs, mask=hits)


def _build_spmm_opt_alg2_buckets_triton_symbolic(row_lengths, dtype):
    device = row_lengths.device
    row_count = int(row_lengths.numel())
    row_index_dtype = torch.int32 if row_count <= _INDEX_LIMIT_INT32 else torch.int64
    bucket_count = len(_spmm_opt_alg2_bucket_specs(dtype))
    if bucket_count != 5:
        raise RuntimeError("Alg2S symbolic builder expects five alg2 buckets")

    counts = torch.zeros((bucket_count,), dtype=torch.int64, device=device)
    block_m = 256
    grid = (triton.cdiv(row_count, block_m),)
    if row_count > 0:
        _spmm_opt_alg2_symbolic_count_kernel[grid](
            row_lengths,
            counts,
            row_count,
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=1,
        )
    offsets = torch.empty_like(counts)
    offsets[0] = 0
    if bucket_count > 1:
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)

    rows_flat = torch.empty((row_count,), dtype=row_index_dtype, device=device)
    write_counts = torch.zeros_like(counts)
    if row_count > 0:
        _spmm_opt_alg2_symbolic_compact_kernel[grid](
            row_lengths,
            offsets,
            write_counts,
            rows_flat,
            row_count,
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=1,
        )

    counts_cpu = counts.to("cpu").tolist()
    offsets_cpu = offsets.to("cpu").tolist()
    buckets = []
    for spec, count, offset in zip(_spmm_opt_alg2_bucket_specs(dtype), counts_cpu, offsets_cpu):
        count = int(count)
        if count == 0:
            continue
        bucket = {
            "label": spec["label"],
            "kind": spec["kind"],
            "rows": rows_flat.narrow(0, int(offset), count),
            "batch_rows": int(spec["batch_rows"]),
            "block_k": int(spec["block_k"]),
            "block_n_cap": int(spec["block_n_cap"]),
            "segments": int(spec.get("segments", 1)),
        }
        buckets.append(bucket)
    return buckets


@triton.jit
def _spmm_csr_alg2_batched_rows_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    rows_ptr,
    n_bucket_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BATCH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols

    for batch_idx in tl.static_range(0, BATCH):
        ridx = pid_row * BATCH + batch_idx
        active = ridx < n_bucket_rows
        row = tl.load(rows_ptr + ridx, mask=active, other=0)
        start = tl.load(indptr_ptr + row, mask=active, other=0)
        end = tl.load(indptr_ptr + row + 1, mask=active, other=0)
        row_nnz = end - start
        acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
        for chunk_start in tl.range(0, row_nnz, BLOCK_K):
            for kk in tl.static_range(0, BLOCK_K):
                idx = start + chunk_start + kk
                valid = active & (idx < end)
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                acc = acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)
        tl.store(
            c_ptr + row * stride_cm + offs_n * stride_cn,
            acc.to(OUT_DTYPE),
            mask=mask_n & active,
        )


@triton.jit
def _spmm_csr_alg2_row_rows_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    rows_ptr,
    n_bucket_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_bucket_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    for chunk_start in tl.range(0, row_nnz, BLOCK_K):
        for kk in tl.static_range(0, BLOCK_K):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            acc = acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)
    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc.to(OUT_DTYPE), mask=mask_n)


@triton.jit
def _spmm_csr_alg2_segmented_rows_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    rows_ptr,
    n_bucket_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SEGMENTS: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_bucket_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    seg_span = (row_nnz + SEGMENTS - 1) // SEGMENTS
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc0 = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    acc1 = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    acc2 = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    acc3 = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    acc4 = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    acc5 = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    acc6 = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    acc7 = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)

    if SEGMENTS > 0:
        seg_start = start
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_K):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_K):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)
            acc0 = acc0 + chunk_acc

    if SEGMENTS > 1:
        seg_start = start + seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_K):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_K):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)
            acc1 = acc1 + chunk_acc

    if SEGMENTS > 2:
        seg_start = start + 2 * seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_K):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_K):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)
            acc2 = acc2 + chunk_acc

    if SEGMENTS > 3:
        seg_start = start + 3 * seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_K):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_K):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)
            acc3 = acc3 + chunk_acc

    if SEGMENTS > 4:
        seg_start = start + 4 * seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_K):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_K):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)
            acc4 = acc4 + chunk_acc

    if SEGMENTS > 5:
        seg_start = start + 5 * seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_K):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_K):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)
            acc5 = acc5 + chunk_acc

    if SEGMENTS > 6:
        seg_start = start + 6 * seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_K):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_K):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)
            acc6 = acc6 + chunk_acc

    if SEGMENTS > 7:
        seg_start = start + 7 * seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_K):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_K):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val.to(ACC_DTYPE) * b_vals.to(ACC_DTYPE)
            acc7 = acc7 + chunk_acc

    if SEGMENTS == 1:
        acc = acc0
    elif SEGMENTS == 2:
        acc = acc0 + acc1
    elif SEGMENTS <= 4:
        acc = (acc0 + acc1) + (acc2 + acc3)
    else:
        acc = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))
    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc.to(OUT_DTYPE), mask=mask_n)


def prepare_spmm_csr_opt_alg2(data, indices, indptr, shape):
    (
        data,
        kernel_indices,
        kernel_indptr,
        n_rows,
        n_cols,
        row_lengths,
        max_row_nnz,
    ) = _prepare_spmm_csr_matrix(data, indices, indptr, shape)
    return PreparedCsrSpmmOptAlg2(
        data=data,
        kernel_indices=kernel_indices,
        kernel_indptr=kernel_indptr,
        shape=shape,
        n_rows=n_rows,
        n_cols=n_cols,
        row_lengths=row_lengths,
        max_row_nnz=max_row_nnz,
        opt_buckets=[],
    )


def prepare_spmm_csr_opt_alg2_symbolic(data, indices, indptr, shape):
    return prepare_spmm_csr_opt_alg2(data, indices, indptr, shape)


def _validate_spmm_opt_alg2_runtime_inputs(prepared, B, out):
    if not prepared.supports_opt:
        raise TypeError("flagsparse_spmm_csr_opt_alg2 only supports float32 and float64")
    if B is None:
        raise ValueError("B is required")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != prepared.data.device:
        raise ValueError("B must be on the same CUDA device as sparse matrix data")
    if B.dtype != prepared.data.dtype:
        raise TypeError("B dtype must match sparse matrix dtype")
    if B.shape[0] != prepared.n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={prepared.n_cols}, got {B.shape[0]}")
    if out is not None:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.device != prepared.data.device:
            raise ValueError("out must be on the same CUDA device as sparse matrix data")
        if out.shape != (prepared.n_rows, int(B.shape[1])) or out.dtype != prepared.data.dtype:
            raise ValueError("out shape/dtype must match result")


def _spmm_opt_alg2_acc_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _run_spmm_opt_alg2_bucket(prepared, bucket, B, C_out, device_props):
    launch = _resolve_spmm_opt_alg2_launch(bucket, int(B.shape[1]), prepared.data.dtype, device_props)
    rows = bucket["rows"]
    if rows.numel() == 0:
        return launch

    acc_dtype = _spmm_opt_alg2_acc_dtype(prepared.data.dtype)
    out_dtype = tl.float64 if prepared.data.dtype == torch.float64 else tl.float32
    common_kwargs = {
        "num_warps": launch["num_warps"],
        "num_stages": launch["num_stages"],
    }

    if bucket["kind"] == "batched2d":
        grid = (
            triton.cdiv(rows.numel(), launch["batch_rows"]),
            triton.cdiv(B.shape[1], launch["block_n"]),
        )
        _spmm_csr_alg2_batched_rows_kernel[grid](
            prepared.data,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            B,
            C_out,
            rows,
            rows.numel(),
            B.shape[1],
            B.stride(0),
            B.stride(1),
            C_out.stride(0),
            C_out.stride(1),
            BATCH=launch["batch_rows"],
            BLOCK_N=launch["block_n"],
            BLOCK_K=launch["block_k"],
            ACC_DTYPE=acc_dtype,
            OUT_DTYPE=out_dtype,
            **common_kwargs,
        )
    elif bucket["kind"] == "row2d":
        grid = (rows.numel(), triton.cdiv(B.shape[1], launch["block_n"]))
        _spmm_csr_alg2_row_rows_kernel[grid](
            prepared.data,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            B,
            C_out,
            rows,
            rows.numel(),
            B.shape[1],
            B.stride(0),
            B.stride(1),
            C_out.stride(0),
            C_out.stride(1),
            BLOCK_N=launch["block_n"],
            BLOCK_K=launch["block_k"],
            ACC_DTYPE=acc_dtype,
            OUT_DTYPE=out_dtype,
            **common_kwargs,
        )
    else:
        grid = (rows.numel(), triton.cdiv(B.shape[1], launch["block_n"]))
        _spmm_csr_alg2_segmented_rows_kernel[grid](
            prepared.data,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            B,
            C_out,
            rows,
            rows.numel(),
            B.shape[1],
            B.stride(0),
            B.stride(1),
            C_out.stride(0),
            C_out.stride(1),
            BLOCK_N=launch["block_n"],
            BLOCK_K=launch["block_k"],
            SEGMENTS=launch["segments"],
            ACC_DTYPE=acc_dtype,
            OUT_DTYPE=out_dtype,
            **common_kwargs,
        )

    launch["grid_m"] = int(grid[0])
    launch["grid_n"] = int(grid[1])
    return launch


def _triton_spmm_csr_impl_opt_alg2_prepared(prepared, B, opt_buckets=None, return_meta=False):
    device_props = _normalize_spmm_opt_alg2_device_props(prepared.data.device)
    C_out = torch.zeros((prepared.n_rows, int(B.shape[1])), dtype=B.dtype, device=B.device)
    launch_configs = []
    bucket_hits = []
    if opt_buckets is None:
        opt_buckets = prepared.opt_buckets
    for bucket in opt_buckets:
        launch = _run_spmm_opt_alg2_bucket(prepared, bucket, B, C_out, device_props)
        launch_configs.append(launch)
        bucket_hits.append(
            {
                "bucket_label": launch["bucket_label"],
                "kind": launch["kind"],
                "row_count": launch["row_count"],
            }
        )

    meta = None
    if return_meta:
        meta = {
            "device_name": device_props["device_name"],
            "sm_count": device_props["sm_count"],
            "warp_size": device_props["warp_size"],
            "bucket_hits": bucket_hits,
            "launch_configs": launch_configs,
            "max_row_nnz": prepared.max_row_nnz,
            "n_dense_cols": int(B.shape[1]),
        }
    return C_out, meta


def flagsparse_spmm_csr_opt_alg2(
    data=None,
    indices=None,
    indptr=None,
    B=None,
    shape=None,
    prepared=None,
    out=None,
    return_time=False,
    return_meta=False,
):
    """CSR SpMM opt-alg2: hardware-aware bucketed path for CSR @ dense."""

    if prepared is not None and not isinstance(prepared, PreparedCsrSpmmOptAlg2):
        raise TypeError("prepared must be a PreparedCsrSpmmOptAlg2 instance")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_spmm_csr_opt_alg2(data, indices, indptr, shape)
    elif shape is not None:
        resolved_shape = (int(shape[0]), int(shape[1]))
        if resolved_shape != prepared.shape:
            raise ValueError(
                f"shape {resolved_shape} does not match prepared.shape {prepared.shape}"
            )

    _validate_spmm_opt_alg2_runtime_inputs(prepared, B, out)
    B = B.contiguous()

    do_timing = bool(return_time or return_meta)
    symbolic_ms = None
    compute_ms = None
    op_total_ms = None

    if do_timing:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    opt_buckets = _build_spmm_opt_alg2_buckets(prepared.row_lengths, prepared.data.dtype)
    prepared.opt_buckets = opt_buckets
    if do_timing:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        symbolic_ms = (t1 - t0) * 1000.0

    C, meta = _triton_spmm_csr_impl_opt_alg2_prepared(
        prepared,
        B,
        opt_buckets=opt_buckets,
        return_meta=return_meta,
    )
    if do_timing:
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        compute_ms = (t2 - t1) * 1000.0
        op_total_ms = symbolic_ms + compute_ms
    if out is not None:
        out.copy_(C)
        C = out
    if return_meta:
        if meta is None:
            meta = {}
        meta.update(
            {
                "symbolic_ms": symbolic_ms,
                "compute_ms": compute_ms,
                "op_total_ms": op_total_ms,
            }
        )
    if return_time and return_meta:
        return C, op_total_ms, meta
    if return_time:
        return C, op_total_ms
    if return_meta:
        return C, meta
    return C


def flagsparse_spmm_csr_opt_alg2_symbolic(
    data=None,
    indices=None,
    indptr=None,
    B=None,
    shape=None,
    prepared=None,
    out=None,
    return_time=False,
    return_meta=False,
):
    """CSR SpMM opt-alg2 with Triton runtime symbolic bucket construction."""

    if prepared is not None and not isinstance(prepared, PreparedCsrSpmmOptAlg2):
        raise TypeError("prepared must be a PreparedCsrSpmmOptAlg2 instance")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_spmm_csr_opt_alg2_symbolic(data, indices, indptr, shape)
    elif shape is not None:
        resolved_shape = (int(shape[0]), int(shape[1]))
        if resolved_shape != prepared.shape:
            raise ValueError(
                f"shape {resolved_shape} does not match prepared.shape {prepared.shape}"
            )

    _validate_spmm_opt_alg2_runtime_inputs(prepared, B, out)
    B = B.contiguous()

    do_timing = bool(return_time or return_meta)
    symbolic_ms = None
    compute_ms = None
    op_total_ms = None

    if do_timing:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    opt_buckets = _build_spmm_opt_alg2_buckets_triton_symbolic(
        prepared.row_lengths,
        prepared.data.dtype,
    )
    prepared.opt_buckets = opt_buckets
    if do_timing:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        symbolic_ms = (t1 - t0) * 1000.0

    C, meta = _triton_spmm_csr_impl_opt_alg2_prepared(
        prepared,
        B,
        opt_buckets=opt_buckets,
        return_meta=return_meta,
    )
    if do_timing:
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        compute_ms = (t2 - t1) * 1000.0
        op_total_ms = symbolic_ms + compute_ms
    if out is not None:
        out.copy_(C)
        C = out
    if return_meta:
        if meta is None:
            meta = {}
        meta.update(
            {
                "symbolic_ms": symbolic_ms,
                "compute_ms": compute_ms,
                "op_total_ms": op_total_ms,
            }
        )
    if return_time and return_meta:
        return C, op_total_ms, meta
    if return_time:
        return C, op_total_ms
    if return_meta:
        return C, meta
    return C


def _spmm_opt_alg2_reference_error(candidate, reference, value_dtype):
    atol, rtol = _spmm_coo_reference_tolerance(value_dtype)
    if candidate.numel() == 0:
        return 0.0
    diff = torch.abs(candidate - reference)
    denom = atol + rtol * torch.abs(reference)
    return float(torch.max(diff / denom).item())


def benchmark_spmm_opt_alg2_case(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    n_dense_cols=32,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    run_cusparse=True,
):
    """Benchmark CSR SpMM base, opt, and opt-alg2 against torch / sparse backend references."""

    if value_dtype not in SUPPORTED_SPMM_OPT_ALG2_DTYPES:
        raise TypeError("benchmark_spmm_opt_alg2_case only supports float32 and float64")

    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_cols, nnz, value_dtype, index_dtype, device
    )
    B = _build_random_dense((n_cols, n_dense_cols), value_dtype, device)
    shape = (n_rows, n_cols)

    prepared_opt = prepare_spmm_csr_opt(data, indices, indptr, shape)
    prepared_alg2 = prepare_spmm_csr_opt_alg2(data, indices, indptr, shape)

    base_values, base_ms = _benchmark_cuda_op(
        lambda: flagsparse_spmm_csr(data, indices, indptr, B, shape),
        warmup=warmup,
        iters=iters,
    )
    opt_values, opt_ms = _benchmark_cuda_op(
        lambda: flagsparse_spmm_csr_opt(B=B, prepared=prepared_opt),
        warmup=warmup,
        iters=iters,
    )
    alg2_values, alg2_ms = _benchmark_cuda_op(
        lambda: flagsparse_spmm_csr_opt_alg2(B=B, prepared=prepared_alg2),
        warmup=warmup,
        iters=iters,
    )

    indptr64 = indptr.to(torch.int64)
    indices64 = indices.to(torch.int64)
    ref_dtype = torch.float64 if value_dtype == torch.float32 else value_dtype
    csr_ref = torch.sparse_csr_tensor(
        indptr64,
        indices64,
        data.to(ref_dtype),
        size=shape,
        device=device,
    )
    torch_ref = torch.sparse.mm(csr_ref, B.to(ref_dtype)).to(value_dtype)

    torch_ms = None
    try:
        pt_sparse = torch.sparse_csr_tensor(indptr64, indices64, data, size=shape, device=device)
        _, torch_ms = _benchmark_cuda_op(lambda: torch.sparse.mm(pt_sparse, B), warmup=warmup, iters=iters)
    except Exception:
        torch_ms = None

    sparse_backend = None
    sparse_ms = None
    sparse_reason = None
    if run_cusparse:
        if cp is None or cpx_sparse is None:
            sparse_reason = "CuPy/cuSPARSE is not available"
        else:
            try:
                data_cp = _cupy_from_torch(data)
                indices_cp = _cupy_from_torch(indices.to(torch.int64))
                indptr_cp = _cupy_from_torch(indptr)
                B_cp = _cupy_from_torch(B)
                A_csr = cpx_sparse.csr_matrix((data_cp, indices_cp, indptr_cp), shape=shape)
                sparse_backend, sparse_ms = _benchmark_cuda_op(lambda: A_csr @ B_cp, warmup=warmup, iters=iters)
                sparse_backend = _torch_from_cupy(sparse_backend)
            except Exception as exc:
                sparse_reason = str(exc)

    return {
        "parameters": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": nnz,
            "n_dense_cols": n_dense_cols,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
        },
        "performance": {
            "base_ms": base_ms,
            "opt_ms": opt_ms,
            "opt_alg2_ms": alg2_ms,
            "torch_ms": torch_ms,
            "cusparse_ms": sparse_ms,
        },
        "verification": {
            "base_vs_torch_err": _spmm_opt_alg2_reference_error(base_values, torch_ref, value_dtype),
            "opt_vs_torch_err": _spmm_opt_alg2_reference_error(opt_values, torch_ref, value_dtype),
            "opt_alg2_vs_torch_err": _spmm_opt_alg2_reference_error(alg2_values, torch_ref, value_dtype),
            "base_vs_cusparse_err": (
                _spmm_opt_alg2_reference_error(base_values, sparse_backend, value_dtype)
                if sparse_backend is not None
                else None
            ),
            "opt_vs_cusparse_err": (
                _spmm_opt_alg2_reference_error(opt_values, sparse_backend, value_dtype)
                if sparse_backend is not None
                else None
            ),
            "opt_alg2_vs_cusparse_err": (
                _spmm_opt_alg2_reference_error(alg2_values, sparse_backend, value_dtype)
                if sparse_backend is not None
                else None
            ),
        },
        "backend_status": {
            "cusparse_unavailable_reason": sparse_reason,
        },
        "samples": {
            "base_triton": base_values,
            "opt_triton": opt_values,
            "opt_alg2_triton": alg2_values,
            "torch_ref": torch_ref,
            "cusparse_ref": sparse_backend,
        },
    }
