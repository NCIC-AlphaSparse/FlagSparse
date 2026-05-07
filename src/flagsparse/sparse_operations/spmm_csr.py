"""CSR SpMM kernels, helpers, and benchmark entry points."""

from ._common import *
from ._alpha_spmm_alg1_common import _select_alpha_spmm_alg1_warp_and_factor

SUPPORTED_SPMM_VALUE_DTYPES = SUPPORTED_VALUE_DTYPES
def _spmm_relative_threshold(value_dtype):
    if value_dtype == torch.float16:
        return 5e-3
    if value_dtype == torch.bfloat16:
        return 1e-2
    if value_dtype in (torch.float32, torch.complex64):
        return 1e-6
    if value_dtype in (torch.float64, torch.complex128):
        return 1e-12
    return 1e-6


def _spmm_coo_reference_tolerance(value_dtype):
    if value_dtype == torch.float16:
        return 2e-3, 2e-3
    if value_dtype == torch.bfloat16:
        return 1e-1, 1e-1
    if value_dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-2
    if value_dtype in (torch.float64, torch.complex128):
        return 1e-12, 1e-10
    return 1e-6, 1e-5


def _spmm_error_metrics(candidate, reference):
    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate and reference must have the same shape, got {candidate.shape} vs {reference.shape}"
        )

    if candidate.numel() == 0:
        return {
            "max_abs_error": 0.0,
            "max_relative_error": 0.0,
            "sum_relative_error": 0.0,
            "reference_max_magnitude": 0.0,
            "reference_sum_magnitude": 0.0,
        }

    if _is_complex_dtype(reference.dtype):
        candidate_compare = torch.abs(candidate)
        reference_compare = torch.abs(reference)
        abs_diff = torch.abs(candidate_compare - reference_compare)
    else:
        reference_compare = torch.abs(reference)
        abs_diff = torch.abs(candidate - reference)

    max_abs_error = float(torch.max(abs_diff).item())
    reference_max_magnitude = float(torch.max(reference_compare).item())
    sum_abs_error = float(torch.sum(abs_diff).item())
    reference_sum_magnitude = float(torch.sum(reference_compare).item())

    if reference_max_magnitude == 0.0:
        max_relative_error = 0.0 if max_abs_error == 0.0 else float("inf")
    else:
        max_relative_error = max_abs_error / reference_max_magnitude

    if reference_sum_magnitude == 0.0:
        sum_relative_error = 0.0 if sum_abs_error == 0.0 else float("inf")
    else:
        sum_relative_error = sum_abs_error / reference_sum_magnitude

    return {
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
        "sum_relative_error": sum_relative_error,
        "reference_max_magnitude": reference_max_magnitude,
        "reference_sum_magnitude": reference_sum_magnitude,
    }


def _spmm_validation_metrics(candidate, reference):
    metrics = _spmm_error_metrics(candidate, reference)
    threshold = _spmm_relative_threshold(reference.dtype)
    atol, rtol = _tolerance_for_dtype(reference.dtype)
    metrics.update(
        {
            "relative_threshold": threshold,
            "matches_threshold": metrics["max_relative_error"] <= threshold,
            "strict_allclose_match": torch.allclose(
                candidate, reference, atol=atol, rtol=rtol
            ),
        }
    )
    return metrics

@triton.jit
def _spmm_csr_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    n_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if row >= n_rows:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)

    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
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

    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


# Complex-path variant of the same Triton-native CSR base mapping.
@triton.jit
def _spmm_csr_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    b_ri_ptr,
    c_ri_ptr,
    n_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    stride_cr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if row >= n_rows:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    acc_re = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
    acc_im = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)

    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_re = tl.load(data_ri_ptr + idx * 2, mask=valid, other=0.0)
            a_im = tl.load(data_ri_ptr + idx * 2 + 1, mask=valid, other=0.0)
            a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
            b_re = tl.load(
                b_ri_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            b_im = tl.load(
                b_ri_ptr + a_col * stride_bk + offs_n * stride_bn + stride_br,
                mask=mask_n & valid,
                other=0.0,
            )
            acc_re = acc_re + a_re.to(ACC_DTYPE) * b_re.to(ACC_DTYPE) - a_im.to(ACC_DTYPE) * b_im.to(ACC_DTYPE)
            acc_im = acc_im + a_re.to(ACC_DTYPE) * b_im.to(ACC_DTYPE) + a_im.to(ACC_DTYPE) * b_re.to(ACC_DTYPE)

    tl.store(c_ri_ptr + row * stride_cm + offs_n * stride_cn, acc_re, mask=mask_n)
    tl.store(
        c_ri_ptr + row * stride_cm + offs_n * stride_cn + stride_cr,
        acc_im,
        mask=mask_n,
    )

def _prepare_spmm_csr_matrix(data, indices, indptr, shape):
    if len(shape) != 2:
        raise ValueError("shape must be a 2-tuple: (n_rows, n_cols)")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, and indptr must be 1D tensors")

    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError("shape dimensions must be non-negative")
    if indptr.numel() != n_rows + 1:
        raise ValueError(
            f"indptr length must be n_rows+1={n_rows + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")

    if not all(t.is_cuda for t in (data, indices, indptr)):
        raise ValueError("data, indices, and indptr must be CUDA tensors")
    if not all(t.device == data.device for t in (indices, indptr)):
        raise ValueError("data, indices, and indptr must be on the same CUDA device")
    if data.dtype not in SUPPORTED_SPMM_VALUE_DTYPES:
        raise TypeError(
            "data dtype must be one of: float16, bfloat16, float32, float64, complex64, complex128"
        )
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if indptr.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")

    nnz = data.numel()
    if indptr.numel() > 0 and int(indptr[0].item()) != 0:
        raise ValueError("indptr[0] must be 0")
    if indptr.numel() > 0 and int(indptr[-1].item()) != nnz:
        raise ValueError(f"indptr[-1] must equal nnz={nnz}, got {int(indptr[-1].item())}")
    if indptr.numel() > 1 and bool(torch.any(indptr[1:] < indptr[:-1]).item()):
        raise ValueError("indptr must be nondecreasing")
    if nnz > 0:
        min_col = int(indices.min().item())
        max_col = int(indices.max().item())
        if min_col < 0 or max_col >= n_cols:
            raise IndexError("indices out of range for n_cols")
        if max_col > _INDEX_LIMIT_INT32:
            raise ValueError(
                "column indices exceed the int32 range supported by the Triton kernel"
            )

    data = data.contiguous()
    indices = indices.contiguous()
    indptr = indptr.contiguous()

    kernel_indices = indices.to(torch.int32) if indices.dtype == torch.int64 else indices
    kernel_indptr = indptr.to(torch.int64)
    row_lengths = (
        kernel_indptr[1:] - kernel_indptr[:-1]
        if n_rows > 0
        else kernel_indptr.new_empty((0,))
    )
    max_row_nnz = int(row_lengths.max().item()) if n_rows > 0 else 0
    return data, kernel_indices, kernel_indptr, n_rows, n_cols, row_lengths, max_row_nnz


def _prepare_spmm_csr_inputs(data, indices, indptr, B, shape):
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")

    (
        data,
        kernel_indices,
        kernel_indptr,
        n_rows,
        n_cols,
        _row_lengths,
        _max_row_nnz,
    ) = _prepare_spmm_csr_matrix(data, indices, indptr, shape)
    if B.shape[0] != n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={n_cols}, got {B.shape[0]}")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != data.device:
        raise ValueError("B must be on the same CUDA device as the sparse matrix")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")

    B = B.contiguous()
    return data, kernel_indices, kernel_indptr, B, n_rows, n_cols, int(B.shape[1])


def _select_spmm_alg1_warp_and_factor(n_dense_cols):
    # Reuse the AlphaSparse CSR ALG1 dense-N heuristic without claiming
    # that the current Triton base path is a direct structural port.
    return _select_alpha_spmm_alg1_warp_and_factor(n_dense_cols, order_row=True)


def _resolve_spmm_alg1_launch_config(
    n_dense_cols,
    max_row_nnz,
    block_n=None,
    block_nnz=None,
    max_segments=None,
):
    warp_size, factor = _select_spmm_alg1_warp_and_factor(n_dense_cols)

    if block_n is None:
        block_n = warp_size * factor
    if block_nnz is None:
        block_nnz = warp_size

    if block_n <= 0 or block_nnz <= 0:
        raise ValueError("block_n and block_nnz must be positive when provided")
    if max_segments is not None and max_segments <= 0:
        raise ValueError("max_segments must be positive when provided")

    required_segments = triton.cdiv(max_row_nnz, block_nnz) if max_row_nnz > 0 else 0
    if max_segments is not None and required_segments > int(max_segments):
        raise ValueError(
            "row nnz requires more CSR segments than the explicit max_segments override allows: "
            f"required {required_segments}, provided {int(max_segments)}"
        )

    return {
        "block_n": int(block_n),
        "block_nnz": int(block_nnz),
        "max_segments": (None if max_segments is None else int(max_segments)),
        "required_segments": int(required_segments),
        "warp_size": int(warp_size),
        "factor": int(factor),
        "max_row_nnz": int(max_row_nnz),
        "auto_max_segments": max_segments is None,
    }


class PreparedCsrSpmmOpt:
    """Cached CSR metadata for repeated native SpMM-opt calls on the same sparse matrix."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "row_lengths",
        "max_row_nnz",
        "row_buckets",
        "supports_opt",
        "long_part_rows",
        "long_part_starts",
        "long_part_ends",
        "long_row_ids",
        "long_row_part_ptr",
        "long_row_fallback_only",
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
        row_buckets,
        long_part_rows,
        long_part_starts,
        long_part_ends,
        long_row_ids,
        long_row_part_ptr,
        long_row_fallback_only,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.row_lengths = row_lengths
        self.max_row_nnz = int(max_row_nnz)
        self.row_buckets = row_buckets
        self.supports_opt = data.dtype in (torch.float32, torch.float64)
        self.long_part_rows = long_part_rows
        self.long_part_starts = long_part_starts
        self.long_part_ends = long_part_ends
        self.long_row_ids = long_row_ids
        self.long_row_part_ptr = long_row_part_ptr
        self.long_row_fallback_only = bool(long_row_fallback_only)


_SPMM_OPT_LONG_ROW_THRESHOLD = 2048
_SPMM_OPT_SPLIT_BLOCK_NNZ = 256
_SPMM_OPT_ALG2_MID_AVG_NNZ_LIMIT = 16.0

_SPMM_OPT_BUCKET_SPECS_F32 = (
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
        "label": "seg_2048",
        "kind": "row2d_segmented",
        "min_row_nnz": 1025,
        "max_row_nnz": _SPMM_OPT_LONG_ROW_THRESHOLD,
        "batch_rows": 1,
        "block_k": 128,
        "block_n_cap": 128,
        "segments": 8,
    },
)

_SPMM_OPT_BUCKET_SPECS_F64 = (
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
        "label": "seg_2048",
        "kind": "row2d_segmented",
        "min_row_nnz": 1025,
        "max_row_nnz": _SPMM_OPT_LONG_ROW_THRESHOLD,
        "batch_rows": 1,
        "block_k": 64,
        "block_n_cap": 64,
        "segments": 8,
    },
)


def _spmm_opt_bucket_specs(dtype):
    return _SPMM_OPT_BUCKET_SPECS_F64 if dtype == torch.float64 else _SPMM_OPT_BUCKET_SPECS_F32


def _spmm_opt_make_bucket(
    label,
    kind,
    rows,
    execution,
    min_row_nnz,
    max_row_nnz,
    batch_rows=1,
    block_k=64,
    block_nnz=None,
    block_n_cap=64,
    segments=1,
):
    bucket = {
        "label": label,
        "kind": kind,
        "execution": execution,
        "rows": rows,
        "batch_rows": int(batch_rows),
        "block_k": int(block_k),
        "block_nnz": int(block_nnz if block_nnz is not None else block_k),
        "block_n_cap": int(block_n_cap),
        "min_row_nnz": int(min_row_nnz),
        "max_row_nnz": int(max_row_nnz),
    }
    if segments is not None:
        bucket["segments"] = int(segments)
    return bucket


def _select_spmm_opt_block_n(n_dense_cols):
    if n_dense_cols <= 8:
        return 8
    if n_dense_cols <= 16:
        return 16
    if n_dense_cols <= 32:
        return 32
    if n_dense_cols <= 64:
        return 64
    return 128


def _normalize_spmm_opt_device_props(device):
    props = torch.cuda.get_device_properties(device)
    warp_size = int(getattr(props, "warp_size", 32) or 32)
    max_threads_per_block = int(getattr(props, "max_threads_per_block", 1024) or 1024)
    max_threads_per_mp = int(
        getattr(props, "max_threads_per_multi_processor", 2048) or 2048
    )
    shared_memory_per_block = int(getattr(props, "shared_memory_per_block", 0) or 0)
    return {
        "warp_size": max(1, warp_size),
        "max_threads_per_block": max(32, max_threads_per_block),
        "max_threads_per_mp": max(32, max_threads_per_mp),
        "shared_memory_per_block": max(0, shared_memory_per_block),
    }


def _clip_spmm_opt_num_warps(desired, device_props):
    warp_size = int(device_props["warp_size"])
    max_by_block = max(1, int(device_props["max_threads_per_block"]) // warp_size)
    max_by_mp = max(1, int(device_props["max_threads_per_mp"]) // warp_size)
    max_supported = min(16, max_by_block, max_by_mp)
    supported = [value for value in (1, 2, 4, 8, 16) if value <= max_supported]
    if not supported:
        return 1
    desired = max(1, int(desired))
    for value in reversed(supported):
        if value <= desired:
            return value
    return supported[0]


def _resolve_spmm_opt_launch(kind, block_n, block_nnz, batch_rows, dtype, device_props):
    warp_size = int(device_props["warp_size"])
    dense_lane_need = max(1, triton.cdiv(int(block_n), warp_size))
    block_nnz = int(block_nnz)
    batch_rows = int(batch_rows)

    if kind == "batched":
        desired_warps = max(dense_lane_need, 1)
        if block_nnz >= 64 or int(block_n) >= 64:
            desired_warps = max(desired_warps, 2)
        if batch_rows >= 8 and block_n <= 32:
            desired_warps = min(desired_warps, 2)
    elif kind == "split_reduce":
        desired_warps = max(dense_lane_need, 2 if block_n <= 32 else 4)
    elif kind == "split_part":
        desired_warps = max(dense_lane_need, 4)
        if block_n >= 64:
            desired_warps = max(desired_warps, 8)
    else:
        desired_warps = max(dense_lane_need, 2)
        if block_nnz >= 128 or block_n >= 64:
            desired_warps = max(desired_warps, 4)

    if dtype == torch.float64:
        desired_warps = min(desired_warps, 8)
    num_warps = _clip_spmm_opt_num_warps(desired_warps, device_props)

    if dtype == torch.float64:
        num_stages = 1 if kind.startswith("split") or block_n >= 64 else 2
    elif kind == "batched":
        num_stages = 2
    elif kind == "vector":
        num_stages = 2 if block_n <= 64 else 1
    else:
        num_stages = 1 if num_warps >= 8 else 2
    return {"num_warps": int(num_warps), "num_stages": int(num_stages)}


def _select_spmm_opt_block_n_for_bucket(n_dense_cols, block_n_cap):
    return max(8, min(_select_spmm_opt_block_n(n_dense_cols), int(block_n_cap)))


def _select_spmm_opt_2d_num_warps(bucket, block_n, device_props, dtype):
    warp_size = max(1, int(device_props["warp_size"]))
    max_threads_per_block = max(32, int(device_props["max_threads_per_block"]))
    max_threads_per_mp = max(max_threads_per_block, int(device_props["max_threads_per_mp"]))
    dense_lane_need = max(1, triton.cdiv(int(block_n), warp_size))
    kind = bucket["kind"]
    block_k = int(bucket["block_k"])
    segments = int(bucket.get("segments", 1))
    batch_rows = int(bucket.get("batch_rows", 1))

    if kind == "batched2d":
        desired = max(dense_lane_need, 1 if block_k <= 16 else 2)
        if batch_rows >= 8 and block_n >= 64:
            desired = max(desired, 2)
    elif kind == "row2d":
        desired = max(dense_lane_need, 2 if block_k <= 32 else 4)
    else:
        desired = max(dense_lane_need, 4)
        if segments >= 4:
            desired = max(desired, 8 if block_k >= 64 else 4)
        if segments >= 8:
            desired = max(desired, 16 if dtype == torch.float32 and block_n >= 64 else 8)

    if dtype == torch.float64:
        desired = min(desired, 8)
    if max_threads_per_mp < 1536:
        desired = min(desired, 8)
    return _clip_spmm_opt_num_warps(desired, device_props)


def _select_spmm_opt_2d_num_stages(bucket, block_n, num_warps, device_props, dtype):
    kind = bucket["kind"]
    segments = int(bucket.get("segments", 1))
    shared_memory_per_block = int(device_props.get("shared_memory_per_block", 0))
    if dtype == torch.float64:
        stages = 1 if kind == "row2d_segmented" else (2 if block_n <= 32 and num_warps <= 4 else 1)
    elif kind == "batched2d":
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


def _resolve_spmm_opt_2d_launch(bucket, n_dense_cols, dtype, device_props):
    block_n = _select_spmm_opt_block_n_for_bucket(n_dense_cols, bucket["block_n_cap"])
    num_warps = _select_spmm_opt_2d_num_warps(bucket, block_n, device_props, dtype)
    num_stages = _select_spmm_opt_2d_num_stages(bucket, block_n, num_warps, device_props, dtype)
    return {
        "bucket_label": bucket.get("label"),
        "kind": bucket["kind"],
        "block_k": int(bucket["block_k"]),
        "block_n": int(block_n),
        "num_warps": int(num_warps),
        "num_stages": int(num_stages),
        "batch_rows": int(bucket.get("batch_rows", 1)),
        "segments": int(bucket.get("segments", 1)),
    }


def _build_spmm_opt_split_metadata(kernel_indptr, long_rows, part_block_nnz):
    device = kernel_indptr.device
    row_dtype = long_rows.dtype if long_rows.numel() > 0 else torch.int32
    if long_rows.numel() == 0:
        return (
            torch.empty((0,), dtype=row_dtype, device=device),
            torch.empty((0,), dtype=torch.int64, device=device),
            torch.empty((0,), dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
        )

    block = int(part_block_nnz)
    long_rows_i64 = long_rows.to(torch.int64)
    row_starts = kernel_indptr[long_rows_i64].to(torch.int64)
    row_ends = kernel_indptr[long_rows_i64 + 1].to(torch.int64)
    row_lengths = row_ends - row_starts
    part_counts = torch.div(row_lengths + block - 1, block, rounding_mode="floor")
    row_part_ptr = torch.empty((long_rows.numel() + 1,), dtype=torch.int64, device=device)
    row_part_ptr[0] = 0
    row_part_ptr[1:] = torch.cumsum(part_counts, dim=0)

    long_part_rows = torch.repeat_interleave(long_rows.to(row_dtype), part_counts)
    repeated_starts = torch.repeat_interleave(row_starts, part_counts)
    repeated_ends = torch.repeat_interleave(row_ends, part_counts)
    repeated_part_bases = torch.repeat_interleave(row_part_ptr[:-1], part_counts)
    part_ids = torch.arange(long_part_rows.numel(), dtype=torch.int64, device=device)
    in_row_part_offsets = part_ids - repeated_part_bases
    part_starts = repeated_starts + in_row_part_offsets * block
    part_ends = torch.minimum(part_starts + block, repeated_ends)
    return (
        long_part_rows,
        part_starts,
        part_ends,
        row_part_ptr,
    )


def _build_spmm_opt_buckets(row_lengths, dtype, nnz=None):
    device = row_lengths.device
    row_count = int(row_lengths.numel())
    row_index_dtype = torch.int32 if row_count <= _INDEX_LIMIT_INT32 else torch.int64
    all_rows = torch.arange(row_count, device=device, dtype=row_index_dtype)
    buckets = []

    normal_mask = row_lengths <= _SPMM_OPT_LONG_ROW_THRESHOLD
    if dtype == torch.float32:
        short_rows = all_rows[row_lengths <= 64]
        if short_rows.numel() > 0:
            buckets.append(
                _spmm_opt_make_bucket(
                    "short_64_alg2",
                    "batched2d",
                    short_rows,
                    "alg2_2d",
                    0,
                    64,
                    batch_rows=4,
                    block_k=32,
                    block_n_cap=64,
                )
            )
        avg_nnz = (float(nnz) / float(row_count)) if row_count > 0 and nnz is not None else 0.0
        use_mid_alg2 = avg_nnz <= _SPMM_OPT_ALG2_MID_AVG_NNZ_LIMIT
        if use_mid_alg2:
            mid_rows = all_rows[(row_lengths >= 65) & (row_lengths <= 256)]
            if mid_rows.numel() > 0:
                buckets.append(
                    _spmm_opt_make_bucket(
                        "row_256_alg2",
                        "row2d",
                        mid_rows,
                        "alg2_2d",
                        65,
                        256,
                        block_k=64,
                        block_n_cap=64,
                    )
                )
            legacy_mask = (row_lengths >= 257) & normal_mask
            legacy_min = 257
        else:
            legacy_mask = (row_lengths >= 65) & normal_mask
            legacy_min = 65
        legacy_rows = all_rows[legacy_mask]
        if legacy_rows.numel() > 0:
            buckets.append(
                _spmm_opt_make_bucket(
                    "legacy_regular",
                    "vector",
                    legacy_rows,
                    "legacy",
                    legacy_min,
                    _SPMM_OPT_LONG_ROW_THRESHOLD,
                    block_nnz=64,
                    block_n_cap=128,
                )
            )
    else:
        short_rows = all_rows[row_lengths <= 64]
        if short_rows.numel() > 0:
            buckets.append(
                _spmm_opt_make_bucket(
                    "legacy_short_64",
                    "batched",
                    short_rows,
                    "legacy",
                    0,
                    64,
                    batch_rows=4,
                    block_nnz=64,
                    block_n_cap=64,
                )
            )
        regular_rows = all_rows[(row_lengths >= 65) & normal_mask]
        if regular_rows.numel() > 0:
            buckets.append(
                _spmm_opt_make_bucket(
                    "legacy_regular",
                    "vector",
                    regular_rows,
                    "legacy",
                    65,
                    _SPMM_OPT_LONG_ROW_THRESHOLD,
                    block_nnz=64,
                    block_n_cap=128,
                )
            )
    long_rows = all_rows[row_lengths > _SPMM_OPT_LONG_ROW_THRESHOLD]
    if long_rows.numel() > 0:
        buckets.append(
            _spmm_opt_make_bucket(
                "split_long",
                "split",
                long_rows,
                "split",
                _SPMM_OPT_LONG_ROW_THRESHOLD + 1,
                _SPMM_OPT_LONG_ROW_THRESHOLD + 1,
                block_k=_SPMM_OPT_SPLIT_BLOCK_NNZ,
                block_nnz=_SPMM_OPT_SPLIT_BLOCK_NNZ,
                block_n_cap=128,
            )
        )
    return buckets, long_rows


def prepare_spmm_csr_opt(data, indices, indptr, shape):
    (
        data,
        kernel_indices,
        kernel_indptr,
        n_rows,
        n_cols,
        row_lengths,
        max_row_nnz,
    ) = _prepare_spmm_csr_matrix(data, indices, indptr, shape)
    row_index_dtype = torch.int32 if n_rows <= _INDEX_LIMIT_INT32 else torch.int64
    row_buckets = []
    long_rows = torch.empty((0,), dtype=row_index_dtype, device=data.device)
    long_part_rows = torch.empty((0,), dtype=row_index_dtype, device=data.device)
    long_part_starts = torch.empty((0,), dtype=torch.int64, device=data.device)
    long_part_ends = torch.empty((0,), dtype=torch.int64, device=data.device)
    long_row_part_ptr = torch.zeros(1, dtype=torch.int64, device=data.device)
    return PreparedCsrSpmmOpt(
        data=data,
        kernel_indices=kernel_indices,
        kernel_indptr=kernel_indptr,
        shape=shape,
        n_rows=n_rows,
        n_cols=n_cols,
        row_lengths=row_lengths,
        max_row_nnz=max_row_nnz,
        row_buckets=row_buckets,
        long_part_rows=long_part_rows,
        long_part_starts=long_part_starts,
        long_part_ends=long_part_ends,
        long_row_ids=long_rows,
        long_row_part_ptr=long_row_part_ptr,
        long_row_fallback_only=False,
    )


def _build_spmm_csr_opt_runtime_symbolic(prepared):
    row_buckets, long_rows = _build_spmm_opt_buckets(
        prepared.row_lengths,
        prepared.data.dtype,
        nnz=prepared.data.numel(),
    )
    long_part_rows = torch.empty((0,), dtype=long_rows.dtype, device=prepared.data.device)
    long_part_starts = torch.empty((0,), dtype=torch.int64, device=prepared.data.device)
    long_part_ends = torch.empty((0,), dtype=torch.int64, device=prepared.data.device)
    long_row_part_ptr = torch.zeros(1, dtype=torch.int64, device=prepared.data.device)
    if long_rows.numel() > 0:
        (
            long_part_rows,
            long_part_starts,
            long_part_ends,
            long_row_part_ptr,
        ) = _build_spmm_opt_split_metadata(
            prepared.kernel_indptr,
            long_rows,
            part_block_nnz=256,
        )
    return PreparedCsrSpmmOpt(
        data=prepared.data,
        kernel_indices=prepared.kernel_indices,
        kernel_indptr=prepared.kernel_indptr,
        shape=prepared.shape,
        n_rows=prepared.n_rows,
        n_cols=prepared.n_cols,
        row_lengths=prepared.row_lengths,
        max_row_nnz=prepared.max_row_nnz,
        row_buckets=row_buckets,
        long_part_rows=long_part_rows,
        long_part_starts=long_part_starts,
        long_part_ends=long_part_ends,
        long_row_ids=long_rows,
        long_row_part_ptr=long_row_part_ptr,
        long_row_fallback_only=False,
    )


def _triton_spmm_csr_impl(
    data,
    indices,
    indptr,
    B,
    n_rows,
    n_dense_cols,
    block_n,
    block_nnz,
):
    device = data.device
    dtype = data.dtype
    if n_rows == 0 or n_dense_cols == 0 or B.shape[0] == 0:
        return torch.zeros((n_rows, n_dense_cols), dtype=dtype, device=device)

    if not _is_complex_dtype(dtype):
        compute_dtype = dtype
        data_in = data
        B_in = B
        if dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
            data_in = data.to(torch.float32)
            B_in = B.to(torch.float32)
        elif dtype == torch.float32:
            compute_dtype = torch.float64
            data_in = data.to(torch.float64)
            B_in = B.to(torch.float64)

        C_compute = torch.empty((n_rows, n_dense_cols), dtype=compute_dtype, device=device)
        grid = (n_rows, triton.cdiv(n_dense_cols, block_n))
        acc_dtype = tl.float64 if compute_dtype == torch.float64 else tl.float32
        _spmm_csr_real_kernel[grid](
            data_in,
            indices,
            indptr,
            B_in,
            C_compute,
            n_rows,
            n_dense_cols,
            B_in.stride(0),
            B_in.stride(1),
            C_compute.stride(0),
            C_compute.stride(1),
            BLOCK_N=block_n,
            BLOCK_NNZ=block_nnz,
            ACC_DTYPE=acc_dtype,
        )
        if compute_dtype != dtype:
            C_compute = C_compute.to(dtype)
        return C_compute

    data_ri = torch.view_as_real(data).contiguous().reshape(-1)
    B_ri = torch.view_as_real(B).contiguous()
    C_ri = torch.empty((n_rows, n_dense_cols, 2), dtype=B_ri.dtype, device=device)
    grid = (n_rows, triton.cdiv(n_dense_cols, block_n))
    acc_dtype = tl.float64 if B_ri.dtype == torch.float64 else tl.float32
    _spmm_csr_complex_kernel[grid](
        data_ri,
        indices,
        indptr,
        B_ri,
        C_ri,
        n_rows,
        n_dense_cols,
        B_ri.stride(0),
        B_ri.stride(1),
        B_ri.stride(2),
        C_ri.stride(0),
        C_ri.stride(1),
        C_ri.stride(2),
        BLOCK_N=block_n,
        BLOCK_NNZ=block_nnz,
        ACC_DTYPE=acc_dtype,
    )
    return torch.view_as_complex(C_ri.contiguous())


@triton.jit
def _spmm_csr_batched_rows_f32_kernel(
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
    BLOCK_NNZ: tl.constexpr,
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
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = start + chunk_start + kk
                valid = active & (idx < end)
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                acc = acc + a_val * b_vals
        tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n & active)


@triton.jit
def _spmm_csr_batched_rows_f64_kernel(
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
    BLOCK_NNZ: tl.constexpr,
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
        acc = tl.zeros([BLOCK_N], dtype=tl.float64)
        for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = start + chunk_start + kk
                valid = active & (idx < end)
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                acc = acc + a_val * b_vals
        tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n & active)


@triton.jit
def _spmm_csr_vector_rows_f32_kernel(
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
    BLOCK_NNZ: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid_row)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            acc = acc + a_val * b_vals
    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_vector_rows_f64_kernel(
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
    BLOCK_NNZ: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid_row)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            acc = acc + a_val * b_vals
    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_opt_batched2d_kernel(
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
        tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc.to(OUT_DTYPE), mask=mask_n & active)


@triton.jit
def _spmm_csr_opt_row2d_kernel(
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
def _spmm_csr_opt_segmented2d_kernel(
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


@triton.jit
def _spmm_csr_split_part_f32_kernel(
    data_ptr,
    indices_ptr,
    b_ptr,
    workspace_ptr,
    part_starts_ptr,
    part_ends_ptr,
    n_parts,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_wm,
    stride_wn,
    BLOCK_N: tl.constexpr,
):
    pid_part = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_part >= n_parts:
        return
    start = tl.load(part_starts_ptr + pid_part)
    end = tl.load(part_ends_ptr + pid_part)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for idx in tl.range(start, end):
        a_val = tl.load(data_ptr + idx)
        a_col = tl.load(indices_ptr + idx)
        b_vals = tl.load(
            b_ptr + a_col * stride_bk + offs_n * stride_bn,
            mask=mask_n,
            other=0.0,
        )
        acc = acc + a_val * b_vals
    tl.store(workspace_ptr + pid_part * stride_wm + offs_n * stride_wn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_split_part_f64_kernel(
    data_ptr,
    indices_ptr,
    b_ptr,
    workspace_ptr,
    part_starts_ptr,
    part_ends_ptr,
    n_parts,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_wm,
    stride_wn,
    BLOCK_N: tl.constexpr,
):
    pid_part = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_part >= n_parts:
        return
    start = tl.load(part_starts_ptr + pid_part)
    end = tl.load(part_ends_ptr + pid_part)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    for idx in tl.range(start, end):
        a_val = tl.load(data_ptr + idx)
        a_col = tl.load(indices_ptr + idx)
        b_vals = tl.load(
            b_ptr + a_col * stride_bk + offs_n * stride_bn,
            mask=mask_n,
            other=0.0,
        )
        acc = acc + a_val * b_vals
    tl.store(workspace_ptr + pid_part * stride_wm + offs_n * stride_wn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_split_reduce_f32_kernel(
    workspace_ptr,
    out_ptr,
    long_rows_ptr,
    row_part_ptr_ptr,
    n_long_rows,
    n_dense_cols,
    stride_wm,
    stride_wn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_long_rows:
        return
    row = tl.load(long_rows_ptr + pid_row)
    part_start = tl.load(row_part_ptr_ptr + pid_row)
    part_end = tl.load(row_part_ptr_ptr + pid_row + 1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for rel_part in tl.range(0, part_end - part_start):
        part_id = part_start + rel_part
        vals = tl.load(
            workspace_ptr + part_id * stride_wm + offs_n * stride_wn,
            mask=mask_n,
            other=0.0,
        )
        acc = acc + vals
    tl.store(out_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_split_reduce_f64_kernel(
    workspace_ptr,
    out_ptr,
    long_rows_ptr,
    row_part_ptr_ptr,
    n_long_rows,
    n_dense_cols,
    stride_wm,
    stride_wn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_long_rows:
        return
    row = tl.load(long_rows_ptr + pid_row)
    part_start = tl.load(row_part_ptr_ptr + pid_row)
    part_end = tl.load(row_part_ptr_ptr + pid_row + 1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    for rel_part in tl.range(0, part_end - part_start):
        part_id = part_start + rel_part
        vals = tl.load(
            workspace_ptr + part_id * stride_wm + offs_n * stride_wn,
            mask=mask_n,
            other=0.0,
        )
        acc = acc + vals
    tl.store(out_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_stable_batched_f32_kernel(
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
    BLOCK_NNZ: tl.constexpr,
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
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        comp = tl.zeros([BLOCK_N], dtype=tl.float32)
        for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = start + chunk_start + kk
                valid = active & (idx < end)
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                prod = a_val.to(tl.float32) * b_vals.to(tl.float32)
                y = prod - comp
                t = acc + y
                comp = (t - acc) - y
                acc = t
        tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n & active)


@triton.jit
def _spmm_csr_stable_batched_f64_kernel(
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
    BLOCK_NNZ: tl.constexpr,
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
        acc = tl.zeros([BLOCK_N], dtype=tl.float64)
        comp = tl.zeros([BLOCK_N], dtype=tl.float64)
        for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = start + chunk_start + kk
                valid = active & (idx < end)
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                prod = a_val.to(tl.float64) * b_vals.to(tl.float64)
                y = prod - comp
                t = acc + y
                comp = (t - acc) - y
                acc = t
        tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n & active)


@triton.jit
def _spmm_csr_stable_vector_f32_kernel(
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
    BLOCK_NNZ: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid_row)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            acc = acc + a_val.to(tl.float64) * b_vals.to(tl.float64)
    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc.to(tl.float32), mask=mask_n)


@triton.jit
def _spmm_csr_stable_vector_f64_kernel(
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
    BLOCK_NNZ: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid_row)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            acc = acc + a_val.to(tl.float64) * b_vals.to(tl.float64)
    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_stable_split_part_f32_kernel(
    data_ptr,
    indices_ptr,
    b_ptr,
    workspace_ptr,
    part_starts_ptr,
    part_ends_ptr,
    n_parts,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_wm,
    stride_wn,
    BLOCK_N: tl.constexpr,
):
    pid_part = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_part >= n_parts:
        return
    start = tl.load(part_starts_ptr + pid_part)
    end = tl.load(part_ends_ptr + pid_part)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    for idx in tl.range(start, end):
        a_val = tl.load(data_ptr + idx)
        a_col = tl.load(indices_ptr + idx)
        b_vals = tl.load(
            b_ptr + a_col * stride_bk + offs_n * stride_bn,
            mask=mask_n,
            other=0.0,
        )
        acc = acc + a_val.to(tl.float64) * b_vals.to(tl.float64)
    tl.store(workspace_ptr + pid_part * stride_wm + offs_n * stride_wn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_stable_split_part_f64_kernel(
    data_ptr,
    indices_ptr,
    b_ptr,
    workspace_ptr,
    part_starts_ptr,
    part_ends_ptr,
    n_parts,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_wm,
    stride_wn,
    BLOCK_N: tl.constexpr,
):
    pid_part = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_part >= n_parts:
        return
    start = tl.load(part_starts_ptr + pid_part)
    end = tl.load(part_ends_ptr + pid_part)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    for idx in tl.range(start, end):
        a_val = tl.load(data_ptr + idx)
        a_col = tl.load(indices_ptr + idx)
        b_vals = tl.load(
            b_ptr + a_col * stride_bk + offs_n * stride_bn,
            mask=mask_n,
            other=0.0,
        )
        acc = acc + a_val.to(tl.float64) * b_vals.to(tl.float64)
    tl.store(workspace_ptr + pid_part * stride_wm + offs_n * stride_wn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_stable_split_reduce_f32_kernel(
    workspace_ptr,
    out_ptr,
    long_rows_ptr,
    row_part_ptr_ptr,
    n_long_rows,
    n_dense_cols,
    stride_wm,
    stride_wn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_long_rows:
        return
    row = tl.load(long_rows_ptr + pid_row)
    part_start = tl.load(row_part_ptr_ptr + pid_row)
    part_end = tl.load(row_part_ptr_ptr + pid_row + 1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    for rel_part in tl.range(0, part_end - part_start):
        part_id = part_start + rel_part
        vals = tl.load(
            workspace_ptr + part_id * stride_wm + offs_n * stride_wn,
            mask=mask_n,
            other=0.0,
        )
        acc = acc + vals.to(tl.float64)
    tl.store(out_ptr + row * stride_cm + offs_n * stride_cn, acc.to(tl.float32), mask=mask_n)


@triton.jit
def _spmm_csr_stable_split_reduce_f64_kernel(
    workspace_ptr,
    out_ptr,
    long_rows_ptr,
    row_part_ptr_ptr,
    n_long_rows,
    n_dense_cols,
    stride_wm,
    stride_wn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_long_rows:
        return
    row = tl.load(long_rows_ptr + pid_row)
    part_start = tl.load(row_part_ptr_ptr + pid_row)
    part_end = tl.load(row_part_ptr_ptr + pid_row + 1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    for rel_part in tl.range(0, part_end - part_start):
        part_id = part_start + rel_part
        vals = tl.load(
            workspace_ptr + part_id * stride_wm + offs_n * stride_wn,
            mask=mask_n,
            other=0.0,
        )
        acc = acc + vals.to(tl.float64)
    tl.store(out_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


def _run_spmm_opt_legacy_bucket(prepared, bucket, B, C_out, block_n, device_props):
    rows = bucket["rows"]
    if rows.numel() == 0:
        return
    dtype = prepared.data.dtype
    kind = bucket["kind"]
    block_nnz = int(bucket.get("block_nnz", bucket.get("block_k", 64)))
    batch_rows = int(bucket.get("batch_rows", 1))
    launch = _resolve_spmm_opt_launch(kind, block_n, block_nnz, batch_rows, dtype, device_props)
    if kind == "batched":
        kernel = (
            _spmm_csr_stable_batched_f64_kernel
            if dtype == torch.float64
            else _spmm_csr_stable_batched_f32_kernel
        )
        grid = (triton.cdiv(rows.numel(), batch_rows), triton.cdiv(B.shape[1], block_n))
        kernel[grid](
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
            BATCH=batch_rows,
            BLOCK_N=block_n,
            BLOCK_NNZ=block_nnz,
            num_warps=launch["num_warps"],
            num_stages=launch["num_stages"],
        )
        return

    kernel = (
        _spmm_csr_stable_vector_f64_kernel
        if dtype == torch.float64
        else _spmm_csr_stable_vector_f32_kernel
    )
    grid = (rows.numel(), triton.cdiv(B.shape[1], block_n))
    kernel[grid](
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
        BLOCK_N=block_n,
        BLOCK_NNZ=block_nnz,
        num_warps=launch["num_warps"],
        num_stages=launch["num_stages"],
    )


def _run_spmm_opt_bucket(prepared, bucket, B, C_out, block_n, device_props):
    rows = bucket["rows"]
    if rows.numel() == 0:
        return
    if bucket.get("execution") != "alg2_2d":
        _run_spmm_opt_legacy_bucket(prepared, bucket, B, C_out, block_n, device_props)
        return
    dtype = prepared.data.dtype
    launch = _resolve_spmm_opt_2d_launch(bucket, int(B.shape[1]), dtype, device_props)
    acc_dtype = tl.float64 if dtype == torch.float64 else tl.float32
    out_dtype = tl.float64 if dtype == torch.float64 else tl.float32
    common_kwargs = {
        "num_warps": launch["num_warps"],
        "num_stages": launch["num_stages"],
    }

    if bucket["kind"] == "batched2d":
        grid = (
            triton.cdiv(rows.numel(), launch["batch_rows"]),
            triton.cdiv(B.shape[1], launch["block_n"]),
        )
        _spmm_csr_opt_batched2d_kernel[grid](
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
        return

    if bucket["kind"] == "row2d":
        grid = (rows.numel(), triton.cdiv(B.shape[1], launch["block_n"]))
        _spmm_csr_opt_row2d_kernel[grid](
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
        return

    grid = (rows.numel(), triton.cdiv(B.shape[1], launch["block_n"]))
    _spmm_csr_opt_segmented2d_kernel[grid](
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


def _run_spmm_opt_split_bucket(prepared, B, C_out, block_n, device_props):
    if prepared.long_part_rows.numel() == 0:
        return False
    workspace = torch.empty(
        (prepared.long_part_rows.numel(), B.shape[1]),
        dtype=B.dtype,
        device=B.device,
    )
    split_kernel = (
        _spmm_csr_split_part_f64_kernel
        if B.dtype == torch.float64
        else _spmm_csr_split_part_f32_kernel
    )
    reduce_kernel = (
        _spmm_csr_split_reduce_f64_kernel
        if B.dtype == torch.float64
        else _spmm_csr_split_reduce_f32_kernel
    )
    split_launch = _resolve_spmm_opt_launch(
        "split_part", block_n, 256, 1, B.dtype, device_props
    )
    reduce_launch = _resolve_spmm_opt_launch(
        "split_reduce", block_n, 1, 1, B.dtype, device_props
    )
    split_grid = (
        prepared.long_part_rows.numel(),
        triton.cdiv(B.shape[1], block_n),
    )
    split_kernel[split_grid](
        prepared.data,
        prepared.kernel_indices,
        B,
        workspace,
        prepared.long_part_starts,
        prepared.long_part_ends,
        prepared.long_part_rows.numel(),
        B.shape[1],
        B.stride(0),
        B.stride(1),
        workspace.stride(0),
        workspace.stride(1),
        BLOCK_N=block_n,
        num_warps=split_launch["num_warps"],
        num_stages=split_launch["num_stages"],
    )
    reduce_grid = (
        prepared.long_row_ids.numel(),
        triton.cdiv(B.shape[1], block_n),
    )
    reduce_kernel[reduce_grid](
        workspace,
        C_out,
        prepared.long_row_ids,
        prepared.long_row_part_ptr,
        prepared.long_row_ids.numel(),
        B.shape[1],
        workspace.stride(0),
        workspace.stride(1),
        C_out.stride(0),
        C_out.stride(1),
        BLOCK_N=block_n,
        num_warps=reduce_launch["num_warps"],
        num_stages=reduce_launch["num_stages"],
    )
    return False


def _run_spmm_opt_bucket_stable(prepared, bucket, B, C_out, block_n):
    rows = bucket["rows"]
    if rows.numel() == 0:
        return
    dtype = prepared.data.dtype
    kernel_map = {
        ("vector", torch.float32): _spmm_csr_stable_vector_f32_kernel,
        ("vector", torch.float64): _spmm_csr_stable_vector_f64_kernel,
    }
    # Diagnose-only stable path: run every non-split opt bucket through the
    # row-per-program fp64-accum vector kernel. Runtime buckets now use
    # alg2-style 2D kinds, so keep this adapter independent of bucket naming.
    kind = "vector"

    grid = (rows.numel(), triton.cdiv(B.shape[1], block_n))
    kernel = kernel_map[(kind, dtype)]
    kernel[grid](
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
        BLOCK_N=block_n,
        BLOCK_NNZ=int(bucket.get("block_nnz", bucket.get("block_k", 64))),
    )


def _run_spmm_opt_split_bucket_stable(prepared, B, C_out, block_n):
    if prepared.long_part_rows.numel() == 0:
        return False
    workspace = torch.empty(
        (prepared.long_part_rows.numel(), B.shape[1]),
        dtype=torch.float64,
        device=B.device,
    )
    split_kernel = (
        _spmm_csr_stable_split_part_f64_kernel
        if B.dtype == torch.float64
        else _spmm_csr_stable_split_part_f32_kernel
    )
    reduce_kernel = (
        _spmm_csr_stable_split_reduce_f64_kernel
        if B.dtype == torch.float64
        else _spmm_csr_stable_split_reduce_f32_kernel
    )
    split_grid = (
        prepared.long_part_rows.numel(),
        triton.cdiv(B.shape[1], block_n),
    )
    split_kernel[split_grid](
        prepared.data,
        prepared.kernel_indices,
        B,
        workspace,
        prepared.long_part_starts,
        prepared.long_part_ends,
        prepared.long_part_rows.numel(),
        B.shape[1],
        B.stride(0),
        B.stride(1),
        workspace.stride(0),
        workspace.stride(1),
        BLOCK_N=block_n,
    )
    reduce_grid = (
        prepared.long_row_ids.numel(),
        triton.cdiv(B.shape[1], block_n),
    )
    reduce_kernel[reduce_grid](
        workspace,
        C_out,
        prepared.long_row_ids,
        prepared.long_row_part_ptr,
        prepared.long_row_ids.numel(),
        B.shape[1],
        workspace.stride(0),
        workspace.stride(1),
        C_out.stride(0),
        C_out.stride(1),
        BLOCK_N=block_n,
    )
    return False


_SPMM_OPT_CANDIDATE_BUCKET_SPECS_F32 = (
    {"label": "short_16", "kind": "micro", "min_row_nnz": 0, "max_row_nnz": 16, "block_nnz": 16},
    {"label": "short_32", "kind": "micro", "min_row_nnz": 17, "max_row_nnz": 32, "block_nnz": 32},
    {"label": "short_64", "kind": "segmented", "min_row_nnz": 33, "max_row_nnz": 64, "block_nnz": 64, "segments": 2},
    {"label": "short_128", "kind": "segmented", "min_row_nnz": 65, "max_row_nnz": 128, "block_nnz": 128, "segments": 4},
    {"label": "vector_192", "kind": "segmented", "min_row_nnz": 129, "max_row_nnz": 192, "block_nnz": 64, "segments": 2},
    {"label": "vector_256", "kind": "segmented", "min_row_nnz": 193, "max_row_nnz": 256, "block_nnz": 64, "segments": 4},
    {"label": "vector_512", "kind": "segmented", "min_row_nnz": 257, "max_row_nnz": 512, "block_nnz": 64, "segments": 4},
    {"label": "vector_1024", "kind": "segmented", "min_row_nnz": 513, "max_row_nnz": 1024, "block_nnz": 32, "segments": 8},
    {"label": "vector_2048", "kind": "segmented", "min_row_nnz": 1025, "max_row_nnz": 2048, "block_nnz": 32, "segments": 8},
)

_SPMM_OPT_CANDIDATE_BUCKET_SPECS_F64 = (
    {"label": "short_16", "kind": "segmented", "min_row_nnz": 0, "max_row_nnz": 16, "block_nnz": 16, "segments": 2},
    {"label": "short_32", "kind": "segmented", "min_row_nnz": 17, "max_row_nnz": 32, "block_nnz": 32, "segments": 2},
    {"label": "short_64", "kind": "segmented", "min_row_nnz": 33, "max_row_nnz": 64, "block_nnz": 64, "segments": 2},
    {"label": "short_128", "kind": "segmented", "min_row_nnz": 65, "max_row_nnz": 128, "block_nnz": 128, "segments": 4},
    {"label": "vector_256", "kind": "segmented", "min_row_nnz": 129, "max_row_nnz": 256, "block_nnz": 64, "segments": 4},
    {"label": "vector_512", "kind": "segmented", "min_row_nnz": 257, "max_row_nnz": 512, "block_nnz": 64, "segments": 4},
    {"label": "vector_1024", "kind": "segmented", "min_row_nnz": 513, "max_row_nnz": 1024, "block_nnz": 32, "segments": 8},
    {"label": "vector_2048", "kind": "segmented", "min_row_nnz": 1025, "max_row_nnz": 2048, "block_nnz": 32, "segments": 8},
)

_SPMM_OPT_CANDIDATE_SPLIT_THRESHOLD_F32 = 2048
_SPMM_OPT_CANDIDATE_SPLIT_THRESHOLD_F64 = 2048


def _select_spmm_opt_candidate_specs(dtype):
    return (
        _SPMM_OPT_CANDIDATE_BUCKET_SPECS_F64
        if dtype == torch.float64
        else _SPMM_OPT_CANDIDATE_BUCKET_SPECS_F32
    )


def _select_spmm_opt_candidate_split_threshold(dtype):
    return (
        _SPMM_OPT_CANDIDATE_SPLIT_THRESHOLD_F64
        if dtype == torch.float64
        else _SPMM_OPT_CANDIDATE_SPLIT_THRESHOLD_F32
    )


@triton.jit
def _spmm_csr_candidate_short_micro_kernel(
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
    BLOCK_NNZ: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid_row)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    offs_k = start + tl.arange(0, BLOCK_NNZ)
    mask_k = offs_k < end
    a_vals = tl.load(data_ptr + offs_k, mask=mask_k, other=0.0).to(ACC_DTYPE)
    cols = tl.load(indices_ptr + offs_k, mask=mask_k, other=0)
    b_ptrs = b_ptr + cols[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b_mask = mask_k[:, None] & mask_n[None, :]
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0).to(ACC_DTYPE)
    prod = a_vals[:, None] * b_vals
    acc = tl.sum(prod, axis=0)
    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_candidate_stable_short_f32_kernel(
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
    BLOCK_NNZ: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid_row)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    comp = tl.zeros([BLOCK_N], dtype=tl.float64)
    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            prod = a_val.to(tl.float64) * b_vals.to(tl.float64)
            y = prod - comp
            t = acc + y
            comp = (t - acc) - y
            acc = t
    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc.to(tl.float32), mask=mask_n)


@triton.jit
def _spmm_csr_candidate_stable_short_f64_kernel(
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
    BLOCK_NNZ: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid_row)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    acc = tl.zeros([BLOCK_N], dtype=tl.float64)
    comp = tl.zeros([BLOCK_N], dtype=tl.float64)
    for chunk_start in tl.range(0, row_nnz, BLOCK_NNZ):
        for kk in tl.static_range(0, BLOCK_NNZ):
            idx = start + chunk_start + kk
            valid = idx < end
            a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
            a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
            b_vals = tl.load(
                b_ptr + a_col * stride_bk + offs_n * stride_bn,
                mask=mask_n & valid,
                other=0.0,
            )
            prod = a_val.to(tl.float64) * b_vals.to(tl.float64)
            y = prod - comp
            t = acc + y
            comp = (t - acc) - y
            acc = t
    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


@triton.jit
def _spmm_csr_candidate_segmented_rows_kernel(
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
    BLOCK_NNZ: tl.constexpr,
    SEGMENTS: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_row >= n_bucket_rows:
        return
    row = tl.load(rows_ptr + pid_row)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_dense_cols
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    row_nnz = end - start
    seg_span = (row_nnz + SEGMENTS - 1) // SEGMENTS
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
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_NNZ):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val * b_vals
            acc0 = acc0 + chunk_acc

    if SEGMENTS > 1:
        seg_start = start + seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_NNZ):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val * b_vals
            acc1 = acc1 + chunk_acc

    if SEGMENTS > 2:
        seg_start = start + 2 * seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_NNZ):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val * b_vals
            acc2 = acc2 + chunk_acc

    if SEGMENTS > 3:
        seg_start = start + 3 * seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_NNZ):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val * b_vals
            acc3 = acc3 + chunk_acc

    if SEGMENTS > 4:
        seg_start = start + 4 * seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_NNZ):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val * b_vals
            acc4 = acc4 + chunk_acc

    if SEGMENTS > 5:
        seg_start = start + 5 * seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_NNZ):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val * b_vals
            acc5 = acc5 + chunk_acc

    if SEGMENTS > 6:
        seg_start = start + 6 * seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_NNZ):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val * b_vals
            acc6 = acc6 + chunk_acc

    if SEGMENTS > 7:
        seg_start = start + 7 * seg_span
        seg_end = tl.minimum(end, seg_start + seg_span)
        for chunk_offset in tl.range(0, seg_end - seg_start, BLOCK_NNZ):
            chunk_acc = tl.zeros([BLOCK_N], dtype=ACC_DTYPE)
            for kk in tl.static_range(0, BLOCK_NNZ):
                idx = seg_start + chunk_offset + kk
                valid = idx < seg_end
                a_val = tl.load(data_ptr + idx, mask=valid, other=0.0)
                a_col = tl.load(indices_ptr + idx, mask=valid, other=0)
                b_vals = tl.load(
                    b_ptr + a_col * stride_bk + offs_n * stride_bn,
                    mask=mask_n & valid,
                    other=0.0,
                )
                chunk_acc = chunk_acc + a_val * b_vals
            acc7 = acc7 + chunk_acc

    if SEGMENTS == 1:
        acc = acc0
    elif SEGMENTS == 2:
        acc = acc0 + acc1
    elif SEGMENTS <= 4:
        acc_left = acc0 + acc1
        acc_right = acc2 + acc3
        acc = acc_left + acc_right
    else:
        acc01 = acc0 + acc1
        acc23 = acc2 + acc3
        acc45 = acc4 + acc5
        acc67 = acc6 + acc7
        acc = (acc01 + acc23) + (acc45 + acc67)
    tl.store(c_ptr + row * stride_cm + offs_n * stride_cn, acc, mask=mask_n)


def _build_spmm_opt_candidate_buckets(prepared):
    row_lengths = prepared.row_lengths
    device = row_lengths.device
    row_count = int(row_lengths.numel())
    max_row_index = row_count - 1
    row_index_dtype = torch.int32 if max_row_index <= _INDEX_LIMIT_INT32 else torch.int64
    all_rows = torch.arange(row_count, device=device, dtype=row_index_dtype)
    buckets = []
    for spec in _select_spmm_opt_candidate_specs(prepared.data.dtype):
        lower = int(spec["min_row_nnz"])
        upper = int(spec["max_row_nnz"])
        if lower <= 0:
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
            "block_nnz": int(spec["block_nnz"]),
            "min_row_nnz": lower,
            "max_row_nnz": upper,
        }
        if "segments" in spec:
            bucket["segments"] = int(spec["segments"])
        buckets.append(bucket)

    split_rows = all_rows[row_lengths > _select_spmm_opt_candidate_split_threshold(prepared.data.dtype)]
    if split_rows.numel() > 0:
        buckets.append({"label": "split", "kind": "split", "rows": split_rows, "block_nnz": 256})
    return buckets


def _candidate_bucket_uses_stable_short_kernel(bucket):
    if bucket["kind"] == "split":
        return False
    return int(bucket.get("max_row_nnz", 0)) <= 256


def _run_spmm_opt_bucket_candidate(prepared, bucket, B, C_out, block_n):
    rows = bucket["rows"]
    if rows.numel() == 0:
        return
    dtype = prepared.data.dtype
    if _candidate_bucket_uses_stable_short_kernel(bucket):
        grid = (rows.numel(), triton.cdiv(B.shape[1], block_n))
        kernel = (
            _spmm_csr_candidate_stable_short_f64_kernel
            if dtype == torch.float64
            else _spmm_csr_candidate_stable_short_f32_kernel
        )
        kernel[grid](
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
            BLOCK_N=block_n,
            BLOCK_NNZ=bucket["block_nnz"],
        )
        return
    acc_dtype = tl.float64 if dtype == torch.float64 else tl.float32
    grid = (rows.numel(), triton.cdiv(B.shape[1], block_n))
    if bucket["kind"] == "micro":
        _spmm_csr_candidate_short_micro_kernel[grid](
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
            BLOCK_N=block_n,
            BLOCK_NNZ=bucket["block_nnz"],
            ACC_DTYPE=acc_dtype,
        )
        return
    _spmm_csr_candidate_segmented_rows_kernel[grid](
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
        BLOCK_N=block_n,
        BLOCK_NNZ=bucket["block_nnz"],
        SEGMENTS=bucket["segments"],
        ACC_DTYPE=acc_dtype,
    )


def _run_spmm_opt_split_bucket_candidate(prepared, split_rows, B, C_out, block_n):
    if split_rows.numel() == 0:
        return False
    (
        long_part_rows,
        long_part_starts,
        long_part_ends,
        long_row_part_ptr,
    ) = _build_spmm_opt_split_metadata(
        prepared.kernel_indptr,
        split_rows.to(prepared.kernel_indptr.device),
        part_block_nnz=256,
    )
    if long_part_rows.numel() == 0:
        return False
    workspace = torch.empty((long_part_rows.numel(), B.shape[1]), dtype=B.dtype, device=B.device)
    split_kernel = _spmm_csr_split_part_f64_kernel if B.dtype == torch.float64 else _spmm_csr_split_part_f32_kernel
    reduce_kernel = _spmm_csr_split_reduce_f64_kernel if B.dtype == torch.float64 else _spmm_csr_split_reduce_f32_kernel
    split_grid = (long_part_rows.numel(), triton.cdiv(B.shape[1], block_n))
    split_kernel[split_grid](
        prepared.data,
        prepared.kernel_indices,
        B,
        workspace,
        long_part_starts,
        long_part_ends,
        long_part_rows.numel(),
        B.shape[1],
        B.stride(0),
        B.stride(1),
        workspace.stride(0),
        workspace.stride(1),
        BLOCK_N=block_n,
    )
    reduce_grid = (split_rows.numel(), triton.cdiv(B.shape[1], block_n))
    reduce_kernel[reduce_grid](
        workspace,
        C_out,
        split_rows,
        long_row_part_ptr,
        split_rows.numel(),
        B.shape[1],
        workspace.stride(0),
        workspace.stride(1),
        C_out.stride(0),
        C_out.stride(1),
        BLOCK_N=block_n,
    )
    return False


def _triton_spmm_csr_impl_opt_prepared_candidate(prepared, B):
    if not prepared.supports_opt:
        raise TypeError("spmm opt only supports float32 and float64")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != prepared.data.device:
        raise ValueError("B must be on the same CUDA device as the sparse matrix")
    if B.dtype != prepared.data.dtype:
        raise TypeError("B dtype must match sparse matrix dtype")
    if B.shape[0] != prepared.n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={prepared.n_cols}, got {B.shape[0]}")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")
    B = B.contiguous()
    block_n = _select_spmm_opt_block_n(int(B.shape[1]))
    C_out = torch.zeros((prepared.n_rows, int(B.shape[1])), dtype=B.dtype, device=B.device)
    long_row_fallback_used = False
    for bucket in _build_spmm_opt_candidate_buckets(prepared):
        if bucket["kind"] == "split":
            long_row_fallback_used = _run_spmm_opt_split_bucket_candidate(
                prepared,
                bucket["rows"],
                B,
                C_out,
                block_n,
            )
            continue
        _run_spmm_opt_bucket_candidate(prepared, bucket, B, C_out, block_n)
    return C_out, long_row_fallback_used


def _flagsparse_spmm_csr_opt_candidate_for_diagnose(prepared, B):
    C_out, _ = _triton_spmm_csr_impl_opt_prepared_candidate(prepared, B)
    return C_out


def _triton_spmm_csr_impl_opt_prepared(prepared, B):
    if not prepared.supports_opt:
        raise TypeError("spmm opt only supports float32 and float64")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != prepared.data.device:
        raise ValueError("B must be on the same CUDA device as the sparse matrix")
    if B.dtype != prepared.data.dtype:
        raise TypeError("B dtype must match sparse matrix dtype")
    if B.shape[0] != prepared.n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={prepared.n_cols}, got {B.shape[0]}")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")
    B = B.contiguous()
    block_n = _select_spmm_opt_block_n(int(B.shape[1]))
    C_out = torch.zeros((prepared.n_rows, int(B.shape[1])), dtype=B.dtype, device=B.device)
    long_row_fallback_used = False
    device_props = _normalize_spmm_opt_device_props(prepared.data.device)
    for bucket in prepared.row_buckets:
        if bucket["kind"] == "split":
            long_row_fallback_used = _run_spmm_opt_split_bucket(
                prepared, B, C_out, block_n, device_props
            )
            continue
        _run_spmm_opt_bucket(prepared, bucket, B, C_out, block_n, device_props)
    return C_out, long_row_fallback_used


def _triton_spmm_csr_impl_opt_prepared_stable(prepared, B):
    if not prepared.supports_opt:
        raise TypeError("spmm opt only supports float32 and float64")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != prepared.data.device:
        raise ValueError("B must be on the same CUDA device as the sparse matrix")
    if B.dtype != prepared.data.dtype:
        raise TypeError("B dtype must match sparse matrix dtype")
    if B.shape[0] != prepared.n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={prepared.n_cols}, got {B.shape[0]}")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")
    B = B.contiguous()
    block_n = _select_spmm_opt_block_n(int(B.shape[1]))
    C_out = torch.zeros((prepared.n_rows, int(B.shape[1])), dtype=B.dtype, device=B.device)
    long_row_fallback_used = False
    for bucket in prepared.row_buckets:
        if bucket["kind"] == "split":
            long_row_fallback_used = _run_spmm_opt_split_bucket_stable(prepared, B, C_out, block_n)
            continue
        _run_spmm_opt_bucket_stable(prepared, bucket, B, C_out, block_n)
    return C_out, long_row_fallback_used


def _flagsparse_spmm_csr_opt_stable_for_diagnose(prepared, B):
    C_out, _ = _triton_spmm_csr_impl_opt_prepared_stable(prepared, B)
    return C_out

def flagsparse_spmm_csr(
    data,
    indices,
    indptr,
    B,
    shape,
    block_n=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    return_time=False,
):
    """CSR SpMM: C = A @ B using Triton.

    A is provided as CSR arrays; B is a dense CUDA tensor with shape (n_cols, n_dense_cols).
    This is the current Triton-native CSR base path for the row-major,
    non-transpose subset. It borrows the dense-N launch heuristic from
    AlphaSparse CSR ALG1 (`csrspmm_rb_sr`) but is not a direct structural
    port of the CUDA kernel.
    """
    if block_n is not None and block_n <= 0:
        raise ValueError("block_n must be positive when provided")
    if block_nnz is not None and block_nnz <= 0:
        raise ValueError("block_nnz must be positive when provided")
    if max_segments is not None and max_segments <= 0:
        raise ValueError("max_segments must be positive when provided")

    data, kernel_indices, kernel_indptr, B, n_rows, _, n_dense_cols = _prepare_spmm_csr_inputs(
        data, indices, indptr, B, shape
    )
    max_row_nnz = (
        int(torch.max(kernel_indptr[1:] - kernel_indptr[:-1]).item())
        if n_rows > 0
        else 0
    )
    launch = _resolve_spmm_alg1_launch_config(
        n_dense_cols,
        max_row_nnz,
        block_n=block_n,
        block_nnz=block_nnz,
        max_segments=max_segments,
    )

    if out is not None:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.device != data.device:
            raise ValueError("out must be on the same CUDA device as the inputs")
        if out.shape != (n_rows, n_dense_cols) or out.dtype != data.dtype:
            raise ValueError("out shape/dtype must match result")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C = _triton_spmm_csr_impl(
        data,
        kernel_indices,
        kernel_indptr,
        B,
        n_rows,
        n_dense_cols,
        block_n=launch["block_n"],
        block_nnz=launch["block_nnz"],
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if out is not None:
        out.copy_(C)
        C = out
    if return_time:
        return C, elapsed_ms
    return C


def flagsparse_spmm_csr_opt(
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
    """CSR SpMM-opt: native float32/float64 bucketed path for CSR @ dense."""
    if prepared is not None and not isinstance(prepared, PreparedCsrSpmmOpt):
        raise TypeError("prepared must be a PreparedCsrSpmmOpt instance")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_spmm_csr_opt(data, indices, indptr, shape)
    elif shape is not None:
        resolved_shape = (int(shape[0]), int(shape[1]))
        if resolved_shape != prepared.shape:
            raise ValueError(
                f"shape {resolved_shape} does not match prepared.shape {prepared.shape}"
            )
    if B is None:
        raise ValueError("B is required")
    if not prepared.supports_opt:
        raise TypeError("flagsparse_spmm_csr_opt only supports float32 and float64")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != prepared.data.device:
        raise ValueError("B must be on the same CUDA device as sparse matrix data")
    if out is not None:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.device != prepared.data.device:
            raise ValueError("out must be on the same CUDA device as sparse matrix data")
        if out.shape != (prepared.n_rows, int(B.shape[1])) or out.dtype != prepared.data.dtype:
            raise ValueError("out shape/dtype must match result")
    do_timing = bool(return_time or return_meta)
    symbolic_ms = None
    compute_ms = None
    op_total_ms = None
    if do_timing:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    runtime_prepared = _build_spmm_csr_opt_runtime_symbolic(prepared)
    if do_timing:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        symbolic_ms = (t1 - t0) * 1000.0
    C, _long_row_fallback_used = _triton_spmm_csr_impl_opt_prepared(runtime_prepared, B)
    if do_timing:
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        compute_ms = (t2 - t1) * 1000.0
        op_total_ms = symbolic_ms + compute_ms
    if out is not None:
        out.copy_(C)
        C = out
    if return_meta:
        meta = {
            "symbolic_ms": symbolic_ms,
            "compute_ms": compute_ms,
            "op_total_ms": op_total_ms,
            "long_part_count": int(runtime_prepared.long_part_rows.numel()),
            "long_row_count": int(runtime_prepared.long_row_ids.numel()),
            "bucket_count": int(len(runtime_prepared.row_buckets)),
        }
        if return_time:
            return C, op_total_ms, meta
        return C, meta
    if return_time:
        return C, op_total_ms
    return C


def _spmm_opt_reference_error(candidate, reference, value_dtype):
    atol, rtol = _spmm_coo_reference_tolerance(value_dtype)
    if candidate.numel() == 0:
        return 0.0
    diff = torch.abs(candidate - reference)
    denom = atol + rtol * torch.abs(reference)
    return float(torch.max(diff / denom).item())


def benchmark_spmm_opt_case(
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
    """Benchmark SpMM base vs opt against the same high-precision PyTorch reference."""
    if value_dtype not in (torch.float32, torch.float64):
        raise TypeError("benchmark_spmm_opt_case only supports float32 and float64")
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_cols, nnz, value_dtype, index_dtype, device
    )
    B = _build_random_dense((n_cols, n_dense_cols), value_dtype, device)
    shape = (n_rows, n_cols)
    prepared = prepare_spmm_csr_opt(data, indices, indptr, shape)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    base_values = flagsparse_spmm_csr(data, indices, indptr, B, shape)
    torch.cuda.synchronize()
    base_first_call_ms = (time.perf_counter() - t0) * 1000.0
    base_values, base_ms = _benchmark_cuda_op(
        lambda: flagsparse_spmm_csr(data, indices, indptr, B, shape),
        warmup=warmup,
        iters=iters,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    opt_values = flagsparse_spmm_csr_opt(B=B, prepared=prepared)
    torch.cuda.synchronize()
    opt_first_call_ms = (time.perf_counter() - t0) * 1000.0
    opt_values, opt_ms = _benchmark_cuda_op(
        lambda: flagsparse_spmm_csr_opt(B=B, prepared=prepared),
        warmup=warmup,
        iters=iters,
    )

    indptr64 = indptr.to(torch.int64)
    indices64 = indices.to(torch.int64)
    csr_ref = torch.sparse_csr_tensor(
        indptr64,
        indices64,
        data.to(torch.float64 if value_dtype == torch.float32 else value_dtype),
        size=shape,
        device=device,
    )
    ref = torch.sparse.mm(
        csr_ref,
        B.to(torch.float64 if value_dtype == torch.float32 else value_dtype),
    ).to(value_dtype)

    pt_ms = None
    try:
        pt_sparse = torch.sparse_csr_tensor(
            indptr64,
            indices64,
            data,
            size=shape,
            device=device,
        )
        pt_op = lambda: torch.sparse.mm(pt_sparse, B)
        _, pt_ms = _benchmark_cuda_op(pt_op, warmup=warmup, iters=iters)
    except Exception:
        pt_ms = None

    cu_ms = None
    if run_cusparse and cp is not None and cpx_sparse is not None:
        try:
            data_cp = _cupy_from_torch(data)
            indices_cp = _cupy_from_torch(indices.to(torch.int64))
            indptr_cp = _cupy_from_torch(indptr)
            B_cp = _cupy_from_torch(B)
            A_csr = cpx_sparse.csr_matrix((data_cp, indices_cp, indptr_cp), shape=shape)
            _, cu_ms = _benchmark_cuda_op(lambda: A_csr @ B_cp, warmup=warmup, iters=iters)
        except Exception:
            cu_ms = None

    err_base = _spmm_opt_reference_error(base_values, ref, value_dtype)
    err_opt = _spmm_opt_reference_error(opt_values, ref, value_dtype)
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
            "base_first_call_ms": base_first_call_ms,
            "opt_ms": opt_ms,
            "opt_first_call_ms": opt_first_call_ms,
            "pt_ms": pt_ms,
            "cu_ms": cu_ms,
            "opt_vs_base": (base_ms / opt_ms if opt_ms and opt_ms > 0 else None),
            "opt_vs_pt": (pt_ms / opt_ms if pt_ms is not None and opt_ms > 0 else None),
            "opt_vs_cu": (cu_ms / opt_ms if cu_ms is not None and opt_ms > 0 else None),
        },
        "verification": {
            "err_base": err_base,
            "err_opt": err_opt,
            "base_ok": err_base <= 1.0,
            "opt_ok": err_opt <= 1.0,
            "status": ("PASS" if err_opt <= 1.0 else "FAIL"),
        },
        "backend_status": {
            "long_row_fallback_used": bool(prepared.long_row_fallback_only),
            "flagsparse_internal_route": "csr-opt-bucketed",
        },
        "samples": {
            "base": base_values,
            "opt": opt_values,
            "reference": ref,
        },
    }


def benchmark_spmm_case(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    n_dense_cols=32,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_n=None,
    block_nnz=None,
    max_segments=None,
    run_cusparse=True,
):
    """Benchmark Triton CSR SpMM vs PyTorch sparse.mm and CuPy/cuSPARSE CSR @ dense."""
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_cols, nnz, value_dtype, index_dtype, device
    )
    B = _build_random_dense((n_cols, n_dense_cols), value_dtype, device)
    shape = (n_rows, n_cols)
    max_row_nnz = int(torch.max(indptr[1:] - indptr[:-1]).item()) if n_rows > 0 else 0
    launch = _resolve_spmm_alg1_launch_config(
        n_dense_cols,
        max_row_nnz,
        block_n=block_n,
        block_nnz=block_nnz,
        max_segments=max_segments,
    )

    triton_kwargs = {
        "data": data,
        "indices": indices,
        "indptr": indptr,
        "B": B,
        "shape": shape,
        "block_n": launch["block_n"],
        "block_nnz": launch["block_nnz"],
        "max_segments": launch["max_segments"],
        "return_time": False,
    }

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = flagsparse_spmm_csr(**triton_kwargs)
    torch.cuda.synchronize()
    triton_first_call_ms = (time.perf_counter() - t0) * 1000.0
    triton_C, triton_ms = _benchmark_cuda_op(
        lambda: flagsparse_spmm_csr(**triton_kwargs),
        warmup=warmup,
        iters=iters,
    )

    indptr64 = indptr.to(torch.int64)
    indices64 = indices.to(torch.int64)
    row_indices = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )

    pytorch_reason = None
    pytorch_values = None
    pytorch_ms = None
    pytorch_format = "CSR"
    try:
        csr_pt = torch.sparse_csr_tensor(indptr64, indices64, data, size=shape, device=device)
        pytorch_op = lambda: torch.sparse.mm(csr_pt, B)
        if value_dtype in (torch.float16, torch.bfloat16):
            csr_ref = torch.sparse_csr_tensor(indptr64, indices64, data.to(torch.float32), size=shape, device=device)
            expected = torch.sparse.mm(csr_ref, B.to(torch.float32)).to(value_dtype)
        elif value_dtype == torch.float32:
            csr_ref = torch.sparse_csr_tensor(indptr64, indices64, data.to(torch.float64), size=shape, device=device)
            expected = torch.sparse.mm(csr_ref, B.to(torch.float64)).to(value_dtype)
        elif value_dtype == torch.complex64:
            csr_ref = torch.sparse_csr_tensor(indptr64, indices64, data.to(torch.complex128), size=shape, device=device)
            expected = torch.sparse.mm(csr_ref, B.to(torch.complex128)).to(value_dtype)
        else:
            expected = torch.sparse.mm(csr_pt, B)
    except Exception as exc:
        pytorch_format = "COO"
        pytorch_reason = f"CSR fallback: {exc}"
        coo = torch.sparse_coo_tensor(
            torch.stack([row_indices, indices64]),
            data,
            shape,
            device=device,
        ).coalesce()
        pytorch_op = lambda: torch.sparse.mm(coo, B)
        if value_dtype in (torch.float16, torch.bfloat16):
            expected = torch.sparse.mm(coo.to(torch.float32), B.to(torch.float32)).to(value_dtype)
        elif value_dtype == torch.float32:
            expected = torch.sparse.mm(coo.to(torch.float64), B.to(torch.float64)).to(value_dtype)
        elif value_dtype == torch.complex64:
            expected = torch.sparse.mm(coo.to(torch.complex128), B.to(torch.complex128)).to(value_dtype)
        else:
            expected = torch.sparse.mm(coo, B)

    pytorch_values = expected
    try:
        pytorch_values, pytorch_ms = _benchmark_cuda_op(
            pytorch_op, warmup=warmup, iters=iters
        )
    except Exception as exc:
        pytorch_reason = str(exc) if pytorch_reason is None else f"{pytorch_reason}; timing: {exc}"

    triton_metrics = _spmm_validation_metrics(triton_C, expected)
    triton_match = triton_metrics["strict_allclose_match"]

    cusparse_ms = None
    cusparse_match = None
    cusparse_reason = None
    cusparse_values = None
    cusparse_metrics = None
    _cupy_supported_dtypes = (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    if run_cusparse:
        if cp is None or cpx_sparse is None:
            cusparse_reason = "CuPy/cuSPARSE is not available"
        elif value_dtype not in _cupy_supported_dtypes:
            cusparse_reason = "float16/bfloat16 not supported by CuPy sparse; skipped"
        else:
            try:
                data_cp = _cupy_from_torch(data)
                indices_cp = _cupy_from_torch(indices.to(torch.int64))
                indptr_cp = _cupy_from_torch(indptr)
                B_cp = _cupy_from_torch(B)
                A_csr = cpx_sparse.csr_matrix(
                    (data_cp, indices_cp, indptr_cp), shape=shape
                )
                cusparse_values_cp, cusparse_ms = _benchmark_cuda_op(
                    lambda: A_csr @ B_cp, warmup=warmup, iters=iters
                )
                cusparse_values = _torch_from_cupy(cusparse_values_cp)
                cusparse_metrics = _spmm_validation_metrics(cusparse_values, expected)
                cusparse_match = cusparse_metrics["strict_allclose_match"]
            except Exception as exc:
                cusparse_reason = str(exc)

    triton_speedup_vs_pytorch = (
        pytorch_ms / triton_ms if (pytorch_ms is not None and triton_ms > 0) else None
    )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms if (cusparse_ms is not None and triton_ms > 0) else None
    )
    threshold = _spmm_relative_threshold(value_dtype)
    return {
        "parameters": {
            "format": "csr",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": nnz,
            "n_dense_cols": n_dense_cols,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
            "block_n": launch["block_n"],
            "block_nnz": launch["block_nnz"],
            "max_segments": launch["max_segments"],
            "required_segments": launch["required_segments"],
            "alg1_warp_size": launch["warp_size"],
            "alg1_factor": launch["factor"],
            "auto_max_segments": launch["auto_max_segments"],
            "run_cusparse": run_cusparse,
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "triton_first_call_ms": triton_first_call_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": triton_speedup_vs_pytorch,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_reference": triton_match,
            "triton_match_pytorch": triton_match,
            "triton_max_error": triton_metrics["max_abs_error"],
            "triton_max_abs_error": triton_metrics["max_abs_error"],
            "triton_max_relative_error": triton_metrics["max_relative_error"],
            "triton_sum_relative_error": triton_metrics["sum_relative_error"],
            "triton_relative_threshold": triton_metrics["relative_threshold"],
            "triton_strict_allclose_match": triton_metrics["strict_allclose_match"],
            "pytorch_match_reference": True,
            "pytorch_max_error": 0.0,
            "pytorch_max_abs_error": 0.0,
            "pytorch_max_relative_error": 0.0,
            "pytorch_sum_relative_error": 0.0,
            "pytorch_relative_threshold": threshold,
            "cusparse_match_reference": cusparse_match,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": (cusparse_metrics["max_abs_error"] if cusparse_metrics is not None else None),
            "cusparse_max_abs_error": (cusparse_metrics["max_abs_error"] if cusparse_metrics is not None else None),
            "cusparse_max_relative_error": (cusparse_metrics["max_relative_error"] if cusparse_metrics is not None else None),
            "cusparse_sum_relative_error": (cusparse_metrics["sum_relative_error"] if cusparse_metrics is not None else None),
            "cusparse_relative_threshold": (cusparse_metrics["relative_threshold"] if cusparse_metrics is not None else threshold),
            "cusparse_strict_allclose_match": (cusparse_metrics["strict_allclose_match"] if cusparse_metrics is not None else None),
        },
        "backend_status": {
            "pytorch_unavailable_reason": pytorch_reason,
            "pytorch_sparse_format": pytorch_format,
            "cusparse_unavailable_reason": cusparse_reason,
            "flagsparse_internal_route": "csr-base-alg1-inspired",
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_C,
            "reference": expected,
            "cusparse": cusparse_values,
        },
    }
def comprehensive_spmm_test(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    n_dense_cols=32,
    dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_n=None,
    block_nnz=None,
    max_segments=None,
    run_cusparse=True,
):
    """Full SpMM benchmark entry for one configuration."""
    return benchmark_spmm_case(
        n_rows=n_rows,
        n_cols=n_cols,
        nnz=nnz,
        n_dense_cols=n_dense_cols,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        block_n=block_n,
        block_nnz=block_nnz,
        max_segments=max_segments,
        run_cusparse=run_cusparse,
    )
