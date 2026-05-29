"""CSR SpMM kernels, helpers, and benchmark entry points."""

import math

from ._common import *
from ._alpha_spmm_alg1_common import _select_alpha_spmm_alg1_warp_and_factor
from dataclasses import dataclass

SUPPORTED_SPMM_VALUE_DTYPES = SUPPORTED_VALUE_DTYPES

SPMM_OP_NON = 0
SPMM_OP_TRANS = 1
SPMM_OP_CONJ_TRANS = 2
SPMM_OP_NAMES = {
    SPMM_OP_NON: "non",
    SPMM_OP_TRANS: "trans",
    SPMM_OP_CONJ_TRANS: "conj",
}
_SPMM_OP_NAME_TO_CODE = {name: code for code, name in SPMM_OP_NAMES.items()}


def _normalize_spmm_op(op=None, transpose=False):
    if op is None:
        return SPMM_OP_TRANS if bool(transpose) else SPMM_OP_NON
    if isinstance(op, str):
        token = op.strip().lower()
        if token not in _SPMM_OP_NAME_TO_CODE:
            raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
        return _SPMM_OP_NAME_TO_CODE[token]
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj") from exc
    if op_code not in SPMM_OP_NAMES:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
    return op_code


def _spmm_op_to_name(op):
    op_code = _normalize_spmm_op(op)
    return SPMM_OP_NAMES[op_code]


def _spmm_op_transposes(op):
    return _normalize_spmm_op(op) in (SPMM_OP_TRANS, SPMM_OP_CONJ_TRANS)


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


def _transpose_csr_for_spmm(data, indices, indptr, shape):
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


def _materialize_spmm_csr_op(data, indices, indptr, shape, op_code):
    if op_code == SPMM_OP_NON:
        return data, indices, indptr, shape
    data_op = data
    if op_code == SPMM_OP_CONJ_TRANS and _is_complex_dtype(data.dtype):
        data_op = data.conj()
        if hasattr(data_op, "resolve_conj"):
            data_op = data_op.resolve_conj()
    return _transpose_csr_for_spmm(data_op, indices, indptr, shape)


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


def _normalize_spmm_base_device_props(device):
    props = torch.cuda.get_device_properties(device)
    warp_size = int(getattr(props, "warp_size", 32) or 32)
    max_threads_per_block = int(getattr(props, "max_threads_per_block", 1024) or 1024)
    max_threads_per_mp = int(
        getattr(props, "max_threads_per_multi_processor", 2048) or 2048
    )
    return {
        "warp_size": max(1, warp_size),
        "max_threads_per_block": max(32, max_threads_per_block),
        "max_threads_per_mp": max(32, max_threads_per_mp),
    }


def _clip_spmm_base_num_warps(desired, device_props):
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


def _resolve_spmm_base_triton_launch(
    dtype,
    n_dense_cols,
    max_row_nnz,
    block_n=None,
    block_nnz=None,
    max_segments=None,
    device_props=None,
):
    launch = _resolve_spmm_alg1_launch_config(
        n_dense_cols,
        max_row_nnz,
        block_n=block_n,
        block_nnz=block_nnz,
        max_segments=max_segments,
    )
    if device_props is None:
        return {
            **launch,
            "num_warps": 1,
            "num_stages": 2,
        }

    n_dense_cols = int(n_dense_cols)
    max_row_nnz = int(max_row_nnz)
    block_n = int(launch["block_n"])

    if n_dense_cols <= 16:
        desired_warps = 1
    elif n_dense_cols <= 32:
        desired_warps = 2
    elif n_dense_cols <= 64:
        desired_warps = 4
    else:
        desired_warps = 8 if max_row_nnz >= 512 else 4

    if block_n >= 128:
        desired_warps = max(desired_warps, 4)
    if max_row_nnz >= 1024 and n_dense_cols > 32:
        desired_warps = max(desired_warps, 8)

    if dtype in (torch.float64, torch.complex128):
        desired_warps = min(desired_warps, 8)
    elif dtype in (torch.float16, torch.bfloat16, torch.float32, torch.complex64):
        desired_warps = min(desired_warps, 16)

    num_warps = _clip_spmm_base_num_warps(desired_warps, device_props)
    if n_dense_cols > 64 or block_n >= 128:
        num_stages = 1
    elif dtype in (torch.float64, torch.complex128) and max_row_nnz >= 512:
        num_stages = 1
    else:
        num_stages = 2

    return {
        **launch,
        "num_warps": int(num_warps),
        "num_stages": int(num_stages),
    }


@dataclass(frozen=True)
class SpmmCsrAlgorithm:
    """Registered CSR SpMM route for the AlphaSparse-style run API."""

    name: str
    display_name: str
    supported_ops: tuple
    supported_dtypes: tuple
    run: object


class SpmmCsrAlgorithmUnavailable(RuntimeError):
    """Raised when a registered SpMM CSR route is unavailable in this runtime."""


class PreparedCsrSpmmRoute:
    """Matrix-level CSR SpMM route preparation shared by route algorithms."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "row_lengths",
        "max_row_nnz",
        "nnz",
        "avg_nnz_per_row",
        "op",
        "alg",
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
        op,
        alg,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.row_lengths = row_lengths
        self.max_row_nnz = int(max_row_nnz)
        self.nnz = int(data.numel())
        self.avg_nnz_per_row = float(self.nnz) / float(max(1, self.n_rows))
        self.op = str(op)
        self.alg = str(alg)


def _normalize_spmm_csr_alg(alg):
    token = "auto" if alg is None else str(alg).strip().lower()
    if token in ("base", "csr", "csr_base"):
        return "csr_base"
    if token in ("alg1", "csr_alg1", "spmm_csr_alg1"):
        return "spmm_csr_alg1"
    if token in ("alg2", "csr_alg2", "spmm_csr_alg2"):
        return "spmm_csr_alg2"
    if token == "auto":
        return "auto"
    return token


def _normalize_dense_layout(layout):
    token = "row" if layout is None else str(layout).strip().lower()
    if token in ("auto", "default"):
        return "row"
    if token in ("row", "row_major", "row-major", "c", "c_order"):
        return "row"
    if token in ("col", "column", "col_major", "column_major", "col-major", "column-major", "f", "fortran"):
        return "col"
    raise ValueError("dense_layout must be one of: auto, row, col")


def _is_col_major_2d(tensor):
    return (
        torch.is_tensor(tensor)
        and tensor.ndim == 2
        and tensor.stride(0) == 1
        and tensor.stride(1) >= max(1, int(tensor.shape[0]))
    )


def _dense_layout_name(tensor):
    if not torch.is_tensor(tensor) or tensor.ndim != 2:
        return "unknown"
    if tensor.is_contiguous():
        return "row"
    if _is_col_major_2d(tensor):
        return "col"
    return "strided"


def _empty_dense_layout(shape, dtype, device, layout):
    layout = _normalize_dense_layout(layout)
    rows, cols = int(shape[0]), int(shape[1])
    if layout == "col":
        return torch.empty_strided((rows, cols), (1, max(1, rows)), dtype=dtype, device=device)
    return torch.empty((rows, cols), dtype=dtype, device=device)


def _zeros_dense_layout(shape, dtype, device, layout):
    out = _empty_dense_layout(shape, dtype, device, layout)
    out.zero_()
    return out


def _materialize_dense_layout(tensor, layout):
    layout = _normalize_dense_layout(layout)
    if tensor.ndim != 2:
        raise ValueError("dense layout materialization expects a 2D tensor")
    if layout == "row":
        return tensor.contiguous()
    if _is_col_major_2d(tensor):
        return tensor
    out = _empty_dense_layout(tensor.shape, tensor.dtype, tensor.device, layout)
    out.copy_(tensor)
    return out


def _validate_spmm_route_runtime_inputs(prepared, B):
    if not isinstance(prepared, PreparedCsrSpmmRoute):
        raise TypeError("prepared must be a PreparedCsrSpmmRoute instance")
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
    return B


def _spmm_csr_route_from_materialized(prepared, data, indices, indptr, shape, op_name):
    kernel_indptr = indptr.to(torch.int64)
    row_lengths = (
        kernel_indptr[1:] - kernel_indptr[:-1]
        if int(shape[0]) > 0
        else kernel_indptr.new_empty((0,))
    )
    max_row_nnz = int(row_lengths.max().item()) if int(shape[0]) > 0 else 0
    kernel_indices = indices.to(torch.int32) if indices.dtype == torch.int64 else indices
    return PreparedCsrSpmmRoute(
        data=data.contiguous(),
        kernel_indices=kernel_indices.contiguous(),
        kernel_indptr=kernel_indptr.contiguous(),
        shape=shape,
        n_rows=int(shape[0]),
        n_cols=int(shape[1]),
        row_lengths=row_lengths,
        max_row_nnz=max_row_nnz,
        op=op_name,
        alg=prepared.alg,
    )


def _materialize_spmm_csr_route_op(prepared, op_name, *, timing=False):
    if op_name == "non":
        return prepared, 0.0 if timing else None

    start = torch.cuda.Event(enable_timing=True) if timing else None
    end = torch.cuda.Event(enable_timing=True) if timing else None
    if start is not None:
        start.record()
    data, indices, indptr, shape = _materialize_spmm_csr_op(
        prepared.data,
        prepared.kernel_indices,
        prepared.kernel_indptr,
        prepared.shape,
        _normalize_spmm_op(op_name),
    )
    runtime_prepared = _spmm_csr_route_from_materialized(
        prepared,
        data,
        indices,
        indptr,
        shape,
        op_name,
    )
    if end is not None:
        end.record()
        torch.cuda.synchronize()
        return runtime_prepared, start.elapsed_time(end)
    return runtime_prepared, None


def _run_spmm_csr_base_route(prepared, B, *, timing=False, diagnostics=False, dense_layout="row"):
    B = _validate_spmm_route_runtime_inputs(prepared, B)
    dense_layout = _normalize_dense_layout(dense_layout)
    if dense_layout == "col" and _is_complex_dtype(prepared.data.dtype):
        raise SpmmCsrAlgorithmUnavailable("csr_base col-major layout does not support complex dtypes yet")
    B = _materialize_dense_layout(B, dense_layout)
    C_out = _empty_dense_layout(
        (prepared.n_rows, int(B.shape[1])),
        prepared.data.dtype,
        prepared.data.device,
        dense_layout,
    )
    device_props = _normalize_spmm_base_device_props(prepared.data.device)
    launch = _resolve_spmm_base_triton_launch(
        prepared.data.dtype,
        int(B.shape[1]),
        prepared.max_row_nnz,
        device_props=device_props,
    )
    process_cpu_ms = 0.0
    process_gpu_ms = 0.0 if timing else None
    compute_ms = None
    if timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    C = _triton_spmm_csr_impl(
        prepared.data,
        prepared.kernel_indices,
        prepared.kernel_indptr,
        B,
        prepared.n_rows,
        int(B.shape[1]),
        block_n=launch["block_n"],
        block_nnz=launch["block_nnz"],
        num_warps=launch["num_warps"],
        num_stages=launch["num_stages"],
        out=C_out,
        dense_layout=dense_layout,
    )
    if timing:
        end.record()
        torch.cuda.synchronize()
        compute_ms = start.elapsed_time(end)
    meta = {
        "alg": "csr_base",
        "display_name": "Base",
        "op": prepared.op,
        "process_cpu_ms": process_cpu_ms,
        "process_gpu_ms": process_gpu_ms,
        "compute_ms": compute_ms,
        "dense_layout": dense_layout,
        "b_stride": tuple(int(v) for v in B.stride()),
        "c_stride": tuple(int(v) for v in C.stride()),
        "output_layout": _dense_layout_name(C),
    }
    if diagnostics:
        meta["diagnostics"] = {
            "launch_config_scope": "matrix",
            "launch_config_count": 1,
            "bucket_count": 0,
            "long_row_count": 0,
            "launch_version": "csr_base_v1",
            "block_n": launch["block_n"],
            "block_nnz": launch["block_nnz"],
            "num_warps": launch["num_warps"],
            "num_stages": launch["num_stages"],
            "required_segments": launch["required_segments"],
            "warp_size": launch["warp_size"],
            "factor": launch["factor"],
            "dense_layout": dense_layout,
            "b_stride": tuple(int(v) for v in B.stride()),
            "c_stride": tuple(int(v) for v in C.stride()),
            "output_layout": _dense_layout_name(C),
        }
    return C, meta


def _build_alpha_spmm_alg1_route_prepared(prepared):
    from .alpha_spmm_alg1 import PreparedAlphaSpmmAlg1

    return PreparedAlphaSpmmAlg1(
        data=prepared.data,
        kernel_indices=prepared.kernel_indices,
        kernel_indptr=prepared.kernel_indptr,
        shape=prepared.shape,
        n_rows=prepared.n_rows,
        n_cols=prepared.n_cols,
        row_lengths=prepared.row_lengths,
        max_row_nnz=prepared.max_row_nnz,
    )


def _run_alpha_spmm_alg1_tle_route(
    prepared,
    B,
    *,
    route_name,
    availability_fn_name,
    unavailable_reason_fn_name,
    build_meta_fn_name,
    run_fn_name,
    timing=False,
    diagnostics=False,
):
    B = _validate_spmm_route_runtime_inputs(prepared, B)
    from . import alpha_spmm_alg1 as alpha_mod
    availability_fn = getattr(alpha_mod, availability_fn_name)
    unavailable_reason_fn = getattr(alpha_mod, unavailable_reason_fn_name)
    if not availability_fn():
        raise SpmmCsrAlgorithmUnavailable(unavailable_reason_fn())

    alpha_prepared = _build_alpha_spmm_alg1_route_prepared(prepared)
    build_meta_fn = getattr(alpha_mod, build_meta_fn_name)
    run_fn = getattr(alpha_mod, run_fn_name)
    launch_meta = build_meta_fn(alpha_prepared, B)

    process_cpu_ms = 0.0
    process_gpu_ms = 0.0 if timing else None
    compute_ms = None
    if timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    C = run_fn(B=B, prepared=alpha_prepared, meta=launch_meta)
    if timing:
        end.record()
        torch.cuda.synchronize()
        compute_ms = start.elapsed_time(end)
    meta = {
        "alg": route_name,
        "display_name": route_name,
        "op": prepared.op,
        "process_cpu_ms": process_cpu_ms,
        "process_gpu_ms": process_gpu_ms,
        "compute_ms": compute_ms,
    }
    if diagnostics:
        meta["diagnostics"] = {
            "launch_config_scope": "matrix",
            "launch_config_count": 1,
            "bucket_count": 0,
            "long_row_count": 0,
            "launch_version": launch_meta.get("launch_version"),
            "block_n": launch_meta.get("block_cols"),
            "block_nnz": launch_meta.get("warp_size"),
            "num_warps": launch_meta.get("num_warps"),
            "num_stages": launch_meta.get("num_stages"),
            "warp_size": launch_meta.get("warp_size"),
            "factor": launch_meta.get("factor"),
            "block_rows": launch_meta.get("block_rows"),
            "block_cols": launch_meta.get("block_cols"),
            "grid_m": launch_meta.get("grid_m"),
            "grid_n": launch_meta.get("grid_n"),
        }
    return C, meta


def _run_alpha_spmm_alg1_tle_opt_route(prepared, B, *, timing=False, diagnostics=False):
    return _run_alpha_spmm_alg1_tle_route(
        prepared,
        B,
        route_name="alpha_alg1_tle_opt",
        availability_fn_name="is_alpha_spmm_alg1_tle_opt_available",
        unavailable_reason_fn_name="alpha_spmm_alg1_tle_opt_unavailable_reason",
        build_meta_fn_name="build_alpha_spmm_alg1_tle_opt_meta",
        run_fn_name="flagsparse_alpha_spmm_alg1_tle_opt",
        timing=timing,
        diagnostics=diagnostics,
    )


def _run_alpha_spmm_alg1_tle_opt2_route(prepared, B, *, timing=False, diagnostics=False):
    return _run_alpha_spmm_alg1_tle_route(
        prepared,
        B,
        route_name="alpha_alg1_tle_opt2",
        availability_fn_name="is_alpha_spmm_alg1_tle_opt2_available",
        unavailable_reason_fn_name="alpha_spmm_alg1_tle_opt2_unavailable_reason",
        build_meta_fn_name="build_alpha_spmm_alg1_tle_opt2_meta",
        run_fn_name="flagsparse_alpha_spmm_alg1_tle_opt2",
        timing=timing,
        diagnostics=diagnostics,
    )


class PreparedSpmmCsrAlg1Plan:
    """Runtime execution plan for the route-native SpMM CSR Alg1 path."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "row_buckets",
        "long_part_rows",
        "long_part_starts",
        "long_part_ends",
        "long_row_ids",
        "long_row_part_ptr",
        "process_cpu_ms",
        "process_gpu_ms",
        "bucket_count",
        "long_row_count",
        "long_part_count",
    )

    def __init__(
        self,
        data,
        kernel_indices,
        kernel_indptr,
        shape,
        row_buckets,
        long_part_rows,
        long_part_starts,
        long_part_ends,
        long_row_ids,
        long_row_part_ptr,
        process_cpu_ms,
        process_gpu_ms,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(shape[0])
        self.n_cols = int(shape[1])
        self.row_buckets = row_buckets
        self.long_part_rows = long_part_rows
        self.long_part_starts = long_part_starts
        self.long_part_ends = long_part_ends
        self.long_row_ids = long_row_ids
        self.long_row_part_ptr = long_row_part_ptr
        self.process_cpu_ms = float(process_cpu_ms)
        self.process_gpu_ms = process_gpu_ms
        self.bucket_count = int(len(row_buckets))
        self.long_row_count = int(long_row_ids.numel())
        self.long_part_count = int(long_part_rows.numel())


@triton.jit
def _spmm_csr_alg1_process_count_kernel(
    row_lengths_ptr,
    bucket_counts_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_rows
    lens = tl.load(row_lengths_ptr + offs, mask=mask, other=0)
    bucket = tl.where(
        lens <= 32,
        0,
        tl.where(lens <= 128, 1, tl.where(lens <= 512, 2, tl.where(lens <= 2048, 3, 4))),
    )
    for bid in tl.static_range(0, 5):
        hits = mask & (bucket == bid)
        count = tl.sum(tl.where(hits, 1, 0), axis=0)
        tl.atomic_add(bucket_counts_ptr + bid, count, sem="relaxed")


@triton.jit
def _spmm_csr_alg1_process_compact_kernel(
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
    bucket = tl.where(
        lens <= 32,
        0,
        tl.where(lens <= 128, 1, tl.where(lens <= 512, 2, tl.where(lens <= 2048, 3, 4))),
    )
    for bid in tl.static_range(0, 5):
        hits = mask & (bucket == bid)
        ranks = tl.cumsum(tl.where(hits, 1, 0), axis=0) - 1
        local_count = tl.sum(tl.where(hits, 1, 0), axis=0)
        base = tl.atomic_add(bucket_write_counts_ptr + bid, local_count, sem="relaxed")
        offset = tl.load(bucket_offsets_ptr + bid)
        tl.store(rows_flat_ptr + offset + base + ranks, offs, mask=hits)


def _spmm_csr_alg1_empty_split_metadata(device, row_index_dtype):
    return (
        torch.empty((0,), dtype=row_index_dtype, device=device),
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.zeros(1, dtype=torch.int64, device=device),
    )


def _spmm_csr_alg1_build_bucket_descriptors(rows_flat, counts, offsets):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    counts_cpu = counts.cpu().tolist()
    offsets_cpu = offsets.cpu().tolist()
    row_index_dtype = rows_flat.dtype
    device = rows_flat.device
    buckets = []
    long_rows = torch.empty((0,), dtype=row_index_dtype, device=device)
    lower = 0
    for spec, count, offset in zip(_SPMM_CSR_ALG1_BUCKET_SPECS, counts_cpu, offsets_cpu):
        upper = spec["max_row_nnz"]
        count = int(count)
        if count == 0:
            if upper is not None:
                lower = upper
            continue
        rows = rows_flat.narrow(0, int(offset), count)
        if upper is None:
            max_row_nnz = _SPMM_OPT_LONG_ROW_THRESHOLD + 1
            label = "spmm_csr_alg1_split_long"
            execution = "split"
        else:
            max_row_nnz = upper
            label = f"spmm_csr_alg1_{spec['kind']}_{upper}"
            execution = "alg1"
        buckets.append(
            _spmm_opt_make_bucket(
                label,
                spec["kind"],
                rows,
                execution,
                lower + 1 if lower > 0 else 0,
                max_row_nnz,
                batch_rows=spec["batch_rows"],
                block_k=spec["block_nnz"],
                block_nnz=spec["block_nnz"],
                block_n_cap=128,
            )
        )
        if spec["kind"] == "split":
            long_rows = rows
        if upper is not None:
            lower = upper
    process_cpu_ms = (time.perf_counter() - t0) * 1000.0
    return buckets, long_rows, process_cpu_ms


def _spmm_csr_alg1_build_process_plan(prepared, *, timing=False):
    if prepared.data.dtype not in (torch.float32, torch.float64):
        raise TypeError("spmm_csr_alg1 only supports float32 and float64")
    device = prepared.data.device
    row_count = int(prepared.row_lengths.numel())
    row_index_dtype = torch.int32 if row_count <= _INDEX_LIMIT_INT32 else torch.int64
    bucket_count = len(_SPMM_CSR_ALG1_BUCKET_SPECS)
    if bucket_count != 5:
        raise RuntimeError("spmm_csr_alg1 expects five bucket specs")

    counts = torch.zeros((bucket_count,), dtype=torch.int64, device=device)
    offsets = torch.empty_like(counts)
    rows_flat = torch.empty((row_count,), dtype=row_index_dtype, device=device)
    write_counts = torch.zeros_like(counts)
    block_m = 256
    grid = (triton.cdiv(row_count, block_m),)

    start = torch.cuda.Event(enable_timing=True) if timing else None
    end = torch.cuda.Event(enable_timing=True) if timing else None
    if start is not None:
        start.record()
    if row_count > 0:
        _spmm_csr_alg1_process_count_kernel[grid](
            prepared.row_lengths,
            counts,
            row_count,
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=1,
        )
    offsets[0] = 0
    if bucket_count > 1:
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)
    if row_count > 0:
        _spmm_csr_alg1_process_compact_kernel[grid](
            prepared.row_lengths,
            offsets,
            write_counts,
            rows_flat,
            row_count,
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=1,
        )
    if end is not None:
        end.record()
        torch.cuda.synchronize()
        process_gpu_ms = start.elapsed_time(end)
    else:
        torch.cuda.synchronize()
        process_gpu_ms = None

    row_buckets, long_rows, process_cpu_ms = _spmm_csr_alg1_build_bucket_descriptors(
        rows_flat,
        counts,
        offsets,
    )

    split_start = torch.cuda.Event(enable_timing=True) if timing else None
    split_end = torch.cuda.Event(enable_timing=True) if timing else None
    if split_start is not None:
        split_start.record()
    if long_rows.numel() > 0:
        (
            long_part_rows,
            long_part_starts,
            long_part_ends,
            long_row_part_ptr,
        ) = _build_spmm_opt_split_metadata(
            prepared.kernel_indptr,
            long_rows,
            part_block_nnz=_SPMM_OPT_SPLIT_BLOCK_NNZ,
        )
    else:
        (
            long_part_rows,
            long_part_starts,
            long_part_ends,
            long_row_part_ptr,
        ) = _spmm_csr_alg1_empty_split_metadata(device, row_index_dtype)
    if split_end is not None:
        split_end.record()
        torch.cuda.synchronize()
        process_gpu_ms = float(process_gpu_ms or 0.0) + split_start.elapsed_time(split_end)

    return PreparedSpmmCsrAlg1Plan(
        data=prepared.data,
        kernel_indices=prepared.kernel_indices,
        kernel_indptr=prepared.kernel_indptr,
        shape=prepared.shape,
        row_buckets=row_buckets,
        long_part_rows=long_part_rows,
        long_part_starts=long_part_starts,
        long_part_ends=long_part_ends,
        long_row_ids=long_rows,
        long_row_part_ptr=long_row_part_ptr,
        process_cpu_ms=process_cpu_ms,
        process_gpu_ms=process_gpu_ms,
    )


def _spmm_csr_alg1_run_bucket(plan, bucket, B, C_out, block_n, device_props):
    rows = bucket["rows"]
    if rows.numel() == 0:
        return
    dtype = plan.data.dtype
    kind = bucket["kind"]
    kernel_map = {
        ("batched", torch.float32): _spmm_csr_batched_rows_f32_kernel,
        ("batched", torch.float64): _spmm_csr_batched_rows_f64_kernel,
        ("vector", torch.float32): _spmm_csr_vector_rows_f32_kernel,
        ("vector", torch.float64): _spmm_csr_vector_rows_f64_kernel,
    }
    if (kind, dtype) not in kernel_map:
        raise TypeError(f"unsupported spmm_csr_alg1 bucket kind/dtype: {kind}/{dtype}")

    batch_rows = int(bucket.get("batch_rows", 1))
    block_n = _select_spmm_opt_block_n_for_bucket(
        int(B.shape[1]),
        bucket.get("block_n_cap", block_n),
    )
    block_nnz = int(bucket["block_nnz"])
    launch = _resolve_spmm_opt_launch(
        kind,
        block_n,
        block_nnz,
        batch_rows,
        dtype,
        device_props,
    )
    kernel = kernel_map[(kind, dtype)]

    if kind == "batched":
        grid = (triton.cdiv(rows.numel(), batch_rows), triton.cdiv(B.shape[1], block_n))
        kernel[grid](
            plan.data,
            plan.kernel_indices,
            plan.kernel_indptr,
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

    grid = (rows.numel(), triton.cdiv(B.shape[1], block_n))
    kernel[grid](
        plan.data,
        plan.kernel_indices,
        plan.kernel_indptr,
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


def _spmm_csr_alg1_run_split_bucket(plan, B, C_out, block_n, device_props):
    if plan.long_part_rows.numel() == 0:
        return
    block_n = _select_spmm_opt_block_n_for_bucket(int(B.shape[1]), block_n)
    workspace = torch.empty(
        (plan.long_part_rows.numel(), B.shape[1]),
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
        "split_part",
        block_n,
        _SPMM_OPT_SPLIT_BLOCK_NNZ,
        1,
        B.dtype,
        device_props,
    )
    reduce_launch = _resolve_spmm_opt_launch(
        "split_reduce",
        block_n,
        _SPMM_OPT_SPLIT_BLOCK_NNZ,
        1,
        B.dtype,
        device_props,
    )
    split_grid = (
        plan.long_part_rows.numel(),
        triton.cdiv(B.shape[1], block_n),
    )
    split_kernel[split_grid](
        plan.data,
        plan.kernel_indices,
        B,
        workspace,
        plan.long_part_starts,
        plan.long_part_ends,
        plan.long_part_rows.numel(),
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
        plan.long_row_ids.numel(),
        triton.cdiv(B.shape[1], block_n),
    )
    reduce_kernel[reduce_grid](
        workspace,
        C_out,
        plan.long_row_ids,
        plan.long_row_part_ptr,
        plan.long_row_ids.numel(),
        B.shape[1],
        workspace.stride(0),
        workspace.stride(1),
        C_out.stride(0),
        C_out.stride(1),
        BLOCK_N=block_n,
        num_warps=reduce_launch["num_warps"],
        num_stages=reduce_launch["num_stages"],
    )


def _spmm_csr_alg1_compute(plan, B):
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != plan.data.device:
        raise ValueError("B must be on the same CUDA device as sparse matrix data")
    if B.dtype != plan.data.dtype:
        raise TypeError("B dtype must match sparse matrix dtype")
    if B.shape[0] != plan.n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={plan.n_cols}, got {B.shape[0]}")
    B = B.contiguous()
    block_n = _select_spmm_opt_block_n(int(B.shape[1]))
    C_out = torch.zeros((plan.n_rows, int(B.shape[1])), dtype=B.dtype, device=B.device)
    device_props = _normalize_spmm_opt_device_props(plan.data.device)
    for bucket in plan.row_buckets:
        if bucket["kind"] == "split":
            _spmm_csr_alg1_run_split_bucket(plan, B, C_out, block_n, device_props)
            continue
        _spmm_csr_alg1_run_bucket(plan, bucket, B, C_out, block_n, device_props)
    return C_out


def _run_spmm_csr_alg1_route(prepared, B, *, timing=False, diagnostics=False):
    B = _validate_spmm_route_runtime_inputs(prepared, B)
    plan = _spmm_csr_alg1_build_process_plan(prepared, timing=bool(timing))
    compute_ms = None
    if timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    C = _spmm_csr_alg1_compute(plan, B)
    if timing:
        end.record()
        torch.cuda.synchronize()
        compute_ms = start.elapsed_time(end)
    meta = {
        "alg": "spmm_csr_alg1",
        "display_name": "Alg1",
        "op": prepared.op,
        "process_cpu_ms": plan.process_cpu_ms,
        "process_gpu_ms": plan.process_gpu_ms if timing else None,
        "compute_ms": compute_ms,
    }
    if diagnostics:
        meta["diagnostics"] = {
            "launch_config_scope": "bucket",
            "launch_config_count": plan.bucket_count,
            "bucket_count": plan.bucket_count,
            "long_row_count": plan.long_row_count,
            "launch_version": "spmm_csr_alg1_v1",
            "block_n": None,
            "block_nnz": None,
            "num_warps": None,
            "num_stages": None,
            "long_part_count": plan.long_part_count,
        }
    return C, meta


_SPMM_CSR_ALG2_BUCKET_SPECS_F32 = (
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

_SPMM_CSR_ALG2_BUCKET_SPECS_F64 = (
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


def _spmm_csr_alg2_bucket_specs(dtype):
    return (
        _SPMM_CSR_ALG2_BUCKET_SPECS_F64
        if dtype == torch.float64
        else _SPMM_CSR_ALG2_BUCKET_SPECS_F32
    )


def _normalize_spmm_csr_alg2_device_props(device):
    props = torch.cuda.get_device_properties(device)
    warp_size = int(getattr(props, "warp_size", 32) or 32)
    sm_count = int(getattr(props, "multi_processor_count", 0) or 0)
    max_threads_per_mp = int(getattr(props, "max_threads_per_multi_processor", 2048) or 2048)
    max_threads_per_block = int(getattr(props, "max_threads_per_block", 1024) or 1024)
    shared_memory_per_block = int(getattr(props, "shared_memory_per_block", 0) or 0)
    return {
        "device_name": str(getattr(props, "name", "cuda")),
        "warp_size": max(1, warp_size),
        "sm_count": max(0, sm_count),
        "max_threads_per_mp": max(32, max_threads_per_mp),
        "max_threads_per_block": max(32, max_threads_per_block),
        "shared_memory_per_block": max(0, shared_memory_per_block),
        "capability": (
            int(getattr(props, "major", 0) or 0),
            int(getattr(props, "minor", 0) or 0),
        ),
    }


def _select_spmm_csr_alg2_block_n(n_dense_cols, block_n_cap):
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


def _select_spmm_csr_alg2_num_warps(bucket, block_n, device_props, dtype):
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


def _select_spmm_csr_alg2_num_stages(bucket, block_n, num_warps, device_props, dtype):
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


def _resolve_spmm_csr_alg2_launch(bucket, n_dense_cols, dtype, device_props):
    block_n = _select_spmm_csr_alg2_block_n(n_dense_cols, bucket["block_n_cap"])
    num_warps = _select_spmm_csr_alg2_num_warps(bucket, block_n, device_props, dtype)
    num_stages = _select_spmm_csr_alg2_num_stages(
        bucket,
        block_n,
        num_warps,
        device_props,
        dtype,
    )
    return {
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


class PreparedSpmmCsrAlg2Plan:
    """Runtime execution plan for the route-native SpMM CSR Alg2 path."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "row_buckets",
        "process_cpu_ms",
        "process_gpu_ms",
        "bucket_count",
        "long_row_count",
        "launch_configs",
    )

    def __init__(
        self,
        data,
        kernel_indices,
        kernel_indptr,
        shape,
        row_buckets,
        process_cpu_ms,
        process_gpu_ms,
        long_row_count,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(shape[0])
        self.n_cols = int(shape[1])
        self.row_buckets = row_buckets
        self.process_cpu_ms = float(process_cpu_ms)
        self.process_gpu_ms = process_gpu_ms
        self.bucket_count = int(len(row_buckets))
        self.long_row_count = int(long_row_count)
        self.launch_configs = []


@triton.jit
def _spmm_csr_alg2_process_count_kernel(
    row_lengths_ptr,
    bucket_counts_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_rows
    lens = tl.load(row_lengths_ptr + offs, mask=mask, other=0)
    bucket = tl.where(
        lens <= 16,
        0,
        tl.where(lens <= 64, 1, tl.where(lens <= 256, 2, tl.where(lens <= 1024, 3, 4))),
    )
    for bid in tl.static_range(0, 5):
        hits = mask & (bucket == bid)
        count = tl.sum(tl.where(hits, 1, 0), axis=0)
        tl.atomic_add(bucket_counts_ptr + bid, count, sem="relaxed")


@triton.jit
def _spmm_csr_alg2_process_compact_kernel(
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
    bucket = tl.where(
        lens <= 16,
        0,
        tl.where(lens <= 64, 1, tl.where(lens <= 256, 2, tl.where(lens <= 1024, 3, 4))),
    )
    for bid in tl.static_range(0, 5):
        hits = mask & (bucket == bid)
        ranks = tl.cumsum(tl.where(hits, 1, 0), axis=0) - 1
        local_count = tl.sum(tl.where(hits, 1, 0), axis=0)
        base = tl.atomic_add(bucket_write_counts_ptr + bid, local_count, sem="relaxed")
        offset = tl.load(bucket_offsets_ptr + bid)
        tl.store(rows_flat_ptr + offset + base + ranks, offs, mask=hits)


def _spmm_csr_alg2_build_bucket_descriptors(rows_flat, counts, offsets, dtype):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    counts_cpu = counts.cpu().tolist()
    offsets_cpu = offsets.cpu().tolist()
    buckets = []
    long_row_count = 0
    for spec, count, offset in zip(_spmm_csr_alg2_bucket_specs(dtype), counts_cpu, offsets_cpu):
        count = int(count)
        if count == 0:
            continue
        rows = rows_flat.narrow(0, int(offset), count)
        if spec["label"] == "seg_long":
            long_row_count = count
        buckets.append(
            {
                "label": spec["label"],
                "kind": spec["kind"],
                "rows": rows,
                "batch_rows": int(spec["batch_rows"]),
                "block_k": int(spec["block_k"]),
                "block_n_cap": int(spec["block_n_cap"]),
                "segments": int(spec.get("segments", 1)),
            }
        )
    process_cpu_ms = (time.perf_counter() - t0) * 1000.0
    return buckets, long_row_count, process_cpu_ms


def _spmm_csr_alg2_build_process_plan(prepared, *, timing=False):
    if prepared.data.dtype not in (torch.float32, torch.float64):
        raise TypeError("spmm_csr_alg2 only supports float32 and float64")
    device = prepared.data.device
    row_count = int(prepared.row_lengths.numel())
    row_index_dtype = torch.int32 if row_count <= _INDEX_LIMIT_INT32 else torch.int64
    bucket_count = len(_spmm_csr_alg2_bucket_specs(prepared.data.dtype))
    if bucket_count != 5:
        raise RuntimeError("spmm_csr_alg2 expects five bucket specs")

    counts = torch.zeros((bucket_count,), dtype=torch.int64, device=device)
    offsets = torch.empty_like(counts)
    rows_flat = torch.empty((row_count,), dtype=row_index_dtype, device=device)
    write_counts = torch.zeros_like(counts)
    block_m = 256
    grid = (triton.cdiv(row_count, block_m),)

    start = torch.cuda.Event(enable_timing=True) if timing else None
    end = torch.cuda.Event(enable_timing=True) if timing else None
    if start is not None:
        start.record()
    if row_count > 0:
        _spmm_csr_alg2_process_count_kernel[grid](
            prepared.row_lengths,
            counts,
            row_count,
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=1,
        )
    offsets[0] = 0
    if bucket_count > 1:
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)
    if row_count > 0:
        _spmm_csr_alg2_process_compact_kernel[grid](
            prepared.row_lengths,
            offsets,
            write_counts,
            rows_flat,
            row_count,
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=1,
        )
    if end is not None:
        end.record()
        torch.cuda.synchronize()
        process_gpu_ms = start.elapsed_time(end)
    else:
        torch.cuda.synchronize()
        process_gpu_ms = None

    row_buckets, long_row_count, process_cpu_ms = _spmm_csr_alg2_build_bucket_descriptors(
        rows_flat,
        counts,
        offsets,
        prepared.data.dtype,
    )
    return PreparedSpmmCsrAlg2Plan(
        data=prepared.data,
        kernel_indices=prepared.kernel_indices,
        kernel_indptr=prepared.kernel_indptr,
        shape=prepared.shape,
        row_buckets=row_buckets,
        process_cpu_ms=process_cpu_ms,
        process_gpu_ms=process_gpu_ms,
        long_row_count=long_row_count,
    )


def _spmm_csr_alg2_kernel_bundle():
    from .spmm_csr_opt_alg2 import (
        _spmm_csr_alg2_batched_rows_kernel,
        _spmm_csr_alg2_row_rows_kernel,
        _spmm_csr_alg2_segmented_rows_kernel,
    )

    return (
        _spmm_csr_alg2_batched_rows_kernel,
        _spmm_csr_alg2_row_rows_kernel,
        _spmm_csr_alg2_segmented_rows_kernel,
    )


def _spmm_csr_alg2_acc_dtype(dtype):
    return tl.float64 if dtype == torch.float64 else tl.float32


def _spmm_csr_alg2_run_bucket(plan, bucket, B, C_out, device_props, kernels):
    launch = _resolve_spmm_csr_alg2_launch(
        bucket,
        int(B.shape[1]),
        plan.data.dtype,
        device_props,
    )
    rows = bucket["rows"]
    if rows.numel() == 0:
        return launch

    batched_kernel, row_kernel, segmented_kernel = kernels
    acc_dtype = _spmm_csr_alg2_acc_dtype(plan.data.dtype)
    out_dtype = tl.float64 if plan.data.dtype == torch.float64 else tl.float32
    common_kwargs = {
        "num_warps": launch["num_warps"],
        "num_stages": launch["num_stages"],
    }

    if bucket["kind"] == "batched2d":
        grid = (
            triton.cdiv(rows.numel(), launch["batch_rows"]),
            triton.cdiv(B.shape[1], launch["block_n"]),
        )
        batched_kernel[grid](
            plan.data,
            plan.kernel_indices,
            plan.kernel_indptr,
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
        row_kernel[grid](
            plan.data,
            plan.kernel_indices,
            plan.kernel_indptr,
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
        segmented_kernel[grid](
            plan.data,
            plan.kernel_indices,
            plan.kernel_indptr,
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


def _spmm_csr_alg2_compute(plan, B):
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != plan.data.device:
        raise ValueError("B must be on the same CUDA device as sparse matrix data")
    if B.dtype != plan.data.dtype:
        raise TypeError("B dtype must match sparse matrix dtype")
    if B.shape[0] != plan.n_cols:
        raise ValueError(f"B.shape[0] must be n_cols={plan.n_cols}, got {B.shape[0]}")

    B = B.contiguous()
    C_out = torch.zeros((plan.n_rows, int(B.shape[1])), dtype=B.dtype, device=B.device)
    device_props = _normalize_spmm_csr_alg2_device_props(plan.data.device)
    kernels = _spmm_csr_alg2_kernel_bundle()
    plan.launch_configs.clear()
    for bucket in plan.row_buckets:
        launch = _spmm_csr_alg2_run_bucket(plan, bucket, B, C_out, device_props, kernels)
        plan.launch_configs.append(launch)
    return C_out


def _run_spmm_csr_alg2_route(prepared, B, *, timing=False, diagnostics=False):
    B = _validate_spmm_route_runtime_inputs(prepared, B)
    plan = _spmm_csr_alg2_build_process_plan(prepared, timing=bool(timing))
    compute_ms = None
    if timing:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    C = _spmm_csr_alg2_compute(plan, B)
    if timing:
        end.record()
        torch.cuda.synchronize()
        compute_ms = start.elapsed_time(end)
    meta = {
        "alg": "spmm_csr_alg2",
        "display_name": "Alg2",
        "op": prepared.op,
        "process_cpu_ms": plan.process_cpu_ms,
        "process_gpu_ms": plan.process_gpu_ms if timing else None,
        "compute_ms": compute_ms,
    }
    if diagnostics:
        first_launch = plan.launch_configs[0] if plan.launch_configs else {}
        meta["diagnostics"] = {
            "launch_config_scope": "bucket",
            "launch_config_count": len(plan.launch_configs),
            "bucket_count": plan.bucket_count,
            "long_row_count": plan.long_row_count,
            "long_part_count": 0,
            "launch_version": "spmm_csr_alg2_v1",
            "block_n": first_launch.get("block_n"),
            "block_nnz": first_launch.get("block_k"),
            "num_warps": first_launch.get("num_warps"),
            "num_stages": first_launch.get("num_stages"),
            "grid_m": first_launch.get("grid_m"),
            "grid_n": first_launch.get("grid_n"),
        }
    return C, meta


SPMM_CSR_ALGORITHMS = {
    "csr_base": SpmmCsrAlgorithm(
        name="csr_base",
        display_name="Base",
        supported_ops=tuple(SPMM_OP_NAMES.values()),
        supported_dtypes=SUPPORTED_SPMM_VALUE_DTYPES,
        run=_run_spmm_csr_base_route,
    ),
    "alpha_alg1_tle_opt": SpmmCsrAlgorithm(
        name="alpha_alg1_tle_opt",
        display_name="TLEOpt",
        supported_ops=tuple(SPMM_OP_NAMES.values()),
        supported_dtypes=(torch.float32, torch.float64),
        run=_run_alpha_spmm_alg1_tle_opt_route,
    ),
    "alpha_alg1_tle_opt2": SpmmCsrAlgorithm(
        name="alpha_alg1_tle_opt2",
        display_name="TLEOpt2",
        supported_ops=tuple(SPMM_OP_NAMES.values()),
        supported_dtypes=(torch.float32, torch.float64),
        run=_run_alpha_spmm_alg1_tle_opt2_route,
    ),
    "spmm_csr_alg1": SpmmCsrAlgorithm(
        name="spmm_csr_alg1",
        display_name="Alg1",
        supported_ops=tuple(SPMM_OP_NAMES.values()),
        supported_dtypes=(torch.float32, torch.float64),
        run=_run_spmm_csr_alg1_route,
    ),
    "spmm_csr_alg2": SpmmCsrAlgorithm(
        name="spmm_csr_alg2",
        display_name="Alg2",
        supported_ops=tuple(SPMM_OP_NAMES.values()),
        supported_dtypes=(torch.float32, torch.float64),
        run=_run_spmm_csr_alg2_route,
    ),
}


def resolve_spmm_csr_algorithm(alg, op, dtype):
    token = _normalize_spmm_csr_alg(alg)
    if token == "auto":
        token = "csr_base"
    if token not in SPMM_CSR_ALGORITHMS:
        supported = ", ".join(sorted(SPMM_CSR_ALGORITHMS))
        raise ValueError(f"unsupported CSR SpMM algorithm {alg!r}; supported: auto, {supported}")
    algorithm = SPMM_CSR_ALGORITHMS[token]
    op_name = _spmm_op_to_name(op)
    if op_name not in algorithm.supported_ops:
        raise ValueError(f"algorithm {token!r} does not support op {op_name!r}")
    if dtype not in algorithm.supported_dtypes:
        raise TypeError(f"algorithm {token!r} does not support dtype {dtype}")
    return algorithm


def list_spmm_csr_algorithms(op=None, dtype=None):
    op_name = None if op is None else _spmm_op_to_name(op)
    names = []
    for name, algorithm in SPMM_CSR_ALGORITHMS.items():
        if op_name is not None and op_name not in algorithm.supported_ops:
            continue
        if dtype is not None and dtype not in algorithm.supported_dtypes:
            continue
        names.append(name)
    return tuple(names)


def prepare_spmm_csr_route(data, indices, indptr, shape, *, op="non", alg="auto"):
    """Prepare matrix-level CSR SpMM metadata for the route-based run API."""
    op_code = _normalize_spmm_op(op)
    op_name = _spmm_op_to_name(op_code)
    (
        data,
        kernel_indices,
        kernel_indptr,
        n_rows,
        n_cols,
        row_lengths,
        max_row_nnz,
    ) = _prepare_spmm_csr_matrix(data, indices, indptr, shape)
    resolved_alg = _normalize_spmm_csr_alg(alg)
    if resolved_alg != "auto":
        resolve_spmm_csr_algorithm(resolved_alg, op_name, data.dtype)
    return PreparedCsrSpmmRoute(
        data=data,
        kernel_indices=kernel_indices,
        kernel_indptr=kernel_indptr,
        shape=shape,
        n_rows=n_rows,
        n_cols=n_cols,
        row_lengths=row_lengths,
        max_row_nnz=max_row_nnz,
        op=op_name,
        alg=resolved_alg,
    )


def flagsparse_spmm_csr_run(
    prepared,
    B,
    *,
    alg=None,
    op=None,
    dense_layout="auto",
    return_time=False,
    return_meta=False,
    timing=False,
    diagnostics=False,
):
    """Run a registered CSR SpMM route.

    The default operator timing follows the AlphaSparse-style benchmark policy:
    ``gpu_ms`` is a CUDA event around the route run, while process CPU time is
    reported separately by the route. Stage timing is only collected when
    ``timing=True``.
    """
    if not isinstance(prepared, PreparedCsrSpmmRoute):
        raise TypeError("prepared must be a PreparedCsrSpmmRoute instance")
    op_name = prepared.op if op is None else _spmm_op_to_name(op)
    alg_name = prepared.alg if alg is None else _normalize_spmm_csr_alg(alg)
    algorithm = resolve_spmm_csr_algorithm(alg_name, op_name, prepared.data.dtype)
    dense_layout = _normalize_dense_layout(dense_layout)
    if dense_layout == "col" and algorithm.name != "csr_base":
        raise SpmmCsrAlgorithmUnavailable(
            "col-major layout is currently supported only by csr_base"
        )
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

    start = torch.cuda.Event(enable_timing=True) if (return_time or return_meta) else None
    end = torch.cuda.Event(enable_timing=True) if (return_time or return_meta) else None
    if start is not None:
        torch.cuda.synchronize()
        start.record()
    runtime_prepared, op_process_gpu_ms = _materialize_spmm_csr_route_op(
        prepared,
        op_name,
        timing=bool(timing),
    )
    if algorithm.name == "csr_base":
        C, route_meta = algorithm.run(
            runtime_prepared,
            B,
            timing=bool(timing),
            diagnostics=bool(diagnostics),
            dense_layout=dense_layout,
        )
    else:
        C, route_meta = algorithm.run(
            runtime_prepared,
            B,
            timing=bool(timing),
            diagnostics=bool(diagnostics),
        )
    if end is not None:
        end.record()
        torch.cuda.synchronize()
        gpu_ms = start.elapsed_time(end)
    else:
        gpu_ms = None

    process_cpu_ms = float(route_meta.get("process_cpu_ms", 0.0) or 0.0)
    route_process_gpu_ms = route_meta.get("process_gpu_ms")
    if timing:
        process_gpu_ms = float(op_process_gpu_ms or 0.0) + float(route_process_gpu_ms or 0.0)
    else:
        process_gpu_ms = None
    operator_ms = (process_cpu_ms + float(gpu_ms)) if gpu_ms is not None else None
    meta = None
    if return_meta:
        meta = {
            "alg": algorithm.name,
            "display_name": algorithm.display_name,
            "op": op_name,
            "operator_ms": operator_ms,
            "gpu_ms": gpu_ms,
            "process_cpu_ms": process_cpu_ms,
            "dense_layout": route_meta.get("dense_layout", dense_layout),
            "b_stride": route_meta.get("b_stride"),
            "c_stride": route_meta.get("c_stride"),
            "output_layout": route_meta.get("output_layout"),
        }
        if timing:
            meta["process_gpu_ms"] = process_gpu_ms
            meta["compute_ms"] = route_meta.get("compute_ms")
        if diagnostics and "diagnostics" in route_meta:
            meta["diagnostics"] = route_meta["diagnostics"]
    if return_time and return_meta:
        return C, operator_ms, meta
    if return_time:
        return C, operator_ms
    if return_meta:
        return C, meta
    return C


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

_SPMM_OPT_BUCKET_SPECS = (
    {"max_row_nnz": 32, "kind": "batched", "batch_rows": 8, "block_nnz": 32},
    {"max_row_nnz": 128, "kind": "batched", "batch_rows": 4, "block_nnz": 64},
    {"max_row_nnz": 512, "kind": "vector", "batch_rows": 1, "block_nnz": 128},
    {"max_row_nnz": _SPMM_OPT_LONG_ROW_THRESHOLD, "kind": "vector", "batch_rows": 1, "block_nnz": 128},
    {"max_row_nnz": None, "kind": "split", "batch_rows": 1, "block_nnz": _SPMM_OPT_SPLIT_BLOCK_NNZ},
)
_SPMM_CSR_ALG1_BUCKET_SPECS = (
    {"max_row_nnz": 32, "kind": "batched", "batch_rows": 8, "block_nnz": 32},
    {"max_row_nnz": 128, "kind": "batched", "batch_rows": 4, "block_nnz": 64},
    {"max_row_nnz": 512, "kind": "vector", "batch_rows": 1, "block_nnz": 128},
    {"max_row_nnz": _SPMM_OPT_LONG_ROW_THRESHOLD, "kind": "vector", "batch_rows": 1, "block_nnz": 128},
    {"max_row_nnz": None, "kind": "split", "batch_rows": 1, "block_nnz": _SPMM_OPT_SPLIT_BLOCK_NNZ},
)

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
    long_rows = torch.empty((0,), dtype=row_index_dtype, device=device)
    lower = 0
    for spec in _SPMM_OPT_BUCKET_SPECS:
        upper = spec["max_row_nnz"]
        if upper is None:
            mask = row_lengths > lower
            max_row_nnz = _SPMM_OPT_LONG_ROW_THRESHOLD + 1
            label = "split_long"
            execution = "split"
        elif lower == 0:
            mask = row_lengths <= upper
            max_row_nnz = upper
            label = f"{spec['kind']}_{upper}"
            execution = "legacy"
        else:
            mask = (row_lengths > lower) & (row_lengths <= upper)
            max_row_nnz = upper
            label = f"{spec['kind']}_{upper}"
            execution = "legacy"
        rows = all_rows[mask]
        if rows.numel() == 0:
            if upper is not None:
                lower = upper
            continue
        buckets.append(
            _spmm_opt_make_bucket(
                label,
                spec["kind"],
                rows,
                execution,
                lower + 1 if lower > 0 else 0,
                max_row_nnz,
                batch_rows=spec["batch_rows"],
                block_k=spec["block_nnz"],
                block_nnz=spec["block_nnz"],
                block_n_cap=128,
            )
        )
        if spec["kind"] == "split":
            long_rows = rows
        if upper is not None:
            lower = upper
    return buckets, long_rows


@triton.jit
def _spmm_opt_alg1_symbolic_count_kernel(
    row_lengths_ptr,
    bucket_counts_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_rows
    lens = tl.load(row_lengths_ptr + offs, mask=mask, other=0)
    bucket = tl.where(
        lens <= 32,
        0,
        tl.where(lens <= 128, 1, tl.where(lens <= 512, 2, tl.where(lens <= 2048, 3, 4))),
    )
    for bid in tl.static_range(0, 5):
        hits = mask & (bucket == bid)
        count = tl.sum(tl.where(hits, 1, 0), axis=0)
        tl.atomic_add(bucket_counts_ptr + bid, count, sem="relaxed")


@triton.jit
def _spmm_opt_alg1_symbolic_compact_kernel(
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
    bucket = tl.where(
        lens <= 32,
        0,
        tl.where(lens <= 128, 1, tl.where(lens <= 512, 2, tl.where(lens <= 2048, 3, 4))),
    )
    for bid in tl.static_range(0, 5):
        hits = mask & (bucket == bid)
        ranks = tl.cumsum(tl.where(hits, 1, 0), axis=0) - 1
        local_count = tl.sum(tl.where(hits, 1, 0), axis=0)
        base = tl.atomic_add(bucket_write_counts_ptr + bid, local_count, sem="relaxed")
        offset = tl.load(bucket_offsets_ptr + bid)
        tl.store(rows_flat_ptr + offset + base + ranks, offs, mask=hits)


def _build_spmm_opt_alg1_buckets_triton_symbolic(row_lengths, dtype, nnz=None):
    device = row_lengths.device
    row_count = int(row_lengths.numel())
    row_index_dtype = torch.int32 if row_count <= _INDEX_LIMIT_INT32 else torch.int64
    bucket_count = len(_SPMM_OPT_BUCKET_SPECS)
    if bucket_count != 5:
        raise RuntimeError("Alg1 Triton symbolic builder expects five alg1 buckets")

    counts = torch.zeros((bucket_count,), dtype=torch.int64, device=device)
    block_m = 256
    grid = (triton.cdiv(row_count, block_m),)
    if row_count > 0:
        _spmm_opt_alg1_symbolic_count_kernel[grid](
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
        _spmm_opt_alg1_symbolic_compact_kernel[grid](
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
    long_rows = torch.empty((0,), dtype=row_index_dtype, device=device)
    lower = 0
    for spec, count, offset in zip(_SPMM_OPT_BUCKET_SPECS, counts_cpu, offsets_cpu):
        upper = spec["max_row_nnz"]
        count = int(count)
        if count == 0:
            if upper is not None:
                lower = upper
            continue
        rows = rows_flat.narrow(0, int(offset), count)
        if upper is None:
            max_row_nnz = _SPMM_OPT_LONG_ROW_THRESHOLD + 1
            label = "split_long"
            execution = "split"
        else:
            max_row_nnz = upper
            label = f"{spec['kind']}_{upper}"
            execution = "legacy"
        buckets.append(
            _spmm_opt_make_bucket(
                label,
                spec["kind"],
                rows,
                execution,
                lower + 1 if lower > 0 else 0,
                max_row_nnz,
                batch_rows=spec["batch_rows"],
                block_k=spec["block_nnz"],
                block_nnz=spec["block_nnz"],
                block_n_cap=128,
            )
        )
        if spec["kind"] == "split":
            long_rows = rows
        if upper is not None:
            lower = upper
    return buckets, long_rows


def prepare_spmm_csr_opt_alg1(data, indices, indptr, shape):
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


def prepare_spmm_csr_opt(data, indices, indptr, shape):
    return prepare_spmm_csr_opt_alg1(data, indices, indptr, shape)


def prepare_spmm_csr_opt_alg1_preprocess(data, indices, indptr, shape):
    return prepare_spmm_csr_opt_alg1(data, indices, indptr, shape)


def _build_spmm_csr_opt_runtime_symbolic_with_builder(prepared, bucket_builder):
    row_buckets, long_rows = bucket_builder(
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


def _build_spmm_csr_opt_runtime_symbolic(prepared):
    return _build_spmm_csr_opt_runtime_symbolic_with_builder(
        prepared,
        _build_spmm_opt_buckets,
    )


def _build_spmm_csr_opt_runtime_symbolic_triton(prepared):
    return _build_spmm_csr_opt_runtime_symbolic_with_builder(
        prepared,
        _build_spmm_opt_alg1_buckets_triton_symbolic,
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
    num_warps,
    num_stages,
    out=None,
    dense_layout="row",
):
    device = data.device
    dtype = data.dtype
    dense_layout = _normalize_dense_layout(dense_layout)
    if out is not None:
        if out.shape != (int(n_rows), int(n_dense_cols)) or out.dtype != dtype:
            raise ValueError("out shape/dtype must match result")
        if out.device != device:
            raise ValueError("out must be on the same CUDA device as data")
    if n_rows == 0 or n_dense_cols == 0 or B.shape[0] == 0:
        if out is not None:
            out.zero_()
            return out
        return _zeros_dense_layout((n_rows, n_dense_cols), dtype, device, dense_layout)

    if not _is_complex_dtype(dtype):
        compute_dtype = dtype
        data_in = data
        B_in = B
        if dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
            data_in = data.to(torch.float32)
            B_in = _materialize_dense_layout(B.to(torch.float32), dense_layout)
        elif dtype == torch.float32:
            compute_dtype = torch.float64
            data_in = data.to(torch.float64)
            B_in = _materialize_dense_layout(B.to(torch.float64), dense_layout)

        C_compute = (
            out
            if out is not None and compute_dtype == dtype
            else _empty_dense_layout((n_rows, n_dense_cols), compute_dtype, device, dense_layout)
        )
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
            num_warps=num_warps,
            num_stages=num_stages,
        )
        if compute_dtype != dtype:
            C_cast = C_compute.to(dtype)
            if out is not None:
                out.copy_(C_cast)
                return out
            if dense_layout == "col":
                C_out = _empty_dense_layout((n_rows, n_dense_cols), dtype, device, dense_layout)
                C_out.copy_(C_cast)
                return C_out
            return C_cast
        return C_compute

    if dense_layout == "col" or (out is not None and _is_col_major_2d(out)):
        raise SpmmCsrAlgorithmUnavailable("csr_base col-major layout does not support complex dtypes yet")
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
        num_warps=num_warps,
        num_stages=num_stages,
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


def _run_spmm_opt_bucket(prepared, bucket, B, C_out, block_n, device_props):
    rows = bucket["rows"]
    if rows.numel() == 0:
        return
    dtype = prepared.data.dtype
    kind = bucket["kind"]
    kernel_map = {
        ("batched", torch.float32): _spmm_csr_batched_rows_f32_kernel,
        ("batched", torch.float64): _spmm_csr_batched_rows_f64_kernel,
        ("vector", torch.float32): _spmm_csr_vector_rows_f32_kernel,
        ("vector", torch.float64): _spmm_csr_vector_rows_f64_kernel,
    }
    if (kind, dtype) not in kernel_map:
        raise TypeError(f"unsupported SpMM opt bucket kind/dtype: {kind}/{dtype}")

    batch_rows = int(bucket.get("batch_rows", 1))
    block_n = _select_spmm_opt_block_n_for_bucket(
        int(B.shape[1]), bucket.get("block_n_cap", block_n)
    )
    block_nnz = int(bucket["block_nnz"])
    launch = _resolve_spmm_opt_launch(
        kind,
        block_n,
        block_nnz,
        batch_rows,
        dtype,
        device_props,
    )
    kernel = kernel_map[(kind, dtype)]

    if kind == "batched":
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


def _run_spmm_opt_split_bucket(prepared, B, C_out, block_n, device_props):
    if prepared.long_part_rows.numel() == 0:
        return False
    block_n = _select_spmm_opt_block_n_for_bucket(int(B.shape[1]), block_n)
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
        "split_part",
        block_n,
        _SPMM_OPT_SPLIT_BLOCK_NNZ,
        1,
        B.dtype,
        device_props,
    )
    reduce_launch = _resolve_spmm_opt_launch(
        "split_reduce",
        block_n,
        _SPMM_OPT_SPLIT_BLOCK_NNZ,
        1,
        B.dtype,
        device_props,
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
    transpose=None,
    op=None,
    return_meta=False,
):
    """CSR SpMM using Triton.

    op: 0/'non' for A @ B, 1/'trans' for A.T @ B,
    2/'conj' for A.conj().T @ B.
    """
    op_explicit = op is not None
    op_code = _normalize_spmm_op(
        op,
        transpose=False if transpose is None else bool(transpose),
    )
    if (
        op_explicit
        and transpose is not None
        and bool(transpose) != _spmm_op_transposes(op_code)
    ):
        raise ValueError("transpose conflicts with op")
    if block_n is not None and block_n <= 0:
        raise ValueError("block_n must be positive when provided")
    if block_nnz is not None and block_nnz <= 0:
        raise ValueError("block_nnz must be positive when provided")
    if max_segments is not None and max_segments <= 0:
        raise ValueError("max_segments must be positive when provided")

    (
        data,
        kernel_indices,
        kernel_indptr,
        n_rows,
        n_cols,
        _row_lengths,
        _max_row_nnz,
    ) = _prepare_spmm_csr_matrix(data, indices, indptr, shape)
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense tensor")
    effective_n_rows = n_rows
    effective_n_cols = n_cols
    if _spmm_op_transposes(op_code):
        effective_n_rows, effective_n_cols = n_cols, n_rows
    if B.shape[0] != effective_n_cols:
        raise ValueError(
            f"B.shape[0] must be n_cols={effective_n_cols}, got {B.shape[0]}"
        )
    if not B.is_cuda:
        raise ValueError("B must be a CUDA tensor")
    if B.device != data.device:
        raise ValueError("B must be on the same CUDA device as the sparse matrix")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")
    B = B.contiguous()
    n_dense_cols = int(B.shape[1])

    if out is not None:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.device != data.device:
            raise ValueError("out must be on the same CUDA device as the inputs")
        if out.shape != (effective_n_rows, n_dense_cols) or out.dtype != data.dtype:
            raise ValueError("out shape/dtype must match result")

    do_timing = bool(return_time or return_meta)
    symbolic_ms = 0.0 if do_timing else None
    compute_ms = None
    op_total_ms = None

    if do_timing:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    if _spmm_op_transposes(op_code):
        data, indices, indptr, shape = _materialize_spmm_csr_op(
            data,
            indices,
            indptr,
            shape,
            op_code,
        )
        (
            data,
            kernel_indices,
            kernel_indptr,
            n_rows,
            n_cols,
            _row_lengths,
            _max_row_nnz,
        ) = _prepare_spmm_csr_matrix(data, indices, indptr, shape)
    if do_timing:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        symbolic_ms = (t1 - t0) * 1000.0 if _spmm_op_transposes(op_code) else 0.0

    max_row_nnz = (
        int(torch.max(kernel_indptr[1:] - kernel_indptr[:-1]).item())
        if n_rows > 0
        else 0
    )
    device_props = _normalize_spmm_base_device_props(data.device)
    launch = _resolve_spmm_base_triton_launch(
        data.dtype,
        n_dense_cols,
        max_row_nnz,
        block_n=block_n,
        block_nnz=block_nnz,
        max_segments=max_segments,
        device_props=device_props,
    )

    if do_timing:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    C = _triton_spmm_csr_impl(
        data,
        kernel_indices,
        kernel_indptr,
        B,
        n_rows,
        n_dense_cols,
        block_n=launch["block_n"],
        block_nnz=launch["block_nnz"],
        num_warps=launch["num_warps"],
        num_stages=launch["num_stages"],
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
        meta = {
            "symbolic_ms": symbolic_ms,
            "compute_ms": compute_ms,
            "op_total_ms": op_total_ms,
            "op": _spmm_op_to_name(op_code),
        }
        if return_time:
            return C, op_total_ms, meta
        return C, meta
    if return_time:
        return C, op_total_ms
    return C


def _flagsparse_spmm_csr_opt_alg1_impl(
    data=None,
    indices=None,
    indptr=None,
    B=None,
    shape=None,
    prepared=None,
    out=None,
    return_time=False,
    return_meta=False,
    runtime_symbolic_builder=_build_spmm_csr_opt_runtime_symbolic_triton,
    api_name="flagsparse_spmm_csr_opt_alg1",
):
    if prepared is not None and not isinstance(prepared, PreparedCsrSpmmOpt):
        raise TypeError("prepared must be a PreparedCsrSpmmOpt instance")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_spmm_csr_opt_alg1(data, indices, indptr, shape)
    elif shape is not None:
        resolved_shape = (int(shape[0]), int(shape[1]))
        if resolved_shape != prepared.shape:
            raise ValueError(
                f"shape {resolved_shape} does not match prepared.shape {prepared.shape}"
            )
    if B is None:
        raise ValueError("B is required")
    if not prepared.supports_opt:
        raise TypeError(f"{api_name} only supports float32 and float64")
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
    runtime_prepared = runtime_symbolic_builder(prepared)
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


def flagsparse_spmm_csr_opt_alg1(
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
    """CSR SpMM-opt alg1: native float32/float64 bucketed path for CSR @ dense."""
    return _flagsparse_spmm_csr_opt_alg1_impl(
        data=data,
        indices=indices,
        indptr=indptr,
        B=B,
        shape=shape,
        prepared=prepared,
        out=out,
        return_time=return_time,
        return_meta=return_meta,
        runtime_symbolic_builder=_build_spmm_csr_opt_runtime_symbolic_triton,
        api_name="flagsparse_spmm_csr_opt_alg1",
    )


def flagsparse_spmm_csr_opt_alg1_preprocess(
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
    """Compatibility alias for CSR SpMM-opt alg1 with runtime preprocessing."""
    return _flagsparse_spmm_csr_opt_alg1_impl(
        data=data,
        indices=indices,
        indptr=indptr,
        B=B,
        shape=shape,
        prepared=prepared,
        out=out,
        return_time=return_time,
        return_meta=return_meta,
        runtime_symbolic_builder=_build_spmm_csr_opt_runtime_symbolic_triton,
        api_name="flagsparse_spmm_csr_opt_alg1_preprocess",
    )


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
    """Compatibility alias for CSR SpMM-opt alg1."""
    return flagsparse_spmm_csr_opt_alg1(
        data=data,
        indices=indices,
        indptr=indptr,
        B=B,
        shape=shape,
        prepared=prepared,
        out=out,
        return_time=return_time,
        return_meta=return_meta,
    )


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
    op="non",
):
    """Benchmark Triton CSR SpMM vs PyTorch sparse.mm and CuPy/cuSPARSE CSR @ dense."""
    op_code = _normalize_spmm_op(op)
    op_name = _spmm_op_to_name(op_code)
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_cols, nnz, value_dtype, index_dtype, device
    )
    b_rows = n_rows if _spmm_op_transposes(op_code) else n_cols
    B = _build_random_dense((b_rows, n_dense_cols), value_dtype, device)
    shape = (n_rows, n_cols)
    materialized = _materialize_spmm_csr_op(data, indices, indptr, shape, op_code)
    launch_indptr = materialized[2]
    launch_shape = materialized[3]
    launch_n_rows, _launch_n_cols = int(launch_shape[0]), int(launch_shape[1])
    max_row_nnz = (
        int(torch.max(launch_indptr[1:] - launch_indptr[:-1]).item())
        if launch_n_rows > 0
        else 0
    )
    launch = _resolve_spmm_base_triton_launch(
        value_dtype,
        n_dense_cols,
        max_row_nnz,
        block_n=block_n,
        block_nnz=block_nnz,
        max_segments=max_segments,
        device_props=_normalize_spmm_base_device_props(device),
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
        "op": op_name,
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
        if op_code == SPMM_OP_NON:
            pytorch_op = lambda: torch.sparse.mm(csr_pt, B)
        elif op_code == SPMM_OP_TRANS:
            pytorch_op = lambda: torch.sparse.mm(csr_pt.transpose(0, 1), B)
        else:
            pytorch_op = lambda: torch.sparse.mm(csr_pt.conj().transpose(0, 1), B)
        if value_dtype in (torch.float16, torch.bfloat16):
            csr_ref = torch.sparse_csr_tensor(indptr64, indices64, data.to(torch.float32), size=shape, device=device)
            expected = (
                torch.sparse.mm(csr_ref, B.to(torch.float32))
                if op_code == SPMM_OP_NON
                else torch.sparse.mm(
                    csr_ref.transpose(0, 1)
                    if op_code == SPMM_OP_TRANS
                    else csr_ref.conj().transpose(0, 1),
                    B.to(torch.float32),
                )
            ).to(value_dtype)
        elif value_dtype == torch.float32:
            csr_ref = torch.sparse_csr_tensor(indptr64, indices64, data.to(torch.float64), size=shape, device=device)
            expected = (
                torch.sparse.mm(csr_ref, B.to(torch.float64))
                if op_code == SPMM_OP_NON
                else torch.sparse.mm(
                    csr_ref.transpose(0, 1)
                    if op_code == SPMM_OP_TRANS
                    else csr_ref.conj().transpose(0, 1),
                    B.to(torch.float64),
                )
            ).to(value_dtype)
        elif value_dtype == torch.complex64:
            csr_ref = torch.sparse_csr_tensor(indptr64, indices64, data.to(torch.complex128), size=shape, device=device)
            expected = (
                torch.sparse.mm(csr_ref, B.to(torch.complex128))
                if op_code == SPMM_OP_NON
                else torch.sparse.mm(
                    csr_ref.transpose(0, 1)
                    if op_code == SPMM_OP_TRANS
                    else csr_ref.conj().transpose(0, 1),
                    B.to(torch.complex128),
                )
            ).to(value_dtype)
        else:
            expected = pytorch_op()
    except Exception as exc:
        pytorch_format = "COO"
        pytorch_reason = f"CSR fallback: {exc}"
        coo = torch.sparse_coo_tensor(
            torch.stack([row_indices, indices64]),
            data,
            shape,
            device=device,
        ).coalesce()
        if op_code == SPMM_OP_NON:
            pytorch_op = lambda: torch.sparse.mm(coo, B)
        elif op_code == SPMM_OP_TRANS:
            pytorch_op = lambda: torch.sparse.mm(coo.transpose(0, 1), B)
        else:
            pytorch_op = lambda: torch.sparse.mm(coo.conj().transpose(0, 1), B)
        if value_dtype in (torch.float16, torch.bfloat16):
            coo_ref = coo.to(torch.float32)
            expected = (
                torch.sparse.mm(coo_ref, B.to(torch.float32))
                if op_code == SPMM_OP_NON
                else torch.sparse.mm(
                    coo_ref.transpose(0, 1)
                    if op_code == SPMM_OP_TRANS
                    else coo_ref.conj().transpose(0, 1),
                    B.to(torch.float32),
                )
            ).to(value_dtype)
        elif value_dtype == torch.float32:
            coo_ref = coo.to(torch.float64)
            expected = (
                torch.sparse.mm(coo_ref, B.to(torch.float64))
                if op_code == SPMM_OP_NON
                else torch.sparse.mm(
                    coo_ref.transpose(0, 1)
                    if op_code == SPMM_OP_TRANS
                    else coo_ref.conj().transpose(0, 1),
                    B.to(torch.float64),
                )
            ).to(value_dtype)
        elif value_dtype == torch.complex64:
            coo_ref = coo.to(torch.complex128)
            expected = (
                torch.sparse.mm(coo_ref, B.to(torch.complex128))
                if op_code == SPMM_OP_NON
                else torch.sparse.mm(
                    coo_ref.transpose(0, 1)
                    if op_code == SPMM_OP_TRANS
                    else coo_ref.conj().transpose(0, 1),
                    B.to(torch.complex128),
                )
            ).to(value_dtype)
        else:
            expected = pytorch_op()

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
                A_eff = A_csr
                if op_code == SPMM_OP_TRANS:
                    A_eff = A_csr.transpose().tocsr()
                elif op_code == SPMM_OP_CONJ_TRANS:
                    A_eff = A_csr.transpose().conj().tocsr()
                cusparse_values_cp, cusparse_ms = _benchmark_cuda_op(
                    lambda: A_eff @ B_cp, warmup=warmup, iters=iters
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
            "op": op_name,
            "warmup": warmup,
            "iters": iters,
            "block_n": launch["block_n"],
            "block_nnz": launch["block_nnz"],
            "max_segments": launch["max_segments"],
            "required_segments": launch["required_segments"],
            "alg1_warp_size": launch["warp_size"],
            "alg1_factor": launch["factor"],
            "base_num_warps": launch["num_warps"],
            "base_num_stages": launch["num_stages"],
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
    op="non",
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
        op=op,
    )
