"""Gather and scatter (Triton kernels + cuSPARSE-style baselines)."""

from ._common import *

import triton
import triton.language as tl

SUPPORTED_SCATTER_VALUE_DTYPES = SUPPORTED_VALUE_DTYPES


def _scatter_dtype_error_message():
    supported = ", ".join(
        str(dtype).replace("torch.", "") for dtype in SUPPORTED_SCATTER_VALUE_DTYPES
    )
    return (
        "Scatter benchmark supports value dtypes: "
        f"{supported}."
    )


@triton.jit
def _gather_real_kernel(
    sparse_values_ptr,
    dense_values_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nnz
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    gathered_values = tl.load(dense_values_ptr + indices, mask=mask, other=0.0)
    tl.store(sparse_values_ptr + offsets, gathered_values, mask=mask)


@triton.jit
def _gather_complex_kernel(
    sparse_values_ri_ptr,
    dense_values_ri_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nnz
    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)

    dense_offsets = indices * 2
    sparse_offsets = offsets * 2

    gathered_real = tl.load(dense_values_ri_ptr + dense_offsets, mask=mask, other=0.0)
    gathered_imag = tl.load(dense_values_ri_ptr + dense_offsets + 1, mask=mask, other=0.0)

    tl.store(sparse_values_ri_ptr + sparse_offsets, gathered_real, mask=mask)
    tl.store(sparse_values_ri_ptr + sparse_offsets + 1, gathered_imag, mask=mask)


@triton.jit
def _scatter_real_kernel(
    dense_values_ptr,
    sparse_values_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nnz

    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    values = tl.load(sparse_values_ptr + offsets, mask=mask, other=0.0)
    tl.store(dense_values_ptr + indices, values, mask=mask)


@triton.jit
def _scatter_complex_kernel(
    dense_values_ri_ptr,
    sparse_values_ri_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nnz

    indices = tl.load(indices_ptr + offsets, mask=mask, other=0)
    dense_offsets = indices * 2
    sparse_offsets = offsets * 2

    values_real = tl.load(sparse_values_ri_ptr + sparse_offsets, mask=mask, other=0.0)
    values_imag = tl.load(sparse_values_ri_ptr + sparse_offsets + 1, mask=mask, other=0.0)

    tl.store(dense_values_ri_ptr + dense_offsets, values_real, mask=mask)
    tl.store(dense_values_ri_ptr + dense_offsets + 1, values_imag, mask=mask)


def _triton_gather_impl(dense_vector, kernel_indices, block_size=1024):
    nnz = kernel_indices.numel()
    if nnz == 0:
        return torch.empty(0, dtype=dense_vector.dtype, device=dense_vector.device)

    grid = lambda meta: (triton.cdiv(nnz, meta["BLOCK_SIZE"]),)

    if not _is_complex_dtype(dense_vector.dtype):
        sparse_values = torch.empty(nnz, dtype=dense_vector.dtype, device=dense_vector.device)
        _gather_real_kernel[grid](
            sparse_values,
            dense_vector,
            kernel_indices,
            nnz,
            BLOCK_SIZE=block_size,
        )
        return sparse_values

    sparse_values = torch.empty(nnz, dtype=dense_vector.dtype, device=dense_vector.device)
    dense_values_ri = torch.view_as_real(dense_vector).reshape(-1)
    sparse_values_ri = torch.view_as_real(sparse_values).reshape(-1)

    _gather_complex_kernel[grid](
        sparse_values_ri,
        dense_values_ri,
        kernel_indices,
        nnz,
        BLOCK_SIZE=block_size,
    )
    return sparse_values


def _triton_scatter_impl(
    sparse_values,
    kernel_indices,
    dense_size,
    out=None,
    block_size=1024,
    reset_output=True,
    index_fallback_policy="auto",
    return_metadata=False,
):
    index_fallback_policy = str(index_fallback_policy).lower()
    if index_fallback_policy not in ("auto", "strict"):
        raise ValueError("index_fallback_policy must be 'auto' or 'strict'")

    if out is None:
        dense_values = torch.zeros(
            dense_size, dtype=sparse_values.dtype, device=sparse_values.device
        )
    else:
        dense_values = out
        if reset_output:
            dense_values.zero_()

    nnz = kernel_indices.numel()
    scatter_meta = {
        "index_fallback_applied": False,
        "index_fallback_reason": None,
        "kernel_index_dtype": str(kernel_indices.dtype).replace("torch.", ""),
    }
    if nnz == 0:
        if return_metadata:
            return dense_values, scatter_meta
        return dense_values

    try:
        _launch_triton_scatter_kernel(
            dense_values,
            sparse_values,
            kernel_indices,
            nnz,
            block_size=block_size,
        )
    except Exception as exc:
        if kernel_indices.dtype != torch.int64 or index_fallback_policy != "auto":
            raise RuntimeError(
                f"Triton scatter failed for index dtype {kernel_indices.dtype}: "
                f"{exc.__class__.__name__}: {str(exc)}"
            ) from exc

        max_index = int(kernel_indices.max().item()) if nnz > 0 else -1
        if max_index > _INDEX_LIMIT_INT32:
            raise RuntimeError(
                "Triton scatter failed for int64 indices, and int32 fallback is invalid: "
                f"max index {max_index} exceeds int32 range"
            ) from exc

        fallback_indices = kernel_indices.to(torch.int32)
        try:
            _launch_triton_scatter_kernel(
                dense_values,
                sparse_values,
                fallback_indices,
                nnz,
                block_size=block_size,
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                "Triton scatter failed for int64 indices, and int32 fallback also failed: "
                f"{fallback_exc.__class__.__name__}: {str(fallback_exc)}"
            ) from fallback_exc

        scatter_meta["index_fallback_applied"] = True
        scatter_meta["index_fallback_reason"] = (
            f"int64 kernel launch failed: {exc.__class__.__name__}: {str(exc)}"
        )
        scatter_meta["kernel_index_dtype"] = "int32"

    if return_metadata:
        return dense_values, scatter_meta
    return dense_values


def _launch_triton_scatter_kernel(
    dense_values, sparse_values, kernel_indices, nnz, block_size=1024
):
    grid = lambda meta: (triton.cdiv(nnz, meta["BLOCK_SIZE"]),)
    if not _is_complex_dtype(sparse_values.dtype):
        _scatter_real_kernel[grid](
            dense_values,
            sparse_values,
            kernel_indices,
            nnz,
            BLOCK_SIZE=block_size,
        )
        return
    dense_values_ri = torch.view_as_real(dense_values).reshape(-1)
    sparse_values_ri = torch.view_as_real(sparse_values).reshape(-1)
    _scatter_complex_kernel[grid](
        dense_values_ri,
        sparse_values_ri,
        kernel_indices,
        nnz,
        BLOCK_SIZE=block_size,
    )


def _validate_gather_value_dtype(dense_vector, op_name):
    return None


def _cusparse_spmv(selector_matrix, dense_vector):
    if cp is not None and cpx_sparse is not None and isinstance(selector_matrix, cpx_sparse.spmatrix):
        if torch.is_tensor(dense_vector):
            out_dtype = dense_vector.dtype
            dense_for_compute = (
                dense_vector.to(torch.float32)
                if dense_vector.dtype == torch.bfloat16
                else dense_vector
            )
            dense_cp = _cupy_from_torch(dense_for_compute)
            out_cp = selector_matrix @ dense_cp
            out_torch = _torch_from_cupy(out_cp)
            if out_dtype == torch.bfloat16:
                out_torch = out_torch.to(torch.bfloat16)
            return out_torch

        if cp is not None and isinstance(dense_vector, cp.ndarray):
            return selector_matrix @ dense_vector

        raise TypeError("dense_vector must be torch.Tensor or cupy.ndarray")

    # Fallback path: torch sparse SpMV (still CUDA-backed).
    if torch.is_tensor(selector_matrix) and selector_matrix.is_sparse:
        if not torch.is_tensor(dense_vector):
            raise TypeError("dense_vector must be torch.Tensor for torch sparse fallback")
        out_dtype = dense_vector.dtype
        dense_for_compute = (
            dense_vector.to(torch.float32)
            if dense_vector.dtype == torch.bfloat16
            else dense_vector
        )
        out = torch.sparse.mm(selector_matrix, dense_for_compute.unsqueeze(1)).squeeze(1)
        if out_dtype == torch.bfloat16:
            out = out.to(torch.bfloat16)
        return out

    if cp is None or cpx_sparse is None:
        raise RuntimeError(
            "CuPy is not available and torch sparse fallback selector is not provided"
        )
    raise TypeError(
        "selector_matrix must be a cupyx sparse matrix or torch sparse tensor"
    )


def _make_gather_selector_matrix(indices, dense_size, value_dtype):
    if cp is not None and cpx_sparse is not None:
        rows_cp = cp.arange(indices.numel(), dtype=cp.int64)
        cols_cp = _cupy_from_torch(indices.to(torch.int64))
        vals_cp = cp.ones(indices.numel(), dtype=_cupy_dtype_from_torch(value_dtype))
        return cpx_sparse.coo_matrix(
            (vals_cp, (rows_cp, cols_cp)),
            shape=(indices.numel(), dense_size),
        )

    rows = torch.arange(indices.numel(), dtype=torch.int64, device=indices.device)
    cols = indices.to(torch.int64)
    coords = torch.stack([rows, cols], dim=0)
    values = torch.ones(indices.numel(), dtype=value_dtype, device=indices.device)
    return torch.sparse_coo_tensor(
        coords, values, size=(indices.numel(), dense_size), device=indices.device
    ).coalesce()


def _make_scatter_selector_matrix(indices, dense_size, value_dtype):
    if cp is not None and cpx_sparse is not None:
        rows_cp = _cupy_from_torch(indices.to(torch.int64))
        cols_cp = cp.arange(indices.numel(), dtype=cp.int64)
        vals_cp = cp.ones(indices.numel(), dtype=_cupy_dtype_from_torch(value_dtype))
        return cpx_sparse.coo_matrix(
            (vals_cp, (rows_cp, cols_cp)),
            shape=(dense_size, indices.numel()),
        )

    rows = indices.to(torch.int64)
    cols = torch.arange(indices.numel(), dtype=torch.int64, device=indices.device)
    coords = torch.stack([rows, cols], dim=0)
    values = torch.ones(indices.numel(), dtype=value_dtype, device=indices.device)
    return torch.sparse_coo_tensor(
        coords, values, size=(dense_size, indices.numel()), device=indices.device
    ).coalesce()


def _pytorch_scatter_impl(sparse_values, indices, dense_size, out=None, reset_output=True):
    if out is None:
        dense_values = torch.zeros(
            dense_size, dtype=sparse_values.dtype, device=sparse_values.device
        )
    else:
        dense_values = out
        if reset_output:
            dense_values.zero_()
    dense_values.index_copy_(0, indices.to(torch.int64), sparse_values)
    return dense_values


def flagsparse_gather(a, indices, out=None, mode="raise", block_size=1024, return_time=False):
    """CuPy-style gather (take): out = a[indices]."""
    if mode != "raise":
        raise NotImplementedError("Only mode='raise' is currently supported")

    dense_vector, dense_backend = _to_torch_tensor(a, "a")
    indices_tensor, _ = _to_torch_tensor(indices, "indices")
    dense_vector, indices_tensor, kernel_indices = _prepare_inputs(dense_vector, indices_tensor)
    _validate_gather_value_dtype(dense_vector, "flagsparse_gather")

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    sparse_values = _triton_gather_impl(dense_vector, kernel_indices, block_size=block_size)
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0

    if out is not None:
        out_tensor, _ = _to_torch_tensor(out, "out")
        if out_tensor.shape != sparse_values.shape:
            raise ValueError("out shape must match gather output shape")
        if out_tensor.dtype != sparse_values.dtype:
            raise TypeError("out dtype must match gather output dtype")
        out_tensor.copy_(sparse_values)
        result = out if dense_backend == "cupy" else out_tensor
    else:
        result = _to_backend_like(sparse_values, a)

    if return_time:
        return result, execution_time_ms
    return result


def flagsparse_scatter(
    a,
    indices,
    values,
    mode="raise",
    block_size=1024,
    return_time=False,
    reset_output=True,
    dtype_policy="auto",
    index_fallback_policy="auto",
):
    """CuPy-style scatter (put): a[indices] = values (in-place)."""
    if mode != "raise":
        raise NotImplementedError("Only mode='raise' is currently supported")

    dense_tensor, dense_backend = _to_torch_tensor(a, "a")
    values_tensor, _ = _to_torch_tensor(values, "values")
    indices_tensor, _ = _to_torch_tensor(indices, "indices")
    values_tensor, _, kernel_indices, dense_size, _ = _prepare_scatter_inputs(
        values_tensor,
        indices_tensor,
        dense_size=dense_tensor.numel(),
        out=dense_tensor,
        dtype_policy=dtype_policy,
        return_metadata=True,
    )

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    _ = _triton_scatter_impl(
        values_tensor,
        kernel_indices,
        dense_size=dense_size,
        out=dense_tensor,
        block_size=block_size,
        reset_output=reset_output,
        index_fallback_policy=index_fallback_policy,
    )
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0

    if dense_backend == "cupy":
        # DLPack view updates dense_tensor and cupy array shares memory.
        pass

    if return_time:
        return execution_time_ms
    return None


# Backward compatibility wrappers.
def triton_cusparse_gather(dense_vector, indices, block_size=1024):
    return flagsparse_gather(
        dense_vector, indices, block_size=block_size, return_time=True
    )


def triton_cusparse_scatter(
    sparse_values,
    indices,
    dense_size=None,
    out=None,
    block_size=1024,
    reset_output=True,
    dtype_policy="auto",
    index_fallback_policy="auto",
):
    sparse_values_t, sparse_backend = _to_torch_tensor(sparse_values, "sparse_values")
    indices_t, _ = _to_torch_tensor(indices, "indices")
    if out is None:
        if dense_size is None:
            dense_size = int(indices_t.max().item()) + 1 if indices_t.numel() > 0 else 0
        out = torch.zeros(
            int(dense_size), dtype=sparse_values_t.dtype, device=sparse_values_t.device
        )
    elapsed_ms = flagsparse_scatter(
        out,
        indices_t,
        sparse_values_t,
        block_size=block_size,
        return_time=True,
        reset_output=reset_output,
        dtype_policy=dtype_policy,
        index_fallback_policy=index_fallback_policy,
    )
    if sparse_backend == "cupy":
        return _to_backend_like(out, sparse_values), elapsed_ms
    return out, elapsed_ms


def pytorch_index_gather(dense_vector, indices):
    """Baseline gather using PyTorch native indexing."""
    dense_vector, indices, _ = _prepare_inputs(dense_vector, indices)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    sparse_values = dense_vector[indices]
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    return sparse_values, execution_time_ms


def pytorch_index_scatter(
    sparse_values, indices, dense_size=None, out=None, reset_output=True, dtype_policy="auto"
):
    """Baseline scatter using PyTorch index_copy_."""
    sparse_values, indices, _, dense_size, _ = _prepare_scatter_inputs(
        sparse_values,
        indices,
        dense_size=dense_size,
        out=out,
        dtype_policy=dtype_policy,
        return_metadata=True,
    )
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    dense_values = _pytorch_scatter_impl(
        sparse_values, indices, dense_size, out=out, reset_output=reset_output
    )
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    return dense_values, execution_time_ms


def cusparse_spmv_gather(dense_vector, indices, selector_matrix=None):
    """Equivalent gather baseline via cuSPARSE-backed COO SpMV."""
    dense_vector, indices, _ = _prepare_inputs(dense_vector, indices)
    _validate_gather_value_dtype(dense_vector, "cusparse_spmv_gather")
    skip_reason = _cusparse_baseline_skip_reason(dense_vector.dtype)
    if skip_reason:
        raise RuntimeError(skip_reason)

    if selector_matrix is None:
        selector_matrix = _make_gather_selector_matrix(
            indices, dense_vector.numel(), dense_vector.dtype
        )

    try:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        sparse_values = _cusparse_spmv(selector_matrix, dense_vector)
        torch.cuda.synchronize()
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    except Exception as exc:
        raise RuntimeError(
            "cuSPARSE gather baseline is unavailable in this PyTorch/CUDA environment"
        ) from exc

    return sparse_values, execution_time_ms, selector_matrix


def cusparse_spmv_scatter(
    sparse_values, indices, dense_size=None, selector_matrix=None, dtype_policy="auto"
):
    """Equivalent scatter baseline via cuSPARSE-backed COO SpMV."""
    sparse_values, indices, _, dense_size, _ = _prepare_scatter_inputs(
        sparse_values,
        indices,
        dense_size=dense_size,
        out=None,
        dtype_policy=dtype_policy,
        return_metadata=True,
    )
    skip_reason = _cusparse_baseline_skip_reason(sparse_values.dtype)
    if skip_reason:
        raise RuntimeError(skip_reason)

    if selector_matrix is None:
        selector_matrix = _make_scatter_selector_matrix(indices, dense_size, sparse_values.dtype)

    try:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        dense_values = _cusparse_spmv(selector_matrix, sparse_values)
        torch.cuda.synchronize()
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    except Exception as exc:
        raise RuntimeError(
            "cuSPARSE scatter baseline is unavailable in this PyTorch/CUDA environment"
        ) from exc

    return dense_values, execution_time_ms, selector_matrix
