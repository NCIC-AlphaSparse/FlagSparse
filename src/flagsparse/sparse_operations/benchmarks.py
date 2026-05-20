"""Benchmarks for gather, scatter, SpMV, SpMM, SpGEMM, SDDMM, and SpSM."""

from ._common import *

from .gather_scatter import (
    SUPPORTED_SCATTER_VALUE_DTYPES,
    _scatter_dtype_error_message,
    _cusparse_spmv,
    _make_gather_selector_matrix,
    _make_scatter_selector_matrix,
    _pytorch_scatter_impl,
    _triton_gather_impl,
    _triton_scatter_impl,
)
from .spmv_csr import (
    SPMV_OP_NON,
    SPMV_OP_TRANS,
    _normalize_spmv_op,
    _spmv_op_to_name,
    _spmv_op_transposes,
    flagsparse_spmv_csr,
    prepare_spmv_csr,
)
from .spmm_csr import (
    benchmark_spmm_case,
    benchmark_spmm_opt_case,
    comprehensive_spmm_test,
)
from .spmm_csr_opt_alg2 import benchmark_spmm_opt_alg2_case
from .spgemm_csr import benchmark_spgemm_case
from .sddmm_csr import benchmark_sddmm_case
from .spsm import benchmark_spsm_case


def _cupy_spmv_op_matrix(matrix, op_code):
    if op_code == SPMV_OP_NON:
        return matrix
    if op_code == SPMV_OP_TRANS:
        return matrix.T
    matrix_conj = matrix.conj() if hasattr(matrix, "conj") else matrix.conjugate()
    return matrix_conj.T


def _apply_cupy_spmv_op(matrix, vector, op_code):
    return _cupy_spmv_op_matrix(matrix, op_code) @ vector


def _apply_torch_sparse_spmv_op(matrix, vector_2d, op_code):
    if op_code == SPMV_OP_NON:
        return torch.sparse.mm(matrix, vector_2d).squeeze(1)
    if op_code == SPMV_OP_TRANS:
        return torch.sparse.mm(matrix.transpose(0, 1), vector_2d).squeeze(1)
    return torch.sparse.mm(matrix.conj().transpose(0, 1), vector_2d).squeeze(1)


def _normalize_dtype_name(value):
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).replace("torch.", "")


def _resolve_scatter_benchmark_dtype(value_dtype, dtype_policy):
    requested_name = _normalize_dtype_name(value_dtype)
    effective_dtype, fallback_applied, fallback_reason = _resolve_scatter_value_dtype(
        value_dtype, dtype_policy=dtype_policy
    )
    if effective_dtype not in SUPPORTED_SCATTER_VALUE_DTYPES:
        raise TypeError(_scatter_dtype_error_message())
    return requested_name, effective_dtype, bool(fallback_applied), fallback_reason


def benchmark_gather_case(
    dense_size=65536,
    nnz=4096,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_size=1024,
    run_cusparse=True,
):
    """Benchmark Triton vs PyTorch indexing vs cuSPARSE-backed COO SpMV."""
    device = torch.device("cuda")
    dense_vector = _build_random_dense(dense_size, value_dtype, device)
    indices = _build_indices(nnz, dense_size, index_dtype, device, unique=False)

    dense_vector, indices, kernel_indices = _prepare_inputs(dense_vector, indices)
    expected = dense_vector[indices]

    pytorch_op = lambda: dense_vector[indices]
    triton_op = lambda: _triton_gather_impl(dense_vector, kernel_indices, block_size=block_size)

    pytorch_values, pytorch_ms = _benchmark_cuda_op(pytorch_op, warmup=warmup, iters=iters)
    triton_values, triton_ms = _benchmark_cuda_op(triton_op, warmup=warmup, iters=iters)

    atol, rtol = _tolerance_for_dtype(value_dtype)
    triton_match = torch.allclose(triton_values, expected, atol=atol, rtol=rtol)
    triton_max_error = (
        float(torch.max(torch.abs(triton_values - expected)).item())
        if nnz > 0
        else 0.0
    )

    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    if run_cusparse:
        skip_reason = _cusparse_baseline_skip_reason(value_dtype)
        if skip_reason:
            cusparse_reason = skip_reason
        else:
            try:
                selector_matrix = _make_gather_selector_matrix(
                    indices, dense_vector.numel(), dense_vector.dtype
                )
                cusparse_op = lambda: _cusparse_spmv(selector_matrix, dense_vector)
                cusparse_values, cusparse_ms = _benchmark_cuda_op(
                    cusparse_op, warmup=warmup, iters=iters
                )
                cusparse_match = torch.allclose(
                    cusparse_values, expected, atol=atol, rtol=rtol
                )
                cusparse_max_error = (
                    float(torch.max(torch.abs(cusparse_values - expected)).item())
                    if nnz > 0
                    else 0.0
                )
            except Exception as exc:
                cusparse_reason = str(exc)

    triton_speedup_vs_pytorch = (
        pytorch_ms / triton_ms if triton_ms > 0 else float("inf")
    )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms
        if (cusparse_ms is not None and triton_ms > 0)
        else None
    )

    return {
        "parameters": {
            "dense_size": dense_size,
            "nnz": nnz,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": triton_speedup_vs_pytorch,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_pytorch": triton_match,
            "triton_max_error": triton_max_error,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_values,
        },
    }


def benchmark_scatter_case(
    dense_size=65536,
    nnz=4096,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_size=1024,
    run_cusparse=True,
    unique_indices=True,
    reset_output=True,
    dtype_policy="auto",
    index_fallback_policy="auto",
):
    """Benchmark Triton scatter vs PyTorch index_copy vs cuSPARSE-backed COO SpMV."""
    device = torch.device("cuda")
    requested_value_dtype, requested_effective_dtype, fallback_applied, fallback_reason = (
        _resolve_scatter_benchmark_dtype(value_dtype, dtype_policy)
    )
    sparse_values = _build_random_dense(nnz, requested_effective_dtype, device)
    indices = _build_indices(nnz, dense_size, index_dtype, device, unique=unique_indices)

    sparse_values, indices, kernel_indices, dense_size, prep_meta = _prepare_scatter_inputs(
        sparse_values,
        indices,
        dense_size=dense_size,
        out=None,
        dtype_policy=dtype_policy,
        return_metadata=True,
    )
    effective_value_dtype = prep_meta["effective_value_dtype"]
    fallback_applied = fallback_applied or prep_meta["fallback_applied"]
    fallback_reason = fallback_reason or prep_meta["fallback_reason"]

    base_out = _build_random_dense(dense_size, sparse_values.dtype, device)
    expected = _pytorch_scatter_impl(
        sparse_values,
        indices,
        dense_size,
        out=base_out.clone(),
        reset_output=reset_output,
    )

    pytorch_out = base_out.clone()
    triton_probe_out = base_out.clone()
    _, triton_index_meta = _triton_scatter_impl(
        sparse_values,
        kernel_indices,
        dense_size=dense_size,
        out=triton_probe_out,
        block_size=block_size,
        reset_output=reset_output,
        index_fallback_policy=index_fallback_policy,
        return_metadata=True,
    )
    effective_kernel_indices = kernel_indices
    if triton_index_meta["index_fallback_applied"] and kernel_indices.dtype == torch.int64:
        effective_kernel_indices = kernel_indices.to(torch.int32)
    triton_out = base_out.clone()

    pytorch_op = lambda: _pytorch_scatter_impl(
        sparse_values,
        indices,
        dense_size,
        out=pytorch_out,
        reset_output=reset_output,
    )
    triton_op = lambda: _triton_scatter_impl(
        sparse_values,
        effective_kernel_indices,
        dense_size=dense_size,
        out=triton_out,
        block_size=block_size,
        reset_output=reset_output,
        index_fallback_policy="strict",
    )

    pytorch_values, pytorch_ms = _benchmark_cuda_op(pytorch_op, warmup=warmup, iters=iters)
    triton_values, triton_ms = _benchmark_cuda_op(triton_op, warmup=warmup, iters=iters)

    atol, rtol = _tolerance_for_dtype(effective_value_dtype)
    triton_match = torch.allclose(triton_values, expected, atol=atol, rtol=rtol)
    triton_max_error = (
        float(torch.max(torch.abs(triton_values - expected)).item())
        if dense_size > 0
        else 0.0
    )

    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    if run_cusparse:
        if not reset_output:
            skip_reason = "cuSPARSE scatter baseline only matches reset_output=True semantics"
        else:
            skip_reason = _cusparse_baseline_skip_reason(sparse_values.dtype)
        if skip_reason:
            cusparse_reason = skip_reason
        else:
            try:
                selector_matrix = _make_scatter_selector_matrix(
                    indices, dense_size, sparse_values.dtype
                )
                cusparse_op = lambda: _cusparse_spmv(selector_matrix, sparse_values)
                cusparse_values, cusparse_ms = _benchmark_cuda_op(
                    cusparse_op, warmup=warmup, iters=iters
                )
                cusparse_match = torch.allclose(
                    cusparse_values, expected, atol=atol, rtol=rtol
                )
                cusparse_max_error = (
                    float(torch.max(torch.abs(cusparse_values - expected)).item())
                    if dense_size > 0
                    else 0.0
                )
            except Exception as exc:
                cusparse_reason = str(exc)

    triton_speedup_vs_pytorch = (
        pytorch_ms / triton_ms if triton_ms > 0 else float("inf")
    )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms
        if (cusparse_ms is not None and triton_ms > 0)
        else None
    )

    return {
        "parameters": {
            "dense_size": dense_size,
            "nnz": nnz,
            "value_dtype": requested_value_dtype,
            "effective_value_dtype": str(effective_value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
            "unique_indices": unique_indices,
            "reset_output": bool(reset_output),
            "dtype_policy": str(dtype_policy).lower(),
            "fallback_applied": bool(fallback_applied),
            "index_fallback_policy": str(index_fallback_policy).lower(),
            "kernel_index_dtype": triton_index_meta["kernel_index_dtype"],
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": triton_speedup_vs_pytorch,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_pytorch": triton_match,
            "triton_max_error": triton_max_error,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
            "fallback_reason": fallback_reason,
            "index_fallback_applied": bool(triton_index_meta["index_fallback_applied"]),
            "index_fallback_reason": triton_index_meta["index_fallback_reason"],
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_values,
        },
    }


def benchmark_spmv_case(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_nnz=256,
    max_segments=None,
    run_cusparse=True,
    transpose=False,
    op=None,
    index_fallback_policy="auto",
):
    """Benchmark Triton CSR SpMV vs cuSPARSE (CuPy CSR @ x)."""
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_cols, nnz, value_dtype, index_dtype, device
    )
    op_code = _normalize_spmv_op(op, transpose=transpose)
    if op is not None and bool(transpose) and op_code == SPMV_OP_NON:
        raise ValueError("transpose=True conflicts with op=non")
    transpose = _spmv_op_transposes(op_code)
    op_name = _spmv_op_to_name(op_code)
    x_size = n_rows if transpose else n_cols
    y_size = n_cols if transpose else n_rows
    x = _build_random_dense(x_size, value_dtype, device)
    shape = (n_rows, n_cols)
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
    triton_op = lambda: flagsparse_spmv_csr(
        x=x,
        prepared=prepared,
        return_time=False,
    )
    triton_y, triton_ms = _benchmark_cuda_op(triton_op, warmup=warmup, iters=iters)
    _cupy_supported_dtypes = (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    if (
        cp is not None
        and cpx_sparse is not None
        and value_dtype in _cupy_supported_dtypes
    ):
        data_cp = _cupy_from_torch(data)
        indices_cp = _cupy_from_torch(indices.to(torch.int64))
        indptr_cp = _cupy_from_torch(indptr)
        x_cp = _cupy_from_torch(x)
        A_csr = cpx_sparse.csr_matrix(
            (data_cp, indices_cp, indptr_cp), shape=shape
        )
        ref_y = _apply_cupy_spmv_op(A_csr, x_cp, op_code)
        expected = _torch_from_cupy(ref_y)
    else:
        row_indices = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        col_ind = indices.to(torch.int64)
        coo = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_ind]),
            data,
            shape,
            device=device,
        ).coalesce()
        x_2d = x.unsqueeze(1)
        if value_dtype in (torch.float16, torch.bfloat16):
            coo_f32 = coo.to(torch.float32)
            x_2d_f32 = x_2d.to(torch.float32)
            expected = _apply_torch_sparse_spmv_op(
                coo_f32, x_2d_f32, op_code
            ).to(value_dtype)
        else:
            expected = _apply_torch_sparse_spmv_op(coo, x_2d, op_code)
    atol, rtol = _tolerance_for_dtype(value_dtype)
    triton_match = torch.allclose(triton_y, expected, atol=atol, rtol=rtol)
    triton_max_error = (
        float(torch.max(torch.abs(triton_y - expected)).item())
        if y_size > 0
        else 0.0
    )
    cusparse_ms = None
    cusparse_match = None
    cusparse_max_error = None
    cusparse_reason = None
    if (
        run_cusparse
        and cp is not None
        and cpx_sparse is not None
        and value_dtype in _cupy_supported_dtypes
    ):
        skip_reason = _cusparse_baseline_skip_reason(value_dtype)
        if skip_reason:
            cusparse_reason = skip_reason
        else:
            try:
                A_op = _cupy_spmv_op_matrix(A_csr, op_code)
                cusparse_op = lambda: _torch_from_cupy(
                    A_op @ _cupy_from_torch(x)
                )
                cusparse_values, cusparse_ms = _benchmark_cuda_op(
                    cusparse_op, warmup=warmup, iters=iters
                )
                cusparse_match = torch.allclose(
                    cusparse_values, expected, atol=atol, rtol=rtol
                )
                cusparse_max_error = (
                    float(torch.max(torch.abs(cusparse_values - expected)).item())
                    if y_size > 0
                    else 0.0
                )
            except Exception as exc:
                cusparse_reason = str(exc)
    elif run_cusparse and value_dtype not in _cupy_supported_dtypes:
        cusparse_reason = (
            "float16/bfloat16 not supported by CuPy sparse; skipped"
        )
    triton_speedup_vs_cusparse = (
        cusparse_ms / triton_ms
        if (cusparse_ms is not None and triton_ms > 0)
        else None
    )
    return {
        "parameters": {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": nnz,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "op": op_name,
            "transpose": transpose,
            "index_fallback_policy": str(index_fallback_policy).lower(),
            "warmup": warmup,
            "iters": iters,
        },
        "performance": {
            "triton_ms": triton_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_cusparse": triton_speedup_vs_cusparse,
        },
        "verification": {
            "triton_match_reference": triton_match,
            "triton_max_error": triton_max_error,
            "cusparse_match_reference": cusparse_match,
            "cusparse_max_error": cusparse_max_error,
        },
        "backend_status": {
            "cusparse_unavailable_reason": cusparse_reason,
        },
        "samples": {"triton": triton_y, "reference": expected},
    }


def benchmark_performance(
    dense_size=65536,
    nnz=4096,
    dtype=torch.float32,
    index_dtype=torch.int32,
):
    """Backward-compatible benchmark entry."""
    result = benchmark_gather_case(
        dense_size=dense_size,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=10,
        iters=100,
        run_cusparse=False,
    )
    return {
        "triton_time_ms": result["performance"]["triton_ms"],
        "results_match": result["verification"]["triton_match_pytorch"],
        "dtype": str(dtype),
        "index_dtype": str(index_dtype),
        "dense_size": dense_size,
        "nnz": nnz,
    }


def comprehensive_gather_test(
    dense_size=100000,
    nnz=10000,
    dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    run_cusparse=True,
):
    """Full test entry for one configuration."""
    return benchmark_gather_case(
        dense_size=dense_size,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        run_cusparse=run_cusparse,
    )


def comprehensive_scatter_test(
    dense_size=100000,
    nnz=10000,
    dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    run_cusparse=True,
    unique_indices=True,
    reset_output=True,
    dtype_policy="auto",
    index_fallback_policy="auto",
):
    """Full scatter test entry for one configuration."""
    return benchmark_scatter_case(
        dense_size=dense_size,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        warmup=warmup,
        iters=iters,
        run_cusparse=run_cusparse,
        unique_indices=unique_indices,
        reset_output=reset_output,
        dtype_policy=dtype_policy,
        index_fallback_policy=index_fallback_policy,
    )


def comprehensive_spsm_test(
    fmt="csr",
    n_rows=1024,
    n_rhs=32,
    nnz=8192,
    dtype=torch.float32,
    index_dtype=torch.int32,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    warmup=10,
    iters=50,
):
    """Full SpSM benchmark entry for one configuration."""
    return benchmark_spsm_case(
        fmt=fmt,
        n_rows=n_rows,
        n_rhs=n_rhs,
        nnz=nnz,
        value_dtype=dtype,
        index_dtype=index_dtype,
        alpha=alpha,
        lower=lower,
        unit_diagonal=unit_diagonal,
        warmup=warmup,
        iters=iters,
    )
