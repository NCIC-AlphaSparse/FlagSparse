"""Shared imports, dtypes, and helpers for FlagSparse sparse ops."""

import time

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError(
        "Runtime dependencies are missing. Install them manually: pip install torch triton"
    ) from exc

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except ImportError:
    cp = None
    cpx_sparse = None

_SUPPORTED_VALUE_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
]
SUPPORTED_VALUE_DTYPES = tuple(_SUPPORTED_VALUE_DTYPES)
SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)
_INDEX_LIMIT_INT32 = 2**31 - 1

# Star-import exposes only non-underscore names unless listed here.
__all__ = (
    "SUPPORTED_VALUE_DTYPES",
    "SUPPORTED_INDEX_DTYPES",
    "_INDEX_LIMIT_INT32",
    "_is_complex_dtype",
    "_resolve_scatter_value_dtype",
    "_component_dtype_for_complex",
    "_tolerance_for_dtype",
    "_require_cupy",
    "_cupy_dtype_from_torch",
    "_cupy_from_torch",
    "_torch_from_cupy",
    "_to_torch_tensor",
    "_to_backend_like",
    "_cusparse_baseline_skip_reason",
    "_build_random_dense",
    "_build_indices",
    "_build_random_csr",
    "_validate_common_inputs",
    "_prepare_inputs",
    "_prepare_scatter_inputs",
    "_benchmark_cuda_op",
    "cp",
    "cpx_sparse",
    "time",
    "torch",
    "triton",
    "tl",
)

def _is_complex_dtype(value_dtype):
    return value_dtype in (torch.complex64, torch.complex128)


def _resolve_scatter_value_dtype(value_dtype, dtype_policy="auto"):
    dtype_policy = str(dtype_policy).lower()
    if dtype_policy not in ("auto", "strict"):
        raise ValueError("dtype_policy must be 'auto' or 'strict'")
    if isinstance(value_dtype, str):
        token = value_dtype.strip().lower()
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
            "complex64": torch.complex64,
            "complex128": torch.complex128,
        }
        if token not in mapping:
            raise TypeError(f"Unsupported dtype token: {value_dtype}")
        value_dtype = mapping[token]
    return value_dtype, False, None


def _component_dtype_for_complex(value_dtype):
    if value_dtype == torch.complex64:
        return torch.float32
    if value_dtype == torch.complex128:
        return torch.float64
    raise TypeError(f"Unsupported complex dtype: {value_dtype}")


def _tolerance_for_dtype(value_dtype):
    if value_dtype == torch.float16:
        return 2e-3, 2e-3
    if value_dtype == torch.bfloat16:
        return 1e-1, 1e-1
    if value_dtype in (torch.float32, torch.complex64):
        return 1e-6, 1e-5
    if value_dtype in (torch.float64, torch.complex128):
        return 1e-10, 1e-8
    return 1e-6, 1e-5


def _require_cupy():
    if cp is None or cpx_sparse is None:
        raise RuntimeError(
            "CuPy is required for cuSPARSE baseline. "
            "Install a CUDA-matched wheel, for example: pip install cupy-cuda12x"
        )


def _cupy_dtype_from_torch(torch_dtype):
    _require_cupy()
    mapping = {
        torch.float16: cp.float16,
        # Keep cuSPARSE baseline stable for bf16 by computing in fp32 on CuPy path.
        torch.bfloat16: cp.float32,
        torch.float32: cp.float32,
        torch.float64: cp.float64,
        torch.complex64: cp.complex64,
        torch.complex128: cp.complex128,
        torch.int32: cp.int32,
        torch.int64: cp.int64,
    }
    if torch_dtype not in mapping:
        raise TypeError(f"Unsupported dtype conversion to CuPy: {torch_dtype}")
    return mapping[torch_dtype]


def _cupy_from_torch(tensor):
    _require_cupy()
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def _torch_from_cupy(array):
    try:
        dlpack_capsule = array.toDlpack()
    except AttributeError:
        dlpack_capsule = array.to_dlpack()
    return torch.utils.dlpack.from_dlpack(dlpack_capsule)


def _to_torch_tensor(x, name):
    if torch.is_tensor(x):
        return x, "torch"
    if cp is not None and isinstance(x, cp.ndarray):
        return _torch_from_cupy(x), "cupy"
    raise TypeError(f"{name} must be a torch.Tensor or cupy.ndarray")


def _to_backend_like(torch_tensor, ref_obj):
    if cp is not None and isinstance(ref_obj, cp.ndarray):
        return _cupy_from_torch(torch_tensor)
    return torch_tensor


def _cusparse_baseline_skip_reason(value_dtype):
    if value_dtype == torch.bfloat16:
        return "bfloat16 is not supported by the cuSPARSE baseline path; skipped"
    if cp is None and value_dtype == torch.float16:
        return "float16 is not supported by torch sparse fallback when CuPy is unavailable; skipped"
    return None


def _build_random_dense(dense_size, value_dtype, device):
    if value_dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(dense_size, dtype=value_dtype, device=device)
    if _is_complex_dtype(value_dtype):
        component_dtype = _component_dtype_for_complex(value_dtype)
        real = torch.randn(dense_size, dtype=component_dtype, device=device)
        imag = torch.randn(dense_size, dtype=component_dtype, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"Unsupported value dtype: {value_dtype}")


def _build_indices(nnz, dense_size, index_dtype, device, unique=False):
    if unique and nnz <= dense_size:
        return torch.randperm(dense_size, device=device)[:nnz].to(index_dtype)
    return torch.randint(0, dense_size, (nnz,), dtype=index_dtype, device=device)


def _build_random_csr(n_rows, n_cols, nnz, value_dtype, index_dtype, device):
    if nnz <= 0 or n_rows <= 0 or n_cols <= 0:
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
        return (
            torch.empty(0, dtype=value_dtype, device=device),
            torch.empty(0, dtype=index_dtype, device=device),
            indptr,
        )
    row_choices = torch.randint(0, n_rows, (nnz,), device=device)
    row_choices, _ = torch.sort(row_choices)
    nnz_per_row = torch.bincount(row_choices, minlength=n_rows)
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    indices = torch.randint(0, n_cols, (nnz,), dtype=index_dtype, device=device)
    data = _build_random_dense(nnz, value_dtype, device)
    return data, indices, indptr


def _validate_common_inputs(dense_vector, indices):
    if dense_vector.ndim != 1:
        raise ValueError("dense_vector must be a 1D tensor")
    if indices.ndim != 1:
        raise ValueError("indices must be a 1D tensor")
    if not dense_vector.is_cuda or not indices.is_cuda:
        raise ValueError("dense_vector and indices must both be CUDA tensors")
    if dense_vector.dtype not in SUPPORTED_VALUE_DTYPES:
        raise TypeError(
            f"dense_vector dtype must be one of: {', '.join(str(dt) for dt in SUPPORTED_VALUE_DTYPES)}"
        )
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")


def _prepare_inputs(dense_vector, indices):
    _validate_common_inputs(dense_vector, indices)

    dense_vector = dense_vector.contiguous()
    indices = indices.contiguous()

    max_index = -1
    if indices.numel() > 0:
        if torch.any(indices < 0).item():
            raise IndexError("indices must be non-negative")
        max_index = int(indices.max().item())
        if max_index >= dense_vector.numel():
            raise IndexError(
                f"indices out of range: max index {max_index}, dense size {dense_vector.numel()}"
            )

    kernel_indices = indices
    if indices.dtype == torch.int64:
        if max_index > _INDEX_LIMIT_INT32:
            raise ValueError(
                f"int64 index value {max_index} exceeds Triton int32 kernel range"
            )
        kernel_indices = indices.to(torch.int32)

    return dense_vector, indices, kernel_indices


def _prepare_scatter_inputs(
    sparse_values, indices, dense_size=None, out=None, dtype_policy="auto", return_metadata=False
):
    if sparse_values.ndim != 1:
        raise ValueError("sparse_values must be a 1D tensor")
    if indices.ndim != 1:
        raise ValueError("indices must be a 1D tensor")
    if sparse_values.numel() != indices.numel():
        raise ValueError("sparse_values and indices must have the same number of elements")
    if not sparse_values.is_cuda or not indices.is_cuda:
        raise ValueError("sparse_values and indices must both be CUDA tensors")
    if sparse_values.dtype not in SUPPORTED_VALUE_DTYPES:
        raise TypeError(
            f"sparse_values dtype must be one of: {', '.join(str(dt) for dt in SUPPORTED_VALUE_DTYPES)}"
        )
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")

    requested_value_dtype = sparse_values.dtype
    effective_value_dtype, fallback_applied, fallback_reason = _resolve_scatter_value_dtype(
        requested_value_dtype, dtype_policy=dtype_policy
    )
    if effective_value_dtype != sparse_values.dtype:
        sparse_values = sparse_values.to(effective_value_dtype)

    sparse_values = sparse_values.contiguous()
    indices = indices.contiguous()

    if dense_size is None:
        dense_size = int(indices.max().item()) + 1 if indices.numel() > 0 else 0
    dense_size = int(dense_size)
    if dense_size < 0:
        raise ValueError("dense_size must be non-negative")

    max_index = -1
    if indices.numel() > 0:
        if torch.any(indices < 0).item():
            raise IndexError("indices must be non-negative")
        max_index = int(indices.max().item())
        if max_index >= dense_size:
            raise IndexError(
                f"indices out of range: max index {max_index}, dense size {dense_size}"
            )

    kernel_indices = indices

    if out is not None:
        if out.ndim != 1:
            raise ValueError("out must be a 1D tensor")
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.dtype != sparse_values.dtype:
            raise TypeError("out dtype must match sparse_values dtype")
        if out.numel() != dense_size:
            raise ValueError("out size must equal dense_size")
        if out.device != sparse_values.device:
            raise ValueError("out must be on the same device as sparse_values")

    metadata = {
        "requested_value_dtype": requested_value_dtype,
        "effective_value_dtype": sparse_values.dtype,
        "fallback_applied": bool(fallback_applied),
        "fallback_reason": fallback_reason,
        "dtype_policy": str(dtype_policy).lower(),
    }
    if return_metadata:
        return sparse_values, indices, kernel_indices, dense_size, metadata
    return sparse_values, indices, kernel_indices, dense_size


def _benchmark_cuda_op(op, warmup, iters):
    warmup = max(0, int(warmup))
    iters = max(1, int(iters))

    output = None
    for _ in range(warmup):
        output = op()

    torch.cuda.synchronize()
    if cp is not None:
        cp.cuda.runtime.deviceSynchronize()
    start_time = time.perf_counter()
    for _ in range(iters):
        output = op()
    torch.cuda.synchronize()
    if cp is not None:
        cp.cuda.runtime.deviceSynchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / iters
    return output, elapsed_ms
