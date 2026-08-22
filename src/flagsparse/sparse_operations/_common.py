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

"""Shared imports, dtypes, and helpers for FlagSparse sparse ops."""

import ctypes
import statistics
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

# DCU/ROCm backend: the vendor reference library is hipSPARSE, reached through the
# `hip-python` bindings. The import is optional in exactly the same way CuPy is, so a
# CUDA-only install keeps working unchanged.
_HIP_IMPORT_ERROR = None
try:
    from hip import hip, hipsparse
    from hip._util.types import Pointer as HipPointer
except Exception as exc:  # pragma: no cover - depends on the installed runtime
    hip = None
    hipsparse = None
    HipPointer = None
    _HIP_IMPORT_ERROR = exc

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
# torch.version.hip is set only by ROCm/DCU builds of PyTorch, so it is the cheapest
# reliable way to pick the vendor backend without touching a device.
_IS_ROCM_RUNTIME = getattr(torch.version, "hip", None) is not None
_CUPY_SPMV_SUPPORTED_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

# Star-import exposes only non-underscore names unless listed here.
__all__ = (
    "SUPPORTED_VALUE_DTYPES",
    "SUPPORTED_INDEX_DTYPES",
    "_INDEX_LIMIT_INT32",
    "_is_complex_dtype",
    "_resolve_scatter_value_dtype",
    "_component_dtype_for_complex",
    "_tolerance_for_dtype",
    "_is_rocm_runtime",
    "_is_hipsparse_available",
    "_require_cupy",
    "_cupy_dtype_from_torch",
    "_cupy_from_torch",
    "_torch_from_cupy",
    "_to_torch_tensor",
    "_to_backend_like",
    "_cusparse_baseline_skip_reason",
    "_cupy_cusparse_spmv_skip_reason",
    # hipSPARSE / HIP glue shared by the per-operator DCU reference paths.
    "_hip_check_result",
    "_hip_lookup",
    "_hip_runtime_event_available",
    "_hip_event_elapsed_ms",
    "_destroy_hip_event",
    "_hipsparse_lookup",
    "_hipsparse_unavailable_reason",
    "_hipsparse_value_type",
    "_hipsparse_scalar",
    "_hipsparse_index_type",
    "_hipsparse_spmv_operation",
    "_hipsparse_spmm_order",
    "_hipsparse_spmm_algorithm",
    "_hipsparse_create_coo_descriptor",
    "_hipsparse_create_csr_descriptor",
    "_hipsparse_create_dnmat_descriptor",
    # Consumed by benchmarks.py through `from ._common import *`.
    "_prepare_spmv_csr_ref_hipsparse",
    "_run_spmv_csr_ref_hipsparse_prepared",
    "_destroy_spmv_csr_ref_hipsparse_prepared",
    "_normalize_spmv_reference_op",
    "_normalize_sparse_reference_op",
    "_spmv_reference_compute_dtype",
    "_sparse_reference_compute_dtype",
    "_cast_spmv_reference_output",
    "_cast_sparse_reference_output",
    "_apply_torch_sparse_matmul_op",
    "_apply_torch_sparse_spmv_op",
    "_cupy_spmv_op_matrix",
    "_apply_cupy_sparse_matmul_op",
    "_apply_cupy_spmv_op",
    "_spmv_csr_sparse_ref_backend",
    "_spmv_csr_reference_backend",
    "_spmv_csr_ref_pytorch",
    "_spmv_csr_ref_cupy",
    "spmv_csr_ref_hipsparse",
    "_spmv_csr_reference",
    "_benchmark_spmv_csr_sparse_ref",
    "_spmv_coo_sparse_ref_backend",
    "_spmv_coo_reference_backend",
    "_spmv_coo_ref_pytorch",
    "_spmv_coo_ref_cupy",
    "spmv_coo_ref_hipsparse",
    "_spmv_coo_reference",
    "_benchmark_spmv_coo_sparse_ref",
    "_build_random_dense",
    "_build_indices",
    "_build_random_csr",
    "_validate_common_inputs",
    "_prepare_inputs",
    "_prepare_scatter_inputs",
    "_benchmark_cuda_op",
    "_torch_current_stream_ptr",
    "_hip_event_record_stream",
    "_hip_event_record_current_stream",
    "_benchmark_prepared_hip_event_op",
    "_benchmark_prepared_cuda_op",
    "_benchmark_cuda_graph_op",
    "cp",
    "cpx_sparse",
    "hip",
    "hipsparse",
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


def _is_rocm_runtime():
    return bool(_IS_ROCM_RUNTIME)


def _is_hipsparse_available():
    return hip is not None and hipsparse is not None and HipPointer is not None


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


def _normalize_spmv_reference_op(op):
    if op is None:
        return "non"
    if isinstance(op, str):
        token = op.strip().lower()
        if token in ("non", "trans", "conj"):
            return token
        raise ValueError("op must be one of: non, trans, conj")
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: non, trans, conj") from exc
    mapping = {0: "non", 1: "trans", 2: "conj"}
    if op_code not in mapping:
        raise ValueError("op must be one of: non, trans, conj")
    return mapping[op_code]


def _normalize_sparse_reference_op(op):
    return _normalize_spmv_reference_op(op)


def _spmv_reference_compute_dtype(value_dtype):
    if value_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if value_dtype == torch.float32:
        return torch.float64
    if value_dtype == torch.complex64:
        return torch.complex128
    return value_dtype


def _sparse_reference_compute_dtype(value_dtype):
    return _spmv_reference_compute_dtype(value_dtype)


def _cast_spmv_reference_output(y_ref, out_dtype):
    return y_ref.to(out_dtype) if y_ref.dtype != out_dtype else y_ref


def _cast_sparse_reference_output(y_ref, out_dtype):
    return _cast_spmv_reference_output(y_ref, out_dtype)


def _apply_torch_sparse_matmul_op(matrix, rhs, op):
    op_name = _normalize_sparse_reference_op(op)
    rhs_is_vector = rhs.ndim == 1
    rhs_2d = rhs.unsqueeze(1) if rhs_is_vector else rhs
    if op_name == "non":
        result = torch.sparse.mm(matrix, rhs_2d)
    elif op_name == "trans":
        result = torch.sparse.mm(matrix.transpose(0, 1), rhs_2d)
    else:
        result = torch.sparse.mm(matrix.conj().transpose(0, 1), rhs_2d)
    return result.squeeze(1) if rhs_is_vector else result


def _apply_torch_sparse_spmv_op(matrix, vector_2d, op):
    return _apply_torch_sparse_matmul_op(matrix, vector_2d, op)


def _cupy_spmv_op_matrix(matrix, op):
    op_name = _normalize_spmv_reference_op(op)
    if op_name == "non":
        return matrix
    if op_name == "trans":
        return matrix.T
    matrix_conj = matrix.conj() if hasattr(matrix, "conj") else matrix.conjugate()
    return matrix_conj.T


def _apply_cupy_sparse_matmul_op(matrix, rhs, op):
    return _cupy_spmv_op_matrix(matrix, op) @ rhs


def _apply_cupy_spmv_op(matrix, vector, op):
    return _apply_cupy_sparse_matmul_op(matrix, vector, op)


def _hipsparse_status_success(status):
    try:
        return int(status) == 0
    except Exception:
        pass
    value = getattr(status, "value", None)
    if value is not None:
        try:
            return int(value) == 0
        except Exception:
            pass
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name.upper().endswith("SUCCESS")
    text = str(status).strip()
    return text == "0" or text.upper().endswith("SUCCESS")


def _hip_check_result(result, call_name):
    payload = None
    status = result
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError(f"{call_name} returned an empty result")
        status = result[0]
        if len(result) == 2:
            payload = result[1]
        elif len(result) > 2:
            payload = result[1:]
    if not _hipsparse_status_success(status):
        raise RuntimeError(f"{call_name} failed: {status}")
    return payload


def _hip_runtime_event_available():
    if hip is None:
        return False
    required_symbols = (
        "hipEventCreate",
        "hipEventRecord",
        "hipEventSynchronize",
        "hipEventElapsedTime",
        "hipEventDestroy",
    )
    return all(hasattr(hip, symbol) for symbol in required_symbols)


def _hip_event_elapsed_ms(start_evt, stop_evt):
    raw = hip.hipEventElapsedTime(start_evt, stop_evt)
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise RuntimeError(f"hipEventElapsedTime returned unexpected result: {raw!r}")
    status, elapsed_ms = raw
    if not _hipsparse_status_success(status):
        raise RuntimeError(f"hipEventElapsedTime failed: {status}")
    try:
        return float(elapsed_ms)
    except Exception as exc:
        raise RuntimeError(
            f"hipEventElapsedTime returned a non-numeric elapsed value: {raw!r}"
        ) from exc


def _destroy_hip_event(evt):
    if evt is None or hip is None or not hasattr(hip, "hipEventDestroy"):
        return
    try:
        _hip_check_result(hip.hipEventDestroy(evt), "hipEventDestroy")
    except Exception:
        pass


def _torch_current_stream_ptr():
    stream = torch.cuda.current_stream()
    for attr_name in ("cuda_stream", "hip_stream"):
        stream_ptr = getattr(stream, attr_name, None)
        if callable(stream_ptr):
            stream_ptr = stream_ptr()
        if stream_ptr is not None:
            return int(stream_ptr)
    return None


def _hip_event_record_stream(evt, stream, call_name):
    if stream == "current":
        stream_ptr = _torch_current_stream_ptr()
        if stream_ptr is None:
            raise RuntimeError("could not resolve torch current CUDA/HIP stream pointer")
    else:
        stream_ptr = int(stream)
    if stream_ptr == 0:
        stream_args = (0, ctypes.c_void_p(0))
    else:
        stream_args = (ctypes.c_void_p(stream_ptr), stream_ptr)
    last_error = None
    for stream_arg in stream_args:
        try:
            _hip_check_result(hip.hipEventRecord(evt, stream_arg), call_name)
            return
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"{call_name} failed for stream {stream_ptr}: {last_error}"
    ) from last_error


def _hip_event_record_current_stream(evt, call_name):
    _hip_event_record_stream(evt, "current", call_name)


def _hipsparse_lookup(container_name, attr_names):
    container = getattr(hipsparse, container_name, None) if hipsparse is not None else None
    search_spaces = [container, hipsparse]
    for space in search_spaces:
        if space is None:
            continue
        for attr_name in attr_names:
            if hasattr(space, attr_name):
                return getattr(space, attr_name)
    names = ", ".join(attr_names)
    raise RuntimeError(f"Unable to resolve hipSPARSE attribute {names}")


def _hip_lookup(container_name, attr_names):
    container = getattr(hip, container_name, None) if hip is not None else None
    search_spaces = [container, hip]
    for space in search_spaces:
        if space is None:
            continue
        for attr_name in attr_names:
            if hasattr(space, attr_name):
                return getattr(space, attr_name)
    names = ", ".join(attr_names)
    raise RuntimeError(f"Unable to resolve HIP attribute {names}")


def _hipsparse_unavailable_reason():
    if _is_hipsparse_available():
        return None
    if not _is_rocm_runtime():
        return "hip-python/hipSPARSE requires a ROCm runtime"
    reason = "ROCm runtime detected but hip-python/hipSPARSE is unavailable"
    if _HIP_IMPORT_ERROR is not None:
        reason += f": {_HIP_IMPORT_ERROR}"
    return reason


def _hipsparse_create_coo_descriptor(
    n_rows,
    n_cols,
    nnz,
    row_ptr,
    col_ptr,
    values_ptr,
    index_type,
    index_base,
    value_type,
):
    # hip-python returns the sparse descriptor as a payload rather than filling
    # an output slot, unlike hipsparseCreateDnVec/DnMat which keep the
    # out-parameter form. Some wrappers also split the COO index type in two.
    attempts = (
        (
            n_rows,
            n_cols,
            nnz,
            row_ptr,
            col_ptr,
            values_ptr,
            index_type,
            index_base,
            value_type,
        ),
        (
            n_rows,
            n_cols,
            nnz,
            row_ptr,
            col_ptr,
            values_ptr,
            index_type,
            index_type,
            index_base,
            value_type,
        ),
    )
    last_error = None
    for args in attempts:
        try:
            return _hip_check_result(
                hipsparse.hipsparseCreateCoo(*args), "hipsparseCreateCoo"
            )
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise RuntimeError(
            f"hipsparseCreateCoo wrapper signature mismatch: {last_error}"
        ) from last_error
    raise RuntimeError("hipsparseCreateCoo wrapper signature mismatch")


def _hipsparse_create_csr_descriptor(
    n_rows,
    n_cols,
    nnz,
    row_ptr,
    col_ptr,
    values_ptr,
    row_index_type,
    col_index_type,
    index_base,
    value_type,
):
    # As with COO, the CSR descriptor comes back as a payload instead of being
    # written into a leading output slot.
    return _hip_check_result(
        hipsparse.hipsparseCreateCsr(
            n_rows,
            n_cols,
            nnz,
            row_ptr,
            col_ptr,
            values_ptr,
            row_index_type,
            col_index_type,
            index_base,
            value_type,
        ),
        "hipsparseCreateCsr",
    )


def _hipsparse_spmm_order(order_name, context):
    mapping = {
        "row": ("HIPSPARSE_ORDER_ROW",),
        "col": ("HIPSPARSE_ORDER_COL",),
    }
    if order_name not in mapping:
        raise RuntimeError(f"{context} does not support dense order={order_name}")
    return _hipsparse_lookup("hipsparseOrder_t", mapping[order_name])


def _hipsparse_spmm_algorithm(format_name):
    format_name = str(format_name).lower()
    mapping = {
        "csr": (
            "HIPSPARSE_SPMM_CSR_ALG1",
            "HIPSPARSE_SPMM_ALG_DEFAULT",
        ),
        "coo": (
            "HIPSPARSE_SPMM_COO_ALG1",
            "HIPSPARSE_COOMM_ALG1",
            "HIPSPARSE_SPMM_ALG_DEFAULT",
        ),
    }
    if format_name not in mapping:
        raise RuntimeError(f"hipSPARSE SpMM does not support format={format_name}")
    return _hipsparse_lookup("hipsparseSpMMAlg_t", mapping[format_name])


def _hipsparse_create_dnmat_descriptor(
    mat_ref,
    rows,
    cols,
    ld,
    values_ptr,
    value_type,
    order,
):
    attempts = (
        (mat_ref, rows, cols, ld, values_ptr, value_type, order),
    )
    last_error = None
    for args in attempts:
        try:
            return _hip_check_result(
                hipsparse.hipsparseCreateDnMat(*args), "hipsparseCreateDnMat"
            )
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise RuntimeError(
            f"hipsparseCreateDnMat wrapper signature mismatch: {last_error}"
        ) from last_error
    raise RuntimeError("hipsparseCreateDnMat wrapper signature mismatch")


class _HipFloatComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class _HipDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def _hipsparse_value_type(value_dtype):
    mapping = {
        torch.float32: ("HIP_R_32F",),
        torch.float64: ("HIP_R_64F",),
        torch.complex64: ("HIP_C_32F", "HIP_C_32FC"),
        torch.complex128: ("HIP_C_64F", "HIP_C_64FC"),
    }
    if value_dtype not in mapping:
        raise RuntimeError(f"hipSPARSE CSR SpMV has no dtype mapping for {value_dtype}")
    return _hip_lookup("hipDataType", mapping[value_dtype])


def _hipsparse_scalar(value_dtype, real, imag=0.0):
    if value_dtype == torch.float32:
        return ctypes.c_float(float(real))
    if value_dtype == torch.float64:
        return ctypes.c_double(float(real))
    if value_dtype == torch.complex64:
        scalar_type = getattr(hip, "hipComplex", None) or getattr(
            hip, "hipFloatComplex", None
        )
        if scalar_type is not None:
            return scalar_type(float(real), float(imag))
        return _HipFloatComplex(float(real), float(imag))
    if value_dtype == torch.complex128:
        scalar_type = getattr(hip, "hipDoubleComplex", None) or getattr(
            hip, "hipComplexDouble", None
        )
        if scalar_type is not None:
            return scalar_type(float(real), float(imag))
        return _HipDoubleComplex(float(real), float(imag))
    raise RuntimeError(f"hipSPARSE CSR SpMV does not support {value_dtype}")


def _hipsparse_index_type(index_dtype, context):
    mapping = {
        torch.int32: ("HIPSPARSE_INDEX_32I",),
        torch.int64: ("HIPSPARSE_INDEX_64I",),
    }
    if index_dtype not in mapping:
        raise RuntimeError(
            f"{context} has no supported index dtype mapping for {index_dtype}"
        )
    return _hipsparse_lookup("hipsparseIndexType_t", mapping[index_dtype])


def _hipsparse_spmv_operation(op, context):
    op_name = _normalize_spmv_reference_op(op)
    mapping = {
        "non": ("HIPSPARSE_OPERATION_NON_TRANSPOSE",),
        "trans": ("HIPSPARSE_OPERATION_TRANSPOSE",),
        "conj": ("HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE",),
    }
    if op_name not in mapping:
        raise RuntimeError(f"{context} does not support op={op_name}")
    return _hipsparse_lookup("hipsparseOperation_t", mapping[op_name])


def _hipsparse_spmv_csr_skip_reason(value_dtype, index_dtype, op="non"):
    op_name = _normalize_spmv_reference_op(op)
    if not _is_rocm_runtime():
        return "hipSPARSE CSR SpMV reference requires a ROCm runtime"
    unavailable_reason = _hipsparse_unavailable_reason()
    if unavailable_reason is not None:
        return unavailable_reason
    required_symbols = (
        "hipsparseCreate",
        "hipsparseDestroy",
        "hipsparseCreateCsr",
        "hipsparseCreateDnVec",
        "hipsparseDestroyDnVec",
        "hipsparseDestroySpMat",
        "hipsparseSpMV_bufferSize",
        "hipsparseSpMV",
    )
    for symbol in required_symbols:
        if not hasattr(hipsparse, symbol):
            return f"hipSPARSE CSR SpMV direct API is unavailable: missing {symbol}"
    if value_dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        return f"hipSPARSE CSR SpMV has no supported value dtype mapping for {value_dtype}"
    if index_dtype not in (torch.int32, torch.int64):
        return f"hipSPARSE CSR SpMV has no supported index dtype mapping for {index_dtype}"
    try:
        _ = _hipsparse_value_type(value_dtype)
        _ = _hipsparse_scalar(value_dtype, 1.0, 0.0)
        _ = _hipsparse_scalar(value_dtype, 0.0, 0.0)
        _ = _hipsparse_index_type(index_dtype, "hipSPARSE CSR SpMV")
        _ = _hipsparse_spmv_operation(op_name, "hipSPARSE CSR SpMV")
    except Exception as exc:
        return str(exc)
    return None


def _hipsparse_spmv_coo_direct_skip_reason(value_dtype, index_dtype, op="non"):
    op_name = _normalize_sparse_reference_op(op)
    if not _is_rocm_runtime():
        return "hipSPARSE COO SpMV reference requires a ROCm runtime"
    unavailable_reason = _hipsparse_unavailable_reason()
    if unavailable_reason is not None:
        return unavailable_reason
    required_symbols = (
        "hipsparseCreate",
        "hipsparseDestroy",
        "hipsparseCreateCoo",
        "hipsparseCreateDnVec",
        "hipsparseDestroyDnVec",
        "hipsparseDestroySpMat",
        "hipsparseSpMV_bufferSize",
        "hipsparseSpMV",
    )
    for symbol in required_symbols:
        if not hasattr(hipsparse, symbol):
            return f"hipSPARSE COO SpMV direct API is unavailable: missing {symbol}"
    if value_dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        return f"hipSPARSE COO SpMV has no supported value dtype mapping for {value_dtype}"
    if index_dtype not in (torch.int32, torch.int64):
        return f"hipSPARSE COO SpMV has no supported index dtype mapping for {index_dtype}"
    try:
        _ = _hipsparse_value_type(value_dtype)
        _ = _hipsparse_scalar(value_dtype, 1.0, 0.0)
        _ = _hipsparse_scalar(value_dtype, 0.0, 0.0)
        _ = _hipsparse_index_type(index_dtype, "hipSPARSE COO SpMV")
        _ = _hipsparse_spmv_operation(op_name, "hipSPARSE COO SpMV")
    except Exception as exc:
        return str(exc)
    return None


def _cupy_cusparse_spmv_skip_reason(value_dtype):
    """Why the CuPy/cuSPARSE vendor baseline cannot serve this dtype."""
    if cp is None or cpx_sparse is None:
        return "CuPy/cuSPARSE is not available"
    if value_dtype not in _CUPY_SPMV_SUPPORTED_VALUE_DTYPES:
        return f"{value_dtype} is not supported by the CuPy/cuSPARSE SpMV baseline"
    return None


def _spmv_csr_sparse_ref_backend(value_dtype, index_dtype, op="non"):
    """Pick the vendor sparse library for a CSR SpMV reference, per backend.

    Returns (backend, reason). ``backend`` is None when no vendor library can serve
    the request, in which case ``reason`` explains why and callers fall back to the
    portable torch.sparse path.
    """
    op_name = _normalize_spmv_reference_op(op)
    if _is_rocm_runtime():
        skip_reason = _hipsparse_spmv_csr_skip_reason(value_dtype, index_dtype, op=op_name)
        if skip_reason is None:
            return "hipsparse", None
        return None, skip_reason
    skip_reason = _cupy_cusparse_spmv_skip_reason(value_dtype)
    if skip_reason is None:
        return "cupy_cusparse", None
    return None, skip_reason


def _spmv_csr_reference_backend(value_dtype, index_dtype, op="non"):
    sparse_backend, reason = _spmv_csr_sparse_ref_backend(value_dtype, index_dtype, op=op)
    if sparse_backend is not None:
        return sparse_backend, None
    return "torch", reason


def _spmv_csr_ref_pytorch(
    data,
    indices,
    indptr,
    x,
    shape,
    out_dtype=None,
    op="non",
    return_compute=False,
    reference_compute_dtype=False,
):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    if reference_compute_dtype:
        compute_dtype = _spmv_reference_compute_dtype(out_dtype)
    elif out_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = out_dtype
    device = data.device
    data_ref = data.to(compute_dtype)
    x_ref = x.to(compute_dtype)
    op_name = _normalize_spmv_reference_op(op)
    try:
        csr_ref = torch.sparse_csr_tensor(
            indptr.to(torch.int64),
            indices.to(torch.int64),
            data_ref,
            size=shape,
            device=device,
        )
        y_ref = _apply_torch_sparse_spmv_op(csr_ref, x_ref.unsqueeze(1), op_name)
    except Exception:
        n_rows = int(shape[0])
        row_ind = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr.to(torch.int64)[1:] - indptr.to(torch.int64)[:-1],
        )
        coo_ref = torch.sparse_coo_tensor(
            torch.stack([row_ind, indices.to(torch.int64)]),
            data_ref,
            shape,
            device=device,
        ).coalesce()
        y_ref = _apply_torch_sparse_spmv_op(coo_ref, x_ref.unsqueeze(1), op_name)
    if return_compute:
        return y_ref
    return _cast_spmv_reference_output(y_ref, out_dtype)


def _spmv_csr_ref_cupy(
    data,
    indices,
    indptr,
    x,
    shape,
    out_dtype=None,
    op="non",
    reference_compute_dtype=False,
):
    _require_cupy()
    out_dtype = data.dtype if out_dtype is None else out_dtype
    compute_dtype = (
        _spmv_reference_compute_dtype(out_dtype)
        if reference_compute_dtype
        else out_dtype
    )
    op_name = _normalize_spmv_reference_op(op)
    data_ref = data.to(compute_dtype)
    x_ref = x.to(compute_dtype)
    data_cp = _cupy_from_torch(data_ref)
    ind_cp = _cupy_from_torch(indices.to(torch.int64))
    ptr_cp = _cupy_from_torch(indptr.to(torch.int64))
    x_cp = _cupy_from_torch(x_ref)
    matrix = cpx_sparse.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
    y_ref = _apply_cupy_spmv_op(matrix, x_cp, op_name)
    return _cast_spmv_reference_output(_torch_from_cupy(y_ref), out_dtype)


def _spmv_coo_sparse_ref_backend(value_dtype, index_dtype, op="non"):
    op_name = _normalize_sparse_reference_op(op)
    if _is_rocm_runtime():
        direct_reason = _hipsparse_spmv_coo_direct_skip_reason(
            value_dtype, index_dtype, op=op_name
        )
        if direct_reason is None:
            return "hipsparse", None
        return None, direct_reason
    skip_reason = _cupy_cusparse_spmv_skip_reason(value_dtype)
    if skip_reason is None:
        return "cupy_cusparse", None
    return None, skip_reason


def _spmv_coo_reference_backend(value_dtype, index_dtype, op="non"):
    sparse_backend, reason = _spmv_coo_sparse_ref_backend(value_dtype, index_dtype, op=op)
    if sparse_backend is not None:
        return sparse_backend, None
    return "torch", reason


def _spmv_coo_ref_pytorch(
    data,
    row,
    col,
    x,
    shape,
    out_dtype=None,
    op="non",
    return_compute=False,
    reference_compute_dtype=False,
):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    if reference_compute_dtype:
        compute_dtype = _sparse_reference_compute_dtype(out_dtype)
    elif out_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = out_dtype
    data_ref = data.to(compute_dtype)
    x_ref = x.to(compute_dtype)
    coo_ref = torch.sparse_coo_tensor(
        torch.stack([row.to(torch.int64), col.to(torch.int64)]),
        data_ref,
        shape,
        device=data.device,
    ).coalesce()
    y_ref = _apply_torch_sparse_matmul_op(coo_ref, x_ref, op)
    if return_compute:
        return y_ref
    return _cast_sparse_reference_output(y_ref, out_dtype)


def _spmv_coo_ref_cupy(
    data,
    row,
    col,
    x,
    shape,
    out_dtype=None,
    op="non",
    reference_compute_dtype=False,
):
    _require_cupy()
    out_dtype = data.dtype if out_dtype is None else out_dtype
    compute_dtype = (
        _sparse_reference_compute_dtype(out_dtype)
        if reference_compute_dtype
        else out_dtype
    )
    data_ref = data.to(compute_dtype)
    x_ref = x.to(compute_dtype)
    data_cp = _cupy_from_torch(data_ref)
    row_cp = _cupy_from_torch(row.to(torch.int64))
    col_cp = _cupy_from_torch(col.to(torch.int64))
    x_cp = _cupy_from_torch(x_ref)
    matrix = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
    y_ref = _apply_cupy_sparse_matmul_op(matrix, x_cp, op)
    return _cast_sparse_reference_output(_torch_from_cupy(y_ref), out_dtype)


def _prepare_spmv_coo_ref_hipsparse(
    data,
    row,
    col,
    x,
    shape,
    out=None,
    op="non",
):
    op_name = _normalize_sparse_reference_op(op)
    skip_reason = _hipsparse_spmv_coo_direct_skip_reason(
        data.dtype, row.dtype, op=op_name
    )
    if skip_reason is not None:
        raise RuntimeError(skip_reason)
    if not all(torch.is_tensor(t) for t in (data, row, col, x)):
        raise TypeError("data, row, col, x must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, x)):
        raise ValueError("data, row, col, x must all be CUDA tensors")
    if not all(t.device == data.device for t in (row, col, x)):
        raise ValueError("data, row, col, x must be on the same CUDA device")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1 or x.ndim != 1:
        raise ValueError("data, row, col, x must all be 1D tensors")
    if row.numel() != data.numel() or col.numel() != data.numel():
        raise ValueError("data, row, col must have the same length")
    if row.dtype != col.dtype:
        raise ValueError(
            "hipSPARSE COO SpMV direct reference requires row/col to share the same index dtype"
        )

    n_rows, n_cols = int(shape[0]), int(shape[1])
    x_size = n_cols if op_name == "non" else n_rows
    y_size = n_rows if op_name == "non" else n_cols
    if x.numel() != x_size:
        raise ValueError(f"x length must be {x_size} for op={op_name}")

    data = data.contiguous()
    row = row.contiguous()
    col = col.contiguous()
    x = x.contiguous()
    value_type = _hipsparse_value_type(data.dtype)
    alpha = _hipsparse_scalar(data.dtype, 1.0, 0.0)
    beta = _hipsparse_scalar(data.dtype, 0.0, 0.0)
    index_type = _hipsparse_index_type(row.dtype, "hipSPARSE COO SpMV")
    op_enum = _hipsparse_spmv_operation(op_name, "hipSPARSE COO SpMV")

    y = out
    if y is None:
        y = torch.zeros(y_size, dtype=data.dtype, device=data.device)
    else:
        if not torch.is_tensor(y):
            raise TypeError("out must be a torch.Tensor")
        if not y.is_cuda or y.device != data.device:
            raise ValueError("out must be a CUDA tensor on the same device as data")
        if y.dtype != data.dtype or y.shape != (y_size,):
            raise ValueError("out must match the result shape and dtype")
        if not y.is_contiguous():
            raise ValueError("out must be contiguous")
        y.zero_()

    if y_size == 0:
        return {
            "backend": "hipsparse",
            "buffer_size": 0,
            "format": "coo",
            "y": y,
            "empty": True,
        }

    handle = None
    spmat = None
    vecx = None
    vecy = None
    workspace = 0
    workspace_allocated = False
    try:
        handle = _hip_check_result(hipsparse.hipsparseCreate(), "hipsparseCreate")
        ptr_type = type(handle)

        vecx = ptr_type()
        vecy = ptr_type()
        vecx_ref = vecx.createRef()
        vecy_ref = vecy.createRef()

        row_ptr = HipPointer.fromObj(row.data_ptr())
        col_ptr = HipPointer.fromObj(col.data_ptr())
        values_ptr = HipPointer.fromObj(data.data_ptr())
        x_ptr = HipPointer.fromObj(x.data_ptr())
        y_ptr = HipPointer.fromObj(y.data_ptr())

        index_base = _hipsparse_lookup(
            "hipsparseIndexBase_t", ("HIPSPARSE_INDEX_BASE_ZERO",)
        )
        alg = _hipsparse_lookup(
            "hipsparseSpMVAlg_t",
            ("HIPSPARSE_SPMV_ALG_DEFAULT", "HIPSPARSE_MV_ALG_DEFAULT"),
        )

        spmat = _hipsparse_create_coo_descriptor(
            n_rows,
            n_cols,
            int(data.numel()),
            row_ptr,
            col_ptr,
            values_ptr,
            index_type,
            index_base,
            value_type,
        )
        _hip_check_result(
            hipsparse.hipsparseCreateDnVec(vecx_ref, x_size, x_ptr, value_type),
            "hipsparseCreateDnVec(x)",
        )
        _hip_check_result(
            hipsparse.hipsparseCreateDnVec(vecy_ref, y_size, y_ptr, value_type),
            "hipsparseCreateDnVec(y)",
        )

        size_out = ctypes.c_size_t()
        _hip_check_result(
            hipsparse.hipsparseSpMV_bufferSize(
                handle,
                op_enum,
                alpha,
                spmat,
                vecx,
                beta,
                vecy,
                value_type,
                alg,
                size_out,
            ),
            "hipsparseSpMV_bufferSize",
        )
        buffer_size = int(size_out.value)
        if buffer_size > 0:
            workspace = _hip_check_result(hip.hipMalloc(buffer_size), "hipMalloc")
            workspace_allocated = True
        else:
            workspace = 0
        return {
            "backend": "hipsparse",
            "buffer_size": buffer_size,
            "format": "coo",
            "handle": handle,
            "spmat": spmat,
            "vecx": vecx,
            "vecy": vecy,
            "workspace": workspace,
            "workspace_allocated": workspace_allocated,
            "op_enum": op_enum,
            "alpha": alpha,
            "beta": beta,
            "value_type": value_type,
            "alg": alg,
            "y": y,
            "empty": False,
        }
    finally:
        if handle is None and vecy is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroyDnVec(vecy), "hipsparseDestroyDnVec(y)"
                )
            except Exception:
                pass
        if handle is None and vecx is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroyDnVec(vecx), "hipsparseDestroyDnVec(x)"
                )
            except Exception:
                pass
        if handle is None and spmat is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroySpMat(spmat), "hipsparseDestroySpMat"
                )
            except Exception:
                pass
        if handle is None and workspace_allocated:
            try:
                _hip_check_result(hip.hipFree(workspace), "hipFree")
            except Exception:
                pass


def _run_spmv_coo_ref_hipsparse_prepared(state):
    if state.get("empty"):
        return state["y"]
    _hip_check_result(
        hipsparse.hipsparseSpMV(
            state["handle"],
            state["op_enum"],
            state["alpha"],
            state["spmat"],
            state["vecx"],
            state["beta"],
            state["vecy"],
            state["value_type"],
            state["alg"],
            state["workspace"],
        ),
        "hipsparseSpMV",
    )
    return state["y"]


def _destroy_spmv_coo_ref_hipsparse_prepared(state):
    vecy = state.get("vecy")
    vecx = state.get("vecx")
    spmat = state.get("spmat")
    workspace_allocated = bool(state.get("workspace_allocated"))
    workspace = state.get("workspace", 0)
    handle = state.get("handle")
    if vecy is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroyDnVec(vecy), "hipsparseDestroyDnVec(y)"
            )
        except Exception:
            pass
    if vecx is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroyDnVec(vecx), "hipsparseDestroyDnVec(x)"
            )
        except Exception:
            pass
    if spmat is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroySpMat(spmat), "hipsparseDestroySpMat"
            )
        except Exception:
            pass
    if workspace_allocated:
        try:
            _hip_check_result(hip.hipFree(workspace), "hipFree")
        except Exception:
            pass
    if handle is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroy(handle), "hipsparseDestroy")
        except Exception:
            pass


def spmv_coo_ref_hipsparse(
    data,
    row,
    col,
    x,
    shape,
    out=None,
    op="non",
    return_metadata=False,
):
    state = _prepare_spmv_coo_ref_hipsparse(
        data,
        row,
        col,
        x,
        shape,
        out=out,
        op=op,
    )
    try:
        y = _run_spmv_coo_ref_hipsparse_prepared(state)
        metadata = {
            "backend": "hipsparse",
            "buffer_size": int(state.get("buffer_size", 0)),
            "format": "coo",
        }
        if return_metadata:
            return y, metadata
        return y
    finally:
        _destroy_spmv_coo_ref_hipsparse_prepared(state)


def _prepare_spmv_csr_ref_hipsparse(
    data,
    indices,
    indptr,
    x,
    shape,
    out=None,
    op="non",
):
    op_name = _normalize_spmv_reference_op(op)
    skip_reason = _hipsparse_spmv_csr_skip_reason(data.dtype, indices.dtype, op=op_name)
    if skip_reason is not None:
        raise RuntimeError(skip_reason)
    if not all(torch.is_tensor(t) for t in (data, indices, indptr, x)):
        raise TypeError("data, indices, indptr, x must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, indices, indptr, x)):
        raise ValueError("data, indices, indptr, x must all be CUDA tensors")
    if not all(t.device == data.device for t in (indices, indptr, x)):
        raise ValueError("data, indices, indptr, x must be on the same CUDA device")

    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1 or x.ndim != 1:
        raise ValueError("data, indices, indptr, x must all be 1D tensors")
    if indices.numel() != data.numel():
        raise ValueError("data and indices must have the same length")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    x_size = n_cols if op_name == "non" else n_rows
    y_size = n_rows if op_name == "non" else n_cols
    if x.numel() != x_size:
        raise ValueError(f"x length must be {x_size} for op={op_name}")

    data = data.contiguous()
    indices = indices.contiguous()
    indptr = indptr.contiguous()
    x = x.contiguous()
    value_type = _hipsparse_value_type(data.dtype)
    alpha = _hipsparse_scalar(data.dtype, 1.0, 0.0)
    beta = _hipsparse_scalar(data.dtype, 0.0, 0.0)
    row_index_type = _hipsparse_index_type(
        indptr.dtype, "hipSPARSE CSR SpMV row offsets"
    )
    col_index_type = _hipsparse_index_type(
        indices.dtype, "hipSPARSE CSR SpMV column indices"
    )
    op_enum = _hipsparse_spmv_operation(op_name, "hipSPARSE CSR SpMV")

    y = out
    if y is None:
        y = torch.zeros(y_size, dtype=data.dtype, device=data.device)
    else:
        if not torch.is_tensor(y):
            raise TypeError("out must be a torch.Tensor")
        if not y.is_cuda or y.device != data.device:
            raise ValueError("out must be a CUDA tensor on the same device as data")
        if y.dtype != data.dtype or y.shape != (y_size,):
            raise ValueError("out must match the result shape and dtype")
        if not y.is_contiguous():
            raise ValueError("out must be contiguous")
        y.zero_()

    if y_size == 0:
        return {
            "backend": "hipsparse",
            "buffer_size": 0,
            "y": y,
            "empty": True,
        }

    handle = None
    spmat = None
    vecx = None
    vecy = None
    workspace = 0
    workspace_allocated = False
    try:
        handle = _hip_check_result(hipsparse.hipsparseCreate(), "hipsparseCreate")
        ptr_type = type(handle)

        # hip-python dense descriptor outputs must be created via createRef().
        vecx = ptr_type()
        vecy = ptr_type()
        vecx_ref = vecx.createRef()
        vecy_ref = vecy.createRef()

        # CSR pointers must wrap tensor.data_ptr(), not the tensor object itself.
        row_ptr = HipPointer.fromObj(indptr.data_ptr())
        col_ind = HipPointer.fromObj(indices.data_ptr())
        values_ptr = HipPointer.fromObj(data.data_ptr())
        x_ptr = HipPointer.fromObj(x.data_ptr())
        y_ptr = HipPointer.fromObj(y.data_ptr())

        index_base = _hipsparse_lookup(
            "hipsparseIndexBase_t", ("HIPSPARSE_INDEX_BASE_ZERO",)
        )
        alg = _hipsparse_lookup(
            "hipsparseSpMVAlg_t",
            ("HIPSPARSE_SPMV_ALG_DEFAULT", "HIPSPARSE_MV_ALG_DEFAULT"),
        )

        spmat = _hipsparse_create_csr_descriptor(
            n_rows,
            n_cols,
            int(data.numel()),
            row_ptr,
            col_ind,
            values_ptr,
            row_index_type,
            col_index_type,
            index_base,
            value_type,
        )
        _hip_check_result(
            hipsparse.hipsparseCreateDnVec(vecx_ref, x_size, x_ptr, value_type),
            "hipsparseCreateDnVec(x)",
        )
        _hip_check_result(
            hipsparse.hipsparseCreateDnVec(vecy_ref, y_size, y_ptr, value_type),
            "hipsparseCreateDnVec(y)",
        )

        # alpha/beta must be passed directly, not via ctypes.byref(...).
        # Current hip-python exposes hipsparseSpMV_bufferSize with an explicit
        # c_size_t output slot instead of returning the size as a payload.
        size_out = ctypes.c_size_t()
        _hip_check_result(
            hipsparse.hipsparseSpMV_bufferSize(
                handle,
                op_enum,
                alpha,
                spmat,
                vecx,
                beta,
                vecy,
                value_type,
                alg,
                size_out,
            ),
            "hipsparseSpMV_bufferSize",
        )
        buffer_size = int(size_out.value)
        if buffer_size > 0:
            workspace = _hip_check_result(
                hip.hipMalloc(buffer_size), "hipMalloc"
            )
            workspace_allocated = True
        else:
            # hipSPARSE may legitimately report buffer_size == 0 for small inputs.
            workspace = 0
        return {
            "backend": "hipsparse",
            "buffer_size": buffer_size,
            "handle": handle,
            "spmat": spmat,
            "vecx": vecx,
            "vecy": vecy,
            "workspace": workspace,
            "workspace_allocated": workspace_allocated,
            "op_enum": op_enum,
            "alpha": alpha,
            "beta": beta,
            "value_type": value_type,
            "alg": alg,
            "y": y,
            "empty": False,
        }
    finally:
        if handle is None and vecy is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroyDnVec(vecy), "hipsparseDestroyDnVec(y)"
                )
            except Exception:
                pass
        if handle is None and vecx is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroyDnVec(vecx), "hipsparseDestroyDnVec(x)"
                )
            except Exception:
                pass
        if handle is None and spmat is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseDestroySpMat(spmat), "hipsparseDestroySpMat"
                )
            except Exception:
                pass
        if handle is None and workspace_allocated:
            try:
                _hip_check_result(hip.hipFree(workspace), "hipFree")
            except Exception:
                pass


def _run_spmv_csr_ref_hipsparse_prepared(state):
    if state.get("empty"):
        return state["y"]
    _hip_check_result(
        hipsparse.hipsparseSpMV(
            state["handle"],
            state["op_enum"],
            state["alpha"],
            state["spmat"],
            state["vecx"],
            state["beta"],
            state["vecy"],
            state["value_type"],
            state["alg"],
            state["workspace"],
        ),
        "hipsparseSpMV",
    )
    return state["y"]


def _destroy_spmv_csr_ref_hipsparse_prepared(state):
    vecy = state.get("vecy")
    vecx = state.get("vecx")
    spmat = state.get("spmat")
    workspace_allocated = bool(state.get("workspace_allocated"))
    workspace = state.get("workspace", 0)
    handle = state.get("handle")
    if vecy is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroyDnVec(vecy), "hipsparseDestroyDnVec(y)"
            )
        except Exception:
            pass
    if vecx is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroyDnVec(vecx), "hipsparseDestroyDnVec(x)"
            )
        except Exception:
            pass
    if spmat is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroySpMat(spmat), "hipsparseDestroySpMat"
            )
        except Exception:
            pass
    if workspace_allocated:
        try:
            _hip_check_result(hip.hipFree(workspace), "hipFree")
        except Exception:
            pass
    if handle is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroy(handle), "hipsparseDestroy")
        except Exception:
            pass


def spmv_csr_ref_hipsparse(
    data,
    indices,
    indptr,
    x,
    shape,
    out=None,
    op="non",
    return_metadata=False,
):
    state = _prepare_spmv_csr_ref_hipsparse(
        data,
        indices,
        indptr,
        x,
        shape,
        out=out,
        op=op,
    )
    try:
        y = _run_spmv_csr_ref_hipsparse_prepared(state)
        metadata = {"backend": "hipsparse", "buffer_size": int(state.get("buffer_size", 0))}
        if return_metadata:
            return y, metadata
        return y
    finally:
        _destroy_spmv_csr_ref_hipsparse_prepared(state)


def _spmv_csr_reference(
    data,
    indices,
    indptr,
    x,
    shape,
    out_dtype=None,
    op="non",
    reference_compute_dtype=False,
    return_metadata=False,
):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    op_name = _normalize_spmv_reference_op(op)
    backend, fallback_reason = _spmv_csr_reference_backend(
        data.dtype, indices.dtype, op=op_name
    )
    if backend == "hipsparse":
        result = spmv_csr_ref_hipsparse(
            data,
            indices,
            indptr,
            x,
            shape,
            op=op_name,
        )
    elif backend == "cupy_cusparse":
        result = _spmv_csr_ref_cupy(
            data,
            indices,
            indptr,
            x,
            shape,
            out_dtype=out_dtype,
            op=op_name,
            reference_compute_dtype=reference_compute_dtype,
        )
    else:
        result = _spmv_csr_ref_pytorch(
            data,
            indices,
            indptr,
            x,
            shape,
            out_dtype=out_dtype,
            op=op_name,
            reference_compute_dtype=reference_compute_dtype,
        )
    metadata = {"backend": backend, "fallback_reason": fallback_reason}
    if return_metadata:
        return result, metadata
    return result


def _benchmark_spmv_csr_sparse_ref(
    data,
    indices,
    indptr,
    x,
    shape,
    warmup,
    iters,
    op="non",
    include_csc=False,
):
    op_name = _normalize_spmv_reference_op(op)
    backend, reason = _spmv_csr_sparse_ref_backend(data.dtype, indices.dtype, op=op_name)
    result = {
        "backend": backend,
        "values": None,
        "ms": None,
        "csc_ms": None,
        "reason": reason,
    }
    if backend is None:
        return result
    if backend == "hipsparse":
        values, ms = _benchmark_prepared_cuda_op(
            lambda: _prepare_spmv_csr_ref_hipsparse(
                data, indices, indptr, x, shape, op=op_name
            ),
            _run_spmv_csr_ref_hipsparse_prepared,
            _destroy_spmv_csr_ref_hipsparse_prepared,
            warmup=warmup,
            iters=iters,
        )
        result["values"] = values
        result["ms"] = ms
        result["reason"] = None
        return result

    data_cp = _cupy_from_torch(data)
    ind_cp = _cupy_from_torch(indices.to(torch.int64))
    ptr_cp = _cupy_from_torch(indptr.to(torch.int64))
    x_cp = _cupy_from_torch(x)
    matrix = cpx_sparse.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
    values_cp, ms = _benchmark_cuda_op(
        lambda: _apply_cupy_spmv_op(matrix, x_cp, op_name),
        warmup=warmup,
        iters=iters,
    )
    result["values"] = _torch_from_cupy(values_cp)
    result["ms"] = ms
    result["reason"] = None
    if include_csc:
        matrix_csc = matrix.tocsc()
        values_csc, csc_ms = _benchmark_cuda_op(
            lambda: _apply_cupy_spmv_op(matrix_csc, x_cp, op_name),
            warmup=warmup,
            iters=iters,
        )
        _ = values_csc
        result["csc_ms"] = csc_ms
    return result


def _spmv_coo_reference(
    data,
    row,
    col,
    x,
    shape,
    out_dtype=None,
    op="non",
    reference_compute_dtype=False,
    return_metadata=False,
):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    op_name = _normalize_sparse_reference_op(op)
    backend, fallback_reason = _spmv_coo_reference_backend(
        data.dtype, row.dtype, op=op_name
    )
    if backend == "hipsparse":
        result = spmv_coo_ref_hipsparse(
            data,
            row,
            col,
            x,
            shape,
            op=op_name,
        )
    elif backend == "cupy_cusparse":
        result = _spmv_coo_ref_cupy(
            data,
            row,
            col,
            x,
            shape,
            out_dtype=out_dtype,
            op=op_name,
            reference_compute_dtype=reference_compute_dtype,
        )
    else:
        result = _spmv_coo_ref_pytorch(
            data,
            row,
            col,
            x,
            shape,
            out_dtype=out_dtype,
            op=op_name,
            reference_compute_dtype=reference_compute_dtype,
        )
    metadata = {"backend": backend, "fallback_reason": fallback_reason}
    if return_metadata:
        return result, metadata
    return result


def _benchmark_spmv_coo_sparse_ref(
    data,
    row,
    col,
    x,
    shape,
    warmup,
    iters,
    op="non",
):
    op_name = _normalize_sparse_reference_op(op)
    backend, reason = _spmv_coo_sparse_ref_backend(data.dtype, row.dtype, op=op_name)
    result = {
        "backend": backend,
        "values": None,
        "ms": None,
        "reason": reason,
    }
    if backend is None:
        return result
    if backend == "hipsparse":
        values, ms = _benchmark_prepared_cuda_op(
            lambda: _prepare_spmv_coo_ref_hipsparse(
                data, row, col, x, shape, op=op_name
            ),
            _run_spmv_coo_ref_hipsparse_prepared,
            _destroy_spmv_coo_ref_hipsparse_prepared,
            warmup=warmup,
            iters=iters,
        )
        result["values"] = values
        result["ms"] = ms
        result["reason"] = None
        return result

    data_cp = _cupy_from_torch(data)
    row_cp = _cupy_from_torch(row.to(torch.int64))
    col_cp = _cupy_from_torch(col.to(torch.int64))
    x_cp = _cupy_from_torch(x)
    matrix = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
    values_cp, ms = _benchmark_cuda_op(
        lambda: _apply_cupy_sparse_matmul_op(matrix, x_cp, op_name),
        warmup=warmup,
        iters=iters,
    )
    result["values"] = _torch_from_cupy(values_cp)
    result["ms"] = ms
    result["reason"] = None
    return result


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
    sparse_values,
    indices,
    dense_size=None,
    out=None,
    dtype_policy="auto",
    return_metadata=False,
):
    if sparse_values.ndim != 1:
        raise ValueError("sparse_values must be a 1D tensor")
    if indices.ndim != 1:
        raise ValueError("indices must be a 1D tensor")
    if sparse_values.numel() != indices.numel():
        raise ValueError(
            "sparse_values and indices must have the same number of elements"
        )
    if not sparse_values.is_cuda or not indices.is_cuda:
        raise ValueError("sparse_values and indices must both be CUDA tensors")
    if sparse_values.dtype not in SUPPORTED_VALUE_DTYPES:
        raise TypeError(
            f"sparse_values dtype must be one of: {', '.join(str(dt) for dt in SUPPORTED_VALUE_DTYPES)}"
        )
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")

    requested_value_dtype = sparse_values.dtype
    effective_value_dtype, fallback_applied, fallback_reason = (
        _resolve_scatter_value_dtype(requested_value_dtype, dtype_policy=dtype_policy)
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


def _benchmark_prepared_cuda_op(prepare_fn, run_fn, destroy_fn, warmup, iters):
    warmup = max(0, int(warmup))
    iters = max(1, int(iters))
    state = None
    start_ev = None
    stop_ev = None
    try:
        state = prepare_fn()
        output = None
        for _ in range(warmup):
            output = run_fn(state)

        if _is_rocm_runtime() and _hip_runtime_event_available():
            torch.cuda.synchronize()
            start_ev = _hip_check_result(hip.hipEventCreate(), "hipEventCreate(start)")
            stop_ev = _hip_check_result(hip.hipEventCreate(), "hipEventCreate(stop)")
            _hip_check_result(hip.hipEventRecord(start_ev, 0), "hipEventRecord(start)")
            for _ in range(iters):
                output = run_fn(state)
            _hip_check_result(hip.hipEventRecord(stop_ev, 0), "hipEventRecord(stop)")
            _hip_check_result(
                hip.hipEventSynchronize(stop_ev), "hipEventSynchronize(stop)"
            )
            return output, _hip_event_elapsed_ms(start_ev, stop_ev) / iters

        torch.cuda.synchronize()
        start_ev_torch = torch.cuda.Event(enable_timing=True)
        end_ev_torch = torch.cuda.Event(enable_timing=True)
        start_ev_torch.record()
        for _ in range(iters):
            output = run_fn(state)
        end_ev_torch.record()
        torch.cuda.synchronize()
        return output, start_ev_torch.elapsed_time(end_ev_torch) / iters
    finally:
        _destroy_hip_event(stop_ev)
        _destroy_hip_event(start_ev)
        if state is not None:
            destroy_fn(state)


def _benchmark_prepared_hip_event_op(
    prepare_fn, run_fn, destroy_fn, warmup, iters, event_stream="current"
):
    if not _is_rocm_runtime() or not _hip_runtime_event_available():
        return _benchmark_prepared_cuda_op(
            prepare_fn, run_fn, destroy_fn, warmup=warmup, iters=iters
        )

    warmup = max(0, int(warmup))
    iters = max(1, int(iters))
    state = None
    start_evt = None
    stop_evt = None
    try:
        state = prepare_fn()
        output = None
        for _ in range(warmup):
            output = run_fn(state)

        torch.cuda.synchronize()
        start_evt = _hip_check_result(hip.hipEventCreate(), "hipEventCreate(start)")
        stop_evt = _hip_check_result(hip.hipEventCreate(), "hipEventCreate(stop)")
        _hip_event_record_stream(start_evt, event_stream, "hipEventRecord(start)")
        for _ in range(iters):
            output = run_fn(state)
        _hip_event_record_stream(stop_evt, event_stream, "hipEventRecord(stop)")
        _hip_check_result(hip.hipEventSynchronize(stop_evt), "hipEventSynchronize(stop)")
        return output, _hip_event_elapsed_ms(start_evt, stop_evt) / iters
    finally:
        _destroy_hip_event(stop_evt)
        _destroy_hip_event(start_evt)
        if state is not None:
            destroy_fn(state)


def _benchmark_cuda_graph_op(
    op,
    *,
    graph_batch=100,
    warmup=20,
    repeats=10,
    capture_setup=None,
):
    """Measure allocation-free CUDA work without Python/FFI launch gaps."""
    graph_batch = max(1, int(graph_batch))
    warmup = max(0, int(warmup))
    repeats = max(1, int(repeats))

    torch.cuda.synchronize()
    capture_stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.stream(capture_stream):
        if capture_setup is not None:
            capture_setup()
        op()
    capture_stream.synchronize()

    with torch.cuda.graph(graph, stream=capture_stream):
        for _ in range(graph_batch):
            op()
    capture_stream.synchronize()

    with torch.cuda.stream(capture_stream):
        for _ in range(warmup):
            graph.replay()
    capture_stream.synchronize()

    samples_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        with torch.cuda.stream(capture_stream):
            start.record()
            graph.replay()
            end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)) / graph_batch)

    return statistics.median(samples_ms)
