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

"""Sparse triangular matrix-matrix solve (SpSM) for CSR/COO."""

from collections import OrderedDict

import ctypes

from . import _common as _common_mod
from ._common import *

HipPointer = _common_mod.HipPointer

SUPPORTED_SPSM_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)
SUPPORTED_SPSM_INDEX_DTYPES = (torch.int32,)
SPSM_NON_TRANS_PRIMARY_COMBOS = (
    ("csr", torch.float32, torch.int32),
    ("csr", torch.float64, torch.int32),
    ("csr", torch.complex64, torch.int32),
    ("csr", torch.complex128, torch.int32),
    ("coo", torch.float32, torch.int32),
    ("coo", torch.float64, torch.int32),
    ("coo", torch.complex64, torch.int32),
    ("coo", torch.complex128, torch.int32),
)
_SPSM_PREPROCESS_CACHE = OrderedDict()
_SPSM_PREPROCESS_CACHE_SIZE = 8
_SPSM_POLLING_RHS_TILE = 1024


def _try_set_hipsparse_current_stream(handle):
    set_stream = getattr(hipsparse, "hipsparseSetStream", None)
    if set_stream is None:
        return "hipSPARSE binding does not expose hipsparseSetStream"

    stream_ptr = _torch_current_stream_ptr()
    if stream_ptr is None:
        return "could not resolve torch current CUDA/HIP stream pointer"

    stream_args = []
    if HipPointer is not None:
        try:
            stream_args.append(HipPointer.fromObj(stream_ptr))
        except Exception:
            pass
    stream_args.extend([stream_ptr, ctypes.c_void_p(stream_ptr)])

    last_error = None
    for stream_arg in stream_args:
        try:
            _hip_check_result(set_stream(handle, stream_arg), "hipsparseSetStream")
            return None
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
    return f"hipsparseSetStream failed for torch current stream: {last_error}"


def _hipsparse_csrsm2_functions(value_dtype):
    prefix = {
        torch.float32: "S",
        torch.float64: "D",
        torch.complex64: "C",
        torch.complex128: "Z",
    }.get(value_dtype)
    if prefix is None:
        raise TypeError(f"hipSPARSE csrsm2 does not support {value_dtype}")
    return (
        getattr(hipsparse, f"hipsparse{prefix}csrsm2_bufferSizeExt"),
        getattr(hipsparse, f"hipsparse{prefix}csrsm2_analysis"),
        getattr(hipsparse, f"hipsparse{prefix}csrsm2_solve"),
    )


def _hipsparse_create_descriptor(attr_names, context, handle=None):
    ptr_type = type(handle) if handle is not None else None
    last_error = None
    for attr_name in attr_names:
        create_fn = getattr(hipsparse, attr_name, None) if hipsparse is not None else None
        if create_fn is None:
            continue
        try:
            raw = create_fn()
            if ptr_type is not None and isinstance(raw, ptr_type):
                return raw
            if hasattr(raw, "createRef"):
                return raw
            payload = _hip_check_result(raw, context)
            if payload is not None:
                return payload
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc

        if ptr_type is None:
            continue
        descr = ptr_type()
        attempts = []
        if hasattr(descr, "createRef"):
            attempts.append((descr.createRef(),))
        attempts.append((descr,))
        for args in attempts:
            try:
                payload = _hip_check_result(create_fn(*args), context)
                return payload if payload is not None else descr
            except TypeError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
    raise RuntimeError(f"{context} failed: {last_error}") from last_error


def _hipsparse_csrsm2_skip_reason(value_dtype, index_dtype, indptr_dtype=None):
    indptr_dtype = index_dtype if indptr_dtype is None else indptr_dtype
    if not _is_rocm_runtime():
        return "hipSPARSE csrsm2 requires a ROCm/DCU runtime"
    unavailable_reason = _hipsparse_unavailable_reason()
    if unavailable_reason is not None:
        return unavailable_reason
    if value_dtype not in SUPPORTED_SPSM_VALUE_DTYPES:
        return f"hipSPARSE csrsm2 has no value dtype mapping for {value_dtype}"
    if index_dtype != torch.int32 or indptr_dtype != torch.int32:
        return "hipSPARSE csrsm2 requires int32 CSR indices and row offsets"
    for symbol in ("hipMalloc", "hipFree"):
        if hip is None or not hasattr(hip, symbol):
            return f"hipSPARSE csrsm2 is unavailable: missing HIP runtime {symbol}"
    required = (
        "hipsparseCreate",
        "hipsparseDestroy",
        "hipsparseCreateMatDescr",
        "hipsparseDestroyMatDescr",
        "hipsparseSetMatType",
        "hipsparseSetMatIndexBase",
        "hipsparseSetMatFillMode",
        "hipsparseSetMatDiagType",
        "hipsparseCreateCsrsm2Info",
        "hipsparseDestroyCsrsm2Info",
    )
    for symbol in required:
        if not hasattr(hipsparse, symbol):
            return f"hipSPARSE csrsm2 is unavailable: missing {symbol}"
    try:
        _hipsparse_csrsm2_functions(value_dtype)
        _hipsparse_lookup(
            "hipsparseOperation_t", ("HIPSPARSE_OPERATION_NON_TRANSPOSE",)
        )
        _hipsparse_lookup(
            "hipsparseMatrixType_t", ("HIPSPARSE_MATRIX_TYPE_GENERAL",)
        )
        _hipsparse_lookup(
            "hipsparseIndexBase_t", ("HIPSPARSE_INDEX_BASE_ZERO",)
        )
        _hipsparse_lookup("hipsparseFillMode_t", ("HIPSPARSE_FILL_MODE_LOWER",))
        _hipsparse_lookup(
            "hipsparseDiagType_t", ("HIPSPARSE_DIAG_TYPE_NON_UNIT",)
        )
        _hipsparse_lookup(
            "hipsparseSolvePolicy_t",
            (
                "HIPSPARSE_SOLVE_POLICY_NO_LEVEL",
                "HIPSPARSE_SOLVE_POLICY_USE_LEVEL",
            ),
        )
        _hipsparse_scalar(value_dtype, 1.0, 0.0)
    except Exception as exc:
        return str(exc)
    return None


def _spsm_csr_sparse_ref_backend(value_dtype, index_dtype, indptr_dtype=None):
    reason = _hipsparse_csrsm2_skip_reason(
        value_dtype, index_dtype, indptr_dtype
    )
    if reason is None:
        return "hipsparse_csrsm2", None
    return None, reason


def _destroy_spsm_csr_ref_hipsparse_prepared(state):
    if not state:
        return
    if state.get("workspace_allocated"):
        try:
            _hip_check_result(hip.hipFree(state["workspace"]), "hipFree")
        except Exception:
            pass
    info = state.get("info")
    if info is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroyCsrsm2Info(info),
                "hipsparseDestroyCsrsm2Info",
            )
        except Exception:
            pass
    descr = state.get("descr")
    if descr is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseDestroyMatDescr(descr),
                "hipsparseDestroyMatDescr",
            )
        except Exception:
            pass
    handle = state.get("handle")
    if handle is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroy(handle), "hipsparseDestroy")
        except Exception:
            pass


def _time_hipsparse_call_ms(call):
    start_evt = None
    stop_evt = None
    if _hip_runtime_event_available():
        try:
            start_evt = _hip_check_result(hip.hipEventCreate(), "hipEventCreate(start)")
            stop_evt = _hip_check_result(hip.hipEventCreate(), "hipEventCreate(stop)")
            _hip_check_result(hip.hipEventRecord(start_evt, 0), "hipEventRecord(start)")
            call()
            _hip_check_result(hip.hipEventRecord(stop_evt, 0), "hipEventRecord(stop)")
            _hip_check_result(
                hip.hipEventSynchronize(stop_evt), "hipEventSynchronize(stop)"
            )
            return _hip_event_elapsed_ms(start_evt, stop_evt)
        finally:
            _destroy_hip_event(stop_evt)
            _destroy_hip_event(start_evt)

    torch.cuda.synchronize()
    start_evt_torch = torch.cuda.Event(enable_timing=True)
    stop_evt_torch = torch.cuda.Event(enable_timing=True)
    start_evt_torch.record()
    call()
    stop_evt_torch.record()
    torch.cuda.synchronize()
    return float(start_evt_torch.elapsed_time(stop_evt_torch))


def _spsm_filtered_avg_ms(times):
    if not times:
        return None
    times = [float(t) for t in times]
    if len(times) == 1:
        return times[0]
    ordered = sorted(times)
    n = len(ordered)
    if n % 2 == 0:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    else:
        median = ordered[n // 2]
    lo = median * 0.9
    hi = median * 1.1
    kept = [t for t in ordered if lo <= t <= hi]
    return sum(kept) / len(kept) if kept else median


def _prepare_spsm_csr_ref_hipsparse(
    data,
    indices,
    indptr,
    B,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
):
    reason = _hipsparse_csrsm2_skip_reason(
        data.dtype, indices.dtype, indptr.dtype
    )
    if reason is not None:
        raise RuntimeError(reason)
    if not all(torch.is_tensor(t) for t in (data, indices, indptr, B)):
        raise TypeError("data, indices, indptr, B must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, indices, indptr, B)):
        raise ValueError("hipSPARSE csrsm2 inputs must all be GPU tensors")
    if not all(t.device == data.device for t in (indices, indptr, B)):
        raise ValueError("hipSPARSE csrsm2 inputs must be on the same device")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, and indptr must be 1D")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense matrix")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError("hipSPARSE csrsm2 requires a square sparse matrix")
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    if data.numel() != indices.numel():
        raise ValueError("data and indices must contain the same number of entries")
    if B.shape[0] != n_rows:
        raise ValueError(f"B.shape[0] must equal n_rows={n_rows}")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match sparse value dtype")

    state = {
        "backend": "hipsparse_csrsm2",
        "handle": None,
        "descr": None,
        "info": None,
        "workspace": 0,
        "workspace_allocated": False,
    }
    try:
        data = data.contiguous()
        indices = indices.contiguous()
        indptr = indptr.contiguous()
        # csrsm2 uses column-major dense storage. B.T contiguous has that layout.
        rhs_col_major = B.transpose(0, 1).contiguous()
        solution_col_major = rhs_col_major.clone()
        buffer_size_fn, analysis_fn, solve_fn = _hipsparse_csrsm2_functions(
            data.dtype
        )

        handle = _hip_check_result(hipsparse.hipsparseCreate(), "hipsparseCreate")
        state["handle"] = handle
        state["stream_binding_warning"] = _try_set_hipsparse_current_stream(handle)
        descr = _hipsparse_create_descriptor(
            ("hipsparseCreateMatDescr",),
            "hipsparseCreateMatDescr",
            handle=handle,
        )
        state["descr"] = descr
        _hip_check_result(
            hipsparse.hipsparseSetMatType(
                descr,
                _hipsparse_lookup(
                    "hipsparseMatrixType_t", ("HIPSPARSE_MATRIX_TYPE_GENERAL",)
                ),
            ),
            "hipsparseSetMatType",
        )
        _hip_check_result(
            hipsparse.hipsparseSetMatIndexBase(
                descr,
                _hipsparse_lookup(
                    "hipsparseIndexBase_t", ("HIPSPARSE_INDEX_BASE_ZERO",)
                ),
            ),
            "hipsparseSetMatIndexBase",
        )
        _hip_check_result(
            hipsparse.hipsparseSetMatFillMode(
                descr,
                _hipsparse_lookup(
                    "hipsparseFillMode_t",
                    ("HIPSPARSE_FILL_MODE_LOWER",)
                    if lower
                    else ("HIPSPARSE_FILL_MODE_UPPER",),
                ),
            ),
            "hipsparseSetMatFillMode",
        )
        _hip_check_result(
            hipsparse.hipsparseSetMatDiagType(
                descr,
                _hipsparse_lookup(
                    "hipsparseDiagType_t",
                    ("HIPSPARSE_DIAG_TYPE_UNIT",)
                    if unit_diagonal
                    else ("HIPSPARSE_DIAG_TYPE_NON_UNIT",),
                ),
            ),
            "hipsparseSetMatDiagType",
        )
        info = _hipsparse_create_descriptor(
            ("hipsparseCreateCsrsm2Info",),
            "hipsparseCreateCsrsm2Info",
            handle=handle,
        )
        state["info"] = info

        alpha = _hipsparse_scalar(data.dtype, 1.0, 0.0)
        op_n = _hipsparse_lookup(
            "hipsparseOperation_t", ("HIPSPARSE_OPERATION_NON_TRANSPOSE",)
        )
        policy = _hipsparse_lookup(
            "hipsparseSolvePolicy_t",
            (
                "HIPSPARSE_SOLVE_POLICY_NO_LEVEL",
                "HIPSPARSE_SOLVE_POLICY_USE_LEVEL",
            ),
        )
        dense_ptr = HipPointer.fromObj(solution_col_major.data_ptr())
        common_args = (
            handle,
            0,
            op_n,
            op_n,
            n_rows,
            int(B.shape[1]),
            int(data.numel()),
            alpha,
            descr,
            HipPointer.fromObj(data.data_ptr()),
            HipPointer.fromObj(indptr.data_ptr()),
            HipPointer.fromObj(indices.data_ptr()),
            dense_ptr,
            n_rows,
            info,
            policy,
        )
        size_out = ctypes.c_size_t()
        _hip_check_result(
            buffer_size_fn(*common_args, size_out),
            "hipsparseXcsrsm2_bufferSizeExt",
        )
        buffer_size = int(size_out.value)
        if buffer_size > 0:
            state["workspace"] = _hip_check_result(
                hip.hipMalloc(buffer_size), "hipMalloc(csrsm2 workspace)"
            )
            state["workspace_allocated"] = True
        analysis_ms = _time_hipsparse_call_ms(
            lambda: _hip_check_result(
                analysis_fn(*common_args, state["workspace"]),
                "hipsparseXcsrsm2_analysis",
            )
        )
        state.update(
            {
                "analysis_ms": analysis_ms,
                "buffer_size": buffer_size,
                "data": data,
                "indices": indices,
                "indptr": indptr,
                "rhs": rhs_col_major,
                "solution_col_major": solution_col_major,
                "solve_fn": solve_fn,
                "solve_args": common_args,
            }
        )
        return state
    except Exception:
        _destroy_spsm_csr_ref_hipsparse_prepared(state)
        raise


def _run_spsm_csr_ref_hipsparse_prepared(state):
    state["solution_col_major"].copy_(state["rhs"])
    _hip_check_result(
        state["solve_fn"](*state["solve_args"], state["workspace"]),
        "hipsparseXcsrsm2_solve",
    )
    return state["solution_col_major"].transpose(0, 1).contiguous()


def _spsm_csr_ref_hipsparse(
    data,
    indices,
    indptr,
    B,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    return_metadata=False,
):
    state = _prepare_spsm_csr_ref_hipsparse(
        data,
        indices,
        indptr,
        B,
        shape,
        lower=lower,
        unit_diagonal=unit_diagonal,
    )
    try:
        values = _run_spsm_csr_ref_hipsparse_prepared(state)
        metadata = {
            "backend": "hipsparse_csrsm2",
            "analysis_ms": float(state.get("analysis_ms", 0.0)),
            "buffer_size": int(state.get("buffer_size", 0)),
            "format": "csr",
        }
        return (values, metadata) if return_metadata else values
    finally:
        _destroy_spsm_csr_ref_hipsparse_prepared(state)


def _benchmark_spsm_csr_sparse_ref(
    data,
    indices,
    indptr,
    B,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    warmup=0,
    iters=1,
):
    backend, reason = _spsm_csr_sparse_ref_backend(
        data.dtype, indices.dtype, indptr.dtype
    )
    result = {
        "backend": backend,
        "values": None,
        "ms": None,
        "reason": reason,
    }
    if backend is None:
        return result
    state = None
    try:
        values = None
        for _ in range(warmup):
            state = _prepare_spsm_csr_ref_hipsparse(
                data,
                indices,
                indptr,
                B,
                shape,
                lower=lower,
                unit_diagonal=unit_diagonal,
            )
            try:
                values = _run_spsm_csr_ref_hipsparse_prepared(state)
            finally:
                _destroy_spsm_csr_ref_hipsparse_prepared(state)
                state = None
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            state = _prepare_spsm_csr_ref_hipsparse(
                data,
                indices,
                indptr,
                B,
                shape,
                lower=lower,
                unit_diagonal=unit_diagonal,
            )
            try:
                values = _run_spsm_csr_ref_hipsparse_prepared(state)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start_time) * 1000.0)
            finally:
                _destroy_spsm_csr_ref_hipsparse_prepared(state)
                state = None
        result["values"] = values
        result["ms"] = _spsm_filtered_avg_ms(times)
        result["reason"] = None
    except Exception as exc:
        result["reason"] = str(exc)
    finally:
        _destroy_spsm_csr_ref_hipsparse_prepared(state)
    return result


def _clear_spsm_preprocess_cache():
    _SPSM_PREPROCESS_CACHE.clear()


def _is_non_transpose(op):
    op_str = str(op).upper()
    return op_str in ("NON_TRANS", "NON_TRANSPOSE", "N")


def _validate_spsm_non_trans_combo(fmt_name, value_dtype, index_dtype):
    combo = (str(fmt_name).lower(), value_dtype, index_dtype)
    if combo not in SPSM_NON_TRANS_PRIMARY_COMBOS:
        raise TypeError(
            f"{fmt_name} SpSM currently supports NON_TRANS combinations: "
            "CSR/COO with float32/float64/complex64/complex128 and int32"
        )


def _validate_spsm_op_and_layout(opA, opB, major):
    if not _is_non_transpose(opA):
        raise NotImplementedError("Only op(A)=NON_TRANS is supported")
    if not _is_non_transpose(opB):
        raise NotImplementedError("Only op(B)=NON_TRANS is supported")
    if str(major).lower() != "row":
        raise NotImplementedError("Only row-major dense layout is supported")


def _prepare_spsm_csr_inputs(data, indices, indptr, B, shape, opA, opB, major):
    if not all(torch.is_tensor(t) for t in (data, indices, indptr, B)):
        raise TypeError("data, indices, indptr, B must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, indices, indptr, B)):
        raise ValueError("data, indices, indptr, B must all be CUDA tensors")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense matrix")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"SpSM requires square A, got shape={shape}")
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if B.shape[0] != n_rows:
        raise ValueError(f"B.shape[0] must equal n_rows={n_rows}")
    _validate_spsm_op_and_layout(opA, opB, major)
    if data.dtype not in SUPPORTED_SPSM_VALUE_DTYPES:
        raise TypeError("data dtype must be float32, float64, complex64, or complex128")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")
    if indices.dtype not in SUPPORTED_SPSM_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32")
    if indptr.dtype not in SUPPORTED_SPSM_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32")
    if indices.dtype != indptr.dtype:
        raise TypeError("indices and indptr dtype must match for SpSM CSR")
    input_index_dtype = indices.dtype
    _validate_spsm_non_trans_combo("csr", data.dtype, input_index_dtype)
    return (
        data.contiguous(),
        indices.contiguous(),
        indptr.contiguous(),
        B.contiguous(),
        n_rows,
        n_cols,
    )


def _prepare_spsm_coo_inputs(data, row, col, B, shape, opA, opB, major):
    if not all(torch.is_tensor(t) for t in (data, row, col, B)):
        raise TypeError("data, row, col, B must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, B)):
        raise ValueError("data, row, col, B must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, col must have same length")
    if int(data.numel()) > _INDEX_LIMIT_INT32:
        raise ValueError("nnz exceeds int32 kernel range")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"SpSM requires square A, got shape={shape}")
    if B.ndim != 2 or B.shape[0] != n_rows:
        raise ValueError("B must be 2D and B.shape[0] == n_rows")
    _validate_spsm_op_and_layout(opA, opB, major)
    if data.dtype not in SUPPORTED_SPSM_VALUE_DTYPES:
        raise TypeError("data dtype must be float32, float64, complex64, or complex128")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")
    if (
        row.dtype not in SUPPORTED_SPSM_INDEX_DTYPES
        or col.dtype not in SUPPORTED_SPSM_INDEX_DTYPES
    ):
        raise TypeError("row/col dtype must be torch.int32")
    if row.dtype != col.dtype:
        raise TypeError("row and col dtype must match for SpSM COO")
    _validate_spsm_non_trans_combo("coo", data.dtype, row.dtype)
    return (
        data.contiguous(),
        row.contiguous(),
        col.contiguous(),
        B.contiguous(),
        n_rows,
        n_cols,
    )


def _complex_interleaved_view(tensor):
    # Tensor.values() of a sparse CSR matrix is a strided *view* whose base is the
    # sparse tensor; torch.view_as_real can't derive strides through a sparse base
    # (it raises "Sparse CSR tensors do not have strides"), and .contiguous()
    # returns the same view when already contiguous. Materialize an owned copy.
    base = tensor._base
    if base is not None and base.layout != torch.strided:
        tensor = tensor.clone()
    return torch.view_as_real(tensor.contiguous()).reshape(-1).contiguous()


def _tensor_cache_token(tensor):
    try:
        storage_ptr = int(tensor.untyped_storage().data_ptr())
    except Exception:
        storage_ptr = 0
    return (
        str(tensor.device),
        str(tensor.dtype),
        tuple(int(v) for v in tensor.shape),
        int(tensor.numel()),
        storage_ptr,
        int(getattr(tensor, "_version", 0)),
    )


def _spsm_cache_get(cache, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _spsm_cache_put(cache, key, value, max_entries):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _spsm_preprocess_cache_key(fmt_name, tensors, shape, lower, unit_diagonal):
    return (
        str(fmt_name).lower(),
        bool(lower),
        bool(unit_diagonal),
        int(shape[0]),
        int(shape[1]),
        *(_tensor_cache_token(t) for t in tensors),
    )


def _prepare_spsm_kernel_row_ptr(indptr64):
    if indptr64.numel() == 0:
        return indptr64.to(torch.int32)
    if int(indptr64[-1].item()) <= _INDEX_LIMIT_INT32:
        return indptr64.to(torch.int32)
    return indptr64


def _alpha_is_one(alpha):
    if torch.is_tensor(alpha):
        if alpha.numel() != 1:
            return False
        return bool((alpha.detach().cpu() == 1).item())
    return alpha == 1 or alpha == 1.0


def _alpha_to_host_scalar(alpha):
    if torch.is_tensor(alpha):
        if alpha.numel() != 1:
            raise ValueError("alpha tensor must be scalar")
        value = alpha.detach().cpu().item()
        if isinstance(value, complex):
            return value
        return float(value)
    if isinstance(alpha, complex):
        return alpha
    return float(alpha)


@triton.jit
def _spsm_extract_diag_kernel_real(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ptr,
    n_rows,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    row = tl.program_id(0)
    if USE_FP64_ACC:
        diag = tl.full((), 1.0, tl.float64)
    else:
        diag = tl.full((), 1.0, tl.float32)
    if not UNIT_DIAG:
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        p = start
        while p < end:
            col = tl.load(indices_ptr + p)
            if col == row:
                val = tl.load(data_ptr + p)
                if USE_FP64_ACC:
                    diag = val.to(tl.float64)
                else:
                    diag = val.to(tl.float32)
            p += 1
    tl.store(diag_ptr + row, diag)


@triton.jit
def _spsm_extract_diag_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ri_ptr,
    n_rows,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    row = tl.program_id(0)
    if USE_FP64_ACC:
        diag_re = tl.full((), 1.0, tl.float64)
        diag_im = tl.full((), 0.0, tl.float64)
    else:
        diag_re = tl.full((), 1.0, tl.float32)
        diag_im = tl.full((), 0.0, tl.float32)
    if not UNIT_DIAG:
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        p = start
        while p < end:
            col = tl.load(indices_ptr + p)
            if col == row:
                diag_re = tl.load(data_ri_ptr + p * 2)
                diag_im = tl.load(data_ri_ptr + p * 2 + 1)
                if USE_FP64_ACC:
                    diag_re = diag_re.to(tl.float64)
                    diag_im = diag_im.to(tl.float64)
                else:
                    diag_re = diag_re.to(tl.float32)
                    diag_im = diag_im.to(tl.float32)
            p += 1
    tl.store(diag_ri_ptr + row * 2, diag_re)
    tl.store(diag_ri_ptr + row * 2 + 1, diag_im)


@triton.jit
def _spsm_csr_polling_kernel_real(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ptr,
    work_ptr,
    done_ptr,
    n_rows,
    n_rhs,
    stride_work0,
    alpha,
    BLOCK_RHS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if LOWER:
        row = pid_row
    else:
        row = n_rows - 1 - pid_row
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    row_i64 = row.to(tl.int64)
    rhs_offsets_i64 = rhs_offsets.to(tl.int64)
    rhs_mask = rhs_offsets < n_rhs

    if USE_FP64_ACC:
        rhs = tl.load(
            work_ptr + row_i64 * stride_work0 + rhs_offsets_i64,
            mask=rhs_mask,
            other=0.0,
        ).to(tl.float64)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float64)
        local_sum = rhs * alpha_val
    else:
        rhs = tl.load(
            work_ptr + row_i64 * stride_work0 + rhs_offsets_i64,
            mask=rhs_mask,
            other=0.0,
        ).to(tl.float32)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float32)
        local_sum = rhs * alpha_val

    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    flag_base = pid_rhs.to(tl.int64) * n_rows

    if LOWER:
        p = start
        while p < end:
            col = tl.load(indices_ptr + p)
            if col < row:
                val = tl.load(data_ptr + p)
                if USE_FP64_ACC:
                    val = val.to(tl.float64)
                else:
                    val = val.to(tl.float32)
                dep_flag_ptr = done_ptr + flag_base + col.to(tl.int64)
                ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                while ready == 0:
                    ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                x_vals = tl.load(
                    work_ptr + col.to(tl.int64) * stride_work0 + rhs_offsets_i64,
                    mask=rhs_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
                if USE_FP64_ACC:
                    x_vals = x_vals.to(tl.float64)
                else:
                    x_vals = x_vals.to(tl.float32)
                local_sum -= val * x_vals
            p += 1
    else:
        p = end - 1
        while p >= start:
            col = tl.load(indices_ptr + p)
            if col > row:
                val = tl.load(data_ptr + p)
                if USE_FP64_ACC:
                    val = val.to(tl.float64)
                else:
                    val = val.to(tl.float32)
                dep_flag_ptr = done_ptr + flag_base + col.to(tl.int64)
                ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                while ready == 0:
                    ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                x_vals = tl.load(
                    work_ptr + col.to(tl.int64) * stride_work0 + rhs_offsets_i64,
                    mask=rhs_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
                if USE_FP64_ACC:
                    x_vals = x_vals.to(tl.float64)
                else:
                    x_vals = x_vals.to(tl.float32)
                local_sum -= val * x_vals
            p -= 1

    diag = tl.load(diag_ptr + row)
    if USE_FP64_ACC:
        diag = diag.to(tl.float64)
    else:
        diag = diag.to(tl.float32)
    if UNIT_DIAG:
        out = local_sum
    else:
        out = local_sum / diag
    out = tl.where(out == out, out, 0.0)
    tl.store(
        work_ptr + row_i64 * stride_work0 + rhs_offsets_i64,
        out,
        mask=rhs_mask,
        cache_modifier=".wt",
    )
    # Every warp must finish publishing its RHS slice before the row is marked
    # ready for dependent programs.
    tl.debug_barrier()
    tl.atomic_or(done_ptr + flag_base + row_i64, 1, sem="release")


@triton.jit
def _spsm_csr_polling_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ri_ptr,
    work_ri_ptr,
    done_ptr,
    n_rows,
    n_rhs,
    stride_work0,
    alpha_re,
    alpha_im,
    BLOCK_RHS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if LOWER:
        row = pid_row
    else:
        row = n_rows - 1 - pid_row
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    row_i64 = row.to(tl.int64)
    rhs_offsets_i64 = rhs_offsets.to(tl.int64)
    rhs_mask = rhs_offsets < n_rhs
    base = (row_i64 * stride_work0 + rhs_offsets_i64) * 2

    rhs_re = tl.load(work_ri_ptr + base, mask=rhs_mask, other=0.0)
    rhs_im = tl.load(work_ri_ptr + base + 1, mask=rhs_mask, other=0.0)
    if USE_FP64_ACC:
        rhs_re = rhs_re.to(tl.float64)
        rhs_im = rhs_im.to(tl.float64)
        alpha_re_v = tl.full((BLOCK_RHS,), alpha_re, tl.float64)
        alpha_im_v = tl.full((BLOCK_RHS,), alpha_im, tl.float64)
        sum_re = rhs_re * alpha_re_v - rhs_im * alpha_im_v
        sum_im = rhs_re * alpha_im_v + rhs_im * alpha_re_v
    else:
        rhs_re = rhs_re.to(tl.float32)
        rhs_im = rhs_im.to(tl.float32)
        alpha_re_v = tl.full((BLOCK_RHS,), alpha_re, tl.float32)
        alpha_im_v = tl.full((BLOCK_RHS,), alpha_im, tl.float32)
        sum_re = rhs_re * alpha_re_v - rhs_im * alpha_im_v
        sum_im = rhs_re * alpha_im_v + rhs_im * alpha_re_v

    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    flag_base = pid_rhs.to(tl.int64) * n_rows

    if LOWER:
        p = start
        while p < end:
            col = tl.load(indices_ptr + p)
            if col < row:
                val_re = tl.load(data_ri_ptr + p * 2)
                val_im = tl.load(data_ri_ptr + p * 2 + 1)
                if USE_FP64_ACC:
                    val_re = val_re.to(tl.float64)
                    val_im = val_im.to(tl.float64)
                else:
                    val_re = val_re.to(tl.float32)
                    val_im = val_im.to(tl.float32)
                dep_flag_ptr = done_ptr + flag_base + col.to(tl.int64)
                ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                while ready == 0:
                    ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                x_base = (col.to(tl.int64) * stride_work0 + rhs_offsets_i64) * 2
                x_re = tl.load(
                    work_ri_ptr + x_base,
                    mask=rhs_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
                x_im = tl.load(
                    work_ri_ptr + x_base + 1,
                    mask=rhs_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
                if USE_FP64_ACC:
                    x_re = x_re.to(tl.float64)
                    x_im = x_im.to(tl.float64)
                else:
                    x_re = x_re.to(tl.float32)
                    x_im = x_im.to(tl.float32)
                sum_re -= val_re * x_re - val_im * x_im
                sum_im -= val_re * x_im + val_im * x_re
            p += 1
    else:
        p = end - 1
        while p >= start:
            col = tl.load(indices_ptr + p)
            if col > row:
                val_re = tl.load(data_ri_ptr + p * 2)
                val_im = tl.load(data_ri_ptr + p * 2 + 1)
                if USE_FP64_ACC:
                    val_re = val_re.to(tl.float64)
                    val_im = val_im.to(tl.float64)
                else:
                    val_re = val_re.to(tl.float32)
                    val_im = val_im.to(tl.float32)
                dep_flag_ptr = done_ptr + flag_base + col.to(tl.int64)
                ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                while ready == 0:
                    ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                x_base = (col.to(tl.int64) * stride_work0 + rhs_offsets_i64) * 2
                x_re = tl.load(
                    work_ri_ptr + x_base,
                    mask=rhs_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
                x_im = tl.load(
                    work_ri_ptr + x_base + 1,
                    mask=rhs_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
                if USE_FP64_ACC:
                    x_re = x_re.to(tl.float64)
                    x_im = x_im.to(tl.float64)
                else:
                    x_re = x_re.to(tl.float32)
                    x_im = x_im.to(tl.float32)
                prod_re = val_re * x_re - val_im * x_im
                prod_im = val_re * x_im + val_im * x_re
                sum_re -= prod_re
                sum_im -= prod_im
            p -= 1

    diag_re = tl.load(diag_ri_ptr + row * 2)
    diag_im = tl.load(diag_ri_ptr + row * 2 + 1)
    if USE_FP64_ACC:
        diag_re = diag_re.to(tl.float64)
        diag_im = diag_im.to(tl.float64)
    else:
        diag_re = diag_re.to(tl.float32)
        diag_im = diag_im.to(tl.float32)
    denom = diag_re * diag_re + diag_im * diag_im
    if UNIT_DIAG:
        out_re = sum_re
        out_im = sum_im
    else:
        out_re = (sum_re * diag_re + sum_im * diag_im) / denom
        out_im = (sum_im * diag_re - sum_re * diag_im) / denom
    out_re = tl.where(out_re == out_re, out_re, 0.0)
    out_im = tl.where(out_im == out_im, out_im, 0.0)

    tl.store(work_ri_ptr + base, out_re, mask=rhs_mask, cache_modifier=".wt")
    tl.store(work_ri_ptr + base + 1, out_im, mask=rhs_mask, cache_modifier=".wt")
    # Match the CUDA __syncthreads()/publish ordering for multi-warp RHS tiles.
    tl.debug_barrier()
    tl.atomic_or(done_ptr + flag_base + row_i64, 1, sem="release")


def _prepare_spsm_rhs_work_buffer(rhs):
    # Library-main solves through a dedicated RHS work buffer. In our current
    # row-major NON_TRANS path that buffer already matches the final layout, so
    # we keep a contiguous in-place work copy rather than doing an extra transpose.
    return rhs.contiguous().clone()


def _spsm_polling_rhs_tile(n_rhs):
    n_rhs = max(1, int(n_rhs))
    return min(_SPSM_POLLING_RHS_TILE, 1 << (n_rhs - 1).bit_length())


def _spsm_polling_num_warps(block_rhs):
    if block_rhs <= 32:
        return 1
    if block_rhs <= 64:
        return 2
    if block_rhs <= 256:
        return 4
    return 8


def _coo_to_csr_sorted_unique(data, row64, col64, n_rows, n_cols):
    if data.numel() == 0:
        return (
            data,
            torch.empty(0, dtype=torch.int64, device=data.device),
            torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device),
        )
    key = row64 * max(1, n_cols) + col64
    if key.numel() == 1 or bool(torch.all(key[1:] > key[:-1]).item()):
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
        nnz_per_row = torch.bincount(row64.to(torch.int64), minlength=n_rows)
        indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
        return data.contiguous(), col64.to(torch.int64).contiguous(), indptr
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    key_s = key[order]
    data_s = data[order]
    unique_key, inverse = torch.unique_consecutive(key_s, return_inverse=True)
    out_nnz = unique_key.numel()
    data_u = torch.zeros(out_nnz, dtype=data.dtype, device=data.device)
    data_u.scatter_add_(0, inverse, data_s)
    row_u = torch.div(unique_key, max(1, n_cols), rounding_mode="floor")
    col_u = unique_key - row_u * max(1, n_cols)
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
    if out_nnz > 0:
        nnz_per_row = torch.bincount(row_u, minlength=n_rows)
        indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    return data_u, col_u.to(torch.int64), indptr


def _prepare_spsm_csr_system(data, indices32, indptr32, n_rows, lower, unit_diagonal):
    plan = {
        "kernel_dep_data": data,
        "kernel_dep_indices32": indices32,
        "kernel_dep_ptr": indptr32,
        "lower_eff": bool(lower),
        "unit_diagonal": bool(unit_diagonal),
    }
    if torch.is_complex(data):
        plan["kernel_dep_data_ri"] = _complex_interleaved_view(data)
    return plan


def _prepare_spsm_coo_system(data, row32, col32, n_rows, n_cols, lower, unit_diagonal):
    data_u, col_u64, row_ptr = _coo_to_csr_sorted_unique(
        data, row32.to(torch.int64), col32.to(torch.int64), n_rows, n_cols
    )
    plan = {
        "kernel_dep_data": data_u,
        "kernel_dep_indices32": col_u64.to(torch.int32),
        "kernel_dep_ptr": _prepare_spsm_kernel_row_ptr(row_ptr),
        "lower_eff": bool(lower),
        "unit_diagonal": bool(unit_diagonal),
    }
    if torch.is_complex(data):
        plan["kernel_dep_data_ri"] = _complex_interleaved_view(data_u)
    return plan


def _resolve_spsm_csr_runtime(
    data, indices, indptr, B, shape, lower, unit_diagonal, opA, opB, major
):
    data, indices64, indptr64, B, n_rows, n_cols = _prepare_spsm_csr_inputs(
        data, indices, indptr, B, shape, opA, opB, major
    )
    cache_key = _spsm_preprocess_cache_key(
        "csr",
        (data, indices64, indptr64),
        shape,
        lower,
        unit_diagonal,
    )
    solve_plan = _spsm_cache_get(_SPSM_PREPROCESS_CACHE, cache_key)
    if solve_plan is None:
        solve_plan = _prepare_spsm_csr_system(
            data, indices64, indptr64, n_rows, lower, unit_diagonal
        )
        _spsm_cache_put(
            _SPSM_PREPROCESS_CACHE, cache_key, solve_plan, _SPSM_PREPROCESS_CACHE_SIZE
        )
    return data, B, n_rows, n_cols, solve_plan


def _resolve_spsm_coo_runtime(
    data, row, col, B, shape, lower, unit_diagonal, opA, opB, major
):
    data, row64, col64, B, n_rows, n_cols = _prepare_spsm_coo_inputs(
        data, row, col, B, shape, opA, opB, major
    )
    cache_key = _spsm_preprocess_cache_key(
        "coo",
        (data, row64, col64),
        shape,
        lower,
        unit_diagonal,
    )
    solve_plan = _spsm_cache_get(_SPSM_PREPROCESS_CACHE, cache_key)
    if solve_plan is None:
        solve_plan = _prepare_spsm_coo_system(
            data, row64, col64, n_rows, n_cols, lower, unit_diagonal
        )
        _spsm_cache_put(
            _SPSM_PREPROCESS_CACHE, cache_key, solve_plan, _SPSM_PREPROCESS_CACHE_SIZE
        )
    return data, B, n_rows, n_cols, solve_plan


def _run_spsm_csr_core(
    data,
    indices32,
    indptr,
    rhs,
    n_rows,
    *,
    data_ri=None,
    alpha=1.0,
    lower=True,
    block_rhs=None,
    unit_diagonal=False,
):
    if rhs.ndim != 2:
        raise ValueError("rhs must be 2D")
    rhs = rhs.contiguous()
    if rhs.shape[0] != n_rows:
        raise ValueError("rhs first dim must equal n_rows")
    n_rhs = int(rhs.shape[1])
    rhs_work = _prepare_spsm_rhs_work_buffer(rhs)
    if n_rows == 0 or n_rhs == 0:
        return rhs_work

    block_rhs_use = (
        int(block_rhs) if block_rhs is not None else _spsm_polling_rhs_tile(n_rhs)
    )
    if (
        block_rhs_use <= 0
        or block_rhs_use > _SPSM_POLLING_RHS_TILE
        or (block_rhs_use & (block_rhs_use - 1)) != 0
    ):
        raise ValueError("block_rhs must be a power of two in [1, 1024]")
    rhs_tiles = triton.cdiv(n_rhs, block_rhs_use)
    num_warps_use = _spsm_polling_num_warps(block_rhs_use)
    done = torch.zeros((rhs_tiles, n_rows), dtype=torch.int32, device=rhs.device)
    is_complex = torch.is_complex(data)
    use_fp64 = data.dtype in (torch.float64, torch.complex128)
    grid = (n_rows, rhs_tiles)
    if is_complex:
        data_ri = data_ri if data_ri is not None else _complex_interleaved_view(data)
        rhs_work_ri = _complex_interleaved_view(rhs_work)
        diag_ri = torch.empty((n_rows * 2,), dtype=data_ri.dtype, device=rhs.device)
        _spsm_extract_diag_kernel_complex[(n_rows,)](
            data_ri,
            indices32,
            indptr,
            diag_ri,
            n_rows=n_rows,
            UNIT_DIAG=bool(unit_diagonal),
            USE_FP64_ACC=use_fp64,
        )
        alpha_re = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
        alpha_im = float(alpha.imag) if isinstance(alpha, complex) else 0.0
        _spsm_csr_polling_kernel_complex[grid](
            data_ri,
            indices32,
            indptr,
            diag_ri,
            rhs_work_ri,
            done,
            n_rows=n_rows,
            n_rhs=n_rhs,
            stride_work0=rhs_work.stride(0),
            alpha_re=alpha_re,
            alpha_im=alpha_im,
            BLOCK_RHS=block_rhs_use,
            USE_FP64_ACC=use_fp64,
            LOWER=bool(lower),
            UNIT_DIAG=bool(unit_diagonal),
            num_warps=num_warps_use,
            num_stages=1,
        )
        return torch.view_as_complex(rhs_work_ri.reshape(n_rows, n_rhs, 2).contiguous())

    diag = torch.empty((n_rows,), dtype=data.dtype, device=rhs.device)
    _spsm_extract_diag_kernel_real[(n_rows,)](
        data,
        indices32,
        indptr,
        diag,
        n_rows=n_rows,
        UNIT_DIAG=bool(unit_diagonal),
        USE_FP64_ACC=use_fp64,
    )
    _spsm_csr_polling_kernel_real[grid](
        data,
        indices32,
        indptr,
        diag,
        rhs_work,
        done,
        n_rows=n_rows,
        n_rhs=n_rhs,
        stride_work0=rhs_work.stride(0),
        alpha=alpha,
        BLOCK_RHS=block_rhs_use,
        USE_FP64_ACC=use_fp64,
        LOWER=bool(lower),
        UNIT_DIAG=bool(unit_diagonal),
        num_warps=num_warps_use,
        num_stages=1,
    )
    return rhs_work


def flagsparse_spsm_csr(
    data,
    indices,
    indptr,
    B,
    shape,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    out=None,
    return_time=False,
):
    data, B, n_rows, _n_cols, solve_plan = _resolve_spsm_csr_runtime(
        data, indices, indptr, B, shape, lower, unit_diagonal, opA, opB, major
    )
    alpha_value = 1.0 if _alpha_is_one(alpha) else _alpha_to_host_scalar(alpha)
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    x = _run_spsm_csr_core(
        solve_plan["kernel_dep_data"],
        solve_plan["kernel_dep_indices32"],
        solve_plan["kernel_dep_ptr"],
        B,
        n_rows,
        data_ri=solve_plan.get("kernel_dep_data_ri"),
        alpha=alpha_value,
        lower=solve_plan["lower_eff"],
        unit_diagonal=solve_plan["unit_diagonal"],
    )
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(x)
        x = out
    if return_time:
        return x, elapsed_ms
    return x


def flagsparse_spsm_coo(
    data,
    row,
    col,
    B,
    shape,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    out=None,
    return_time=False,
):
    data, B, n_rows, _n_cols, solve_plan = _resolve_spsm_coo_runtime(
        data, row, col, B, shape, lower, unit_diagonal, opA, opB, major
    )
    alpha_value = 1.0 if _alpha_is_one(alpha) else _alpha_to_host_scalar(alpha)
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    x = _run_spsm_csr_core(
        solve_plan["kernel_dep_data"],
        solve_plan["kernel_dep_indices32"],
        solve_plan["kernel_dep_ptr"],
        B,
        n_rows,
        data_ri=solve_plan.get("kernel_dep_data_ri"),
        alpha=alpha_value,
        lower=solve_plan["lower_eff"],
        unit_diagonal=solve_plan["unit_diagonal"],
    )
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(x)
        x = out
    if return_time:
        return x, elapsed_ms
    return x


def _analyze_spsm_csr(
    data,
    indices,
    indptr,
    B,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    clear_cache=False,
    return_time=False,
):
    if clear_cache:
        _clear_spsm_preprocess_cache()
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    _resolve_spsm_csr_runtime(
        data, indices, indptr, B, shape, lower, unit_diagonal, opA, opB, major
    )
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms
    return None


def _analyze_spsm_coo(
    data,
    row,
    col,
    B,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    clear_cache=False,
    return_time=False,
):
    if clear_cache:
        _clear_spsm_preprocess_cache()
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    _resolve_spsm_coo_runtime(
        data, row, col, B, shape, lower, unit_diagonal, opA, opB, major
    )
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms
    return None


def benchmark_spsm_case(
    fmt="csr",
    n_rows=1024,
    n_rhs=1024,
    nnz=8192,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    warmup=10,
    iters=50,
):
    """Pure FlagSparse SpSM benchmark entry for one configuration."""
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_rows, nnz, value_dtype, index_dtype, device
    )
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr.to(torch.int64)[1:] - indptr.to(torch.int64)[:-1],
    )
    col_ids = indices.to(torch.int64)
    tri_mask = (col_ids <= row_ids) if lower else (col_ids >= row_ids)
    data = data[tri_mask]
    row_ids = row_ids[tri_mask]
    col_ids = col_ids[tri_mask]
    data, col_ids, indptr = _coo_to_csr_sorted_unique(
        data, row_ids, col_ids, n_rows, n_rows
    )
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    diag_mask = col_ids == row_ids
    diag_present = torch.zeros(n_rows, dtype=torch.bool, device=device)
    if diag_mask.numel() > 0 and bool(torch.any(diag_mask).item()):
        diag_present[row_ids[diag_mask]] = True
    missing_diag = (
        torch.nonzero(~diag_present, as_tuple=False).reshape(-1).to(torch.int64)
    )
    if missing_diag.numel() > 0:
        diag_values = torch.ones(missing_diag.numel(), dtype=value_dtype, device=device)
        data = torch.cat([data, diag_values], dim=0)
        row_ids = torch.cat([row_ids, missing_diag], dim=0)
        col_ids = torch.cat([col_ids, missing_diag], dim=0)
        data, col_ids, indptr = _coo_to_csr_sorted_unique(
            data, row_ids, col_ids, n_rows, n_rows
        )
    B = torch.randn((n_rows, n_rhs), dtype=value_dtype, device=device).contiguous()
    shape = (n_rows, n_rows)

    if str(fmt).lower() == "coo":
        row_ids = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        analysis_ms = _analyze_spsm_coo(
            data,
            row_ids.to(index_dtype),
            col_ids.to(index_dtype),
            B,
            shape,
            lower=lower,
            unit_diagonal=unit_diagonal,
            clear_cache=True,
            return_time=True,
        )
        solve_call = lambda: flagsparse_spsm_coo(
            data,
            row_ids.to(index_dtype),
            col_ids.to(index_dtype),
            B,
            shape,
            alpha=alpha,
            lower=lower,
            unit_diagonal=unit_diagonal,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        )
    else:
        analysis_ms = _analyze_spsm_csr(
            data,
            col_ids.to(index_dtype),
            indptr.to(index_dtype),
            B,
            shape,
            lower=lower,
            unit_diagonal=unit_diagonal,
            clear_cache=True,
            return_time=True,
        )
        solve_call = lambda: flagsparse_spsm_csr(
            data,
            col_ids.to(index_dtype),
            indptr.to(index_dtype),
            B,
            shape,
            alpha=alpha,
            lower=lower,
            unit_diagonal=unit_diagonal,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        )
    C_fs, solve_ms = _benchmark_cuda_op(solve_call, warmup=warmup, iters=iters)
    return {
        "parameters": {
            "format": str(fmt).lower(),
            "n_rows": n_rows,
            "n_rhs": n_rhs,
            "nnz": int(data.numel()),
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "opA": "NON_TRANS",
            "opB": "NON_TRANS",
            "major": "row",
        },
        "performance": {
            "triton_analysis_ms": analysis_ms,
            "triton_solve_ms": solve_ms,
            "triton_time_total_ms": (
                analysis_ms + solve_ms
                if analysis_ms is not None and solve_ms is not None
                else None
            ),
        },
        "samples": {"flagsparse": C_fs},
    }
