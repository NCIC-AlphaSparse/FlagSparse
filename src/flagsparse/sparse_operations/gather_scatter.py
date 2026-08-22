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

"""Gather and scatter (Triton kernels + cuSPARSE-style baselines)."""

from . import _common as _common_mod
from ._common import *

import ctypes
import ctypes.util

HipPointer = _common_mod.HipPointer

import triton
import triton.language as tl

SUPPORTED_SCATTER_VALUE_DTYPES = SUPPORTED_VALUE_DTYPES
DEFAULT_GATHER_BLOCK_SIZE = 256
DEFAULT_GATHER_MAX_PROGRAMS = 2
DEFAULT_GATHER_NUM_WARPS = 8


def _set_hipsparse_stream(handle, stream="current"):
    set_stream = getattr(hipsparse, "hipsparseSetStream", None)
    if set_stream is None:
        return "hipSPARSE binding does not expose hipsparseSetStream"

    if stream == "current":
        stream_ptr = _torch_current_stream_ptr()
        if stream_ptr is None:
            return "could not resolve torch current CUDA/HIP stream pointer"
    else:
        stream_ptr = int(stream)

    if stream_ptr == 0:
        stream_args = [0, ctypes.c_void_p(0)]
    else:
        stream_args = [ctypes.c_void_p(stream_ptr), stream_ptr]
    if stream_ptr != 0 and HipPointer is not None:
        try:
            stream_args.append(HipPointer.fromObj(stream_ptr))
        except Exception:
            pass

    last_error = None
    for stream_arg in stream_args:
        try:
            _hip_check_result(set_stream(handle, stream_arg), "hipsparseSetStream")
            return None
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
    return f"hipsparseSetStream failed for stream {stream_ptr}: {last_error}"


def _set_hipsparse_current_stream(handle):
    return _set_hipsparse_stream(handle, "current")


def _scatter_dtype_error_message():
    supported = ", ".join(
        str(dtype).replace("torch.", "") for dtype in SUPPORTED_SCATTER_VALUE_DTYPES
    )
    return "Scatter benchmark supports value dtypes: " f"{supported}."


@triton.jit
def _gather_real_kernel(
    sparse_values_ptr,
    dense_values_ptr,
    indices_ptr,
    nnz,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    for block_start in tl.range(pid * BLOCK_SIZE, nnz, num_programs * BLOCK_SIZE):
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
    num_programs = tl.num_programs(axis=0)
    for block_start in tl.range(pid * BLOCK_SIZE, nnz, num_programs * BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < nnz
        indices = tl.load(indices_ptr + offsets, mask=mask, other=0)

        dense_offsets = indices * 2
        sparse_offsets = offsets * 2

        gathered_real = tl.load(
            dense_values_ri_ptr + dense_offsets, mask=mask, other=0.0
        )
        gathered_imag = tl.load(
            dense_values_ri_ptr + dense_offsets + 1, mask=mask, other=0.0
        )

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
    values_imag = tl.load(
        sparse_values_ri_ptr + sparse_offsets + 1, mask=mask, other=0.0
    )

    tl.store(dense_values_ri_ptr + dense_offsets, values_real, mask=mask)
    tl.store(dense_values_ri_ptr + dense_offsets + 1, values_imag, mask=mask)


def _triton_gather_impl(
    dense_vector,
    kernel_indices,
    out=None,
    block_size=DEFAULT_GATHER_BLOCK_SIZE,
):
    nnz = kernel_indices.numel()
    if nnz == 0:
        if out is None:
            return torch.empty(0, dtype=dense_vector.dtype, device=dense_vector.device)
        if out.shape != (0,):
            raise ValueError("out shape must match gather output shape")
        if out.dtype != dense_vector.dtype:
            raise TypeError("out dtype must match gather output dtype")
        return out

    grid = lambda meta: (
        min(DEFAULT_GATHER_MAX_PROGRAMS, triton.cdiv(nnz, meta["BLOCK_SIZE"])),
    )

    if not _is_complex_dtype(dense_vector.dtype):
        sparse_values = out
        if sparse_values is None:
            sparse_values = torch.empty(
                nnz, dtype=dense_vector.dtype, device=dense_vector.device
            )
        else:
            if sparse_values.shape != (nnz,):
                raise ValueError("out shape must match gather output shape")
            if sparse_values.dtype != dense_vector.dtype:
                raise TypeError("out dtype must match gather output dtype")
        _gather_real_kernel[grid](
            sparse_values,
            dense_vector,
            kernel_indices,
            nnz,
            BLOCK_SIZE=block_size,
            num_warps=DEFAULT_GATHER_NUM_WARPS,
        )
        return sparse_values

    sparse_values = out
    if sparse_values is None:
        sparse_values = torch.empty(
            nnz, dtype=dense_vector.dtype, device=dense_vector.device
        )
    else:
        if sparse_values.shape != (nnz,):
            raise ValueError("out shape must match gather output shape")
        if sparse_values.dtype != dense_vector.dtype:
            raise TypeError("out dtype must match gather output dtype")
    dense_values_ri = torch.view_as_real(dense_vector).reshape(-1)
    sparse_values_ri = torch.view_as_real(sparse_values).reshape(-1)

    _gather_complex_kernel[grid](
        sparse_values_ri,
        dense_values_ri,
        kernel_indices,
        nnz,
        BLOCK_SIZE=block_size,
        num_warps=DEFAULT_GATHER_NUM_WARPS,
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


def _validate_scatter_value_dtype(sparse_values):
    if sparse_values.dtype not in SUPPORTED_SCATTER_VALUE_DTYPES:
        raise TypeError(_scatter_dtype_error_message())


_CUSPARSE_STATUS_SUCCESS = 0
_CUSPARSE_INDEX_BASE_ZERO = 0
_CUSPARSE_INDEX_32I = 2
_CUSPARSE_INDEX_64I = 3
_CUDA_R_32F = 0
_CUDA_R_64F = 1
_CUDA_R_16F = 2
_CUDA_C_32F = 4
_CUDA_C_64F = 5
_CUSPARSE_LIB = None
_CUSPARSE_LIB_LOAD_ERROR = None


def _cusparse_native_gather_skip_reason(value_dtype):
    if value_dtype == torch.bfloat16:
        return "bfloat16 is not supported by native cusparseGather; skipped"
    return None


def _cuda_data_type_from_torch(torch_dtype):
    mapping = {
        torch.float16: _CUDA_R_16F,
        torch.float32: _CUDA_R_32F,
        torch.float64: _CUDA_R_64F,
        torch.complex64: _CUDA_C_32F,
        torch.complex128: _CUDA_C_64F,
    }
    if torch_dtype not in mapping:
        raise TypeError(f"Unsupported cuSPARSE native gather dtype: {torch_dtype}")
    return mapping[torch_dtype]


def _cusparse_index_type_from_torch(index_dtype):
    if index_dtype == torch.int32:
        return _CUSPARSE_INDEX_32I
    if index_dtype == torch.int64:
        return _CUSPARSE_INDEX_64I
    raise TypeError(f"Unsupported cuSPARSE native gather index dtype: {index_dtype}")


def _load_cusparse_library():
    global _CUSPARSE_LIB, _CUSPARSE_LIB_LOAD_ERROR
    if _CUSPARSE_LIB is not None:
        return _CUSPARSE_LIB
    if _CUSPARSE_LIB_LOAD_ERROR is not None:
        raise RuntimeError(_CUSPARSE_LIB_LOAD_ERROR)

    candidate_names = []
    found_name = ctypes.util.find_library("cusparse")
    if found_name:
        candidate_names.append(found_name)
    candidate_names.extend(["libcusparse.so", "libcusparse.so.12", "libcusparse.so.11"])

    load_error = None
    for name in candidate_names:
        try:
            lib = ctypes.CDLL(name)
            lib.cusparseCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            lib.cusparseCreate.restype = ctypes.c_int
            lib.cusparseDestroy.argtypes = [ctypes.c_void_p]
            lib.cusparseDestroy.restype = ctypes.c_int
            lib.cusparseCreateDnVec.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            lib.cusparseCreateDnVec.restype = ctypes.c_int
            lib.cusparseDestroyDnVec.argtypes = [ctypes.c_void_p]
            lib.cusparseDestroyDnVec.restype = ctypes.c_int
            lib.cusparseCreateSpVec.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.cusparseCreateSpVec.restype = ctypes.c_int
            lib.cusparseDestroySpVec.argtypes = [ctypes.c_void_p]
            lib.cusparseDestroySpVec.restype = ctypes.c_int
            lib.cusparseGather.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            lib.cusparseGather.restype = ctypes.c_int
            if hasattr(lib, "cusparseSetStream"):
                lib.cusparseSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                lib.cusparseSetStream.restype = ctypes.c_int
            _CUSPARSE_LIB = lib
            return lib
        except Exception as exc:
            load_error = exc

    _CUSPARSE_LIB_LOAD_ERROR = (
        "Failed to load libcusparse for native gather baseline"
        + (f": {load_error}" if load_error is not None else "")
    )
    raise RuntimeError(_CUSPARSE_LIB_LOAD_ERROR)


def _check_cusparse_status(status, op_name):
    if int(status) != _CUSPARSE_STATUS_SUCCESS:
        raise RuntimeError(f"{op_name} failed with cuSPARSE status {int(status)}")


def _set_cusparse_stream(lib, handle, *, strict=False):
    if not hasattr(lib, "cusparseSetStream"):
        if strict:
            raise RuntimeError(
                "Loaded cuSPARSE library does not export cusparseSetStream"
            )
        return
    try:
        stream = torch.cuda.current_stream()
        stream_ptr = getattr(stream, "cuda_stream", None)
        if stream_ptr is None:
            if strict:
                raise RuntimeError("Could not obtain the current CUDA stream pointer")
            return
        _check_cusparse_status(
            lib.cusparseSetStream(handle, ctypes.c_void_p(int(stream_ptr))),
            "cusparseSetStream",
        )
    except Exception:
        if strict:
            raise
        return


class _PreparedCusparseNativeGather:
    def __init__(self, dense_vector, indices, out=None):
        dense_vector, indices, _ = _prepare_inputs(dense_vector, indices)
        _validate_gather_value_dtype(dense_vector, "cusparse_native_gather")
        skip_reason = _cusparse_native_gather_skip_reason(dense_vector.dtype)
        if skip_reason:
            raise RuntimeError(skip_reason)

        nnz = int(indices.numel())
        if out is None:
            sparse_values = torch.empty(
                nnz, dtype=dense_vector.dtype, device=dense_vector.device
            )
        else:
            sparse_values = out
            if sparse_values.shape != (nnz,):
                raise ValueError("out shape must match gather output shape")
            if sparse_values.dtype != dense_vector.dtype:
                raise TypeError("out dtype must match gather output dtype")

        self.lib = _load_cusparse_library()
        self.handle = ctypes.c_void_p()
        self.dn_desc = ctypes.c_void_p()
        self.sp_desc = ctypes.c_void_p()
        self.dense_vector = dense_vector
        self.indices = indices
        self.sparse_values = sparse_values
        self.nnz = nnz
        self.closed = False

        if self.nnz == 0:
            return

        try:
            _check_cusparse_status(
                self.lib.cusparseCreate(ctypes.byref(self.handle)),
                "cusparseCreate",
            )
            _set_cusparse_stream(self.lib, self.handle)
            _check_cusparse_status(
                self.lib.cusparseCreateDnVec(
                    ctypes.byref(self.dn_desc),
                    ctypes.c_int64(int(self.dense_vector.numel())),
                    ctypes.c_void_p(int(self.dense_vector.data_ptr())),
                    ctypes.c_int(_cuda_data_type_from_torch(self.dense_vector.dtype)),
                ),
                "cusparseCreateDnVec",
            )
            _check_cusparse_status(
                self.lib.cusparseCreateSpVec(
                    ctypes.byref(self.sp_desc),
                    ctypes.c_int64(int(self.dense_vector.numel())),
                    ctypes.c_int64(self.nnz),
                    ctypes.c_void_p(int(self.indices.data_ptr())),
                    ctypes.c_void_p(int(self.sparse_values.data_ptr())),
                    ctypes.c_int(_cusparse_index_type_from_torch(self.indices.dtype)),
                    ctypes.c_int(_CUSPARSE_INDEX_BASE_ZERO),
                    ctypes.c_int(_cuda_data_type_from_torch(self.dense_vector.dtype)),
                ),
                "cusparseCreateSpVec",
            )
        except Exception:
            self.close()
            raise

    def run(self):
        if self.closed:
            raise RuntimeError("Prepared cuSPARSE gather plan is already closed")
        if self.nnz == 0:
            return self.sparse_values
        _check_cusparse_status(
            self.lib.cusparseGather(self.handle, self.dn_desc, self.sp_desc),
            "cusparseGather",
        )
        return self.sparse_values

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.sp_desc.value:
            try:
                self.lib.cusparseDestroySpVec(self.sp_desc)
            except Exception:
                pass
            self.sp_desc = ctypes.c_void_p()
        if self.dn_desc.value:
            try:
                self.lib.cusparseDestroyDnVec(self.dn_desc)
            except Exception:
                pass
            self.dn_desc = ctypes.c_void_p()
        if self.handle.value:
            try:
                self.lib.cusparseDestroy(self.handle)
            except Exception:
                pass
            self.handle = ctypes.c_void_p()


def _cusparse_native_gather(dense_vector, indices, out=None):
    plan = _PreparedCusparseNativeGather(dense_vector, indices, out=out)
    try:
        return plan.run()
    finally:
        plan.close()


def _hipsparse_gather_scatter_skip_reason(value_dtype, index_dtype, op_name):
    op_label = "Gather" if op_name == "gather" else "Scatter"
    if not _is_rocm_runtime():
        return f"hipSPARSE {op_label} reference requires a ROCm runtime"
    unavailable_reason = _hipsparse_unavailable_reason()
    if unavailable_reason is not None:
        return unavailable_reason
    required_symbols = (
        "hipsparseCreate",
        "hipsparseDestroy",
        "hipsparseCreateSpVec",
        "hipsparseDestroySpVec",
        "hipsparseCreateDnVec",
        "hipsparseDestroyDnVec",
        "hipsparseGather" if op_name == "gather" else "hipsparseScatter",
    )
    for symbol in required_symbols:
        if not hasattr(hipsparse, symbol):
            return f"hipSPARSE {op_label} direct API is unavailable: missing {symbol}"
    try:
        _ = _hipsparse_value_type(value_dtype)
        _ = _hipsparse_index_type(index_dtype, f"hipSPARSE {op_label} SpVec indices")
        _ = _hipsparse_lookup(
            "hipsparseIndexBase_t", ("HIPSPARSE_INDEX_BASE_ZERO",)
        )
    except Exception as exc:
        return str(exc)
    return None


def _prepare_hipsparse_gather_inputs(dense_vector, indices):
    _validate_common_inputs(dense_vector, indices)
    dense_vector = dense_vector.contiguous()
    indices = indices.contiguous()

    if indices.numel() > 0:
        if torch.any(indices < 0).item():
            raise IndexError("indices must be non-negative")
        max_index = int(indices.max().item())
        if max_index >= dense_vector.numel():
            raise IndexError(
                f"indices out of range: max index {max_index}, dense size {dense_vector.numel()}"
            )

    return dense_vector, indices


def _hipsparse_create_spvec_descriptor(
    spvec_ref,
    dense_size,
    nnz,
    indices,
    values,
    index_type,
    index_base,
    value_type,
):
    _hip_check_result(
        hipsparse.hipsparseCreateSpVec(
            spvec_ref,
            dense_size,
            nnz,
            HipPointer.fromObj(indices.data_ptr()),
            HipPointer.fromObj(values.data_ptr()),
            index_type,
            index_base,
            value_type,
        ),
        "hipsparseCreateSpVec",
    )


def _hipsparse_create_dnvec_descriptor(dnvec_ref, size, values, value_type):
    _hip_check_result(
        hipsparse.hipsparseCreateDnVec(
            dnvec_ref,
            size,
            HipPointer.fromObj(values.data_ptr()),
            value_type,
        ),
        "hipsparseCreateDnVec",
    )


def _prepare_hipsparse_gather(dense_vector, indices, out=None):
    dense_vector, indices = _prepare_hipsparse_gather_inputs(dense_vector, indices)
    skip_reason = _hipsparse_gather_scatter_skip_reason(
        dense_vector.dtype, indices.dtype, "gather"
    )
    if skip_reason is not None:
        raise RuntimeError(skip_reason)

    dense_vector = dense_vector.contiguous()
    indices = indices.contiguous()
    nnz = int(indices.numel())
    sparse_values = out
    if sparse_values is None:
        sparse_values = torch.empty(nnz, dtype=dense_vector.dtype, device=dense_vector.device)
    else:
        if not torch.is_tensor(sparse_values):
            raise TypeError("out must be a torch.Tensor")
        if not sparse_values.is_cuda or sparse_values.device != dense_vector.device:
            raise ValueError("out must be a CUDA tensor on the same device as dense_vector")
        if sparse_values.dtype != dense_vector.dtype or sparse_values.shape != (nnz,):
            raise ValueError("out shape/dtype must match gather output")
        if not sparse_values.is_contiguous():
            raise ValueError("out must be contiguous")

    value_type = _hipsparse_value_type(dense_vector.dtype)
    index_type = _hipsparse_index_type(indices.dtype, "hipSPARSE Gather SpVec indices")
    index_base = _hipsparse_lookup(
        "hipsparseIndexBase_t", ("HIPSPARSE_INDEX_BASE_ZERO",)
    )

    handle = None
    spvec = None
    dnvec = None
    try:
        handle = _hip_check_result(hipsparse.hipsparseCreate(), "hipsparseCreate")
        ptr_type = type(handle)
        spvec = ptr_type()
        dnvec = ptr_type()
        _hipsparse_create_spvec_descriptor(
            spvec.createRef(),
            int(dense_vector.numel()),
            nnz,
            indices,
            sparse_values,
            index_type,
            index_base,
            value_type,
        )
        _hipsparse_create_dnvec_descriptor(
            dnvec.createRef(),
            int(dense_vector.numel()),
            dense_vector,
            value_type,
        )
        return {
            "backend": "hipsparse",
            "handle": handle,
            "spvec": spvec,
            "dnvec": dnvec,
            "values": sparse_values,
        }
    finally:
        if handle is None and dnvec is not None:
            try:
                _hip_check_result(hipsparse.hipsparseDestroyDnVec(dnvec), "hipsparseDestroyDnVec")
            except Exception:
                pass
        if handle is None and spvec is not None:
            try:
                _hip_check_result(hipsparse.hipsparseDestroySpVec(spvec), "hipsparseDestroySpVec")
            except Exception:
                pass


def _run_hipsparse_gather_prepared(state):
    _hip_check_result(
        hipsparse.hipsparseGather(state["handle"], state["dnvec"], state["spvec"]),
        "hipsparseGather",
    )
    return state["values"]


def _destroy_hipsparse_gather_prepared(state):
    dnvec = state.get("dnvec")
    spvec = state.get("spvec")
    handle = state.get("handle")
    if dnvec is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroyDnVec(dnvec), "hipsparseDestroyDnVec")
        except Exception:
            pass
    if spvec is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroySpVec(spvec), "hipsparseDestroySpVec")
        except Exception:
            pass
    if handle is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroy(handle), "hipsparseDestroy")
        except Exception:
            pass


def hipsparse_gather(dense_vector, indices, out=None, return_metadata=False):
    state = _prepare_hipsparse_gather(dense_vector, indices, out=out)
    try:
        values = _run_hipsparse_gather_prepared(state)
        if return_metadata:
            return values, {"backend": "hipsparse"}
        return values
    finally:
        _destroy_hipsparse_gather_prepared(state)


def benchmark_hipsparse_gather(dense_vector, indices, warmup, iters, out=None):
    return _benchmark_prepared_cuda_op(
        lambda: _prepare_hipsparse_gather(dense_vector, indices, out=out),
        _run_hipsparse_gather_prepared,
        _destroy_hipsparse_gather_prepared,
        warmup=warmup,
        iters=iters,
    )


def _prepare_hipsparse_scatter(
    sparse_values,
    indices,
    dense_size=None,
    out=None,
    reset_output=True,
    dtype_policy="strict",
):
    sparse_values, indices, _, dense_size, _ = _prepare_scatter_inputs(
        sparse_values,
        indices,
        dense_size=dense_size,
        out=out,
        dtype_policy=dtype_policy,
        return_metadata=True,
    )
    _validate_scatter_value_dtype(sparse_values)
    skip_reason = _hipsparse_gather_scatter_skip_reason(
        sparse_values.dtype, indices.dtype, "scatter"
    )
    if skip_reason is not None:
        raise RuntimeError(skip_reason)

    sparse_values = sparse_values.contiguous()
    indices = indices.contiguous()
    nnz = int(indices.numel())
    if out is None:
        dense_values = torch.zeros(
            dense_size, dtype=sparse_values.dtype, device=sparse_values.device
        )
    else:
        dense_values = out
        if reset_output:
            dense_values.zero_()
    if not dense_values.is_contiguous():
        raise ValueError("out must be contiguous")

    value_type = _hipsparse_value_type(sparse_values.dtype)
    index_type = _hipsparse_index_type(indices.dtype, "hipSPARSE Scatter SpVec indices")
    index_base = _hipsparse_lookup(
        "hipsparseIndexBase_t", ("HIPSPARSE_INDEX_BASE_ZERO",)
    )

    handle = None
    spvec = None
    dnvec = None
    try:
        handle = _hip_check_result(hipsparse.hipsparseCreate(), "hipsparseCreate")
        ptr_type = type(handle)
        spvec = ptr_type()
        dnvec = ptr_type()
        _hipsparse_create_spvec_descriptor(
            spvec.createRef(),
            int(dense_size),
            nnz,
            indices,
            sparse_values,
            index_type,
            index_base,
            value_type,
        )
        _hipsparse_create_dnvec_descriptor(
            dnvec.createRef(),
            int(dense_size),
            dense_values,
            value_type,
        )
        return {
            "backend": "hipsparse",
            "handle": handle,
            "spvec": spvec,
            "dnvec": dnvec,
            "values": dense_values,
        }
    finally:
        if handle is None and dnvec is not None:
            try:
                _hip_check_result(hipsparse.hipsparseDestroyDnVec(dnvec), "hipsparseDestroyDnVec")
            except Exception:
                pass
        if handle is None and spvec is not None:
            try:
                _hip_check_result(hipsparse.hipsparseDestroySpVec(spvec), "hipsparseDestroySpVec")
            except Exception:
                pass


def _run_hipsparse_scatter_prepared(state):
    _hip_check_result(
        hipsparse.hipsparseScatter(state["handle"], state["spvec"], state["dnvec"]),
        "hipsparseScatter",
    )
    return state["values"]


def _destroy_hipsparse_scatter_prepared(state):
    dnvec = state.get("dnvec")
    spvec = state.get("spvec")
    handle = state.get("handle")
    if dnvec is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroyDnVec(dnvec), "hipsparseDestroyDnVec")
        except Exception:
            pass
    if spvec is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroySpVec(spvec), "hipsparseDestroySpVec")
        except Exception:
            pass
    if handle is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroy(handle), "hipsparseDestroy")
        except Exception:
            pass


def hipsparse_scatter(
    sparse_values,
    indices,
    dense_size=None,
    out=None,
    reset_output=True,
    dtype_policy="strict",
    return_metadata=False,
):
    state = _prepare_hipsparse_scatter(
        sparse_values,
        indices,
        dense_size=dense_size,
        out=out,
        reset_output=reset_output,
        dtype_policy=dtype_policy,
    )
    try:
        values = _run_hipsparse_scatter_prepared(state)
        if return_metadata:
            return values, {"backend": "hipsparse"}
        return values
    finally:
        _destroy_hipsparse_scatter_prepared(state)


def benchmark_hipsparse_scatter(
    sparse_values,
    indices,
    dense_size=None,
    out=None,
    reset_output=True,
    dtype_policy="strict",
    warmup=20,
    iters=200,
):
    return _benchmark_prepared_cuda_op(
        lambda: _prepare_hipsparse_scatter(
            sparse_values,
            indices,
            dense_size=dense_size,
            out=out,
            reset_output=reset_output,
            dtype_policy=dtype_policy,
        ),
        _run_hipsparse_scatter_prepared,
        _destroy_hipsparse_scatter_prepared,
        warmup=warmup,
        iters=iters,
    )


def _gather_scatter_sparse_ref_backend(value_dtype, index_dtype, op_name):
    """Pick the vendor library backing the gather/scatter baseline, per backend.

    On DCU/ROCm the reference is hipSPARSE's SpVec gather/scatter; on CUDA it stays
    the cuSPARSE SpMV-with-selector-matrix formulation used by the existing baseline.
    """
    if _is_rocm_runtime():
        reason = _hipsparse_gather_scatter_skip_reason(value_dtype, index_dtype, op_name)
        if reason is None:
            return "hipsparse", None
        return None, reason
    if cp is None or cpx_sparse is None:
        return None, "CuPy/cuSPARSE is not available"
    skip_reason = _cusparse_baseline_skip_reason(value_dtype)
    if skip_reason is not None:
        return None, skip_reason
    return "cupy_cusparse", None


def _cusparse_spmv(selector_matrix, dense_vector):
    if (
        cp is not None
        and cpx_sparse is not None
        and isinstance(selector_matrix, cpx_sparse.spmatrix)
    ):
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
            raise TypeError(
                "dense_vector must be torch.Tensor for torch sparse fallback"
            )
        out_dtype = dense_vector.dtype
        dense_for_compute = (
            dense_vector.to(torch.float32)
            if dense_vector.dtype == torch.bfloat16
            else dense_vector
        )
        out = torch.sparse.mm(selector_matrix, dense_for_compute.unsqueeze(1)).squeeze(
            1
        )
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


def _pytorch_scatter_impl(
    sparse_values, indices, dense_size, out=None, reset_output=True
):
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


def flagsparse_gather(
    a,
    indices,
    out=None,
    mode="raise",
    block_size=DEFAULT_GATHER_BLOCK_SIZE,
    return_time=False,
):
    """CuPy-style gather (take): out = a[indices]."""
    if mode != "raise":
        raise NotImplementedError("Only mode='raise' is currently supported")

    dense_vector, dense_backend = _to_torch_tensor(a, "a")
    indices_tensor, _ = _to_torch_tensor(indices, "indices")
    dense_vector, indices_tensor, kernel_indices = _prepare_inputs(
        dense_vector, indices_tensor
    )
    _validate_gather_value_dtype(dense_vector, "flagsparse_gather")
    out_tensor = None
    if out is not None:
        out_tensor, _ = _to_torch_tensor(out, "out")
        if out_tensor.shape != (int(indices_tensor.numel()),):
            raise ValueError("out shape must match gather output shape")
        if out_tensor.dtype != dense_vector.dtype:
            raise TypeError("out dtype must match gather output dtype")

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    sparse_values = _triton_gather_impl(
        dense_vector,
        kernel_indices,
        out=out_tensor,
        block_size=block_size,
    )
    torch.cuda.synchronize()
    execution_time_ms = (time.perf_counter() - start_time) * 1000.0

    if out is not None:
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
def triton_cusparse_gather(dense_vector, indices, block_size=DEFAULT_GATHER_BLOCK_SIZE):
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
    sparse_values,
    indices,
    dense_size=None,
    out=None,
    reset_output=True,
    dtype_policy="auto",
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
    """Equivalent gather baseline via the backend's vendor sparse library.

    On DCU/ROCm this is hipSPARSE's native SpVec gather; on CUDA it stays the
    cuSPARSE-backed COO SpMV with a selector matrix.
    """
    dense_vector, indices, _ = _prepare_inputs(dense_vector, indices)
    _validate_gather_value_dtype(dense_vector, "cusparse_spmv_gather")
    if _is_rocm_runtime() and selector_matrix is None:
        reason = _hipsparse_gather_scatter_skip_reason(
            dense_vector.dtype, indices.dtype, "gather"
        )
        if reason is None:
            sparse_values, execution_time_ms = benchmark_hipsparse_gather(
                dense_vector, indices, warmup=20, iters=200
            )
            return sparse_values, execution_time_ms, None
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


def cusparse_native_gather(dense_vector, indices, out=None):
    """Native cuSPARSE gather baseline via SpVec/DnVec descriptors."""
    dense_vector, indices, _ = _prepare_inputs(dense_vector, indices)
    try:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        sparse_values = _cusparse_native_gather(dense_vector, indices, out=out)
        torch.cuda.synchronize()
        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
    except Exception as exc:
        raise RuntimeError(
            "Native cuSPARSE gather baseline is unavailable in this PyTorch/CUDA environment"
        ) from exc

    return sparse_values, execution_time_ms


def cusparse_spmv_scatter(
    sparse_values, indices, dense_size=None, selector_matrix=None, dtype_policy="auto"
):
    """Equivalent scatter baseline via the backend's vendor sparse library.

    On DCU/ROCm this is hipSPARSE's native SpVec scatter; on CUDA it stays the
    cuSPARSE-backed COO SpMV with a selector matrix.
    """
    sparse_values, indices, _, dense_size, _ = _prepare_scatter_inputs(
        sparse_values,
        indices,
        dense_size=dense_size,
        out=None,
        dtype_policy=dtype_policy,
        return_metadata=True,
    )
    if _is_rocm_runtime() and selector_matrix is None:
        reason = _hipsparse_gather_scatter_skip_reason(
            sparse_values.dtype, indices.dtype, "scatter"
        )
        if reason is None:
            dense_values, execution_time_ms = benchmark_hipsparse_scatter(
                sparse_values,
                indices,
                dense_size=dense_size,
                reset_output=True,
                warmup=20,
                iters=200,
            )
            return dense_values, execution_time_ms, None
    skip_reason = _cusparse_baseline_skip_reason(sparse_values.dtype)
    if skip_reason:
        raise RuntimeError(skip_reason)

    if selector_matrix is None:
        selector_matrix = _make_scatter_selector_matrix(
            indices, dense_size, sparse_values.dtype
        )

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
