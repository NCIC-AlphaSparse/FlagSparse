"""Native, full-precision Ascend SpMM for prepared COO and CSR matrices."""

import ctypes
from functools import lru_cache
import importlib
import os
from pathlib import Path
import sys
import threading
import time
import warnings

import torch


def _mode():
    mode = os.environ.get("FLAGSPARSE_ASCEND_FP64_SPMM", "auto").strip().lower()
    if mode not in ("auto", "native", "scatter"):
        raise ValueError("FLAGSPARSE_ASCEND_FP64_SPMM must be auto, native or scatter")
    return mode


@lru_cache(maxsize=1)
def _runtime():
    home = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    try:
        acl = importlib.import_module("acl")
    except ImportError:
        location = str(home / "python/site-packages")
        if location not in sys.path:
            sys.path.append(location)
        acl = importlib.import_module("acl")
    compiler = home / "lib64/libacl_op_compiler.so"
    library = ctypes.CDLL(str(compiler) if compiler.exists() else "libacl_op_compiler.so")
    execute = library.aclopCompileAndExecute
    pointer = ctypes.POINTER(ctypes.c_void_p)
    execute.argtypes = [
        ctypes.c_char_p, ctypes.c_int, pointer, pointer, ctypes.c_int,
        pointer, pointer, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_void_p,
    ]
    execute.restype = ctypes.c_int
    return acl, library, execute


class _NativeFP64Spmm:
    def __init__(self, values, rows, columns, shape):
        self.lock = threading.Lock()
        self.closed = False
        self.acl = None
        self.attribute = None
        self.descriptors = []
        self.buffers = []
        self.shape = tuple(int(size) for size in shape)
        self.values = values.contiguous()
        self.coordinates = torch.stack((rows, columns), dim=1).to(torch.int32).contiguous()
        self.device_shape = torch.tensor(self.shape, dtype=torch.int64, device=values.device)
        try:
            self.acl, self.library, self.execute = _runtime()
            for tensor, code in ((self.coordinates, 3), (self.values, 11), (self.device_shape, 9)):
                descriptor, buffer = self._tensor(tensor, code)
                self.descriptors.append(descriptor)
                self.buffers.append(buffer)
            self.attribute = self.acl.op.create_attr()
            if not self.attribute:
                raise RuntimeError("acl.op.create_attr failed")
            for name in ("adjoint_a", "adjoint_b"):
                status = self.acl.op.set_attr_bool(self.attribute, name, False)
                if status:
                    raise RuntimeError(f"acl.op.set_attr_bool({name}) returned {status}")
        except Exception:
            self.close()
            raise

    def _tensor(self, tensor, code):
        descriptor = self.acl.create_tensor_desc(code, list(tensor.shape), 2)
        if not descriptor:
            raise RuntimeError("acl.create_tensor_desc failed")
        buffer = self.acl.create_data_buffer(tensor.data_ptr(), tensor.numel() * tensor.element_size())
        if not buffer:
            self.acl.destroy_tensor_desc(descriptor)
            raise RuntimeError("acl.create_data_buffer failed")
        return descriptor, buffer

    def run(self, dense, out=None):
        with self.lock, torch.npu.device(self.values.device):
            if self.closed:
                raise RuntimeError("native FP64 SpMM plan is closed")
            if dense.dtype != torch.float64:
                raise TypeError("native FP64 SpMM requires a float64 dense operand")
            if dense.ndim != 2 or dense.shape[0] != self.shape[1] or dense.device != self.values.device:
                raise ValueError("native FP64 SpMM dense dimensions/device must match the sparse operand")
            dense = dense.contiguous()
            direct_out = (
                out is not None and out.is_contiguous() and out.dtype == torch.float64
                and out.device == dense.device and tuple(out.shape) == (self.shape[0], dense.shape[1])
                and out.untyped_storage().data_ptr() not in (
                    dense.untyped_storage().data_ptr(), self.values.untyped_storage().data_ptr(),
                )
            )
            output = out if direct_out else torch.empty(
                (self.shape[0], dense.shape[1]), dtype=torch.float64, device=dense.device
            )
            descriptors = []
            buffers = []
            stream = torch.npu.current_stream(dense.device)
            launched = False
            try:
                for tensor in (dense, output):
                    descriptor, buffer = self._tensor(tensor, 11)
                    descriptors.append(descriptor)
                    buffers.append(buffer)
                inputs = (ctypes.c_void_p * 4)(*self.descriptors, descriptors[0])
                input_buffers = (ctypes.c_void_p * 4)(*self.buffers, buffers[0])
                outputs = (ctypes.c_void_p * 1)(descriptors[1])
                output_buffers = (ctypes.c_void_p * 1)(buffers[1])
                launched = True
                status = self.execute(
                    b"SparseTensorDenseMatMul", 4, inputs, input_buffers,
                    1, outputs, output_buffers, self.attribute, 0, 0, None, stream.npu_stream,
                )
                if status:
                    raise RuntimeError(
                        f"SparseTensorDenseMatMul returned {status}: {self.acl.get_recent_err_msg()}"
                    )
            finally:
                try:
                    if launched:
                        stream.synchronize()
                finally:
                    for buffer in buffers:
                        self.acl.destroy_data_buffer(buffer)
                    for descriptor in descriptors:
                        self.acl.destroy_tensor_desc(descriptor)
            if out is not None and output is not out:
                out.copy_(output)
                return out
            return output

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.acl is not None:
                for buffer in self.buffers:
                    self.acl.destroy_data_buffer(buffer)
                for descriptor in self.descriptors:
                    self.acl.destroy_tensor_desc(descriptor)
                if self.attribute is not None:
                    self.acl.op.destroy_attr(self.attribute)
            self.buffers.clear()
            self.descriptors.clear()
            self.closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def prepare_fp64_spmm(values, rows, columns, shape):
    if values.dtype != torch.float64 or values.device.type not in ("npu", "privateuseone"):
        return None
    if (_mode() == "scatter" or not values.numel() or not all(shape)
            or max(shape) > 2**31 - 1 or values.numel() > 2**31 - 1):
        return None
    try:
        with torch.npu.device(values.device):
            return _NativeFP64Spmm(values, rows, columns, shape)
    except Exception as exc:
        if _mode() == "native":
            raise
        warnings.warn(f"Ascend native FP64 SpMM preparation failed; using scatter: {exc}", RuntimeWarning)
        return None


def run_fp64_spmm(prepared, dense, dense_layout, *, timing=False, out=None):
    """Run the prepared native plan; ``out`` receives the result in place.

    Passing ``out`` is what lets a caller keep one output buffer across calls.
    That matters beyond saving an allocation: this operator's run time depends on
    where its B and C land (222 ms at one offset against 155 ms at another for
    the same matrix), so a benchmark comparing against CANN can only hold the
    address constant on both sides by owning C itself.

    ``dense_layout="col"`` still needs a post-transform, which cannot write
    through to ``out``, so that case allocates and copies instead.
    """
    plan = prepared.ascend_fp64_plan
    if plan is None or dense.shape[1] == 0:
        return None
    # The plan is built in the prepared object's __init__, so it only ever
    # describes that object's own orientation: a trans/conj route reaches us as
    # a *separate* prepared built from the materialized transpose (COO
    # materializes before __init__, CSR rebuilds via
    # _spmm_csr_route_from_materialized). Enforce that rather than rely on it --
    # a plan for the other orientation would silently return A @ B for A^T @ B
    # on a square matrix, where the dense-shape check below cannot catch it.
    if plan.shape != (int(prepared.n_rows), int(prepared.n_cols)):
        return None
    if torch.is_grad_enabled() and (prepared.data.requires_grad or dense.requires_grad):
        return None
    started = time.perf_counter() if timing else None
    try:
        if dense_layout == "col":
            result = plan.run(dense).T.contiguous().T
            if out is not None:
                out.copy_(result)
                result = out
        else:
            result = plan.run(dense, out=out)
    except Exception as exc:
        if _mode() == "native":
            raise
        warnings.warn(f"Ascend native FP64 SpMM failed; using scatter: {exc}", RuntimeWarning)
        plan.close()
        prepared.ascend_fp64_plan = None
        return None
    return result, {
        "execution_path": "native",
        "implementation": "cann_sparse_tensor_dense_matmul_fp64",
        "process_cpu_ms": 0.0,
        "process_gpu_ms": 0.0 if timing else None,
        "compute_ms": (time.perf_counter() - started) * 1000 if timing else None,
        "dense_layout": dense_layout,
        "b_stride": tuple(dense.stride()),
        "c_stride": tuple(result.stride()),
        "output_layout": "col" if dense_layout == "col" else "row",
    }


def run_fp64_spmm_once(values, rows, columns, dense, shape, *, sort_rows=True, out=None,
                       dense_layout="row"):
    if values.dtype != torch.float64 or values.device.type not in ("npu", "privateuseone"):
        return None
    if torch.is_grad_enabled() and (values.requires_grad or dense.requires_grad):
        return None
    if _mode() == "scatter" or not values.numel() or not all(shape) or dense.shape[1] == 0:
        return None
    plan = None
    try:
        if sort_rows and rows.numel() > 1 and not bool((rows[1:] >= rows[:-1]).all().item()):
            order = torch.argsort(rows, stable=True)
            rows = rows.index_select(0, order)
            columns = columns.index_select(0, order)
            values = values.contiguous().view(torch.int32).reshape(-1, 2).index_select(0, order)
            values = values.reshape(-1).view(torch.float64)
        plan = prepare_fp64_spmm(values, rows, columns, shape)
        if plan is None:
            return None
        result = plan.run(dense, out=out)
        if out is None and dense_layout == "col":
            result = result.T.contiguous().T
        return result
    except Exception as exc:
        if _mode() == "native":
            raise
        warnings.warn(f"Ascend native FP64 SpMM failed; using scatter: {exc}", RuntimeWarning)
        return None
    finally:
        if plan is not None:
            plan.close()
