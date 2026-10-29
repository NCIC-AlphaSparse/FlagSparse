"""CANN's native COO SparseTensorDenseMatMul benchmark adapter."""

from __future__ import annotations

import ctypes
import importlib
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


NAME = "CANN SparseTensorDenseMatMul"
PERF_FIELDS = [
    "baseline_kind", "vendor_name", "cann_status", "cann_reason", "cann_ms",
    "cann_flagsparse_ms", "speedup_vs_cann", "cann_max_error", "cann_prepare_ms",
    "cann_arithmetic_dtype", "cann_i8_promoted_to_i32",
]


def skip_reason():
    namespace = getattr(torch, "npu", None)
    if namespace is None or not namespace.is_available():
        return "CANN sparse baseline requires an available Ascend NPU"
    return None


def mark_row(row):
    row.update(baseline_kind="cann", vendor_name=NAME, cann_status="UNAVAILABLE",
               cann_reason="CANN baseline has not completed correctness validation")


def record_result(row, plan, elapsed_ms, error, flagsparse_ms):
    row.update(cann_status="PASS", cann_reason="", cann_ms=elapsed_ms,
               cann_flagsparse_ms=flagsparse_ms, cann_max_error=error,
               cann_prepare_ms=plan.prepare_ms, cann_arithmetic_dtype=str(plan.dtype).replace("torch.", ""),
               cann_i8_promoted_to_i32=plan.i8_promoted_to_i32,
               speedup_vs_cann=elapsed_ms / flagsparse_ms if flagsparse_ms and flagsparse_ms > 0 else None)


class SparsePlan:
    def __init__(self, coordinates, values, shape, dense, output, transpose):
        self.descriptors = []
        self.buffers = []
        self.attribute = None
        self.acl = None
        self.closed = False
        self.launched = False
        self.output = output
        self.dtype = output.dtype
        self.inputs = [coordinates, values, shape, dense]
        home = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
        try:
            self.acl = importlib.import_module("acl")
        except ImportError:
            sys.path.insert(0, str(home / "python/site-packages"))
            self.acl = importlib.import_module("acl")
        try:
            codes = {torch.float32: 0, torch.float16: 1, torch.int32: 3,
                     torch.int64: 9, torch.float64: 11, torch.complex64: 16, torch.complex128: 17}
            for tensor in [*self.inputs, self.output]:
                descriptor = self.acl.create_tensor_desc(codes[tensor.dtype], list(tensor.shape), 2)
                if not descriptor:
                    raise RuntimeError("acl.create_tensor_desc failed")
                self.descriptors.append(descriptor)
                buffer = self.acl.create_data_buffer(tensor.data_ptr(), tensor.numel() * tensor.element_size())
                if not buffer:
                    raise RuntimeError("acl.create_data_buffer failed")
                self.buffers.append(buffer)
            self.attribute = self.acl.op.create_attr()
            for key, value in (("adjoint_a", transpose), ("adjoint_b", False)):
                status = self.acl.op.set_attr_bool(self.attribute, key, value)
                if status:
                    raise RuntimeError(f"acl.op.set_attr_bool({key}) returned {status}")
            compiler = home / "lib64/libacl_op_compiler.so"
            self.library = ctypes.CDLL(str(compiler) if compiler.exists() else "libacl_op_compiler.so")
            self.execute = self.library.aclopCompileAndExecute
            pointer = ctypes.POINTER(ctypes.c_void_p)
            self.execute.argtypes = [ctypes.c_char_p, ctypes.c_int, pointer, pointer, ctypes.c_int,
                                    pointer, pointer, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_char_p, ctypes.c_void_p]
            self.execute.restype = ctypes.c_int
            self.input_desc = (ctypes.c_void_p * 4)(*self.descriptors[:-1])
            self.input_buffers = (ctypes.c_void_p * 4)(*self.buffers[:-1])
            self.output_desc = (ctypes.c_void_p * 1)(self.descriptors[-1])
            self.output_buffers = (ctypes.c_void_p * 1)(self.buffers[-1])
        except Exception:
            self.close()
            raise

    def run(self):
        if self.closed:
            raise RuntimeError("CANN sparse baseline plan is closed")
        status = self.execute(b"SparseTensorDenseMatMul", 4, self.input_desc, self.input_buffers,
                              1, self.output_desc, self.output_buffers, self.attribute, 0, 0, None,
                              torch.npu.current_stream(self.output.device).npu_stream)
        if status:
            raise RuntimeError(f"{NAME} returned {status}: {self.acl.get_recent_err_msg()}")
        self.launched = True
        return self.output

    def close(self):
        if self.closed:
            return
        if self.launched:
            torch.npu.synchronize(self.output.device)
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


def _coordinates(fmt, arrays, shape):
    values, indices, offsets = arrays
    nnz = values.numel()
    if fmt == "coo":
        rows = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        columns = offsets.detach().cpu().numpy().astype(np.int64, copy=False)
    elif fmt in ("csr", "csc"):
        indices = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        offsets = offsets.detach().cpu().numpy().astype(np.int64, copy=False)
        count = shape[0 if fmt == "csr" else 1]
        if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != nnz or np.any(np.diff(offsets) < 0):
            raise ValueError("Invalid CSR/CSC offsets")
        major = np.repeat(np.arange(count, dtype=np.int64), np.diff(offsets))
        rows, columns = (major, indices) if fmt == "csr" else (indices, major)
    else:
        raise ValueError(f"Unsupported CANN baseline source format: {fmt}")
    if rows.shape != (nnz,) or columns.shape != (nnz,):
        raise ValueError("COO coordinates do not match values")
    if max(shape) >= 2**31 or np.any(rows < 0) or np.any(rows >= shape[0]) or np.any(columns < 0) or np.any(columns >= shape[1]):
        raise ValueError("Invalid or overflowing int32 COO coordinates")
    return np.stack((rows, columns), axis=1).astype(np.int32)


def prepare_spmm(fmt, arrays, shape, dense, output, op_a="non", op_b="non", compute_dtype=None):
    reason = skip_reason()
    if reason:
        raise RuntimeError(reason)
    if op_a not in ("non", "trans", "conj") or op_b not in ("non", "trans", "conj"):
        raise ValueError("CANN baseline supports non/trans/conj operations")
    supported = (torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128, torch.int32)
    if output.dtype not in supported:
        raise ValueError(f"CANN SparseTensorDenseMatMul does not support {output.dtype}")
    if not output.is_contiguous() or dense.device != output.device or arrays[0].device != output.device:
        raise ValueError("CANN baseline requires contiguous output and same-device operands")
    torch.npu.synchronize(output.device)
    started = time.perf_counter_ns()
    coordinates = _coordinates(fmt, arrays, shape)
    values = arrays[0].detach().cpu().to(output.dtype)
    order = np.argsort(coordinates[:, 0 if op_a == "non" else 1], kind="stable")
    coordinates = coordinates[order]
    values = values[torch.from_numpy(order)]
    rhs = dense.detach().cpu().to(output.dtype)
    if op_a == "trans" and values.is_complex():
        values = values.conj().resolve_conj()
    if op_b != "non":
        rhs = rhs.T
        if op_b == "conj":
            rhs = rhs.conj().resolve_conj()
    expected_rows = shape[1 if op_a == "non" else 0]
    expected_out = shape[0 if op_a == "non" else 1]
    if rhs.ndim != 2 or rhs.shape[0] != expected_rows or tuple(output.shape) != (expected_out, rhs.shape[1]):
        raise ValueError("CANN sparse/dense/output dimensions do not match")
    plan = SparsePlan(torch.from_numpy(coordinates).to(output.device), values.contiguous().to(output.device),
                      torch.tensor(shape, device=output.device, dtype=torch.int64),
                      rhs.contiguous().to(output.device), output, op_a != "non")
    torch.npu.synchronize(output.device)
    plan.prepare_ms = (time.perf_counter_ns() - started) / 1e6
    plan.i8_promoted_to_i32 = arrays[0].dtype == torch.int8
    return plan


def prepare_spmv(fmt, arrays, shape, vector, output, op="non", compute_dtype=None):
    return prepare_spmm(fmt, arrays, shape, vector.reshape(-1, 1), output.reshape(-1, 1), op)
