"""CANN delivery baselines for GEMVI and CSR GEAM on fixed operands."""

from __future__ import annotations

import ctypes
import importlib
import math
import os
import sys
import time
from pathlib import Path

import torch

import ascend_sparse_baseline as sparse_baseline
from flagsparse.sparse_operations._common import _benchmark_cuda_op


PERF_FIELDS = [
    *sparse_baseline.PERF_FIELDS,
    "baseline_required",
    "cann_timing_scope",
    "cann_input_format",
    "cann_output_format",
    "cann_allocation_scope",
    "cann_formulation",
]


def mark_row(row, kind):
    row.update(
        baseline_kind="cann",
        baseline_required="cann",
        cann_status="UNAVAILABLE",
        cann_reason="CANN baseline has not completed correctness validation",
        vendor_name={
            "gemvi": "CANN SparseTensorDenseMatMul",
            "geam": "CANN SparseAdd",
            "axpby": "CANN SparseTensorDenseAdd",
        }[kind],
    )


def measure(row, build, check, dtype, flagsparse_ms, warmup, iters):
    plan = None
    for field in ("cann_ms", "cann_flagsparse_ms", "speedup_vs_cann", "cann_max_error"):
        row.pop(field, None)
    try:
        plan = build()
        row.update(plan.metadata)
        row["cann_prepare_ms"] = plan.prepare_ms
        row["cann_arithmetic_dtype"] = str(dtype).replace("torch.", "")
        tolerance = 1e-4 if dtype in (torch.float32, torch.complex64) else 1e-10
        before = float(check(plan.run()))
        if not math.isfinite(before) or before > tolerance:
            raise RuntimeError(f"CANN pre-timing correctness check failed: {before}")
        _, elapsed_ms = _benchmark_cuda_op(plan.run, warmup, iters)
        after = float(check(plan.run()))
        if not math.isfinite(after) or after > tolerance:
            raise RuntimeError(f"CANN post-timing correctness check failed: {after}")
        if (
            not math.isfinite(elapsed_ms)
            or elapsed_ms <= 0
            or not flagsparse_ms
            or not math.isfinite(flagsparse_ms)
            or flagsparse_ms <= 0
        ):
            raise RuntimeError("CANN comparison requires positive finite timings")
        row.update(
            cann_status="PASS",
            cann_reason="",
            cann_ms=elapsed_ms,
            cann_flagsparse_ms=flagsparse_ms,
            speedup_vs_cann=elapsed_ms / flagsparse_ms,
            cann_max_error=max(before, after),
        )
    except Exception as exc:
        row.update(cann_status="FAIL", cann_reason=f"{type(exc).__name__}: {exc}")
    finally:
        if plan is not None:
            try:
                plan.close()
            except Exception as exc:
                for field in ("cann_ms", "cann_flagsparse_ms", "speedup_vs_cann"):
                    row.pop(field, None)
                row.update(
                    cann_status="FAIL", cann_reason=f"CANN cleanup failed: {exc}"
                )


def prepare_gemvi(matrix, values, indices, output, op="non"):
    if op not in ("non", "trans"):
        raise ValueError("GEMVI CANN baseline supports non/trans only")
    torch.npu.synchronize(matrix.device)
    started = time.perf_counter()
    operand = matrix if op == "non" else matrix.t()
    index_cpu, order = indices.detach().cpu().long().sort()
    coordinates = torch.stack((torch.zeros_like(index_cpu), index_cpu), dim=1).int()
    prepared_values = values.detach().cpu()[order].contiguous().to(matrix.device)
    rhs = operand.t().contiguous()
    shape = torch.tensor([1, operand.shape[1]], dtype=torch.int64, device=matrix.device)
    if tuple(output.shape) != (operand.shape[0],) or not output.is_contiguous():
        raise ValueError(
            "GEMVI CANN baseline requires a contiguous output of the correct size"
        )
    plan = sparse_baseline.SparsePlan(
        coordinates.to(matrix.device),
        prepared_values,
        shape,
        rhs,
        output.reshape(1, -1),
        False,
    )
    torch.npu.synchronize(matrix.device)
    plan.prepare_ms = (time.perf_counter() - started) * 1000
    plan.metadata = dict(
        cann_timing_scope="prepared_native_call",
        cann_formulation="x^T @ op(A)^T; alpha=1,beta=0",
        cann_input_format="COO sparse vector and contiguous transposed dense matrix",
        cann_output_format="dense vector",
        cann_allocation_scope="preallocated output on both sides",
    )
    return plan


class _SparseAddCall:
    def __init__(self, coordinates, values, shape, other_coordinates, other_values):
        self.descriptors = []
        self.buffers = []
        self.launched = False
        self.closed = False
        home = Path(
            os.environ.get(
                "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"
            )
        )
        try:
            self.acl = importlib.import_module("acl")
        except ImportError:
            sys.path.insert(0, str(home / "python/site-packages"))
            self.acl = importlib.import_module("acl")
        self.inputs = [
            coordinates,
            values,
            shape,
            other_coordinates,
            other_values,
            shape,
            torch.tensor(0.0, device=values.device),
        ]
        self.capacity = values.numel() + other_values.numel()
        allocation = max(1, self.capacity)
        self.outputs = [
            torch.empty((allocation, 2), dtype=torch.int64, device=values.device),
            torch.empty(allocation, dtype=values.dtype, device=values.device),
            torch.empty(2, dtype=torch.int64, device=values.device),
        ]
        codes = {
            torch.float32: 0,
            torch.float64: 11,
            torch.int64: 9,
            torch.complex64: 16,
            torch.complex128: 17,
        }
        try:
            for tensor in [*self.inputs, *self.outputs]:
                descriptor = self.acl.create_tensor_desc(
                    codes[tensor.dtype], list(tensor.shape), 2
                )
                if not descriptor:
                    raise RuntimeError("acl.create_tensor_desc failed")
                self.descriptors.append(descriptor)
                buffer = self.acl.create_data_buffer(
                    tensor.data_ptr(), tensor.numel() * tensor.element_size()
                )
                if not buffer:
                    raise RuntimeError("acl.create_data_buffer failed")
                self.buffers.append(buffer)
            library_path = home / "lib64/libacl_op_compiler.so"
            self.library = ctypes.CDLL(str(library_path))
            self.execute = self.library.aclopCompileAndExecuteV2
            pointer = ctypes.POINTER(ctypes.c_void_p)
            self.execute.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                pointer,
                pointer,
                ctypes.c_int,
                pointer,
                pointer,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_void_p,
            ]
            self.execute.restype = ctypes.c_int
            self.input_desc = (ctypes.c_void_p * 7)(*self.descriptors[:7])
            self.input_buffers = (ctypes.c_void_p * 7)(*self.buffers[:7])
            self.output_desc = (ctypes.c_void_p * 3)(*self.descriptors[7:])
            self.output_buffers = (ctypes.c_void_p * 3)(*self.buffers[7:])
        except Exception:
            self.close()
            raise

    def run(self):
        if self.closed:
            raise RuntimeError("CANN SparseAdd call is closed")
        self.launched = True
        status = self.execute(
            b"SparseAdd",
            7,
            self.input_desc,
            self.input_buffers,
            3,
            self.output_desc,
            self.output_buffers,
            None,
            0,
            0,
            None,
            torch.npu.current_stream(self.outputs[0].device).npu_stream,
        )
        if status:
            raise RuntimeError(
                f"CANN SparseAdd returned {status}: {self.acl.get_recent_err_msg()}"
            )
        torch.npu.synchronize(self.outputs[0].device)
        count = self.acl.get_tensor_desc_dim(self.descriptors[8], 0)
        if not 0 <= count <= self.capacity:
            raise RuntimeError(f"CANN SparseAdd returned invalid output length {count}")
        return self.outputs[0][:count], self.outputs[1][:count]

    def close(self):
        if self.closed:
            return
        if self.launched:
            torch.npu.synchronize(self.outputs[0].device)
        for buffer in self.buffers:
            self.acl.destroy_data_buffer(buffer)
        for descriptor in self.descriptors:
            self.acl.destroy_tensor_desc(descriptor)
        self.buffers.clear()
        self.descriptors.clear()
        self.closed = True


class GeamPlan:
    def __init__(self, first, second, shape, alpha, beta):
        torch.npu.synchronize(first[0].device)
        started = time.perf_counter()
        self.first, self.second = first, second
        self.shape = tuple(shape)
        self.alpha, self.beta = alpha, beta
        self.closed = False
        self.prepare_ms = (time.perf_counter() - started) * 1000
        self.metadata = dict(
            cann_timing_scope="csr_to_coo_native_add_to_csr_call",
            cann_input_format="CSR",
            cann_output_format="CSR",
            cann_allocation_scope="allocate native COO outputs and CSR structure per call",
            cann_formulation="SparseAdd(alpha*A,beta*B),thresh=0",
        )

    def run(self):
        if self.closed:
            raise RuntimeError("CANN GEAM plan is closed")
        device = self.first[0].device
        n_rows = self.shape[0]

        def operands(arrays, scalar):
            values, columns, pointer = arrays
            pointer = pointer.long()
            rows = torch.repeat_interleave(
                torch.arange(n_rows, device=device), pointer.diff()
            )
            coordinates = torch.stack((rows, columns.long()), dim=1)
            if scalar != 1:
                if values.is_complex():
                    planes = torch.view_as_real(values.resolve_conj())
                    scalar = complex(scalar)
                    values = torch.view_as_complex(
                        torch.stack(
                            (
                                planes[:, 0] * scalar.real - planes[:, 1] * scalar.imag,
                                planes[:, 0] * scalar.imag + planes[:, 1] * scalar.real,
                            ),
                            dim=1,
                        )
                    )
                else:
                    values = values * scalar
            return coordinates, values

        coordinates, values = operands(self.first, self.alpha)
        other_coordinates, other_values = operands(self.second, self.beta)
        shape = torch.tensor(self.shape, dtype=torch.int64, device=device)
        call = _SparseAddCall(
            coordinates, values, shape, other_coordinates, other_values
        )
        try:
            output_coordinates, output_values = call.run()
            counts = torch.bincount(
                output_coordinates[:, 0].contiguous(), minlength=n_rows
            )
            pointer = torch.zeros(n_rows + 1, dtype=self.first[2].dtype, device=device)
            pointer[1:] = counts.cumsum(0)
            columns = output_coordinates[:, 1].to(torch.int32)
            return output_values, columns, pointer
        finally:
            call.close()

    def close(self):
        self.closed = True


def prepare_geam_csr(first, second, shape, alpha=1.0, beta=1.0):
    return GeamPlan(first, second, shape, alpha, beta)
