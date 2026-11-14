"""Prepared native CANN AXPBY baseline with independent mutable output."""

from __future__ import annotations

import time

import torch

import ascend_gemvi_geam_baseline as delivery_baseline
from ascend_sparse_baseline import SparsePlan


PERF_FIELDS = delivery_baseline.PERF_FIELDS


def expected_after(initial, values, indices, alpha, beta, calls):
    dtype = torch.complex128 if initial.is_complex() else torch.float64
    expected = initial.detach().cpu().to(dtype) * beta**calls
    factor = alpha * (calls if beta == 1 else (1 - beta**calls) / (1 - beta))
    expected.index_add_(
        0, indices.detach().cpu().long(), values.detach().cpu().to(dtype) * factor
    )
    return expected


class AxpbyPlan(SparsePlan):
    def __init__(self, values, indices, target, alpha, beta):
        torch.npu.synchronize(target.device)
        started = time.perf_counter()
        index_cpu, order = indices.detach().cpu().long().sort()
        coordinates = torch.stack((torch.zeros_like(index_cpu), index_cpu), dim=1)
        self.values = values.detach().cpu()[order].contiguous().to(target.device)
        self.target = target
        self.alpha, self.beta = alpha, beta
        self.calls = 0
        scaled = torch.empty_like(self.values)
        dense = torch.empty_like(target).reshape(1, -1)
        output = torch.empty_like(dense)
        shape = torch.tensor([1, target.numel()], dtype=torch.int64, device=target.device)
        super().__init__(
            coordinates.to(target.device), scaled, shape, dense, output, False,
            operator="SparseTensorDenseAdd",
        )
        try:
            torch.npu.synchronize(target.device)
        except Exception:
            self.close()
            raise
        self.prepare_ms = (time.perf_counter() - started) * 1000
        self.metadata = dict(
            cann_timing_scope="prepared_scale_native_add_copy",
            cann_formulation="y = SparseTensorDenseAdd(alpha*x, beta*y)",
            cann_input_format="sorted 1xN COO sparse vector and dense scratch",
            cann_output_format="independent dense output copied back to y",
            cann_allocation_scope="preallocated output and scratch on both sides",
        )

    def run(self):
        if self.closed:
            raise RuntimeError("CANN AXPBY baseline plan is closed")
        torch.mul(self.values, self.alpha, out=self.inputs[1])
        torch.mul(self.target.reshape(1, -1), self.beta, out=self.inputs[3])
        super().run()
        self.target.copy_(self.output.reshape(-1))
        self.calls += 1
        return self.target


def measure(row, values, indices, initial, alpha, beta, flagsparse_ms, warmup, iters):
    delivery_baseline.mark_row(row, "axpby")
    plan = None

    def build():
        nonlocal plan
        plan = AxpbyPlan(values, indices, initial.clone(), alpha, beta)
        return plan

    def check(result):
        expected = expected_after(initial, values, indices, alpha, beta, plan.calls)
        actual = result.detach().cpu().to(expected.dtype)
        return float((actual - expected).abs().max() / max(1.0, float(expected.abs().max())))

    delivery_baseline.measure(row, build, check, initial.dtype, flagsparse_ms, warmup, iters)
