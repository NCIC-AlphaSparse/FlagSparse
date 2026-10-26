# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BI-V packed SDDMM operands must stay live across prepared calls."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import _common


@pytest.mark.sddmm_csr
@pytest.mark.parametrize("indptr_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64], ids=["f32", "c32"])
def test_sddmm_iluvatar_transposed_live_inputs(indptr_dtype, dtype):
    if not _common._is_iluvatar_runtime():
        pytest.skip("BI-V layout specialization")
    # Partial tiles, an empty row, and enough reuse to exercise packing.
    m, n, k = 17, 19, 32
    gen = torch.Generator().manual_seed(20261021)
    rows = torch.arange(m - 1).repeat_interleave(9)
    cols = (torch.arange(rows.numel()) % n).to(torch.int32)
    ptr = torch.cat((torch.arange(m) * 9, torch.tensor([(m - 1) * 9]))).to(indptr_dtype)
    a_cpu = torch.randn(k, m, generator=gen, dtype=dtype)
    b_cpu = torch.randn(k, n, generator=gen, dtype=dtype)
    data_cpu = torch.randn(rows.numel(), generator=gen, dtype=dtype)
    a, b, data = (t.cuda() for t in (a_cpu, b_cpu, data_cpu))
    prepared = fs.prepare_sddmm_csr(cols.cuda(), ptr.cuda(), (m, n), k_hint=k)
    out = torch.empty(rows.numel(), device="cuda", dtype=dtype)
    for step in range(2):
        if step:
            a_cpu.add_(0.25)
            b_cpu.mul_(0.5)
            a.copy_(a_cpu)
            b.copy_(b_cpu)
        ref = 1.25 * (a_cpu.T @ b_cpu)[rows, cols.long()] + 0.5 * data_cpu
        got = fs.flagsparse_sddmm_csr(
            data=data, x=a, y=b, prepared=prepared, out=out,
            alpha=1.25, beta=0.5, op_a="trans", op_b="non", dense_layout="row",
        )
        assert got is out
        torch.testing.assert_close(got.cpu(), ref, rtol=1e-4, atol=1e-4)
