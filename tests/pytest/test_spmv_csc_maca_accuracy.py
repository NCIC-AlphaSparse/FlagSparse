# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA CSC gather accuracy, live values and mutation-aware topology reuse."""
import pytest
import torch

from flagsparse.sparse_operations import spmv_csc as csc
from flagsparse.sparse_operations._common import _is_maca_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device

pytestmark = [pytest.mark.spmv_csc, pytest.mark.skipif(
    not (_is_maca_runtime() and accelerator_available()), reason="MACA required")]


@pytest.mark.parametrize("dtype,op", [(torch.float32, "trans"), (torch.complex64, "conj"),
                                      (torch.int8, "non"), (torch.int8, "trans")])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (73, 41), (1031, 3)])
def test_live_values_and_strided_outputs(dtype, op, index_dtype, shape):
    gen = torch.Generator().manual_seed(631)
    m, n = shape
    a = torch.randint(-4, 5, shape, generator=gen).to(dtype)
    if dtype.is_complex:
        a += 1j * torch.randn(shape, generator=gen)
    if m:
        a[0] = 0
    sparse = a.to_sparse_csc()
    dev = accelerator_device()
    values = sparse.values().to(dev)
    indices = sparse.row_indices().to(device=dev, dtype=index_dtype)
    ptr = sparse.ccol_indices().to(device=dev, dtype=index_dtype)
    input_length, output_length = (n, m) if op == "non" else (m, n)
    x_cpu = torch.randint(-4, 5, (input_length,), generator=gen).to(dtype)
    if dtype.is_complex:
        x_cpu += 1j * torch.randn(input_length, generator=gen)
    x_storage = torch.zeros(2 * input_length, dtype=dtype, device=dev)
    x_storage[::2] = x_cpu.to(dev)
    x = x_storage[::2]
    output_dtype = torch.int32 if dtype == torch.int8 else dtype
    out = torch.empty(2 * output_length, dtype=output_dtype, device=dev)[::2]
    prepared = None if dtype == torch.int8 else csc.prepare_spmv_csc(values, indices, ptr, shape, op=op)
    for scale in (1, 2):
        if scale == 2:
            values.mul_(2)
        args = dict(x=x, out=out)
        args.update(dict(data=values, indices=indices, indptr=ptr, shape=shape, op=op)
                    if prepared is None else dict(prepared=prepared))
        result = csc.flagsparse_spmv_csc(**args)
        assert result is out
        wide = torch.int32 if dtype == torch.int8 else torch.complex128 if dtype.is_complex else torch.float64
        matrix = (a * scale).to(wide)
        matrix = matrix if op == "non" else matrix.T.conj() if op == "conj" else matrix.T
        expected = matrix @ x_cpu.to(wide)
        torch.testing.assert_close(result.cpu().to(wide), expected, atol=3e-4, rtol=1e-4)


@pytest.mark.parametrize("inference", [False, True])
def test_int_topology_mutation_and_exact_large_sum(inference):
    dev = accelerator_device()
    with torch.inference_mode(inference):
        # Sum exceeds 2**24: float32 accumulation would lose integer units.
        values = torch.full((2001,), 127, dtype=torch.int8, device=dev)
        indices = torch.zeros(2001, dtype=torch.int32, device=dev)
        ptr = torch.tensor([0, 2001], dtype=torch.int64, device=dev)
        x = torch.tensor([127], dtype=torch.int8, device=dev)
        for row in (0, 1, 0):
            indices.fill_(row)
            result = csc.flagsparse_spmv_csc(values, indices, ptr, x, (2, 1))
            expected = torch.zeros(2, dtype=torch.int32)
            expected[row] = 2001 * 127 * 127
            torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)


def test_conjugate_view_live_values():
    dev = accelerator_device()
    values = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64, device=dev)
    indices = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    ptr = torch.tensor([0, 2], dtype=torch.int32, device=dev)
    x = torch.tensor([2 - 1j, 1 + 3j], dtype=torch.complex64, device=dev)
    prepared = csc.prepare_spmv_csc(values.conj(), indices, ptr, (2, 1), op="conj")
    for factor in (1, 2):
        if factor == 2:
            values.mul_(2)
        result = csc.flagsparse_spmv_csc(prepared=prepared, x=x.conj())
        torch.testing.assert_close(result, (values * x.conj()).sum().reshape(1))


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_int_strided_topology_and_timing(op):
    dev = accelerator_device()
    values = torch.tensor([2, 0, -3, 0, 4, 0], dtype=torch.int8, device=dev)[::2]
    indices = torch.tensor([0, 9, 1, 9, 0, 9], dtype=torch.int64, device=dev)[::2]
    ptr = torch.tensor([0, 9, 2, 9, 3, 9], dtype=torch.int64, device=dev)[::2]
    x = torch.tensor([5, -2], dtype=torch.int8, device=dev)
    result, elapsed, meta = csc.flagsparse_spmv_csc(
        values, indices, ptr, x, (2, 2), op=op, return_time=True, return_meta=True)
    dense = torch.tensor([[2, 4], [-3, 0]], dtype=torch.int32)
    expected = (dense if op == "non" else dense.T) @ x.cpu().int()
    torch.testing.assert_close(result.cpu(), expected)
    assert elapsed >= 0 and meta["route"] == "maca_csc_int_gather"
