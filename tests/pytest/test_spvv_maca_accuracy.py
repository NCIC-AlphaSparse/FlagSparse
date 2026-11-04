# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA complex SpVV: reduction boundaries and live, strided logical inputs."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations._common import _is_maca_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device

pytestmark = [pytest.mark.spvv, pytest.mark.skipif(
    not (_is_maca_runtime() and accelerator_available()), reason="MACA required")]


def _check(values, indices, y, op, **kwargs):
    lhs = values.cpu().to(torch.complex128)
    if op == "conj":
        lhs = lhs.conj()
    rhs = y.cpu().to(torch.complex128)[indices.cpu().long()]
    expected = (lhs * rhs).sum()
    actual = fs.flagsparse_spvv(values, indices, y, op=op, **kwargs)
    if kwargs.get("return_time"):
        actual, elapsed = actual
        assert elapsed >= 0
    assert actual.shape == () and actual.dtype == torch.complex64
    torch.testing.assert_close(actual.cpu().to(torch.complex128), expected,
                               atol=3e-4, rtol=2e-5)
    return actual


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("op", ["non", "conj"])
@pytest.mark.parametrize("nnz", [0, 1, 63, 65, 1023, 1024, 1025, 4095, 4096, 4097, 16387, 65539])
def test_reduction_boundaries(index_dtype, op, nnz):
    gen = torch.Generator().manual_seed(91 + nnz)
    dev = accelerator_device()
    # Unsorted and duplicate indices are valid for a dot product.
    indices = torch.randint(0, 10007, (nnz,), generator=gen, dtype=index_dtype).to(dev)
    values = torch.randn(nnz, dtype=torch.complex64, generator=gen).to(dev)
    y = torch.randn(10007, dtype=torch.complex64, generator=gen).to(dev)
    _check(values, indices, y, op)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("op", ["non", "conj"])
@pytest.mark.parametrize("logical_view", ["plain", "conj", "neg"])
def test_strided_mutable_inputs(index_dtype, op, logical_view):
    gen = torch.Generator().manual_seed(22)
    dev = accelerator_device()
    values = torch.randn(2 * 4093, dtype=torch.complex64, generator=gen).to(dev)[::2]
    y = torch.randn(2 * 10007, dtype=torch.complex64, generator=gen).to(dev)[::2]
    indices = torch.randint(0, 10007, (2 * 4093,), generator=gen, dtype=index_dtype).to(dev)[::2]
    for step in range(3):
        if step:
            values.mul_(.5 + .25j)
            y.add_(.1 - .3j)
            indices.fill_(step)
        x, dense = values, y
        if logical_view == "conj":
            x, dense = x.conj(), dense.conj()
        elif logical_view == "neg":
            x, dense = torch._neg_view(x), torch._neg_view(dense)
        _check(x, indices, dense, op, return_time=True, out_dtype=torch.complex64)


def test_result_lifetime_empty_and_invalid_indices():
    dev = accelerator_device()
    values = torch.tensor([1 + 2j, -3 + 4j], dtype=torch.complex64, device=dev)
    indices = torch.tensor([0, 2], dtype=torch.int32, device=dev)
    y = torch.tensor([2 - 1j, 3j, 4 + 2j], dtype=torch.complex64, device=dev)
    result = _check(values, indices, y, "non")
    saved = result.cpu().clone()
    values.mul_(2)
    _check(values, indices, y, "non")
    torch.testing.assert_close(result.cpu(), saved)
    _check(values[:0], indices[:0], y[:0], "non", validate=False)
    for invalid in (-1, y.numel()):
        indices[0] = invalid
        with pytest.raises(IndexError, match="indices out of range"):
            fs.flagsparse_spvv(values, indices, y)
