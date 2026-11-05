# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA float32/complex64 SpVV: direct and atomic reductions with live inputs."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations._common import _is_maca_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device

pytestmark = [pytest.mark.spvv, pytest.mark.skipif(
    not (_is_maca_runtime() and accelerator_available()), reason="MACA required")]


def _check(values, indices, y, op, **kwargs):
    wide = torch.complex128 if values.is_complex() else torch.float64
    lhs = values.cpu().to(wide)
    if op == "conj":
        lhs = lhs.conj()
    rhs = y.cpu().to(wide)[indices.cpu().long()]
    expected = (lhs * rhs).sum()
    actual = fs.flagsparse_spvv(values, indices, y, op=op, **kwargs)
    if kwargs.get("return_time"):
        actual, elapsed = actual
        assert elapsed >= 0
    assert actual.shape == () and actual.dtype == values.dtype
    torch.testing.assert_close(actual.cpu().to(wide), expected,
                               atol=3e-4, rtol=2e-5)
    return actual


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("op", ["non", "conj"])
@pytest.mark.parametrize("nnz", [0, 1, 63, 65, 1023, 1024, 1025, 4095, 4096, 4097,
                                 16387, 65535, 65536, 65537, 65539])
def test_reduction_boundaries(dtype, index_dtype, op, nnz):
    gen = torch.Generator().manual_seed(91 + nnz)
    dev = accelerator_device()
    # Unsorted and duplicate indices are valid for a dot product.
    indices = torch.randint(0, 10007, (nnz,), generator=gen, dtype=index_dtype).to(dev)
    values = torch.randn(nnz, dtype=dtype, generator=gen).to(dev)
    y = torch.randn(10007, dtype=dtype, generator=gen).to(dev)
    _check(values, indices, y, op)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("nnz", [4093, 8195])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("op", ["non", "conj"])
@pytest.mark.parametrize("logical_view", ["plain", "conj", "neg"])
def test_strided_mutable_inputs(dtype, nnz, index_dtype, op, logical_view):
    gen = torch.Generator().manual_seed(22)
    dev = accelerator_device()
    values = torch.randn(2 * nnz, dtype=dtype, generator=gen).to(dev)[::2]
    y = torch.randn(2 * 10007, dtype=dtype, generator=gen).to(dev)[::2]
    indices = torch.randint(0, 10007, (2 * nnz,), generator=gen, dtype=index_dtype).to(dev)[::2]
    for step in range(3):
        if step:
            values.mul_(.5 + .25j if dtype.is_complex else .5)
            y.add_(.1 - .3j if dtype.is_complex else .1)
            indices.fill_(step)
        x, dense = values, y
        if logical_view == "conj":
            x, dense = x.conj(), dense.conj()
        elif logical_view == "neg":
            x, dense = torch._neg_view(x), torch._neg_view(dense)
        _check(x, indices, dense, op, return_time=True, out_dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("nnz", [2, 8193])
def test_result_lifetime_empty_and_invalid_indices(dtype, nnz):
    dev = accelerator_device()
    values = torch.full((nnz,), 1 + 2j if dtype.is_complex else 1.5, dtype=dtype, device=dev)
    indices = torch.arange(nnz, dtype=torch.int32, device=dev) % 3
    y = torch.tensor([2 - 1j, 3j, 4 + 2j] if dtype.is_complex else [2., 3., 4.],
                     dtype=dtype, device=dev)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("op", ["non", "conj"])
def test_deterministic_large_reduction(dtype, op):
    gen = torch.Generator().manual_seed(315)
    dev = accelerator_device()
    values = torch.randn(65536, dtype=dtype, generator=gen).to(dev)
    y = torch.randn(10007, dtype=dtype, generator=gen).to(dev)
    indices = torch.randint(0, y.numel(), (values.numel(),), generator=gen,
                            dtype=torch.int32).to(dev)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        first = _check(values, indices, y, op)
        second = _check(values, indices, y, op)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_independent_outputs_on_multiple_streams(dtype):
    dev = accelerator_device()
    gen = torch.Generator().manual_seed(81)
    values = torch.randn(65536, dtype=dtype, generator=gen).to(dev)
    y = torch.randn(8191, dtype=dtype, generator=gen).to(dev)
    indices = torch.randint(0, y.numel(), (values.numel(),), generator=gen,
                            dtype=torch.int32).to(dev)
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    results = []
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            # Distinct live values and outputs, without any shared scratch state.
            results.append(fs.flagsparse_spvv(values * (i + 1), indices, y, validate=False))
    for stream in streams:
        stream.synchronize()
    wide = torch.complex128 if dtype.is_complex else torch.float64
    expected = (values.cpu().to(wide) * y.cpu().to(wide)[indices.cpu().long()]).sum()
    for i, actual in enumerate(results):
        torch.testing.assert_close(actual.cpu().to(wide), expected * (i + 1),
                                   atol=3e-4, rtol=2e-5)
