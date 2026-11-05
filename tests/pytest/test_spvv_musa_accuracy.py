# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MUSA narrow-input SpVV must reduce live operands into independent scalars."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations._common import _ACCEL, _is_mthreads_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device

pytestmark = [pytest.mark.spvv, pytest.mark.skipif(
    not (_is_mthreads_runtime() and accelerator_available()), reason="MUSA required")]


def _check(values, indices, y, op="non", **kwargs):
    expected = (values.cpu().double() * y.cpu().double()[indices.cpu().long()]).sum()
    actual = fs.flagsparse_spvv(values, indices, y, op=op, **kwargs)
    if kwargs.get("return_time"):
        actual, elapsed = actual
        assert elapsed >= 0
    assert actual.shape == () and actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu().double(), expected, atol=3e-4, rtol=2e-5)
    return actual


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("op", ["non", "conj"])
@pytest.mark.parametrize("nnz", [0, 1, 63, 1023, 1024, 1025, 4095, 4096, 4097,
                                 16383, 16384, 16385, 65535, 65536, 65537, 262145,
                                 1048577])
def test_half_reduction_boundaries(index_dtype, op, nnz):
    gen = torch.Generator().manual_seed(91 + nnz)
    dev = accelerator_device()
    indices = torch.randint(0, 10007, (nnz,), generator=gen, dtype=index_dtype).to(dev)
    values = torch.randn(nnz, dtype=torch.float16, generator=gen).to(dev)
    y = torch.randn(10007, dtype=torch.float16, generator=gen).to(dev)
    _check(values, indices, y, op)


@pytest.mark.parametrize("nnz", [4093, 4097, 16385, 65539])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("op", ["non", "conj"])
def test_strided_live_half_inputs(nnz, index_dtype, op):
    gen = torch.Generator().manual_seed(22)
    dev = accelerator_device()
    values = torch.randn(2 * nnz, dtype=torch.float16, generator=gen).to(dev)[::2]
    y = torch.randn(2 * 10007, dtype=torch.float16, generator=gen).to(dev)[::2]
    indices = torch.randint(0, 10007, (2 * nnz,), generator=gen, dtype=index_dtype).to(dev)[::2]
    for step in range(3):
        if step:
            values.mul_(.5)
            y.add_(.1)
            indices.fill_(step)
        _check(values, indices, y, op, return_time=True, out_dtype=torch.float32)


@pytest.mark.parametrize("nnz", [2, 8193])
def test_result_lifetime_and_validation(nnz):
    dev = accelerator_device()
    values = torch.full((nnz,), 1.5, dtype=torch.float16, device=dev)
    indices = torch.arange(nnz, dtype=torch.int32, device=dev) % 3
    y = torch.tensor([2., 3., 4.], dtype=torch.float16, device=dev)
    first = _check(values, indices, y)
    saved = first.cpu().clone()
    values.mul_(2)
    _check(values, indices, y)
    torch.testing.assert_close(first.cpu(), saved, rtol=0, atol=0)
    empty = _check(values[:0], indices[:0], y[:0], validate=False)
    assert empty.item() == 0
    for invalid in (-1, y.numel()):
        indices[0] = invalid
        with pytest.raises(IndexError, match="indices out of range"):
            fs.flagsparse_spvv(values, indices, y)
    with pytest.raises(TypeError, match="has no"):
        fs.flagsparse_spvv(values[:0], indices[:0], y, out_dtype=torch.float16)


@pytest.mark.parametrize("nnz", [4096, 65536])
def test_deterministic_half_reduction(nnz):
    gen = torch.Generator().manual_seed(315)
    dev = accelerator_device()
    values = torch.randn(nnz, dtype=torch.float16, generator=gen).to(dev)
    y = torch.randn(10007, dtype=torch.float16, generator=gen).to(dev)
    indices = torch.randint(0, y.numel(), (nnz,), generator=gen, dtype=torch.int32).to(dev)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        first = _check(values, indices, y)
        second = _check(values, indices, y)
        torch.testing.assert_close(first.cpu(), second.cpu(), rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _check_int8(values, indices, y, out_dtype, op="non", **kwargs):
    expected = (values.cpu().long() * y.cpu().long()[indices.cpu().long()]).sum()
    actual = fs.flagsparse_spvv(values, indices, y, op=op, out_dtype=out_dtype, **kwargs)
    if kwargs.get("return_time"):
        actual, elapsed = actual
        assert elapsed >= 0
    assert actual.shape == () and actual.dtype == out_dtype
    if out_dtype == torch.int32:
        torch.testing.assert_close(actual.cpu(), expected.to(torch.int32), rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual.cpu().double(), expected.double(), rtol=2e-5, atol=3e-4)
    return actual


@pytest.mark.parametrize("out_dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("op", ["non", "conj"])
@pytest.mark.parametrize("nnz", [0, 1, 1023, 1024, 1025, 4096, 16383, 16384,
                                 16385, 65535, 65536, 65537, 1048577])
def test_int8_reduction_boundaries(out_dtype, index_dtype, op, nnz):
    gen = torch.Generator().manual_seed(271 + nnz)
    dev = accelerator_device()
    indices = torch.randint(0, 10007, (nnz,), dtype=index_dtype, generator=gen).to(dev)
    values = torch.randint(-128, 128, (nnz,), dtype=torch.int8, generator=gen).to(dev)
    y = torch.randint(-128, 128, (10007,), dtype=torch.int8, generator=gen).to(dev)
    _check_int8(values, indices, y, out_dtype, op)


@pytest.mark.parametrize("out_dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("nnz", [4093, 16385, 65539])
def test_int8_strides_and_live_inputs(out_dtype, index_dtype, nnz):
    gen = torch.Generator().manual_seed(73)
    dev = accelerator_device()
    values = torch.randint(-16, 17, (2 * nnz,), dtype=torch.int8, generator=gen).to(dev)[::2]
    indices = torch.randint(0, 13, (2 * nnz,), dtype=index_dtype, generator=gen).to(dev)[::2]
    y = torch.randint(-16, 17, (26,), dtype=torch.int8, generator=gen).to(dev)[::2]
    first = _check_int8(values, indices, y, out_dtype)
    saved = first.cpu().clone()
    for step in (1, 2):
        values.add_(1)
        y.sub_(1)
        indices.fill_(step)
        _check_int8(values, indices, y, out_dtype, op="conj", return_time=True)
        torch.testing.assert_close(first.cpu(), saved, rtol=0, atol=0)
    for bad in (-1, y.numel()):
        indices[0] = bad
        with pytest.raises(IndexError, match="indices out of range"):
            fs.flagsparse_spvv(values, indices, y, out_dtype=out_dtype)


@pytest.mark.parametrize("out_dtype", [torch.int32, torch.float32])
def test_int8_large_accumulation_and_streams(out_dtype):
    dev = accelerator_device()
    values = torch.full((262145,), 127, dtype=torch.int8, device=dev)
    indices = torch.zeros(values.numel(), dtype=torch.int32, device=dev)
    y = torch.tensor([127], dtype=torch.int8, device=dev)
    # FP32 must not overflow through an int32 accumulator; int32 wraps as before.
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        first = _check_int8(values, indices, y, out_dtype)
        second = _check_int8(values, indices, y, out_dtype)
        torch.testing.assert_close(first.cpu(), second.cpu(), rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    _ACCEL.synchronize()
    stream = torch.musa.Stream()
    with torch.musa.stream(stream):
        values.fill_(-128)
        y.fill_(127)
        result = fs.flagsparse_spvv(values, indices, y, out_dtype=out_dtype, validate=False)
    stream.synchronize()
    expected = (values.cpu().long() * 127).sum()
    torch.testing.assert_close(result.cpu(), expected.to(out_dtype), rtol=2e-5 if out_dtype == torch.float32 else 0, atol=0)


@pytest.mark.parametrize("nnz", [4096, 65536])
def test_independent_results_on_streams(nnz):
    dev = accelerator_device()
    gen = torch.Generator().manual_seed(81)
    values = torch.randn(nnz, dtype=torch.float16, generator=gen).to(dev)
    y = torch.randn(8191, dtype=torch.float16, generator=gen).to(dev)
    indices = torch.randint(0, y.numel(), (nnz,), generator=gen, dtype=torch.int32).to(dev)
    _ACCEL.synchronize()
    streams = [_ACCEL.Stream(), _ACCEL.Stream()]
    outputs = []
    for i, stream in enumerate(streams):
        with _ACCEL.stream(stream):
            outputs.append(fs.flagsparse_spvv(values * (i + 1), indices, y, validate=False))
    for stream in streams:
        stream.synchronize()
    expected = (values.cpu().double() * y.cpu().double()[indices.cpu().long()]).sum()
    for i, actual in enumerate(outputs):
        torch.testing.assert_close(actual.cpu().double(), expected * (i + 1),
                                   atol=3e-4, rtol=2e-5)
