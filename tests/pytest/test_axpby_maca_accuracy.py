# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA complex Axpby fusion, including live inputs and mutable structure."""
import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations._common import _is_maca_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device

pytestmark = [pytest.mark.axpby, pytest.mark.skipif(
    not (_is_maca_runtime() and accelerator_available()), reason="MACA required")]


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("size", [7, 1031, 65539])
@pytest.mark.parametrize("alpha,beta", [(1.5, .5), (1 + 2j, -.5 + .25j), (0., 1.)])
def test_fused_live_inputs_and_topology(index_dtype, inference, size, alpha, beta):
    dev = accelerator_device()
    with torch.inference_mode(inference):
        indices = torch.tensor([3, 99, 0, 99, 6, 99], dtype=index_dtype, device=dev)[::2]
        values = torch.tensor([1 + 2j, 0, -3 + 1j, 0, 4 - 2j, 0],
                              dtype=torch.complex64, device=dev)[::2]
        y = torch.arange(size, dtype=torch.float32, device=dev).to(torch.complex64) * (1 + .1j)
        for step in range(3):
            if step:
                values.mul_(.5 + .1j)
                indices.copy_(torch.tensor([6, 2, 1] if step == 1 else [3, 0, 6],
                                           dtype=index_dtype, device=dev))
                y.add_(1 - .5j)
            expected = y.cpu().to(torch.complex128) * beta
            expected[indices.cpu().long()] += alpha * values.cpu().to(torch.complex128)
            actual, elapsed = fs.flagsparse_axpby(values, indices, y, alpha, beta, return_time=True)
            assert actual is y and elapsed >= 0
            torch.testing.assert_close(actual.cpu().to(torch.complex128), expected,
                                       rtol=2e-5, atol=2e-4)


@pytest.mark.parametrize("size", [0, 7, 1031])
@pytest.mark.parametrize("beta", [0., 1., -.5 + .25j])
def test_fused_empty_sparse_vector(size, beta):
    dev = accelerator_device()
    values = torch.empty(0, dtype=torch.complex64, device=dev)
    indices = torch.empty(0, dtype=torch.int64, device=dev)
    y = torch.full((size,), 2 - 3j, dtype=torch.complex64, device=dev)
    expected = y.cpu() * beta
    assert fs.flagsparse_axpby(values, indices, y, 1 + 2j, beta) is y
    torch.testing.assert_close(y.cpu(), expected)


def test_fused_dense_unsorted_indices():
    dev = accelerator_device()
    gen = torch.Generator().manual_seed(129)
    indices = torch.randperm(1031, generator=gen).int().to(dev)
    values = torch.randn(1031, dtype=torch.complex64, generator=gen).to(dev)
    y = torch.randn(1031, dtype=torch.complex64, generator=gen).to(dev)
    expected = y.cpu() * (.5 - .2j)
    expected[indices.cpu().long()] += values.cpu() * (1 + .3j)
    fs.flagsparse_axpby(values, indices, y, 1 + .3j, .5 - .2j)
    torch.testing.assert_close(y.cpu(), expected)


def test_missing_entries_ignore_nonfinite_alpha():
    dev = accelerator_device()
    values = torch.ones(1, dtype=torch.complex64, device=dev)
    indices = torch.tensor([1], dtype=torch.int32, device=dev)
    y = torch.ones(5, dtype=torch.complex64, device=dev)
    fs.flagsparse_axpby(values, indices, y, complex(float("nan"), 0), .5)
    torch.testing.assert_close(y.cpu()[torch.tensor([0, 2, 3, 4])],
                               torch.full((4,), .5, dtype=torch.complex64))


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_warm_cache_does_not_bypass_range_validation(index_dtype):
    dev = accelerator_device()
    values = torch.ones(1, dtype=torch.complex64, device=dev)
    indices = torch.tensor([1], dtype=index_dtype, device=dev)
    y = torch.ones(7, dtype=torch.complex64, device=dev)
    fs.flagsparse_axpby(values, indices, y, 1.5, .5)
    indices.fill_(7)
    with pytest.raises(IndexError, match="indices out of range"):
        fs.flagsparse_axpby(values, indices, y, 1.5, .5)
