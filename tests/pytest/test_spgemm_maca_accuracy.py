# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The pure Triton MACA product never falls back to torch sparse or ESC."""

import pytest
import torch

from flagsparse.sparse_operations import spgemm_csr as csr
from flagsparse.sparse_operations._common import _is_maca_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device

pytestmark = [pytest.mark.spgemm_csr, pytest.mark.skipif(
    not (_is_maca_runtime() and accelerator_available()), reason="MACA required"
)]


def _triple(matrix, device, ptr_dtype):
    sparse = matrix.to_sparse_csr()
    return (sparse.values().to(device), sparse.col_indices().int().to(device),
            sparse.crow_indices().to(device=device, dtype=ptr_dtype), tuple(matrix.shape))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("ptr_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(0, 7, 9), (7, 0, 9), (1, 128, 128), (1100, 3, 5)])
def test_pure_triton_edges_and_live_values(monkeypatch, dtype, ptr_dtype, shape):
    device = accelerator_device()
    m, k, n = shape
    gen = torch.Generator().manual_seed(91)
    a = torch.randn(m, k, dtype=dtype, generator=gen)
    b = torch.randn(k, n, dtype=dtype, generator=gen)
    if m > 1:
        a[0] = 0
    args_a, args_b = _triple(a, device, ptr_dtype), _triple(b, device, ptr_dtype)
    prepared = csr.prepare_spgemm_csr(*args_a, *args_b)

    def forbidden(*args, **kwargs):
        raise AssertionError("non-Triton sparse compute was called")

    monkeypatch.setattr(torch.sparse, "mm", forbidden)
    monkeypatch.setattr(csr, "_spgemm_esc_compute_safe", forbidden)
    for scale in (1.0, 2.0):
        if scale == 2:
            args_a[0].mul_(2)
        data, indices, ptr, result_shape = csr.flagsparse_spgemm_csr(prepared=prepared)
        result = torch.sparse_csr_tensor(ptr.cpu(), indices.cpu(), data.cpu(), result_shape)
        torch.testing.assert_close(result.to_dense(), (a * scale) @ b,
                                   atol=1e-5 if dtype == torch.float32 else 1e-11,
                                   rtol=1e-4 if dtype == torch.float32 else 1e-11)
        out = (torch.empty_like(data), torch.empty_like(indices), torch.empty_like(ptr))
        got = csr.flagsparse_spgemm_csr(prepared=prepared, out=out)
        for actual, expected in zip(got[:3], out):
            assert actual is expected
        torch.testing.assert_close(got[0], data)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_duplicate_columns_and_empty_rhs_rows(dtype):
    device = accelerator_device()
    a = (torch.tensor([1., -1., 2., 3.], dtype=dtype, device=device),
         torch.tensor([0, 0, 1, 2], dtype=torch.int32, device=device),
         torch.tensor([0, 3, 4], dtype=torch.int64, device=device), (2, 3))
    b = (torch.tensor([4., -2.], dtype=dtype, device=device),
         torch.tensor([1, 1], dtype=torch.int32, device=device),
         torch.tensor([0, 0, 2, 2], dtype=torch.int64, device=device), (3, 4))
    data, indices, ptr, shape = csr.flagsparse_spgemm_csr(*a, *b)
    result = torch.sparse_csr_tensor(ptr.cpu(), indices.cpu(), data.cpu(), shape)
    expected = torch.tensor([[0., 4., 0., 0.], [0., 0., 0., 0.]], dtype=dtype)
    torch.testing.assert_close(result.to_dense(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_long_row_hash_collisions(dtype):
    device = accelerator_device()
    inner, width = 128, 64
    # Work=8192 gives 16384 hash slots. Every distinct column hashes to the
    # same slot, exercising probing and duplicate insertion across iterations.
    columns = torch.arange(width, dtype=torch.int32, device=device) * 16384
    ad = torch.ones(inner, dtype=dtype, device=device)
    ai = torch.arange(inner, dtype=torch.int32, device=device)
    ap = torch.tensor([0, inner], dtype=torch.int64, device=device)
    bd = torch.ones(inner * width, dtype=dtype, device=device)
    bi = columns.repeat(inner)
    bp = torch.arange(inner + 1, dtype=torch.int64, device=device) * width
    data, indices, ptr, _ = csr.flagsparse_spgemm_csr(
        ad, ai, ap, (1, inner), bd, bi, bp, (inner, 1048576)
    )
    assert ptr.cpu().tolist() == [0, width]
    assert sorted(indices.cpu().tolist()) == columns.cpu().tolist()
    torch.testing.assert_close(data.cpu(), torch.full((width,), float(inner), dtype=dtype))


@pytest.mark.parametrize("n", [0, 1, 1023, 1024, 1025, 1048579])
def test_recursive_prefix_scan(n):
    from flagsparse.sparse_operations._spgemm_maca import _prefix_ptr

    values = torch.arange(n, dtype=torch.int64) % 7
    got = _prefix_ptr(values.to(accelerator_device())).cpu()
    expected = torch.cat((torch.zeros(1, dtype=torch.int64), values.cumsum(0)))
    torch.testing.assert_close(got, expected)


def test_largest_int32_column_is_not_padding():
    device = accelerator_device()
    one = torch.ones(1, device=device)
    ap = torch.tensor([0, 1], dtype=torch.int64, device=device)
    ai = torch.zeros(1, dtype=torch.int32, device=device)
    bi = torch.tensor([2147483647], dtype=torch.int32, device=device)
    data, indices, ptr, _ = csr.flagsparse_spgemm_csr(
        one, ai, ap, (1, 1), one, bi, ap, (1, 2147483648)
    )
    assert ptr.cpu().tolist() == [0, 1]
    assert indices.cpu().tolist() == [2147483647]
    assert data.cpu().tolist() == [1.0]
