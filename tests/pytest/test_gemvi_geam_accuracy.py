# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Edge cases of flagsparse_gemvi and flagsparse_geam_csr beyond the one
delivery case each variant has in test_q4_variants_accuracy.py: layouts, index
types, complex scalars, empty rows and matrices, full overlap, and the inputs
both operators must reject. Checked against a CPU float64 / complex128 golden.
"""

import pytest
import torch

import flagsparse as fs
from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
)

pytestmark = pytest.mark.skipif(not accelerator_available(), reason=ACCELERATOR_REQUIRED)

_TOL = {torch.float32: 1e-4, torch.complex64: 1e-4, torch.float64: 1e-10, torch.complex128: 1e-10}
DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]


def _rand(shape, dtype, gen):
    if dtype.is_complex:
        real = torch.float32 if dtype == torch.complex64 else torch.float64
        return torch.complex(
            torch.randn(shape, dtype=real, generator=gen),
            torch.randn(shape, dtype=real, generator=gen),
        )
    return torch.randn(shape, generator=gen).to(dtype)


def _wide(t):
    return t.detach().cpu().to(torch.complex128 if t.is_complex() else torch.float64)


def _close(got, ref, dtype):
    got, ref = _wide(got), ref.to(_wide(got).dtype)
    scale = max(1.0, float(ref.abs().max())) if ref.numel() else 1.0
    err = float((got - ref).abs().max()) if ref.numel() else 0.0
    assert err <= _TOL[dtype] * scale, f"max error {err:.3g} (scale {scale:.3g})"


# ------------------------------------------------------------------------- gemvi
@pytest.mark.gemvi
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("op", ["non", "trans"])
@pytest.mark.parametrize("layout", ["col", "row"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=str)
def test_gemvi_matches_dense_golden(dtype, op, layout, index_dtype):
    gen = torch.Generator().manual_seed(7)
    dev = accelerator_device()
    m, n = 131, 77
    A = _rand((m, n), dtype, gen)
    n_in, n_out = (n, m) if op == "non" else (m, n)
    idx = torch.randperm(n_in, generator=gen)[:40]
    x, y = _rand((40,), dtype, gen), _rand((n_out,), dtype, gen)
    alpha, beta = (1.5 - 0.25j, 0.5 + 1j) if dtype.is_complex else (1.5, -0.5)
    A_d = A.to(dev)
    if layout == "col":
        A_d = A_d.t().contiguous().t()
    y_d = y.to(dev)
    out = fs.flagsparse_gemvi(
        A_d, x.to(dev), idx.to(index_dtype).to(dev), y_d, op=op, alpha=alpha, beta=beta
    )
    assert out is y_d
    dense_x = torch.zeros(n_in, dtype=_wide(x).dtype)
    dense_x[idx] = _wide(x)
    opA = _wide(A) if op == "non" else _wide(A).t()
    _close(out, alpha * (opA @ dense_x) + beta * _wide(y), dtype)


@pytest.mark.gemvi
def test_gemvi_without_y_and_with_an_empty_x():
    dev = accelerator_device()
    A = torch.randn(9, 5, device=dev)
    empty = torch.empty(0, device=dev)
    no_idx = torch.empty(0, dtype=torch.int32, device=dev)
    assert torch.equal(fs.flagsparse_gemvi(A, empty, no_idx).cpu(), torch.zeros(9))
    idx = torch.tensor([4, 0], dtype=torch.int32, device=dev)
    vals = torch.tensor([2.0, -1.0], device=dev)
    _close(fs.flagsparse_gemvi(A, vals, idx), _wide(A[:, 4] * 2 - A[:, 0]), torch.float32)


@pytest.mark.gemvi
def test_gemvi_rejects_what_cusparse_gemvi_does_not_do():
    dev = accelerator_device()
    A = torch.randn(4, 3, device=dev)
    vals = torch.ones(1, device=dev)
    idx = torch.zeros(1, dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match="conjugate"):
        fs.flagsparse_gemvi(A, vals, idx, op="conj")
    with pytest.raises(ValueError, match="beta"):
        fs.flagsparse_gemvi(A, vals, idx, beta=1.0)
    with pytest.raises(IndexError):
        fs.flagsparse_gemvi(A, vals, torch.tensor([3], dtype=torch.int32, device=dev))
    with pytest.raises(TypeError):
        fs.flagsparse_gemvi(A.half(), vals.half(), idx)


# ---------------------------------------------------------------------- geam_csr
def _csr(dense, dev, ptr_dtype=torch.int32):
    s = dense.to_sparse_csr()
    return (
        s.values().to(dev),
        s.col_indices().to(torch.int32).to(dev),
        s.crow_indices().to(ptr_dtype).to(dev),
    )


def _sparse(m, n, dtype, gen, density):
    mask = torch.rand((m, n), generator=gen) < density
    return torch.where(mask, _rand((m, n), dtype, gen), torch.zeros((), dtype=dtype))


def _check_geam(A, B, alpha, beta, dtype, ptr_dtype=torch.int32):
    dev = accelerator_device()
    m, n = A.shape
    c_data, c_indices, c_indptr = fs.flagsparse_geam_csr(
        *_csr(A, dev, ptr_dtype), *_csr(B, dev, ptr_dtype), (m, n), alpha=alpha, beta=beta
    )
    assert c_indices.dtype == torch.int32 and c_indptr.dtype == ptr_dtype
    ptr, cols = c_indptr.cpu().to(torch.int64), c_indices.cpu().to(torch.int64)
    rows = torch.repeat_interleave(torch.arange(m), ptr.diff())
    same_row = rows[1:] == rows[:-1]
    assert bool((cols[1:][same_row] > cols[:-1][same_row]).all())
    # The stored pattern is the union of the two stored patterns, sums of zero kept.
    union = (A != 0) | (B != 0)
    assert int(c_data.numel()) == int(union.sum())
    assert torch.equal(torch.zeros_like(union).index_put_((rows, cols), torch.tensor(True)), union)
    C = torch.sparse_csr_tensor(ptr, cols, c_data.cpu(), (m, n)).to_dense()
    _close(C, alpha * _wide(A) + beta * _wide(B), dtype)


@pytest.mark.geam_csr
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("ptr_dtype", [torch.int32, torch.int64], ids=str)
def test_geam_matches_dense_golden(dtype, ptr_dtype):
    gen = torch.Generator().manual_seed(11)
    A, B = _sparse(203, 97, dtype, gen, 0.1), _sparse(203, 97, dtype, gen, 0.3)
    A[5], B[6] = 0, 0  # a row only B has, a row only A has
    A[7], B[7] = 0, 0  # a row neither has
    alpha, beta = (0.5 - 2j, 1.25 + 0.5j) if dtype.is_complex else (0.5, 1.25)
    _check_geam(A, B, alpha, beta, dtype, ptr_dtype)


@pytest.mark.geam_csr
def test_geam_long_rows_and_full_overlap():
    gen = torch.Generator().manual_seed(12)
    A = _sparse(17, 3000, torch.float32, gen, 0.5)  # rows past the widest tile
    _check_geam(A, A, 1.0, -1.0, torch.float32)  # every column collides, sums are 0
    _check_geam(A, _sparse(17, 3000, torch.float32, gen, 0.05), 2.0, 3.0, torch.float32)


@pytest.mark.geam_csr
def test_geam_empty_operands():
    gen = torch.Generator().manual_seed(13)
    A = _sparse(40, 30, torch.float64, gen, 0.2)
    zero = torch.zeros(40, 30, dtype=torch.float64)
    _check_geam(A, zero, 1.0, 1.0, torch.float64)
    _check_geam(zero, A, 1.0, 2.0, torch.float64)
    _check_geam(zero, zero, 1.0, 1.0, torch.float64)


@pytest.mark.geam_csr
def test_geam_rejects_unsorted_or_mismatched_input():
    dev = accelerator_device()
    vals = torch.ones(2, device=dev)
    ptr = torch.tensor([0, 2], dtype=torch.int32, device=dev)
    sorted_cols = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    unsorted = torch.tensor([1, 0], dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match="sorted"):
        fs.flagsparse_geam_csr(vals, unsorted, ptr, vals, sorted_cols, ptr, (1, 2))
    with pytest.raises(TypeError):
        other = torch.ones(2, dtype=torch.float64).to(dev)
        fs.flagsparse_geam_csr(vals, sorted_cols, ptr, other, sorted_cols, ptr, (1, 2))
    with pytest.raises(TypeError, match="complex"):
        fs.flagsparse_geam_csr(vals, sorted_cols, ptr, vals, sorted_cols, ptr, (1, 2), alpha=1j)


@pytest.mark.geam_csr
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("ptr_dtype", [torch.int32, torch.int64], ids=str)
def test_geam_large_sorted_keys_and_empty_rows(dtype, ptr_dtype):
    dev = accelerator_device()
    values = _rand((3,), dtype, torch.Generator().manual_seed(67))
    other_values = _rand((3,), dtype, torch.Generator().manual_seed(71))
    columns = torch.tensor([7, 11, 90000], dtype=torch.int32)
    other_columns = torch.tensor([0, 11, 99999], dtype=torch.int32)
    pointer = torch.zeros(70001, dtype=ptr_dtype)
    pointer[30001:] = 2
    pointer[60001:] = 3
    other_pointer = torch.zeros_like(pointer)
    other_pointer[1:] = 1
    other_pointer[30001:] = 2
    other_pointer[60001:] = 3
    alpha, beta = (1.25 - 0.5j, -2 + 1j) if dtype.is_complex else (1.25, -2)
    expected_values = torch.stack((
        beta * other_values[0], alpha * values[0],
        alpha * values[1] + beta * other_values[1],
        alpha * values[2], beta * other_values[2],
    ))
    expected_columns = torch.tensor([0, 7, 11, 90000, 99999], dtype=torch.int32)
    expected_pointer = torch.zeros_like(pointer)
    expected_pointer[1:] = 1
    expected_pointer[30001:] = 3
    expected_pointer[60001:] = 5
    operands = tuple(tensor.to(dev) for tensor in (
        values, columns, pointer, other_values, other_columns, other_pointer,
    ))
    result = fs.flagsparse_geam_csr(
        *operands, (70000, 100000), alpha=alpha, beta=beta)
    _close(result[0], _wide(expected_values), dtype)
    torch.testing.assert_close(result[1].cpu(), expected_columns)
    torch.testing.assert_close(result[2].cpu(), expected_pointer)
