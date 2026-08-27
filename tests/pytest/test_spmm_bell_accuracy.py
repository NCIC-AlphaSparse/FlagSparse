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

import importlib

import pytest
import torch

from flagsparse import flagsparse_spmm_bell, prepare_spmm_bell_route
from tests.pytest.accuracy_utils import close_tolerances


spmm_bell_mod = importlib.import_module("flagsparse.sparse_operations.spmm_bell")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA/ROCm PyTorch device required",
)

SPMM_BELL_MKN_SHAPES = ((7, 5, 4), (16, 32, 8), (17, 19, 7))


def _value_dtype_cases():
    return [
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    ]


def _random_values(shape, dtype, device):
    if dtype in (torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device) * 0.125
    if dtype == torch.complex64:
        return torch.complex(
            torch.randn(shape, dtype=torch.float32, device=device) * 0.125,
            torch.randn(shape, dtype=torch.float32, device=device) * 0.125,
        )
    if dtype == torch.complex128:
        return torch.complex(
            torch.randn(shape, dtype=torch.float64, device=device) * 0.125,
            torch.randn(shape, dtype=torch.float64, device=device) * 0.125,
        )
    raise TypeError(f"unsupported dtype: {dtype}")


def _reference_dtype(dtype):
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _entries_from_dense(dense):
    rows, cols = torch.nonzero(dense != 0, as_tuple=True)
    return {
        (int(row.item()), int(col.item())): dense[row, col].item()
        for row, col in zip(rows, cols)
    }


def _zero_value(dtype):
    return 0j if dtype.is_complex else 0.0


def _entries_to_bell(entries, shape, dtype, index_dtype, block_dim, device):
    M, _K = shape
    mb = (M + block_dim - 1) // block_dim
    blocks = {}
    for (row, col), value in entries.items():
        brow = int(row) // block_dim
        bcol = int(col) // block_dim
        inner_row = int(row) % block_dim
        inner_col = int(col) % block_dim
        block = blocks.setdefault(
            (brow, bcol),
            [_zero_value(dtype) for _ in range(block_dim * block_dim)],
        )
        block[inner_row * block_dim + inner_col] += complex(value) if dtype.is_complex else float(value)
    row_blocks = [[] for _ in range(mb)]
    for key in sorted(blocks):
        row_blocks[key[0]].append(key)
    ell_width_blocks = max(1, max([len(row) for row in row_blocks] or [0]))
    data_values = [_zero_value(dtype)] * (mb * ell_width_blocks * block_dim * block_dim)
    index_values = [-1] * (mb * ell_width_blocks)
    for brow, keys in enumerate(row_blocks):
        for slot, key in enumerate(keys):
            index_values[brow * ell_width_blocks + slot] = key[1]
            base = (brow * ell_width_blocks + slot) * block_dim * block_dim
            data_values[base : base + block_dim * block_dim] = blocks[key]
    data = torch.tensor(data_values, dtype=dtype, device=device)
    data = data.reshape(mb, ell_width_blocks, block_dim, block_dim).contiguous()
    indices = torch.tensor(index_values, dtype=index_dtype, device=device)
    indices = indices.reshape(mb, ell_width_blocks).contiguous()
    return data, indices


def _make_problem(M, K, N, dtype, index_dtype, block_dim):
    device = torch.device("cuda")
    mask = torch.rand((M, K), device=device) < 0.2
    mask[0, 0] = True
    if M > 4 and K > 4:
        mask[M - 1, K - 1] = True
    dense = torch.where(
        mask,
        _random_values((M, K), dtype, device),
        torch.zeros((), dtype=dtype, device=device),
    )
    entries = _entries_from_dense(dense)
    data, indices = _entries_to_bell(entries, (M, K), dtype, index_dtype, block_dim, device)
    B = _random_values((K, N), dtype, device).contiguous()
    return dense, data, indices, B


def _dense_reference(dense, B, dtype):
    ref_dtype = _reference_dtype(dtype)
    return (dense.to(ref_dtype) @ B.to(ref_dtype)).to(dtype)


@pytest.mark.spmm_bell
@pytest.mark.parametrize("M, K, N", SPMM_BELL_MKN_SHAPES)
@pytest.mark.parametrize("name, dtype", _value_dtype_cases())
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("block_dim", [2, 4])
def test_spmm_bell_matches_dense_reference(M, K, N, name, dtype, index_dtype, block_dim):
    del name
    dense, data, indices, B = _make_problem(M, K, N, dtype, index_dtype, block_dim)
    out = flagsparse_spmm_bell(data, indices, B, shape=(M, K), block_dim=block_dim, op="non")
    ref = _dense_reference(dense, B, dtype)
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    assert out.shape == (M, N)


@pytest.mark.spmm_bell
def test_spmm_bell_prepared_path_and_meta():
    M, K, N = 7, 5, 3
    dtype = torch.float32
    dense, data, indices, B = _make_problem(M, K, N, dtype, torch.int64, 2)
    prepared = prepare_spmm_bell_route(data, indices, (M, K), block_dim=2, op="non")
    out, meta = spmm_bell_mod.flagsparse_spmm_bell_run(
        prepared,
        B,
        op="non",
        return_meta=True,
        timing=True,
    )
    ref = _dense_reference(dense, B, dtype)
    rtol, atol = close_tolerances(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    assert meta["alg"] == "spmm_bell_base"
    assert meta["op"] == "non"
    assert meta["logical_shape"] == (M, K)
    assert meta["ell_width_blocks"] == data.shape[1]
    assert meta["process_cpu_ms"] == 0.0
    assert meta["process_gpu_ms"] == 0.0
    assert meta["compute_ms"] >= 0.0


@pytest.mark.spmm_bell
def test_spmm_bell_empty_slots_are_index_minus_one():
    M, K, N = 5, 7, 2
    dtype = torch.float32
    device = torch.device("cuda")
    data = torch.zeros((3, 2, 2, 2), dtype=dtype, device=device)
    indices = torch.full((3, 2), -1, dtype=torch.int64, device=device)
    data[0, 0, 0, 0] = 2.0
    indices[0, 0] = 0
    data[0, 1, 0, 0] = 999.0
    B = torch.randn((K, N), dtype=dtype, device=device)
    out = flagsparse_spmm_bell(data, indices, B, shape=(M, K), block_dim=2, op="non")
    ref = torch.zeros((M, N), dtype=dtype, device=device)
    ref[0] = 2.0 * B[0]
    torch.testing.assert_close(out, ref)


@pytest.mark.spmm_bell
def test_spmm_bell_B_length_mismatch_rejected():
    M, K, N = 8, 10, 3
    _dense, data, indices, _B = _make_problem(M, K, N, torch.float32, torch.int32, 2)
    bad_B = torch.randn((K + 2, N), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="B must have shape"):
        flagsparse_spmm_bell(data, indices, bad_B, shape=(M, K), block_dim=2, op="non")


@pytest.mark.spmm_bell
def test_spmm_bell_prepared_op_mismatch_rejected():
    M, K, N = 8, 10, 3
    _dense, data, indices, B = _make_problem(M, K, N, torch.float32, torch.int32, 2)
    prepared = prepare_spmm_bell_route(data, indices, (M, K), block_dim=2, op="non")
    with pytest.raises(ValueError, match="reserved"):
        spmm_bell_mod.flagsparse_spmm_bell_run(prepared, B, op="trans")


@pytest.mark.spmm_bell
@pytest.mark.parametrize("op", ["trans", "conj"])
def test_spmm_bell_transpose_family_reserved(op):
    M, K, N = 8, 10, 3
    _dense, data, indices, B = _make_problem(M, K, N, torch.float32, torch.int32, 2)
    with pytest.raises(ValueError, match="reserved"):
        flagsparse_spmm_bell(data, indices, B, shape=(M, K), block_dim=2, op=op)
