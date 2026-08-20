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

from flagsparse import flagsparse_spmm_bsr, prepare_spmm_bsr_route
from tests.pytest.accuracy_utils import close_tolerances


spmm_bsr_mod = importlib.import_module("flagsparse.sparse_operations.spmm_bsr")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

BSR_MNK_SHAPES = ((7, 5, 4), (16, 32, 8), (64, 96, 16))


def _value_dtype_cases():
    cases = [
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    ]
    return cases


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


def _dense_to_bsr(dense, index_dtype, block_dim):
    device = dense.device
    M, K = dense.shape
    n_block_rows = (M + block_dim - 1) // block_dim
    rows, cols = torch.nonzero(dense != 0, as_tuple=True)
    blocks = {}
    for row, col in zip(rows.tolist(), cols.tolist()):
        brow = int(row) // block_dim
        bcol = int(col) // block_dim
        inner_row = int(row) % block_dim
        inner_col = int(col) % block_dim
        block = blocks.setdefault(
            (brow, bcol),
            torch.zeros((block_dim, block_dim), dtype=dense.dtype, device=device),
        )
        block[inner_row, inner_col] = dense[row, col]
    row_blocks = [[] for _ in range(n_block_rows)]
    for key in sorted(blocks):
        row_blocks[key[0]].append(key)
    data = []
    indices = []
    indptr = [0]
    for keys in row_blocks:
        for key in keys:
            indices.append(key[1])
            data.append(blocks[key])
        indptr.append(len(indices))
    if data:
        data_tensor = torch.stack(data).contiguous()
    else:
        data_tensor = torch.empty((0, block_dim, block_dim), dtype=dense.dtype, device=device)
    return (
        data_tensor,
        torch.tensor(indices, dtype=index_dtype, device=device),
        torch.tensor(indptr, dtype=index_dtype, device=device),
    )


def _random_bsr_mk(M, K, dtype, index_dtype, block_dim, device):
    p = min(0.25, max(0.06, 32.0 / max(M * K, 1)))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    dense = torch.where(
        mask,
        _random_values((M, K), dtype, device),
        torch.zeros((), dtype=dtype, device=device),
    )
    data, indices, indptr = _dense_to_bsr(dense, index_dtype, block_dim)
    return data, indices, indptr, dense


def _padded_rows(M, block_dim):
    return ((int(M) + int(block_dim) - 1) // int(block_dim)) * int(block_dim)


def _padded_cols(K, block_dim):
    return ((int(K) + int(block_dim) - 1) // int(block_dim)) * int(block_dim)


def _op_transposes(op):
    return op in ("trans", "conj")


def _logical_b_rows(M, K, op):
    return int(M) if _op_transposes(op) else int(K)


def _logical_out_rows(M, K, op):
    return int(K) if _op_transposes(op) else int(M)


def _padded_b_rows(M, K, block_dim, op):
    return _padded_rows(M, block_dim) if _op_transposes(op) else _padded_cols(K, block_dim)


def _padded_out_rows(M, K, block_dim, op):
    return _padded_cols(K, block_dim) if _op_transposes(op) else _padded_rows(M, block_dim)


def _dense_reference(dense, B, dtype, op):
    ref_dtype = _reference_dtype(dtype)
    dense_ref = dense.to(ref_dtype)
    B_ref = B.to(ref_dtype)
    if op == "non":
        return (dense_ref @ B_ref).to(dtype)
    if op == "trans":
        return (dense_ref.T @ B_ref).to(dtype)
    if op == "conj":
        return (dense_ref.conj().T @ B_ref).to(dtype)
    raise ValueError(f"unsupported op: {op}")


def _assert_close(actual, expected, dtype):
    rtol, atol = close_tolerances(dtype)
    ref_dtype = _reference_dtype(dtype)
    assert torch.allclose(
        actual.to(ref_dtype), expected.to(ref_dtype), rtol=rtol, atol=atol
    )


@pytest.mark.spmm_bsr
@pytest.mark.parametrize("M, K, N", BSR_MNK_SHAPES)
@pytest.mark.parametrize(
    "name,dtype", _value_dtype_cases(), ids=[case[0] for case in _value_dtype_cases()]
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("block_dim", [2, 4], ids=["block2", "block4"])
@pytest.mark.parametrize("op", ["non", "trans", "conj"], ids=["non", "trans", "conj"])
def test_spmm_bsr_matches_dense_reference(M, K, N, name, dtype, index_dtype, block_dim, op):
    del name
    device = torch.device("cuda")
    data, indices, indptr, dense = _random_bsr_mk(
        M, K, dtype, index_dtype, block_dim, device
    )
    B = _random_values((_logical_b_rows(M, K, op), N), dtype, device)
    ref = _dense_reference(dense, B, dtype, op)
    out = flagsparse_spmm_bsr(
        data,
        indices,
        indptr,
        B,
        shape=(M, K),
        block_dim=block_dim,
        op=op,
        index_fallback_policy="auto",
    )
    assert out.shape == (_padded_out_rows(M, K, block_dim, op), N)
    _assert_close(out[: _logical_out_rows(M, K, op), :], ref, dtype)


@pytest.mark.spmm_bsr
def test_spmm_bsr_prepared_path_and_meta():
    device = torch.device("cuda")
    M, K, N = 7, 5, 4
    dtype = torch.complex64
    block_dim = 4
    data, indices, indptr, dense = _random_bsr_mk(
        M, K, dtype, torch.int32, block_dim, device
    )
    op = "conj"
    prepared = prepare_spmm_bsr_route(data, indices, indptr, (M, K), block_dim=block_dim, op=op)
    B = _random_values((M, N), dtype, device)
    ref = _dense_reference(dense, B, dtype, op)
    out, meta = spmm_bsr_mod.flagsparse_spmm_bsr_run(
        prepared,
        B,
        op=op,
        return_meta=True,
        timing=True,
    )
    assert out.shape == (_padded_cols(K, block_dim), N)
    assert meta["alg"] == "spmm_bsr_base"
    assert meta["op"] == op
    assert meta["logical_shape"] == (M, K)
    assert meta["padded_shape"] == (_padded_rows(M, block_dim), _padded_cols(K, block_dim))
    assert meta["process_cpu_ms"] == 0.0
    assert meta["process_gpu_ms"] == 0.0
    assert meta["compute_ms"] >= 0.0
    _assert_close(out[:K, :], ref, dtype)


@pytest.mark.spmm_bsr
@pytest.mark.parametrize("op", ["non", "trans"], ids=["non", "trans"])
def test_spmm_bsr_accepts_padded_B(op):
    device = torch.device("cuda")
    M, K, N = 7, 5, 4
    block_dim = 4
    dtype = torch.float32
    data, indices, indptr, dense = _random_bsr_mk(
        M, K, dtype, torch.int32, block_dim, device
    )
    logical_b_rows = _logical_b_rows(M, K, op)
    B = torch.zeros((_padded_b_rows(M, K, block_dim, op), N), dtype=dtype, device=device)
    B[:logical_b_rows, :] = _random_values((logical_b_rows, N), dtype, device)
    ref = _dense_reference(dense, B[:logical_b_rows, :], dtype, op)
    out = flagsparse_spmm_bsr(data, indices, indptr, B, shape=(M, K), block_dim=block_dim, op=op)
    assert out.shape == (_padded_out_rows(M, K, block_dim, op), N)
    _assert_close(out[: _logical_out_rows(M, K, op), :], ref, dtype)


@pytest.mark.spmm_bsr
@pytest.mark.parametrize("op", ["non", "trans"], ids=["non", "trans"])
def test_spmm_bsr_B_length_mismatch_rejected(op):
    device = torch.device("cuda")
    M, K, N = 8, 12, 4
    data, indices, indptr, _dense = _random_bsr_mk(
        M, K, torch.float32, torch.int32, 2, device
    )
    good_rows = _logical_b_rows(M, K, op)
    B = torch.randn((good_rows - 1, N), dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="B.shape\\[0\\] must be"):
        flagsparse_spmm_bsr(data, indices, indptr, B, shape=(8, 12), block_dim=2, op=op)


@pytest.mark.spmm_bsr
@pytest.mark.parametrize("op", ["non", "trans"], ids=["non", "trans"])
def test_spmm_bsr_int64_auto_fallback_to_int32(monkeypatch, op):
    device = torch.device("cuda")
    M, K, N = 7, 5, 4
    dtype = torch.float32
    block_dim = 2
    data, indices, indptr, dense = _random_bsr_mk(
        M, K, dtype, torch.int64, block_dim, device
    )
    B = _random_values((_logical_b_rows(M, K, op), N), dtype, device)
    ref = _dense_reference(dense, B, dtype, op)
    state = {"forced_once": False}
    original = spmm_bsr_mod._triton_spmm_bsr_base_kernel

    def fail_int64_once(prepared, B_in, op_code=None):
        if prepared.kernel_indices.dtype == torch.int64 and not state["forced_once"]:
            state["forced_once"] = True
            raise RuntimeError("forced int64 launch failure")
        return original(prepared, B_in, op_code)

    monkeypatch.setattr(spmm_bsr_mod, "_triton_spmm_bsr_base_kernel", fail_int64_once)
    out = flagsparse_spmm_bsr(
        data,
        indices,
        indptr,
        B,
        shape=(M, K),
        block_dim=block_dim,
        op=op,
        index_fallback_policy="auto",
    )
    assert state["forced_once"] is True
    _assert_close(out[: _logical_out_rows(M, K, op), :], ref, dtype)
