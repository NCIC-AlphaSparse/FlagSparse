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

"""CPU simulation of the Ascend (torch_npu) paths of the q4 operators.

Ascend's Triton lacks what the new kernels use, so every new route has a torch path
there. It cannot run on the CI box, so these tests force the Ascend branch and feed
CPU tensors: the same code, checked against a float64 / complex128 dense product.
"""

import importlib

import pytest

torch = pytest.importorskip("torch", reason="requires torch to exercise dispatch")

vector_ops = importlib.import_module("flagsparse.sparse_operations.vector_ops")
spmv_sell = importlib.import_module("flagsparse.sparse_operations.spmv_sell")
mixed = importlib.import_module("flagsparse.sparse_operations.mixed_spmx")


@pytest.fixture
def ascend(monkeypatch):
    for mod in (vector_ops, spmv_sell, mixed):
        monkeypatch.setattr(mod, "_is_ascend_runtime", lambda: True)
        monkeypatch.setattr(mod, "_is_accel_tensor", lambda t: t.device.type == "cpu")


def _rand(shape, dtype, gen):
    if dtype == torch.int8:
        return torch.randint(-4, 5, shape, dtype=torch.int8, generator=gen)
    if dtype.is_complex:
        return torch.complex(
            torch.randn(shape, generator=gen), torch.randn(shape, generator=gen)
        )
    return torch.randn(shape, generator=gen).to(dtype)


def _sparse(m, k, dtype, gen):
    mask = torch.rand((m, k), generator=gen) < 0.2
    mask[0, 0] = True
    mask[m // 2] = False
    return torch.where(mask, _rand((m, k), dtype, gen), torch.zeros((), dtype=dtype))


def _wide(t):
    return t.to(torch.complex128 if t.is_complex() else torch.float64)


def _close(got, ref, tol):
    scale = max(1.0, float(ref.abs().max()))
    assert float((_wide(got) - ref).abs().max()) <= tol * scale


@pytest.mark.parametrize(
    "dtype, op, want",
    [
        (torch.float16, "non", torch.float32),
        (torch.int8, "non", torch.int32),
        (torch.complex64, "conj", torch.complex64),
    ],
)
def test_spvv_ascend_path(ascend, dtype, op, want):
    gen = torch.Generator().manual_seed(0)
    idx = torch.randperm(200, generator=gen)[:60].to(torch.int32)
    x, y = _rand((60,), dtype, gen), _rand((200,), dtype, gen)
    got = vector_ops.flagsparse_spvv(x, idx, y, op=op)
    xr = _wide(x).conj() if op == "conj" else _wide(x)
    assert got.dtype == want
    _close(got.reshape(1), (xr * _wide(y)[idx.long()]).sum().reshape(1), 1e-3)


def test_axpby_ascend_path(ascend):
    gen = torch.Generator().manual_seed(1)
    idx = torch.randperm(200, generator=gen)[:60].to(torch.int32)
    x, y = _rand((60,), torch.float16, gen), _rand((200,), torch.float16, gen)
    ref = _wide(y) * 0.25
    ref[idx.long()] += 2.0 * _wide(x)
    vector_ops.flagsparse_axpby(x, idx, y, alpha=2.0, beta=0.25)
    _close(y, ref, 2e-3)


@pytest.mark.parametrize(
    "dtype, out_dtype",
    [
        (torch.float32, None),
        (torch.float16, None),
        (torch.complex64, None),
        (torch.int8, torch.int32),
    ],
)
def test_spmv_sell_ascend_path(ascend, dtype, out_dtype):
    gen = torch.Generator().manual_seed(2)
    A, x = _sparse(37, 29, dtype, gen), _rand((29,), dtype, gen)
    vals, cols, offsets = spmv_sell.dense_to_sell(A, 4)
    y = spmv_sell.flagsparse_spmv_sell(
        vals, cols, offsets, x, (37, 29), slice_size=4, out_dtype=out_dtype
    )
    _close(y, _wide(A) @ _wide(x), 2e-3)


@pytest.mark.parametrize(
    "dtype, x_dtype, out_dtype, want",
    [
        (torch.float16, torch.float16, torch.float32, torch.float32),
        (torch.int8, torch.int8, None, torch.int32),
        (torch.int8, torch.int8, torch.float32, torch.float32),
        (torch.float32, torch.complex64, None, torch.complex64),
    ],
)
def test_mixed_csr_spmv_ascend_path(ascend, dtype, x_dtype, out_dtype, want):
    gen = torch.Generator().manual_seed(3)
    A, x = _sparse(41, 23, dtype, gen), _rand((23,), x_dtype, gen)
    s = A.to_sparse_csr()
    y = mixed.spmv_csr_mixed(
        s.values(), s.col_indices(), s.crow_indices(), x, (41, 23), out_dtype=out_dtype
    )
    assert y.dtype == want
    _close(y, _wide(A).to(_wide(x).dtype) @ _wide(x), 1e-3)


@pytest.mark.parametrize(
    "dtype, out_dtype", [(torch.float16, None), (torch.int8, torch.int32)]
)
def test_mixed_coo_spmv_ascend_path(ascend, dtype, out_dtype):
    gen = torch.Generator().manual_seed(4)
    A, x = _sparse(41, 23, dtype, gen), _rand((23,), dtype, gen)
    s = A.to_sparse_coo().coalesce()
    r, c = s.indices()
    y = mixed.spmv_coo_mixed(s.values(), r, c, x, (41, 23), out_dtype=out_dtype)
    _close(y, _wide(A) @ _wide(x), 2e-3)


@pytest.mark.parametrize(
    "dtype, out_dtype", [(torch.float16, torch.float32), (torch.int8, torch.int32)]
)
def test_mixed_csr_spmm_ascend_path(ascend, dtype, out_dtype):
    gen = torch.Generator().manual_seed(5)
    A, B = _sparse(41, 23, dtype, gen), _rand((13, 23), dtype, gen)
    s = A.to_sparse_csr()
    C = mixed.spmm_csr_mixed(
        s.values(),
        s.col_indices(),
        s.crow_indices(),
        B.t(),
        (41, 23),
        out_dtype=out_dtype,
    )
    assert C.dtype == out_dtype
    _close(C, _wide(A) @ _wide(B).t(), 1e-3)


def test_mixed_rejects_combinations_cusparse_lacks():
    assert mixed.spmv_needs_mixed(torch.float32, torch.float32, None, None) is False
    assert mixed.spmv_needs_mixed(torch.float16, torch.float16, None, None) is False
    assert (
        mixed.spmv_needs_mixed(torch.float16, torch.float16, None, None, coo=True)
        is True
    )
    with pytest.raises(TypeError):
        mixed._resolve_types(torch.float32, torch.float32, None, torch.float16, True)
    with pytest.raises(TypeError):
        mixed._resolve_types(torch.int8, torch.int8, None, torch.float16, False)
    with pytest.raises(TypeError):
        mixed._resolve_types(torch.float32, torch.complex64, None, None, False)
