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

"""prepare_spmv_csr_mixed / spmv_csr_mixed_prepared against a CPU golden result.

Every route the plan can pick (cached, plain, the short/long bucket order and the
nnz-balanced segmented scan) on structures that stress it: empty rows, one long
row among short ones (what sends ``auto`` to ``nnz``), 0 and 1 rows, int64
indices, a strided ``out``.
"""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import mixed_spmx
from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
)

pytestmark = pytest.mark.skipif(not accelerator_available(), reason=ACCELERATOR_REQUIRED)

# Routes beyond cached/plain exist only where those kernels are wired.
_EXTRA_ROUTES = mixed_spmx._is_iluvatar_runtime() or mixed_spmx._CUDA_DIRECT_LAUNCH


def _lengths(kind, n_rows):
    gen = torch.Generator().manual_seed(n_rows * 31 + len(kind))
    if kind == "short":
        return torch.randint(0, 5, (n_rows,), generator=gen)
    if kind == "long_tail":  # ASIC_680ks-like: mean <= 4, one row > 128
        lengths = torch.randint(1, 4, (n_rows,), generator=gen)
        lengths[n_rows // 3] = 300
        lengths[n_rows // 2] = 0
        return lengths
    return torch.randint(8, 24, (n_rows,), generator=gen)


def _csr(kind, n_rows, n_cols, dtype):
    gen = torch.Generator().manual_seed(n_rows + n_cols)
    lengths = _lengths(kind, n_rows).clamp(max=n_cols) if n_rows else torch.zeros(0, dtype=torch.long)
    indptr = torch.zeros(n_rows + 1, dtype=torch.long)
    indptr[1:] = torch.cumsum(lengths, 0)
    nnz = int(indptr[-1])
    cols = torch.cat(
        [torch.randperm(n_cols, generator=gen)[: int(n)].sort().values for n in lengths]
    ) if nnz else torch.zeros(0, dtype=torch.long)
    if dtype == torch.int8:
        data = torch.randint(-3, 4, (nnz,), generator=gen).to(dtype)
    else:
        data = torch.randn(nnz, generator=gen).to(dtype)
    return data, cols, indptr


def _vector(n, dtype):
    gen = torch.Generator().manual_seed(n + 7)
    if dtype == torch.int8:
        return torch.randint(-3, 4, (n,), generator=gen).to(dtype)
    if dtype.is_complex:
        return torch.complex(torch.randn(n, generator=gen), torch.randn(n, generator=gen)).to(dtype)
    return torch.randn(n, generator=gen).to(dtype)


def _golden(data, cols, indptr, x, n_rows):
    wide = torch.complex128 if x.is_complex() else torch.float64
    rows = torch.repeat_interleave(torch.arange(n_rows), indptr[1:] - indptr[:-1])
    y = torch.zeros(n_rows, dtype=wide)
    y.index_add_(0, rows, data.to(wide) * x.to(wide)[cols])
    return y


def _check(y, ref):
    got = y.cpu().to(ref.dtype)
    scale = max(1.0, float(ref.abs().max())) if ref.numel() else 1.0
    assert float((got - ref).abs().max() if ref.numel() else 0.0) <= 1e-4 * scale


ROUTES = ["auto", "cached", "plain", "bucket", "nnz"]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("kind", ["short", "long_tail", "medium"])
@pytest.mark.parametrize("n_rows", [0, 1, 257])
def test_real_by_complex_routes_match_golden(route, kind, n_rows):
    if route in ("bucket", "nnz") and not _EXTRA_ROUTES:
        pytest.skip("bucket/nnz routes are wired on CUDA and Iluvatar only")
    if route != "auto" and mixed_spmx._prepared_mixed_uses_call(torch.float32, torch.complex64):
        pytest.skip("this backend keeps its own mixed CSR call route")
    dev = accelerator_device()
    n_cols = 400
    data, cols, indptr = _csr(kind, n_rows, n_cols, torch.float32)
    x = _vector(n_cols, torch.complex64)
    ref = _golden(data, cols, indptr, x, n_rows)
    prepared = fs.prepare_spmv_csr_mixed(
        data.to(dev), cols.to(dev), indptr.to(dev), torch.complex64, (n_rows, n_cols),
        route=route,
    )
    if route == "auto" and kind == "long_tail" and n_rows > 1 and _EXTRA_ROUTES:
        assert prepared.route == "nnz"
    xd = x.to(dev)
    _check(fs.flagsparse_spmv_csr(prepared=prepared, x=xd), ref)
    # Repeated runs into the same out reuse every cached launcher.
    out = torch.full((n_rows,), 7 + 7j, dtype=torch.complex64, device=dev)
    for _ in range(2):
        assert fs.flagsparse_spmv_csr(prepared=prepared, x=xd, out=out) is out
        _check(out, ref)
    buf = torch.full((2 * n_rows,), 7 + 7j, dtype=torch.complex64, device=dev)
    strided = buf[::2]
    assert fs.spmv_csr_mixed_prepared(prepared, xd, out=strided) is strided
    _check(strided, ref)
    if n_rows:
        assert torch.all(buf[1::2].cpu() == 7 + 7j)


@pytest.mark.parametrize(
    "value,x_dtype,out_dtype",
    [
        (torch.float16, torch.float16, torch.float32),
        (torch.int8, torch.int8, torch.int32),
        (torch.int8, torch.int8, torch.float32),
        (torch.float64, torch.complex128, None),
    ],
)
def test_other_mixed_combinations_match_golden(value, x_dtype, out_dtype):
    dev = accelerator_device()
    data, cols, indptr = _csr("medium", 129, 200, value)
    x = _vector(200, x_dtype)
    ref = _golden(data, cols, indptr, x, 129)
    prepared = fs.prepare_spmv_csr_mixed(
        data.to(dev), cols.to(torch.int64).to(dev), indptr.to(dev), x_dtype, (129, 200),
        out_dtype=out_dtype,
    )
    y = fs.flagsparse_spmv_csr(prepared=prepared, x=x.to(dev))
    expected = out_dtype or (x_dtype if x_dtype.is_complex else value)
    assert y.dtype == expected
    tol_ref = ref.to(torch.float64 if not ref.is_complex() else ref.dtype)
    got = y.cpu().to(tol_ref.dtype)
    tol = 2e-2 if value == torch.float16 else 1e-4
    assert float((got - tol_ref).abs().max()) <= tol * max(1.0, float(tol_ref.abs().max()))


def test_prepared_mixed_rejects_misuse():
    dev = accelerator_device()
    data, cols, indptr = _csr("short", 16, 16, torch.float32)
    prepared = fs.prepare_spmv_csr_mixed(
        data.to(dev), cols.to(dev), indptr.to(dev), torch.complex64, (16, 16)
    )
    x = _vector(16, torch.complex64).to(dev)
    with pytest.raises(TypeError):
        fs.flagsparse_spmv_csr(prepared=prepared, x=x.real.contiguous())
    with pytest.raises(ValueError):
        fs.flagsparse_spmv_csr(prepared=prepared, x=x[:8])
    with pytest.raises(ValueError):
        fs.flagsparse_spmv_csr(prepared=prepared, x=x, out=torch.empty(15, dtype=torch.complex64, device=dev))
    with pytest.raises(ValueError):
        fs.flagsparse_spmv_csr(data.to(dev), prepared=prepared, x=x)
    with pytest.raises(ValueError):
        fs.flagsparse_spmv_csr(prepared=prepared, x=x, alg="auto")
    with pytest.raises(ValueError):
        fs.prepare_spmv_csr_mixed(
            data.to(dev), cols.to(dev), indptr.to(dev), torch.complex64, (16, 16), route="bogus"
        )
