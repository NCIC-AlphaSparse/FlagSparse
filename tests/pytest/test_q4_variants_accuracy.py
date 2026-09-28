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

"""The first 42 variants of docs/NEW_OPERATORS_CUSPARSE_12_5.md, one case each.

Every case runs the public operator on the accelerator and compares it with a CPU
float64 / complex128 golden (integer outputs exactly). Case ids are the variant
names from the list; each case carries its operator's marker, so the unified
runner's ``-m <op>`` accuracy pass picks it up.

Variant names follow cuSPARSE: ``<op>_<fmt>_<type>_int_<opA>[_<opB>][_<layout>]``,
``f16f32`` = float16 in, float32 out; ``i8i32`` = int8 in, int32 out; ``f32c32`` =
float32 matrix times a complex64 vector. For SDDMM, opA/opB are cuSPARSE's (A is
M x K, B is K x N): the legacy ``flagsparse_sddmm_csr`` call is cuSPARSE opB=TRANS.
"""

import zlib

import pytest
import torch

import flagsparse as fs
from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
)

pytestmark = pytest.mark.skipif(not accelerator_available(), reason=ACCELERATOR_REQUIRED)

SHAPES = [(64, 48, 16), (257, 129, 33)]
_TOL = {
    torch.float16: 2e-3,
    torch.bfloat16: 1e-2,
    torch.float32: 1e-4,
    torch.complex64: 1e-4,
    torch.float64: 1e-10,
    torch.complex128: 1e-10,
}


def _rand(shape, dtype, gen):
    if dtype == torch.int8:
        return torch.randint(-4, 5, shape, dtype=torch.int8, generator=gen)
    if dtype.is_complex:
        real = torch.float32 if dtype == torch.complex64 else torch.float64
        return torch.complex(
            torch.randn(shape, dtype=real, generator=gen),
            torch.randn(shape, dtype=real, generator=gen),
        )
    return torch.randn(shape, generator=gen).to(dtype)


def _sparse(m, k, dtype, gen, density=0.15):
    mask = torch.rand((m, k), generator=gen) < density
    mask[0, 0] = True
    if m > 2:
        mask[m // 2] = False  # an empty row
    return torch.where(mask, _rand((m, k), dtype, gen), torch.zeros((), dtype=dtype))


def _wide(t):
    return t.to(torch.complex128 if t.is_complex() else torch.float64)


def _op(dense, op):
    return dense if op == "non" else (dense.t() if op == "trans" else dense.conj().t())


def _dev(*tensors):
    dev = accelerator_device()
    return [t.to(dev) for t in tensors]


def _csr(dense):
    s = dense.to_sparse_csr()
    return _dev(s.values(), s.col_indices().to(torch.int32), s.crow_indices().to(torch.int32))


def _coo(dense):
    s = dense.to_sparse_coo().coalesce()
    r, c = s.indices()
    return _dev(s.values(), r.to(torch.int32), c.to(torch.int32))


def _csc(dense):
    s = dense.to_sparse_csc()
    return _dev(s.values(), s.row_indices().to(torch.int32), s.ccol_indices().to(torch.int32))


def _check(result, reference, dtype):
    assert result.dtype == dtype, f"result dtype {result.dtype}, expected {dtype}"
    got = result.detach().cpu()
    if not (dtype.is_floating_point or dtype.is_complex):
        assert torch.equal(got.to(torch.int64), reference.round().to(torch.int64))
        return
    scale = max(1.0, float(reference.abs().max())) if reference.numel() else 1.0
    err = float((_wide(got) - reference).abs().max()) if reference.numel() else 0.0
    assert err <= _TOL[dtype] * scale, f"max error {err:.3g} (scale {scale:.3g})"


# --------------------------------------------------------------------------- cases
def gather(dtype, shape, gen):
    n = shape[0] * 4
    a, idx = _rand((n,), dtype, gen), torch.randint(0, n, (shape[1],), dtype=torch.int32, generator=gen)
    a_d, idx_d = _dev(a, idx)
    return fs.flagsparse_gather(a_d, idx_d), a[idx.long()].to(torch.float64), dtype


def scatter(dtype, shape, gen):
    n = shape[0] * 4
    idx = torch.randperm(n, generator=gen)[: shape[1]].to(torch.int32)
    vals = _rand((shape[1],), dtype, gen)
    dense_d, idx_d, vals_d = _dev(torch.zeros(n, dtype=dtype), idx, vals)
    fs.flagsparse_scatter(dense_d, idx_d, vals_d)
    ref = torch.zeros(n, dtype=torch.float64)
    ref[idx.long()] = vals.to(torch.float64)
    return dense_d, ref, dtype


def axpby(dtype, shape, gen):
    n = shape[0] * 4
    idx = torch.randperm(n, generator=gen)[: shape[1]].to(torch.int32)
    x, y = _rand((shape[1],), dtype, gen), _rand((n,), dtype, gen)
    x_d, idx_d, y_d = _dev(x, idx, y)
    fs.flagsparse_axpby(x_d, idx_d, y_d, alpha=1.5, beta=-0.5)
    ref = _wide(y) * -0.5
    ref[idx.long()] += 1.5 * _wide(x)
    return y_d, ref, dtype


def spvv(dtype, out_dtype, op):
    def build(_unused, shape, gen):
        n = shape[0] * 4
        idx = torch.randperm(n, generator=gen)[: shape[1]].to(torch.int32)
        x, y = _rand((shape[1],), dtype, gen), _rand((n,), dtype, gen)
        result = fs.flagsparse_spvv(*_dev(x, idx, y), op=op)
        xr = _wide(x).conj() if op == "conj" else _wide(x)
        return result.reshape(1), (xr * _wide(y)[idx.long()]).sum().reshape(1), out_dtype
    return build


def spmv_sell(dtype, out_dtype):
    def build(_unused, shape, gen):
        m, k, _ = shape
        A, x = _sparse(m, k, dtype, gen), _rand((k,), dtype, gen)
        vals, cols, offsets = fs.dense_to_sell(A, 8)
        y = fs.flagsparse_spmv_sell(
            *_dev(vals, cols, offsets, x), (m, k), slice_size=8,
            out_dtype=None if out_dtype == _default_out(dtype) else out_dtype,
        )
        return y, _wide(A) @ _wide(x), out_dtype
    return build


def _default_out(dtype):
    return torch.int32 if dtype == torch.int8 else dtype


def spmv_csr(dtype, op="non", out_dtype=None, x_dtype=None):
    def build(_unused, shape, gen):
        m, k, _ = shape
        A = _sparse(m, k, dtype, gen)
        xd = x_dtype or dtype
        x = _rand((k if op == "non" else m,), xd, gen)
        kw = {} if out_dtype is None else {"out_dtype": out_dtype}
        y = fs.flagsparse_spmv_csr(*_csr(A), *_dev(x), shape=(m, k), op=op, **kw)
        ref = _op(_wide(A).to(_wide(x).dtype), op) @ _wide(x)
        return y, ref, out_dtype or xd
    return build


def spmv_coo(dtype, op="non", out_dtype=None):
    def build(_unused, shape, gen):
        m, k, _ = shape
        A = _sparse(m, k, dtype, gen)
        x = _rand((k if op == "non" else m,), dtype, gen)
        kw = {} if out_dtype is None else {"out_dtype": out_dtype}
        y = fs.flagsparse_spmv_coo(*_coo(A), *_dev(x), shape=(m, k), op=op, **kw)
        return y, _op(_wide(A), op) @ _wide(x), out_dtype or dtype
    return build


def spmv_csc(dtype):
    def build(_unused, shape, gen):
        m, k, _ = shape
        A, x = _sparse(m, k, dtype, gen), _rand((k,), dtype, gen)
        y = fs.flagsparse_spmv_csc(*_csc(A), *_dev(x), shape=(m, k))
        return y, _wide(A) @ _wide(x), dtype
    return build


def spmm_csr(dtype, op_a="non", op_b="non", layout="row", out_dtype=None):
    def build(_unused, shape, gen):
        m, k, n = shape
        A = _sparse(m, k, dtype, gen) if op_a == "non" else _sparse(k, m, dtype, gen)
        B = _rand((k, n), dtype, gen) if op_b == "non" else _rand((n, k), dtype, gen)
        (B_d,) = _dev(B)
        if layout == "col":
            B_d = B_d.t().contiguous().t()
        kw = {"op_b": op_b} if op_b != "non" else {}
        if out_dtype is not None:
            kw["out_dtype"] = out_dtype
        C = fs.flagsparse_spmm_csr(*_csr(A), B_d, tuple(A.shape), op=op_a, **kw)
        ref = _op(_wide(A), op_a) @ (_wide(B) if op_b == "non" else _wide(B).t())
        return C, ref, out_dtype or dtype
    return build


def spmm_coo(dtype):
    def build(_unused, shape, gen):
        m, k, n = shape
        A, B = _sparse(m, k, dtype, gen), _rand((k, n), dtype, gen)
        C = fs.flagsparse_spmm_coo(*_coo(A), *_dev(B), (m, k))
        return C, _wide(A) @ _wide(B), dtype
    return build


def spmm_csc(dtype):
    def build(_unused, shape, gen):
        m, k, n = shape
        A, B = _sparse(m, k, dtype, gen), _rand((k, n), dtype, gen)
        C = fs.flagsparse_spmm_csc(*_csc(A), *_dev(B), (m, k))
        return C, _wide(A) @ _wide(B), dtype
    return build


def _padded(m, k, dtype, gen, bd=4):
    return _sparse((m + bd - 1) // bd * bd, (k + bd - 1) // bd * bd, dtype, gen, 0.3)


def spmm_bsr(dtype, bd=4):
    def build(_unused, shape, gen):
        A = _padded(shape[0], shape[1], dtype, gen, bd)
        B = _rand((A.shape[1], shape[2]), dtype, gen)
        s = A.to_sparse_bsr((bd, bd))
        C = fs.flagsparse_spmm_bsr(
            *_dev(s.values(), s.col_indices().to(torch.int32), s.crow_indices().to(torch.int32), B),
            shape=tuple(A.shape), block_dim=bd,
        )
        return C, _wide(A) @ _wide(B), dtype
    return build


def spmm_bell(dtype, bd=4):
    def build(_unused, shape, gen):
        A = _padded(shape[0], shape[1], dtype, gen, bd)
        B = _rand((A.shape[1], shape[2]), dtype, gen)
        s = A.to_sparse_bsr((bd, bd))
        crow, col, val = s.crow_indices(), s.col_indices(), s.values()
        mb = A.shape[0] // bd
        width = max(1, int((crow[1:] - crow[:-1]).max()))
        data = torch.zeros(mb, width, bd, bd, dtype=dtype)
        idx = torch.full((mb, width), -1, dtype=torch.int32)
        for b in range(mb):
            for j, e in enumerate(range(int(crow[b]), int(crow[b + 1]))):
                data[b, j], idx[b, j] = val[e], col[e]
        C = fs.flagsparse_spmm_bell(*_dev(data, idx, B), shape=tuple(A.shape), block_dim=bd, op="non")
        return C, _wide(A) @ _wide(B), dtype
    return build


def sddmm(op_a, op_b, layout):
    def build(_unused, shape, gen):
        m, n, k = shape
        S = _sparse(m, n, torch.float32, gen, 0.2)
        A = _rand((m, k) if op_a == "non" else (k, m), torch.float32, gen)
        B = _rand((k, n) if op_b == "non" else (n, k), torch.float32, gen)
        A_d, B_d = _dev(A, B)
        if layout == "col":
            A_d, B_d = A_d.t().contiguous().t(), B_d.t().contiguous().t()
        vals = fs.flagsparse_sddmm_csr(
            None, *_csr(S)[1:], A_d, B_d, shape=(m, n), op_a=op_a, op_b=op_b, dense_layout=layout
        )
        full = _op(_wide(A), op_a) @ _op(_wide(B), op_b)
        pattern = S.to_sparse_csr()
        rows = torch.repeat_interleave(torch.arange(m), pattern.crow_indices().diff())
        return vals, full[rows, pattern.col_indices()], torch.float32
    return build


f16, f32, c32, i8, i32 = torch.float16, torch.float32, torch.complex64, torch.int8, torch.int32

# (variant, marker, builder, value dtype passed to generic builders)
VARIANTS = [
    ("gather_i8_int", "gather", gather, i8),
    ("scatter_i8_int", "scatter", scatter, i8),
    ("axpby_f16_int", "axpby", axpby, f16),
    ("spmv_sell_f32_int_non", "spmv_sell", spmv_sell(f32, f32), None),
    ("spmv_csr_f16f32_int_non", "spmv_csr", spmv_csr(f16, out_dtype=f32), None),
    ("spvv_f16f32_int_non", "spvv", spvv(f16, f32, "non"), None),
    ("spmv_csr_f16_int_non", "spmv_csr", spmv_csr(f16), None),
    ("spmv_csr_f32c32_int_non", "spmv_csr", spmv_csr(f32, x_dtype=c32), None),
    ("spmv_sell_f16_int_non", "spmv_sell", spmv_sell(f16, f16), None),
    ("spmm_csr_f16f32_int_non_non_row", "spmm_csr", spmm_csr(f16, out_dtype=f32), None),
    ("spmm_csr_f16_int_non_non_row", "spmm_csr", spmm_csr(f16), None),
    ("spmm_csr_f32_int_non_non_col", "spmm_csr", spmm_csr(f32, layout="col"), None),
    ("spmm_csr_f32_int_non_trans_row", "spmm_csr", spmm_csr(f32, op_b="trans"), None),
    ("spmv_csc_f32_int_non", "spmv_csc", spmv_csc(f32), None),
    ("spmv_csr_c32_int_non", "spmv_csr", spmv_csr(c32), None),
    ("spmv_sell_c32_int_non", "spmv_sell", spmv_sell(c32, c32), None),
    ("spvv_c32_int_conj", "spvv", spvv(c32, c32, "conj"), None),
    ("spmv_coo_f16f32_int_non", "spmv_coo", spmv_coo(f16, out_dtype=f32), None),
    ("spmm_csr_c32_int_non_non_row", "spmm_csr", spmm_csr(c32), None),
    ("spmv_coo_f32_int_trans", "spmv_coo", spmv_coo(f32, "trans"), None),
    ("spmv_csr_f32_int_trans", "spmv_csr", spmv_csr(f32, "trans"), None),
    ("spmv_csr_i8f32_int_non", "spmv_csr", spmv_csr(i8, out_dtype=f32), None),
    ("spmv_csr_i8i32_int_non", "spmv_csr", spmv_csr(i8, out_dtype=i32), None),
    ("spmv_sell_i8i32_int_non", "spmv_sell", spmv_sell(i8, i32), None),
    ("spvv_i8i32_int_non", "spvv", spvv(i8, i32, "non"), None),
    ("spmm_csc_f32_int_non_non_row", "spmm_csc", spmm_csc(f32), None),
    ("spmm_csr_i8i32_int_non_non_row", "spmm_csr", spmm_csr(i8, out_dtype=i32), None),
    ("spmv_coo_c32_int_non", "spmv_coo", spmv_coo(c32), None),
    ("spmv_coo_f16_int_non", "spmv_coo", spmv_coo(f16), None),
    ("spmv_csc_c32_int_non", "spmv_csc", spmv_csc(c32), None),
    ("spmv_csc_f16_int_non", "spmv_csc", spmv_csc(f16), None),
    ("sddmm_csr_f32_int_non_non_col", "sddmm_csr", sddmm("non", "non", "col"), None),
    ("sddmm_csr_f32_int_non_trans_row", "sddmm_csr", sddmm("non", "trans", "row"), None),
    ("sddmm_csr_f32_int_trans_non_row", "sddmm_csr", sddmm("trans", "non", "row"), None),
    ("spmm_bell_f32_int_non_non_row", "spmm_bell", spmm_bell(f32), None),
    ("spmm_bsr_f32_int_non_non_row", "spmm_bsr", spmm_bsr(f32), None),
    ("spmm_coo_c32_int_non_non_row", "spmm_coo", spmm_coo(c32), None),
    ("spmm_coo_f16_int_non_non_row", "spmm_coo", spmm_coo(f16), None),
    ("spmm_csr_f32_int_trans_non_row", "spmm_csr", spmm_csr(f32, op_a="trans"), None),
    ("spmv_coo_c32_int_conj", "spmv_coo", spmv_coo(c32, "conj"), None),
    ("spmv_coo_i8i32_int_non", "spmv_coo", spmv_coo(i8, out_dtype=i32), None),
    ("spmv_csr_c32_int_conj", "spmv_csr", spmv_csr(c32, "conj"), None),
]


def test_the_list_has_42_distinct_variants():
    names = [v[0] for v in VARIANTS]
    assert len(names) == 42 and len(set(names)) == 42


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize(
    "name, builder, dtype",
    [
        pytest.param(name, builder, dtype, id=name, marks=getattr(pytest.mark, marker))
        for name, marker, builder, dtype in VARIANTS
    ],
)
def test_q4_variant_matches_cpu_golden(name, builder, dtype, shape):
    # crc32, not hash(): str hashing is salted per process.
    gen = torch.Generator().manual_seed(zlib.crc32(name.encode()))
    result, reference, expected_dtype = builder(dtype, shape, gen)
    _check(result, reference, expected_dtype)
