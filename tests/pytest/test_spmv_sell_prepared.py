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

"""prepare_spmv_sell (SELL-C-sigma re-encoding) against a CPU golden result.

Covers every SELL value/output dtype, structures that stress the re-encoding
(empty rows, one long row among short ones, row counts that are not a multiple
of the slice height), the layout options, value refresh after an in-place write,
rebuild after a structure change, and an inf in ``x`` that only padding touches.
"""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import spmv_sell
from flagsparse.sparse_operations._common import _is_ascend_runtime
from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
)

pytestmark = pytest.mark.skipif(not accelerator_available(), reason=ACCELERATOR_REQUIRED)

_SIGMA_OK = not spmv_sell._is_ascend_runtime()
# BI-V150's fp64 path is unreliable (H2D, strided copy_ rewrote the gaps of a
# strided double/complex128 out in 16 cases, 2026-10-25); its delivery skips fp64.
_NO_FP64 = spmv_sell._is_iluvatar_runtime()

DTYPES = [
    (torch.float32, None),
    (torch.float16, None),
    (torch.float16, torch.float32),
    (torch.bfloat16, torch.float32),
    (torch.int8, None),
    (torch.int8, torch.float32),
    (torch.complex64, None),
    (torch.float64, None),
    (torch.complex128, None),
]
_TOL = {
    torch.float16: 2e-2,
    torch.bfloat16: 2e-2,
    torch.float32: 1e-5,
    torch.float64: 1e-12,
    torch.complex64: 1e-5,
    torch.complex128: 1e-12,
}


def _lengths(kind, n_rows, n_cols, gen):
    if kind == "short":
        lengths = torch.randint(0, 5, (n_rows,), generator=gen)
    elif kind == "long_tail":  # ASIC_680ks-like: mostly 1-3, a 300-long row, empty rows
        lengths = torch.randint(1, 4, (n_rows,), generator=gen)
        if n_rows:
            lengths[n_rows // 3] = 300
            lengths[n_rows // 2] = 0
            lengths[-1] = 0
    else:
        lengths = torch.randint(8, 24, (n_rows,), generator=gen)
    return lengths.clamp(max=n_cols)


def _csr(kind, n_rows, n_cols, dtype, seed=0):
    gen = torch.Generator().manual_seed(seed + 31 * n_rows + len(kind))
    lengths = _lengths(kind, n_rows, n_cols, gen)
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64)
    indptr[1:] = torch.cumsum(lengths, 0)
    cols = [torch.randperm(n_cols, generator=gen)[: int(k)].sort().values for k in lengths]
    indices = torch.cat(cols) if cols and int(indptr[-1]) else torch.zeros(0, dtype=torch.int64)
    data = _random(int(indptr[-1]), dtype, gen)
    return data, indices, indptr


def _random(n, dtype, gen):
    if dtype == torch.int8:
        return torch.randint(-3, 4, (n,), generator=gen).to(dtype)
    if dtype.is_complex:
        parts = torch.randn(n, 2, generator=gen, dtype=torch.float64)
        return torch.complex(parts[:, 0], parts[:, 1]).to(dtype)
    return (0.5 * torch.randn(n, generator=gen)).to(dtype)


def _golden(data, indices, indptr, x, n_rows):
    wide = torch.complex128 if data.is_complex() else torch.float64
    rows = torch.repeat_interleave(torch.arange(n_rows), indptr[1:] - indptr[:-1])
    y = torch.zeros(n_rows, dtype=wide)
    y.index_add_(0, rows, data.to(wide) * x.to(wide)[indices])
    return y


def _sell(data, indices, indptr, n_rows, dev, slice_size=32):
    return fs.csr_to_sell(
        data.to(dev), indices.to(torch.int32).to(dev), indptr.to(torch.int32).to(dev),
        n_rows, slice_size,
    )


def _check(y, ref, out_dtype):
    got = y.cpu().to(ref.dtype)
    if not (out_dtype.is_floating_point or out_dtype.is_complex):
        assert torch.equal(got, ref)
        return
    scale = max(1.0, float(ref.abs().max())) if ref.numel() else 1.0
    err = float((got - ref).abs().max()) if ref.numel() else 0.0
    assert err <= _TOL[out_dtype] * scale, err


def _out_dtype(value, out):
    return out or {torch.int8: torch.int32}.get(value, value)


@pytest.mark.skipif(not _SIGMA_OK, reason="SELL-C-sigma needs Triton")
@pytest.mark.parametrize("value,out", DTYPES, ids=lambda d: str(d).replace("torch.", ""))
@pytest.mark.parametrize("kind", ["short", "long_tail", "medium"])
@pytest.mark.parametrize("n_rows", [1, 70, 257])
def test_prepared_sigma_matches_golden(value, out, kind, n_rows):
    if _NO_FP64 and value in (torch.float64, torch.complex128):
        pytest.skip("fp64 is not supported on BI-V150 (known platform issue)")
    dev = accelerator_device()
    n_cols = 400
    data, indices, indptr = _csr(kind, n_rows, n_cols, value)
    x = _random(n_cols, value, torch.Generator().manual_seed(7))
    ref = _golden(data, indices, indptr, x, n_rows)
    sv, sc, so = _sell(data, indices, indptr, n_rows, dev)
    p = fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, out_dtype=out, route="sigma")
    assert p.route == "sigma" and p.out_dtype == _out_dtype(value, out)
    xd = x.to(dev)
    _check(fs.flagsparse_spmv_sell(prepared=p, x=xd), ref, p.out_dtype)
    target = torch.full((n_rows,), 7, dtype=p.out_dtype, device=dev)
    for _ in range(2):  # repeated runs reuse the cached launcher
        assert fs.flagsparse_spmv_sell(prepared=p, x=xd, out=target) is target
        _check(target, ref, p.out_dtype)
    buf = torch.full((2 * n_rows,), 7, dtype=p.out_dtype, device=dev)
    assert fs.spmv_sell_prepared(p, xd, out=buf[::2]) is not None
    _check(buf[::2], ref, p.out_dtype)
    assert torch.all(buf[1::2].cpu() == 7)


OPTIONS = [
    {"sigma": 32},
    {"sigma": 256},
    {"sigma": 4096},
    {"c": 64, "group": 1, "num_warps": 1},
    {"c": 32, "group": 2, "num_warps": 1},
    {"c": 16, "group": 8, "num_warps": 4},
    {"paired": True},
    {"paired": True, "c": 64, "group": 2, "num_warps": 2},
]


@pytest.mark.skipif(not _SIGMA_OK, reason="SELL-C-sigma needs Triton")
@pytest.mark.parametrize("value", [torch.float32, torch.complex64, torch.int8])
@pytest.mark.parametrize("options", OPTIONS, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()))
@pytest.mark.parametrize("kind", ["short", "long_tail", "medium"])
def test_prepared_layout_options(value, options, kind):
    dev = accelerator_device()
    n_rows, n_cols = 333, 500
    data, indices, indptr = _csr(kind, n_rows, n_cols, value, seed=5)
    x = _random(n_cols, value, torch.Generator().manual_seed(9))
    ref = _golden(data, indices, indptr, x, n_rows)
    sv, sc, so = _sell(data, indices, indptr, n_rows, dev)
    p = fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, route="sigma", **options)
    _check(fs.flagsparse_spmv_sell(prepared=p, x=x.to(dev)), ref, p.out_dtype)


@pytest.mark.skipif(not _SIGMA_OK, reason="SELL-C-sigma needs Triton")
def test_auto_sigma_sorts_only_padded_structures():
    dev = accelerator_device()
    n_rows, n_cols = 4096, 4096
    for kind, sorted_rows in (("long_tail", True), ("medium", True)):
        data, indices, indptr = _csr(kind, n_rows, n_cols, torch.float32, seed=3)
        sv, sc, so = _sell(data, indices, indptr, n_rows, dev)
        p = fs.prepare_spmv_sell(
            sv, sc, so, (n_rows, n_cols), slice_size=32, route="sigma", sigma="auto"
        )
        assert p.n_values <= 1.10 * data.numel() or p.sigma == spmv_sell._SELL_AUTO_MAX_SIGMA
        assert (p.sigma > p.c) == sorted_rows
    # Equal row lengths: nothing to gain, the caller's order is kept.
    indptr = torch.arange(0, 8 * n_rows + 1, 8)
    indices = torch.cat([torch.arange(8) * 37 % n_cols for _ in range(n_rows)])
    sv, sc, so = _sell(torch.ones(8 * n_rows), indices, indptr, n_rows, dev)
    p = fs.prepare_spmv_sell(
        sv, sc, so, (n_rows, n_cols), slice_size=32, route="sigma", sigma="auto"
    )
    assert p.sigma == p.c and p.n_values == 8 * n_rows


@pytest.mark.skipif(not _SIGMA_OK, reason="SELL-C-sigma needs Triton")
def test_empty_matrices():
    dev = accelerator_device()
    for n_rows in (0, 50):
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64)
        sv, sc, so = _sell(torch.zeros(0), torch.zeros(0, dtype=torch.int64), indptr, n_rows, dev)
        p = fs.prepare_spmv_sell(sv, sc, so, (n_rows, 10), slice_size=32, route="sigma")
        y = fs.flagsparse_spmv_sell(prepared=p, x=torch.ones(10, device=dev))
        assert y.shape == (n_rows,) and torch.all(y.cpu() == 0)


@pytest.mark.skipif(not _SIGMA_OK, reason="SELL-C-sigma needs Triton")
def test_padding_never_reads_x():
    """Rows that do not touch column 0 stay finite when x[0] is inf; padding is masked."""
    dev = accelerator_device()
    n_rows, n_cols = 96, 64
    data, indices, indptr = _csr("short", n_rows, n_cols, torch.float32, seed=11)
    keep = indices != 0
    rows = torch.repeat_interleave(torch.arange(n_rows), indptr[1:] - indptr[:-1])[keep]
    indices, data = indices[keep], data[keep]
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64)
    indptr[1:] = torch.cumsum(torch.bincount(rows, minlength=n_rows), 0)
    x = torch.randn(n_cols)
    x[0] = float("inf")
    sv, sc, so = _sell(data, indices, indptr, n_rows, dev)
    p = fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, route="sigma")
    y = fs.flagsparse_spmv_sell(prepared=p, x=x.to(dev)).cpu()
    assert torch.isfinite(y).all()
    _check(y, _golden(data, indices, indptr, x, n_rows), torch.float32)


@pytest.mark.skipif(not _SIGMA_OK, reason="SELL-C-sigma needs Triton")
def test_in_place_value_and_structure_changes_are_seen():
    dev = accelerator_device()
    n_rows, n_cols = 200, 300
    data, indices, indptr = _csr("long_tail", n_rows, n_cols, torch.float32, seed=13)
    x = torch.randn(n_cols)
    sv, sc, so = _sell(data, indices, indptr, n_rows, dev)
    p = fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, route="sigma")
    xd = x.to(dev)
    _check(fs.flagsparse_spmv_sell(prepared=p, x=xd), _golden(data, indices, indptr, x, n_rows), torch.float32)
    sv.mul_(-2.0)  # values only: refreshed through the gather map
    _check(fs.flagsparse_spmv_sell(prepared=p, x=xd), _golden(-2.0 * data, indices, indptr, x, n_rows), torch.float32)
    # Structure change: move every stored column by one (same padding layout).
    shifted = (indices + 1) % n_cols
    sc.copy_(torch.where(sc >= 0, (sc + 1) % n_cols, sc))
    _check(fs.flagsparse_spmv_sell(prepared=p, x=xd), _golden(-2.0 * data, shifted, indptr, x, n_rows), torch.float32)


def test_call_route_matches_unprepared():
    dev = accelerator_device()
    n_rows, n_cols = 150, 120
    data, indices, indptr = _csr("medium", n_rows, n_cols, torch.float32, seed=17)
    x = torch.randn(n_cols).to(dev)
    sv, sc, so = _sell(data, indices, indptr, n_rows, dev)
    p = fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, route="call")
    assert p.route == "call"
    want = fs.flagsparse_spmv_sell(sv, sc, so, x, (n_rows, n_cols), slice_size=32)
    got = fs.flagsparse_spmv_sell(prepared=p, x=x)
    if _is_ascend_runtime():
        torch.testing.assert_close(got.cpu(), want.cpu(), rtol=1e-5, atol=1e-5)
    else:
        assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("strided", [False, True] if _is_ascend_runtime() else [False])
def test_call_route_in_place_value_and_structure_changes_are_seen(dtype, strided):
    dev = accelerator_device()
    sv = torch.arange(1, 129, dtype=torch.float32).to(dtype).to(dev)
    sc = (torch.arange(128, dtype=torch.int32) % 12).to(dev)
    so = torch.tensor([0, 64, 128], dtype=torch.int32, device=dev)
    if strided:
        buffers = [torch.empty(tensor.numel() * 2, dtype=tensor.dtype, device=dev)
                   for tensor in (sv, sc, so)]
        for buffer, tensor in zip(buffers, (sv, sc, so)):
            buffer[::2].copy_(tensor)
        sv, sc, so = (buffer[::2] for buffer in buffers)
    x = torch.arange(1, 13, dtype=torch.float32).to(dtype).to(dev)
    p = fs.prepare_spmv_sell(sv, sc, so, (64, 12), slice_size=32, route="call")

    def check():
        want = fs.flagsparse_spmv_sell(sv, sc, so, x, (64, 12), slice_size=32)
        got = fs.flagsparse_spmv_sell(prepared=p, x=x)
        torch.testing.assert_close(got.cpu(), want.cpu(), rtol=1e-5, atol=1e-5)

    check()
    sv.mul_(-2)
    check()
    sc.copy_((sc + 1) % 12)
    check()
    so[1] = 32
    check()


@pytest.mark.skipif(not _is_ascend_runtime(), reason="Ascend workspace fallback")
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_call_route_fallback_results_own_storage(dtype):
    dev = accelerator_device()
    data, indices, indptr = _csr("short", 64, 48, dtype, seed=41)
    sv, sc, so = _sell(data, indices, indptr, 64, dev)
    p = fs.prepare_spmv_sell(sv, sc, so, (64, 48), slice_size=32, route="call")
    p.ascend_csr_plan = None
    p.ascend_runner = False
    x = torch.ones(48).to(dtype).to(dev)
    first = fs.flagsparse_spmv_sell(prepared=p, x=x)
    saved = first.clone()
    second = fs.flagsparse_spmv_sell(prepared=p, x=x * -2)
    assert torch.equal(first.cpu(), saved.cpu())
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(second.cpu(), first.cpu() * -2)


@pytest.mark.skipif(not _SIGMA_OK, reason="SELL-C-sigma needs Triton")
def test_environment_overrides_and_explicit_arguments(monkeypatch):
    dev = accelerator_device()
    n_rows, n_cols = 128, 128
    data, indices, indptr = _csr("long_tail", n_rows, n_cols, torch.complex64, seed=19)
    sv, sc, so = _sell(data, indices, indptr, n_rows, dev)
    for name, value in (("SIGMA", "64"), ("C", "64"), ("GROUP", "1"), ("WARPS", "1"), ("PAIRED", "1")):
        monkeypatch.setenv(f"FLAGSPARSE_SELL_PREP_{name}", value)
    p = fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, route="sigma")
    assert (p.sigma, p.c, p.group, p.num_warps, p.paired) == (64, 64, 1, 1, True)
    q = fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, route="sigma",
                             sigma=32, c=32, group=4, num_warps=4, paired=False)
    assert (q.sigma, q.c, q.group, q.num_warps, q.paired) == (32, 32, 4, 4, False)
    x = _random(n_cols, torch.complex64, torch.Generator().manual_seed(1))
    ref = _golden(data, indices, indptr, x, n_rows)
    for plan in (p, q):
        _check(fs.flagsparse_spmv_sell(prepared=plan, x=x.to(dev)), ref, torch.complex64)


def test_prepared_sell_rejects_misuse():
    dev = accelerator_device()
    n_rows, n_cols = 64, 48
    data, indices, indptr = _csr("short", n_rows, n_cols, torch.float32, seed=23)
    sv, sc, so = _sell(data, indices, indptr, n_rows, dev)
    p = fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32)
    x = torch.randn(n_cols, device=dev)
    with pytest.raises(ValueError):
        fs.flagsparse_spmv_sell(sv, prepared=p, x=x)
    wrong_dtype = x.cpu().double() if _is_ascend_runtime() else x.double()
    with pytest.raises(TypeError):
        fs.flagsparse_spmv_sell(prepared=p, x=wrong_dtype)
    with pytest.raises(TypeError):
        fs.flagsparse_spmv_sell(prepared=p, x=x.half())
    with pytest.raises(ValueError):
        fs.flagsparse_spmv_sell(prepared=p, x=x[:10])
    with pytest.raises(ValueError):
        fs.flagsparse_spmv_sell(prepared=p, x=x, out=torch.empty(n_rows, dtype=torch.float64, device=dev))
    with pytest.raises(ValueError):
        fs.flagsparse_spmv_sell(prepared=p, x=x, out_dtype=torch.float64)
    with pytest.raises(NotImplementedError):
        fs.flagsparse_spmv_sell(prepared=p, x=x, op="trans")
    with pytest.raises(TypeError):
        fs.flagsparse_spmv_sell(sv, sc, so, x, (n_rows, n_cols))  # slice_size missing
    with pytest.raises(TypeError):
        fs.spmv_sell_prepared(object(), x)
    with pytest.raises(ValueError):
        fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, route="bogus")
    with pytest.raises(TypeError):
        fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, out_dtype=torch.float16)
    if _SIGMA_OK:
        for bad in ({"c": 48}, {"group": 3}, {"num_warps": 3}, {"sigma": 0}):
            with pytest.raises(ValueError):
                fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, route="sigma", **bad)


def _scrambled_band(n, half_width, dtype, seed):
    """A banded square matrix with rows and columns shuffled by one permutation."""
    gen = torch.Generator().manual_seed(seed)
    rows, cols = [], []
    for i in range(n):
        lo, hi = max(0, i - half_width), min(n, i + half_width + 1)
        width = int(torch.randint(1, hi - lo + 1, (1,), generator=gen))
        picked = lo + torch.randperm(hi - lo, generator=gen)[:width]
        rows.append(torch.full((width,), i)), cols.append(picked)
    rows, cols = torch.cat(rows), torch.cat(cols)
    shuffle = torch.randperm(n, generator=gen)
    rows, cols = shuffle[rows], shuffle[cols]
    order = torch.argsort(rows * n + cols)
    rows, cols = rows[order], cols[order]
    indptr = torch.zeros(n + 1, dtype=torch.int64)
    indptr[1:] = torch.cumsum(torch.bincount(rows, minlength=n), 0)
    return _random(rows.numel(), dtype, gen), cols, indptr


try:
    import scipy  # noqa: F401

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@pytest.mark.skipif(not (_SIGMA_OK and _HAS_SCIPY), reason="needs Triton and SciPy")
@pytest.mark.parametrize("value", [torch.float32, torch.complex64, torch.int8, torch.float16])
@pytest.mark.parametrize("reorder", ["rcm", "auto"])
def test_rcm_reorder_matches_golden(value, reorder):
    dev = accelerator_device()
    n = 30000  # median |col - row| of the shuffle ~n/3, over the 4096 auto threshold
    data, indices, indptr = _scrambled_band(n, 3, value, seed=29)
    x = _random(n, value, torch.Generator().manual_seed(31))
    ref = _golden(data, indices, indptr, x, n)
    sv, sc, so = _sell(data, indices, indptr, n, dev)
    p = fs.prepare_spmv_sell(sv, sc, so, (n, n), slice_size=32, route="sigma", reorder=reorder)
    assert p.reorder == "rcm"
    xd = x.to(dev)
    out = torch.empty(n, dtype=p.out_dtype, device=dev)
    for _ in range(2):
        _check(fs.flagsparse_spmv_sell(prepared=p, x=xd, out=out), ref, p.out_dtype)
    if value == torch.float32:
        sv.mul_(3.0)  # value refresh through the gather map, layout unchanged
        _check(fs.flagsparse_spmv_sell(prepared=p, x=xd), _golden(3.0 * data, indices, indptr, x, n), p.out_dtype)


@pytest.mark.skipif(not (_SIGMA_OK and _HAS_SCIPY), reason="needs Triton and SciPy")
def test_rcm_reorder_auto_skips_ordered_and_rectangular():
    dev = accelerator_device()
    n_rows, n_cols = 300, 200
    data, indices, indptr = _csr("medium", n_rows, n_cols, torch.float32, seed=37)
    sv, sc, so = _sell(data, indices, indptr, n_rows, dev)
    p = fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, route="sigma", reorder="auto")
    assert p.reorder == "none" and p.x_order is None
    with pytest.raises(ValueError):
        fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, route="sigma", reorder="rcm")
    with pytest.raises(ValueError):
        fs.prepare_spmv_sell(sv, sc, so, (n_rows, n_cols), slice_size=32, route="sigma", reorder="bogus")
    # A matrix that is already banded gains nothing: auto keeps the order.
    n = 5000
    rows = torch.arange(n).repeat_interleave(3)
    cols = (rows + torch.tensor([-1, 0, 1]).repeat(n)).clamp(0, n - 1)
    order = torch.argsort(rows * n + cols)
    rows, cols = rows[order], cols[order]
    indptr = torch.zeros(n + 1, dtype=torch.int64)
    indptr[1:] = torch.cumsum(torch.bincount(rows, minlength=n), 0)
    sv, sc, so = _sell(torch.ones(rows.numel()), cols, indptr, n, dev)
    p = fs.prepare_spmv_sell(sv, sc, so, (n, n), slice_size=32, route="sigma", reorder="auto")
    assert p.reorder == "none"
