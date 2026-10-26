"""K-wide SELL-C-sigma (``prepare_spmv_sell(k=...)``) and the Iluvatar prepared COO -> SELL route."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import _common
from flagsparse.sparse_operations import spmv_sell as _spmv_sell
from tests.pytest.accuracy_utils import ACCELERATOR_REQUIRED, accelerator_available, accelerator_device


# The sigma kernel is plain Triton: K-wide runs wherever the sigma route does.
pytestmark = pytest.mark.skipif(
    not accelerator_available() or _common._is_ascend_runtime(),
    reason=ACCELERATOR_REQUIRED + " (the SELL-C-sigma route needs Triton)",
)
iluvatar_only = pytest.mark.skipif(
    not _common._is_iluvatar_runtime(), reason="the COO -> SELL route is Iluvatar-only"
)

SELL_PAIRS = [
    (torch.float32, torch.float32),
    (torch.float16, torch.float16),
    (torch.float16, torch.float32),
    (torch.int8, torch.int32),
    (torch.int8, torch.float32),
]
COO_PAIRS = [
    (torch.float16, torch.float16),
    (torch.float16, torch.float32),
    (torch.int8, torch.int32),
]


def _case(value_dtype, n_rows=301, n_cols=257, max_len=41, seed=0):
    """Row lengths 0..max_len (empty rows, rows wider than K, n_rows not a multiple of c)."""
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(0, max_len + 1, (n_rows,), generator=gen)
    lengths[::7] = 0
    rows = torch.repeat_interleave(torch.arange(n_rows), lengths)
    cols = torch.cat([torch.randperm(n_cols, generator=gen)[:int(n)] for n in lengths]).to(torch.int32)
    if value_dtype == torch.int8:
        values = torch.randint(-3, 4, (rows.numel(),), generator=gen).to(torch.int8)
        x = torch.randint(-2, 3, (n_cols,), generator=gen).to(torch.int8)
    else:
        values = (torch.rand(rows.numel(), generator=gen) - 0.5).to(value_dtype)
        x = (torch.rand(n_cols, generator=gen) - 0.5).to(value_dtype)
    ptr = torch.zeros(n_rows + 1, dtype=torch.int32)
    ptr[1:] = torch.cumsum(lengths, 0)
    return values, rows.to(torch.int32), cols, ptr, x, (n_rows, n_cols)


def _reference(values, rows, cols, x, n_rows, out_dtype):
    if out_dtype == torch.int32:
        y = torch.zeros(n_rows, dtype=torch.int64)
        y.index_add_(0, rows.long(), values.long() * x.long()[cols.long()])
        return y
    y = torch.zeros(n_rows, dtype=torch.float64)
    y.index_add_(0, rows.long(), values.double() * x.double()[cols.long()])
    return y


def _check(actual, reference, out_dtype):
    assert actual.dtype == out_dtype
    if out_dtype == torch.int32:
        assert torch.equal(actual.cpu().long(), reference)
    else:
        tol = 2e-3 if out_dtype == torch.float16 else 1e-5
        assert torch.allclose(actual.cpu().double(), reference, rtol=tol, atol=tol)


def _sell(values, cols, ptr, n_rows, device):
    return fs.csr_to_sell(values.to(device), cols.to(device), ptr.to(device), n_rows, 32)


@pytest.mark.parametrize("value_dtype,out_dtype", SELL_PAIRS)
@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
def test_prepared_sell_kwide_matches_reference(value_dtype, out_dtype, k):
    values, rows, cols, ptr, x, shape = _case(value_dtype)
    device = accelerator_device()
    plan = fs.prepare_spmv_sell(
        *_sell(values, cols, ptr, shape[0], device), shape,
        slice_size=32, out_dtype=out_dtype, route="sigma", c=32, group=1, num_warps=1, k=k,
    )
    assert plan.k == k
    actual = fs.spmv_sell_prepared(plan, x.to(device))
    _check(actual, _reference(values, rows, cols, x, shape[0], out_dtype), out_dtype)


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
def test_prepared_sell_kwide_empty_one_slice(k):
    device = accelerator_device()
    values = torch.zeros(0, dtype=torch.float32, device=device)
    cols = torch.zeros(0, dtype=torch.int32, device=device)
    offsets = torch.zeros(2, dtype=torch.int32, device=device)
    x = torch.ones(17, dtype=torch.float32, device=device)
    plan = fs.prepare_spmv_sell(
        values, cols, offsets, (1, 17), slice_size=32, route="sigma",
        c=32, group=1, num_warps=1, k=k,
    )
    actual = fs.spmv_sell_prepared(plan, x)
    assert torch.equal(actual, torch.zeros(1, dtype=torch.float32, device=device))


def test_prepared_sell_kwide_rejects_explicit_invalid_k():
    values, _, cols, ptr, _, shape = _case(torch.float32)
    device = accelerator_device()
    sell = _sell(values, cols, ptr, shape[0], device)
    with pytest.raises(ValueError, match="k > 1"):
        fs.prepare_spmv_sell(*sell, shape, slice_size=32, route="sigma", c=32, group=2, k=2)
    with pytest.raises(ValueError, match="power of two"):
        fs.prepare_spmv_sell(*sell, shape, slice_size=32, route="sigma", c=32, group=1, k=3)
    with pytest.raises(ValueError, match="route='sigma'"):
        fs.prepare_spmv_sell(*sell, shape, slice_size=32, route="call", k=4)
    complex_sell = (sell[0].to(torch.complex64),) + tuple(sell[1:])
    with pytest.raises(ValueError, match="k > 1"):
        fs.prepare_spmv_sell(*complex_sell, shape, slice_size=32, route="sigma", c=32, group=1, k=2)


@pytest.mark.parametrize("route", ["auto", "sigma"])
def test_env_k_does_not_break_complex_or_grouped_plans(monkeypatch, route):
    values, rows, cols, ptr, x, shape = _case(torch.float32)
    device = accelerator_device()
    sell = _sell(values.to(torch.complex64), cols, ptr, shape[0], device)
    monkeypatch.setenv("FLAGSPARSE_SELL_PREP_K", "4")
    plan = fs.prepare_spmv_sell(*sell, shape, slice_size=32, route=route)
    assert plan.k == 1
    actual = fs.spmv_sell_prepared(plan, x.to(torch.complex64).to(device))
    reference = _reference(values, rows, cols, x, shape[0], torch.float64)
    assert torch.allclose(actual.cpu().real.double(), reference, rtol=1e-5, atol=1e-5)
    grouped = fs.prepare_spmv_sell(
        *_sell(values, cols, ptr, shape[0], device), shape, slice_size=32, route="sigma", group=2
    )
    assert grouped.k == 1


@pytest.mark.parametrize("value_dtype,out_dtype", SELL_PAIRS)
def test_sell_plan_variant_shares_layout(value_dtype, out_dtype):
    values, rows, cols, ptr, x, shape = _case(value_dtype)
    device = accelerator_device()
    base = fs.prepare_spmv_sell(
        *_sell(values, cols, ptr, shape[0], device), shape,
        slice_size=32, route="sigma", c=32, group=1, num_warps=1, k=1,
    )
    variant = _spmv_sell._sell_plan_variant(base, k=4, out_dtype=out_dtype)
    assert variant.k == 4 and variant.out_dtype == out_dtype
    assert variant.sell_values.data_ptr() == base.sell_values.data_ptr()
    actual = fs.spmv_sell_prepared(variant, x.to(device))
    _check(actual, _reference(values, rows, cols, x, shape[0], out_dtype), out_dtype)


def _coo(value_dtype, unsorted):
    values, rows, cols, ptr, x, shape = _case(value_dtype, seed=1)
    if unsorted:
        order = torch.randperm(rows.numel(), generator=torch.Generator().manual_seed(2))
        values, rows, cols = values[order], rows[order], cols[order]
    return values, rows, cols, x, shape


@iluvatar_only
@pytest.mark.parametrize("value_dtype,out_dtype", COO_PAIRS)
@pytest.mark.parametrize("mode", ["0", "1", "auto"])
@pytest.mark.parametrize("sort_by_row,unsorted", [(True, False), (True, True), (False, False), (False, True)])
def test_prepared_coo_sell_matches_reference(monkeypatch, value_dtype, out_dtype, mode, sort_by_row, unsorted):
    values, rows, cols, x, shape = _coo(value_dtype, unsorted)
    device = accelerator_device()
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_COO_SELL", mode)
    monkeypatch.delenv("FLAGSPARSE_SELL_PREP_K", raising=False)
    prepared = fs.prepare_spmv_coo(
        values.to(device), rows.to(device), cols.to(device), shape, sort_by_row=sort_by_row
    )
    actual = fs.flagsparse_spmv_coo(prepared=prepared, x=x.to(device), out_dtype=out_dtype)
    _check(actual, _reference(values, rows, cols, x, shape[0], out_dtype), out_dtype)
    choice = prepared.sell_choices.get(out_dtype)
    if mode == "0":
        assert choice is None and not prepared.sell_plans
        return
    if mode == "1":
        assert choice == "sell_k4"
    else:
        assert choice in ("existing", "sell_k2", "sell_k4")
        assert set(prepared.sell_times[out_dtype]) == {"existing", "sell_k2", "sell_k4"}
    # Only the chosen plan is kept.
    assert (out_dtype in prepared.sell_plans) == (choice != "existing")
    if choice != "existing":
        assert prepared.sell_plans[out_dtype].plan.k == int(choice[-1])


@iluvatar_only
def test_prepared_coo_sell_forced_k_from_env(monkeypatch):
    values, rows, cols, x, shape = _coo(torch.float16, False)
    device = accelerator_device()
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_COO_SELL", "1")
    monkeypatch.setenv("FLAGSPARSE_SELL_PREP_K", "2")
    prepared = fs.prepare_spmv_coo(values.to(device), rows.to(device), cols.to(device), shape)
    assert prepared.sell_choices[torch.float16] == "sell_k2"
    actual = fs.flagsparse_spmv_coo(prepared=prepared, x=x.to(device))
    _check(actual, _reference(values, rows, cols, x, shape[0], torch.float16), torch.float16)


@iluvatar_only
@pytest.mark.parametrize("mode", ["0", "1"])
def test_prepared_coo_sell_sees_in_place_updates_like_existing_path(monkeypatch, mode):
    # sort_by_row=False keeps the caller's values tensor; the existing path reads it
    # on every call, so the SELL route must follow in-place updates the same way.
    values, rows, cols, x, shape = _coo(torch.float16, True)
    device = accelerator_device()
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_COO_SELL", mode)
    data = values.to(device)
    prepared = fs.prepare_spmv_coo(data, rows.to(device), cols.to(device), shape, sort_by_row=False)
    fs.flagsparse_spmv_coo(prepared=prepared, x=x.to(device), out_dtype=torch.float32)
    data.mul_(2)
    actual = fs.flagsparse_spmv_coo(prepared=prepared, x=x.to(device), out_dtype=torch.float32)
    _check(actual, _reference(values * 2, rows, cols, x, shape[0], torch.float32), torch.float32)


@iluvatar_only
def test_prepared_coo_sell_rejects_unknown_mode(monkeypatch):
    values, rows, cols, _, shape = _coo(torch.float16, False)
    device = accelerator_device()
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_COO_SELL", "yes")
    with pytest.raises(ValueError, match="FLAGSPARSE_ILUVATAR_COO_SELL"):
        fs.prepare_spmv_coo(values.to(device), rows.to(device), cols.to(device), shape)
