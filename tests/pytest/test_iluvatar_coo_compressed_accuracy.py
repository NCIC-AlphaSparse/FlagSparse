"""Lossless COO storage, dynamic inputs, and representation boundaries."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import _spmv_iluvatar_compressed as compressed
from flagsparse.sparse_operations import _spmv_iluvatar_ell as ell
from flagsparse.sparse_operations._common import _is_iluvatar_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device


pytestmark = pytest.mark.skipif(
    not _is_iluvatar_runtime() or not accelerator_available(), reason="Iluvatar-only storage",
)


def _inputs(mode, index_dtype=torch.int32, n_cols=257, width=10, n_rows=69):
    device = accelerator_device()
    lengths = torch.full((n_rows,), width, device=device, dtype=index_dtype)
    lengths[:2] = 0
    lengths[-2:] = 0
    ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).to(index_dtype)))
    count = int(ptr[-1])
    cols = (torch.arange(count, device=device) * 197 % n_cols).to(index_dtype)
    position = torch.arange(count, device=device)
    if mode == "one":
        real = torch.ones(count, device=device)
    elif mode == "constant":
        real = torch.full((count,), -2.25, device=device)
    elif mode == "integer":
        real = (position % 256 - 128).float()
    elif mode == "dictionary":
        real = (position % 335).float() * 0.125 + 0.03125
    elif mode == "full":
        real = torch.linspace(-1.1, 1.7, count, device=device)
    else:
        raise ValueError(mode)
    values = torch.complex(real, (position % 31 - 15).float() * 0.03)
    vector = torch.randn(n_cols * 2, dtype=torch.complex64, device=device)[::2]
    target = torch.empty(lengths.numel() * 2, dtype=torch.complex64, device=device)[::2]
    return values, cols, ptr, vector, target


def _reference(values, cols, ptr, vector):
    lengths = (ptr[1:] - ptr[:-1]).cpu().long()
    rows = torch.repeat_interleave(torch.arange(lengths.numel()), lengths)
    result = torch.zeros(lengths.numel(), dtype=torch.complex128)
    result.index_add_(0, rows, values.cpu().to(torch.complex128) * vector.cpu().to(torch.complex128)[cols.cpu().long()])
    return result


@pytest.mark.parametrize("mode,encoding", [("one", 4), ("constant", 1), ("integer", 2), ("dictionary", 3), ("full", 0)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("n_cols", [257, 100003])
def test_compressed_modes_live_rhs_values_and_topology(mode, encoding, index_dtype, n_cols):
    values, cols, ptr, vector, target = _inputs(mode, index_dtype, n_cols)
    cache = {"coo_c32_non": True}
    for step in range(4):
        assert ell.run(values, cols, ptr, vector, target, target.numel(), cache)
        torch.testing.assert_close(target.cpu().to(torch.complex128), _reference(values, cols, ptr, vector), rtol=1e-5, atol=1e-5)
        if step == 0:
            assert cache["ell_compressed"]["encoding"] == encoding
            if encoding in (0, 1, 4):
                assert cache["ell_compressed"]["short"] == (n_cols == 257)
        elif step == 1:
            assert cache["ell_compressed"]["encoding"] == 0
        if step == 0:
            values.real.copy_(torch.linspace(-0.7, 1.3, values.numel(), device=values.device))
        elif step == 1:
            values.real.fill_(1.)
            cols[0] = (cols[0] + 17) % n_cols
        else:
            ptr[2] = 1
        values.imag.add_(0.125)
        vector.mul_(0.7 + 0.1j)


@pytest.mark.parametrize("mode", ["one", "constant", "integer", "dictionary", "full"])
@pytest.mark.parametrize("width", [2, 10, 80])
def test_public_prepared_coo_compressed_modes(mode, width):
    values, cols, ptr, vector, target = _inputs(mode, width=width)
    rows = torch.repeat_interleave(torch.arange(69, device=cols.device, dtype=cols.dtype), ptr[1:] - ptr[:-1])
    prepared = fs.prepare_spmv_coo(values, rows, cols, (69, vector.numel()))
    for _step in range(2):
        output = fs.flagsparse_spmv_coo(prepared=prepared, x=vector, out=target)
        assert output is target
        torch.testing.assert_close(output.cpu().to(torch.complex128), _reference(prepared.data_non, prepared.col_non, ptr, vector), rtol=1e-5, atol=1e-5)
        vector.add_(0.15 - 0.1j)
        prepared.data_non.add_(0.125 + 0.25j)


@pytest.mark.parametrize("pattern", ["zeros", "nonfinite", "int_edges"])
@pytest.mark.parametrize("n_cols", [257, 1 << 24, (1 << 24) + 1])
def test_real_encoding_bitwise_round_trip(pattern, n_cols):
    values, cols, ptr, _vector, _target = _inputs("integer")
    if pattern == "zeros":
        values.real[::2] = -0.
        values.real[1::2] = 0.
    elif pattern == "nonfinite":
        patterns = torch.tensor([0x7FC00001, 0x7FC00002, 0x7F800000, -8388608], dtype=torch.int32, device=values.device)
        values.real.copy_(patterns.view(torch.float32)[torch.arange(values.numel(), device=values.device) % 4])
    plan = ell._layout(cols, ptr, 69)
    real, payload, encoding, colbits = compressed._encode_real(values, plan, n_cols)
    if encoding == 4:
        decoded = torch.ones(plan[4].numel(), device=values.device)
    elif encoding == 1:
        decoded = real.expand(plan[4].numel())
    elif encoding == 2:
        decoded = (payload.long() >> colbits).to(torch.int8).float()
    elif encoding == 3:
        codes = (payload.long() & 0xFFFFFFFF) >> colbits
        decoded = real[codes]
    else:
        decoded = real
    assert torch.equal(decoded.contiguous().view(torch.int32), values.real[plan[4]].contiguous().view(torch.int32))
    if encoding in (2, 3):
        assert torch.equal(payload.long() & ((1 << colbits) - 1), plan[0].long())


def test_replaced_sources_and_inference_fallback():
    values, cols, ptr, vector, target = _inputs("one")
    cache = {"coo_c32_non": True}
    assert ell.run(values, cols, ptr, vector, target, 69, cache)
    previous = cache["ell_compressed"]
    values = values.clone() + 0.125j
    cols, ptr = cols.clone(), ptr.clone()
    assert ell.run(values, cols, ptr, vector, target, 69, cache)
    assert cache["ell_compressed"] is not previous
    torch.testing.assert_close(target.cpu().to(torch.complex128), _reference(values, cols, ptr, vector), rtol=1e-5, atol=1e-5)
    with torch.inference_mode():
        assert not ell.run(values.clone(), cols, ptr, vector, target, 69, cache)


@pytest.mark.parametrize("span", [65535, 65536])
def test_narrow_column_range_boundary(span):
    values, cols, ptr, vector, target = _inputs("one", n_cols=span + 48)
    cols[::2], cols[1::2] = 47, 47 + span
    cache = {"coo_c32_non": True}
    assert ell.run(values, cols, ptr, vector, target, 69, cache)
    assert cache["ell_compressed"]["short"] == (span == 65535)
    torch.testing.assert_close(target.cpu().to(torch.complex128), _reference(values, cols, ptr, vector), rtol=1e-5, atol=1e-5)


def test_maximum_packed_row_length():
    values, cols, ptr, vector, target = _inputs("integer", width=255)
    cache = {"coo_c32_non": True}
    assert ell.run(values, cols, ptr, vector, target, 69, cache)
    assert cache["ell_compressed"]["encoding"] == 2
    # 255 float32 additions on an unseeded RHS: the forward error bound is ~3e-5.
    torch.testing.assert_close(target.cpu().to(torch.complex128), _reference(values, cols, ptr, vector), rtol=5e-5, atol=1e-5)


@pytest.mark.parametrize("n_rows", [17, 65, 129])
@pytest.mark.parametrize("width", [16, 31, 32, 33])
def test_grouped_integer_tiles_tail_and_live_inputs(n_rows, width):
    values, cols, ptr, vector, target = _inputs("integer", torch.int64, width=width, n_rows=n_rows)
    cache = {"coo_c32_non": True}
    for step in range(3):
        assert ell.run(values, cols, ptr, vector, target, n_rows, cache)
        assert cache["ell_compressed"]["encoding"] == 2
        torch.testing.assert_close(target.cpu().to(torch.complex128), _reference(values, cols, ptr, vector), rtol=1e-5, atol=1e-5)
        if step == 0:
            values.real.add_(1).clamp_(-128, 127)
            values.imag.add_(0.125)
            vector.mul_(0.7 + 0.1j)
        else:
            cols.add_(17).remainder_(vector.numel())


@pytest.mark.parametrize("maximum,n_rows", [(256, 69), (10, 0xFFFFFF)])
def test_metadata_limits_fall_back(maximum, n_rows):
    values, cols, ptr, vector, target = _inputs("one")
    plan = ell._layout(cols, ptr, 69)
    assert not compressed.run(values, plan, vector, target, n_rows, {}, 16, 4, maximum, values._version)


@pytest.mark.parametrize("mode", ["one", "constant", "full"])
def test_mixed_column_tiles_and_storage_transitions(mode):
    values, cols, ptr, vector, target = _inputs(mode, n_cols=100003, width=44)
    cols.remainder_(257)
    cols[-44::2] = 47
    cols[-43::2] = 100002
    cache = {"coo_c32_non": True}
    for step, expected in enumerate((2, True, False)):
        assert ell.run(values, cols, ptr, vector, target, 69, cache)
        storage = cache["ell_compressed"]
        assert storage["short"] == expected
        if step == 0:
            assert bool(torch.any(storage["bases"] < 0))
            assert bool(torch.any(storage["bases"] >= 0))
        torch.testing.assert_close(target.cpu().to(torch.complex128), _reference(values, cols, ptr, vector), rtol=1e-5, atol=1e-5)
        if step == 0:
            cols.remainder_(257)
        else:
            cols.copy_((torch.arange(cols.numel(), device=cols.device) * 197 % vector.numel()).to(cols.dtype))
        values.imag.add_(0.125)
        vector.mul_(0.7 + 0.1j)


@pytest.mark.parametrize("height", [8, 16])
def test_empty_tiles_do_not_qualify_wide_columns_for_mixed_storage(height):
    device = accelerator_device()
    lengths = torch.zeros(256, dtype=torch.int32, device=device)
    lengths[0] = 44
    ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).int()))
    cols = torch.zeros(44, dtype=torch.int32, device=device)
    cols[1::2] = 100002
    plan = ell._layout(cols, ptr, 256, height)
    columns, _bases, short = compressed._narrow_columns(plan, 100003, height, 44)
    assert columns is plan[0]
    assert short is False
