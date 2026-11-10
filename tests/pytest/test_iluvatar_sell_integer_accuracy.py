"""Iluvatar integer SELL tiles retain exact live operands and output semantics."""

import importlib

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations._common import _is_iluvatar_runtime
from tests.pytest.accuracy_utils import accelerator_available
from tests.pytest.test_iluvatar_sell_storage_accuracy import _reference


pytestmark = pytest.mark.skipif(
    not _is_iluvatar_runtime() or not accelerator_available(), reason="Iluvatar-only integer SELL tiles",
)


def _inputs(lengths, index_dtype):
    rows = lengths.numel()
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
    positions = torch.arange(int(offsets[-1]))
    data = (positions.remainder(256) - 128).to(torch.int8).cuda()
    columns = (positions * 7).remainder(257).to(index_dtype).cuda()
    values, columns, offsets = fs.csr_to_sell(data, columns, offsets.to(index_dtype).cuda(), rows, 32, index_dtype=index_dtype)
    vector = (torch.arange(514, device="cuda").remainder(256) - 128).to(torch.int8)[::2]
    return values, columns, offsets, vector


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("width", [13, 16, 27, 33])
@pytest.mark.parametrize("inference", [False, True])
def test_integer_raw_grouped_live_values_and_strides(index_dtype, width, inference, monkeypatch):
    sell = importlib.import_module("flagsparse.sparse_operations.spmv_sell")
    kernel = sell._musa_spmv_sell_kernel
    launches = []

    class TrackedKernel:
        def __getitem__(self, grid):
            def launch(*args, **options):
                launches.append((options["ROWS"], options["SLOTS"]))
                return kernel[grid](*args, **options)
            return launch

    monkeypatch.setattr(sell, "_musa_spmv_sell_kernel", TrackedKernel())
    with torch.inference_mode(inference):
        lengths = torch.full((257,), width, dtype=torch.int64)
        lengths[::17] = 0
        values, columns, offsets, vector = _inputs(lengths, index_dtype)
        target = torch.empty(514, dtype=torch.int32, device="cuda")[::2]
        for step in range(4):
            output = fs.flagsparse_spmv_sell(values, columns, offsets, vector, (257, 257), slice_size=32, out=target, validate=False)
            assert output is target
            torch.testing.assert_close(output.cpu().double(), _reference(values, columns, offsets, vector, 257), rtol=0, atol=0)
            assert not sell._iluvatar_sell_plan(values, columns, offsets, (257, 257), 32, torch.int32)
            if step == 0:
                values.neg_()
                vector.add_(1)
            elif step == 1:
                valid = columns >= 0
                columns[valid] = (columns[valid] + 19) % 257
            else:
                values, columns, offsets = values.clone(), columns.clone(), offsets.clone()
        assert launches == [(64, 4)] * 4


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("out_dtype", [torch.int32, torch.float32])
def test_integer_sigma_tile_preserves_widening_and_mutation(index_dtype, out_dtype):
    sell = importlib.import_module("flagsparse.sparse_operations.spmv_sell")
    lengths = torch.where(torch.arange(69) % 3 == 0, 61, 19)
    values, columns, offsets, vector = _inputs(lengths, index_dtype)
    target = torch.empty(138, dtype=out_dtype, device="cuda")[::2]
    for step in range(3):
        output = fs.flagsparse_spmv_sell(values, columns, offsets, vector, (69, 257), slice_size=32, out=target, validate=False)
        assert output is target
        torch.testing.assert_close(output.cpu().double(), _reference(values, columns, offsets, vector, 69), rtol=0, atol=0)
        route, prepared = sell._iluvatar_sell_plan(values, columns, offsets, (69, 257), 32, out_dtype)
        assert route == "sigma"
        assert (prepared.c, prepared.k) == ((16, 4) if out_dtype == torch.int32 else (32, 4))
        if step == 0:
            values.add_(1)
            vector.neg_()
        else:
            valid = columns >= 0
            columns[valid] = (columns[valid] + 23) % 257


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_integer_large_short_rows_and_timed_output(index_dtype):
    rows = 1048577
    slices = (rows + 31) // 32
    columns = torch.cat((torch.zeros(32, dtype=index_dtype), torch.ones(32, dtype=index_dtype))).repeat(slices).cuda()
    values = torch.cat((torch.full((32,), -128, dtype=torch.int8), torch.full((32,), 127, dtype=torch.int8))).repeat(slices).cuda()
    offsets = (torch.arange(slices + 1, dtype=index_dtype) * 64).cuda()
    vector = torch.tensor([-128, 127], dtype=torch.int8, device="cuda")
    target = torch.empty(rows, dtype=torch.int32, device="cuda")
    output, elapsed = fs.flagsparse_spmv_sell(values, columns, offsets, vector, (rows, 2), slice_size=32, out=target, return_time=True)
    assert output is target
    assert elapsed > 0
    torch.testing.assert_close(output, torch.full_like(target, 32513), rtol=0, atol=0)
