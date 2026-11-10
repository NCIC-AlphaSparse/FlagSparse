"""SELL compressed storage and dense real tiles retain live operands."""

import importlib

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations._common import _is_iluvatar_runtime
from tests.pytest.accuracy_utils import accelerator_available
from tests.pytest.test_iluvatar_coo_compressed_accuracy import _inputs


pytestmark = pytest.mark.skipif(
    not _is_iluvatar_runtime() or not accelerator_available(), reason="Iluvatar-only SELL routes",
)


def _reference(values, columns, offsets, vector, rows):
    sell = importlib.import_module("flagsparse.sparse_operations.spmv_sell")
    row_ids = sell._sell_entry_rows(columns, offsets, 32).cpu().long()
    columns = columns.cpu().long()
    valid = (columns >= 0) & (row_ids < rows)
    dtype = torch.complex128 if values.is_complex() else torch.float64
    output = torch.zeros(rows, dtype=dtype)
    output.index_add_(0, row_ids[valid], values.cpu().to(dtype)[valid] * vector.cpu().to(dtype)[columns[valid]])
    return output


@pytest.mark.parametrize("mode", ["one", "constant", "integer", "dictionary", "full"])
@pytest.mark.parametrize("width", [10, 32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_sell_compressed_live_values_topology_and_strides(mode, width, index_dtype):
    sell = importlib.import_module("flagsparse.sparse_operations.spmv_sell")
    data, indices, ptr, vector, target = _inputs(mode, index_dtype, width=width)
    values, columns, offsets = fs.csr_to_sell(data, indices, ptr, 69, 32, index_dtype=index_dtype)
    for step in range(4):
        output = fs.flagsparse_spmv_sell(values, columns, offsets, vector, (69, vector.numel()), slice_size=32, out=target, validate=False)
        assert output is target
        torch.testing.assert_close(output.cpu().to(torch.complex128), _reference(values, columns, offsets, vector, 69), rtol=1e-5, atol=1e-5)
        route, prepared = sell._iluvatar_sell_plan(values, columns, offsets, (69, vector.numel()), 32, values.dtype)
        assert route == "csr"
        assert prepared.non_gather_plan[-1]["sell_c32_non"]
        assert "ell_compressed" in prepared.non_gather_plan[-1]
        if step == 0:
            values.real.copy_(torch.linspace(-0.75, 1.25, values.numel(), device=values.device))
            values.imag.add_(0.125)
        elif step == 1:
            valid = columns >= 0
            columns[valid] = (columns[valid] + 17) % vector.numel()
        else:
            values = values.clone() + 0.25j
            columns, offsets = columns.clone(), offsets.clone()
        vector.mul_(0.7 + 0.1j)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int8])
@pytest.mark.parametrize("inference", [False, True])
def test_dense_sell_tile_mutation_and_inference(dtype, inference):
    sell = importlib.import_module("flagsparse.sparse_operations.spmv_sell")
    with torch.inference_mode(inference):
        data, indices, ptr, unused_vector, unused_target = _inputs("integer", width=20)
        data = data.real.to(dtype)
        values, columns, offsets = fs.csr_to_sell(data, indices, ptr, 69, 32)
        vector = (torch.arange(514, device=data.device) % 3 - 1).to(dtype)[::2]
        out_dtype = torch.int32 if dtype == torch.int8 else dtype
        target = torch.empty(138, dtype=out_dtype, device=data.device)[::2]
        for step in range(3):
            output = fs.flagsparse_spmv_sell(values, columns, offsets, vector, (69, 257), slice_size=32, out=target, validate=False)
            assert output is target
            torch.testing.assert_close(output.cpu().double(), _reference(values, columns, offsets, vector, 69), rtol=0, atol=0)
            plan = sell._iluvatar_sell_plan(values, columns, offsets, (69, 257), 32, out_dtype)
            if inference:
                assert not plan
            else:
                assert plan[0] == "sigma"
                expected = {torch.float32: (16, 8), torch.int8: (16, 4), torch.float16: (32, 4)}
                assert (plan[1].c, plan[1].k) == expected[dtype]
            if step == 0:
                values.fill_(2)
                vector.neg_()
            else:
                valid = columns >= 0
                columns[valid] = (columns[valid] + 17) % vector.numel()


@pytest.mark.parametrize("override", ["SPLIT", "BATCH", "LANES", "WARPS", "STAGES"])
def test_sell_compressed_respects_launch_overrides(monkeypatch, override):
    sell = importlib.import_module("flagsparse.sparse_operations.spmv_sell")
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_GATHER_" + override, "0" if override == "SPLIT" else "1")
    data, indices, ptr, vector, target = _inputs("integer", width=10)
    values, columns, offsets = fs.csr_to_sell(data, indices, ptr, 69, 32)
    output = fs.flagsparse_spmv_sell(values, columns, offsets, vector, (69, vector.numel()), slice_size=32, out=target, validate=False)
    torch.testing.assert_close(output.cpu().to(torch.complex128), _reference(values, columns, offsets, vector, 69), rtol=1e-5, atol=1e-5)
    plan = sell._iluvatar_sell_plan(values, columns, offsets, (69, vector.numel()), 32, values.dtype)
    assert "ell_compressed" not in plan[1].non_gather_plan[-1]
