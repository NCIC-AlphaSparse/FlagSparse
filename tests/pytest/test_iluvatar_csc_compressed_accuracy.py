"""Conjugated CSC storage with live values, topology, and dense operands."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import _spmv_iluvatar_compressed as compressed
from flagsparse.sparse_operations import _spmv_iluvatar_ell as ell
from flagsparse.sparse_operations._common import _is_iluvatar_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device
from tests.pytest.test_iluvatar_coo_compressed_accuracy import _inputs, _reference


pytestmark = pytest.mark.skipif(
    not _is_iluvatar_runtime() or not accelerator_available(), reason="Iluvatar-only CSC storage",
)


def _check(values, indices, ptr, vector, output):
    expected = _reference(values.conj(), indices, ptr, vector)
    torch.testing.assert_close(output.cpu().to(torch.complex128), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", ["one", "constant", "integer", "dictionary", "full"])
@pytest.mark.parametrize("width", [2, 10, 32, 80])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_prepared_csc_conjugate_compressed_live_inputs(mode, width, index_dtype):
    values, indices, ptr, vector, target = _inputs(mode, index_dtype, n_cols=257, width=width)
    prepared = fs.prepare_spmv_csc(values, indices, ptr, (vector.numel(), 69), op="conj")
    assert prepared.maca_trans_plan[-1]["csc_c32_conj"]
    for step in range(3):
        output = fs.flagsparse_spmv_csc(prepared=prepared, x=vector, out=target)
        assert output is target
        _check(prepared.data, prepared.kernel_indices, prepared.kernel_indptr, vector, output)
        assert "ell_compressed" in prepared.maca_trans_plan[-1]
        if step == 0:
            prepared.data.real.copy_(torch.linspace(-0.75, 1.25, values.numel(), device=values.device))
            prepared.data.imag.add_(0.125)
            vector.mul_(0.7 + 0.1j)
        else:
            prepared.kernel_indices.add_(17).remainder_(vector.numel())
            prepared.kernel_indptr[2] = 1


@pytest.mark.parametrize("mode", ["one", "constant", "full"])
def test_conjugate_mixed_columns_and_storage_transitions(mode):
    values, indices, ptr, vector, target = _inputs(mode, n_cols=100003, width=44)
    indices.remainder_(257)
    indices[-44::2] = 47
    indices[-43::2] = 100002
    cache = {"csc_c32_conj": True}
    for step, expected in enumerate((2, True, False)):
        assert ell.run(values, indices, ptr, vector, target, 69, cache, conjugate=True)
        assert cache["ell_compressed"]["short"] == expected
        _check(values, indices, ptr, vector, target)
        if step == 0:
            indices.remainder_(257)
        else:
            indices.copy_((torch.arange(indices.numel(), device=indices.device) * 197 % vector.numel()).to(indices.dtype))
        values.imag.add_(0.125)
        vector.mul_(0.7 + 0.1j)


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("lazy_conjugate", [False, True])
def test_conjugate_replacement_and_inference_inputs(inference, lazy_conjugate):
    with torch.inference_mode(inference):
        values, indices, ptr, vector, target = _inputs("integer", width=10)
        if lazy_conjugate:
            values = values.conj()
            vector = vector.conj()
        prepared = fs.prepare_spmv_csc(values, indices, ptr, (vector.numel(), 69), op="conj")
        for step in range(3):
            output = fs.flagsparse_spmv_csc(prepared=prepared, x=vector, out=target)
            _check(prepared.data, prepared.kernel_indices, prepared.kernel_indptr, vector, output)
            if inference:
                assert "ell_compressed" not in prepared.maca_trans_plan[-1]
            if step == 0:
                prepared.data = prepared.data.clone() + 0.25j
                prepared.kernel_indices.copy_((prepared.kernel_indices + 17) % vector.numel())
            else:
                prepared.data.add_(0.125 + 0.25j)
                vector = vector * (0.7 + 0.1j)


@pytest.mark.parametrize("override", ["SPLIT", "BATCH", "LANES", "WARPS", "STAGES"])
def test_conjugate_explicit_launch_overrides_bypass_compression(monkeypatch, override):
    values, indices, ptr, vector, target = _inputs("integer", width=10)
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_GATHER_" + override, "0" if override == "SPLIT" else "1")
    calls = []
    original = compressed.run

    def capture(*arguments, **options):
        calls.append(True)
        return original(*arguments, **options)

    monkeypatch.setattr(compressed, "run", capture)
    prepared = fs.prepare_spmv_csc(values, indices, ptr, (vector.numel(), 69), op="conj")
    output = fs.flagsparse_spmv_csc(prepared=prepared, x=vector, out=target)
    _check(values, indices, ptr, vector, output)
    assert not calls


@pytest.mark.parametrize("op", ["non", "trans"])
def test_other_csc_operations_do_not_enable_compressed_route(op):
    values, indices, ptr, vector, _target = _inputs("integer", width=10)
    prepared = fs.prepare_spmv_csc(values, indices, ptr, (vector.numel(), 69), op=op)
    if prepared.maca_trans_plan is not None:
        assert not prepared.maca_trans_plan[-1].get("csc_c32_conj")


@pytest.mark.parametrize("n_cols", [0, 3])
def test_conjugate_empty_storage(n_cols):
    device = accelerator_device()
    values = torch.empty(0, dtype=torch.complex64, device=device)
    indices = torch.empty(0, dtype=torch.int64, device=device)
    ptr = torch.zeros(n_cols + 1, dtype=torch.int64, device=device)
    vector = torch.ones(7, dtype=torch.complex64, device=device)
    prepared = fs.prepare_spmv_csc(values, indices, ptr, (7, n_cols), op="conj")
    output = fs.flagsparse_spmv_csc(prepared=prepared, x=vector)
    torch.testing.assert_close(output.cpu(), torch.zeros(n_cols, dtype=torch.complex64))


@pytest.mark.parametrize("mode", ["integer", "dictionary", "full"])
def test_storage_rebuilds_when_conjugation_changes(mode):
    values, indices, ptr, vector, target = _inputs(mode, width=10)
    plan = ell._layout(indices, ptr, 69)
    cache = {}
    for conjugate in (False, True, False, True):
        previous = cache.get("ell_compressed")
        assert compressed.run(values, plan, vector.contiguous(), target, 69, cache, 16, 4, 10,
                              values._version, conjugate=conjugate)
        assert cache["ell_compressed"] is not previous
        assert cache["ell_compressed"]["conjugate"] == conjugate
        expected = _reference(values.conj() if conjugate else values, indices, ptr, vector)
        torch.testing.assert_close(target.cpu().to(torch.complex128), expected, rtol=1e-5, atol=1e-5)
