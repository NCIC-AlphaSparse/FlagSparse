# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The SciPy oracle must not travel to the accelerator at fp64 (Iluvatar BI-V150).

CoreX 4.4 silently zeroes an fp64 host-to-device transfer and an on-device cast from
fp64. The spmm_coo and SDDMM benchmarks moved their promoted (fp64) SciPy reference
to the device and cast it there, so on BI-V150 every fp32 row compared a correct
kernel against an all-zero reference: status FAIL, error ~ |value| / atol, and, because
a FAILed row has no usable time, no speedup -- not even the scaled H800 one.

There is no CoreX card in CI, so the failure is emulated: any transfer through
``reference_utils`` at fp64/complex128 to a destination other than the literal
``"cpu"`` returns zeros. Written this way the tests fail on the old code and pass on
the new one; on a real CUDA/ROCm box the destinations are equivalent.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip(
    "torch", reason="tests/ci runs on a CPU-only runner without torch"
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

import reference_utils as ru  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sddmm_bench = _load("test_sddmm")
coo_bench = _load("test_spmm_coo")

WIDE = (torch.float64, torch.complex128)


@pytest.fixture
def corex(monkeypatch):
    """Emulate CoreX 4.4: a wide transfer to the accelerator arrives as zeros."""
    real_as_torch, real_as_torch_at = ru.as_torch, ru.as_torch_at

    def lossy(real):
        def transfer(array, dtype, device):
            out = real(array, dtype, device)
            on_accelerator = not (isinstance(device, str) and device == "cpu")
            return torch.zeros_like(out) if on_accelerator and dtype in WIDE else out

        return transfer

    monkeypatch.setattr(ru, "as_torch", lossy(real_as_torch))
    monkeypatch.setattr(ru, "as_torch_at", lossy(real_as_torch_at))
    for module in (sddmm_bench, coo_bench):
        monkeypatch.setattr(
            module.fs_common, "_use_scipy_accuracy_reference", lambda: True
        )


def _sddmm_case(beta_data=False):
    torch.manual_seed(0)
    n, k = 6, 4
    indptr = torch.tensor([0, 2, 3, 5, 5, 7, 8], dtype=torch.int32)
    indices = torch.tensor([0, 3, 2, 1, 4, 0, 5, 3], dtype=torch.int32)
    x, y = torch.randn(n, k), torch.randn(n, k)
    data = torch.randn(indices.numel()) if beta_data else None
    rows = torch.repeat_interleave(torch.arange(n), (indptr[1:] - indptr[:-1]).long())
    dots = (x[rows] * y[indices.long()]).sum(1)
    return data, indices, indptr, x, y, dots


def test_sddmm_oracle_is_not_zeroed_by_the_wide_transfer(corex):
    data, indices, indptr, x, y, dots = _sddmm_case()
    ref, _ = sddmm_bench._benchmark_reference_sddmm(
        data, indices, indptr, x, y, 1.0, 0.0, torch.float32, 0, 1
    )
    assert bool((ref != 0).any()), "the oracle arrived as zeros"
    assert torch.allclose(ref, dots, atol=1e-5)


def test_sddmm_oracle_with_beta_and_input_values(corex):
    data, indices, indptr, x, y, dots = _sddmm_case(beta_data=True)
    ref, _ = sddmm_bench._benchmark_reference_sddmm(
        data, indices, indptr, x, y, 2.0, 0.5, torch.float32, 0, 1
    )
    assert torch.allclose(ref, 2.0 * dots + 0.5 * data, atol=1e-5)


def test_sddmm_oracle_keeps_its_output_dtype(corex):
    data, indices, indptr, x, y, _ = _sddmm_case()
    ref, _ = sddmm_bench._benchmark_reference_sddmm(
        data, indices, indptr, x.half(), y.half(), 1.0, 0.0, torch.float16, 0, 1
    )
    assert ref.dtype == torch.float16


def _coo_prepared(dtype=torch.float32):
    torch.manual_seed(1)
    n = 5
    row = torch.tensor([0, 0, 1, 2, 4, 4], dtype=torch.int32)
    col = torch.tensor([0, 3, 1, 2, 0, 4], dtype=torch.int32)
    data, B = torch.randn(6, dtype=dtype), torch.randn(n, 3, dtype=dtype)
    dense = torch.zeros(n, n, dtype=dtype)
    dense[row.long(), col.long()] = data
    prepared = {
        "canonical_data": data,
        "canonical_row": row,
        "canonical_col": col,
        "canonical_B": B,
        "n_rows": n,
        "n_cols": n,
        "output_dtype": dtype,
        "native_coo": None,
        "native_B": B,
    }
    return (data, row, col, (n, n), B), prepared, dense @ B


def test_spmm_coo_oracle_is_not_zeroed_by_the_wide_transfer(corex, monkeypatch):
    monkeypatch.setattr(coo_bench.fs_common, "_is_iluvatar_runtime", lambda: False)
    args, prepared, truth = _coo_prepared()
    ref, _op, _fmt, _reason = coo_bench._build_pytorch_reference(
        *args, prepared=prepared
    )
    assert bool((ref != 0).any()), "the oracle arrived as zeros"
    assert ref.dtype == torch.float32
    assert torch.allclose(ref, truth, atol=1e-5)


def test_spmm_coo_has_no_pytorch_baseline_on_iluvatar(corex, monkeypatch):
    monkeypatch.setattr(coo_bench.fs_common, "_is_iluvatar_runtime", lambda: True)
    monkeypatch.setattr(coo_bench.fs_common, "_is_mthreads_runtime", lambda: False)
    args, prepared, truth = _coo_prepared()
    ref, op, _fmt, reason = coo_bench._build_pytorch_reference(*args, prepared=prepared)
    assert op is None, "a latency measured on an unverified op is not a baseline"
    assert "Iluvatar" in reason
    assert torch.allclose(ref, truth, atol=1e-5)  # the oracle itself still works


def test_spmm_coo_keeps_its_pytorch_baseline_elsewhere(corex, monkeypatch):
    monkeypatch.setattr(coo_bench.fs_common, "_is_iluvatar_runtime", lambda: False)
    monkeypatch.setattr(coo_bench.fs_common, "_is_mthreads_runtime", lambda: False)
    args, prepared, _ = _coo_prepared()
    _ref, op, _fmt, reason = coo_bench._build_pytorch_reference(
        *args, prepared=prepared
    )
    assert callable(op) and reason is None


def test_as_torch_at_casts_on_the_cpu_and_moves_once():
    import numpy as np

    out = ru.as_torch_at(np.array([1.5, 2.5], dtype=np.float64), torch.float32, "cpu")
    assert out.dtype == torch.float32 and out.tolist() == [1.5, 2.5]
    c = ru.as_torch_at(np.array([1 + 2j], dtype=np.complex128), torch.complex64, "cpu")
    assert c.dtype == torch.complex64
