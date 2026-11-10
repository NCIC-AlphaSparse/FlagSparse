from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
spgemm = pytest.importorskip("flagsparse.sparse_operations.spgemm_csr")


@pytest.mark.parametrize("iluvatar", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
def test_iluvatar_standard_triton_dispatch(monkeypatch, iluvatar, dtype):
    from flagsparse.sparse_operations import _spgemm_iluvatar

    monkeypatch.setattr(_spgemm_iluvatar, "compute", lambda plan: None)
    monkeypatch.setattr(spgemm, "_is_iluvatar_runtime", lambda: iluvatar)
    monkeypatch.setattr(spgemm, "_is_maca_runtime", lambda: False)
    monkeypatch.setattr(spgemm, "_is_mthreads_runtime", lambda: False)
    monkeypatch.setattr(spgemm, "_is_xpu_runtime", lambda: False)
    monkeypatch.setattr(spgemm, "_TLE_AVAILABLE", True)
    monkeypatch.setattr(spgemm, "_maca_triton_spgemm_compute", lambda plan: "triton")
    monkeypatch.setattr(spgemm, "_spgemm_esc_compute_safe", lambda plan: "esc")
    monkeypatch.setattr(spgemm, "_spgemm_hash_hybrid_compute", lambda plan: "hash")
    plan = object.__new__(spgemm.SpGEMMPrepared)
    plan.a_data = SimpleNamespace(dtype=dtype)
    expected = "hash" if not iluvatar else "esc" if dtype == torch.float64 else "triton"
    assert spgemm._spgemm_compute(plan) == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_iluvatar_symbolic_dispatch(monkeypatch, dtype):
    from flagsparse.sparse_operations import _spgemm_iluvatar

    monkeypatch.setattr(spgemm, "_is_iluvatar_runtime", lambda: True)
    monkeypatch.setattr(_spgemm_iluvatar, "compute", lambda plan: "symbolic")
    plan = object.__new__(spgemm.SpGEMMPrepared)
    plan.a_data = SimpleNamespace(dtype=dtype)
    assert spgemm._spgemm_compute(plan) == "symbolic"
