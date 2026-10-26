"""CPU-only checks of the Iluvatar f32 CSR row-tile config (read once at prepare)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")
spmv_csr = pytest.importorskip("flagsparse.sparse_operations.spmv_csr")

_ENV = (
    "FLAGSPARSE_ILUVATAR_SPMV_F32_ROUTE",
    "FLAGSPARSE_ILUVATAR_F32_ROWS",
    "FLAGSPARSE_ILUVATAR_F32_LANES",
    "FLAGSPARSE_ILUVATAR_F32_WARPS",
    "FLAGSPARSE_ILUVATAR_F32_STAGES",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def test_default_config_and_fixed_steps():
    config = spmv_csr._iluvatar_f32_row_tile_config
    assert config(8) == (16, 8, 1, 1, 1)
    assert config(16) == (16, 8, 1, 1, 2)
    assert config(17) == (16, 8, 1, 1, 0)


def test_segbin_disables_row_tile(monkeypatch):
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_SPMV_F32_ROUTE", "segbin")
    assert spmv_csr._iluvatar_f32_row_tile_config(8) is None


def test_overrides_and_validation(monkeypatch):
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_F32_ROWS", "32")
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_F32_LANES", "4")
    assert spmv_csr._iluvatar_f32_row_tile_config(8) == (32, 4, 1, 1, 2)
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_F32_LANES", "6")
    with pytest.raises(ValueError, match="power of two"):
        spmv_csr._iluvatar_f32_row_tile_config(8)
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_F32_LANES", "64")
    with pytest.raises(ValueError, match="<= 32"):
        spmv_csr._iluvatar_f32_row_tile_config(8)
