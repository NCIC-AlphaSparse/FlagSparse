"""CPU-only checks of the Iluvatar COO c32 route selection (no GPU kernels run)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
spmv_coo = pytest.importorskip("flagsparse.sparse_operations.spmv_coo")


def test_iluvatar_short_complex_coo_uses_nnz_plan(monkeypatch):
    monkeypatch.setattr(spmv_coo, "_is_iluvatar_runtime", lambda: True)
    monkeypatch.setattr(spmv_coo, "_backend_name", lambda: "iluvatar")

    n_rows = 256
    row_lengths = torch.ones(n_rows, dtype=torch.long)
    row_lengths[37] = 210
    row = torch.repeat_interleave(torch.arange(n_rows, dtype=torch.int32), row_lengths)
    col = torch.arange(row.numel(), dtype=torch.int32) % 1024
    data = torch.ones(row.numel(), dtype=torch.complex64)
    prepared = spmv_coo.PreparedCoo(
        data,
        row,
        col,
        (n_rows, 1024),
    )

    assert isinstance(prepared.non_csr_plan, spmv_coo._CooNnzPlan)
    assert prepared.non_csr_plan.ptr.numel() == n_rows + 1
    assert prepared.non_csr_plan.rows.numel() == row.numel()
    assert prepared.non_csr_plan.empty.numel() == 0


def test_iluvatar_coo_c32_row_tile_overrides(monkeypatch):
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_COO_C32_ROWS", "16")
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_COO_C32_BLOCK", "8")
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_COO_C32_WARPS", "2")
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_COO_C32_STAGES", "3")

    tile = spmv_coo._iluvatar_coo_c32_row_tile(1000, 256, 210)
    assert tile == (16, 8, 2, 3)


def test_iluvatar_coo_c32_row_tile_adapts_to_long_tail():
    assert spmv_coo._iluvatar_coo_c32_row_tile(900, 256, 210)[:2] == (16, 8)
    assert spmv_coo._iluvatar_coo_c32_row_tile(700, 256, 12)[:2] == (32, 2)


def test_iluvatar_f32_transpose_uses_torch_scatter_for_long_tail(monkeypatch):
    monkeypatch.setattr(spmv_coo, "_is_iluvatar_runtime", lambda: True)
    monkeypatch.setattr(spmv_coo, "_backend_name", lambda: "iluvatar")

    n_rows = 256
    row_lengths = torch.ones(n_rows, dtype=torch.long)
    row_lengths[37] = 210
    row = torch.repeat_interleave(torch.arange(n_rows, dtype=torch.int32), row_lengths)
    col = torch.arange(row.numel(), dtype=torch.int32) % n_rows
    data = torch.ones(row.numel(), dtype=torch.float32)
    prepared = spmv_coo.PreparedCoo(
        data,
        row,
        col,
        (n_rows, n_rows),
        data_trans=data,
        row_trans=row,
        col_trans=col,
        transpose=True,
        op="trans",
    )

    assert isinstance(prepared.transpose_torch_plan, spmv_coo._CooTorchScatterPlan)
    assert prepared.transpose_torch_plan.rows.dtype == torch.int64
    assert prepared.transpose_torch_plan.cols.dtype == torch.int64


def test_iluvatar_f32_non_transpose_does_not_use_complex_plan(monkeypatch):
    monkeypatch.setattr(spmv_coo, "_is_iluvatar_runtime", lambda: True)
    monkeypatch.setattr(spmv_coo, "_backend_name", lambda: "iluvatar")

    row = torch.arange(8, dtype=torch.int32)
    col = torch.arange(8, dtype=torch.int32)
    data = torch.ones(8, dtype=torch.float32)
    prepared = spmv_coo.PreparedCoo(data, row, col, (8, 8))

    assert prepared.non_csr_plan is None
