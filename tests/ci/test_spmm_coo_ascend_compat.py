# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise Ascend COO compatibility behavior with the pure-Torch fallback."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from flagsparse.sparse_operations import spmm_coo as impl  # noqa: E402


def _fake_prepare_inputs(data, row, col, B, shape, dense_layout="row"):
    return (
        data.contiguous(),
        row.to(torch.int32).contiguous(),
        col.to(torch.int32).contiguous(),
        B.contiguous(),
        int(shape[0]),
        int(shape[1]),
        int(B.shape[1]),
    )


def _configure_ascend_dispatch(monkeypatch):
    monkeypatch.setattr(impl, "_is_ascend_runtime", lambda: True)
    monkeypatch.setattr(impl, "_is_xpu_runtime", lambda: False)
    monkeypatch.setattr(impl, "_use_spmm_coo_ascend_dispatch", lambda: True)
    monkeypatch.setattr(impl, "_prepare_spmm_coo_inputs", _fake_prepare_inputs)
    monkeypatch.setattr(
        impl,
        "_coalesce_coo_entries",
        lambda *args, **kwargs: pytest.fail("Ascend route called sparse coalesce"),
    )
    monkeypatch.setattr(impl, "_ACCEL", SimpleNamespace(synchronize=lambda: None))


def test_public_spmm_coo_float64_sums_duplicates_without_sparse_coalesce(monkeypatch):
    _configure_ascend_dispatch(monkeypatch)
    data = torch.tensor([1.25, -0.5, 2.0, 0.75], dtype=torch.float64)
    row = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    col = torch.tensor([2, 1, 2, 3], dtype=torch.int32)
    B = torch.arange(12, dtype=torch.float64).reshape(4, 3)

    actual = impl.flagsparse_spmm_coo(data, row, col, B, (2, 4))
    dense = torch.zeros((2, 4), dtype=torch.float64)
    dense.index_put_((row.to(torch.int64), col.to(torch.int64)), data, accumulate=True)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, dense @ B)


def test_prepared_spmm_coo_route_preserves_duplicates_without_coalesce(monkeypatch):
    _configure_ascend_dispatch(monkeypatch)
    monkeypatch.setattr(
        impl,
        "_prepare_spmm_coo_matrix",
        lambda data, row, col, shape: (data, row, col, shape),
    )
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    row = torch.tensor([1, 0, 1], dtype=torch.int32)
    col = torch.tensor([2, 1, 2], dtype=torch.int32)

    prepared = impl.prepare_spmm_coo_route(data, row, col, (2, 3))

    assert prepared.nnz == 3
    assert prepared.row.tolist() == [0, 1, 1]
    assert prepared.col.tolist() == [1, 2, 2]


def test_prepared_alg1_route_uses_scatter_for_duplicate_float64_entries(monkeypatch):
    _configure_ascend_dispatch(monkeypatch)
    monkeypatch.setattr(
        impl, "_validate_spmm_coo_route_runtime_inputs", lambda prepared, B, layout: B
    )
    prepared = SimpleNamespace(
        output_dtype=torch.float64,
        data=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        row=torch.tensor([0, 1, 1], dtype=torch.int32),
        col=torch.tensor([1, 2, 2], dtype=torch.int32),
        n_rows=2,
        op="non",
    )
    B = torch.arange(9, dtype=torch.float64).reshape(3, 3)

    actual, meta = impl._run_spmm_coo_alg1_route(prepared, B)

    expected = torch.zeros((2, 3), dtype=torch.float64)
    expected.index_add_(
        0,
        prepared.row.to(torch.int64),
        prepared.data[:, None] * B[prepared.col.to(torch.int64)],
    )
    torch.testing.assert_close(actual, expected)
    assert meta["alg"] == "coo_ascend_scatter"
