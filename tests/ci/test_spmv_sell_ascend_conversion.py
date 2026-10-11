# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for Ascend's complex CSR-to-SELL conversion."""

import pytest

torch = pytest.importorskip("torch")

from flagsparse.sparse_operations import spmv_sell  # noqa: E402


def test_ascend_complex_csr_to_sell_avoids_complex_index_put(monkeypatch):
    monkeypatch.setattr(spmv_sell, "_is_ascend_runtime", lambda: True)
    data = torch.tensor([1 + 2j, 3 - 4j, 5 + 6j], dtype=torch.complex64)
    indices = torch.tensor([0, 2, 1], dtype=torch.int32)
    indptr = torch.tensor([0, 2, 3], dtype=torch.int32)

    values, cols, offsets = spmv_sell.csr_to_sell(
        data, indices, indptr, n_rows=2, slice_size=2
    )

    torch.testing.assert_close(values, torch.tensor([1 + 2j, 5 + 6j, 3 - 4j, 0j]))
    assert torch.equal(cols, torch.tensor([0, 1, 2, -1], dtype=torch.int32))
    assert torch.equal(offsets, torch.tensor([0, 4], dtype=torch.int32))
