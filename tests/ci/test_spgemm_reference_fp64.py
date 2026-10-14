# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The SpGEMM SciPy reference never moves fp64 data to the device.

BI-V150 (CoreX) zeroes an fp64 H2D copy. tests/test_spgemm.py built its reference
with ``as_torch(data, float64, device).to(float32)``, so on Iluvatar every value
arrived as 0, was dropped as an explicit zero, and all ten rows failed with
"sparsity pattern mismatch" (results_iluvatar_65_1014_spgemm_f32).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
reference_utils = pytest.importorskip("reference_utils")
test_spgemm = pytest.importorskip("tests.test_spgemm")


def test_reference_survives_a_device_that_zeroes_fp64(monkeypatch):
    real = reference_utils.as_torch

    def biv150_as_torch(array, dtype, device):
        out = real(array, dtype, device)
        return torch.zeros_like(out) if dtype == torch.float64 else out

    monkeypatch.setattr(reference_utils, "as_torch", biv150_as_torch)
    a = torch.tensor([[1.0, 2.0], [0.0, 3.0]]).to_sparse_csr()
    data, indices, indptr = a.values(), a.col_indices().int(), a.crow_indices()
    result = test_spgemm._build_scipy_spgemm_reference(
        data, indices, indptr, (2, 2), data, indices, indptr, (2, 2)
    )
    values = result[0][0] if isinstance(result[0], tuple) else result[0]
    # A @ A = [[1, 8], [0, 9]]: three stored values, none of them zero
    assert values.dtype == torch.float32
    assert sorted(values.tolist()) == [1.0, 8.0, 9.0]
