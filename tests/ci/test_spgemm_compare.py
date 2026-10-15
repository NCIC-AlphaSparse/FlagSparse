# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU regressions for the SpGEMM sparse-result comparator."""

import pytest

# CI's tests/ci job runs without torch (or triton); skip there like the rest of
# tests/ci instead of failing collection.
torch = pytest.importorskip("torch")
_spgemm_compare_metrics = pytest.importorskip(
    "tests.test_spgemm"
)._spgemm_compare_metrics


def _csr(data, indices, indptr, shape, dtype):
    return (
        torch.tensor(data, dtype=dtype),
        torch.tensor(indices, dtype=torch.int32),
        torch.tensor(indptr, dtype=torch.int64),
        shape,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spgemm_comparison_ignores_explicit_zeros(dtype):
    candidate = _csr(
        [1.0, 0.0, 0.0, 2.0],
        [0, 2, 1, 2],
        [0, 2, 4],
        (2, 3),
        dtype,
    )
    reference = _csr([1.0, 2.0], [0, 2], [0, 1, 2], (2, 3), dtype)

    metrics = _spgemm_compare_metrics(candidate, reference, dtype)

    assert metrics["pattern_ok"]
    assert metrics["pass"]
    assert metrics["reason"] == "ok"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spgemm_comparison_accepts_empty_after_zero_elision(dtype):
    candidate = _csr([0.0, 0.0], [0, 2], [0, 1, 2], (2, 3), dtype)
    reference = _csr([], [], [0, 0, 0], (2, 3), dtype)

    metrics = _spgemm_compare_metrics(candidate, reference, dtype)

    assert metrics["pattern_ok"]
    assert metrics["pass"]
    assert metrics["err_ratio"] == 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spgemm_comparison_still_rejects_nonzero_pattern_mismatch(dtype):
    candidate = _csr([1.0], [0], [0, 1], (1, 3), dtype)
    reference = _csr([1.0], [1], [0, 1], (1, 3), dtype)

    metrics = _spgemm_compare_metrics(candidate, reference, dtype)

    assert not metrics["pattern_ok"]
    assert not metrics["pass"]
    assert "sparsity pattern mismatch" in metrics["reason"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_spgemm_comparison_still_checks_values(dtype):
    candidate = _csr([1.0], [0], [0, 1], (1, 1), dtype)
    reference = _csr([2.0], [0], [0, 1], (1, 1), dtype)

    metrics = _spgemm_compare_metrics(candidate, reference, dtype)

    assert metrics["pattern_ok"]
    assert not metrics["pass"]
    assert metrics["reason"] == "value mismatch"
