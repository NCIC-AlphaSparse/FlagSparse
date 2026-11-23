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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
def test_ascend_rectangular_rhs_transpose_keeps_dtype(monkeypatch, dtype):
    from tests import test_spgemm as benchmark

    monkeypatch.setattr(benchmark.ast_common, "_is_ascend_runtime", lambda: True)
    dense = torch.tensor([[1, 0, 2], [0, 3, 4]], dtype=dtype)
    sparse = dense.to_sparse_csr()

    values, columns, pointer, shape = benchmark._build_spgemm_rhs(
        sparse.values(),
        sparse.col_indices().int(),
        sparse.crow_indices(),
        tuple(dense.shape),
        "A_AT",
    )

    result = torch.sparse_csr_tensor(pointer, columns.long(), values, shape).to_dense()
    assert values.dtype == dtype
    assert columns.dtype == torch.int32
    assert pointer.dtype == torch.int64
    torch.testing.assert_close(result, dense.t())


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


@pytest.mark.parametrize("extra_zero", [False, True])
def test_spgemm_comparison_checks_shared_zero_against_subnormal(extra_zero):
    dtype = torch.float32
    values = [0.0, 0.0] if extra_zero else [0.0]
    candidate = _csr(values, list(range(len(values))), [0, len(values)], (1, 2), dtype)
    reference = _csr([1e-42], [0], [0, 1], (1, 2), dtype)

    for left, right in [(candidate, reference), (reference, candidate)]:
        metrics = _spgemm_compare_metrics(left, right, dtype)
        assert metrics["pass"]
        assert metrics["max_abs_error"] == reference[0][0].item()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_spgemm_comparison_rejects_shared_zero_against_nonfinite(value):
    dtype = torch.float32
    zero = _csr([0.0], [0], [0, 1], (1, 1), dtype)
    other = _csr([value], [0], [0, 1], (1, 1), dtype)

    for left, right in [(zero, other), (other, zero)]:
        metrics = _spgemm_compare_metrics(left, right, dtype)
        assert not metrics["pass"]
        assert metrics["reason"] == "value mismatch"
