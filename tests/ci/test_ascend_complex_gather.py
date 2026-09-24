# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Complex gather must avoid sending a complex dtype to Ascend indexing APIs."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flagsparse.sparse_operations import gather_scatter as impl


def _benchmark_module():
    path = ROOT / "benchmark" / "benchmark_ascend.py"
    spec = importlib.util.spec_from_file_location("_ascend_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_ascend_complex_gather_splits_real_and_imag_and_supports_out(
    monkeypatch, dtype
):
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    dense = torch.complex(
        torch.arange(8, dtype=real_dtype),
        -torch.arange(8, dtype=real_dtype) - 0.5,
    )
    indices = torch.tensor([5, 1, 5, 0], dtype=torch.int32)
    expected = dense[indices.to(torch.int64)]

    monkeypatch.setattr(impl, "_is_ascend_runtime", lambda: True)
    monkeypatch.setattr(impl, "_is_accel_tensor", lambda tensor: True)
    result = impl.flagsparse_gather(dense, indices)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)

    out = torch.empty_like(expected)
    actual_out = impl.flagsparse_gather(dense, indices, out=out)
    assert actual_out is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_pytorch_complex_gather_baseline_uses_component_offsets(dtype):
    benchmark = _benchmark_module()
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    dense = torch.complex(
        torch.arange(8, dtype=real_dtype), torch.ones(8, dtype=real_dtype)
    )
    indices = torch.tensor([7, 2, 0], dtype=torch.int32)

    actual = benchmark._torch_gather(torch, dense, indices)

    torch.testing.assert_close(actual, dense[indices.to(torch.int64)], rtol=0, atol=0)


def test_coo_pytorch_baselines_match_scipy_for_duplicate_coordinates():
    benchmark = _benchmark_module()
    row = torch.tensor([1, 0, 1, 1], dtype=torch.int32)
    col = torch.tensor([2, 1, 2, 0], dtype=torch.int32)
    data = torch.tensor([1.5, -2.0, 0.25, 3.0], dtype=torch.float64)
    x = torch.arange(3, dtype=torch.float64)
    B = torch.arange(6, dtype=torch.float64).reshape(3, 2)

    spmv = benchmark._torch_npu_coo_spmv(data, row, col, x, 2)
    spmm = benchmark._torch_npu_coo_spmm(data, row, col, B, 2)

    expected_spmv = torch.tensor([-2.0, 3.5], dtype=torch.float64)
    expected_spmm = torch.tensor([[-4.0, -6.0], [7.0, 11.75]], dtype=torch.float64)
    torch.testing.assert_close(spmv, expected_spmv)
    torch.testing.assert_close(spmm, expected_spmm)


def test_benchmark_refuses_to_time_mismatched_candidate_or_baseline():
    benchmark = _benchmark_module()
    correct = torch.tensor([1.0, 2.0]).numpy()
    wrong = torch.tensor([1.0, 3.0]).numpy()

    assert benchmark._outputs_pass_accuracy(correct, correct, torch.float32)
    assert not benchmark._outputs_pass_accuracy(wrong, correct, torch.float32)
    assert benchmark._outputs_pass_accuracy(
        np.array([np.nan, np.inf]),
        np.array([np.nan, np.inf]),
        torch.float32,
    )
