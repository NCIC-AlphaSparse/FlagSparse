# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""tools/sddmm_sweep.py: the CPU oracle and the launch-config override it relies on."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip(
    "torch", reason="tests/ci runs on a CPU-only runner without torch"
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import sddmm_sweep  # noqa: E402


def _case(n_rows=7, n_cols=9, k=5, seed=0):
    torch.manual_seed(seed)
    dense_mask = torch.rand(n_rows, n_cols) < 0.4
    dense_mask[3] = False  # an empty row
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64)
    indptr[1:] = torch.cumsum(dense_mask.sum(1), 0)
    indices = dense_mask.nonzero()[:, 1].to(torch.int32)
    x, y = torch.randn(n_rows, k), torch.randn(n_cols, k)
    truth = (x @ y.T)[dense_mask]
    return indices, indptr, x, y, truth.to(torch.float64)


def test_cpu_reference_matches_the_dense_product():
    indices, indptr, x, y, truth = _case()
    got = sddmm_sweep._cpu_reference(indices, indptr, x, y)
    assert got.dtype == torch.float64
    assert torch.allclose(got, truth, atol=1e-5)


@pytest.mark.parametrize("chunk", [1, 3, 7, 1000])
def test_cpu_reference_is_independent_of_the_chunk_size(chunk):
    indices, indptr, x, y, truth = _case(seed=1)
    got = sddmm_sweep._cpu_reference(indices, indptr, x, y, chunk=chunk)
    assert torch.allclose(got, truth, atol=1e-5)


def test_cpu_reference_of_an_empty_pattern():
    indices = torch.zeros(0, dtype=torch.int32)
    indptr = torch.zeros(4, dtype=torch.int64)
    got = sddmm_sweep._cpu_reference(
        indices, indptr, torch.randn(3, 2), torch.randn(3, 2)
    )
    assert got.numel() == 0


def test_verdict_rejects_a_zero_or_nonfinite_result():
    _, _, _, _, truth = _case()
    assert sddmm_sweep._verdict(truth.float(), truth)[0] == "PASS"
    assert sddmm_sweep._verdict(torch.zeros_like(truth).float(), truth)[0] == "FAIL"
    bad = truth.float().clone()
    bad[0] = float("nan")
    assert sddmm_sweep._verdict(bad, truth)[0] == "FAIL"


def test_force_and_restore_the_launch_config():
    original = sddmm_sweep.sddmm_mod._resolve_sddmm_launch_config
    sddmm_sweep._force((16, 8, 2))
    try:
        forced = sddmm_sweep.sddmm_mod._resolve_sddmm_launch_config(
            256, mean_row_len=40.0, value_dtype=torch.float32
        )
        assert forced == (16, 8, 2)
    finally:
        sddmm_sweep._restore()
    assert sddmm_sweep.sddmm_mod._resolve_sddmm_launch_config is original


def test_the_default_config_is_the_one_the_operator_would_pick():
    default = sddmm_sweep._DEFAULT_RESOLVE(
        256, mean_row_len=25.0, value_dtype=torch.float32
    )
    assert default == (512, 32, 4)  # the wide branch the BI-V150 sweep is about
    assert sddmm_sweep._DEFAULT_RESOLVE(
        256, mean_row_len=3.0, value_dtype=torch.float32
    ) == (64, 32, 8)


def test_int_list_parsing():
    assert sddmm_sweep._ints("32, 64,,128") == [32, 64, 128]
