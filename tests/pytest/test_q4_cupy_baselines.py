# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Correctness and coverage of the CuPy supplementary benchmark paths."""
from pathlib import Path

import pytest
import torch

from tests.pytest.accuracy_utils import accelerator_available
from flagsparse.sparse_operations._common import _backend_name

pytestmark = pytest.mark.skipif(not accelerator_available() or _backend_name() != "cuda",
                                reason="NVIDIA CUDA and CuPy required")


@pytest.mark.parametrize("op,dtype", [("spvv", torch.float16), ("spvv", torch.int8),
                                      ("spvv", torch.complex64), ("axpby", torch.float16)])
def test_vector_cupy_baseline_precision(op, dtype, monkeypatch):
    pytest.importorskip("cupy")
    from tools import benchmark_q4_baselines as bench
    monkeypatch.setenv("FLAGSPARSE_BENCH_CUPY", "1")
    row = bench.vector._run_case(op, dtype, "conj" if dtype.is_complex else "non",
                                 32768, 1024, 1, 1, torch.Generator().manual_seed(0))
    assert row["status"] == "PASS"
    assert row.get("cupy_ms", 0) > 0, row.get("cupy_reason")
    assert row["triton_speedup_vs_cupy"] > 0


def test_scatter_cupy_graph_correctness():
    pytest.importorskip("cupy")
    from tools import benchmark_q4_baselines as bench
    row = bench.run_scatter(32768, 1024, 1, 1, torch.Generator().manual_seed(0))
    assert row["status"] == "PASS" and row["cupy_max_error"] == 0
    assert row["cupy_ms"] > 0 and row["speedup_vs_cupy"] > 0


def test_real_sparse_complex_vector_cupy_retains_imaginary_part(monkeypatch):
    pytest.importorskip("cupy")
    from tools import benchmark_q4_baselines as bench
    q4 = bench.q4
    monkeypatch.setenv("FLAGSPARSE_BENCH_CUPY", "1")
    selected = [v for v in q4.VARIANTS["spmv_csr"] if v.name == "spmv_csr_f32c32_int_non"]
    monkeypatch.setitem(q4.VARIANTS, "spmv_csr", selected)
    path = Path(__file__).resolve().parents[1] / "data/q4_worker_smoke.mtx"
    row, = q4.run_variants("spmv_csr", [path], 1, 1)
    assert row["status"] == "PASS"
    assert row["cupy_compute_dtype"] == "complex64"
    assert row.get("cupy_ms", 0) > 0, row.get("cupy_reason")
