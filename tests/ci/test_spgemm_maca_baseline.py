# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA performance references stay independent of the SciPy accuracy oracle."""

import pytest

torch = pytest.importorskip("torch")
bench = pytest.importorskip("tests.test_spgemm")


def _case(dtype=torch.float32):
    a = torch.tensor([[1.0, 2.0], [0.0, 3.0]], dtype=dtype).to_sparse_csr()
    args = (a.values(), a.col_indices().int(), a.crow_indices(), (2, 2))
    oracle = bench._build_scipy_spgemm_reference(*args, *args)
    return args, oracle


def _timer(monkeypatch):
    monkeypatch.setattr(
        bench.ast_ops, "_benchmark_cuda_op", lambda op, warmup, iters: (op(), 2.0)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_csr_baseline_is_timed_and_validated(monkeypatch, dtype):
    _timer(monkeypatch)
    args, oracle = _case(dtype)
    state = bench._benchmark_maca_torch_spgemm_baseline(*args, *args, oracle, 1, 3)
    assert state["pytorch_baseline_status"] == "PASS"
    assert state["pytorch_baseline_format"] == "CSR"
    assert state["pytorch_ms"] == 2.0
    assert state["pytorch_reason"] is None


@pytest.mark.parametrize("value", [99.0, float("nan"), float("inf")])
def test_wrong_baseline_never_publishes_latency(monkeypatch, value):
    _timer(monkeypatch)
    args, oracle = _case()
    bad = torch.full((2, 2), value).to_sparse_csr()
    monkeypatch.setattr(torch.sparse, "mm", lambda a, b: bad)
    state = bench._benchmark_maca_torch_spgemm_baseline(*args, *args, oracle, 0, 1)
    assert state["pytorch_baseline_status"] == "FAIL"
    assert state["pytorch_ms"] is None
    assert state["pytorch_reason"]


def test_baseline_exception_is_reported(monkeypatch):
    _timer(monkeypatch)
    args, oracle = _case()

    def fail(*args):
        raise RuntimeError("sparse multiply failed")

    monkeypatch.setattr(torch.sparse, "mm", fail)
    state = bench._benchmark_maca_torch_spgemm_baseline(*args, *args, oracle, 0, 1)
    assert state["pytorch_ms"] is None
    assert state["pytorch_baseline_status"] == "ERROR"
    assert state["pytorch_reason"] == "RuntimeError: sparse multiply failed"


@pytest.mark.parametrize("latency", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_latency_is_not_a_baseline(monkeypatch, latency):
    args, oracle = _case()
    monkeypatch.setattr(
        bench.ast_ops, "_benchmark_cuda_op", lambda op, warmup, iters: (op(), latency)
    )
    state = bench._benchmark_maca_torch_spgemm_baseline(*args, *args, oracle, 0, 1)
    assert state["pytorch_ms"] is None
    assert state["pytorch_baseline_status"] == "ERROR"


def test_baseline_validates_the_timed_output(monkeypatch):
    args, oracle = _case()
    bad = torch.ones((2, 2)).to_sparse_csr()

    def timer(op, warmup, iters):
        op()
        return bad, 2.0

    monkeypatch.setattr(bench.ast_ops, "_benchmark_cuda_op", timer)
    state = bench._benchmark_maca_torch_spgemm_baseline(*args, *args, oracle, 1, 1)
    assert state["pytorch_ms"] is None
    assert state["pytorch_baseline_status"] == "FAIL"


@pytest.mark.parametrize(
    "maca,vendor,enabled",
    [(True, "torch", True), (True, None, False), (False, "torch", False)],
)
@pytest.mark.parametrize("baseline_ok", [True, False])
def test_matrix_runner_keeps_accuracy_separate(
    monkeypatch, maca, vendor, enabled, baseline_ok
):
    args, oracle = _case()
    monkeypatch.setattr(bench, "accelerator_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(bench, "load_mtx_to_csr_torch", lambda *a, **kw: args)
    monkeypatch.setattr(bench.ast_common, "_use_scipy_accuracy_reference", lambda: True)
    monkeypatch.setattr(bench.ast_common, "_is_maca_runtime", lambda: maca)
    monkeypatch.setattr(
        bench.ast_common, "_expected_vendor_sparse_backend", lambda: vendor
    )
    monkeypatch.setattr(bench, "_cleanup_reference_pools", lambda: None)
    monkeypatch.setattr(
        bench, "_benchmark_flagsparse_spgemm", lambda *a, **kw: (oracle, 1.0, 1.0, {})
    )
    calls = []

    def baseline(*a):
        calls.append(a)
        return {
            "pytorch_ms": 2.0 if baseline_ok else None,
            "pytorch_baseline_status": "PASS" if baseline_ok else "FAIL",
            "pytorch_reason": None if baseline_ok else "wrong baseline",
        }

    monkeypatch.setattr(bench, "_benchmark_maca_torch_spgemm_baseline", baseline)
    result = bench.run_one_mtx("mock.mtx", run_cusparse=False)
    assert bool(calls) == enabled
    assert result["status"] == "PASS"
    assert result["triton_ok_pt"]
    assert result["error"] is None
    assert result["pytorch_ms"] == (2.0 if enabled and baseline_ok else None)
    row = bench._spgemm_csv_row(result, torch.float32, torch.int32)
    assert row["triton_speedup_vs_pytorch"] == (
        2.0 if enabled and baseline_ok else None
    )
