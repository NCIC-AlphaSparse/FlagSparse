# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark scripts must not lose the FlagSparse time to a broken torch op (MUSA).

results_musa_split_65_20261014 left two variants NoBaseline because the Python half
measured nothing: tests/test_spmm_csc.py crashed on torch.sparse.mm (MUSA registers
no sparse matmul) and benchmark_scatter_case raised on muDNN's missing int8 support.
Both run here on the CPU, no accelerator needed.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("triton")
test_spmm_csc = pytest.importorskip("test_spmm_csc")
benchmarks = pytest.importorskip("flagsparse.sparse_operations.benchmarks")


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.float16])
def test_scipy_csc_reference_matches_dense(op, dtype):
    gen = torch.Generator().manual_seed(3)
    dense = torch.randn(5, 4, generator=gen)
    dense[dense.abs() < 0.6] = 0
    if dtype.is_complex:
        dense = torch.complex(dense, torch.randn(5, 4, generator=gen) * (dense != 0))
    dense = dense.to(dtype)
    csc = dense.to(torch.float64 if not dtype.is_complex else torch.complex128)
    csc = csc.to_sparse_csc()
    data = csc.values().to(dtype)
    indices, indptr = csc.row_indices().int(), csc.ccol_indices()
    a_op = {"non": dense, "trans": dense.t(), "conj": dense.conj().t()}[op]
    B = torch.randn(a_op.shape[1], 3, generator=gen).to(dtype)
    got = test_spmm_csc._scipy_spmm_csc_reference(
        data, indices, indptr, B, (5, 4), dtype, op
    )
    wide = torch.complex128 if dtype.is_complex else torch.float64
    want = (a_op.to(wide) @ B.to(wide)).to(dtype)
    assert got.dtype == dtype and got.shape == want.shape
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


def test_scatter_benchmark_keeps_the_flagsparse_time_without_a_torch_baseline(
    monkeypatch,
):
    # The device is the CPU; FlagSparse's kernel is replaced by an exact CPU
    # scatter, and the PyTorch timing raises the way muDNN does for int8.
    monkeypatch.setattr(benchmarks, "_ACCEL_DEVICE_TYPE", "cpu")

    def fake_triton(values, indices, dense_size, out=None, reset_output=True, **kw):
        if reset_output:
            out.zero_()
        out.index_copy_(0, indices.to(torch.int64), values)
        meta = {
            "index_fallback_applied": False,
            "index_fallback_reason": None,
            "kernel_index_dtype": "int32",
        }
        return (out, meta) if kw.get("return_metadata") else out

    def mudnn(*_a, **_k):
        raise RuntimeError(
            "Now muDNN only support float64, float32, half, bfloat16, int32, and int64"
        )

    def prepare(values, indices, dense_size, **_kw):
        meta = {
            "effective_value_dtype": values.dtype,
            "fallback_applied": False,
            "fallback_reason": None,
        }
        return values, indices, indices, dense_size, meta

    monkeypatch.setattr(benchmarks, "_prepare_scatter_inputs", prepare)
    monkeypatch.setattr(benchmarks, "_triton_scatter_impl", fake_triton)
    monkeypatch.setattr(benchmarks, "_benchmark_cuda_graph_op", mudnn)
    monkeypatch.setattr(
        benchmarks, "_benchmark_scatter_triton_op", lambda op, **kw: (op(), 0.5)[1]
    )
    result = benchmarks.benchmark_scatter_case(
        dense_size=256,
        nnz=64,
        value_dtype=torch.int8,
        warmup=1,
        iters=1,
        run_cusparse=False,
    )
    assert result["performance"]["triton_ms"] == 0.5
    assert result["performance"]["pytorch_ms"] is None
    assert result["performance"]["triton_speedup_vs_pytorch"] is None
    assert "muDNN" in result["backend_status"]["pytorch_unavailable_reason"]
    assert result["verification"]["triton_match_pytorch"] is True
    assert result["verification"]["triton_max_error"] == 0.0
