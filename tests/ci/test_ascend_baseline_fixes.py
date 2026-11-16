from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from flagsparse.sparse_operations import sddmm_csr  # noqa: E402


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("provided_output", [False, True])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (0.731, -0.317), (-0.25, 1.5)])
@pytest.mark.parametrize("transposed", [False, True])
def test_sddmm_half_accumulates_before_narrowing(
    monkeypatch, prepared, output_dtype, provided_output, alpha, beta, transposed
):
    monkeypatch.setattr(sddmm_csr, "_is_ascend_runtime", lambda: True)
    monkeypatch.setattr(sddmm_csr, "_is_accel_tensor", lambda tensor: True)
    monkeypatch.setattr(sddmm_csr, "_ACCEL", SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(sddmm_csr, "_ASCEND_SDDMM_CHUNK_NNZ", 2)
    generator = torch.Generator().manual_seed(97)
    lhs = torch.randn(5, 63, generator=generator).half()
    rhs = torch.randn(7, 63, generator=generator).half()
    values = torch.randn(6, generator=generator).half()
    indices = torch.tensor([0, 5, 1, 2, 4, 6], dtype=torch.int32)
    pointer = torch.tensor([0, 2, 2, 3, 5, 6], dtype=torch.int32)
    rows = torch.tensor([0, 0, 2, 3, 3, 4])
    expected = (
        (lhs.double()[rows] * rhs.double()[indices.long()]).sum(1) * alpha
        + beta * values.double()
    ).to(output_dtype)
    plan = (
        sddmm_csr.prepare_sddmm_csr(indices, pointer, (5, 7), k_hint=63)
        if prepared
        else None
    )
    target = torch.empty_like(expected) if provided_output else None
    result = sddmm_csr.flagsparse_sddmm_csr(
        values,
        indices,
        pointer,
        lhs.t() if transposed else lhs,
        rhs if transposed else rhs.t(),
        shape=(5, 7),
        prepared=plan,
        alpha=alpha,
        beta=beta,
        out=target,
        op_a="trans" if transposed else "non",
        op_b="trans" if transposed else "non",
        out_dtype=output_dtype,
    )
    assert result.dtype == output_dtype
    if provided_output:
        assert result is target
    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)


def _baseline_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    import ascend_axpby_baseline
    import ascend_gemvi_geam_baseline

    return ascend_axpby_baseline, ascend_gemvi_geam_baseline


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, -0.5])
@pytest.mark.parametrize("calls", [0, 1, 2, 27])
def test_axpby_iterative_reference(monkeypatch, beta, calls):
    baseline, _ = _baseline_modules(monkeypatch)
    initial = torch.tensor([1 + 2j, -3j, 4, 5j], dtype=torch.complex64)
    values = torch.tensor([2j, -1 + 1j, 0.5j], dtype=torch.complex64)
    indices = torch.tensor([3, 0, 3], dtype=torch.int32)
    expected = initial.cdouble()
    for _ in range(calls):
        expected.mul_(beta).index_add_(0, indices.long(), values.cdouble() * 1.5)
    torch.testing.assert_close(
        baseline.expected_after(initial, values, indices, 1.5, beta, calls), expected
    )


@pytest.mark.parametrize("broken", [False, True])
def test_axpby_measure_checks_mutable_state_and_fails_closed(monkeypatch, broken):
    baseline, delivery = _baseline_modules(monkeypatch)
    plans = []
    initial = torch.tensor([1 + 2j, -3j, 4, 5j], dtype=torch.complex64)
    values = torch.tensor([2j, -1 + 1j], dtype=torch.complex64)
    indices = torch.tensor([3, 0], dtype=torch.int32)

    class Plan:
        def __init__(self, values, indices, target, alpha, beta):
            self.target = target
            self.calls = 0
            self.closed = False
            self.prepare_ms = 0.1
            self.metadata = {}
            plans.append(self)

        def run(self):
            self.calls += 1
            if broken and self.calls > 1:
                return self.target.fill_(float("nan"))
            self.target.mul_(0.5).index_add_(0, indices.long(), values * 1.5)
            return self.target

        def close(self):
            self.closed = True

    def benchmark(run, warmup, iters):
        for _ in range(warmup + iters):
            result = run()
        return result, 0.5

    monkeypatch.setattr(baseline, "AxpbyPlan", Plan)
    monkeypatch.setattr(delivery, "_benchmark_cuda_op", benchmark)
    row = {}
    baseline.measure(row, values, indices, initial, 1.5, 0.5, 0.25, 2, 3)
    assert row["baseline_required"] == "cann"
    assert row["vendor_name"] == "CANN SparseTensorDenseAdd"
    assert plans[0].calls == 7
    assert plans[0].closed
    assert row["cann_status"] == ("FAIL" if broken else "PASS")
    if broken:
        assert "cann_ms" not in row
        assert "speedup_vs_cann" not in row
    else:
        assert row["speedup_vs_cann"] == 2.0
    torch.testing.assert_close(initial, torch.tensor([1 + 2j, -3j, 4, 5j]))


@pytest.mark.parametrize("shape", [(31, 23), (31, 31)])
@pytest.mark.parametrize(
    "name",
    [
        "spmv_csr_f16f32_int_trans",
        "spmv_csr_f32c32_int_trans",
        "spmv_csr_i8i32_int_trans",
        "spmm_csr_f16f32_int_trans_non_row",
        "spmv_csc_i8i32_int_non",
    ],
)
def test_q4_mixed_routes_prepare_correct_operation_once(
    monkeypatch, tmp_path, shape, name
):
    import scipy.io
    import scipy.sparse

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    import q4_variant_bench as bench

    monkeypatch.setattr(bench, "_is_ascend_runtime", lambda: True)
    monkeypatch.setattr(bench, "_vendor", lambda *args: (None, None))
    monkeypatch.setattr(bench, "_time_vendor", lambda *args: None)
    monkeypatch.setattr(bench, "_time_pytorch", lambda *args: None)
    monkeypatch.setattr(bench, "_benchmark_cuda_op", lambda run, *args: (run(), 1.0))
    monkeypatch.delenv("FLAGSPARSE_BENCH_CUPY", raising=False)
    monkeypatch.delenv("FLAGSPARSE_BENCH_CAPI", raising=False)
    variant = next(
        variant
        for variants in bench.VARIANTS.values()
        for variant in variants
        if variant.name == name
    )
    matrix_path = tmp_path / "matrix.mtx"
    scipy.io.mmwrite(
        matrix_path,
        scipy.sparse.random(*shape, density=0.15, random_state=107, format="csr"),
    )
    matrix = bench._Matrix(matrix_path, "cpu")
    calls, preparations = [], []

    def prepare(values, indices, pointer, *args, op, **kwargs):
        assert op == variant.op_a
        assert kwargs.get("out_dtype", variant.out) == variant.out
        preparations.append(op)
        constructor = (
            scipy.sparse.csc_matrix if variant.fmt == "csc" else scipy.sparse.csr_matrix
        )
        sparse = constructor(
            (bench._wide_np(values), indices.numpy(), pointer.numpy()), shape
        )
        return SimpleNamespace(sparse=bench._op_sp(sparse, op), acc_dtype=variant.out)

    def run(prepared, dense=None, *, x=None):
        reference = prepared.sparse @ bench._wide_np(x if dense is None else dense)
        calls.append(variant.op_a)
        return torch.from_numpy(reference).to(variant.out)

    monkeypatch.setattr(bench.fs, f"flagsparse_{variant.kind}_{variant.fmt}", run)
    monkeypatch.setattr(bench.fs, "prepare_spmv_csc", prepare)
    monkeypatch.setattr(bench.mixed_spmx, "prepare_spmv_csr_mixed", prepare)
    monkeypatch.setattr(bench.mixed_spmx, "prepare_spmm_csr_mixed", prepare)
    monkeypatch.setattr(bench.mixed_spmx, "spmm_csr_mixed_prepared", run)
    row = {}
    bench._KINDS[variant.kind](
        variant, matrix, torch.Generator().manual_seed(97), 1, 2, row
    )
    assert calls == [variant.op_a, variant.op_a]
    assert preparations == [variant.op_a]
    assert row["timed"] == "prepared"
    assert row["ours_prepare_ms"] > 0
    assert row["max_error"] < 1e-6
