"""MACA SDDMM local PyTorch baselines stay independent of FlagSparse accuracy."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.sparse")
import q4_variant_bench as q4  # noqa: E402  (after the importorskip guards above)


@pytest.mark.parametrize("variant", q4.VARIANTS["sddmm_csr"], ids=lambda v: v.name)
@pytest.mark.parametrize("failure", ["wrong", "exception", "valid"])
@pytest.mark.parametrize("maca", [True, False])
def test_maca_sddmm_baseline_all_variants(monkeypatch, variant, failure, maca):
    csr = sp.csr_matrix((np.ones(4), ([0, 0, 2, 2], [0, 3, 1, 3])), shape=(3, 4))
    mat = SimpleNamespace(
        csr=csr,
        shape=csr.shape,
        dev=torch.device("cpu"),
        cols=torch.tensor(csr.indices),
        ptr=torch.tensor(csr.indptr),
        rows=torch.tensor([0, 0, 2, 2]),
    )

    def ours(**kw):
        a, b = q4._op_t(kw["x"], variant.op_a), q4._op_t(kw["y"], variant.op_b)
        return (a.float() @ b.float() if a.dtype == torch.float16 else a @ b)[
            mat.rows, mat.cols.long()
        ]

    def sampled(pattern, a, b, **kw):
        if failure == "exception":
            raise RuntimeError("sampled_addmm unavailable")
        values = (a @ b)[mat.rows, mat.cols.long()]
        return values + 100 if failure == "wrong" else values

    monkeypatch.setattr(q4, "_is_maca_runtime", lambda: maca)
    monkeypatch.setattr(q4, "_is_ascend_runtime", lambda: False)
    monkeypatch.setattr(
        q4,
        "fs",
        SimpleNamespace(
            prepare_sddmm_csr=lambda *a, **k: object(), flagsparse_sddmm_csr=ours
        ),
    )
    monkeypatch.setattr(q4, "_vendor", lambda *a, **k: (None, None))
    monkeypatch.setattr(q4, "_time_vendor", lambda *a, **k: None)
    monkeypatch.setattr(torch.sparse, "sampled_addmm", sampled)
    monkeypatch.setattr(q4, "_benchmark_cuda_op", lambda op, *a: (op(), 1.0))
    row = {}
    q4._sddmm(variant, mat, torch.Generator().manual_seed(20261103), 1, 3, row)
    assert row["max_error"] < 1e-6
    if not maca and failure == "wrong":
        assert "pytorch_ms" not in row
        assert row["pytorch_reason"]
        assert row["pytorch_formulation"] == "sampled_addmm"
        return
    assert row["pytorch_ms"] == 1.0
    assert row["pytorch_max_error"] < 1e-6
    assert "pytorch_reason" not in row
    assert row["pytorch_formulation"] == (
        "sampled_addmm" if failure == "valid" else "gather/multiply/sum"
    )
    assert ("pytorch_sampled_addmm_reason" in row) == (failure != "valid")
    assert row["pytorch_compute_dtype"] == (
        "float32" if variant.value == torch.float16 else q4._name(variant.value)
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 100.0])
@pytest.mark.parametrize("timed_only", [False, True])
def test_invalid_outputs_never_publish_latency(monkeypatch, bad, timed_only):
    ref = np.array([1.0, 2.0])
    row = {"status": "PASS", "pytorch_ms": 99.0, "pytorch_reason": "stale"}
    builds = []

    def build(fallback):
        builds.append(fallback)
        return lambda: torch.tensor(ref if timed_only else [bad, bad])

    monkeypatch.setattr(
        q4, "_benchmark_cuda_op", lambda *a: (torch.tensor([bad, bad]), 1.0)
    )
    q4._time_maca_sddmm_pytorch(row, build, 1, 3, ref)
    assert builds == [False, True]
    assert "pytorch_ms" not in row
    assert row["pytorch_reason"]
    assert row["pytorch_sampled_addmm_reason"]
    assert row["status"] == "PASS"


@pytest.mark.parametrize("latency", [None, 0, -1, float("nan"), float("inf")])
def test_invalid_timing_never_published(monkeypatch, latency):
    ref = np.ones(2)
    row = {}
    monkeypatch.setattr(q4, "_benchmark_cuda_op", lambda *a: (torch.ones(2), latency))
    q4._time_maca_sddmm_pytorch(row, lambda **k: lambda: torch.ones(2), 0, 1, ref)
    assert "pytorch_ms" not in row
    assert "invalid latency" in row["pytorch_reason"]


def test_bad_timed_sparse_output_retries_valid_fallback(monkeypatch):
    ref = np.ones(2)
    row = {}
    attempts = []

    def build(fallback):
        attempts.append(fallback)
        return lambda: torch.ones(2)

    def timer(op, *args):
        return (op() if attempts[-1] else torch.full((2,), float("nan"))), 2.0

    monkeypatch.setattr(q4, "_benchmark_cuda_op", timer)
    q4._time_maca_sddmm_pytorch(row, build, 0, 1, ref)
    assert attempts == [False, True]
    assert row["pytorch_ms"] == 2.0
    assert "pytorch_reason" not in row


def test_wrong_shape_is_rejected(monkeypatch):
    row = {}
    q4._time_maca_sddmm_pytorch(
        row, lambda **k: lambda: torch.ones(2, 1), 0, 1, np.ones(2)
    )
    assert "pytorch_ms" not in row
    assert "shape mismatch" in row["pytorch_reason"]


def test_fallback_exception_is_reported_without_latency():
    row = {"status": "PASS"}

    def build(fallback):
        raise RuntimeError("fallback failed" if fallback else "sampled failed")

    q4._time_maca_sddmm_pytorch(row, build, 0, 1, np.ones(2))
    assert "pytorch_ms" not in row
    assert "sampled failed" in row["pytorch_sampled_addmm_reason"]
    assert "fallback failed" in row["pytorch_reason"]
    assert row["status"] == "PASS"


def test_empty_output_can_be_a_valid_baseline(monkeypatch):
    row = {}
    monkeypatch.setattr(q4, "_benchmark_cuda_op", lambda op, *a: (op(), 1.0))
    q4._time_maca_sddmm_pytorch(
        row, lambda **k: lambda: torch.empty(0), 0, 1, np.empty(0)
    )
    assert row["pytorch_ms"] == 1.0
    assert row["pytorch_max_error"] == 0
