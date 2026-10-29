"""Complex-safe Ascend SDDMM benchmark reference and dispatch contracts."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.sparse")

import q4_variant_bench as q4  # noqa: E402  (after the importorskip guards above)


@pytest.mark.parametrize("width", [1, 7, 32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("conjugated", [False, True])
def test_real_plane_reference_matches_complex128(
    monkeypatch, width, index_dtype, column_major, conjugated
):
    generator = torch.Generator().manual_seed(20261029)
    left = torch.complex(
        torch.randn(4, width, generator=generator),
        torch.randn(4, width, generator=generator),
    )
    right = torch.complex(
        torch.randn(5, width, generator=generator),
        torch.randn(5, width, generator=generator),
    )
    if column_major:
        left, right = left.T.contiguous().T, right.T.contiguous().T
    if conjugated:
        left, right = left.conj(), right.conj()
    indices = torch.tensor([0, 4, 1, 1, 3], dtype=index_dtype)
    indptr = torch.tensor([0, 2, 2, 4, 5], dtype=index_dtype)
    rows = torch.tensor([0, 0, 2, 2, 3])
    reference = (left.to(torch.complex128) @ right.to(torch.complex128).T)[
        rows, indices.long()
    ]
    select = torch.Tensor.index_select

    def real_only(tensor, *args, **kwargs):
        assert not tensor.is_complex()
        return select(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "index_select", real_only)
    result = q4._ascend_complex_sddmm_reference(indices, indptr, left, right)
    assert result.dtype == torch.complex64
    torch.testing.assert_close(
        result.to(torch.complex128), reference, atol=2e-6, rtol=2e-6
    )
    left = left.resolve_conj().mul(-2)
    second = q4._ascend_complex_sddmm_reference(indices, indptr, left, right)
    torch.testing.assert_close(second, -2 * result, atol=2e-6, rtol=2e-6)


def test_real_plane_reference_handles_all_empty_rows():
    indices = torch.empty(0, dtype=torch.int32)
    indptr = torch.zeros(5, dtype=torch.int64)
    result = q4._ascend_complex_sddmm_reference(
        indices,
        indptr,
        torch.ones(4, 7, dtype=torch.complex64),
        torch.ones(5, 7, dtype=torch.complex64),
    )
    assert result.shape == (0,) and result.dtype == torch.complex64


@pytest.fixture
def benchmark_case(monkeypatch):
    matrix = sp.csr_matrix((np.ones(4), ([0, 0, 2, 2], [0, 3, 1, 3])), shape=(3, 4))
    mat = SimpleNamespace(
        csr=matrix,
        shape=matrix.shape,
        dev=torch.device("cpu"),
        cols=torch.tensor(matrix.indices, dtype=torch.int32),
        ptr=torch.tensor(matrix.indptr, dtype=torch.int32),
        rows=torch.tensor([0, 0, 2, 2], dtype=torch.int32),
    )

    def ours(**kwargs):
        left = q4._op_t(kwargs["x"], kwargs["op_a"])
        right = q4._op_t(kwargs["y"], kwargs["op_b"])
        return (left @ right)[mat.rows.long(), mat.cols.long()]

    def unavailable(*args, **kwargs):
        raise NotImplementedError("sampled_addmm unavailable")

    monkeypatch.setattr(
        q4,
        "fs",
        SimpleNamespace(
            prepare_sddmm_csr=lambda *args, **kwargs: object(),
            flagsparse_sddmm_csr=ours,
        ),
    )
    monkeypatch.setattr(torch.sparse, "sampled_addmm", unavailable)
    monkeypatch.setattr(q4, "_vendor", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(q4, "_time_vendor", lambda *args, **kwargs: None)
    monkeypatch.setattr(q4, "_benchmark_cuda_op", lambda call, *args: (call(), 1.0))
    return mat


@pytest.mark.parametrize("ascend", [True, False])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.float32])
def test_real_plane_fallback_is_ascend_complex_only(
    monkeypatch, benchmark_case, ascend, dtype
):
    monkeypatch.setattr(q4, "_is_ascend_runtime", lambda: ascend)
    variant = q4.Variant("test", "sddmm", "csr", dtype, dtype)
    row = {}
    q4._sddmm(variant, benchmark_case, torch.Generator().manual_seed(29), 0, 1, row)
    assert row["pytorch_ms"] == 1.0
    assert row["pytorch_max_error"] < 1e-6
    assert row["pytorch_formulation"] == "gather/multiply/sum"
    assert row.get("pytorch_complex_strategy") == (
        "real_planes" if ascend and dtype.is_complex else None
    )
