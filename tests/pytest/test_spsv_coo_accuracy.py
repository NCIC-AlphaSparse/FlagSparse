import pytest
import torch

from flagsparse import (
    FlagSparseSpSVDescr,
    flagsparse_spsv_analysis_coo,
    flagsparse_spsv_coo,
    flagsparse_spsv_create_workspace,
    flagsparse_spsv_preprocess_coo,
    flagsparse_spsv_solve_coo,
)
import flagsparse.sparse_operations.spsv as fs_spsv_impl

from tests.pytest.param_shapes import SPSV_N
from tests.pytest.test_spsv_csr_accuracy import (
    _apply_ref_op,
    _build_triangular,
    _dense_ref_spsv,
    _dtype_id,
    _effective_upper,
    _rand_like,
    _tol,
    _transpose_arg,
    NON_TRANS_DTYPES,
    SUPPORTED_COMPLEX_DTYPES,
    TRANS_CONJ_MODES,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("op_mode", TRANS_CONJ_MODES)
def test_spsv_coo_transpose_family_complex128_routes_through_csr(n, op_mode):
    device = torch.device("cuda")
    dtype = torch.complex128
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    x_ref = torch.linalg.solve_triangular(
        _apply_ref_op(A, op_mode), b.unsqueeze(-1), upper=_effective_upper(True, op_mode)
    ).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=_transpose_arg(op_mode),
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_spsv_coo_non_trans_supported_combos(n, dtype, index_dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    x_ref = torch.linalg.solve_triangular(
        A.to(dtype), b.to(dtype).unsqueeze(-1), upper=False
    ).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_spsv_coo_non_trans_upper_supported_combos(n, dtype, index_dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=False)
    b = _rand_like(dtype, (n,), device)
    x_ref = torch.linalg.solve_triangular(
        A.to(dtype), b.to(dtype).unsqueeze(-1), upper=True
    ).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=False,
        unit_diagonal=False,
        transpose=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_non_trans_routes_through_csr(monkeypatch):
    device = torch.device("cuda")
    dtype = torch.complex64
    index_dtype = torch.int64
    n = SPSV_N[0]
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    x_ref = torch.linalg.solve_triangular(
        A, b.unsqueeze(-1), upper=False
    ).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    called = {"hit": False}
    real_csr_impl = fs_spsv_impl.flagsparse_spsv_csr

    def _wrapped_flagsparse_spsv_csr(*args, **kwargs):
        called["hit"] = True
        return real_csr_impl(*args, **kwargs)

    monkeypatch.setattr(fs_spsv_impl, "flagsparse_spsv_csr", _wrapped_flagsparse_spsv_csr)

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
    )

    rtol, atol = _tol(dtype)
    assert called["hit"]
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_to_csr_keeps_duplicates():
    device = torch.device("cuda")
    data = torch.tensor([1.0, 2.0, 3.0], device=device)
    row = torch.tensor([0, 0, 1], dtype=torch.int64, device=device)
    col = torch.tensor([1, 1, 0], dtype=torch.int64, device=device)

    data_csr, indices_csr, indptr_csr = fs_spsv_impl._coo2csr_for_spsv(
        data, row, col, 2, assume_ordered=False
    )

    assert data_csr.numel() == 3
    assert indices_csr.tolist() == [1, 1, 0]
    assert indptr_csr.tolist() == [0, 2, 3]


@pytest.mark.spsv
def test_spsv_coo_analysis_workspace_solve_matches_direct():
    device = torch.device("cuda")
    dtype = torch.complex64
    n = SPSV_N[0]
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    descr = flagsparse_spsv_analysis_coo(
        data,
        row.to(torch.int64),
        col.to(torch.int64),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
    )
    assert isinstance(descr, FlagSparseSpSVDescr)
    assert descr.format == "coo"
    assert descr.canonical_format == "csr"

    workspace = flagsparse_spsv_create_workspace(descr)
    x_via_descr = flagsparse_spsv_solve_coo(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_coo(
        data,
        row.to(torch.int64),
        col.to(torch.int64),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_explicit_roc_route_matches_dense():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_roc",
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("dtype", SUPPORTED_COMPLEX_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("solve_kind", ["csr_roc", "csr_cw_levelschd", "alg2"])
def test_spsv_coo_explicit_complex_level_routes_match_dense(dtype, solve_kind):
    device = torch.device("cuda")
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind=solve_kind,
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("dtype", SUPPORTED_COMPLEX_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("solve_kind", ["csr_nnz_balance", "alg3"])
def test_spsv_coo_explicit_complex_nnz_balance_routes_match_dense(dtype, solve_kind):
    device = torch.device("cuda")
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind=solve_kind,
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_explicit_levelschd_route_matches_dense():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_cw_levelschd",
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_explicit_nnz_balance_route_matches_dense():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 96
    A = torch.tril(torch.randn(n, n, dtype=dtype, device=device) * 0.02)
    A = A + torch.eye(n, dtype=dtype, device=device) * 3.0
    b = _rand_like(dtype, (n,), device)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_nnz_balance",
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_roc_analysis_workspace_solve_matches_direct():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    descr = flagsparse_spsv_analysis_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_roc",
    )
    assert descr.solve_kind == "csr_roc"
    assert descr.canonical_format == "csr"
    workspace = flagsparse_spsv_preprocess_coo(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_coo(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_roc",
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_levelschd_analysis_workspace_solve_matches_direct():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    descr = flagsparse_spsv_analysis_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_cw_levelschd",
    )
    assert descr.solve_kind == "csr_cw_levelschd"
    assert descr.canonical_format == "csr"
    workspace = flagsparse_spsv_preprocess_coo(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_coo(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_cw_levelschd",
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_nnz_balance_analysis_workspace_solve_matches_direct():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 96
    A = torch.tril(torch.randn(n, n, dtype=dtype, device=device) * 0.02)
    A = A + torch.eye(n, dtype=dtype, device=device) * 3.0
    b = _rand_like(dtype, (n,), device)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    descr = flagsparse_spsv_analysis_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_nnz_balance",
    )
    assert descr.solve_kind == "csr_nnz_balance"
    assert descr.canonical_format == "csr"
    workspace = flagsparse_spsv_preprocess_coo(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_coo(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_nnz_balance",
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("dtype", SUPPORTED_COMPLEX_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("solve_kind", ["csr_nnz_balance", "alg3"])
def test_spsv_coo_complex_nnz_balance_analysis_workspace_matches_direct(dtype, solve_kind):
    device = torch.device("cuda")
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    descr = flagsparse_spsv_analysis_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind=solve_kind,
    )
    assert descr.solve_kind == "csr_nnz_balance"
    assert descr.canonical_format == "csr"
    workspace = flagsparse_spsv_preprocess_coo(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_coo(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind=solve_kind,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_spsv_coo_non_trans_unit_supported_combos(n, dtype, index_dtype, lower):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=lower)
    b = _rand_like(dtype, (n,), device)
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=lower, unit_diagonal=True)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=lower,
        unit_diagonal=True,
        transpose=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize("op_mode", TRANS_CONJ_MODES)
def test_spsv_coo_unit_transpose_family_supported_combos(
    n, dtype, index_dtype, lower, op_mode
):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=lower)
    b = _rand_like(dtype, (n,), device)
    x_ref = _dense_ref_spsv(
        A.to(dtype),
        b.to(dtype),
        lower=lower,
        op_mode=op_mode,
        unit_diagonal=True,
    )

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices()
    data = A_coo.values()

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=lower,
        unit_diagonal=True,
        transpose=_transpose_arg(op_mode),
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)
