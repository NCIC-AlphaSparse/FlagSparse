import pytest
import torch

from flagsparse import flagsparse_spsv_coo
import flagsparse.sparse_operations.spsv as fs_spsv_impl

from tests.pytest.param_shapes import SPSV_N
from tests.pytest.test_spsv_csr_accuracy import (
    _apply_ref_op,
    _build_triangular,
    _dtype_id,
    _effective_upper,
    _rand_like,
    _tol,
    _transpose_arg,
    NON_TRANS_DTYPES,
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
