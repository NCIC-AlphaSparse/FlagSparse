import pytest
import torch

from flagsparse import flagsparse_spsv_csr

from tests.pytest.param_shapes import SPSV_N

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    from cupyx.scipy.sparse.linalg import spsolve_triangular as cpx_spsolve_triangular
except Exception:
    cp = None
    cpx_sparse = None
    cpx_spsolve_triangular = None


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
SUPPORTED_COMPLEX_DTYPES = [torch.complex64, torch.complex128]

SUPPORTED_DTYPES = [torch.float32, torch.float64, *SUPPORTED_COMPLEX_DTYPES]
NON_TRANS_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
TRANS_CONJ_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
TRANS_CONJ_MODES = ["TRANS", "CONJ"]


def _dtype_id(dtype):
    return str(dtype).replace("torch.", "")


def _tol(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-3
    return 1e-10, 1e-8


def _rand_like(dtype, shape, device):
    if dtype in (torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
    base = torch.float32 if dtype == torch.complex64 else torch.float64
    r = torch.randn(shape, dtype=base, device=device)
    i = torch.randn(shape, dtype=base, device=device)
    return torch.complex(r, i)

def _apply_ref_op(A, op_mode):
    if op_mode == "TRANS":
        return A.transpose(-2, -1)
    if op_mode == "CONJ":
        return A.transpose(-2, -1).conj() if torch.is_complex(A) else A.transpose(-2, -1)
    return A


def _effective_upper(lower, op_mode):
    return lower if op_mode in ("TRANS", "CONJ") else not lower


def _effective_lower_for_op(lower, op_mode):
    return (not lower) if op_mode in ("TRANS", "CONJ") else lower


def _transpose_arg(op_mode):
    if op_mode == "NON":
        return False
    return op_mode


def _cupy_apply_op(A_cp, op_mode):
    if op_mode == "TRANS":
        return A_cp.transpose().tocsr()
    if op_mode == "CONJ":
        return A_cp.transpose().conj().tocsr()
    return A_cp


def _build_triangular(n, dtype, device, lower=True):
    off = _rand_like(dtype, (n, n), device) * 0.02
    A = torch.tril(off) if lower else torch.triu(off)
    if torch.is_complex(A):
        diag = (torch.rand(n, device=device, dtype=A.real.dtype) + 2.0).to(A.real.dtype)
        A = A + torch.diag(torch.complex(diag, torch.zeros_like(diag)))
    else:
        diag = torch.rand(n, device=device, dtype=A.dtype) + 2.0
        A = A + torch.diag(diag)
    return A


def _cupy_csr_from_torch(data, indices, indptr, shape):
    if cp is None or cpx_sparse is None:
        return None
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data.contiguous()))
    idx_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64).contiguous()))
    ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr.to(torch.int64).contiguous()))
    return cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)


def _cupy_ref_spsv(A_cp, b_t, *, lower, unit_diagonal=False):
    if cp is None or cpx_spsolve_triangular is None:
        return None
    b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b_t.contiguous()))
    x_cp = cpx_spsolve_triangular(A_cp, b_cp, lower=lower, unit_diagonal=unit_diagonal)
    x_t = torch.utils.dlpack.from_dlpack(x_cp.toDlpack())
    return x_t.to(b_t.dtype)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["float32", "float64"],
)
def test_spsv_csr_lower_matches_dense(n, dtype):
    # Keep the original baseline test case untouched in semantics.
    device = torch.device("cuda")
    base = torch.tril(torch.randn(n, n, dtype=dtype, device=device))
    eye = torch.eye(n, dtype=dtype, device=device)
    A = base + eye * (float(n) * 0.5 + 2.0)
    b = torch.randn(n, dtype=dtype, device=device)
    x_ref = torch.linalg.solve_triangular(
        A, b.unsqueeze(-1), upper=False
    ).squeeze(-1)
    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices().to(torch.int64)
    x = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
    )
    rtol = 1e-4 if dtype == torch.float32 else 1e-10
    atol = 1e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_spsv_csr_non_trans_supported_combos(n, dtype, index_dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    x_ref = torch.linalg.solve_triangular(
        A.to(dtype), b.to(dtype).unsqueeze(-1), upper=False
    ).squeeze(-1)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(index_dtype)
    indptr = Asp.crow_indices().to(index_dtype)

    x = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
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
@pytest.mark.parametrize("dtype", TRANS_CONJ_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("op_mode", TRANS_CONJ_MODES)
def test_spsv_csr_transpose_family_supported_combos(n, dtype, index_dtype, op_mode):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    A_ref = A.to(dtype)
    b_ref = b.to(dtype)
    x_ref = torch.linalg.solve_triangular(
        _apply_ref_op(A_ref, op_mode), b_ref.unsqueeze(-1), upper=_effective_upper(True, op_mode)
    ).squeeze(-1)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(index_dtype)
    indptr = Asp.crow_indices().to(index_dtype)

    x = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=_transpose_arg(op_mode),
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.skipif(
    cp is None or cpx_sparse is None or cpx_spsolve_triangular is None,
    reason="CuPy/cuSPARSE required",
)
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
def test_spsv_csr_matches_cusparse_non_trans(n, dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)
    A_cp = _cupy_csr_from_torch(data, indices, indptr, (n, n))

    x_non = flagsparse_spsv_csr(
        data, indices, indptr, b, (n, n), lower=True, unit_diagonal=False, transpose=False
    )
    x_non_ref = _cupy_ref_spsv(A_cp, b, lower=True, unit_diagonal=False)

    rtol, atol = _tol(dtype)
    assert torch.allclose(x_non, x_non_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.skipif(
    cp is None or cpx_sparse is None or cpx_spsolve_triangular is None,
    reason="CuPy/cuSPARSE required",
)
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", TRANS_CONJ_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("op_mode", TRANS_CONJ_MODES)
def test_spsv_csr_matches_cusparse_transpose_family(n, dtype, index_dtype, op_mode):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(index_dtype)
    indptr = Asp.crow_indices().to(index_dtype)
    A_cp = _cupy_csr_from_torch(data, indices, indptr, (n, n))

    x_trans = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=_transpose_arg(op_mode),
    )
    x_trans_ref = _cupy_ref_spsv(
        _cupy_apply_op(A_cp, op_mode),
        b,
        lower=_effective_lower_for_op(True, op_mode),
        unit_diagonal=False,
    )

    rtol, atol = _tol(dtype)
    assert torch.allclose(x_trans, x_trans_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
def test_spsv_csr_non_trans_upper_supported_combos(n, dtype, index_dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=False)
    b = _rand_like(dtype, (n,), device)
    x_ref = torch.linalg.solve_triangular(
        A.to(dtype), b.to(dtype).unsqueeze(-1), upper=True
    ).squeeze(-1)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(index_dtype)
    indptr = Asp.crow_indices().to(index_dtype)

    x = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        (n, n),
        lower=False,
        unit_diagonal=False,
        transpose=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", TRANS_CONJ_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("op_mode", TRANS_CONJ_MODES)
def test_spsv_csr_upper_transpose_family_supported_combos(n, dtype, index_dtype, op_mode):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=False)
    b = _rand_like(dtype, (n,), device)
    A_ref = A.to(dtype)
    b_ref = b.to(dtype)
    x_ref = torch.linalg.solve_triangular(
        _apply_ref_op(A_ref, op_mode), b_ref.unsqueeze(-1), upper=_effective_upper(False, op_mode)
    ).squeeze(-1)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(index_dtype)
    indptr = Asp.crow_indices().to(index_dtype)

    x = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        (n, n),
        lower=False,
        unit_diagonal=False,
        transpose=_transpose_arg(op_mode),
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.skipif(
    cp is None or cpx_sparse is None or cpx_spsolve_triangular is None,
    reason="CuPy/cuSPARSE required",
)
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
def test_spsv_csr_matches_cusparse_upper_non_trans(n, dtype):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=False)
    b = _rand_like(dtype, (n,), device)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)
    A_cp = _cupy_csr_from_torch(data, indices, indptr, (n, n))

    x_non = flagsparse_spsv_csr(
        data, indices, indptr, b, (n, n), lower=False, unit_diagonal=False, transpose=False
    )
    x_non_ref = _cupy_ref_spsv(A_cp, b, lower=False, unit_diagonal=False)

    rtol, atol = _tol(dtype)
    assert torch.allclose(x_non, x_non_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.skipif(
    cp is None or cpx_sparse is None or cpx_spsolve_triangular is None,
    reason="CuPy/cuSPARSE required",
)
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", TRANS_CONJ_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("op_mode", TRANS_CONJ_MODES)
def test_spsv_csr_matches_cusparse_upper_transpose_family(n, dtype, index_dtype, op_mode):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=False)
    b = _rand_like(dtype, (n,), device)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(index_dtype)
    indptr = Asp.crow_indices().to(index_dtype)
    A_cp = _cupy_csr_from_torch(data, indices, indptr, (n, n))

    x_trans = flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        (n, n),
        lower=False,
        unit_diagonal=False,
        transpose=_transpose_arg(op_mode),
    )
    x_trans_ref = _cupy_ref_spsv(
        _cupy_apply_op(A_cp, op_mode),
        b,
        lower=_effective_lower_for_op(False, op_mode),
        unit_diagonal=False,
    )

    rtol, atol = _tol(dtype)
    assert torch.allclose(x_trans, x_trans_ref, rtol=rtol, atol=atol)
