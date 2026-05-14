import pytest
import torch

from flagsparse import (
    FlagSparseDnVecDescr,
    FlagSparseSpMatDescr,
    FlagSparseSpSVDescr,
    FlagSparseSpSVHandle,
    FlagSparseSpSVWorkspace,
    flagsparse_create_dnvec,
    flagsparse_create_spmat_csr,
    flagsparse_create_spsv_handle,
    flagsparse_spsv_analysis_csr,
    flagsparse_spsv_analysis_ex,
    flagsparse_spsv_buffer_size,
    flagsparse_spsv_buffer_size_ex,
    flagsparse_spsv_create_workspace,
    flagsparse_spsv_csr,
    flagsparse_spsv_preprocess_csr,
    flagsparse_spsv_solve_ex,
    flagsparse_spsv_solve_csr,
)
import flagsparse.sparse_operations.spsv as fs_spsv_impl

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


def _dense_ref_spsv(A, b, *, lower, op_mode="NON", unit_diagonal=False):
    A_eff = _apply_ref_op(A, op_mode)
    x = torch.linalg.solve_triangular(
        A_eff,
        b.unsqueeze(-1),
        upper=_effective_upper(lower, op_mode),
        unitriangular=unit_diagonal,
    )
    return x.squeeze(-1)


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
def test_spsv_csr_rejects_matrix_rhs():
    device = torch.device("cuda")
    dtype = torch.float32
    n = SPSV_N[0]
    A = _build_triangular(n, dtype, device, lower=True)
    b = torch.randn(n, 2, dtype=dtype, device=device)

    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)

    with pytest.raises(ValueError, match="DnVec"):
        flagsparse_spsv_csr(
            data,
            indices,
            indptr,
            b,
            (n, n),
            lower=True,
            unit_diagonal=False,
            transpose=False,
        )


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_spsv_csr_non_trans_unit_supported_combos(n, dtype, index_dtype, lower):
    device = torch.device("cuda")
    A = _build_triangular(n, dtype, device, lower=lower)
    b = _rand_like(dtype, (n,), device)
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=lower, unit_diagonal=True)

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
        lower=lower,
        unit_diagonal=True,
        transpose=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", TRANS_CONJ_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize("op_mode", TRANS_CONJ_MODES)
def test_spsv_csr_unit_transpose_family_supported_combos(
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
        lower=lower,
        unit_diagonal=True,
        transpose=_transpose_arg(op_mode),
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_csr_complex_non_trans_defaults_to_cw_route():
    device = torch.device("cuda")
    n = SPSV_N[0]
    dtype = torch.complex64
    A = _build_triangular(n, dtype, device, lower=True)
    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)
    b = _rand_like(dtype, (n,), device)

    _, _, _, trans_mode, _, _, solve_plan = fs_spsv_impl._resolve_spsv_csr_runtime(
        data, indices, indptr, b, (n, n), True, False, False
    )
    selected = fs_spsv_impl._select_spsv_runtime_plan(solve_plan, trans_mode)
    assert selected["solve_kind"] == "csr_cw"


@pytest.mark.spsv
@pytest.mark.parametrize("op_mode", TRANS_CONJ_MODES)
def test_spsv_csr_transpose_family_defaults_to_cw_route(op_mode):
    device = torch.device("cuda")
    n = SPSV_N[0]
    dtype = torch.complex128
    A = _build_triangular(n, dtype, device, lower=True)
    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)
    b = _rand_like(dtype, (n,), device)

    _, _, _, trans_mode, _, _, solve_plan = fs_spsv_impl._resolve_spsv_csr_runtime(
        data, indices, indptr, b, (n, n), True, _transpose_arg(op_mode), False
    )
    selected = fs_spsv_impl._select_spsv_runtime_plan(solve_plan, trans_mode)
    assert selected["solve_kind"] == "transpose_cw"


@pytest.mark.spsv
def test_spsv_internal_cw_worker_count_limits_narrow_frontier():
    matrix_stats = {
        "max_frontier": 3,
        "avg_frontier": 2.5,
        "frontier_ratio": 0.006,
        "num_levels": 4096,
        "avg_nnz_per_row": 4096.0,
    }
    worker_count = fs_spsv_impl._resolve_cw_worker_count(4096, matrix_stats, 1)
    assert worker_count <= 4


@pytest.mark.spsv
def test_spsv_auto_route_promotes_dense_real_lower_to_nnz_balance():
    matrix_stats = {
        "num_levels": 256,
        "max_frontier": 1,
        "avg_frontier": 1.0,
        "frontier_ratio": 1.0 / 256.0,
        "avg_nnz_per_row": 128.0,
        "max_nnz_per_row": 256,
    }
    route = fs_spsv_impl._choose_spsv_nontrans_auto_route(
        256,
        matrix_stats,
        lower=True,
        unit_diagonal=False,
        value_dtype=torch.float64,
    )
    assert route == "csr_nnz_balance"


@pytest.mark.spsv
def test_spsv_levelschd_analysis_builds_sorted_row_map():
    device = torch.device("cuda")
    data = torch.tensor([4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 4.0, 7.0], dtype=torch.float64, device=device)
    indices = torch.tensor([0, 0, 1, 0, 2, 1, 2, 3], dtype=torch.int64, device=device)
    indptr = torch.tensor([0, 1, 3, 5, 8], dtype=torch.int64, device=device)
    meta = fs_spsv_impl._build_spsv_level_schedule_metadata(
        indices,
        indptr,
        4,
        lower=True,
        unit_diagonal=False,
    )
    assert meta["row_map32"].tolist() == [0, 1, 2, 3]
    assert meta["level_ptr32"].tolist() == [0, 1, 3, 4]
    assert meta["indegree_init32"].tolist() == [1, 2, 2, 3]
    assert meta["matrix_stats"]["num_levels"] == 3
    assert meta["matrix_stats"]["max_frontier"] == 2


@pytest.mark.spsv
def test_spsv_nnz_balance_analysis_builds_row_idx_and_indegree():
    device = torch.device("cuda")
    indices = torch.tensor([0, 0, 1, 0, 2, 1, 2, 3], dtype=torch.int64, device=device)
    indptr = torch.tensor([0, 1, 3, 5, 8], dtype=torch.int64, device=device)
    meta = fs_spsv_impl._build_spsv_nnz_balance_metadata(
        indices,
        indptr,
        4,
        lower=True,
        unit_diagonal=False,
    )
    assert meta["indegree_init32"].tolist() == [1, 2, 2, 3]
    assert meta["csr_row_idx32"].tolist() == [0, 1, 1, 2, 2, 3, 3, 3]


@pytest.mark.spsv
def test_spsv_auto_route_promotes_wide_frontier_real_lower_to_levelschd():
    matrix_stats = {
        "num_levels": 48,
        "max_frontier": 64,
        "avg_frontier": 12.0,
        "frontier_ratio": 0.125,
        "avg_nnz_per_row": 12.0,
        "max_nnz_per_row": 48,
    }
    route = fs_spsv_impl._choose_spsv_nontrans_auto_route(
        512,
        matrix_stats,
        lower=True,
        unit_diagonal=False,
        value_dtype=torch.float32,
    )
    assert route == "csr_cw_levelschd"


@pytest.mark.spsv
def test_spsv_csr_transpose_descriptor_keeps_preprocess_metadata():
    device = torch.device("cuda")
    dtype = torch.float64
    n = SPSV_N[0]
    A = _build_triangular(n, dtype, device, lower=True)
    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)

    descr = flagsparse_spsv_analysis_csr(
        data,
        indices,
        indptr,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose="TRANS",
    )
    assert descr.solve_kind == "transpose_cw"
    assert descr.storage_view == "csr_as_csc"
    assert descr.solve_plan.get("transpose_indegree_init") is None
    assert descr.solve_plan.get("transpose_diag") is None
    assert descr.solve_plan.get("transpose_dep_start") is None
    assert descr.solve_plan.get("transpose_dep_end") is None


@pytest.mark.spsv
@pytest.mark.parametrize("op_mode", TRANS_CONJ_MODES)
def test_spsv_csr_transpose_public_solve_uses_transpose_kernel(monkeypatch, op_mode):
    device = torch.device("cuda")
    dtype = torch.complex128
    n = SPSV_N[0]
    A = _build_triangular(n, dtype, device, lower=True)
    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)
    b = _rand_like(dtype, (n,), device)

    called = {"transpose_complex": False}
    real_impl = fs_spsv_impl._triton_spsv_csr_transpose_cw_vector_complex

    def _wrapped(*args, **kwargs):
        called["transpose_complex"] = True
        return real_impl(*args, **kwargs)

    monkeypatch.setattr(
        fs_spsv_impl,
        "_triton_spsv_csr_transpose_cw_vector_complex",
        _wrapped,
    )

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
    x_ref = _dense_ref_spsv(
        A.to(dtype),
        b.to(dtype),
        lower=True,
        op_mode=op_mode,
        unit_diagonal=False,
    )
    rtol, atol = _tol(dtype)
    assert called["transpose_complex"]
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_csr_explicit_roc_route_matches_dense():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    Asp = A.to_sparse_csr()

    x = flagsparse_spsv_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        solve_kind="csr_roc",
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("dtype", SUPPORTED_COMPLEX_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("solve_kind", ["csr_roc", "csr_cw_levelschd", "alg2"])
def test_spsv_csr_explicit_complex_level_routes_match_dense(dtype, solve_kind):
    device = torch.device("cuda")
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    Asp = A.to_sparse_csr()

    x = flagsparse_spsv_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        solve_kind=solve_kind,
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("dtype", SUPPORTED_COMPLEX_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("solve_kind", ["csr_nnz_balance", "alg3"])
def test_spsv_csr_explicit_complex_nnz_balance_routes_match_dense(dtype, solve_kind):
    device = torch.device("cuda")
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    Asp = A.to_sparse_csr()

    x = flagsparse_spsv_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        solve_kind=solve_kind,
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_csr_explicit_roc_analysis_builds_only_level_metadata():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    Asp = A.to_sparse_csr()
    descr = flagsparse_spsv_analysis_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_roc",
    )
    assert descr.solve_kind == "csr_roc"
    assert int(descr.solve_plan["level_row_map32"].numel()) == n
    assert int(descr.solve_plan["nnz_balance_row_idx32"].numel()) == 0
    assert int(descr.solve_plan["nnz_balance_indegree32"].numel()) == 0


@pytest.mark.spsv
def test_spsv_csr_explicit_levelschd_route_matches_dense():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    Asp = A.to_sparse_csr()

    x = flagsparse_spsv_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        solve_kind="csr_cw_levelschd",
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_csr_explicit_levelschd_analysis_builds_only_level_metadata():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    Asp = A.to_sparse_csr()
    descr = flagsparse_spsv_analysis_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_cw_levelschd",
    )
    assert descr.solve_kind == "csr_cw_levelschd"
    assert int(descr.solve_plan["level_row_map32"].numel()) == n
    assert int(descr.solve_plan["nnz_balance_row_idx32"].numel()) == 0
    assert int(descr.solve_plan["nnz_balance_indegree32"].numel()) == 0


@pytest.mark.spsv
def test_spsv_csr_explicit_nnz_balance_route_matches_dense():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 96
    A = torch.tril(torch.randn(n, n, dtype=dtype, device=device) * 0.02)
    A = A + torch.eye(n, dtype=dtype, device=device) * 3.0
    b = _rand_like(dtype, (n,), device)
    Asp = A.to_sparse_csr()

    x = flagsparse_spsv_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        solve_kind="csr_nnz_balance",
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x, x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_csr_explicit_nnz_balance_analysis_builds_only_nnz_metadata():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 96
    A = torch.tril(torch.randn(n, n, dtype=dtype, device=device) * 0.02)
    A = A + torch.eye(n, dtype=dtype, device=device) * 3.0
    Asp = A.to_sparse_csr()
    descr = flagsparse_spsv_analysis_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_nnz_balance",
    )
    assert descr.solve_kind == "csr_nnz_balance"
    assert int(descr.solve_plan["level_row_map32"].numel()) == 0
    assert int(descr.solve_plan["nnz_balance_row_idx32"].numel()) == int(Asp.values().numel())
    assert int(descr.solve_plan["nnz_balance_indegree32"].numel()) == n


@pytest.mark.spsv
def test_spsv_csr_roc_analysis_workspace_solve_matches_direct():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    Asp = A.to_sparse_csr()

    descr = flagsparse_spsv_analysis_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_roc",
    )
    assert descr.solve_kind == "csr_roc"
    workspace = flagsparse_spsv_preprocess_csr(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_csr(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        solve_kind="csr_roc",
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("dtype", SUPPORTED_COMPLEX_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("solve_kind", ["csr_roc", "csr_cw_levelschd", "alg2"])
def test_spsv_csr_complex_level_route_analysis_workspace_matches_direct(dtype, solve_kind):
    device = torch.device("cuda")
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    Asp = A.to_sparse_csr()

    descr = flagsparse_spsv_analysis_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind=solve_kind,
    )
    workspace = flagsparse_spsv_preprocess_csr(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_csr(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        solve_kind=solve_kind,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("dtype", SUPPORTED_COMPLEX_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("solve_kind", ["csr_nnz_balance", "alg3"])
def test_spsv_csr_complex_nnz_balance_analysis_workspace_matches_direct(dtype, solve_kind):
    device = torch.device("cuda")
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    Asp = A.to_sparse_csr()

    descr = flagsparse_spsv_analysis_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind=solve_kind,
    )
    assert descr.solve_kind == "csr_nnz_balance"
    workspace = flagsparse_spsv_preprocess_csr(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_csr(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        solve_kind=solve_kind,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_csr_levelschd_analysis_workspace_solve_matches_direct():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, device, lower=True)
    b = _rand_like(dtype, (n,), device)
    Asp = A.to_sparse_csr()

    descr = flagsparse_spsv_analysis_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_cw_levelschd",
    )
    assert descr.solve_kind == "csr_cw_levelschd"
    workspace = flagsparse_spsv_preprocess_csr(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_csr(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        solve_kind="csr_cw_levelschd",
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_csr_nnz_balance_analysis_workspace_solve_matches_direct():
    device = torch.device("cuda")
    dtype = torch.float64
    n = 96
    A = torch.tril(torch.randn(n, n, dtype=dtype, device=device) * 0.02)
    A = A + torch.eye(n, dtype=dtype, device=device) * 3.0
    b = _rand_like(dtype, (n,), device)
    Asp = A.to_sparse_csr()

    descr = flagsparse_spsv_analysis_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_nnz_balance",
    )
    assert descr.solve_kind == "csr_nnz_balance"
    workspace = flagsparse_spsv_preprocess_csr(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_csr(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        solve_kind="csr_nnz_balance",
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_csr_analysis_workspace_solve_matches_direct():
    device = torch.device("cuda")
    dtype = torch.float64
    n = SPSV_N[0]
    A = _build_triangular(n, dtype, device, lower=True)
    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)
    b = _rand_like(dtype, (n,), device)

    descr = flagsparse_spsv_analysis_csr(
        data,
        indices,
        indptr,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
    )
    assert isinstance(descr, FlagSparseSpSVDescr)
    assert descr.solve_kind == "csr_cw"
    assert descr.buffer_size == flagsparse_spsv_buffer_size((n, n), dtype, format="csr")

    workspace = flagsparse_spsv_create_workspace(descr)
    assert isinstance(workspace, FlagSparseSpSVWorkspace)
    assert workspace.buffer_size == descr.buffer_size

    x_direct = flagsparse_spsv_csr(
        data, indices, indptr, b, (n, n), lower=True, unit_diagonal=False
    )
    x_via_descr = flagsparse_spsv_solve_csr(descr, b, workspace=workspace)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("op_mode", TRANS_CONJ_MODES)
def test_spsv_csr_transpose_analysis_workspace_route(op_mode):
    device = torch.device("cuda")
    dtype = torch.complex128
    n = SPSV_N[0]
    A = _build_triangular(n, dtype, device, lower=True)
    Asp = A.to_sparse_csr()
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int32)
    indptr = Asp.crow_indices().to(torch.int32)
    b = _rand_like(dtype, (n,), device)

    descr = flagsparse_spsv_analysis_csr(
        data,
        indices,
        indptr,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=_transpose_arg(op_mode),
    )
    assert descr.solve_kind == "transpose_cw"
    assert descr.route_name == "transpose_cw"
    assert descr.solve_plan.get("transpose_lower") is True
    layout_names = [entry["name"] for entry in descr.workspace_layout]
    assert layout_names == ["residual", "indegree", "row_counter"]

    workspace = flagsparse_spsv_create_workspace(descr)
    x_via_descr = flagsparse_spsv_solve_csr(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_csr(
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
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_csr_descriptor_exposes_cuda_style_fields():
    device = torch.device("cuda")
    dtype = torch.float64
    n = SPSV_N[0]
    A = _build_triangular(n, dtype, device, lower=True)
    Asp = A.to_sparse_csr()
    descr = flagsparse_spsv_analysis_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
    )
    assert descr.fill_mode == "lower"
    assert descr.diag_type == "non_unit"
    assert descr.matrix_type == "triangular"
    assert descr.index_base == 0
    assert descr.storage_view == "csr"


@pytest.mark.spsv
def test_spsv_csr_preprocess_initializes_workspace():
    device = torch.device("cuda")
    dtype = torch.float64
    n = SPSV_N[0]
    A = _build_triangular(n, dtype, device, lower=True)
    Asp = A.to_sparse_csr()
    descr = flagsparse_spsv_analysis_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose="TRANS",
        solve_kind="transpose_cw",
    )
    workspace = flagsparse_spsv_create_workspace(descr)
    workspace = flagsparse_spsv_preprocess_csr(descr, workspace=workspace)
    assert isinstance(workspace, FlagSparseSpSVWorkspace)
    indegree_expected = torch.zeros(n, dtype=torch.int32, device=device)
    row_ids = torch.repeat_interleave(
        torch.arange(n, device=device, dtype=torch.int64),
        Asp.crow_indices().to(torch.int64)[1:] - Asp.crow_indices().to(torch.int64)[:-1],
    )
    mask = Asp.col_indices().to(torch.int64) <= row_ids
    if bool(torch.any(mask).item()):
        counts = torch.bincount(
            Asp.col_indices().to(torch.int64)[mask], minlength=n
        ).to(torch.int32)
        indegree_expected.copy_(counts)
    assert torch.equal(workspace.buffers["indegree"], indegree_expected)


@pytest.mark.spsv
def test_spsv_ex_interfaces_match_direct_route():
    device = torch.device("cuda")
    dtype = torch.float64
    n = SPSV_N[0]
    A = _build_triangular(n, dtype, device, lower=True)
    Asp = A.to_sparse_csr()
    b = _rand_like(dtype, (n,), device)

    handle = flagsparse_create_spsv_handle(device=device)
    mat = flagsparse_create_spmat_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
    )
    vec = flagsparse_create_dnvec(b)
    assert isinstance(handle, FlagSparseSpSVHandle)
    assert isinstance(mat, FlagSparseSpMatDescr)
    assert isinstance(vec, FlagSparseDnVecDescr)

    descr = flagsparse_spsv_analysis_ex(
        handle,
        False,
        1,
        mat,
        vec,
        compute_dtype=torch.float64,
    )
    workspace = flagsparse_spsv_preprocess_csr(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_ex = flagsparse_spsv_solve_ex(handle, False, 1, mat, vec, descr=descr, workspace=workspace)
    x_direct = flagsparse_spsv_csr(
        Asp.values(),
        Asp.col_indices().to(torch.int32),
        Asp.crow_indices().to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
    )
    assert flagsparse_spsv_buffer_size_ex(handle, False, 1, mat, vec) == descr.buffer_size
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_ex, x_direct, rtol=rtol, atol=atol)


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
