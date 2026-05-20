import pytest
import torch

from flagsparse import (
    flagsparse_alpha_spmm_alg1_tle,
    flagsparse_alpha_spmm_alg1_tle_opt,
    flagsparse_alpha_spmm_alg1,
    is_alpha_spmm_alg1_tle_opt_available,
    is_alpha_spmm_alg1_tle_available,
    prepare_alpha_spmm_alg1_tle_opt,
    prepare_alpha_spmm_alg1_tle,
    prepare_alpha_spmm_alg1,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _random_csr_mk(M, K, dtype, device):
    denom = max(M * K, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = torch.randn(M, K, dtype=dtype, device=device) * mask.to(dtype=dtype)
    return vals.to_sparse_csr()


def _tol(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-10, 1e-8


def _reference(Asp, B, dtype):
    if dtype == torch.float32:
        Asp64 = torch.sparse_csr_tensor(
            crow_indices=Asp.crow_indices(),
            col_indices=Asp.col_indices(),
            values=Asp.values().double(),
            size=Asp.shape,
            dtype=torch.float64,
            device=Asp.device,
        )
        return torch.sparse.mm(Asp64, B.double()).float()
    return torch.sparse.mm(Asp, B)


@pytest.mark.alpha_spmm_alg1
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_alpha_spmm_alg1_matches_torch(dtype):
    device = torch.device("cuda")
    M, K, N = 96, 80, 48
    Asp = _random_csr_mk(M, K, dtype, device)
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B = torch.randn(K, N, dtype=dtype, device=device)
    out = flagsparse_alpha_spmm_alg1(data, indices, indptr, B, (M, K))
    ref = _reference(Asp, B, dtype)
    atol, rtol = _tol(dtype)
    assert torch.allclose(out, ref, atol=atol, rtol=rtol)


@pytest.mark.alpha_spmm_alg1
@pytest.mark.skipif(not is_alpha_spmm_alg1_tle_available(), reason="TLE runtime unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_alpha_spmm_alg1_tle_matches_torch(dtype):
    device = torch.device("cuda")
    M, K, N = 96, 80, 48
    Asp = _random_csr_mk(M, K, dtype, device)
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B = torch.randn(K, N, dtype=dtype, device=device)
    out = flagsparse_alpha_spmm_alg1_tle(data, indices, indptr, B, (M, K))
    ref = _reference(Asp, B, dtype)
    atol, rtol = _tol(dtype)
    assert torch.allclose(out, ref, atol=atol, rtol=rtol)


@pytest.mark.alpha_spmm_alg1
@pytest.mark.skipif(not is_alpha_spmm_alg1_tle_opt_available(), reason="TLEOpt runtime unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_alpha_spmm_alg1_tle_opt_matches_torch(dtype):
    device = torch.device("cuda")
    M, K, N = 96, 80, 48
    Asp = _random_csr_mk(M, K, dtype, device)
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B = torch.randn(K, N, dtype=dtype, device=device)
    out = flagsparse_alpha_spmm_alg1_tle_opt(data, indices, indptr, B, (M, K))
    ref = _reference(Asp, B, dtype)
    atol, rtol = _tol(dtype)
    assert torch.allclose(out, ref, atol=atol, rtol=rtol)


@pytest.mark.alpha_spmm_alg1
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_alpha_spmm_alg1_prepare_matches_raw(dtype):
    device = torch.device("cuda")
    M, K, N = 72, 64, 33
    Asp = _random_csr_mk(M, K, dtype, device)
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B = torch.randn(K, N, dtype=dtype, device=device)
    prepared = prepare_alpha_spmm_alg1(data, indices, indptr, (M, K))
    raw_out = flagsparse_alpha_spmm_alg1(data, indices, indptr, B, (M, K))
    prepared_out = flagsparse_alpha_spmm_alg1(B=B, prepared=prepared)
    atol, rtol = _tol(dtype)
    assert torch.allclose(raw_out, prepared_out, atol=atol, rtol=rtol)


@pytest.mark.alpha_spmm_alg1
@pytest.mark.skipif(not is_alpha_spmm_alg1_tle_available(), reason="TLE runtime unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_alpha_spmm_alg1_tle_prepare_matches_raw(dtype):
    device = torch.device("cuda")
    M, K, N = 72, 64, 33
    Asp = _random_csr_mk(M, K, dtype, device)
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B = torch.randn(K, N, dtype=dtype, device=device)
    prepared = prepare_alpha_spmm_alg1_tle(data, indices, indptr, (M, K))
    raw_out = flagsparse_alpha_spmm_alg1_tle(data, indices, indptr, B, (M, K))
    prepared_out = flagsparse_alpha_spmm_alg1_tle(B=B, prepared=prepared)
    atol, rtol = _tol(dtype)
    assert torch.allclose(raw_out, prepared_out, atol=atol, rtol=rtol)


@pytest.mark.alpha_spmm_alg1
@pytest.mark.skipif(not is_alpha_spmm_alg1_tle_opt_available(), reason="TLEOpt runtime unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_alpha_spmm_alg1_tle_opt_prepare_matches_raw(dtype):
    device = torch.device("cuda")
    M, K, N = 72, 64, 33
    Asp = _random_csr_mk(M, K, dtype, device)
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B = torch.randn(K, N, dtype=dtype, device=device)
    prepared = prepare_alpha_spmm_alg1_tle_opt(data, indices, indptr, (M, K))
    raw_out = flagsparse_alpha_spmm_alg1_tle_opt(data, indices, indptr, B, (M, K))
    prepared_out = flagsparse_alpha_spmm_alg1_tle_opt(B=B, prepared=prepared)
    atol, rtol = _tol(dtype)
    assert torch.allclose(raw_out, prepared_out, atol=atol, rtol=rtol)


@pytest.mark.alpha_spmm_alg1
@pytest.mark.parametrize(
    "dense_cols, expected_warp_size, expected_factor",
    [
        (1, 4, 1),
        (4, 4, 1),
        (5, 8, 1),
        (8, 8, 1),
        (9, 16, 1),
        (16, 16, 1),
        (17, 32, 1),
        (32, 32, 1),
        (33, 32, 2),
        (64, 32, 2),
        (65, 32, 4),
    ],
)
def test_alpha_spmm_alg1_launch_heuristics_match_alphasparse(
    dense_cols,
    expected_warp_size,
    expected_factor,
):
    device = torch.device("cuda")
    dtype = torch.float32
    M, K = 40, 48
    Asp = _random_csr_mk(M, K, dtype, device)
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B = torch.randn(K, dense_cols, dtype=dtype, device=device)
    _, meta = flagsparse_alpha_spmm_alg1(
        data,
        indices,
        indptr,
        B,
        (M, K),
        return_meta=True,
    )
    assert meta["warp_size"] == expected_warp_size
    assert meta["factor"] == expected_factor
    assert meta["block_cols"] == expected_warp_size * expected_factor
    assert meta["route"] == "alpha_spmm_alg1"

    if is_alpha_spmm_alg1_tle_opt_available():
        _, opt_meta = flagsparse_alpha_spmm_alg1_tle_opt(
            data,
            indices,
            indptr,
            B,
            (M, K),
            return_meta=True,
        )
        assert opt_meta["warp_size"] == expected_warp_size
        assert opt_meta["factor"] == expected_factor
        assert opt_meta["block_cols"] == expected_warp_size * expected_factor
        assert opt_meta["route"] == "alpha_spmm_alg1_tle_opt"
        assert opt_meta["loop_strategy"] == "block_local_row_nnz"


@pytest.mark.alpha_spmm_alg1
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_alpha_spmm_alg1_handles_empty_rows_and_tail_columns(dtype):
    device = torch.device("cuda")
    n_rows, n_cols, dense_cols = 5, 9, 17
    data = torch.tensor([1.5, -2.0, 3.0, 4.0], dtype=dtype, device=device)
    indices = torch.tensor([0, 4, 2, 8], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 2, 2, 3, 3, 4], dtype=torch.int64, device=device)
    B = torch.randn(n_cols, dense_cols, dtype=dtype, device=device)
    Asp = torch.sparse_csr_tensor(indptr, indices.to(torch.int64), data, size=(n_rows, n_cols), device=device)
    out = flagsparse_alpha_spmm_alg1(data, indices, indptr, B, (n_rows, n_cols))
    ref = _reference(Asp, B, dtype)
    atol, rtol = _tol(dtype)
    assert torch.allclose(out, ref, atol=atol, rtol=rtol)


@pytest.mark.alpha_spmm_alg1
@pytest.mark.skipif(not is_alpha_spmm_alg1_tle_available(), reason="TLE runtime unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_alpha_spmm_alg1_tle_handles_empty_rows_and_tail_columns(dtype):
    device = torch.device("cuda")
    n_rows, n_cols, dense_cols = 5, 9, 17
    data = torch.tensor([1.5, -2.0, 3.0, 4.0], dtype=dtype, device=device)
    indices = torch.tensor([0, 4, 2, 8], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 2, 2, 3, 3, 4], dtype=torch.int64, device=device)
    B = torch.randn(n_cols, dense_cols, dtype=dtype, device=device)
    Asp = torch.sparse_csr_tensor(indptr, indices.to(torch.int64), data, size=(n_rows, n_cols), device=device)
    out = flagsparse_alpha_spmm_alg1_tle(data, indices, indptr, B, (n_rows, n_cols))
    ref = _reference(Asp, B, dtype)
    atol, rtol = _tol(dtype)
    assert torch.allclose(out, ref, atol=atol, rtol=rtol)


@pytest.mark.alpha_spmm_alg1
@pytest.mark.skipif(not is_alpha_spmm_alg1_tle_opt_available(), reason="TLEOpt runtime unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_alpha_spmm_alg1_tle_opt_handles_empty_rows_and_tail_columns(dtype):
    device = torch.device("cuda")
    n_rows, n_cols, dense_cols = 5, 9, 17
    data = torch.tensor([1.5, -2.0, 3.0, 4.0], dtype=dtype, device=device)
    indices = torch.tensor([0, 4, 2, 8], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 2, 2, 3, 3, 4], dtype=torch.int64, device=device)
    B = torch.randn(n_cols, dense_cols, dtype=dtype, device=device)
    Asp = torch.sparse_csr_tensor(indptr, indices.to(torch.int64), data, size=(n_rows, n_cols), device=device)
    out = flagsparse_alpha_spmm_alg1_tle_opt(data, indices, indptr, B, (n_rows, n_cols))
    ref = _reference(Asp, B, dtype)
    atol, rtol = _tol(dtype)
    assert torch.allclose(out, ref, atol=atol, rtol=rtol)
