import pytest
import torch

from flagsparse import (
    flagsparse_spmm_csr_opt_alg2,
    prepare_spmm_csr_opt_alg2,
)

from tests.pytest.accuracy_utils import close_tolerances
from tests.pytest.param_shapes import MNK_SHAPES, SPMM_OPT_DTYPES, SPMM_OPT_DTYPE_IDS


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


_DENSE_COLS = (1, 8, 32, 96)
_INDEX_DTYPES = (torch.int32, torch.int64)
_INDEX_DTYPE_IDS = ("int32", "int64")


def _random_csr_mk(M, K, dtype, device):
    denom = max(M * K, 1)
    p = min(0.35, max(0.08, 48.0 / denom))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = torch.randn(M, K, dtype=dtype, device=device) * mask.to(dtype=dtype)
    return vals.to_sparse_csr()


def _tol(dtype):
    return close_tolerances(dtype)


def _torch_reference(Asp, data, indices, indptr, B, shape, dtype, index_dtype):
    if dtype == torch.float32:
        Asp64 = torch.sparse_csr_tensor(
            crow_indices=indptr.to(torch.int64),
            col_indices=indices.to(torch.int64),
            values=data.double(),
            size=shape,
            dtype=torch.float64,
            device=B.device,
        )
        return torch.sparse.mm(Asp64, B.double()).float()
    return torch.sparse.mm(
        torch.sparse_csr_tensor(
            crow_indices=indptr.to(torch.int64),
            col_indices=indices.to(torch.int64),
            values=data,
            size=shape,
            dtype=dtype,
            device=B.device,
        ),
        B,
    )


@pytest.mark.spmm_csr_opt_alg2
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPMM_OPT_DTYPES, ids=SPMM_OPT_DTYPE_IDS)
@pytest.mark.parametrize("dense_cols", _DENSE_COLS)
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES, ids=_INDEX_DTYPE_IDS)
def test_spmm_csr_opt_alg2_matches_torch(M, N, K, dtype, dense_cols, index_dtype):
    device = torch.device("cuda")
    Asp = _random_csr_mk(M, K, dtype, device)
    data = Asp.values()
    indices = Asp.col_indices().to(index_dtype)
    indptr = Asp.crow_indices().to(index_dtype)
    B = torch.randn(K, dense_cols, dtype=dtype, device=device)
    ref = _torch_reference(Asp, data, indices, indptr, B, (M, K), dtype, index_dtype)

    out = flagsparse_spmm_csr_opt_alg2(data, indices, indptr, B, (M, K))
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)


@pytest.mark.spmm_csr_opt_alg2
@pytest.mark.parametrize("dtype", SPMM_OPT_DTYPES, ids=SPMM_OPT_DTYPE_IDS)
def test_spmm_csr_opt_alg2_prepared_out_and_meta(dtype):
    device = torch.device("cuda")
    M, K, N = 16, 32, 24
    Asp = _random_csr_mk(M, K, dtype, device)
    data = Asp.values()
    indices = Asp.col_indices().to(torch.int64)
    indptr = Asp.crow_indices().to(torch.int64)
    B = torch.randn(K, N, dtype=dtype, device=device)
    prepared = prepare_spmm_csr_opt_alg2(data, indices, indptr, (M, K))

    out_buf = torch.empty((M, N), dtype=dtype, device=device)
    out, meta = flagsparse_spmm_csr_opt_alg2(
        B=B,
        prepared=prepared,
        out=out_buf,
        return_meta=True,
    )
    ref = _torch_reference(Asp, data, indices, indptr, B, (M, K), dtype, torch.int64)
    rtol, atol = _tol(dtype)
    assert out is out_buf
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)
    assert meta["device_name"]
    assert isinstance(meta["sm_count"], int)
    assert isinstance(meta["warp_size"], int)
    assert meta["max_row_nnz"] == prepared.max_row_nnz
    assert meta["n_dense_cols"] == N
    assert meta["bucket_hits"]
    assert meta["launch_configs"]


@pytest.mark.spmm_csr_opt_alg2
def test_spmm_csr_opt_alg2_return_time_and_meta_shapes():
    device = torch.device("cuda")
    dtype = torch.float32
    M, K, N = 8, 16, 7
    Asp = _random_csr_mk(M, K, dtype, device)
    data = Asp.values()
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B = torch.randn(K, N, dtype=dtype, device=device)
    out, elapsed_ms, meta = flagsparse_spmm_csr_opt_alg2(
        data=data,
        indices=indices,
        indptr=indptr,
        B=B,
        shape=(M, K),
        return_time=True,
        return_meta=True,
    )
    assert out.shape == (M, N)
    assert isinstance(elapsed_ms, float)
    assert meta["n_dense_cols"] == N
