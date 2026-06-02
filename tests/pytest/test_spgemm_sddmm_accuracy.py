import pytest
import torch

from flagsparse import flagsparse_sddmm_csr, flagsparse_spgemm_csr

from tests.pytest.accuracy_utils import close_tolerances
from tests.pytest.param_shapes import (
    SDDMM_DTYPES,
    SDDMM_DTYPE_IDS,
    SDDMM_MNK_SHAPES,
    SPGEMM_DTYPES,
    SPGEMM_DTYPE_IDS,
    SPGEMM_MNK_SHAPES,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _random_csr(rows, cols, dtype, device):
    denom = max(rows * cols, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(rows, cols, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = torch.randn(rows, cols, dtype=dtype, device=device) * mask.to(dtype=dtype)
    return vals.to_sparse_csr()


def _csr_to_dense(data, indices, indptr, shape):
    csr = torch.sparse_csr_tensor(
        indptr,
        indices,
        data,
        size=shape,
        dtype=data.dtype,
        device=data.device,
    )
    return csr.to_dense()


def _tol(dtype):
    return close_tolerances(dtype)


@pytest.mark.spgemm_csr
@pytest.mark.parametrize("M, N, K", SPGEMM_MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPGEMM_DTYPES, ids=SPGEMM_DTYPE_IDS)
def test_spgemm_csr_matches_torch(M, N, K, dtype):
    device = torch.device("cuda")
    A = _random_csr(M, K, dtype, device)
    B = _random_csr(K, N, dtype, device)
    c_data, c_indices, c_indptr, c_shape = flagsparse_spgemm_csr(
        A.values(),
        A.col_indices().to(torch.int32),
        A.crow_indices(),
        (M, K),
        B.values(),
        B.col_indices().to(torch.int32),
        B.crow_indices(),
        (K, N),
    )
    got = _csr_to_dense(c_data, c_indices, c_indptr, c_shape)
    ref = torch.sparse.mm(A, B.to_dense())
    rtol, atol = _tol(dtype)
    assert torch.allclose(got, ref, rtol=rtol, atol=atol)


@pytest.mark.sddmm_csr
@pytest.mark.parametrize("M, N, K", SDDMM_MNK_SHAPES)
@pytest.mark.parametrize("dtype", SDDMM_DTYPES, ids=SDDMM_DTYPE_IDS)
def test_sddmm_csr_matches_sampled_dense_reference(M, N, K, dtype):
    device = torch.device("cuda")
    pattern = _random_csr(M, N, dtype, device)
    indices = pattern.col_indices().to(torch.int32)
    indptr = pattern.crow_indices()
    data = pattern.values()
    x = torch.randn(M, K, dtype=dtype, device=device)
    y = torch.randn(N, K, dtype=dtype, device=device)
    alpha = 1.25
    beta = 0.5

    got = flagsparse_sddmm_csr(
        data=data,
        indices=indices,
        indptr=indptr,
        x=x,
        y=y,
        shape=(M, N),
        alpha=alpha,
        beta=beta,
    )
    row_ids = torch.repeat_interleave(
        torch.arange(M, dtype=torch.int64, device=device),
        indptr[1:] - indptr[:-1],
    )
    ref = alpha * torch.sum(x[row_ids] * y[indices.to(torch.int64)], dim=1) + beta * data
    rtol, atol = _tol(dtype)
    assert torch.allclose(got, ref, rtol=rtol, atol=atol)
