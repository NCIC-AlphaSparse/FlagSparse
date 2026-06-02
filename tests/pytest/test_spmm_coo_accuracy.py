import pytest
import torch

from flagsparse import flagsparse_spmm_coo

from tests.pytest.accuracy_utils import close_tolerances
from tests.pytest.param_shapes import MNK_SHAPES, SPMM_OPT_DTYPES, SPMM_OPT_DTYPE_IDS


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _random_coo_mk(M, K, dtype, device):
    denom = max(M * K, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = torch.randn(M, K, dtype=dtype, device=device) * mask.to(dtype=dtype)
    return vals.to_sparse_coo().coalesce()


def _tol(dtype):
    return close_tolerances(dtype)


@pytest.mark.spmm_coo
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPMM_OPT_DTYPES, ids=SPMM_OPT_DTYPE_IDS)
def test_spmm_coo_matches_torch(M, N, K, dtype):
    device = torch.device("cuda")
    Asp = _random_coo_mk(M, K, dtype, device)
    indices = Asp.indices()
    data = Asp.values()
    row = indices[0].contiguous()
    col = indices[1].contiguous()
    B = torch.randn(K, N, dtype=dtype, device=device)
    if dtype == torch.float32:
        ref = torch.sparse.mm(Asp.double(), B.double()).float()
    else:
        ref = torch.sparse.mm(Asp, B)
    out = flagsparse_spmm_coo(data, row, col, B, (M, K))
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)
