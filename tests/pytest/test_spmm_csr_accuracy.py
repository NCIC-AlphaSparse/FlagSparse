import pytest
import torch

from flagsparse import (
    flagsparse_spmm_csr,
    flagsparse_spmm_csr_opt,
    prepare_spmm_csr_opt,
)

from tests.pytest.param_shapes import (
    MNK_SHAPES,
    SPMM_FLOAT_DTYPES,
    SPMM_FLOAT_DTYPE_IDS,
    SPMM_OPT_DTYPES,
    SPMM_OPT_DTYPE_IDS,
)
from tests.pytest.accuracy_utils import close_tolerances


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _random_csr_mk(M, K, dtype, device):
    denom = max(M * K, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = torch.randn(M, K, dtype=dtype, device=device) * mask.to(dtype=dtype)
    return vals.to_sparse_csr()


def _plain_sparse_values(sparse_tensor):
    return sparse_tensor.values().clone()


def _random_dense(shape, dtype, device):
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.complex64:
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if dtype == torch.complex128:
        real = torch.randn(shape, dtype=torch.float64, device=device)
        imag = torch.randn(shape, dtype=torch.float64, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"unsupported dtype: {dtype}")


def _reference_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _apply_dense_op(dense, op):
    if op == "non":
        return dense
    if op == "trans":
        return dense.t()
    if op == "conj":
        return dense.conj().t()
    raise ValueError(f"unsupported op: {op}")


def _tol(dtype):
    return close_tolerances(dtype)


SPMM_OP_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
SPMM_OP_DTYPE_IDS = ("float32", "float64", "complex64", "complex128")


@pytest.mark.spmm_csr
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPMM_FLOAT_DTYPES, ids=SPMM_FLOAT_DTYPE_IDS)
def test_spmm_csr_matches_torch(M, N, K, dtype):
    if dtype == torch.bfloat16 and not (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ):
        pytest.skip("bfloat16 not supported on this GPU")
    device = torch.device("cuda")
    Asp = _random_csr_mk(M, K, dtype, device)
    data = _plain_sparse_values(Asp)
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B = torch.randn(K, N, dtype=dtype, device=device)
    if dtype == torch.float32:
        Asp64 = torch.sparse_csr_tensor(
            crow_indices=indptr,
            col_indices=indices,
            values=data.double(),
            size=(M, K),
            dtype=torch.float64,
            device=device,
        )
        ref = torch.sparse.mm(Asp64, B.double()).float()
    else:
        ref = torch.sparse.mm(Asp, B)
    out = flagsparse_spmm_csr(data, indices, indptr, B, (M, K))
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)


@pytest.mark.spmm_csr
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPMM_OP_DTYPES, ids=SPMM_OP_DTYPE_IDS)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize("op", ["non", "trans", "conj"], ids=["non", "trans", "conj"])
def test_spmm_csr_op_matches_dense_reference(M, N, K, dtype, index_dtype, op):
    device = torch.device("cuda")
    Asp = _random_csr_mk(M, K, dtype, device)
    data = _plain_sparse_values(Asp)
    indices = Asp.col_indices().to(index_dtype)
    indptr = Asp.crow_indices().to(index_dtype)
    dense = Asp.to_dense()
    effective_cols = M if op in ("trans", "conj") else K
    B = _random_dense((effective_cols, N), dtype, device)
    ref_dtype = _reference_dtype(dtype)
    ref = (_apply_dense_op(dense, op).to(ref_dtype) @ B.to(ref_dtype)).to(dtype)
    out = flagsparse_spmm_csr(data, indices, indptr, B, (M, K), op=op)
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)


@pytest.mark.spmm_csr
def test_spmm_csr_return_meta_times_transpose_path():
    device = torch.device("cuda")
    Asp = _random_csr_mk(8, 12, torch.float32, device)
    data = _plain_sparse_values(Asp)
    indices = Asp.col_indices().to(torch.int64)
    indptr = Asp.crow_indices().to(torch.int64)
    B = torch.randn(8, 4, dtype=torch.float32, device=device)
    out, elapsed_ms, meta = flagsparse_spmm_csr(
        data,
        indices,
        indptr,
        B,
        (8, 12),
        op="trans",
        return_time=True,
        return_meta=True,
    )
    assert out.shape == (12, 4)
    assert meta["op"] == "trans"
    assert meta["symbolic_ms"] >= 0.0
    assert meta["compute_ms"] >= 0.0
    assert elapsed_ms == pytest.approx(meta["op_total_ms"])
    assert meta["op_total_ms"] == pytest.approx(
        meta["symbolic_ms"] + meta["compute_ms"]
    )


@pytest.mark.spmm_csr
def test_spmm_csr_non_meta_has_zero_symbolic_time():
    device = torch.device("cuda")
    Asp = _random_csr_mk(8, 12, torch.float32, device)
    data = _plain_sparse_values(Asp)
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B = torch.randn(12, 4, dtype=torch.float32, device=device)
    out, meta = flagsparse_spmm_csr(
        data,
        indices,
        indptr,
        B,
        (8, 12),
        op="non",
        return_meta=True,
    )
    assert out.shape == (8, 4)
    assert meta["op"] == "non"
    assert meta["symbolic_ms"] == 0.0


@pytest.mark.spmm_csr
def test_spmm_csr_op_validation_errors():
    device = torch.device("cuda")
    Asp = _random_csr_mk(8, 12, torch.float32, device)
    data = _plain_sparse_values(Asp)
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B_non = torch.randn(12, 4, dtype=torch.float32, device=device)
    B_trans = torch.randn(8, 4, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="op must be one of"):
        flagsparse_spmm_csr(data, indices, indptr, B_non, (8, 12), op="bad")
    with pytest.raises(ValueError, match="transpose conflicts with op"):
        flagsparse_spmm_csr(
            data, indices, indptr, B_non, (8, 12), op="non", transpose=True
        )
    with pytest.raises(ValueError, match="transpose conflicts with op"):
        flagsparse_spmm_csr(
            data, indices, indptr, B_trans, (8, 12), op="trans", transpose=False
        )
    with pytest.raises(ValueError, match="B.shape\\[0\\] must be n_cols=8"):
        flagsparse_spmm_csr(data, indices, indptr, B_non, (8, 12), op="trans")
    with pytest.raises(ValueError, match="out shape/dtype must match result"):
        flagsparse_spmm_csr(
            data,
            indices,
            indptr,
            B_trans,
            (8, 12),
            op="trans",
            out=torch.empty((8, 4), dtype=torch.float32, device=device),
        )


@pytest.mark.spmm_csr_opt_alg1
@pytest.mark.spmm_csr_opt
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPMM_OPT_DTYPES, ids=SPMM_OPT_DTYPE_IDS)
def test_spmm_csr_opt_matches_torch(M, N, K, dtype):
    device = torch.device("cuda")
    Asp = _random_csr_mk(M, K, dtype, device)
    data = _plain_sparse_values(Asp)
    indices = Asp.col_indices()
    indptr = Asp.crow_indices()
    B = torch.randn(K, N, dtype=dtype, device=device)
    if dtype == torch.float32:
        Asp64 = torch.sparse_csr_tensor(
            crow_indices=indptr,
            col_indices=indices,
            values=data.double(),
            size=(M, K),
            dtype=torch.float64,
            device=device,
        )
        ref = torch.sparse.mm(Asp64, B.double()).float()
    else:
        ref = torch.sparse.mm(Asp, B)
    prepared = prepare_spmm_csr_opt(data, indices, indptr, (M, K))
    out = flagsparse_spmm_csr_opt(B=B, prepared=prepared)
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)
