import pytest
import torch

from flagsparse import (
    flagsparse_spmv_coo,
    flagsparse_spmv_coo_tocsr,
    prepare_spmv_coo,
    prepare_spmv_coo_tocsr,
)
import flagsparse.sparse_operations.spmv_coo as spmv_coo_mod

from tests.pytest.accuracy_utils import close_tolerances
from tests.pytest.param_shapes import (
    SPMV_COO_DTYPES,
    SPMV_COO_DTYPE_IDS,
    SPMV_MN_SHAPES,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


_TOCSR_DTYPES = (torch.float32, torch.float64)
_TOCSR_DTYPE_IDS = ("float32", "float64")


def _random_dense(shape, dtype, device):
    if dtype in (torch.float32, torch.float64):
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
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _random_coo_mn(M, N, dtype, device):
    denom = max(M * N, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, N, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    dense = torch.where(
        mask,
        _random_dense((M, N), dtype, device),
        torch.zeros((), dtype=dtype, device=device),
    )
    return dense.to_sparse_coo().coalesce(), dense


def _tol(dtype):
    return close_tolerances(dtype)


def _make_x(length, dtype, device):
    return _random_dense((length,), dtype, device)


def _op_transposes(op):
    return op in ("trans", "conj")


def _apply_dense_op(dense, op):
    if op == "non":
        return dense
    if op == "trans":
        return dense.t()
    if op == "conj":
        return dense.conj().t()
    raise ValueError(f"unsupported op: {op}")


def _assert_close(actual, expected, dtype):
    rtol, atol = _tol(dtype)
    ref_dtype = _reference_dtype(dtype)
    assert torch.allclose(actual.to(ref_dtype), expected.to(ref_dtype), rtol=rtol, atol=atol)


@pytest.mark.spmv_coo
@pytest.mark.parametrize("M, N", SPMV_MN_SHAPES)
@pytest.mark.parametrize("dtype", SPMV_COO_DTYPES, ids=SPMV_COO_DTYPE_IDS)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("op", ["non", "trans", "conj"], ids=["non", "trans", "conj"])
def test_spmv_coo_matches_dense_reference(M, N, dtype, index_dtype, op):
    device = torch.device("cuda")
    Asp, dense = _random_coo_mn(M, N, dtype, device)
    data = Asp.values()
    indices = Asp.indices()
    row = indices[0].to(index_dtype).contiguous()
    col = indices[1].to(index_dtype).contiguous()
    x_len = M if _op_transposes(op) else N
    x = _make_x(x_len, dtype, device)
    ref_dtype = _reference_dtype(dtype)
    ref = (_apply_dense_op(dense, op).to(ref_dtype) @ x.to(ref_dtype)).to(dtype)
    out = flagsparse_spmv_coo(data, row, col, x, shape=(M, N), op=op)
    _assert_close(out, ref, dtype)


@pytest.mark.spmv_coo
def test_spmv_coo_prepared_reuses_structure_across_ops():
    device = torch.device("cuda")
    Asp, dense = _random_coo_mn(8, 10, torch.complex64, device)
    data = Asp.values()
    indices = Asp.indices()
    row = indices[0].to(torch.int32).contiguous()
    col = indices[1].to(torch.int32).contiguous()
    prepared = prepare_spmv_coo(data, row, col, (8, 10))

    x_non = _make_x(10, torch.complex64, device)
    ref_non = dense.to(torch.complex128) @ x_non.to(torch.complex128)
    out_non = flagsparse_spmv_coo(x=x_non, prepared=prepared, op="non")
    _assert_close(out_non, ref_non.to(torch.complex64), torch.complex64)

    x_trans = _make_x(8, torch.complex64, device)
    ref_trans = dense.t().to(torch.complex128) @ x_trans.to(torch.complex128)
    out_trans = flagsparse_spmv_coo(x=x_trans, prepared=prepared, op="trans")
    _assert_close(out_trans, ref_trans.to(torch.complex64), torch.complex64)

    ref_conj = dense.conj().t().to(torch.complex128) @ x_trans.to(torch.complex128)
    out_conj = flagsparse_spmv_coo(x=x_trans, prepared=prepared, op="conj")
    _assert_close(out_conj, ref_conj.to(torch.complex64), torch.complex64)


@pytest.mark.spmv_coo
@pytest.mark.parametrize("op", ["trans", "conj"], ids=["trans", "conj"])
def test_spmv_coo_runtime_launch_matches_public_api_for_ops(op):
    device = torch.device("cuda")
    Asp, _dense = _random_coo_mn(7, 9, torch.complex64, device)
    data = Asp.values()
    indices = Asp.indices()
    row = indices[0].to(torch.int32).contiguous()
    col = indices[1].to(torch.int32).contiguous()
    x = _make_x(7, torch.complex64, device)

    expected = flagsparse_spmv_coo(data, row, col, x, shape=(7, 9), op=op)
    launch = spmv_coo_mod._prepare_spmv_coo_launch_from_raw(
        data=data,
        row=row,
        col=col,
        shape=(7, 9),
        sort_by_row=True,
        op=op,
    )
    actual = spmv_coo_mod._run_spmv_coo_prepared_with_fallback(
        launch,
        x,
        block_size=256,
        num_warps=4,
        block_inner=128,
    )
    _assert_close(actual, expected, torch.complex64)


@pytest.mark.spmv_coo
def test_spmv_coo_prepared_explicit_transpose_conflict_rejected():
    device = torch.device("cuda")
    Asp, _dense = _random_coo_mn(8, 10, torch.float32, device)
    data = Asp.values()
    indices = Asp.indices()
    row = indices[0].to(torch.int32).contiguous()
    col = indices[1].to(torch.int32).contiguous()
    prepared = prepare_spmv_coo(data, row, col, (8, 10), transpose=True)
    x = torch.randn(10, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="transpose conflicts with op"):
        flagsparse_spmv_coo(x=x, prepared=prepared, op="non", transpose=True)


@pytest.mark.spmv_coo
def test_spmv_coo_int64_auto_fallback_to_int32(monkeypatch):
    device = torch.device("cuda")
    Asp, dense = _random_coo_mn(12, 9, torch.float32, device)
    data = Asp.values()
    indices = Asp.indices()
    row = indices[0].to(torch.int64).contiguous()
    col = indices[1].to(torch.int64).contiguous()
    x = torch.randn(9, dtype=torch.float32, device=device)
    ref = dense.to(torch.float64) @ x.to(torch.float64)
    state = {"forced_once": False}
    original = spmv_coo_mod._triton_spmv_coo_kernel

    def fail_int64_once(prepared, x_in, block_size, num_warps, block_inner):
        if prepared.row.dtype == torch.int64 and not state["forced_once"]:
            state["forced_once"] = True
            raise RuntimeError("forced int64 launch failure")
        return original(prepared, x_in, block_size, num_warps, block_inner)

    monkeypatch.setattr(spmv_coo_mod, "_triton_spmv_coo_kernel", fail_int64_once)
    out = flagsparse_spmv_coo(
        data,
        row,
        col,
        x,
        shape=(12, 9),
        index_fallback_policy="auto",
    )
    assert state["forced_once"]
    rtol, atol = _tol(torch.float32)
    assert torch.allclose(out.to(torch.float64), ref, rtol=rtol, atol=atol)


@pytest.mark.spmv_coo_tocsr
@pytest.mark.parametrize("M, N", SPMV_MN_SHAPES)
@pytest.mark.parametrize("dtype", _TOCSR_DTYPES, ids=_TOCSR_DTYPE_IDS)
def test_spmv_coo_tocsr_matches_torch(M, N, dtype):
    device = torch.device("cuda")
    Asp, _dense = _random_coo_mn(M, N, dtype, device)
    Asp = Asp.coalesce()
    data = Asp.values()
    indices = Asp.indices()
    row = indices[0].contiguous()
    col = indices[1].contiguous()
    x = torch.randn(N, dtype=dtype, device=device)
    ref = torch.sparse.mm(Asp, x.unsqueeze(1)).squeeze(1)
    out = flagsparse_spmv_coo_tocsr(data, row, col, x, shape=(M, N))
    rtol, atol = _tol(dtype)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)


@pytest.mark.spmv_coo_tocsr
def test_spmv_coo_tocsr_prepared_path_matches_torch():
    device = torch.device("cuda")
    M, N = 8, 10
    dtype = torch.float32
    Asp, _dense = _random_coo_mn(M, N, dtype, device)
    Asp = Asp.coalesce()
    data = Asp.values()
    indices = Asp.indices()
    row = indices[0].contiguous()
    col = indices[1].contiguous()
    x = torch.randn(N, dtype=dtype, device=device)
    prepared = prepare_spmv_coo_tocsr(data, row, col, (M, N))
    ref = torch.sparse.mm(Asp, x.unsqueeze(1)).squeeze(1)
    out = flagsparse_spmv_coo_tocsr(x=x, prepared=prepared)
    rtol, atol = _tol(torch.float32)
    assert torch.allclose(out, ref, rtol=rtol, atol=atol)
