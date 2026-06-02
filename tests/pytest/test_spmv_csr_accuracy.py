import importlib

import pytest
import torch

from flagsparse import flagsparse_spmv_csr
from tests.pytest.accuracy_utils import close_tolerances
from tests.pytest.param_shapes import SPMV_MN_SHAPES


spmv_mod = importlib.import_module("flagsparse.sparse_operations.spmv_csr")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _value_dtype_cases():
    cases = [
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    ]
    return [(name, dtype) for name, dtype in cases if dtype is not None]


def _skip_unavailable_dtype(name, dtype):
    if dtype == torch.bfloat16 and not (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ):
        pytest.skip("bfloat16 not supported on this GPU")


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


def _tol(dtype):
    return close_tolerances(dtype)


def _random_csr_mn(M, N, dtype, index_dtype, device):
    denom = max(M * N, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(M, N, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    dense = torch.where(mask, _random_dense((M, N), dtype, device), torch.zeros((), dtype=dtype, device=device))
    rows, cols = torch.nonzero(mask, as_tuple=True)
    data = dense[rows, cols].contiguous()
    row_counts = torch.bincount(rows, minlength=M)
    indptr = torch.zeros(M + 1, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(row_counts, dim=0)
    return data, cols.to(index_dtype).contiguous(), indptr.to(index_dtype), dense


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


@pytest.mark.spmv_csr
@pytest.mark.parametrize("M, N", SPMV_MN_SHAPES)
@pytest.mark.parametrize("name,dtype", _value_dtype_cases(), ids=[c[0] for c in _value_dtype_cases()])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("op", ["non", "trans", "conj"], ids=["non", "trans", "conj"])
def test_spmv_csr_matches_dense_reference(M, N, name, dtype, index_dtype, op):
    _skip_unavailable_dtype(name, dtype)
    device = torch.device("cuda")
    data, indices, indptr, dense = _random_csr_mn(M, N, dtype, index_dtype, device)
    transpose = _op_transposes(op)
    x_len = M if transpose else N
    x = _make_x(x_len, dtype, device)
    ref_dtype = _reference_dtype(dtype)
    ref_mat = _apply_dense_op(dense, op)
    ref = (ref_mat.to(ref_dtype) @ x.to(ref_dtype)).to(dtype)
    out = flagsparse_spmv_csr(
        data,
        indices,
        indptr,
        x,
        shape=(M, N),
        op=op,
        index_fallback_policy="auto",
    )
    _assert_close(out, ref, dtype)


@pytest.mark.spmv_csr
def test_spmv_csr_prepared_transpose_mismatch_rejected():
    device = torch.device("cuda")
    data, indices, indptr, _dense = _random_csr_mn(8, 10, torch.float32, torch.int32, device)
    prepared = spmv_mod.prepare_spmv_csr(data, indices, indptr, (8, 10), transpose=True)
    x = torch.randn(8, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="does not match prepared.transpose"):
        flagsparse_spmv_csr(x=x, prepared=prepared, transpose=False)


@pytest.mark.spmv_csr
def test_spmv_csr_prepared_op_mismatch_rejected():
    device = torch.device("cuda")
    data, indices, indptr, _dense = _random_csr_mn(8, 10, torch.complex64, torch.int32, device)
    prepared = spmv_mod.prepare_spmv_csr(data, indices, indptr, (8, 10), op="conj")
    x = _make_x(8, torch.complex64, device)
    with pytest.raises(ValueError, match="does not match prepared.op"):
        flagsparse_spmv_csr(x=x, prepared=prepared, op="trans")


@pytest.mark.spmv_csr
def test_spmv_csr_int64_auto_fallback_to_int32(monkeypatch):
    device = torch.device("cuda")
    data, indices, indptr, dense = _random_csr_mn(12, 9, torch.float32, torch.int64, device)
    x = torch.randn(9, dtype=torch.float32, device=device)
    ref = dense.to(torch.float64) @ x.to(torch.float64)
    state = {"forced_once": False}
    original = spmv_mod._triton_spmv_csr_impl_prepared

    def fail_int64_once(prepared, x_in):
        if prepared.kernel_indices.dtype == torch.int64 and not state["forced_once"]:
            state["forced_once"] = True
            raise RuntimeError("forced int64 launch failure")
        return original(prepared, x_in)

    monkeypatch.setattr(spmv_mod, "_triton_spmv_csr_impl_prepared", fail_int64_once)
    out = flagsparse_spmv_csr(
        data,
        indices,
        indptr,
        x,
        shape=(12, 9),
        index_fallback_policy="auto",
    )
    assert state["forced_once"]
    rtol, atol = _tol(torch.float32)
    assert torch.allclose(out.to(torch.float64), ref, rtol=rtol, atol=atol)


@pytest.mark.spmv_csr
def test_spmv_csr_int64_strict_no_fallback(monkeypatch):
    device = torch.device("cuda")
    data, indices, indptr, _dense = _random_csr_mn(12, 9, torch.float32, torch.int64, device)
    x = torch.randn(9, dtype=torch.float32, device=device)

    def fail_int64(prepared, x_in):
        if prepared.kernel_indices.dtype == torch.int64:
            raise RuntimeError("forced int64 launch failure")
        return spmv_mod._triton_spmv_csr_impl_prepared(prepared, x_in)

    monkeypatch.setattr(spmv_mod, "_triton_spmv_csr_impl_prepared", fail_int64)
    with pytest.raises(RuntimeError, match="forced int64 launch failure"):
        flagsparse_spmv_csr(
            data,
            indices,
            indptr,
            x,
            shape=(12, 9),
            index_fallback_policy="strict",
        )


@pytest.mark.spmv_csr
def test_spmv_csr_int64_auto_does_not_fallback_when_index_exceeds_int32(monkeypatch):
    device = torch.device("cuda")
    limit = spmv_mod._INDEX_LIMIT_INT32
    data = torch.ones(1, dtype=torch.float32, device=device)
    prepared = spmv_mod.PreparedCsrSpmv(
        data=data,
        kernel_indices=torch.tensor([limit + 1], dtype=torch.int64, device=device),
        kernel_indptr=torch.tensor([0, 1], dtype=torch.int64, device=device),
        shape=(1, limit + 2),
        n_rows=1,
        n_cols=limit + 2,
        block_nnz=256,
        max_segments=1,
        max_row_nnz=1,
        opt_buckets=[],
        transpose=False,
        index_fallback_policy="auto",
    )

    def fail_launch(_prepared, _x, use_opt=False):
        raise RuntimeError("forced native int64 failure")

    monkeypatch.setattr(spmv_mod, "_run_spmv_prepared", fail_launch)
    x = torch.empty(0, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError, match="int32 fallback is unsafe"):
        spmv_mod._run_spmv_prepared_with_fallback(prepared, x, use_opt=False)
