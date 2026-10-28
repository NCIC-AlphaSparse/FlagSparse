from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="requires torch to exercise dispatch")

from flagsparse.sparse_operations import spmv_csr as spmv_csr_ops  # noqa: E402


@pytest.fixture
def ascend_csr_cpu(monkeypatch):
    monkeypatch.setattr(spmv_csr_ops, "_is_ascend_runtime", lambda: True)
    monkeypatch.setattr(
        spmv_csr_ops, "_is_accel_tensor", lambda tensor: tensor.device.type == "cpu"
    )
    monkeypatch.setattr(
        spmv_csr_ops, "_ACCEL", SimpleNamespace(synchronize=lambda: None)
    )


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex64])
def test_ascend_prepared_csr_preserves_operation(ascend_csr_cpu, dtype, op):
    real = torch.tensor([2, -1, 4], dtype=torch.float64)
    if dtype.is_complex:
        imag = torch.tensor([1, 2, -3], dtype=torch.float64)
        values = torch.complex(real, imag).to(dtype)
        rhs = torch.tensor(
            [1 + 2j, -3 + 1j] if op != "non" else [1 + 2j, -3 + 1j, 2 - 4j],
            dtype=dtype,
        )
    else:
        values = real
        rhs = torch.tensor([1, -3] if op != "non" else [1, -3, 2], dtype=dtype)
    indices = torch.tensor([0, 2, 1], dtype=torch.int32)
    indptr = torch.tensor([0, 2, 3], dtype=torch.int32)
    shape = (2, 3)
    dense = torch.zeros(shape, dtype=dtype)
    dense[0, 0] = values[0]
    dense[0, 2] = values[1]
    dense[1, 1] = values[2]
    prepared = spmv_csr_ops.PreparedCsrSpmv(
        data=values,
        kernel_indices=indices,
        kernel_indptr=indptr,
        shape=shape,
        n_rows=2,
        n_cols=3,
        block_nnz=256,
        max_segments=1,
        max_row_nnz=2,
        op=op,
    )
    if op == "non":
        expected = dense @ rhs
    elif op == "trans":
        expected = dense.t() @ rhs
    else:
        expected = dense.conj().t() @ rhs

    torch.testing.assert_close(
        spmv_csr_ops.flagsparse_spmv_csr(prepared=prepared, x=rhs), expected
    )
    torch.testing.assert_close(
        spmv_csr_ops.flagsparse_spmv_csr(prepared=prepared, x=rhs, op=op), expected
    )


@pytest.mark.parametrize(
    ("prepared_op", "call_op", "transpose"),
    [("non", "trans", None), ("trans", "non", None), ("conj", None, False)],
)
def test_ascend_prepared_csr_rejects_operation_conflicts(
    ascend_csr_cpu, prepared_op, call_op, transpose
):
    values = torch.tensor([1.0], dtype=torch.float32)
    indices = torch.tensor([0], dtype=torch.int32)
    indptr = torch.tensor([0, 1], dtype=torch.int32)
    prepared = spmv_csr_ops.PreparedCsrSpmv(
        data=values,
        kernel_indices=indices,
        kernel_indptr=indptr,
        shape=(1, 1),
        n_rows=1,
        n_cols=1,
        block_nnz=256,
        max_segments=1,
        max_row_nnz=1,
        op=prepared_op,
    )
    with pytest.raises(ValueError, match="does not match prepared"):
        spmv_csr_ops.flagsparse_spmv_csr(
            prepared=prepared,
            x=torch.tensor([2.0]),
            op=call_op,
            transpose=transpose,
        )
