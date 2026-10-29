"""Real-device correctness for the benchmark-only CANN sparse adapter."""

import pytest
import torch

from tests import ascend_sparse_baseline as cann
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device
from flagsparse.sparse_operations._common import _is_ascend_runtime


pytestmark = pytest.mark.skipif(
    not (_is_ascend_runtime() and accelerator_available()), reason="real Ascend runtime required"
)


def _operands(fmt, dtype):
    values = torch.tensor([1, -2, 3], dtype=dtype)
    if dtype.is_complex:
        values += torch.tensor([0.5, -1.5, 2], dtype=dtype) * 1j
    matrix = torch.zeros(3, 4, dtype=dtype)
    matrix[1, 0], matrix[1, 2], matrix[2, 1] = values
    if fmt == "coo":
        indices = torch.tensor([1, 1, 2], dtype=torch.int32)
        offsets = torch.tensor([0, 2, 1], dtype=torch.int64)
    elif fmt == "csr":
        indices = torch.tensor([0, 2, 1], dtype=torch.int32)
        offsets = torch.tensor([0, 0, 2, 3], dtype=torch.int64)
    else:
        values = values[[0, 2, 1]]
        indices = torch.tensor([1, 2, 1], dtype=torch.int64)
        offsets = torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32)
    return matrix, tuple(tensor.to(accelerator_device()) for tensor in (values, indices, offsets))


def _operation(tensor, operation):
    return tensor if operation == "non" else tensor.T if operation == "trans" else tensor.conj().T


@pytest.mark.parametrize("fmt", ["coo", "csr", "csc"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.complex64, torch.int8])
@pytest.mark.parametrize("op_a", ["non", "trans", "conj"])
@pytest.mark.parametrize("op_b", ["non", "trans", "conj"])
def test_native_cann_rectangular_empty_rows_and_transposes(fmt, dtype, op_a, op_b):
    matrix, arrays = _operands(fmt, dtype)
    rows, columns = _operation(matrix, op_a).shape
    rhs = torch.arange(columns * 2).reshape(columns, 2).remainder(5).sub(2).to(dtype)
    if dtype.is_complex:
        rhs += 0.25j
    stored_rhs = _operation(rhs, op_b).resolve_conj().contiguous()
    output_dtype = torch.int32 if dtype == torch.int8 else dtype
    target = torch.empty((rows, 2), device=accelerator_device(), dtype=output_dtype)
    wide = torch.complex128 if dtype.is_complex else torch.int64 if dtype == torch.int8 else torch.float64
    golden = _operation(matrix.to(wide), op_a) @ rhs.to(wide)
    plan = cann.prepare_spmm(fmt, arrays, matrix.shape, stored_rhs.to(target.device), target, op_a, op_b)
    try:
        for repeat in range(2):
            actual = plan.run()
            assert actual.data_ptr() == target.data_ptr()
            if dtype == torch.int8:
                assert torch.equal(actual.cpu().to(wide), golden)
            else:
                torch.testing.assert_close(actual.cpu().to(wide), golden, rtol=2e-3, atol=2e-3)
        assert plan.prepare_ms > 0
        assert plan.i8_promoted_to_i32 == (dtype == torch.int8)
    finally:
        plan.close()
        plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.run()


@pytest.mark.parametrize("dtype,out_dtype", [(torch.float16, torch.float16), (torch.float16, torch.float32),
                                           (torch.float32, torch.float32), (torch.complex64, torch.complex64),
                                           (torch.int8, torch.int32)])
def test_native_cann_spmv_reuses_output_and_preserves_empty_row(dtype, out_dtype):
    matrix, arrays = _operands("csr", dtype)
    vector = torch.tensor([1, 2, -1, 0], dtype=dtype, device=accelerator_device())
    target = torch.empty(3, dtype=out_dtype, device=vector.device)
    plan = cann.prepare_spmv("csr", arrays, matrix.shape, vector, target)
    try:
        plan.run()
        wide = torch.complex128 if dtype.is_complex else torch.float64
        golden = matrix.to(wide) @ vector.cpu().to(wide)
        torch.testing.assert_close(target.cpu().to(wide), golden, rtol=2e-3, atol=2e-3)
        assert target.cpu()[0] == 0
    finally:
        plan.close()


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_cann_spmm_prepares_column_major_rhs(fmt, dtype):
    matrix, arrays = _operands(fmt, dtype)
    rhs = torch.arange(12, dtype=dtype).reshape(3, 4).T
    assert not rhs.is_contiguous()
    target = torch.empty((3, 3), device=accelerator_device(), dtype=dtype)
    plan = cann.prepare_spmm(fmt, arrays, matrix.shape, rhs.to(target.device), target)
    try:
        actual = plan.run().cpu()
        torch.testing.assert_close(actual, matrix @ rhs)
    finally:
        plan.close()
