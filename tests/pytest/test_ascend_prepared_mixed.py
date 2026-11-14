from types import SimpleNamespace

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import mixed_spmx, spmv_csc
from flagsparse.sparse_operations._common import _is_ascend_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device


@pytest.fixture(params=["cpu_simulation", "npu"])
def device(request, monkeypatch):
    if request.param == "npu":
        if not _is_ascend_runtime() or not accelerator_available():
            pytest.skip("requires an Ascend NPU")
        return accelerator_device()
    for module in (mixed_spmx, spmv_csc):
        monkeypatch.setattr(module, "_is_ascend_runtime", lambda: True)
        monkeypatch.setattr(module, "_is_accel_tensor", lambda tensor: True)
        monkeypatch.setattr(module, "_ACCEL", SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(mixed_spmx, "_ascend_compile", lambda function: function)
    monkeypatch.setattr(
        spmv_csc, "_get_device_backend_info",
        lambda device=None: {"backend": "ascend", "device_warp_size": 32},
    )
    monkeypatch.setattr(
        mixed_spmx, "_ascend_index_add",
        lambda target, dimension, indices, values: target.index_add_(dimension, indices, values),
    )
    return "cpu"


def operands(shape, dtype, empty):
    matrix = torch.zeros(shape, dtype=dtype)
    if not empty:
        matrix[0, 1] = 3
        matrix[0, 3] = -2
        matrix[2, 0] = 4
        matrix[3, 1] = -1
    sparse = matrix.to_sparse_csr()
    return matrix, sparse.values(), sparse.col_indices().int(), sparse.crow_indices().int()


@pytest.mark.parametrize("shape", [(5, 7), (5, 5)])
@pytest.mark.parametrize("dtype,out_dtype", [
    (torch.float16, torch.float32),
    (torch.float32, torch.complex64),
    (torch.int8, torch.int32),
])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("empty", [False, True])
def test_prepared_mixed_spmv_preserves_operation_and_dynamic_rhs(
    device, monkeypatch, shape, dtype, out_dtype, op, empty
):
    matrix, values, indices, pointer = operands(shape, dtype, empty)
    if dtype == torch.int8:
        matrix, values = matrix * 10, values * 10
    rhs_dtype = torch.complex64 if out_dtype == torch.complex64 else dtype
    effective = matrix if op == "non" else matrix.t()
    rhs = torch.arange(effective.shape[1] * 2).remainder(7).sub(3).to(rhs_dtype)[::2]
    if rhs.is_complex():
        rhs += 0.25j
    values, indices, pointer, rhs = (
        tensor.to(device) for tensor in (values, indices, pointer, rhs)
    )
    plan = mixed_spmx.prepare_spmv_csr_mixed(
        values, indices, pointer, rhs_dtype, shape, op=op, out_dtype=out_dtype
    )
    assert (plan.n_rows, plan.n_cols) == tuple(effective.shape)
    target = torch.empty(effective.shape[0] * 2, dtype=out_dtype, device=device)[::2]

    def no_expansion(*args, **kwargs):
        pytest.fail("prepared execution must not expand structural indices")

    monkeypatch.setattr(mixed_spmx, "_csr_row_ids", no_expansion)
    previous = None
    for _ in range(2):
        result = fs.flagsparse_spmv_csr(prepared=plan, x=rhs)
        wide = torch.complex128 if rhs.is_complex() else torch.float64
        expected = effective.to(wide) @ rhs.detach().cpu().to(wide)
        torch.testing.assert_close(result.cpu().to(wide), expected, atol=1e-5, rtol=1e-5)
        if previous is not None:
            torch.testing.assert_close(previous[0].cpu(), previous[1])
        previous = result, result.cpu().clone()
        returned = fs.flagsparse_spmv_csr(prepared=plan, x=rhs, out=target)
        assert returned is target
        torch.testing.assert_close(returned.cpu(), result.cpu())
        rhs.add_(1)
    with pytest.raises(ValueError, match="shape/device"):
        fs.flagsparse_spmv_csr(prepared=plan, x=torch.zeros(plan.n_cols + 1, dtype=rhs_dtype, device=device))


@pytest.mark.parametrize("shape", [(5, 7), (5, 5)])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("column_major", [False, True])
def test_prepared_half_spmm_preserves_operation_and_dynamic_rhs(
    device, monkeypatch, shape, op, empty, column_major
):
    matrix, values, indices, pointer = operands(shape, torch.float16, empty)
    effective = matrix if op == "non" else matrix.t()
    rhs = torch.arange(effective.shape[1] * 6).reshape(effective.shape[1], 6).remainder(9).sub(4).half()
    if column_major:
        rhs = rhs.t().contiguous().t()
    values, indices, pointer, rhs = (
        tensor.to(device) for tensor in (values, indices, pointer, rhs)
    )
    plan = mixed_spmx.prepare_spmm_csr_mixed(
        values, indices, pointer, shape, op=op, out_dtype=torch.float32
    )
    target = torch.empty(effective.shape[0], 12, dtype=torch.float32, device=device)[:, ::2]

    def no_expansion(*args, **kwargs):
        pytest.fail("prepared execution must not expand structural indices")

    monkeypatch.setattr(mixed_spmx, "_csr_row_ids", no_expansion)
    previous = None
    for _ in range(2):
        expected = effective.double() @ rhs.cpu().double()
        result = mixed_spmx.spmm_csr_mixed_prepared(plan, rhs)
        torch.testing.assert_close(result.cpu().double(), expected)
        if previous is not None:
            torch.testing.assert_close(previous[0].cpu(), previous[1])
        previous = result, result.cpu().clone()
        returned = mixed_spmx.spmm_csr_mixed_prepared(plan, rhs, out=target)
        assert returned is target
        torch.testing.assert_close(returned.cpu(), result.cpu())
        rhs.add_(1)


@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.int32, torch.float32])
def test_prepared_int8_csc_accumulates_exactly_and_refreshes(
    device, monkeypatch, op, empty, out_dtype
):
    matrix, *_ = operands((5, 7), torch.int8, empty)
    matrix *= 10
    sparse = matrix.to_sparse_csc()
    values, indices, pointer = (
        tensor.to(device) for tensor in (
            sparse.values(), sparse.row_indices().int(), sparse.ccol_indices().int()
        )
    )
    effective = matrix if op == "non" else matrix.t()
    rhs = (torch.arange(effective.shape[1], dtype=torch.int8) * 10).to(device)
    plan = fs.prepare_spmv_csc(values, indices, pointer, matrix.shape, op=op, x=rhs)
    target = torch.empty(effective.shape[0] * 2, dtype=out_dtype, device=device)[::2]

    def no_expansion(*args, **kwargs):
        pytest.fail("unchanged CSC structure must reuse prepared indices")

    monkeypatch.setattr(spmv_csc, "_ascend_spmv_csc_indices", no_expansion)
    for step in range(3):
        if step == 1:
            rhs.add_(1)
        if step == 2:
            values.mul_(2)
            effective = effective * 2
        result = fs.flagsparse_spmv_csc(prepared=plan, x=rhs, out_dtype=out_dtype)
        expected = effective.int() @ rhs.cpu().int()
        assert torch.equal(result.cpu(), expected.to(out_dtype))
        returned = fs.flagsparse_spmv_csc(prepared=plan, x=rhs, out=target, out_dtype=out_dtype)
        assert returned is target
        assert torch.equal(returned.cpu(), expected.to(out_dtype))
    if out_dtype == torch.int32:
        assert fs.flagsparse_spmv_csc(prepared=plan, x=rhs).dtype == torch.int32


def test_prepared_transpose_does_not_enable_other_backends(monkeypatch):
    monkeypatch.setattr(mixed_spmx, "_is_ascend_runtime", lambda: False)
    monkeypatch.setattr(mixed_spmx, "_is_accel_tensor", lambda tensor: True)
    _, values, indices, pointer = operands((5, 7), torch.float16, False)
    with pytest.raises(NotImplementedError, match="requires Ascend"):
        mixed_spmx.prepare_spmv_csr_mixed(values, indices, pointer, torch.float16, (5, 7), op="trans")
    with pytest.raises(NotImplementedError, match="requires Ascend"):
        mixed_spmx.prepare_spmm_csr_mixed(values, indices, pointer, (5, 7), op="trans")
