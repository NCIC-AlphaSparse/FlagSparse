import importlib

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations._common import _is_iluvatar_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device


pytestmark = pytest.mark.skipif(
    not _is_iluvatar_runtime() or not accelerator_available(),
    reason="requires an Iluvatar accelerator",
)


@pytest.mark.parametrize("op", ["trans", "conj"])
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_csc_nnz_cross_block_conjugation(op, inference, dtype):
    device = accelerator_device()
    with torch.inference_mode(inference):
        values = torch.arange(256, dtype=torch.float32, device=device).to(dtype)
        if dtype.is_complex:
            values.add_(2j)
        indices = torch.arange(256, dtype=torch.int32, device=device)
        ptr = torch.full((129,), 256, dtype=torch.int64, device=device)
        ptr[0] = 0
        vector = torch.ones(514, dtype=dtype, device=device)[::2]
        prepared = fs.prepare_spmv_csc(values, indices, ptr, (257, 128), op=op)
        assert prepared.iluvatar_nnz_plan is not None
        for _ in range(2):
            effective = values.cpu().conj() if op == "conj" else values.cpu()
            reference = torch.zeros(128, dtype=dtype)
            reference[0] = (effective * vector.cpu()[:256]).sum()
            output = fs.flagsparse_spmv_csc(prepared=prepared, x=vector)
            torch.testing.assert_close(output.cpu(), reference)
            values.mul_(1 + 1j if dtype.is_complex else 2)
            vector.add_(2 - 1j if dtype.is_complex else 2)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_csc_real_csr_delegate_live_values(index_dtype):
    device = accelerator_device()
    values = torch.arange(1, 25, dtype=torch.float32, device=device)
    indices = torch.arange(8, dtype=index_dtype, device=device).repeat(3)
    ptr = torch.arange(0, 25, 8, dtype=index_dtype, device=device)
    vector = torch.arange(16, dtype=torch.float32, device=device)[::2]
    prepared = fs.prepare_spmv_csc(values, indices, ptr, (8, 3), op="trans")
    assert prepared.csr_delegate is not None
    for _ in range(2):
        reference = (values.cpu().reshape(3, 8) * vector.cpu()).sum(1)
        output = fs.flagsparse_spmv_csc(prepared=prepared, x=vector)
        torch.testing.assert_close(output.cpu(), reference)
        values.add_(3)
        vector.mul_(2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("inference", [False, True])
def test_spgemm_symbolic_topology_values_and_output_ownership(dtype, inference):
    device = accelerator_device()
    with torch.inference_mode(inference):
        left = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device)
        right = torch.tensor([2, 3, 4, 5], dtype=dtype, device=device)
        if dtype.is_complex:
            left.add_(1j)
            right.add_(2j)
        left_cols = torch.tensor([0, 2, 1, 2], dtype=torch.int32, device=device)
        left_ptr = torch.tensor([0, 2, 2, 4], dtype=torch.int64, device=device)
        right_cols = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
        right_ptr = torch.tensor([0, 1, 2, 4], dtype=torch.int64, device=device)
        prepared = fs.prepare_spgemm_csr(
            left, left_cols, left_ptr, (3, 3), right, right_cols, right_ptr, (3, 2),
        )
        for step in range(3):
            left_dense = torch.sparse_csr_tensor(left_ptr.cpu(), left_cols.cpu(), left.cpu(), (3, 3)).to_dense()
            right_dense = torch.sparse_csr_tensor(right_ptr.cpu(), right_cols.cpu(), right.cpu(), (3, 2)).to_dense()
            values, indices, ptr, shape = fs.flagsparse_spgemm_csr(prepared=prepared)
            result = torch.sparse_csr_tensor(ptr.cpu(), indices.cpu(), values.cpu(), shape).to_dense()
            torch.testing.assert_close(result, left_dense @ right_dense)
            indices.zero_()
            ptr.zero_()
            values.zero_()
            left.mul_(2)
            right.add_(1)
            if not inference:
                left_cols.copy_(left_cols.flip(0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_spgemm_symbolic_budget_fallback(monkeypatch, dtype):
    from flagsparse.sparse_operations import _spgemm_iluvatar
    from flagsparse.sparse_operations._rocm_sparse_cache import TensorPlanCache

    monkeypatch.setattr(_spgemm_iluvatar, "_MAX_PRODUCTS", 1)
    monkeypatch.setattr(_spgemm_iluvatar, "_SYMBOLIC_CACHE", TensorPlanCache(maxsize=1))
    device = accelerator_device()
    values = torch.ones(4, dtype=dtype, device=device)
    indices = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
    ptr = torch.tensor([0, 2, 4], dtype=torch.int64, device=device)
    prepared = fs.prepare_spgemm_csr(values, indices, ptr, (2, 2), values, indices, ptr, (2, 2))
    assert _spgemm_iluvatar.compute(prepared) is None
    data, cols, rows, shape = fs.flagsparse_spgemm_csr(prepared=prepared)
    result = torch.sparse_csr_tensor(rows.cpu(), cols.cpu(), data.cpu(), shape).to_dense()
    torch.testing.assert_close(result, torch.full((2, 2), 2, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8, torch.complex64])
@pytest.mark.parametrize("inference", [False, True])
def test_raw_sell_cached_padding_value_and_topology_mutation(dtype, inference):
    device = accelerator_device()
    sell = importlib.import_module("flagsparse.sparse_operations.spmv_sell")
    with torch.inference_mode(inference):
        values = torch.ones(41, dtype=dtype, device=device)
        if dtype.is_complex:
            values.add_(1j)
        cols = torch.arange(41, dtype=torch.int32, device=device)
        ptr = torch.full((66,), 41, dtype=torch.int64, device=device)
        ptr[0] = 0
        encoded, indices, offsets = fs.csr_to_sell(values, cols, ptr, 65, 32)
        vector = torch.ones(41, dtype=dtype, device=device)
        output_dtype = torch.int32 if dtype == torch.int8 else dtype
        target = torch.empty(130, dtype=output_dtype, device=device)[::2]
        for step in range(3):
            rows = sell._sell_entry_rows(indices, offsets, 32)
            valid = (indices >= 0) & (rows < 65)
            reference = torch.zeros(65, dtype=output_dtype)
            reference.index_add_(0, rows[valid].cpu().long(),
                                 encoded[valid].cpu().to(output_dtype) * vector[indices[valid].long()].cpu().to(output_dtype))
            output, elapsed = fs.flagsparse_spmv_sell(
                encoded, indices, offsets, vector, (65, 41), slice_size=32,
                out=target, validate=False, return_time=True,
            )
            assert output is target
            assert elapsed >= 0
            torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
            encoded.mul_(2)
            vector.add_(1)
            if step == 0:
                indices[1] = 0
                encoded[1] = 3


@pytest.mark.parametrize("inference", [False, True])
def test_prepared_coo_real_complex_unsorted_out_and_live_values(inference):
    device = accelerator_device()
    with torch.inference_mode(inference):
        values = torch.tensor([2., 3., 4.], device=device)
        rows = torch.tensor([2, 0, 2], dtype=torch.int32, device=device)
        cols = torch.tensor([1, 2, 0], dtype=torch.int32, device=device)
        vector = torch.tensor([1 + 2j, 2 - 1j, 3 + 1j], dtype=torch.complex64, device=device)
        target = torch.empty(6, dtype=torch.complex64, device=device)[::2]
        prepared = fs.prepare_spmv_coo(values, rows, cols, (3, 3), sort_by_row=False)
        for _ in range(2):
            dense = torch.zeros(3, 3, dtype=torch.complex64)
            dense[rows.cpu().long(), cols.cpu().long()] = values.cpu().to(torch.complex64)
            output, elapsed = fs.flagsparse_spmv_coo(
                prepared=prepared, x=vector, out=target, return_time=True,
            )
            assert output is target
            assert elapsed >= 0
            torch.testing.assert_close(output.cpu(), dense @ vector.cpu())
            values.mul_(2)
            rows.copy_(rows.roll(1))
            vector.mul_(2)


def test_cached_sell_integer_sum_above_float32_precision():
    device = accelerator_device()
    values = torch.full((1401,), 127, dtype=torch.int8, device=device)
    values[-1] = 1
    cols = torch.zeros(1401, dtype=torch.int32, device=device)
    ptr = torch.full((66,), 1401, dtype=torch.int64, device=device)
    ptr[0] = 0
    encoded, indices, offsets = fs.csr_to_sell(values, cols, ptr, 65, 32)
    vector = torch.tensor([127], dtype=torch.int8, device=device)
    target = torch.empty(130, dtype=torch.int32, device=device)[::2]
    for _ in range(2):
        output = fs.flagsparse_spmv_sell(
            encoded, indices, offsets, vector, (65, 1), slice_size=32,
            out=target, validate=False,
        )
        reference = torch.zeros(65, dtype=torch.int32)
        reference[0] = (values.cpu().to(torch.int64) * int(vector.item())).sum().to(torch.int32)
        torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
        vector.sub_(1)


@pytest.mark.parametrize("shape", [(0, 3, 2), (3, 0, 2), (3, 2, 0)])
def test_symbolic_spgemm_empty_dimensions(shape):
    device = accelerator_device()
    rows, inner, cols = shape
    data = torch.empty(0, device=device)
    indices = torch.empty(0, dtype=torch.int32, device=device)
    left_ptr = torch.zeros(rows + 1, dtype=torch.int64, device=device)
    right_ptr = torch.zeros(inner + 1, dtype=torch.int64, device=device)
    prepared = fs.prepare_spgemm_csr(
        data, indices, left_ptr, (rows, inner), data, indices, right_ptr, (inner, cols),
    )
    values, columns, ptr, result_shape = fs.flagsparse_spgemm_csr(prepared=prepared)
    assert result_shape == (rows, cols)
    assert values.numel() == columns.numel() == 0
    torch.testing.assert_close(ptr.cpu(), torch.zeros(rows + 1, dtype=ptr.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_raw_spgemm_int32_ptr_symbolic_reuse_and_invalidation(monkeypatch, dtype):
    from flagsparse.sparse_operations import _spgemm_iluvatar
    from flagsparse.sparse_operations._rocm_sparse_cache import TensorPlanCache

    monkeypatch.setattr(_spgemm_iluvatar, "_SYMBOLIC_CACHE", TensorPlanCache(maxsize=1))
    build = _spgemm_iluvatar._build_symbolic
    calls = []

    def record(prepared):
        calls.append(prepared)
        return build(prepared)

    monkeypatch.setattr(_spgemm_iluvatar, "_build_symbolic", record)
    device = accelerator_device()
    values = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    indices = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    ptr = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    for step in range(3):
        dense = torch.sparse_csr_tensor(ptr.cpu(), indices.cpu(), values.cpu(), (2, 2)).to_dense()
        data, cols, rows, shape = fs.flagsparse_spgemm_csr(
            values, indices, ptr, (2, 2), values, indices, ptr, (2, 2),
        )
        result = torch.sparse_csr_tensor(rows.cpu(), cols.cpu(), data.cpu(), shape).to_dense()
        torch.testing.assert_close(result, dense @ dense)
        assert len(calls) == (2 if step == 2 else 1)
        values.mul_(2)
        if step == 1:
            ptr[1] = 1
