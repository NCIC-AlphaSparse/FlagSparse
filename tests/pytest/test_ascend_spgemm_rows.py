import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import _spgemm_ascend, spgemm_csr
from flagsparse.sparse_operations._common import _is_ascend_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device


@pytest.fixture
def device():
    if not _is_ascend_runtime() or not accelerator_available():
        pytest.skip("requires an Ascend NPU")
    return accelerator_device()


def csr_operand(values, columns, pointer, shape, dtype, device):
    real = torch.tensor(values, dtype=torch.float32)
    if dtype.is_complex:
        real = torch.complex(real, real.flip(0) * 0.25)
    data = real.to(dtype)
    columns = torch.tensor(columns, dtype=torch.int32)
    pointer = torch.tensor(pointer, dtype=torch.int64)
    dense = torch.sparse_csr_tensor(pointer, columns.long(), data, shape).to_dense()
    return (data.to(device), columns.to(device), pointer.to(device), shape), dense


@pytest.mark.parametrize("dtype", ["f32", "c32"])
def test_h800_comparison_uses_full_call_for_target(dtype):
    from tools.ascend_spgemm_opt_bench import h800_comparison

    result = h800_comparison("GL7d14", dtype, full_ms=1000, prepared_ms=1)
    assert result["h800_reference_available"]
    assert result["h800_scaled_ms"] == pytest.approx(result["h800_vendor_ms"] * 3050 / 1600)
    assert result["full_h800_scaled_speedup"] == pytest.approx(result["h800_scaled_ms"] / 1000)
    assert result["prepared_h800_scaled_speedup"] > 0.8
    assert not result["full_target_met"]
    boundary = h800_comparison("GL7d14", dtype, result["target_full_ms"], prepared_ms=1)
    assert boundary["full_h800_scaled_speedup"] == pytest.approx(0.8)


def test_h800_comparison_does_not_substitute_unmatched_matrix():
    from tools.ascend_spgemm_opt_bench import h800_comparison

    result = h800_comparison("unmatched_spgemm_matrix", "f32", full_ms=1, prepared_ms=1)
    assert not result["h800_reference_available"]
    assert "full_target_met" not in result


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("pointer_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("native", [False, True])
def test_row_local_spgemm_duplicates_empty_rows_and_dynamic_values(device, monkeypatch, dtype, pointer_dtype, native):
    monkeypatch.setattr(_spgemm_ascend, "NATIVE_NUMERIC_THRESHOLD", 0 if native else 4096)
    first, first_dense = csr_operand(
        [2, -1, 3, 0, 4, -2], [2, 0, 2, 1, 3, 0], [0, 3, 3, 4, 6],
        (4, 5), dtype, device,
    )
    second, second_dense = csr_operand(
        [1, 2, -3, 4, 5, -2, 1], [3, 1, 3, 0, 2, 2, 1], [0, 3, 3, 5, 7, 7],
        (5, 6), dtype, device,
    )
    first = first[:2] + (first[2].to(pointer_dtype), first[3])
    second = second[:2] + (second[2].to(pointer_dtype), second[3])
    plan = fs.prepare_spgemm_csr(*first, *second)
    assert plan.ascend_plan is not None
    assert (plan.ascend_plan["product_destination"] is not None) == native

    def no_global_expansion(*args, **kwargs):
        pytest.fail("prepared SpGEMM must not redo symbolic analysis")

    monkeypatch.setattr(torch, "repeat_interleave", no_global_expansion)
    monkeypatch.setattr(torch, "argsort", no_global_expansion)
    monkeypatch.setattr(torch, "sort", no_global_expansion)
    monkeypatch.setattr(_spgemm_ascend, "prepare_symbolic", no_global_expansion)
    monkeypatch.setattr(spgemm_csr, "_spgemm_esc_compute_safe", no_global_expansion)
    previous = None
    for iteration in range(2):
        output = fs.flagsparse_spgemm_csr(prepared=plan)
        assert output[1].dtype == torch.int32
        assert output[2].dtype == torch.int64
        dense = torch.sparse_csr_tensor(output[2].cpu(), output[1].long().cpu(), output[0].cpu(), output[3]).to_dense()
        wide = torch.complex128 if dtype.is_complex else torch.float64
        expected = first_dense.to(wide) @ second_dense.to(wide)
        torch.testing.assert_close(dense.to(wide), expected, rtol=1e-5, atol=1e-5)
        for row in range(output[3][0]):
            columns = output[1].cpu()[output[2][row].item():output[2][row + 1].item()]
            assert torch.all(columns[1:] > columns[:-1])
        if previous is not None:
            torch.testing.assert_close(previous[0][0].cpu(), previous[1])
        previous = output, output[0].cpu().clone()
        buffers = tuple(torch.empty_like(tensor) for tensor in output[:3])
        returned = fs.flagsparse_spgemm_csr(prepared=plan, out=buffers)
        for actual, target, expected_tensor in zip(returned, buffers, output):
            assert actual is target
            torch.testing.assert_close(actual.cpu(), expected_tensor.cpu())
        first[0].mul_(2)
        first_dense.mul_(2)
        second[0].mul_(-0.5)
        second_dense.mul_(-0.5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("shape", [(0, 4, 3), (5, 0, 3), (4, 3, 0), (4, 3, 5)])
def test_row_local_spgemm_empty(device, dtype, shape):
    rows, inner, columns = shape
    first, _ = csr_operand([], [], [0] * (rows + 1), (rows, inner), dtype, device)
    second, _ = csr_operand([], [], [0] * (inner + 1), (inner, columns), dtype, device)
    output = fs.flagsparse_spgemm_csr(*first, *second)
    assert output[0].numel() == output[1].numel() == 0
    torch.testing.assert_close(output[2].cpu(), torch.zeros(rows + 1, dtype=torch.int64))
    assert output[3] == (rows, columns)


@pytest.mark.parametrize("products", [17, 65, 257, 1025, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_row_local_spgemm_bucket_boundaries(device, products, dtype):
    first, first_dense = csr_operand([2], [0], [0, 1], (1, 1), dtype, device)
    second, second_dense = csr_operand(
        list(range(products)), [offset % 13 for offset in range(products)],
        [0, products], (1, 13), dtype, device,
    )
    output = fs.flagsparse_spgemm_csr(*first, *second)
    dense = torch.sparse_csr_tensor(output[2].cpu(), output[1].long().cpu(), output[0].cpu(), output[3]).to_dense()
    wide = torch.complex128 if dtype.is_complex else torch.float64
    torch.testing.assert_close(dense.to(wide), first_dense.to(wide) @ second_dense.to(wide), atol=1e-3, rtol=1e-5)


def test_row_local_spgemm_scratch_and_row_limits(monkeypatch):
    columns = torch.tensor([0], dtype=torch.int32)
    pointer = torch.tensor([0, 1], dtype=torch.int64)
    second_pointer = torch.tensor([0, 17], dtype=torch.int64)
    monkeypatch.setattr(_spgemm_ascend, "MAX_ROW_PRODUCTS", 16)
    assert _spgemm_ascend.prepare_structure(columns, pointer, second_pointer) is None
    monkeypatch.setattr(_spgemm_ascend, "MAX_ROW_PRODUCTS", 32)
    monkeypatch.setattr(_spgemm_ascend, "MAX_PRODUCTS", 16)
    assert _spgemm_ascend.prepare_structure(columns, pointer, second_pointer) is None


def test_row_local_spgemm_checks_total_before_narrow_prefix(monkeypatch):
    columns = torch.zeros(8, dtype=torch.int32)
    pointer = torch.tensor([0, 8], dtype=torch.int64)
    second_pointer = torch.tensor([0, 2**29], dtype=torch.int64)

    def reject_prefix(*args, **kwargs):
        pytest.fail('overflowing product total must be rejected before the int32 prefix')

    monkeypatch.setattr(torch, 'cumsum', reject_prefix)
    assert _spgemm_ascend.prepare_structure(columns, pointer, second_pointer) is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_row_local_spgemm_maximum_multiplicity(device, dtype):
    first, first_dense = csr_operand([2], [0], [0, 1], (1, 1), dtype, device)
    second, second_dense = csr_operand(
        [1] * 4096, [0] * 4096, [0, 4096], (1, 1), dtype, device,
    )
    output = fs.flagsparse_spgemm_csr(*first, *second)
    torch.testing.assert_close(output[0].cpu(), (first_dense @ second_dense).reshape(-1))


@pytest.mark.parametrize("last", [2**30, 2**31])
def test_row_local_spgemm_rejects_overflowing_source_offsets(last):
    columns = torch.tensor([0], dtype=torch.int32)
    pointer = torch.tensor([0, 1], dtype=torch.int64)
    second_pointer = torch.tensor([0, last], dtype=torch.int64)
    assert _spgemm_ascend.prepare_structure(columns, pointer, second_pointer) is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_row_local_spgemm_prepared_structure_is_not_exposed(device, dtype):
    first, first_dense = csr_operand([2, -1, 3], [0, 1, 0], [0, 2, 3], (2, 2), dtype, device)
    second, second_dense = csr_operand([1, -2, 4], [0, 1, 1], [0, 2, 3], (2, 2), dtype, device)
    prepared = fs.prepare_spgemm_csr(*first, *second)
    output = fs.flagsparse_spgemm_csr(prepared=prepared)
    output[0].fill_(123)
    output[1].zero_()
    output[2].zero_()
    repeated = fs.flagsparse_spgemm_csr(prepared=prepared)
    dense = torch.sparse_csr_tensor(
        repeated[2].cpu(), repeated[1].long().cpu(), repeated[0].cpu(), repeated[3],
    ).to_dense()
    torch.testing.assert_close(dense, first_dense @ second_dense)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_row_local_spgemm_underflow_and_cancellation_validation(device, dtype):
    import numpy as np
    import scipy.sparse
    from tools.ascend_spgemm_opt_bench import structural_reference, validate

    first, _ = csr_operand([1e-30, 1, -1], [0, 1, 2], [0, 3], (1, 3), dtype, device)
    second, _ = csr_operand([1e-20, 1, 1], [0, 1, 1], [0, 1, 2, 3], (3, 2), dtype, device)
    left = scipy.sparse.csr_matrix(tuple(tensor.cpu().numpy() for tensor in first[:3]), shape=first[3])
    right = scipy.sparse.csr_matrix(tuple(tensor.cpu().numpy() for tensor in second[:3]), shape=second[3])
    wide = np.complex128 if dtype.is_complex else np.float64
    golden = left.astype(wide) @ right.astype(wide)
    structure = structural_reference(left, right)
    output = fs.flagsparse_spgemm_csr(*first, *second)
    assert validate(output, golden, structure)["pattern_ok"]
    output[0][1] += 1
    with pytest.raises(AssertionError, match="outside rtol"):
        validate(output, golden, structure)
    missing = output[0][:1], output[1][:1], torch.tensor([0, 1], dtype=torch.int64, device=device), output[3]
    with pytest.raises(AssertionError, match="symbolic reference"):
        validate(missing, golden, structure)
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_row_local_spgemm_falls_back_for_wide_rows(device, monkeypatch, dtype):
    first, first_dense = csr_operand([2], [0], [0, 1], (1, 1), dtype, device)
    second, second_dense = csr_operand([1, -2, 3], [2, 0, 2], [0, 3], (1, 4), dtype, device)
    monkeypatch.setattr(_spgemm_ascend, "MAX_ROW_PRODUCTS", 2)
    plan = fs.prepare_spgemm_csr(*first, *second)
    assert plan.ascend_plan is None
    output = fs.flagsparse_spgemm_csr(prepared=plan)
    dense = torch.sparse_csr_tensor(output[2].cpu(), output[1].long().cpu(), output[0].cpu(), output[3]).to_dense()
    torch.testing.assert_close(dense, first_dense @ second_dense)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_row_local_spgemm_large_exact_columns(device, dtype):
    values = torch.tensor([1.0, 2.0, -0.5], dtype=dtype, device=device)
    if dtype.is_complex:
        values += 0.25j
    columns = torch.tensor([16777213, 16777212, 16777213], dtype=torch.int32, device=device)
    pointer = torch.tensor([0, 3], dtype=torch.int64, device=device)
    first = (torch.tensor([2], dtype=dtype, device=device),
             torch.tensor([0], dtype=torch.int32, device=device),
             torch.tensor([0, 1], dtype=torch.int64, device=device), (1, 1))
    second = values, columns, pointer, (1, 2**24)
    plan = fs.prepare_spgemm_csr(*first, *second)
    assert plan.ascend_plan is not None
    output = fs.flagsparse_spgemm_csr(prepared=plan)
    torch.testing.assert_close(output[1].cpu(), torch.tensor([16777212, 16777213], dtype=torch.int32))
    expected = torch.stack((values[1] * 2, (values[0] + values[2]) * 2))
    torch.testing.assert_close(output[0].cpu(), expected.cpu())
    wider = fs.prepare_spgemm_csr(*first, *second[:3], (1, 2**24 + 1))
    assert wider.ascend_plan is None


def test_row_local_spgemm_shards_large_launches(device, monkeypatch):
    monkeypatch.setattr(_spgemm_ascend, "MAX_PROGRAMS", 7)
    first, first_dense = csr_operand([2] * 33, [0] * 33, list(range(34)), (33, 1), torch.float32, device)
    second, second_dense = csr_operand(
        list(range(65)), [offset % 13 for offset in range(65)], [0, 65],
        (1, 13), torch.float32, device,
    )
    plan = fs.prepare_spgemm_csr(*first, *second)
    assert len(plan.ascend_plan["groups"]) > 1
    for rows, block, tile_rows in plan.ascend_plan["groups"]:
        programs = (rows.numel() + tile_rows - 1) // tile_rows
        column_tiles = (block + min(block, 256) - 1) // min(block, 256)
        assert programs * column_tiles <= 7
    for outputs, tile_rows, products in plan.ascend_plan["numeric_groups"]:
        programs = (outputs.numel() + tile_rows - 1) // tile_rows
        assert programs * ((products + 255) // 256) <= 7
    output = fs.flagsparse_spgemm_csr(prepared=plan)
    dense = torch.sparse_csr_tensor(output[2].cpu(), output[1].long().cpu(), output[0].cpu(), output[3]).to_dense()
    torch.testing.assert_close(dense, first_dense @ second_dense)


def test_row_local_spgemm_survives_allocator_reuse(device):
    for dtype in (torch.float32, torch.complex64, torch.float32, torch.complex64):
        first, first_dense = csr_operand(
            [2, -1, 3, 0, 4, -2], [2, 0, 2, 1, 3, 0], [0, 3, 3, 4, 6],
            (4, 5), dtype, device,
        )
        second, second_dense = csr_operand(
            list(range(65)), [offset % 13 for offset in range(65)],
            [0, 13, 26, 39, 52, 65], (5, 13), dtype, device,
        )
        output = fs.flagsparse_spgemm_csr(*first, *second)
        unused = torch.empty(1024 * 1024, dtype=dtype, device=device)
        unused.fill_(123)
        del unused
        dense = torch.sparse_csr_tensor(output[2].cpu(), output[1].long().cpu(), output[0].cpu(), output[3]).to_dense()
        torch.testing.assert_close(dense, first_dense @ second_dense)
        del output, first, second
        torch.npu.empty_cache()


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_row_local_spgemm_long_input_with_mostly_empty_rhs_rows(device, dtype):
    first, first_dense = csr_operand([1] * 513, [0] * 512 + [1], [0, 513], (1, 2), dtype, device)
    second, second_dense = csr_operand([2], [0], [0, 0, 1], (2, 1), dtype, device)
    output = fs.flagsparse_spgemm_csr(*first, *second)
    torch.testing.assert_close(output[0].cpu(), (first_dense @ second_dense).reshape(-1))


@pytest.mark.parametrize("rows", [28, 29, 156])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_row_local_spgemm_padded_rows_and_empty_inner_rows(device, rows, dtype):
    first, first_dense = csr_operand(
        [1] * (rows * 31), list(range(31)) * rows, list(range(0, (rows + 1) * 31, 31)),
        (rows, 31), dtype, device,
    )
    second, second_dense = csr_operand(
        list(range(1, 18)), list(range(17)), [0] + [17] * 31,
        (31, 17), dtype, device,
    )
    output = fs.flagsparse_spgemm_csr(*first, *second)
    dense = torch.sparse_csr_tensor(output[2].cpu(), output[1].long().cpu(), output[0].cpu(), output[3]).to_dense()
    torch.testing.assert_close(dense, first_dense @ second_dense)
