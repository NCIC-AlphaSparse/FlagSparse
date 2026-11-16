import pytest
import torch
import flagsparse as fs

from flagsparse.sparse_operations import _spgemm_ascend, _spgemm_ascend_padded, spgemm_csr
from flagsparse.sparse_operations._common import _is_ascend_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device
from tests.pytest.test_ascend_spgemm_rows import csr_operand


@pytest.fixture
def device():
    if not _is_ascend_runtime() or not accelerator_available():
        pytest.skip('requires an Ascend NPU')
    return accelerator_device()


def prepare_candidate(first, second):
    inputs = spgemm_csr._prepare_spgemm_csr_inputs(*first, *second)
    structure = _spgemm_ascend.prepare_structure(inputs[1], inputs[2], inputs[6])
    assert structure is not None
    prepared = spgemm_csr.SpGEMMPrepared(
        *inputs, structure['work'], torch.empty(0, dtype=torch.int32, device=inputs[0].device), True,
    )
    prepared.ascend_plan = structure
    return prepared, _spgemm_ascend_padded.prepare(prepared)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('products', [1, 17, 65, 129])
def test_padded_spgemm_row_boundaries_and_duplicate_columns(device, dtype, products):
    first, first_dense = csr_operand([2], [0], [0, 1], (1, 1), dtype, device)
    second, second_dense = csr_operand(
        list(range(products)), [offset % 13 for offset in range(products)], [0, products], (1, 13), dtype, device,
    )
    prepared, plan = prepare_candidate(first, second)
    assert plan is not None
    values, columns, pointer = _spgemm_ascend_padded.compute(prepared, plan)
    actual = torch.sparse_csr_tensor(pointer.cpu(), columns.cpu().long(), values.cpu(), (1, 13)).to_dense()
    wide = torch.complex128 if dtype.is_complex else torch.float64
    torch.testing.assert_close(actual.to(wide), first_dense.to(wide) @ second_dense.to(wide), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_padded_spgemm_dynamic_values_and_independent_outputs(device, dtype):
    first, first_dense = csr_operand([2, -1, 3, 0, 4, -2], [2, 0, 2, 1, 3, 0],
                                     [0, 3, 3, 4, 6], (4, 5), dtype, device)
    second, second_dense = csr_operand([1, 2, -3, 4, 5, -2, 1], [3, 1, 3, 0, 2, 2, 1],
                                       [0, 3, 3, 5, 7, 7], (5, 6), dtype, device)
    prepared, plan = prepare_candidate(first, second)
    assert plan is not None
    values, columns, pointer = _spgemm_ascend_padded.compute(prepared, plan)
    previous = values.cpu().clone()
    columns.zero_()
    pointer.zero_()
    first[0].mul_(2)
    output = _spgemm_ascend_padded.compute(prepared, plan)
    actual = torch.sparse_csr_tensor(output[2].cpu(), output[1].cpu().long(), output[0].cpu(), (4, 6)).to_dense()
    torch.testing.assert_close(actual, (first_dense * 2) @ second_dense, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(values.cpu(), previous)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('shape', [(0, 4, 3), (5, 0, 3), (4, 3, 0), (4, 3, 5)])
def test_padded_spgemm_empty(device, dtype, shape):
    rows, inner, columns = shape
    first, unused = csr_operand([], [], [0] * (rows + 1), (rows, inner), dtype, device)
    second, unused = csr_operand([], [], [0] * (inner + 1), (inner, columns), dtype, device)
    prepared, plan = prepare_candidate(first, second)
    assert plan is not None
    output = _spgemm_ascend_padded.compute(prepared, plan)
    assert output[0].numel() == output[1].numel() == 0
    torch.testing.assert_close(output[2].cpu(), torch.zeros(rows + 1, dtype=torch.int64))


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_padded_spgemm_rejects_excessive_padding(device, dtype):
    first, unused = csr_operand([1], [0], [0, 1], (1, 1), dtype, device)
    second, unused = csr_operand([1] * 257, [0] * 257, [0, 257], (1, 1), dtype, device)
    prepared, plan = prepare_candidate(first, second)
    assert plan is None


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_padded_spgemm_underflow_and_cancellation(device, dtype):
    import numpy as np
    import scipy.sparse
    from tests.pytest.spgemm_reference import structural_reference, validate

    first, unused = csr_operand([1e-30, 1, -1], [0, 1, 2], [0, 3], (1, 3), dtype, device)
    second, unused = csr_operand([1e-20, 1, 1], [0, 1, 1], [0, 1, 2, 3], (3, 2), dtype, device)
    left = scipy.sparse.csr_matrix(tuple(tensor.cpu().numpy() for tensor in first[:3]), shape=first[3])
    right = scipy.sparse.csr_matrix(tuple(tensor.cpu().numpy() for tensor in second[:3]), shape=second[3])
    wide = np.complex128 if dtype.is_complex else np.float64
    prepared, plan = prepare_candidate(first, second)
    output = (*_spgemm_ascend_padded.compute(prepared, plan), (1, 2))
    assert validate(output, left.astype(wide) @ right.astype(wide), structural_reference(left, right))['pattern_ok']


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_padded_spgemm_public_dispatch(device, monkeypatch, dtype):
    monkeypatch.setattr(_spgemm_ascend, 'MIN_PADDED_PRODUCTS', 0)
    first, first_dense = csr_operand([2, -1, 3], [0, 1, 0], [0, 2, 3], (2, 2), dtype, device)
    second, second_dense = csr_operand([1, -2, 4], [0, 1, 1], [0, 2, 3], (2, 2), dtype, device)
    prepared = fs.prepare_spgemm_csr(*first, *second)
    assert 'padded' in prepared.ascend_plan
    output = fs.flagsparse_spgemm_csr(prepared=prepared)
    actual = torch.sparse_csr_tensor(output[2].cpu(), output[1].cpu().long(), output[0].cpu(), (2, 2)).to_dense()
    torch.testing.assert_close(actual, first_dense @ second_dense, rtol=1e-4, atol=1e-4)
    full = fs.flagsparse_spgemm_csr(*first, *second)
    for actual_tensor, expected_tensor in zip(full[:3], output[:3]):
        torch.testing.assert_close(actual_tensor.cpu(), expected_tensor.cpu())


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_padded_spgemm_persistent_launch_limit(device, monkeypatch, dtype):
    monkeypatch.setattr(_spgemm_ascend, 'MAX_PROGRAMS', 7)
    first, first_dense = csr_operand([2] * 33, [0] * 33, list(range(34)), (33, 1), dtype, device)
    second, second_dense = csr_operand(list(range(65)), [offset % 13 for offset in range(65)],
                                       [0, 65], (1, 13), dtype, device)
    prepared, plan = prepare_candidate(first, second)
    assert plan['max_programs'] == 7
    output = _spgemm_ascend_padded.compute(prepared, plan)
    actual = torch.sparse_csr_tensor(output[2].cpu(), output[1].cpu().long(), output[0].cpu(), (33, 13)).to_dense()
    torch.testing.assert_close(actual, first_dense @ second_dense, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_padded_spgemm_persistent_groups_exceed_old_launch_limit(device, dtype):
    row_count = 32768 if dtype.is_complex else 65536
    first_values = torch.arange(row_count, dtype=torch.float32, device=device).remainder(17).to(dtype)
    first = (
        first_values, torch.zeros(row_count, dtype=torch.int32, device=device),
        torch.arange(row_count + 1, dtype=torch.int64, device=device), (row_count, 1),
    )
    second, unused_dense = csr_operand([2], [0], [0, 1], (1, 1), dtype, device)
    prepared, plan = prepare_candidate(first, second)
    assert len(prepared.ascend_plan['groups']) == 1
    assert len(plan['groups']) == 1
    assert plan['pointer'].dtype == torch.int64
    for factor in (1, 2):
        if factor == 2:
            first_values.mul_(2)
        values, columns, pointer = _spgemm_ascend_padded.compute(prepared, plan)
        torch.testing.assert_close(values.cpu(), first_values.cpu() * second[0][0].cpu(), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(columns.cpu(), first[1].cpu())
        torch.testing.assert_close(pointer.cpu(), first[2].cpu())


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_padded_spgemm_native_unique_and_inactive_tail_positions(device, monkeypatch, dtype):
    row_count = 129
    degree = 32
    width = 64
    first, first_dense = csr_operand(
        [1] * (row_count * degree), list(range(degree)) * row_count,
        list(range(0, (row_count + 1) * degree, degree)), (row_count, degree), dtype, device,
    )
    second, second_dense = csr_operand(
        [offset % 5 - 2 for offset in range(degree * width)],
        [offset % 13 for offset in range(degree * width)],
        list(range(0, (degree + 1) * width, width)), (degree, 13), dtype, device,
    )
    original = _spgemm_ascend_padded._native_unique
    selected = []

    def recorded(*args):
        selected.append(args[0].numel())
        return original(*args)

    monkeypatch.setattr(_spgemm_ascend_padded, '_native_unique', recorded)
    prepared, plan = prepare_candidate(first, second)
    assert selected and min(selected) >= 262144
    for rows, block, unused_degree, unused_columns, unused_positions, ends in plan['groups']:
        counts = plan['pointer'][rows + 1] - plan['pointer'][rows]
        inactive = torch.arange(block, device=device)[None, :] >= counts[:, None]
        ends.masked_fill_(inactive, 2147483647)
    for factor in (1, 2):
        if factor == 2:
            first[0].mul_(2)
        values, columns, pointer = _spgemm_ascend_padded.compute(prepared, plan)
        actual = torch.sparse_csr_tensor(pointer.cpu(), columns.cpu().long(), values.cpu(), (row_count, 13)).to_dense()
        torch.testing.assert_close(actual, (first_dense * factor) @ second_dense, rtol=1e-4, atol=1e-4)




@pytest.mark.parametrize('degree,width', [(128, 128), (256, 256)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_padded_spgemm_wide_chunked_products_dynamic_values_and_empty_rows(device, degree, width, dtype):
    right_pointer = [0]
    right_values = []
    right_columns = []
    for row in range(degree):
        length = width if row == 0 else 8
        right_columns.extend(offset % 65 for offset in range(length))
        right_values.extend((offset % 7 - 3) * .125 for offset in range(length))
        right_pointer.append(len(right_values))
    first_pointer = [0, 0, 1, 4]
    first_columns = [0, 0, 1, 2]
    for row in range(9):
        first_columns.extend(range(degree))
        first_pointer.append(len(first_columns))
    first_values = [(offset % 11 - 5) * .25 for offset in range(len(first_columns))]
    first, first_dense = csr_operand(
        first_values, first_columns, first_pointer, (len(first_pointer) - 1, degree), dtype, device,
    )
    second, second_dense = csr_operand(
        right_values, right_columns, right_pointer, (degree, 65), dtype, device,
    )
    prepared, plan = prepare_candidate(first, second)
    assert plan is not None
    assert any(group[2] * width > 8192 for group in plan['groups'])
    wide = torch.complex128 if dtype.is_complex else torch.float64
    expected = first_dense.to(wide) @ second_dense.to(wide)
    for factor in (1, 6):
        if factor == 6:
            first[0].mul_(2)
            second[0].mul_(3)
        values, columns, pointer = _spgemm_ascend_padded.compute(prepared, plan)
        assert pointer.dtype == torch.int64
        assert columns.dtype == torch.int32
        assert pointer[0].item() == pointer[1].item() == 0
        actual = torch.sparse_csr_tensor(pointer.cpu(), columns.cpu().long(), values.cpu(), first_dense.shape[:1] + (65,)).to_dense()
        torch.testing.assert_close(actual.to(wide), expected * factor, rtol=1e-4, atol=1e-4)

def test_padded_spgemm_wide_complex_chunk_grid_exceeds_launch_limit(device):
    row_count = 8192
    degree = 65
    first = (
        torch.full((row_count * degree,), 1 + .25j, dtype=torch.complex64, device=device),
        torch.arange(degree, dtype=torch.int32, device=device).repeat(row_count),
        torch.arange(row_count + 1, dtype=torch.int64, device=device) * degree,
        (row_count, degree),
    )
    second, second_dense = csr_operand(
        [2] * (128 + degree - 1), [offset % degree for offset in range(128)] + list(range(1, degree)),
        [0] + list(range(128, 128 + degree)), (degree, degree), torch.complex64, device,
    )
    prepared, plan = prepare_candidate(first, second)
    assert plan is not None
    assert len(plan['groups']) == 1
    assert plan['groups'][0][0].numel() == row_count
    values, columns, pointer = _spgemm_ascend_padded.compute(prepared, plan)
    actual = torch.sparse_csr_tensor(pointer.cpu(), columns.cpu().long(), values.cpu(), (row_count, degree)).to_dense()
    expected = ((1 + .25j) * second_dense.sum(dim=0)).expand(row_count, degree)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize('block', [64, 512, 2048, 4096])
def test_padded_spgemm_bounded_reduction_random_values_and_long_segments(device, block):
    import numpy as np

    generator = np.random.default_rng(31)
    row_count = 9
    keys = np.sort(generator.integers(0, block // 4, (row_count, block)), axis=1).astype(np.float32)
    keys[0] = np.arange(block)
    keys[1] = 0
    values = generator.uniform(-2, 2, (2, row_count, block)).astype(np.float32)
    ends = np.full((row_count, block), 2147483647, dtype=np.int32)
    pointer = [0]
    reference = []
    for row in range(row_count):
        starts = np.flatnonzero(np.concatenate(([True], keys[row, 1:] != keys[row, :-1])))
        stops = np.concatenate((starts[1:] - 1, [block - 1]))
        ends[row, :len(stops)] = stops
        pointer.append(pointer[-1] + len(stops))
        reference.append(np.add.reduceat(values[:, row].astype(np.float64), starts, axis=1))
    expected = torch.from_numpy(np.concatenate(reference, axis=1))
    keys_tensor = torch.from_numpy(keys).to(device)
    values_tensor = torch.from_numpy(values).to(device)
    ends_tensor = torch.from_numpy(ends).to(device)
    rows_tensor = torch.arange(row_count, dtype=torch.int32, device=device)
    pointer_tensor = torch.tensor(pointer, dtype=torch.int64, device=device)
    output = torch.empty((2, pointer[-1]), dtype=torch.float32, device=device)
    for factor in (1, 2):
        if factor == 2:
            values_tensor.mul_(2)
        reduce = _spgemm_ascend_padded._scan_reduce if block > 2048 else _spgemm_ascend_padded._bounded_reduce
        reduce[(3, 2)](
            keys_tensor, values_tensor, ends_tensor, rows_tensor, pointer_tensor, output,
            pointer[-1], row_count, 3, block, enable_fp_fusion=False, multibuffer=False,
        )
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu().double(), expected * factor, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('block', [128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize('row_count', [0, 1, 9])
def test_padded_spgemm_native_compaction_matches_reference(device, monkeypatch, block, row_count):
    monkeypatch.setattr(_spgemm_ascend, 'MAX_PROGRAMS', 3)
    generator = torch.Generator().manual_seed(47)
    lengths = torch.randint(0, block + 1, (row_count,), generator=generator, dtype=torch.int32)
    if row_count:
        lengths[0] = block
    if row_count > 1:
        lengths[1] = 0
    keys = torch.randint(0, block // 4, (row_count, block), generator=generator).float().sort(dim=1).values
    if row_count:
        keys[0] = 0
    if row_count > 2:
        lengths[2] = block
        keys[2] = torch.arange(block) + (2**24 - block)
    lane = torch.arange(block)[None, :]
    keys = torch.where(lane < lengths[:, None], keys, 2147483647.)
    positions = torch.randint(2**24, 2**30, (row_count, block), generator=generator, dtype=torch.int32)
    order = torch.stack([torch.randperm(block, generator=generator) for unused in range(row_count)]) if row_count else torch.empty((0, block), dtype=torch.int64)
    following = torch.cat((keys[:, 1:], torch.full((row_count, 1), -1.)), dim=1)
    tails = (keys != following) & (lane < lengths[:, None])
    expected_ends = torch.where(tails, lane, block - 1).sort(dim=1).values
    expected = (positions.gather(1, order), keys.gather(1, expected_ends), expected_ends.int(), tails.sum(dim=1))
    actual = _spgemm_ascend_padded._native_unique(
        keys.to(device), lengths.to(device), positions.to(device), order.to(device),
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor.cpu(), expected_tensor, rtol=0, atol=0)


@pytest.mark.parametrize('count', [1, 1023, 1024, 1025, 65537, 524281])
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_padded_spgemm_store_counts_unique_rows_and_tail(device, count, dtype):
    import triton

    generator = torch.Generator().manual_seed(31)
    rows = torch.randperm(count + 17, generator=generator)[:count]
    values = torch.arange(count, dtype=dtype).remainder(4097)
    expected = torch.zeros(count + 17, dtype=dtype)
    expected.index_copy_(0, rows, values)
    target = torch.zeros(count + 17, dtype=dtype, device=device)
    _spgemm_ascend_padded._store_counts[(triton.cdiv(count, 1024),)](
        rows.to(device), values.to(device), target, count, 1024, multibuffer=False,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(target.cpu(), expected)
