import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from flagsparse.sparse_operations import geam_csr, gemvi  # noqa: E402


@pytest.fixture
def ascend_cpu(monkeypatch):
    for module in (gemvi, geam_csr):
        monkeypatch.setattr(module, "_is_ascend_runtime", lambda: True)
        monkeypatch.setattr(
            module, "_is_accel_tensor", lambda tensor: tensor.device.type == "cpu"
        )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
@pytest.mark.parametrize("op", ["non", "trans"])
@pytest.mark.parametrize("empty", [False, True])
def test_gemvi_ascend_dtype_and_empty_vector(ascend_cpu, dtype, op, empty):
    generator = torch.Generator().manual_seed(37)
    matrix = torch.randn(13, 11, dtype=dtype, generator=generator).t()
    indices = torch.tensor([] if empty else [1, 3, 6], dtype=torch.int32)
    values = torch.randn(indices.numel(), dtype=dtype, generator=generator)
    output_size = matrix.shape[0 if op == "non" else 1]
    target = torch.randn(output_size * 2, dtype=dtype, generator=generator)[::2]
    alpha, beta = (1.5 - 0.25j, 0.5 + 1j) if dtype.is_complex else (1.5, -0.5)
    operand = matrix if op == "non" else matrix.t()
    expected = alpha * (operand[:, indices.long()] @ values) + beta * target

    result = gemvi.flagsparse_gemvi(
        matrix, values, indices, target, op=op, alpha=alpha, beta=beta
    )

    assert result is target
    assert result.dtype == dtype
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("empty", [False, True])
def test_geam_ascend_complex_scalars_and_pattern(ascend_cpu, dtype, empty):
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(3, dtype=dtype, generator=generator)
    columns = torch.tensor([0, 2, 1], dtype=torch.int32)
    pointer = torch.tensor([0, 2, 2, 3], dtype=torch.int64)
    other_values = values[:0] if empty else values.conj().resolve_conj()
    other_columns = columns[:0] if empty else columns
    other_pointer = torch.zeros_like(pointer) if empty else pointer
    alpha, beta = 0.5 - 2j, 1.25 + 0.5j
    expected = alpha * values + (0 if empty else beta * other_values)

    result, result_columns, result_pointer = geam_csr.flagsparse_geam_csr(
        values,
        columns,
        pointer,
        other_values,
        other_columns,
        other_pointer,
        (3, 4),
        alpha=alpha,
        beta=beta,
    )

    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(result_columns, columns)
    torch.testing.assert_close(result_pointer, pointer)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
@pytest.mark.parametrize("pointer_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("empty", ["none", "first", "second", "both"])
def test_geam_rank_merge_union_and_strided_inputs(
    ascend_cpu, dtype, pointer_dtype, empty
):
    values = torch.tensor([2, 3, 4, -5], dtype=dtype)
    other_values = torch.tensor([-2, 6, 7, 8], dtype=dtype)
    if dtype.is_complex:
        values += torch.tensor([1j, -2j, 3j, 4j], dtype=dtype)
        other_values += torch.tensor([-1j, 5j, -6j, 7j], dtype=dtype)
    columns = torch.tensor([1, 4, 0, 2], dtype=torch.int32)
    other_columns = torch.tensor([1, 3, 0, 3], dtype=torch.int32)
    pointer = torch.tensor([0, 0, 2, 2, 3, 4, 4], dtype=pointer_dtype)
    other_pointer = torch.tensor([0, 0, 2, 3, 3, 3, 4], dtype=torch.int64)
    if empty in ("first", "both"):
        values, columns, pointer = values[:0], columns[:0], torch.zeros_like(pointer)
    if empty in ("second", "both"):
        other_values, other_columns, other_pointer = (
            other_values[:0],
            other_columns[:0],
            torch.zeros_like(other_pointer),
        )
    alpha, beta = (0.5 - 1j, -2 + 0.25j) if dtype.is_complex else (0.5, -2)
    expected_rows = [{} for _ in range(6)]
    for data, indices, indptr, scalar in (
        (values, columns, pointer, alpha),
        (other_values, other_columns, other_pointer, beta),
    ):
        for row in range(6):
            for position in range(int(indptr[row]), int(indptr[row + 1])):
                column = int(indices[position])
                expected_rows[row][column] = (
                    expected_rows[row].get(column, 0) + scalar * data[position].item()
                )
    expected_values, expected_columns, expected_pointer = [], [], [0]
    for row in expected_rows:
        for column, value in sorted(row.items()):
            expected_values.append(value)
            expected_columns.append(column)
        expected_pointer.append(len(expected_columns))

    def strided(tensor):
        storage = torch.empty(tensor.numel() * 2, dtype=tensor.dtype)
        storage[::2] = tensor
        return storage[::2]

    operands = tuple(
        strided(tensor)
        for tensor in (
            values,
            columns,
            pointer,
            other_values,
            other_columns,
            other_pointer,
        )
    )
    originals = tuple(tensor.clone() for tensor in operands)
    result = geam_csr.flagsparse_geam_csr(*operands, (6, 5), alpha=alpha, beta=beta)
    expected = (
        torch.tensor(expected_values, dtype=dtype),
        torch.tensor(expected_columns, dtype=torch.int32),
        torch.tensor(expected_pointer, dtype=pointer_dtype),
    )
    for actual, reference in zip(result, expected):
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, reference)
    for actual, original in zip(operands, originals):
        torch.testing.assert_close(actual, original)


@pytest.mark.parametrize("shape", [(0, 0), (0, 7), (5, 0)])
def test_geam_rank_merge_zero_dimensions(ascend_cpu, shape):
    values = torch.empty(0)
    columns = torch.empty(0, dtype=torch.int32)
    pointer = torch.zeros(shape[0] + 1, dtype=torch.int64)
    result = geam_csr.flagsparse_geam_csr(
        values, columns, pointer, values, columns, pointer, shape
    )
    for actual, expected in zip(result, (values, columns, pointer)):
        torch.testing.assert_close(actual, expected)


def test_geam_rank_merge_recomputes_values_and_pattern(ascend_cpu):
    values = torch.tensor([2.0, 3.0])
    other_values = torch.tensor([-2.0, 4.0])
    columns = torch.tensor([0, 2], dtype=torch.int32)
    other_columns = torch.tensor([0, 3], dtype=torch.int32)
    pointer = torch.tensor([0, 2], dtype=torch.int32)

    def run():
        return geam_csr.flagsparse_geam_csr(
            values, columns, pointer, other_values, other_columns, pointer, (1, 4)
        )

    first = run()
    torch.testing.assert_close(first[0], torch.tensor([0.0, 3.0, 4.0]))
    values.add_(1)
    other_columns[0] = 1
    second = run()
    torch.testing.assert_close(second[0], torch.tensor([3.0, -2.0, 4.0, 4.0]))
    torch.testing.assert_close(second[1], torch.arange(4, dtype=torch.int32))
    torch.testing.assert_close(first[0], torch.tensor([0.0, 3.0, 4.0]))


def test_geam_rank_merge_does_not_sort_expand_rows_or_copy_to_host():
    tree = ast.parse(Path(geam_csr.__file__).read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_torch_geam"
    )
    calls = [
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert not set(calls) & {
        "sort",
        "argsort",
        "unique_consecutive",
        "repeat_interleave",
        "index_copy_",
        "index_add_",
        "cpu",
        "numpy",
        "to_dense",
    }
    assert calls.count("item") == 1
