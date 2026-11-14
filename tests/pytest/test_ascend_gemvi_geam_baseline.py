import pytest
import torch

from tests import ascend_gemvi_geam_baseline as cann
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device
from flagsparse.sparse_operations._common import _is_ascend_runtime


pytestmark = pytest.mark.skipif(
    not (_is_ascend_runtime() and accelerator_available()),
    reason="real Ascend runtime required",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("op", ["non", "trans"])
@pytest.mark.parametrize("empty", [False, True])
def test_native_gemvi(dtype, op, empty):
    device = accelerator_device()
    generator = torch.Generator().manual_seed(73)
    matrix = torch.randn(11, 9, dtype=dtype, generator=generator).t()
    indices = torch.tensor([] if empty else [5, 0, 3], dtype=torch.int32)
    values = torch.randn(indices.numel(), dtype=dtype, generator=generator)
    operand = matrix if op == "non" else matrix.t()
    expected = operand[:, indices.long()] @ values
    target = torch.empty(operand.shape[0], dtype=dtype, device=device)
    plan = cann.prepare_gemvi(
        matrix.to(device), values.to(device), indices.to(device), target, op
    )
    try:
        for _ in range(2):
            torch.testing.assert_close(
                plan.run().flatten().cpu(), expected, atol=1e-5, rtol=1e-5
            )
    finally:
        plan.close()
        plan.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("empty", ["none", "second", "both"])
@pytest.mark.parametrize("pointer_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("scaled", [False, True])
def test_native_geam_csr_structure_and_cancellation(
    dtype, empty, pointer_dtype, scaled
):
    device = accelerator_device()
    values = torch.tensor([2, 3, 4], dtype=dtype)
    if dtype.is_complex:
        values += torch.tensor([1j, 2j, -1j], dtype=dtype)
    columns = torch.tensor([0, 2, 1], dtype=torch.int32)
    pointer = torch.tensor([0, 2, 2, 3], dtype=pointer_dtype)
    other_values = torch.tensor([-2, 5, 6], dtype=dtype)
    if dtype.is_complex:
        other_values += torch.tensor([-1j, 1j, 2j], dtype=dtype)
    other_columns = torch.tensor([0, 1, 3], dtype=torch.int32)
    other_pointer = torch.tensor([0, 1, 2, 3], dtype=pointer_dtype)
    if empty != "none":
        other_values, other_columns, other_pointer = (
            other_values[:0],
            other_columns[:0],
            torch.zeros_like(pointer),
        )
    if empty == "both":
        values, columns, pointer = values[:0], columns[:0], torch.zeros_like(pointer)
    alpha, beta = (0.5, -2) if scaled else (1, 1)
    if scaled and dtype.is_complex:
        alpha, beta = 0.5 - 1j, -2 + 0.25j
    if empty == "none":
        expected_values = torch.stack(
            (
                alpha * values[0] + beta * other_values[0],
                alpha * values[1],
                beta * other_values[1],
                alpha * values[2],
                beta * other_values[2],
            )
        )
        expected_columns = torch.tensor([0, 2, 1, 1, 3], dtype=torch.int32)
        expected_pointer = torch.tensor([0, 2, 3, 5], dtype=pointer_dtype)
    else:
        expected_values, expected_columns, expected_pointer = (
            alpha * values,
            columns,
            pointer,
        )
    first = tuple(tensor.to(device) for tensor in (values, columns, pointer))
    second = tuple(
        tensor.to(device) for tensor in (other_values, other_columns, other_pointer)
    )
    plan = cann.prepare_geam_csr(first, second, (3, 4), alpha, beta)
    try:
        output = plan.run()
        repeated = plan.run()
        for actual in (output, repeated):
            for tensor, expected in zip(
                actual, (expected_values, expected_columns, expected_pointer)
            ):
                assert tensor.dtype == expected.dtype
                torch.testing.assert_close(tensor.cpu(), expected)
    finally:
        plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.run()
