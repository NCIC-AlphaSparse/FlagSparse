import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import sddmm_csr
from flagsparse.sparse_operations._common import _is_ascend_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device


pytestmark = [
    pytest.mark.sddmm_csr,
    pytest.mark.skipif(
        not _is_ascend_runtime() or not accelerator_available(),
        reason="requires an Ascend NPU",
    ),
]


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("provided_output", [False, True])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (0.731, -0.317), (-0.25, 1.5)])
@pytest.mark.parametrize("transposed", [False, True])
def test_half_sddmm_accumulates_in_float32(
    monkeypatch, prepared, output_dtype, provided_output, alpha, beta, transposed
):
    monkeypatch.setattr(sddmm_csr, "_ASCEND_SDDMM_CHUNK_NNZ", 2)
    generator = torch.Generator().manual_seed(97)
    lhs_cpu = torch.randn(5, 63, generator=generator).half()
    rhs_cpu = torch.randn(7, 63, generator=generator).half()
    data_cpu = torch.randn(6, generator=generator).half()
    columns_cpu = torch.tensor([0, 5, 1, 2, 4, 6], dtype=torch.int32)
    rows_cpu = torch.tensor([0, 0, 2, 3, 3, 4])
    expected = (
        (lhs_cpu.double()[rows_cpu] * rhs_cpu.double()[columns_cpu.long()]).sum(1) * alpha
        + beta * data_cpu.double()
    ).to(output_dtype)
    device = accelerator_device()
    lhs, rhs, data, columns = (
        tensor.to(device) for tensor in (lhs_cpu, rhs_cpu, data_cpu, columns_cpu)
    )
    pointer = torch.tensor([0, 2, 2, 3, 5, 6], dtype=torch.int32, device=device)
    plan = fs.prepare_sddmm_csr(columns, pointer, (5, 7), k_hint=63) if prepared else None
    target = torch.empty(6, dtype=output_dtype, device=device) if provided_output else None
    result = fs.flagsparse_sddmm_csr(
        data, columns, pointer,
        lhs.t() if transposed else lhs,
        rhs if transposed else rhs.t(),
        shape=(5, 7), prepared=plan, alpha=alpha, beta=beta, out=target,
        op_a="trans" if transposed else "non",
        op_b="trans" if transposed else "non", out_dtype=output_dtype,
    )
    if provided_output:
        assert result is target
    assert result.dtype == output_dtype
    torch.testing.assert_close(result.cpu(), expected, atol=2e-6, rtol=2e-6)
