"""Ownership and repeated-call checks for the ROCm NNZ-balanced CSR kernel."""

import pytest
import torch

from flagsparse.sparse_operations import _common
from flagsparse.sparse_operations import _spmv_csr_nnz
from flagsparse.sparse_operations.spmv_csr import prepare_spmv_csr, flagsparse_spmv_csr_run

pytestmark = pytest.mark.skipif(
    not _common._is_rocm_runtime() or not torch.cuda.is_available(),
    reason="ROCm kernel",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("lengths", [
    [],
    [0] * 513,
    [1] + [0] * 700,
    [255, 0, 2, 0, 255, 1, 128, 256, 0, 7],
    [1, 819, 0, 17, 512, 0, 3],
])
def test_nnz_partition_ownership_and_live_inputs(dtype, index_dtype, lengths):
    _common._ACCEL.manual_seed(23)
    counts = torch.tensor(lengths, dtype=torch.int64)
    rp_cpu = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    nnz = int(rp_cpu[-1])
    columns = torch.arange(nnz, dtype=torch.int64) % 37
    values = torch.randn(nnz, dtype=dtype, device="cuda")
    rp = rp_cpu.to(device="cuda", dtype=index_dtype)
    ci = columns.to(device="cuda", dtype=index_dtype)
    prepared = prepare_spmv_csr(values, ci, rp, (len(lengths), 37), alg="row_tile")
    # Exercise even tiny and pathological inputs directly; automatic selection
    # deliberately limits this kernel to measured large-matrix structures.
    prepared.rocm_nnz_plan = _spmv_csr_nnz.prepare(prepared)
    out = torch.empty(len(lengths), dtype=dtype, device="cuda")
    rows = torch.repeat_interleave(torch.arange(len(lengths)), counts)
    for _ in range(2):
        values.mul_(0.75)
        x = torch.randn(37, dtype=dtype, device="cuda")
        out.fill_(float("nan"))
        reference = torch.zeros(len(lengths), dtype=torch.float64)
        reference.index_add_(0, rows, values.cpu().double() * x.cpu().double()[columns])
        result = flagsparse_spmv_csr_run(prepared, x, out=out)
        assert result.data_ptr() == out.data_ptr()
        torch.testing.assert_close(
            result.cpu(), reference.to(dtype),
            rtol=1e-3 if dtype == torch.float32 else 1e-5,
            atol=1.3e-6 if dtype == torch.float32 else 1e-7,
        )
