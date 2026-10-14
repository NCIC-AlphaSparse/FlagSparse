# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA auto gathers must read updated and inference tensor values."""

import pytest
import torch

from flagsparse.sparse_operations import spmv_csr as csr
from flagsparse.sparse_operations._common import _is_maca_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device

pytestmark = [pytest.mark.spmv_csr, pytest.mark.skipif(
    not (_is_maca_runtime() and accelerator_available()), reason="MACA required"
)]


@pytest.mark.parametrize("dtype", [torch.float16, torch.complex64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
@pytest.mark.parametrize("inference", [False, True])
def test_maca_gather_live_values(dtype, index_dtype, op, inference):
    dev = accelerator_device()
    gen = torch.Generator().manual_seed(47)
    # Empty rows, an uneven tail, duplicate indices and a long row exercise
    # both masks and the streaming loop in the MACA component-pair kernel.
    ptr = torch.tensor([0, 0, 3, 132, 134], dtype=index_dtype)
    cols = torch.randint(0, 7, (134,), generator=gen).to(index_dtype)
    values = torch.randn(134, generator=gen).to(dtype)
    if dtype.is_complex:
        values += 1j * torch.randn(134, generator=gen)
    size = 7 if op == "non" else 4
    x_cpu = torch.randn(size, generator=gen).to(dtype)
    if dtype.is_complex:
        x_cpu += 1j * torch.randn(size, generator=gen)
    with torch.inference_mode(inference):
        values_d, cols_d, ptr_d = (t.to(dev) for t in (values, cols, ptr))
        x = x_cpu.to(dev)
        prepared = csr.prepare_spmv_csr(values_d, cols_d, ptr_d, (4, 7), op=op)
        out = torch.empty(4 if op == "non" else 7, dtype=dtype, device=dev)
        for scale, x_scale in [(1, 1), (2, 1), (2, 0.5)]:
            if scale == 2 and x_scale == 1:
                values_d.mul_(2)
            elif x_scale == 0.5:
                x.mul_(0.5)
            actual = csr.flagsparse_spmv_csr_run(prepared, x, out=out)
            assert actual is out
            wide = torch.complex128 if dtype.is_complex else torch.float64
            dense = torch.sparse_csr_tensor(ptr.long(), cols.long(),
                                           values.to(wide) * scale, (4, 7)).to_dense()
            if op != "non":
                dense = dense.t() if op == "trans" else dense.conj().t()
            gold = dense @ (x_cpu.to(wide) * x_scale)
            torch.testing.assert_close(actual.cpu().to(wide), gold,
                                       rtol=2e-3 if dtype == torch.float16 else 1e-4,
                                       atol=2e-3 if dtype == torch.float16 else 1e-4)
