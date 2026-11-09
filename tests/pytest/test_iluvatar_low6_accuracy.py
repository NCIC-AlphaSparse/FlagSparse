import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import _spmv_iluvatar_ell as ell
from flagsparse.sparse_operations._common import _is_iluvatar_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device


pytestmark = pytest.mark.skipif(
    not _is_iluvatar_runtime() or not accelerator_available(),
    reason="requires an Iluvatar accelerator",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_ell_live_values_strides_empty_rows(dtype, conjugate, inference):
    device = accelerator_device()
    with torch.inference_mode(inference):
        lengths = torch.full((65,), 16, dtype=torch.int32, device=device)
        lengths[0] = 0
        lengths[-1] = 33
        ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).int()))
        count = int(ptr[-1])
        cols = torch.arange(count, device=device, dtype=torch.int32) % 79
        values = (torch.arange(count, device=device) % 7 - 3).to(dtype)
        if dtype.is_complex:
            values.add_(2j)
        vector = torch.ones(158, dtype=torch.complex64, device=device)[::2]
        target = torch.empty(130, dtype=torch.complex64, device=device)[::2]
        cache = {}
        for step in range(3):
            active = ell.run(values, cols, ptr, vector, target, 65, cache, conjugate)
            assert active is not inference
            if active:
                rows = torch.repeat_interleave(torch.arange(65), lengths.cpu().long())
                data = values.cpu().conj() if conjugate and dtype.is_complex else values.cpu()
                reference = torch.zeros(65, dtype=torch.complex128)
                reference.index_add_(0, rows, data.to(torch.complex128) * vector.cpu()[cols.cpu().long()].to(torch.complex128))
                torch.testing.assert_close(target.cpu().to(torch.complex128), reference, rtol=1e-5, atol=1e-5)
            values.add_(1)
            vector.add_(1 + 1j)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inference", [False, True])
def test_csr_int_transpose_exact_and_mutated_topology(index_dtype, inference):
    device = accelerator_device()
    with torch.inference_mode(inference):
        count = 1201
        values = torch.full((count,), 127, dtype=torch.int8, device=device)
        values[-1] = 1
        vector = values.clone()
        cols = torch.zeros(count, dtype=index_dtype, device=device)
        ptr = torch.arange(count + 1, dtype=index_dtype, device=device)
        target = torch.empty(4, dtype=torch.int32, device=device)[::2]
        for step in range(3):
            reference = torch.zeros(2, dtype=torch.int64)
            reference.index_add_(0, cols.cpu().long(), values.cpu().long() * vector.cpu().long())
            output = fs.flagsparse_spmv_csr(values, cols, ptr, vector, (count, 2), op="trans", out=target)
            assert output is target
            torch.testing.assert_close(output.cpu(), reference.int(), rtol=0, atol=0)
            values[-1] = 3
            vector[-1] = 5
            if step == 1:
                cols[0] = 1


@pytest.mark.parametrize("fmt", ["csr", "coo", "csc", "sell"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_complex_ell_public_mutations(fmt, dtype):
    device = accelerator_device()
    rows = torch.arange(65).repeat_interleave(16)
    cols = torch.arange(rows.numel()) % 79
    values = (torch.arange(rows.numel()) % 7 - 3).to(dtype).to(device)
    if dtype.is_complex:
        values.add_(1j)
    vector = torch.ones(79, dtype=torch.complex64, device=device)
    ptr = torch.arange(0, values.numel() + 1, 16, dtype=torch.int32, device=device)
    indices = cols.to(device=device, dtype=torch.int32)
    if fmt == "csr":
        prepared = fs.prepare_spmv_csr(values, indices, ptr, (65, 79)) if dtype.is_complex else None
        def call():
            return fs.flagsparse_spmv_csr(prepared=prepared, x=vector) if prepared else fs.flagsparse_spmv_csr(values, indices, ptr, vector, (65, 79))
    elif fmt == "coo":
        prepared = fs.prepare_spmv_coo(values, rows.to(device=device, dtype=torch.int32), indices, (65, 79))
        values = prepared.data_non
        def call():
            return fs.flagsparse_spmv_coo(prepared=prepared, x=vector)
    elif fmt == "csc":
        prepared = fs.prepare_spmv_csc(values, indices, ptr, (79, 65), op="conj") if dtype.is_complex else None
        def call():
            return fs.flagsparse_spmv_csc(prepared=prepared, x=vector) if prepared else fs.flagsparse_spmv_csc(values, indices, ptr, vector, (79, 65), op="trans")
    else:
        if not dtype.is_complex:
            pytest.skip("real SELL has a separate existing path")
        encoded, sell_cols, offsets = fs.csr_to_sell(values, indices, ptr, 65, 32)
        def call():
            return fs.flagsparse_spmv_sell(encoded, sell_cols, offsets, vector, (65, 79), slice_size=32, validate=False)
    for step in range(2):
        reference = torch.zeros(65, dtype=torch.complex128)
        data = values.cpu().to(torch.complex128)
        if fmt == "csc" and dtype.is_complex:
            data = data.conj()
        reference.index_add_(0, rows, data * vector.cpu()[cols].to(torch.complex128))
        torch.testing.assert_close(call().cpu().to(torch.complex128), reference, rtol=1e-5, atol=1e-5)
        values.mul_(2)
        vector.add_(1 + 1j)
        if fmt == "sell":
            encoded.mul_(2)
