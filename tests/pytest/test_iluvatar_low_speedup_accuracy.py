import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import mixed_spmx
from flagsparse.sparse_operations._common import _is_iluvatar_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device

pytestmark = pytest.mark.skipif(
    not _is_iluvatar_runtime() or not accelerator_available(),
    reason="requires an Iluvatar accelerator",
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("op", ["trans", "conj"])
@pytest.mark.parametrize("inference", [False, True])
def test_csc_transpose_gather_updates(dtype, op, inference):
    device = accelerator_device()
    with torch.inference_mode(inference):
        values = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device)
        if dtype.is_complex:
            values.add_(1j)
        indices = torch.tensor([0, 2, 1, 2], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 2, 2, 4], dtype=torch.int32, device=device)
        vector = torch.arange(6, dtype=torch.float32, device=device)[::2].to(dtype)
        prepared = fs.prepare_spmv_csc(values, indices, ptr, (3, 3), op=op)
        for _ in range(2):
            dense = torch.zeros(3, 3, dtype=dtype)
            dense[0, 0], dense[2, 0], dense[1, 2], dense[2, 2] = values.cpu()
            reference = (dense.T if op == "trans" else dense.mH) @ vector.cpu()
            output = fs.flagsparse_spmv_csc(prepared=prepared, x=vector)
            torch.testing.assert_close(output.cpu(), reference)
            values.mul_(2)
            vector.add_(1)


@pytest.mark.parametrize("op", ["non", "trans"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_csc_integer_exact_mutation_and_strided_output(op, index_dtype):
    device = accelerator_device()
    count = 1201
    values = torch.full((count,), 127, dtype=torch.int8, device=device)
    values[-1] = 1
    vector = values.clone()
    if op == "non":
        indices = torch.zeros(count, dtype=index_dtype, device=device)
        ptr = torch.arange(count + 1, dtype=index_dtype, device=device)
        shape = (1, count)
    else:
        indices = torch.arange(count, dtype=index_dtype, device=device)
        ptr = torch.tensor([0, count], dtype=index_dtype, device=device)
        shape = (count, 1)
    target = torch.empty(2, dtype=torch.int32, device=device)[::2]
    for _ in range(2):
        reference = (values.cpu().to(torch.int64) * vector.cpu().to(torch.int64)).sum().reshape(1).to(torch.int32)
        output = fs.flagsparse_spmv_csc(values, indices, ptr, vector, shape, op=op, out=target)
        assert output is target
        torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)
        values[-1] = 3
        vector[-1] = 5


def test_csc_integer_topology_cache_invalidation():
    device = accelerator_device()
    values = torch.tensor([2, 3], dtype=torch.int8, device=device)
    indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    vector = torch.tensor([5, 7], dtype=torch.int8, device=device)
    output = fs.flagsparse_spmv_csc(values, indices, ptr, vector, (2, 2))
    torch.testing.assert_close(output.cpu(), torch.tensor([10, 21], dtype=torch.int32))
    indices.copy_(indices.flip(0))
    output = fs.flagsparse_spmv_csc(values, indices, ptr, vector, (2, 2))
    torch.testing.assert_close(output.cpu(), torch.tensor([21, 10], dtype=torch.int32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("width", [0, 4, 65])
def test_spgemm_triton_product_and_value_updates(dtype, width):
    device = accelerator_device()
    inner = 65
    cpu_a = torch.ones(2, inner, dtype=dtype)
    cpu_a[1].zero_()
    cpu_b = torch.ones(inner, width, dtype=dtype)
    if dtype.is_complex:
        cpu_a.mul_(1 + 2j)
        cpu_b.mul_(3 - 1j)
    sparse_a, sparse_b = cpu_a.to_sparse_csr(), cpu_b.to_sparse_csr()
    args_a = (sparse_a.values().to(device), sparse_a.col_indices().to(device=device, dtype=torch.int32),
              sparse_a.crow_indices().to(device), cpu_a.shape)
    args_b = (sparse_b.values().to(device), sparse_b.col_indices().to(device=device, dtype=torch.int32),
              sparse_b.crow_indices().to(device), cpu_b.shape)
    prepared = fs.prepare_spgemm_csr(*args_a, *args_b)
    for scale in (1, 2):
        if scale == 2:
            args_a[0].mul_(2)
        values, indices, ptr, shape = fs.flagsparse_spgemm_csr(prepared=prepared)
        result = torch.sparse_csr_tensor(ptr.cpu(), indices.cpu(), values.cpu(), shape).to_dense()
        torch.testing.assert_close(result, (cpu_a * scale) @ cpu_b)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8, torch.complex64])
@pytest.mark.parametrize("slice_size", [5, 16, 32])
def test_sell_wide_partial_slices(dtype, slice_size):
    device = accelerator_device()
    rows, cols = 37, 13
    cpu_dense = torch.arange(rows * cols).reshape(rows, cols).remainder(7).to(dtype)
    if dtype.is_complex:
        cpu_dense.mul_(1 + 1j)
    cpu_dense[::3].zero_()
    sparse = cpu_dense.to_sparse_csr()
    values = sparse.values().to(device)
    indices = sparse.col_indices().to(device=device, dtype=torch.int32)
    ptr = sparse.crow_indices().to(device=device, dtype=torch.int32)
    sell_values, sell_cols, offsets = fs.csr_to_sell(values, indices, ptr, rows, slice_size)
    vector = torch.ones(cols, dtype=dtype, device=device)
    out_dtype = torch.int32 if dtype == torch.int8 else dtype
    target = torch.empty(rows * 2, dtype=out_dtype, device=device)[::2]
    for validate in (False, True):
        output = fs.flagsparse_spmv_sell(
            sell_values, sell_cols, offsets, vector, (rows, cols),
            slice_size=slice_size, out=target, validate=validate,
        )
        assert output is target
        torch.testing.assert_close(output.cpu(), cpu_dense.sum(1).to(out_dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8, torch.complex64])
@pytest.mark.parametrize("layout", ["col", "padded"])
def test_spmm_strides_and_rhs_updates(dtype, layout):
    device = accelerator_device()
    cpu_dense = torch.tensor([[1, 0, 2], [0, 0, 0], [3, 4, 0]], dtype=dtype)
    if dtype.is_complex:
        cpu_dense.mul_(1 + 1j)
    sparse = cpu_dense.to_sparse_csr()
    values, indices, ptr = (sparse.values().to(device),
                            sparse.col_indices().to(device=device, dtype=torch.int32),
                            sparse.crow_indices().to(device=device, dtype=torch.int32))
    rhs = (torch.ones(32, 3, dtype=dtype, device=device).T if layout == "col"
           else torch.ones(3, 64, dtype=dtype, device=device)[:, ::2])
    out_dtype = torch.float32 if dtype == torch.float16 else torch.int32 if dtype == torch.int8 else dtype
    target = torch.empty(3, 64, dtype=out_dtype, device=device)[:, ::2]
    for _ in range(2):
        output = fs.flagsparse_spmm_csr(values, indices, ptr, rhs, cpu_dense.shape, out=target)
        assert output is target
        reference = cpu_dense.to(out_dtype) @ rhs.cpu().to(out_dtype)
        torch.testing.assert_close(output.cpu(), reference)
        rhs.mul_(2)


@pytest.mark.parametrize(
    "dtype,op",
    [
        (torch.float16, "non"),
        (torch.complex64, "non"),
        (torch.float32, "trans"),
        (torch.complex64, "trans"),
        (torch.complex64, "conj"),
    ],
)
def test_spmm_csr_run_writes_caller_output(dtype, op):
    device = accelerator_device()
    cpu_dense = torch.tensor([[1, 0, 2], [0, 0, 0], [3, 4, 0]], dtype=dtype)
    if dtype.is_complex:
        cpu_dense.mul_(1 + 1j)
    sparse = cpu_dense.to_sparse_csr()
    values, indices, ptr = (sparse.values().to(device),
                            sparse.col_indices().to(device=device, dtype=torch.int32),
                            sparse.crow_indices().to(device=device, dtype=torch.int32))
    rhs = torch.ones(3, 32, dtype=dtype, device=device)
    prepared = fs.prepare_spmm_csr_route(values, indices, ptr, cpu_dense.shape, op=op)
    target = torch.empty(3, 32, dtype=dtype, device=device)
    output = fs.flagsparse_spmm_csr_run(prepared, rhs, op=op, out=target)
    assert output is target
    matrix = {"non": cpu_dense, "trans": cpu_dense.T, "conj": cpu_dense.mH}[op]
    torch.testing.assert_close(target.cpu(), matrix @ rhs.cpu())


def test_mixed_csr_explicit_plain_route_keeps_its_kernel():
    device = accelerator_device()
    n_rows, width = 64, 16
    cols = (torch.arange(n_rows * width, device=device) * 7 % n_rows).to(torch.int32)
    ptr = torch.arange(n_rows + 1, dtype=torch.int32, device=device) * width
    values = torch.arange(1, n_rows * width + 1, dtype=torch.float32, device=device)
    vector = torch.complex(torch.ones(n_rows, device=device), torch.full((n_rows,), 2., device=device))
    reference = torch.zeros(n_rows, dtype=torch.complex128)
    rows = torch.arange(n_rows).repeat_interleave(width)
    reference.index_add_(0, rows, values.cpu().to(torch.complex128) * vector.cpu().to(torch.complex128)[cols.cpu().long()])
    for route, uses_ell in (("plain", False), ("cached", True)):
        prepared = mixed_spmx.prepare_spmv_csr_mixed(
            values, cols, ptr, torch.complex64, (n_rows, n_rows), route=route,
        )
        output = mixed_spmx.spmv_csr_mixed_prepared(prepared, vector)
        assert ("ell_layout" in prepared.iluvatar_ell_plan) == uses_ell
        torch.testing.assert_close(output.cpu().to(torch.complex128), reference, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("op", ["trans", "conj"])
def test_csc_spmm_transpose_cache(op):
    device = accelerator_device()
    values = torch.tensor([1., 2., 3.], device=device)
    indices = torch.tensor([0, 2, 1], dtype=torch.int32, device=device)
    ptr = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    rhs = torch.ones(3, 32, device=device)
    for _ in range(2):
        cpu_dense = torch.zeros(3, 2)
        cpu_dense[0, 0], cpu_dense[2, 0], cpu_dense[1, 1] = values.cpu()
        output = fs.flagsparse_spmm_csc(values, indices, ptr, rhs, (3, 2), op=op)
        torch.testing.assert_close(output.cpu(), cpu_dense.T @ rhs.cpu())
        values.mul_(2)
        indices.copy_(indices.roll(1))
        rhs.mul_(2)


def test_coo_real_complex_unsorted_and_mutation():
    device = accelerator_device()
    values = torch.tensor([2., 3., 4.], device=device)
    rows = torch.tensor([2, 0, 2], dtype=torch.int32, device=device)
    cols = torch.tensor([1, 2, 0], dtype=torch.int32, device=device)
    vector = torch.tensor([1 + 2j, 2 - 1j, 3 + 1j], dtype=torch.complex64, device=device)
    for _ in range(2):
        dense = torch.zeros(3, 3, dtype=torch.complex64)
        dense[rows.cpu().long(), cols.cpu().long()] = values.cpu().to(torch.complex64)
        output = fs.flagsparse_spmv_coo(values, rows, cols, vector, (3, 3))
        torch.testing.assert_close(output.cpu(), dense @ vector.cpu())
        values.mul_(2)
        rows.copy_(rows.roll(1))
        vector.mul_(2)


@pytest.mark.parametrize("op", ["non", "trans"])
@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_csc_integer_empty_dimensions(op, shape):
    device = accelerator_device()
    values = torch.empty(0, dtype=torch.int8, device=device)
    indices = torch.empty(0, dtype=torch.int32, device=device)
    ptr = torch.zeros(shape[1] + 1, dtype=torch.int32, device=device)
    vector = torch.ones(shape[1] if op == "non" else shape[0], dtype=torch.int8, device=device)
    output = fs.flagsparse_spmv_csc(values, indices, ptr, vector, shape, op=op)
    reference = torch.zeros(shape[0] if op == "non" else shape[1], dtype=torch.int32)
    torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)


def test_csc_integer_inference_values_stay_live():
    device = accelerator_device()
    with torch.inference_mode():
        values = torch.tensor([2, 3], dtype=torch.int8, device=device)
        indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
        ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        vector = torch.tensor([5, 7], dtype=torch.int8, device=device)
        for scale in (1, 2):
            if scale == 2:
                values.mul_(2)
            output = fs.flagsparse_spmv_csc(values, indices, ptr, vector, (2, 2))
            reference = torch.tensor([10 * scale, 21 * scale], dtype=torch.int32)
            torch.testing.assert_close(output.cpu(), reference, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["non", "trans"])
def test_csc_integer_timing_and_route_metadata(op):
    device = accelerator_device()
    values = torch.tensor([2, 3], dtype=torch.int8, device=device)
    indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    vector = torch.tensor([5, 7], dtype=torch.int8, device=device)
    output, elapsed, metadata = fs.flagsparse_spmv_csc(
        values, indices, ptr, vector, (2, 2), op=op, return_time=True, return_meta=True,
    )
    torch.testing.assert_close(output.cpu(), torch.tensor([10, 21], dtype=torch.int32), rtol=0, atol=0)
    assert elapsed >= 0
    assert metadata["route"] == "iluvatar_csc_int_gather"
