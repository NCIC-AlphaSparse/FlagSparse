# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MUSA row-owned routes must read live values and invalidate changed topology."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations._common import _is_mthreads_runtime
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device

pytestmark = pytest.mark.skipif(
    not accelerator_available() or not _is_mthreads_runtime(), reason="requires MUSA"
)


def test_musa_coo_transpose_repeated_row_ownership():
    gen = torch.Generator().manual_seed(123)
    rows = torch.randint(0, 257, (4930,), generator=gen).int()
    cols = torch.randint(0, 129, (4930,), generator=gen).int()
    values = torch.randn(4930, generator=gen)
    rhs = torch.randn(257, generator=gen)
    expected = torch.zeros(129, dtype=torch.float64)
    expected.index_add_(0, cols.long(), values.double() * rhs.double()[rows.long()])
    dev = accelerator_device()
    for _ in range(30):
        actual = fs.flagsparse_spmv_coo(
            values.to(dev), rows.to(dev), cols.to(dev), rhs.to(dev),
            (257, 129), op="trans",
        )
        torch.testing.assert_close(actual.cpu().double(), expected, rtol=1e-5, atol=1e-5)


def test_musa_nnz_integer_cross_block_large_sum():
    dev = accelerator_device()
    length, n_rows = 1101, 512
    ptr = torch.full((n_rows + 1,), length, dtype=torch.int32, device=dev)
    ptr[0] = 0
    cols = torch.zeros(length, dtype=torch.int32, device=dev)
    values = torch.full((length,), 127, dtype=torch.int8, device=dev)
    rhs = torch.tensor([127], dtype=torch.int8, device=dev)
    actual = fs.flagsparse_spmv_csr(values, cols, ptr, rhs, (n_rows, 1))
    expected = torch.zeros(n_rows, dtype=torch.int32)
    expected[0] = length * 127 * 127
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_musa_nnz_real_by_complex_live_cross_block(dtype):
    dev = accelerator_device()
    ptr = torch.tensor([0, 601, 700] + [700] * 254, dtype=torch.int32, device=dev)
    cols = (torch.arange(700, device=dev) % 7).int()
    values = (torch.arange(700, device=dev) % 5 - 2).to(dtype)
    rhs = torch.complex(torch.arange(7, device=dev).to(dtype),
                        torch.arange(7, device=dev).to(dtype) - 2)
    for _ in range(3):
        host = torch.sparse_csr_tensor(ptr.cpu().long(), cols.cpu().long(),
                                       values.cpu().to(torch.complex128), (256, 7))
        expected = host @ rhs.cpu().to(torch.complex128)
        actual = fs.flagsparse_spmv_csr(values, cols, ptr, rhs, (256, 7))
        torch.testing.assert_close(actual.cpu().to(torch.complex128), expected,
                                   atol=1e-5, rtol=1e-5)
        values.add_(1)
        rhs.mul_(1j)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(7, 11, 2), (71, 31, 9), (5, 33, 31)])
def test_musa_real_by_complex_row_tiles(dtype, shape):
    m, n, width = shape
    dev = accelerator_device()
    values = torch.arange(m * width, device=dev).to(dtype).div_(13)
    cols = (torch.arange(m * width, device=dev) % n).int()
    ptr = torch.arange(m + 1, dtype=torch.int32, device=dev) * width
    rhs = torch.complex(torch.arange(n, device=dev).to(dtype).div_(11),
                        torch.arange(n, device=dev).to(dtype).div_(7))
    host = torch.sparse_csr_tensor(ptr.cpu().long(), cols.cpu().long(),
                                   values.cpu().to(torch.complex128), (m, n))
    expected = host @ rhs.cpu().to(torch.complex128)
    actual = fs.flagsparse_spmv_csr(values, cols, ptr, rhs, (m, n))
    tolerance = 1e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(actual.cpu().to(torch.complex128), expected,
                               rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("indirect", [False, True])
def test_musa_half_gather_strided_output(indirect):
    from flagsparse.sparse_operations import _spmv_csr_transpose as gather
    dev = accelerator_device()
    values = torch.tensor([1, 2, 3], dtype=torch.float16, device=dev)
    cols = torch.tensor([0, 2, 1], dtype=torch.int32, device=dev)
    ptr = torch.tensor([0, 2, 3, 3], dtype=torch.int32, device=dev)
    order = torch.tensor([2, 1, 0], dtype=torch.int32, device=dev) if indirect else None
    if indirect:
        values = values.flip(0)
    rhs = torch.tensor([2, 3, 4], dtype=torch.float16, device=dev)
    backing = torch.full((6,), -99, dtype=torch.float16, device=dev)
    out = backing[::2]
    plan = gather.make_plan(cols, ptr, order, 3)
    actual = gather.gather(values, rhs, plan, 3, out=out)
    assert actual is out
    torch.testing.assert_close(actual.cpu(), torch.tensor([10, 9, 0], dtype=torch.float16))
    assert torch.equal(backing[1::2].cpu(), torch.full((3,), -99, dtype=torch.float16))


@pytest.mark.parametrize("fmt", ["coo", "csc"])
@pytest.mark.parametrize("matrix_rhs", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
def test_musa_mixed_live_values_and_indices(fmt, matrix_rhs, dtype):
    if fmt == "csc" and dtype == torch.int8:
        pytest.skip("CSC int8 is not supported")
    dev = accelerator_device()
    rows = torch.tensor([2, 0, 2, 0], dtype=torch.int32, device=dev)
    cols = torch.tensor([1, 0, 2, 0], dtype=torch.int32, device=dev)
    values = torch.tensor([2, 1, 3, -1], dtype=dtype, device=dev)
    rhs = torch.ones((3, 7) if matrix_rhs else (3,), dtype=dtype, device=dev)
    out_dtype = torch.int32 if dtype == torch.int8 else torch.float32
    if fmt == "csc":
        rows = torch.tensor([0, 0, 2, 2], dtype=torch.int32, device=dev)
        cols = torch.tensor([0, 2, 3, 4], dtype=torch.int32, device=dev)

    def run():
        kwargs = {"out_dtype": out_dtype}
        if fmt == "csc":
            if matrix_rhs:
                return fs.flagsparse_spmm_csc(values, rows, cols, rhs, (3, 3))
            return fs.flagsparse_spmv_csc(values, rows, cols, rhs, (3, 3), **kwargs)
        op = fs.flagsparse_spmm_coo if matrix_rhs else fs.flagsparse_spmv_coo
        return op(values, rows, cols, rhs, (3, 3), **kwargs)

    def check():
        a = torch.zeros(3, 3, dtype=torch.float64)
        rr, cc, vv = rows.cpu().long(), cols.cpu().long(), values.cpu().double()
        if fmt == "csc":
            cc = torch.repeat_interleave(torch.arange(3), cc[1:] - cc[:-1])
        a.index_put_((rr, cc), vv, accumulate=True)
        expected = a @ rhs.cpu().double()
        torch.testing.assert_close(run().cpu().double(), expected, rtol=0, atol=1e-5)

    check()
    check()
    values.add_(1)
    check()
    rows[0] = 1
    check()
    rhs.mul_(2)
    check()
    if fmt == "csc":
        cols[1] = 1
        check()
        with torch.inference_mode():
            rows, cols, values = rows.clone(), cols.clone(), values.clone()
            check()
            values.add_(1)
            cols[1] = 2
            check()


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_musa_csc_half_spmm_strides_and_out(index_dtype):
    dev = accelerator_device()
    values = torch.tensor([1, 3, 2], dtype=torch.float16, device=dev)
    indices = torch.tensor([0, 2, 0], dtype=index_dtype, device=dev)
    indptr = torch.tensor([0, 2, 2, 3], dtype=index_dtype, device=dev)
    rhs = torch.arange(42, dtype=torch.float16, device=dev).reshape(7, 6).T[::2]
    backing = torch.full((3, 14), -99, dtype=torch.float16, device=dev)
    out = backing[:, ::2]
    expected_a = torch.tensor([[1, 0, 2], [0, 0, 0], [3, 0, 0]], dtype=torch.float32)
    for _ in range(2):
        actual = fs.flagsparse_spmm_csc(values, indices, indptr, rhs, (3, 3), out=out)
        assert actual is out
        expected = (expected_a @ rhs.cpu().float()).half()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert torch.equal(backing[:, 1::2].cpu(), torch.full((3, 7), -99, dtype=torch.float16))
        rhs.add_(1)
    actual, elapsed = fs.flagsparse_spmm_csc(values, indices, indptr, rhs, (3, 3), return_time=True)
    torch.testing.assert_close(actual.cpu(), (expected_a @ rhs.cpu().float()).half())
    assert elapsed >= 0
    with pytest.raises(ValueError, match="transpose conflicts"):
        fs.flagsparse_spmm_csc(values, indices, indptr, rhs, (3, 3), op="non", transpose=True)
    with pytest.raises(ValueError, match="policy"):
        fs.flagsparse_spmm_csc(values, indices, indptr, rhs, (3, 3), index_fallback_policy="invalid")


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_musa_scatter_triton_strides_and_live_inputs(dtype, reset, index_dtype):
    from flagsparse.sparse_operations.gather_scatter import _triton_scatter_impl

    dev = accelerator_device()
    values = torch.tensor([1, 99, 2, 99, 3, 99], dtype=dtype, device=dev)[::2]
    indices = torch.tensor([4, 99, 1, 99, 3, 99], dtype=index_dtype, device=dev)[::2]
    backing = torch.full((12,), -9, dtype=dtype, device=dev)
    out = backing[::2]
    for _ in range(3):
        expected = out.cpu()
        if reset:
            expected.zero_()
        expected.index_copy_(0, indices.cpu().long(), values.cpu())
        actual, meta = _triton_scatter_impl(values, indices, 6, out=out,
                                            reset_output=reset, return_metadata=True)
        assert actual is out
        assert not meta["index_fallback_applied"]
        assert torch.equal(actual.cpu(), expected)
        assert torch.equal(backing[1::2].cpu(), torch.full((6,), -9, dtype=dtype))
        values.add_(1)
        indices[0] = (int(indices[0]) + 1) % 3


@pytest.mark.parametrize("musa,block", [(False, None), (True, None), (True, 256)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
def test_musa_scatter_triton_scope_and_explicit_tuning(monkeypatch, musa, block, dtype):
    from flagsparse.sparse_operations import gather_scatter as scatter

    values = torch.tensor([1, 2], dtype=dtype)
    indices = torch.tensor([2, 0], dtype=torch.int32)
    out = torch.zeros(3, dtype=dtype)
    launches = []

    def launch(dense, sparse, index, nnz, block_size=None):
        launches.append(block_size)
        dense.index_copy_(0, index.long(), sparse)

    monkeypatch.setattr(scatter, "_is_mthreads_runtime", lambda: musa)
    monkeypatch.setattr(scatter, "_musa_zero_vector", lambda dense: dense.zero_())
    monkeypatch.setattr(scatter, "_launch_triton_scatter_kernel", launch)
    result, meta = scatter._triton_scatter_impl(values, indices, 3, out=out,
                                               block_size=block, return_metadata=True)
    assert result is out
    assert launches == [scatter._resolve_scatter_block_size(block, dtype, 2)]
    assert meta["kernel_index_dtype"] == "int32"
    assert not meta["index_fallback_applied"]
    assert torch.equal(result, torch.tensor([2, 0, 1], dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64,
                                    torch.complex64, torch.complex128, torch.int8])
def test_musa_sell_parallel_partial_slice_and_padding(dtype):
    dense = torch.tensor([[1, 0, 2], [0, 0, 0], [-2, 3, 4]], dtype=dtype)
    csr = dense.to_sparse_csr()
    dev = accelerator_device()
    values, cols, ptr = fs.csr_to_sell(
        csr.values().to(dev), csr.col_indices().int().to(dev),
        csr.crow_indices().int().to(dev), 3, 2,
    )
    x = torch.tensor([1, 2, -1], dtype=dtype, device=dev)
    got = fs.flagsparse_spmv_sell(values, cols, ptr, x, (3, 3), slice_size=2)
    torch.testing.assert_close(got.cpu().to(torch.complex128),
                               dense.to(torch.complex128) @ x.cpu().to(torch.complex128))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int8])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_musa_sell_compact_live_inputs_and_strides(dtype, index_dtype):
    from flagsparse.sparse_operations import spmv_sell as sell

    dev = accelerator_device()
    m, n, slice_size = 19, 11, 8
    dense = torch.zeros(m, n, dtype=dtype)
    dense[0] = torch.arange(n).to(dtype)
    dense[9, 3] = 2
    dense[18, 7] = -1
    csr = dense.to_sparse_csr()
    values, cols, offsets = fs.csr_to_sell(
        csr.values().to(dev), csr.col_indices().to(device=dev, dtype=index_dtype),
        csr.crow_indices().to(device=dev, dtype=index_dtype), m, slice_size,
    )
    rhs = (torch.arange(2 * n, device=dev) % 7).to(dtype)[::2]
    out_dtype = torch.int32 if dtype == torch.int8 else dtype
    backing = torch.full((2 * m,), -99, device=dev, dtype=out_dtype)
    out = backing[::2]

    def check():
        host = torch.zeros(m, n, dtype=torch.float64)
        rr = sell._sell_entry_rows(cols, offsets, slice_size).cpu()
        cc = cols.cpu().long()
        valid = (cc >= 0) & (rr < m)
        host.index_put_((rr[valid], cc[valid]), values.cpu().double()[valid], accumulate=True)
        result = fs.flagsparse_spmv_sell(values, cols, offsets, rhs, (m, n),
                                          slice_size=slice_size, out=out)
        assert result is out
        torch.testing.assert_close(result.cpu().double(), host @ rhs.cpu().double(),
                                   atol=0, rtol=0)
        assert torch.equal(backing[1::2].cpu(), torch.full((m,), -99, dtype=out_dtype))

    check()
    assert sell._MUSA_SELL_PLAN_CACHE.get((values, cols, offsets), (m, n, slice_size))["csr"]
    check()
    values.add_(1)
    check()
    cols[0] = 2
    check()
    offsets[1] -= slice_size
    check()
    rhs.add_(1)
    check()
    timed, elapsed = fs.flagsparse_spmv_sell(values, cols, offsets, rhs, (m, n),
                                            slice_size=slice_size, return_time=True)
    torch.testing.assert_close(timed, out)
    assert elapsed >= 0
    with torch.inference_mode():
        values, cols, offsets = values.clone(), cols.clone(), offsets.clone()
        check()
        values.add_(1)
        cols[0] = 1
        check()
    cols = cols.clone()
    cols[0] = n
    with pytest.raises(IndexError, match="out of range"):
        fs.flagsparse_spmv_sell(values, cols, offsets, rhs, (m, n), slice_size=slice_size)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("op", ["trans", "conj"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_musa_csc_transposed_row_gather(dtype, op, index_dtype):
    from flagsparse.sparse_operations import spmm_csc as csc

    dev = accelerator_device()
    values = torch.tensor([1, 3, 2], dtype=dtype, device=dev)
    if dtype.is_complex:
        values.add_(1j)
    indices = torch.tensor([0, 2, 0], dtype=index_dtype, device=dev)
    indptr = torch.tensor([0, 2, 2, 3], dtype=index_dtype, device=dev)
    rhs = torch.arange(42, device=dev).to(dtype).reshape(7, 6).T[::2]
    if dtype.is_complex:
        rhs = (rhs + 2j).conj()
    backing = torch.full((3, 14), -99, dtype=dtype, device=dev)
    out = backing[:, ::2]
    prepared = csc.prepare_spmm_csc_route(values, indices, indptr, (3, 3), op=op)
    assert prepared.musa_trans_gather
    for _ in range(2):
        dense = torch.zeros(3, 3, dtype=torch.complex128)
        dense[0, 0], dense[2, 0], dense[0, 2] = values.cpu().to(torch.complex128)
        matrix = dense.mH if op == "conj" else dense.T
        expected = matrix @ rhs.cpu().to(torch.complex128)
        actual, meta = fs.flagsparse_spmm_csc(prepared=prepared, B=rhs, out=out,
                                              return_meta=True)
        assert actual is out
        torch.testing.assert_close(actual.cpu().to(torch.complex128), expected)
        assert meta["block_n"] == 8 and meta["block_nnz"] == 4
        torch.testing.assert_close(backing[:, 1::2].cpu(), torch.full((3, 7), -99, dtype=dtype))
        values.add_(1)
        rhs = rhs + 1
    explicit = csc.prepare_spmm_csc_route(values, indices, indptr, (3, 3), op=op,
                                          block_nnz=4)
    assert not explicit.musa_trans_gather
    named = csc.prepare_spmm_csc_route(values, indices, indptr, (3, 3), op=op,
                                      alg="spmm_csc_base")
    assert named.musa_trans_gather


def test_musa_sell_cached_validation():
    dev = accelerator_device()
    values = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device=dev)
    cols = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=dev)
    offsets = torch.tensor([0, 4], dtype=torch.int32, device=dev)
    rhs = torch.ones(2, dtype=torch.float32, device=dev)
    kwargs = dict(shape=(2, 2), slice_size=2)
    fs.flagsparse_spmv_sell(values, cols, offsets, rhs, validate=False, **kwargs)
    fs.flagsparse_spmv_sell(values, cols, offsets, rhs, **kwargs)
    with pytest.raises(ValueError, match="x length"):
        fs.flagsparse_spmv_sell(values, cols, offsets, rhs[:1], **kwargs)
    with pytest.raises(TypeError, match="dtype"):
        fs.flagsparse_spmv_sell(values, cols, offsets, rhs.half(), **kwargs)
    cols[0] = 2
    with pytest.raises(IndexError, match="out of range"):
        fs.flagsparse_spmv_sell(values, cols, offsets, rhs, **kwargs)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_musa_sell_complex_cached_live_inputs(index_dtype):
    from flagsparse.sparse_operations import spmv_sell as sell

    dev = accelerator_device()
    m, n, slice_size = 5, 4, 2
    dense = torch.tensor([[1 + 2j, 0, 3 - 1j, 0], [0, 0, 0, 0],
                          [0, 2j, 0, -1], [3, 0, 0, 0], [0, 0, 1j, 0]])
    csr = dense.to_sparse_csr()
    values, cols, offsets = fs.csr_to_sell(
        csr.values().to(dev), csr.col_indices().to(device=dev, dtype=index_dtype),
        csr.crow_indices().to(device=dev, dtype=index_dtype), m, slice_size,
    )
    backing_values = torch.empty(values.numel() * 2, dtype=values.dtype, device=dev)
    backing_values[::2] = values
    values = backing_values[::2]
    rhs = torch.tensor([1 + 1j, 2 - 1j, 3, -2j], device=dev)
    backing_out = torch.full((2 * m,), -99, dtype=torch.complex64, device=dev)
    out = backing_out[::2]

    def check(validate=True):
        rr = sell._sell_entry_rows(cols, offsets, slice_size).cpu()
        cc = cols.cpu().long()
        valid = (rr < m) & (cc >= 0)
        host = torch.zeros(m, n, dtype=torch.complex128)
        host.index_put_((rr[valid], cc[valid]),
                       values.cpu().to(torch.complex128)[valid], accumulate=True)
        actual = fs.flagsparse_spmv_sell(values, cols, offsets, rhs, (m, n),
                                          slice_size=slice_size, out=out, validate=validate)
        assert actual is out
        torch.testing.assert_close(actual.cpu().to(torch.complex128),
                                   host @ rhs.cpu().to(torch.complex128))
        torch.testing.assert_close(backing_out[1::2].cpu(),
                                   torch.full((m,), -99, dtype=torch.complex64))

    check(False)
    check()
    state = sell._MUSA_SELL_PLAN_CACHE.get((values, cols, offsets), (m, n, slice_size))
    assert state is not None and state["validated"] and not state["csr"]
    values.add_(1j)
    check()
    values.data = (values + 2).data
    check()
    cols[0] = 1
    check()
    offsets[1] -= slice_size
    check()
    rhs.mul_(1j)
    check()
    with torch.inference_mode():
        values, cols, offsets = values.clone(), cols.clone(), offsets.clone()
        check()
        values.add_(1)
        check()
    cols = cols.clone()
    cols[0] = n
    with pytest.raises(IndexError, match="out of range"):
        fs.flagsparse_spmv_sell(values, cols, offsets, rhs, (m, n), slice_size=slice_size)


@pytest.mark.parametrize("sorted_rows", [False, True])
def test_musa_coo_cache_live_storage(sorted_rows):
    dev = accelerator_device()
    rows = torch.tensor([0, 0, 2, 2] if sorted_rows else [2, 0, 2, 0],
                        dtype=torch.int32, device=dev)
    cols = torch.tensor([0, 1, 2, 1], dtype=torch.int32, device=dev)
    values = torch.tensor([1, 2, 3, 4], dtype=torch.float16, device=dev)
    rhs = torch.tensor([2, 3, 4], dtype=torch.float16, device=dev)

    def check():
        expected = torch.zeros(3, dtype=torch.float64)
        expected.index_add_(0, rows.cpu().long(),
                            values.cpu().double() * rhs.cpu().double()[cols.cpu().long()])
        result = fs.flagsparse_spmv_coo(values, rows, cols, rhs, (3, 3),
                                       out_dtype=torch.float32)
        torch.testing.assert_close(result.cpu().double(), expected, rtol=0, atol=0)

    check()
    values.add_(1)
    check()
    values.data = (values + 2).data
    check()
    cols.data = ((cols + 1) % 3).data
    check()
    rows[0] = 1
    check()
    with torch.inference_mode():
        values, rows, cols = values.clone(), rows.clone(), cols.clone()
        check()
        rows[0] = 0
        values.add_(1)
        check()


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
def test_musa_nnz_route_mutation_and_strided_output(dtype):
    dev = accelerator_device()
    ptr = torch.tensor([0, 161, 200] + [200] * 62, dtype=torch.int32, device=dev)
    cols = (torch.arange(200, device=dev) % 7).int()
    values = (torch.arange(200, device=dev) % 5 - 2).to(dtype)
    rhs = torch.arange(7, device=dev).to(dtype)
    out_dtype = torch.int32 if dtype == torch.int8 else torch.float32
    out = torch.full((128,), -99, dtype=out_dtype, device=dev)[::2]

    def check():
        host = torch.sparse_csr_tensor(ptr.cpu().long(), cols.cpu().long(),
                                       values.cpu().double(), (64, 7))
        actual = fs.flagsparse_spmv_csr(values, cols, ptr, rhs, (64, 7),
                                         out=out, out_dtype=out_dtype)
        assert actual is out
        torch.testing.assert_close(actual.cpu().double(), host @ rhs.cpu().double(),
                                   atol=1e-5, rtol=0)

    check()
    check()
    values.add_(1)
    rhs.mul_(2)
    check()
    ptr[1] = 140
    cols[0] = 6
    check()
    with torch.inference_mode():
        ptr = ptr.clone()
        check()
        ptr[1] = 150
        check()


def test_musa_half_gather_strides_and_live_input():
    from flagsparse.sparse_operations.gather_scatter import _triton_gather_impl
    dev = accelerator_device()
    dense = torch.arange(16, device=dev, dtype=torch.float16)[::2]
    indices = torch.tensor([7, 1, 1, 0], dtype=torch.int32, device=dev)
    out = torch.empty(8, device=dev, dtype=torch.float16)[::2]
    assert _triton_gather_impl(dense, indices, out) is out
    torch.testing.assert_close(out.cpu(), torch.tensor([14, 2, 2, 0], dtype=torch.float16))
    dense.add_(1)
    _triton_gather_impl(dense, indices, out)
    torch.testing.assert_close(out.cpu(), torch.tensor([15, 3, 3, 1], dtype=torch.float16))
    with pytest.raises(ValueError, match="shape"):
        _triton_gather_impl(dense, indices, out[:1])
    with pytest.raises(TypeError, match="dtype"):
        _triton_gather_impl(dense, indices, out.float())


@pytest.mark.parametrize("musa", [False, True])
def test_musa_scope_output_self_copy(monkeypatch, musa):
    from flagsparse.sparse_operations import mixed_spmx

    class Output:
        copies = 0

        def copy_(self, other):
            assert other is self
            self.copies += 1

    monkeypatch.setattr(mixed_spmx, "_is_mthreads_runtime", lambda: musa)
    out = Output()
    assert mixed_spmx._finish(out, out, None, False) is out
    assert out.copies == (0 if musa else 1)


@pytest.mark.parametrize("musa", [False, True])
@pytest.mark.parametrize("integer", [False, True])
def test_musa_scope_row_gather_configuration(monkeypatch, musa, integer):
    import triton.language as tl
    from flagsparse.sparse_operations import _spmm_row_gather

    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((args, kwargs))
            return launch

    monkeypatch.setattr(_spmm_row_gather, "_is_mthreads_runtime", lambda: musa)
    monkeypatch.setattr(_spmm_row_gather, "_kernel", Kernel())
    dtype = torch.int8 if integer else torch.float16
    values = torch.ones(64, dtype=dtype)
    cols = torch.zeros(64, dtype=torch.int32)
    ptr = torch.tensor([0, 16, 32, 48, 64], dtype=torch.int32)
    rhs = torch.ones((1, 7), dtype=dtype)
    out = torch.empty((4, 7), dtype=torch.int32 if integer else torch.float32)
    assert _spmm_row_gather.compute(values, cols, ptr, rhs, out) is out
    args, kwargs = launches[0]
    assert args[-3:] == ((8, 4, 8) if musa else (4, 8, 8))
    assert kwargs["ACC"] == (tl.int32 if musa and integer else tl.float32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inference", [False, True])
def test_musa_axpby_fused_live_inputs(dtype, index_dtype, inference):
    from flagsparse.sparse_operations import vector_ops

    dev = accelerator_device()
    with torch.inference_mode(inference):
        indices = torch.tensor([3, 99, 0, 99, 6, 99], dtype=index_dtype, device=dev)[::2]
        values = torch.tensor([1, 99, -2, 99, 3, 99], dtype=dtype, device=dev)[::2]
        y = torch.arange(9, dtype=dtype, device=dev)
        for iteration in range(4):
            expected = y.cpu().float().mul_(-0.25)
            expected.index_add_(0, indices.cpu().long(), values.cpu().float() * 1.5)
            actual, elapsed = fs.flagsparse_axpby(values, indices, y, 1.5, -0.25, return_time=True)
            assert actual is y
            assert elapsed >= 0
            torch.testing.assert_close(y.cpu(), expected.to(dtype), rtol=0, atol=0)
            values.add_(0.5)
            indices[0] = (iteration + 1) % 3 + 3
        if not inference:
            indices.data = indices.clone()
            expected = y.cpu().float().mul_(0.5)
            expected.index_add_(0, indices.cpu().long(), values.cpu().float() * -2)
            fs.flagsparse_axpby(values, indices, y, -2, 0.5, validate=False)
            torch.testing.assert_close(y.cpu(), expected.to(dtype), rtol=0, atol=0)
        assert len(vector_ops._MUSA_AXPBY_INDEX_CACHE._entries) <= 8


@pytest.mark.parametrize("size", [0, 19])
def test_musa_axpby_fused_empty_sparse_vector(size):
    dev = accelerator_device()
    y = torch.arange(size, dtype=torch.float16, device=dev)
    expected = y.cpu() * 0.5
    got = fs.flagsparse_axpby(torch.empty(0, dtype=y.dtype, device=dev),
                             torch.empty(0, dtype=torch.int32, device=dev), y, 2, 0.5)
    assert got is y
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("musa", [False, True])
def test_musa_axpby_fused_backend_scope(monkeypatch, musa):
    from flagsparse.sparse_operations import vector_ops

    values = torch.tensor([2], dtype=torch.float16)
    indices = torch.tensor([1], dtype=torch.int32)
    y = torch.ones(3, dtype=torch.float16)
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append("generic")

    monkeypatch.setattr(vector_ops, "_is_mthreads_runtime", lambda: musa)
    monkeypatch.setattr(vector_ops, "_is_rocm_runtime", lambda: False)
    monkeypatch.setattr(vector_ops, "_is_ascend_runtime", lambda: False)
    monkeypatch.setattr(vector_ops, "_is_accel_tensor", lambda _: True)
    monkeypatch.setattr(vector_ops, "_musa_axpby", lambda *args: calls.append("musa"))
    monkeypatch.setattr(vector_ops, "_scale_real_kernel", Kernel())
    monkeypatch.setattr(vector_ops, "_axpy_real_kernel", Kernel())
    fs.flagsparse_axpby(values, indices, y, 1.5, 0.5, validate=False)
    assert calls == (["musa"] if musa else ["generic", "generic"])


@pytest.fixture
def musa_launch_cache(monkeypatch):
    from collections import OrderedDict
    from flagsparse.sparse_operations import _musa_spmv_launch as cache

    monkeypatch.setattr(cache, "_LAUNCHERS", OrderedDict())
    monkeypatch.setattr(cache, "_RAW_STREAM", lambda device: 123)

    class Compiled:
        def __getitem__(self, grid):
            def run(*args, stream=None):
                kernel.calls.append(("compiled", grid, args))
            return run

    class Kernel:
        debug = False
        pre_run_hooks = ()

        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def run(*args, **kwargs):
                assert kwargs == {"num_warps": 4, "num_stages": 1}
                self.calls.append(("jit", grid, args))
                return Compiled()
            return run

    kernel = Kernel()
    return cache, kernel


def test_musa_launch_cache_specializations(musa_launch_cache):
    from types import SimpleNamespace

    cache, kernel = musa_launch_cache

    def tensor(address, dtype=torch.float16, device="musa:0"):
        return SimpleNamespace(dtype=dtype, device=torch.device(device),
                               data_ptr=lambda: address)

    a, b = tensor(256), tensor(512)
    cache.launch(kernel, (2,), (a, b), (17,), (32,))
    cache.launch(kernel, (2, 1, 1), (tensor(768), tensor(1024)), (17,), (32,))
    assert [call[0] for call in kernel.calls] == ["jit", "compiled"]

    for tensors, scalars, constants, grid in [
        ((tensor(258), b), (17,), (32,), (2,)),
        ((tensor(256, torch.float32), b), (17,), (32,), (2,)),
        ((a, tensor(512, device="musa:1")), (17,), (32,), (2,)),
        ((a, a), (17,), (32,), (2,)),
        ((a, b), (18,), (32,), (2,)),
        ((a, b), (17,), (64,), (2,)),
        ((a, b), (17,), (32,), (3,)),
    ]:
        cache.launch(kernel, grid, tensors, scalars, constants)
        assert kernel.calls[-1][0] == "jit"
    kernel.debug = True
    cache.launch(kernel, (2,), (a, b), (17,), (32,))
    assert kernel.calls[-1][0] == "jit"
    assert len(cache._LAUNCHERS) == 9


def test_musa_launch_cache_lru_eviction(musa_launch_cache, monkeypatch):
    cache, kernel = musa_launch_cache
    monkeypatch.setattr(cache, "_MAX_LAUNCHERS", 2)
    tensors = (torch.empty(4),)
    for size in [1, 2, 1, 3, 2]:
        cache.launch(kernel, (size,), tensors, (), ())
        assert len(cache._LAUNCHERS) <= 2
    assert [call[0] for call in kernel.calls] == [
        "jit", "jit", "compiled", "jit", "jit",
    ]


@pytest.mark.parametrize("mode", ["pre_run", "debug", "instrumentation", "stages"])
def test_musa_launch_cache_debug_fallback(musa_launch_cache, monkeypatch, mode):
    from triton import knobs

    cache, kernel = musa_launch_cache
    tensors = (torch.empty(4),)
    cache.launch(kernel, (1,), tensors, (), ())
    if mode == "pre_run":
        monkeypatch.setattr(kernel, "pre_run_hooks", (lambda: None,))
    elif mode == "debug":
        monkeypatch.setattr(knobs.runtime, "debug", True)
    elif mode == "instrumentation":
        monkeypatch.setattr(knobs.compilation, "instrumentation_mode", "test")
    else:
        monkeypatch.setattr(knobs.runtime, "add_stages_inspection_hook", lambda: None)
    cache.launch(kernel, (1,), tensors, (), ())
    assert [call[0] for call in kernel.calls] == ["jit", "jit"]
    assert len(cache._LAUNCHERS) == 1


@pytest.mark.parametrize("fmt", ["coo", "sell"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
def test_musa_cached_launch_live_buffers_and_stream(fmt, dtype):
    from flagsparse.sparse_operations._common import _ACCEL

    dev = accelerator_device()
    dense = torch.tensor([[1, 0, 2], [0, 0, 0], [0, -2, 0]], dtype=dtype)
    rows = torch.tensor([0, 0, 2], dtype=torch.int32, device=dev)
    cols = torch.tensor([0, 2, 1], dtype=torch.int32, device=dev)
    values = torch.tensor([1, 2, -2], dtype=dtype, device=dev)
    sell_values, sell_cols, offsets = (t.to(dev) for t in fs.dense_to_sell(dense, 2))
    outputs = (torch.float16, torch.float32) if dtype == torch.float16 else (
        torch.int32, torch.float32
    )
    original = None

    def run(rhs, out_dtype, out=None):
        if fmt == "coo":
            return fs.flagsparse_spmv_coo(values, rows, cols, rhs, (3, 3),
                                          out=out, out_dtype=out_dtype)
        return fs.flagsparse_spmv_sell(sell_values, sell_cols, offsets, rhs, (3, 3),
                                       slice_size=2, out=out, out_dtype=out_dtype)

    stream = _ACCEL.Stream()
    stream.wait_stream(_ACCEL.current_stream())
    for out_dtype in outputs:
        for offset in (0, 1):
            with _ACCEL.stream(stream):
                rhs = torch.empty(3 + offset, dtype=dtype, device=dev)[offset:]
                rhs.copy_(torch.tensor([2, 3, 4], dtype=dtype, device=dev))
                first = run(rhs, out_dtype)
                assert original is None or first.data_ptr() != original.data_ptr()
                original = first
                rhs.add_(1)
                out = torch.empty(3 + offset, dtype=out_dtype, device=dev)[offset:]
                actual = run(rhs, out_dtype, out)
                assert actual is out
                # Exercise a cached launch on the default stream as well.
                stream.synchronize()
            second = run(rhs, out_dtype)
            _ACCEL.synchronize()
            torch.testing.assert_close(first.cpu().double(),
                                       dense.double() @ torch.tensor([2., 3., 4.]).double(),
                                       rtol=0, atol=0)
            expected = dense.double() @ torch.tensor([3., 4., 5.]).double()
            torch.testing.assert_close(actual.cpu().double(), expected, rtol=0, atol=0)
            torch.testing.assert_close(second.cpu().double(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("fmt", ["csr", "sell"])
@pytest.mark.parametrize("musa", [False, True])
def test_musa_cached_launch_backend_scope(monkeypatch, fmt, musa):
    from flagsparse.sparse_operations import mixed_spmx, spmv_sell

    module = mixed_spmx if fmt == "csr" else spmv_sell
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append("generic")

    for name in ("_is_rocm_runtime", "_is_ascend_runtime", "_is_maca_runtime"):
        monkeypatch.setattr(module, name, lambda: False)
    monkeypatch.setattr(module, "_is_mthreads_runtime", lambda: musa)
    monkeypatch.setattr(module, "_is_accel_tensor", lambda _: True)
    monkeypatch.setattr(module, "_launch_musa_spmv", lambda *args: calls.append("musa"))
    values = torch.ones(2, dtype=torch.float16)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    rhs = torch.ones(2, dtype=torch.float16)
    if fmt == "csr":
        monkeypatch.setattr(module, "_csr_spmv_mixed_kernel", Kernel())
        module.spmv_csr_mixed(values, indices, offsets, rhs, (1, 2),
                              out_dtype=torch.float32)
    else:
        monkeypatch.setattr(module, "_spmv_sell_real_kernel", Kernel())
        module.flagsparse_spmv_sell(values, indices, offsets, rhs, (1, 2), slice_size=1)
    assert calls == (["musa"] if musa else ["generic"])


@pytest.mark.parametrize("dtype,out_dtype", [
    (torch.float16, torch.float16), (torch.float16, torch.float32),
    (torch.int8, torch.int32), (torch.int8, torch.float32),
])
@pytest.mark.parametrize("width", [2, 9, 29])
def test_musa_triton_coo_live_partial_rows(monkeypatch, dtype, out_dtype, width):
    from flagsparse.sparse_operations import mixed_spmx as mixed

    dev = accelerator_device()
    launches = []
    original = mixed._launch_musa_spmv

    def triton_launch(kernel, *args):
        from triton.runtime.jit import JITFunction
        assert isinstance(kernel, JITFunction)
        launches.append(kernel)
        return original(kernel, *args)

    monkeypatch.setattr(mixed, "_launch_musa_spmv", triton_launch)
    m, n = 67, 41
    lengths = torch.tensor([width if r % 7 else 0 for r in range(m)])
    rr = torch.repeat_interleave(torch.arange(m), lengths).int()
    cc = (torch.arange(rr.numel()) * 7 % n).int()
    vv = (torch.arange(rr.numel()) % 5 - 2).to(dtype)
    # Start unsorted, with noncontiguous sparse operands and vector.
    rows = torch.stack((rr.flip(0), rr), 1).to(dev)[:, 0]
    cols = torch.stack((cc.flip(0), cc), 1).to(dev)[:, 0]
    values = torch.stack((vv.flip(0), vv), 1).to(dev)[:, 0]
    rhs = torch.arange(n * 2, dtype=torch.int32).to(dtype).to(dev)[::2]
    backing = torch.full((m * 2,), -99, dtype=out_dtype, device=dev)
    out = backing[::2]

    def check():
        reference = torch.zeros(m, dtype=torch.float64)
        reference.index_add_(0, rows.cpu().long(), values.cpu().double() *
                             rhs.cpu().double()[cols.cpu().long()])
        actual = fs.flagsparse_spmv_coo(values, rows, cols, rhs, (m, n),
                                        out=out, out_dtype=out_dtype)
        assert launches
        assert actual is out
        torch.testing.assert_close(actual.cpu(), reference.to(out_dtype), rtol=0, atol=0)
        assert torch.equal(backing[1::2].cpu(), torch.full((m,), -99, dtype=out_dtype))

    check()
    values.add_(1)
    rhs.add_(1)
    check()
    values.data = (values + 1).data
    cols.data = ((cols + 1) % n).data
    check()
    rows[0] = 0
    check()
    with torch.inference_mode():
        values, rows, cols = values.clone(), rows.clone(), cols.clone()
        check()
        values.sub_(1)
        cols[0] = 3
        check()


@pytest.mark.parametrize("kind", ["fixed", "nnz", "short"])
@pytest.mark.parametrize("strided_out", [False, True])
def test_musa_half_coo_bound_launch_live_storage_and_stream(kind, strided_out):
    from flagsparse.sparse_operations._common import _ACCEL
    from flagsparse.sparse_operations import mixed_spmx as mixed

    dev = accelerator_device()
    m, n, nnz = (67, 19, 493) if kind == "fixed" else (1025, 19, 1101)
    rows = (torch.arange(nnz) % m).int().to(dev)
    if kind == "nnz":
        rows.zero_()
    cols = (torch.arange(nnz) % n).int().to(dev)
    values = (torch.arange(nnz) % 5 - 2).half().to(dev)
    backing = torch.full((m * 2 + 2,), -99, dtype=torch.float32, device=dev)
    out = backing[1:2 * m:2] if strided_out else backing[1:m + 1]
    stream = _ACCEL.Stream()
    stream.wait_stream(_ACCEL.current_stream())

    def check(rhs, target=None):
        reference = torch.zeros(m, dtype=torch.float64)
        reference.index_add_(0, rows.cpu().long(), values.cpu().double() *
                             rhs.cpu().double()[cols.cpu().long()])
        result = fs.flagsparse_spmv_coo(values, rows, cols, rhs, (m, n),
                                        out=target, out_dtype=torch.float32)
        _ACCEL.synchronize()
        if target is not None:
            assert result is target
        torch.testing.assert_close(result.cpu().double(), reference, rtol=0, atol=0)
        return result

    for offset in (0, 1):
        with _ACCEL.stream(stream):
            rhs = torch.arange(n + offset, device=dev).half()[offset:]
            first = check(rhs)
            preserved = first.clone()
            rhs.add_(1)
            check(rhs, out)
            torch.testing.assert_close(first, preserved, rtol=0, atol=0)
            stream.synchronize()
        check(rhs)
        # Unsupported operand types must still fail after warming the cache.
        with pytest.raises(TypeError, match="not a supported combination"):
            fs.flagsparse_spmv_coo(values, rows, cols, rhs.float(), (m, n),
                                   out_dtype=torch.float32)
        check(rhs)
        values.add_(1)
        rows[0] = m - 1
        check(rhs)
        # Replacing storage does not increment the Tensor's version counter.
        values.data = (values - 1).data
        cols.data = ((cols + 1) % n).data
        check(rhs, out)
    state = mixed._MUSA_COO_CSR_CACHE.get((values, rows, cols), (m, n))
    assert state is not None and state[3]["half_f32"][-1]
    assert torch.equal(backing[:1].cpu(), torch.tensor([-99.]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8, torch.complex64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_musa_indexed_copies_never_call_native_indexing(monkeypatch, dtype, index_dtype):
    from flagsparse.sparse_operations import gather_scatter as copies

    dev = accelerator_device()
    dense = torch.arange(18, dtype=torch.int32).to(dtype).to(dev)[::2]
    indices = torch.tensor([7, 2, 0], dtype=index_dtype, device=dev)
    sparse = torch.empty(6, dtype=dtype, device=dev)[::2]
    target = torch.full((18,), -9, dtype=dtype, device=dev)[::2]

    def forbidden(*args, **kwargs):
        raise AssertionError("MUSA indexed copies must execute Triton kernels")

    monkeypatch.setattr(torch, "index_select", forbidden)
    monkeypatch.setattr(torch, "gather", forbidden)
    monkeypatch.setattr(torch.Tensor, "index_select", forbidden)
    monkeypatch.setattr(torch.Tensor, "index_copy_", forbidden)
    copies._triton_gather_impl(dense, indices, out=sparse)
    copies._triton_scatter_impl(sparse, indices, 9, out=target, reset_output=True)
    assert torch.equal(sparse.cpu(), torch.tensor([14, 4, 0], dtype=dtype))
    assert torch.equal(target.cpu(), torch.tensor([0, 0, 4, 0, 0, 0, 0, 14, 0], dtype=dtype))
    assert fs.flagsparse_gather(dense, indices, out=sparse) is sparse
    fs.flagsparse_scatter(target, indices, sparse, reset_output=True)
    assert torch.equal(sparse.cpu(), torch.tensor([14, 4, 0], dtype=dtype))
    assert torch.equal(target.cpu(), torch.tensor([0, 0, 4, 0, 0, 0, 0, 14, 0], dtype=dtype))
