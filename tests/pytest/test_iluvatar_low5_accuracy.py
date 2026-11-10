"""Regression coverage for paired complex gathers and prepared mixed COO."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations._common import _is_iluvatar_runtime
from flagsparse.sparse_operations import _spmv_iluvatar_pair as pair
from flagsparse.sparse_operations import _spmv_iluvatar_ell as ell
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device


pytestmark = pytest.mark.skipif(
    not _is_iluvatar_runtime() or not accelerator_available(), reason="Iluvatar-only routes"
)


def reference(values, cols, ptr, vector, conjugate=False):
    lengths = (ptr[1:] - ptr[:-1]).cpu().long()
    rows = torch.repeat_interleave(torch.arange(lengths.numel()), lengths)
    data = values.cpu().to(torch.complex128)
    if conjugate:
        data = data.conj()
    result = torch.zeros(lengths.numel(), dtype=torch.complex128)
    result.index_add_(0, rows, data * vector.cpu().to(torch.complex128)[cols.cpu().long()])
    return result


@pytest.mark.parametrize("width", [2, 6, 40])
@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_pair_live_values_rhs_indices_and_strides(width, conjugate, inference):
    device = accelerator_device()
    with torch.inference_mode(inference):
        lengths = torch.full((67,), width, dtype=torch.int32, device=device)
        lengths[0] = 0
        lengths[-1] = width + 1
        ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).int()))
        count = int(ptr[-1])
        cols = torch.arange(count, device=device, dtype=torch.int32) % 83
        values = (torch.arange(count, device=device) % 7 - 3).to(torch.complex64) + 2j
        vector = torch.ones(166, dtype=torch.complex64, device=device)[::2]
        target = torch.full((134,), complex(float("nan"), 0), dtype=torch.complex64, device=device)[::2]
        cache = dict(avg_row_nnz=count / 67, max_row_nnz=width + 1)
        for step in range(3):
            output = pair.run(values, cols, ptr, vector, 67, cache, conjugate, target)
            if inference:
                assert output is None
            else:
                assert output is target
                torch.testing.assert_close(output.cpu().to(torch.complex128), reference(values, cols, ptr, vector, conjugate), rtol=1e-5, atol=1e-5)
            values.add_(1 + 2j)
            vector.add_(1 - 1j)
            cols[0] = (cols[0] + 1) % 83


@pytest.mark.parametrize("sorted_rows", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inference", [False, True])
def test_prepared_mixed_coo_mutations(sorted_rows, index_dtype, inference):
    device = accelerator_device()
    with torch.inference_mode(inference):
        rows = torch.arange(67, device=device, dtype=index_dtype).repeat_interleave(10)
        cols = torch.arange(rows.numel(), device=device, dtype=index_dtype) % 83
        values = (torch.arange(rows.numel(), device=device) % 7 - 3).float()
        permutation = torch.arange(rows.numel() - 1, -1, -1, device=device)
        plan = fs.prepare_spmv_coo(values[permutation], rows[permutation], cols[permutation], (67, 83), sort_by_row=sorted_rows)
        vector = torch.ones(166, dtype=torch.complex64, device=device)[::2]
        target = torch.empty(134, dtype=torch.complex64, device=device)[::2]
        for step in range(3):
            result = torch.zeros(67, dtype=torch.complex128)
            result.index_add_(0, plan.row_non.cpu().long(), plan.data_non.cpu().double() * vector.cpu().to(torch.complex128)[plan.col_non.cpu().long()])
            output = fs.flagsparse_spmv_coo(prepared=plan, x=vector, out=target)
            assert output is target
            torch.testing.assert_close(output.cpu().to(torch.complex128), result, rtol=1e-5, atol=1e-5)
            plan.data_non.add_(1)
            plan.row_non[0] = 66
            plan.col_non[0] = 82
            vector.add_(1 + 1j)


@pytest.mark.parametrize("width", [2, 6, 40])
@pytest.mark.parametrize("inference", [False, True])
def test_csc_conjugate_public_updates(width, inference):
    device = accelerator_device()
    with torch.inference_mode(inference):
        ptr = torch.arange(68, dtype=torch.int32, device=device) * width
        cols = torch.arange(67 * width, dtype=torch.int32, device=device) % 83
        values = (torch.arange(cols.numel(), device=device) % 7 - 3).to(torch.complex64) + 1j
        vector = torch.ones(166, dtype=torch.complex64, device=device)[::2]
        plan = fs.prepare_spmv_csc(values, cols, ptr, (83, 67), op="conj")
        for step in range(3):
            output = fs.flagsparse_spmv_csc(prepared=plan, x=vector)
            torch.testing.assert_close(output.cpu().to(torch.complex128), reference(values, cols, ptr, vector, True), rtol=1e-5, atol=1e-5)
            values.mul_(2)
            vector.add_(1 + 1j)


@pytest.mark.parametrize("n_rows,mean,maximum", [(0, 0, 0), (100, 20, 32), (100, 3.5, 210), (100, 8, 300)])
def test_pair_structural_fallback(n_rows, mean, maximum):
    assert not pair.eligible(n_rows, mean, maximum)


@pytest.mark.parametrize("inference", [False, True])
def test_unrolled_mixed_live_values_and_strided_output(inference):
    from flagsparse.sparse_operations import mixed_spmx

    device = accelerator_device()
    with torch.inference_mode(inference):
        n_rows = 262145
        ptr = torch.arange(n_rows + 1, device=device, dtype=torch.int32) * 9
        cols = torch.arange(n_rows * 9, device=device, dtype=torch.int32) % 13
        values = torch.ones(cols.numel(), device=device)
        vector = torch.ones(26, dtype=torch.complex64, device=device)[::2]
        target = torch.empty(n_rows * 2, dtype=torch.complex64, device=device)[::2]
        prepared = mixed_spmx.prepare_spmv_csr_mixed(values, cols, ptr, vector.dtype, (n_rows, 13))
        for step in range(2):
            output = fs.flagsparse_spmv_csr(prepared=prepared, x=vector, out=target)
            assert output is target
            expected = complex(9 * (step + 1), 9 * step * (step + 1))
            torch.testing.assert_close(output, torch.full_like(output, expected), rtol=1e-5, atol=1e-5)
            values.add_(1)
            vector.add_(1j)


@pytest.mark.parametrize("fmt", ["csr", "coo", "sell"])
@pytest.mark.parametrize("width", [2, 6, 40])
def test_original_complex_routes_and_live_values(fmt, width):
    device = accelerator_device()
    ptr = torch.arange(68, device=device, dtype=torch.int32) * width
    cols = torch.arange(67 * width, device=device, dtype=torch.int32) % 83
    values = (torch.arange(cols.numel(), device=device) % 7 - 3).to(torch.complex64) + 1j
    vector = torch.ones(83, dtype=torch.complex64, device=device)
    if fmt == "coo":
        rows = torch.arange(67, device=device, dtype=torch.int32).repeat_interleave(width)
        prepared = fs.prepare_spmv_coo(values, rows, cols, (67, 83))
        values = prepared.data_non
        def call():
            return fs.flagsparse_spmv_coo(prepared=prepared, x=vector)
    elif fmt == "csr":
        prepared = fs.prepare_spmv_csr(values, cols, ptr, (67, 83))
        def call():
            return fs.flagsparse_spmv_csr(prepared=prepared, x=vector)
    else:
        encoded, indices, offsets = fs.csr_to_sell(values, cols, ptr, 67, 32)
        def call():
            return fs.flagsparse_spmv_sell(encoded, indices, offsets, vector, (67, 83), slice_size=32, validate=False)
    for step in range(2):
        torch.testing.assert_close(call().cpu().to(torch.complex128), reference(values, cols, ptr, vector), rtol=1e-5, atol=1e-5)
        values.mul_(2)
        vector.add_(1 + 1j)
        if fmt == "sell":
            encoded.mul_(2)


@pytest.mark.parametrize("width", [2, 6, 10, 20, 80])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_paired_sell_live_layout_values_and_strides(width, dtype, conjugate, inference):
    device = accelerator_device()
    with torch.inference_mode(inference):
        lengths = torch.full((67,), width, dtype=torch.int32, device=device)
        lengths[0] = 0
        lengths[-1] = width + 1
        ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).int()))
        count = int(ptr[-1])
        cols = torch.arange(count, device=device, dtype=torch.int32) % 83
        values = (torch.arange(count, device=device) % 7 - 3).to(dtype)
        if dtype.is_complex:
            values = values + 2j
        vector = torch.ones(166, dtype=torch.complex64, device=device)[::2]
        target = torch.empty(134, dtype=torch.complex64, device=device)[::2]
        cache = {}
        selected = not inference and (dtype.is_complex or width > 8)
        for step in range(3):
            handled = ell.run(values, cols, ptr, vector, target, 67, cache, conjugate)
            assert handled == selected
            if handled:
                torch.testing.assert_close(target.cpu().to(torch.complex128), reference(values, cols, ptr, vector, conjugate), rtol=1e-5, atol=1e-5)
            values.add_(1)
            vector.add_(1 - 1j)
            cols.add_(1).remainder_(83)
            ptr[1].add_(1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_paired_sell_replaced_sources(dtype):
    device = accelerator_device()
    ptr = torch.arange(68, dtype=torch.int32, device=device) * 10
    cols = torch.arange(670, dtype=torch.int32, device=device) % 83
    values = torch.ones(670, dtype=dtype, device=device)
    vector = torch.arange(83, dtype=torch.float32, device=device).to(torch.complex64) + 1j
    target = torch.empty(67, dtype=torch.complex64, device=device)
    cache = {}
    for step in range(3):
        assert ell.run(values, cols, ptr, vector, target, 67, cache)
        torch.testing.assert_close(target.cpu().to(torch.complex128), reference(values, cols, ptr, vector), rtol=1e-5, atol=1e-5)
        values = values + 1
        cols = (cols + 1).remainder(83)
        ptr = ptr.clone()


@pytest.mark.parametrize("option,dtype", [
    ("GATHER_SPLIT", torch.complex64),
    ("GATHER_BATCH", torch.complex64),
    ("GATHER_LANES", torch.complex64),
    ("GATHER_WARPS", torch.complex64),
    ("GATHER_STAGES", torch.complex64),
    ("MIXED_ROWS", torch.float32),
    ("MIXED_BLOCK", torch.float32),
    ("MIXED_WARPS", torch.float32),
])
def test_paired_sell_respects_explicit_overrides(monkeypatch, option, dtype):
    device = accelerator_device()
    monkeypatch.setenv("FLAGSPARSE_ILUVATAR_" + option, "0" if option == "GATHER_SPLIT" else "2")
    ptr = torch.arange(68, dtype=torch.int32, device=device) * 10
    cols = torch.arange(670, dtype=torch.int32, device=device) % 83
    values = torch.ones(670, dtype=dtype, device=device)
    vector = torch.ones(83, dtype=torch.complex64, device=device)
    target = torch.empty(67, dtype=torch.complex64, device=device)
    assert not ell.run(values, cols, ptr, vector, target, 67, {})


@pytest.mark.parametrize("cache_modifier", ["", ".cg"])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("prefix_block", [64, 128])
@pytest.mark.parametrize("block", [128, 256])
@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_streaming_complex_nnz_prefix_boundaries(cache_modifier, paired, prefix_block, block, conjugate, index_dtype):
    import triton
    from flagsparse.sparse_operations._spmv_csr_nnz import _complex_nnz_kernel

    device = accelerator_device()
    lengths = torch.tensor([0, 1, 127, 128, 129, 511, 2751, 3, 0], dtype=index_dtype, device=device)
    ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).to(index_dtype)))
    count = int(ptr[-1])
    cols = torch.arange(count, dtype=index_dtype, device=device) % 83
    rows = torch.repeat_interleave(torch.arange(9, dtype=torch.int32, device=device), lengths.long())
    empty = torch.nonzero(lengths == 0).flatten().int()
    values = ((torch.arange(count, device=device) % 7 - 3).float() / 7).to(torch.complex64) + 2j
    vector = (torch.arange(83, device=device) % 11).to(torch.complex64) / 11 + 1j
    target = torch.empty(9, dtype=torch.complex64, device=device)
    for step in range(2):
        _complex_nnz_kernel[(triton.cdiv(max(count, empty.numel()), block),)](
            torch.view_as_real(values), cols, ptr, rows, torch.view_as_real(vector),
            torch.view_as_real(target), empty, empty.numel(), count, block,
            CONJUGATE=conjugate, CACHE_MODIFIER=cache_modifier, num_warps=1,
            PAIR_LOADS=paired, PREFIX_BLOCK=prefix_block,
        )
        torch.testing.assert_close(target.cpu().to(torch.complex128), reference(values, cols, ptr, vector, conjugate), rtol=1e-5, atol=1e-5)
        values.add_(1 + 1j)
        vector.add_(1 - 1j)
        cols.add_(1).remainder_(83)


@pytest.mark.parametrize("width", [6, 20, 80])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inference", [False, True])
def test_coo_pipeline_live_values_topology_and_strides(width, index_dtype, inference):
    device = accelerator_device()
    with torch.inference_mode(inference):
        lengths = torch.full((67,), width, dtype=index_dtype, device=device)
        lengths[0] = 0
        lengths[-1] = width + 1
        ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).to(index_dtype)))
        count = int(ptr[-1])
        cols = torch.arange(count, dtype=index_dtype, device=device) % 83
        values = (torch.arange(count, device=device) % 7 - 3).to(torch.complex64) + 2j
        vector = torch.ones(166, dtype=torch.complex64, device=device)[::2]
        target = torch.empty(134, dtype=torch.complex64, device=device)[::2]
        cache = {"coo_c32_non": True}
        for step in range(3):
            handled = ell.run(values, cols, ptr, vector, target, 67, cache)
            assert handled != inference
            if handled:
                torch.testing.assert_close(target.cpu().to(torch.complex128), reference(values, cols, ptr, vector), rtol=1e-5, atol=1e-5)
            values.add_(1 + 2j)
            vector.add_(1 - 1j)
            cols.add_(1).remainder_(83)
            ptr[1].add_(1)


@pytest.mark.parametrize("width", [6, 20, 80])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_coo_pipeline_public_route(width, index_dtype):
    device = accelerator_device()
    lengths = torch.full((67,), width, dtype=index_dtype, device=device)
    lengths[0] = 0
    ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).to(index_dtype)))
    rows = torch.repeat_interleave(torch.arange(67, dtype=index_dtype, device=device), lengths.long())
    cols = torch.arange(rows.numel(), dtype=index_dtype, device=device) % 83
    values = (torch.arange(rows.numel(), device=device) % 7 - 3).to(torch.complex64) + 2j
    vector = torch.ones(166, dtype=torch.complex64, device=device)[::2]
    target = torch.empty(134, dtype=torch.complex64, device=device)[::2]
    prepared = fs.prepare_spmv_coo(values, rows, cols, (67, 83))
    assert prepared.non_csr_plan[-1]["coo_c32_non"]
    for step in range(3):
        expected = reference(prepared.data_non, prepared.col_non, ptr, vector)
        output = fs.flagsparse_spmv_coo(prepared=prepared, x=vector, out=target)
        assert output is target
        torch.testing.assert_close(output.cpu().to(torch.complex128), expected, rtol=1e-5, atol=1e-5)
        prepared.data_non.add_(1 + 2j)
        vector.add_(1 - 1j)


@pytest.mark.parametrize("n_rows", [67, 4099, 16403])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("conjugate", [False, True])
def test_coo_sell_window_boundaries(n_rows, index_dtype, conjugate):
    device = accelerator_device()
    lengths = torch.arange(n_rows, device=device, dtype=index_dtype) % 17
    ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).to(index_dtype)))
    count = int(ptr[-1])
    cols = torch.arange(count, device=device, dtype=index_dtype) % 83
    values = (torch.arange(count, device=device) % 7 - 3).to(torch.complex64) + 2j
    vector = torch.ones(166, dtype=torch.complex64, device=device)[::2]
    target = torch.empty(n_rows * 2, dtype=torch.complex64, device=device)[::2]
    cache = {"coo_c32_non": True}
    for step in range(2):
        assert ell.run(values, cols, ptr, vector, target, n_rows, cache, conjugate)
        torch.testing.assert_close(target.cpu().to(torch.complex128), reference(values, cols, ptr, vector, conjugate), rtol=1e-5, atol=1e-5)
        sigma = 4096 if conjugate else 16384
        windows = torch.arange(n_rows, device=device) // sigma
        expected = torch.argsort(windows * (int(lengths.max()) + 1) + int(lengths.max()) - lengths, stable=True)
        assert torch.equal(cache["ell_layout"][3][:n_rows].long(), expected)
        values.add_(1 + 2j)
        vector.add_(1 - 1j)
        cols.add_(1).remainder_(83)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("conjugate", [False, True])
def test_streaming_csc_public_long_rows_live_inputs(index_dtype, conjugate):
    device = accelerator_device()
    lengths = torch.full((263,), 2, dtype=index_dtype, device=device)
    lengths[0] = 0
    lengths[-1] = 2751
    ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0).to(index_dtype)))
    count = int(ptr[-1])
    cols = torch.arange(count, dtype=index_dtype, device=device) % 83
    values = (torch.arange(count, device=device) % 7 - 3).to(torch.complex64) + 2j
    vector = torch.ones(166, dtype=torch.complex64, device=device)[::2]
    prepared = fs.prepare_spmv_csc(values, cols, ptr, (83, 263), op="conj" if conjugate else "trans")
    assert prepared.iluvatar_nnz_plan is not None
    for step in range(3):
        output = fs.flagsparse_spmv_csc(prepared=prepared, x=vector)
        torch.testing.assert_close(output.cpu().to(torch.complex128), reference(values, cols, ptr, vector, conjugate), rtol=1e-5, atol=1e-5)
        values.add_(1 + 1j)
        vector.add_(1 - 1j)
        cols.add_(1).remainder_(83)
