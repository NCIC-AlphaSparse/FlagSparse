# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import importlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tests.pytest.accuracy_utils import accelerator_available, accelerator_device

csr = importlib.import_module("flagsparse.sparse_operations.spmv_csr")
sell = importlib.import_module("flagsparse.sparse_operations.spmv_sell")
pytestmark = pytest.mark.skipif(not accelerator_available(), reason="accelerator required")


def test_spgemm_matrix_worker_reaches_compute(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = tmp_path / "worker.pt"
    proc = subprocess.run([
        sys.executable, str(root / "tests/test_spgemm.py"), "--_matrix-worker",
        "--_worker-mtx", str(root / "tests/data/q4_worker_smoke.mtx"),
        "--_worker-output", str(result), "--warmup", "1", "--iters", "1",
        "--no-cusparse", "--no-ref-isolated-retry",
    ], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = torch.load(result, weights_only=False)
    assert payload["success"]
    assert payload["entry"]["triton_ms"] is not None
    assert payload["entry"]["error"] is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("op", ["trans", "conj"])
def test_auto_transpose_reuses_structure_and_refreshes_values(dtype, index_dtype, op, monkeypatch):
    device = accelerator_device()
    # Empty rows, duplicate columns, a row crossing several reduction tiles.
    lengths = torch.tensor([0, 1, 129, 0, 3], device=device)
    ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(index_dtype)
    col = (torch.arange(133, device=device) % 7).to(index_dtype)
    values = torch.randn(133, dtype=dtype, device=device)
    x = torch.randn(5, dtype=dtype, device=device)
    prepared = csr.prepare_spmv_csr(values, col, ptr, (5, 9), op=op)
    def unexpected(*args, **kwargs):
        raise AssertionError("auto transpose rebuilt CSR")
    monkeypatch.setattr(csr, "_transpose_csr_for_spmv", unexpected)
    rows = torch.repeat_interleave(torch.arange(5, device=device), lengths)
    out = torch.empty(18, device=device, dtype=dtype)[::2]
    for _ in range(2):
        expected = torch.zeros(9, device=device, dtype=dtype)
        expected.index_add_(0, col.long(), (values.conj() if op == "conj" else values) * x[rows])
        # A contiguous output view exercises caller-owned output storage.
        target = out.contiguous()
        actual, meta = csr.flagsparse_spmv_csr_run(prepared, x, out=target, timing=True, return_meta=True)
        assert actual is target
        assert meta["transpose_strategy"] == "prepared_structure_gather"
        assert meta["process_gpu_ms"] == 0
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        values.mul_(0.5)
        x.neg_()


def test_auto_transpose_inference_values_are_read_live():
    device = accelerator_device()
    with torch.inference_mode():
        values = torch.tensor([1., 2., 3.], device=device)
        cols = torch.tensor([0, 1, 0], device=device, dtype=torch.int32)
        ptr = torch.tensor([0, 2, 3], device=device, dtype=torch.int32)
        prepared = csr.prepare_spmv_csr(values, cols, ptr, (2, 2), op="trans")
        x = torch.tensor([2., 4.], device=device)
        for scale in (1., 2.):
            actual = csr.flagsparse_spmv_csr_run(prepared, x)
            torch.testing.assert_close(actual, torch.tensor([14., 4.], device=device) * scale)
            values.mul_(2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.complex64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layout", ["row", "col"])
@pytest.mark.parametrize("width", [1, 31, 32, 33])
def test_batched_spmm_gather_strides_and_tail(dtype, index_dtype, layout, width):
    from flagsparse.sparse_operations import _spmm_row_gather
    device = accelerator_device()
    # A long first row, duplicate columns, empty interior and final rows.
    ptr = torch.tensor([0, 97, 97, 100, 100, 101, 101], device=device, dtype=index_dtype)
    cols = (torch.arange(101, device=device) % 19).to(index_dtype)
    values = torch.randn(101, dtype=dtype, device=device) * .1
    dense = torch.randn(19, width, dtype=dtype, device=device)
    b = dense.t().contiguous().t() if layout == "col" else dense
    backing = torch.empty((6, width + 3) if layout == "row" else (width + 3, 6),
                          dtype=dtype, device=device)
    out = backing[:, :width] if layout == "row" else backing[:width, :].t()
    got = _spmm_row_gather.compute(values, cols, ptr, b, out)
    rows = torch.repeat_interleave(torch.arange(6, device=device), (ptr[1:] - ptr[:-1]).long())
    wide = torch.complex128 if dtype.is_complex else torch.float64
    expected = torch.zeros(6, width, dtype=wide, device=device)
    expected.index_add_(0, rows, values.to(wide)[:, None] * b.to(wide)[cols.long()])
    assert got is out
    torch.testing.assert_close(got, expected.to(dtype), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.complex64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_non_spmv_empty_rows_duplicate_columns_and_values_updates(dtype, index_dtype):
    import flagsparse as fs
    device = accelerator_device()
    dense = torch.zeros(17, 13, dtype=dtype)
    dense[1, :3], dense[4, :7], dense[16, :2] = 1, -1, 2
    sparse = dense.to_sparse_csr()
    values, cols, ptr = (sparse.values().to(device), sparse.col_indices().to(device, index_dtype),
                         sparse.crow_indices().to(device, index_dtype))
    prepared = fs.prepare_spmv_csr(values, cols, ptr, (17, 13))
    x = torch.randn(13, dtype=dtype, device=device)
    for scale in (1., 2.):
        got = fs.flagsparse_spmv_csr(prepared=prepared, x=x)
        wide = torch.complex128 if dtype.is_complex else torch.float64
        expected = (dense.to(wide).to(device) @ x.to(wide) * scale).to(dtype)
        torch.testing.assert_close(got, expected, atol=2e-3, rtol=2e-3)
        values.mul_(2)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("op", ["non", "conj"])
@pytest.mark.parametrize("nnz", [0, 1, 1024, 1025, 65536])
def test_spvv_packed_partial_reduction(dtype, index_dtype, op, nnz):
    import flagsparse as fs
    device = accelerator_device()
    values = torch.randn(nnz, device=device, dtype=dtype)
    indices = torch.randperm(max(nnz * 2, 1), device=device)[:nnz].to(index_dtype)
    y = torch.randn(max(nnz * 2, 1), device=device, dtype=dtype)
    actual = fs.flagsparse_spvv(values, indices, y, op=op)
    wide = values.to(torch.complex128)
    if op == "conj":
        wide = wide.conj()
    expected = (wide * y.to(torch.complex128)[indices.long()]).sum()
    torch.testing.assert_close(actual, expected.to(dtype), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_csc_half_skew_empty_columns_and_caller_output(index_dtype, out_dtype):
    import flagsparse as fs
    device = accelerator_device()
    lengths = torch.tensor([0, 1, 257, 0, 3], device=device)
    ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(index_dtype)
    rows = (torch.arange(261, device=device) % 19).to(index_dtype)
    values = torch.randn(261, dtype=torch.float16, device=device) * .1
    x = torch.randn(5, dtype=torch.float16, device=device)
    cols = torch.repeat_interleave(torch.arange(5, device=device), lengths)
    expected = torch.zeros(23, device=device, dtype=torch.float64)
    expected.index_add_(0, rows.long(), values.double() * x.double()[cols])
    out = torch.empty(23, dtype=out_dtype, device=device)
    actual, elapsed, meta = fs.flagsparse_spmv_csc(values, rows, ptr, x, (23, 5),
        out=out, return_time=True, return_meta=True)
    assert actual is out and elapsed >= 0 and meta["route"] == "mixed"
    torch.testing.assert_close(actual, expected.to(out_dtype), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("slice_size", [1, 3, 16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128, torch.int8])
def test_sell_group_tail_and_uneven_widths(slice_size, dtype):
    device = accelerator_device()
    dense = torch.zeros(259, 97, dtype=dtype)
    dense[1, :3] = 2
    dense[31, :65] = 1
    dense[64, :7] = -1
    dense[258, :5] = 3
    data = dense.to_sparse_csr()
    v, c, p = (t.to(device) for t in (data.values(), data.col_indices(), data.crow_indices()))
    sv, sc, so = sell.csr_to_sell(v, c, p, 259, slice_size)
    x = torch.ones(97, device=device, dtype=dtype)
    actual = sell.flagsparse_spmv_sell(sv, sc, so, x, (259, 97), slice_size=slice_size)
    expected = dense.to(torch.int32 if dtype == torch.int8 else dtype).sum(1).to(actual.dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,out_dtype", [(torch.int8, torch.int32), (torch.int8, torch.float32),
                                             (torch.float16, torch.float32)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("width", [1, 31, 32, 33, 129])
@pytest.mark.parametrize("nnz", [0, 19, 97])
def test_coo_mixed_batched_unsorted_duplicates_strides(dtype, out_dtype, index_dtype, width, nnz):
    import flagsparse as fs
    device = accelerator_device()
    # Unsorted rows, duplicate coordinates, empty rows, partial nonzero tiles.
    rows = (torch.arange(nnz, device=device) * 7 % 5).to(index_dtype)
    cols = (torch.arange(nnz, device=device) * 3 % 11).to(index_dtype)
    values = (torch.arange(nnz, device=device) % 7 - 3).to(dtype)
    b = (torch.arange(11 * width, device=device).reshape(width, 11) % 7 - 3).t().to(dtype)
    out = torch.empty(width + 3, 8, dtype=out_dtype, device=device)[:width].t()
    got = fs.flagsparse_spmm_coo(values, rows, cols, b, (8, 11), out=out, out_dtype=out_dtype)
    expected = torch.zeros(8, width, dtype=torch.float64, device=device)
    expected.index_add_(0, rows.long(), values.double()[:, None] * b.double()[cols.long()])
    assert got is out
    torch.testing.assert_close(got, expected.to(out_dtype), rtol=0, atol=0)
