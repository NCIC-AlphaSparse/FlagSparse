# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Sparse reuse: mutated inputs, inference tensors, and API outputs.

Written for the DCU plan caches, but every case checks the public API against a
CPU result, so it is backend-neutral. It used to skip everywhere except ROCm,
which left MACA's spmv_coo (and the other operators here) without this coverage.
"""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations._common import (
    _backend_name,
    _is_maca_runtime,
    _is_rocm_runtime,
)
from tests.pytest.accuracy_utils import accelerator_available, accelerator_device

# Enabled where it has been run: ROCm (BW1000), MACA (C550, 18 passed) and CUDA.
# The runner's accuracy phase collects these by operator marker, so opening it to
# a backend that has never run it would change that backend's delivery result.
_VERIFIED_BACKEND = _is_rocm_runtime() or _is_maca_runtime() or _backend_name() == "cuda"

pytestmark = pytest.mark.skipif(
    not accelerator_available() or not _VERIFIED_BACKEND,
    reason="reuse coverage not yet verified on this backend (ROCm, MACA, CUDA only)",
)


@pytest.mark.spmv_coo
@pytest.mark.parametrize(
    "dtype,out_dtype",
    [
        (torch.float16, torch.float16),
        (torch.float16, torch.float32),
        (torch.int8, torch.int32),
        (torch.int8, torch.float32),
    ],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_mixed_coo_reuse_tracks_all_inputs(dtype, out_dtype, index_dtype):
    device = accelerator_device()
    values = torch.tensor([1, 2, -3, 4], dtype=dtype, device=device)
    rows = torch.tensor([2, 0, 2, 1], dtype=index_dtype, device=device)
    cols = torch.tensor([1, 0, 0, 2], dtype=index_dtype, device=device)
    x = torch.tensor([2, -1, 3], dtype=dtype, device=device)
    out = torch.empty(6, dtype=out_dtype, device=device)[::2]

    def check():
        ref = torch.zeros(3, dtype=torch.float64)
        ref.index_add_(
            0,
            rows.cpu().long(),
            values.cpu().double() * x.cpu().double()[cols.cpu().long()],
        )
        y = fs.flagsparse_spmv_coo(values, rows, cols, x, (3, 3), out=out)
        assert y is out
        torch.testing.assert_close(y.cpu(), ref.to(out_dtype))

    check()
    check()
    values.add_(1)
    check()
    cols.copy_(torch.tensor([2, 1, 0, 0], dtype=index_dtype, device=device))
    check()
    rows.copy_(torch.tensor([0, 1, 0, 2], dtype=index_dtype, device=device))
    check()
    x.add_(1)
    check()


@pytest.mark.spmv_csc
@pytest.mark.parametrize(
    "return_time,return_meta",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_half_csc_reuse_preserves_wide_output_and_return_shape(
    return_time, return_meta
):
    device = accelerator_device()
    values = torch.tensor([1, 2, 3, 4], dtype=torch.float16, device=device)
    rows = torch.tensor([0, 1, 0, 2], dtype=torch.int32, device=device)
    ptr = torch.tensor([0, 2, 3, 4], dtype=torch.int32, device=device)
    x = torch.tensor([2, 3, 4], dtype=torch.float16, device=device)
    out = torch.empty(6, dtype=torch.float32, device=device)[::2]
    for _ in range(2):
        result = fs.flagsparse_spmv_csc(
            values,
            rows,
            ptr,
            x,
            (3, 3),
            out=out,
            return_time=return_time,
            return_meta=return_meta,
        )
        if return_time or return_meta:
            assert len(result) == 1 + int(return_time) + int(return_meta)
            y = result[0]
        else:
            y = result
        assert y is out
        ref = torch.zeros(3, dtype=torch.float32)
        columns = torch.repeat_interleave(
            torch.arange(3), (ptr[1:] - ptr[:-1]).cpu().long()
        )
        ref.index_add_(
            0, rows.cpu().long(), values.cpu().float() * x.cpu().float()[columns]
        )
        torch.testing.assert_close(y.cpu(), ref)
        rows.copy_(torch.tensor([2, 0, 1, 0], dtype=torch.int32, device=device))
        values.add_(1)


@pytest.mark.spmm_csc
@pytest.mark.parametrize("dense_cols", [1, 17, 32, 65])
def test_complex_csc_gather_reuse_and_prepared_value_updates(dense_cols):
    device = accelerator_device()
    values = torch.tensor(
        [1 + 2j, 2 - 1j, 3 + 1j, -4 + 2j], dtype=torch.complex64, device=device
    )
    rows = torch.tensor([0, 1, 0, 2], dtype=torch.int32, device=device)
    ptr = torch.tensor([0, 2, 3, 4], dtype=torch.int32, device=device)
    B = (
        torch.complex(torch.randn(dense_cols, 3), torch.randn(dense_cols, 3))
        .t()
        .to(device)
    )
    prepared = fs.prepare_spmm_csc_route(values, rows, ptr, (4, 3))
    for _ in range(2):
        dense = torch.zeros(4, 3, dtype=torch.complex64)
        dense[rows.cpu().long(), torch.tensor([0, 0, 1, 2])] = values.cpu()
        ref = dense @ B.cpu()
        for kwargs in ({}, {"prepared": prepared}):
            y = fs.flagsparse_spmm_csc(values, rows, ptr, B, (4, 3), **kwargs)
            torch.testing.assert_close(y.cpu(), ref, rtol=1e-4, atol=1e-4)
        values.mul_(2)


@pytest.mark.spmv_coo
def test_inference_coo_recomputes_after_mutation_and_handles_empty_rows():
    device = accelerator_device()
    with torch.inference_mode():
        values = torch.tensor([1, 2], dtype=torch.float16, device=device)
        rows = torch.tensor([1, 0], dtype=torch.int32, device=device)
        cols = torch.tensor([0, 1], dtype=torch.int32, device=device)
        x = torch.tensor([2, 3], dtype=torch.float16, device=device)
        torch.testing.assert_close(
            fs.flagsparse_spmv_coo(values, rows, cols, x, (2, 2)).cpu(),
            torch.tensor([6, 2], dtype=torch.float16),
        )
        values.add_(1)
        torch.testing.assert_close(
            fs.flagsparse_spmv_coo(values, rows, cols, x, (2, 2)).cpu(),
            torch.tensor([9, 4], dtype=torch.float16),
        )
        empty = values[:0]
        y = fs.flagsparse_spmv_coo(empty, rows[:0], cols[:0], x, (0, 2))
        assert y.shape == (0,)


@pytest.mark.spgemm_csr
def test_spgemm_repeated_calls_track_topology_and_value_mutation():
    device = accelerator_device()
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    ai = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    ap = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    b = torch.tensor([4.0, 5.0, 6.0], device=device)
    bi = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    bp = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)

    def dense(values, indices, ptr):
        result = torch.zeros(2, 2)
        rows = torch.repeat_interleave(
            torch.arange(2), (ptr[1:] - ptr[:-1]).cpu().long()
        )
        result.index_put_((rows, indices.cpu().long()), values.cpu(), accumulate=True)
        return result

    for stage in range(4):
        values, indices, ptr, shape = fs.flagsparse_spgemm_csr(
            a, ai, ap, (2, 2), b, bi, bp, (2, 2)
        )
        assert shape == (2, 2)
        torch.testing.assert_close(
            dense(values, indices, ptr), dense(a, ai, ap) @ dense(b, bi, bp)
        )
        if stage == 0:
            a.add_(1)
            b.mul_(2)
        elif stage == 1:
            ai.copy_(torch.tensor([1, 1, 0], dtype=torch.int32, device=device))
        elif stage == 2:
            bi.copy_(torch.tensor([1, 0, 0], dtype=torch.int32, device=device))
