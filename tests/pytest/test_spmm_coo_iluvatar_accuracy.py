# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BI-V prepared SpMM retains coalescing and reads live operands."""

import pytest
import torch

import flagsparse as fs
from flagsparse.sparse_operations import _common


@pytest.mark.spmm_coo
@pytest.mark.parametrize("n", [0, 17, 32, 65])
@pytest.mark.parametrize("layout", ["row", "col"])
def test_spmm_coo_iluvatar_f16_duplicates_and_live_rhs(n, layout):
    if not _common._is_iluvatar_runtime():
        pytest.skip("BI-V half row route")
    # Duplicate sum 2048 + 1 - 2048 must retain the middle value in fp32.
    row = torch.tensor([0, 0, 0, 3, 8], dtype=torch.int32)
    col = torch.tensor([1, 1, 1, 0, 2], dtype=torch.int32)
    data = torch.tensor([2048, 1, -2048, 2, -3], dtype=torch.float16)
    dense = torch.zeros(9, 4)
    dense.index_put_((row.long(), col.long()), data.float(), accumulate=True)
    prepared = fs.prepare_spmm_coo_route(data.cuda(), row.cuda(), col.cuda(), (9, 4))
    gen = torch.Generator().manual_seed(90210)
    rhs_cpu = torch.randn((4, n), generator=gen).half()
    rhs = rhs_cpu.cuda()
    if layout == "col":
        rhs = rhs.T.contiguous().T
    for step in range(2):
        if step:
            rhs_cpu.add_(0.5)
            rhs.copy_(rhs_cpu)
        got = fs.flagsparse_spmm_coo_run(prepared, rhs, dense_layout=layout)
        ref = (dense @ rhs_cpu.float()).half()
        torch.testing.assert_close(got.cpu(), ref, rtol=1e-3, atol=1e-3)


@pytest.mark.spmm_coo
@pytest.mark.parametrize("pattern", ["short", "dense", "long", "empty"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layout", ["row", "col"])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_spmm_coo_iluvatar_c32_live_operands(pattern, index_dtype, layout, op):
    if not _common._is_iluvatar_runtime():
        pytest.skip("BI-V complex row route")
    generator = torch.Generator().manual_seed(1022)
    dense = torch.randn((13, 137), dtype=torch.complex64, generator=generator)
    lengths = {
        "short": [0, 1, 2, 3, 9, 0, 2, 1, 4, 0, 3, 2, 0],
        "dense": [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0],
        "long": [0, 1, 0, 3, 137, 0, 0, 1, 0, 2, 0, 0, 0],
        "empty": [0] * 13,
    }[pattern]
    mask = torch.arange(137)[None, :] < torch.tensor(lengths)[:, None]
    dense = torch.where(mask, dense, 0)
    sparse = dense.to_sparse_coo().coalesce()
    row, col = sparse.indices()
    values = sparse.values().clone()
    if values.numel():
        duplicate = values[:1] * 0.25
        values[:1].mul_(0.75)
        values = torch.cat((values, duplicate))
        row, col = torch.cat((row, row[:1])), torch.cat((col, col[:1]))
    prepared = fs.prepare_spmm_coo_route(
        values.cuda(), row.to(device="cuda", dtype=index_dtype),
        col.to(device="cuda", dtype=index_dtype), dense.shape, op=op,
    )
    matrix = dense if op == "non" else dense.T if op == "trans" else dense.conj().T
    reference_matrix = matrix.to(torch.complex128)
    rhs_cpu = torch.randn((matrix.shape[1], 32), dtype=torch.complex64, generator=generator)
    storage = torch.empty((matrix.shape[1], 64), dtype=torch.complex64, device="cuda")
    rhs = storage[:, ::2]
    if layout == "col":
        storage = torch.empty((64, matrix.shape[1]), dtype=torch.complex64, device="cuda")
        rhs = storage[::2].T
    for step in range(2):
        if step:
            rhs_cpu.add_(0.5 - 0.25j)
            prepared.data.mul_(0.75 + 0.5j)
            reference_matrix.mul_(0.75 + 0.5j)
        rhs.copy_(rhs_cpu)
        got, meta = fs.flagsparse_spmm_coo_run(
            prepared, rhs, dense_layout=layout, return_meta=True, diagnostics=True,
        )
        reference = reference_matrix @ rhs_cpu.to(torch.complex128)
        torch.testing.assert_close(got.cpu().to(torch.complex128), reference,
                                   rtol=1e-4, atol=1e-4)
        if prepared.nnz:
            diagnostics = meta["diagnostics"]
            assert diagnostics["launch_version"] == "coo_biv_complex_rows_v1"
            expected_tile = (2, 1) if prepared.nnz <= 8 * prepared.n_rows else (1, 2)
            assert (diagnostics["batch_rows"], diagnostics["block_nnz"]) == expected_tile
            assert diagnostics["num_warps"] == 1


@pytest.mark.spmm_coo
@pytest.mark.parametrize("n", [0, 1, 17, 65])
@pytest.mark.parametrize("layout", ["row", "col"])
def test_spmm_coo_iluvatar_c32_other_widths(n, layout):
    if not _common._is_iluvatar_runtime():
        pytest.skip("BI-V complex row route")
    dense = torch.tensor([[1 + 2j, 0], [0, 0], [2 - 3j, -1j]], dtype=torch.complex64)
    sparse = dense.to_sparse_coo().coalesce()
    row, col = sparse.indices()
    prepared = fs.prepare_spmm_coo_route(
        sparse.values().cuda(), row.int().cuda(), col.int().cuda(), dense.shape,
    )
    generator = torch.Generator().manual_seed(1022)
    rhs_cpu = torch.randn((2, n), dtype=torch.complex64, generator=generator)
    got = fs.flagsparse_spmm_coo_run(prepared, rhs_cpu.cuda(), dense_layout=layout)
    reference = dense.to(torch.complex128) @ rhs_cpu.to(torch.complex128)
    torch.testing.assert_close(got.cpu().to(torch.complex128), reference,
                               rtol=1e-4, atol=1e-4)
