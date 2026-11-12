# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The per-slice complex SELL route is confined to MUSA complex64 C=32."""

import importlib

import pytest

torch = pytest.importorskip("torch")
sell = importlib.import_module("flagsparse.sparse_operations.spmv_sell")


@pytest.mark.parametrize("musa", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128, torch.float32])
@pytest.mark.parametrize("slice_size", [16, 32])
@pytest.mark.parametrize("width", [2, 9])
def test_per_slice_scope(monkeypatch, musa, dtype, slice_size, width):
    monkeypatch.setattr(sell, "_is_mthreads_runtime", lambda: musa)
    for name in (
        "_is_maca_runtime",
        "_is_rocm_runtime",
        "_is_iluvatar_runtime",
        "_is_ascend_runtime",
    ):
        monkeypatch.setattr(sell, name, lambda: False)
    m, n = 65, 7
    slices = (m + slice_size - 1) // slice_size
    values = torch.ones(slices * slice_size * width, dtype=dtype)
    cols = torch.zeros(values.numel(), dtype=torch.int32)
    offsets = torch.arange(slices + 1, dtype=torch.int32) * (slice_size * width)
    dims = (m, n, slice_size, slices)
    monkeypatch.setattr(sell, "_validate_spmv_sell", lambda *a: dims)
    monkeypatch.setattr(
        sell,
        "_musa_sell_plan",
        lambda *a: {"dims": dims, "arrays": (values, cols, offsets), "csr": ()},
    )
    calls = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *a, **kw: calls.append("generic")

    for name in (
        "_musa_spmv_sell_kernel",
        "_spmv_sell_complex_parallel_kernel",
        "_spmv_sell_complex_kernel",
        "_spmv_sell_real_kernel",
        "_spmv_sell_wide_kernel",
    ):
        monkeypatch.setattr(sell, name, Kernel(name))
    monkeypatch.setattr(
        sell, "_launch_musa_spmv", lambda kernel, *a, **kw: calls.append(kernel.name)
    )
    sell.flagsparse_spmv_sell(
        values,
        cols,
        offsets,
        torch.ones(n, dtype=dtype),
        (m, n),
        slice_size=slice_size,
        validate=False,
    )
    expected = "generic"
    if musa:
        if dtype == torch.complex64 and slice_size == 32 and width > 4.5:
            expected = "_spmv_sell_complex_parallel_kernel"
        elif dtype == torch.float32 and slice_size == 32 and width > 4.5:
            expected = "_spmv_sell_wide_kernel"
        else:
            expected = "_musa_spmv_sell_kernel"
    assert calls == [expected]
