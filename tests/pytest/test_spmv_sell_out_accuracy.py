# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SpMV SELL writes the caller's ``out`` on every backend.

Kept OUT of test_q4_variants_accuracy.py on purpose: Iluvatar's runner command selects
that whole module with ``-k "test_q4_variant or ..."``, so a float64 case placed there
bypasses its ``not float64`` filter -- and BI-V150's fp64 H2D copy returns zeros.
"""

import pytest
import torch

import flagsparse as fs
from tests.pytest.accuracy_utils import ACCELERATOR_REQUIRED, accelerator_available
from tests.pytest.test_q4_variants_accuracy import _check, _dev, _rand, _sparse, _wide

pytestmark = pytest.mark.skipif(
    not accelerator_available(), reason=ACCELERATOR_REQUIRED
)


@pytest.mark.spmv_sell
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=str)
@pytest.mark.parametrize("strided_out", [False, True], ids=["dense_out", "strided_out"])
def test_spmv_sell_writes_caller_out(dtype, strided_out):
    # out= must be filled on every backend (Ascend computes into a fresh tensor),
    # including a strided view, and FP64 values must keep FP64 accumulation.
    gen = torch.Generator().manual_seed(7)
    m, k = 257, 129
    A, x = _sparse(m, k, dtype, gen), _rand((k,), dtype, gen)
    vals, cols, offsets = _dev(*fs.dense_to_sell(A, 8))
    x = _dev(x)[0]
    storage = torch.full((2 * m,), -7.0, dtype=dtype, device=x.device)
    out = storage[::2] if strided_out else storage[:m]
    y = fs.flagsparse_spmv_sell(vals, cols, offsets, x, (m, k), slice_size=8, out=out)
    assert y is out
    _check(out, _wide(A) @ _wide(x.cpu()), dtype)
    if strided_out:
        assert bool((storage[1::2] == -7.0).all())
