# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The transposed CSR SpMV value cache must reorder complex values portably.

``_spmv_csr_transpose.gather`` materialises ``values`` in transpose order once
per tensor version. It did so with ``values[order]``, and Moore Threads has no
complex kernel for advanced indexing (``"IndexMusa" not implemented for
'ComplexFloat'``), so spmv_csr_c32_int_conj failed on MUSA while CUDA was
green. The reorder has to go through ``_common._gather_values``.

The test runs the real ``gather`` on CPU with a tensor subclass that rejects
complex tensor-indexing the way MUSA does, and a no-op launch in place of the
Triton kernel -- only the host-side value cache is under test.
"""

import importlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Imported by name rather than with a from-import: that line would pass 79
# columns, which isort and black wrap in opposite directions forever.
transpose = importlib.import_module("flagsparse.sparse_operations._spmv_csr_transpose")


class MusaLikeTensor(torch.Tensor):
    """Raises on complex advanced indexing, as torch_musa does."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if (
            func is torch.Tensor.__getitem__
            and args[0].is_complex()
            and isinstance(args[1], torch.Tensor)
        ):
            raise RuntimeError("\"IndexMusa\" not implemented for 'ComplexFloat'")
        return super().__torch_function__(func, types, args, kwargs or {})


class _Launch:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append(args)

        return launch


def _plan(order):
    n = order.numel()
    cols = torch.zeros(n, dtype=torch.int32)
    ptr = torch.tensor([0, n], dtype=torch.int32)
    rows = torch.zeros(1, dtype=torch.int32)
    return cols, ptr, order, rows, 4, 1, {}


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_cached_transpose_values_avoid_complex_indexing(monkeypatch, dtype):
    launch = _Launch()
    monkeypatch.setattr(transpose, "_gather", launch)
    real = torch.arange(6, dtype=torch.float64)
    plain = torch.complex(real, -real - 0.5).to(dtype)
    values = plain.as_subclass(MusaLikeTensor)
    order = torch.tensor([4, 0, 5, 2, 1, 3])
    with pytest.raises(RuntimeError, match="IndexMusa"):
        values[order]

    plan = _plan(order)
    x = torch.ones(1, dtype=dtype)
    transpose.gather(values, x, plan, n_rows=1)

    cached = plan[-1]["values"]
    torch.testing.assert_close(
        cached.as_subclass(torch.Tensor), plain[order], rtol=0, atol=0
    )
    assert len(launch.calls) == 1


def test_real_values_keep_the_direct_index(monkeypatch):
    monkeypatch.setattr(transpose, "_gather", _Launch())
    values = torch.arange(6, dtype=torch.float32)
    order = torch.tensor([4, 0, 5, 2, 1, 3])
    plan = _plan(order)

    transpose.gather(values, torch.ones(1), plan, n_rows=1)

    torch.testing.assert_close(plan[-1]["values"], values[order])
