# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The prepared SpMM entries accept ``out=`` and return that buffer.

Added for the Ascend native FP64 route, whose run time depends on where its B
and C sit: 222 ms at one byte offset against 155 ms at another for the same
matrix. The CANN baseline binds its output once, so while our prepared API
allocated a new result every call the two sides drew different C addresses and
the ratio read ~0.72 for reasons that had nothing to do with the computation.
Owning the buffer is what lets a benchmark hold both sides at one address.

Routes that cannot write through to ``out`` must still honour it, so the public
entry copies for them; ``_route_takes_out`` is what keeps a route from being
handed an argument it would silently drop.
"""

import pytest

torch = pytest.importorskip(
    "torch", reason="tests/ci runs on a CPU-only runner without torch"
)
spmm_coo = pytest.importorskip("flagsparse.sparse_operations.spmm_coo")
spmm_csr = pytest.importorskip("flagsparse.sparse_operations.spmm_csr")


def test_public_entries_expose_out():
    import inspect

    for entry in (spmm_coo.flagsparse_spmm_coo_run, spmm_csr.flagsparse_spmm_csr_run):
        assert "out" in inspect.signature(entry).parameters, entry.__name__


def test_route_takes_out_distinguishes_the_routes():
    """Only the route reaching the native FP64 plan declares ``out``.

    If a route that ignores it were to declare it, the public entry would skip
    its copy and hand the caller a tensor that is not the buffer they passed.
    """
    assert spmm_csr._route_takes_out(spmm_coo._run_spmm_coo_rowrun_route)
    assert not spmm_csr._route_takes_out(spmm_coo._run_spmm_coo_atomic_route)
    assert not spmm_csr._route_takes_out(spmm_coo._run_spmm_coo_alg1_route)
    assert spmm_csr._route_takes_out(spmm_csr._run_spmm_csr_base_route)


@pytest.mark.parametrize(
    "bad,expected",
    [
        ("not a tensor", TypeError),
        ("wrong_shape", ValueError),
        ("wrong_dtype", ValueError),
        ("non_contiguous", ValueError),
    ],
)
def test_validate_spmm_out_rejects_mismatches(bad, expected):
    """A bad buffer has to fail at the boundary.

    The native route's own guard would merely decline it, fall back to an
    internal allocation and return a different tensor -- a silently unfilled
    ``out`` is worse than an error.
    """
    from types import SimpleNamespace

    prepared = SimpleNamespace(
        n_rows=4, n_cols=3, data=torch.zeros(1, dtype=torch.float32)
    )
    dense = torch.zeros((3, 2), dtype=torch.float32)
    buffers = {
        "not a tensor": object(),
        "wrong_shape": torch.zeros((4, 5), dtype=torch.float32),
        "wrong_dtype": torch.zeros((4, 2), dtype=torch.float64),
        "non_contiguous": torch.zeros((2, 4), dtype=torch.float32).T,
    }

    with pytest.raises(expected):
        spmm_csr._validate_spmm_out(prepared, dense, buffers[bad], "row")


def test_validate_spmm_out_accepts_a_matching_buffer():
    from types import SimpleNamespace

    prepared = SimpleNamespace(
        n_rows=4, n_cols=3, data=torch.zeros(1, dtype=torch.float32)
    )
    dense = torch.zeros((3, 2), dtype=torch.float32)
    out = torch.zeros((4, 2), dtype=torch.float32)

    assert spmm_csr._validate_spmm_out(prepared, dense, out, "row") is None


def test_validate_spmm_out_prefers_the_prepared_output_dtype():
    """COO names its result dtype (mixed i8->i32); CSR follows its values."""
    from types import SimpleNamespace

    mixed = SimpleNamespace(
        n_rows=4,
        n_cols=3,
        data=torch.zeros(1, dtype=torch.int8),
        output_dtype=torch.int32,
    )
    dense = torch.zeros((3, 2), dtype=torch.int8)

    assert (
        spmm_csr._validate_spmm_out(
            mixed, dense, torch.zeros((4, 2), dtype=torch.int32), "row"
        )
        is None
    )
    with pytest.raises(ValueError):
        spmm_csr._validate_spmm_out(
            mixed, dense, torch.zeros((4, 2), dtype=torch.int8), "row"
        )
