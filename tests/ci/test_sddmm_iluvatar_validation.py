# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""The fused Iluvatar CSR pattern check must agree with the generic checks.

The SDDMM benchmark times ``prepare + validate + kernel`` on every call, and the
generic validation is ~8 small torch launches plus host syncs. On BI-V150 a single
Triton kernel does the same check. It is only a fast path for VALID input: anything
it flags falls through to the generic checks, which raise the specific error. These
tests need a real accelerator (a Triton kernel and accelerator tensors), so they skip
on the CPU-only runner and run on a CUDA box or on the BI-V150 itself.
"""

import importlib

import pytest

torch = pytest.importorskip(
    "torch", reason="tests/ci runs on a CPU-only runner without torch"
)
if not torch.cuda.is_available():
    pytest.skip(
        "needs an accelerator to run the Triton kernel", allow_module_level=True
    )

sddmm = importlib.import_module("flagsparse.sparse_operations.sddmm_csr")
DEV = "cuda"


def _pattern(n_rows, n_cols, density=0.3, seed=0, indptr_dtype=torch.int64):
    gen = torch.Generator().manual_seed(seed)
    mask = torch.rand(n_rows, n_cols, generator=gen) < density
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64)
    indptr[1:] = torch.cumsum(mask.sum(1), 0)
    indices = mask.nonzero()[:, 1].to(torch.int32)
    return (
        indices.to(DEV),
        indptr.to(indptr_dtype).to(DEV),
        int(indices.numel()),
    )


def _fast(indices, indptr, n_rows, n_cols, nnz):
    return sddmm._iluvatar_sddmm_csr_pattern_is_valid(
        indices, indptr.to(torch.int64), n_rows, n_cols, nnz
    )


@pytest.mark.parametrize(
    "n_rows,n_cols",
    [(1, 1), (5, 7), (255, 9), (256, 9), (257, 9), (600, 300), (3000, 50)],
)
def test_valid_patterns_pass_including_block_boundaries(n_rows, n_cols):
    indices, indptr, nnz = _pattern(n_rows, n_cols)
    assert _fast(indices, indptr, n_rows, n_cols, nnz)


def test_empty_patterns_pass():
    empty = torch.zeros(0, dtype=torch.int32, device=DEV)
    assert _fast(empty, torch.zeros(5, dtype=torch.int64, device=DEV), 4, 6, 0)
    assert _fast(empty, torch.zeros(1, dtype=torch.int64, device=DEV), 0, 6, 0)


def _corrupt(kind, n_rows=700, n_cols=40):
    indices, indptr, nnz = _pattern(n_rows, n_cols, 0.2, seed=3)
    indices, indptr = indices.clone(), indptr.clone()
    if kind == "ptr0":
        indptr[0] = 1
    elif kind == "ptr_last_big":
        indptr[-1] = nnz + 1
    elif kind == "ptr_last_small":
        indptr[-1] = nnz - 1
    elif kind == "decreasing_mid":
        indptr[n_rows // 2] = indptr[n_rows // 2 + 1] + 1
    elif kind == "decreasing_last_block":
        indptr[n_rows - 1] = indptr[n_rows] + 1
    elif kind == "col_negative":
        indices[nnz // 2] = -1
    elif kind == "col_too_big_last":
        indices[nnz - 1] = n_cols
    elif kind == "col_too_big_first":
        indices[0] = n_cols + 5
    return indices, indptr, nnz, n_rows, n_cols


KINDS = [
    "ptr0",
    "ptr_last_big",
    "ptr_last_small",
    "decreasing_mid",
    "decreasing_last_block",
    "col_negative",
    "col_too_big_last",
    "col_too_big_first",
]


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_of_violation_is_caught(kind):
    indices, indptr, nnz, n_rows, n_cols = _corrupt(kind)
    assert not _fast(indices, indptr, n_rows, n_cols, nnz)


@pytest.mark.parametrize(
    "kind,error",
    [
        ("ptr0", ValueError),
        ("ptr_last_big", ValueError),
        ("decreasing_mid", ValueError),
        ("col_negative", IndexError),
        ("col_too_big_last", IndexError),
    ],
)
def test_invalid_input_still_raises_the_generic_error(monkeypatch, kind, error):
    # The fast path only ever accepts; a rejected pattern reaches the specific checks.
    monkeypatch.setattr(sddmm, "_is_iluvatar_runtime", lambda: True)
    indices, indptr, nnz, n_rows, n_cols = _corrupt(kind)
    with pytest.raises(error):
        sddmm._prepare_sddmm_csr_pattern(indices, indptr, (n_rows, n_cols))


@pytest.mark.parametrize("indptr_dtype", [torch.int32, torch.int64])
def test_valid_input_is_returned_unchanged(monkeypatch, indptr_dtype):
    monkeypatch.setattr(sddmm, "_is_iluvatar_runtime", lambda: True)
    indices, indptr, _ = _pattern(500, 60, indptr_dtype=indptr_dtype)
    got_indices, got_indptr, shape = sddmm._prepare_sddmm_csr_pattern(
        indices, indptr, (500, 60)
    )
    assert shape == (500, 60)
    assert got_indptr.dtype == torch.int64
    assert torch.equal(got_indices, indices)


def test_the_fast_path_is_only_taken_on_iluvatar(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sddmm,
        "_iluvatar_sddmm_csr_pattern_is_valid",
        lambda *a: calls.append(a) or True,
    )
    indices, indptr, _ = _pattern(50, 20)
    monkeypatch.setattr(sddmm, "_is_iluvatar_runtime", lambda: False)
    sddmm._prepare_sddmm_csr_pattern(indices, indptr, (50, 20))
    assert not calls
    monkeypatch.setattr(sddmm, "_is_iluvatar_runtime", lambda: True)
    sddmm._prepare_sddmm_csr_pattern(indices, indptr, (50, 20))
    assert len(calls) == 1
