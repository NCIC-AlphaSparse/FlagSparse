# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SDDMM has no vendor baseline on Iluvatar, so the H800 scaled baseline supplies it.

With ``FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse`` (the right choice for CSR SpMV/SpMM,
where torch.sparse silently returns zeros) the SDDMM benchmark's "cusparse" column is
really ``torch.sparse.sampled_addmm``, which cannot be trusted on BI-V150. If it were
reported as a baseline and returned zeros, every SDDMM row would fail on ``cu_status``
and the scaled baseline would never be consulted. So Iluvatar reports "no baseline".
"""

import importlib

import pytest

torch = pytest.importorskip(
    "torch", reason="tests/ci runs on a CPU-only runner without torch"
)

# `flagsparse.sparse_operations.sddmm_csr` is also the name of the public function.
sddmm = importlib.import_module("flagsparse.sparse_operations.sddmm_csr")

F32, I32 = torch.float32, torch.int32


def _select(monkeypatch, *, vendor, iluvatar):
    monkeypatch.setattr(sddmm, "_expected_vendor_sparse_backend", lambda: vendor)
    monkeypatch.setattr(sddmm, "_is_iluvatar_runtime", lambda: iluvatar)
    monkeypatch.setattr(sddmm, "_is_ascend_runtime", lambda: False)


def test_iluvatar_with_cupy_cusparse_reports_no_sddmm_baseline(monkeypatch):
    _select(monkeypatch, vendor="cupy_cusparse", iluvatar=True)
    backend, reason = sddmm._sddmm_csr_sparse_ref_backend(F32, I32)
    assert backend is None
    assert "no verified" in reason


def test_no_baseline_holds_for_every_dtype_the_iluvatar_run_can_ask_for(monkeypatch):
    _select(monkeypatch, vendor="cupy_cusparse", iluvatar=True)
    for dtype in (torch.float16, torch.float32):
        for index in (torch.int32, torch.int64):
            assert sddmm._sddmm_csr_sparse_ref_backend(dtype, index)[0] is None


def test_other_cuda_compatible_backends_keep_their_cusparse_baseline(monkeypatch):
    # CUDA (and MetaX with CuPy) still time and cross-check the vendor SDDMM.
    _select(monkeypatch, vendor="cupy_cusparse", iluvatar=False)
    assert sddmm._sddmm_csr_sparse_ref_backend(F32, I32) == ("cupy_cusparse", None)


def test_iluvatar_with_torch_is_still_reported_as_not_wired(monkeypatch):
    _select(monkeypatch, vendor="torch", iluvatar=True)
    backend, reason = sddmm._sddmm_csr_sparse_ref_backend(F32, I32)
    assert backend is None
    assert "not wired" in reason
