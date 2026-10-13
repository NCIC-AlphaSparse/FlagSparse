# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU simulation coverage for Ascend CSC and SpGEMM fallbacks."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from flagsparse.sparse_operations import sddmm_csr as sddmm_ops  # noqa: E402
from flagsparse.sparse_operations import spgemm_csr as spgemm_ops  # noqa: E402
from flagsparse.sparse_operations import spmm_csc as spmm_csc_ops  # noqa: E402
from flagsparse.sparse_operations import spmv_csc as spmv_csc_ops  # noqa: E402


def _no_gpu_backend_info(device=None):
    # What _get_device_backend_info reports with no CUDA device. Without this,
    # the CPU tensors below make it query CUDA properties on a GPU box and fail
    # with "Expected a cuda device, but got: cpu".
    return {
        "backend": "unavailable",
        "device_name": "",
        "device_warp_size": 32,
        "max_threads_per_block": 1024,
        "max_threads_per_multi_processor": 0,
        "multi_processor_count": 0,
    }


def _patch_ascend(monkeypatch, module):
    monkeypatch.setattr(module, "_is_ascend_runtime", lambda: True)
    if hasattr(module, "_get_device_backend_info"):
        monkeypatch.setattr(module, "_get_device_backend_info", _no_gpu_backend_info)
    monkeypatch.setattr(
        module, "_is_accel_tensor", lambda tensor: tensor.device.type == "cpu"
    )
    monkeypatch.setattr(module, "_ACCEL", SimpleNamespace(synchronize=lambda: None))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_ascend_csc_spmv_fallback(monkeypatch, dtype, op):
    _patch_ascend(monkeypatch, spmv_csc_ops)
    values = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    if dtype.is_complex:
        values = torch.complex(
            values, torch.tensor([1, -2, 2, -1], dtype=torch.float32)
        ).to(dtype)
    else:
        values = values.to(dtype)
    rows = torch.tensor([0, 1, 0, 2], dtype=torch.int32)
    ptr = torch.tensor([0, 2, 3, 4], dtype=torch.int32)
    x = (torch.arange(3, dtype=torch.float32) + 1).to(dtype)
    out = spmv_csc_ops.flagsparse_spmv_csc(values, rows, ptr, x, (3, 3), op=op)
    dense = torch.zeros((3, 3), dtype=dtype)
    cols = torch.repeat_interleave(torch.arange(3), torch.tensor([2, 1, 1]))
    dense[rows.long(), cols] = values
    expected = (
        dense @ x
        if op == "non"
        else (dense.t() if op == "trans" else dense.conj().t()) @ x
    )
    torch.testing.assert_close(out, expected)


def test_ascend_csc_spmm_and_spgemm_fallback(monkeypatch):
    _patch_ascend(monkeypatch, spmm_csc_ops)
    _patch_ascend(monkeypatch, spgemm_ops)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rows = torch.tensor([0, 1, 0, 2], dtype=torch.int32)
    ptr = torch.tensor([0, 2, 3, 4], dtype=torch.int32)
    B = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    out = spmm_csc_ops.flagsparse_spmm_csc(values, rows, ptr, B, (3, 3), op="non")
    dense = torch.zeros((3, 3))
    dense[
        rows.long(), torch.repeat_interleave(torch.arange(3), torch.tensor([2, 1, 1]))
    ] = values
    torch.testing.assert_close(out, dense @ B)

    a_data = torch.tensor([1.0, 2.0, 3.0])
    a_idx = torch.tensor([0, 1, 1], dtype=torch.int32)
    a_ptr = torch.tensor([0, 2, 3], dtype=torch.int64)
    b_data = torch.tensor([4.0, 5.0, 6.0])
    b_idx = torch.tensor([0, 1, 1], dtype=torch.int32)
    b_ptr = torch.tensor([0, 1, 3], dtype=torch.int64)
    result = spgemm_ops.flagsparse_spgemm_csr(
        a_data, a_idx, a_ptr, (2, 2), b_data, b_idx, b_ptr, (2, 2)
    )
    a_dense = torch.zeros((2, 2))
    a_dense.index_put_((torch.tensor([0, 0, 1]), a_idx.long()), a_data, accumulate=True)
    b_dense = torch.zeros((2, 2))
    b_dense.index_put_((torch.tensor([0, 1, 1]), b_idx.long()), b_data, accumulate=True)
    expected = a_dense @ b_dense
    got = torch.zeros((2, 2))
    row_ids = torch.repeat_interleave(torch.arange(2), result[2][1:] - result[2][:-1])
    got[row_ids, result[1].long()] = result[0]
    torch.testing.assert_close(got, expected)


def test_ascend_sddmm_complex_uses_component_safe_gather(monkeypatch):
    _patch_ascend(monkeypatch, sddmm_ops)
    values = torch.zeros(4, dtype=torch.complex64)
    rows = torch.tensor([0, 1, 0, 2], dtype=torch.int32)
    ptr = torch.tensor([0, 2, 3, 4], dtype=torch.int32)
    x = torch.complex(
        torch.arange(9, dtype=torch.float32).reshape(3, 3), torch.ones(3, 3)
    )
    y = torch.complex(
        torch.arange(9, dtype=torch.float32).reshape(3, 3) + 1, torch.ones(3, 3) * 2
    )
    out = sddmm_ops.flagsparse_sddmm_csr(values, rows, ptr, x, y, shape=(3, 3))
    row_ids = torch.tensor([0, 0, 1, 2])
    expected = torch.sum(x[row_ids] * y[rows.long()], dim=1)
    torch.testing.assert_close(out, expected)
