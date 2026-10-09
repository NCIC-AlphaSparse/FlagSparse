# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU simulation coverage for the Ascend COO SpMV fallback."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="requires torch to exercise dispatch")

from flagsparse.sparse_operations import spmv_coo as spmv_coo_ops  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128, torch.complex64])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_ascend_index_add_fallback_preserves_coo_operations(monkeypatch, dtype, op):
    monkeypatch.setattr(spmv_coo_ops, "_is_ascend_runtime", lambda: True)
    monkeypatch.setattr(
        spmv_coo_ops, "_is_accel_tensor", lambda tensor: tensor.device.type == "cpu"
    )
    monkeypatch.setattr(
        spmv_coo_ops, "_ACCEL", SimpleNamespace(synchronize=lambda: None)
    )

    # A nonzero imaginary part is required for the "conj" case to actually
    # exercise conjugation: conj(a + 0j) == a + 0j, so an all-real complex
    # tensor cannot distinguish "conj" from "trans" and would pass even with a
    # missing/incorrect conjugation (this previously let a real bug in
    # flagsparse_spmv_coo's Ascend conj path through undetected).
    real = torch.tensor([2, -1, 3, 4], dtype=torch.float64)
    x_real = torch.tensor([1, 2, -3], dtype=torch.float64)
    if dtype.is_complex:
        imag = torch.tensor([1, 2, -1, 3], dtype=torch.float64)
        x_imag = torch.tensor([-2, 1, 2], dtype=torch.float64)
        values = torch.complex(real, imag).to(dtype)
        x = torch.complex(x_real, x_imag).to(dtype)
    else:
        values = real.to(dtype)
        x = x_real.to(dtype)
    row = torch.tensor([0, 0, 1, 2], dtype=torch.int32)
    col = torch.tensor([0, 2, 1, 0], dtype=torch.int32)
    dense = torch.zeros((3, 3), dtype=dtype)
    dense.index_put_((row.to(torch.int64), col.to(torch.int64)), values)

    if op == "non":
        expected = dense @ x
    elif op == "trans":
        expected = dense.t() @ x
    else:
        expected = dense.conj().t() @ x

    out = spmv_coo_ops.flagsparse_spmv_coo(values, row, col, x, shape=(3, 3), op=op)

    torch.testing.assert_close(out, expected)
