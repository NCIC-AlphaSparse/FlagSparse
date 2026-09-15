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

import pytest
import torch

from flagsparse import flagsparse_spsm_coo, flagsparse_spsm_csr

from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
    close_tolerances,
    golden_device,
)
from tests.pytest.param_shapes import SPSM_N_RHS

pytestmark = pytest.mark.skipif(not accelerator_available(), reason=ACCELERATOR_REQUIRED)


SPSM_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
SPSM_DTYPE_IDS = ("float32", "float64", "complex64", "complex128")


def _build_triangular_dense(n, dtype, device, lower, unit_diagonal):
    """Dense triangular matrix on ``device``.

    Tests pass ``golden_device()``: the reference is ``torch.linalg.solve_triangular``
    and the sparse conversions below are ``to_sparse_csr``/``to_sparse_coo``, none of
    which is dependable on MUSA.  Only the operator's inputs are copied to the
    accelerator.  See ``accuracy_utils.golden_device()``.
    """
    base = torch.randn(n, n, dtype=dtype, device=device) * 0.02
    base = torch.tril(base) if lower else torch.triu(base)
    eye = torch.eye(n, dtype=dtype, device=device)
    if unit_diagonal:
        return base * (1 - eye) + eye
    return base + eye * (float(n) * 0.5 + 2.0)


def _tol(dtype):
    return close_tolerances(dtype)


@pytest.mark.spsm
@pytest.mark.spsm_csr
@pytest.mark.parametrize("n, n_rhs", SPSM_N_RHS)
@pytest.mark.parametrize("dtype", SPSM_DTYPES, ids=SPSM_DTYPE_IDS)
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize(
    "unit_diagonal", [False, True], ids=["explicit_diag", "unit_diag"]
)
def test_spsm_csr_matches_dense(n, n_rhs, dtype, lower, unit_diagonal):
    device = accelerator_device()
    A = _build_triangular_dense(n, dtype, golden_device(), lower, unit_diagonal)
    B = torch.randn(n, n_rhs, dtype=dtype, device=golden_device())
    ref = torch.linalg.solve_triangular(
        A,
        B,
        upper=not lower,
        unitriangular=unit_diagonal,
    )
    Acsr = A.to_sparse_csr()
    out = flagsparse_spsm_csr(
        Acsr.values().to(device),
        Acsr.col_indices().to(torch.int32).to(device),
        Acsr.crow_indices().to(torch.int32).to(device),
        B.to(device),
        (n, n),
        lower=lower,
        unit_diagonal=unit_diagonal,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(out.to(ref.device), ref, rtol=rtol, atol=atol)


@pytest.mark.spsm
@pytest.mark.spsm_coo
@pytest.mark.parametrize("n, n_rhs", SPSM_N_RHS)
@pytest.mark.parametrize("dtype", SPSM_DTYPES, ids=SPSM_DTYPE_IDS)
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize(
    "unit_diagonal", [False, True], ids=["explicit_diag", "unit_diag"]
)
def test_spsm_coo_matches_dense(n, n_rhs, dtype, lower, unit_diagonal):
    device = accelerator_device()
    A = _build_triangular_dense(n, dtype, golden_device(), lower, unit_diagonal)
    B = torch.randn(n, n_rhs, dtype=dtype, device=golden_device())
    ref = torch.linalg.solve_triangular(
        A,
        B,
        upper=not lower,
        unitriangular=unit_diagonal,
    )
    Acoo = A.to_sparse_coo().coalesce()
    indices = Acoo.indices()
    out = flagsparse_spsm_coo(
        Acoo.values().to(device),
        indices[0].to(torch.int32).contiguous().to(device),
        indices[1].to(torch.int32).contiguous().to(device),
        B.to(device),
        (n, n),
        lower=lower,
        unit_diagonal=unit_diagonal,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(out.to(ref.device), ref, rtol=rtol, atol=atol)


@pytest.mark.spsm
@pytest.mark.spsm_csr
def test_spsm_csr_rejects_unsupported_index_dtype():
    device = accelerator_device()
    n = 8
    A = _build_triangular_dense(
        n, torch.float32, golden_device(), lower=True, unit_diagonal=False
    )
    B = torch.randn(n, 4, dtype=torch.float32, device=golden_device())
    Acsr = A.to_sparse_csr()
    with pytest.raises(TypeError, match="indices dtype must be torch.int32"):
        flagsparse_spsm_csr(
            Acsr.values().to(device),
            Acsr.col_indices().to(torch.int64).to(device),
            Acsr.crow_indices().to(torch.int64).to(device),
            B.to(device),
            (n, n),
        )


@pytest.mark.spsm
@pytest.mark.spsm_coo
def test_spsm_coo_rejects_unsupported_index_dtype():
    device = accelerator_device()
    n = 8
    A = _build_triangular_dense(
        n, torch.float32, golden_device(), lower=True, unit_diagonal=False
    )
    B = torch.randn(n, 4, dtype=torch.float32, device=golden_device())
    Acoo = A.to_sparse_coo().coalesce()
    indices = Acoo.indices()
    with pytest.raises(TypeError, match="row/col dtype must be torch.int32"):
        flagsparse_spsm_coo(
            Acoo.values().to(device),
            indices[0].to(torch.int64).contiguous().to(device),
            indices[1].to(torch.int64).contiguous().to(device),
            B.to(device),
            (n, n),
        )


@pytest.mark.spsm
@pytest.mark.spsm_csr
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"opA": "TRANS"}, "Only op\\(A\\)=NON_TRANS is supported"),
        ({"opB": "TRANS"}, "Only op\\(B\\)=NON_TRANS is supported"),
        ({"major": "col"}, "Only row-major dense layout is supported"),
    ],
    ids=["opA_trans", "opB_trans", "col_major"],
)
def test_spsm_csr_rejects_unsupported_ops_and_layout(kwargs, match):
    device = accelerator_device()
    n = 8
    A = _build_triangular_dense(
        n, torch.float32, golden_device(), lower=True, unit_diagonal=False
    )
    B = torch.randn(n, 4, dtype=torch.float32, device=golden_device())
    Acsr = A.to_sparse_csr()
    with pytest.raises(NotImplementedError, match=match):
        flagsparse_spsm_csr(
            Acsr.values().to(device),
            Acsr.col_indices().to(torch.int32).to(device),
            Acsr.crow_indices().to(torch.int32).to(device),
            B.to(device),
            (n, n),
            **kwargs,
        )


@pytest.mark.spsm
@pytest.mark.spsm_coo
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"opA": "TRANS"}, "Only op\\(A\\)=NON_TRANS is supported"),
        ({"opB": "TRANS"}, "Only op\\(B\\)=NON_TRANS is supported"),
        ({"major": "col"}, "Only row-major dense layout is supported"),
    ],
    ids=["opA_trans", "opB_trans", "col_major"],
)
def test_spsm_coo_rejects_unsupported_ops_and_layout(kwargs, match):
    device = accelerator_device()
    n = 8
    A = _build_triangular_dense(
        n, torch.float32, golden_device(), lower=True, unit_diagonal=False
    )
    B = torch.randn(n, 4, dtype=torch.float32, device=golden_device())
    Acoo = A.to_sparse_coo().coalesce()
    indices = Acoo.indices()
    with pytest.raises(NotImplementedError, match=match):
        flagsparse_spsm_coo(
            Acoo.values().to(device),
            indices[0].to(torch.int32).contiguous().to(device),
            indices[1].to(torch.int32).contiguous().to(device),
            B.to(device),
            (n, n),
            **kwargs,
        )
