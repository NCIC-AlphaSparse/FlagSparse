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

from tests.pytest.accuracy_utils import ACCELERATOR_REQUIRED, accelerator_available, accelerator_device

from flagsparse import (
    FlagSparseSpSVDescr,
    flagsparse_spsv_analysis_coo,
    flagsparse_spsv_coo,
    flagsparse_spsv_create_workspace,
    flagsparse_spsv_preprocess_coo,
    flagsparse_spsv_solve_coo,
)
import flagsparse.sparse_operations.spsv as fs_spsv_impl
from flagsparse.sparse_operations import _common as common
from tests import reference_utils

from tests.pytest.param_shapes import SPSV_N
from tests.pytest.accuracy_utils import golden_device
from tests.pytest.test_spsv_csr_accuracy import (
    _apply_ref_op,
    _build_triangular,
    _dense_ref_spsv,
    _dtype_id,
    _effective_upper,
    _rand_like,
    _tol,
    _transpose_arg,
    NON_TRANS_DTYPES,
    SUPPORTED_COMPLEX_DTYPES,
    TRANS_CONJ_DTYPES,
    TRANS_CONJ_MODES,
)

pytestmark = [
    pytest.mark.skipif(not accelerator_available(), reason=ACCELERATOR_REQUIRED),
    pytest.mark.spsv_coo,
]


def _solve_triangular(A, B, *, upper, unitriangular=False):
    """Use the CPU SciPy triangular oracle for XPU correctness checks."""
    if common._use_scipy_accuracy_reference():
        matrix = A.to_sparse_csr()
        scipy_matrix = reference_utils.scipy_csr(
            matrix.values(), matrix.col_indices(), matrix.crow_indices(), A.shape,
            reference_utils.reference_dtype(A.dtype),
        )
        solved = reference_utils.triangular_solve(
            scipy_matrix, B.squeeze(-1), reference_utils.reference_dtype(B.dtype),
            lower=not upper, unit_diagonal=unitriangular,
        )
        # Back to the operand dtype: the oracle solves at the reference dtype,
        # and allclose() refuses to compare float32 against float64.
        return (
            reference_utils.as_torch(
                solved, reference_utils.reference_dtype(B.dtype), golden_device()
            )
            .to(B.dtype)
            .unsqueeze(-1)
        )
    return torch.linalg.solve_triangular(A, B, upper=upper, unitriangular=unitriangular)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("op_mode", TRANS_CONJ_MODES)
def test_spsv_coo_transpose_family_complex128_routes_through_csr(n, op_mode):
    device = accelerator_device()
    dtype = torch.complex128
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())
    x_ref = _solve_triangular(
        _apply_ref_op(A, op_mode),
        b.unsqueeze(-1),
        upper=_effective_upper(True, op_mode),
    ).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=_transpose_arg(op_mode),
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", TRANS_CONJ_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
def test_spsv_coo_trans_supported_combos(n, dtype, index_dtype):
    device = accelerator_device()
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())
    A_ref = A.to(dtype)
    b_ref = b.to(dtype)
    x_ref = _solve_triangular(
        _apply_ref_op(A_ref, "TRANS"),
        b_ref.unsqueeze(-1),
        upper=_effective_upper(True, "TRANS"),
    ).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=_transpose_arg("TRANS"),
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", TRANS_CONJ_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
def test_spsv_coo_upper_trans_supported_combos(n, dtype, index_dtype):
    device = accelerator_device()
    A = _build_triangular(n, dtype, golden_device(), lower=False)
    b = _rand_like(dtype, (n,), golden_device())
    A_ref = A.to(dtype)
    b_ref = b.to(dtype)
    x_ref = _solve_triangular(
        _apply_ref_op(A_ref, "TRANS"),
        b_ref.unsqueeze(-1),
        upper=_effective_upper(False, "TRANS"),
    ).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=False,
        unit_diagonal=False,
        transpose=_transpose_arg("TRANS"),
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", TRANS_CONJ_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
def test_spsv_coo_conj_supported_combos(n, dtype, index_dtype):
    device = accelerator_device()
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())
    A_ref = A.to(dtype)
    b_ref = b.to(dtype)
    x_ref = _solve_triangular(
        _apply_ref_op(A_ref, "CONJ"),
        b_ref.unsqueeze(-1),
        upper=_effective_upper(True, "CONJ"),
    ).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=_transpose_arg("CONJ"),
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", TRANS_CONJ_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
def test_spsv_coo_upper_conj_supported_combos(n, dtype, index_dtype):
    device = accelerator_device()
    A = _build_triangular(n, dtype, golden_device(), lower=False)
    b = _rand_like(dtype, (n,), golden_device())
    A_ref = A.to(dtype)
    b_ref = b.to(dtype)
    x_ref = _solve_triangular(
        _apply_ref_op(A_ref, "CONJ"),
        b_ref.unsqueeze(-1),
        upper=_effective_upper(False, "CONJ"),
    ).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=False,
        unit_diagonal=False,
        transpose=_transpose_arg("CONJ"),
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
def test_spsv_coo_non_trans_supported_combos(n, dtype, index_dtype):
    device = accelerator_device()
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())
    x_ref = _solve_triangular(
        A.to(dtype), b.to(dtype).unsqueeze(-1), upper=False
    ).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
def test_spsv_coo_non_trans_upper_supported_combos(n, dtype, index_dtype):
    device = accelerator_device()
    A = _build_triangular(n, dtype, golden_device(), lower=False)
    b = _rand_like(dtype, (n,), golden_device())
    x_ref = _solve_triangular(
        A.to(dtype), b.to(dtype).unsqueeze(-1), upper=True
    ).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=False,
        unit_diagonal=False,
        transpose=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_non_trans_routes_through_csr(monkeypatch):
    device = accelerator_device()
    dtype = torch.complex64
    index_dtype = torch.int64
    n = SPSV_N[0]
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())
    x_ref = _solve_triangular(A, b.unsqueeze(-1), upper=False).squeeze(-1)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    called = {"hit": False}
    real_csr_impl = fs_spsv_impl.flagsparse_spsv_csr

    def _wrapped_flagsparse_spsv_csr(*args, **kwargs):
        called["hit"] = True
        return real_csr_impl(*args, **kwargs)

    monkeypatch.setattr(
        fs_spsv_impl, "flagsparse_spsv_csr", _wrapped_flagsparse_spsv_csr
    )

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
    )

    rtol, atol = _tol(dtype)
    assert called["hit"]
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_to_csr_keeps_duplicates():
    device = accelerator_device()
    data = torch.tensor([1.0, 2.0, 3.0], device=device)
    row = torch.tensor([0, 0, 1], dtype=torch.int64, device=device)
    col = torch.tensor([1, 1, 0], dtype=torch.int64, device=device)

    data_csr, indices_csr, indptr_csr = fs_spsv_impl._coo2csr_for_spsv(
        data, row, col, 2, assume_ordered=False
    )

    assert data_csr.numel() == 3
    assert indices_csr.tolist() == [1, 1, 0]
    assert indptr_csr.tolist() == [0, 2, 3]


@pytest.mark.spsv
def test_spsv_coo_analysis_workspace_solve_matches_direct():
    device = accelerator_device()
    dtype = torch.complex64
    n = SPSV_N[0]
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    descr = flagsparse_spsv_analysis_coo(
        data,
        row.to(torch.int64),
        col.to(torch.int64),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
    )
    assert isinstance(descr, FlagSparseSpSVDescr)
    assert descr.format == "coo"
    assert descr.canonical_format == "csr"

    workspace = flagsparse_spsv_create_workspace(descr)
    x_via_descr = flagsparse_spsv_solve_coo(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_coo(
        data,
        row.to(torch.int64),
        col.to(torch.int64),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_explicit_roc_route_matches_dense():
    device = accelerator_device()
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_roc",
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("dtype", SUPPORTED_COMPLEX_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("solve_kind", ["csr_roc", "csr_cw_levelschd", "alg2"])
def test_spsv_coo_explicit_complex_level_routes_match_dense(dtype, solve_kind):
    device = accelerator_device()
    n = 64
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind=solve_kind,
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("dtype", SUPPORTED_COMPLEX_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("solve_kind", ["csr_nnz_balance", "alg3"])
def test_spsv_coo_explicit_complex_nnz_balance_routes_match_dense(dtype, solve_kind):
    device = accelerator_device()
    n = 64
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind=solve_kind,
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_explicit_levelschd_route_matches_dense():
    device = accelerator_device()
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_cw_levelschd",
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_explicit_nnz_balance_route_matches_dense():
    device = accelerator_device()
    dtype = torch.float64
    n = 96
    A = torch.tril(torch.randn(n, n, dtype=dtype, device=device) * 0.02)
    A = A + torch.eye(n, dtype=dtype, device=device) * 3.0
    b = _rand_like(dtype, (n,), golden_device())

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_nnz_balance",
    )
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=True, unit_diagonal=False)
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_roc_analysis_workspace_solve_matches_direct():
    device = accelerator_device()
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    descr = flagsparse_spsv_analysis_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_roc",
    )
    assert descr.solve_kind == "csr_roc"
    assert descr.canonical_format == "csr"
    workspace = flagsparse_spsv_preprocess_coo(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_coo(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_roc",
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_levelschd_analysis_workspace_solve_matches_direct():
    device = accelerator_device()
    dtype = torch.float64
    n = 64
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    descr = flagsparse_spsv_analysis_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_cw_levelschd",
    )
    assert descr.solve_kind == "csr_cw_levelschd"
    assert descr.canonical_format == "csr"
    workspace = flagsparse_spsv_preprocess_coo(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_coo(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_cw_levelschd",
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
def test_spsv_coo_nnz_balance_analysis_workspace_solve_matches_direct():
    device = accelerator_device()
    dtype = torch.float64
    n = 96
    A = torch.tril(torch.randn(n, n, dtype=dtype, device=device) * 0.02)
    A = A + torch.eye(n, dtype=dtype, device=device) * 3.0
    b = _rand_like(dtype, (n,), golden_device())

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    descr = flagsparse_spsv_analysis_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_nnz_balance",
    )
    assert descr.solve_kind == "csr_nnz_balance"
    assert descr.canonical_format == "csr"
    workspace = flagsparse_spsv_preprocess_coo(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_coo(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind="csr_nnz_balance",
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("dtype", SUPPORTED_COMPLEX_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("solve_kind", ["csr_nnz_balance", "alg8"])
def test_spsv_coo_complex_nnz_balance_analysis_workspace_matches_direct(
    dtype, solve_kind
):
    device = accelerator_device()
    n = 64
    A = _build_triangular(n, dtype, golden_device(), lower=True)
    b = _rand_like(dtype, (n,), golden_device())

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    descr = flagsparse_spsv_analysis_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind=solve_kind,
    )
    assert descr.solve_kind == "csr_nnz_balance"
    assert descr.canonical_format == "csr"
    workspace = flagsparse_spsv_preprocess_coo(
        descr, workspace=flagsparse_spsv_create_workspace(descr)
    )
    x_via_descr = flagsparse_spsv_solve_coo(descr, b, workspace=workspace)
    x_direct = flagsparse_spsv_coo(
        data,
        row.to(torch.int32),
        col.to(torch.int32),
        b,
        (n, n),
        lower=True,
        unit_diagonal=False,
        transpose=False,
        solve_kind=solve_kind,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x_via_descr, x_direct, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_spsv_coo_non_trans_unit_supported_combos(n, dtype, index_dtype, lower):
    device = accelerator_device()
    A = _build_triangular(n, dtype, golden_device(), lower=lower)
    b = _rand_like(dtype, (n,), golden_device())
    x_ref = _dense_ref_spsv(A.to(dtype), b.to(dtype), lower=lower, unit_diagonal=True)

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=lower,
        unit_diagonal=True,
        transpose=False,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)


@pytest.mark.spsv
@pytest.mark.parametrize("n", SPSV_N)
@pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize(
    "index_dtype", [torch.int32, torch.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize("op_mode", TRANS_CONJ_MODES)
def test_spsv_coo_unit_transpose_family_supported_combos(
    n, dtype, index_dtype, lower, op_mode
):
    device = accelerator_device()
    A = _build_triangular(n, dtype, golden_device(), lower=lower)
    b = _rand_like(dtype, (n,), golden_device())
    x_ref = _dense_ref_spsv(
        A.to(dtype),
        b.to(dtype),
        lower=lower,
        op_mode=op_mode,
        unit_diagonal=True,
    )

    A_coo = A.to_sparse_coo().coalesce()
    row, col = A_coo.indices().to(device)
    data = A_coo.values().clone().to(device)
    b = b.to(device)

    x = flagsparse_spsv_coo(
        data,
        row.to(index_dtype),
        col.to(index_dtype),
        b,
        (n, n),
        lower=lower,
        unit_diagonal=True,
        transpose=_transpose_arg(op_mode),
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(x.to(x_ref.device), x_ref, rtol=rtol, atol=atol)
