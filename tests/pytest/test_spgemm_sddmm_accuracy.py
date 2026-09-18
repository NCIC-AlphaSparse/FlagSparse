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

from flagsparse import flagsparse_sddmm_csr, flagsparse_spgemm_csr
from flagsparse.sparse_operations import _common as common
from tests import reference_utils

from tests.pytest.accuracy_utils import (
    ACCELERATOR_REQUIRED,
    accelerator_available,
    accelerator_device,
    close_tolerances,
    golden_device,
)
from tests.pytest.param_shapes import (
    SDDMM_DTYPES,
    SDDMM_DTYPE_IDS,
    SDDMM_MNK_SHAPES,
    SPGEMM_DTYPES,
    SPGEMM_DTYPE_IDS,
    SPGEMM_MNK_SHAPES,
)

pytestmark = pytest.mark.skipif(not accelerator_available(), reason=ACCELERATOR_REQUIRED)

_SYNTHETIC_VALUE_SCALE = 0.125


def _random_csr(rows, cols, dtype, device, value_scale=_SYNTHETIC_VALUE_SCALE):
    """Random sparse CSR matrix on ``device``.

    Tests that build a reference pass ``golden_device()``: the SpGEMM reference is
    ``torch.sparse.mm`` (no working implementation on MUSA) and the SDDMM reference
    is advanced indexing (no complex kernel there).  See ``golden_device()``.
    """
    denom = max(rows * cols, 1)
    p = min(0.25, max(0.06, 32.0 / denom))
    mask = torch.rand(rows, cols, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    vals = (
        torch.randn(rows, cols, dtype=dtype, device=device)
        * value_scale
        * mask.to(dtype=dtype)
    )
    return vals.to_sparse_csr()


def _csr_to_dense(data, indices, indptr, shape):
    # torch_musa can store CSR tensors but does not register SparseCsrmusa
    # ``to_dense``. The result is an oracle artifact, so materialize it on the
    # CPU rather than exercising a second vendor sparse implementation.
    csr = torch.sparse_csr_tensor(
        indptr.cpu(),
        indices.cpu(),
        data.cpu(),
        size=shape,
        dtype=data.dtype,
        device=golden_device(),
    )
    return csr.to_dense()


def _tol(dtype):
    return close_tolerances(dtype)


@pytest.mark.spgemm_csr
@pytest.mark.parametrize("M, N, K", SPGEMM_MNK_SHAPES)
@pytest.mark.parametrize("dtype", SPGEMM_DTYPES, ids=SPGEMM_DTYPE_IDS)
@pytest.mark.parametrize(
    "indptr_dtype", [torch.int32, torch.int64], ids=["ptr32", "ptr64"]
)
def test_spgemm_csr_matches_torch(M, N, K, dtype, indptr_dtype):
    device = accelerator_device()
    golden = golden_device()
    A = _random_csr(M, K, dtype, golden)
    B = _random_csr(K, N, dtype, golden)
    if common._use_scipy_accuracy_reference():
        left = reference_utils.scipy_csr(
            A.values(), A.col_indices(), A.crow_indices(), (M, K), dtype
        )
        right = reference_utils.scipy_csr(
            B.values(), B.col_indices(), B.crow_indices(), (K, N), dtype
        )
        # spgemm() returns a SciPy sparse matrix; densify it before torch sees it.
        ref = reference_utils.as_torch(
            reference_utils.spgemm(left, right).toarray(), dtype, golden
        )
    else:
        ref = torch.sparse.mm(A, B.to_dense())
    c_data, c_indices, c_indptr, c_shape = flagsparse_spgemm_csr(
        A.values().to(device),
        A.col_indices().to(torch.int32).to(device),
        A.crow_indices().to(indptr_dtype).to(device),
        (M, K),
        B.values().to(device),
        B.col_indices().to(torch.int32).to(device),
        B.crow_indices().to(indptr_dtype).to(device),
        (K, N),
    )
    got = _csr_to_dense(c_data, c_indices, c_indptr, c_shape)
    rtol, atol = _tol(dtype)
    assert torch.allclose(got.to(ref.device), ref, rtol=rtol, atol=atol)


@pytest.mark.spgemm_csr
def test_spgemm_csr_rejects_unsupported_index_and_value_dtypes():
    device = accelerator_device()
    # Build the CSR inputs on CPU and upload only the triples: MUSA does not
    # register a dense-to-CSR conversion, so constructing them on the device
    # failed before the dtype check this test exists for was ever reached.
    golden = golden_device()
    A = _random_csr(8, 10, torch.float32, golden)
    B = _random_csr(10, 6, torch.float32, golden)
    with pytest.raises(TypeError, match="a_indices dtype must be torch.int32"):
        flagsparse_spgemm_csr(
            A.values().to(device),
            A.col_indices().to(torch.int64).to(device),
            A.crow_indices().to(device),
            (8, 10),
            B.values().to(device),
            B.col_indices().to(torch.int32).to(device),
            B.crow_indices().to(device),
            (10, 6),
        )

    A_complex = _random_csr(8, 10, torch.complex64, golden)
    with pytest.raises(
        TypeError, match="a_data dtype must be torch.float32 or torch.float64"
    ):
        flagsparse_spgemm_csr(
            A_complex.values().to(device),
            A_complex.col_indices().to(torch.int32).to(device),
            A_complex.crow_indices().to(device),
            (8, 10),
            B.values().to(device),
            B.col_indices().to(torch.int32).to(device),
            B.crow_indices().to(device),
            (10, 6),
        )


@pytest.mark.sddmm_csr
@pytest.mark.parametrize("M, N, K", SDDMM_MNK_SHAPES)
@pytest.mark.parametrize("dtype", SDDMM_DTYPES, ids=SDDMM_DTYPE_IDS)
@pytest.mark.parametrize(
    "indptr_dtype", [torch.int32, torch.int64], ids=["ptr32", "ptr64"]
)
def test_sddmm_csr_matches_sampled_dense_reference(M, N, K, dtype, indptr_dtype):
    device = accelerator_device()
    golden = golden_device()
    pattern = _random_csr(M, N, dtype, golden)
    indices = pattern.col_indices().to(torch.int32)
    indptr = pattern.crow_indices().to(indptr_dtype)
    data = pattern.values()
    x = torch.randn(M, K, dtype=dtype, device=golden) * _SYNTHETIC_VALUE_SCALE
    y = torch.randn(N, K, dtype=dtype, device=golden) * _SYNTHETIC_VALUE_SCALE
    alpha = 1.25
    beta = 0.5

    row_ids = torch.repeat_interleave(
        torch.arange(M, dtype=torch.int64, device=golden),
        indptr[1:] - indptr[:-1],
    )
    if common._use_scipy_accuracy_reference():
        sampled = reference_utils.sddmm_csr_values(indices, indptr, x, y, dtype)
        ref = torch.as_tensor(sampled, dtype=dtype) * alpha + data.cpu() * beta
    else:
        ref = alpha * torch.sum(x[row_ids] * y[indices.to(torch.int64)], dim=1) + beta * data
    got = flagsparse_sddmm_csr(
        data=data.to(device),
        indices=indices.to(device),
        indptr=indptr.to(device),
        x=x.to(device),
        y=y.to(device),
        shape=(M, N),
        alpha=alpha,
        beta=beta,
    )
    rtol, atol = _tol(dtype)
    assert torch.allclose(got.to(ref.device), ref, rtol=rtol, atol=atol)


@pytest.mark.sddmm_csr
def test_sddmm_csr_rejects_unsupported_index_and_value_dtypes():
    device = accelerator_device()
    # Same fix as the SpGEMM sibling above, which it was missed alongside: build
    # the pattern on CPU (MUSA registers no aten::_to_sparse_csr) and upload
    # only the triples the operator actually takes.
    pattern = _random_csr(8, 10, torch.float32, golden_device())
    data = pattern.values().to(device)
    indptr = pattern.crow_indices().to(device)
    col_indices = pattern.col_indices().to(device)
    x = torch.randn(8, 4, dtype=torch.float32, device=device)
    y = torch.randn(10, 4, dtype=torch.float32, device=device)
    with pytest.raises(TypeError, match="indices dtype must be torch.int32"):
        flagsparse_sddmm_csr(
            data=data,
            indices=col_indices.to(torch.int64),
            indptr=indptr,
            x=x,
            y=y,
            shape=(8, 10),
        )

    data_complex = data.to(torch.complex64)
    with pytest.raises(
        TypeError, match="x dtype must be torch.float32 or torch.float64"
    ):
        flagsparse_sddmm_csr(
            data=data_complex,
            indices=col_indices.to(torch.int32),
            indptr=indptr,
            x=x.to(torch.complex64),
            y=y.to(torch.complex64),
            shape=(8, 10),
        )
