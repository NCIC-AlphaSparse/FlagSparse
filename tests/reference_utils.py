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

"""SciPy correctness references for the benchmark CLIs, computed on CPU.

CUDA and ROCm check against their vendor sparse library plus torch. Every other
backend uses these instead, because torch.sparse there is not a reference but
another thing under test: MACA returns non-finite output on the fp32 CSR path,
MUSA registers no sparse matmul at all in any layout or dtype. A reference that
is itself broken reports the kernel as wrong -- the most expensive false alarm
there is.

Which one is in force is decided centrally, not here:
``_common._use_scipy_accuracy_reference()``, overridable with
FLAGSPARSE_ACCURACY_REFERENCE=auto|scipy|torch.

TIMING IS NOT AFFECTED. The PyTorch baseline column keeps being measured on the
accelerator; only the value the kernel is *compared against* moves to CPU SciPy.
"""

import numpy as np
import torch


_PROMOTED = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float64,
    torch.complex64: torch.complex128,
}


def reference_dtype(dtype):
    """Accumulate the reference one step wider than the operator computes in."""
    return _PROMOTED.get(dtype, dtype)


def _numpy(tensor, dtype):
    return tensor.to(dtype).detach().cpu().numpy()


def scipy_csr(data, indices, indptr, shape, dtype):
    """SciPy CSR at `dtype`, from the same arrays the operator was given."""
    import scipy.sparse as sp

    return sp.csr_matrix(
        (
            _numpy(data, dtype),
            indices.detach().cpu().numpy().astype(np.int64, copy=False),
            indptr.detach().cpu().numpy().astype(np.int64, copy=False),
        ),
        shape=tuple(int(v) for v in shape),
    )


def scipy_coo(data, rows, cols, shape, dtype):
    """SciPy COO at `dtype`. Duplicates are summed, matching the COO kernels."""
    import scipy.sparse as sp

    matrix = sp.coo_matrix(
        (
            _numpy(data, dtype),
            (
                rows.detach().cpu().numpy().astype(np.int64, copy=False),
                cols.detach().cpu().numpy().astype(np.int64, copy=False),
            ),
        ),
        shape=tuple(int(v) for v in shape),
    )
    return matrix.tocsr()


def apply_op(matrix, op):
    """A, A.T or A.conj().T -- the three operand forms the operators accept."""
    op = _normalize_op(op)
    if op == "non":
        return matrix
    if op == "trans":
        return matrix.transpose()
    return matrix.conj().transpose()


def _normalize_op(op):
    token = str(op or "non").strip().lower()
    if token in ("non", "n", "none", "no_trans", "non_transpose"):
        return "non"
    if token in ("trans", "t", "transpose"):
        return "trans"
    if token in ("conj", "c", "conj_transpose", "conjugate"):
        return "conj"
    raise ValueError(f"unsupported op for the SciPy reference: {op!r}")


def as_torch(array, dtype, device):
    """Back to torch at the reference dtype; the caller casts to the output dtype."""
    return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)


def spmv(matrix, x, dtype, *, op="non"):
    """y = op(A) @ x, with x a 1-D torch vector."""
    return apply_op(matrix, op) @ _numpy(x, dtype)


def spmm(matrix, dense, dtype, *, op="non"):
    """C = op(A) @ B, with B a 2-D torch tensor."""
    return apply_op(matrix, op) @ _numpy(dense, dtype)


def spgemm(a_matrix, b_matrix, *, op="non"):
    """C = op(A) @ B, both sparse. Returns a SciPy CSR."""
    return (apply_op(a_matrix, op) @ b_matrix).tocsr()


def sddmm(matrix, left, right, dtype, *, op="non"):
    """Sampled dense-dense product: (op(X) @ Y.T) masked by A's pattern.

    Returns the values in A's CSR order, so the caller compares value arrays
    rather than materialising a dense product.
    """
    pattern = apply_op(matrix, op).tocoo()
    x = _numpy(left, dtype)
    y = _numpy(right, dtype)
    rows = pattern.row.astype(np.int64, copy=False)
    cols = pattern.col.astype(np.int64, copy=False)
    return np.einsum("ij,ij->i", x[rows], y[cols])


def sddmm_csr_values(indices, indptr, left, right, dtype):
    """Sampled dot products in CSR order, straight from the CSR arrays.

    SDDMM never materialises a sparse operand, so this takes the pattern rather
    than a matrix: vals[i] = dot(left[row_of(i)], right[indices[i]]). The caller
    applies alpha/beta, matching sddmm_csr._sddmm_reference().
    """
    idx = indices.detach().cpu().numpy().astype(np.int64, copy=False)
    ptr = indptr.detach().cpu().numpy().astype(np.int64, copy=False)
    if idx.size == 0:
        return np.zeros(0, dtype=_numpy(left[:0], dtype).dtype)
    rows = np.repeat(np.arange(ptr.size - 1, dtype=np.int64), np.diff(ptr))
    x = _numpy(left, dtype)
    y = _numpy(right, dtype)
    return np.einsum("ij,ij->i", x[rows], y[idx])


def triangular_solve(matrix, rhs, dtype, *, lower, unit_diagonal, op="non"):
    """Solve op(A) @ Z = rhs for a triangular A. rhs may be 1-D or 2-D.

    SciPy's spsolve_triangular takes the triangle from `lower`, so transposing
    the operand flips it -- getting that wrong silently solves the other system
    and the error shows up as a wrong answer rather than an exception.
    """
    from scipy.sparse.linalg import spsolve_triangular

    op = _normalize_op(op)
    operand = apply_op(matrix, op).tocsr()
    effective_lower = lower if op == "non" else not lower
    if unit_diagonal:
        operand = operand.copy()
        operand.setdiag(np.ones(min(operand.shape), dtype=operand.dtype))
    b = _numpy(rhs, dtype)
    return spsolve_triangular(operand, b, lower=effective_lower, unit_diagonal=False)
