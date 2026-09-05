"""Fast MatrixMarket loading backed by scipy.io.mmread (C-accelerated).

The historical per-line Python parsers in the benchmark scripts took minutes on
large SuiteSparse matrices (pure-Python readlines + a per-nonzero dict loop),
while scipy reads the same files in well under a second. scipy.io.mmread already
expands symmetric / skew-symmetric / hermitian matrices and materializes
``pattern`` matrices with unit values, matching the previous loaders' semantics,
so callers get equivalent CSR/COO/CSC tensors — just orders of magnitude faster.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.io import mmread


class NonSquareMatrixError(ValueError):
    """Raised when a triangular solve benchmark receives a non-square matrix."""

    def __init__(self, operator, shape):
        self.operator = str(operator)
        self.shape = (int(shape[0]), int(shape[1]))
        super().__init__(
            f"{self.operator} requires a square matrix, got {self.shape}"
        )


def read_scipy_csr(file_path):
    """Return a canonical scipy CSR (symmetry expanded, duplicates summed, sorted)."""
    csr = mmread(str(file_path)).tocsr()
    csr.sum_duplicates()
    csr.sort_indices()
    return csr


def _values_to_torch(data_np, dtype, device):
    if np.iscomplexobj(data_np) and dtype not in (torch.complex64, torch.complex128):
        data_np = data_np.real
    return torch.tensor(np.ascontiguousarray(data_np), dtype=dtype, device=device)


def load_csr(file_path, dtype=torch.float32, device=None):
    """(data, indices, indptr, (n_rows, n_cols)) as torch tensors — CSR layout."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csr = read_scipy_csr(file_path)
    data = _values_to_torch(csr.data, dtype, device)
    indices = torch.tensor(
        np.ascontiguousarray(csr.indices.astype(np.int64)), dtype=torch.int64, device=device
    )
    indptr = torch.tensor(
        np.ascontiguousarray(csr.indptr.astype(np.int64)), dtype=torch.int64, device=device
    )
    return data, indices, indptr, (int(csr.shape[0]), int(csr.shape[1]))


def load_csc(file_path, dtype=torch.float32, device=None):
    """(data, indices(row), indptr(col), (n_rows, n_cols)) — CSC layout."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csc = mmread(str(file_path)).tocsc()
    csc.sum_duplicates()
    csc.sort_indices()
    data = _values_to_torch(csc.data, dtype, device)
    indices = torch.tensor(
        np.ascontiguousarray(csc.indices.astype(np.int64)), dtype=torch.int64, device=device
    )
    indptr = torch.tensor(
        np.ascontiguousarray(csc.indptr.astype(np.int64)), dtype=torch.int64, device=device
    )
    return data, indices, indptr, (int(csc.shape[0]), int(csc.shape[1]))


def load_coo(file_path, dtype=torch.float32, device=None):
    """(data, rows, cols, (n_rows, n_cols)) — coalesced COO in row-major order."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csr = read_scipy_csr(file_path)  # row-major, duplicates summed, cols sorted
    coo = csr.tocoo()
    data = _values_to_torch(coo.data, dtype, device)
    rows = torch.tensor(
        np.ascontiguousarray(coo.row.astype(np.int64)), dtype=torch.int64, device=device
    )
    cols = torch.tensor(
        np.ascontiguousarray(coo.col.astype(np.int64)), dtype=torch.int64, device=device
    )
    return data, rows, cols, (int(csr.shape[0]), int(csr.shape[1]))


def read_scipy_coo_entries(file_path):
    """Return (rows, cols, vals) int64/float64 numpy arrays — symmetry expanded."""
    coo = mmread(str(file_path)).tocoo()
    return (
        coo.row.astype(np.int64),
        coo.col.astype(np.int64),
        np.ascontiguousarray(coo.data),
    )


def load_csr_spsv(file_path, dtype=torch.float32, device=None, lower=True):
    """Well-conditioned triangular-ready CSR for SpSV/SpSM-style solves.

    Reproduces the previous loader's normalization: ensure a structural diagonal,
    scale every row by its abs-sum ``s`` and set the diagonal to ``(s + 1) / s``
    so the system is strictly diagonally dominant. For real dtypes this matches
    the old per-row math exactly; complex dtypes get the same real-valued,
    diagonally dominant matrix (values differ from the old complex branch but the
    solve stays well conditioned, which is all the benchmark needs). ``lower`` is
    accepted for signature compatibility; triangular extraction is done by the
    caller downstream.
    """
    import scipy.sparse as sp

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = mmread(str(file_path)).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    if A.shape[0] != A.shape[1]:
        raise NonSquareMatrixError("SpSV", A.shape)
    if np.iscomplexobj(A.data):
        A = A.real.tocsr()
    A = A.astype(np.float64)
    n = A.shape[0]

    # Ensure a structural diagonal (default value 1.0 where missing).
    coo = A.tocoo()
    has_diag = np.zeros(n, dtype=bool)
    dmask = coo.row == coo.col
    has_diag[coo.row[dmask]] = True
    missing = np.where(~has_diag)[0]
    if missing.size:
        A = (A + sp.csr_matrix((np.ones(missing.size), (missing, missing)), shape=A.shape)).tocsr()
        A.sort_indices()

    s = np.asarray(np.abs(A).sum(axis=1)).ravel()
    s[s == 0.0] = 1.0
    R = (sp.diags(1.0 / s) @ A).tocsr()
    R.setdiag((s + 1.0) / s)
    R.sort_indices()

    real = R.data
    if dtype in (torch.complex64, torch.complex128):
        data = torch.tensor(np.ascontiguousarray(real), device=device).to(dtype)
    else:
        data = torch.tensor(np.ascontiguousarray(real), device=device).to(dtype)
    indices = torch.tensor(
        np.ascontiguousarray(R.indices.astype(np.int64)), dtype=torch.int64, device=device
    )
    indptr = torch.tensor(
        np.ascontiguousarray(R.indptr.astype(np.int64)), dtype=torch.int64, device=device
    )
    return data, indices, indptr, (int(R.shape[0]), int(R.shape[1]))


def load_entries(file_path):
    """{(row, col): value} dict + (n_rows, n_cols), symmetry expanded / pattern=1.0.

    Matches the old ``load_mtx_entries`` output but sources the nonzeros from the
    C-accelerated scipy reader instead of a per-line Python parse."""
    csr = read_scipy_csr(file_path)
    coo = csr.tocoo()
    vals = coo.data
    if np.iscomplexobj(vals):
        vals = vals.real
    entries = {
        (int(r), int(c)): float(v)
        for r, c, v in zip(coo.row.tolist(), coo.col.tolist(), vals.tolist())
    }
    return entries, (int(csr.shape[0]), int(csr.shape[1]))
