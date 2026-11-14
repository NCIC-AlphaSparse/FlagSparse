"""CPU symbolic and numeric references for SpGEMM accuracy tests."""

import numpy as np
import scipy.sparse


def structural_reference(first, second):
    left, right = first.copy(), second.copy()
    left.data = np.ones(left.nnz, dtype=np.bool_)
    right.data = np.ones(right.nnz, dtype=np.bool_)
    structure = left @ right
    structure.sort_indices()
    return structure


def validate(output, golden, structure):
    values, columns, pointer, shape = output
    if tuple(shape) != golden.shape:
        raise AssertionError("SpGEMM output shape differs from CPU reference")
    actual = scipy.sparse.csr_matrix(
        (values.cpu().numpy(), columns.cpu().numpy(), pointer.cpu().numpy()),
        shape=shape,
    )
    pattern_ok = np.array_equal(actual.indptr, structure.indptr) and np.array_equal(
        actual.indices, structure.indices
    )
    if not pattern_ok:
        raise AssertionError(
            "SpGEMM output pattern differs from CPU symbolic reference"
        )
    if not np.isfinite(actual.data).all():
        raise AssertionError("SpGEMM output contains nonfinite values")
    difference = (actual.astype(golden.dtype) - golden).tocoo()
    errors = np.abs(difference.data)
    maximum = float(errors.max(initial=0))
    scale = float(np.abs(golden.data).max(initial=0))
    normalized = maximum / max(scale, 1.0)
    if difference.nnz:
        expected = np.asarray(golden[difference.row, difference.col]).reshape(-1)
        violations = errors > 1e-4 + 1e-4 * np.abs(expected)
        if violations.any():
            raise AssertionError(
                f"SpGEMM has {int(violations.sum())} entries outside rtol=1e-4, atol=1e-4"
            )
    return dict(
        pattern_ok=True,
        normalized_error=normalized,
        max_abs_error=maximum,
        nnz=actual.nnz,
    )
