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

"""Shared accuracy assertions for FlagSparse pytest suites.

The policy follows the FlagGems-style convention:
- numeric compute operators compare against a CPU float64 golden reference,
  cast back to the dtype under test before assertion;
- exact or logical operators compare against a CPU int32-style golden reference
  with equality.
"""


def accelerator_available() -> bool:
    """Whether an accelerator this build targets is usable.

    Not ``torch.cuda.is_available()``: MUSA and Ascend are separate torch device types
    supplied by out-of-tree extensions, so on those backends ``torch.cuda`` is
    unavailable and a cuda-only guard skips every test silently.  ``_ACCEL`` resolves to
    torch.cuda / torch.musa / torch.npu for the active backend.
    """
    try:
        from flagsparse.sparse_operations._common import _ACCEL

        return bool(_ACCEL.is_available())
    except Exception:
        import torch

        return bool(torch.cuda.is_available())


ACCELERATOR_REQUIRED = "an accelerator (CUDA/ROCm/MACA/MUSA/Ascend) is required"


def accelerator_device_type() -> str:
    """torch device type for the active backend: cuda / musa / npu.

    MUSA and Ascend register their own torch device types, so a hardcoded
    ``torch.device("cuda")`` raises NotImplementedError there even though the backend
    dispatches correctly.  On CUDA/ROCm/MACA this returns "cuda" and is a no-op.
    """
    try:
        from flagsparse.sparse_operations._common import _ACCEL_DEVICE_TYPE

        return str(_ACCEL_DEVICE_TYPE)
    except Exception:
        return "cuda"


ACCELERATOR_DEVICE_TYPE = accelerator_device_type()


def accelerator_device():
    """``torch.device`` for the active accelerator; use instead of ``torch.device("cuda")``."""
    import torch as _torch

    return _torch.device(ACCELERATOR_DEVICE_TYPE)


GOLDEN_DEVICE = "cpu"


def golden_device():
    """Device the dense golden reference is built and evaluated on: always CPU.

    The module docstring's policy ("compare against a CPU float64 golden reference")
    is not a stylistic preference -- a reference computed on the accelerator tests the
    vendor's dense library, not FlagSparse.  Measured on Moore Threads (MTT S5000,
    torch_musa 2.7.1) with ``tools/probe_accel_capabilities.py``: muDNN has no
    ``where`` for float64/complex, no ``sum`` for complex, and no 2-D-by-1-D matmul for
    float64/complex, so every dtype except float32 failed while *building the input or
    the reference*, before the operator under test ran.  Keeping the reference on CPU
    removes that entire class and costs nothing at these shapes (<= 160x1024).

    Only the tensors actually handed to a FlagSparse operator belong on
    ``accelerator_device()``.
    """
    import torch as _torch

    return _torch.device(GOLDEN_DEVICE)


def _accel_supports_bf16() -> bool:
    """bfloat16 support on the active accelerator.

    ``is_bf16_supported`` is a torch.cuda API; torch.musa and torch.npu may not expose
    it, so a missing attribute is treated as "unknown" and reported as supported --
    the test then runs and fails loudly if the dtype really is unavailable, which beats
    skipping silently on every non-CUDA backend.
    """
    try:
        from flagsparse.sparse_operations._common import _ACCEL

        probe = getattr(_ACCEL, "is_bf16_supported", None)
        return True if probe is None else bool(probe())
    except Exception:
        return False

import torch


def is_mthreads_backend() -> bool:
    """Whether the active FlagSparse dispatch is the MUSA backend."""
    try:
        from flagsparse.sparse_operations._common import _backend_name

        return _backend_name() == "mthreads"
    except Exception:
        return False


def scipy_sparse_mm(data, indices, indptr, shape, rhs, *, layout="csr", op="non"):
    """Compute a sparse reference on CPU with SciPy and return a CPU torch tensor.

    Accuracy tests use this only for MUSA, whose PyTorch sparse matmul operators are
    not registered. Inputs may live on MUSA; conversion is deliberately through CPU.
    """
    import numpy as np
    import scipy.sparse as sp

    values = data.detach().cpu().numpy()
    rhs_np = rhs.detach().cpu().numpy()
    if layout == "csr":
        matrix = sp.csr_matrix(
            (values, indices.detach().cpu().numpy(), indptr.detach().cpu().numpy()),
            shape=shape,
        )
    elif layout == "csc":
        matrix = sp.csc_matrix(
            (values, indices.detach().cpu().numpy(), indptr.detach().cpu().numpy()),
            shape=shape,
        )
    elif layout == "coo":
        row, col = indices
        matrix = sp.coo_matrix(
            (values, (row.detach().cpu().numpy(), col.detach().cpu().numpy())),
            shape=shape,
        ).tocsr()
    else:
        raise ValueError(f"unsupported sparse layout: {layout}")

    if op == "trans":
        matrix = matrix.transpose()
    elif op == "conj":
        matrix = matrix.getH() if np.iscomplexobj(values) else matrix.transpose()
    elif op != "non":
        raise ValueError(f"unsupported op: {op}")
    result = matrix.dot(rhs_np)
    return torch.from_numpy(np.asarray(result))


def scipy_sparse_product(
    left_data, left_indices, left_indptr, left_shape,
    right_data, right_indices, right_indptr, right_shape,
):
    """Multiply two CSR matrices on CPU and return a CPU torch sparse COO tensor."""
    import numpy as np
    import scipy.sparse as sp

    left = sp.csr_matrix(
        (
            left_data.detach().cpu().numpy(),
            left_indices.detach().cpu().numpy(),
            left_indptr.detach().cpu().numpy(),
        ),
        shape=left_shape,
    )
    right = sp.csr_matrix(
        (
            right_data.detach().cpu().numpy(),
            right_indices.detach().cpu().numpy(),
            right_indptr.detach().cpu().numpy(),
        ),
        shape=right_shape,
    )
    coo = (left @ right).tocoo()
    indices = torch.from_numpy(
        np.stack((coo.row, coo.col), axis=0).astype("int64")
    )
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()


def scipy_bsr_mm(data, indices, indptr, shape, rhs, *, op="non"):
    """Compute BSR SpMV/SpMM on CPU with SciPy."""
    import numpy as np
    import scipy.sparse as sp

    matrix = sp.bsr_matrix(
        (
            data.detach().cpu().numpy(),
            indices.detach().cpu().numpy(),
            indptr.detach().cpu().numpy(),
        ),
        shape=shape,
    )
    if op == "trans":
        matrix = matrix.transpose()
    elif op == "conj":
        matrix = matrix.getH() if np.iscomplexobj(data.detach().cpu().numpy()) else matrix.transpose()
    elif op != "non":
        raise ValueError(f"unsupported op: {op}")
    return torch.from_numpy(np.asarray(matrix.dot(rhs.detach().cpu().numpy())))


def scipy_triangular_solve(matrix, rhs, *, lower, unit_diagonal, op="non"):
    """Solve a triangular system on CPU with SciPy for MUSA accuracy tests."""
    import numpy as np
    from scipy.linalg import solve_triangular

    dense = matrix.detach().cpu().numpy()
    if op == "trans":
        dense = dense.T
    elif op == "conj":
        dense = dense.conj().T
    elif op != "non":
        raise ValueError(f"unsupported op: {op}")
    result = solve_triangular(
        dense,
        rhs.detach().cpu().numpy(),
        lower=(not lower if op in ("trans", "conj") else lower),
        unit_diagonal=unit_diagonal,
    )
    return torch.from_numpy(np.asarray(result))


def _optional_dtype(name):
    return getattr(torch, name, None)


_TOLERANCE_BY_DTYPE = {
    torch.bool: 0,
    torch.uint8: 0,
    torch.int8: 0,
    torch.int16: 0,
    torch.int32: 0,
    torch.int64: 0,
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
    torch.float64: 1e-7,
    torch.complex64: 1.3e-6,
    torch.complex128: 1e-7,
}

for _name, _tol in {
    "float8_e4m3fn": 1e-3,
    "float8_e5m2": 1e-3,
    "float8_e4m3fnuz": 1e-3,
    "float8_e5m2fnuz": 1e-3,
}.items():
    _dtype = _optional_dtype(_name)
    if _dtype is not None:
        _TOLERANCE_BY_DTYPE[_dtype] = _tol

TOLERANCE_BY_DTYPE = dict(_TOLERANCE_BY_DTYPE)


def tolerance_for_dtype(dtype, default=1e-4):
    """Return the centralized absolute/relative tolerance for a torch dtype."""
    return TOLERANCE_BY_DTYPE.get(dtype, default)


def close_tolerances(dtype, default=1e-4):
    """Return ``(rtol, atol)`` using the shared FlagGems-style dtype tolerance."""
    tolerance = tolerance_for_dtype(dtype, default=default)
    return tolerance, tolerance


def golden_reference_close(reference, dtype):
    """Cast a CPU-FP64 golden reference to the dtype being validated."""
    if not torch.is_tensor(reference):
        raise TypeError("reference must be a torch.Tensor")
    return reference.to(dtype=dtype)


def golden_reference_equal(reference):
    """Cast an exact-comparison golden reference to CPU int32."""
    if not torch.is_tensor(reference):
        raise TypeError("reference must be a torch.Tensor")
    return reference.to(device="cpu", dtype=torch.int32)


def gems_assert_close(res, ref, dtype, equal_nan=False, reduce_dim=1, atol=None):
    """Assert approximate equality using the centralized dtype tolerance policy."""
    del reduce_dim  # Kept for FlagGems-compatible call sites.
    tolerance = tolerance_for_dtype(dtype) if atol is None else atol
    expected = golden_reference_close(ref, dtype).to(device=res.device)
    torch.testing.assert_close(
        res,
        expected,
        atol=tolerance,
        rtol=tolerance,
        equal_nan=equal_nan,
    )


def gems_assert_equal(res, ref, equal_nan=False):
    """Assert exact equality for exact/logical outputs."""
    expected = golden_reference_equal(ref).to(device=res.device)
    torch.testing.assert_close(res, expected, atol=0, rtol=0, equal_nan=equal_nan)
