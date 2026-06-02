"""Shared accuracy assertions for FlagSparse pytest suites.

The policy follows the FlagGems-style convention:
- numeric compute operators compare against a CPU float64 golden reference,
  cast back to the dtype under test before assertion;
- exact or logical operators compare against a CPU int32-style golden reference
  with equality.
"""

import torch


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
