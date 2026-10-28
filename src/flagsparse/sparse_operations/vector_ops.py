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

"""Sparse-vector level-1 operators: Axpby and SpVV (cusparseAxpby / cusparseSpVV).

A sparse vector is ``(values, indices)`` over a dense vector of length ``y.numel()``;
indices are unique, as cuSPARSE requires.

Type rules follow cuSPARSE: half types compute in float32 and SpVV returns the
compute type (``f16/bf16 -> f32``, ``i8 -> i32``, otherwise the value type).

Backend notes:
- Complex values go through ``view_as_real`` and explicit real/imag arithmetic.
  Moore Threads has no complex ``sum`` or advanced-indexing kernel, so nothing here
  asks the runtime for either.
- SpVV reduces per-block partials in the real compute dtype, then sums that short
  buffer. No atomics, so it needs neither fp16 nor complex atomic support.
- Ascend's Triton lacks the ``shmem`` extension the generic kernels rely on
  elsewhere; these operators take a pure torch_npu path there.
"""

from ._common import *

import time

import triton
import triton.language as tl

from ._rocm_sparse_cache import TensorPlanCache

AXPBY_VALUE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)
SPVV_VALUE_DTYPES = AXPBY_VALUE_DTYPES + (torch.int8,)
_VECTOR_BLOCK = 1024
_MUSA_AXPBY_INDEX_CACHE = TensorPlanCache()
_ILUVATAR_AXPBY_INDEX_CACHE = TensorPlanCache()


class PreparedAxpby:
    """Prepared Ascend Axpby operands for repeated in-place updates."""

    __slots__ = (
        "values",
        "indices",
        "compute_values",
        "workspace",
        "n",
        "compute_dtype",
        "value_dtype",
        "device",
        "alpha",
        "beta",
    )

    def __init__(self, values, indices, y, alpha, beta):
        self.values = values
        self.indices = indices
        self.compute_dtype = _vector_compute_dtype(values.dtype)
        self.compute_values = (
            values.to(self.compute_dtype)
            if self.compute_dtype != values.dtype
            else values
        )
        self.workspace = (
            torch.empty(y.shape, dtype=self.compute_dtype, device=y.device)
            if self.compute_dtype != values.dtype or _is_complex_dtype(values.dtype)
            else None
        )
        self.n = int(y.numel())
        self.value_dtype = values.dtype
        self.device = values.device
        self.alpha = alpha
        self.beta = beta


class PreparedSpvv:
    """Prepared Ascend SpVV operands for repeated sparse dot products."""

    __slots__ = (
        "values",
        "indices",
        "compute_values",
        "value_dtype",
        "compute_dtype",
        "device",
        "nnz",
        "op",
    )

    def __init__(self, values, indices, op):
        self.values = values
        self.indices = indices
        self.value_dtype = values.dtype
        self.compute_dtype = _vector_compute_dtype(values.dtype)
        self.compute_values = values.to(self.compute_dtype) if self.compute_dtype != values.dtype else values
        self.device = values.device
        self.nnz = int(values.numel())
        self.op = op


def prepare_axpby(values, indices, y, alpha=1.0, beta=1.0, *, validate=True):
    """Prepare Ascend Axpby metadata and reusable compute buffers."""
    if not _is_ascend_runtime():
        raise ValueError("prepared Axpby is currently supported on Ascend only")
    kernel_idx = _validate_sparse_vector(
        values, indices, y, AXPBY_VALUE_DTYPES, "axpby", check_range=validate
    )
    if not y.is_contiguous():
        raise ValueError("axpby: y must be contiguous (it is updated in place)")
    return PreparedAxpby(
        values.contiguous(),
        kernel_idx.to(torch.int64).contiguous(),
        y,
        alpha,
        beta,
    )


def prepare_spvv(values, indices, y, op="non", *, validate=True):
    """Prepare Ascend SpVV metadata and reusable compute operands."""
    if not _is_ascend_runtime():
        raise ValueError("prepared SpVV is currently supported on Ascend only")
    op = _normalize_vector_op(op)
    kernel_idx = _validate_sparse_vector(
        values, indices, y, SPVV_VALUE_DTYPES, "spvv", check_range=validate
    )
    return PreparedSpvv(values.contiguous(), kernel_idx.to(torch.int64).contiguous(), op)


def _run_prepared_axpby(prepared, y, alpha=None, beta=None):
    if y.device != prepared.device or y.dtype != prepared.value_dtype:
        raise ValueError("y must match the prepared Axpby operands")
    if y.ndim != 1 or y.numel() != prepared.n or not y.is_contiguous():
        raise ValueError("y must be a contiguous vector matching the prepared Axpby plan")
    values = prepared.compute_values
    compute = prepared.compute_dtype
    alpha = prepared.alpha if alpha is None else alpha
    beta = prepared.beta if beta is None else beta
    if prepared.workspace is None:
        y.mul_(beta)
        if values.numel():
            y.index_add_(0, prepared.indices, values * alpha)
        return y
    workspace = prepared.workspace
    workspace.copy_(y)
    workspace.mul_(beta)
    update = values * alpha
    if _is_complex_dtype(compute):
        torch.view_as_real(workspace).index_add_(
            0, prepared.indices, torch.view_as_real(update)
        )
    else:
        workspace.index_add_(0, prepared.indices, update)
    y.copy_(workspace)
    return y


def _vector_compute_dtype(value_dtype):
    if value_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if value_dtype == torch.int8:
        return torch.int32
    return value_dtype


def _real_dtype(dtype):
    return _component_dtype_for_complex(dtype) if _is_complex_dtype(dtype) else dtype


def _normalize_vector_op(op):
    token = "non" if op is None else str(op).strip().lower()
    aliases = {"n": "non", "non_trans": "non", "c": "conj", "conj_trans": "conj"}
    token = aliases.get(token, token)
    if token not in ("non", "conj"):
        raise ValueError("op must be 'non' or 'conj'")
    return token


def _validate_sparse_vector(values, indices, y, supported, name, check_range=True):
    for tensor, label in ((values, "values"), (indices, "indices"), (y, "y")):
        if not torch.is_tensor(tensor):
            raise TypeError(f"{label} must be a torch.Tensor")
        if tensor.ndim != 1:
            raise ValueError(f"{label} must be a 1D tensor")
    if not (_is_accel_tensor(values) and _is_accel_tensor(indices) and _is_accel_tensor(y)):
        raise ValueError(f"{name}: values, indices and y must be accelerator tensors")
    if len({values.device, indices.device, y.device}) != 1:
        raise ValueError(f"{name}: values, indices and y must be on one device")
    if values.numel() != indices.numel():
        raise ValueError(f"{name}: values and indices must have the same length")
    if values.dtype not in supported:
        names = ", ".join(str(d).replace("torch.", "") for d in supported)
        raise TypeError(f"{name} supports value dtypes: {names}")
    if y.dtype != values.dtype:
        raise TypeError(f"{name}: y dtype must match values dtype")
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if check_range and indices.numel() > 0:
        # One host sync for both bounds (min().item() + max().item() was two, ~50 us).
        lo, hi = (int(v) for v in torch.stack(torch.aminmax(indices)).tolist())
        if lo < 0 or hi >= y.numel():
            raise IndexError(f"{name}: indices out of range for y of length {y.numel()}")
    return indices.to(torch.int32) if indices.dtype == torch.int64 else indices


def _split_scalar(value, complex_dtype):
    value = complex(value)
    if not complex_dtype and value.imag != 0:
        raise TypeError("a complex scalar needs complex values")
    return value.real, value.imag


# ---------------------------------------------------------------------------
# Axpby: y = alpha * x + beta * y
# ---------------------------------------------------------------------------


@triton.jit
def _scale_real_kernel(y_ptr, n, beta, COMPUTE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(COMPUTE)
    tl.store(y_ptr + offs, (y * beta).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _scale_complex_kernel(y_ri_ptr, n, beta_re, beta_im, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    yr = tl.load(y_ri_ptr + 2 * offs, mask=mask, other=0.0)
    yi = tl.load(y_ri_ptr + 2 * offs + 1, mask=mask, other=0.0)
    tl.store(y_ri_ptr + 2 * offs, yr * beta_re - yi * beta_im, mask=mask)
    tl.store(y_ri_ptr + 2 * offs + 1, yr * beta_im + yi * beta_re, mask=mask)


@triton.jit
def _axpy_real_kernel(
    y_ptr, x_ptr, idx_ptr, nnz, alpha, COMPUTE: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(COMPUTE)
    y = tl.load(y_ptr + idx, mask=mask, other=0.0).to(COMPUTE)
    tl.store(y_ptr + idx, (y + alpha * x).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _axpy_complex_kernel(
    y_ri_ptr, x_ri_ptr, idx_ptr, nnz, alpha_re, alpha_im, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    xr = tl.load(x_ri_ptr + 2 * offs, mask=mask, other=0.0)
    xi = tl.load(x_ri_ptr + 2 * offs + 1, mask=mask, other=0.0)
    yr = tl.load(y_ri_ptr + 2 * idx, mask=mask, other=0.0)
    yi = tl.load(y_ri_ptr + 2 * idx + 1, mask=mask, other=0.0)
    tl.store(y_ri_ptr + 2 * idx, yr + alpha_re * xr - alpha_im * xi, mask=mask)
    tl.store(y_ri_ptr + 2 * idx + 1, yi + alpha_re * xi + alpha_im * xr, mask=mask)


@triton.jit
def _musa_axpby_inverse_kernel(inverse, indices, nnz, n, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(indices + pos, pos < nnz, -1)
    tl.store(inverse + row, pos, (pos < nnz) & (row >= 0) & (row < n))


@triton.jit
def _musa_axpby_kernel(y, values, inverse, n, alpha, beta, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(inverse + row, row < n, -1)
    value = tl.load(values + pos, (row < n) & (pos >= 0), 0).to(tl.float32)
    previous = tl.load(y + row, row < n, 0).to(tl.float32)
    tl.store(y + row, beta * previous + alpha * value, row < n)


def _musa_axpby(values, indices, y, alpha, beta):
    n = y.numel()
    if not n:
        return
    inverse = _MUSA_AXPBY_INDEX_CACHE.get((indices,), (n,))
    if inverse is None:
        inverse = torch.full((n,), -1, dtype=torch.int32, device=y.device)
        nnz = indices.numel()
        if nnz:
            _musa_axpby_inverse_kernel[((nnz + 1023) // 1024,)](
                inverse, indices.contiguous(), nnz, n, BLOCK=1024,
                num_warps=4,
            )
        _MUSA_AXPBY_INDEX_CACHE.put((indices,), (n,), inverse)
    _musa_axpby_kernel[((n + 2047) // 2048,)](
        y, values, inverse, n, alpha, beta, BLOCK=2048, num_warps=4,
    )


def _iluvatar_axpby(values, indices, y, alpha, beta):
    """BI-V fused Axpby using a cached sparse-index inverse map."""
    n = y.numel()
    if not n:
        return
    inverse = _ILUVATAR_AXPBY_INDEX_CACHE.get((indices,), (n,))
    if inverse is None:
        inverse = torch.full((n,), -1, dtype=torch.int32, device=y.device)
        nnz = indices.numel()
        if nnz:
            _musa_axpby_inverse_kernel[((nnz + 1023) // 1024,)](
                inverse, indices.contiguous(), nnz, n, BLOCK=1024,
                num_warps=4,
            )
        _ILUVATAR_AXPBY_INDEX_CACHE.put((indices,), (n,), inverse)
    _musa_axpby_kernel[((n + 2047) // 2048,)](
        y, values, inverse, n, alpha, beta, BLOCK=2048, num_warps=4,
    )


def _torch_axpby(values, indices, y, alpha, beta):
    """Ascend path, also the reference shape of the Triton path."""
    compute = _vector_compute_dtype(y.dtype)
    idx = indices.to(torch.int64)
    if _is_ascend_runtime() and compute == y.dtype and not _is_complex_dtype(y.dtype):
        y.mul_(beta)
        if values.numel():
            y.index_add_(0, idx, values * alpha)
        return y
    acc = y.to(compute) if compute != y.dtype else y.clone()
    acc.mul_(beta)
    update = values.to(compute) * alpha
    if _is_complex_dtype(compute):
        # index_add_ on complex has no NPU/MUSA kernel; add the two real planes.
        torch.view_as_real(acc).index_add_(0, idx, torch.view_as_real(update))
    else:
        acc.index_add_(0, idx, update)
    y.copy_(acc)
    return y


def flagsparse_axpby(
    values=None,
    indices=None,
    y=None,
    alpha=None,
    beta=None,
    return_time=False,
    *,
    validate=True,
    prepared=None,
):
    """In-place ``y = alpha * x + beta * y`` for sparse ``x = (values, indices)``.

    Mirrors ``cusparseAxpby``; half types are computed in float32. Returns ``y``
    (and the elapsed milliseconds when ``return_time``). ``validate=False`` skips the
    index range check, a host sync that cuSPARSE does not perform either.
    """
    if prepared is not None:
        if not _is_ascend_runtime():
            raise ValueError("prepared Axpby is currently supported on Ascend only")
        if not isinstance(prepared, PreparedAxpby):
            raise TypeError("prepared must be a PreparedAxpby")
        if values is not None or indices is not None:
            raise ValueError("pass either prepared or values/indices, not both")
        if y is None:
            raise ValueError("y is required when using a prepared Axpby plan")
        if return_time:
            _ACCEL.synchronize()
            t0 = time.perf_counter()
        result = _run_prepared_axpby(prepared, y, alpha=alpha, beta=beta)
        if return_time:
            _ACCEL.synchronize()
            return result, (time.perf_counter() - t0) * 1000.0
        return result
    if any(v is None for v in (values, indices, y)):
        raise ValueError("values, indices and y are required when prepared is not provided")
    alpha = 1.0 if alpha is None else alpha
    beta = 1.0 if beta is None else beta
    kernel_idx = _validate_sparse_vector(
        values, indices, y, AXPBY_VALUE_DTYPES, "axpby", check_range=validate
    )
    if not y.is_contiguous():
        raise ValueError("axpby: y must be contiguous (it is updated in place)")
    is_complex = _is_complex_dtype(y.dtype)
    a_re, a_im = _split_scalar(alpha, is_complex)
    b_re, b_im = _split_scalar(beta, is_complex)
    values = values.contiguous()
    t0 = None
    if return_time:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    if (
        _is_mthreads_runtime() and y.dtype in (torch.float16, torch.bfloat16)
        and y.numel() <= 4 * 1024 * 1024
    ):
        # Cache index ownership only; every call reads the current values/y.
        _musa_axpby(values, indices, y, a_re, b_re)
    elif _is_iluvatar_runtime() and y.dtype in (torch.float16, torch.float32):
        _iluvatar_axpby(values, kernel_idx, y, a_re, b_re)
    elif _is_rocm_runtime() and y.dtype in (torch.float16, torch.bfloat16):
        # Native half indexed add avoids a second launch on gfx936. Indices
        # are unique, as required by the sparse-vector API.
        y.mul_(b_re)
        if values.numel():
            y.index_add_(0, kernel_idx.to(torch.int64), values * a_re)
    elif _is_ascend_runtime():
        _torch_axpby(values, kernel_idx, y, complex(a_re, a_im) if is_complex else a_re,
                     complex(b_re, b_im) if is_complex else b_re)
    else:
        n, nnz = y.numel(), values.numel()
        if is_complex:
            y_ri = torch.view_as_real(y).reshape(-1)
            if (b_re, b_im) != (1.0, 0.0) and n:
                _scale_complex_kernel[(triton.cdiv(n, _VECTOR_BLOCK),)](
                    y_ri, n, b_re, b_im, BLOCK=_VECTOR_BLOCK
                )
            if nnz:
                x_ri = torch.view_as_real(values).reshape(-1)
                _axpy_complex_kernel[(triton.cdiv(nnz, _VECTOR_BLOCK),)](
                    y_ri, x_ri, kernel_idx, nnz, a_re, a_im, BLOCK=_VECTOR_BLOCK
                )
        else:
            compute = tl.float64 if y.dtype == torch.float64 else tl.float32
            if b_re != 1.0 and n:
                _scale_real_kernel[(triton.cdiv(n, _VECTOR_BLOCK),)](
                    y, n, b_re, COMPUTE=compute, BLOCK=_VECTOR_BLOCK
                )
            if nnz:
                _axpy_real_kernel[(triton.cdiv(nnz, _VECTOR_BLOCK),)](
                    y, values, kernel_idx, nnz, a_re, COMPUTE=compute, BLOCK=_VECTOR_BLOCK
                )
    if return_time:
        _ACCEL.synchronize()
        return y, (time.perf_counter() - t0) * 1000.0
    return y


# ---------------------------------------------------------------------------
# SpVV: result = sum op(x)[k] * y[idx[k]]
# ---------------------------------------------------------------------------


@triton.jit
def _spvv_real_kernel(
    part_ptr, x_ptr, y_ptr, idx_ptr, nnz, ACC: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    x = tl.load(x_ptr + offs, mask=mask, other=0).to(ACC)
    y = tl.load(y_ptr + idx, mask=mask, other=0).to(ACC)
    tl.store(part_ptr + pid, tl.sum(x * y, axis=0))


@triton.jit
def _spvv_complex_kernel(
    part_re_ptr, part_im_ptr, x_ri_ptr, y_ri_ptr, idx_ptr, nnz,
    CONJ: tl.constexpr, BLOCK: tl.constexpr, PACKED: tl.constexpr = False,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    xr = tl.load(x_ri_ptr + 2 * offs, mask=mask, other=0.0)
    xi = tl.load(x_ri_ptr + 2 * offs + 1, mask=mask, other=0.0)
    if CONJ:
        xi = -xi
    yr = tl.load(y_ri_ptr + 2 * idx, mask=mask, other=0.0)
    yi = tl.load(y_ri_ptr + 2 * idx + 1, mask=mask, other=0.0)
    pos = 2 * pid if PACKED else pid
    tl.store(part_re_ptr + pos, tl.sum(xr * yr - xi * yi, axis=0))
    tl.store(part_im_ptr + pos + (1 if PACKED else 0), tl.sum(xr * yi + xi * yr, axis=0))


@triton.jit
def _spvv_complex_finalize(P, Y, N, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    real = tl.load(P + 2 * offsets, offsets < N, other=0.)
    imag = tl.load(P + 2 * offsets + 1, offsets < N, other=0.)
    tl.store(Y, tl.sum(real, 0))
    tl.store(Y + 1, tl.sum(imag, 0))


def _torch_spvv(values, indices, y, op):
    compute = _vector_compute_dtype(values.dtype)
    idx = indices.to(torch.int64)
    if _is_complex_dtype(compute):
        if _is_ascend_runtime():
            gathered = y.index_select(0, idx)
            lhs = values.conj() if op == "conj" else values
            return (lhs * gathered).sum()
        # Real-plane gathers and sums: no complex index/sum kernel needed.
        x_ri = torch.view_as_real(values)
        y_ri = torch.view_as_real(y).index_select(0, idx)
        xr, xi = x_ri[:, 0], x_ri[:, 1]
        if op == "conj":
            xi = -xi
        yr, yi = y_ri[:, 0], y_ri[:, 1]
        return torch.complex((xr * yr - xi * yi).sum(), (xr * yi + xi * yr).sum())
    gathered = y.index_select(0, idx).to(compute)
    lhs = values.to(compute)
    if _is_ascend_runtime() and compute == torch.float32:
        return torch.dot(lhs, gathered)
    return (lhs * gathered).sum(dtype=compute)


def _run_prepared_spvv(prepared, y):
    if y.device != prepared.device or y.dtype != prepared.value_dtype:
        raise ValueError("y must match the prepared SpVV operands")
    if y.ndim != 1:
        raise ValueError("y must be a 1D vector")
    values = prepared.compute_values
    gathered = y.index_select(0, prepared.indices)
    lhs = values.conj() if prepared.op == "conj" and values.is_complex() else values
    return (lhs * gathered.to(prepared.compute_dtype)).sum(dtype=prepared.compute_dtype)


def flagsparse_spvv(
    values=None,
    indices=None,
    y=None,
    op="non",
    return_time=False,
    *,
    validate=True,
    prepared=None,
):
    """Sparse-dense dot product ``sum(op(x)[k] * y[indices[k]])`` (``cusparseSpVV``).

    ``op`` is ``"non"`` or ``"conj"``. Returns a 0-d tensor of the compute dtype:
    float32 for float16/bfloat16, int32 for int8, otherwise the value dtype.
    ``validate=False`` skips the index range check (a host sync).
    """
    if prepared is not None:
        if not _is_ascend_runtime():
            raise ValueError("prepared SpVV is currently supported on Ascend only")
        if not isinstance(prepared, PreparedSpvv):
            raise TypeError("prepared must be a PreparedSpvv")
        if values is not None or indices is not None:
            raise ValueError("pass either prepared or values/indices, not both")
        if y is None:
            raise ValueError("y is required when using a prepared SpVV plan")
        if return_time:
            _ACCEL.synchronize()
            t0 = time.perf_counter()
        result = _run_prepared_spvv(prepared, y)
        if return_time:
            _ACCEL.synchronize()
            return result, (time.perf_counter() - t0) * 1000.0
        return result
    if any(v is None for v in (values, indices, y)):
        raise ValueError("values, indices and y are required when prepared is not provided")
    op = _normalize_vector_op(op)
    kernel_idx = _validate_sparse_vector(
        values, indices, y, SPVV_VALUE_DTYPES, "spvv", check_range=validate
    )
    values = values.contiguous()
    y = y.contiguous()
    compute = _vector_compute_dtype(values.dtype)
    t0 = None
    if return_time:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    # Match the ROCm PyTorch baseline for the variants whose Triton reduction
    # is launch-bound.  Keeping the expression native is also important for
    # complex values: the real-plane fallback needs two extra reductions.
    if _is_rocm_runtime() and values.dtype in (torch.float16, torch.complex64, torch.int8):
        idx = kernel_idx.to(torch.int64)
        rhs = y.index_select(0, idx).to(compute)
        lhs = values.to(compute)
        if values.dtype == torch.complex64:
            # vdot fuses the complex multiply and reduction into one device
            # reduction; unlike ``(conj(x) * y).sum()`` it avoids a temporary.
            result = torch.vdot(lhs, rhs) if op == "conj" else torch.dot(lhs, rhs)
        elif values.dtype == torch.int8:
            result = (lhs * rhs).sum(dtype=compute)
        else:
            result = torch.dot(lhs, rhs)
    elif _is_ascend_runtime():
        if kernel_idx.dtype != torch.int64:
            kernel_idx = kernel_idx.to(torch.int64).contiguous()
        result = _torch_spvv(values, kernel_idx, y, op)
    else:
        nnz = values.numel()
        n_parts = max(1, triton.cdiv(nnz, _VECTOR_BLOCK))
        real = _real_dtype(compute)
        # MACA benefits from the same packed real/imaginary partial layout as
        # CUDA: one reduction kernel replaces two independent ``Tensor.sum``
        # launches and avoids a temporary complex construction.  Keep this
        # opt-in backend-specific because other Triton targets have different
        # complex lowering behavior.
        if (
            _is_complex_dtype(compute)
            and (
                _backend_name() == "cuda"
                or _is_maca_runtime()
                or _is_iluvatar_runtime()
            )
            and n_parts <= 1024
        ):
            result = torch.empty((), dtype=compute, device=y.device)
            target = torch.view_as_real(result)
            if not nnz:
                result.zero_()
            else:
                part = target if n_parts == 1 else torch.empty((n_parts, 2), dtype=real, device=y.device)
                _spvv_complex_kernel[(n_parts,)](
                    part, part, torch.view_as_real(values), torch.view_as_real(y),
                    kernel_idx, nnz, CONJ=(op == "conj"), BLOCK=_VECTOR_BLOCK, PACKED=True)
                if n_parts > 1:
                    _spvv_complex_finalize[(1,)](part, target, n_parts,
                                                BLOCK=triton.next_power_of_2(n_parts))
        elif _is_complex_dtype(compute):
            # Each program writes its own slot; only the nnz == 0 case needs zeros.
            alloc = torch.empty if nnz else torch.zeros
            part_re = alloc(n_parts, dtype=real, device=y.device)
            part_im = alloc(n_parts, dtype=real, device=y.device)
            if nnz:
                _spvv_complex_kernel[(n_parts,)](
                    part_re, part_im,
                    torch.view_as_real(values).reshape(-1),
                    torch.view_as_real(y).reshape(-1),
                    kernel_idx, nnz, CONJ=(op == "conj"), BLOCK=_VECTOR_BLOCK,
                )
            result = torch.complex(part_re.sum(), part_im.sum())
        else:
            # ``op="conj"`` on real data is the identity, as in cuSPARSE.
            part = (torch.empty if nnz else torch.zeros)(n_parts, dtype=compute, device=y.device)
            if nnz:
                acc = {torch.int32: tl.int32, torch.float64: tl.float64}.get(compute, tl.float32)
                _spvv_real_kernel[(n_parts,)](
                    part, values, y, kernel_idx, nnz, ACC=acc, BLOCK=_VECTOR_BLOCK
                )
            result = part.sum(dtype=compute)
    if return_time:
        _ACCEL.synchronize()
        return result, (time.perf_counter() - t0) * 1000.0
    return result
