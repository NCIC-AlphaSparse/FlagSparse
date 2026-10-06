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


def _torch_axpby(values, indices, y, alpha, beta):
    """Ascend path, also the reference shape of the Triton path."""
    compute = _vector_compute_dtype(y.dtype)
    acc = y.to(compute) if compute != y.dtype else y.clone()
    acc.mul_(beta)
    idx = indices.to(torch.int64)
    update = values.to(compute) * alpha
    if _is_complex_dtype(compute):
        # index_add_ on complex has no NPU/MUSA kernel; add the two real planes.
        torch.view_as_real(acc).index_add_(0, idx, torch.view_as_real(update))
    else:
        acc.index_add_(0, idx, update)
    y.copy_(acc)
    return y


def flagsparse_axpby(
    values, indices, y, alpha=1.0, beta=1.0, return_time=False, *, validate=True
):
    """In-place ``y = alpha * x + beta * y`` for sparse ``x = (values, indices)``.

    Mirrors ``cusparseAxpby``; half types are computed in float32. Returns ``y``
    (and the elapsed milliseconds when ``return_time``). ``validate=False`` skips the
    index range check, a host sync that cuSPARSE does not perform either.
    """
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
    if _is_ascend_runtime():
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
        # Real-plane gathers and sums: no complex index/sum kernel needed.
        x_ri = torch.view_as_real(values)
        y_ri = torch.view_as_real(y).index_select(0, idx)
        xr, xi = x_ri[:, 0], x_ri[:, 1]
        if op == "conj":
            xi = -xi
        yr, yi = y_ri[:, 0], y_ri[:, 1]
        return torch.complex((xr * yr - xi * yi).sum(), (xr * yi + xi * yr).sum())
    prod = values.to(compute) * y.index_select(0, idx).to(compute)
    return prod.sum(dtype=compute)


def flagsparse_spvv(values, indices, y, op="non", return_time=False, *, validate=True):
    """Sparse-dense dot product ``sum(op(x)[k] * y[indices[k]])`` (``cusparseSpVV``).

    ``op`` is ``"non"`` or ``"conj"``. Returns a 0-d tensor of the compute dtype:
    float32 for float16/bfloat16, int32 for int8, otherwise the value dtype.
    ``validate=False`` skips the index range check (a host sync).
    """
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
    if _is_ascend_runtime():
        result = _torch_spvv(values, kernel_idx, y, op)
    else:
        nnz = values.numel()
        n_parts = max(1, triton.cdiv(nnz, _VECTOR_BLOCK))
        real = _real_dtype(compute)
        if _is_complex_dtype(compute) and _backend_name() == "cuda" and n_parts <= 1024:
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
