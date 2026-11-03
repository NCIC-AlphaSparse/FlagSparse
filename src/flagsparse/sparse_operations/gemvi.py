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

"""Dense matrix times sparse vector (cusparse<t>gemvi).

``y = alpha * op(A) @ x + beta * y`` with ``A`` dense and ``x = (values, indices)``
a sparse vector over ``op(A)``'s columns. ``op`` is ``"non"`` or ``"trans"``;
cuSPARSE's gemvi has no conjugate transpose, so neither does this.

``A`` is read through its strides: cuSPARSE wants it column-major (``lda >= m``),
which is the layout the kernels coalesce best on, but any 2D view works.

- ``non``: each program owns a block of output rows and gathers, per stored
  ``x`` entry, the matching column of ``A`` (contiguous for column-major ``A``).
- ``trans``: each program owns a block of output columns and gathers the rows of
  ``A`` that ``x`` selects.

Neither direction writes the same output element twice, so there are no atomics.
Complex values go through ``view_as_real``. Ascend takes a torch_npu path.

On CUDA the kernels go through the cached direct launcher (_musa_spmv_launch):
Triton's JIT dispatch costs more host time than a short gemvi runs.
"""

from ._common import *

import time

import triton
import triton.language as tl

from ._musa_spmv_launch import launch as _launch_direct
from .mixed_spmx import _CUDA_DIRECT_LAUNCH

GEMVI_VALUE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
_BLOCK_OUT = 64
_BLOCK_K = 32


def _normalize_gemvi_op(op):
    token = "non" if op is None else str(op).strip().lower()
    token = {"n": "non", "non_trans": "non", "0": "non", "t": "trans", "1": "trans"}.get(
        token, token
    )
    if token in ("conj", "c", "conj_trans", "2"):
        raise ValueError("gemvi has no conjugate transpose (cusparse<t>gemvi has none)")
    if token not in ("non", "trans"):
        raise ValueError("op must be 'non' or 'trans'")
    return token


@triton.jit
def _gemvi_real_kernel(
    y_ptr, a_ptr, x_ptr, idx_ptr, n_out, nnz,
    stride_out, stride_k, alpha, beta, k_chunk, part_stride,
    HAS_BETA: tl.constexpr, ACC: tl.constexpr,
    BLOCK_OUT: tl.constexpr, BLOCK_K: tl.constexpr, OUT_INNER: tl.constexpr,
):
    # stride_out walks op(A)'s rows (the outputs), stride_k its columns, so the
    # same kernel serves both directions: trans just swaps the two strides.
    # Axis 1 splits x's entries into k_chunk-long runs; with more than one run
    # each writes a partial sum at y_ptr + split * part_stride (alpha = 1, no
    # beta) and _gemvi_reduce_kernel finishes.
    split = tl.program_id(1)
    k_begin = split * k_chunk
    k_end = tl.minimum(nnz, k_begin + k_chunk)
    out = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    out_mask = out < n_out
    acc = tl.zeros((BLOCK_OUT,), dtype=ACC)
    for k0 in range(k_begin, k_end, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_mask = ks < k_end
        idx = tl.load(idx_ptr + ks, mask=k_mask, other=0)
        xv = tl.load(x_ptr + ks, mask=k_mask, other=0).to(ACC)
        if OUT_INNER:
            # Outputs are adjacent in memory (column-major A, op='non'): put them on
            # the tile's last axis, which is the one Triton spreads lanes across.
            a = tl.load(
                a_ptr + idx[:, None].to(tl.int64) * stride_k + out[None, :] * stride_out,
                mask=k_mask[:, None] & out_mask[None, :],
                other=0,
            ).to(ACC)
            acc += tl.sum(a * xv[:, None], axis=0)
        else:
            a = tl.load(
                a_ptr + out[:, None] * stride_out + idx[None, :].to(tl.int64) * stride_k,
                mask=out_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(ACC)
            acc += tl.sum(a * xv[None, :], axis=1)
    y_ptr += split * part_stride
    result = acc * alpha
    if HAS_BETA:
        result += tl.load(y_ptr + out, mask=out_mask, other=0).to(ACC) * beta
    tl.store(y_ptr + out, result.to(y_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def _gemvi_complex_kernel(
    y_ri_ptr, a_ri_ptr, x_ri_ptr, idx_ptr, n_out, nnz,
    stride_out, stride_k, alpha_re, alpha_im, beta_re, beta_im, k_chunk, part_stride,
    HAS_BETA: tl.constexpr, ACC: tl.constexpr,
    BLOCK_OUT: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Strides are in real elements of the (..., 2) view; the imaginary part is +1.
    split = tl.program_id(1)
    k_begin = split * k_chunk
    k_end = tl.minimum(nnz, k_begin + k_chunk)
    out = tl.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    out_mask = out < n_out
    acc_re = tl.zeros((BLOCK_OUT,), dtype=ACC)
    acc_im = tl.zeros((BLOCK_OUT,), dtype=ACC)
    for k0 in range(k_begin, k_end, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_mask = ks < k_end
        idx = tl.load(idx_ptr + ks, mask=k_mask, other=0)
        x_re = tl.load(x_ri_ptr + 2 * ks, mask=k_mask, other=0.0).to(ACC)
        x_im = tl.load(x_ri_ptr + 2 * ks + 1, mask=k_mask, other=0.0).to(ACC)
        # Outputs on the tile's first axis even for column-major A: measured faster
        # than the real kernel's OUT_INNER layout or a (re, im) trailing axis
        # (RTX 5090, 16384x8192 / 65536x4096 / 4096x65536: 1.0x vs 0.6-0.8x cuSPARSE).
        base = a_ri_ptr + out[:, None] * stride_out + idx[None, :].to(tl.int64) * stride_k
        mask = out_mask[:, None] & k_mask[None, :]
        a_re = tl.load(base, mask=mask, other=0.0).to(ACC)
        a_im = tl.load(base + 1, mask=mask, other=0.0).to(ACC)
        acc_re += tl.sum(a_re * x_re[None, :] - a_im * x_im[None, :], axis=1)
        acc_im += tl.sum(a_re * x_im[None, :] + a_im * x_re[None, :], axis=1)
    y_ri_ptr += split * part_stride
    res_re = acc_re * alpha_re - acc_im * alpha_im
    res_im = acc_re * alpha_im + acc_im * alpha_re
    if HAS_BETA:
        y_re = tl.load(y_ri_ptr + 2 * out, mask=out_mask, other=0.0).to(ACC)
        y_im = tl.load(y_ri_ptr + 2 * out + 1, mask=out_mask, other=0.0).to(ACC)
        res_re += y_re * beta_re - y_im * beta_im
        res_im += y_re * beta_im + y_im * beta_re
    tl.store(y_ri_ptr + 2 * out, res_re, mask=out_mask)
    tl.store(y_ri_ptr + 2 * out + 1, res_im, mask=out_mask)


@triton.jit
def _gemvi_reduce_kernel(
    y_ptr, part_ptr, n_out, n_split, alpha_re, alpha_im, beta_re, beta_im,
    HAS_BETA: tl.constexpr, COMPLEX: tl.constexpr, ACC: tl.constexpr, BLOCK: tl.constexpr,
):
    """y = alpha * sum(partials) + beta * y; complex arrays are (re, im) views."""
    out = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = out < n_out
    width = 2 if COMPLEX else 1
    acc_re = tl.zeros((BLOCK,), dtype=ACC)
    acc_im = tl.zeros((BLOCK,), dtype=ACC)
    for s in range(n_split):
        base = part_ptr + s * width * n_out + width * out
        acc_re += tl.load(base, mask=mask, other=0.0).to(ACC)
        if COMPLEX:
            acc_im += tl.load(base + 1, mask=mask, other=0.0).to(ACC)
    res_re = acc_re * alpha_re - acc_im * alpha_im
    res_im = acc_re * alpha_im + acc_im * alpha_re
    if HAS_BETA:
        y_re = tl.load(y_ptr + width * out, mask=mask, other=0.0).to(ACC)
        res_re += y_re * beta_re
        res_im += y_re * beta_im
        if COMPLEX:
            y_im = tl.load(y_ptr + 2 * out + 1, mask=mask, other=0.0).to(ACC)
            res_re -= y_im * beta_im
            res_im += y_im * beta_re
    tl.store(y_ptr + width * out, res_re.to(y_ptr.dtype.element_ty), mask=mask)
    if COMPLEX:
        tl.store(y_ptr + 2 * out + 1, res_im.to(y_ptr.dtype.element_ty), mask=mask)


def _cdiv(a, b):
    # triton.cdiv is a ConstexprFunction since Triton 3.6: ~1 us per host call,
    # and this path makes five of them per gemvi.
    return -(-a // b)


_SM_COUNT = {}


def _sm_count(device):
    key = (device.type, device.index)
    if key not in _SM_COUNT:
        try:
            props = _ACCEL.get_device_properties(device)
            _SM_COUNT[key] = int(getattr(props, "multi_processor_count", 0)) or 64
        except Exception:
            _SM_COUNT[key] = 64
    return _SM_COUNT[key]


# Below this many bytes of A to read the call is host-bound (a few microseconds of
# GPU work): a second launch for the reduction would cost more than it saves.
_SPLIT_MIN_BYTES = 16 << 20


def _split_count(n_out, nnz, element_size, device):
    """How many runs to cut x's entries into so the grid fills the device.

    One program per BLOCK_OUT outputs leaves most SMs idle once n_out is a few
    thousand. Measured on an RTX 5090 (170 SMs), float32, column-major A:
    4096 x 65536 with 4096 entries took 97 us unsplit, 22 us in 5 runs and 14 us
    in 10-20; 16384 x 8192 with 2048 entries 91 us unsplit, 75 us in 5 runs. So
    aim for ~8 programs per SM, keep each run several BLOCK_K steps long, and
    do not split a call too small to be anything but host-bound.
    """
    if n_out * nnz * element_size <= _SPLIT_MIN_BYTES:
        return 1
    out_blocks = _cdiv(n_out, _BLOCK_OUT)
    target = 8 * _sm_count(device)
    if out_blocks >= target:
        return 1
    return max(1, min(target // out_blocks, nnz // (4 * _BLOCK_K)))


def _launch(kernel, grid, tensors, scalars, constants):
    if _CUDA_DIRECT_LAUNCH:
        _launch_direct(kernel, grid, tensors, scalars, constants, num_warps=4, cuda=True)
    else:
        kernel[grid](*tensors, *scalars, *constants, num_warps=4)


def _validate_gemvi(A, values, indices, y, op, beta, check_range):
    for tensor, label in ((A, "A"), (values, "values"), (indices, "indices")):
        if not torch.is_tensor(tensor):
            raise TypeError(f"{label} must be a torch.Tensor")
        if not _is_accel_tensor(tensor):
            raise ValueError(f"{label} must be an accelerator tensor")
    if A.ndim != 2:
        raise ValueError("A must be a 2D tensor")
    if values.ndim != 1 or indices.ndim != 1 or values.numel() != indices.numel():
        raise ValueError("values and indices must be 1D tensors of the same length")
    if A.dtype not in GEMVI_VALUE_DTYPES:
        raise TypeError(f"A dtype must be one of {GEMVI_VALUE_DTYPES}")
    if values.dtype != A.dtype:
        raise TypeError("values dtype must match A dtype")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if values.device != A.device or indices.device != A.device:
        raise ValueError("A, values and indices must be on the same device")
    m, n = int(A.shape[0]), int(A.shape[1])
    n_out, n_in = (m, n) if op == "non" else (n, m)
    if y is None:
        if beta != 0:
            raise ValueError("y is required when beta is non-zero")
    else:
        if not torch.is_tensor(y) or y.ndim != 1 or y.numel() != n_out:
            raise ValueError(f"y must be a 1D tensor of length {n_out}")
        if y.dtype != A.dtype or y.device != A.device:
            raise ValueError("y must match A's dtype and device")
    if check_range and indices.numel():
        bounds = torch.stack((indices.min(), indices.max())).to(torch.int64).cpu()
        if int(bounds[0]) < 0 or int(bounds[1]) >= n_in:
            raise IndexError(f"indices must lie in [0, {n_in})")
    if n_in > _INDEX_LIMIT_INT32 and indices.dtype == torch.int32:
        raise ValueError("int32 indices cannot address this A")
    return n_out


def _torch_gemvi(A, values, indices, y, op, alpha, beta):
    idx = indices.to(torch.int64)
    if op == "non":
        result = A.index_select(1, idx) @ values
    else:
        result = values @ A.index_select(0, idx)
    result = result * alpha
    if y is not None and beta != 0:
        result = result + y * beta
    return result


def flagsparse_gemvi(
    A,
    values,
    indices,
    y=None,
    *,
    op="non",
    alpha=1.0,
    beta=0.0,
    validate=True,
    return_time=False,
):
    """``y = alpha * op(A) @ x + beta * y`` for dense ``A``, sparse ``x`` (cusparse<t>gemvi).

    ``op="non"``: ``A`` is m x n, ``x`` indexes n columns, ``y`` has m entries;
    ``op="trans"``: ``x`` indexes m rows, ``y`` has n entries. float32, float64,
    complex64 and complex128, accumulated in the value type. ``y`` is updated in
    place and returned when given; it is required when ``beta != 0``.
    ``validate=False`` skips the index range check (a host sync).
    """
    op = _normalize_gemvi_op(op)
    n_out = _validate_gemvi(A, values, indices, y, op, beta, validate)
    t0 = None
    if return_time:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    if _is_ascend_runtime():
        result = _torch_gemvi(A, values, indices, y, op, alpha, beta)
        if y is not None:
            y.copy_(result)
            result = y
    else:
        target = y if y is not None and y.is_contiguous() else torch.empty(
            n_out, dtype=A.dtype, device=A.device
        )
        if y is not None and target is not y and beta != 0:
            target.copy_(y)
        nnz = int(values.numel())
        idx = indices.to(torch.int32) if indices.dtype == torch.int64 else indices
        idx = idx.contiguous()
        x = values.contiguous()
        s0, s1 = A.stride()
        stride_out, stride_k = (s0, s1) if op == "non" else (s1, s0)
        has_beta = beta != 0
        if n_out:
            complex_ = A.is_complex()
            if complex_:
                acc = tl.float64 if A.dtype == torch.complex128 else tl.float32
                part_dtype = torch.float64 if A.dtype == torch.complex128 else torch.float32
            else:
                acc = tl.float64 if A.dtype == torch.float64 else tl.float32
                part_dtype = torch.float64 if A.dtype == torch.float64 else torch.float32
            a_c, b_c = complex(alpha), complex(beta)
            n_split = _split_count(n_out, nnz, A.element_size(), A.device)
            k_chunk = _cdiv(max(1, nnz), n_split)
            k_chunk = _cdiv(k_chunk, _BLOCK_K) * _BLOCK_K
            n_split = _cdiv(max(1, nnz), k_chunk)
            width = 2 if complex_ else 1
            if n_split > 1:
                part = torch.empty(n_split * n_out * width, dtype=part_dtype, device=A.device)
                first_out, first_alpha, first_beta, first_has_beta = part, 1.0, 0.0, False
                part_stride = n_out * width
            else:
                first_out = torch.view_as_real(target) if complex_ else target
                first_alpha, first_beta, first_has_beta = a_c, b_c, has_beta
                part_stride = 0
            if complex_:
                kernel = _gemvi_complex_kernel
                fa, fb = complex(first_alpha), complex(first_beta)
                tensors = (first_out, torch.view_as_real(A), torch.view_as_real(x), idx)
                scalars = (n_out, nnz, 2 * stride_out, 2 * stride_k,
                           fa.real, fa.imag, fb.real, fb.imag, k_chunk, part_stride)
            else:
                kernel = _gemvi_real_kernel
                tensors = (first_out, A, x, idx)
                scalars = (n_out, nnz, stride_out, stride_k,
                           float(complex(first_alpha).real), float(complex(first_beta).real),
                           k_chunk, part_stride)
            constants = (first_has_beta, acc, _BLOCK_OUT, _BLOCK_K)
            if not complex_:
                # OUT_INNER: unit stride along the outputs, i.e. column-major A with
                # op='non' or row-major A with op='trans'.
                constants += (stride_out == 1,)
            grid = (_cdiv(n_out, _BLOCK_OUT), n_split)
            _launch(kernel, grid, tensors, scalars, constants)
            if n_split > 1:
                y_view = torch.view_as_real(target) if complex_ else target
                _launch(
                    _gemvi_reduce_kernel, (_cdiv(n_out, 256),), (y_view, part),
                    (n_out, n_split, a_c.real, a_c.imag, b_c.real, b_c.imag),
                    (has_beta, complex_, acc, 256),
                )
        if y is not None and target is not y:
            y.copy_(target)
            target = y
        result = target
    if return_time:
        _ACCEL.synchronize()
        return result, (time.perf_counter() - t0) * 1000.0
    return result
