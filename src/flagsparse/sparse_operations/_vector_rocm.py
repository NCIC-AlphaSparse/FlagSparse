# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused sparse-vector reductions and updates on ROCm."""

import torch
import triton
import triton.language as tl

from ._rocm_launch import can_launch, launch as _launch
from ._rocm_sparse_cache import TensorPlanCache
from ._vector_cuda import _axpby_real, _dot, _dot_workspace, _stream
from ._vector_maca import _axpby_complex, _inverse


_OWNERS = TensorPlanCache()
_LAST = {}


def _signature(tensors):
    return tuple((t.data_ptr(), t.dtype, t.device, t.shape, t.stride(), t.is_conj(), t.is_neg()) for t in tensors)


def _remember(kind, runner, sources, options, inputs, extra, partial=None):
    if runner is None or torch.cuda.is_current_stream_capturing():
        return
    # Views of contiguous operands share live storage. Never bind a converted copy.
    if any(not t.is_contiguous() or t.is_conj() or t.is_neg() for t in sources):
        return
    try:
        version = sources[1]._version if kind == "axpby" else None
    except RuntimeError:
        return
    stream = _stream(sources[0].device)
    _LAST[kind] = (sources, options, _signature(sources), stream, version,
                   runner, inputs, extra, partial)


def try_bound(kind, values, indices, y, options):
    entry = _LAST.get(kind)
    if entry is None:
        return None
    sources, saved, signature, stream, version, runner, inputs, extra, partial = entry
    if not (values is sources[0] and indices is sources[1] and y is sources[2]
            and options == saved and _signature(sources) == signature):
        return None
    if (torch.cuda.current_device() != y.device.index
            or _stream(y.device) != stream
            or torch.cuda.is_current_stream_capturing()):
        return None
    if kind == "axpby":
        kernel = _axpby_complex if y.is_complex() else _axpby_real
        if version != indices._version or not can_launch(kernel):
            return None
        runner(*inputs, *extra, stream=stream)
        return y
    if not can_launch(_dot):
        return None
    dtype, parts, n = extra
    result = torch.empty((), dtype=dtype, device=y.device)
    # The compiled launcher reads addresses. A complex scalar already owns
    # the same two adjacent floats as its real view.
    work, counter = partial if parts > 1 else (result, result)
    runner(*inputs, work, counter, result, n, stream=stream)
    return result


def spvv(values, indices, y, conjugate, dtype, requested_dtype):
    result = torch.empty((), dtype=dtype, device=y.device)
    n = values.numel()
    block = 256
    parts = max(1, triton.cdiv(n, block))
    complex_ = values.is_complex()
    target = torch.view_as_real(result) if complex_ else result
    partial, counter = _dot_workspace(y, target, parts, complex_)
    x = values.resolve_conj().resolve_neg()
    dense = y.resolve_conj().resolve_neg()
    if complex_:
        x, dense = torch.view_as_real(x), torch.view_as_real(dense)
    acc = tl.int32 if dtype == torch.int32 else tl.float32
    runner = _launch(_dot, parts, (x, indices.contiguous(), dense, partial, counter, target), (n,),
            (complex_, conjugate, acc, block, parts, triton.next_power_of_2(parts)))
    if indices.is_contiguous():
        _remember("spvv", runner, (values, indices, y), ("conj" if conjugate else "non", requested_dtype),
                  (x, indices, dense), (dtype, parts, n), (partial, counter))
    return result


def axpby(values, indices, y, ar, ai, br, bi):
    n = y.numel()
    if not n:
        return
    if torch.cuda.current_device() != y.device.index:
        with torch.cuda.device(y.device):
            return axpby(values, indices, y, ar, ai, br, bi)
    capturing = torch.cuda.is_current_stream_capturing()
    stream = _stream(y.device)
    # Capture the inverse itself so each replay reads the graph's current indices.
    owner = None if capturing else _OWNERS.get((indices,), (n, stream))
    if owner is None:
        owner = torch.full((n,), -1, dtype=torch.int32, device=y.device)
        if indices.numel():
            _inverse[(triton.cdiv(indices.numel(), 256),)](
                owner, indices.contiguous(), indices.numel(), n, 256, num_warps=4
            )
        if not capturing:
            _OWNERS.put((indices,), (n, stream), owner)
    if y.is_complex():
        kernel = _axpby_complex
        tensors = (torch.view_as_real(y), torch.view_as_real(values), owner)
        scalars, constants = (n, ar, ai, br, bi), (ai != 0., bi != 0., 512)
    else:
        kernel = _axpby_real
        tensors = (y, values, owner)
        scalars, constants = (n, ar, br), (1024, y.dtype == torch.float16)
    block = constants[-1] if y.is_complex() else constants[0]
    runner = _launch(kernel, triton.cdiv(n, block), tensors, scalars, constants)
    _remember("axpby", runner, (values, indices, y), (complex(ar, ai), complex(br, bi)),
              tensors, scalars)
