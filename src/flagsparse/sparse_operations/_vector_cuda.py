# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CUDA sparse-vector updates with one writer per dense output element."""

from types import SimpleNamespace
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from ._musa_spmv_launch import fast_path_allowed, launch
from ._rocm_sparse_cache import TensorPlanCache
from ._vector_maca import _axpby_complex, _inverse


_OWNERS = TensorPlanCache()
_HOT = None
_RAW_STREAM = None
_DOT_HOT = None
_DOT_SCRATCH = OrderedDict()


def _stream(device):
    global _RAW_STREAM
    if _RAW_STREAM is None:
        from torch._C import _cuda_getCurrentRawStream

        _RAW_STREAM = _cuda_getCurrentRawStream
    return _RAW_STREAM(device.index)


def _signature(values, indices, y):
    try:
        return (
            values.data_ptr(),
            indices.data_ptr(),
            y.data_ptr(),
            values.dtype,
            indices.dtype,
            y.dtype,
            values.device,
            indices.device,
            y.device,
            values.shape,
            indices.shape,
            y.shape,
            values.stride(),
            indices.stride(),
            y.stride(),
            indices._version,
        )
    except RuntimeError:
        return None


def try_axpby(values, indices, y, alpha, beta):
    """Reuse validation/compilation only while tensor metadata is unchanged."""
    global _HOT
    hot = _HOT
    if hot is None or not (
        values is hot.values and indices is hot.indices and y is hot.y
    ):
        return False
    if not isinstance(alpha, (int, float, complex, type(None))) or not isinstance(
        beta, (int, float, complex, type(None))
    ):
        return False
    if (
        complex(1 if alpha is None else alpha) != hot.alpha
        or complex(1 if beta is None else beta) != hot.beta
    ):
        return False
    if _signature(values, indices, y) != hot.signature:
        _HOT = None
        return False
    stream = _RAW_STREAM(hot.device_index)
    if stream != hot.stream:
        return False
    if not fast_path_allowed(hot.bound.kernel):
        return False
    if hot.raw is not None:
        try:
            hot.raw(
                *hot.bound.grid,
                stream,
                hot.bound.compiled.function,
                False,
                False,
                None,
                None,
                *hot.args,
            )
            return True
        except TypeError:
            # Signature rejection happens before launch; retain the supported
            # compiled wrapper on Triton versions with a different raw ABI.
            hot.raw = None
    hot.bound(hot.tensors, hot.scalars, hot.constants, hot.addresses, checked=True)
    return True


@triton.jit
def _axpby_real(Y, V, OWNER, N, ALPHA, BETA, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < N
    source = tl.load(OWNER + row, valid, -1)
    value = tl.load(V + source, valid & (source >= 0), 0.0).to(tl.float32)
    previous = tl.load(Y + row, valid, 0.0).to(tl.float32)
    update = tl.where(source >= 0, ALPHA * value, 0.0)
    tl.store(Y + row, BETA * previous + update, valid)


def axpby(values, indices, y, ar, ai, br, bi):
    """Reuse only index ownership; read current values and y on every call."""
    global _HOT
    n = y.numel()
    if not n or (not values.numel() and (br, bi) == (1.0, 0.0)):
        return
    stream = _stream(y.device)
    owner = _OWNERS.get((indices,), (n, stream))
    if owner is None:
        owner = torch.full((n,), -1, dtype=torch.int32, device=y.device)
        nnz = indices.numel()
        if nnz:
            _inverse[(triton.cdiv(nnz, 256),)](
                owner, indices.contiguous(), nnz, n, 256, num_warps=4
            )
        _OWNERS.put((indices,), (n, stream), owner)
    if y.is_complex():
        kernel, grid = _axpby_complex, (triton.cdiv(n, 512),)
        tensors = (torch.view_as_real(y), torch.view_as_real(values), owner)
        scalars, constants = (n, ar, ai, br, bi), (ai != 0.0, bi != 0.0, 512)
    else:
        kernel, grid = _axpby_real, (triton.cdiv(n, 1024),)
        tensors, scalars, constants = (y, values, owner), (n, ar, br), (1024,)
    if not hasattr(triton, "knobs"):
        kernel[grid](*tensors, *scalars, *constants, num_warps=4)
        return
    bound = launch(kernel, grid, tensors, scalars, constants, num_warps=4, cuda=True)
    signature = _signature(values, indices, y)
    if bound is not None and bound.direct and signature is not None:
        entry = bound.compiled.run
        raw = (
            entry.launch
            if hasattr(entry, "launch")
            and getattr(entry, "global_scratch_size", None) == 0
            and getattr(entry, "profile_scratch_size", None) == 0
            and getattr(entry, "launch_cooperative_grid", None) is False
            and getattr(entry, "launch_pdl", None) is False
            else None
        )
        addresses = tuple(t.data_ptr() for t in tensors)
        _HOT = SimpleNamespace(
            values=values,
            indices=indices,
            y=y,
            signature=signature,
            alpha=complex(ar, ai),
            beta=complex(br, bi),
            bound=bound,
            tensors=tensors,
            scalars=scalars,
            constants=constants,
            addresses=addresses,
            stream=stream,
            device_index=y.device.index,
            raw=raw,
            args=(
                bound.compiled.packed_metadata,
                None,
                None,
                None,
                *addresses,
                *scalars,
                *constants,
            ),
        )


@triton.jit
def _dot(
    V,
    INDICES,
    Y,
    P,
    COUNT,
    OUT,
    NNZ,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
    NPARTS: tl.constexpr,
    REDUCE: tl.constexpr,
):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < NNZ
    idx = tl.load(INDICES + pos, valid, 0)
    if COMPLEX:
        parts = tl.arange(0, 2)
        vp = tl.load(V + 2 * pos[:, None] + parts[None, :], valid[:, None], 0).to(ACC)
        yp = tl.load(Y + 2 * idx[:, None] + parts[None, :], valid[:, None], 0).to(ACC)
        vr, vi = tl.split(vp)
        yr, yi = tl.split(yp)
        if CONJ:
            vi = -vi
        tl.store(P + 2 * tl.program_id(0), tl.sum(vr * yr - vi * yi, 0))
        tl.store(P + 2 * tl.program_id(0) + 1, tl.sum(vr * yi + vi * yr, 0))
    else:
        v = tl.load(V + pos, valid, 0).to(ACC)
        y = tl.load(Y + idx, valid, 0).to(ACC)
        tl.store(P + tl.program_id(0), tl.sum(v * y, 0))
    if NPARTS > 1:
        # Publish every partial before taking a ticket. The last publisher
        # performs an ordered reduction; no block waits for unscheduled work.
        tl.debug_barrier()
        ticket = tl.atomic_add(COUNT, 1, sem="acq_rel")
        if ticket == NPARTS - 1:
            offsets = tl.arange(0, REDUCE)
            width: tl.constexpr = 2 if COMPLEX else 1
            re = tl.load(P + width * offsets, offsets < NPARTS, 0, volatile=True)
            tl.store(OUT, tl.sum(re, 0))
            if COMPLEX:
                im = tl.load(P + 2 * offsets + 1, offsets < NPARTS, 0, volatile=True)
                tl.store(OUT + 1, tl.sum(im, 0))
            tl.debug_barrier()
            tl.atomic_xchg(COUNT, 0, sem="release")


def spvv(values, indices, y, conjugate, dtype):
    """Deterministic reductions with a single launch for small sparse dots."""
    global _DOT_HOT
    if values.device.index != torch._C._cuda_getDevice():
        with torch.cuda.device(values.device):
            return spvv(values, indices, y, conjugate, dtype)
    sources = (values, indices, y)
    n = values.numel()
    result = torch.empty((), dtype=dtype, device=y.device)
    complex_ = values.is_complex()
    acc = {
        torch.float64: tl.float64,
        torch.complex128: tl.float64,
        torch.int32: tl.int32,
    }.get(dtype, tl.float32)
    block = max(32, triton.next_power_of_2(n)) if n <= 16384 else 512
    parts = max(1, triton.cdiv(n, block))
    target = torch.view_as_real(result) if complex_ else result
    partial, counter = _dot_workspace(y, target, parts, complex_)
    values, y = values.resolve_conj().resolve_neg(), y.resolve_conj().resolve_neg()
    tensors = (
        torch.view_as_real(values) if complex_ else values,
        indices,
        torch.view_as_real(y) if complex_ else y,
        partial,
        counter,
        target,
    )
    constants = (complex_, conjugate, acc, block, parts, triton.next_power_of_2(parts))
    first = launch(
        _dot,
        (parts,),
        tensors,
        (n,),
        constants,
        num_warps=8 if block > 4096 else 4,
        cuda=True,
    )
    signature = _signature(*sources)
    if (
        first is not None
        and not torch.cuda.is_current_stream_capturing()
        and indices.dtype == torch.int32
        and signature is not None
        and all(not t.is_conj() and not t.is_neg() for t in sources)
    ):
        _DOT_HOT = SimpleNamespace(
            sources=sources,
            signature=signature,
            dtype=dtype,
            conjugate=conjugate,
            n=n,
            parts=parts,
            complex=complex_,
            first=first,
            inputs=tensors[:3],
            constants=constants,
            workspace=(partial, counter) if parts > 1 else None,
            stream=_stream(y.device),
            addresses=tuple(t.data_ptr() for t in tensors[:5]),
        )
    return result


def try_spvv(values, indices, y, op, out_dtype):
    hot = _DOT_HOT
    if hot is None or not (
        values is hot.sources[0] and indices is hot.sources[1] and y is hot.sources[2]
    ):
        return None
    if y.device.index != torch._C._cuda_getDevice():
        return None
    if op not in ("non", "conj") or (op == "conj") != hot.conjugate:
        return None
    if out_dtype is not None and out_dtype != hot.dtype:
        return None
    default = {
        torch.float16: torch.float32,
        torch.bfloat16: torch.float32,
        torch.int8: torch.int32,
    }.get(values.dtype, values.dtype)
    if out_dtype is None and default != hot.dtype:
        return None
    if (
        _signature(values, indices, y) != hot.signature
        or values.is_conj()
        or y.is_conj()
        or values.is_neg()
        or y.is_neg()
        or not fast_path_allowed(hot.first.kernel)
        or _stream(y.device) != hot.stream
        or (hot.workspace is not None and torch.cuda.is_current_stream_capturing())
    ):
        return None
    result = torch.empty((), dtype=hot.dtype, device=y.device)
    raw = hot.first.cuda_entry
    if raw is not None:
        address = result.data_ptr()
        addresses = (
            *hot.addresses[:3],
            *(hot.addresses[3:] if hot.workspace is not None else (address, address)),
            address,
        )
        entry, function, metadata = raw
        try:
            entry(
                *hot.first.grid,
                hot.stream,
                function,
                False,
                False,
                None,
                None,
                metadata,
                None,
                None,
                None,
                *addresses,
                hot.n,
                *hot.constants,
            )
            return result
        except TypeError:
            hot.first.cuda_entry = None
    target = torch.view_as_real(result) if hot.complex else result
    partial, counter = (target, target) if hot.workspace is None else hot.workspace
    hot.first(
        (*hot.inputs, partial, counter, target), (hot.n,), hot.constants, checked=True
    )
    return result


def _dot_workspace(y, target, parts, complex_):
    if parts == 1:
        return target, target
    key = (y.device, _stream(y.device), target.dtype, parts, complex_)
    capturing = torch.cuda.is_current_stream_capturing()
    scratch = None if capturing else _DOT_SCRATCH.get(key)
    if scratch is None:
        scratch = (
            torch.empty(
                (parts, 2) if complex_ else (parts,),
                dtype=target.dtype,
                device=y.device,
            ),
            torch.zeros((), dtype=torch.int32, device=y.device),
        )
        if not capturing:
            _DOT_SCRATCH[key] = scratch
            while len(_DOT_SCRATCH) > 8:
                _DOT_SCRATCH.popitem(last=False)
    else:
        _DOT_SCRATCH.move_to_end(key)
    return scratch


@triton.jit
def _scatter_reset(Y, V, OWNER, N, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    parts = tl.arange(0, 2)
    valid = row < N
    pos = tl.load(OWNER + row, valid, -1)
    pair = tl.load(
        V + 2 * pos[:, None] + parts[None, :], (valid & (pos >= 0))[:, None], 0.0
    )
    tl.store(Y + 2 * row[:, None] + parts[None, :], pair, valid[:, None])


def scatter_reset(values, indices, y):
    """Fuse reset and scatter by assigning one writer to every dense element."""
    n = y.numel()
    values = values.resolve_conj().resolve_neg()
    if not n:
        return y
    options = (n, _stream(y.device))
    owner = _OWNERS.get((indices,), options)
    if owner is None:
        owner = torch.full((n,), -1, dtype=torch.int32, device=y.device)
        if indices.numel():
            _inverse[(triton.cdiv(indices.numel(), 256),)](
                owner, indices, indices.numel(), n, 256, num_warps=4
            )
        # Captured initialisation has not executed yet. Its private buffers
        # must not be reused by an eager call made before graph replay.
        if not torch.cuda.is_current_stream_capturing():
            _OWNERS.put((indices,), options, owner)
    launch(
        _scatter_reset,
        (triton.cdiv(n, 256),),
        (torch.view_as_real(y), torch.view_as_real(values), owner),
        (n,),
        (256,),
        num_warps=4,
        cuda=True,
    )
    return y
