# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CSR products on K100 using a fresh workgroup-local hash per output row.

FlagTree 3.1 on HCU supports Triton extern libraries. The library below
provides shared-memory allocation and atomic operations for the hash table.
Triton handles synchronization, tiling, and global memory operations.
"""

import hashlib
from functools import lru_cache

import torch
import triton
import triton.language as tl

from triton.language import core
from triton.runtime.cache import get_cache_manager

from ._rocm_sparse_cache import TensorPlanCache

_OPERANDS = TensorPlanCache(maxsize=4)


@lru_cache(maxsize=16)
def available(device):
    if not triton.__version__.startswith("3.1."):
        return False
    with torch.cuda.device(device):
        target = triton.runtime.driver.active.get_current_target()
        return target.backend == "hip" and target.arch == "gfx926"


def operand(data, cols, ptr, shape, tag):
    from .spgemm_csr import _validate_csr

    stream = torch.cuda.current_stream(data.device).cuda_stream
    options = (tuple(shape), data.shape, data.dtype, data.device, stream)
    state = _OPERANDS.get((cols, ptr), options)
    if state is None:
        m, n, pointer = _validate_csr(data, cols, ptr, shape, tag)
        pointer = pointer.contiguous()
        maximum = int((pointer[1:] - pointer[:-1]).max()) if m else 0
        state = cols.contiguous(), pointer, (m, n), maximum
        _OPERANDS.put((cols, ptr), options, state)
    return state


def config(a_shape, b_shape, nnz_a, nnz_b, max_a, max_b):
    m, k = a_shape
    n = b_shape[1]
    if not m or not k or not n or not nnz_a or not nnz_b:
        return None
    if max(m, n, k, nnz_a, nnz_b) >= 2**30:
        return None
    mean_a, mean_b = nnz_a / m, nnz_b / k
    if max_a > 32 * mean_a or max_b > 32 * mean_b:
        return None
    if mean_b > 2 * mean_a:
        cap = min(2048, triton.next_power_of_2(max_a * max_b))
    elif mean_a >= 20 and max_a <= 2 * mean_a and max_b <= 2 * mean_b:
        cap = 256
    else:
        cap = min(2048, max(32, triton.next_power_of_2(int(mean_a * mean_b + 0.5))))
        if cap == 256 and max_a * max_b > 4 * cap and mean_a * mean_b > 0.9 * cap:
            cap = 512
    if m * cap >= 2**29:
        return None
    rows = max(1, 512 // cap)
    tile = 4 if cap <= 32 else 8 if cap <= 64 else 16
    return rows, cap, tile


@lru_cache(maxsize=16)
def _lds_library(slots, complex_values):
    """Allocate the requested table size so small products can share a CU."""
    arrays = {"key": "i32", "real": "float"}
    if complex_values:
        arrays["imag"] = "float"
    clear_imag = """
    %i = getelementptr float, ptr addrspace(3) @imag, i32 %index
    store float 0.0, ptr addrspace(3) %i""" if complex_values else ""
    sum_imag = """
    %i = getelementptr float, ptr addrspace(3) @imag, i32 %slot
    %b = atomicrmw fadd ptr addrspace(3) %i, float %vi monotonic""" if complex_values else ""
    ir = []
    for field, typ in arrays.items():
        array = f"[{slots} x {typ}]"
        ir.append(f"""
@{field} = internal addrspace(3) global {array} undef, align 4
define {typ} @__flagsparse_{field}(i32 %index) alwaysinline {{
    %p = getelementptr {array}, ptr addrspace(3) @{field}, i32 0, i32 %index
    %v = load {typ}, ptr addrspace(3) %p
    ret {typ} %v
}}
""")
    ir.append(f"""
define i32 @__flagsparse_clear(i32 %index) alwaysinline {{
    %k = getelementptr i32, ptr addrspace(3) @key, i32 %index
    %r = getelementptr float, ptr addrspace(3) @real, i32 %index
    store i32 -1, ptr addrspace(3) %k
    store float 0.0, ptr addrspace(3) %r{clear_imag}
    ret i32 0
}}
define i32 @__flagsparse_add(i32 %base, i32 %cap, i32 %col, float %vr, float %vi) alwaysinline {{
entry:
    %mask = sub i32 %cap, 1
    %skip = icmp slt i32 %col, 0
    br i1 %skip, label %done, label %probe
probe:
    %step = phi i32 [ 0, %entry ], [ %next, %collision ]
    %hash = add i32 %col, %step
    %bucket = and i32 %hash, %mask
    %slot = add i32 %base, %bucket
    %k = getelementptr i32, ptr addrspace(3) @key, i32 %slot
    %claim = cmpxchg ptr addrspace(3) %k, i32 -1, i32 %col monotonic monotonic
    %old = extractvalue {{i32, i1}} %claim, 0
    %empty = extractvalue {{i32, i1}} %claim, 1
    %same = icmp eq i32 %old, %col
    %found = or i1 %empty, %same
    br i1 %found, label %sum, label %collision
collision:
    %next = add i32 %step, 1
    %full = icmp eq i32 %next, %cap
    br i1 %full, label %overflow, label %probe
sum:
    %r = getelementptr float, ptr addrspace(3) @real, i32 %slot
    %a = atomicrmw fadd ptr addrspace(3) %r, float %vr monotonic{sum_imag}
    br label %done
done:
    ret i32 0
overflow:
    ret i32 -1
}}
""")
    source = "\n".join(ir)
    cache = get_cache_manager(hashlib.sha256(source.encode()).hexdigest())
    return cache.get_file("spgemm_lds.ll") or cache.put(source, "spgemm_lds.ll", binary=False)


@core.extern
def _lds_clear(index, _builder=None):
    return core.extern_elementwise("", "", [index], {(core.int32,): ("__flagsparse_clear", core.int32)},
                                   is_pure=False, _builder=_builder)


@core.extern
def _lds_add(base, capacity, column, real, imag, _builder=None):
    args = (core.int32, core.int32, core.int32, core.float32, core.float32)
    return core.extern_elementwise("", "", [base, capacity, column, real, imag],
                                   {args: ("__flagsparse_add", core.int32)}, is_pure=False, _builder=_builder)


@core.extern
def _lds_key(index, _builder=None):
    return core.extern_elementwise("", "", [index], {(core.int32,): ("__flagsparse_key", core.int32)},
                                   is_pure=False, _builder=_builder)


@core.extern
def _lds_real(index, _builder=None):
    return core.extern_elementwise("", "", [index], {(core.int32,): ("__flagsparse_real", core.float32)},
                                   is_pure=False, _builder=_builder)


@core.extern
def _lds_imag(index, _builder=None):
    return core.extern_elementwise("", "", [index], {(core.int32,): ("__flagsparse_imag", core.float32)},
                                   is_pure=False, _builder=_builder)


@triton.jit
def _accumulate(A, AI, AP, B, BI, BP, COLS, VALUES, COUNTS, OVERFLOW, M,
                ROWS: tl.constexpr, CAP: tl.constexpr, TILE: tl.constexpr,
                COMPLEX: tl.constexpr):
    group = tl.program_id(0)
    rows = group * ROWS + tl.arange(0, ROWS)
    slots = tl.arange(0, ROWS * CAP)
    _lds_clear(slots)
    tl.debug_barrier()

    start = tl.load(AP + rows, rows < M, 0).to(tl.int32)
    end = tl.load(AP + rows + 1, rows < M, 0).to(tl.int32)
    lane = tl.arange(0, ROWS * TILE * TILE)
    local_row = lane // (TILE * TILE)
    row = group * ROWS + local_row
    left_start = tl.load(AP + row, row < M, 0).to(tl.int32)
    left_end = tl.load(AP + row + 1, row < M, 0).to(tl.int32)
    failed = tl.full((ROWS * TILE * TILE,), 0, tl.int32)
    for left in range(0, tl.max(end - start, 0), TILE):
        pos = left_start + left + (lane // TILE) % TILE
        active = pos < left_end
        inner = tl.load(AI + pos, active, 0)
        if COMPLEX:
            ar = tl.load(A + 2 * pos, active, 0.)
            ai = tl.load(A + 2 * pos + 1, active, 0.)
        else:
            ar = tl.load(A + pos, active, 0.)
        first = tl.load(BP + inner, active, 0).to(tl.int32)
        stop = tl.load(BP + inner + 1, active, 0).to(tl.int32)
        for right in range(0, tl.max(stop - first, 0), TILE):
            pos_b = first + right + lane % TILE
            valid = pos_b < stop
            col = tl.load(BI + pos_b, valid, -1)
            if COMPLEX:
                br = tl.load(B + 2 * pos_b, valid, 0.)
                bi = tl.load(B + 2 * pos_b + 1, valid, 0.)
                vr, vi = ar * br - ai * bi, ar * bi + ai * br
            else:
                vr, vi = ar * tl.load(B + pos_b, valid, 0.), 0.
            failed |= _lds_add(local_row * CAP, CAP, col, vr, vi)
    if tl.min(failed, 0) < 0:
        tl.atomic_or(OVERFLOW, 1)
    tl.debug_barrier()
    cols = tl.reshape(_lds_key(slots), (ROWS, CAP))
    re = tl.reshape(_lds_real(slots), (ROWS, CAP))
    present = cols >= 0
    offsets = tl.cumsum(present.to(tl.int32), 1) - 1
    dest = rows[:, None].to(tl.int64) * CAP + offsets
    tl.store(COLS + dest, cols, present)
    if COMPLEX:
        im = tl.reshape(_lds_imag(slots), (ROWS, CAP))
        tl.store(VALUES + 2 * dest, re, present)
        tl.store(VALUES + 2 * dest + 1, im, present)
    else:
        tl.store(VALUES + dest, re, present)
    tl.store(COUNTS + rows + 1, tl.sum(present.to(tl.int32), 1), rows < M)


@triton.jit
def _compact(COLS, VALUES, PTR, C, V, M, CAP: tl.constexpr, R: tl.constexpr, K: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    start = tl.load(PTR + rows, rows < M, 0)
    end = tl.load(PTR + rows + 1, rows < M, 0)
    slot = tl.program_id(1) * K + tl.arange(0, K)
    active = slot[None, :] < (end - start)[:, None]
    src = rows[:, None] * CAP + slot[None, :]
    dst = start[:, None] + slot[None, :]
    tl.store(C + dst, tl.load(COLS + src, active, 0), active)
    tl.store(V + dst, tl.load(VALUES + src, active, 0), active)


def compute(prepared):
    rows, cap, tile = prepared.rocm_hash_config
    m = prepared.n_rows
    a = prepared.a_data.resolve_conj().resolve_neg()
    b = prepared.b_data.resolve_conj().resolve_neg()
    complex_values = a.is_complex()
    counts = torch.zeros(m + 2, dtype=torch.int32, device=a.device)
    cols = torch.empty(m * cap, dtype=torch.int32, device=a.device)
    vals = a.new_empty(m * cap)
    _accumulate[(triton.cdiv(m, rows),)](
        torch.view_as_real(a) if complex_values else a, prepared.a_indices, prepared.a_indptr,
        torch.view_as_real(b) if complex_values else b, prepared.b_indices, prepared.b_indptr,
        cols, torch.view_as_real(vals) if complex_values else vals, counts, counts[m + 1:],
        m, rows, cap, tile, complex_values,
        extern_libs={"flagsparse": _lds_library(rows * cap, complex_values)},
        num_warps=4, enable_fp_fusion=False,
    )
    ptr = counts.cumsum(0, dtype=torch.int64)
    # Pack the overflow flag with the last row count in the same host transfer.
    nnz, with_flag = ptr[-2:].tolist()
    if nnz != with_flag:
        return None
    out_cols = cols.new_empty(nnz)
    out_vals = vals.new_empty(nnz)
    compact_rows = 4 if 32 < cap <= 128 else 8
    width = min(256, cap)
    # For complex output, copy both FP32 components together as one int64.
    _compact[(triton.cdiv(m, compact_rows), triton.cdiv(cap, width))](
        cols, vals.view(torch.int64) if complex_values else vals, ptr, out_cols,
        out_vals.view(torch.int64) if complex_values else out_vals,
        m, cap, compact_rows, width, num_warps=4,
    )
    return out_vals, out_cols, ptr[:-1]
