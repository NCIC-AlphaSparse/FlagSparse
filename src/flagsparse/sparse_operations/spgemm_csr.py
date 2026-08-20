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

"""CSR SpGEMM (A@B) with two-phase structure/value build."""

import ctypes

from . import _common as _common_mod
from ._common import *

HipPointer = _common_mod.HipPointer

SUPPORTED_SPGEMM_VALUE_DTYPES = (torch.float32, torch.float64)


class SpGEMMPrepared:
    """Prepared CSR metadata for repeated SpGEMM runs.

    ``a_row_work[i]`` is the number of scalar products row ``i`` expands to
    (``sum_k nnz(B[A_col_k, :])``), which the compute path uses to size hash
    tables and to route over-wide rows to the ESC fallback. ``a_pref`` is the
    matching within-row exclusive prefix sum over those per-nonzero counts.
    """

    __slots__ = (
        "a_data",
        "a_indices",
        "a_indptr",
        "a_shape",
        "b_data",
        "b_indices",
        "b_indptr",
        "b_shape",
        "n_rows",
        "n_inner",
        "n_cols",
        "a_row_work",
        "a_pref",
        "row_work_ready",
    )

    def __init__(
        self,
        a_data,
        a_indices,
        a_indptr,
        a_shape,
        b_data,
        b_indices,
        b_indptr,
        b_shape,
        a_row_work,
        a_pref,
        row_work_ready,
    ):
        self.a_data = a_data
        self.a_indices = a_indices
        self.a_indptr = a_indptr
        self.a_shape = (int(a_shape[0]), int(a_shape[1]))
        self.b_data = b_data
        self.b_indices = b_indices
        self.b_indptr = b_indptr
        self.b_shape = (int(b_shape[0]), int(b_shape[1]))
        self.n_rows = self.a_shape[0]
        self.n_inner = self.a_shape[1]
        self.n_cols = self.b_shape[1]
        self.a_row_work = a_row_work
        # per-A-nonzero exclusive prefix of B row lengths, restarted each row;
        # lets a kernel map a flat product index q -> (A nonzero, B position)
        self.a_pref = a_pref
        self.row_work_ready = bool(row_work_ready)


def _validate_csr(data, indices, indptr, shape, tag):
    if len(shape) != 2:
        raise ValueError(f"{tag}_shape must be a 2-tuple")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError(f"{tag} data/indices/indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError(f"{tag}_shape dimensions must be non-negative")
    if indptr.numel() != n_rows + 1:
        raise ValueError(
            f"{tag}_indptr length must be n_rows+1={n_rows + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError(f"{tag}_data and {tag}_indices must have the same length")
    if not data.is_cuda or not indices.is_cuda or not indptr.is_cuda:
        raise ValueError(f"{tag} tensors must be CUDA tensors")
    if data.dtype not in SUPPORTED_SPGEMM_VALUE_DTYPES:
        raise TypeError(f"{tag}_data dtype must be torch.float32 or torch.float64")
    if indices.dtype != torch.int32:
        raise TypeError(f"{tag}_indices dtype must be torch.int32")
    if indptr.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{tag}_indptr dtype must be torch.int32 or torch.int64")

    nnz = int(data.numel())
    indptr_i64 = indptr.to(torch.int64)
    # Gather every check into one device->host transfer; done separately these
    # were ~5 syncs per operand, which dominated prepare on short-row matrices.
    device = indptr_i64.device
    zero = torch.zeros((), dtype=torch.int64, device=device)
    checks = torch.stack(
        [
            indptr_i64[0] if indptr_i64.numel() > 0 else zero,
            indptr_i64[-1] if indptr_i64.numel() > 0 else zero,
            torch.any(indptr_i64[1:] < indptr_i64[:-1]).to(torch.int64)
            if indptr_i64.numel() > 1
            else zero,
            indices.min().to(torch.int64) if nnz > 0 else zero,
            indices.max().to(torch.int64) if nnz > 0 else zero,
        ]
    ).tolist()
    first, last, unsorted, min_col, max_col = checks
    if indptr_i64.numel() > 0 and first != 0:
        raise ValueError(f"{tag}_indptr[0] must be 0")
    if indptr_i64.numel() > 0 and last != nnz:
        raise ValueError(f"{tag}_indptr[-1] must equal nnz={nnz}")
    if unsorted:
        raise ValueError(f"{tag}_indptr must be nondecreasing")
    if nnz > 0 and (min_col < 0 or max_col >= n_cols):
        raise IndexError(f"{tag}_indices out of range for n_cols={n_cols}")
    return n_rows, n_cols, indptr_i64


def _prepare_spgemm_csr_inputs(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
):
    a_rows, a_cols, a_indptr64 = _validate_csr(
        a_data, a_indices, a_indptr, a_shape, "a"
    )
    b_rows, b_cols, b_indptr64 = _validate_csr(
        b_data, b_indices, b_indptr, b_shape, "b"
    )
    if a_cols != b_rows:
        raise ValueError(
            f"shape mismatch for A@B: A is {a_rows}x{a_cols}, B is {b_rows}x{b_cols}"
        )
    if a_data.device != b_data.device:
        raise ValueError("A and B tensors must be on the same CUDA device")
    if a_data.dtype != b_data.dtype:
        raise TypeError("A and B value dtype must match")

    a_data = a_data.contiguous()
    a_indices = a_indices.contiguous()
    a_indptr64 = a_indptr64.contiguous()
    b_data = b_data.contiguous()
    b_indices = b_indices.contiguous()
    b_indptr64 = b_indptr64.contiguous()
    return (
        a_data,
        a_indices,
        a_indptr64,
        (a_rows, a_cols),
        b_data,
        b_indices,
        b_indptr64,
        (b_rows, b_cols),
    )


def _build_row_product_metadata(a_indices, a_indptr, b_indptr):
    """Per-row product counts and the within-row prefix, from one scan.

    ``row_work[i]`` is how many scalar products row ``i`` expands to and
    ``pref[t]`` is how many the nonzeros before ``t`` (same row) contribute, so
    a kernel can binary-search a flat product index back to its A nonzero. Both
    fall out of the same exclusive scan over the per-nonzero B row lengths --
    computing them separately (as a row-work kernel plus this) duplicated the
    whole gather.
    """
    device = a_indices.device
    nnz_a = int(a_indices.numel())
    n_rows = int(a_indptr.numel()) - 1
    if nnz_a == 0:
        return (
            torch.zeros(max(n_rows, 0), dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        )
    b_len = (b_indptr[1:] - b_indptr[:-1]).to(torch.int64)
    seg = b_len[a_indices.to(torch.int64)]
    excl = torch.zeros(nnz_a + 1, dtype=torch.int64, device=device)
    excl[1:] = torch.cumsum(seg, 0)
    row_base = excl[a_indptr[:-1]]
    a_deg = a_indptr[1:] - a_indptr[:-1]
    pref = (excl[:nnz_a] - torch.repeat_interleave(row_base, a_deg)).to(torch.int32)
    row_work = (excl[a_indptr[1:]] - row_base).to(torch.int32)
    return row_work.contiguous(), pref.contiguous()


def prepare_spgemm_csr(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    block_nnz=256,
    analyze_rows=True,
):
    if block_nnz <= 0:
        raise ValueError("block_nnz must be positive")
    (
        a_data,
        a_indices,
        a_indptr,
        a_shape,
        b_data,
        b_indices,
        b_indptr,
        b_shape,
    ) = _prepare_spgemm_csr_inputs(
        a_data,
        a_indices,
        a_indptr,
        a_shape,
        b_data,
        b_indices,
        b_indptr,
        b_shape,
    )
    n_rows = int(a_shape[0])
    if n_rows == 0:
        row_work = torch.empty(0, dtype=torch.int32, device=a_data.device)
        a_pref = torch.empty(0, dtype=torch.int32, device=a_data.device)
        row_work_ready = True
    elif analyze_rows:
        row_work, a_pref = _build_row_product_metadata(a_indices, a_indptr, b_indptr)
        row_work_ready = True
    else:
        # Row work is recomputed on demand by the compute path when not analyzed.
        row_work = torch.zeros(n_rows, dtype=torch.int32, device=a_data.device)
        a_pref = _build_row_product_metadata(a_indices, a_indptr, b_indptr)[1]
        row_work_ready = False
    return SpGEMMPrepared(
        a_data=a_data,
        a_indices=a_indices,
        a_indptr=a_indptr,
        a_shape=a_shape,
        b_data=b_data,
        b_indices=b_indices,
        b_indptr=b_indptr,
        b_shape=b_shape,
        a_row_work=row_work,
        a_pref=a_pref,
        row_work_ready=row_work_ready,
    )


def _spgemm_esc_compute(prepared):
    """C = A @ B via a single global expand-sort-reduce (ESC).

    Every scalar product A[i,k]*B[k,j] is materialized once, keyed by
    row*n_cols+col, sorted globally, then duplicate (row,col) pairs are summed.
    The result is emitted directly as a canonical CSR (columns sorted per row).
    This replaces the former per-row-bucket Python orchestration and is an order
    of magnitude faster while producing identical output.
    """
    a_data = prepared.a_data
    a_indices = prepared.a_indices
    a_indptr = prepared.a_indptr
    b_data = prepared.b_data
    b_indices = prepared.b_indices
    b_indptr = prepared.b_indptr
    n_rows = prepared.n_rows
    n_cols = prepared.n_cols
    device = a_data.device
    dtype = a_data.dtype

    empty_indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
    nnz_a = a_data.numel()
    if nnz_a == 0 or n_rows == 0:
        return (
            torch.empty(0, dtype=dtype, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            empty_indptr,
        )

    a_indptr64 = a_indptr.to(torch.int64)
    b_indptr64 = b_indptr.to(torch.int64)
    a_row = torch.repeat_interleave(
        torch.arange(n_rows, device=device), a_indptr64[1:] - a_indptr64[:-1]
    )
    k = a_indices.to(torch.int64)  # A column == B row
    b_len = b_indptr64[1:] - b_indptr64[:-1]
    seg = b_len[k]  # number of products contributed by each A nonzero
    total = int(seg.sum().item())
    if total == 0:
        return (
            torch.empty(0, dtype=dtype, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            empty_indptr,
        )

    # expanded[p] -> the A nonzero that produced it, and its offset within B[k,:]
    ap = torch.repeat_interleave(torch.arange(nnz_a, device=device), seg)
    base = torch.cumsum(seg, 0) - seg
    within = torch.arange(total, device=device) - torch.repeat_interleave(base, seg)
    bpos = b_indptr64[k][ap] + within
    out_col = b_indices[bpos].to(torch.int64)
    out_val = a_data[ap] * b_data[bpos]
    out_row = a_row[ap]

    key = out_row * n_cols + out_col
    order = torch.argsort(key)
    key_sorted = key[order]
    val_sorted = out_val[order]
    uniq_key, inverse = torch.unique_consecutive(key_sorted, return_inverse=True)
    c_data = torch.zeros(uniq_key.numel(), dtype=dtype, device=device)
    c_data.index_add_(0, inverse, val_sorted)
    c_rows = torch.div(uniq_key, n_cols, rounding_mode="floor")
    c_indices = (uniq_key - c_rows * n_cols).to(torch.int32)

    c_indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
    row_counts = torch.bincount(c_rows, minlength=n_rows)
    c_indptr[1:] = torch.cumsum(row_counts, dim=0)
    return c_data, c_indices, c_indptr


# ---------------------------------------------------------------------------
# TLE shared-memory hash acceleration (float32 medium-row rows)
#
# For rows whose expanded work fits a shared-memory hash table, a single
# program per row accumulates C[i,:] entirely in shared memory (open-addressing
# hash keyed by column, value accumulator). This is dramatically faster than
# the global expand-sort-reduce (ESC) for medium-width rows. Rows that are too
# small (hash init/occupancy overhead dominates) or too wide (exceed the
# largest shared-memory table) are routed to the chunked ESC fallback instead.
#
# TLE is a FlagTree Triton extension and may be unavailable; the import is
# guarded and the whole path degrades to pure ESC when it (or a non-float32
# dtype, or an out-of-int32-range shape) rules it out.
# ---------------------------------------------------------------------------
try:
    import triton.experimental.tle.language as _tle

    _TLE_AVAILABLE = True
except Exception:  # pragma: no cover - environment without FlagTree TLE
    _tle = None
    _TLE_AVAILABLE = False

# (max_row_work, hash_capacity, inner_block, num_warps); capacity is a power of
# two (load factor <=0.75) and stays within the 100KB shared-memory budget
# (8192*4B*2 tables). A row is sized into the first bucket whose max_row_work it
# fits. Rows start at the size their row work suggests (clamped to the largest
# table) and are re-counted one size up if they overflow; only rows still
# overflowing at the largest table fall back to the chunked ESC.
_SPGEMM_HASH_BUCKETS = (
    (48, 64, 32, 2),
    (192, 256, 32, 2),
    (768, 1024, 64, 4),
    (3072, 4096, 128, 8),
    (6144, 8192, 256, 16),
)
_SPGEMM_HASH_LOAD = 0.75  # max hash-table load factor
_SPGEMM_HASH_SMEM_BUDGET = 96 * 1024  # per-program shared-memory budget
_SPGEMM_SINGLE_PASS_MAX_BYTES = 512 * 1024 * 1024  # over-allocation cap
_SPGEMM_ESC_CHUNK_PRODUCTS = 8_000_000  # peak-memory bound for chunked ESC


if _TLE_AVAILABLE:

    @triton.jit
    def _spgemm_hash_count_kernel(
        a_ind,
        a_ptr,
        a_pref,
        b_ind,
        b_ptr,
        rows_ptr,
        nrows,
        rw_ptr,
        row_nnz_ptr,
        ovf_ptr,
        CAP: tl.constexpr,
        BLOCK: tl.constexpr,
        USE_ROWS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= nrows:
            return
        if USE_ROWS:
            row = tl.load(rows_ptr + pid)
        else:
            row = pid
        rw = tl.load(rw_ptr + row)
        keys = _tle.gpu.alloc([CAP], tl.int32, scope=_tle.gpu.smem)
        tl.store(
            _tle.gpu.local_ptr(keys, (tl.arange(0, CAP),)),
            tl.full((CAP,), -1, tl.int32),
        )
        a_s = tl.load(a_ptr + row)
        a_e = tl.load(a_ptr + row + 1)
        fail = 0
        q0 = 0
        # stop as soon as the table overflows: this row's count is discarded
        # (it falls back to ESC), so probing the rest of it is pure waste --
        # on hub rows a full table costs CAP probes per product otherwise.
        while (q0 < rw) & (fail == 0):
            q = q0 + tl.arange(0, BLOCK)
            m = q < rw
            # locate the A nonzero owning product q: last ap with pref[ap] <= q
            lo = tl.full((BLOCK,), a_s, tl.int32)
            hi = tl.full((BLOCK,), a_e, tl.int32)
            while tl.sum((hi - lo > 1).to(tl.int32)) > 0:
                mid = (lo + hi) // 2
                pv = tl.load(a_pref + mid, mask=(mid < a_e), other=0x7FFFFFFF)
                take = pv <= q
                lo = tl.where(take, mid, lo)
                hi = tl.where(take, hi, mid)
            ap = lo
            k = tl.load(a_ind + ap, mask=m, other=0)
            pv = tl.load(a_pref + ap, mask=m, other=0)
            bpos = tl.load(b_ptr + k, mask=m, other=0) + (q - pv)
            j = tl.load(b_ind + bpos, mask=m, other=-1)
            h = (j.to(tl.uint32) * 2654435761).to(tl.int32) & (CAP - 1)
            done = (~m) | (j < 0)
            it = 0
            while (tl.sum((~done).to(tl.int32)) > 0) & (it < CAP):
                cur = tl.atomic_cas(
                    _tle.gpu.local_ptr(keys, (h,)),
                    tl.full((BLOCK,), -1, tl.int32),
                    j,
                )
                done = done | (cur == -1) | (cur == j)
                j = tl.where(done, -1, j)
                h = (h + 1) & (CAP - 1)
                it += 1
            # a lane left unplaced means the table was too small for this row
            fail = fail | (tl.sum((~done).to(tl.int32)) > 0).to(tl.int32)
            q0 += BLOCK
        kk = tl.load(_tle.gpu.local_ptr(keys, (tl.arange(0, CAP),)))
        cnt = tl.sum((kk != -1).to(tl.int32))
        if fail:
            tl.store(ovf_ptr + row, 1)
            tl.store(row_nnz_ptr + row, 0)
        else:
            tl.store(ovf_ptr + row, 0)
            tl.store(row_nnz_ptr + row, cnt)

    @triton.jit
    def _spgemm_hash_fill_kernel(
        a_data,
        a_ind,
        a_ptr,
        a_pref,
        b_data,
        b_ind,
        b_ptr,
        rows_ptr,
        nrows,
        rw_ptr,
        c_indptr,
        c_ind,
        c_data,
        row_nnz_ptr,
        ovf_ptr,
        CAP: tl.constexpr,
        BLOCK: tl.constexpr,
        USE_ROWS: tl.constexpr,
        REPORT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= nrows:
            return
        if USE_ROWS:
            row = tl.load(rows_ptr + pid)
        else:
            row = pid
        rw = tl.load(rw_ptr + row)
        keys = _tle.gpu.alloc([CAP], tl.int32, scope=_tle.gpu.smem)
        accum = _tle.gpu.alloc([CAP], tl.float32, scope=_tle.gpu.smem)
        tl.store(
            _tle.gpu.local_ptr(keys, (tl.arange(0, CAP),)),
            tl.full((CAP,), -1, tl.int32),
        )
        tl.store(
            _tle.gpu.local_ptr(accum, (tl.arange(0, CAP),)),
            tl.zeros((CAP,), tl.float32),
        )
        a_s = tl.load(a_ptr + row)
        a_e = tl.load(a_ptr + row + 1)
        fail = 0
        q0 = 0
        while (q0 < rw) & (fail == 0):
            q = q0 + tl.arange(0, BLOCK)
            m = q < rw
            lo = tl.full((BLOCK,), a_s, tl.int32)
            hi = tl.full((BLOCK,), a_e, tl.int32)
            while tl.sum((hi - lo > 1).to(tl.int32)) > 0:
                mid = (lo + hi) // 2
                pv = tl.load(a_pref + mid, mask=(mid < a_e), other=0x7FFFFFFF)
                take = pv <= q
                lo = tl.where(take, mid, lo)
                hi = tl.where(take, hi, mid)
            ap = lo
            k = tl.load(a_ind + ap, mask=m, other=0)
            av = tl.load(a_data + ap, mask=m, other=0.0)
            pv = tl.load(a_pref + ap, mask=m, other=0)
            bpos = tl.load(b_ptr + k, mask=m, other=0) + (q - pv)
            j = tl.load(b_ind + bpos, mask=m, other=-1)
            bv = tl.load(b_data + bpos, mask=m, other=0.0)
            prod = av * bv
            h = (j.to(tl.uint32) * 2654435761).to(tl.int32) & (CAP - 1)
            done = (~m) | (j < 0)
            it = 0
            while (tl.sum((~done).to(tl.int32)) > 0) & (it < CAP):
                cur = tl.atomic_cas(
                    _tle.gpu.local_ptr(keys, (h,)),
                    tl.full((BLOCK,), -1, tl.int32),
                    j,
                )
                hit = (cur == -1) | (cur == j)
                da = (~done) & hit
                tl.atomic_add(
                    _tle.gpu.local_ptr(accum, (h,)),
                    tl.where(da, prod, 0.0),
                    mask=da,
                )
                done = done | hit
                j = tl.where(done, -1, j)
                h = (h + 1) & (CAP - 1)
                it += 1
            fail = fail | (tl.sum((~done).to(tl.int32)) > 0).to(tl.int32)
            q0 += BLOCK
        sl = tl.arange(0, CAP)
        kk = tl.load(_tle.gpu.local_ptr(keys, (sl,)))
        vv = tl.load(_tle.gpu.local_ptr(accum, (sl,)))
        ne = kk != -1
        pos = tl.load(c_indptr + row) + (
            tl.cumsum(ne.to(tl.int32), 0) - ne.to(tl.int32)
        )
        tl.store(c_ind + pos, kk, mask=ne)
        tl.store(c_data + pos, vv, mask=ne)
        if REPORT:
            # single-pass mode: this pass also serves as the counting pass
            tl.store(ovf_ptr + row, fail)
            tl.store(
                row_nnz_ptr + row,
                tl.where(fail != 0, 0, tl.sum(ne.to(tl.int32))),
            )

    # float64 twin of the fill kernel: TLE's smem alloc needs a literal
    # dtype, so the accumulator type cannot be a constexpr parameter.
    @triton.jit
    def _spgemm_hash_fill_kernel_f64(
        a_data,
        a_ind,
        a_ptr,
        a_pref,
        b_data,
        b_ind,
        b_ptr,
        rows_ptr,
        nrows,
        rw_ptr,
        c_indptr,
        c_ind,
        c_data,
        row_nnz_ptr,
        ovf_ptr,
        CAP: tl.constexpr,
        BLOCK: tl.constexpr,
        USE_ROWS: tl.constexpr,
        REPORT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= nrows:
            return
        if USE_ROWS:
            row = tl.load(rows_ptr + pid)
        else:
            row = pid
        rw = tl.load(rw_ptr + row)
        keys = _tle.gpu.alloc([CAP], tl.int32, scope=_tle.gpu.smem)
        accum = _tle.gpu.alloc([CAP], tl.float64, scope=_tle.gpu.smem)
        tl.store(
            _tle.gpu.local_ptr(keys, (tl.arange(0, CAP),)),
            tl.full((CAP,), -1, tl.int32),
        )
        tl.store(
            _tle.gpu.local_ptr(accum, (tl.arange(0, CAP),)),
            tl.zeros((CAP,), tl.float64),
        )
        a_s = tl.load(a_ptr + row)
        a_e = tl.load(a_ptr + row + 1)
        fail = 0
        q0 = 0
        while (q0 < rw) & (fail == 0):
            q = q0 + tl.arange(0, BLOCK)
            m = q < rw
            lo = tl.full((BLOCK,), a_s, tl.int32)
            hi = tl.full((BLOCK,), a_e, tl.int32)
            while tl.sum((hi - lo > 1).to(tl.int32)) > 0:
                mid = (lo + hi) // 2
                pv = tl.load(a_pref + mid, mask=(mid < a_e), other=0x7FFFFFFF)
                take = pv <= q
                lo = tl.where(take, mid, lo)
                hi = tl.where(take, hi, mid)
            ap = lo
            k = tl.load(a_ind + ap, mask=m, other=0)
            av = tl.load(a_data + ap, mask=m, other=0.0)
            pv = tl.load(a_pref + ap, mask=m, other=0)
            bpos = tl.load(b_ptr + k, mask=m, other=0) + (q - pv)
            j = tl.load(b_ind + bpos, mask=m, other=-1)
            bv = tl.load(b_data + bpos, mask=m, other=0.0)
            prod = av * bv
            h = (j.to(tl.uint32) * 2654435761).to(tl.int32) & (CAP - 1)
            done = (~m) | (j < 0)
            it = 0
            while (tl.sum((~done).to(tl.int32)) > 0) & (it < CAP):
                cur = tl.atomic_cas(
                    _tle.gpu.local_ptr(keys, (h,)),
                    tl.full((BLOCK,), -1, tl.int32),
                    j,
                )
                hit = (cur == -1) | (cur == j)
                da = (~done) & hit
                tl.atomic_add(
                    _tle.gpu.local_ptr(accum, (h,)),
                    tl.where(da, prod, 0.0),
                    mask=da,
                )
                done = done | hit
                j = tl.where(done, -1, j)
                h = (h + 1) & (CAP - 1)
                it += 1
            fail = fail | (tl.sum((~done).to(tl.int32)) > 0).to(tl.int32)
            q0 += BLOCK
        sl = tl.arange(0, CAP)
        kk = tl.load(_tle.gpu.local_ptr(keys, (sl,)))
        vv = tl.load(_tle.gpu.local_ptr(accum, (sl,)))
        ne = kk != -1
        pos = tl.load(c_indptr + row) + (
            tl.cumsum(ne.to(tl.int32), 0) - ne.to(tl.int32)
        )
        tl.store(c_ind + pos, kk, mask=ne)
        tl.store(c_data + pos, vv, mask=ne)
        if REPORT:
            # single-pass mode: this pass also serves as the counting pass
            tl.store(ovf_ptr + row, fail)
            tl.store(
                row_nnz_ptr + row,
                tl.where(fail != 0, 0, tl.sum(ne.to(tl.int32))),
            )


def _spgemm_row_products(prepared):
    """Products (expanded work) per output row: ``sum_k nnz(B[A_col_k, :])``.

    Returned as int32 (the dtype ``prepare`` stores) so the compute path does
    not pay an int32 -> int64 -> int32 round trip over every row on each call.
    Callers that sum it must accumulate in int64.
    """
    if prepared.row_work_ready and prepared.a_row_work.numel() == prepared.n_rows:
        return prepared.a_row_work
    return _build_row_product_metadata(
        prepared.a_indices, prepared.a_indptr, prepared.b_indptr
    )[0]


def _spgemm_esc_expand_rows(prepared, rows):
    """Reduced (crow, ccol, cval) for the given ascending row ids, in CSR order.

    Materializes exactly the products contributed by ``rows``, keys them by
    row*n_cols+col, sorts, and sums duplicates. Output is globally row-sorted
    and column-sorted within each row (canonical CSR order for these rows).
    """
    device = prepared.a_data.device
    dtype = prepared.a_data.dtype
    n_cols = prepared.n_cols
    a_data, a_indices = prepared.a_data, prepared.a_indices
    b_data, b_indices = prepared.b_data, prepared.b_indices
    a_indptr64 = prepared.a_indptr.to(torch.int64)
    b_indptr64 = prepared.b_indptr.to(torch.int64)

    a_deg = a_indptr64[1:] - a_indptr64[:-1]
    sel_deg = a_deg[rows]
    n_a_sel = int(sel_deg.sum().item())
    empty = (
        torch.empty(0, dtype=torch.int64, device=device),
        torch.empty(0, dtype=torch.int32, device=device),
        torch.empty(0, dtype=dtype, device=device),
    )
    if n_a_sel == 0:
        return empty
    sel_starts = a_indptr64[rows]
    ap_local = torch.repeat_interleave(
        torch.arange(rows.numel(), device=device), sel_deg
    )
    a_base = torch.cumsum(sel_deg, 0) - sel_deg
    within_a = torch.arange(n_a_sel, device=device) - torch.repeat_interleave(
        a_base, sel_deg
    )
    ap = sel_starts[ap_local] + within_a
    arow = rows[ap_local]

    k = a_indices[ap].to(torch.int64)
    b_len = b_indptr64[1:] - b_indptr64[:-1]
    seg = b_len[k]
    total = int(seg.sum().item())
    if total == 0:
        return empty
    aap = torch.repeat_interleave(torch.arange(ap.numel(), device=device), seg)
    base = torch.cumsum(seg, 0) - seg
    within_b = torch.arange(total, device=device) - torch.repeat_interleave(base, seg)
    bpos = b_indptr64[k][aap] + within_b
    out_col = b_indices[bpos].to(torch.int64)
    out_val = a_data[ap][aap] * b_data[bpos]
    out_row = arow[aap]

    key = out_row * n_cols + out_col
    order = torch.argsort(key)
    key_sorted = key[order]
    val_sorted = out_val[order]
    uniq_key, inverse = torch.unique_consecutive(key_sorted, return_inverse=True)
    cval = torch.zeros(uniq_key.numel(), dtype=dtype, device=device)
    cval.index_add_(0, inverse, val_sorted)
    crow = torch.div(uniq_key, n_cols, rounding_mode="floor")
    ccol = (uniq_key - crow * n_cols).to(torch.int32)
    return crow, ccol, cval


def _spgemm_esc_rows_chunked(prepared, rows, rw, budget=_SPGEMM_ESC_CHUNK_PRODUCTS):
    """ESC over ``rows`` split into contiguous chunks bounded by ``budget``
    products, so peak memory stays bounded regardless of total work."""
    device = prepared.a_data.device
    dtype = prepared.a_data.dtype
    if rows.numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=dtype, device=device),
        )
    prod = rw[rows].to(torch.int64)
    total = int(prod.sum().item())
    if total <= budget:
        return _spgemm_esc_expand_rows(prepared, rows)
    csum = torch.cumsum(prod, 0)
    chunk_id = torch.div(torch.clamp(csum - 1, min=0), budget, rounding_mode="floor")
    boundaries = (
        torch.nonzero(chunk_id[1:] != chunk_id[:-1], as_tuple=False).flatten() + 1
    )
    edges = [0] + (boundaries + 0).tolist() + [rows.numel()]
    crow_parts, ccol_parts, cval_parts = [], [], []
    for i in range(len(edges) - 1):
        sub = rows[edges[i] : edges[i + 1]]
        if sub.numel() == 0:
            continue
        cr, cc, cv = _spgemm_esc_expand_rows(prepared, sub)
        if cr.numel():
            crow_parts.append(cr)
            ccol_parts.append(cc)
            cval_parts.append(cv)
        torch.cuda.empty_cache()
    if not crow_parts:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=dtype, device=device),
        )
    return (
        torch.cat(crow_parts),
        torch.cat(ccol_parts),
        torch.cat(cval_parts),
    )


def _spgemm_assemble_row_sorted(crow, ccol, cval, n_rows, dtype, device):
    """(crow, ccol, cval) already in CSR order -> (c_data, c_indices, c_indptr)."""
    c_indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
    if crow.numel():
        c_indptr[1:] = torch.cumsum(torch.bincount(crow, minlength=n_rows), dim=0)
    return cval.to(dtype), ccol.to(torch.int32), c_indptr


def _spgemm_esc_compute_safe(prepared):
    """Global ESC that never exceeds the chunk memory budget."""
    rw = _spgemm_row_products(prepared)
    total = int(rw.sum(dtype=torch.int64).item())
    if total <= _SPGEMM_ESC_CHUNK_PRODUCTS:
        return _spgemm_esc_compute(prepared)
    device = prepared.a_data.device
    rows = torch.nonzero(rw > 0, as_tuple=False).flatten()
    crow, ccol, cval = _spgemm_esc_rows_chunked(prepared, rows, rw)
    return _spgemm_assemble_row_sorted(
        crow, ccol, cval, prepared.n_rows, prepared.a_data.dtype, device
    )


def _spgemm_hash_hybrid_compute(prepared):
    """Hybrid: TLE shared-memory hash for rows that fit a table, chunked ESC
    for the rest. Returns ``None`` when hashing does not apply (caller then
    uses the ESC fallback).

    Table sizes start from the row-work upper bound *clamped to the largest
    table* rather than routing wide rows straight to ESC -- row work
    over-estimates the distinct column count by up to ~37x, so many "wide" rows
    actually fit. Rows that genuinely overflow are re-counted at the next size
    up; only those still overflowing at the largest table fall back to ESC. The
    fill phase is then sized from the exact per-row nnz the count produced.
    """
    if not _TLE_AVAILABLE:
        return None
    dtype = prepared.a_data.dtype
    if dtype == torch.float32:
        fill_kernel = _spgemm_hash_fill_kernel
        acc_bytes = 4
    elif dtype == torch.float64:
        fill_kernel = _spgemm_hash_fill_kernel_f64
        acc_bytes = 8
    else:
        return None
    n_rows = prepared.n_rows
    n_cols = prepared.n_cols
    nnz_a = prepared.a_data.numel()
    if n_rows == 0 or nnz_a == 0:
        return None
    if (
        n_cols > _INDEX_LIMIT_INT32
        or prepared.b_data.numel() > _INDEX_LIMIT_INT32
        or nnz_a > _INDEX_LIMIT_INT32
    ):
        return None

    device = prepared.a_data.device
    rw = _spgemm_row_products(prepared)
    if int(rw.sum(dtype=torch.int64).item()) == 0:
        return None
    if int(rw.max().item()) > _INDEX_LIMIT_INT32:
        return None
    pref = prepared.a_pref
    if pref is None or pref.numel() != nnz_a:
        pref = _build_row_product_metadata(
            prepared.a_indices, prepared.a_indptr, prepared.b_indptr
        )[1]

    ai = prepared.a_indices.to(torch.int32).contiguous()
    ap = prepared.a_indptr.to(torch.int32).contiguous()
    bi = prepared.b_indices.to(torch.int32).contiguous()
    bp = prepared.b_indptr.to(torch.int32).contiguous()
    a_data = prepared.a_data.contiguous()
    b_data = prepared.b_data.contiguous()
    rw32 = rw.contiguous()
    dummy = torch.empty(1, dtype=torch.int32, device=device)

    row_nnz = torch.zeros(n_rows, dtype=torch.int64, device=device)
    ovf = torch.zeros(n_rows, dtype=torch.int32, device=device)

    # fp64 tables hold keys(4B)+accum(8B) per slot, so the largest sizes may not
    # fit the shared-memory budget; drop the levels that do not.
    ncaps = len(_SPGEMM_HASH_BUCKETS)
    while (
        ncaps > 1
        and _SPGEMM_HASH_BUCKETS[ncaps - 1][1] * (4 + acc_bytes)
        > _SPGEMM_HASH_SMEM_BUDGET
    ):
        ncaps -= 1

    # A single-pass variant (fill reports its own counts, writing into
    # rw-sized slots, then compact) was measured a net loss: 0.589 -> 0.528
    # average, because the compaction gather over nnz(C) costs more than the
    # counting pass it saves. It did help ASIC_680ks (0.12 -> 0.36), whose row
    # work spans four table sizes, so a narrower gate may still be worthwhile.
    # The kernels keep their REPORT path for that experiment.
    # Starting table size per row from its row-work bound, clamped to the
    # largest table. Rows below the top level have row work within that table's
    # load bound and so cannot overflow; only clamped rows can, and those go
    # straight to ESC. (Starting everyone at the smallest table and escalating
    # was measured ~8x slower -- the wasted passes fall on the widest rows.)
    level = torch.zeros(n_rows, dtype=torch.int64, device=device)
    for ci, (max_rw, _cap, _blk, _w) in enumerate(_SPGEMM_HASH_BUCKETS[:ncaps]):
        level = torch.where(rw > max_rw, torch.full_like(level, ci + 1), level)
    level = level.clamp(max=ncaps - 1)

    lvl_min = int(level.min().item())
    lvl_max = int(level.max().item())
    if lvl_min == lvl_max:
        # uniform: one launch over all rows, no row-list materialisation
        _max_rw, cap, blk, warps = _SPGEMM_HASH_BUCKETS[lvl_min]
        _spgemm_hash_count_kernel[(n_rows,)](
            ai,
            ap,
            pref,
            bi,
            bp,
            dummy,
            n_rows,
            rw32,
            row_nnz,
            ovf,
            CAP=cap,
            BLOCK=blk,
            num_warps=warps,
            USE_ROWS=False,
        )
    else:
        for ci in range(lvl_min, lvl_max + 1):
            sel = torch.nonzero(level == ci, as_tuple=False).flatten().to(torch.int32)
            if sel.numel() == 0:
                continue
            _max_rw, cap, blk, warps = _SPGEMM_HASH_BUCKETS[ci]
            _spgemm_hash_count_kernel[(sel.numel(),)](
                ai,
                ap,
                pref,
                bi,
                bp,
                sel,
                sel.numel(),
                rw32,
                row_nnz,
                ovf,
                CAP=cap,
                BLOCK=blk,
                num_warps=warps,
                USE_ROWS=True,
            )
    pend = torch.nonzero(ovf, as_tuple=False).flatten()

    # rows too wide for the largest table -> chunked ESC
    crow = ccol = cval = None
    if pend.numel():
        # A Gustavson dense-accumulator (SPA) alternative to this ESC fallback
        # was measured to make no difference: on Stanford / net150 the wide rows
        # are only 20% / 44% of the runtime, and SPA cost the same as ESC on
        # them. The bottleneck is the hash path on the merely-wide rows.
        crow, ccol, cval = _spgemm_esc_rows_chunked(prepared, pend, rw)
        if crow.numel():
            row_nnz += torch.bincount(crow, minlength=n_rows)

    c_indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
    c_indptr[1:] = torch.cumsum(row_nnz, dim=0)
    nnz_c = int(c_indptr[-1].item())
    c_indices = torch.empty(nnz_c, dtype=torch.int32, device=device)
    c_data = torch.empty(nnz_c, dtype=dtype, device=device)

    # fill, sized from the exact counts rather than the row-work bound
    hashed = row_nnz > 0
    if pend.numel():
        keep = torch.ones(n_rows, dtype=torch.bool, device=device)
        keep[pend] = False
        hashed = hashed & keep
    if bool(hashed.any()):
        max_nnz = int(row_nnz[hashed].max().item())
        single = None
        if pend.numel() == 0:
            for _mr, cap, blk, warps in _SPGEMM_HASH_BUCKETS[:ncaps]:
                if max_nnz <= cap * _SPGEMM_HASH_LOAD:
                    single = (cap, blk, warps)
                    break
        if single is not None:
            cap, blk, warps = single
            fill_kernel[(n_rows,)](
                a_data,
                ai,
                ap,
                pref,
                b_data,
                bi,
                bp,
                dummy,
                n_rows,
                rw32,
                c_indptr,
                c_indices,
                c_data,
                row_nnz,
                ovf,
                CAP=cap,
                BLOCK=blk,
                num_warps=warps,
                USE_ROWS=False,
                REPORT=False,
            )
        else:
            lo = 0
            for _mr, cap, blk, warps in _SPGEMM_HASH_BUCKETS[:ncaps]:
                hi = int(cap * _SPGEMM_HASH_LOAD)
                sel = (
                    torch.nonzero(
                        hashed & (row_nnz > lo) & (row_nnz <= hi), as_tuple=False
                    )
                    .flatten()
                    .to(torch.int32)
                )
                lo = hi
                if sel.numel() == 0:
                    continue
                fill_kernel[(sel.numel(),)](
                    a_data,
                    ai,
                    ap,
                    pref,
                    b_data,
                    bi,
                    bp,
                    sel,
                    sel.numel(),
                    rw32,
                    c_indptr,
                    c_indices,
                    c_data,
                    row_nnz,
                    ovf,
                    CAP=cap,
                    BLOCK=blk,
                    num_warps=warps,
                    USE_ROWS=True,
                    REPORT=False,
                )

    if crow is not None and crow.numel():
        starts = c_indptr[crow]
        first = torch.searchsorted(crow, crow)
        pos = starts + (torch.arange(crow.numel(), device=device) - first)
        c_indices[pos] = ccol
        c_data[pos] = cval.to(dtype)

    return c_data, c_indices, c_indptr


def _spgemm_compute(prepared):
    """Dispatch: hybrid hash when worthwhile, otherwise memory-safe ESC."""
    if _TLE_AVAILABLE and prepared.a_data.dtype in (torch.float32, torch.float64):
        try:
            result = _spgemm_hash_hybrid_compute(prepared)
        except (triton.runtime.errors.OutOfResources, torch.cuda.OutOfMemoryError):
            # genuinely out of shared memory / device memory: ESC still works
            result = None
        if result is not None:
            return result
    return _spgemm_esc_compute_safe(prepared)


_SPGEMM_EMPTY_STAGE_META = {
    "count_ms": None,
    "fill_ms": None,
    "bucket_ms_short": None,
    "bucket_ms_medium": None,
    "bucket_ms_long": None,
    "bucket_count_ms_short": None,
    "bucket_count_ms_medium": None,
    "bucket_count_ms_long": None,
    "bucket_fill_ms_short": None,
    "bucket_fill_ms_medium": None,
    "bucket_fill_ms_long": None,
    "bucket_nrows_short": 0,
    "bucket_nrows_medium": 0,
    "bucket_nrows_long": 0,
    "long_row_sliced_count": 0,
}


def _run_spgemm_prepared(prepared, out=None, profile=False, measure_stage=False):
    if out is not None:
        if not isinstance(out, (tuple, list)) or len(out) != 3:
            raise TypeError("out must be a tuple/list of (data, indices, indptr)")
        out_data, out_indices, out_indptr = out
        if not out_data.is_cuda or not out_indices.is_cuda or not out_indptr.is_cuda:
            raise ValueError("out data/indices/indptr must be CUDA tensors")
        if (
            out_data.device != prepared.a_data.device
            or out_indices.device != prepared.a_data.device
            or out_indptr.device != prepared.a_data.device
        ):
            raise ValueError(
                "out data/indices/indptr must be on the same CUDA device as computed C"
            )
        if (
            out_indptr.shape != (prepared.n_rows + 1,)
            or out_indptr.dtype != torch.int64
        ):
            raise ValueError("out indptr shape/dtype must match computed C indptr")
    else:
        out_data = out_indices = out_indptr = None

    if measure_stage:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    c_data, c_indices, c_indptr = _spgemm_compute(prepared)
    if measure_stage:
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - t0) * 1000.0
    else:
        total_ms = None

    nnz_c = int(c_data.numel())
    if out_data is not None:
        if out_data.shape != (nnz_c,) or out_data.dtype != prepared.a_data.dtype:
            raise ValueError("out data shape/dtype must match computed C data")
        if out_indices.shape != (nnz_c,) or out_indices.dtype != torch.int32:
            raise ValueError("out indices shape/dtype must match computed C indices")
        out_data.copy_(c_data)
        out_indices.copy_(c_indices)
        out_indptr.copy_(c_indptr)
        c_data, c_indices, c_indptr = out_data, out_indices, out_indptr

    meta = dict(_SPGEMM_EMPTY_STAGE_META)
    meta["count_ms"] = total_ms
    meta["fill_ms"] = 0.0 if total_ms is not None else None
    return c_data, c_indices, c_indptr, meta


def flagsparse_spgemm_csr(
    a_data=None,
    a_indices=None,
    a_indptr=None,
    a_shape=None,
    b_data=None,
    b_indices=None,
    b_indptr=None,
    b_shape=None,
    prepared=None,
    out=None,
    return_time=False,
    return_meta=False,
):
    """CSR SpGEMM: C = A @ B with CSR output (Triton-only main path)."""
    prepare_ms = 0.0
    if prepared is None:
        if any(
            x is None
            for x in (
                a_data,
                a_indices,
                a_indptr,
                a_shape,
                b_data,
                b_indices,
                b_indptr,
                b_shape,
            )
        ):
            raise ValueError(
                "A/B CSR tensors and shapes are required when prepared is not provided"
            )
        if return_meta:
            torch.cuda.synchronize()
            t_prepare0 = time.perf_counter()
        prepared = prepare_spgemm_csr(
            a_data,
            a_indices,
            a_indptr,
            a_shape,
            b_data,
            b_indices,
            b_indptr,
            b_shape,
        )
        if return_meta:
            torch.cuda.synchronize()
            prepare_ms = (time.perf_counter() - t_prepare0) * 1000.0
    elif not isinstance(prepared, SpGEMMPrepared):
        raise TypeError("prepared must be a SpGEMMPrepared instance")

    elapsed_ms = None
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    c_data, c_indices, c_indptr, stage_meta = _run_spgemm_prepared(
        prepared,
        out=out,
        profile=bool(return_meta),
        measure_stage=bool(return_meta),
    )
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    result = (c_data, c_indices, c_indptr, (prepared.n_rows, prepared.n_cols))
    if return_meta:
        # The per-bucket keys are retained for the benchmark's report schema; the
        # ESC / TLE-hash compute path does not populate them.
        meta = {"prepare_ms": prepare_ms, **stage_meta}
        if return_time:
            meta["triton_ms"] = elapsed_ms
            return result, elapsed_ms, meta
        return result, meta
    if return_time:
        return result, elapsed_ms
    return result


def _csr_to_sorted_pairs(data, indices, indptr, n_cols):
    n_rows = int(indptr.numel()) - 1
    row_counts = indptr[1:] - indptr[:-1]
    rows = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        row_counts,
    )
    cols = indices.to(torch.int64)
    keys = rows * max(1, int(n_cols)) + cols
    if keys.numel() == 0:
        return keys, data
    order = torch.argsort(keys)
    return keys[order], data[order]


def _spgemm_pairwise_summary(candidate, reference, value_dtype):
    c_data, c_indices, c_indptr, c_shape = candidate
    r_data, r_indices, r_indptr, r_shape = reference
    if c_shape != r_shape:
        return {
            "match": False,
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "status": f"shape mismatch {c_shape} vs {r_shape}",
        }
    c_keys, c_vals = _csr_to_sorted_pairs(c_data, c_indices, c_indptr, c_shape[1])
    r_keys, r_vals = _csr_to_sorted_pairs(r_data, r_indices, r_indptr, r_shape[1])
    if c_keys.numel() != r_keys.numel():
        return {
            "match": False,
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "status": f"nnz mismatch {c_keys.numel()} vs {r_keys.numel()}",
        }
    if c_keys.numel() > 0 and not torch.equal(c_keys, r_keys):
        return {
            "match": False,
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "status": "sparsity pattern mismatch",
        }
    if c_vals.numel() == 0:
        return {
            "match": True,
            "max_abs_error": 0.0,
            "max_relative_error": 0.0,
            "status": "ok",
        }
    abs_diff = torch.abs(c_vals - r_vals)
    max_abs = float(torch.max(abs_diff).item())
    ref_max = float(torch.max(torch.abs(r_vals)).item())
    max_rel = 0.0 if ref_max == 0.0 else max_abs / ref_max
    atol, rtol = _tolerance_for_dtype(value_dtype)
    match = bool(torch.allclose(c_vals, r_vals, atol=atol, rtol=rtol))
    return {
        "match": match,
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "status": "ok" if match else "value mismatch",
    }


def _to_torch_csr(data, indices, indptr, shape):
    return torch.sparse_csr_tensor(
        indptr.to(torch.int64),
        indices.to(torch.int64),
        data,
        size=shape,
        device=data.device,
    )


def _torch_sparse_to_csr(tensor):
    if tensor.layout == torch.sparse_csr:
        indptr = tensor.crow_indices().to(torch.int64).contiguous()
        indices = tensor.col_indices().to(torch.int32).contiguous()
        data = tensor.values().contiguous()
        shape = (int(tensor.shape[0]), int(tensor.shape[1]))
        return data, indices, indptr, shape
    if tensor.layout == torch.sparse_coo:
        t = tensor.coalesce()
        rows = t.indices()[0].to(torch.int64)
        cols = t.indices()[1].to(torch.int64)
        vals = t.values()
        n_rows, n_cols = int(t.shape[0]), int(t.shape[1])
        if rows.numel() == 0:
            return (
                torch.empty(0, dtype=vals.dtype, device=vals.device),
                torch.empty(0, dtype=torch.int32, device=vals.device),
                torch.zeros(n_rows + 1, dtype=torch.int64, device=vals.device),
                (n_rows, n_cols),
            )
        key = rows * max(1, n_cols) + cols
        order = torch.argsort(key)
        rows = rows[order]
        cols = cols[order]
        vals = vals[order]
        row_counts = torch.bincount(rows, minlength=n_rows)
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=vals.device)
        indptr[1:] = torch.cumsum(row_counts, dim=0)
        return vals, cols.to(torch.int32), indptr, (n_rows, n_cols)
    raise TypeError(f"Unsupported sparse layout: {tensor.layout}")


def _hipsparse_spgemm_algorithm():
    return _hipsparse_lookup(
        "hipsparseSpGEMMAlg_t",
        ("HIPSPARSE_SPGEMM_DEFAULT", "HIPSPARSE_SPGEMM_ALG_DEFAULT"),
    )


def _hipsparse_spgemm_csr_skip_reason(
    value_dtype,
    a_indices_dtype,
    a_indptr_dtype,
    b_indices_dtype=None,
    b_indptr_dtype=None,
):
    b_indices_dtype = a_indices_dtype if b_indices_dtype is None else b_indices_dtype
    b_indptr_dtype = a_indptr_dtype if b_indptr_dtype is None else b_indptr_dtype
    if not _is_rocm_runtime():
        return "hipSPARSE CSR SpGEMM reference requires a ROCm runtime"
    unavailable_reason = _hipsparse_unavailable_reason()
    if unavailable_reason is not None:
        return unavailable_reason
    required_symbols = (
        "hipsparseCreate",
        "hipsparseDestroy",
        "hipsparseCreateCsr",
        "hipsparseDestroySpMat",
        "hipsparseSpGEMM_createDescr",
        "hipsparseSpGEMM_destroyDescr",
        "hipsparseSpGEMM_workEstimation",
        "hipsparseSpGEMM_compute",
        "hipsparseSpGEMM_copy",
        "hipsparseSpMatGetSize",
        "hipsparseCsrSetPointers",
    )
    for symbol in required_symbols:
        if not hasattr(hipsparse, symbol):
            return f"hipSPARSE CSR SpGEMM direct API is unavailable: missing {symbol}"
    if value_dtype not in SUPPORTED_SPGEMM_VALUE_DTYPES:
        return f"hipSPARSE CSR SpGEMM has no supported value dtype mapping for {value_dtype}"
    try:
        _ = _hipsparse_value_type(value_dtype)
        _ = _hipsparse_scalar(value_dtype, 1.0, 0.0)
        _ = _hipsparse_scalar(value_dtype, 0.0, 0.0)
        _ = _hipsparse_index_type(
            a_indptr_dtype, "hipSPARSE CSR SpGEMM A row offsets"
        )
        _ = _hipsparse_index_type(
            a_indices_dtype, "hipSPARSE CSR SpGEMM A column indices"
        )
        _ = _hipsparse_index_type(
            b_indptr_dtype, "hipSPARSE CSR SpGEMM B row offsets"
        )
        _ = _hipsparse_index_type(
            b_indices_dtype, "hipSPARSE CSR SpGEMM B column indices"
        )
        _ = _hipsparse_lookup(
            "hipsparseOperation_t", ("HIPSPARSE_OPERATION_NON_TRANSPOSE",)
        )
        _ = _hipsparse_lookup(
            "hipsparseIndexBase_t", ("HIPSPARSE_INDEX_BASE_ZERO",)
        )
        _ = _hipsparse_spgemm_algorithm()
    except Exception as exc:
        return str(exc)
    return None


def _hipsparse_spgemm_create_descr():
    result = hipsparse.hipsparseSpGEMM_createDescr()
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "hipsparseSpGEMM_createDescr wrapper returned an unexpected result"
        )
    status, descr = result
    _hip_check_result(status, "hipsparseSpGEMM_createDescr")
    return descr


def _hipsparse_create_csr_descriptor(
    spmat_ref,
    n_rows,
    n_cols,
    nnz,
    indptr,
    indices,
    data,
    row_index_type,
    col_index_type,
    index_base,
    value_type,
):
    _hip_check_result(
        hipsparse.hipsparseCreateCsr(
            spmat_ref,
            n_rows,
            n_cols,
            nnz,
            HipPointer.fromObj(indptr.data_ptr()),
            HipPointer.fromObj(indices.data_ptr()),
            HipPointer.fromObj(data.data_ptr()),
            row_index_type,
            col_index_type,
            index_base,
            value_type,
        ),
        "hipsparseCreateCsr",
    )


def _hipsparse_spgemm_work_estimation(
    handle,
    op_enum,
    alpha,
    mat_a,
    mat_b,
    beta,
    mat_c,
    value_type,
    alg,
    descr,
    workspace,
):
    size_out = ctypes.c_size_t()
    _hip_check_result(
        hipsparse.hipsparseSpGEMM_workEstimation(
            handle,
            op_enum,
            op_enum,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            value_type,
            alg,
            descr,
            size_out,
            workspace,
        ),
        "hipsparseSpGEMM_workEstimation",
    )
    return int(size_out.value)


def _hipsparse_spgemm_compute(
    handle,
    op_enum,
    alpha,
    mat_a,
    mat_b,
    beta,
    mat_c,
    value_type,
    alg,
    descr,
    workspace,
):
    size_out = ctypes.c_size_t()
    _hip_check_result(
        hipsparse.hipsparseSpGEMM_compute(
            handle,
            op_enum,
            op_enum,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            value_type,
            alg,
            descr,
            size_out,
            workspace,
        ),
        "hipsparseSpGEMM_compute",
    )
    return int(size_out.value)


def _prepare_spgemm_csr_ref_hipsparse(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
):
    skip_reason = _hipsparse_spgemm_csr_skip_reason(
        a_data.dtype,
        a_indices.dtype,
        a_indptr.dtype,
        b_indices.dtype,
        b_indptr.dtype,
    )
    if skip_reason is not None:
        raise RuntimeError(skip_reason)
    if not all(
        torch.is_tensor(t)
        for t in (a_data, a_indices, a_indptr, b_data, b_indices, b_indptr)
    ):
        raise TypeError("all CSR inputs must be torch.Tensor")
    if not all(t.is_cuda for t in (a_data, a_indices, a_indptr, b_data, b_indices, b_indptr)):
        raise ValueError("all CSR inputs must be CUDA tensors")
    if not all(
        t.device == a_data.device
        for t in (a_indices, a_indptr, b_data, b_indices, b_indptr)
    ):
        raise ValueError("all CSR inputs must be on the same CUDA device")
    if a_data.dtype != b_data.dtype:
        raise TypeError("A and B value dtypes must match for direct hipSPARSE SpGEMM")
    if a_data.ndim != 1 or a_indices.ndim != 1 or a_indptr.ndim != 1:
        raise ValueError("A CSR tensors must be 1D")
    if b_data.ndim != 1 or b_indices.ndim != 1 or b_indptr.ndim != 1:
        raise ValueError("B CSR tensors must be 1D")
    if a_data.numel() != a_indices.numel():
        raise ValueError("A data and indices must have the same length")
    if b_data.numel() != b_indices.numel():
        raise ValueError("B data and indices must have the same length")

    m, k_a = int(a_shape[0]), int(a_shape[1])
    k_b, n = int(b_shape[0]), int(b_shape[1])
    if k_a != k_b:
        raise ValueError(f"inner dimensions must match, got {k_a} and {k_b}")
    if a_indptr.numel() != m + 1:
        raise ValueError(f"A indptr length must be n_rows+1={m + 1}")
    if b_indptr.numel() != k_b + 1:
        raise ValueError(f"B indptr length must be n_rows+1={k_b + 1}")

    a_data = a_data.contiguous()
    a_indices = a_indices.contiguous()
    a_indptr = a_indptr.contiguous()
    b_data = b_data.contiguous()
    b_indices = b_indices.contiguous()
    b_indptr = b_indptr.contiguous()

    c_indptr = torch.zeros(m + 1, dtype=a_indptr.dtype, device=a_data.device)
    # Some hip-python wrappers still validate pointer objects even when nnz == 0.
    c_indices_placeholder = torch.empty(1, dtype=a_indices.dtype, device=a_data.device)
    c_data_placeholder = torch.empty(1, dtype=a_data.dtype, device=a_data.device)

    value_type = _hipsparse_value_type(a_data.dtype)
    alpha = _hipsparse_scalar(a_data.dtype, 1.0, 0.0)
    beta = _hipsparse_scalar(a_data.dtype, 0.0, 0.0)
    a_row_index_type = _hipsparse_index_type(
        a_indptr.dtype, "hipSPARSE CSR SpGEMM A row offsets"
    )
    a_col_index_type = _hipsparse_index_type(
        a_indices.dtype, "hipSPARSE CSR SpGEMM A column indices"
    )
    b_row_index_type = _hipsparse_index_type(
        b_indptr.dtype, "hipSPARSE CSR SpGEMM B row offsets"
    )
    b_col_index_type = _hipsparse_index_type(
        b_indices.dtype, "hipSPARSE CSR SpGEMM B column indices"
    )
    c_col_index_type = _hipsparse_index_type(
        a_indices.dtype, "hipSPARSE CSR SpGEMM C column indices"
    )
    op_enum = _hipsparse_lookup(
        "hipsparseOperation_t", ("HIPSPARSE_OPERATION_NON_TRANSPOSE",)
    )
    index_base = _hipsparse_lookup(
        "hipsparseIndexBase_t", ("HIPSPARSE_INDEX_BASE_ZERO",)
    )
    alg = _hipsparse_spgemm_algorithm()

    handle = None
    mat_a = None
    mat_b = None
    mat_c = None
    descr = None
    buffer1 = 0
    buffer2 = 0
    buffer1_allocated = False
    buffer2_allocated = False
    try:
        handle = _hip_check_result(hipsparse.hipsparseCreate(), "hipsparseCreate")
        ptr_type = type(handle)

        mat_a = ptr_type()
        mat_b = ptr_type()
        mat_c = ptr_type()
        _hipsparse_create_csr_descriptor(
            mat_a.createRef(),
            m,
            k_a,
            int(a_data.numel()),
            a_indptr,
            a_indices,
            a_data,
            a_row_index_type,
            a_col_index_type,
            index_base,
            value_type,
        )
        _hipsparse_create_csr_descriptor(
            mat_b.createRef(),
            k_b,
            n,
            int(b_data.numel()),
            b_indptr,
            b_indices,
            b_data,
            b_row_index_type,
            b_col_index_type,
            index_base,
            value_type,
        )
        _hipsparse_create_csr_descriptor(
            mat_c.createRef(),
            m,
            n,
            0,
            c_indptr,
            c_indices_placeholder,
            c_data_placeholder,
            a_row_index_type,
            c_col_index_type,
            index_base,
            value_type,
        )
        descr = _hipsparse_spgemm_create_descr()

        buffer1_size = _hipsparse_spgemm_work_estimation(
            handle,
            op_enum,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            value_type,
            alg,
            descr,
            0,
        )
        if buffer1_size > 0:
            buffer1 = _hip_check_result(hip.hipMalloc(buffer1_size), "hipMalloc")
            buffer1_allocated = True
        _ = _hipsparse_spgemm_work_estimation(
            handle,
            op_enum,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            value_type,
            alg,
            descr,
            buffer1,
        )

        buffer2_size = _hipsparse_spgemm_compute(
            handle,
            op_enum,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            value_type,
            alg,
            descr,
            0,
        )
        if buffer2_size > 0:
            buffer2 = _hip_check_result(hip.hipMalloc(buffer2_size), "hipMalloc")
            buffer2_allocated = True
        _ = _hipsparse_spgemm_compute(
            handle,
            op_enum,
            alpha,
            mat_a,
            mat_b,
            beta,
            mat_c,
            value_type,
            alg,
            descr,
            buffer2,
        )

        rows_out = ctypes.c_int64()
        cols_out = ctypes.c_int64()
        nnz_out = ctypes.c_int64()
        _hip_check_result(
            hipsparse.hipsparseSpMatGetSize(mat_c, rows_out, cols_out, nnz_out),
            "hipsparseSpMatGetSize",
        )
        nnz_c = int(nnz_out.value)
        c_indices = torch.empty(nnz_c, dtype=a_indices.dtype, device=a_data.device)
        c_data = torch.empty(nnz_c, dtype=a_data.dtype, device=a_data.device)
        _hip_check_result(
            hipsparse.hipsparseCsrSetPointers(
                mat_c,
                HipPointer.fromObj(c_indptr.data_ptr()),
                HipPointer.fromObj(c_indices.data_ptr()),
                HipPointer.fromObj(c_data.data_ptr()),
            ),
            "hipsparseCsrSetPointers",
        )
        return {
            "backend": "hipsparse",
            "buffer1_size": buffer1_size,
            "buffer2_size": buffer2_size,
            "compute_size_out": ctypes.c_size_t(buffer2_size),
            "handle": handle,
            "mat_a": mat_a,
            "mat_b": mat_b,
            "mat_c": mat_c,
            "descr": descr,
            "buffer1": buffer1,
            "buffer2": buffer2,
            "buffer1_allocated": buffer1_allocated,
            "buffer2_allocated": buffer2_allocated,
            "op_enum": op_enum,
            "alpha": alpha,
            "beta": beta,
            "value_type": value_type,
            "alg": alg,
            "result": (c_data, c_indices, c_indptr, (m, n)),
        }
    finally:
        if handle is None and descr is not None:
            try:
                _hip_check_result(
                    hipsparse.hipsparseSpGEMM_destroyDescr(descr),
                    "hipsparseSpGEMM_destroyDescr",
                )
            except Exception:
                pass
        for name, mat in (
            ("hipsparseDestroySpMat(C)", mat_c),
            ("hipsparseDestroySpMat(B)", mat_b),
            ("hipsparseDestroySpMat(A)", mat_a),
        ):
            if handle is None and mat is not None:
                try:
                    _hip_check_result(hipsparse.hipsparseDestroySpMat(mat), name)
                except Exception:
                    pass
        if handle is None and buffer2_allocated:
            try:
                _hip_check_result(hip.hipFree(buffer2), "hipFree(buffer2)")
            except Exception:
                pass
        if handle is None and buffer1_allocated:
            try:
                _hip_check_result(hip.hipFree(buffer1), "hipFree(buffer1)")
            except Exception:
                pass


def _run_spgemm_csr_ref_hipsparse_prepared(state):
    _hip_check_result(
        hipsparse.hipsparseSpGEMM_compute(
            state["handle"],
            state["op_enum"],
            state["op_enum"],
            state["alpha"],
            state["mat_a"],
            state["mat_b"],
            state["beta"],
            state["mat_c"],
            state["value_type"],
            state["alg"],
            state["descr"],
            state["compute_size_out"],
            state["buffer2"],
        ),
        "hipsparseSpGEMM_compute",
    )
    _hip_check_result(
        hipsparse.hipsparseSpGEMM_copy(
            state["handle"],
            state["op_enum"],
            state["op_enum"],
            state["alpha"],
            state["mat_a"],
            state["mat_b"],
            state["beta"],
            state["mat_c"],
            state["value_type"],
            state["alg"],
            state["descr"],
        ),
        "hipsparseSpGEMM_copy",
    )
    return state["result"]


def _destroy_spgemm_csr_ref_hipsparse_prepared(state):
    descr = state.get("descr")
    mat_c = state.get("mat_c")
    mat_b = state.get("mat_b")
    mat_a = state.get("mat_a")
    buffer2_allocated = bool(state.get("buffer2_allocated"))
    buffer1_allocated = bool(state.get("buffer1_allocated"))
    buffer2 = state.get("buffer2", 0)
    buffer1 = state.get("buffer1", 0)
    handle = state.get("handle")
    if descr is not None:
        try:
            _hip_check_result(
                hipsparse.hipsparseSpGEMM_destroyDescr(descr),
                "hipsparseSpGEMM_destroyDescr",
            )
        except Exception:
            pass
    for name, mat in (
        ("hipsparseDestroySpMat(C)", mat_c),
        ("hipsparseDestroySpMat(B)", mat_b),
        ("hipsparseDestroySpMat(A)", mat_a),
    ):
        if mat is not None:
            try:
                _hip_check_result(hipsparse.hipsparseDestroySpMat(mat), name)
            except Exception:
                pass
    if buffer2_allocated:
        try:
            _hip_check_result(hip.hipFree(buffer2), "hipFree(buffer2)")
        except Exception:
            pass
    if buffer1_allocated:
        try:
            _hip_check_result(hip.hipFree(buffer1), "hipFree(buffer1)")
        except Exception:
            pass
    if handle is not None:
        try:
            _hip_check_result(hipsparse.hipsparseDestroy(handle), "hipsparseDestroy")
        except Exception:
            pass


def _spgemm_csr_ref_hipsparse(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    return_metadata=False,
):
    state = _prepare_spgemm_csr_ref_hipsparse(
        a_data,
        a_indices,
        a_indptr,
        a_shape,
        b_data,
        b_indices,
        b_indptr,
        b_shape,
    )
    try:
        result = _run_spgemm_csr_ref_hipsparse_prepared(state)
        metadata = {
            "backend": "hipsparse",
            "buffer1_size": int(state.get("buffer1_size", 0)),
            "buffer2_size": int(state.get("buffer2_size", 0)),
        }
        if return_metadata:
            return result, metadata
        return result
    finally:
        _destroy_spgemm_csr_ref_hipsparse_prepared(state)


def _spgemm_csr_sparse_ref_backend(
    value_dtype=None,
    a_indices_dtype=None,
    a_indptr_dtype=None,
    b_indices_dtype=None,
    b_indptr_dtype=None,
):
    if _is_rocm_runtime():
        reason = _hipsparse_spgemm_csr_skip_reason(
            value_dtype,
            a_indices_dtype,
            a_indptr_dtype,
            b_indices_dtype,
            b_indptr_dtype,
        )
        if reason is None:
            return "hipsparse", None
        return None, reason
    if cp is None or cpx_sparse is None:
        return None, "CuPy/cuSPARSE is not available"
    return "cupy_cusparse", None


def _benchmark_spgemm_csr_sparse_ref(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    warmup,
    iters,
    value_dtype,
):
    backend, reason = _spgemm_csr_sparse_ref_backend(
        a_data.dtype,
        a_indices.dtype,
        a_indptr.dtype,
        b_indices.dtype,
        b_indptr.dtype,
    )
    result = {
        "backend": backend,
        "values": None,
        "ms": None,
        "reason": reason,
    }
    if backend is None:
        return result

    if backend == "hipsparse":
        values, ms = _benchmark_prepared_cuda_op(
            lambda: _prepare_spgemm_csr_ref_hipsparse(
                a_data,
                a_indices,
                a_indptr,
                a_shape,
                b_data,
                b_indices,
                b_indptr,
                b_shape,
            ),
            _run_spgemm_csr_ref_hipsparse_prepared,
            _destroy_spgemm_csr_ref_hipsparse_prepared,
            warmup=warmup,
            iters=iters,
        )
        result["values"] = values
        result["ms"] = ms
        result["reason"] = None
        return result

    a_cp = cpx_sparse.csr_matrix(
        (
            _cupy_from_torch(a_data),
            _cupy_from_torch(a_indices.to(torch.int64)),
            _cupy_from_torch(a_indptr.to(torch.int64)),
        ),
        shape=a_shape,
    )
    b_cp = cpx_sparse.csr_matrix(
        (
            _cupy_from_torch(b_data),
            _cupy_from_torch(b_indices.to(torch.int64)),
            _cupy_from_torch(b_indptr.to(torch.int64)),
        ),
        shape=b_shape,
    )
    c_cp, ms = _benchmark_cuda_op(lambda: a_cp @ b_cp, warmup=warmup, iters=iters)
    c_coo = c_cp.tocoo()
    rows = _torch_from_cupy(c_coo.row).to(torch.int64)
    cols = _torch_from_cupy(c_coo.col).to(torch.int64)
    vals = _torch_from_cupy(c_coo.data).to(value_dtype)
    c_t = torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        vals,
        (int(a_shape[0]), int(b_shape[1])),
        device=a_data.device,
    ).coalesce()
    result["values"] = _torch_sparse_to_csr(c_t)
    result["ms"] = ms
    result["reason"] = None
    return result


def benchmark_spgemm_case(
    n_rows=1024,
    n_inner=1024,
    n_cols=1024,
    nnz_a=16384,
    nnz_b=16384,
    value_dtype=torch.float32,
    warmup=10,
    iters=30,
    run_cusparse=True,
):
    """Benchmark CSR SpGEMM and compare with torch/cuSPARSE baselines."""
    if value_dtype not in SUPPORTED_SPGEMM_VALUE_DTYPES:
        raise TypeError("value_dtype must be torch.float32 or torch.float64")
    device = torch.device("cuda")
    a_data, a_indices, a_indptr = _build_random_csr(
        n_rows, n_inner, nnz_a, value_dtype, torch.int32, device
    )
    b_data, b_indices, b_indptr = _build_random_csr(
        n_inner, n_cols, nnz_b, value_dtype, torch.int32, device
    )

    prepared = prepare_spgemm_csr(
        a_data,
        a_indices,
        a_indptr,
        (n_rows, n_inner),
        b_data,
        b_indices,
        b_indptr,
        (n_inner, n_cols),
    )
    op = lambda: flagsparse_spgemm_csr(prepared=prepared, return_time=False)
    triton_result, triton_ms = _benchmark_cuda_op(op, warmup=warmup, iters=iters)

    a_t = _to_torch_csr(a_data, a_indices, a_indptr, (n_rows, n_inner))
    b_t = _to_torch_csr(b_data, b_indices, b_indptr, (n_inner, n_cols))

    pytorch_reason = None
    pytorch_ms = None
    pytorch_result = None
    try:
        torch_op = lambda: torch.sparse.mm(a_t, b_t)
        pytorch_sparse, pytorch_ms = _benchmark_cuda_op(
            torch_op, warmup=warmup, iters=iters
        )
        pytorch_result = _torch_sparse_to_csr(pytorch_sparse)
    except Exception as exc:
        pytorch_reason = str(exc)
        a_coo = a_t.to_sparse_coo().coalesce()
        b_coo = b_t.to_sparse_coo().coalesce()
        torch_op = lambda: torch.sparse.mm(a_coo, b_coo)
        pytorch_sparse, pytorch_ms = _benchmark_cuda_op(
            torch_op, warmup=warmup, iters=iters
        )
        pytorch_result = _torch_sparse_to_csr(pytorch_sparse)

    triton_summary = _spgemm_pairwise_summary(
        triton_result, pytorch_result, value_dtype
    )

    cusparse_ms = None
    cusparse_reason = None
    cusparse_match = None
    sparse_ref_backend, sparse_ref_reason = _spgemm_csr_sparse_ref_backend(
        value_dtype, a_indices.dtype, a_indptr.dtype, b_indices.dtype, b_indptr.dtype
    )
    if run_cusparse and sparse_ref_backend == "hipsparse":
        # DCU/ROCm: hipSPARSE SpGEMM stands in for the cuSPARSE baseline.
        try:
            c_ref, cusparse_ms = _benchmark_prepared_cuda_op(
                lambda: _prepare_spgemm_csr_ref_hipsparse(
                    a_data,
                    a_indices,
                    a_indptr,
                    (n_rows, n_inner),
                    b_data,
                    b_indices,
                    b_indptr,
                    (n_inner, n_cols),
                ),
                _run_spgemm_csr_ref_hipsparse_prepared,
                _destroy_spgemm_csr_ref_hipsparse_prepared,
                warmup=warmup,
                iters=iters,
            )
            cusparse_match = _spgemm_pairwise_summary(
                triton_result, c_ref, value_dtype
            )["match"]
        except Exception as exc:
            cusparse_reason = str(exc)
    elif run_cusparse:
        if cp is None or cpx_sparse is None:
            cusparse_reason = sparse_ref_reason or "CuPy/cuSPARSE is not available"
        else:
            try:
                a_cp = cpx_sparse.csr_matrix(
                    (
                        _cupy_from_torch(a_data),
                        _cupy_from_torch(a_indices.to(torch.int64)),
                        _cupy_from_torch(a_indptr.to(torch.int64)),
                    ),
                    shape=(n_rows, n_inner),
                )
                b_cp = cpx_sparse.csr_matrix(
                    (
                        _cupy_from_torch(b_data),
                        _cupy_from_torch(b_indices.to(torch.int64)),
                        _cupy_from_torch(b_indptr.to(torch.int64)),
                    ),
                    shape=(n_inner, n_cols),
                )
                c_cp, cusparse_ms = _benchmark_cuda_op(
                    lambda: a_cp @ b_cp, warmup=warmup, iters=iters
                )
                c_coo = c_cp.tocoo()
                rows = _torch_from_cupy(c_coo.row).to(torch.int64)
                cols = _torch_from_cupy(c_coo.col).to(torch.int64)
                vals = _torch_from_cupy(c_coo.data).to(value_dtype)
                c_t = torch.sparse_coo_tensor(
                    torch.stack([rows, cols]), vals, (n_rows, n_cols), device=device
                ).coalesce()
                c_ref = _torch_sparse_to_csr(c_t)
                cusparse_match = _spgemm_pairwise_summary(
                    triton_result, c_ref, value_dtype
                )["match"]
            except Exception as exc:
                cusparse_reason = str(exc)

    return {
        "parameters": {
            "n_rows": n_rows,
            "n_inner": n_inner,
            "n_cols": n_cols,
            "nnz_a": nnz_a,
            "nnz_b": nnz_b,
            "value_dtype": str(value_dtype),
            "warmup": warmup,
            "iters": iters,
        },
        "performance": {
            "triton_ms": triton_ms,
            "pytorch_ms": pytorch_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": (
                pytorch_ms / triton_ms if (pytorch_ms and triton_ms > 0) else None
            ),
            "triton_speedup_vs_cusparse": (
                cusparse_ms / triton_ms if (cusparse_ms and triton_ms > 0) else None
            ),
        },
        "verification": {
            "triton_match_pytorch": triton_summary["match"],
            "triton_max_abs_error": triton_summary["max_abs_error"],
            "triton_max_relative_error": triton_summary["max_relative_error"],
            "cusparse_match_pytorch": cusparse_match,
        },
        "backend_status": {
            "pytorch_unavailable_reason": pytorch_reason,
            "cusparse_unavailable_reason": cusparse_reason,
        },
        "samples": {
            "triton": triton_result,
            "pytorch": pytorch_result,
        },
    }
