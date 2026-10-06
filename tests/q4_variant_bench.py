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

"""q4 variants of existing operators, for the operators' own benchmark scripts.

docs/NEW_OPERATORS_CUSPARSE_12_5.md lists mixed-precision, integer, transposed and
layout variants of spmv_csr / spmv_coo / spmv_csc / spmm_csr / spmm_coo / sddmm_csr.
Their scripts pass ``--q4-variants``; this module measures each variant on each
.mtx matrix and appends one row per (variant, matrix) to the script's CSV, under
that script's own column names (``COLUMN_MAPS``), tagged with a ``variant`` column.
The script's original rows are left exactly as they were.

Timing matches the scripts' own rows: where the operator has a prepare step
(prepare_spmv_csr / _coo / _csc, prepare_spmm_*_route, prepare_sddmm_csr) it runs
once outside the timed window, as cuSPARSE's descriptor setup does; the mixed
precision / int8 paths have no prepare step and are timed as one call. The row's
``timed`` column says which.

Per row: FlagSparse, native cuSPARSE with the same type combination (CUDA only;
``tests/cusparse_generic_baseline.py``), and PyTorch on the same matrix. Accuracy
is checked against a CPU float64 / complex128 SciPy product; int outputs exactly.
"""

import csv
import glob
import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

import cusparse_generic_baseline as cusparse_baseline
import flagsparse as fs
from benchmark_utils import accelerator_device
from flagsparse.sparse_operations import mixed_spmx
from flagsparse.sparse_operations._common import _benchmark_cuda_op
from mtx_fast import read_scipy_csr

DENSE_COLS = 32  # N for SpMM, K for SDDMM
_SDDMM_REF_CHUNK = 1 << 20


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str  # spmv / spmm / sddmm
    fmt: str  # csr / coo / csc
    value: torch.dtype
    out: torch.dtype
    op_a: str = "non"
    op_b: str = "non"
    layout: str = "row"
    x: torch.dtype = None  # SpMV vector dtype when it differs from the matrix's


f16, f32, c32, i8, i32 = torch.float16, torch.float32, torch.complex64, torch.int8, torch.int32

VARIANTS = {
    "spmv_csr": [
        Variant("spmv_csr_f16f32_int_non", "spmv", "csr", f16, f32),
        Variant("spmv_csr_f16_int_non", "spmv", "csr", f16, f16),
        Variant("spmv_csr_f32c32_int_non", "spmv", "csr", f32, c32, x=c32),
        Variant("spmv_csr_c32_int_non", "spmv", "csr", c32, c32),
        Variant("spmv_csr_f32_int_trans", "spmv", "csr", f32, f32, op_a="trans"),
        Variant("spmv_csr_i8f32_int_non", "spmv", "csr", i8, f32),
        Variant("spmv_csr_i8i32_int_non", "spmv", "csr", i8, i32),
        Variant("spmv_csr_c32_int_conj", "spmv", "csr", c32, c32, op_a="conj"),
    ],
    "spmv_coo": [
        Variant("spmv_coo_f16f32_int_non", "spmv", "coo", f16, f32),
        Variant("spmv_coo_f32_int_trans", "spmv", "coo", f32, f32, op_a="trans"),
        Variant("spmv_coo_c32_int_non", "spmv", "coo", c32, c32),
        Variant("spmv_coo_f16_int_non", "spmv", "coo", f16, f16),
        Variant("spmv_coo_c32_int_conj", "spmv", "coo", c32, c32, op_a="conj"),
        Variant("spmv_coo_i8i32_int_non", "spmv", "coo", i8, i32),
    ],
    "spmv_csc": [
        Variant("spmv_csc_f32_int_non", "spmv", "csc", f32, f32),
        Variant("spmv_csc_c32_int_non", "spmv", "csc", c32, c32),
        Variant("spmv_csc_f16_int_non", "spmv", "csc", f16, f16),
    ],
    "spmm_csr": [
        Variant("spmm_csr_f16f32_int_non_non_row", "spmm", "csr", f16, f32),
        Variant("spmm_csr_f16_int_non_non_row", "spmm", "csr", f16, f16),
        Variant("spmm_csr_f32_int_non_non_col", "spmm", "csr", f32, f32, layout="col"),
        Variant("spmm_csr_f32_int_non_trans_row", "spmm", "csr", f32, f32, op_b="trans"),
        Variant("spmm_csr_c32_int_non_non_row", "spmm", "csr", c32, c32),
        Variant("spmm_csr_i8i32_int_non_non_row", "spmm", "csr", i8, i32),
        Variant("spmm_csr_f32_int_trans_non_row", "spmm", "csr", f32, f32, op_a="trans"),
    ],
    # Tagged here so the row carries a cuSPARSE Blocked-ELL baseline: the script's own
    # CuPy baseline has no Blocked-ELL SpMM, so its rows have no speedup on CUDA.
    "spmm_bell": [
        Variant("spmm_bell_f32_int_non_non_row", "spmm", "bell", f32, f32),
    ],
    "spmm_coo": [
        Variant("spmm_coo_c32_int_non_non_row", "spmm", "coo", c32, c32),
        Variant("spmm_coo_f16_int_non_non_row", "spmm", "coo", f16, f16),
        Variant("spmm_coo_i8i32_int_non_non_row", "spmm", "coo", i8, i32),
    ],
    "spmm_csc": [
        Variant("spmm_csc_f16_int_non_non_row", "spmm", "csc", f16, f16),
    ],
    "sddmm_csr": [
        Variant("sddmm_csr_f32_int_non_non_col", "sddmm", "csr", f32, f32, layout="col"),
        Variant("sddmm_csr_f32_int_non_trans_row", "sddmm", "csr", f32, f32, op_b="trans"),
        Variant("sddmm_csr_f32_int_trans_non_row", "sddmm", "csr", f32, f32, op_a="trans"),
        Variant("sddmm_csr_c32_int_non_non_row", "sddmm", "csr", c32, c32),
        Variant("sddmm_csr_f16_int_non_non_row", "sddmm", "csr", f16, f16),
    ],
}

# generic key -> the column each script already uses for it. Keys missing from a
# map keep their generic name (the header is widened for them).
COLUMN_MAPS = {
    "spmv_csr": {
        "value_dtype": "dtype", "n_rows": "m", "n_cols": "n", "ours_ms": "ms",
        "speedup_vs_vendor": "speedup_vs_vendor", "speedup_vs_pytorch": "triton_speedup_vs_pytorch",
        "vendor_reason": "vendor_reason", "error": "reason",
    },
    "spmv_coo": {
        "ours_ms": "opt_ms", "vendor_ms": "cusparse_ms", "speedup_vs_vendor": "opt_speedup_vs_cusparse",
        "speedup_vs_pytorch": "opt_speedup_vs_pytorch", "max_error": "err_opt", "vendor_max_error": "err_cu",
    },
    "spmv_csc": {
        "ours_ms": "csc_ms", "vendor_ms": "cusparse_ms", "speedup_vs_vendor": "csc_speedup_vs_cusparse",
        "speedup_vs_pytorch": "csc_speedup_vs_pytorch", "max_error": "err",
    },
    "spmm_csr": {
        "ours_ms": "triton_ms", "vendor_ms": "cusparse_ms", "speedup_vs_vendor": "triton_speedup_vs_cusparse",
        "speedup_vs_pytorch": "triton_speedup_vs_pytorch", "vendor_max_error": "err_cu",
    },
    "spmm_bell": {
        "value_dtype": "dtype", "ours_ms": "ms", "speedup_vs_vendor": "speedup_vs_vendor",
        "speedup_vs_pytorch": "triton_speedup_vs_pytorch", "vendor_max_error": "err_vs_vendor",
        "error": "reason",
    },
    "spmm_coo": {
        "value_dtype": "dtype", "ours_ms": "ms", "vendor_ms": "cusparse_ms", "pytorch_ms": "torch_ms",
        "speedup_vs_vendor": "cusparse_vs_alg_speedup", "speedup_vs_pytorch": "triton_speedup_vs_pytorch",
        "vendor_max_error": "err_vs_cusparse", "vendor_reason": "cusparse_reason", "error": "reason",
    },
    "sddmm_csr": {
        "ours_ms": "triton_ms", "vendor_ms": "cusparse_ms", "speedup_vs_vendor": "triton_speedup_vs_cusparse",
        "speedup_vs_pytorch": "triton_speedup_vs_pytorch", "vendor_max_error": "err_cu",
        "vendor_reason": "cu_reason", "dense_cols": "k",
    },
}
# Script columns that must hold a fixed value on q4 rows (e.g. their ``alg``).
CONSTANTS = {
    "spmv_csr": {"alg": "q4"},
    "spmm_coo": {"alg": "q4"},
    "spmm_bell": {"alg": "q4", "block_dim": 2, "layout": "row", "index_dtype": "int32"},
}
BELL_BLOCK = 2  # the runner's spmm_bell command measures --block-dims 2
# Blocked-ELL pads every block row to the widest one; past this much combined
# data+index storage the conversion is skipped rather than risk an OOM. Mirrors
# tests/test_spmm_bell.py's DEFAULT_MAX_BELL_STORAGE_MB (2048 MiB, data+index) --
# that test already clears ASIC_680ks.mtx at this threshold, whereas the old
# values-only 1 GiB cap here rejected it on every backend (MUSA/DCU/MACA).
_BELL_MAX_STORAGE_BYTES = 2048 * 1024 * 1024

_TOL = {torch.float16: 2e-3, torch.bfloat16: 1e-2, torch.float32: 1e-4, torch.complex64: 1e-4}


def _name(dtype):
    return str(dtype).replace("torch.", "")


def _compute_dtype(v):
    if v.x is not None:  # real A x complex x
        return v.x
    if v.value in (torch.float16, torch.bfloat16):
        return torch.float32
    if v.value == torch.int8:
        return v.out
    return v.value


def _wide_np(t):
    return t.detach().cpu().to(torch.complex128 if t.is_complex() else torch.float64).numpy()


def _rand(shape, dtype, gen):
    if dtype == torch.int8:
        return torch.randint(-3, 4, shape, dtype=dtype, generator=gen)
    if dtype.is_complex:
        return torch.randn(shape, dtype=dtype, generator=gen)
    return torch.randn(shape, generator=gen).to(dtype)


def _matrix_values(csr, dtype, gen):
    """Values in ``dtype``: int8 is drawn (.mtx values do not fit), halves are scaled."""
    if dtype == torch.int8:
        return torch.randint(-3, 4, (csr.nnz,), dtype=dtype, generator=gen)
    data = np.asarray(csr.data)
    if np.iscomplexobj(data) and not dtype.is_complex:
        data = data.real
    t = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float64))
    if dtype.is_complex:
        t = torch.complex(t, torch.randn(csr.nnz, dtype=torch.float64, generator=gen))
    elif dtype in (torch.float16, torch.bfloat16):
        t = t / max(1.0, float(t.abs().max()))
    return t.to(dtype)


def _error(got, ref):
    got = _wide_np(got) if torch.is_tensor(got) else np.asarray(got)
    scale = max(1.0, float(np.abs(ref).max())) if ref.size else 1.0
    return float(np.abs(got - ref).max() / scale) if ref.size else 0.0


def _passes(out_dtype, err):
    if not (out_dtype.is_floating_point or out_dtype.is_complex):
        return err == 0.0
    return err <= _TOL.get(out_dtype, 1e-10)


def _ratio(base, ours):
    return None if base is None or not ours else base / ours


def _op_sp(a, op):
    return a if op == "non" else (a.T if op == "trans" else a.conj().T)


def _op_t(t, op):
    return t if op == "non" else (t.t() if op == "trans" else t.conj().t())


def _col_major(t):
    return t.t().contiguous().t()


def _sparse_torch(ptr, cols, values, shape, op):
    """PyTorch CSR of op(A), built outside the timed window."""
    a = torch.sparse_csr_tensor(ptr.long(), cols.long(), values, shape)
    if op == "non":
        return a
    t = a.to_sparse_coo().t()
    if op == "conj":
        t = t.conj().resolve_conj()
    return t.coalesce().to_sparse_csr()


class _Matrix:
    """One .mtx file on the device, in each layout the variants need."""

    def __init__(self, path, dev):
        self.path = path
        self.csr = read_scipy_csr(path)
        self.shape = tuple(int(s) for s in self.csr.shape)
        self.dev = dev
        self.ptr = torch.from_numpy(self.csr.indptr.astype(np.int32)).to(dev)
        self.cols = torch.from_numpy(self.csr.indices.astype(np.int32)).to(dev)
        rows = np.repeat(np.arange(self.shape[0], dtype=np.int32), np.diff(self.csr.indptr))
        self.rows = torch.from_numpy(rows).to(dev)
        csc = self.csr.tocsc()
        csc.sort_indices()
        self.csc_order = torch.from_numpy(
            sp.csr_matrix(
                (np.arange(self.csr.nnz, dtype=np.float64) + 1, self.csr.indices, self.csr.indptr), self.shape
            ).tocsc().data.astype(np.int64) - 1
        )
        self.csc_ptr = torch.from_numpy(csc.indptr.astype(np.int32)).to(dev)
        self.csc_rows = torch.from_numpy(csc.indices.astype(np.int32)).to(dev)

    def bell(self, values, bd):
        """Blocked-ELL of this matrix: (data[mb, width, bd, bd], block cols[mb, width]).

        Built on the device from the CSR arrays (torch.unique over block keys); rows and
        columns are padded to multiples of ``bd``.
        """
        m, n = self.shape
        mb, nb = -(-m // bd), -(-n // bd)
        rows, cols = self.rows.long(), self.cols.long()
        keys = (rows // bd) * nb + cols // bd
        uniq, inverse = torch.unique(keys, return_inverse=True)
        brow, bcol = uniq // nb, uniq % nb
        counts = torch.bincount(brow, minlength=mb)
        width = max(1, int(counts.max().item())) if uniq.numel() else 1
        stored_values = mb * width * bd * bd
        index_values = mb * width
        storage_bytes = stored_values * values.element_size() + index_values * 4  # index is int32
        if storage_bytes > _BELL_MAX_STORAGE_BYTES:
            raise MemoryError(
                f"Blocked-ELL would store {stored_values} values + {index_values} indices "
                f"({storage_bytes / (1024.0 * 1024.0):.1f} MiB > "
                f"{_BELL_MAX_STORAGE_BYTES / (1024.0 * 1024.0):.1f} MiB guard)"
            )
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(uniq.numel(), device=keys.device) - starts[brow]
        data = torch.zeros(mb, width, bd, bd, dtype=values.dtype, device=values.device)
        data[brow[inverse], slot[inverse], rows % bd, cols % bd] = values
        index = torch.full((mb, width), -1, dtype=torch.int32, device=values.device)
        index[brow, slot] = bcol.to(torch.int32)
        return data, index, (mb * bd, nb * bd)

    def arrays(self, fmt, values):
        """(values, a, b) in cusparse_generic_baseline's order for ``fmt``."""
        if fmt == "csr":
            return values, self.cols, self.ptr
        if fmt == "coo":
            return values, self.rows, self.cols
        return values[self.csc_order.to(values.device)].contiguous(), self.csc_rows, self.csc_ptr

    def scipy(self, values_cpu):
        return sp.csr_matrix((_wide_np(values_cpu), self.csr.indices, self.csr.indptr), self.shape)


# A baseline is only a baseline if it computes the right thing. Loose enough for
# TF32 (cuSPARSE's float32 Blocked-ELL SpMM runs on tensor cores, ~5e-4), tight
# enough to reject a result that silently was not computed: cuSPARSE Blocked-ELL
# SpMM on roadNet-TX (696,692 block rows) returns in 3 us with every output 0 and
# no error status.
_VENDOR_MAX_ERROR = 1e-2


def _vendor(row, build):
    reason = cusparse_baseline.skip_reason()
    if reason is None:
        try:
            plan, check = build()
            plan.run()
            err = check()
            row["vendor_max_error"] = err
            if err <= _VENDOR_MAX_ERROR:
                return plan, None
            plan.close()
            reason = f"cuSPARSE result off by {err:.3g} (relative); baseline not used"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
    row["vendor_reason"] = reason
    return None, reason


def _time_vendor(row, plan, warmup, iters):
    if plan is not None:
        _, row["vendor_ms"] = _benchmark_cuda_op(plan.run, warmup, iters)
        plan.close()


def _time_pytorch(row, build, warmup, iters):
    try:
        op = build()
        _, row["pytorch_ms"] = _benchmark_cuda_op(op, warmup, iters)
    except Exception as exc:  # e.g. MUSA registers no sparse matmul, no sparse int8
        row["pytorch_reason"] = f"{type(exc).__name__}: {exc}"


def _time_cupy(row, mat, values, dense, op_a, op_b, warmup, iters, reference):
    """CuPy CSR of op(A), with the same outside-timing setup as PyTorch."""
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cps
        compute = torch.float32 if values.dtype in (torch.float16, torch.bfloat16, torch.int8) else values.dtype
        compute = torch.promote_types(compute, dense.dtype)
        row["cupy_compute_dtype"] = _name(compute)
        a = cps.csr_matrix((cp.from_dlpack(values.to(compute)),
                            cp.from_dlpack(mat.cols), cp.from_dlpack(mat.ptr)), shape=mat.shape)
        if op_a != "non":
            a = (a.conj().T if op_a == "conj" else a.T).tocsr()
        b = cp.from_dlpack(_op_t(dense, op_b).to(compute).contiguous())
        if b.ndim == 2:
            # CuPy converts a row-major RHS inside sparse matmul. Hoist it out,
            # just like its sparse-format setup and our prepared layout setup.
            b = cp.asfortranarray(b)
        row["cupy_rhs_layout"] = "column-major" if b.ndim == 2 else "vector"
        run = lambda: a @ b
        output, elapsed = _benchmark_cuda_op(run, warmup, iters)
        error = _error(cp.asnumpy(output), reference)
        row["cupy_max_error"] = error
        if error > _TOL.get(compute, 1e-10):
            raise RuntimeError(f"CuPy correctness check failed: {error}")
        row["cupy_ms"] = elapsed
    except Exception as exc:
        row["cupy_reason"] = f"{type(exc).__name__}: {exc}"


def _spmv(v, mat, gen, warmup, iters, row):
    m, n = mat.shape
    x_len = n if v.op_a == "non" else m
    y_len = m if v.op_a == "non" else n
    vals_cpu = _matrix_values(mat.csr, v.value, gen)
    x_cpu = _rand((x_len,), v.x or v.value, gen)
    ref = _op_sp(mat.scipy(vals_cpu), v.op_a) @ _wide_np(x_cpu)
    vals, x = vals_cpu.to(mat.dev), x_cpu.to(mat.dev)
    default_out = v.x or ({torch.int8: torch.int32}.get(v.value, v.value))
    kw = {} if v.out == default_out else {"out_dtype": v.out}
    mixed = mixed_spmx.spmv_needs_mixed(
        v.value, x.dtype, None, kw.get("out_dtype"), coo=v.fmt == "coo"
    ) or (v.fmt == "csc" and v.value in (torch.float16, torch.bfloat16, torch.int8))
    csc_vals = mat.arrays("csc", vals)[0] if v.fmt == "csc" else None
    if mixed:
        row["timed"] = "call"
        if v.fmt == "csr":
            ours = lambda: fs.flagsparse_spmv_csr(vals, mat.cols, mat.ptr, x, shape=mat.shape, op=v.op_a, **kw)  # noqa: E731
        elif v.fmt == "coo":
            ours = lambda: fs.flagsparse_spmv_coo(vals, mat.rows, mat.cols, x, shape=mat.shape, op=v.op_a, **kw)  # noqa: E731
        else:
            ours = lambda: fs.flagsparse_spmv_csc(csc_vals, mat.csc_rows, mat.csc_ptr, x, shape=mat.shape, op=v.op_a, **kw)  # noqa: E731
    else:
        row["timed"] = "prepared"
        if v.fmt == "csr":
            prepared = fs.prepare_spmv_csr(vals, mat.cols, mat.ptr, mat.shape, op=v.op_a)
            ours = lambda: fs.flagsparse_spmv_csr(prepared=prepared, x=x)  # noqa: E731
        elif v.fmt == "coo":
            prepared = fs.prepare_spmv_coo(vals, mat.rows, mat.cols, mat.shape, op=v.op_a)
            ours = lambda: fs.flagsparse_spmv_coo(prepared=prepared, x=x)  # noqa: E731
        else:
            prepared = fs.prepare_spmv_csc(csc_vals, mat.csc_rows, mat.csc_ptr, mat.shape, op=v.op_a)
            ours = lambda: fs.flagsparse_spmv_csc(prepared=prepared, x=x)  # noqa: E731
    first = ours()
    row["max_error"] = _error(first, ref)
    row["out_dtype"] = _name(first.dtype)
    _, row["ours_ms"] = _benchmark_cuda_op(ours, warmup, iters)

    def build_vendor():
        y = torch.empty(y_len, dtype=v.out, device=mat.dev)
        plan = cusparse_baseline.prepare_spmv(v.fmt, mat.arrays(v.fmt, vals), mat.shape, x, y, v.op_a, _compute_dtype(v))
        return plan, lambda: _error(y, ref)

    plan, _ = _vendor(row, build_vendor)
    _time_vendor(row, plan, warmup, iters)

    def build_torch():
        c = _compute_dtype(v) if v.value in (torch.int8, torch.float16, torch.bfloat16) else (v.x or v.value)
        c = torch.float32 if c == torch.int32 else c
        a = _sparse_torch(mat.ptr, mat.cols, vals.to(c), mat.shape, v.op_a)
        xc = x.to(c)
        return lambda: a @ xc

    _time_pytorch(row, build_torch, warmup, iters)
    if os.environ.get("FLAGSPARSE_BENCH_CUPY") == "1":
        _time_cupy(row, mat, vals, x, v.op_a, "non", warmup, iters, ref)
    return v.out


def _spmm(v, mat, gen, warmup, iters, row):
    m, n = mat.shape
    k_eff, m_eff = (n, m) if v.op_a == "non" else (m, n)
    vals_cpu = _matrix_values(mat.csr, v.value, gen)
    b_cpu = _rand((k_eff, DENSE_COLS) if v.op_b == "non" else (DENSE_COLS, k_eff), v.value, gen)
    ref = _op_sp(mat.scipy(vals_cpu), v.op_a) @ _op_t(torch.from_numpy(_wide_np(b_cpu)), v.op_b).numpy()
    vals = vals_cpu.to(mat.dev)
    B = b_cpu.to(mat.dev)
    if v.layout == "col":
        B = _col_major(B)
    default_out = {torch.int8: torch.int32}.get(v.value, v.value)
    if v.fmt == "bell":
        return _spmm_bell(v, mat, vals, vals_cpu, B, ref, warmup, iters, row)
    kw = {} if v.out == default_out else {"out_dtype": v.out}
    if v.fmt == "csc":
        row["timed"] = "call"
        csc_vals, csc_rows, csc_ptr = mat.arrays("csc", vals)
        ours = lambda: fs.flagsparse_spmm_csc(
            csc_vals, csc_rows, csc_ptr, _op_t(B, v.op_b), mat.shape, op=v.op_a, **kw
        )
    elif v.fmt in ("csr", "coo") and mixed_spmx.spmm_needs_mixed(
        v.value, None, kw.get("out_dtype")
    ):
        row["timed"] = "call"
        if v.fmt == "csr":
            ours = lambda: fs.flagsparse_spmm_csr(  # noqa: E731
                vals, mat.cols, mat.ptr, B, mat.shape, op=v.op_a, op_b=v.op_b, **kw
            )
        else:
            ours = lambda: fs.flagsparse_spmm_coo(  # noqa: E731
                vals,
                mat.rows,
                mat.cols,
                B,
                mat.shape,
                op=v.op_a,
                dense_layout=v.layout,
                **kw,
            )
    else:
        row["timed"] = "prepared"
        B_eff = _op_t(B, v.op_b)  # op(B) as a strided view; the run reads its strides
        if v.fmt == "csr":
            prepared = fs.prepare_spmm_csr_route(vals, mat.cols, mat.ptr, mat.shape, op=v.op_a)
            ours = lambda: fs.flagsparse_spmm_csr_run(prepared, B_eff, alg="auto")  # noqa: E731
        else:
            prepared = fs.prepare_spmm_coo_route(vals, mat.rows, mat.cols, mat.shape, op=v.op_a)
            ours = lambda: fs.flagsparse_spmm_coo_run(prepared, B_eff, alg="auto")  # noqa: E731
    first = ours()
    first = first[0] if isinstance(first, tuple) else first
    row["max_error"] = _error(first, ref)
    row["out_dtype"] = _name(first.dtype)
    _, row["ours_ms"] = _benchmark_cuda_op(ours, warmup, iters)

    def build_vendor():
        C = torch.empty(m_eff, DENSE_COLS, dtype=v.out, device=mat.dev)
        if v.layout == "col":
            C = _col_major(C)
        plan = cusparse_baseline.prepare_spmm(
            v.fmt, mat.arrays(v.fmt, vals), mat.shape, B, C, v.op_a, v.op_b, _compute_dtype(v)
        )
        return plan, lambda: _error(C, ref)

    plan, _ = _vendor(row, build_vendor)
    _time_vendor(row, plan, warmup, iters)

    def build_torch():
        c = torch.float32 if v.value in (torch.int8, torch.float16, torch.bfloat16) else v.value
        a = _sparse_torch(mat.ptr, mat.cols, vals.to(c), mat.shape, v.op_a)
        bc = _op_t(B, v.op_b).to(c)
        return lambda: torch.sparse.mm(a, bc)

    _time_pytorch(row, build_torch, warmup, iters)
    if os.environ.get("FLAGSPARSE_BENCH_CUPY") == "1":
        _time_cupy(row, mat, vals, B, v.op_a, v.op_b, warmup, iters, ref)
    if (os.environ.get("FLAGSPARSE_BENCH_CAPI") == "1"
            and v.fmt == "csr" and v.value == torch.float32 and v.op_a == "trans"):
        from flagsparse_capi_bench import prepare_spmm
        target = torch.empty(m_eff, DENSE_COLS, dtype=v.out, device=mat.dev)
        run, close = prepare_spmm(vals, mat.cols, mat.ptr, mat.shape, B, target, v.op_a, v.op_b)
        try:
            result, elapsed = _benchmark_cuda_op(run, warmup, iters)
            row["capi_max_error"] = _error(result, ref)
            if not _passes(v.out, row["capi_max_error"]):
                raise RuntimeError("C API correctness check failed")
            row["capi_ms"] = elapsed
            row["capi_speedup_vs_pytorch"] = _ratio(row.get("pytorch_ms"), elapsed)
            row["capi_speedup_vs_cupy"] = _ratio(row.get("cupy_ms"), elapsed)
        finally:
            close()
    return v.out


def _spmm_bell(v, mat, vals, vals_cpu, B, ref, warmup, iters, row):
    """Blocked-ELL SpMM on the zero-padded matrix; the first n_rows rows are compared."""
    bd = BELL_BLOCK
    m, _ = mat.shape
    data, index, padded = mat.bell(vals, bd)
    B_pad = torch.zeros(padded[1], B.shape[1], dtype=B.dtype, device=B.device)
    B_pad[: B.shape[0]] = B
    row["timed"] = "prepared"
    prepared = fs.prepare_spmm_bell_route(data, index, padded, block_dim=bd)

    def ours():
        out = fs.flagsparse_spmm_bell_run(prepared, B_pad)
        return out[0] if isinstance(out, tuple) else out

    first = ours()
    row["max_error"] = _error(first[:m], ref)
    row["out_dtype"] = _name(first.dtype)
    _, row["ours_ms"] = _benchmark_cuda_op(ours, warmup, iters)

    def build_vendor():
        mb, width = index.shape
        ell_values = data.permute(0, 2, 1, 3).reshape(mb * bd, width * bd).contiguous()
        C = torch.empty(padded[0], B.shape[1], dtype=v.out, device=B.device)
        plan = cusparse_baseline.prepare_spmm(
            "bell", (ell_values, index, bd), padded, B_pad, C, "non", "non", _compute_dtype(v)
        )
        return plan, lambda: _error(C[:m], ref)

    plan, _ = _vendor(row, build_vendor)
    _time_vendor(row, plan, warmup, iters)

    def build_torch():
        a = _sparse_torch(mat.ptr, mat.cols, vals.to(v.value), mat.shape, "non")
        return lambda: torch.sparse.mm(a, B)

    _time_pytorch(row, build_torch, warmup, iters)
    return v.out


def _sddmm_ref(mat, a_eff, b_eff):
    """Compute stored-entry dot products in float64/complex128 chunks."""
    rows = mat.rows.cpu().long().numpy()
    cols = mat.cols.cpu().long().numpy()
    # Complex q4 SDDMM must retain the imaginary component.  The previous
    # float64 buffer silently discarded it while assigning the einsum result,
    # making a correct complex kernel look like a benchmark failure.
    out_dtype = np.complex128 if np.iscomplexobj(a_eff) or np.iscomplexobj(b_eff) else np.float64
    out = np.empty(rows.size, dtype=out_dtype)
    for lo in range(0, rows.size, _SDDMM_REF_CHUNK):
        hi = min(lo + _SDDMM_REF_CHUNK, rows.size)
        out[lo:hi] = np.einsum("ij,ij->i", a_eff[rows[lo:hi]], b_eff[:, cols[lo:hi]].T)
    return out


def _sddmm(v, mat, gen, warmup, iters, row):
    m, n = mat.shape
    kd = DENSE_COLS
    a_cpu = _rand((m, kd) if v.op_a == "non" else (kd, m), v.value, gen)
    b_cpu = _rand((kd, n) if v.op_b == "non" else (n, kd), v.value, gen)
    a_eff = _op_t(torch.from_numpy(_wide_np(a_cpu)), v.op_a).numpy()
    b_eff = _op_t(torch.from_numpy(_wide_np(b_cpu)), v.op_b).numpy()
    ref = _sddmm_ref(mat, a_eff, b_eff)
    A, B = a_cpu.to(mat.dev), b_cpu.to(mat.dev)
    if v.layout == "col":
        A, B = _col_major(A), _col_major(B)

    row["timed"] = "prepared"
    prepared = fs.prepare_sddmm_csr(mat.cols, mat.ptr, mat.shape, k_hint=kd)

    def ours():
        return fs.flagsparse_sddmm_csr(
            x=A, y=B, prepared=prepared, op_a=v.op_a, op_b=v.op_b, dense_layout=v.layout
        )

    first = ours()
    row["max_error"] = _error(first, ref)
    row["out_dtype"] = _name(first.dtype)
    row["dense_cols"] = kd
    _, row["ours_ms"] = _benchmark_cuda_op(ours, warmup, iters)

    def build_vendor():
        out_vals = torch.empty(mat.csr.nnz, dtype=v.out, device=mat.dev)
        plan = cusparse_baseline.prepare_sddmm(
            (None, mat.cols, mat.ptr), mat.shape, A, B, out_vals, v.op_a, v.op_b, v.value
        )
        return plan, lambda: _error(out_vals, ref)

    plan, _ = _vendor(row, build_vendor)
    _time_vendor(row, plan, warmup, iters)

    def build_torch():
        compute = torch.float32 if v.value == torch.float16 else v.value
        pattern = torch.sparse_csr_tensor(
            mat.ptr.long(), mat.cols.long(), torch.zeros(mat.csr.nnz, dtype=compute, device=mat.dev), mat.shape
        )
        row["pytorch_compute_dtype"] = _name(compute)
        a_op, b_op = _op_t(A, v.op_a).to(compute), _op_t(B, v.op_b).to(compute)
        return lambda: torch.sparse.sampled_addmm(pattern, a_op, b_op, beta=0.0)

    _time_pytorch(row, build_torch, warmup, iters)
    if os.environ.get("FLAGSPARSE_BENCH_CUPY") == "1":
        _time_cupy_sddmm(row, mat, A, B, v.op_a, v.op_b, warmup, iters, ref)
    return v.out


def _time_cupy_sddmm(row, mat, A, B, op_a, op_b, warmup, iters, reference):
    """CuPy gather/product/reduction; CuPy has no native sampled_addmm API.

    Chunking bounds temporary memory; gathering and reduction remain timed.
    Complex multiplication is not conjugated, matching sampled_addmm.
    """
    try:
        import cupy as cp
        compute = torch.float32 if A.dtype == torch.float16 else A.dtype
        a = cp.from_dlpack(_op_t(A, op_a).to(compute).contiguous())
        bt = cp.from_dlpack(_op_t(B, op_b).t().to(compute).contiguous())
        rows, cols = cp.from_dlpack(mat.rows), cp.from_dlpack(mat.cols)
        row["cupy_compute_dtype"] = _name(compute)
        row["cupy_method"] = "chunked gather, multiply, sum (not native cuSPARSE SDDMM)"
        row["cupy_chunk_nnz"] = 262144

        def run():
            out = cp.empty(cols.size, dtype=a.dtype)
            for lo in range(0, cols.size, 262144):
                hi = min(lo + 262144, cols.size)
                out[lo:hi] = cp.sum(a[rows[lo:hi]] * bt[cols[lo:hi]], axis=1)
            return out

        output, elapsed = _benchmark_cuda_op(run, warmup, iters)
        error = _error(cp.asnumpy(output), reference)
        row["cupy_max_error"] = error
        if error > _TOL.get(compute, 1e-10):
            raise RuntimeError(f"CuPy correctness check failed: {error}")
        row["cupy_ms"] = elapsed
    except Exception as exc:
        row["cupy_reason"] = f"{type(exc).__name__}: {exc}"


_KINDS = {"spmv": _spmv, "spmm": _spmm, "sddmm": _sddmm}


def matrix_paths(inputs):
    paths = []
    for item in inputs or []:
        item = str(item)
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "**", "*.mtx"), recursive=True)))
        elif item.endswith(".mtx"):
            paths.append(item)
    return list(dict.fromkeys(paths))


def run_variants(op, inputs, warmup, iters, seed=2026, log=print):
    """Generic rows for every q4 variant of ``op`` on every .mtx in ``inputs``."""
    dev = accelerator_device()
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for path in matrix_paths(inputs):
        try:
            mat = _Matrix(path, dev)
        except Exception as exc:
            log(f"[q4] {os.path.basename(path)}: cannot load: {exc}")
            continue
        for v in VARIANTS[op]:
            row = {
                "variant": v.name,
                "matrix": os.path.basename(path),
                "value_dtype": _name(v.value),
                "index_dtype": "int32",
                "indptr_dtype": "int32",
                "op": v.op_a,
                "op_b": v.op_b,
                "layout": v.layout,
                "n_rows": mat.shape[0],
                "n_cols": mat.shape[1],
                "nnz": mat.csr.nnz,
            }
            try:
                out_dtype = _KINDS[v.kind](v, mat, gen, warmup, iters, row)
                row["status"] = "PASS" if _passes(out_dtype, row["max_error"]) else "FAIL"
                if row["status"] == "FAIL":
                    row["error"] = f"max error {row['max_error']:.3g} over tolerance"
            except Exception as exc:
                row["status"] = "ERROR"
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["speedup_vs_vendor"] = _ratio(row.get("vendor_ms"), row.get("ours_ms"))
            row["speedup_vs_pytorch"] = _ratio(row.get("pytorch_ms"), row.get("ours_ms"))
            rows.append(row)
            log(
                f"[q4] {row['matrix']} {v.name}: {row['status']} ours={row.get('ours_ms')} "
                f"vendor={row.get('vendor_ms')} torch={row.get('pytorch_ms')} {row.get('error') or ''}"
            )
    return rows


def append_to_csv(op, csv_path, rows):
    """Append ``rows`` to the script's CSV under its column names, widening the header."""
    mapping = COLUMN_MAPS[op]
    translated = []
    for row in rows:
        out = {mapping.get(k, k): v for k, v in row.items()}
        out.update(CONSTANTS.get(op, {}))
        translated.append(out)
    existing, fields = [], []
    if os.path.isfile(csv_path) and os.path.getsize(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            existing = list(reader)
    for row in translated:
        fields += [k for k in row if k not in fields]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
        for row in translated:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    return len(translated)


def run_and_append(op, inputs, csv_path, warmup, iters):
    """Entry point for the scripts: measure ``op``'s q4 variants, append to ``csv_path``."""
    if not csv_path:
        print("[q4] --q4-variants needs the script's CSV output option; skipped")
        return 0
    rows = run_variants(op, inputs, warmup, iters)
    count = append_to_csv(op, csv_path, rows)
    failed = sum(1 for r in rows if r.get("status") != "PASS")
    print(f"[q4] appended {count} {op} q4-variant row(s) to {csv_path}; {failed} not PASS")
    return count
