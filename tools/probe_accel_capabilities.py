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

"""Per-dtype capability probe for a FlagSparse accelerator backend.

Answers one question: for each dtype, which layer of the stack actually works on
this card -- torch allocation, torch dense math (the vendor DNN library), Triton
codegen, torch.sparse (the vendor baseline), or the FlagSparse operator itself.

Motivation: on Moore Threads (MUSA) every ``test_spmv_csr_accuracy`` case failed
inside the *test's own* dense reference (``muDNN NOT_SUPPORTED ... DOUBLE``,
``SetMUTensorDType Unsupported tensor dtype: ComplexFloat``), so the FlagSparse
kernel was never reached and its real dtype coverage stayed unknown.  This script
separates those layers instead of guessing which one a test failure belongs to.

Usage (on the card, from the repo root)::

    export PYTHONPATH=$PWD/src
    export FLAGSPARSE_BACKEND=mthreads          # or leave unset to auto-detect
    python tools/probe_accel_capabilities.py                  # isolated, default
    python tools/probe_accel_capabilities.py --no-isolate     # one process, faster
    python tools/probe_accel_capabilities.py --json out.json
    python tools/probe_accel_capabilities.py --dtypes float32,float64
    python tools/probe_accel_capabilities.py --groups torch,triton

Isolated mode (the default) runs every check in its own subprocess under a hard
timeout, so a wedged kernel or a fault that disables the vendor runtime costs one
row instead of every row after it.  The child's stderr is captured and attached,
because vendor libraries print their real diagnosis there (the ``muDNN ... ERROR#
NOT_SUPPORTED`` line) while the python exception only says ``MmCall failed``.
"""

import argparse
import json
import os
import subprocess
import sys
import traceback

# ---------------------------------------------------------------------------
# dtype table
# ---------------------------------------------------------------------------

DTYPE_NAMES = [
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]

DEFAULT_DTYPES = ["float32", "float64", "complex64", "complex128"]


def _dtype(name):
    import torch

    return getattr(torch, name, None)


def _is_complex(name):
    return name.startswith("complex")


def _component_name(name):
    """Real component dtype of a complex dtype -- what the Triton path really uses."""
    return {"complex64": "float32", "complex128": "float64"}.get(name, name)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _device():
    """Accelerator device for the active backend, via the repo's own abstraction."""
    import torch

    try:
        from flagsparse.sparse_operations._common import _ACCEL_DEVICE_TYPE

        return torch.device(str(_ACCEL_DEVICE_TYPE))
    except Exception:
        return torch.device("cuda")


def _sync():
    try:
        from flagsparse.sparse_operations._common import _ACCEL

        _ACCEL.synchronize()
    except Exception:
        pass


def _randn(shape, name, device):
    """Allocate on the device without going through ops the card may not support.

    Built on CPU and copied, deliberately: ``torch.complex(re, im)`` executed on the
    device is itself one of the things under test, so the allocation check must not
    depend on it.
    """
    import torch

    dt = _dtype(name)
    if _is_complex(name):
        comp = _dtype(_component_name(name))
        re = torch.randn(shape, dtype=comp)
        im = torch.randn(shape, dtype=comp)
        return torch.complex(re, im).to(device)
    return torch.randn(shape, dtype=torch.float32).to(dt).to(device)


# ---------------------------------------------------------------------------
# checks: (group, name, fn) -- fn(name, device) raises on failure
# ---------------------------------------------------------------------------

CHECKS = []


def check(group, name, note=""):
    def deco(fn):
        CHECKS.append((group, name, note, fn))
        return fn

    return deco


# --- group: alloc ----------------------------------------------------------


@check("alloc", "empty/zeros", "torch.zeros on device")
def _c_zeros(name, device):
    import torch

    t = torch.zeros(64, dtype=_dtype(name), device=device)
    _sync()
    assert t.numel() == 64


@check("alloc", "h2d copy", "CPU tensor .to(device)")
def _c_h2d(name, device):
    t = _randn(64, name, device)
    _sync()
    assert t.device.type == device.type


@check("alloc", "d2h copy", ".cpu() readback")
def _c_d2h(name, device):
    t = _randn(64, name, device).cpu()
    assert t.numel() == 64


@check("alloc", "torch.complex on device", "complex construction executed on card")
def _c_complex_build(name, device):
    import torch

    if not _is_complex(name):
        raise Skip("real dtype")
    comp = _dtype(_component_name(name))
    re = torch.randn(64, dtype=comp, device=device)
    im = torch.randn(64, dtype=comp, device=device)
    t = torch.complex(re, im)
    _sync()
    assert t.dtype == _dtype(name)


# --- group: torch (dense math -- the vendor DNN library) -------------------


@check("torch", "randn on device", "device RNG in this dtype")
def _c_randn_dev(name, device):
    import torch

    if _is_complex(name):
        raise Skip("randn has no complex form here; covered by torch.complex")
    t = torch.randn(64, dtype=_dtype(name), device=device)
    _sync()
    assert t.numel() == 64


@check("torch", "add/mul", "elementwise")
def _c_elementwise(name, device):
    a = _randn(64, name, device)
    b = _randn(64, name, device)
    c = a * b + a
    _sync()
    assert c.numel() == 64


@check("torch", "where", "ternary -- killed float64 in _random_csr_mn")
def _c_where(name, device):
    import torch

    a = _randn((16, 16), name, device)
    mask = torch.rand((16, 16), device=device) < 0.5
    out = torch.where(mask, a, torch.zeros((), dtype=_dtype(name), device=device))
    _sync()
    assert out.shape == (16, 16)


@check("torch", "sum", "reduction")
def _c_sum(name, device):
    a = _randn(256, name, device)
    s = a.sum()
    _sync()
    assert s.numel() == 1


@check("torch", "cast to ref dtype", "fp32->fp64 / c64->c128 upcast the oracle needs")
def _c_cast(name, device):
    # .get(name, name) on purpose: an unlisted dtype must not raise KeyError from
    # the probe itself and be read as a hardware limitation.
    ref = {
        "float16": "float32",
        "bfloat16": "float32",
        "float32": "float64",
        "complex64": "complex128",
    }.get(name, name)
    a = _randn(64, name, device).to(_dtype(ref))
    _sync()
    assert a.dtype == _dtype(ref)


@check("torch", "matvec (@)", "muDNN MatMul -- the 135-line oracle")
def _c_matvec(name, device):
    a = _randn((32, 24), name, device)
    x = _randn(24, name, device)
    y = a @ x
    _sync()
    assert y.shape == (32,)


@check("torch", "matmul (@)", "dense x dense")
def _c_matmul(name, device):
    a = _randn((32, 24), name, device)
    b = _randn((24, 16), name, device)
    c = a @ b
    _sync()
    assert c.shape == (32, 16)


@check("torch", "oracle upcast matvec", "exactly what the test does: (A.to(ref) @ x.to(ref))")
def _c_oracle(name, device):
    ref = {"float32": "float64", "complex64": "complex128"}.get(name, name)
    a = _randn((32, 24), name, device)
    x = _randn(24, name, device)
    y = (a.to(_dtype(ref)) @ x.to(_dtype(ref))).to(_dtype(name))
    _sync()
    assert y.shape == (32,)


@check("torch", "matvec as 2D (@)", "same math with x reshaped (N,1) -- gemv vs gemm dispatch")
def _c_matvec_2d(name, device):
    a = _randn((32, 24), name, device)
    x = _randn((24, 1), name, device)
    y = a @ x
    _sync()
    assert y.shape == (32, 1)


@check("torch", "oracle upcast as 2D", "the oracle, but through the gemm path")
def _c_oracle_2d(name, device):
    ref = {"float32": "float64", "complex64": "complex128"}.get(name, name)
    a = _randn((32, 24), name, device)
    x = _randn((24, 1), name, device)
    y = (a.to(_dtype(ref)) @ x.to(_dtype(ref))).to(_dtype(name))
    _sync()
    assert y.shape == (32, 1)


@check("torch", "nonzero/bincount/cumsum", "CSR construction in _random_csr_mn")
def _c_csr_build(name, device):
    import torch

    mask = torch.rand((16, 16), device=device) < 0.5
    rows, cols = torch.nonzero(mask, as_tuple=True)
    counts = torch.bincount(rows, minlength=16)
    indptr = torch.zeros(17, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(counts, dim=0)
    _sync()
    assert int(indptr[-1].item()) == int(rows.numel())


@check("torch", "advanced indexing", "values[order] -- the IndexMusa gap _gather_values works around")
def _c_index(name, device):
    import torch

    a = _randn(256, name, device)
    order = torch.randperm(256, device=device)
    out = a[order]
    _sync()
    assert out.numel() == 256


@check("torch", "index_select", "the other form of the same gather")
def _c_index_select(name, device):
    import torch

    a = _randn(256, name, device)
    idx = torch.randperm(256, device=device)
    out = a.index_select(0, idx)
    _sync()
    assert out.numel() == 256


@check("torch", "index_copy_", "the scatter direction of the same op")
def _c_index_copy(name, device):
    import torch

    a = _randn(256, name, device)
    vals = _randn(64, name, device)
    idx = torch.randperm(256, device=device)[:64]
    a.index_copy_(0, idx, vals)
    _sync()


@check("torch", "2-D gather rows", "dense[rows, cols] -- how CSR data is extracted")
def _c_index_2d(name, device):
    import torch

    a = _randn((32, 24), name, device)
    rows = torch.randint(0, 32, (50,), device=device)
    cols = torch.randint(0, 24, (50,), device=device)
    out = a[rows, cols]
    _sync()
    assert out.numel() == 50


@check("torch", "view_as_real/complex", "how the Triton complex path splits components")
def _c_view_as_real(name, device):
    import torch

    if not _is_complex(name):
        raise Skip("real dtype")
    a = _randn(64, name, device)
    ri = torch.view_as_real(a).reshape(-1)
    back = torch.view_as_complex(ri.reshape(64, 2))
    _sync()
    assert back.dtype == a.dtype


@check("torch", "allclose", "the assertion itself")
def _c_allclose(name, device):
    import torch

    a = _randn(64, name, device)
    ok = torch.allclose(a, a.clone())
    _sync()
    assert ok


# --- group: triton ---------------------------------------------------------

_TRITON_SRC = '''
import triton
import triton.language as tl


@triton.jit
def k_copy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m, other=0.0), mask=m)


@triton.jit
def k_where_sum(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m, other=0.0)
    v = tl.where(m, v * v, tl.zeros([BLOCK], dtype=v.dtype))
    tl.store(y_ptr + tl.program_id(0), tl.sum(v, axis=0))


@triton.jit
def k_atomic(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.atomic_add(y_ptr + 0, tl.sum(v, axis=0))


@triton.jit
def k_cas_loop(ptr, out_ptr, CAP: tl.constexpr, BLOCK: tl.constexpr):
    """Bounded atomic_cas claim loop -- the shape of spgemm_csr's hash fill."""
    offs = tl.arange(0, BLOCK)
    done = offs < 0
    it = 0
    while (tl.sum((~done).to(tl.int32)) > 0) & (it < CAP):
        # cmp/val must be TENSORS of the pointer's dtype: scalar Python ints give
        # "tt.atomic_cas op failed to verify that cmp type matches ptr type".
        expect = tl.zeros([BLOCK], dtype=tl.int32)
        claim = tl.full([BLOCK], 1, tl.int32)
        cur = tl.atomic_cas(ptr + offs, expect, claim)
        done = done | (cur == 0)
        it += 1
    tl.store(out_ptr + offs, done.to(tl.int32))


@triton.jit
def k_while_reduce(out_ptr, n, BLOCK: tl.constexpr):
    """while-loop whose condition is a CROSS-LANE reduction, with no iteration cap.

    spgemm_csr's per-row binary search is exactly this shape and is the only
    uncapped loop in the wedging path.  Terminates in log2(n) steps on healthy
    hardware because hi-lo>1 guarantees lo < mid < hi.
    """
    offs = tl.arange(0, BLOCK)
    lo = tl.zeros([BLOCK], dtype=tl.int32)
    hi = tl.full([BLOCK], n, tl.int32)
    while tl.sum((hi - lo > 1).to(tl.int32)) > 0:
        mid = (lo + hi) // 2
        take = mid <= offs
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid)
    tl.store(out_ptr + offs, lo)


@triton.jit
def _seg_add(row_a, val_a, row_b, val_b):
    same = row_a == row_b
    return row_b, tl.where(same, val_a + val_b, val_b)


@triton.jit
def k_assoc_scan(row_ptr, val_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    row = tl.load(row_ptr + offs, mask=m, other=0)
    val = tl.load(val_ptr + offs, mask=m, other=0.0)
    _, acc = tl.associative_scan((row, val), axis=0, combine_fn=_seg_add)
    tl.store(out_ptr + offs, acc, mask=m)
'''

_triton_mod = []


def _triton_kernels():
    """Materialise the probe kernels as a real module and import it.

    @triton.jit reads the decorated function's source file, so exec()'d kernels
    raise "@jit functions should be defined in a Python file".  Writing the source
    to a temp file keeps this probe a SINGLE file to copy onto a card -- an earlier
    two-file version reported the whole triton group as FAIL on MUSA purely because
    the sibling module had not been copied across, which reads exactly like a
    hardware verdict.
    """
    if _triton_mod:
        return _triton_mod[0]
    import importlib.util
    import tempfile

    d = tempfile.mkdtemp(prefix="flagsparse_probe_")
    path = os.path.join(d, "_probe_triton_kernels.py")
    with open(path, "w") as fh:
        fh.write(_TRITON_SRC)
    spec = importlib.util.spec_from_file_location("_probe_triton_kernels", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_probe_triton_kernels"] = mod
    spec.loader.exec_module(mod)
    _triton_mod.append(mod)
    return mod


def _triton_dtype_name(name):
    """Triton operates on the real component dtype for complex, like spmv_csr does."""
    return _component_name(name) if _is_complex(name) else name


@check("triton", "load/store", "basic codegen for this dtype")
def _c_tl_copy(name, device):
    import torch

    k = _triton_kernels()
    dn = _triton_dtype_name(name)
    x = _randn(256, dn, device)
    y = torch.zeros(256, dtype=_dtype(dn), device=device)
    k.k_copy[(1,)](x, y, 256, BLOCK=256)
    _sync()
    assert torch.allclose(x.cpu(), y.cpu())


@check("triton", "where+sum", "tl.where / tl.sum reduction")
def _c_tl_where_sum(name, device):
    import torch

    k = _triton_kernels()
    dn = _triton_dtype_name(name)
    x = _randn(256, dn, device)
    y = torch.zeros(1, dtype=_dtype(dn), device=device)
    k.k_where_sum[(1,)](x, y, 256, BLOCK=256)
    _sync()
    got = float(y.cpu()[0])
    want = float((x.cpu().double() ** 2).sum())
    assert abs(got - want) <= 1e-2 * max(1.0, abs(want)), f"{got} vs {want}"


@check("triton", "atomic_add", "tl.atomic_add in this dtype")
def _c_tl_atomic(name, device):
    import torch

    k = _triton_kernels()
    dn = _triton_dtype_name(name)
    x = _randn(256, dn, device)
    y = torch.zeros(1, dtype=_dtype(dn), device=device)
    k.k_atomic[(1,)](x, y, 256, BLOCK=256)
    _sync()
    assert y.cpu().isfinite().all()


@check("triton", "atomic_cas loop", "spgemm_csr hash fill: bounded CAS claim loop")
def _c_tl_cas(name, device):
    import torch

    k = _triton_kernels()
    slots = torch.zeros(256, dtype=torch.int32, device=device)
    out = torch.zeros(256, dtype=torch.int32, device=device)
    k.k_cas_loop[(1,)](slots, out, CAP=16, BLOCK=256)
    _sync()
    assert int(out.sum().item()) > 0, "no lane ever won its slot"


@check("triton", "while + cross-lane reduce", "spgemm_csr binary search: UNCAPPED while")
def _c_tl_while(name, device):
    import torch

    k = _triton_kernels()
    out = torch.zeros(256, dtype=torch.int32, device=device)
    k.k_while_reduce[(1,)](out, 256, BLOCK=256)
    _sync()
    assert out.cpu().max().item() >= 0


@check("triton", "associative_scan", "segbin core -- Ascend could not lower this")
def _c_tl_scan(name, device):
    import torch

    k = _triton_kernels()
    dn = _triton_dtype_name(name)
    rows = torch.arange(256, device=device, dtype=torch.int32) // 8
    val = _randn(256, dn, device)
    out = torch.zeros(256, dtype=_dtype(dn), device=device)
    k.k_assoc_scan[(1,)](rows, val, out, 256, BLOCK=256)
    _sync()
    assert out.cpu().isfinite().all()


# --- group: torch_sparse (the MUSA vendor baseline) ------------------------


@check("torch_sparse", "sparse_csr_tensor", "construction")
def _c_sparse_build(name, device):
    import torch

    crow = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    col = torch.tensor([0, 2, 1], dtype=torch.int32, device=device)
    val = _randn(3, name, device)
    a = torch.sparse_csr_tensor(crow, col, val, size=(2, 3))
    _sync()
    assert a.shape == (2, 3)


@check("torch_sparse", "sparse mv", "torch.sparse baseline used on mthreads")
def _c_sparse_mv(name, device):
    import torch

    crow = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    col = torch.tensor([0, 2, 1], dtype=torch.int32, device=device)
    val = _randn(3, name, device)
    a = torch.sparse_csr_tensor(crow, col, val, size=(2, 3))
    x = _randn((3, 1), name, device)
    y = a @ x
    _sync()
    assert y.shape == (2, 1)


@check("torch_sparse", "sparse mm (2D rhs)", "does a 2D rhs dodge whatever mv hits")
def _c_sparse_mm(name, device):
    import torch

    crow = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    col = torch.tensor([0, 2, 1], dtype=torch.int32, device=device)
    val = _randn(3, name, device)
    a = torch.sparse_csr_tensor(crow, col, val, size=(2, 3))
    b = _randn((3, 4), name, device)
    y = torch.sparse.mm(a, b)
    _sync()
    assert y.shape == (2, 4)


@check("torch_sparse", "sparse COO mm", "the COO layout as a fallback baseline")
def _c_sparse_coo_mm(name, device):
    import torch

    idx = torch.tensor([[0, 0, 1], [0, 2, 1]], dtype=torch.int64, device=device)
    val = _randn(3, name, device)
    a = torch.sparse_coo_tensor(idx, val, (2, 3)).coalesce()
    b = _randn((3, 4), name, device)
    y = torch.sparse.mm(a, b)
    _sync()
    assert y.shape == (2, 4)


# --- group: routing (which code path a passing check actually took) --------


@check("routing", "spmv_csr kernel choice", "segbin vs rowpar -- what the OK row ran")
def _c_route_kernel(name, device):
    import flagsparse.sparse_operations.spmv_csr as M

    fn = getattr(M, "_spmv_csr_default_backend", None)
    if fn is None:
        raise Skip("no _spmv_csr_default_backend in this build")
    raise Skip(f"kernel={fn()}")


@check("routing", "spmv_csr uses triton", "assert the Triton impl really ran for this dtype")
def _c_route_triton(name, device):
    import torch

    import flagsparse.sparse_operations.spmv_csr as M
    from flagsparse import flagsparse_spmv_csr

    original = M._triton_spmv_csr_impl_prepared
    seen = {"hit": False}

    def spy(prepared, x_in):
        seen["hit"] = True
        return original(prepared, x_in)

    M._triton_spmv_csr_impl_prepared = spy
    try:
        M_, N = 32, 24
        crow = torch.arange(0, M_ + 1, dtype=torch.int32, device=device) * 2
        col = (torch.arange(M_ * 2, dtype=torch.int32, device=device) % N).contiguous()
        flagsparse_spmv_csr(
            _randn(M_ * 2, name, device),
            col,
            crow,
            _randn(N, name, device),
            shape=(M_, N),
            op="non",
        )
        _sync()
    finally:
        M._triton_spmv_csr_impl_prepared = original
    assert seen["hit"], "operator completed WITHOUT the Triton impl (silent fallback)"


# --- group: flagsparse (end to end) ---------------------------------------


@check("flagsparse", "spmv_csr non", "the operator under test, no dense oracle")
def _c_fs_spmv(name, device):
    import torch

    from flagsparse import flagsparse_spmv_csr

    M, N = 32, 24
    crow = torch.arange(0, M + 1, dtype=torch.int32, device=device) * 2
    col = (torch.arange(M * 2, dtype=torch.int32, device=device) % N).contiguous()
    val = _randn(M * 2, name, device)
    x = _randn(N, name, device)
    y = flagsparse_spmv_csr(val, col, crow, x, shape=(M, N), op="non")
    _sync()
    assert y.shape == (M,)
    assert y.cpu().isfinite().all()


@check("flagsparse", "spmv_csr vs CPU oracle", "correctness with the reference on CPU")
def _c_fs_spmv_ref(name, device):
    import torch

    from flagsparse import flagsparse_spmv_csr

    M, N = 32, 24
    crow = torch.arange(0, M + 1, dtype=torch.int32) * 2
    col = (torch.arange(M * 2, dtype=torch.int32) % N).contiguous()
    val_c = _randn(M * 2, name, torch.device("cpu"))
    x_c = _randn(N, name, torch.device("cpu"))
    dense = torch.zeros((M, N), dtype=_dtype(name))
    for r in range(M):
        for j in range(2 * r, 2 * r + 2):
            dense[r, int(col[j])] += val_c[j]
    ref_dt = {"float32": torch.float64, "complex64": torch.complex128}.get(
        name, _dtype(name)
    )
    ref = (dense.to(ref_dt) @ x_c.to(ref_dt)).to(_dtype(name))
    y = flagsparse_spmv_csr(
        val_c.to(device),
        col.to(device),
        crow.to(device),
        x_c.to(device),
        shape=(M, N),
        op="non",
    )
    _sync()
    tol = 1e-3 if name in ("float32", "complex64", "float16", "bfloat16") else 1e-7
    assert torch.allclose(y.cpu().to(ref_dt), ref.to(ref_dt), rtol=tol, atol=tol)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


class Skip(Exception):
    """Check does not apply to this dtype."""


def run_one(group, cname, dtype_name):
    """Run a single check; return a result dict. Never raises."""
    for g, n, note, fn in CHECKS:
        if g == group and n == cname:
            break
    else:
        return {"status": "ERROR", "detail": f"unknown check {group}/{cname}"}

    if _dtype(dtype_name) is None:
        return {"status": "SKIP", "detail": f"torch has no {dtype_name}"}

    device = _device()
    try:
        fn(dtype_name, device)
    except Skip as exc:
        return {"status": "SKIP", "detail": str(exc)}
    except BaseException as exc:  # noqa: BLE001 - a probe reports, never propagates
        return {
            "status": "FAIL",
            "detail": f"{type(exc).__name__}: {exc}".strip().replace("\n", " | ")[:400],
            "traceback": traceback.format_exc()[-1500:],
        }
    return {"status": "OK", "detail": ""}


def run_isolated(group, cname, dtype_name, timeout):
    """Run one check in a fresh process, capturing the vendor's stderr."""
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--child",
        "--group-one",
        group,
        "--check-one",
        cname,
        "--dtype-one",
        dtype_name,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "HANG",
            "detail": f"no result within {timeout}s (killed)",
            "stderr": "",
        }
    out = (proc.stdout or "").strip().splitlines()
    payload = None
    for line in reversed(out):
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except ValueError:
                continue
    stderr = (proc.stderr or "").strip()
    if payload is None:
        return {
            "status": "CRASH",
            "detail": f"child exited {proc.returncode} with no result",
            "stderr": stderr[-1500:],
        }
    payload["stderr"] = stderr[-1500:]
    return payload


_STATUS_MARK = {
    "OK": "ok",
    "FAIL": "FAIL",
    "SKIP": "-",
    "HANG": "HANG",
    "CRASH": "CRASH",
    "ERROR": "ERR",
}


def backend_banner():
    lines = []
    try:
        import torch

        lines.append(f"torch             : {torch.__version__}")
    except Exception as exc:
        lines.append(f"torch             : import failed: {exc}")
        return lines
    try:
        import flagsparse.sparse_operations._common as C

        lines.append(f"backend           : {C._backend_name()}")
        lines.append(f"accel device type : {C._accel_device_type()}")
        lines.append(f"vendor sparse lib : {C._vendor_sparse_library()}")
        reason = C._accel_fallback_reason()
        lines.append(f"fallback reason   : {reason}")
        if reason:
            lines.append(
                "  !! fallback is active -- the numbers below are NOT this backend's"
            )
    except Exception as exc:
        lines.append(f"flagsparse probe  : failed: {type(exc).__name__}: {exc}")
    try:
        import triton

        lines.append(f"triton            : {triton.__version__}")
    except Exception as exc:
        lines.append(f"triton            : unavailable: {exc}")
    try:
        from flagsparse.sparse_operations._common import _ACCEL

        lines.append(f"device count      : {_ACCEL.device_count()}")
        try:
            lines.append(f"device name       : {_ACCEL.get_device_name(0)}")
        except Exception:
            pass
    except Exception:
        pass
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--dtypes",
        default=",".join(DEFAULT_DTYPES),
        help=f"comma-separated; any of {','.join(DTYPE_NAMES)}, or 'all'",
    )
    ap.add_argument("--groups", default="", help="comma-separated group filter")
    ap.add_argument(
        "--no-isolate",
        action="store_true",
        help="run everything in this process (faster, but one wedge loses the rest)",
    )
    ap.add_argument("--timeout", type=float, default=180.0, help="per-check seconds")
    ap.add_argument("--json", default="", help="write full results here")
    # child-mode plumbing
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--group-one", default="", help=argparse.SUPPRESS)
    ap.add_argument("--check-one", default="", help=argparse.SUPPRESS)
    ap.add_argument("--dtype-one", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.child:
        res = run_one(args.group_one, args.check_one, args.dtype_one)
        res.pop("traceback", None) if res.get("status") == "OK" else None
        print(json.dumps(res))
        return 0

    dtypes = DTYPE_NAMES if args.dtypes == "all" else args.dtypes.split(",")
    dtypes = [d.strip() for d in dtypes if d.strip()]
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    for line in backend_banner():
        print(line)
    print()
    mode = "one process" if args.no_isolate else f"isolated, {args.timeout:.0f}s timeout"
    print(f"mode              : {mode}")
    print()

    selected = [c for c in CHECKS if not groups or c[0] in groups]
    width = max(len(f"{g}/{n}") for g, n, _, _ in selected) + 2
    header = "check".ljust(width) + "".join(d.rjust(13) for d in dtypes)
    print(header)
    print("-" * len(header))

    results = {}
    last_group = None
    for group, cname, note, _fn in selected:
        if last_group is not None and group != last_group:
            print()
        last_group = group
        row = f"{group}/{cname}".ljust(width)
        for dt in dtypes:
            if args.no_isolate:
                res = run_one(group, cname, dt)
            else:
                res = run_isolated(group, cname, dt, args.timeout)
            results[f"{group}/{cname}/{dt}"] = res
            row += _STATUS_MARK.get(res["status"], res["status"]).rjust(13)
        print(row)
        if note:
            print(" " * 2 + f"# {note}")

    print()
    print("=" * 78)
    print("failures in detail (vendor stderr included -- read it literally)")
    print("=" * 78)
    any_fail = False
    for key, res in results.items():
        if res["status"] in ("OK", "SKIP"):
            continue
        any_fail = True
        print(f"\n--- {key}  [{res['status']}]")
        if res.get("detail"):
            print(f"    {res['detail']}")
        err = (res.get("stderr") or "").strip()
        if err:
            for line in err.splitlines()[-12:]:
                print(f"    | {line}")
    if not any_fail:
        print("\n(none)")

    # Some checks report a fact rather than pass/fail (which kernel got selected,
    # why a dtype does not apply). Those land as SKIP and would otherwise be
    # invisible in the table, which is where a routing surprise hides.
    notes = {}
    for key, res in results.items():
        if res["status"] != "SKIP" or not res.get("detail"):
            continue
        check_name = key.rsplit("/", 1)[0]
        notes.setdefault((check_name, res["detail"]), []).append(key.rsplit("/", 1)[1])
    if notes:
        print()
        print("=" * 78)
        print("notes (skipped checks that still carry information)")
        print("=" * 78)
        for (check_name, detail), dts in notes.items():
            print(f"  {check_name} [{','.join(dts)}]: {detail}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"dtypes": dtypes, "results": results}, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
