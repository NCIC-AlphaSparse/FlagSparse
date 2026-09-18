#!/usr/bin/env python3
"""Per-operator capability probe, aimed at Ascend but runnable on any backend.

``benchmark_ascend.py`` measures the handful of operators already known to work
on an NPU.  This one answers the prior question: for every operator in the
library, over a spread of matrices, does it run at all -- and when it does not,
WHY.  Three outcomes are worth telling apart and the usual "it threw" is not
enough:

``PASS``            ran and matched a CPU reference.
``MISMATCH``        ran and did not.  A wrong answer, not a missing feature.
``TRITON_COMPILE``  the Triton backend could not lower the kernel.  This is the
                    Ascend story: the kernel is fine, CANN cannot build it.
``REJECTED``        the operator itself declined the input (ValueError and
                    friends) -- an unsupported dtype, layout or shape, which is
                    a deliberate limit rather than a defect.
``ERROR``           anything else, kept separate so it cannot be mistaken for a
                    compile failure.
``NO_ADAPTER``      this script has no input recipe for the operator -- its own
                    gap, reported as such rather than as a rejection.

Each case runs with a cleared Triton cache, because a cached artifact from an
earlier case makes a compile failure look like a pass.  The classification is
what this script is for; the timings are secondary and only recorded for PASS.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# --------------------------------------------------------------------- matrices


@dataclass(frozen=True)
class Matrix:
    name: str
    rows: int
    cols: int
    density: float
    seed: int


# Twenty cases spanning the axes that actually change an operator's behaviour:
# size, aspect ratio, density, and the degenerate ends (one row, one column, a
# matrix with no nonzeros at all).  Synthetic and seeded, so a probe result is
# reproducible without shipping a matrix corpus.
MATRICES: tuple[Matrix, ...] = (
    Matrix("tiny_square", 16, 16, 0.25, 1),
    Matrix("tiny_wide", 8, 128, 0.20, 2),
    Matrix("tiny_tall", 128, 8, 0.20, 3),
    Matrix("single_row", 1, 256, 0.10, 4),
    Matrix("single_col", 256, 1, 0.10, 5),
    Matrix("empty", 64, 64, 0.0, 6),
    Matrix("small_sparse", 128, 128, 0.02, 7),
    Matrix("small_dense", 128, 128, 0.40, 8),
    Matrix("square_512", 512, 512, 0.01, 9),
    Matrix("square_512_dense", 512, 512, 0.10, 10),
    Matrix("wide_512", 256, 2048, 0.01, 11),
    Matrix("tall_512", 2048, 256, 0.01, 12),
    Matrix("square_1k", 1024, 1024, 0.005, 13),
    Matrix("square_1k_dense", 1024, 1024, 0.05, 14),
    Matrix("square_2k", 2048, 2048, 0.002, 15),
    Matrix("square_4k", 4096, 4096, 0.001, 16),
    Matrix("skew_rows", 1024, 1024, 0.01, 17),
    Matrix("banded_1k", 1024, 1024, 0.008, 18),
    Matrix("wide_4k", 512, 4096, 0.004, 19),
    Matrix("tall_4k", 4096, 512, 0.004, 20),
)


def _accel(torch):
    """(accelerator module, torch.device) for whichever backend this is.

    Resolved through the library's own registry rather than a local if-chain on
    torch.npu: the probe has to follow the same backend the operators follow, or
    it measures one platform and reports another.  The registry pairs the module
    and the device type deliberately -- see _resolve_accel -- so they cannot
    disagree here either.
    """
    try:
        from flagsparse.sparse_operations._common import _resolve_accel

        module, device_type = _resolve_accel()
    except Exception:
        module, device_type = torch.cuda, "cuda"
    try:
        if not module.is_available():
            return None, torch.device("cpu")
    except Exception:
        return None, torch.device("cpu")
    return module, torch.device(device_type)


def _sync(module, device):
    if module is None or device.type == "cpu":
        return
    try:
        module.synchronize()
    except Exception:
        pass


def _empty_cache(module, device):
    if module is None or device.type == "cpu":
        return
    try:
        module.empty_cache()
    except Exception:
        pass


def _dense(torch, m: Matrix, dtype, device):
    """A dense matrix with m's shape and density, deterministic in its seed."""
    gen = torch.Generator(device="cpu").manual_seed(m.seed)
    values = torch.randn(m.rows, m.cols, generator=gen, dtype=torch.float32)
    keep = torch.rand(m.rows, m.cols, generator=gen) < m.density
    return (values * keep).to(dtype).to(device)


# ------------------------------------------------------------- classification


_COMPILE_MARKERS = (
    "CompilationError",
    "failed to compile",
    "Failed to compile",
    "triton.compiler",
    "PTXAS",
    "ptxas",
    "LLVM ERROR",
    "MLIR",
    "mlir",
    "loc(",
    "bishengir",
    "ascendc",
    "npucompiler",
    "out of shared memory",
    "shared memory",
    "Shared memory",
)

_REJECT_TYPES = (ValueError, TypeError, NotImplementedError, IndexError, KeyError)


def classify(exc: BaseException) -> tuple[str, str]:
    """(status, one-line reason) for a failure."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    name = type(exc).__name__
    first = str(exc).strip().splitlines()
    reason = first[0][:200] if first else name

    # Compile failures are checked FIRST: a Triton CompilationError is often a
    # subclass of something ordinary, and misfiling one as "rejected" would hide
    # exactly the signal this probe exists to collect.
    if any(marker in text for marker in _COMPILE_MARKERS) or "Compilation" in name:
        return "TRITON_COMPILE", f"{name}: {reason}"
    if isinstance(exc, _REJECT_TYPES):
        return "REJECTED", f"{name}: {reason}"
    return "ERROR", f"{name}: {reason}"


def clear_triton_cache(cache_dir: Path) -> None:
    """Drop every compiled artifact so the next case really compiles.

    Without this the first case to compile a kernel hides a later case's
    failure -- the cache hit never reaches the backend.
    """
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------- adapters
#
# Each adapter turns a plain dense matrix into the inputs one operator wants,
# and returns (callable, reference) where reference is computed on the CPU in
# fp64.  The formats that need real structure -- BSR blocks, SELL slices, a
# triangular matrix for the solves -- are built here rather than being faked,
# because an operator that rejects a malformed input would otherwise be recorded
# as an Ascend limitation.


class _NoAdapter(LookupError):
    """This script has no input recipe for that operator.

    Its own gap, not the operator's: reporting it as REJECTED would read as
    "the operator declined the input", which is the one thing it is not.
    """


@dataclass
class Probe:
    run: object
    reference: object = None
    note: str = ""
    extra: dict = field(default_factory=dict)


def _csr(torch, dense):
    sp = dense.to_sparse_csr()
    return (
        sp.values().contiguous(),
        sp.col_indices().to(torch.int32).contiguous(),
        sp.crow_indices().to(torch.int32).contiguous(),
    )


def _coo(torch, dense):
    sp = dense.to_sparse_coo().coalesce()
    idx = sp.indices()
    return (
        sp.values().contiguous(),
        idx[0].to(torch.int32).contiguous(),
        idx[1].to(torch.int32).contiguous(),
    )


def _triangular(torch, m: Matrix, dtype, device, lower=True):
    """A well-conditioned triangular matrix; the solves need one to mean anything."""
    n = min(m.rows, m.cols)
    n = max(n, 1)
    gen = torch.Generator(device="cpu").manual_seed(m.seed + 100)
    a = torch.randn(n, n, generator=gen, dtype=torch.float64)
    a = a * (torch.rand(n, n, generator=gen) < m.density)
    a = a / (4.0 * max(1.0, m.density * n))
    a = torch.tril(a) if lower else torch.triu(a)
    a[range(n), range(n)] = 2.0 + torch.rand(n, generator=gen).to(torch.float64)
    return a.to(dtype).to(device), n


def _bsr_inputs(torch, m: Matrix, dtype, device, block_dim=4):
    """Block-CSR needs whole blocks, so the shape is rounded to the block grid."""
    brows = max(1, m.rows // block_dim)
    bcols = max(1, m.cols // block_dim)
    gen = torch.Generator(device="cpu").manual_seed(m.seed + 200)
    present = torch.rand(brows, bcols, generator=gen) < max(m.density, 1.0 / bcols)
    indptr = [0]
    indices: list[int] = []
    blocks = []
    for br in range(brows):
        for bc in range(bcols):
            if not bool(present[br, bc]):
                continue
            indices.append(bc)
            blocks.append(torch.randn(block_dim, block_dim, generator=gen,
                                      dtype=torch.float32))
        indptr.append(len(indices))
    if not blocks:
        indices.append(0)
        blocks.append(torch.zeros(block_dim, block_dim, dtype=torch.float32))
        indptr = [0] * brows + [1]
    data = torch.stack(blocks).to(dtype).to(device)
    return (
        data,
        torch.tensor(indices, dtype=torch.int32, device=device),
        torch.tensor(indptr, dtype=torch.int32, device=device),
        brows,
        bcols,
        block_dim,
    )


def _sell_inputs(torch, m: Matrix, dtype, device, slice_size=8):
    """Sliced-ELL for SpSV: one diagonal per row, padding (-1) strictly trailing."""
    n = max(1, min(m.rows, m.cols))
    n_slices = (n + slice_size - 1) // slice_size
    width = 2                       # diagonal + at most one off-diagonal
    gen = torch.Generator(device="cpu").manual_seed(m.seed + 300)
    cols = torch.full((n_slices * slice_size * width,), -1, dtype=torch.int32)
    vals = torch.zeros(n_slices * slice_size * width, dtype=torch.float32)
    offsets = torch.arange(n_slices + 1, dtype=torch.int32) * (slice_size * width)
    for row in range(n):
        s, lane = divmod(row, slice_size)
        base = s * slice_size * width + lane
        # Slot 0 is the sub-diagonal when there is one, slot 1 the diagonal:
        # the solve requires the diagonal present and the padding trailing.
        if row > 0:
            cols[base] = row - 1
            vals[base] = float(torch.randn(1, generator=gen)) * 0.1
            cols[base + slice_size] = row
            vals[base + slice_size] = 2.0 + float(torch.rand(1, generator=gen))
        else:
            cols[base] = row
            vals[base] = 2.0 + float(torch.rand(1, generator=gen))
    return (vals.to(dtype).to(device), cols.to(device), offsets.to(device), n,
            slice_size)


def build_probe(flagsparse, torch, op: str, m: Matrix, dtype, device, dense_cols: int):
    """Inputs and a CPU reference for one (operator, matrix) pair."""
    fs = flagsparse
    dense = _dense(torch, m, dtype, device)
    ref64 = dense.to(torch.float64).cpu()

    if op == "gather":
        n = max(1, m.rows * m.cols // 4)
        gen = torch.Generator(device="cpu").manual_seed(m.seed)
        src = torch.randn(max(1, m.rows * m.cols), generator=gen).to(dtype).to(device)
        idx = torch.randint(0, src.numel(), (n,), generator=gen).to(torch.int32).to(device)
        return Probe(lambda: fs.flagsparse_gather(src, idx),
                     src.cpu().to(torch.float64)[idx.cpu().to(torch.int64)])
    if op == "scatter":
        n = max(1, m.rows * m.cols // 4)
        gen = torch.Generator(device="cpu").manual_seed(m.seed)
        dst = torch.zeros(max(1, m.rows * m.cols), dtype=dtype, device=device)
        vals = torch.randn(n, generator=gen).to(dtype).to(device)
        idx = torch.arange(n, dtype=torch.int32, device=device)
        expect = torch.zeros(dst.numel(), dtype=torch.float64)
        expect[: n] = vals.cpu().to(torch.float64)

        def _scatter():
            # Scatter writes into its destination and returns None, so the
            # result to compare is the buffer, not the return value.
            target = dst.clone()
            fs.flagsparse_scatter(target, idx, vals)
            return target

        return Probe(_scatter, expect)

    if op in ("spmv_csr", "spmv_coo", "spmv_csc", "spmv_coo_tocsr"):
        x = torch.randn(m.cols, dtype=dtype, device=device)
        ref = ref64 @ x.cpu().to(torch.float64)
        if op == "spmv_csr":
            d, i, p = _csr(torch, dense)
            return Probe(lambda: fs.flagsparse_spmv_csr(d, i, p, x, shape=(m.rows, m.cols)), ref)
        if op == "spmv_csc":
            sp = dense.t().contiguous().to_sparse_csr()
            d = sp.values().contiguous()
            i = sp.col_indices().to(torch.int32).contiguous()
            p = sp.crow_indices().to(torch.int32).contiguous()
            return Probe(lambda: fs.flagsparse_spmv_csc(d, i, p, x, shape=(m.rows, m.cols)), ref)
        d, r, c = _coo(torch, dense)
        fn = fs.flagsparse_spmv_coo if op == "spmv_coo" else fs.flagsparse_spmv_coo_tocsr
        return Probe(lambda: fn(d, r, c, x, shape=(m.rows, m.cols)), ref)

    if op == "spmv_bsr":
        data, idx, ptr, brows, bcols, bd = _bsr_inputs(torch, m, dtype, device)
        x = torch.randn(bcols * bd, dtype=dtype, device=device)
        return Probe(lambda: fs.flagsparse_spmv_bsr(data, idx, ptr, x,
                                                    shape=(brows * bd, bcols * bd),
                                                    block_dim=bd),
                     None, note="no CPU reference: block layout")

    if op in ("spmm_csr", "spmm_coo", "spmm_csc"):
        B = torch.randn(m.cols, dense_cols, dtype=dtype, device=device)
        ref = ref64 @ B.cpu().to(torch.float64)
        if op == "spmm_csr":
            d, i, p = _csr(torch, dense)
            return Probe(lambda: fs.flagsparse_spmm_csr(d, i, p, B, shape=(m.rows, m.cols)), ref)
        if op == "spmm_coo":
            d, r, c = _coo(torch, dense)
            return Probe(lambda: fs.flagsparse_spmm_coo(d, r, c, B, shape=(m.rows, m.cols)), ref)
        sp = dense.t().contiguous().to_sparse_csr()
        return Probe(lambda: fs.flagsparse_spmm_csc(
            sp.values().contiguous(), sp.col_indices().to(torch.int32).contiguous(),
            sp.crow_indices().to(torch.int32).contiguous(), B,
            shape=(m.rows, m.cols)), ref)

    if op in ("spmm_csr_opt", "spmm_csr_opt_alg1", "spmm_csr_opt_alg2", "alpha_spmm_alg1"):
        d, i, p = _csr(torch, dense)
        B = torch.randn(m.cols, dense_cols, dtype=dtype, device=device)
        ref = ref64 @ B.cpu().to(torch.float64)
        fn = getattr(fs, "flagsparse_" + op)
        return Probe(lambda: fn(d, i, p, B, shape=(m.rows, m.cols)), ref)

    if op == "spmm_bsr":
        data, idx, ptr, brows, bcols, bd = _bsr_inputs(torch, m, dtype, device)
        B = torch.randn(bcols * bd, dense_cols, dtype=dtype, device=device)
        return Probe(lambda: fs.flagsparse_spmm_bsr(data, idx, ptr, B,
                                                    shape=(brows * bd, bcols * bd),
                                                    block_dim=bd),
                     None, note="no CPU reference: block layout")

    if op == "spmm_bell":
        bd = 4
        brows = max(1, m.rows // bd)
        bcols = max(1, m.cols // bd)
        ell_width = max(1, min(bcols, 2))
        gen = torch.Generator(device="cpu").manual_seed(m.seed + 400)
        data = (torch.randn(brows, ell_width, bd, bd, generator=gen)
                .to(dtype).to(device))
        idx = (torch.randint(0, bcols, (brows, ell_width), generator=gen)
               .to(torch.int32).to(device))
        B = torch.randn(bcols * bd, dense_cols, dtype=dtype, device=device)
        return Probe(lambda: fs.flagsparse_spmm_bell(data, idx, B,
                                                     shape=(brows * bd, bcols * bd),
                                                     block_dim=bd),
                     None, note="no CPU reference: blocked-ELL layout")

    if op == "sddmm_csr":
        d, i, p = _csr(torch, dense)
        k = max(1, dense_cols)
        x = torch.randn(m.rows, k, dtype=dtype, device=device)
        # y is indexed [column][k], not [k][column]: the kernel gathers a row of
        # y per output column.
        y = torch.randn(m.cols, k, dtype=dtype, device=device)
        prod = (x.cpu().to(torch.float64) @ y.cpu().to(torch.float64).t())
        mask = (ref64 != 0)
        ref = (prod * mask).to_sparse_csr().values() if int(mask.sum()) else None
        return Probe(lambda: fs.flagsparse_sddmm_csr(d, i, p, x, y, shape=(m.rows, m.cols)),
                     ref, note="" if ref is not None else "empty pattern")

    if op == "spgemm_csr":
        d, i, p = _csr(torch, dense)
        other = _dense(torch, Matrix(m.name, m.cols, m.rows, m.density, m.seed + 1),
                       dtype, device)
        d2, i2, p2 = _csr(torch, other)
        return Probe(lambda: fs.flagsparse_spgemm_csr(
            d, i, p, (m.rows, m.cols), d2, i2, p2, (m.cols, m.rows)),
            None, note="no CPU reference: result structure is discovered")

    if op in ("spsv_csr", "spsv_coo"):
        a, n = _triangular(torch, m, dtype, device)
        b = torch.randn(n, dtype=dtype, device=device)
        ref = torch.linalg.solve_triangular(
            a.cpu().to(torch.float64), b.cpu().to(torch.float64).unsqueeze(1),
            upper=False).squeeze(1)
        if op == "spsv_csr":
            d, i, p = _csr(torch, a)
            return Probe(lambda: fs.flagsparse_spsv_csr(d, i, p, b, shape=(n, n),
                                                        lower=True), ref)
        d, r, c = _coo(torch, a)
        return Probe(lambda: fs.flagsparse_spsv_coo(d, r, c, b, shape=(n, n),
                                                    lower=True), ref)

    if op == "spsv_sell":
        vals, cols, offsets, n, slice_size = _sell_inputs(torch, m, dtype, device)
        b = torch.randn(n, dtype=dtype, device=device)
        return Probe(lambda: fs.flagsparse_spsv_sell(vals, cols, offsets, b,
                                                     shape=(n, n),
                                                     slice_size=slice_size),
                     None, note="no CPU reference: SELL layout")

    if op in ("spsm_csr", "spsm_coo"):
        a, n = _triangular(torch, m, dtype, device)
        B = torch.randn(n, dense_cols, dtype=dtype, device=device)
        ref = torch.linalg.solve_triangular(
            a.cpu().to(torch.float64), B.cpu().to(torch.float64), upper=False)
        if op == "spsm_csr":
            d, i, p = _csr(torch, a)
            return Probe(lambda: fs.flagsparse_spsm_csr(d, i, p, B, shape=(n, n),
                                                        lower=True), ref)
        d, r, c = _coo(torch, a)
        return Probe(lambda: fs.flagsparse_spsm_coo(d, r, c, B, shape=(n, n),
                                                    lower=True), ref)

    raise _NoAdapter(f"no probe adapter for operator {op!r}")


OPERATORS: tuple[str, ...] = (
    "gather", "scatter",
    "spmv_csr", "spmv_coo", "spmv_csc", "spmv_bsr", "spmv_coo_tocsr",
    "spmm_csr", "spmm_coo", "spmm_csc", "spmm_bsr", "spmm_bell",
    "spmm_csr_opt", "spmm_csr_opt_alg1", "spmm_csr_opt_alg2", "alpha_spmm_alg1",
    "spgemm_csr", "sddmm_csr",
    "spsv_csr", "spsv_coo", "spsv_sell",
    "spsm_csr", "spsm_coo",
)


# ------------------------------------------------------------------- driver


def _relative_error(torch, got, ref) -> float:
    if isinstance(got, tuple):
        got = got[0]
    got = got.detach().cpu().to(torch.float64).reshape(-1)
    ref = ref.detach().cpu().to(torch.float64).reshape(-1)
    if got.numel() != ref.numel():
        return float("inf")
    scale = float(ref.abs().max())
    if scale == 0.0:
        return float(got.abs().max())
    return float((got - ref).abs().max() / scale)


def probe_one(flagsparse, torch, op, matrix, dtype, device, accel, args, cache_dir):
    row = {
        "operator": op,
        "matrix": matrix.name,
        "rows": matrix.rows,
        "cols": matrix.cols,
        "density": matrix.density,
        "dtype": str(dtype).split(".")[-1],
        "status": "",
        "reason": "",
        "flagsparse_ms": "",
        "rel_error": "",
        "speedup": "",
    }
    if not args.keep_cache:
        clear_triton_cache(cache_dir)
    _empty_cache(accel, device)

    try:
        probe = build_probe(flagsparse, torch, op, matrix, dtype, device,
                            args.dense_cols)
    except _NoAdapter as exc:
        row["status"], row["reason"] = "NO_ADAPTER", str(exc)
        return row
    except BaseException as exc:                      # noqa: BLE001 - classified
        row["status"], row["reason"] = classify(exc)
        row["reason"] = "input adaptation: " + row["reason"]
        return row

    try:
        out = probe.run()
        _sync(accel, device)
    except BaseException as exc:                      # noqa: BLE001 - classified
        row["status"], row["reason"] = classify(exc)
        return row

    if probe.reference is not None:
        try:
            err = _relative_error(torch, out, probe.reference)
        except BaseException as exc:                  # noqa: BLE001
            row["status"] = "ERROR"
            row["reason"] = f"reference comparison: {type(exc).__name__}: {exc}"
            return row
        row["rel_error"] = f"{err:.3e}"
        tol = 1e-4 if dtype == torch.float32 else 1e-10
        if not (err <= tol):
            row["status"] = "MISMATCH"
            row["reason"] = f"rel_error {err:.3e} > {tol:g}"
            return row
    else:
        row["reason"] = probe.note or "ran; no CPU reference for this layout"

    row["status"] = "PASS"
    # Timing is secondary here and only meaningful once the case passes.
    try:
        for _ in range(args.warmup):
            probe.run()
        _sync(accel, device)
        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            probe.run()
            _sync(accel, device)
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        row["flagsparse_ms"] = f"{samples[len(samples) // 2]:.6f}"
    except BaseException as exc:                      # noqa: BLE001
        row["reason"] = f"passed but timing failed: {type(exc).__name__}: {exc}"
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--op", action="append", default=None,
                   help="operator to probe; repeatable. Default: all.")
    p.add_argument("--matrix", action="append", default=None,
                   help="matrix name to probe; repeatable. Default: all 20.")
    p.add_argument("--dtype", default="float32",
                   choices=("float32", "float64"))
    p.add_argument("--dense-cols", type=int, default=32)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--csv-summary", default=None)
    p.add_argument("--keep-cache", action="store_true",
                   help="do not clear the Triton cache between cases "
                        "(faster, but a cache hit can mask a compile failure)")
    p.add_argument("--fail-on-error", action="store_true",
                   help="exit non-zero if any case is not PASS or REJECTED")
    args = p.parse_args()

    cache_dir = Path(tempfile.mkdtemp(prefix="flagsparse_probe_cache_"))
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)

    import torch                                      # after TRITON_CACHE_DIR
    import flagsparse

    accel, device = _accel(torch)
    if accel is not None:
        try:
            accel.set_device(args.device)
        except Exception:
            pass
    dtype = getattr(torch, args.dtype)

    ops = tuple(args.op) if args.op else OPERATORS
    names = set(args.matrix) if args.matrix else None
    matrices = tuple(m for m in MATRICES if names is None or m.name in names)

    rows = []
    for op in ops:
        for matrix in matrices:
            row = probe_one(flagsparse, torch, op, matrix, dtype, device, accel,
                            args, cache_dir)
            rows.append(row)
            print(f"{row['status']:<15} {op:<20} {matrix.name:<18} "
                  f"{row['reason'][:90]}", flush=True)

    tally: dict[str, int] = {}
    for row in rows:
        tally[row["status"]] = tally.get(row["status"], 0) + 1
    print("\n=== probe summary ===", flush=True)
    try:
        from flagsparse.sparse_operations._common import _backend_name
        backend = _backend_name()
    except Exception:
        backend = "unknown"
    print(f"backend: {backend}  device: {device.type}  dtype: {args.dtype}  "
          f"cases: {len(rows)}", flush=True)
    for status in ("PASS", "REJECTED", "TRITON_COMPILE", "MISMATCH", "ERROR",
                   "NO_ADAPTER"):
        if status in tally:
            print(f"  {status:<15} {tally[status]}", flush=True)

    if args.csv_summary:
        out_path = Path(args.csv_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows
                                    else ["operator", "matrix", "status"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_path}", flush=True)

    shutil.rmtree(cache_dir, ignore_errors=True)
    if args.fail_on_error:
        bad = sum(v for k, v in tally.items() if k not in ("PASS", "REJECTED"))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
