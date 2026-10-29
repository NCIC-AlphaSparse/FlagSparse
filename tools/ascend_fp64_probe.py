#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Probe why Ascend FP64 SpMM reads 0.53x in delivery but 0.99x in its own bench.

The delivery table reports spmm_coo_f64 at 0.529x against the CANN baseline while
the FP64 optimisation report measured 0.992x for the same variant. Both compare
against CANN ``SparseTensorDenseMatMul``, so one of three things differs. This
probe measures all three on the ten delivery matrices and prints them side by
side:

  1. Input order. ``benchmark_ascend.py`` builds its COO straight from
     ``mmread(...).tocoo()``, i.e. the raw Matrix Market order, while the FP64
     bench fed ``tocsr().sort_indices().tocoo()``. With unsorted rows the public
     one-shot path runs an nnz-wide ``argsort`` plus three ``index_select`` calls
     on every call, inside the timed region. Column ``row_sorted`` says which it
     is here.
  2. Timing boundary. The CANN adapter reports its own setup separately in
     ``cann_prepare_ms`` and times only ``plan.run()``, while the delivery number
     times the whole public call. Columns ``public`` / ``prepared`` / ``cann``
     quantify that: ``cann/prepared`` is the symmetric comparison.
  3. Whether the sort is needed at all. Column ``shuffled`` runs the native path
     on randomly permuted coordinates and checks the result against a CPU golden.
     TensorFlow's SparseTensorDenseMatMul does not require ordered indices; if
     CANN's agrees, the per-call argsort can go and the public number should
     reach the prepared one.

Reading the result:
  cann/prepared ~= 0.99 and shuffled OK  -> drop the sort (one default), public
                                            should follow to ~0.99
  cann/prepared ~= 0.99 and shuffled bad -> the sort is required; 0.53x is the
                                            real one-shot cost and only the
                                            comparison boundary can change
  row_sorted all True                    -> the argsort was not the cause; the
                                            public/prepared gap says what is

What it cannot tell: whether dropping the sort is safe for duplicate coordinates
or an all-empty matrix -- it uses the ten deduplicated delivery matrices and one
random permutation each. The GE operator is also known to leave nnz=0 output
uninitialised, which this does not exercise.

Usage (Ascend box, repo root; pick a free card):
    ASCEND_RT_VISIBLE_DEVICES=1 python3 tools/ascend_fp64_probe.py
    ASCEND_RT_VISIBLE_DEVICES=1 python3 tools/ascend_fp64_probe.py --input tests/data

Read-only: writes no files and does not touch src/. Exit status 0 when every
matrix was measured.
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.io
import torch

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import flagsparse as fs  # noqa: E402
from tests import ascend_sparse_baseline as cann  # noqa: E402

MATRICES = [
    "ASIC_680ks",
    "GL7d14",
    "NACA0015",
    "amazon0601",
    "auto",
    "cage12",
    "cfd2",
    "filter3D",
    "roadNet-TX",
    "wave",
]
NATIVE_TAG = "cann_sparse_tensor_dense_matmul_fp64"


def _require_npu():
    """Import torch_npu here, not at module scope, so --help works anywhere."""
    try:
        import torch_npu  # noqa: F401  (registers the npu device)
    except ImportError as exc:
        raise SystemExit(f"this probe needs torch_npu on an Ascend box: {exc}")
    if not getattr(torch, "npu", None) or not torch.npu.is_available():
        raise SystemExit("no Ascend NPU is available to this process")


def _timed(call, warmup, iters):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        started = time.perf_counter_ns()
        call()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return sum(samples) / len(samples)


def _one(path, width, warmup, iters):
    # The delivery harness reads exactly this order; see benchmark_ascend.py.
    raw = scipy.io.mmread(path).tocoo()
    csr = raw.tocsr()
    csr.sum_duplicates()
    if csr.nnz != raw.nnz:
        raise AssertionError("duplicate coordinates; golden is not comparable")
    row_sorted = bool((np.diff(raw.row) >= 0).all())

    generator = torch.Generator().manual_seed(20261028)
    rhs = torch.randn((csr.shape[1], width), dtype=torch.float64, generator=generator)
    dense = rhs.to("npu")
    golden = torch.from_numpy(np.asarray(csr @ rhs.numpy()))

    data = torch.from_numpy(raw.data.astype(np.float64)).to("npu")
    row = torch.from_numpy(raw.row.astype(np.int32)).to("npu")
    col = torch.from_numpy(raw.col.astype(np.int32)).to("npu")

    # Strict mode: a preparation or launch failure raises instead of silently
    # falling back, so a time below is never the scatter path in disguise.
    os.environ["FLAGSPARSE_ASCEND_FP64_SPMM"] = "native"
    _, meta = fs.flagsparse_spmm_coo(
        data, row, col, dense, csr.shape, op="non", return_meta=True
    )
    if meta.get("implementation") != NATIVE_TAG:
        raise AssertionError(f"not the native route: {meta}")

    public_ms = _timed(
        lambda: fs.flagsparse_spmm_coo(data, row, col, dense, csr.shape, op="non"),
        warmup,
        iters,
    )
    prepared = fs.prepare_spmm_coo_route(data, row, col, csr.shape, op="non")
    prepared_ms = _timed(
        lambda: fs.flagsparse_spmm_coo_run(prepared, dense), warmup, iters
    )

    out = torch.empty((csr.shape[0], width), dtype=torch.float64, device="npu")
    plan = cann.prepare_spmm("coo", (data, row, col), csr.shape, dense, out)
    try:
        cann_ms = _timed(plan.run, warmup, iters)
    finally:
        plan.close()

    order = torch.randperm(data.numel(), device="npu")
    shuffled = fs.flagsparse_spmm_coo(
        data.index_select(0, order),
        row.index_select(0, order),
        col.index_select(0, order),
        dense,
        csr.shape,
        op="non",
    )
    ordered = fs.flagsparse_spmm_coo(data, row, col, dense, csr.shape, op="non")

    return dict(
        row_sorted=row_sorted,
        public=public_ms,
        prepared=prepared_ms,
        cann=cann_ms,
        shuffled_error=(shuffled.cpu() - golden).abs().max().item(),
        ordered_error=(ordered.cpu() - golden).abs().max().item(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("tests/data"))
    parser.add_argument("--matrices", default=",".join(MATRICES))
    parser.add_argument("--width", type=int, default=32, help="dense columns (N)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    args = parser.parse_args(argv)
    _require_npu()

    print(
        f"{'matrix':12s}{'row_sorted':>11s}{'public':>9s}{'prepared':>9s}{'cann':>9s}"
        f"{'cann/pub':>9s}{'cann/prep':>10s}{'shuffled':>10s}{'ordered':>10s}"
    )
    public_ratios, prepared_ratios, failures = [], [], 0
    for name in args.matrices.split(","):
        try:
            got = _one(args.input / f"{name}.mtx", args.width, args.warmup, args.iters)
        except Exception:
            failures += 1
            print(f"{name[:11]:12s}  FAILED")
            traceback.print_exc(limit=3)
            continue
        public_ratios.append(got["cann"] / got["public"])
        prepared_ratios.append(got["cann"] / got["prepared"])
        verdict = (
            "OK" if got["shuffled_error"] < 1e-9 else f"{got['shuffled_error']:.1e}"
        )
        print(
            f"{name[:11]:12s}{str(got['row_sorted']):>11s}{got['public']:9.1f}"
            f"{got['prepared']:9.1f}{got['cann']:9.1f}{got['cann'] / got['public']:9.3f}"
            f"{got['cann'] / got['prepared']:10.3f}{verdict:>10s}"
            f"{got['ordered_error']:10.1e}"
        )

    if public_ratios:
        count = len(public_ratios)
        print(
            f"\narithmetic mean over {count} matrices:"
            f"  cann/public = {sum(public_ratios) / count:.4f}"
            f"   cann/prepared = {sum(prepared_ratios) / count:.4f}"
        )
        print(
            "delivery reports spmm_coo_f64 = 0.529x (cann/public);"
            " the FP64 report's 0.992x fed sorted input"
        )
    return 1 if failures or not public_ratios else 0


if __name__ == "__main__":
    sys.exit(main())
