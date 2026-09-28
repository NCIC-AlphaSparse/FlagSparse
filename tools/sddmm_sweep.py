#!/usr/bin/env python3
"""Sweep the SDDMM launch config on the ACTIVE accelerator and check every result.

``_resolve_sddmm_launch_config`` was tuned on sm_120 and knows nothing about the
card it runs on. On Iluvatar BI-V150 (warp size 64) the two matrices that take its
wide branch (mean row length >= 16: BLOCK_P=512, BLOCK_K=32, 4 warps) are 17-44x
slower than on H800 while every other matrix is 5-8x slower. This measures which
(BLOCK_P, BLOCK_K, num_warps) is actually fastest per matrix and k on the card in
front of you, so a per-backend profile can be fitted from data instead of guessed.

Each configuration is also compared with the CPU SciPy result (the same oracle the
accuracy suite uses), so a wrong kernel shows up here rather than as a mystery FAIL
in the runner.

    python3 tools/sddmm_sweep.py                       # delivery matrices, k=32..256
    python3 tools/sddmm_sweep.py --matrices cfd2,cage12 --k 32,256 --csv sweep.csv
    python3 tools/sddmm_sweep.py --block-p 32,64,128,256 --warps 2,4,8   # narrower grid

Run it on the target card (FLAGSPARSE_BACKEND=iluvatar ...), not on the H800.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for entry in (ROOT / "src", ROOT / "tests", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from benchmark_utils import accelerator_device  # noqa: E402
from test_spmm import load_mtx_to_csr_torch  # noqa: E402

import flagsparse as ast  # noqa: E402
from flagsparse.sparse_operations import _common as fs_common  # noqa: E402
from tools.delivery_variants import load_delivery_matrices  # noqa: E402

sddmm_mod = __import__("flagsparse.sparse_operations.sddmm_csr", fromlist=["sddmm_csr"])

_DEFAULT_RESOLVE = sddmm_mod._resolve_sddmm_launch_config


def _ints(text):
    return [int(t) for t in str(text).split(",") if t.strip()]


def _matrix_paths(args):
    base = Path(args.input)
    if base.is_file():
        return [base]
    names = (
        [n.strip() for n in args.matrices.split(",") if n.strip()]
        if args.matrices
        else load_delivery_matrices()
    )
    paths = []
    for name in names:
        candidate = base / (name if name.endswith(".mtx") else f"{name}.mtx")
        if not candidate.is_file():
            raise SystemExit(f"no such matrix: {candidate}")
        paths.append(candidate)
    return paths


def _force(config):
    """Make the kernel launch with exactly ``config`` = (block_p, block_k, warps)."""
    sddmm_mod._resolve_sddmm_launch_config = (
        lambda k, mean_row_len=None, value_dtype=None: config
    )


def _restore():
    sddmm_mod._resolve_sddmm_launch_config = _DEFAULT_RESOLVE


def _time(fn, warmup, iters):
    _out, ms = sddmm_mod._benchmark_cuda_op(fn, warmup, iters)
    return ms


def _measure(config, indices, indptr, shape, x, y, warmup, iters):
    """(ms, output on device) for one launch config, or raises."""
    _force(config)
    try:
        prepared = ast.prepare_sddmm_csr(indices, indptr, shape, k_hint=int(x.shape[1]))
        run = lambda: ast.flagsparse_sddmm_csr(  # noqa: E731
            data=None, x=x, y=y, alpha=1.0, beta=0.0, prepared=prepared
        )
        out = run()
        if isinstance(out, tuple):
            out = out[0]
        ms = _time(run, warmup, iters)
        return ms, out
    finally:
        _restore()


def _cpu_reference(indices, indptr, x, y, chunk=1 << 20):
    """fp64 sampled dot products on the CPU, in chunks.

    ``reference_utils.sddmm_csr_values`` expands an (nnz, k) array; for auto.mtx at
    k=256 that is ~13 GB per operand, enough to be OOM-killed on a small host.
    """
    idx = indices.detach().cpu().numpy().astype(np.int64)
    ptr = indptr.detach().cpu().numpy().astype(np.int64)
    rows = np.repeat(np.arange(ptr.size - 1, dtype=np.int64), np.diff(ptr))
    xn = x.detach().cpu().double().numpy()
    yn = y.detach().cpu().double().numpy()
    out = np.empty(idx.size, dtype=np.float64)
    for start in range(0, idx.size, chunk):
        stop = min(start + chunk, idx.size)
        out[start:stop] = np.einsum(
            "ij,ij->i", xn[rows[start:stop]], yn[idx[start:stop]]
        )
    return torch.as_tensor(out)


def _verdict(out, reference):
    got = out.detach().cpu().to(torch.float64)
    if not torch.isfinite(got).all():
        return "FAIL", float("inf")
    err = float(torch.max(torch.abs(got - reference)).item())
    ok = torch.allclose(got, reference, atol=1e-3, rtol=1e-3)
    return ("PASS" if ok else "FAIL"), err


def _fmt(value, width=8, digits=3):
    return (
        f"{value:>{width}.{digits}f}" if value is not None else " " * (width - 1) + "-"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", nargs="?", default=str(ROOT / "tests" / "data"))
    parser.add_argument(
        "--matrices", help="comma-separated names (default: the delivery set)"
    )
    parser.add_argument("--k", default="32,64,128,256")
    parser.add_argument("--block-p", default="16,32,64,128,256,512")
    parser.add_argument("--block-k", default="16,32,64")
    parser.add_argument("--warps", default="2,4,8")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--csv", help="write every measurement here")
    parser.add_argument(
        "--no-check", action="store_true", help="skip the CPU comparison"
    )
    args = parser.parse_args(argv)

    device = accelerator_device()
    grid = list(
        itertools.product(_ints(args.block_p), _ints(args.block_k), _ints(args.warps))
    )
    print(
        f"backend={fs_common._backend_name()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else device} "
        f"configs={len(grid)} k={args.k}",
        flush=True,
    )

    records = []
    best_lines = []
    for path in _matrix_paths(args):
        _v, indices, indptr, shape = load_mtx_to_csr_torch(
            str(path), dtype=torch.float32, device=device
        )
        indices = indices.to(torch.int32)
        n_rows, n_cols = shape
        nnz = int(indices.numel())
        mean_row = nnz / max(1, n_rows)
        for k in _ints(args.k):
            torch.manual_seed(0)
            x = torch.randn((n_rows, k), dtype=torch.float32).to(device)
            y = torch.randn((n_cols, k), dtype=torch.float32).to(device)
            reference = None
            if not args.no_check:
                reference = _cpu_reference(indices, indptr, x, y)
            default = _DEFAULT_RESOLVE(
                k, mean_row_len=mean_row, value_dtype=torch.float32
            )
            rows = []
            for config in dict.fromkeys([default, *grid]):
                rec = {
                    "matrix": path.stem,
                    "k": k,
                    "nnz": nnz,
                    "n_rows": n_rows,
                    "mean_row": round(mean_row, 2),
                    "block_p": config[0],
                    "block_k": config[1],
                    "warps": config[2],
                    "is_default": config == default,
                    "ms": None,
                    "status": "",
                    "max_abs_err": None,
                    "error": "",
                }
                try:
                    ms, out = _measure(
                        config, indices, indptr, shape, x, y, args.warmup, args.iters
                    )
                    rec["ms"] = ms
                    if reference is not None:
                        rec["status"], rec["max_abs_err"] = _verdict(out, reference)
                    del out
                except Exception as exc:  # a config the compiler rejects is data too
                    rec["status"] = "ERROR"
                    rec["error"] = (
                        f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}"
                    )
                    if "-v" in sys.argv:
                        traceback.print_exc()
                rows.append(rec)
                records.append(rec)
            ok = [
                r for r in rows if r["ms"] is not None and r["status"] in ("PASS", "")
            ]
            base = next(r for r in rows if r["is_default"])
            if not ok:
                print(
                    f"{path.stem:12s} k={k:<4d} 没有可用的配置（默认配置状态 {base['status']} {base['error']}）",
                    flush=True,
                )
                continue
            top = min(ok, key=lambda r: r["ms"])
            gain = base["ms"] / top["ms"] if base["ms"] else float("nan")
            traffic = nnz * 2 * k * 4 / 1e6
            best_lines.append((path.stem, k, mean_row, base, top, gain))
            print(
                f"{path.stem:12s} k={k:<4d} row={mean_row:5.1f} | 默认 {base['block_p']}/{base['block_k']}/{base['warps']} "
                f"{_fmt(base['ms'])} ms {base['status']:5s} | 最优 {top['block_p']}/{top['block_k']}/{top['warps']} "
                f"{_fmt(top['ms'])} ms ({traffic / top['ms']:.0f} GB/s 朴素) | 提升 {gain:.2f}x",
                flush=True,
            )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"wrote {args.csv} ({len(records)} rows)")

    wrong = [r for r in records if r["status"] == "FAIL"]
    errors = [r for r in records if r["status"] == "ERROR"]
    print(f"\n配置总数 {len(records)}: FAIL {len(wrong)}, ERROR {len(errors)}")
    if wrong:
        print("!! 有配置结果和 CPU SciPy 不一致——先查正确性，再谈性能。例：", wrong[0])
    if best_lines:
        gains = [g for *_, g in best_lines if g == g]
        print(
            f"最优 vs 当前默认：几何均值提升 {math.exp(statistics.mean(math.log(g) for g in gains)):.2f}x"
        )
        print("\n按每行非零数分档，哪个配置总用时最短（供拟合 BI-V150 的配置）：")
        for lo, hi in ((0, 8), (8, 16), (16, 1e9)):
            bucket = {r["matrix"] for r in records if lo <= r["mean_row"] < hi}
            cand = {}
            for r in records:
                if (
                    r["matrix"] in bucket
                    and r["ms"] is not None
                    and r["status"] in ("PASS", "")
                ):
                    cand.setdefault(
                        (r["block_p"], r["block_k"], r["warps"]), []
                    ).append((r["matrix"], r["k"], r["ms"]))
            full = {
                c: v
                for c, v in cand.items()
                if len(v) == len({(m, k) for m, k, _ in v}) and len(v) >= len(bucket)
            }
            if full:
                # geometric mean over the cases where every config was measured
                scores = {
                    c: math.exp(statistics.mean(math.log(t) for *_, t in v))
                    for c, v in full.items()
                }
                best = min(scores, key=scores.get)
                print(
                    f"  mean_row [{lo:g}, {hi:g}): {sorted(bucket)} -> BLOCK_P={best[0]} BLOCK_K={best[1]} num_warps={best[2]}"
                )
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
