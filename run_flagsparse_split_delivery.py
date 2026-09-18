#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the 40-variant delivery report with accuracy from pytest and
performance from the C API.

Why split: on a backend with no Python-side vendor sparse library -- MUSA is
the first case, torch.sparse has no working matmul there, see
docs/MUSA.md section 3 -- run_flagsparse_pytest.py's own ``--phase
performance`` has nothing to compute a speedup against; every row's baseline
column is empty by construction (see
flagsparse.sparse_operations._common._mthreads_vendor_sparse_library()).  The
C API side has its own vendor baseline
(capi/ctest/baseline/<backend>/baseline.cpp, muSPARSE on MUSA) and its own
delivery projection (capi/tools/write_summary.py, same summary.json schema),
so performance is more honest to source there instead.  Accuracy stays on the
Python side: it is the larger, more exercised suite (see
modified/<BACKEND>.md) and already defaults to a CPU/SciPy reference on
non-CUDA backends
(flagsparse.sparse_operations._common._use_scipy_accuracy_reference()).

Flag names match run_flagsparse_pytest.py wherever the same concept applies
(--mode, --gpus, --results-dir, --benchmark-input, --timeout, --strict), so a
runbook written against one transfers to the other.  ``--benchmark-warmup``/
``--benchmark-iters`` are deliberately NOT exposed here: the C API benchmark
harness hardcodes its own warmup/iters (capi/ctest/common.hpp: kWarmup=10,
kIters=100) with no env/CLI override, so a flag that implied control over it
would be misleading.

This always runs the full 40-variant delivery set on both sides (no
--ops/--delivery-only): narrowing one side without the other would produce a
report where the two halves cover different variants, and prompt.md's rule
against trimming the 40-variant list to dodge a failing operator applies
here too.

Run from the repository root:

    python3 run_flagsparse_split_delivery.py \\
        --mode normal --benchmark-input /root/gcx/matrix \\
        --timeout 3600 --results-dir pytest_results_mthreads_split_20260917

Produces, same as run_flagsparse_pytest.py's own delivery run:
    <results-dir>/summary.json              accuracy only, from pytest
    <capi-bench-out>/summary.json           performance only, from the C API
    <results-dir>/summary_split.json        merged: accuracy from the first,
                                             performance from the second
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CAPI_SRC_DIR = PROJECT_ROOT / "capi"

# The pytest side's FLAGSPARSE_BACKEND env var and the C API's -DBACKEND=
# cmake cache variable (capi/CMakeLists.txt:23) spell the same backend
# differently -- see capi/ctest/CMakeLists.txt's FLAGSPARSE_CTEST_PROFILE
# table for the authoritative pairing. Extend this when adding a backend here.
CAPI_BACKEND_BY_PYTEST_BACKEND = {
    "mthreads": "MUSA",
    "rocm": "ROCM",
    "metax": "MACA",
    "cuda": "CUDA",
    "ascend": "NPU",
    "xpu": "XPU",
}


def log(message: str) -> None:
    print(f"[run_flagsparse_split_delivery] {message}", flush=True)


def run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = False) -> int:
    log("+ " + " ".join(str(part) for part in cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if proc.returncode != 0:
        log(f"exit code {proc.returncode}")
        if check:
            raise SystemExit(proc.returncode)
    return proc.returncode


def run_pytest_accuracy(args: argparse.Namespace, results_dir: Path) -> int:
    """40-variant accuracy via run_flagsparse_pytest.py; SciPy reference on
    non-CUDA backends is that script's own default, not something this
    wrapper sets."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_flagsparse_pytest.py"),
        "--phase", "accuracy",
        "--mode", args.mode,
        "--delivery-only",
        "--gpus", args.gpus,
        "--results-dir", str(results_dir),
    ]
    if args.timeout:
        cmd += ["--timeout", str(args.timeout)]
    if args.strict:
        cmd.append("--strict")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["FLAGSPARSE_BACKEND"] = args.backend
    return run(cmd, env=env)


def configure_and_build_capi(args: argparse.Namespace) -> int:
    if args.skip_capi_build:
        log(f"--skip-capi-build: not touching {args.capi_build_dir}")
        return 0
    capi_backend = CAPI_BACKEND_BY_PYTEST_BACKEND.get(args.backend, args.backend.upper())
    cmake_cmd = [
        "cmake", "-S", str(CAPI_SRC_DIR), "-B", str(PROJECT_ROOT / args.capi_build_dir),
        "-G", "Ninja",
        f"-DBACKEND={capi_backend}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if args.musa_home:
        cmake_cmd.append(f"-DMUSA_HOME={args.musa_home}")
    # 900s (the CMake default) is enough for the synthetic corpus but not for
    # --benchmark-input pointed at a real matrix directory; see
    # capi/ctest/CMakeLists.txt's FLAGSPARSE_CTEST_TIMEOUT and
    # modified/MUSA.md section 10. --timeout 0 (disabled) leaves the cached
    # default alone rather than passing a literal 0 through to ctest.
    if args.timeout:
        cmake_cmd.append(f"-DFLAGSPARSE_CTEST_TIMEOUT={args.timeout}")
    rc = run(cmake_cmd, check=True)
    if rc != 0:
        return rc
    return run(["cmake", "--build", str(PROJECT_ROOT / args.capi_build_dir), "-j"], check=True)


def run_capi_benchmark(args: argparse.Namespace, bench_out: Path) -> int:
    """40-variant performance via `ctest -R benchmark`, vendor baseline
    (muSPARSE on MUSA -- capi/ctest/baseline/<backend>/baseline.cpp), then
    project onto the delivery list the same way capi/docs/MUSA.md's own
    recipe does (tools/write_summary.py + tools/check_manifest.py)."""
    bench_out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["FLAGSPARSE_MATRIX_DIR"] = str(Path(args.benchmark_input).resolve())
    env["FLAGSPARSE_BENCH_OUT"] = str(bench_out.resolve())
    rc = run(
        ["ctest", "--test-dir", str(PROJECT_ROOT / args.capi_build_dir),
         "-R", "benchmark", "--output-on-failure"],
        env=env,
    )
    run([sys.executable, str(CAPI_SRC_DIR / "tools" / "write_summary.py"),
         "--bench-dir", str(bench_out), "--out", str(bench_out)])
    check_manifest_cmd = [
        sys.executable, str(CAPI_SRC_DIR / "tools" / "check_manifest.py"),
        "--bench-dir", str(bench_out),
    ]
    if args.strict:
        check_manifest_cmd.append("--strict")
    run(check_manifest_cmd)
    return rc


def merge_summaries(py_summary_path: Path, capi_summary_path: Path, out_path: Path) -> None:
    """One summary.json: accuracy from the pytest run, performance from the
    C API run.

    Deliberately NOT the C API's own per-row accuracy (the host-fp64 check
    that gates whether a benchmark row counts toward its speedup --
    capi/ctest/common.cpp's write_accuracy_json()/BenchReport::add(), "The
    gate: a speedup is written only over an answer we checked and believed").
    That is a correctness gate internal to the C API's timing methodology,
    not the delivery accuracy result. Both files share one schema
    (capi/tools/write_summary.py's docstring: "the SAME schema FlagSparse's
    Python runner emits"), so the merge is a shallow per-variant overlay.
    """
    if not py_summary_path.exists():
        log(f"no {py_summary_path}; skipping merge (pytest accuracy phase produced nothing)")
        return
    if not capi_summary_path.exists():
        log(f"no {capi_summary_path}; skipping merge (C API performance phase produced nothing)")
        return
    py_summary = json.loads(py_summary_path.read_text(encoding="utf-8"))
    capi_summary = json.loads(capi_summary_path.read_text(encoding="utf-8"))
    py_result = py_summary.get("result", {})
    capi_result = capi_summary.get("result", {})

    merged_result: dict[str, object] = {}
    for variant_id in sorted(set(py_result) | set(capi_result)):
        capi_entry = capi_result.get(variant_id) or {}
        py_entry = py_result.get(variant_id) or {}
        entry = copy.deepcopy(capi_entry)
        # Every entry gets both keys regardless of which side(s) had the variant:
        # a variant missing from the C API run (e.g. its benchmark crashed before
        # writing any row) must still report performance: NOT_CONFIGURED rather
        # than silently dropping the key, or a consumer indexing entry["performance"]
        # breaks on exactly the variants most worth flagging.
        entry["accuracy"] = py_entry.get("accuracy") or entry.get("accuracy") or {"status": "NOT_CONFIGURED"}
        entry.setdefault("performance", {"status": "NOT_CONFIGURED"})
        entry.setdefault("customized", py_entry.get("customized", capi_entry.get("customized", True)))
        entry.setdefault("labels", py_entry.get("labels") or capi_entry.get("labels") or [])
        merged_result[variant_id] = entry

    merged = {
        "timestamp": capi_summary.get("timestamp") or py_summary.get("timestamp"),
        "env": {"pytest_accuracy": py_summary.get("env"), "capi_performance": capi_summary.get("env")},
        "result": merged_result,
        "sources": {"accuracy": str(py_summary_path), "performance": str(capi_summary_path)},
    }
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    total = len(merged_result)
    acc_pass = sum(1 for e in merged_result.values() if e.get("accuracy", {}).get("status") == "Passed")
    perf_pass = sum(1 for e in merged_result.values() if e.get("performance", {}).get("status") == "Passed")
    perf_nobase = sum(1 for e in merged_result.values() if e.get("performance", {}).get("status") == "NoBaseline")
    log(f"wrote {out_path}: {total} variants, accuracy Passed={acc_pass}, performance Passed={perf_pass}, "
        f"NoBaseline={perf_nobase} (ran and passed, no vendor baseline to compare)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend", default="mthreads",
        help="FLAGSPARSE_BACKEND value for the pytest side (default: mthreads). "
             "Mapped to the C API -DBACKEND= value via an internal table.",
    )
    parser.add_argument("--mode", default="quick", choices=("quick", "normal"),
                         help="Same as run_flagsparse_pytest.py --mode; pytest side only.")
    parser.add_argument("--gpus", default="0",
                         help="Same as run_flagsparse_pytest.py --gpus; pytest side only.")
    parser.add_argument("--results-dir", default=None,
                         help="pytest accuracy results dir (default: pytest_results_<backend>_split_accuracy).")
    parser.add_argument(
        "--benchmark-input", default="tests/data",
        help="Matrix file or directory. Same default and same caveat as "
             "run_flagsparse_pytest.py: not passing this falls back to the 3 "
             "tiny matrices under tests/data. Feeds FLAGSPARSE_MATRIX_DIR for "
             "the C API benchmark.",
    )
    parser.add_argument(
        "--timeout", type=int, default=0,
        help="Per-phase timeout in seconds; 0 disables (same meaning as "
             "run_flagsparse_pytest.py --timeout). Applied to both the pytest "
             "--timeout and the C API's -DFLAGSPARSE_CTEST_TIMEOUT.",
    )
    parser.add_argument("--strict", action="store_true",
                         help="Same as run_flagsparse_pytest.py --strict; passed to both sides.")
    parser.add_argument("--capi-build-dir", default="capi/build",
                         help="Relative to the repo root (default: capi/build).")
    parser.add_argument("--capi-bench-out", default=None,
                         help="Default: capi/build/bench_<backend>_split.")
    parser.add_argument("--musa-home", default=None,
                         help="Overrides MUSA_HOME for the C API cmake configure step.")
    parser.add_argument("--skip-capi-build", action="store_true",
                         help="Reuse the existing --capi-build-dir instead of reconfiguring/rebuilding.")
    parser.add_argument("--skip-accuracy", action="store_true", help="Skip the pytest accuracy phase.")
    parser.add_argument("--skip-performance", action="store_true", help="Skip the C API performance phase.")
    parser.add_argument("--out", default=None,
                         help="Merged summary.json path (default: <results-dir>/summary_split.json).")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(
        f"pytest_results_{args.backend}_split_accuracy"
    )
    bench_out = Path(args.capi_bench_out) if args.capi_bench_out else Path(
        f"capi/build/bench_{args.backend}_split"
    )

    accuracy_rc = 0
    if args.skip_accuracy:
        log("--skip-accuracy: not running the pytest phase")
    else:
        accuracy_rc = run_pytest_accuracy(args, results_dir)

    performance_rc = 0
    if args.skip_performance:
        log("--skip-performance: not running the C API phase")
    else:
        build_rc = configure_and_build_capi(args)
        if build_rc != 0:
            return build_rc
        performance_rc = run_capi_benchmark(args, bench_out)

    if not args.skip_accuracy and not args.skip_performance:
        out_path = Path(args.out) if args.out else results_dir / "summary_split.json"
        merge_summaries(results_dir / "summary.json", bench_out / "summary.json", out_path)

    if accuracy_rc != 0:
        log(f"pytest accuracy phase exited {accuracy_rc}")
    if performance_rc != 0:
        log(f"C API performance phase exited {performance_rc}")
    return accuracy_rc or performance_rc


if __name__ == "__main__":
    raise SystemExit(main())
