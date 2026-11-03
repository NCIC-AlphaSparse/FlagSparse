#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""One command for the 105-variant delivery run, on every backend.

    setsid timeout -s KILL 43200 python3 -u run_flagsparse_delivery.py \\
        > delivery_105.log 2>&1 < /dev/null &

The backend is detected the way the package itself detects it
(flagsparse.sparse_operations._common._backend_name(); FLAGSPARSE_BACKEND, when
set, wins), and everything that used to differ between the per-backend commands
in docs/debug/ comes from PROFILES below instead of the command line:

* which runner: run_flagsparse_pytest.py --delivery-only (accuracy and
  performance from pytest), or run_flagsparse_split_delivery.py on MUSA, where
  torch.sparse has no matmul and the performance half comes from the C API's
  muSPARSE baseline;
* --h800-reference for backends whose vendor library leaves rows without a
  baseline (not CUDA, whose baseline is cuSPARSE itself, nor DCU, by choice);
* the vendor environment variables the Iluvatar and Ascend commands exported;
* per-backend benchmark/pytest restrictions: Iluvatar cannot run fp64 (its H2D
  copy silently returns zeros), MetaX C550 has no usable SDDMM reference.

DCU, MetaX and Iluvatar used to pass an explicit --ops list instead of
--delivery-only, which skips the delivery narrowing: spmv_csr then averaged all
seven algorithms into the delivery number (docs/debug/DCU.md). Every backend now
takes --delivery-only, which selects the same 15 parent operators.

Machine setup stays outside the command: source the vendor toolkit (Ascend's
set_env.sh, MUSA_HOME for the C API build), and pick cards with FLAGSPARSE_GPUS
(physical ids, the runners' --gpus; default 0) or the vendor's own visibility
variable. Afterwards the delivery table is printed and written to
<results>/delivery_table.txt.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Shared by every backend: the git-tracked delivery matrices (tests/data holds
# exactly the 10), the timing the delivery reports have always used.
COMMON_ARGS = (
    "--phase", "both",
    "--mode", "normal",
    "--delivery-only",
    "--benchmark-input", "tests/data",
    "--benchmark-warmup", "5",
    "--benchmark-iters", "20",
)


@dataclass(frozen=True)
class Profile:
    runner: str = "pytest"  # "pytest" or "split"
    h800_reference: bool = False
    timeout: int = 3600  # per operator and phase, the runners' --timeout
    env: dict[str, str] = field(default_factory=dict)  # defaults; the caller's values win
    pytest_args: str | None = None
    op_benchmark_args: tuple[str, ...] = ()


# Values carried over unchanged from the 2026-11-02 retest commands in
# docs/debug/<BACKEND>.md; the reasons are recorded there.
PROFILES: dict[str, Profile] = {
    "cuda": Profile(),
    "rocm": Profile(),
    "metax": Profile(
        h800_reference=True,
        # SDDMM's four K values are not narrowed, so one phase runs long.
        timeout=4500,
        # torch.sparse.sampled_addmm is wrong on C550 (docs/MACA.md 7.3).
        op_benchmark_args=("sddmm_csr=--no-cusparse",),
    ),
    "mthreads": Profile(runner="split", h800_reference=True),
    "iluvatar": Profile(
        h800_reference=True,
        env={
            "FLAGSPARSE_BACKEND": "iluvatar",
            "FLAGSPARSE_ILUVATAR_VENDOR": "cupy_cusparse",
        },
        # fp64 H2D silently returns zeros on BI-V150: keep fp64/complex128 cases
        # out, but every q4 variant case in (they are checked on the CPU).
        pytest_args=(
            '-k "test_q4_variant or (((float or half or f16 or f32) and not '
            "(double or float64 or bfloat16 or complex64 or complex128)) or "
            '(complex64 and (gather or scatter)))"'
        ),
        op_benchmark_args=(
            "gather=--value-dtypes float16,float32,complex64",
            "scatter=--value-dtypes float16,float32,complex64,int8",
            "spmv_csr=--dtypes float32 --alg auto",
            "spmv_coo=--dtypes float32",
            "spmv_csc=--dtypes float32,complex64",
            "spmm_csr=--dtypes float16,float32",
            "spmm_coo=--dtypes float16,float32",
            "spmm_csc=--dtypes float32,complex64",
            "sddmm_csr=--dtype float32",
            "spgemm_csr=--dtypes float32",
        ),
    ),
    "ascend": Profile(
        h800_reference=True,
        env={"FLAGSPARSE_BACKEND": "ascend", "FLAGSPARSE_ASCEND_VENDOR": "torch"},
    ),
}


def detect_backend() -> str:
    """The package's own answer, in a child so this process never touches a device.

    src/ goes first on the path: a plain `python3` here otherwise imports any
    older flagsparse installed under ~/.local instead of this checkout.
    """
    override = os.environ.get("FLAGSPARSE_BACKEND", "").strip().lower()
    if override:
        return override
    # _backend_name() falls back to "cuda" when no vendor plugin imports -- which
    # is also what an Ascend box without set_env.sh looks like -- so the device
    # has to be there too, or the run would take the CUDA profile on an NPU.
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from flagsparse.sparse_operations._common import _ACCEL, _backend_name; "
        "print(_backend_name(), bool(_ACCEL.is_available()))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "src")],
        capture_output=True, text=True, cwd=ROOT,
    )
    fields = proc.stdout.split()[-2:] if proc.returncode == 0 else []
    if len(fields) != 2 or fields[1] != "True":
        raise SystemExit(
            "backend detection found no usable accelerator "
            f"({' '.join(fields) or 'import failed'}): is the vendor toolkit sourced? "
            f"Set FLAGSPARSE_BACKEND or pass --backend to skip detection.\n{proc.stderr.strip()}"
        )
    return fields[0]


def build_command(
    backend: str, profile: Profile, results_dir: str, gpus: str
) -> list[str]:
    if profile.runner == "split":
        # The split runner takes the pytest-side flags it shares, always runs the
        # full delivery set, and times the C API with its own warmup/iters.
        cmd = [
            sys.executable, "-u", "run_flagsparse_split_delivery.py",
            "--backend", backend,
            "--mode", "normal",
            "--benchmark-input", "tests/data",
        ]
    else:
        cmd = [sys.executable, "-u", "run_flagsparse_pytest.py", *COMMON_ARGS]
    cmd += ["--gpus", gpus, "--timeout", str(profile.timeout)]
    if profile.h800_reference:
        cmd.append("--h800-reference")
    if profile.pytest_args:
        cmd.append(f"--pytest-args={profile.pytest_args}")
    cmd += [f"--op-benchmark-args={value}" for value in profile.op_benchmark_args]
    cmd += ["--results-dir", results_dir]
    return cmd


def build_env(profile: Profile, tmp_dir: str | None) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in profile.env.items():
        env.setdefault(key, value)
    if tmp_dir:
        # A private scratch dir per run, as the DCU command made by hand: a
        # shared /tmp filled by other jobs has broken Triton's ROCm clang step.
        for key in ("TMPDIR", "TMP", "TEMP"):
            env[key] = tmp_dir
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backend", choices=sorted(PROFILES), default=None,
        help="Skip detection (default: FLAGSPARSE_BACKEND, else auto-detected).",
    )
    parser.add_argument(
        "--gpus", default=os.environ.get("FLAGSPARSE_GPUS", "0"),
        help="Physical device ids, comma-separated (default: $FLAGSPARSE_GPUS or 0).",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Default: results_<backend>_105_<YYYYMMDD_HHMM>, never an existing one.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved environment and command, run nothing.",
    )
    args = parser.parse_args(argv)

    backend = args.backend or detect_backend()
    profile = PROFILES.get(backend)
    if profile is None:
        parser.error(
            f"no delivery profile for backend {backend!r}; known: {', '.join(sorted(PROFILES))}"
        )
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = args.results_dir or f"results_{backend}_105_{stamp}"
    if (ROOT / results_dir).exists():
        # A rerun into an existing directory re-projects the summary over stale
        # rows; a delivery run always starts in a fresh one.
        parser.error(f"{results_dir} already exists; pick a new --results-dir")

    cmd = build_command(backend, profile, results_dir, args.gpus)
    set_env = {k: v for k, v in profile.env.items() if k not in os.environ}
    print(f"[delivery] backend={backend} runner={profile.runner} results={results_dir}", flush=True)
    for key, value in set_env.items():
        print(f"[delivery] env {key}={value}", flush=True)
    print("[delivery] + " + shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0

    with tempfile.TemporaryDirectory(prefix=f"flagsparse-{backend}-105.") as tmp_dir:
        rc = subprocess.call(cmd, cwd=ROOT, env=build_env(profile, tmp_dir))
    table = subprocess.run(
        [sys.executable, "tools/delivery_table.py", results_dir, "--show-baseline"],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(table.stdout + table.stderr, flush=True)
    if (ROOT / results_dir).is_dir():
        (ROOT / results_dir / "delivery_table.txt").write_text(
            table.stdout + table.stderr, encoding="utf-8"
        )
    print(f"[delivery] runner exit={rc}; table in {results_dir}/delivery_table.txt", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
