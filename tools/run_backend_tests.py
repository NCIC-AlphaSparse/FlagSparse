#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the test suite owned by one FlagSparse backend profile."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITES_DIR = ROOT / "tests" / "backends"


def load_suite(backend: str) -> dict[str, object]:
    """Load and minimally validate one backend-owned suite description."""
    path = SUITES_DIR / backend / "suite.json"
    if not path.is_file():
        raise ValueError(f"unknown backend {backend!r}; expected {path}")
    with path.open(encoding="utf-8") as stream:
        suite = json.load(stream)
    required = {
        "backend",
        "selector",
        "capi_backend",
        "capi_buildable",
        "pytest_paths",
        "benchmark_runner",
    }
    missing = required.difference(suite)
    if missing:
        raise ValueError(
            f"{path} is missing required keys: {', '.join(sorted(missing))}"
        )
    if suite["backend"] != backend:
        raise ValueError(
            f"{path} declares backend {suite['backend']!r}, expected {backend!r}"
        )
    return suite


def backend_choices() -> list[str]:
    return sorted(path.parent.name for path in SUITES_DIR.glob("*/suite.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=backend_choices())
    parser.add_argument(
        "--phase", choices=("accuracy", "benchmark", "both"), default="accuracy"
    )
    parser.add_argument("--mode", choices=("quick", "normal"), default="quick")
    parser.add_argument(
        "--ops", help="Comma-separated operator ids passed to the shared runner."
    )
    parser.add_argument("--gpus", help="GPU ids passed to the shared runner.")
    parser.add_argument(
        "--results-dir", help="Result directory passed to the shared runner."
    )
    parser.add_argument(
        "--benchmark-input", help="Benchmark input passed to the shared runner."
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        help="Benchmark warmup passed to the shared runner.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        help="Benchmark iterations passed to the shared runner.",
    )
    args, extra = parser.parse_known_args()

    suite = load_suite(args.backend)
    runner_phase = "performance" if args.phase == "benchmark" else args.phase
    command = [
        sys.executable,
        str(ROOT / str(suite["benchmark_runner"])),
        "--phase",
        runner_phase,
        "--mode",
        args.mode,
    ]
    for option, value in (
        ("--ops", args.ops),
        ("--gpus", args.gpus),
        ("--results-dir", args.results_dir),
        ("--benchmark-input", args.benchmark_input),
        ("--benchmark-warmup", args.benchmark_warmup),
        ("--benchmark-iters", args.benchmark_iters),
    ):
        if value is not None:
            command.extend((option, str(value)))
    command.extend(extra)

    env = os.environ.copy()
    env["FLAGSPARSE_BACKEND"] = str(suite["selector"])
    source_dir = str(ROOT / "src")
    env["PYTHONPATH"] = source_dir + os.pathsep + env.get("PYTHONPATH", "")
    print(
        f"backend={suite['backend']} selector={suite['selector']} "
        f"capi={suite['capi_backend']} buildable={suite['capi_buildable']}",
        flush=True,
    )
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
