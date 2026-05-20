from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one synthetic GPU benchmark suite and capture logs/artifacts."
    )
    parser.add_argument(
        "--suite",
        default="quick",
        choices=[
            "quick",
            "full-synthetic",
            "gather",
            "scatter",
            "spmv",
            "spmv-coo",
            "spmm",
            "spmm-coo",
            "spsv",
            "spsm",
        ],
        help="Benchmark suite to execute.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations where supported.")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations where supported.")
    parser.add_argument(
        "--with-cusparse",
        action="store_true",
        help="Include cuSPARSE baselines for suites that support them.",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark_results",
        help="Directory to write logs and generated CSV artifacts.",
    )
    return parser.parse_args()


def _command_specs(
    args: argparse.Namespace, results_dir: Path
) -> List[Tuple[str, List[str]]]:
    no_cusparse = [] if args.with_cusparse else ["--no-cusparse"]
    commands: Dict[str, List[str]] = {
        "gather": [
            "tests/test_gather.py",
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--csv-summary",
            str(results_dir / "gather_summary.csv"),
            "--csv-samples",
            str(results_dir / "gather_samples.csv"),
        ],
        "scatter": [
            "tests/test_scatter.py",
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--csv-summary",
            str(results_dir / "scatter_summary.csv"),
            "--csv-samples",
            str(results_dir / "scatter_samples.csv"),
        ],
        "spmv": [
            "tests/test_spmv.py",
            "--synthetic",
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            *no_cusparse,
        ],
        "spmv-coo": [
            "tests/test_spmv_coo.py",
            "--synthetic",
        ],
        "spmm": [
            "tests/test_spmm.py",
            "--synthetic",
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            *no_cusparse,
        ],
        "spmm-coo": [
            "tests/test_spmm_coo.py",
            "--synthetic",
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            *no_cusparse,
        ],
        "spsv": [
            "tests/test_spsv.py",
            "--synthetic",
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ],
        "spsm": [
            "tests/test_spsm.py",
            "--synthetic",
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ],
    }
    suites: Dict[str, List[str]] = {
        "quick": ["gather", "scatter", "spmv", "spmm"],
        "full-synthetic": ["gather", "scatter", "spmv", "spmv-coo", "spmm", "spmm-coo", "spsv", "spsm"],
    }
    selected = suites.get(args.suite, [args.suite])
    return [(name, commands[name]) for name in selected]


def _gpu_metadata() -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "platform": platform.platform(),
        "python": sys.version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        metadata["torch_import_error"] = repr(exc)
        return metadata

    metadata["torch_version"] = getattr(torch, "__version__", "unknown")
    metadata["cuda_available"] = bool(torch.cuda.is_available())
    metadata["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        metadata["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return metadata


def _run_command(name: str, script_args: List[str], log_path: Path) -> int:
    full_cmd = [sys.executable, *script_args]
    print(f"==> {name}: {' '.join(full_cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(full_cmd)}\n\n")
        process = subprocess.Popen(
            full_cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return process.wait()


def main() -> int:
    args = _parse_args()
    results_dir = (ROOT / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    metadata = _gpu_metadata()
    metadata["suite"] = args.suite
    metadata["warmup"] = args.warmup
    metadata["iters"] = args.iters
    metadata["with_cusparse"] = args.with_cusparse
    commands = _command_specs(args, results_dir)
    metadata["commands"] = [{"name": name, "argv": [sys.executable, *argv]} for name, argv in commands]
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    failures: List[Dict[str, object]] = []
    for name, argv in commands:
        log_path = results_dir / f"{name}.log"
        return_code = _run_command(name, argv, log_path)
        if return_code != 0:
            failures.append({"name": name, "return_code": return_code, "log": str(log_path)})

    if failures:
        (results_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        for failure in failures:
            print(
                f"Benchmark '{failure['name']}' failed with exit code {failure['return_code']} "
                f"(see {failure['log']}).",
                file=sys.stderr,
            )
        return 1

    print(f"Benchmark artifacts written to {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
