#!/usr/bin/env python3
"""Run FlagSparse accuracy and performance suites per operator.

The operator inventory comes from ``conf/operators.yaml`` by default.  Each
configured operator is run as an isolated subprocess on one requested GPU, with
accuracy first and performance second, then all results are summarized together.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

try:
    from openpyxl import Workbook
except Exception:
    Workbook = None


ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SUMMARY_RE = re.compile(r"(\d+)\s+([A-Za-z_]+)")
SUMMARY_LOCK = threading.Lock()
TIMEOUT_RETURN_CODE = -100
DEFAULT_EXCLUDED_OPS = {
    "alpha_spmm_alg1",
    "spmv_coo_tocsr",
    "spsv_descriptor_api",
    "sparse_format_constructors",
}


@dataclass(frozen=True)
class OperatorTestConfig:
    accuracy_marker: str | None = None
    performance_cmd: tuple[str, ...] | None = None


PERFORMANCE_COMMANDS: dict[str, tuple[str, ...]] = {
    "gather": (
        "tests/test_gather.py",
        "--csv-summary",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "scatter": (
        "tests/test_scatter.py",
        "--csv-summary",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spmv_csr": (
        "tests/test_spmv.py",
        "{input}",
        "--csv-csr",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spmv_coo": (
        "tests/test_spmv_coo.py",
        "{input}",
        "--csv-coo",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spmv_coo_tocsr": (
        "tests/test_spmv_coo.py",
        "{input}",
        "--csv-tocsr",
        "{csv}",
        "--dtypes",
        "float32,float64",
        "--ops",
        "non",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spmm_csr": (
        "tests/test_spmm.py",
        "{input}",
        "--csv",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spmm_coo": (
        "tests/test_spmm_coo.py",
        "{input}",
        "--csv",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spmm_csr_opt": (
        "tests/test_spmm_opt.py",
        "{input}",
        "--csv",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spmm_csr_opt_alg1": (
        "tests/test_spmm_opt.py",
        "{input}",
        "--csv",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spmm_csr_opt_alg2": (
        "tests/test_spmm_opt_alg2.py",
        "--synthetic",
        "--csv",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "alpha_spmm_alg1": (
        "tests/test_alpha_spmm_alg1.py",
        "--synthetic",
        "--csv",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spgemm_csr": (
        "tests/test_spgemm.py",
        "{input}",
        "--csv",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "sddmm_csr": (
        "tests/test_sddmm.py",
        "{input}",
        "--csv",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spsv_csr": (
        "tests/test_spsv.py",
        "{input}",
        "--csv-csr",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spsv_coo": (
        "tests/test_spsv.py",
        "{input}",
        "--csv-coo",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spsm_csr": (
        "tests/test_spsm.py",
        "{input}",
        "--csv-csr",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spsm_coo": (
        "tests/test_spsm.py",
        "{input}",
        "--csv-coo",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
}


OP_TEST_CONFIGS: dict[str, OperatorTestConfig] = {
    "gather": OperatorTestConfig("gather", PERFORMANCE_COMMANDS["gather"]),
    "scatter": OperatorTestConfig("scatter", PERFORMANCE_COMMANDS["scatter"]),
    "spmv_csr": OperatorTestConfig("spmv_csr", PERFORMANCE_COMMANDS["spmv_csr"]),
    "spmv_coo": OperatorTestConfig("spmv_coo", PERFORMANCE_COMMANDS["spmv_coo"]),
    "spmv_coo_tocsr": OperatorTestConfig(
        "spmv_coo_tocsr", PERFORMANCE_COMMANDS["spmv_coo_tocsr"]
    ),
    "spmm_csr": OperatorTestConfig("spmm_csr", PERFORMANCE_COMMANDS["spmm_csr"]),
    "spmm_coo": OperatorTestConfig("spmm_coo", PERFORMANCE_COMMANDS["spmm_coo"]),
    "spmm_csr_opt": OperatorTestConfig(
        "spmm_csr_opt", PERFORMANCE_COMMANDS["spmm_csr_opt"]
    ),
    "spmm_csr_opt_alg1": OperatorTestConfig(
        "spmm_csr_opt_alg1", PERFORMANCE_COMMANDS["spmm_csr_opt_alg1"]
    ),
    "spmm_csr_opt_alg2": OperatorTestConfig(
        "spmm_csr_opt_alg2", PERFORMANCE_COMMANDS["spmm_csr_opt_alg2"]
    ),
    "alpha_spmm_alg1": OperatorTestConfig(
        "alpha_spmm_alg1", PERFORMANCE_COMMANDS["alpha_spmm_alg1"]
    ),
    "spgemm_csr": OperatorTestConfig("spgemm_csr", PERFORMANCE_COMMANDS["spgemm_csr"]),
    "sddmm_csr": OperatorTestConfig("sddmm_csr", PERFORMANCE_COMMANDS["sddmm_csr"]),
    "spsv_csr": OperatorTestConfig("spsv_csr", PERFORMANCE_COMMANDS["spsv_csr"]),
    "spsv_coo": OperatorTestConfig("spsv_coo", PERFORMANCE_COMMANDS["spsv_coo"]),
    "spsm_csr": OperatorTestConfig("spsm_csr", PERFORMANCE_COMMANDS["spsm_csr"]),
    "spsm_coo": OperatorTestConfig("spsm_coo", PERFORMANCE_COMMANDS["spsm_coo"]),
}


def now_ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_operator_catalog(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        return _parse_operator_catalog_fallback(text)
    data = yaml.safe_load(text) or {}
    ops = data.get("ops", [])
    if not isinstance(ops, list):
        raise ValueError(f"{path} must contain a top-level 'ops' list")
    return [op for op in ops if isinstance(op, dict) and op.get("id")]


def _parse_operator_catalog_fallback(text: str) -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    in_stages = False
    for line in text.splitlines():
        match = re.match(r"^  - id:\s*([A-Za-z0-9_]+)\s*$", line)
        if match:
            if current is not None:
                catalog.append(current)
            current = {"id": match.group(1), "stages": []}
            in_stages = False
            continue
        if current is None:
            continue
        if re.match(r"^    stages:\s*$", line):
            in_stages = True
            continue
        if re.match(r"^    [A-Za-z_][A-Za-z0-9_-]*:\s*", line):
            in_stages = False
            continue
        if in_stages:
            stage_match = re.match(r"^      - ([A-Za-z0-9_-]+):", line)
            if stage_match:
                current.setdefault("stages", [])
                current["stages"].append({stage_match.group(1): ""})
    if current is not None:
        catalog.append(current)
    return catalog


def _stage_name(op: dict[str, object]) -> str | None:
    stages = op.get("stages", [])
    if not isinstance(stages, list) or not stages:
        return None
    latest = stages[-1]
    if isinstance(latest, dict) and latest:
        return str(next(iter(latest.keys())))
    return None


def read_ops(
    *,
    project_root: Path,
    operators_yaml: str,
    op_list: str | None,
    ops_arg: str | None,
    stages_arg: str,
    start: str | None,
) -> list[str]:
    if ops_arg:
        return [op.strip().lstrip("_") for op in ops_arg.split(",") if op.strip()]

    if op_list:
        with open(op_list, encoding="utf-8") as handle:
            return [
                line.strip().lstrip("_")
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]

    yaml_path = Path(operators_yaml)
    if not yaml_path.is_absolute():
        yaml_path = project_root / yaml_path
    catalog = load_operator_catalog(yaml_path)

    requested_stages = {
        item.strip() for item in stages_arg.split(",") if item.strip()
    } or {"all"}
    if "all" in requested_stages:
        requested_stages = {"alpha", "beta", "stable"}

    result = []
    for op in catalog:
        op_id = str(op["id"]).strip()
        if op_id in DEFAULT_EXCLUDED_OPS:
            continue
        if start and op_id < start:
            continue
        stage = _stage_name(op)
        if stage and stage not in requested_stages:
            continue
        result.append(op_id)
    return result


def parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid --gpus value: {value}") from exc
    if not gpus:
        raise SystemExit("no GPUs provided")
    return gpus


def parse_pytest_summary(text: str) -> dict[str, int]:
    clean = ANSI_RE.sub("", text)
    counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
    }
    for match in SUMMARY_RE.finditer(clean):
        key = match.group(2).lower()
        if key == "error":
            key = "errors"
        if key in counts:
            counts[key] = int(match.group(1))
    counts["total"] = (
        counts["passed"]
        + counts["failed"]
        + counts["skipped"]
        + counts["errors"]
        + counts["xfailed"]
        + counts["xpassed"]
    )
    return counts


def status_from_pytest_counts(counts: dict[str, int], returncode: int) -> str:
    has_summary = any(
        counts[key]
        for key in ("passed", "failed", "skipped", "errors", "xfailed", "xpassed")
    )
    if returncode == 5 and not has_summary:
        return "NO_TESTS"
    if returncode not in (0, 5) and not has_summary:
        return "CRASH"
    if counts["failed"] or counts["errors"] or returncode not in (0, 5):
        return "FAIL"
    if counts["passed"] or counts["xfailed"]:
        return "PASS"
    if counts["skipped"]:
        return "SKIP"
    return "NO_TESTS"


def _base_env(project_root: Path, gpu_id: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    pythonpath = [str(project_root / "src"), str(project_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def run_subprocess(
    cmd: list[str],
    *,
    project_root: Path,
    env: dict[str, str],
    timeout: int,
) -> tuple[int, str, float, bool]:
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout if timeout > 0 else None)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(proc)
        stdout, stderr = proc.communicate()
        returncode = TIMEOUT_RETURN_CODE
    duration = time.monotonic() - start
    return returncode, (stdout or "") + "\n" + (stderr or ""), duration, timed_out


def _not_configured(op: str, phase: str, reason: str) -> dict[str, object]:
    return {
        "operator": op,
        "phase": phase,
        "configured": False,
        "status": "NOT_CONFIGURED",
        "reason": reason,
        "returncode": None,
        "duration_sec": 0.0,
    }


def run_accuracy(
    *,
    project_root: Path,
    op: str,
    gpu_id: int,
    marker: str | None,
    mode: str,
    op_dir: Path,
    extra_pytest_args: list[str],
    timeout: int,
) -> dict[str, object]:
    if not marker:
        return _not_configured(op, "accuracy", "no pytest marker mapping")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/pytest",
        "-m",
        marker,
        "--mode",
        mode,
        "-vs",
        "-p",
        "no:cacheprovider",
        *extra_pytest_args,
    ]
    returncode, output, duration, timed_out = run_subprocess(
        cmd,
        project_root=project_root,
        env=_base_env(project_root, gpu_id),
        timeout=timeout,
    )
    log_path = op_dir / "accuracy.log"
    log_path.write_text(output, encoding="utf-8")

    counts = parse_pytest_summary(output)
    status = "TIMEOUT" if timed_out else status_from_pytest_counts(counts, returncode)
    return {
        "operator": op,
        "phase": "accuracy",
        "configured": True,
        "marker": marker,
        "status": status,
        "returncode": returncode,
        "duration_sec": duration,
        "command": cmd,
        "log_path": str(log_path),
        **counts,
    }


def _resolve_path(project_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path


def render_performance_command(
    template: tuple[str, ...],
    *,
    project_root: Path,
    op_dir: Path,
    benchmark_input: Path | None,
    warmup: int,
    iters: int,
    extra_args: list[str],
) -> tuple[list[str], Path]:
    csv_path = op_dir / "performance.csv"
    rendered = [sys.executable]
    for token in template:
        if token == "{input}":
            if benchmark_input is not None:
                rendered.append(str(benchmark_input))
            continue
        rendered.append(
            token.format(
                csv=str(csv_path),
                input=str(benchmark_input) if benchmark_input is not None else "",
                warmup=warmup,
                iters=iters,
            )
        )
    rendered.extend(extra_args)
    if not Path(rendered[1]).is_absolute():
        rendered[1] = str(project_root / rendered[1])
    return rendered, csv_path


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NONE", "NULL"}:
        return None
    if text.endswith("x"):
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def summarize_performance_csv(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary: dict[str, object] = {"data_path": str(path), "row_count": len(rows)}
    if not rows:
        return summary

    if all({"dtype", "shape", "speedup"} <= set(row) for row in rows):
        try:
            from benchmark.performance_utils import two_level_average_speedup

            summary["two_level_speedup"] = two_level_average_speedup(rows)
            summary["speedup"] = summary["two_level_speedup"].get("overall")
        except Exception as exc:
            summary["speedup_summary_error"] = str(exc)

    speedup_values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if "speedup" not in key.lower():
                continue
            number = _to_float(value)
            if number is not None:
                speedup_values.setdefault(key, []).append(number)

    if speedup_values:
        by_column = {
            key: statistics.mean(values) for key, values in speedup_values.items()
        }
        summary["speedup_by_column"] = by_column
        if "speedup" not in summary:
            preferred = [
                "speedup",
                "triton_speedup_vs_pytorch",
                "opt_speedup_vs_pytorch",
                "base_vs_alg2_speedup",
                "base_vs_alg1_speedup",
                "pytorch_speedup_total",
                "pytorch_speedup_solve",
            ]
            for key in preferred:
                if key in by_column:
                    summary["speedup"] = by_column[key]
                    break
            else:
                first_key = sorted(by_column)[0]
                summary["speedup"] = by_column[first_key]
    return summary


def run_performance(
    *,
    project_root: Path,
    op: str,
    gpu_id: int,
    template: tuple[str, ...] | None,
    op_dir: Path,
    benchmark_input: Path | None,
    warmup: int,
    iters: int,
    extra_args: list[str],
    timeout: int,
) -> dict[str, object]:
    if not template:
        return _not_configured(op, "performance", "no performance command mapping")

    cmd, csv_path = render_performance_command(
        template,
        project_root=project_root,
        op_dir=op_dir,
        benchmark_input=benchmark_input,
        warmup=warmup,
        iters=iters,
        extra_args=extra_args,
    )
    returncode, output, duration, timed_out = run_subprocess(
        cmd,
        project_root=project_root,
        env=_base_env(project_root, gpu_id),
        timeout=timeout,
    )
    log_path = op_dir / "performance.log"
    log_path.write_text(output, encoding="utf-8")

    if timed_out:
        status = "TIMEOUT"
    elif returncode != 0:
        status = "FAIL"
    elif "CUDA is not available" in output:
        status = "SKIP"
    elif not csv_path.exists():
        status = "NO_TESTS"
    else:
        status = "PASS"

    result: dict[str, object] = {
        "operator": op,
        "phase": "performance",
        "configured": True,
        "status": status,
        "returncode": returncode,
        "duration_sec": duration,
        "command": cmd,
        "log_path": str(log_path),
        "data_path": str(csv_path) if csv_path.exists() else None,
    }
    if csv_path.exists():
        try:
            result.update(summarize_performance_csv(csv_path))
        except Exception as exc:
            result["csv_parse_error"] = str(exc)
    return result


def requested_phases(phase_arg: str) -> tuple[str, ...]:
    if phase_arg == "both":
        return ("accuracy", "performance")
    return (phase_arg,)


def run_one_op(
    *,
    project_root: Path,
    op: str,
    gpu_id: int,
    phase_arg: str,
    mode: str,
    results_dir: Path,
    benchmark_input: Path | None,
    benchmark_warmup: int,
    benchmark_iters: int,
    timeout: int,
    extra_pytest_args: list[str],
    extra_benchmark_args: list[str],
) -> dict[str, object]:
    op_dir = results_dir / op
    ensure_dir(op_dir)
    config = OP_TEST_CONFIGS.get(op, OperatorTestConfig())
    result: dict[str, object] = {"operator": op, "gpu": gpu_id}

    for phase in requested_phases(phase_arg):
        if phase == "accuracy":
            result["accuracy"] = run_accuracy(
                project_root=project_root,
                op=op,
                gpu_id=gpu_id,
                marker=config.accuracy_marker,
                mode=mode,
                op_dir=op_dir,
                extra_pytest_args=extra_pytest_args,
                timeout=timeout,
            )
        elif phase == "performance":
            result["performance"] = run_performance(
                project_root=project_root,
                op=op,
                gpu_id=gpu_id,
                template=config.performance_cmd,
                op_dir=op_dir,
                benchmark_input=benchmark_input,
                warmup=benchmark_warmup,
                iters=benchmark_iters,
                extra_args=extra_benchmark_args,
                timeout=timeout,
            )
    return result


def run_gpu_ops(
    *,
    project_root: Path,
    gpu_id: int,
    ops: list[str],
    phase_arg: str,
    mode: str,
    results_dir: Path,
    benchmark_input: Path | None,
    benchmark_warmup: int,
    benchmark_iters: int,
    timeout: int,
    extra_pytest_args: list[str],
    extra_benchmark_args: list[str],
    results: list[dict[str, object]],
) -> None:
    for op in ops:
        result = run_one_op(
            project_root=project_root,
            op=op,
            gpu_id=gpu_id,
            phase_arg=phase_arg,
            mode=mode,
            results_dir=results_dir,
            benchmark_input=benchmark_input,
            benchmark_warmup=benchmark_warmup,
            benchmark_iters=benchmark_iters,
            timeout=timeout,
            extra_pytest_args=extra_pytest_args,
            extra_benchmark_args=extra_benchmark_args,
        )
        with SUMMARY_LOCK:
            results.append(result)
            write_summary(results, results_dir)
        parts = []
        for phase in requested_phases(phase_arg):
            phase_result = result.get(phase, {})
            parts.append(f"{phase}={phase_result.get('status', 'MISSING')}")
        print(f"[GPU {gpu_id}] {op}: " + " ".join(parts), flush=True)


def _phase_rows(results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for result in sorted(results, key=lambda item: str(item["operator"])):
        for phase in ("accuracy", "performance"):
            phase_result = result.get(phase)
            if not isinstance(phase_result, dict):
                continue
            rows.append(
                {
                    "operator": result.get("operator"),
                    "gpu": result.get("gpu"),
                    "phase": phase,
                    "status": phase_result.get("status"),
                    "configured": phase_result.get("configured"),
                    "passed": phase_result.get("passed", ""),
                    "failed": phase_result.get("failed", ""),
                    "skipped": phase_result.get("skipped", ""),
                    "errors": phase_result.get("errors", ""),
                    "total": phase_result.get("total", ""),
                    "returncode": phase_result.get("returncode", ""),
                    "duration_sec": phase_result.get("duration_sec", ""),
                    "row_count": phase_result.get("row_count", ""),
                    "speedup": phase_result.get("speedup", ""),
                    "log_path": phase_result.get("log_path", ""),
                    "data_path": phase_result.get("data_path", ""),
                    "reason": phase_result.get("reason", ""),
                    "command": shlex.join(phase_result.get("command", []))
                    if phase_result.get("command")
                    else "",
                }
            )
    return rows


def _totals(rows: list[dict[str, object]]) -> dict[str, object]:
    by_status: dict[str, int] = {}
    by_phase: dict[str, dict[str, int]] = {}
    for row in rows:
        status = str(row.get("status") or "UNKNOWN")
        phase = str(row.get("phase") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        by_phase.setdefault(phase, {})
        by_phase[phase][status] = by_phase[phase].get(status, 0) + 1
    return {"by_status": by_status, "by_phase": by_phase}


def write_summary(results: list[dict[str, object]], results_dir: Path) -> None:
    ordered = sorted(results, key=lambda item: str(item["operator"]))
    rows = _phase_rows(ordered)
    json_path = results_dir / "summary.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "totals": _totals(rows),
                "results": ordered,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = results_dir / "summary.csv"
    headers = [
        "operator",
        "gpu",
        "phase",
        "status",
        "configured",
        "passed",
        "failed",
        "skipped",
        "errors",
        "total",
        "returncode",
        "duration_sec",
        "row_count",
        "speedup",
        "log_path",
        "data_path",
        "reason",
        "command",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})

    if Workbook is None:
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(headers)
    for row in rows:
        ws.append([row.get(key, "") for key in headers])
    wb.save(results_dir / "summary.xlsx")


def _should_fail(results: list[dict[str, object]], strict: bool) -> bool:
    failing_statuses = {"FAIL", "CRASH", "TIMEOUT", "NO_TESTS"}
    if strict:
        failing_statuses.add("NOT_CONFIGURED")
    for row in _phase_rows(results):
        if row.get("status") in failing_statuses:
            return True
    return False


def _print_ops(ops: list[str]) -> None:
    for op in ops:
        config = OP_TEST_CONFIGS.get(op, OperatorTestConfig())
        accuracy = config.accuracy_marker or "-"
        performance = "yes" if config.performance_cmd else "-"
        print(f"{op}\taccuracy={accuracy}\tperformance={performance}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operators-yaml", default="conf/operators.yaml")
    parser.add_argument("--op-list", default=None, help="File with one operator id per line.")
    parser.add_argument("--ops", default=None, help="Comma-separated operator ids; overrides YAML/list files.")
    parser.add_argument("--stages", default="all", help="Comma-separated stages from operators.yaml, or all.")
    parser.add_argument("--start", default=None, help="Start from this operator id when reading YAML.")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids for CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--mode", default="quick", choices=("quick", "normal"))
    parser.add_argument(
        "--phase",
        default="both",
        choices=("accuracy", "performance", "both"),
        help="Which phase to run for each operator.",
    )
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--pytest-args",
        default="",
        help="Extra pytest args appended to every accuracy invocation.",
    )
    parser.add_argument(
        "--benchmark-input",
        default="tests/data",
        help="Matrix file or directory passed to performance scripts that need input.",
    )
    parser.add_argument(
        "--benchmark-args",
        default="",
        help="Extra args appended to every performance invocation.",
    )
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Per-phase timeout in seconds; 0 disables timeout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat NOT_CONFIGURED operators as failures.",
    )
    parser.add_argument(
        "--list-ops",
        action="store_true",
        help="Print resolved operators and configured phases, then exit.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    ops = read_ops(
        project_root=project_root,
        operators_yaml=args.operators_yaml,
        op_list=args.op_list,
        ops_arg=args.ops,
        stages_arg=args.stages,
        start=args.start,
    )
    if not ops:
        raise SystemExit("no operators to run")
    if args.list_ops:
        _print_ops(ops)
        return 0

    gpus = parse_gpus(args.gpus)
    results_dir = (
        Path(args.results_dir).resolve()
        if args.results_dir
        else project_root / f"pytest_results_{now_ts()}"
    )
    ensure_dir(results_dir)

    benchmark_input = _resolve_path(project_root, args.benchmark_input)
    extra_pytest_args = shlex.split(args.pytest_args) if args.pytest_args else []
    extra_benchmark_args = (
        shlex.split(args.benchmark_args) if args.benchmark_args else []
    )

    tasks = {gpu: [] for gpu in gpus}
    for index, op in enumerate(ops):
        tasks[gpus[index % len(gpus)]].append(op)

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(
                run_gpu_ops,
                project_root=project_root,
                gpu_id=gpu,
                ops=gpu_ops,
                phase_arg=args.phase,
                mode=args.mode,
                results_dir=results_dir,
                benchmark_input=benchmark_input,
                benchmark_warmup=args.benchmark_warmup,
                benchmark_iters=args.benchmark_iters,
                timeout=args.timeout,
                extra_pytest_args=extra_pytest_args,
                extra_benchmark_args=extra_benchmark_args,
                results=results,
            )
            for gpu, gpu_ops in tasks.items()
            if gpu_ops
        ]
        for future in as_completed(futures):
            future.result()

    write_summary(results, results_dir)
    return 1 if _should_fail(results, args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
