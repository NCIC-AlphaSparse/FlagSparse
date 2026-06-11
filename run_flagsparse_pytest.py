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
import html
import importlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
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
PYTEST_CASE_RE = re.compile(
    r"^(?P<nodeid>\S+::\S+)\s+"
    r"(?P<status>PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\b"
    r"(?P<message>.*)$"
)
SUMMARY_LOCK = threading.Lock()
TIMEOUT_RETURN_CODE = -100
DEFAULT_EXCLUDED_OPS = {
    "alpha_spmm_alg1",
    "spmv_coo_tocsr",
    "spsv_descriptor_api",
    "sparse_format_constructors",
}
STATUS_TO_FLAGGEMS = {
    "PASS": "Passed",
    "FAIL": "Failed",
    "SKIP": "Skipped",
    "TIMEOUT": "Timeout",
    "NO_TESTS": "NotFound",
    "CRASH": "Error",
    "NOT_CONFIGURED": "NotFound",
    "Passed": "Passed",
    "Failed": "Failed",
    "Skipped": "Skipped",
    "Timeout": "Timeout",
    "NotFound": "NotFound",
    "Error": "Error",
}
PYTEST_STATUS_TO_FLAGGEMS = {
    "PASSED": "Passed",
    "FAILED": "Failed",
    "SKIPPED": "Skipped",
    "ERROR": "Error",
    "XFAIL": "Skipped",
    "XPASS": "Passed",
}
PERFORMANCE_METRIC_COLUMNS = {
    "latency_base",
    "latency",
    "speedup",
    "triton_ms",
    "pytorch_ms",
    "cusparse_ms",
    "cupy_ms",
    "base_ms",
    "alg1_ms",
    "alg2_ms",
    "analysis_ms",
    "solve_ms",
    "total_ms",
    "max_abs_err",
    "max_rel_err",
}
PERFORMANCE_SPEEDUP_SCHEMAS = (
    ("speedup", "latency_base", "latency"),
    ("triton_speedup_vs_pytorch", "pytorch_ms", "triton_ms"),
    ("triton_speedup_vs_cusparse", "cusparse_ms", "triton_ms"),
    ("triton_speedup_vs_cupy", "cupy_ms", "triton_ms"),
    ("opt_speedup_vs_pytorch", "pytorch_ms", "opt_ms"),
    ("opt_speedup_vs_cusparse", "cusparse_ms", "opt_ms"),
    ("opt_vs_base", "base_ms", "opt_ms"),
    ("base_vs_alg2_speedup", "base_ms", "alg2_ms"),
    ("base_vs_alg1_speedup", "base_ms", "alg1_ms"),
    ("torch_vs_alg2_speedup", "torch_ms", "alg2_ms"),
    ("torch_vs_alg1_speedup", "torch_ms", "alg1_ms"),
    ("cusparse_vs_alg2_speedup", "cusparse_ms", "alg2_ms"),
    ("cusparse_vs_alg1_speedup", "cusparse_ms", "alg1_ms"),
    ("pytorch_speedup_solve", "pytorch_ms", "solve_ms"),
    ("cusparse_speedup_solve", "cusparse_ms", "solve_ms"),
    ("pytorch_speedup_total", "pytorch_ms", "triton_total_ms"),
    ("cusparse_speedup_total", "cusparse_ms", "triton_total_ms"),
)


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


def _capture_text(cmd: list[str], *, cwd: Path, timeout: int = 5) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _module_version(
    distribution: str, module_name: str | None = None
) -> dict[str, object]:
    info: dict[str, object] = {"installed": False, "version": None}
    try:
        info["version"] = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        pass
    except Exception as exc:
        info["version_error"] = str(exc)

    try:
        module = importlib.import_module(module_name or distribution)
    except Exception as exc:
        info["import_error"] = str(exc)
        return info

    info["installed"] = True
    info["version"] = getattr(module, "__version__", info.get("version"))
    return info


def collect_env_info(project_root: Path) -> dict[str, object]:
    """Collect lightweight run metadata for FlagGems-style summaries."""
    env: dict[str, object] = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": {
            "torch": _module_version("torch"),
            "triton": _module_version("triton"),
            "flagsparse": _module_version("flagsparse"),
        },
        "git": {
            "commit": _capture_text(["git", "rev-parse", "HEAD"], cwd=project_root),
            "branch": _capture_text(
                ["git", "branch", "--show-current"], cwd=project_root
            ),
            "status": _capture_text(["git", "status", "--short"], cwd=project_root),
        },
    }

    try:
        torch = importlib.import_module("torch")
    except Exception:
        return env

    cuda: dict[str, object] = {
        "available": bool(torch.cuda.is_available()),
        "version": getattr(torch.version, "cuda", None),
        "device_count": 0,
        "devices": [],
    }
    if cuda["available"]:
        try:
            cuda["device_count"] = torch.cuda.device_count()
            devices = []
            for index in range(int(cuda["device_count"])):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": props.name,
                        "capability": list(torch.cuda.get_device_capability(index)),
                        "total_memory": props.total_memory,
                    }
                )
            cuda["devices"] = devices
        except Exception as exc:
            cuda["error"] = str(exc)
    env["cuda"] = cuda
    return env


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


def _extract_pytest_failures(text: str) -> list[str]:
    clean = ANSI_RE.sub("", text)
    failures: list[str] = []
    for line in clean.splitlines():
        stripped = line.strip()
        if stripped.startswith(("FAILED ", "ERROR ")):
            failures.append(stripped)
    return failures


def _parse_test_parameters(nodeid: str) -> dict[str, str]:
    if "[" not in nodeid or not nodeid.endswith("]"):
        return {}
    param_text = nodeid.rsplit("[", 1)[1][:-1]
    if not param_text:
        return {}
    return {
        f"param_{index}": value for index, value in enumerate(param_text.split("-"))
    }


def parse_pytest_cases(text: str) -> list[dict[str, object]]:
    clean = ANSI_RE.sub("", text)
    cases: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for line in clean.splitlines():
        stripped = line.strip()
        match = PYTEST_CASE_RE.match(stripped)
        if not match:
            continue
        nodeid = match.group("nodeid")
        status_raw = match.group("status")
        key = (nodeid, status_raw)
        if key in seen:
            continue
        seen.add(key)
        file_part, _, test_name = nodeid.partition("::")
        cases.append(
            {
                "nodeid": nodeid,
                "file": file_part,
                "name": test_name,
                "status": PYTEST_STATUS_TO_FLAGGEMS.get(status_raw, status_raw),
                "status_raw": status_raw,
                "parameters": _parse_test_parameters(nodeid),
                "message": match.group("message").strip() or None,
            }
        )
    return cases


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


def _flaggems_status(status: object) -> str:
    return STATUS_TO_FLAGGEMS.get(str(status or ""), str(status or "Unknown"))


def _duration(phase_result: dict[str, object]) -> object:
    return phase_result.get("duration", phase_result.get("duration_sec"))


def _exit_code(phase_result: dict[str, object]) -> object:
    return phase_result.get("exit_code", phase_result.get("returncode"))


def _data_file(phase_result: dict[str, object]) -> object:
    return phase_result.get("data_file", phase_result.get("data_path"))


def _stdout_log_file(phase_result: dict[str, object]) -> object:
    return phase_result.get("stdout_log_path", phase_result.get("log_path"))


def _stderr_log_file(phase_result: dict[str, object]) -> object:
    return phase_result.get("stderr_log_path", "")


def _json_data_file(phase_result: dict[str, object]) -> object:
    return phase_result.get("data_file")


def _accuracy_details(phase_result: dict[str, object]) -> dict[str, object]:
    parsed_details = phase_result.get("details")
    if isinstance(parsed_details, dict):
        return parsed_details

    details: dict[str, object] = {}
    failures = phase_result.get("failures") or []
    if failures:
        details["failed"] = failures
    tests = phase_result.get("tests") or []
    if tests:
        details["tests"] = tests
    marker = phase_result.get("marker")
    if marker:
        details["marker"] = marker
    reason = phase_result.get("reason")
    if reason:
        details["reason"] = reason
    return details


def _performance_details(phase_result: dict[str, object]) -> dict[str, object]:
    details: dict[str, object] = {}
    for key in (
        "row_count",
        "speedup",
        "speedup_by_column",
        "two_level_speedup",
        "speedup_summary_error",
        "csv_parse_error",
    ):
        if key in phase_result:
            details[key] = phase_result[key]
    records = phase_result.get("records")
    if records:
        details["records"] = records
    reason = phase_result.get("reason")
    if reason:
        details["reason"] = reason
    return details


def _phase_summary(phase_result: dict[str, object]) -> dict[str, object]:
    phase = phase_result.get("phase")
    if phase == "performance":
        summary = {
            "row_count": phase_result.get("row_count", 0),
            "speedup": phase_result.get("speedup", 0),
        }
        for key in (
            "speedup_by_column",
            "two_level_speedup",
            "speedup_summary_error",
            "csv_parse_error",
        ):
            if key in phase_result:
                summary[key] = phase_result[key]
        return summary

    return {
        "passed": phase_result.get("passed", 0),
        "failed": phase_result.get("failed", 0),
        "skipped": phase_result.get("skipped", 0),
        "errors": phase_result.get("errors", 0),
        "xfailed": phase_result.get("xfailed", 0),
        "xpassed": phase_result.get("xpassed", 0),
        "total": phase_result.get("total", 0),
    }


def _flaggems_accuracy_result(phase_result: dict[str, object]) -> dict[str, object]:
    status = str(phase_result.get("status") or "UNKNOWN")
    result = {
        "total": phase_result.get("total", 0),
        "skipped": phase_result.get("skipped", 0),
        "failed": phase_result.get("failed", 0),
        "passed": phase_result.get("passed", 0),
        "details": _accuracy_details(phase_result),
        "status": _flaggems_status(status),
        "duration": _duration(phase_result),
        "exit_code": _exit_code(phase_result),
    }
    if phase_result.get("errors"):
        result["errors"] = phase_result.get("errors")
    if "data_file" in phase_result:
        result["data_file"] = _json_data_file(phase_result)
    return result


def _flaggems_performance_result(phase_result: dict[str, object]) -> dict[str, object]:
    status = str(phase_result.get("status") or "UNKNOWN")
    result = {
        "duration": _duration(phase_result),
        "exit_code": _exit_code(phase_result),
        "data": phase_result.get("data", {}),
        "status": _flaggems_status(status),
    }
    if "data_file" in phase_result:
        result["data_file"] = _json_data_file(phase_result)
    if phase_result.get("test_case") is not None:
        result["test_case"] = phase_result.get("test_case")
    if phase_result.get("reason"):
        result["reason"] = phase_result.get("reason")
    return result


def _flaggems_phase_result(phase_result: dict[str, object]) -> dict[str, object]:
    if phase_result.get("phase") == "performance":
        return _flaggems_performance_result(phase_result)
    return _flaggems_accuracy_result(phase_result)


def parse_accuracy_json(path: Path) -> dict[str, object]:
    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "total": 0,
            "skipped": 0,
            "failed": 0,
            "passed": 0,
            "status": "Error",
            "details": {"error": f"Invalid JSON in {path}: {exc}"},
            "errors": 1,
        }

    skipped: dict[str, list[str]] = {}
    failed: dict[str, list[str]] = {}
    passed = 0
    skipped_with_issue = False
    for test_case, item in raw_data.items():
        if not isinstance(item, dict):
            continue
        params = [test_case[: test_case.find("[")] if "[" in test_case else test_case]
        for key, value in (item.get("params") or {}).items():
            params.append(f"{key}={str(value).replace(' ', '')}")
        param_str = ":".join(params)
        result = str(item.get("result") or "").lower()
        if result == "passed":
            passed += 1
        elif result == "skipped":
            reason = str(item.get("reason") or "Unknown")
            if "Issue" in reason:
                skipped_with_issue = True
            skipped.setdefault(reason, []).append(param_str)
        else:
            reason = str(item.get("reason") or "Unknown")
            failed.setdefault(reason, []).append(param_str)

    skipped_count = sum(len(items) for items in skipped.values())
    failed_count = sum(len(items) for items in failed.values())
    total = passed + skipped_count + failed_count
    details: dict[str, object] = {}
    if failed:
        details["failed"] = failed
    if skipped:
        details["skipped"] = skipped

    if failed_count:
        status = "Failed"
    elif skipped_with_issue:
        status = "Failed"
    elif skipped_count:
        status = "Skipped"
    elif passed:
        status = "Passed"
    else:
        status = "NotFound"

    return {
        "total": total,
        "skipped": skipped_count,
        "failed": failed_count,
        "passed": passed,
        "details": details,
        "status": status,
    }


def _phase_detail_for_file(phase_result: dict[str, object]) -> dict[str, object]:
    status = str(phase_result.get("status") or "UNKNOWN")
    phase = phase_result.get("phase")
    result = {
        **_flaggems_phase_result(phase_result),
        "status_raw": status,
        "phase": phase,
        "configured": phase_result.get("configured"),
        "returncode": phase_result.get("returncode"),
        "duration_sec": phase_result.get("duration_sec"),
        "command": phase_result.get("command", []),
        "log_path": phase_result.get("log_path"),
        "stdout_log_path": phase_result.get("stdout_log_path"),
        "stderr_log_path": phase_result.get("stderr_log_path"),
        "data_path": phase_result.get("data_path"),
        "csv_file": phase_result.get("csv_file"),
        "reason": phase_result.get("reason"),
        "summary": _phase_summary(phase_result),
    }
    if phase == "accuracy":
        result["tests"] = phase_result.get("tests", [])
    if phase == "performance":
        result["benchmark"] = phase_result.get("benchmark", {})
        result["details"] = _performance_details(phase_result)
    return result


def write_phase_result(
    op_dir: Path, phase: str, phase_result: dict[str, object]
) -> Path:
    path = op_dir / f"{phase}_result.json"
    detail_path = op_dir / f"{phase}_detail.json"
    phase_result["result_path"] = str(path)
    if path.exists() and op_dir.name == phase_result.get("operator"):
        phase_result["data_file"] = str(path.relative_to(op_dir.parent))
    elif path.exists():
        phase_result["data_file"] = path.name
    if phase == "performance" and phase_result.get("csv_file") is None:
        phase_result["csv_file"] = phase_result.get("data_path")
    detail_path.write_text(
        json.dumps(_phase_detail_for_file(phase_result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


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
) -> tuple[int, str, str, float, bool]:
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
    return returncode, stdout or "", stderr or "", duration, timed_out


def _not_configured(op: str, phase: str, reason: str) -> dict[str, object]:
    return {
        "operator": op,
        "phase": phase,
        "configured": False,
        "status": "NOT_CONFIGURED",
        "reason": reason,
        "returncode": None,
        "exit_code": None,
        "duration_sec": 0.0,
        "duration": 0.0,
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

    result_path = op_dir / "accuracy_result.json"
    if result_path.exists():
        result_path.unlink()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/pytest",
        "-m",
        marker,
        "--mode",
        mode,
        "--record",
        "json",
        "--output",
        str(result_path),
        "-vs",
        "-p",
        "no:cacheprovider",
        *extra_pytest_args,
    ]
    returncode, stdout, stderr, duration, timed_out = run_subprocess(
        cmd,
        project_root=project_root,
        env=_base_env(project_root, gpu_id),
        timeout=timeout,
    )
    output = stdout + ("\n" if stdout and stderr else "") + stderr
    stdout_path = op_dir / "accuracy_stdout.log"
    stderr_path = op_dir / "accuracy_stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    counts = parse_pytest_summary(output)
    status = "TIMEOUT" if timed_out else status_from_pytest_counts(counts, returncode)
    failures = _extract_pytest_failures(output)
    cases = parse_pytest_cases(output)
    parsed: dict[str, object] = {}
    data_file: str | None = None
    if result_path.exists():
        parsed = parse_accuracy_json(result_path)
        data_file = str(result_path.relative_to(op_dir.parent))
        status = str(parsed.get("status") or status)
    else:
        parsed = {
            "passed": counts["passed"],
            "failed": counts["failed"],
            "skipped": counts["skipped"],
            "errors": counts["errors"],
            "total": counts["total"],
        }
    result = {
        "operator": op,
        "phase": "accuracy",
        "configured": True,
        "marker": marker,
        "status": status,
        "returncode": returncode,
        "exit_code": returncode,
        "duration_sec": duration,
        "duration": duration,
        "command": cmd,
        "stdout_log_path": str(stdout_path),
        "stderr_log_path": str(stderr_path),
        "log_path": str(stdout_path),
        "failures": failures,
        "tests": cases,
        **counts,
        **parsed,
    }
    if data_file is not None:
        result["data_file"] = data_file
    return result


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


def _row_dtype(row: dict[str, str]) -> str:
    return str(
        row.get("dtype")
        or row.get("value_dtype")
        or row.get("value_dtype_req")
        or row.get("value_dtype_compute")
        or row.get("data_dtype")
        or row.get("x_dtype")
        or row.get("B_dtype")
        or "unknown"
    )


def _row_shape(row: dict[str, str], index: int) -> str:
    for key in ("shape", "matrix", "name", "case", "m,n,k", "M,N,K"):
        value = row.get(key)
        if value:
            return str(value)
    dims = [row.get(key) for key in ("M", "N", "K") if row.get(key)]
    if dims:
        return "x".join(str(item) for item in dims)
    dims = [row.get(key) for key in ("rows", "cols", "nnz") if row.get(key)]
    if dims:
        return "x".join(str(item) for item in dims)
    return f"row_{index}"


def _detail_shape(row: dict[str, str], index: int, seen: set[str]) -> str:
    base = _row_shape(row, index)
    parts = [base]
    for key in (
        "index_dtype",
        "op",
        "opA",
        "transpose",
        "format",
        "dense_cols",
        "n_rhs",
        "mode",
        "case_id",
        "seed",
    ):
        value = row.get(key)
        if value:
            parts.append(f"{key}={value}")
    candidate = "|".join(parts)
    if candidate not in seen:
        seen.add(candidate)
        return candidate
    fallback = f"{candidate}|row={index}"
    seen.add(fallback)
    return fallback


def _is_metric_column(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in PERFORMANCE_METRIC_COLUMNS
        or lowered.endswith("_ms")
        or "speedup" in lowered
        or "latency" in lowered
        or "err" in lowered
    )


def _performance_metric_record(row: dict[str, str]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    metadata: dict[str, str] = {}
    for key, value in row.items():
        if value is None or value == "":
            continue
        number = _to_float(value)
        if number is not None and _is_metric_column(key):
            metrics[key] = number
        else:
            metadata[key] = value
    return {"metrics": metrics, "metadata": metadata, "raw": row}


def _performance_schema(row: dict[str, str]) -> tuple[str, str | None, str | None]:
    for speedup_key, base_key, latency_key in PERFORMANCE_SPEEDUP_SCHEMAS:
        if row.get(speedup_key):
            return speedup_key, base_key, latency_key
    for key in row:
        if "speedup" in key.lower():
            return key, None, None
    return "speedup", None, None


def _benchmark_json_detail(row: dict[str, str], index: int) -> dict[str, object]:
    speedup_key, base_key, latency_key = _performance_schema(row)
    base = _to_float(row.get(base_key)) if base_key else None
    latency = _to_float(row.get(latency_key)) if latency_key else None
    speedup = _to_float(row.get(speedup_key))
    result = {
        "shape_detail": _row_shape(row, index),
        "latency_base": 0.0 if base is None else base,
        "latency": 0.0 if latency is None else latency,
        "speedup": 0.0 if speedup is None else speedup,
    }
    for key, value in row.items():
        if key in result or value in (None, ""):
            continue
        number = _to_float(value)
        result[key] = number if number is not None else value
    return result


def benchmark_json_from_csv(
    op: str, csv_path: Path, *, status: str = "passed", test_case: str = "csv"
) -> dict[str, object]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_dtype: dict[str, list[dict[str, object]]] = {}
    for index, row in enumerate(rows):
        by_dtype.setdefault(_row_dtype(row), []).append(
            _benchmark_json_detail(row, index)
        )

    details = [
        {"dtype": dtype, "result": dtype_rows}
        for dtype, dtype_rows in sorted(by_dtype.items())
    ]
    return {op: {"result": status, "test_case": test_case, "details": details}}


def write_benchmark_json_from_csv(op: str, csv_path: Path, json_path: Path) -> None:
    json_path.write_text(
        json.dumps(benchmark_json_from_csv(op, csv_path), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _flaggems_perf_data(rows: list[dict[str, str]]) -> dict[str, object]:
    grouped: dict[str, dict[str, object]] = {}
    totals: dict[str, list[float]] = {}
    seen_by_dtype: dict[str, set[str]] = {}
    for index, row in enumerate(rows):
        dtype = _row_dtype(row)
        shape = _detail_shape(row, index, seen_by_dtype.setdefault(dtype, set()))
        speedup_key, base_key, latency_key = _performance_schema(row)
        speedup = _to_float(row.get(speedup_key))
        base = _to_float(row.get(base_key)) if base_key else None
        latency = _to_float(row.get(latency_key)) if latency_key else None

        dtype_entry = grouped.setdefault(
            dtype, {"result": "OK", "details": {}, "speedup": 0}
        )
        detail: dict[str, object] = {
            "base": base,
            "gems": latency,
            "speedup": speedup,
        }
        for key in ("status", "matrix_status"):
            if row.get(key):
                detail[key] = row[key]
                if row[key].upper() in {"FAIL", "ERROR"}:
                    dtype_entry["result"] = "Failed"
        dtype_entry["details"][shape] = detail  # type: ignore[index]
        if speedup is not None:
            totals.setdefault(dtype, []).append(speedup)

    for dtype, values in totals.items():
        if values:
            grouped[dtype]["speedup"] = statistics.mean(values)
    return grouped


def parse_performance_json(op: str, path: Path) -> dict[str, object]:
    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "Error",
            "reason": f"Invalid JSON in {path}: {exc}",
            "data": {},
        }

    data = raw_data.get(op, {})
    if not data:
        return {"status": "NotFound", "data": {}}

    result = str(data.get("result", "NotFound"))
    if result.lower() in {"failed", "skipped"}:
        return {
            "status": result.title(),
            "reason": data.get("reason", "Unknown"),
            "test_case": data.get("test_case", "Unknown"),
            "data": {},
        }

    bench_res: dict[str, object] = {}
    for item in data.get("details", []):
        dtype = str(item.get("dtype", "unknown"))
        details: dict[str, object] = {}
        total = 0.0
        count = 0
        for res in item.get("result", []):
            shape = str(res.get("shape_detail", "Unknown")).replace(" ", "")
            speedup = _to_float(res.get("speedup")) or 0.0
            details[shape] = {
                "base": _to_float(res.get("latency_base")) or 0.0,
                "gems": _to_float(res.get("latency")) or 0.0,
                "speedup": speedup,
            }
            total += speedup
            count += 1
        bench_res[dtype] = {
            "result": "OK" if details else "Unknown",
            "details": details,
            "speedup": total / count if count else 0,
        }

    return {
        "status": result.title(),
        "data": bench_res,
        "test_case": data.get("test_case", "Unknown"),
    }


def performance_records_by_dtype_shape(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, list[dict[str, object]]]]:
    grouped: dict[str, dict[str, list[dict[str, object]]]] = {}
    for index, row in enumerate(rows):
        dtype = _row_dtype(row)
        shape = _row_shape(row, index)
        grouped.setdefault(dtype, {}).setdefault(shape, []).append(
            _performance_metric_record(row)
        )
    return grouped


def _performance_records_by_dtype_shape(
    rows: list[dict[str, str]],
) -> dict[str, object]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        dtype = _row_dtype(row)
        shape = _row_shape(row, len(grouped))
        grouped.setdefault(dtype, {}).setdefault(shape, []).append(row)
    return grouped


def summarize_performance_csv(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary: dict[str, object] = {
        "data_path": str(path),
        "csv_file": str(path),
        "row_count": len(rows),
        "records": rows,
        "records_by_dtype_shape": _performance_records_by_dtype_shape(rows),
        "benchmark": performance_records_by_dtype_shape(rows),
        "data": _flaggems_perf_data(rows),
        "test_case": "csv",
    }
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

    result_path = op_dir / "performance_result.json"
    if result_path.exists():
        result_path.unlink()
    cmd, csv_path = render_performance_command(
        template,
        project_root=project_root,
        op_dir=op_dir,
        benchmark_input=benchmark_input,
        warmup=warmup,
        iters=iters,
        extra_args=extra_args,
    )
    returncode, stdout, stderr, duration, timed_out = run_subprocess(
        cmd,
        project_root=project_root,
        env=_base_env(project_root, gpu_id),
        timeout=timeout,
    )
    output = stdout + ("\n" if stdout and stderr else "") + stderr
    stdout_path = op_dir / "performance_stdout.log"
    stderr_path = op_dir / "performance_stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

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
        "exit_code": returncode,
        "duration_sec": duration,
        "duration": duration,
        "command": cmd,
        "stdout_log_path": str(stdout_path),
        "stderr_log_path": str(stderr_path),
        "log_path": str(stdout_path),
        "data_path": str(csv_path) if csv_path.exists() else None,
    }
    if csv_path.exists():
        try:
            write_benchmark_json_from_csv(op, csv_path, result_path)
            result.update(summarize_performance_csv(csv_path))
            result.update(parse_performance_json(op, result_path))
            result["data_file"] = str(result_path.relative_to(op_dir.parent))
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
            write_phase_result(op_dir, "accuracy", result["accuracy"])
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
            write_phase_result(op_dir, "performance", result["performance"])
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
    env_info: dict[str, object],
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
            write_summary(results, results_dir, env_info)
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
                    "exit_code": _exit_code(phase_result),
                    "returncode": phase_result.get("returncode", ""),
                    "duration": _duration(phase_result),
                    "duration_sec": phase_result.get("duration_sec", ""),
                    "row_count": phase_result.get("row_count", ""),
                    "speedup": phase_result.get("speedup", ""),
                    "data_file": _data_file(phase_result),
                    "log_path": phase_result.get("log_path", ""),
                    "stdout_log_path": phase_result.get("stdout_log_path", ""),
                    "stderr_log_path": phase_result.get("stderr_log_path", ""),
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


def _operator_summary(result: dict[str, object]) -> dict[str, object]:
    entry: dict[str, object] = {"customized": False}
    for phase in ("accuracy", "performance"):
        phase_result = result.get(phase)
        if not isinstance(phase_result, dict):
            continue
        entry[phase] = _flaggems_phase_result(phase_result)
    return entry


def _flaggems_summary(
    *,
    results: list[dict[str, object]],
    rows: list[dict[str, object]],
    env_info: dict[str, object],
) -> dict[str, object]:
    generated_at = _dt.datetime.now().isoformat(timespec="seconds")
    ordered = sorted(results, key=lambda item: str(item["operator"]))
    return {
        "timestamp": generated_at,
        "env": env_info,
        "result": {
            str(result["operator"]): _operator_summary(result) for result in ordered
        },
    }


def _html_text(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _read_text_for_html(path_value: object) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Unable to read {path}: {exc}"


def _format_html_json(value: object) -> str:
    try:
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    except TypeError:
        return str(value)


FLAGGEMS_HTML_STYLE = """<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 20px;
    color: #333;
}
h3 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 6px;
}
h4 {
    color: #34495e;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 20px;
    font-size: 14px;
}
thead th {
    background-color: #3498db;
    color: #fff;
    padding: 8px 10px;
    text-align: left;
}
tbody td {
    padding: 6px 10px;
    border-bottom: 1px solid #ddd;
}
tbody tr:nth-child(even) {
    background-color: #f9f9f9;
}
tbody tr:hover {
    background-color: #eaf2f8;
}
tt {
    background-color: #ecf0f1;
    padding: 2px 4px;
    border-radius: 3px;
}
blockquote {
    margin: 4px 0;
    padding: 4px 8px;
    border-left: 3px solid #e74c3c;
    background-color: #fdf2f2;
}
details summary {
    cursor: pointer;
    color: #2980b9;
    text-decoration: underline;
}
details summary:hover {
    color: #1a5276;
}
pre.log-content {
    max-height: 400px;
    overflow: auto;
    background-color: #2d2d2d;
    color: #f8f8f2;
    padding: 10px;
    border-radius: 0 4px 4px 4px;
    font-size: 12px;
    white-space: pre-wrap;
    word-wrap: break-word;
    margin-top: 0;
}
.log-tabs {
    margin-top: 6px;
}
.tab-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 2px;
    border-bottom: 2px solid #3498db;
}
.tab-btn {
    padding: 4px 10px;
    border: 1px solid #ccc;
    border-bottom: none;
    background: #ecf0f1;
    cursor: pointer;
    font-size: 12px;
    border-radius: 4px 4px 0 0;
}
.tab-btn.active {
    background: #3498db;
    color: #fff;
    border-color: #3498db;
}
.tab-btn:hover:not(.active) {
    background: #d5e8f7;
}
.tab-panel {
    border: 1px solid #ccc;
    border-top: none;
}
.table-stats {
    padding: 8px 10px;
    margin-bottom: 8px;
    background-color: #eaf2f8;
    border-radius: 4px;
    font-size: 14px;
    color: #2c3e50;
}
.table-stats span {
    font-weight: bold;
}
.table-wrapper {
    overflow-x: auto;
    overflow-y: auto;
    max-height: 85vh;
    position: relative;
}
thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    background-color: #3498db;
}
.sticky-col {
    position: sticky;
    background-color: inherit;
    z-index: 2;
}
.sticky-col-0 {
    left: 0;
    min-width: 40px;
}
.sticky-col-1 {
    left: 40px;
    min-width: 120px;
    border-right: 2px solid #bdc3c7;
}
thead .sticky-col {
    z-index: 4;
}
thead .sticky-col-0 {
    background-color: #3498db;
}
thead .sticky-col-1 {
    background-color: #3498db;
}
tbody tr .sticky-col {
    background-color: #fff;
}
tbody tr:nth-child(even) .sticky-col {
    background-color: #f9f9f9;
}
tbody tr:hover .sticky-col {
    background-color: #eaf2f8;
}
.filter-select {
    width: 100%;
    margin-top: 4px;
    font-size: 12px;
    padding: 2px;
}
.sort-btn {
    border: none;
    background: transparent;
    color: #fff;
    cursor: pointer;
    font-size: 14px;
    padding: 0 2px;
}
.sort-btn:hover {
    color: #f1c40f;
}
</style>"""


FLAGGEMS_HTML_SCRIPT = """<script>
function switchTab(containerId, idx) {
  var container = document.getElementById(containerId);
  var btns = container.querySelectorAll('.tab-btn');
  var panels = container.querySelectorAll('.tab-panel');
  for (var i = 0; i < btns.length; i++) {
    btns[i].classList.remove('active');
    panels[i].style.display = 'none';
  }
  btns[idx].classList.add('active');
  panels[idx].style.display = '';
}

var originalOrder = [];
(function() {
  var table = document.getElementById('main-table');
  if (!table) return;
  var rows = table.tBodies[0].rows;
  for (var i = 0; i < rows.length; i++) originalOrder.push(rows[i]);
})();

function getCellText(row, col) {
  var cell = row.cells[col];
  if (!cell) return '';
  var summary = cell.querySelector('summary');
  if (summary) return summary.textContent.trim();
  return cell.textContent.trim();
}

function filterTable() {
  var table = document.getElementById('main-table');
  if (!table) return;
  var selects = table.querySelectorAll('thead .filter-select');
  var accVal = selects[0] ? selects[0].value : '';
  var perfVal = selects[1] ? selects[1].value : '';
  var rows = table.tBodies[0].rows;
  var visible = 0;
  for (var i = 0; i < rows.length; i++) {
    var show = true;
    if (accVal && getCellText(rows[i], 2) !== accVal) show = false;
    if (perfVal && getCellText(rows[i], 4) !== perfVal) show = false;
    rows[i].style.display = show ? '' : 'none';
    if (show) visible++;
  }
  var el = document.getElementById('visible-count');
  if (el) el.textContent = visible;
}

function sortTable(col, dir) {
  var table = document.getElementById('main-table');
  if (!table) return;
  var tbody = table.tBodies[0];
  var rows = Array.from(tbody.rows);
  if (dir === 'reset') {
    originalOrder.forEach(function(r) { tbody.appendChild(r); });
    return;
  }
  rows.sort(function(a, b) {
    var va = parseFloat(getCellText(a, col)) || 0;
    var vb = parseFloat(getCellText(b, col)) || 0;
    return dir === 'asc' ? va - vb : vb - va;
  });
  rows.forEach(function(r) { tbody.appendChild(r); });
}
</script>"""


HTML_SPEEDUP_DTYPES = (
    ("fp16", ("float16", "torch.float16", "half")),
    ("fp32", ("float32", "torch.float32", "float")),
    ("bf16", ("bfloat16", "torch.bfloat16")),
    ("int16", ("int16", "torch.int16")),
    ("int32", ("int32", "torch.int32")),
    ("int8", ("int8", "torch.int8")),
    ("uint8", ("uint8", "torch.uint8")),
    ("int64", ("int64", "torch.int64", "long")),
    ("bool", ("bool", "torch.bool")),
    ("cf64", ("complex64", "torch.complex64", "cfloat", "cf64")),
    ("float8_e4m3fn", ("float8_e4m3fn", "torch.float8_e4m3fn")),
    ("float8_e5m2", ("float8_e5m2", "torch.float8_e5m2")),
)


def _phase_counts_text(phase_result: dict[str, object] | None) -> str:
    if not isinstance(phase_result, dict):
        return ""
    if phase_result.get("phase") == "performance":
        row_count = phase_result.get("row_count")
        speedup = phase_result.get("speedup")
        if row_count not in (None, ""):
            parts = [f"rows={row_count}"]
            if speedup not in (None, ""):
                parts.append(f"speedup={speedup}")
            return " / ".join(parts)
        return ""
    return (
        f"{phase_result.get('passed', 0)}/"
        f"{phase_result.get('failed', 0)}/"
        f"{phase_result.get('skipped', 0)}"
    )


def _phase_status_for_html(phase_result: dict[str, object] | None) -> str:
    if not isinstance(phase_result, dict):
        return "NotFound"
    return _flaggems_status(str(phase_result.get("status") or "UNKNOWN"))


def _phase_note_for_html(phase_result: dict[str, object] | None) -> str:
    if not isinstance(phase_result, dict):
        return ""
    reason = phase_result.get("reason")
    if reason:
        return str(reason)
    details = phase_result.get("details")
    if isinstance(details, dict):
        failed = details.get("failed")
        if failed:
            return _format_html_json(failed)
        error = details.get("error")
        if error:
            return str(error)
    return ""


def _html_status_filter() -> str:
    options = ["Failed", "NotFound", "Passed", "Skipped", "Timeout"]
    return (
        '<select class="filter-select" onchange="filterTable()"><option value="">All</option> '
        + " ".join(
            f'<option value="{_html_text(option)}">{_html_text(option)}</option>'
            for option in options
        )
        + "</select>"
    )


def _env_value(env_info: dict[str, object], *path: str) -> object:
    value: object = env_info
    for key in path:
        if not isinstance(value, dict):
            return ""
        value = value.get(key, "")
    return value


def _html_environment_rows(env_info: dict[str, object], generated_at: str) -> list[str]:
    platform_info = _env_value(env_info, "platform", "platform")
    os_info = platform_info or " ".join(
        str(part)
        for part in (
            _env_value(env_info, "platform", "system"),
            _env_value(env_info, "platform", "release"),
        )
        if part
    )
    cuda_info = _env_value(env_info, "cuda", "version")
    device = "cuda" if _env_value(env_info, "cuda", "available") else ""
    devices = _env_value(env_info, "cuda", "devices")
    if isinstance(devices, list) and devices:
        first_device = devices[0]
        if isinstance(first_device, dict) and first_device.get("name"):
            device = (
                f"{device}:{first_device['name']}" if device else first_device["name"]
            )
    rows = [
        ("Time", generated_at),
        ("Architecture", _env_value(env_info, "platform", "machine")),
        ("OS", os_info),
        ("Python", _env_value(env_info, "python", "version")),
        ("Torch", _env_value(env_info, "packages", "torch", "version")),
        ("Triton", _env_value(env_info, "packages", "triton", "version")),
        ("FlagSparse", _env_value(env_info, "packages", "flagsparse", "version")),
        ("CUDA", cuda_info),
        ("Device", device),
    ]
    return [
        f"<tr><td>{_html_text(key)}</td><td><tt>{_html_text(value)}</tt></td></tr>"
        for key, value in rows
        if value not in (None, "")
    ]


def _format_speedup_for_html(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return ""
    return f" {number:.3f}"


def _dtype_bucket(dtype: object) -> str | None:
    text = str(dtype or "").strip().lower()
    if not text:
        return None
    for display, aliases in HTML_SPEEDUP_DTYPES:
        if text in aliases or text.replace("torch.", "") in aliases:
            return display
    return None


def _performance_speedups_for_html(
    phase_result: dict[str, object] | None,
) -> tuple[float | None, dict[str, float]]:
    if not isinstance(phase_result, dict):
        return None, {}

    by_dtype: dict[str, list[float]] = {}
    data = phase_result.get("data")
    if isinstance(data, dict):
        for dtype, dtype_data in data.items():
            bucket = _dtype_bucket(dtype)
            if not bucket or not isinstance(dtype_data, dict):
                continue
            number = _to_float(dtype_data.get("speedup"))
            if number is not None:
                by_dtype.setdefault(bucket, []).append(number)

    if not by_dtype:
        records = phase_result.get("records")
        if isinstance(records, list):
            for row in records:
                if not isinstance(row, dict):
                    continue
                bucket = _dtype_bucket(_row_dtype(row))
                if not bucket:
                    continue
                speedup_key, _, _ = _performance_schema(row)
                number = _to_float(row.get(speedup_key))
                if number is not None:
                    by_dtype.setdefault(bucket, []).append(number)

    dtype_speedups = {
        dtype: statistics.mean(values) for dtype, values in by_dtype.items() if values
    }
    all_values = [value for values in by_dtype.values() for value in values]
    overall = (
        _to_float(phase_result.get("speedup"))
        if phase_result.get("speedup") not in (None, "")
        else None
    )
    if overall is None and all_values:
        overall = statistics.mean(all_values)
    return overall, dtype_speedups


def _html_tab_group(group_id: str, tabs: list[tuple[str, str]]) -> str:
    safe_tabs = [(name, content) for name, content in tabs if content is not None]
    if not safe_tabs:
        return "<em>No details</em>"
    buttons = []
    panels = []
    for index, (name, content) in enumerate(safe_tabs):
        active = " active" if index == 0 else ""
        display = "" if index == 0 else ' style="display:none"'
        buttons.append(
            f'<button class="tab-btn{active}" onclick="switchTab(\'{group_id}\', {index})">'
            f"{_html_text(name)}</button>"
        )
        panels.append(
            f'<div class="tab-panel"{display}><pre class="log-content">'
            f"{_html_text(content)}</pre></div>"
        )
    return (
        f'<div class="log-tabs" id="{_html_text(group_id)}">'
        f'<div class="tab-buttons">{"".join(buttons)}</div>'
        f"{''.join(panels)}</div>"
    )


def _phase_html_details(
    phase: str,
    op: str,
    phase_result: dict[str, object] | None,
    results_dir: Path,
) -> str:
    if not isinstance(phase_result, dict):
        return "<em>Not configured</em>"
    tabs: list[tuple[str, str]] = []
    result_path = phase_result.get("result_path")
    if result_path:
        tabs.append((Path(str(result_path)).name, _read_text_for_html(result_path)))
    elif phase_result.get("data_file"):
        tabs.append(
            (
                Path(str(phase_result["data_file"])).name,
                _format_html_json(_flaggems_phase_result(phase_result)),
            )
        )
    stdout_path = _stdout_log_file(phase_result)
    if stdout_path:
        tabs.append((Path(str(stdout_path)).name, _read_text_for_html(stdout_path)))
    if not tabs:
        tabs.append((f"{phase}.json", _format_html_json(phase_result)))
    phase_id = "acc" if phase == "accuracy" else "perf"
    return _html_tab_group(f"tabs-{phase_id}-{op}", tabs)


def write_result_html(
    results: list[dict[str, object]],
    results_dir: Path,
    env_info: dict[str, object],
) -> None:
    ordered = sorted(results, key=lambda item: str(item["operator"]))
    generated_at = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env_rows = _html_environment_rows(env_info, generated_at)
    table_stats_html = (
        f'<div class="table-stats">Total: <span id="total-count">{len(ordered)}</span> '
        f'| Showing: <span id="visible-count">{len(ordered)}</span></div>'
    )
    speedup_header_html = (
        "<th>OPAverageSpeedUp<br>"
        '<button class="sort-btn" onclick="sortTable(5, \'asc\')">&#9650;</button>'
        '<button class="sort-btn" onclick="sortTable(5, \'desc\')">&#9660;</button>'
        '<button class="sort-btn" onclick="sortTable(5, \'reset\')">&#8634;</button>'
        "</th>"
    )
    dtype_header_html = "".join(
        f"<th>{_html_text(dtype)}</th>" for dtype, _ in HTML_SPEEDUP_DTYPES
    )
    op_rows = []
    for index, result in enumerate(ordered, start=1):
        op = str(result["operator"])
        accuracy = result.get("accuracy")
        performance = result.get("performance")
        acc_status = _phase_status_for_html(accuracy)
        perf_status = _phase_status_for_html(performance)
        avg_speedup, dtype_speedups = _performance_speedups_for_html(performance)
        note = _phase_note_for_html(accuracy) or _phase_note_for_html(performance)
        dtype_cells = "".join(
            f"<td>{_html_text(_format_speedup_for_html(dtype_speedups.get(dtype)))}</td>"
            for dtype, _ in HTML_SPEEDUP_DTYPES
        )
        op_rows.append(
            "<tr>"
            f'<td class="sticky-col sticky-col-0">{index}</td>'
            f'<td class="sticky-col sticky-col-1">{_html_text(op)}</td>'
            f"<td><details><summary>{_html_text(acc_status)}</summary>"
            f"{_phase_html_details('accuracy', op, accuracy, results_dir)}</details></td>"
            f"<td>{_html_text(_phase_counts_text(accuracy))}</td>"
            f"<td><details><summary>{_html_text(perf_status)}</summary>"
            f"{_phase_html_details('performance', op, performance, results_dir)}</details></td>"
            f"<td>{_html_text(_format_speedup_for_html(avg_speedup))}</td>"
            f"{dtype_cells}"
            f"<td>{_html_text(note)}</td>"
            "</tr>"
        )

    html_text = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FlagSparse Test Report</title>
{FLAGGEMS_HTML_STYLE}
</head>
<body>
<h3>Test Environment</h3>
<table>
<thead><tr><th>Env</th><th>Setting</th></tr></thead>
<tbody>
{"".join(env_rows)}
</tbody></table>
<h3>Test Result</h3>
{table_stats_html}
<div class="table-wrapper">
<table id="main-table">
<thead>
<tr>
<th class="sticky-col sticky-col-0">No.</th>
<th class="sticky-col sticky-col-1">ID</th>
<th>AccRes<br>{_html_status_filter()}</th>
<th>AccStat(pass/fail/skip)</th>
<th>PerfRes<br>{_html_status_filter()}</th>
{speedup_header_html}
{dtype_header_html}
<th>Note</th>
</tr>
</thead>
<tbody>
{"".join(op_rows)}
</tbody></table>
</div>
{FLAGGEMS_HTML_SCRIPT}
</body>
</html>
"""
    (results_dir / "result.html").write_text(html_text, encoding="utf-8")


def _compat_summary(
    *,
    results: list[dict[str, object]],
    rows: list[dict[str, object]],
    env_info: dict[str, object],
) -> dict[str, object]:
    generated_at = _dt.datetime.now().isoformat(timespec="seconds")
    ordered = sorted(results, key=lambda item: str(item["operator"]))
    return {
        "generated_at": generated_at,
        "timestamp": generated_at,
        "env": env_info,
        "totals": _totals(rows),
        "results": ordered,
        "result": {
            str(result["operator"]): _operator_summary(result) for result in ordered
        },
    }


def write_summary(
    results: list[dict[str, object]],
    results_dir: Path,
    env_info: dict[str, object],
) -> None:
    ordered = sorted(results, key=lambda item: str(item["operator"]))
    rows = _phase_rows(ordered)
    json_path = results_dir / "summary.json"
    json_path.write_text(
        json.dumps(
            _flaggems_summary(results=ordered, rows=rows, env_info=env_info),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    compat_json_path = results_dir / "summary_flat.json"
    compat_json_path.write_text(
        json.dumps(
            _compat_summary(results=ordered, rows=rows, env_info=env_info),
            indent=2,
            sort_keys=True,
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
        "exit_code",
        "returncode",
        "duration",
        "duration_sec",
        "row_count",
        "speedup",
        "data_file",
        "log_path",
        "stdout_log_path",
        "stderr_log_path",
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
        write_result_html(ordered, results_dir, env_info)
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(headers)
    for row in rows:
        ws.append([row.get(key, "") for key in headers])
    wb.save(results_dir / "summary.xlsx")
    write_result_html(ordered, results_dir, env_info)


def _should_fail(results: list[dict[str, object]], strict: bool) -> bool:
    failing_statuses = {
        "FAIL",
        "CRASH",
        "TIMEOUT",
        "NO_TESTS",
        "Failed",
        "Error",
        "Timeout",
        "NotFound",
    }
    if strict:
        failing_statuses.update({"NOT_CONFIGURED", "NotConfigured"})
    for row in _phase_rows(results):
        if row.get("status") in failing_statuses:
            return True
    return False


def _print_ops(ops: list[str], phase_arg: str = "both") -> None:
    for op in ops:
        config = OP_TEST_CONFIGS.get(op, OperatorTestConfig())
        accuracy = config.accuracy_marker or "-"
        performance = "yes" if config.performance_cmd else "-"
        if phase_arg == "accuracy":
            print(f"{op}\taccuracy={accuracy}")
        elif phase_arg == "performance":
            print(f"{op}\tperformance={performance}")
        else:
            print(f"{op}\taccuracy={accuracy}\tperformance={performance}")


def main(
    default_phase: str = "both",
    expose_phase_arg: bool = True,
    description: str | None = None,
    include_accuracy_args: bool = True,
    include_performance_args: bool = True,
) -> int:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--operators-yaml", default="conf/operators.yaml")
    parser.add_argument(
        "--op-list", default=None, help="File with one operator id per line."
    )
    parser.add_argument(
        "--ops",
        default=None,
        help="Comma-separated operator ids; overrides YAML/list files.",
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages from operators.yaml, or all.",
    )
    parser.add_argument(
        "--start", default=None, help="Start from this operator id when reading YAML."
    )
    parser.add_argument(
        "--gpus", default="0", help="Comma-separated GPU ids for CUDA_VISIBLE_DEVICES."
    )
    if include_accuracy_args:
        parser.add_argument("--mode", default="quick", choices=("quick", "normal"))
    if expose_phase_arg:
        parser.add_argument(
            "--phase",
            default=default_phase,
            choices=("accuracy", "performance", "both"),
            help="Which phase to run for each operator.",
        )
    parser.add_argument("--results-dir", default=None)
    if include_accuracy_args:
        parser.add_argument(
            "--pytest-args",
            default="",
            help="Extra pytest args appended to every accuracy invocation.",
        )
    if include_performance_args:
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
    phase_arg = args.phase if expose_phase_arg else default_phase

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
        _print_ops(ops, phase_arg=phase_arg)
        return 0

    gpus = parse_gpus(args.gpus)
    results_dir = (
        Path(args.results_dir).resolve()
        if args.results_dir
        else project_root / f"pytest_results_{now_ts()}"
    )
    ensure_dir(results_dir)

    mode = args.mode if include_accuracy_args else "quick"
    benchmark_input = (
        _resolve_path(project_root, args.benchmark_input)
        if include_performance_args
        else None
    )
    benchmark_warmup = args.benchmark_warmup if include_performance_args else 5
    benchmark_iters = args.benchmark_iters if include_performance_args else 20
    extra_pytest_args = (
        shlex.split(args.pytest_args)
        if include_accuracy_args and args.pytest_args
        else []
    )
    extra_benchmark_args = (
        shlex.split(args.benchmark_args)
        if include_performance_args and args.benchmark_args
        else []
    )
    env_info = collect_env_info(project_root)

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
                phase_arg=phase_arg,
                mode=mode,
                results_dir=results_dir,
                benchmark_input=benchmark_input,
                benchmark_warmup=benchmark_warmup,
                benchmark_iters=benchmark_iters,
                timeout=args.timeout,
                extra_pytest_args=extra_pytest_args,
                extra_benchmark_args=extra_benchmark_args,
                env_info=env_info,
                results=results,
            )
            for gpu, gpu_ops in tasks.items()
            if gpu_ops
        ]
        for future in as_completed(futures):
            future.result()

    write_summary(results, results_dir, env_info)
    return 1 if _should_fail(results, args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
