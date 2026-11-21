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

"""Core performance cases run by FlagOS optest."""

import json
import os
import shlex
from pathlib import Path

import pytest

from run_flagsparse_pytest import (
    DELIVERY_BENCHMARK_ARGS,
    GENERIC_BENCHMARK_BACKENDS,
    PERFORMANCE_COMMANDS,
    _uses_generic_benchmark_script,
    run_performance,
)
from tools.delivery_variants import load_delivery_variants

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_MATRIX = PROJECT_ROOT / "tests" / "data" / "wave.mtx"
CORE_ARGS = {
    "gather": [
        "--cases",
        "32768:1024",
        "--value-dtypes",
        "float32",
        "--index-dtypes",
        "int32",
    ],
    "spmv_csr": ["--dtypes", "float32", "--index-dtypes", "int32", "--ops", "non"],
    "spmm_csr": ["--dtypes", "float32", "--index-dtypes", "int32", "--ops", "non"],
}

pytestmark = pytest.mark.performance
DELIVERY_DIR = os.environ.get("FLAGSPARSE_OPTEST_DELIVERY_DIR")


def _benchmark_cases():
    if DELIVERY_DIR:
        parents = dict.fromkeys(
            variant["operator"] for variant in load_delivery_variants()
        )
        return [pytest.param(op, marks=getattr(pytest.mark, op)) for op in parents]
    return [
        pytest.param("gather", marks=pytest.mark.gather),
        pytest.param("spmv_csr", marks=[pytest.mark.spmv_csr, pytest.mark.spmv_perf]),
        pytest.param("spmm_csr", marks=[pytest.mark.spmm_csr, pytest.mark.spmm_perf]),
    ]


@pytest.mark.parametrize("op", _benchmark_cases())
def test_core_performance(op, tmp_path, record_benchmark, monkeypatch):
    import run_flagsparse_pytest as runner
    from flagsparse.sparse_operations import _common as runtime

    if not runtime._accel_module().is_available():
        pytest.skip("An accelerator is required for FlagSparse performance tests")
    backend = runtime._backend_name()
    monkeypatch.setenv("FLAGSPARSE_BACKEND", backend)
    monkeypatch.setattr(
        runner,
        "_SCALED_BASELINE",
        json.loads(os.environ.get("FLAGSPARSE_OPTEST_SCALED_BASELINE", "null")),
    )
    if DELIVERY_DIR:
        benchmark_input = Path(os.environ["FLAGSPARSE_OPTEST_BENCHMARK_INPUT"])
        extra_args = (
            list(DELIVERY_BENCHMARK_ARGS.get(op, ()))
            if _uses_generic_benchmark_script(op)
            else []
        )
        if backend == "iluvatar":
            from run_flagsparse_delivery import PROFILES

            for setting in PROFILES[backend].op_benchmark_args:
                operator, _, arguments = setting.partition("=")
                if operator == op:
                    extra_args.extend(shlex.split(arguments))
        timeout = 0  # Optest enforces the deadline for the whole benchmark phase.
    else:
        benchmark_input = None if op == "gather" else CORE_MATRIX
        extra_args = CORE_ARGS[op] if backend in GENERIC_BENCHMARK_BACKENDS else []
        timeout = 300
    result = run_performance(
        project_root=PROJECT_ROOT,
        op=op,
        gpu_id=None,
        template=PERFORMANCE_COMMANDS[op],
        op_dir=tmp_path,
        benchmark_input=benchmark_input,
        warmup=5,
        iters=20,
        extra_args=extra_args,
        timeout=timeout,
    )
    if DELIVERY_DIR:
        (Path(DELIVERY_DIR) / f"{op}.json").write_text(
            json.dumps(result, default=str), encoding="utf-8"
        )
    status = result["status"].upper()
    if status in {"SKIP", "SKIPPED", "NOT_CONFIGURED"}:
        pytest.skip(result.get("reason") or f"{op} is unavailable on {backend}")
    if status not in {"PASS", "PASSED"} or result.get("csv_parse_error"):
        logs = "\n".join(
            path.read_text(errors="replace")[-3000:]
            for path in tmp_path.glob("performance_*.log")
        )
        pytest.fail(
            f"{op} benchmark returned {status}: {result.get('csv_parse_error', '')}\n{logs}"
        )

    raw_report = json.loads((tmp_path / "performance_result.json").read_text())
    details = raw_report[op]["details"]
    for group in details:
        for row in group["result"]:
            if variant := row.get("variant"):
                row["shape_detail"] += f"|variant={variant}"
            if row["speedup"] is None:
                group["native_baseline_skip_reason"] = (
                    "Some cases lack a local baseline"
                )
    assert any(
        row["latency"] is not None and row["latency"] > 0
        for group in details
        for row in group["result"]
    ), f"{op} recorded no measurements"
    record_benchmark(details)
