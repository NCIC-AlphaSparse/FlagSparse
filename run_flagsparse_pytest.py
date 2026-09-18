#!/usr/bin/env python3

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

from tools.delivery_variants import load_delivery_variants

# XPU compiler diagnostics can include an MLIR reproducer and exceed the
# csv module's conservative 128 KiB default.  They are part of the execution
# result, so retaining them must not turn a TRITON_COMPILE row into PASS.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

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
    # spmm_csr now routes to the optimized (alg1) kernel by default, so the
    # standalone opt operators are dropped from the default CI run. They remain
    # configured and can still be run explicitly via --ops.
    "spmm_csr_opt",
    "spmm_csr_opt_alg1",
    "spmm_csr_opt_alg2",
}
STATUS_TO_FLAGGEMS = {
    "PASS": "Passed",
    "FAIL": "Failed",
    "SKIP": "Skipped",
    # Capability-probe verdicts. A kernel the backend declined is a skip; one
    # that would not compile is a failure -- neither may read as a pass.
    "REJECTED": "Skipped",
    "TRITON_COMPILE": "Failed",
    "MISMATCH": "Failed",
    "ERROR": "Error",
    "MIXED": "Error",
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
    # Kernel ran and passed, but no vendor baseline exists for this dtype.
    # Emitted by capi/tools/write_summary.py; kept here so the two tables match.
    "NoBaseline": "NoBaseline",
}
# Benchmark arguments that narrow each delivery parent's default CSV sweep to
# what the delivery variants actually name: int32 indices and the `non` op. The
# default sweeps also cover int64 and trans/conj (spmm_csr: 720 rows, 120 of
# them delivery), which on MetaX C550 ran past a 900 s --timeout. Injected only
# under --delivery-only, BEFORE --benchmark-args/--op-benchmark-args, so an
# explicit user flag still wins (argparse keeps the last value). Parents absent
# here already sweep only delivery axes (sddmm_csr, spgemm_csr, spsm_csr).
DELIVERY_BENCHMARK_ARGS: dict[str, tuple[str, ...]] = {
    "gather": (
        "--index-dtypes",
        "int32",
        "--value-dtypes",
        "float16,float32,float64,complex64,complex128",
    ),
    "scatter": ("--index-dtypes", "int32"),
    "spmv_csr": ("--index-dtype", "int32", "--ops", "non"),
    "spmv_coo": ("--index-dtypes", "int32", "--ops", "non"),
    "spmm_csr": ("--index-dtypes", "int32", "--ops", "non"),
    "spmm_coo": ("--index-dtypes", "int32"),
    "spsv_csr": ("--index-dtypes", "int32", "--ops", "NON"),
    "spsv_coo": ("--index-dtypes", "int32", "--ops", "NON"),
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
    "csc_ms",
    "bsr_ms",
    "base_ms",
    "alg1_ms",
    "alg2_ms",
    "analysis_ms",
    "solve_ms",
    "total_ms",
    "max_abs_err",
    "max_rel_err",
}
# VENDOR FIRST, PyTorch as the fallback. The runner reports the first non-empty
# match, so within every family the vendor pair is listed ahead of the PyTorch
# pair: on CUDA and ROCm the number is FlagSparse against cuSPARSE / hipSPARSE,
# and only an operator the vendor library does not implement falls through to
# the PyTorch metric. A backend with no vendor column at all -- MACA, MUSA, XPU --
# falls through for every operator, which is the same rule, not an exception.
#
# This ordering used to be the other way round for the `triton_*` family while a
# comment further down claimed the vendor metric won; it did not, and every CUDA
# speedup in the delivery report was against PyTorch.
#
# A speedup name can appear more than once with different column layouts --
# spmm_coo reports triton_speedup_vs_pytorch over torch_ms/ms while spmm_csc uses
# pytorch_ms/ms -- so each layout is listed and _performance_schema prefers the
# entry whose measurement columns the row actually carries.
PERFORMANCE_SPEEDUP_SCHEMAS = (
    ("speedup", "latency_base", "latency"),
    ("triton_speedup_vs_cusparse", "cusparse_ms", "triton_ms"),
    ("triton_speedup_vs_cupy", "cupy_ms", "triton_ms"),
    # spmm_coo carries BOTH triton_speedup_vs_pytorch (over torch_ms) and
    # cusparse_vs_alg_speedup (over cusparse_ms). The vendor one has to be ahead
    # of the PyTorch ones or that operator alone keeps reporting against torch
    # while everything else reports against cuSPARSE -- vendor-first holding for
    # ten operators and quietly not for the eleventh.
    ("cusparse_vs_alg2_speedup", "cusparse_ms", "alg2_ms"),
    ("cusparse_vs_alg1_speedup", "cusparse_ms", "alg1_ms"),
    ("cusparse_vs_alg_speedup", "cusparse_ms", "ms"),
    ("triton_speedup_vs_pytorch", "pytorch_ms", "triton_ms"),
    ("triton_speedup_vs_pytorch", "pytorch_ms", "ms"),
    ("triton_speedup_vs_pytorch", "torch_ms", "ms"),
    ("triton_speedup_vs_pytorch_api", "pytorch_api_ms", "triton_ms"),
    ("csc_speedup_vs_cusparse", "cusparse_ms", "csc_ms"),
    ("csc_speedup_vs_pytorch", "pytorch_ms", "csc_ms"),
    ("bsr_speedup_vs_cusparse", "cusparse_ms", "bsr_ms"),
    ("bsr_speedup_vs_pytorch", "pytorch_ms", "bsr_ms"),
    ("opt_speedup_vs_cusparse", "cusparse_ms", "opt_ms"),
    ("opt_speedup_vs_pytorch", "pytorch_ms", "opt_ms"),
    ("opt_vs_base", "base_ms", "opt_ms"),
    ("base_vs_alg2_speedup", "base_ms", "alg2_ms"),
    ("base_vs_alg1_speedup", "base_ms", "alg1_ms"),
    ("torch_vs_alg2_speedup", "torch_ms", "alg2_ms"),
    ("torch_vs_alg1_speedup", "torch_ms", "alg1_ms"),
    ("torch_vs_alg_speedup", "torch_ms", "ms"),
    ("cusparse_speedup_solve", "cusparse_ms", "solve_ms"),
    ("pytorch_speedup_solve", "pytorch_ms", "solve_ms"),
    ("cusparse_speedup_total", "cusparse_ms", "triton_total_ms"),
    ("pytorch_speedup_total", "pytorch_ms", "triton_total_ms"),
    ("scipy_vs_alg_speedup", "scipy_cpu_ms", "ms"),
    ("prepared_speedup_vs_pytorch", "pytorch_ms", "prepared_ms"),
    # SpSV and SpSM spell their columns in the vendor's own casing, and SpSV
    # names the CuPy route explicitly. Column lookup is case-insensitive (see
    # _row_value), so "cuSPARSE_ms" resolves against "cusparse_ms" here, but a
    # DIFFERENT name still needs its own entry -- which is why the two below
    # exist and why these operators used to report a speedup with both
    # latencies reading 0.00 ms.
    ("FlagSparse_vs_CuPy/cuSPARSE_speedup", "CuPy/cuSPARSE_ms", "flagsparse_ms"),
    ("FlagSparse_vs_cuSPARSE_speedup", "cusparse_ms", "flagsparse_ms"),
    # SpSV names its columns after the ACTIVE vendor label, so DCU writes
    # hipSPARSE_ms and FlagSparse_vs_hipSPARSE_all_speedup (the `_all` suffix is
    # ROCm-only: that path reports bufferSize/analysis/solve separately). Without
    # these two entries the DCU rows matched nothing and fell through to the
    # PyTorch metric -- vendor-first would have held on CUDA and quietly not on DCU.
    ("FlagSparse_vs_hipSPARSE_all_speedup", "hipsparse_ms", "flagsparse_ms"),
    ("FlagSparse_vs_hipSPARSE_speedup", "hipsparse_ms", "flagsparse_ms"),
    ("FlagSparse_vs_vendor_speedup", "cusparse_ms", "flagsparse_ms"),
    ("FlagSparse_vs_vendor_speedup", "hipsparse_ms", "flagsparse_ms"),
    # Backends whose vendor is torch.sparse, or which have none, write a neutral
    # `vendor_ms` rather than a column named after the PyTorch label or "N/A".
    ("FlagSparse_vs_vendor_speedup", "vendor_ms", "flagsparse_ms"),
    ("FlagSparse_vs_PyTorch_all_speedup", "pytorch_ms", "flagsparse_ms"),
    ("FlagSparse_vs_PyTorch_speedup", "pytorch_ms", "flagsparse_ms"),
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
    "spmv_csc": (
        "tests/test_spmv_csc.py",
        "{input}",
        "--csv-csc",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spmv_bsr": (
        "tests/test_spmv_bsr.py",
        "{input}",
        "--csv-bsr",
        "{csv}",
        "--ops",
        "non,trans,conj",
        "--alg",
        "compare",
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
    "spmm_bsr": (
        "tests/test_spmm_bsr.py",
        "{input}",
        "--csv-bsr",
        "{csv}",
        "--ops",
        "non,trans,conj",
        "--block-dims",
        "2",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spmm_bell": (
        "tests/test_spmm_bell.py",
        "{input}",
        "--csv-bell",
        "{csv}",
        "--ops",
        "non",
        "--block-dims",
        "2",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    ),
    "spmm_csc": (
        "tests/test_spmm_csc.py",
        "{input}",
        "--csv-csc",
        "{csv}",
        "--ops",
        "non,trans,conj",
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
    "spsv_sell": (
        "tests/test_spsv_sell.py",
        "{input}",
        "--csv",
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


# Ascend uses its own entry points instead of the CUDA-oriented per-operator
# scripts.  Selected only when FLAGSPARSE_BACKEND=ascend; every other backend keeps
# PERFORMANCE_COMMANDS untouched.
#
# Two of them, and the split is deliberate:
#
#   benchmark_ascend.py        measures against a torch-npu baseline, and only
#                              these five operators have one.  Where a baseline
#                              exists, a measured speedup beats a bare timing.
#   benchmark_ascend_probe.py  runs every other operator over the 20-matrix
#                              spread and classifies the outcome -- PASS,
#                              REJECTED, TRITON_COMPILE, MISMATCH, ERROR.  On a
#                              backend where kernels may not lower at all, "did
#                              it run, and if not why" is the measurement that
#                              matters, and the previous mapping simply had no
#                              entry for these operators: every one of them came
#                              back as "no performance command mapping", which
#                              reads the same as a pass.
# ---------------------------------------------------------------------------
# Backend registry, read from the library rather than restated here.
#
# The harness has to run on CUDA and on every domestic accelerator, and the list
# of those is the library's business, not the runner's.  Importing it means a
# platform added to flagsparse reaches the accuracy and performance phases
# without a second edit here -- and, more to the point, cannot silently disagree
# with what the operators themselves detect.
#
# The fallback list exists so the runner still works against a checkout whose
# library predates the registry; it is deliberately the same order.
# ---------------------------------------------------------------------------
def _load_backend_names() -> tuple[str, ...]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
        from flagsparse.sparse_operations._common import backend_specs

        return tuple(spec.name for spec in backend_specs())
    except Exception:
        return ("cuda", "rocm", "metax", "mthreads", "ascend", "xpu", "gcu", "mlu")


SUPPORTED_BACKENDS: tuple[str, ...] = _load_backend_names()

# Backends whose operators go through the generic CUDA-oriented per-operator
# benchmark scripts.  A backend absent from here needs its own entry below --
# and the point of listing them is that "not yet wired up" becomes visible
# instead of arriving as an empty result.
GENERIC_BENCHMARK_BACKENDS: tuple[str, ...] = ("cuda", "rocm", "metax", "mthreads")

# Backends with no per-operator benchmark of their own yet.  They fall back to
# the capability probe, which reports PASS / REJECTED / TRITON_COMPILE per
# operator -- on a platform where a kernel may not lower at all, that is the
# measurement that matters, and it beats reporting nothing.
PROBE_ONLY_BACKENDS: tuple[str, ...] = ("gcu", "mlu")

# The XPU SDK has no cuSPARSE-shaped generic sparse API, so these five run
# through benchmark_xpu.py instead of the probe.  The script compares each
# FlagSparse result with an equivalent PyTorch-XPU expression and reports both
# latencies and their ratio; it is not a vendor/XDNN sparse-library comparison.
XPU_BASELINE_OPS: tuple[str, ...] = (
    "gather",
    "scatter",
    "spmv_csr",
    "spmm_csr",
    "sddmm_csr",
)


ASCEND_BASELINE_OPS: tuple[str, ...] = (
    "gather",
    "scatter",
    "spmv_csr",
    "spmm_csr",
    "sddmm_csr",
)

# The operators whose accuracy runs through benchmark_ascend_accuracy.py instead
# of the pytest suite. Deliberately DERIVED from ASCEND_BASELINE_OPS rather than
# repeating the same five names: the two sets have to move together, and the drop
# this came from carried a second hardcoded copy that would drift the first time
# either list changed.
ASCEND_ACCURACY_OPS: frozenset[str] = frozenset(ASCEND_BASELINE_OPS)

ASCEND_PROBE_OPS: tuple[str, ...] = (
    "spmv_coo",
    "spmv_csc",
    "spmv_bsr",
    "spmv_coo_tocsr",
    "spmm_coo",
    "spmm_csc",
    "spmm_bsr",
    "spmm_bell",
    "spmm_csr_opt",
    "spmm_csr_opt_alg1",
    "spmm_csr_opt_alg2",
    "alpha_spmm_alg1",
    "spgemm_csr",
    "spsv_csr",
    "spsv_coo",
    "spsv_sell",
    "spsm_csr",
    "spsm_coo",
)

# Every operator, routed to the capability probe. Used by any backend that has
# no per-operator benchmark of its own; the probe itself is backend-neutral and
# resolves the device from the same registry.
PROBE_PERFORMANCE_COMMANDS: dict[str, tuple[str, ...]] = {}


def _xpu_baseline_command(op: str) -> tuple[str, ...]:
    """Measured baseline for the five operators PyTorch-XPU can express."""
    return (
        "benchmark/benchmark_xpu.py",
        "--op",
        op,
        "--device",
        "{device}",
        # XPU baselines are isolated one matrix per subprocess by
        # _run_bell_per_matrix(), so `{input}` is a file on this path.
        "--matrix",
        "{input}",
        "--csv-summary",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    )


def _probe_command(op: str) -> tuple[str, ...]:
    return (
        "benchmark/benchmark_ascend_probe.py",
        "--op",
        "{op}",
        "--device",
        "{device}",
        "--csv-summary",
        "{csv}",
        "--warmup",
        "{warmup}",
        "--iters",
        "{iters}",
    )


ASCEND_PERFORMANCE_COMMANDS: dict[str, tuple[str, ...]] = {
    **{
        op: (
            "benchmark/benchmark_ascend.py",
            "--op",
            "{op}",
            # The .mtx file or directory from --benchmark-input; without it the
            # script silently measured its built-in synthetic case instead.
            "--input",
            "{input}",
            "--device",
            "{device}",
            "--csv-summary",
            "{csv}",
            # bf16 is not a delivery dtype; measuring it only costs time.
            "--dtypes",
            "float16,float32,float64",
            "--warmup",
            "{warmup}",
            "--iters",
            "{iters}",
        )
        for op in ASCEND_BASELINE_OPS
    },
    **{
        op: (
            "benchmark/benchmark_ascend_probe.py",
            "--op",
            "{op}",
            "--device",
            "{device}",
            "--csv-summary",
            "{csv}",
            "--warmup",
            "{warmup}",
            "--iters",
            "{iters}",
        )
        for op in ASCEND_PROBE_OPS
    },
}


OP_TEST_CONFIGS: dict[str, OperatorTestConfig] = {
    "gather": OperatorTestConfig("gather", PERFORMANCE_COMMANDS["gather"]),
    "scatter": OperatorTestConfig("scatter", PERFORMANCE_COMMANDS["scatter"]),
    "spmv_csr": OperatorTestConfig("spmv_csr", PERFORMANCE_COMMANDS["spmv_csr"]),
    "spmv_coo": OperatorTestConfig("spmv_coo", PERFORMANCE_COMMANDS["spmv_coo"]),
    "spmv_csc": OperatorTestConfig("spmv_csc", PERFORMANCE_COMMANDS["spmv_csc"]),
    "spmv_bsr": OperatorTestConfig("spmv_bsr", PERFORMANCE_COMMANDS["spmv_bsr"]),
    "spmv_coo_tocsr": OperatorTestConfig(
        "spmv_coo_tocsr", PERFORMANCE_COMMANDS["spmv_coo_tocsr"]
    ),
    "spmm_csr": OperatorTestConfig("spmm_csr", PERFORMANCE_COMMANDS["spmm_csr"]),
    "spmm_coo": OperatorTestConfig("spmm_coo", PERFORMANCE_COMMANDS["spmm_coo"]),
    "spmm_bsr": OperatorTestConfig("spmm_bsr", PERFORMANCE_COMMANDS["spmm_bsr"]),
    "spmm_bell": OperatorTestConfig("spmm_bell", PERFORMANCE_COMMANDS["spmm_bell"]),
    "spmm_csc": OperatorTestConfig("spmm_csc", PERFORMANCE_COMMANDS["spmm_csc"]),
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
    "spsv_sell": OperatorTestConfig("spsv_sell", PERFORMANCE_COMMANDS["spsv_sell"]),
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
            # FlagTree ships the ``triton`` module, so its own version only lives
            # in the ``flagtree`` distribution metadata (no importable module).
            "flagtree": _module_version("flagtree"),
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


def _flag_gems_env_info(env_info: dict[str, object]) -> dict[str, object]:
    """Project rich runner metadata onto the strict FlagGems env schema."""
    python_info = env_info.get("python")
    platform_info = env_info.get("platform")
    packages = env_info.get("packages")
    cuda = env_info.get("cuda")
    python_info = python_info if isinstance(python_info, dict) else {}
    platform_info = platform_info if isinstance(platform_info, dict) else {}
    packages = packages if isinstance(packages, dict) else {}
    cuda = cuda if isinstance(cuda, dict) else {}

    try:
        os_release = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        os_release = {}

    torch_package = packages.get("torch")
    triton_package = packages.get("triton")
    flagsparse_package = packages.get("flagsparse")
    flagtree_package = packages.get("flagtree")
    torch_package = torch_package if isinstance(torch_package, dict) else {}
    triton_package = triton_package if isinstance(triton_package, dict) else {}
    flagsparse_package = (
        flagsparse_package if isinstance(flagsparse_package, dict) else {}
    )
    flagtree_package = flagtree_package if isinstance(flagtree_package, dict) else {}

    devices = cuda.get("devices")
    devices = devices if isinstance(devices, list) else []
    first_device = devices[0] if devices and isinstance(devices[0], dict) else {}
    cuda_available = bool(cuda.get("available"))

    try:
        triton = importlib.import_module("triton")
        triton_has_config = hasattr(triton, "Config")
    except Exception:
        triton_has_config = False

    return {
        "architecture": str(platform_info.get("machine") or ""),
        "os_name": str(
            os_release.get("ID") or platform_info.get("system") or ""
        ).lower(),
        "os_release": str(
            os_release.get("VERSION_ID") or platform_info.get("release") or ""
        ),
        "python": str(python_info.get("version") or ""),
        "torch": {
            "version": str(torch_package.get("version") or ""),
            "cuda_available": cuda_available,
            "device_name": str(first_device.get("name") or ""),
            "device_count": int(cuda.get("device_count") or 0),
        },
        # FlagTree distribution version (reserved as null by the reference
        # FlagGems summary); emit the real version when the package is present.
        "flagtree": (
            str(flagtree_package.get("version"))
            if flagtree_package.get("version")
            else None
        ),
        "triton": {
            "version": str(triton_package.get("version") or ""),
            "has_config": triton_has_config,
        },
        "flag_gems": {
            # FlagSparse is the package under test; retain the FlagGems field name
            # required by the external summary contract.
            "version": str(flagsparse_package.get("version") or ""),
            "vendor": "nvidia" if cuda_available else "cpu",
            "device": "cuda" if cuda_available else "cpu",
        },
    }


def load_operator_catalog(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        return _parse_operator_catalog_fallback(text)
    data = yaml.safe_load(text) or {}
    ops = data.get("ops")
    if isinstance(ops, list):
        return [op for op in ops if isinstance(op, dict) and op.get("id")]

    # The C API owns the delivery manifest (capi/conf/operators.yaml). Its schema
    # intentionally carries C-API details, but the runner needs only an id and
    # the reporting scope.
    wrapper_ops = data.get("operators", [])
    if isinstance(wrapper_ops, list):
        return [
            {
                "id": op["id"],
                "labels": op.get("labels", []),
                "reporting": op.get("reporting"),
                "delivery_dtypes": op.get("delivery_dtypes"),
                "dtypes": op.get("dtypes", []),
            }
            for op in wrapper_ops
            if isinstance(op, dict) and op.get("id")
        ]
    raise ValueError(f"{path} must contain a top-level 'ops' or 'operators' list")


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
        reporting_match = re.match(r"^    reporting:\s*([A-Za-z0-9_-]+)\s*$", line)
        if reporting_match:
            current["reporting"] = reporting_match.group(1)
            in_stages = False
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


def _delivery_parent_operators(yaml_path: Path) -> set[str] | None:
    """Operators backing the delivery variants declared in `yaml_path`.

    None when that manifest declares no variants, which is how the caller tells
    the two manifest shapes apart rather than guessing from the file name.
    """
    try:
        variants = load_delivery_variants(yaml_path)
    except Exception:
        return None
    parents = {str(v["operator"]) for v in variants if v.get("operator")}
    return parents or None


def read_ops(
    *,
    project_root: Path,
    operators_yaml: str,
    op_list: str | None,
    ops_arg: str | None,
    stages_arg: str,
    start: str | None,
    delivery_only: bool = False,
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
    delivery_parents = _delivery_parent_operators(yaml_path) if delivery_only else None

    requested_stages = {
        item.strip() for item in stages_arg.split(",") if item.strip()
    } or {"all"}
    if "all" in requested_stages:
        requested_stages = {"alpha", "beta", "stable"}

    result = []
    for op in catalog:
        op_id = str(op["id"]).strip()
        if delivery_only:
            # Two manifest shapes, one meaning: run exactly what the delivery
            # report will have rows for. The repo manifest says so with
            # `delivery_variants` (this operator backs one of them); the C API
            # manifest says so per entry with `reporting: delivery`.
            if delivery_parents is not None:
                if op_id not in delivery_parents:
                    continue
            elif op.get("reporting") != "delivery":
                continue
        if op_id in DEFAULT_EXCLUDED_OPS:
            continue
        if start and op_id < start:
            continue
        stage = _stage_name(op)
        if stage and stage not in requested_stages:
            continue
        result.append(op_id)
    return result


def read_operator_metadata(
    *, project_root: Path, operators_yaml: str
) -> dict[str, dict[str, object]]:
    yaml_path = Path(operators_yaml)
    if not yaml_path.is_absolute():
        yaml_path = project_root / yaml_path
    metadata: dict[str, dict[str, object]] = {}
    for item in load_operator_catalog(yaml_path):
        op_id = str(item["id"]).strip()
        labels = item.get("labels")
        metadata[op_id] = {
            "customized": True,
            "labels": list(labels) if isinstance(labels, list) else [],
        }
    return metadata


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
    return {
        "total": int(phase_result.get("total") or 0),
        "skipped": int(phase_result.get("skipped") or 0),
        "failed": int(phase_result.get("failed") or 0),
        "passed": int(phase_result.get("passed") or 0),
        "details": _accuracy_details(phase_result),
        "status": _flaggems_status(status),
        "exit_code": int(_exit_code(phase_result) or 0),
        "duration": float(_duration(phase_result) or 0.0),
        "data_file": str(_json_data_file(phase_result) or ""),
    }


def _flag_gems_dtype(dtype: object) -> str:
    value = str(dtype)
    aliases = {
        "float16": "fp16",
        "torch.float16": "fp16",
        "half": "fp16",
        "float32": "fp32",
        "torch.float32": "fp32",
        "float": "fp32",
        "float64": "fp64",
        "torch.float64": "fp64",
        "double": "fp64",
        "bfloat16": "bf16",
        "torch.bfloat16": "bf16",
    }
    return aliases.get(value, value)


def _strict_flag_gems_perf_data(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, object] = {}
    for dtype, dtype_value in value.items():
        if not isinstance(dtype_value, dict):
            continue
        details_value = dtype_value.get("details")
        details_value = details_value if isinstance(details_value, dict) else {}
        details: dict[str, object] = {}
        for shape, metrics_value in details_value.items():
            metrics = metrics_value if isinstance(metrics_value, dict) else {}
            details[str(shape)] = {
                "base": float(metrics.get("base") or 0.0),
                "gems": float(metrics.get("gems") or 0.0),
                "speedup": float(metrics.get("speedup") or 0.0),
            }
        normalized[_flag_gems_dtype(dtype)] = {
            "result": str(dtype_value.get("result") or "Unknown"),
            "details": details,
            "speedup": float(dtype_value.get("speedup") or 0.0),
        }
    return normalized


def _flaggems_performance_result(phase_result: dict[str, object]) -> dict[str, object]:
    status = str(phase_result.get("status") or "UNKNOWN")
    return {
        "duration": float(_duration(phase_result) or 0.0),
        "exit_code": int(_exit_code(phase_result) or 0),
        "data_file": str(_json_data_file(phase_result) or ""),
        "data": _strict_flag_gems_perf_data(phase_result.get("data")),
        "status": _flaggems_status(status),
        "test_case": str(phase_result.get("test_case") or "Unknown"),
    }


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

    return summarize_accuracy_cases(raw_data)


def summarize_accuracy_cases(raw_data: dict[str, object]) -> dict[str, object]:
    """Classify recorded pytest cases into counts, a status and failure details.

    The delivery projection feeds a dtype slice of the same artifact through
    here, so an operator row and a variant row can never disagree about what
    the same set of cases means.
    """
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
    if os.environ.get("FLAGSPARSE_BACKEND", "").strip().lower() == "ascend":
        # torch_npu ignores CUDA_VISIBLE_DEVICES: without this every child landed
        # on physical NPU 0 whatever --gpus said (measured on 910B, 2026-09-17).
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    pythonpath = [str(project_root / "src"), str(project_root)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _subprocess_device_id(gpu_id: int) -> int:
    """The device ordinal to pass a child whose visible devices _base_env masked.

    Backends whose runtime honours that mask renumber the one visible device to
    0, so passing the physical id would point past it: Ascend reads
    ASCEND_RT_VISIBLE_DEVICES, and XPU's torch_xmlir shim reads
    CUDA_VISIBLE_DEVICES. Every other backend keeps the id it was given.
    """
    backend = os.environ.get("FLAGSPARSE_BACKEND", "").strip().lower()
    return 0 if backend in ("ascend", "xpu") else gpu_id


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

    accuracy_env = _base_env(project_root, gpu_id)
    # XPU correctness is always evaluated against the CPU SciPy oracle.  This is
    # deliberately independent of the GPU-side PyTorch expression timed by the
    # performance phase.
    if os.environ.get("FLAGSPARSE_BACKEND", "").strip().lower() == "xpu":
        accuracy_env["FLAGSPARSE_ACCURACY_REFERENCE"] = "scipy"

    result_path = op_dir / "accuracy_result.json"
    if result_path.exists():
        result_path.unlink()

    # Ascend routes five operators through a dedicated probe instead of the pytest
    # suite. The reason is the one that governs this whole backend: on Ascend the
    # usual failure is that CANN cannot lower the kernel, not that the answer is
    # wrong, and the pytest suite's references (torch advanced indexing, torch
    # .sparse) are themselves unavailable there for several dtypes. The probe uses
    # SciPy/NumPy references instead, which run on the host and stay valid.
    #
    # It emits the SAME accuracy_result.json the pytest path does, so everything
    # downstream -- parse_accuracy_json, the summary, the data_file field -- is
    # unchanged. This branch swaps the producer, not the format.
    if (
        os.environ.get("FLAGSPARSE_BACKEND", "").strip().lower() == "ascend"
        and op in ASCEND_ACCURACY_OPS
    ):
        cmd = [
            sys.executable,
            "benchmark/benchmark_ascend_accuracy.py",
            "--op",
            op,
            "--device",
            str(_subprocess_device_id(gpu_id)),
            "--output",
            str(result_path),
        ]
        returncode, stdout, stderr, duration, timed_out = run_subprocess(
            cmd,
            project_root=project_root,
            env=accuracy_env,
            timeout=timeout,
        )
        stdout_path = op_dir / "accuracy_stdout.log"
        stderr_path = op_dir / "accuracy_stderr.log"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        parsed = parse_accuracy_json(result_path) if result_path.exists() else {}
        status = (
            "TIMEOUT"
            if timed_out
            else (
                "PASS"
                if returncode == 0 and parsed.get("status") == "Passed"
                else "FAIL"
            )
        )
        return {
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
            "data_file": str(result_path.relative_to(op_dir.parent)),
            **parsed,
        }

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
        env=accuracy_env,
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
        status = resolve_status_with_parsed(
            status,
            parsed.get("status"),
            returncode=returncode,
            timed_out=timed_out,
        )
        # ``parsed`` is splatted into the result below, so keep its status in
        # sync or it would overwrite the one just resolved.
        parsed["status"] = status
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


# Operators whose sweep is run one matrix per subprocess, so a single hung matrix is
# skippable instead of killing the whole operator's results.
PER_MATRIX_PERFORMANCE_OPS = frozenset({"spmm_bell"})
# On Ascend these two also run one .mtx per subprocess: real matrices differ by
# orders of magnitude, and one slow input must not erase the rows that finished.
ASCEND_PER_MATRIX_PERFORMANCE_OPS = frozenset({"spmm_csr", "sddmm_csr"})


def parse_op_benchmark_args(values: list[str]) -> dict[str, list[str]]:
    """Parse repeatable ``OP=ARGS`` performance-script argument overrides."""
    parsed: dict[str, list[str]] = {}
    for value in values:
        op, separator, args = str(value).partition("=")
        op = op.strip()
        if not separator or not op:
            raise ValueError(
                f"invalid --op-benchmark-args value {value!r}; expected OP=ARGS"
            )
        if not args.strip():
            raise ValueError(
                f"invalid --op-benchmark-args value {value!r}; ARGS must not be empty"
            )
        parsed.setdefault(op, []).extend(shlex.split(args))
    return parsed


def render_performance_command(
    template: tuple[str, ...],
    *,
    project_root: Path,
    op_dir: Path,
    benchmark_input: Path | None,
    warmup: int,
    iters: int,
    op: str,
    device: int,
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
                op=op,
                device=device,
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


def _row_value(row: dict[str, str], key: str | None):
    """Read a CSV column, tolerating the vendor's own casing.

    SpSV and SpSM write `FlagSparse_ms`, `PyTorch_ms`, `cuSPARSE_ms`; the schema
    table spells them lowercase. A case-sensitive `row.get()` missed every one of
    them, so those ten delivery variants reported a speedup with both latencies
    reading 0.00 ms -- a number that looks measured and is not.
    """
    if key is None:
        return None
    if key in row:
        return row[key]
    lowered = key.lower()
    for name, value in row.items():
        if name.lower() == lowered:
            return value
    return None


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
    # Two passes.  A speedup name can appear more than once with different column
    # layouts -- spmm_coo reports triton_speedup_vs_pytorch over torch_ms/ms while
    # spmm_csc reports it over pytorch_ms/ms -- so prefer an entry whose measurement
    # columns the row actually carries.  Without this the first entry always wins and
    # the other operator's rows fail the completeness check and vanish from aggregates.
    for speedup_key, base_key, latency_key in PERFORMANCE_SPEEDUP_SCHEMAS:
        if not _row_value(row, speedup_key):
            continue
        if all(key is None or _row_value(row, key) for key in (base_key, latency_key)):
            return speedup_key, base_key, latency_key
    for speedup_key, base_key, latency_key in PERFORMANCE_SPEEDUP_SCHEMAS:
        if _row_value(row, speedup_key):
            return speedup_key, base_key, latency_key
    for key, value in row.items():
        if "speedup" in key.lower() and value:
            return key, None, None
    for key in row:
        if "speedup" in key.lower():
            return key, None, None
    return "speedup", None, None


def _performance_matrix_key(row: dict[str, str]) -> str | None:
    """The matrix/case identifier used to remove interrupted work."""

    for key in ("matrix", "path", "name", "case_id", "case"):
        value = str(row.get(key) or "").strip()
        if value:
            return Path(value).name.lower()
    return None


def _interrupted_matrix_key(output: str, rows: list[dict[str, str]]) -> str | None:
    """Identify a process's in-flight matrix, falling back to the final CSV row."""

    patterns = (
        r"(?m)^\s*RUNNING:\s*([^|\r\n]+)",
        r"(?m)^\[SpGEMM\]\s+\(\d+/\d+\)\s+(.+?)\s*$",
    )
    for pattern in patterns:
        matches = re.findall(pattern, output)
        if matches:
            return Path(str(matches[-1]).strip()).name.lower()
    return _performance_matrix_key(rows[-1]) if rows else None


def filter_interrupted_performance_rows(
    rows: list[dict[str, str]],
    *,
    output: str,
    returncode: int,
    timed_out: bool,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Drop an in-flight matrix from a partial performance artifact.

    Benchmark scripts stream completed rows so a killed subprocess still leaves a useful
    CSV, but a row from the active matrix may survive when the process dies between
    algorithms or operation modes.  Exclude that whole matrix from summaries while
    leaving the source CSV untouched.
    """

    metadata: dict[str, object] = {
        "raw_row_count": len(rows),
        "excluded_row_count": 0,
        "excluded_matrix_keys": [],
    }
    if not (timed_out or returncode < 0) or not rows:
        return rows, metadata

    matrix_key = _interrupted_matrix_key(output, rows)
    if matrix_key is None:
        return rows, metadata
    filtered = [row for row in rows if _performance_matrix_key(row) != matrix_key]
    metadata["excluded_row_count"] = len(rows) - len(filtered)
    metadata["excluded_matrix_keys"] = [matrix_key]
    return filtered, metadata


def _performance_row_status_is_usable(row: dict[str, str]) -> bool:
    """Whether a benchmark row is eligible for aggregate metrics."""

    status = str(row.get("status") or row.get("matrix_status") or "").strip()
    return not status or status.upper() in {"PASS", "PASSED", "OK", "SUCCESS"}


def _capability_probe_status(rows: list[dict[str, str]]) -> str:
    """Keep a probe's semantic status when its subprocess exits successfully.

    The probe exits 0 whether the kernel ran, was declined or failed to compile
    -- the verdict is in the rows. Taking the exit code would report every one
    of them as a pass.
    """

    states = {
        str(row.get("status") or "").strip().upper()
        for row in rows
        if str(row.get("status") or "").strip()
    }
    if not states:
        return "NO_TESTS"
    if len(states) == 1:
        return states.pop()
    return "MIXED"


def _performance_row_has_complete_speedup(row: dict[str, str]) -> bool:
    """Require a passing row and both measurements behind its speedup value.

    A speedup alone is not enough: FAIL rows and rows whose baseline is missing were
    being folded into the per-dtype totals.
    """

    if not _performance_row_status_is_usable(row):
        return False
    speedup_key, base_key, latency_key = _performance_schema(row)
    speedup = _to_float(_row_value(row, speedup_key))
    if speedup is None or speedup <= 0:
        return False
    for key in (base_key, latency_key):
        if key is None:
            continue
        measurement = _to_float(_row_value(row, key))
        if measurement is None or measurement <= 0:
            return False
    return True


def _benchmark_json_detail(row: dict[str, str], index: int) -> dict[str, object]:
    speedup_key, base_key, latency_key = _performance_schema(row)
    base = _to_float(_row_value(row, base_key))
    latency = _to_float(_row_value(row, latency_key))
    speedup = _to_float(_row_value(row, speedup_key))
    # ``None`` means "no measurement", which is not the same claim as a
    # measured 0.0: benchmarks whose CSV carries no speedup column (SpMV CSC,
    # SpMM CSC/BSR on DCU, where no vendor baseline runs) used to be reported
    # as a 0.00x speedup, indistinguishable from a real result.  The HTML
    # renderer already prints an empty cell for None.
    result = {
        "shape_detail": _row_shape(row, index),
        "latency_base": base,
        "latency": latency,
        "speedup": speedup,
    }
    for key, value in row.items():
        if key in result or value in (None, ""):
            continue
        number = _to_float(value)
        result[key] = number if number is not None else value
    return result


def benchmark_json_from_csv(
    op: str,
    csv_path: Path,
    *,
    status: str = "passed",
    test_case: str = "csv",
    rows: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    if rows is None:
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


def write_benchmark_json_from_csv(
    op: str,
    csv_path: Path,
    json_path: Path,
    *,
    status: str = "passed",
    test_case: str = "csv",
    rows: list[dict[str, str]] | None = None,
) -> None:
    json_path.write_text(
        json.dumps(
            benchmark_json_from_csv(
                op, csv_path, status=status, test_case=test_case, rows=rows
            ),
            indent=2,
            sort_keys=True,
        ),
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
        speedup = _to_float(_row_value(row, speedup_key))
        base = _to_float(_row_value(row, base_key))
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
        if _performance_row_has_complete_speedup(row):
            totals.setdefault(dtype, []).append(speedup)

    for dtype, values in totals.items():
        if values:
            grouped[dtype]["speedup"] = statistics.mean(values)
    return grouped


def resolve_status_with_parsed(
    base_status: str,
    parsed_status: object,
    *,
    returncode: int,
    timed_out: bool,
) -> str:
    """Let parsed artifacts refine a status without masking a process failure.

    A run killed by ``--timeout`` (or one that crashed) can still leave a
    partial CSV/JSON behind, and the summary parsed from it reports only the
    rows that made it to disk.  Letting that upgrade the phase to ``Passed``
    hides a truncated result, so a process-level failure always wins.
    """

    if timed_out or returncode != 0:
        return base_status
    return str(parsed_status or base_status)


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


def summarize_performance_csv(
    path: Path,
    *,
    rows: list[dict[str, str]] | None = None,
    raw_row_count: int | None = None,
    excluded_row_count: int = 0,
    excluded_matrix_keys: list[str] | None = None,
) -> dict[str, object]:
    if rows is None:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if raw_row_count is None:
        raw_row_count = len(rows)
    summary: dict[str, object] = {
        "data_path": str(path),
        "csv_file": str(path),
        "row_count": len(rows),
        "raw_row_count": raw_row_count,
        "excluded_row_count": excluded_row_count,
        "excluded_matrix_keys": excluded_matrix_keys or [],
        "records": rows,
        "records_by_dtype_shape": _performance_records_by_dtype_shape(rows),
        "benchmark": performance_records_by_dtype_shape(rows),
        "data": _flaggems_perf_data(rows),
        "test_case": "csv",
    }
    if not rows:
        return summary

    # Aggregate only over rows that passed and carry both measurements behind their
    # speedup: FAIL rows and rows with a missing baseline used to be folded in.
    speedup_rows = [row for row in rows if _performance_row_has_complete_speedup(row)]
    summary["speedup_row_count"] = len(speedup_rows)
    if speedup_rows and all(
        {"dtype", "shape", "speedup"} <= set(row) for row in speedup_rows
    ):
        try:
            from benchmark.performance_utils import two_level_average_speedup

            summary["two_level_speedup"] = two_level_average_speedup(speedup_rows)
            summary["speedup"] = summary["two_level_speedup"].get("overall")
        except Exception as exc:
            summary["speedup_summary_error"] = str(exc)

    speedup_values: dict[str, list[float]] = {}
    for row in speedup_rows:
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
            # Vendor-first, same order as the per-row choice: this list used to
            # put triton_speedup_vs_pytorch first, so the operator-level number
            # was against PyTorch even where every row had a cuSPARSE baseline.
            preferred = [key for key, _, _ in PERFORMANCE_SPEEDUP_SCHEMAS] + [
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
    backend = os.environ.get("FLAGSPARSE_BACKEND", "").strip().lower()
    if backend == "ascend":
        # Every operator has an Ascend entry now, so a missing one is a real gap
        # in the table rather than an operator that quietly gets skipped.
        template = ASCEND_PERFORMANCE_COMMANDS.get(op)
        if template is None:
            return _not_configured(
                op,
                "performance",
                "no Ascend performance/probe command mapping for this operator",
            )
    elif backend == "xpu":
        template = (
            _xpu_baseline_command(op) if op in XPU_BASELINE_OPS else _probe_command(op)
        )
    elif backend in PROBE_ONLY_BACKENDS:
        # These platforms have no per-operator benchmark yet. Running the probe
        # reports which operators work and why the rest do not, which is a real
        # result; the alternative was an empty performance phase that reads the
        # same as a pass.
        template = _probe_command(op)
    elif backend and backend not in GENERIC_BENCHMARK_BACKENDS:
        return _not_configured(
            op,
            "performance",
            f"backend {backend!r} is in the registry but has no performance "
            "routing yet: add it to GENERIC_BENCHMARK_BACKENDS or "
            "PROBE_ONLY_BACKENDS",
        )
    if not template:
        return _not_configured(op, "performance", "no performance command mapping")

    # _base_env exposes exactly one physical accelerator to each child, which
    # XPU and Ascend then see as device 0 regardless of the id --gpus selected.
    script_device = _subprocess_device_id(gpu_id)

    if (
        (
            op in PER_MATRIX_PERFORMANCE_OPS
            or (backend == "xpu" and op in XPU_BASELINE_OPS)
            or (backend == "ascend" and op in ASCEND_PER_MATRIX_PERFORMANCE_OPS)
        )
        and benchmark_input is not None
        and benchmark_input.is_dir()
    ):
        return _run_bell_per_matrix(
            project_root=project_root,
            op=op,
            gpu_id=gpu_id,
            script_device=script_device,
            template=template,
            op_dir=op_dir,
            benchmark_input=benchmark_input,
            warmup=warmup,
            iters=iters,
            extra_args=extra_args,
            timeout=timeout,
        )

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
        op=op,
        device=script_device,
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
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            rows, filter_metadata = filter_interrupted_performance_rows(
                raw_rows,
                output=output,
                returncode=returncode,
                timed_out=timed_out,
            )
            # The probe and the XPU baseline exit 0 whether the kernel ran,
            # was declined or failed to compile: their verdict lives in the CSV
            # rows, so the exit code must not overwrite it with a pass.
            status_from_csv = Path(template[0]).name in {
                "benchmark_ascend_probe.py",
                "benchmark_xpu.py",
            }
            if status_from_csv and status == "PASS":
                status = _capability_probe_status(rows)
            json_status = (
                status.lower()
                if status_from_csv and status not in {"PASS", "PASSED"}
                else "passed"
            )
            write_benchmark_json_from_csv(
                op, csv_path, result_path, rows=rows, status=json_status
            )
            result.update(
                summarize_performance_csv(csv_path, rows=rows, **filter_metadata)
            )
            parsed = parse_performance_json(op, result_path)
            parsed["status"] = (
                status
                if status_from_csv
                else resolve_status_with_parsed(
                    status,
                    parsed.get("status"),
                    returncode=returncode,
                    timed_out=timed_out,
                )
            )
            result.update(parsed)
            result["data_file"] = str(result_path.relative_to(op_dir.parent))
        except Exception as exc:
            result["csv_parse_error"] = str(exc)
    return result


def _run_bell_per_matrix(
    *,
    project_root: Path,
    op: str,
    gpu_id: int,
    script_device: int,
    template: tuple[str, ...],
    op_dir: Path,
    benchmark_input: Path,
    warmup: int,
    iters: int,
    extra_args: list[str],
    timeout: int,
) -> dict[str, object]:
    """Run BELL matrices in isolated processes so one hung case is skippable."""

    result_path = op_dir / "performance_result.json"
    case_dir = op_dir / "bell_cases"
    ensure_dir(case_dir)
    matrix_paths = sorted(benchmark_input.glob("*.mtx"))
    if not matrix_paths:
        return _not_configured(op, "performance", "no .mtx files found")

    all_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    commands: list[list[str]] = []
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    timed_out_matrices: list[str] = []
    failed_matrices: list[str] = []
    total_duration = 0.0

    for index, matrix_path in enumerate(matrix_paths):
        matrix_dir = case_dir / f"{index:04d}_{matrix_path.stem}"
        ensure_dir(matrix_dir)
        cmd, csv_path = render_performance_command(
            template,
            project_root=project_root,
            op_dir=matrix_dir,
            benchmark_input=matrix_path,
            warmup=warmup,
            iters=iters,
            op=op,
            device=script_device,
            extra_args=extra_args,
        )
        commands.append(cmd)
        returncode, stdout, stderr, duration, timed_out = run_subprocess(
            cmd,
            project_root=project_root,
            env=_base_env(project_root, gpu_id),
            timeout=timeout,
        )
        total_duration += duration
        stdout_parts.append(f"===== {matrix_path.name} =====\n{stdout}")
        stderr_parts.append(f"===== {matrix_path.name} =====\n{stderr}")
        if timed_out:
            timed_out_matrices.append(matrix_path.name)
            continue
        if returncode != 0:
            failed_matrices.append(matrix_path.name)
            continue
        if not csv_path.exists():
            failed_matrices.append(matrix_path.name)
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for field in reader.fieldnames:
                    if field not in fieldnames:
                        fieldnames.append(field)
            all_rows.extend(reader)

    csv_path = op_dir / "performance.csv"
    if fieldnames:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(all_rows)

    stdout = "\n".join(stdout_parts)
    stderr = "\n".join(stderr_parts)
    stdout_path = op_dir / "performance_stdout.log"
    stderr_path = op_dir / "performance_stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    if timed_out_matrices:
        status = "TIMEOUT"
    elif failed_matrices:
        status = "FAIL"
    elif not csv_path.exists():
        status = "NO_TESTS"
    else:
        status = "PASS"
    result: dict[str, object] = {
        "operator": op,
        "phase": "performance",
        "configured": True,
        "status": status,
        "returncode": TIMEOUT_RETURN_CODE
        if timed_out_matrices
        else (1 if failed_matrices else 0),
        "exit_code": TIMEOUT_RETURN_CODE
        if timed_out_matrices
        else (1 if failed_matrices else 0),
        "duration_sec": total_duration,
        "duration": total_duration,
        "command": commands[0] if commands else [],
        "commands": commands,
        "stdout_log_path": str(stdout_path),
        "stderr_log_path": str(stderr_path),
        "log_path": str(stdout_path),
        "data_path": str(csv_path) if csv_path.exists() else None,
        "timed_out_matrices": timed_out_matrices,
        "failed_matrices": failed_matrices,
        "matrix_count": len(matrix_paths),
        "completed_matrix_count": len(matrix_paths)
        - len(timed_out_matrices)
        - len(failed_matrices),
    }
    if csv_path.exists():
        # Each timed-out matrix ran in its own process, so rows from completed
        # matrices are already independent and must not be filtered by the
        # aggregate stdout's last matrix name.
        rows = all_rows
        filter_metadata = {
            "excluded_row_count": 0,
            "excluded_matrix_keys": [],
        }
        write_benchmark_json_from_csv(op, csv_path, result_path, rows=rows)
        result.update(summarize_performance_csv(csv_path, rows=rows, **filter_metadata))
        parsed = parse_performance_json(op, result_path)
        result.update(parsed)
        if (
            status == "PASS"
            and Path(template[0]).name
            in {"benchmark_ascend_probe.py", "benchmark_xpu.py"}
        ):
            result["status"] = _capability_probe_status(rows)
        else:
            result["status"] = status
        result["data_file"] = str(result_path.relative_to(op_dir.parent))
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
    op_benchmark_args: dict[str, list[str]],
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
                extra_args=[*extra_benchmark_args, *op_benchmark_args.get(op, [])],
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
    op_benchmark_args: dict[str, list[str]],
    env_info: dict[str, object],
    operator_metadata: dict[str, dict[str, object]],
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
            op_benchmark_args=op_benchmark_args,
        )
        result.update(operator_metadata.get(op, {"customized": True, "labels": []}))
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
                    "command": (
                        shlex.join(phase_result.get("command", []))
                        if phase_result.get("command")
                        else ""
                    ),
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
    entry: dict[str, object] = {"customized": bool(result.get("customized", True))}
    operator = str(result.get("operator") or "")
    for phase in ("accuracy", "performance"):
        phase_result = result.get(phase)
        if not isinstance(phase_result, dict):
            phase_result = {
                "operator": operator,
                "phase": phase,
                "status": "NOT_CONFIGURED",
                "duration": 0.0,
                "exit_code": 0,
                "data_file": "",
            }
        entry[phase] = _flaggems_phase_result(phase_result)
    labels = result.get("labels")
    entry["labels"] = (
        [str(label) for label in labels] if isinstance(labels, list) else []
    )
    return entry


_DELIVERY_DTYPE_TOKENS = {
    "f16": ("torch.float16", "float16", "half", "fp16"),
    "f32": ("torch.float32", "float32", "float", "fp32"),
    "f64": ("torch.float64", "float64", "double", "fp64"),
    "c32": ("torch.complex64", "complex64", "c32"),
    "c64": ("torch.complex128", "complex128", "c64"),
}
_DELIVERY_PERF_DTYPES = {
    "f16": ("fp16", "float16", "half"),
    "f32": ("fp32", "float32", "float"),
    "f64": ("fp64", "float64", "double"),
    "c32": ("complex64", "c32"),
    "c64": ("complex128", "c64"),
}


def _delivery_not_configured_phase(phase: str, reason: str) -> dict[str, object]:
    return {
        "operator": "",
        "phase": phase,
        "status": "NOT_CONFIGURED",
        "reason": reason,
        "duration": 0.0,
        "exit_code": 0,
        "data_file": "",
    }


def _delivery_accuracy_phase(
    phase_result: dict[str, object], dtype: str
) -> dict[str, object]:
    """Restrict recorded pytest cases to one delivery dtype when available."""
    path_value = phase_result.get("result_path")
    if not path_value:
        return _delivery_not_configured_phase("accuracy", "no accuracy artifact")
    try:
        raw = json.loads(Path(str(path_value)).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _delivery_not_configured_phase(
            "accuracy", f"cannot read accuracy artifact: {exc}"
        )

    tokens = _DELIVERY_DTYPE_TOKENS[dtype]
    selected: dict[str, object] = {}
    for nodeid, item in raw.items():
        if not isinstance(item, dict):
            continue
        values = [str(nodeid).lower()]
        values.extend(
            str(value).lower() for value in (item.get("params") or {}).values()
        )
        words = set(re.findall(r"[a-z0-9.]+", " ".join(values)))
        if any(token in words for token in tokens):
            selected[nodeid] = item
    if not selected:
        return _delivery_not_configured_phase(
            "accuracy", f"the shared pytest suite recorded no {dtype} cases"
        )

    # Keep the parent's provenance -- the raw artifact and the two logs -- so a
    # variant row in summary.csv and result.html still points at the pytest run
    # it was sliced from. Only counts, status and details narrow to `dtype`.
    projected = dict(phase_result)
    for stale in ("errors", "xfailed", "xpassed", "failures", "tests"):
        projected.pop(stale, None)
    projected.update(summarize_accuracy_cases(selected))
    projected["phase"] = "accuracy"
    projected["duration"] = phase_result.get("duration", 0.0)
    projected["exit_code"] = phase_result.get("exit_code", 0)
    projected["data_file"] = phase_result.get("data_file", "")
    return projected


def _uses_generic_benchmark_script(op: str) -> bool:
    """Whether run_performance will launch the op's own tests/test_*.py script.

    DELIVERY_BENCHMARK_ARGS are flags of those scripts. XPU, the probe backends
    and some Ascend entries run a different script that would reject them, so
    the delivery narrowing must not reach those commands.
    """
    config = OP_TEST_CONFIGS.get(op)
    if config is None or not config.performance_cmd:
        return False
    backend = os.environ.get("FLAGSPARSE_BACKEND", "").strip().lower()
    if backend == "ascend":
        template = ASCEND_PERFORMANCE_COMMANDS.get(op)
        return bool(template) and template[0] == config.performance_cmd[0]
    return not backend or backend in GENERIC_BENCHMARK_BACKENDS


def _is_delivery_performance_row(row: dict[str, str]) -> bool:
    """True for a CSV row on the delivery axes: int32 indices, `non` op.

    Columns a benchmark does not write are not constraints. Without this filter
    a variant named ..._int_non averaged its int64 and trans/conj rows in too.
    """
    index_dtype = str(_row_value(row, "index_dtype") or "").strip().lower()
    if index_dtype and index_dtype.replace("torch.", "") != "int32":
        return False
    for key in ("op", "opA"):
        op = str(_row_value(row, key) or "").strip().lower()
        if op and op != "non":
            return False
    transpose = str(_row_value(row, "transpose") or "").strip().lower()
    return transpose not in {"true", "1"}


def _delivery_performance_phase(
    phase_result: dict[str, object], dtype: str
) -> dict[str, object]:
    """Restrict CSV-derived performance data to one delivery variant."""
    records = phase_result.get("records")
    delivery_rows = None
    if isinstance(records, list) and records:
        delivery_rows = [
            row
            for row in records
            if isinstance(row, dict) and _is_delivery_performance_row(row)
        ]
        data = _strict_flag_gems_perf_data(_flaggems_perf_data(delivery_rows))
    else:
        data = _strict_flag_gems_perf_data(phase_result.get("data"))
    selected = {
        key: value
        for key, value in data.items()
        if key.lower() in _DELIVERY_PERF_DTYPES[dtype]
    }
    if not selected:
        # A process-level timeout has no dtype rows to project, but it did run:
        # report Timeout rather than the NotFound an unconfigured variant gets.
        if str(phase_result.get("status") or "").upper() == "TIMEOUT":
            result = dict(phase_result)
            result["data"] = {}
            return result
        return _delivery_not_configured_phase(
            "performance", f"the benchmark recorded no {dtype} rows"
        )
    result = dict(phase_result)
    result["data"] = selected
    if delivery_rows is not None:
        result["records"] = delivery_rows
        result["delivery_row_count"] = len(delivery_rows)
        result["non_delivery_row_count"] = len(records) - len(delivery_rows)
    return result


def _delivery_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Project group-level runner results onto the fixed delivery registry."""
    by_operator = {str(result.get("operator")): result for result in results}
    projected: list[dict[str, object]] = []
    for variant in load_delivery_variants():
        parent = by_operator.get(variant["operator"])
        result: dict[str, object] = {
            "operator": variant["id"],
            "variant": variant,
            "customized": True,
            "labels": ["flagsparse", "delivery", variant["format"], variant["dtype"]],
        }
        if parent is None:
            reason = f"delivery parent {variant['operator']!r} was not selected"
            result["accuracy"] = _delivery_not_configured_phase("accuracy", reason)
            result["performance"] = _delivery_not_configured_phase(
                "performance", reason
            )
        else:
            accuracy = parent.get("accuracy")
            performance = parent.get("performance")
            result["accuracy"] = (
                _delivery_accuracy_phase(accuracy, variant["dtype"])
                if isinstance(accuracy, dict)
                else _delivery_not_configured_phase(
                    "accuracy", "accuracy phase was not run"
                )
            )
            result["performance"] = (
                _delivery_performance_phase(performance, variant["dtype"])
                if isinstance(performance, dict)
                else _delivery_not_configured_phase(
                    "performance", "performance phase was not run"
                )
            )
        projected.append(result)
    return projected


def _flaggems_summary(
    *,
    results: list[dict[str, object]],
    rows: list[dict[str, object]],
    env_info: dict[str, object],
) -> dict[str, object]:
    generated_at = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ordered = sorted(results, key=lambda item: str(item["operator"]))
    return {
        "timestamp": generated_at,
        "env": _flag_gems_env_info(env_info),
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


# The short spellings (fp16/fp32/fp64/bf16) are what _flaggems_perf_data keys
# `data` by, so without them a variant's own numbers matched no column and the
# row fell back to operator-wide records -- f32 and f64 rows showed the same
# mixed values, and there was no fp64 column at all for the f64 variants.
HTML_SPEEDUP_DTYPES = (
    ("fp16", ("float16", "torch.float16", "half", "fp16")),
    ("fp32", ("float32", "torch.float32", "float", "fp32")),
    ("fp64", ("float64", "torch.float64", "double", "fp64")),
    ("bf16", ("bfloat16", "torch.bfloat16", "bf16")),
    ("int16", ("int16", "torch.int16")),
    ("int32", ("int32", "torch.int32")),
    ("int8", ("int8", "torch.int8")),
    ("uint8", ("uint8", "torch.uint8")),
    ("int64", ("int64", "torch.int64", "long")),
    ("bool", ("bool", "torch.bool")),
    ("cf64", ("complex64", "torch.complex64", "cfloat", "cf64")),
    ("cf128", ("complex128", "torch.complex128", "cdouble", "cf128")),
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

    # A delivery-variant row carries its own `data`, but the phase-level
    # "speedup" next to it is the whole operator's. Average what this row
    # actually shows instead of printing the operator's number beside it.
    if by_dtype:
        values = [value for values in by_dtype.values() for value in values]
        return statistics.mean(values), {
            dtype: statistics.mean(vals) for dtype, vals in by_dtype.items()
        }

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
    # `results` is already projected onto the delivery registry by the caller;
    # projecting twice would look every variant up by its own id and report the
    # whole registry as not configured.
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
    ordered = _delivery_results(results)
    rows = _phase_rows(ordered)
    json_path = results_dir / "summary.json"
    json_path.write_text(
        json.dumps(
            _flaggems_summary(results=ordered, rows=rows, env_info=env_info),
            indent=2,
        )
        + "\n",
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


def warn_if_backend_fell_back() -> str | None:
    """Say so, loudly, when the selected backend is not the one that will run.

    Nothing in this pipeline checked this before. A box without the vendor plugin
    answers FLAGSPARSE_BACKEND=<vendor> by running every kernel on torch.cuda,
    and the whole suite then passes 40/40 with CUDA numbers under the vendor's
    name -- a result that is worse than a failure, because it looks like one that
    can be reported.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
        from flagsparse.sparse_operations._common import _accel_fallback_reason
    except Exception:
        return None
    reason = _accel_fallback_reason()
    if reason:
        banner = "!" * 78
        print(f"\n{banner}\nWARNING: {reason}\n{banner}\n", file=sys.stderr, flush=True)
    return reason


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
        "--delivery-only",
        action="store_true",
        help=(
            "Run exactly the operators the delivery report has rows for: the "
            "parents of conf/operators.yaml's delivery_variants, or the "
            "reporting: delivery entries of a C API manifest. Ignored by "
            "--ops and --op-list. Also narrows each benchmark sweep to the "
            "delivery axes (int32 indices, `non` op; see "
            "DELIVERY_BENCHMARK_ARGS); explicit --benchmark-args / "
            "--op-benchmark-args still override."
        ),
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
        parser.add_argument(
            "--op-benchmark-args",
            action="append",
            default=[],
            metavar="OP=ARGS",
            help="Extra args appended only to one performance script; repeatable.",
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
    warn_if_backend_fell_back()
    ops = read_ops(
        project_root=project_root,
        operators_yaml=args.operators_yaml,
        op_list=args.op_list,
        ops_arg=args.ops,
        stages_arg=args.stages,
        start=args.start,
        delivery_only=args.delivery_only,
    )
    operator_metadata = read_operator_metadata(
        project_root=project_root, operators_yaml=args.operators_yaml
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
    try:
        op_benchmark_args = (
            parse_op_benchmark_args(args.op_benchmark_args)
            if include_performance_args
            else {}
        )
    except ValueError as exc:
        parser.error(str(exc))
    unknown_op_args = sorted(set(op_benchmark_args) - set(ops))
    if unknown_op_args:
        parser.error(
            "--op-benchmark-args names an operator not selected by --ops: "
            + ", ".join(unknown_op_args)
        )
    extra_benchmark_args = (
        shlex.split(args.benchmark_args)
        if include_performance_args and args.benchmark_args
        else []
    )
    if args.delivery_only and include_performance_args:
        # Fold the global args into each op's list so the delivery defaults can
        # sit in front of both: explicit --benchmark-args/--op-benchmark-args win.
        for op in ops:
            defaults = (
                DELIVERY_BENCHMARK_ARGS.get(op, ())
                if _uses_generic_benchmark_script(op)
                else ()
            )
            op_benchmark_args[op] = [
                *defaults,
                *extra_benchmark_args,
                *op_benchmark_args.get(op, []),
            ]
            if defaults:
                print(
                    f"delivery-only: {op} benchmark narrowed with {' '.join(defaults)}",
                    flush=True,
                )
        extra_benchmark_args = []
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
                op_benchmark_args=op_benchmark_args,
                env_info=env_info,
                operator_metadata=operator_metadata,
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
