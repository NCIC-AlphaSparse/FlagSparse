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
"""The Ascend benchmark's implementation labels must match the operator sources.

On Ascend every delivery operator falls back to torch_npu, and the performance
baseline is also PyTorch-NPU, so both sides of the speedup ratio are the same
library.  ``benchmark_ascend.ASCEND_OPERATOR_IMPLEMENTATION`` records that so a
reader of the CSV is not left inferring it from a column called ``triton_ms``.
A hand-maintained table drifts, so this pins it to the guards in ``src/``.
"""

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES = PROJECT_ROOT / "src" / "flagsparse" / "sparse_operations"

# Where each delivery operator's Ascend fallback lives.  gather and scatter share one
# module; spmv_coo/spmm_coo have their own.
OPERATOR_MODULE = {
    "gather": "gather_scatter.py",
    "scatter": "gather_scatter.py",
    "spmv_csr": "spmv_csr.py",
    "spmv_coo": "spmv_coo.py",
    "spmm_csr": "spmm_csr.py",
    "spmm_coo": "spmm_coo.py",
    "sddmm_csr": "sddmm_csr.py",
}


def _labels():
    """Read the table without importing the script (it needs torch at import time)."""
    source = (PROJECT_ROOT / "benchmark" / "benchmark_ascend.py").read_text(
        encoding="utf-8"
    )
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ASCEND_OPERATOR_IMPLEMENTATION"
            for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("ASCEND_OPERATOR_IMPLEMENTATION not found")


def test_every_delivery_operator_is_labelled():
    assert set(_labels()) == set(OPERATOR_MODULE)


@pytest.mark.parametrize("operator", sorted(OPERATOR_MODULE))
def test_a_torch_npu_label_has_a_matching_guard(operator):
    """A 'torch_npu' label must correspond to a real _is_ascend_runtime() branch."""
    label = _labels()[operator]
    guarded = "_is_ascend_runtime()" in (SOURCES / OPERATOR_MODULE[operator]).read_text(
        encoding="utf-8"
    )
    if label == "torch_npu":
        assert guarded, (
            f"{operator} is labelled torch_npu but "
            f"{OPERATOR_MODULE[operator]} has no _is_ascend_runtime() branch"
        )
    else:
        assert label == "triton", f"unknown implementation label {label!r}"


def test_the_csv_carries_the_label():
    """The column has to reach the CSV, not just the results dict."""
    source = (PROJECT_ROOT / "benchmark" / "benchmark_ascend.py").read_text(
        encoding="utf-8"
    )
    assert '"implementation": _ascend_implementation(name)' in source
    assert '"implementation": item.get("implementation"' in source
    assert '"implementation",' in source, "missing from the csv fieldnames"
