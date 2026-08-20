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

"""CPU-only smoke tests for script entrypoint help text."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if os.environ.get("FLAGSPARSE_TRITON_SMOKE") != "1":
    pytest.skip(
        "triton smoke is opt-in and excluded from CPU-only CI", allow_module_level=True
    )

SCRIPTS = [
    "run_flagsparse_accuracy.py",
    "run_flagsparse_performance.py",
    "run_flagsparse_pytest.py",
    "tests/test_spmv.py",
    "tests/test_spmv_coo.py",
    "tests/test_spmv_csc.py",
    "tests/test_spmv_bsr.py",
    "tests/test_spmv_bsr_scipy.py",
    "tests/test_spmm.py",
    "tests/test_spmm_csc.py",
    "tests/test_spmm_bsr.py",
    "tests/test_spgemm.py",
    "tests/test_spsv.py",
    "tests/test_spsv_sell.py",
    "tests/test_spsm.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_help_runs(script):
    env = os.environ.copy()
    pythonpath = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    proc = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "usage:" in combined.lower()
