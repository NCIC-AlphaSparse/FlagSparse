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
    "run_flagsparse_pytest.py",
    "tests/test_spmv.py",
    "tests/test_spmv_coo.py",
    "tests/test_spmm.py",
    "tests/test_spgemm.py",
    "tests/test_spsv.py",
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
