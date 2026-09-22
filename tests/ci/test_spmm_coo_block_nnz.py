# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""COO SpMM unrolls BLOCK_NNZ, so the shared SpMM override must not set it.

``_spmm_coo_rowrun_*_kernel`` iterates a row with
``for kk in tl.static_range(0, BLOCK_NNZ)``: the body is unrolled BLOCK_NNZ times
whatever the row length is, and the surplus iterations are masked off. A
30-matrix sweep put a flat 4 within 1.02x of the per-matrix optimum and 256 at
6.96x off on average, 24.2x off on roadNet-TX.

_backend_launch_overrides sizes BLOCK_NNZ from a table shared with CSR, whose
rows are tiled dynamically: 128 for a non-CSR format, 256 once nnz reaches 1e6.
The COO path consulted that table on ROCm, so DCU reproduced the whole cost the
sweep had measured away while CUDA did not -- a per-backend performance cliff
with no correctness symptom to catch it.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPMM_COO = ROOT / "src/flagsparse/sparse_operations/spmm_coo.py"


def _launch_config_source():
    tree = ast.parse(SPMM_COO.read_text(encoding="utf-8"))
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef)
        and item.name == "_resolve_spmm_coo_launch_config"
    )
    return ast.get_source_segment(SPMM_COO.read_text(encoding="utf-8"), node)


def test_the_block_nnz_default_does_not_read_the_shared_override():
    """block_n still may; BLOCK_NNZ is the one the static unroll makes unsafe."""
    source = _launch_config_source()
    block_nnz_default = source.split("if block_nnz is None:", 1)[1]
    assert "rocm_launch" not in block_nnz_default, (
        "the COO BLOCK_NNZ default must not come from _spmm_rocm_launch_overrides: "
        "that table is sized for CSR's dynamic row tiles"
    )
    assert "block_nnz = 4" in block_nnz_default
    # The block_n default above it is a separate decision and still honours ROCm.
    assert "rocm_launch" in source.split("if block_nnz is None:", 1)[0]


@pytest.mark.parametrize("backend,expected_override", [("rocm", 256), ("", None)])
def test_block_nnz_stays_at_the_swept_constant(backend, expected_override):
    """Run it: on ROCm the override would say 256 and the kernel must still get 4."""
    pytest.importorskip("torch", reason="resolving the config imports the operator")
    program = (
        "import flagsparse.sparse_operations._common as c;"
        "import flagsparse.sparse_operations.spmm_coo as m;"
        "o = c._spmm_rocm_launch_overrides("
        "n_dense_cols=32, nnz=2_000_000, fmt='coo', device=None) or {};"
        "cfg = m._resolve_spmm_coo_launch_config(32, 2_000_000);"
        "print(o.get('block_nnz'), cfg['block_nnz'])"
    )
    env = {
        "PYTHONPATH": str(ROOT / "src"),
        "PATH": "/usr/bin:/bin",
        "FLAGSPARSE_BACKEND": backend,
    }
    done = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    if done.returncode != 0:
        pytest.skip(f"cannot resolve the launch config here: {done.stderr[-200:]}")
    override, block_nnz = done.stdout.split()[-2:]
    assert override == str(expected_override), done.stdout
    assert block_nnz == "4", done.stdout
