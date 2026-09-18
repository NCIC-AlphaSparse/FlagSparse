# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-backend overrides shadow the shared operators; without one, sharing wins.

REWRITTEN for the single-implementation layout. The previous version asserted
that `spmv_csr.__implementation_module__` was always
`...backends.<backend>.spmv_csr`, which held only while every backend carried a
full copy of every operator -- 240,752 lines of which 19 in 20 files were
byte-identical. The operators live once now, and a backend directory holds only
what genuinely diverges, so the assertions here are:

  * with no override, every backend resolves to the SHARED module;
  * all eight registered backends resolve (gcu and mlu used to raise);
  * an override, when one exists, does shadow the shared module.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# These tests drive real subprocesses that import flagsparse, which imports
# torch. The CPU-only CI runner installs tooling only (tools/ci/requirements-ci),
# so without this they fail there rather than skip -- and nobody noticed because
# `make ci` runs format-check first and was failing before it ever reached
# test-ci. Machines that have torch -- dev boxes and every backend box -- still
# run them.
pytest.importorskip("torch", reason="tests/ci runs on a CPU-only runner without torch")


ROOT = Path(__file__).resolve().parents[2]
BACKENDS_DIR = ROOT / "src" / "flagsparse" / "sparse_operations" / "backends"

# Every backend in _common._BACKEND_SPECS, and the directory it maps to.
# gcu and mlu are here deliberately: they are declared slots with no tuned
# kernels, and "declared" has to mean "runs on the shared implementation"
# rather than "raises".
EXPECTED = {
    "cuda": "cuda",
    "rocm": "rocm",
    "metax": "maca",
    "mthreads": "musa",
    "ascend": "ascend",
    "xpu": "xpu",
    "gcu": "gcu",
    "mlu": "mlu",
}


def _run(code, selector):
    env = os.environ.copy()
    env["FLAGSPARSE_BACKEND"] = selector
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


@pytest.mark.parametrize(("selector", "implementation"), EXPECTED.items())
def test_backend_resolves(selector, implementation):
    """Each registered backend maps to its override directory name."""
    out = _run(
        """
        from flagsparse.sparse_operations import _dispatch
        print(_dispatch.selected_backend())
        """,
        selector,
    )
    assert out[0] == implementation


@pytest.mark.parametrize("selector", sorted(EXPECTED))
def test_no_override_uses_shared_module(selector):
    """With an empty backend directory, the operator is the shared one."""
    out = _run(
        """
        from flagsparse.sparse_operations import _dispatch
        print(_dispatch.operator_module("spmv_csr").__name__)
        print(len(_dispatch.installed_overrides()))
        """,
        selector,
    )
    assert out[0] == "flagsparse.sparse_operations.spmv_csr"
    assert out[1] == "0"


def test_backend_directories_are_empty():
    """The override directories carry no operator files.

    A file appearing here is not forbidden -- it is the documented last resort --
    but it must be a deliberate, reviewed act. This test is what makes a
    re-introduced bulk copy fail loudly instead of passing unnoticed.
    """
    stray = sorted(
        str(p.relative_to(ROOT))
        for p in BACKENDS_DIR.rglob("*.py")
        if p.name != "__init__.py"
    )
    # Message kept on one line on purpose: black 24.10 and ruff 0.15 wrap a
    # multi-line assert message in opposite directions and overwrite each other.
    assert stray == [], f"overrides under backends/ must be deliberate: {stray}"


def test_override_shadows_shared_module(tmp_path, monkeypatch):
    """A file in backends/<backend>/ does take over from the shared module."""
    target = BACKENDS_DIR / "cuda" / "_override_probe.py"
    target.write_text(
        "# Copyright 2026 FlagOS Contributors\nMARKER = 'override'\n",
        encoding="utf-8",
    )
    try:
        out = _run(
            """
            from flagsparse.sparse_operations import _dispatch
            print(_dispatch.override_path("_override_probe") is not None)
            """,
            "cuda",
        )
        assert out[0] == "True"
    finally:
        target.unlink()
