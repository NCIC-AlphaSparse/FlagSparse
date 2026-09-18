# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pin the per-backend baselines: what is measured against, what is trusted.

Two different questions, deliberately answered separately:

  performance baseline   the vendor sparse library whose timing goes in the
                         report next to FlagSparse's own
  accuracy reference     the value the kernel is COMPARED AGAINST

CUDA and ROCm answer both with their vendor library plus torch. Every other
backend compares against SciPy on the CPU, because torch.sparse there is not a
reference but another thing under test -- MACA returns non-finite output on the
fp32 CSR path, MUSA registers no sparse matmul at all. A reference that is
itself broken reports the kernel as wrong, which is the most expensive kind of
false alarm.

The regression this guards is narrower than it looks: `_vendor_sparse_library()`
used to end in an unconditional `return "cupy_cusparse"`, so every backend that
was not explicitly listed -- xpu, gcu, mlu -- claimed CuPy on hardware that has
never had it.
"""

import json
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

# vendor baseline, accuracy reference. None means "no vendor baseline", and the
# report says N/A with a reason rather than inventing a number.
EXPECTED = {
    "cuda": ("cupy_cusparse", "torch"),
    "rocm": ("hipsparse", "torch"),
    # metax is probed, not pinned -- see test_maca_baseline_is_probed_not_assumed.
    "mthreads": (None, "scipy"),
    "ascend": ("torch", "scipy"),
    "xpu": ("torch", "scipy"),
    "gcu": (None, "scipy"),
    "mlu": (None, "scipy"),
}

_PROBE = """
    import importlib, json, os, sys

    sys.path.insert(0, "src")
    result = {}
    for backend in %r:
        os.environ["FLAGSPARSE_BACKEND"] = backend
        common = importlib.import_module("flagsparse.sparse_operations._common")
        importlib.reload(common)
        result[backend] = {
            "vendor": common._vendor_sparse_library(),
            "reference": "scipy" if common._use_scipy_accuracy_reference() else "torch",
            "cupy": common._is_cupy_available(),
        }
    print(json.dumps(result))
"""


def _probe(backends):
    env = os.environ.copy()
    env.pop("FLAGSPARSE_ACCURACY_REFERENCE", None)
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PROBE % (list(backends),))],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout)


def test_every_backend_reports_the_agreed_baselines():
    probed = _probe(list(EXPECTED) + ["metax"])
    for backend, (vendor, reference) in EXPECTED.items():
        assert probed[backend]["vendor"] == vendor, backend
        assert probed[backend]["reference"] == reference, backend


def test_maca_baseline_is_probed_not_assumed():
    """MACA takes CuPy when it is really installed, and torch when it is not.

    The C550 this was brought up on has no CuPy at all, so pinning either value
    would be wrong on one of the two machines.
    """
    probed = _probe(["metax"])["metax"]
    assert probed["vendor"] == ("cupy_cusparse" if probed["cupy"] else "torch")
    assert probed["reference"] == "scipy"


def test_accuracy_reference_override_is_honoured():
    """Forcing SciPy on CUDA is how that path gets exercised before it ships.

    Nobody here has a Moore Threads or Kunlunxin box; without the override the
    SciPy references would reach those machines never having run once.
    """
    code = """
        import sys
        sys.path.insert(0, "src")
        from flagsparse.sparse_operations import _common
        print(_common._use_scipy_accuracy_reference())
    """
    for value, expected in (("scipy", "True"), ("torch", "False")):
        env = os.environ.copy()
        env["FLAGSPARSE_BACKEND"] = "cuda"
        env["FLAGSPARSE_ACCURACY_REFERENCE"] = value
        out = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert out.stdout.strip() == expected


def _schema_for(row):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_runner_for_schema", ROOT / "run_flagsparse_pytest.py"
    )
    runner = importlib.util.module_from_spec(spec)
    sys.modules["_runner_for_schema"] = runner
    spec.loader.exec_module(runner)
    return runner._performance_schema(row)


def test_reported_speedup_prefers_the_vendor_library_over_pytorch():
    """Vendor first on CUDA and DCU; PyTorch only where the vendor has no column.

    Checked on constructed rows because the shape of the bug is invisible without
    the hardware: every script names its columns differently, and SpSV names them
    after the ACTIVE vendor label, so DCU writes `FlagSparse_vs_hipSPARSE_all_speedup`
    (the `_all` suffix is ROCm-only). A table with no entry for that spelling makes
    the DCU rows fall through to the PyTorch metric while CUDA keeps reporting the
    vendor -- the same policy silently meaning two different things per platform.

    The ordering itself was wrong once already: `triton_speedup_vs_pytorch` sat
    ahead of `triton_speedup_vs_cusparse` under a comment claiming the vendor won,
    so every CUDA speedup in the delivery report was against PyTorch.
    """
    cases = {
        # CUDA: the historical `cusparse_ms` column carries whichever vendor ran.
        "cuda spmv": (
            {
                "triton_ms": "1.0",
                "cusparse_ms": "4.0",
                "pytorch_ms": "7.0",
                "triton_speedup_vs_cusparse": "4.0",
                "triton_speedup_vs_pytorch": "7.0",
            },
            "cusparse_ms",
        ),
        "dcu spsv": (
            {
                "FlagSparse_ms": "1.0",
                "hipSPARSE_ms": "3.0",
                "PyTorch_ms": "9.0",
                "FlagSparse_vs_hipSPARSE_all_speedup": "3.0",
                "FlagSparse_vs_PyTorch_all_speedup": "9.0",
            },
            "hipsparse_ms",
        ),
        "dcu spsm": (
            {
                "FlagSparse_ms": "2.0",
                "cuSPARSE_ms": "",
                "hipSPARSE_ms": "8.0",
                "FlagSparse_vs_vendor_speedup": "4.0",
            },
            "hipsparse_ms",
        ),
        # The vendor does not implement this operator: PyTorch is the fallback,
        # which is the policy working, not an exception to it.
        "vendor column empty": (
            {
                "triton_ms": "1.0",
                "cusparse_ms": "",
                "pytorch_ms": "7.0",
                "triton_speedup_vs_pytorch": "7.0",
            },
            "pytorch_ms",
        ),
    }
    for label, (row, expected_base) in cases.items():
        _, base_key, _ = _schema_for(row)
        assert base_key == expected_base, f"{label}: got {base_key}"


def test_every_out_of_tree_backend_reports_its_own_fallback():
    """Selecting a backend whose plugin is absent must say so, for ALL of them.

    `_accel_fallback_reason()` used to name mthreads and ascend by hand, so
    FLAGSPARSE_BACKEND=xpu on a box with no Kunlunxin plugin reported backend
    "xpu", ran every kernel on torch.cuda, and answered None here -- the one
    check the backend docs tell people to run before trusting a number. gcu and
    mlu had the same hole.

    XPU is the case that makes the namespace check insufficient on its own:
    upstream PyTorch ships `torch.xpu` for Intel GPUs, so the namespace exists
    and only the missing vendor plugin gives it away.

    This runs on a CUDA box, which is exactly where the mistake happens: every
    out-of-tree backend named here is one this machine cannot actually be.
    """
    code = """
        import sys
        sys.path.insert(0, "src")
        from flagsparse.sparse_operations import _common
        print(_common._accel_fallback_reason() or "")
    """
    for backend in ("mthreads", "ascend", "xpu", "gcu", "mlu"):
        env = os.environ.copy()
        env["FLAGSPARSE_BACKEND"] = backend
        out = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        reason = out.stdout.strip()
        assert reason, f"{backend}: no fallback reason on a CUDA box"
        assert backend in reason, f"{backend}: reason does not name it: {reason}"

    # CUDA, ROCm and MACA legitimately run on torch.cuda: silence is correct.
    for backend in ("cuda", "rocm", "metax"):
        env = os.environ.copy()
        env["FLAGSPARSE_BACKEND"] = backend
        out = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert not out.stdout.strip(), f"{backend}: unexpected fallback warning"


def test_fallback_reason_requires_an_available_vendor_device(monkeypatch):
    """A loaded plugin alone must not bless an unusable device namespace."""
    from flagsparse.sparse_operations import _common

    class UnavailableXPU:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(_common, "_backend_name", lambda: "xpu")
    monkeypatch.setattr(_common, "_vendor_plugin_present", lambda spec: True)
    monkeypatch.setattr(_common, "_xpu_cuda_shim_present", lambda spec: False)
    monkeypatch.setattr(_common.torch, "xpu", UnavailableXPU())

    assert _common._resolve_accel() == (_common.torch.cuda, "cuda")
    reason = _common._accel_fallback_reason()
    assert reason is not None
    assert "xpu" in reason
    assert "no available device" in reason


def test_torch_xmlir_xpu_uses_the_cuda_shim(monkeypatch):
    """FlagTree's XPU plugin deliberately exposes its accelerator as CUDA."""
    from flagsparse.sparse_operations import _common

    monkeypatch.setattr(_common, "_backend_name", lambda: "xpu")
    monkeypatch.setattr(_common, "_xpu_cuda_shim_present", lambda spec: True)
    monkeypatch.setattr(_common.torch.cuda, "is_available", lambda: True)

    assert _common._resolve_accel() == (_common.torch.cuda, "cuda")
    assert _common._accel_fallback_reason() is None
