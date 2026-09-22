# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Which oracle a backend uses does not decide whether it has a timing baseline.

The SpMM scripts gated the PyTorch baseline on _use_scipy_accuracy_reference(),
which is true for every backend outside CUDA and ROCm. Two different questions
were being answered by one predicate:

    is the torch.sparse VALUE trustworthy here   -- false on MACA (its fp32 CSR
                                                    path returns non-finite)
    does a torch.sparse route RUN here           -- still true on MACA, and its
                                                    latency is that backend's
                                                    delivery baseline

Only MUSA answers no to both: it registers no sparse matmul in any layout or
dtype. The conflation left all four SpMM delivery variants on MACA reporting
`Passed` with an empty speedup, and the MACA CSV even carried the MUSA wording.

Asserted on the source: importing these modules needs torch and an accelerator,
so a behavioural test would skip in CI -- the one environment that has to hold
this line. The behaviour itself was checked by running both scripts under
FLAGSPARSE_BACKEND=metax, where pytorch_ms went from empty to measured.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SPMM_SCRIPTS = ("tests/test_spmm.py", "tests/test_spmm_coo.py")


def _calls_in(node):
    return {
        child.func.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }


def _function(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


@pytest.mark.parametrize("script", SPMM_SCRIPTS)
def test_the_baseline_decision_names_the_musa_predicate(script):
    """A backend without a sparse matmul is the only one with nothing to time."""
    calls = _calls_in(_function(script, "_build_pytorch_reference"))
    assert "_is_mthreads_runtime" in calls, (
        f"{script}: _build_pytorch_reference must decide the PyTorch baseline on "
        "_is_mthreads_runtime(), not on which accuracy oracle the backend uses"
    )


@pytest.mark.parametrize("script", SPMM_SCRIPTS)
def test_the_oracle_choice_is_still_made_where_it_belongs(script):
    """The fix must not have taken SciPy away from the backends that need it."""
    calls = _calls_in(_function(script, "_build_pytorch_reference"))
    assert "_use_scipy_accuracy_reference" in calls, script


def test_no_spmm_script_hardcodes_the_musa_wording_for_every_backend():
    """The MACA CSV reported "unavailable on MUSA" while running on a C550."""
    for script in SPMM_SCRIPTS:
        source = (ROOT / script).read_text(encoding="utf-8")
        assert "baseline unavailable on MUSA" not in source, script


def test_the_csr_script_reports_why_a_baseline_is_missing():
    """`-` in the report has to be traceable to a reason, not to a silent skip."""
    source = (ROOT / "tests/test_spmm.py").read_text(encoding="utf-8")
    assert "if pytorch_op is None:" in source
    assert 'result.setdefault("pytorch_reason"' in source
