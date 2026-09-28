#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check the q4 variants' Ascend dispatch on a CUDA box.

Ascend's Triton cannot lower most FlagSparse kernels, so every q4 route needs a
torch_npu path there. This forces every ``sparse_operations`` module onto its Ascend
branch (``_is_ascend_runtime`` -> True), makes any Triton kernel launch raise, and
runs each of the 42 variants of tests/pytest/test_q4_variants_accuracy.py against
its CPU golden. Per variant it prints one of:

  OK        took a torch path and matched the golden
  TRITON    still launches a Triton kernel under Ascend dispatch (named)
  WRONG     took a torch path but the result is off
  ERROR     raised something else

What it cannot tell: whether torch_npu itself supports those torch operations on
the real card (e.g. int8 ``index_add_``) -- that needs the hardware.

Usage (CUDA box, repo root):
    PYTHONPATH=src python3 tools/q4_ascend_dispatch_check.py
Exit status 0 when every variant is OK.
"""

import importlib
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch  # noqa: E402
import triton.runtime.jit as triton_jit  # noqa: E402

import flagsparse.sparse_operations as ops  # noqa: E402


def _force_ascend():
    for info in pkgutil.iter_modules(ops.__path__):
        module = importlib.import_module(f"flagsparse.sparse_operations.{info.name}")
        if hasattr(module, "_is_ascend_runtime"):
            module._is_ascend_runtime = lambda: True

    def refuse(self, *args, **kwargs):
        raise RuntimeError(f"TRITON kernel launched: {self.fn.__name__}")

    triton_jit.JITFunction.run = refuse


def main():
    _force_ascend()
    import tests.pytest.test_q4_variants_accuracy as cases

    results = {}
    for name, _marker, builder, dtype in cases.VARIANTS:
        gen = torch.Generator().manual_seed(1)
        try:
            result, reference, expected = builder(dtype, cases.SHAPES[0], gen)
        except Exception as exc:
            text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            if "TRITON kernel launched" in text:
                results[name] = "TRITON  " + text.split(": ")[-1]
            else:
                results[name] = f"ERROR   {type(exc).__name__}: {text[:100]}"
            continue
        try:
            cases._check(result, reference, expected)
            results[name] = "OK"
        except AssertionError as exc:
            results[name] = f"WRONG   {str(exc).splitlines()[0][:100]}"
    for name, verdict in results.items():
        print(f"{name:34s} {verdict}")
    ok = sum(v == "OK" for v in results.values())
    print(f"\n{ok}/{len(results)} variants take a correct torch path under Ascend dispatch")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
