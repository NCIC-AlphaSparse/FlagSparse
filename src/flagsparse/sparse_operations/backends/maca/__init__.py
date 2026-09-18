# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA overrides for the shared Triton operators.

EMPTY ON PURPOSE. The operators live once, in
:mod:`flagsparse.sparse_operations`; this package holds only the files that
genuinely diverge for MetaX MACA. A module dropped in here shadows the shared one
of the same name -- see _dispatch.operator_module().

Before adding a file here, prefer the cheaper mechanisms the shared
implementation already uses, in this order:

  1. a `tl.constexpr` that folds away (HAS_BETA, SEG_IS_ROW, ...)
  2. a tuning profile entry (_MACA_SPSV_PROFILES is per DEVICE MODEL, a
     granularity a per-backend directory cannot express)
  3. another `@triton.jit` variant in the same file, beside the others
  4. a fallback dispatch table (SPMM_COO_ASCEND_DISPATCH)

A whole-file override is the last resort: it is the only one of the five
that makes a divergence invisible in a diff.
"""

__all__ = ()
