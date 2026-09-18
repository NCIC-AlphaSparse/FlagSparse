# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-backend OVERRIDES for the shared Triton operators.

The operators themselves live once, in :mod:`flagsparse.sparse_operations`.
These packages are empty by default and hold only files that genuinely diverge
for one backend; `_dispatch.operator_module()` prefers an override and falls
back to the shared module.

WHY EMPTY RATHER THAN A FULL COPY PER BACKEND. This tree briefly held a complete
copy of every operator for six backends -- 240,752 lines, of which 19 of every 20
files were byte-identical across all six (the only difference was one docstring
line). That layout costs more than it buys:

  * a kernel fix has to be applied six times;
  * `gcu` and `mlu` were dropped on the way, because a backend without a
    directory became a hard RuntimeError instead of falling back;
  * it cannot express per-DEVICE-MODEL tuning (_MACA_SPSV_PROFILES keys off
    _maca_device_model(), which is finer than "the backend");
  * worst, it makes divergence INVISIBLE: when two copies differ, a diff cannot
    say whether that is a deliberate port or a missed upstream fix. This repo has
    already lost work that way.

So divergence is expressed, in order of preference, as a folding `tl.constexpr`,
a tuning profile entry, another `@triton.jit` variant beside its siblings, or a
fallback dispatch table -- each of which NAMES the thing that differs. A file in
here is the last resort, and being the only occupant of an otherwise empty
directory is exactly the visibility that resort should have.
"""

__all__ = ("ascend", "cuda", "gcu", "maca", "mlu", "musa", "rocm", "xpu")
