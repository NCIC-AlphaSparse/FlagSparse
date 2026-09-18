# Backend Triton Implementations

Each directory owns the Triton operator source selected for one runtime:

| Runtime selector | Implementation directory |
| --- | --- |
| `cuda` | `cuda/` |
| `rocm` | `rocm/` |
| `metax` | `maca/` |
| `mthreads` | `musa/` |
| `ascend` | `ascend/` |
| `xpu` | `xpu/` |

Public imports remain stable. For example,
`flagsparse.sparse_operations.spmv_csr` dispatches to exactly one of
`backends/<backend>/spmv_csr.py` when the package is imported. This keeps the
C API shims and existing Python callers on their original module paths while
letting Triton read backend-owned source files.

The six directories start from the same compatible API baseline. Backend-specific
launch tuning, lowering workarounds, and fallbacks belong in the selected
directory, not in the public compatibility entry points. XPU is a separate
implementation directory, but still requires validation on Kunlunxin hardware.
