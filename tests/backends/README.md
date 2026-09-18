# Backend-owned test suites

Each backend owns one suite description in this directory.  The descriptions
select the runtime override, C API status, and the shared accuracy and benchmark
implementations.  Keeping the operator cases in `tests/pytest` and `tests/`
shared prevents six copies of the same oracle from diverging.

Run a backend suite from the repository root:

```bash
python tools/run_backend_tests.py --backend cuda --phase accuracy --mode quick
python tools/run_backend_tests.py --backend rocm --phase both --ops spmv_csr,spmm_csr
```

The command always sets `FLAGSPARSE_BACKEND` from the selected suite.  A suite
does not claim that its accelerator is available: it records whether the C API
can currently be built, while actual hardware and SDK checks remain the
responsibility of that backend's runtime.
