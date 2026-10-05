# Q4 C API Coverage Handoff (2026-10-02)

## Re-verified 2026-10-05: 45/45 reproduced from a clean `capi_results/`

`capi/capi_results/` is gitignored (`capi/.gitignore`: "Test artifacts... measurements
of one machine at one moment, not source"), so the "45/45 measured, 0
NotCoveredYet" claim below does **not** persist across a fresh checkout or a
new session — only `summary_q4.json` existed on disk this time, with no
`*_benchmark.json` files, so `write_summary_q4.py` initially reported 39
`NotCoveredYet` again (looking like a regression from this doc's claim). It
was not one: every one of those 39 resolved by simply re-running the already-
implemented benchmark binaries with `FLAGSPARSE_BENCH_OUT=capi_results` set
(`test_spmm`, `test_sddmm`, `test_spmv`, `test_scatter`, `test_spgemm`,
`test_axpby`, `test_spvv`), then re-running `write_summary_q4.py`. No source
code changes were needed; the dispatch and benchmark-axis code this doc
describes was already correct, the JSON files just weren't on disk yet in
this session. Re-ran it end to end and reproduced the same **45/45 measured,
0 NotCoveredYet, 0 dispatch-only** result claimed below.

**Takeaway for the next session**: if `write_summary_q4.py` shows anything
less than 45/45, re-run the benchmark binaries listed above with
`FLAGSPARSE_BENCH_OUT` pointed at `capi_results` before assuming any
dispatch work is missing -- check `capi_results/*.json` exists first.

## Latest batch: CSR transpose SpMM

Implemented `spmm_csr_f32_int_trans_non_row` end to end:

- Added `capi/flagsparse_codegen/spmm_transpose.py`, an f32 CSR transpose
  scatter kernel. Each source-row/nnz-segment/output-column tile atomically
  accumulates into `C[col, :]`; `scale_dense()` applies `beta*C` first.
- Extended `capi/src/ops/spmm.cpp` validation and dispatch for CSR f32
  `opA=TRANSPOSE`, with the correct `A^T (A.cols x A.rows) * B` dimensions and
  arbitrary dense strides supported by the route.
- Added `SpMMAccuracy.CsrTransposeFloat32MatchesHostReference`, including
  non-unit alpha/beta and row-major B/C.
- Extended the SpMM corpus benchmark to build transpose-shaped B/C operands and
  retain the exact q4 variant tag.

Verification on CUDA:

- `cmake --build capi/build -j2` passed.
- Focused transpose accuracy test passed.
- Full synthetic SpMM benchmark passed as a test process and emitted 9 rows for
  the new variant. All rows now have a real cuSPARSE baseline: three small-
  matrix rows pass strict accuracy, and six larger rows pass the relaxed tier
  because cuSPARSE also misses the strict CPU-oracle tolerance.
- `scatter_i8_int` benchmark: 21/21 rows passed, including all three q4 rows;
  CUDA cuSPARSE geometric-mean speedup was 0.6756x.
- `spgemm_csr_f32_int_non_non` benchmark: 9/9 rows completed, with two
  relaxed-accuracy rows and one large row intentionally unchecked; the two
  vendor-comparable rows average 1.782x over cuSPARSE.
- `write_summary_q4.py` now reports all 45 registered q4 variants as measured
  through the C API, with 0 `NotCoveredYet` and 0 dispatch-only entries.
- Registry/benchmark wiring tests: 36/36 passed.

## This batch

Implemented and device-verified the 13 previously uncovered variants:

- `axpby_f16_int`
- `spmm_csc_{c32,f16,f32}_int_non_non_row`
- `spmv_sell_{c32,f16,f32}_int_non`
- `spvv_c32_int_conj`
- `spmv_{csr,coo}_f32_int_trans`
- `spmv_{csr,coo}_c32_int_conj`
- `sddmm_csr_c32_int_non_non_row`

The code paths are registered in `capi/conf/operators.yaml`. Dispatch that is
implemented and verified but not attributed to a benchmark row is reported as
`DispatchVerified` by `capi/tools/write_summary_q4.py`, distinct from both
measured variants and actual implementation gaps.

Key implementation details:

- `spmv_transpose.py` scatters CSR row segments or COO entries into the
  transposed output with atomics. `scale_dense()` applies beta first. This q4
  transpose route supports f32, and c32 conjugate-transpose negates the sparse
  value's imaginary component before complex multiplication.
- CSC SpMM is the q4 non/non row-major identity-alpha/beta path. It clears C and
  uses the existing Python CSC real/complex Triton kernels; f16, f32, and c32
  each have a hand-built matrix test.
- SDDMM c32 uses interleaved real/imag storage and the existing
  `_sddmm_csr_complex_kernel`. Its alpha and beta must be real-valued, matching
  that kernel's contract.
- AXPBY supports the requested fp16 C API path through the existing fp16 vector
  scale/axpy kernels.

Device validation on the available CUDA backend:

- `test_spmv`: the two new transpose/conjugate accuracy tests pass for both CSR
  and COO; the unsupported f64 transpose boundary test also passes.
- `test_spmm --gtest_filter='SpMMAccuracy.Csc*RowMajorIdentity'`: 3/3 pass.
- `test_sddmm --gtest_filter='SDDMMAccuracy.Complex64MatchesHostReference'`:
  1/1 passes against a complex host reference.
- Earlier in this implementation sequence: `test_spmv` SELL tests 4/4,
  `test_spvv` 3/3, and `test_axpby` 1/1 passed.

These tests validate device dispatch and numerical results. The subsequent
benchmark sweep now has rows for all 13 variants; entries without a matching
vendor baseline are still reported as measured without a speedup claim.

## Current verification and remaining issues

The exact live classification is generated by:

```bash
python3 capi/tools/write_summary_q4.py --bench-dir . --out capi/capi_results
```

At this point, there are no `NotCoveredYet` q4 variants in the generated
summary. The full affected accuracy suite passes, so the remaining non-green
benchmark rows are numerical-policy issues rather than missing dispatch:

- Ordinary fp16 SpMV/SpMM now models the kernels correctly: fp16 inputs are
  widened to fp32 for accumulation and rounded only on store. CSR/COO fp16,
  fp16->fp32 mixed, and SELL fp16 benchmark rows pass.
- The remaining strict failures are confined to order-dependent reductions:
  CSC non-transpose f16/f32/c32 SpMM, SELL f32 SpMV, real-f32-by-complex32
  SpMV, and a few large CSR/COO f32/c32 or transpose rows. CSC and mixed routes
  have no matching cuSPARSE baseline; rows with a vendor baseline are accepted
  by the spec's relaxed tier when cuSPARSE fails strict accuracy too. These are
  deliberately deferred per the current handoff decision.
- `spgemm_csr_f32_int_non_non` has two relaxed rows and one intentionally
  unchecked large row; all three execute and the focused accuracy suite passes.

Consult `capi/capi_results/summary_q4.json` for per-variant reason keys; do not
infer implementation state from the older benchmark-only `CONFIRMED` table.

## Verification completed

The following checks passed on CUDA after the fp16 oracle correction:

```bash
cmake --build capi/build -j2
./capi/build/ctest/accuracy/test_axpby
./capi/build/ctest/accuracy/test_scatter
./capi/build/ctest/accuracy/test_spvv
./capi/build/ctest/accuracy/test_spmv
./capi/build/ctest/accuracy/test_spmm
./capi/build/ctest/accuracy/test_sddmm
./capi/build/ctest/accuracy/test_spgemm
python3 capi/tools/write_summary_q4.py --bench-dir . --out capi/capi_results
pytest -q tests/ci/test_delivery_variant_registry.py tests/ci/test_q4_benchmark_wiring.py
git diff --check
```

## Backend Test Commands

The Q4 backend policy is intentionally split by platform:

- MUSA uses the split delivery runner: Python provides accuracy, while the
  C API provides performance and the muSPARSE baseline.
- CUDA can run the C API CTest suite directly.
- ROCm, MACA, Ascend, XPU, and Iluvatar currently use the Python backend
  runner; their C API profiles are registered but not buildable yet.

MUSA delivery run from the repository root:

```bash
python3 run_flagsparse_split_delivery.py \
  --backend mthreads --mode normal \
  --benchmark-input /path/to/mtx \
  --timeout 3600 \
  --results-dir pytest_results_mthreads_split
```

For a direct MUSA C API run after configuring `capi/build-musa`:

```bash
ctest --test-dir capi/build-musa -L capi --output-on-failure
```

Python-only backend runs use the same command shape; replace `BACKEND` with
`rocm`, `maca`, `ascend`, `xpu`, or `iluvatar`:

```bash
python3 tools/run_backend_tests.py \
  --backend BACKEND --phase both --mode normal
```

The backend profile files in `tests/backends/*/suite.json` and
`capi/ctest/backends/*.cmake` are the source of truth for this split.

The worktree contains changes from earlier Q4 batches as well as this batch;
preserve unrelated edits and do not reset/checkout files to clean them up.
