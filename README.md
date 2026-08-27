# FlagSparse

GPU sparse operations package (SpMV, SpMM, SpGEMM, SDDMM, gather, scatter, sparse formats).

## Install

```bash
pip install . --no-deps --no-build-isolation
```

Use `--no-build-isolation` to avoid downloading build deps when offline.

Runtime dependencies (install when needed):

```bash
pip install torch triton cupy-cuda12x
```

## Backends (CUDA / DCU)

FlagSparse dispatches its **vendor reference and baseline** paths on the detected
runtime; the Triton kernels themselves are unchanged across backends.

| Runtime | Detected by | Vendor sparse library | Python binding |
| --- | --- | --- | --- |
| NVIDIA CUDA | `torch.version.hip is None` | cuSPARSE | CuPy (`cupy-cuda12x`) |
| DCU / ROCm | `torch.version.hip is not None` | hipSPARSE | `hip-python` |

On a DCU/ROCm host, install the hip-python bindings matching your ROCm release:

```bash
pip install hip-python
```

Both bindings are optional. When neither is importable the benchmarks fall back to the
portable `torch.sparse` reference and report the reason in the `*_reason` /
`backend_status` fields rather than failing.

Selection happens in `_*_sparse_ref_backend()` helpers, which return
`("hipsparse" | "cupy_cusparse" | None, reason)`:

- `flagsparse.sparse_operations._common` - SpMV CSR/COO
- `.spmm_csr` / `.spmm_coo` / `.spgemm_csr` / `.gather_scatter` - the remaining operators

### Running the tests on DCU

Prefix every command with `PYTHONPATH=src`, and first confirm you are not running a stale
installed copy — the most common DCU pitfall:

```bash
python -c "import flagsparse; print(flagsparse.__file__)"   # must be <repo>/src/flagsparse/__init__.py
```

**1. Diagnose before benchmarking.** A hipSPARSE misuse hangs instead of raising, so probe
phase by phase rather than starting with `--op all`:

```bash
python tests/diagnose_hipsparse_ref.py --op env        # environment probe, touches no operator
python tests/diagnose_hipsparse_ref.py --timing-only   # HIP event timing chain
python tests/diagnose_hipsparse_ref.py --op spmv-csr   # then one operator at a time
python tests/diagnose_hipsparse_ref.py --op all        # only once every single probe passes
```

**2. Correctness suite.**

```bash
PYTHONPATH=src python -m pytest tests/pytest -q
```

SpSV and SpSM currently deadlock in the GPU kernel on DCU (see the known-limits section of
[docs/DCU_TESTING.md](docs/DCU_TESTING.md)), so exclude them for a run that terminates:

```bash
PYTHONPATH=src python -m pytest tests/pytest -q \
  --ignore=tests/pytest/test_spsv_csr_accuracy.py \
  --ignore=tests/pytest/test_spsv_coo_accuracy.py \
  --ignore=tests/pytest/test_spsv_sell_accuracy.py \
  --ignore=tests/pytest/test_spsm_accuracy.py
```

DCU baseline: `984 passed / 1 failed` in ~60 s, with 851 SpSV/SpSM tests excluded. The single
failure is `spmv_coo` / `spmv_csc` tolerance jitter — a different dtype parameter each run, and
it passes when run alone. Only a failure whose parameter stays fixed is a real one. CUDA
baseline for the full suite: `1613 passed / 3 failed`.

**3. Policy/contract tests** — no GPU needed, runs in seconds:

```bash
python -m pytest tests/ci -q     # expect 39 passed / 3 skipped
```

**4. Per-operator benchmarks:**

```bash
M=matrix   # any directory of .mtx files
python tests/test_spmv.py     $M --warmup 2 --iters 5
python tests/test_spmm.py     $M --warmup 2 --iters 5
python tests/test_spgemm.py   $M --warmup 2 --iters 5
python tests/test_spmm_coo.py $M --warmup 2 --iters 5
```

Make sure no other job is competing for the GPU before trusting the timings.

**5. Unified runner.** `run_flagsparse_pytest.py` has no backend awareness, and its default
sweep includes `spsv_csr`, `spsv_coo`, `spsv_sell`, `spsm_csr`, and `spsm_coo` — all five
deadlock on DCU. `--timeout` also defaults to `0` (disabled), so the run would hang forever
rather than move on. Name the operators explicitly and set a timeout as a backstop:

```bash
python run_flagsparse_pytest.py --phase both --mode quick --benchmark-input matrix \
  --timeout 3600 \
  --ops gather,scatter,spmv_csr,spmv_coo,spmv_csc,spmv_bsr,spmm_csr,spmm_coo,spmm_bsr,spmm_bell,spmm_csc,spgemm_csr,sddmm_csr
```

That op list is the full `--list-ops` set minus the five solver entries. `--timeout 3600` is
only a backstop: anything that does hang is recorded as `TIMEOUT` and the sweep continues
instead of stalling. Measured on DCU, a full 30-matrix sweep takes ~3.3 h in total, and the
heaviest per-operator benchmarks (`spmv_bsr`, `spmm_coo`, `spmm_bsr`, `spmm_bell`, `spmm_csc`) each need
more than 1800 s to finish their matrix x dtype grid — hence the 3600 s budget.

Note that `--gpus 0,1` does not help on its own: it splits the operators into two queues, and
whichever queue holds SpSV/SpSM still blocks.

For the full DCU bring-up procedure — environment checks, the stale-install trap, how to
confirm hipSPARSE was actually selected, known limits, and a troubleshooting table — see
[docs/DCU_TESTING.md](docs/DCU_TESTING.md).

## Layout

- `src/flagsparse/` - core package (`sparse_operations/` is emitted as several `.py` modules from string literals in `flagsparse.py`)
- `tests/` - pytest tests
- `benchmark/` - performance benchmarks

## Tests

Run from project root, or `cd tests` then run scripts (paths like `../matrix` for .mtx dir).

The commands below are the repository's documented invocation standard. CPU-only install, build, help-text, and smoke paths are checked in CI; GPU-specific examples are documented but not executed there unless you opt into the triton smoke job locally.

**Operator test runners** - YAML-driven accuracy/performance runs by operator:

```bash
python run_flagsparse_accuracy.py --list-ops
python run_flagsparse_accuracy.py --mode quick --gpus 0
python run_flagsparse_performance.py --ops spmv_csr,spmm_csr --benchmark-input matrix --benchmark-warmup 5 --benchmark-iters 20
python run_flagsparse_pytest.py --phase both --mode quick --gpus 0,1 --benchmark-input matrix --results-dir pytest_results
```

By default, `run_flagsparse_accuracy.py` and `run_flagsparse_performance.py` read operator ids from `conf/operators.yaml`, filter by `--stages`, and distribute operators across `--gpus`. `run_flagsparse_pytest.py --phase both` remains available when one command should run both phases. `--ops` and `--op-list` override the YAML selection. The default sweep excludes manual-test entries `alpha_spmm_alg1` and `spmv_coo_tocsr`; include them explicitly with `--ops` or `--op-list` when needed. Helper APIs such as `spsv_descriptor_api` and `sparse_format_constructors` are not operator test entries.

The accuracy phase launches `pytest tests/pytest -m <operator marker> --mode quick|normal --record json --output <op>/accuracy_result.json` and uses synthetic CUDA data. The performance phase launches the configured `tests/test_*.py` benchmark command for each operator; MatrixMarket-backed commands receive `--benchmark-input` (default `tests/data`, or pass `matrix` for the local matrix directory), and the CSV output is also normalized into a FlagGems-style `<op>/performance_result.json`. Results are written under `pytest_results_<timestamp>/` unless `--results-dir` is provided. Each operator directory contains `accuracy_stdout.log`, `accuracy_stderr.log`, `accuracy_result.json`, `accuracy_detail.json`, `performance_stdout.log`, `performance_stderr.log`, `performance.csv`, `performance_result.json`, and `performance_detail.json` when those phases run. The root `summary.json` uses the FlagGems `timestamp` / `env` / `result` structure. FlagSparse-only fields such as GPU id, commands, logs, totals, parsed pytest cases, and normalized benchmark records are kept in `summary_flat.json` and the per-operator `*_detail.json` files. `summary.csv` and optional `summary.xlsx` provide table-friendly views, and `result.html` is generated automatically for browser inspection. The generated `result.html` is rendered from `summary_flat.json`; `summary.json` remains the compact FlagGems-compatible summary for external tools.

**Direct pytest accuracy suite** - development-oriented accuracy checks, selectable by marker:

```bash
pytest tests/pytest --mode quick
pytest tests/pytest --mode normal -m "spmv_csr or spmm_csr"
pytest tests/pytest --mode quick -m "spmv_coo_tocsr"
```

When adding or changing an operator test entry, keep the implementation/API registration, `conf/operators.yaml` entry, pytest marker in `pytest.ini`, accuracy test, performance command, and public replacement/export registration in sync.

**test_spmv.py** - CSR SpMV (SuiteSparse `.mtx`, synthetic, or CSR CSV export):

```bash
python tests/test_spmv.py <dir_or_file.mtx>              # batch run, default float32
python tests/test_spmv.py <dir/> --dtype float64         # optional: --index-dtype int32|int64, --warmup, --iters, --no-cusparse
python tests/test_spmv.py --synthetic                    # synthetic benchmark
python tests/test_spmv.py <dir/> --csv-csr results.csv   # all value×index dtypes -> one CSV (per-matrix lines while running)
```

**test_spmv_coo.py** - COO SpMV (requires `--synthetic` or `--csv-coo`; no standalone `.mtx` batch):

```bash
python tests/test_spmv_coo.py --synthetic
python tests/test_spmv_coo.py <dir/> --csv-coo out.csv
```

**test_spmv_opt.py** - SpMV baseline vs optimised A/B (`float32` / `float64` only):

```bash
python tests/test_spmv_opt.py <dir_or_file.mtx> [...]
python tests/test_spmv_opt.py <dir/> --csv out.csv
```

**test_spmv_bsr.py** - native BSR SpMV with padded block-grid output:

```bash
python tests/test_spmv_bsr.py --synthetic --ops non,trans,conj
python tests/test_spmv_bsr.py <dir/> --csv-bsr out.csv --block-dims 2,4 --ops non,trans,conj --alg compare
# correctness uses BSR-expanded COO as the exact reference; PyTorch BSR is a baseline only.
# --alg blockrow_reduce runs the non-only block-row tile reduction path; compare keeps trans/conj on base.
```

**test_spmm.py** - CSR SpMM (`.mtx` batch, synthetic, or `--csv`):

```bash
python tests/test_spmm.py <dir_or_file.mtx>
python tests/test_spmm.py --synthetic                    # optional: --ops non,trans,conj
python tests/test_spmm.py <dir/> --csv results.csv      # float32/float64/complex64/complex128 + int32/int64 + ops grid
# common options: --dtype, --index-dtype, --ops, --dense-cols, --block-n, --block-nnz, --max-segments, --warmup, --iters, --no-cusparse
# CSR SpMM supports op="non" (A @ B), op="trans" (A.T @ B), and op="conj" (A.conj().T @ B).
```

**test_spmm_opt.py** - CSR SpMM baseline vs optimised A/B:

```bash
python tests/test_spmm_opt.py <dir_or_file.mtx> --dense-cols 32
python tests/test_spmm_opt.py <dir/> --csv spmm_opt.csv  # optional: --dtype float32|float64, --dense-cols
# common options: --dtype, --dense-cols, --warmup, --iters
```

**test_spmm_coo.py** - native COO SpMM:

```bash
python tests/test_spmm_coo.py <dir_or_file.mtx>
python tests/test_spmm_coo.py --synthetic                # optional: --op non|trans|conj|all, --route rowrun|atomic|compare
python tests/test_spmm_coo.py <dir/> --csv out.csv      # only --route rowrun or atomic (not compare); optional: --op all
# same tuning flags as CSR SpMM where applicable: --op, --dense-cols, --block-n, --block-nnz, --warmup, --iters, --no-cusparse
```

**test_spmm_bsr.py** - native BSR SpMM with padded block-grid output:

```bash
python tests/test_spmm_bsr.py --synthetic --block-dims 2 --ops non
python tests/test_spmm_bsr.py <dir/> --csv-bsr out.csv --block-dims 2 --ops non --dense-cols 32
# correctness uses the same BSR arrays expanded to COO as Ref=torch_spmm_coo; PyTorch/CuPy BSR are same-format baselines only when available.
```

**test_sddmm.py** - CSR SDDMM (`.mtx` batch or `--csv`):

```bash
python tests/test_sddmm.py <dir_or_file.mtx> --k 64
python tests/test_sddmm.py <dir/> --csv out.csv          # optional: --dtype float32|float64, --acc_mode f32|f64, --k 64
# common options: --dtype, --index-dtype, --acc_mode, --k, --alpha, --beta, --warmup, --iters, --no-cupy-ref, --skip-api-checks
```

**test_spgemm.py** - CSR SpGEMM (`.mtx` batch or `--csv`):

```bash
python tests/test_spgemm.py <dir_or_file.mtx> --input-mode auto
python tests/test_spgemm.py <dir/> --csv results.csv     # optional: --dtype float32|float64, --input-mode auto|a_equals_b|a_at, --compare-device cpu|gpu
# common options: --dtype, --index-dtype, --warmup, --iters, --input-mode, --adaptive-loops, --no-cusparse, --ref-blocked-retry, --ref-isolated-retry, --ref-block-rows, --compare-device, --run-api-checks
```

**test_spsv.py** - SpSV (triangular solve; **square** matrices only). CSR and COO share this script; there is **no** `test_spsv_coo.py`.

**test_spsv_sell.py** - lower, UNIT/NON_UNIT, real/complex, native column-major
SELL SpSV with NON/TRANS/CONJ operation modes. Its CSV and
terminal fields follow the CSR SpSV output. `FlagSparse_ms` and `cuSPARSE_ms`
both cover every per-call preparation/analysis plus solve; static descriptors
and SELL conversion are outside the timed interval. The direct
`flagsparse_spsv_sell` API defaults to ALG1; use `--alg_num 2` or the explicit
`flagsparse_spsv_analysis_sell` + `flagsparse_spsv_solve_sell` lifecycle for
the slice-cooperative ALG2 path. TRANS/CONJ use a dedicated reverse-dependency
kernel and do not accept `--alg_num` or `--alg2-workers`.

```bash
python tests/test_spsv.py --synthetic
python tests/test_spsv.py <dir/> --csv-csr spsv.csv
python tests/test_spsv.py <dir/> --csv-coo out.csv      # same CSV columns as CSR
pytest -q -s tests/test_spsv_sell.py
python tests/test_spsv_sell.py <dir_or_file.mtx> --csv sell_alg1.csv --slice-size 32 --alg_num 1
python tests/test_spsv_sell.py <dir_or_file.mtx> --csv sell_alg2.csv --slice-size 32 --alg_num 2
python tests/test_spsv_sell.py <dir_or_file.mtx> --csv sell_unit.csv --unit-diagonal
python tests/test_spsv_sell.py --csv sell_trans.csv --dtype float32 --slice-size 32 --ops TRANS <dir_or_file.mtx>
python tests/test_spsv_sell.py --csv sell_conj.csv --dtype complex64 --slice-size 32 --ops CONJ <dir_or_file.mtx>
python tests/test_spsv_sell.py <dir_or_file.mtx> --csv sell_complex.csv --dtype complex
# Optional ALG2 tuning: append --alg2-workers 32|64|128|256|512
```

**test_spsm.py** - SpSM (triangular matrix-matrix solve; **square** matrices only):

```bash
python tests/test_spsm.py --synthetic --n 512 --rhs 1024
python tests/test_spsm.py <dir/> --csv-csr spsm_csr.csv --rhs 1024
python tests/test_spsm.py <dir/> --csv-coo spsm_coo.csv --rhs 1024
```

**test_gather.py** / **test_scatter.py** - gather/scatter benchmarks (pytest or `python tests/test_gather.py`).

Accuracy suites should use `tests/pytest/accuracy_utils.py` for FlagGems-style
golden reference and tolerance policy. Numeric compute operators compare against
CPU-FP64 golden references cast back to the dtype under test, while exact/logical
outputs compare against CPU int32 references.

## CI/CD

- `.github/workflows/ci.yml` is CPU-only and runs compile, format checks, lint, source-critical static checks, build, install, and smoke tests on GitHub-hosted runners.
- The smoke set now covers installed-wheel validation, packaging metadata, public API surface, operator registry consistency, shared runtime policy helpers, CLI `--help`, and README command snippets.
- `conf/operators.yaml` is the FlagGems-style operator interface registry for public FlagSparse sparse operators used by the unified test runner.
- `.github/workflows/nightly-cpu.yml` is a `main`-branch-only nightly CPU check that repeats the package, lint, and shared-runtime smoke tests.
- `.github/workflows/release.yml` builds source and wheel artifacts, then attaches them to GitHub Releases on `v*` tags.
- `.github/workflows/triton-smoke.yml` is a manual opt-in job for triton-dependent smoke checks.
- `.github/workflows/gpu-ci.yml` is a manual GPU accuracy smoke workflow that runs on the `test-flagsparse` Actions Runner Controller scale set.
- `.github/workflows/gpu-benchmark.yml` adds an Actions button for synthetic GPU benchmark runs on the `test-flagsparse` Actions Runner Controller scale set.
- `.github/workflows/release-drafter.yml` keeps draft release notes current from merged PRs.
- `make help` lists the local entry points.
- `make ci` / `make check` run the same CPU-only pipeline used by CI.
- `make format-check`, `make lint`, and `make lint-src` are the non-GPU quality gates for CI formatting, CI helper lint, and critical package-source static checks.
- `make smoke` is the CPU smoke stage alias.
- `make release-check` / `make release` build, validate, and checksum release artifacts.
- `make triton-smoke` and `make triton-deps` are opt-in local targets for the triton-dependent runtime checks.
- `make gpu-env-check` validates CUDA visibility through `tools/ci/check_gpu_environment.py` on a GPU runner.
- `make gpu-benchmark` runs the quick synthetic benchmark suite on a CUDA machine.
- `python tools/ci/run_gpu_benchmark.py --suite quick` mirrors the manual GPU benchmark workflow locally on a CUDA machine.
- `python tools/ci/run_gpu_benchmark.py --suite full --matrix-dir tests/data` runs the full benchmark matrix, including `.mtx`-backed SpGEMM and SDDMM suites against the repository test matrices.
- `tools/ci/requirements-ci.lock.txt` and `tools/ci/requirements-triton-smoke.lock.txt` are the pinned local dependency bundles behind those make targets.
- `.github/dependabot.yml` keeps GitHub Actions and Python dependency updates visible.
- `.github/ISSUE_TEMPLATE/` keeps issue entry points structured for bugs and feature requests.
- The CI dependency bundle now stays on packaging and test tooling only; triton-dependent smoke is opt-in through `FLAGSPARSE_TRITON_SMOKE=1`.
- Release artifacts now ship with a generated `SHA256SUMS` manifest and a matching checksum verification step in CI.
- PR quality gates are implemented through the default CPU CI workflow; configure branch protection in GitHub to require the `CI / Build and smoke test` check before merge.
- GPU accuracy and benchmark scripts still require CUDA hardware; the GPU workflows are manual and only run on a self-hosted GPU runner.

## Performance

- `benchmark/performance_utils.py` defines the pytest-style performance base class, default metrics (`latency_base`, `latency`, `speedup`), median timing, warmup/iteration controls, CUDA synchronization, CSV record helpers, and the two-level average speedup rule.
- `benchmark/attri_util.py` and `benchmark/core_shapes.yaml` keep default and special shape grids centralized.
- `benchmark/summary_for_plot.py` reads recorded benchmark CSV files and reports the two-level speedup summary.
- `benchmark/test_sparse_perf.py` is an opt-in pytest entry point; real GPU runs remain manual or self-hosted because GitHub-hosted runners do not provide CUDA GPUs.
- `tests/data/*.mtx` can be used as the default MatrixMarket smoke dataset for mtx-backed GPU benchmark suites.

## License

This project is licensed under the [Apache (Version 2.0) license](./LICENSE).
