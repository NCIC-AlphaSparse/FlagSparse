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

## Layout

- `src/flagsparse/` - core package (`sparse_operations/` is emitted as several `.py` modules from string literals in `flagsparse.py`)
- `tests/` - pytest tests
- `benchmark/` - performance benchmarks

## Tests

Run from project root, or `cd tests` then run scripts (paths like `../matrix` for .mtx dir).

The commands below are the repository's documented invocation standard. CPU-only install, build, help-text, and smoke paths are checked in CI; GPU-specific examples are documented but not executed there unless you opt into the triton smoke job locally.

**pytest accuracy suite** - small synthetic CUDA cases, selectable by operator marker:

```bash
pytest tests/pytest --mode quick
pytest tests/pytest --mode normal -m "spmv_csr or spmm_csr"
python run_flagsparse_pytest.py --mode quick --ops gather,spmv_csr,spmm_csr --gpus 0
python run_flagsparse_pytest.py --op-list ops.txt --gpus 0,1 --results-dir pytest_results
```

The runner writes per-operator `accuracy.log` files plus `summary.json`, `summary.csv`, and `summary.xlsx` when `openpyxl` is installed.

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

**test_spmm.py** - CSR SpMM (`.mtx` batch, synthetic, or `--csv`):

```bash
python tests/test_spmm.py <dir_or_file.mtx>
python tests/test_spmm.py --synthetic                    # optional: --skip-api-checks, --skip-alg1-coverage
python tests/test_spmm.py <dir/> --csv results.csv      # float32/float64 + int32 in CSV; per-matrix console output
# common options: --dtype, --index-dtype, --dense-cols, --block-n, --block-nnz, --max-segments, --warmup, --iters, --no-cusparse
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
python tests/test_spmm_coo.py --synthetic                # optional: --route rowrun|atomic|compare, --skip-api-checks, --skip-coo-coverage
python tests/test_spmm_coo.py <dir/> --csv out.csv      # only --route rowrun or atomic (not compare)
# same tuning flags as CSR SpMM where applicable: --dense-cols, --block-n, --block-nnz, --warmup, --iters, --no-cusparse
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

```bash
python tests/test_spsv.py --synthetic
python tests/test_spsv.py <dir/> --csv-csr spsv.csv
python tests/test_spsv.py <dir/> --csv-coo out.csv      # same CSV columns as CSR
```

**test_spsm.py** - SpSM (triangular matrix-matrix solve; **square** matrices only):

```bash
python tests/test_spsm.py --synthetic --n 512 --rhs 32
python tests/test_spsm.py <dir/> --csv-csr spsm_csr.csv --rhs 32
python tests/test_spsm.py <dir/> --csv-coo spsm_coo.csv --rhs 32
```

**test_gather.py** / **test_scatter.py** - gather/scatter benchmarks (pytest or `python tests/test_gather.py`).

Accuracy suites should use `tests/pytest/accuracy_utils.py` for FlagGems-style
golden reference and tolerance policy. Numeric compute operators compare against
CPU-FP64 golden references cast back to the dtype under test, while exact/logical
outputs compare against CPU int32 references.

## CI/CD

- `.github/workflows/ci.yml` is CPU-only and runs compile, format checks, lint, source-critical static checks, build, install, and smoke tests on GitHub-hosted runners.
- The smoke set now covers installed-wheel validation, packaging metadata, public API surface, operator registry consistency, shared runtime policy helpers, CLI `--help`, and README command snippets.
- `conf/operators.yaml` is the FlagGems-style operator interface registry for public FlagSparse sparse operators and sparse-format helpers.
- `.github/workflows/nightly-cpu.yml` is a `main`-branch-only nightly CPU check that repeats the package, lint, and shared-runtime smoke tests.
- `.github/workflows/release.yml` builds source and wheel artifacts, then attaches them to GitHub Releases on `v*` tags.
- `.github/workflows/triton-smoke.yml` is a manual opt-in job for triton-dependent smoke checks.
- `.github/workflows/gpu-ci.yml` is a manual GPU accuracy smoke workflow for a self-hosted runner labeled `self-hosted`, `linux`, and `gpu`.
- `.github/workflows/gpu-benchmark.yml` adds an Actions button for synthetic GPU benchmark runs on a self-hosted runner labeled `self-hosted`, `linux`, and `gpu`.
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
