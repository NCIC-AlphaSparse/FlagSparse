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

## License

This project is licensed under the [Apache (Version 2.0) license](./LICENSE).
