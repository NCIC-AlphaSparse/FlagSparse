# tests/pytest 参数化用例说明 / Parametrized Test Notes

## 中文

### 运行方式

`tests/pytest` 是面向算子正确性的 CUDA pytest 集合。形状和 dtype 网格主要来自 `param_shapes.py`，`--mode quick|normal` 由 `conftest.py` 切换。项目推荐用 `run_flagsparse_pytest.py` 作为统一入口：它从 `conf/operators.yaml` 读取算子列表，在指定 GPU 上对每个算子分别运行精度和/或性能测试，并汇总结果。

常用命令：

```bash
python run_flagsparse_pytest.py --list-ops
python run_flagsparse_pytest.py --phase accuracy --mode quick --gpus 0
python run_flagsparse_pytest.py --phase both --mode quick --gpus 0,1 --benchmark-input matrix --results-dir pytest_results
python run_flagsparse_pytest.py --phase performance --ops spmv_csr,spmm_csr --benchmark-input matrix
pytest tests/pytest --mode quick -m "spmv_csr or spmm_csr"
```

`run_flagsparse_pytest.py` 默认按 `conf/operators.yaml` 的算子 id 选择测试项，可用 `--stages`、`--start` 过滤；`--ops` 和 `--op-list` 会覆盖 YAML 选择。默认全量 sweep 排除手工测试项 `alpha_spmm_alg1`、`spmv_coo_tocsr`，需要时显式指定即可运行。`spsv_descriptor_api`、`sparse_format_constructors` 这类辅助接口不是算子测试项。

精度阶段按算子 marker 单独启动 pytest，设置每个子进程的 `CUDA_VISIBLE_DEVICES`，通过 `--record json --output <op>/accuracy_result.json` 保存 FlagGems 风格 case 级原始 JSON，同时保存 `accuracy.log` 和 `accuracy_detail.json`。性能阶段按算子启动对应 `tests/test_*.py` benchmark 命令，依赖矩阵的命令使用 `--benchmark-input` 指向 MatrixMarket 文件或目录（默认 `tests/data`，本地完整性能测试通常传 `matrix`），CSV 输出会规范化成 FlagGems 风格的 `<op>/performance_result.json`；各算子目录保存 `performance.log`、`performance.csv` 和 `performance_detail.json`。runner 根目录输出 FlagGems 结构的 `summary.json`（包含 `timestamp`、`env`、`result`）；GPU id、命令、日志、totals、pytest case 明细和规范化 benchmark 记录等 FlagSparse 扩展字段写入 `summary_flat.json` 和各算子的 `*_detail.json`。同时保留 `summary.csv` 和可选的 `summary.xlsx`。如果子进程非正常退出且没有 pytest summary，runner 会标记为 `CRASH`。

### 精度策略

精度测试统一使用 `tests/pytest/accuracy_utils.py` 中的 dtype 容差表和 helper。测试文件可以保留本地 `_tol(dtype)` 包装，但不能在各文件中自定义不同的浮点容差。计算类算子以 CPU-FP64 golden reference 为基准，并在断言前转换为被测 dtype；精确/逻辑类输出使用 `atol=0, rtol=0` 判等。

新增或修改算子测试项时，最小闭环包括：算子实现代码、精度测试代码、性能测试代码、`conf/operators.yaml` 注册、`pytest.ini` marker，以及公开替换/导出注册。

### 数据构造

所有参数化正确性用例都在 `torch.device("cuda")` 上构造合成数据，不依赖 `.mtx`、ODPS 或外部矩阵文件。

- Gather/Scatter：使用 `GATHER_SCATTER_SHAPES` 和 `FLOAT_DTYPES`，参考分别为 PyTorch indexing 与 `index_copy_`。
- CSR/COO SpMV：使用合成稀疏矩阵，参考为 `torch.sparse.mm`。
- CSR/COO SpMM、SpSM、SpGEMM、SDDMM：使用小规模合成矩阵，参考为 PyTorch dense/sparse 运算或对应的采样 dense reference。
- SpSV：使用加强对角的三角矩阵，参考为 `torch.linalg.solve_triangular`；可选 CuPy/cuSPARSE reference 仅在依赖可用时运行。

### SPSV 覆盖范围

`test_spsv_csr_accuracy.py` 当前覆盖：

- CSR baseline：`float32` / `float64` lower triangular、non-unit diagonal。
- CSR non-trans 必过矩阵：`float32` / `float64` / `complex64` / `complex128`，索引为 `int32` 与 `int64`。
- CSR transpose：`float32` / `float64` / `complex64` / `complex128`，索引为 `int32` 与 `int64`。
- CSR CuPy/cuSPARSE 对照：non-trans 不包含 `complex64`；transpose 包含 `complex64`；依赖不可用时 skip。
- COO：`float32` / `float64`，覆盖 `auto` / `direct` / `csr` mode；另测 unsorted+duplicate COO 在 `auto` 下回退、在 `direct` 下拒绝。
- 当前 pytest 不修改或倒逼 SPSV 算子能力；`complex64 non-trans`、bf16、`unit_diagonal=True` 和 2D RHS 不作为本轮必过项。

## English

### Running

`tests/pytest` is the CUDA accuracy suite for FlagSparse operators. Shape and dtype grids live mostly in `param_shapes.py`; `--mode quick|normal` is handled by `conftest.py`. The recommended entry point is `run_flagsparse_pytest.py`: it reads operator ids from `conf/operators.yaml`, runs accuracy and/or performance for each operator on the requested GPUs, and summarizes the results.

Common commands:

```bash
python run_flagsparse_pytest.py --list-ops
python run_flagsparse_pytest.py --phase accuracy --mode quick --gpus 0
python run_flagsparse_pytest.py --phase both --mode quick --gpus 0,1 --benchmark-input matrix --results-dir pytest_results
python run_flagsparse_pytest.py --phase performance --ops spmv_csr,spmm_csr --benchmark-input matrix
pytest tests/pytest --mode quick -m "spmv_csr or spmm_csr"
```

`run_flagsparse_pytest.py` selects operators from `conf/operators.yaml` by default and supports `--stages` and `--start` filters. `--ops` and `--op-list` override YAML selection. The default full sweep excludes manual-test entries `alpha_spmm_alg1` and `spmv_coo_tocsr`; include them explicitly when needed. Helper APIs such as `spsv_descriptor_api` and `sparse_format_constructors` are not operator test entries.

The accuracy phase launches one pytest subprocess per operator marker, sets per-process `CUDA_VISIBLE_DEVICES`, and writes FlagGems-style case-level raw JSON via `--record json --output <op>/accuracy_result.json`, plus `accuracy.log` and `accuracy_detail.json`. The performance phase launches the configured `tests/test_*.py` benchmark command per operator; MatrixMarket-backed commands use `--benchmark-input` for the matrix file or directory (default `tests/data`, or `matrix` for the local full matrix directory), and CSV output is normalized into a FlagGems-style `<op>/performance_result.json`. Each operator directory also stores `performance.log`, `performance.csv`, and `performance_detail.json`. The root runner output uses the FlagGems `summary.json` structure with `timestamp`, `env`, and `result`. FlagSparse-only fields such as GPU id, commands, logs, totals, parsed pytest cases, and normalized benchmark records are written to `summary_flat.json` and per-operator `*_detail.json` files. `summary.csv` and optional `summary.xlsx` are also kept. If a subprocess exits abnormally without a pytest summary, the runner reports `CRASH`.

### Accuracy Policy

Accuracy tests use the shared dtype tolerance table and helpers in `tests/pytest/accuracy_utils.py`. Test files may keep local `_tol(dtype)` wrappers, but they should not define per-file floating-point tolerances. Numeric compute operators compare against CPU-FP64 golden references cast back to the dtype under test before assertion; exact/logical outputs use `atol=0, rtol=0` equality checks.

When adding or changing an operator test entry, keep the minimum loop complete: operator implementation, accuracy test, performance test, `conf/operators.yaml` registration, `pytest.ini` marker, and public replacement/export registration.

### Data Construction

All parametrized accuracy tests build synthetic tensors on `torch.device("cuda")`; they do not read `.mtx`, ODPS, or external matrix files.

- Gather/Scatter use `GATHER_SCATTER_SHAPES` and `FLOAT_DTYPES`; references are PyTorch indexing and `index_copy_`.
- CSR/COO SpMV use synthetic sparse matrices; references use `torch.sparse.mm`.
- CSR/COO SpMM, SpSM, SpGEMM, and SDDMM use small synthetic matrices and PyTorch dense/sparse or sampled dense references.
- SpSV uses diagonally strengthened triangular matrices; references use `torch.linalg.solve_triangular`; optional CuPy/cuSPARSE references run only when available.

### SPSV Coverage

`test_spsv_csr_accuracy.py` currently covers:

- CSR baseline: `float32` / `float64` lower triangular, non-unit diagonal.
- CSR non-trans required matrix: `float32` / `float64` / `complex64` / `complex128` with `int32` and `int64` indices.
- CSR transpose: `float32` / `float64` / `complex64` / `complex128` with `int32` and `int64` indices.
- CSR CuPy/cuSPARSE reference: non-trans excludes `complex64`; transpose includes `complex64`; skipped when dependencies are unavailable.
- COO: `float32` / `float64`, covering `auto` / `direct` / `csr` modes; unsorted+duplicate COO is checked for auto fallback and direct-mode rejection.
- This pytest suite does not modify or force new SPSV operator capability; `complex64` non-trans, bf16, `unit_diagonal=True`, and 2D RHS are not required pass cases in this iteration.
