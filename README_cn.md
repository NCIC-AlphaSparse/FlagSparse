# FlagSparse

GPU 稀疏运算库（SpMV、SpMM、SpGEMM、SDDMM、gather、scatter、多种稀疏格式）。

## 安装

```bash
pip install . --no-deps --no-build-isolation
```

离线时可加 `--no-build-isolation` 避免拉取构建依赖。

运行时依赖（按需安装）：

```bash
pip install torch triton cupy-cuda12x
```

## 目录说明

- `src/flagsparse/` - 核心包（`sparse_operations/` 由 `flagsparse.py` 内嵌字符串生成多个 `.py`）
- `tests/` - pytest 测试
- `benchmark/` - 性能基准

## 测试用法

在项目根目录执行，或先 `cd tests` 再运行脚本（.mtx 目录可用 `../matrix` 等相对路径）。

**pytest accuracy suite** - 小规模合成 CUDA 用例，可按算子 marker 选择：

```bash
pytest tests/pytest --mode quick
pytest tests/pytest --mode normal -m "spmv_csr or spmm_csr"
python run_flagsparse_pytest.py --mode quick --ops gather,spmv_csr,spmm_csr --gpus 0
python run_flagsparse_pytest.py --op-list ops.txt --gpus 0,1 --results-dir pytest_results
```

runner 会为每个算子写入 `accuracy.log`，并生成 `summary.json`、`summary.csv`；安装 `openpyxl` 时还会生成 `summary.xlsx`。

**test_spmv.py** - CSR SpMV（SuiteSparse `.mtx`、合成数据或 CSR CSV）：

```bash
python tests/test_spmv.py <目录或文件.mtx>               # 批量跑，默认 float32
python tests/test_spmv.py <目录/> --dtype float64        # 可选：--index-dtype int32|int64、--warmup、--iters、--no-cusparse
python tests/test_spmv.py --synthetic                    # 合成基准
python tests/test_spmv.py <目录/> --csv-csr results.csv  # 全部 value×index dtype 写入一个 CSV（运行过程中逐矩阵打印）
```

**test_spmv_coo.py** - COO SpMV（需 `--synthetic` 或 `--csv-coo`，不能单独批量跑 .mtx）：

```bash
python tests/test_spmv_coo.py --synthetic
python tests/test_spmv_coo.py <目录/> --csv-coo out.csv
```

**test_spmv_opt.py** - SpMV 基线 vs 优化对比（仅 `float32` / `float64`）：

```bash
python tests/test_spmv_opt.py <目录或文件.mtx> [...]
python tests/test_spmv_opt.py <目录/> --csv out.csv
```

**test_spmm.py** - CSR SpMM（`.mtx` 批量、合成或 `--csv`）：

```bash
python tests/test_spmm.py <目录或文件.mtx>
python tests/test_spmm.py --synthetic                    # 可选：--skip-api-checks、--skip-alg1-coverage
python tests/test_spmm.py <目录/> --csv results.csv     # CSV 内为 float32/float64 + int32；控制台逐矩阵输出
# 常用选项：--dtype、--index-dtype、--dense-cols、--block-n、--block-nnz、--max-segments、--warmup、--iters、--no-cusparse
```

**test_spmm_opt.py** - CSR SpMM 基线与优化版 A/B 对比：

```bash
python tests/test_spmm_opt.py <目录或文件.mtx> --dense-cols 32
python tests/test_spmm_opt.py <目录/> --csv spmm_opt.csv  # 可选：--dtype float32|float64、--dense-cols
# 常用选项：--dtype、--dense-cols、--warmup、--iters
```

**test_spmm_coo.py** - 原生 COO SpMM：

```bash
python tests/test_spmm_coo.py <目录或文件.mtx>
python tests/test_spmm_coo.py --synthetic                # 可选：--route rowrun|atomic|compare、--skip-api-checks、--skip-coo-coverage
python tests/test_spmm_coo.py <目录/> --csv out.csv     # 仅支持 --route rowrun 或 atomic（compare 不能配 --csv）
# 与 CSR SpMM 类似的调参：--dense-cols、--block-n、--block-nnz、--warmup、--iters、--no-cusparse
```

**test_sddmm.py** - CSR SDDMM（`.mtx` 批量或 `--csv`）：

```bash
python tests/test_sddmm.py <目录或文件.mtx> --k 64
python tests/test_sddmm.py <目录/> --csv out.csv         # 可选：--dtype float32|float64、--acc_mode f32|f64、--k 64
# 常用选项：--dtype、--index-dtype、--acc_mode、--k、--alpha、--beta、--warmup、--iters、--no-cupy-ref、--skip-api-checks
```

**test_spgemm.py** - CSR SpGEMM（`.mtx` 批量或 `--csv`）：

```bash
python tests/test_spgemm.py <目录或文件.mtx> --input-mode auto
python tests/test_spgemm.py <目录/> --csv results.csv    # 可选：--dtype float32|float64、--input-mode auto|a_equals_b|a_at、--compare-device cpu|gpu
# 常用选项：--dtype、--index-dtype、--warmup、--iters、--input-mode、--adaptive-loops、--no-cusparse、--ref-blocked-retry、--ref-isolated-retry、--ref-block-rows、--compare-device、--run-api-checks
```

**test_spsv.py** - SpSV（三角求解；**仅方阵**）。CSR 与 COO 共用本脚本；**不存在** `test_spsv_coo.py`。

```bash
python tests/test_spsv.py --synthetic
python tests/test_spsv.py <目录/> --csv-csr spsv.csv
python tests/test_spsv.py <目录/> --csv-coo out.csv     # 列与 CSR 相同
```

**test_spsm.py** - SpSM（三角矩阵-稠密矩阵求解；**仅方阵**）：

```bash
python tests/test_spsm.py --synthetic --n 512 --rhs 32
python tests/test_spsm.py <目录/> --csv-csr spsm_csr.csv --rhs 32
python tests/test_spsm.py <目录/> --csv-coo spsm_coo.csv --rhs 32
```

**test_gather.py** / **test_scatter.py** - gather/scatter 基准（pytest 或 `python tests/test_gather.py`）。

## 授权许可

本项目采用 [Apache (Version 2.0) license](./LICENSE) 许可证授权。
