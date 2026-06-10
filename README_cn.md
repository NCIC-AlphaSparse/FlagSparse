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

**统一算子测试 runner** - 按 YAML 算子清单逐算子运行精度/性能测试：

```bash
python run_flagsparse_pytest.py --list-ops
python run_flagsparse_pytest.py --phase accuracy --mode quick --gpus 0
python run_flagsparse_pytest.py --phase both --mode quick --gpus 0,1 --benchmark-input matrix --results-dir pytest_results
python run_flagsparse_pytest.py --phase performance --ops spmv_csr,spmm_csr --benchmark-input matrix --benchmark-warmup 5 --benchmark-iters 20
```

默认情况下，`run_flagsparse_pytest.py` 从 `conf/operators.yaml` 读取算子 id，可用 `--stages` 过滤，并按 `--gpus` 把算子分配到不同 GPU。`--ops` 和 `--op-list` 会覆盖 YAML 选择。默认全量测试会排除手工测试项 `alpha_spmm_alg1` 和 `spmv_coo_tocsr`；需要运行时用 `--ops` 或 `--op-list` 显式指定。`spsv_descriptor_api`、`sparse_format_constructors` 这类辅助接口不是算子测试项。

精度阶段会启动 `pytest tests/pytest -m <operator marker> --mode quick|normal --record json --output <op>/accuracy_result.json`，使用合成 CUDA 数据。性能阶段会按算子启动对应的 `tests/test_*.py` benchmark 命令；依赖 MatrixMarket 矩阵的命令接收 `--benchmark-input`（默认 `tests/data`，本地矩阵目录可传 `matrix`），CSV 输出也会规范化成 FlagGems 风格的 `<op>/performance_result.json`。结果默认写入 `pytest_results_<timestamp>/`，也可通过 `--results-dir` 指定。每个算子目录在对应阶段运行后包含 `accuracy.log`、`accuracy_result.json`、`accuracy_detail.json`、`performance.log`、`performance.csv`、`performance_result.json` 和 `performance_detail.json`。根目录 `summary.json` 使用 FlagGems 的 `timestamp` / `env` / `result` 结构。GPU id、命令、日志、totals、pytest case 明细和规范化 benchmark 记录等 FlagSparse 扩展字段保存在 `summary_flat.json` 和各算子的 `*_detail.json` 中。`summary.csv` 和可选 `summary.xlsx` 用于表格查看。

**直接运行 pytest 精度测试** - 面向开发调试的小规模正确性用例，可按 marker 选择：

```bash
pytest tests/pytest --mode quick
pytest tests/pytest --mode normal -m "spmv_csr or spmm_csr"
pytest tests/pytest --mode quick -m "spmv_coo_tocsr"
```

新增或修改算子测试项时，需要同步维护算子实现/API 注册、`conf/operators.yaml` 注册、`pytest.ini` marker、精度测试、性能命令以及公开替换/导出注册。

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
python tests/test_spmm.py --synthetic                    # 可选：--ops non,trans,conj
python tests/test_spmm.py <目录/> --csv results.csv     # CSV 覆盖 float32/float64/complex64/complex128 + int32/int64 + ops
# 常用选项：--dtype、--index-dtype、--ops、--dense-cols、--block-n、--block-nnz、--max-segments、--warmup、--iters、--no-cusparse
# CSR SpMM 支持 op="non" (A @ B)、op="trans" (A.T @ B)、op="conj" (A.conj().T @ B)。
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
python tests/test_spsm.py --synthetic --n 512 --rhs 1024
python tests/test_spsm.py <目录/> --csv-csr spsm_csr.csv --rhs 1024
python tests/test_spsm.py <目录/> --csv-coo spsm_coo.csv --rhs 1024
```

**test_gather.py** / **test_scatter.py** - gather/scatter 基准（pytest 或 `python tests/test_gather.py`）。

精度测试应使用 `tests/pytest/accuracy_utils.py` 中的统一断言和容差策略。计算类型算子以 CPU-FP64
作为 Golden Reference，并在断言前转换为被测 dtype；精确/逻辑类输出以 CPU int32 结果作为判等基准。

## CI/CD

- `.github/workflows/ci.yml` 是默认 CPU CI，在 GitHub-hosted runner 上执行编译检查、格式检查、静态检查、源码严重错误检查、构建、安装校验和 smoke 测试。
- smoke 测试覆盖已安装 wheel 校验、打包元数据、公开 API、算子接口注册表一致性、共享运行时策略、CLI `--help` 和 README 命令片段。
- `conf/operators.yaml` 是参考 FlagGems 风格维护的公开 FlagSparse 稀疏算子接口注册表，并作为统一测试 runner 的默认算子清单。
- `.github/workflows/nightly-cpu.yml` 是 main 分支夜间 CPU 检查，复用默认 CI 流程。
- `.github/workflows/release.yml` 在 `v*` tag 上构建源码包和 wheel，校验发布产物并上传 GitHub Release。
- `.github/workflows/triton-smoke.yml` 是手动触发的 Triton smoke 检查。
- `.github/workflows/gpu-ci.yml` 是手动触发的 GPU 精度 smoke 检查，依赖带 `self-hosted`、`linux`、`gpu` 标签的 runner。
- `.github/workflows/gpu-benchmark.yml` 是手动触发的 GPU 性能检查，依赖带 `self-hosted`、`linux`、`gpu` 标签的 runner。
- `make ci` / `make check` 运行默认 CPU CI 流程。
- `make format-check`、`make lint`、`make lint-src` 分别对应格式检查、CI 脚本静态检查和源码严重错误检查。
- `make release-check` / `make release` 构建、校验并生成发布产物 checksum。
- `make gpu-env-check` 通过 `tools/ci/check_gpu_environment.py` 在 GPU runner 上检查 CUDA 可见性。
- `make gpu-benchmark` 在 CUDA 机器上运行 quick 合成性能套件。
- `python tools/ci/run_gpu_benchmark.py --suite full --matrix-dir tests/data` 可运行完整 GPU benchmark 集合，其中 `spgemm`、`sddmm` 会使用仓库内的 `.mtx` 测试矩阵。
- PR 门禁由默认 CPU CI workflow 提供；需要在 GitHub 分支保护中把 `CI / Build and smoke test` 设置为必需检查。
- GPU 精度、GPU 性能和 Triton smoke 属于手动/可选流程，当前不进入默认 CPU 门禁。

## 性能测试

- `benchmark/performance_utils.py` 提供 pytest 风格性能测试基类，统一默认指标（`latency_base`、`latency`、`speedup`）、median 统计、warmup/iteration 配置、CUDA 同步、CSV 记录和两级平均加速比规则。
- `benchmark/attri_util.py` 和 `benchmark/core_shapes.yaml` 集中维护默认形状和特殊形状。
- `benchmark/summary_for_plot.py` 用于解析记录文件并输出两级平均加速比统计。
- `benchmark/test_sparse_perf.py` 是可选 pytest 入口；真实 GPU 性能测试仍需手动或 self-hosted GPU runner 执行。
- `tests/data/*.mtx` 可作为依赖矩阵输入的 GPU benchmark 默认烟测数据集。

## 授权许可

本项目采用 [Apache (Version 2.0) license](./LICENSE) 许可证授权。
