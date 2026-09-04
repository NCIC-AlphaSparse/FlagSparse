<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

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

## 后端（CUDA / DCU / MetaX / 摩尔线程 / 昇腾）

FlagSparse 按检测到的运行时对**厂商参考实现与基线**进行分发；Triton 内核本身在各后端保持不变。

| 运行时 | 判定方式 | 厂商稀疏库 | Python 绑定 |
| --- | --- | --- | --- |
| NVIDIA CUDA | `torch.version.hip is None` | cuSPARSE | CuPy（`cupy-cuda12x`） |
| DCU / ROCm | `torch.version.hip is not None` | hipSPARSE | `hip-python` |
| MetaX / MACA（C550） | 见 `_detect_maca_runtime()` | 暂用 CuPy/cuSPARSE 兼容路径 | CuPy |
| 摩尔线程 / MUSA | `torch.musa` 可用 | **`torch.sparse`**（暂定） | torch_musa |
| 昇腾 / CANN（910B） | `torch.npu` 可用 | **ops-sparse**，缺失时回落 `torch.sparse` | torch_npu |

> ⚠️ **摩尔线程与昇腾不是 CUDA 兼容的**：torch 把它们暴露成独立设备类型
> （`musa` / `npu`），`torch.cuda.*` 和 `Tensor.is_cuda` 都不适用。
> **设备抽象迁移已完成**：`sparse_operations/` 下 269 处 CUDA 专属调用已换成
> `_ACCEL.*`（189 处）和 `_is_accel_tensor()`（82 处）。在 CUDA / ROCm / MACA 上
> `_ACCEL is torch.cuda`，因此对这三个后端是**逐字等价**，内核一行未改。

MACA 与 CUDA 源码兼容，机器上 `torch.version.cuda` 有值而 `torch.version.hip` 为 `None`，
**靠 ROCm 那套判据分不出 MetaX 和 NVIDIA**。检测按顺序尝试：`FLAGSPARSE_BACKEND` 环境变量、
MetaX 专有的 `torch.version` 属性、MACA SDK 环境变量（`MACA_PATH` / `MACA_HOME`）、
设备名（先认厂商串，再认 `c550` / `c500` 型号串）。
**在真机上先显式指定，等自动探测确认无误后再依赖它：**

```bash
export FLAGSPARSE_BACKEND=metax     # cuda | rocm | metax | mthreads | ascend
export FLAGSPARSE_MACA_MODEL=c550   # 覆盖型号检测
export FLAGSPARSE_MACA_VENDOR=none  # CuPy 不可用时跳过厂商基线

# 摩尔线程 / 昇腾的基线选择
export FLAGSPARSE_MTHREADS_VENDOR=torch      # torch | musparse | none
export FLAGSPARSE_ASCEND_VENDOR=ops_sparse   # ops_sparse | torch | none
```

调优参数按型号分档，在 `spmv_csr.py` 的 `_MACA_SPMV_PROFILES` 和 `spsv.py` 的
`_MACA_SPSV_PROFILES`。目前只有 **C550**（FlagTree metax 后端的参考型号）一档，
内容是从 CUDA 平移的；识别不出的型号会回落到它。每个参数旁注了 gfx936 上实测的
ROCm 值，是有意不继承的。

在 DCU/ROCm 机器上，安装与 ROCm 版本匹配的 hip-python：

```bash
pip install hip-python
```

两种绑定都是可选的。若都不可用，基准测试会回退到通用的 `torch.sparse` 参考实现，
并在 `*_reason` / `backend_status` 字段中给出原因，而不是直接报错。

分发逻辑位于各 `_*_sparse_ref_backend()` 辅助函数，返回
`("hipsparse" | "cupy_cusparse" | None, reason)`：

- `flagsparse.sparse_operations._common` —— SpMV CSR/COO
- `.spmm_csr` / `.spmm_coo` / `.spgemm_csr` / `.gather_scatter` —— 其余算子

### 在 DCU 上跑测试

所有命令都要带 `PYTHONPATH=src`。开工先确认没跑到旧的已安装包 —— 这是 DCU 上最容易踩的坑：

```bash
python -c "import flagsparse; print(flagsparse.__file__)"   # 必须指向 <仓库>/src/flagsparse/__init__.py
```

**1. 先 diagnose，再跑基准。** hipSPARSE 用错时是挂住而不是报错，所以要逐级探，
不要一上来就 `--op all`：

```bash
python tests/diagnose_hipsparse_ref.py --op env        # 先探环境，不碰任何算子
python tests/diagnose_hipsparse_ref.py --timing-only   # HIP 事件计时链
python tests/diagnose_hipsparse_ref.py --op spmv-csr   # 再一次一个算子
python tests/diagnose_hipsparse_ref.py --op all        # 单点全部通过后才跑这个
```

**2. 正确性套件。**

```bash
PYTHONPATH=src python -m pytest tests/pytest -q
```

SpSV 和 SpSM 目前在 DCU 上会 GPU 内核死锁（见 [docs/DCU_TESTING.md](docs/DCU_TESTING.md)
已知限制一节），需要排除掉才能跑完：

```bash
PYTHONPATH=src python -m pytest tests/pytest -q \
  --ignore=tests/pytest/test_spsv_csr_accuracy.py \
  --ignore=tests/pytest/test_spsv_coo_accuracy.py \
  --ignore=tests/pytest/test_spsv_sell_accuracy.py \
  --ignore=tests/pytest/test_spsm_accuracy.py
```

DCU 基准线：排除 851 个 SpSV/SpSM 用例后 `984 passed / 1 failed`，约 60 秒。
那个失败是 `spmv_coo` / `spmv_csc` 的容差抖动 —— 每次失败的 dtype 参数都不一样，
且单独跑就过；参数固定不变的失败才是真问题。CUDA 全量基准线：`1613 passed / 3 failed`。

**3. 策略/契约类测试** —— 不需要 GPU，秒级：

```bash
python -m pytest tests/ci -q     # 期望 39 passed / 3 skipped
```

**4. 逐算子基准：**

```bash
M=matrix   # 任意 .mtx 目录
python tests/test_spmv.py     $M --warmup 2 --iters 5
python tests/test_spmm.py     $M --warmup 2 --iters 5
python tests/test_spgemm.py   $M --warmup 2 --iters 5
python tests/test_spmm_coo.py $M --warmup 2 --iters 5
```

看时间数据前先确认没有别的任务在争抢 GPU。

**5. 统一运行器。** `run_flagsparse_pytest.py` 没有后端感知，默认算子清单包含
`spsv_csr`、`spsv_coo`、`spsv_sell`、`spsm_csr`、`spsm_coo` —— 这五个在 DCU 上都会死锁。
而且 `--timeout` 默认是 `0`（关闭），一旦卡住就是无限等待而不会跳过。
所以在 DCU 上要显式指定算子，并加超时兜底：

```bash
python run_flagsparse_pytest.py --phase both --mode quick --benchmark-input matrix \
  --timeout 3600 \
  --ops gather,scatter,spmv_csr,spmv_coo,spmv_csc,spmv_bsr,spmm_csr,spmm_coo,spmm_bsr,spmm_csc,spgemm_csr,sddmm_csr
```

这个算子清单就是 `--list-ops` 的全集去掉那五个求解器条目。`--timeout 3600` 只是兜底：
真卡住的会记成 `TIMEOUT` 并继续往下跑，而不是整轮停摆。DCU 实测 30 个矩阵跑完全程约 3.3 小时，
其中 `spmv_bsr`、`spmm_coo`、`spmm_bsr`、`spmm_csc` 四个算子的矩阵 x dtype 网格单个就超过 1800 秒，
所以预算给到 3600。

注意 `--gpus 0,1` 单独用没有用 —— 它只是把算子分成两条队列，含 SpSV/SpSM 的那条照样堵死。

DCU 上的完整验证流程（环境检查、旧安装包陷阱、如何确认真的走了 hipSPARSE、
已知限制、排查速查表）见 [docs/DCU_TESTING.md](docs/DCU_TESTING.md)。

## 目录说明

- `src/flagsparse/` - 核心包（`sparse_operations/` 由 `flagsparse.py` 内嵌字符串生成多个 `.py`）
- `tests/` - pytest 测试
- `benchmark/` - 性能基准

## 测试用法

在项目根目录执行，或先 `cd tests` 再运行脚本（.mtx 目录可用 `../matrix` 等相对路径）。

**算子测试 runner** - 按 YAML 算子清单逐算子运行精度/性能测试：

```bash
python run_flagsparse_accuracy.py --list-ops
python run_flagsparse_accuracy.py --mode quick --gpus 0
python run_flagsparse_performance.py --ops spmv_csr,spmm_csr --benchmark-input matrix --benchmark-warmup 5 --benchmark-iters 20
python run_flagsparse_pytest.py --phase both --mode quick --gpus 0,1 --benchmark-input matrix --results-dir pytest_results
```

默认情况下，`run_flagsparse_accuracy.py` 和 `run_flagsparse_performance.py` 从 `conf/operators.yaml` 读取算子 id，可用 `--stages` 过滤，并按 `--gpus` 把算子分配到不同 GPU。需要一个命令同时跑两个阶段时，仍可使用 `run_flagsparse_pytest.py --phase both`。`--ops` 和 `--op-list` 会覆盖 YAML 选择。默认全量测试会排除手工测试项 `alpha_spmm_alg1` 和 `spmv_coo_tocsr`；需要运行时用 `--ops` 或 `--op-list` 显式指定。`spsv_descriptor_api`、`sparse_format_constructors` 这类辅助接口不是算子测试项。

精度阶段会启动 `pytest tests/pytest -m <operator marker> --mode quick|normal --record json --output <op>/accuracy_result.json`，使用合成 CUDA 数据。性能阶段会按算子启动对应的 `tests/test_*.py` benchmark 命令；依赖 MatrixMarket 矩阵的命令接收 `--benchmark-input`（默认 `tests/data`，本地矩阵目录可传 `matrix`），CSV 输出也会规范化成 FlagGems 风格的 `<op>/performance_result.json`。结果默认写入 `pytest_results_<timestamp>/`，也可通过 `--results-dir` 指定。每个算子目录在对应阶段运行后包含 `accuracy_stdout.log`、`accuracy_stderr.log`、`accuracy_result.json`、`accuracy_detail.json`、`performance_stdout.log`、`performance_stderr.log`、`performance.csv`、`performance_result.json` 和 `performance_detail.json`。根目录 `summary.json` 使用 FlagGems 的 `timestamp` / `env` / `result` 结构。GPU id、命令、日志、totals、pytest case 明细和规范化 benchmark 记录等 FlagSparse 扩展字段保存在 `summary_flat.json` 和各算子的 `*_detail.json` 中。`summary.csv` 和可选 `summary.xlsx` 用于表格查看，并会自动生成可在浏览器查看的 `result.html`。自动生成的 `result.html` 使用 `summary_flat.json` 渲染；`summary.json` 保持为面向外部工具的精简 FlagGems 兼容汇总。

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

**test_spmv_bsr.py** - 原生 BSR SpMV，输出按 block-grid padding：

```bash
python tests/test_spmv_bsr.py --synthetic --ops non,trans,conj
python tests/test_spmv_bsr.py <目录/> --csv-bsr out.csv --block-dims 2,4 --ops non,trans,conj --alg compare
# correctness 使用 BSR 展开的 COO 作为精确 reference；PyTorch BSR 只作为 baseline。
# --alg blockrow_reduce 运行仅支持 non 的 block-row tile reduction 路径；compare 对 trans/conj 保持 base。
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
python tests/test_spmm_coo.py --synthetic                # 可选：--op non|trans|conj|all、--route rowrun|atomic|compare
python tests/test_spmm_coo.py <目录/> --csv out.csv     # 仅支持 --route rowrun 或 atomic（compare 不能配 --csv）；可选：--op all
# 与 CSR SpMM 类似的调参：--op、--dense-cols、--block-n、--block-nnz、--warmup、--iters、--no-cusparse
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

**test_spsv_sell.py** - 下三角、UNIT/NON_UNIT、实数/复数、原生列主序 SELL SpSV，
支持 NON/TRANS/CONJ 操作模式。CSV 和终端字段
遵循 CSR SpSV 输出；`FlagSparse_ms` 和 `cuSPARSE_ms` 都覆盖每次调用的准备/
分析加求解，静态 descriptor 与 SELL 转换不计时。直接
`flagsparse_spsv_sell` API 默认使用 ALG1；使用 `--alg_num 2` 或显式
`flagsparse_spsv_analysis_sell` + `flagsparse_spsv_solve_sell` 生命周期可启用
slice-cooperative ALG2 路径。TRANS/CONJ 使用专用反向依赖 kernel，且不接受
`--alg_num` 或 `--alg2-workers`。

```bash
python tests/test_spsv.py --synthetic
python tests/test_spsv.py <目录/> --csv-csr spsv.csv
python tests/test_spsv.py <目录/> --csv-coo out.csv     # 列与 CSR 相同
pytest -q -s tests/test_spsv_sell.py
python tests/test_spsv_sell.py <目录或文件.mtx> --csv sell_alg1.csv --slice-size 32 --alg_num 1
python tests/test_spsv_sell.py <目录或文件.mtx> --csv sell_alg2.csv --slice-size 32 --alg_num 2
python tests/test_spsv_sell.py <目录或文件.mtx> --csv sell_unit.csv --unit-diagonal
python tests/test_spsv_sell.py --csv sell_trans.csv --dtype float32 --slice-size 32 --ops TRANS <目录或文件.mtx>
python tests/test_spsv_sell.py --csv sell_conj.csv --dtype complex64 --slice-size 32 --ops CONJ <目录或文件.mtx>
python tests/test_spsv_sell.py <目录或文件.mtx> --csv sell_complex.csv --dtype complex
# 可选 ALG2 调优：追加 --alg2-workers 32|64|128|256|512
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
