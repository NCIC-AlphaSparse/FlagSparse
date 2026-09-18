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

> **要在非 NVIDIA 加速卡上做 bring-up？** 从 [`prompt.md`](prompt.md) 开始，再读
> `docs/<后端>.md`（那台机器怎么跑）和 `modified/<后端>.md`（前人改了什么、为什么）。
> 本文记录的是 CUDA 参照路径。

## 安装

```bash
pip install . --no-deps --no-build-isolation
```

离线时可加 `--no-build-isolation` 避免拉取构建依赖。

运行时依赖（按需安装）：

```bash
pip install torch triton cupy-cuda12x
```

## 复现交付测试（40 个变体 × 30 个矩阵）

交付物就是这一轮跑出的 `summary.json`：`conf/operators.yaml` 里登记的 **40 个交付变体**
（`delivery_variants`；交付清单共 42 个，`sddmm_csr` 的 c32/c64 等复数内核）的**精度和性能**，
性能用一个装有 **30 个 MatrixMarket 矩阵**的目录测，热身 5 次、计时 20 次。各后端用同一条命令，
下表列出各后端要改的地方，完整做法见各 `docs/<后端>.md` 的"交付复现"一节。

```bash
export PYTHONPATH=$PWD/src
python3 -c "import flagsparse; print(flagsparse.__file__)"   # 必须是 <仓库>/src/flagsparse/__init__.py

timeout -s KILL 21600 python3 run_flagsparse_pytest.py \
  --phase both --mode normal --delivery-only --gpus 0 --timeout 3600 \
  --benchmark-input <30 个 .mtx 所在目录> --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir pytest_results_<后端>_delivery

python3 tools/delivery_table.py pytest_results_<后端>_delivery      # 打印 40 行结果表，缺变体时退出码为 1
```

| 参数 | 为什么不能省 |
|---|---|
| `--delivery-only` | 只跑 40 个变体背后的 11 个父算子，并把每个 benchmark 的 sweep 收窄到交付范围（int32 索引、`non` 操作）。不加会读 18 个算子的超集，并跑完整的 int64/trans/conj 组合 |
| `--mode normal` | `quick` 每类 shape 只留一个，少跑约四成用例，还恰好跳过两个历史上出过问题的用例 |
| `--timeout 3600` | 每个算子每个阶段的超时（逐矩阵运行的路径是每个矩阵）。默认 `0` 表示内核挂住就永远等下去 |
| 外层 `timeout -s KILL` | 三角求解死锁时进程卡在驱动里，Ctrl-C 送不进去 |

| 后端 | 运行前设置 | 与上面命令的差别 | 加速比的分母 | 精度参考 | 完整做法 |
|---|---|---|---|---|---|
| CUDA | 无（需 `pip install cupy-cuda12x`） | 无 | cuSPARSE（CuPy） | cuSPARSE + torch | 本节 |
| DCU / ROCm | `pip install hip-python` | SpSV/SpSM 可能死锁（结果记为 `Timeout`） | hipSPARSE | hipSPARSE + torch | [docs/DCU.md](docs/DCU.md) 0.5 节 |
| 沐曦 C550 | `FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_VENDOR=none` | 加 `--benchmark-args=--no-cusparse`；SDDMM 的 K sweep 要 `--timeout 4500` | PyTorch | SciPy（CPU） | [docs/MACA.md](docs/MACA.md) 0.5 节 |
| 摩尔线程 | `FLAGSPARSE_BACKEND=mthreads` | 改用 `run_flagsparse_split_delivery.py`（性能取自 C API） | muSPARSE（C API） | SciPy（CPU） | [docs/MUSA.md](docs/MUSA.md) 0.5 节 |
| 昇腾 910B | CANN 的 `set_env.sh`；`FLAGSPARSE_BACKEND=ascend FLAGSPARSE_ASCEND_VENDOR=torch` | 只能用 `--gpus 6,7` | PyTorch-NPU（5 个算子；其余只做能力探测） | SciPy（CPU） | [docs/ASCEND.md](docs/ASCEND.md) "交付复现" |
| 昆仑芯 XPU | `FLAGSPARSE_BACKEND=xpu FLAGTREE_BACKEND=xpu TRITON_BACKEND=xpu` | 无 | PyTorch-XPU（5 个算子；其余只做能力探测） | SciPy（CPU） | [docs/XPU.md](docs/XPU.md) 1.5 节 |

**`86a09cd`（2026-09-18）之前跑出的结果不能拿来比较**：旧版把 int64、trans/conj 的行也平均进了
`..._int_non` 变体，还把没有加速比的行当作 0。请用当前 runner 重跑，见 `prompt.md` 第 2 节。

`summary.json` 里各状态的含义：`Passed` / `Failed` 为实测结果；`Timeout` 表示算子跑了但触发了
`--timeout`；`NotFound` 表示没有该变体的行（没跑，或本后端没有配置）；`NoBaseline`（C API）表示
内核跑通且精度通过，只是厂商库没有这个 dtype/算子可比。CUDA 上（RTX 5090，2026-09-17，sweep
收窄之前）全程 79 分钟，最慢的是 SpSV 和 SpGEMM，各约 20 分钟。

## 后端（已注册 8 个）

FlagSparse 按检测到的运行时对**厂商参考实现与基线**进行分发；Triton 内核本身在各后端保持不变。

| 运行时 | 判定方式 | 厂商稀疏库 | Python 绑定 |
| --- | --- | --- | --- |
| NVIDIA CUDA | `torch.version.hip is None` | cuSPARSE | CuPy（`cupy-cuda12x`） |
| DCU / ROCm | `torch.version.hip is not None` | hipSPARSE | `hip-python` |
| MetaX / MACA（C550） | 见 `_detect_maca_runtime()` | CuPy 真装了就用，否则 `torch` —— **探测而非假定**：做 bring-up 的那台 C550 没有 CuPy | CuPy |
| 摩尔线程 / MUSA | `torch.musa` 可用 | **默认无** —— 那里 `torch.sparse` 能建 CSR/COO 张量但没注册 sparse matmul，MTT S5000 实测 | torch_musa |
| 昇腾 / CANN（910B） | `torch.npu` 可用 | **`torch`** —— ops-sparse 仍可用环境变量选回做 A/B | torch_npu |
| 昆仑芯 XPU | 厂商插件可 import（`torch_xmlir` / `torch_xpu`） | **无** —— XDNN 是固定算子集而非描述符 API，没有可绑的通用入口 | torch_xmlir |
| 燧原 GCU | `torch_gcu` 可 import | 暂无 | torch_gcu |
| 寒武纪 MLU | **只能由环境变量选中**，从不自动探测：它是通用备用槽位，没有自己条目的厂商可以从这里走 | 暂无 | torch_mlu |

后三个没有自己的算子内核：`backends/{xpu,gcu,mlu}/` 下只有 `__init__.py`，走共享实现。
昆仑芯那条"插件必须可 import"不是多余的谨慎 —— 上游 PyTorch 给 Intel GPU 也装了
`torch.xpu` 命名空间，只看命名空间会把一块 Intel 卡认成昆仑芯。

> ⚠️ **摩尔线程与昇腾不是 CUDA 兼容的**：torch 把它们暴露成独立设备类型
> （`musa` / `npu`），`torch.cuda.*` 和 `Tensor.is_cuda` 都不适用。
> **设备抽象迁移已完成**：算子模块里**不再有任何 `torch.cuda.*` 直接调用**，
> 一律走 `_ACCEL.*` 和 `_is_accel_tensor()`；仅剩的 11 处在 `_common.py` 自己的运行时
> 探测里（设备名判定与 OOM 兜底）。在 CUDA / ROCm / MACA 上 `_ACCEL is torch.cuda`，
> 因此对这三个后端是**逐字等价**，内核一行未改。`tests/` 下直接跑的 benchmark 脚本
> 从 `tests/benchmark_utils.py` 取同一对名字。

MACA 与 CUDA 源码兼容，机器上 `torch.version.cuda` 有值而 `torch.version.hip` 为 `None`，
**靠 ROCm 那套判据分不出 MetaX 和 NVIDIA**。检测按顺序尝试：`FLAGSPARSE_BACKEND` 环境变量、
MetaX 专有的 `torch.version` 属性、MACA SDK 环境变量（`MACA_PATH` / `MACA_HOME`）、
设备名（先认厂商串，再认 `c550` / `c500` 型号串）。
**在真机上先显式指定，等自动探测确认无误后再依赖它：**

```bash
export FLAGSPARSE_BACKEND=metax     # cuda | rocm | metax | mthreads | ascend | xpu | gcu | mlu
export FLAGSPARSE_MACA_MODEL=c550   # 覆盖型号检测
export FLAGSPARSE_MACA_VENDOR=none  # CuPy 不可用时跳过厂商基线

# 摩尔线程 / 昇腾的基线选择
export FLAGSPARSE_MTHREADS_VENDOR=none       # none（默认）| torch | musparse
export FLAGSPARSE_ASCEND_VENDOR=torch        # torch（默认）| ops_sparse | none
export FLAGSPARSE_XPU_VENDOR=torch           # torch（默认）| none

# 正确性参考。CUDA 与 ROCm 比对各自的厂商库加 torch；其余后端一律比对 CPU 上的 SciPy ——
# 那些平台上的 torch.sparse 本身就是被测对象，不是参考。在 CUDA 机器上强制 "scipy"，
# 是这条路径在送到没人能跑的硬件之前唯一的验证办法。
export FLAGSPARSE_ACCURACY_REFERENCE=auto    # auto（默认）| scipy | torch
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

SpSV 和 SpSM 目前在 DCU 上会 GPU 内核死锁（见 [docs/DCU.md](docs/DCU.md)
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

**5. 统一运行器。** 交付测试用[复现交付测试](#复现交付测试40-个变体--30-个矩阵)一节的命令。
其中的 `--timeout 3600` 保证死锁的 SpSV/SpSM 不会让整轮停摆：该算子记为 `Timeout`，然后继续往下跑。
注意 `--gpus 0,1` 单独用没有用 —— 它只是把算子分成两条队列，含 SpSV/SpSM 的那条照样要等到超时。

DCU 上的完整验证流程（环境检查、旧安装包陷阱、如何确认真的走了 hipSPARSE、
已知限制、排查速查表）见 [docs/DCU.md](docs/DCU.md)。

### 在 MetaX / MACA C550 上跑测试

**交付测试**（40 个变体）用[复现交付测试](#复现交付测试40-个变体--30-个矩阵)一节的命令，
按表中沐曦那一行设置；确切命令见 [docs/MACA.md](docs/MACA.md) 0.5 节。

下面这条是另一回事：连同 CSC、BSR 等其他算子一起的**全量 sweep**，PyTorch 基线，30 个
MatrixMarket 矩阵。由于当前内核在该平台可能卡住，暂不包含 SpSV 和 SpSM；SpMM BELL 也暂时
排除——单个矩阵的耗时可能超过其余算子之和：

```bash
PYTHONPATH=src python -u run_flagsparse_pytest.py --phase both --mode quick --gpus 0 \
  --ops gather,scatter,spmv_csr,spmv_coo,spmv_csc,spmv_bsr,spmm_csr,spmm_coo,spmm_bsr,spmm_csc,spgemm_csr,sddmm_csr \
  --benchmark-input /root/gcx/matrix --benchmark-warmup 5 --benchmark-iters 20 \
  --benchmark-args=--no-cusparse --op-benchmark-args=spmv_bsr=--resume \
  --timeout 7200 --results-dir pytest_results_metax_runner_both_w5_i20
```

`--benchmark-args` 会追加到所有性能脚本；仅某个算子支持的参数使用可重复的
`--op-benchmark-args=算子名=参数`。上例只向 BSR 脚本传入 `--resume`，不会影响其他算子；
参数部分含空格时再整体引用，例如 `--op-benchmark-args="spmv_bsr=--dtypes float64"`。
BSR 脚本会保留已完成的 `PASS`/`FAIL` case、重试此前的 `ERROR` case，
并且只对精度为 `PASS` 的行给出 `bsr_speedup_vs_pytorch`。

结果目录包含各算子的精度、性能文件和合并汇总。后端检查、基线说明和 C550 已知限制见
[docs/MACA.md](docs/MACA.md)。

在 MetaX 上，`test_spmm_csc.py` 使用 PyTorch 原生 CSC 作为性能基线：直接构造
`torch.sparse_csc_tensor` 后调用 `torch.sparse.mm`。CSC 格式构造不计入计时，CSV 中记录为
`pytorch_ms`，并计算 `triton_speedup_vs_pytorch`。`--no-cusparse` 只关闭可选的厂商基线，
不会关闭 PyTorch CSC 基线；COO 路径仍只用于精度参考。

### 在昇腾 910B 上跑测试

只能使用 6、7 号 NPU。交付测试用[复现交付测试](#复现交付测试40-个变体--30-个矩阵)一节的命令，
按表中昇腾那一行设置；确切命令、以 PyTorch-NPU 为基线的 5 个算子（其余只做能力探测）、
环境检查和已知限制见 [docs/ASCEND.md](docs/ASCEND.md)。

## 目录说明

| 路径 | 是什么 |
| --- | --- |
| `src/flagsparse/` | 核心包。`sparse_operations/` 下每个算子只有**一份共享实现**；`sparse_operations/backends/<后端>/` 是**默认为空**的覆盖层，只有真正分叉的文件才会出现在里面 |
| `conf/operators.yaml` | 算子清单，也是 **40 个已登记交付变体**（交付清单共 42 个，`sddmm_csr` 的 c32/c64 等复数内核）（`gather_f32_int`、`spmv_csr_f32_int_non` …）的单一真源。Python 侧和 C API 侧都通过 `tools/delivery_variants.py` 读它 |
| `run_flagsparse_pytest.py` | 统一 runner：逐算子跑精度与性能，产出按变体名做键的 `summary.json` / `summary.csv` / `result.html` |
| `tests/pytest/` | 精度用例（各后端共用；golden reference 恒在 CPU） |
| `tests/ci/` | 策略与契约测试，不需要 GPU |
| `tests/*.py` | 逐算子的独立 benchmark CLI，runner 的性能阶段也调用它们 |
| `benchmark/` | 后端能力探测与厂商基线脚本 |
| `capi/` | 兼容 cuSPARSE 的 C API 封装、它的 CTest 套件与各后端基线 |
| `docs/` | 一个后端一份：那台机器怎么跑。索引是 `docs/README.md` |
| `modified/` | 一个后端一份改动台账：在那台机器上改了哪个文件的哪个函数、为什么 —— 各后端之间传描述，不传文件 |
| `prompt.md` | 新后端上手从这里开始 |

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
