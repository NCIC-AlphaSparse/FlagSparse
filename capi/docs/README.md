# FlagSparse C API —— 后端文档

本目录一个后端一份，内容是**环境检查、构建、跑精度和性能测试**，以及每个平台上
已经踩到的坑。想先看整体设计（re-export 而非拷贝内核、bufferSize 而非私自分配、
那五个会折叠掉的 constexpr），回 [../README.md](../README.md)。

| 文档 | BACKEND | 平台 | 这一层的状态 |
|---|---|---|---|
| [CUDA.md](CUDA.md) | `CUDA` | NVIDIA | **可构建**，全量验证过 |
| [MUSA.md](MUSA.md) | `MUSA` | 摩尔线程 | **可构建**（adaptor 已有） |
| [MACA.md](MACA.md) | `MACA` | 沐曦 | 预留槽位，缺 adaptor.cpp |
| [DCU.md](DCU.md) | `HCU` | 海光 | 预留槽位，缺 adaptor.cpp |
| [ASCEND.md](ASCEND.md) | `NPU` | 昇腾 | 预留槽位，缺 adaptor.cpp |

还有三个 BACKEND 名字存在但本目录没有单独文档：`GCU`（燧原）、`MLU`（寒武纪，
**通用备用槽**）、`IX`（天数智芯）—— 都是预留槽位，填法与上面三个相同，见
[../README.md](../README.md) 的平台矩阵。`XPU`（昆仑芯）是另一回事：
`deps/libtriton_jit` 根本没有 XPU 后端，所以 C wrapper 现在编不了它；Python 侧可以。

## 两层，别混淆

这个目录是 C API；它 re-export 的内核住在同一源码树顶层的 FlagSparse `src/` 里。所以每个
平台有**两层**要验证，失败的含义也不同：

| 层 | 测什么 | 怎么跑 | 挂了说明 |
|---|---|---|---|
| Triton 内核 | 内核本身能否在这个后端编译、算对 | `ctest -R pytest` | 该平台的 Triton/编译器问题，C API 一样用不了 |
| C API | 调度层：校验、路由、launch 配置、scratch | `ctest -R accuracy` | 本仓库的问题 |

`ctest --test-dir build` 两层一起跑。`ctest -R <算子名>` 把一个算子的三份
（accuracy / benchmark / pytest）一起选出来 —— **先看 pytest 那份**：内核编不出来的话，
C API 的失败只是它的回声。

## 后端专属测试入口

Python/Triton 与 C API 的测试策略现在按后端拆在同树中：

```
tests/backends/<cuda|rocm|maca|musa|ascend|xpu>/suite.json
capi/ctest/backends/<cuda|rocm|maca|musa|ascend|xpu>.cmake
```

每个目录或文件拥有该后端的运行时 selector、C API 状态和测试入口；算子断言仍在共享的
`tests/pytest` 与 `ctest/{accuracy,benchmark}` 中，避免六份 oracle 漂移。Python 测试用：

```bash
python3 tools/run_backend_tests.py --backend cuda --phase accuracy --mode quick
python3 tools/run_backend_tests.py --backend maca --phase both --ops spmv_csr,spmm_csr
```

该入口从 profile 设置 `FLAGSPARSE_BACKEND`。C API 成功配置后，每个注册的 CTest 都有
后端 label，例如 CUDA：

```bash
ctest --test-dir build -L cuda --output-on-failure
```

当前只有 `cuda`、`musa` profile 允许注册 C API cases；`rocm`、`maca`、`ascend`、`xpu`
明确记录为 Python-only，直到对应 C API adaptor 可构建并完成 profile 状态更新。

## 交付变体清单

Python runner 与本目录的 `tools/write_summary.py` 都读取同一份顶层清单：
`../conf/operators.yaml` 的 `delivery_variants` 字段。当前登记 40 个变体（交付清单共 42 个，`sddmm_csr` 的 c32/c64 等复数内核，见下文"三份文件，三种口径"），两个 `summary.json` 的
`result` key 集合完全相同。未运行的变体保留 key 并标为 `NotFound`，不从其他 dtype 或
算子借用结果。后续扩展时先向该清单加入一个带 `id`、`operator`、`format`、`dtype` 的条目，
再接测试和 benchmark 生产端。

## 这个库离不开 FlagSparse

**运行时硬依赖，不是"建议一起用"。** `flagsparse_codegen/*.py` 里是真的 import：

```python
from flagsparse.sparse_operations.spmm_csr import _spmm_csr_real_kernel
```

libtriton_jit 按路径加载这个 shim、`getattr` 取出内核对象——拿到的**就是 Python 包
自己用的那同一个 JITFunction**。内核因此只有一份、不可能漂移；代价是那一份必须在。

依赖的不是"那个仓库"，而是**可导入的 `flagsparse` 包**。分层看：

| 调用 | 没有 Python 包时 |
|---|---|
| `flagsparseCreate` / `CreateCsr` / `CreateDnMat` / `*_bufferSize` / 各种校验 | **正常工作**，纯 C++ |
| 真正要算的那一步（SpMM / SpSV / …） | `EXECUTION_FAILED`，`last_error` 里是 ImportError |

三种配置实测（`flagsparseGetLastErrorString` 就是为这类问题加的）：

| 配置 | 结果 |
|---|---|
| 默认的同树 `FLAGSPARSE_PYTHON_SRC` | 算对 |
| 指向空目录 | `EXECUTION_FAILED` + `ImportError: ... Something else is shadowing it.` |
| 指向一份 pip 装的旧包 | `CompilationError`：内核签名对不上 |

### 旧副本会静默获胜——除非拦住

第二行那个报错是**故意的**。`_bootstrap` 把 `FLAGSPARSE_PYTHON_SRC` 放到 `sys.path`
最前、清掉 `sys.modules` 里的 `flagsparse.*`，然后**验证解析到的
`flagsparse.__file__` 确实在它之下**，否则直接抛错。

没有这道验证的话，装在 `site-packages` / `dist-packages` 里的旧副本会赢，而报出来的
症状只是 JIT 里一句 `number of argument mismatch: Actual(11), Function Definition(8)`
——**两个副本的名字一个都不提**。这个坑真踩过，排查成本很高。

所以：**`FLAGSPARSE_PYTHON_SRC` 指错会明确报错，但机器上存在旧副本仍可能干扰
Python 层的直接调用**（那一层不经过 `_bootstrap`）。跑 `ctest -R pytest` 前值得确认
`python3 -c "import flagsparse; print(flagsparse.__file__)"` 指向的是你以为的那份。

### 同树组件必须保持配套

```
FlagSparse（Python/Triton 内核）  src/flagsparse/
capi（C API）                     capi/
libtriton_jit（JIT 桥，子模块）   capi/deps/libtriton_jit/
```

依赖是**单向**的：`capi/` 通过 `FLAGSPARSE_PYTHON_SRC` 找到顶层 FlagSparse 内核，反过来
FlagSparse 不依赖 `capi/`，单独跑 `run_flagsparse_pytest.py` 照常工作。

C API 为对齐 cuSPARSE，往 **FlagSparse 侧的内核**里加了参数和 constexpr。它们都设计成
在 Python 路径上折叠掉（传 1/0、`False`），所以 Python 侧生成的代码不变——但**旧的
FlagSparse 配新的 c_fs 会直接 `CompilationError`**，因为 C 侧拼的签名多出参数。

本轮改动涉及的模块：

| 模块 | 加了什么 |
|---|---|
| `spmm_csr` / `spmm_coo` | `alpha` `beta` `HAS_BETA`，COO 另加 `SEG_IS_ROW` |
| `spmv_coo` / `spmv_csc` / `spmv_bsr` | `alpha`（`beta` 走前置件），COO 加 `SEG_IS_ROW`，BSR 加 `SEG_FROM_GRID` |
| `spsv` | `alpha`、`SCAN_BACKWARD`（CSR）、SELL 四个内核的 `alpha` |
| `spsm` | `SPSM_ASCEND_DISPATCH` 及 Ascend 退路 |
| `_common` | `_dense_scale_kernel`、`_dense_copy_kernel`（原子路线的 beta 前置件、SpSM 的跨步拷贝） |

**怎么确认配套**：`ctest --test-dir build` 会把 Python 侧 17 个套件一起跑
（`ctest -R pytest`），不配套当场暴露，而不是等某个算子在生产里报 `CompilationError`。

### 部署要带的东西

```
libflagsparse.so + flagsparse.h        本仓库产物（install 时装到 lib/ 和 include/）
flagsparse_codegen/*.py                本仓库的 shim（装到 share/flagsparse/）
可导入的 flagsparse 包                 FlagSparse 那边，版本要对得上
triton + torch                         内核在运行时 JIT 编译，见下
FLAGSPARSE_PYTHON_SRC                  构建期烤默认值，运行时可覆盖
```

进程里会有 `libpython3.x.so` 和 `libtorch*.so`——**内核是运行时 JIT，不是 AOT 进 .so 的**。
后果：首次遇到某个 (dtype, constexpr) 组合要付一次编译，之后走 Triton 磁盘缓存。
性能测试里的 warmup 就是为这个，而不是走过场。

### 想去掉这个依赖？

理论上有 AOT 一条路（`triton.tools.compile` + `triton.tools.link`），但代价大到需要
先做取舍：constexpr 会被特化死、grid 要调用方自己算、`gfx936`（DCU）编译器**直接不
支持**，而且 113 个内核 × dtype × constexpr 组合必须先定一个很窄的白名单
（单个特化 192 KB CUDA / 56 KB HIP）。现状是有意选的，不是没想过。

## 每个平台都一样的部分

```bash
# 1. 环境：默认使用同一份源码树的 ../src
export FLAGSPARSE_BACKEND=<cuda|rocm|metax|mthreads|ascend|xpu|gcu|mlu>

# 2. 构建
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DBACKEND=<CUDA|MUSA|MACA|NPU|HCU> \
      -DPython_ROOT="$(dirname "$(dirname "$(which python3)")")" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. 精度
ctest --test-dir build -R accuracy --output-on-failure

# 4. 性能（JSON 落盘，每份带 backend + arch）
#    指向真实矩阵语料，并自动对厂商基线求加速比
FLAGSPARSE_MATRIX_DIR=/path/to/mtx FLAGSPARSE_BENCH_OUT=./bench \
    ctest --test-dir build -R benchmark
```

`FLAGSPARSE_MATRIX_DIR` **不设也能跑**：会退回合成形状，并把每一行打上
`corpus=synthetic`。这是故意的——没有语料的开发者仍然得到一个能用的 `ctest`，而读
JSON 的人不可能把那些行误当成真实数据。

**`FLAGSPARSE_PYTHON_SRC` 是构建期烤进去的**，运行时可用同名环境变量覆盖。指错了
不会静默走错 —— `_bootstrap` 会验证解析到的 `flagsparse.__file__` 确实在它下面，
否则直接报错。装在 site-packages 里的旧副本曾经在这里赢过一次，报出来的症状是
JIT 里一句 `number of argument mismatch`，两个副本的名字一个都没提。

## 真实矩阵语料扫描

`ctest -R benchmark` 扫的是合成形状，用来抓 launch 配置的回归。真实矩阵是另一个问题：
**随机矩阵的行长是均匀的，而真实矩阵的行长偏斜才是决定一个 launch 启发式猜得准不准的
东西**。所以另有一个跨算子的工具：

```bash
./build/tools/matrix_sweep --matrix-dir /path/to/matrices --json sweep.json
# 常用开关
  --op spmv --op spsv        # 只扫某些算子，可重复
  --matrix ecology1          # 只扫某些矩阵，可重复
  --warmup 2 --iters 10      # 1140 个格子，默认比 ctest 的 10/100 轻
  --budget-mb 12000          # 稠密操作数超过这个预算就跳过，避免 OOM
```

它对**每个 (变体, 矩阵) 组合**产出一行，状态分五类——这是它和一次性基准的区别：

| 状态 | 含义 |
|---|---|
| `ok` | 测到了 |
| `not_supported` | 这个后端上算子拒绝了它。**记下来而不是丢掉** |
| `skipped_memory` | 稠密操作数放不下预算。**这是扫描工具的限制，不是算子的**，所以单独一类 |
| `skipped_shape` | 矩阵喂不了这个算子（SpGEMM 的 A×A 要求 A 是方阵） |
| `failed` | 跑了但出错，状态名和 `last_error` 都留着 |

读 Matrix Market 时**对称矩阵会展开**（文件只存一个三角）。不展开的话求解器只会被
测到一半的工作量，而那种错误静悄悄——这正是扫描要避免的。

三角求解（SpSV / SpSM）用的是矩阵的**下三角加一个强制对角占优的对角线**：真实矩阵的
三角往往病态得离谱，不这么做量到的是矩阵而不是内核。

工具不注册进 ctest：一次扫描要几分钟，而且依赖仓库里没有的矩阵。


## 算子清单：从哪来，怎么决定报告出什么

这一节是**测试范围的唯一真源**，改清单就改范围，不用动测试代码。

### 三份文件，三种口径

| 文件 | 内容 | 作用 |
|---|---|---|
| `算子列表注册修改.xlsx`（仓库外） | 40 个变体，"新算子列表"一列；**漏了 gather/scatter 的 f16**（见下） | 交付口径的原始来源 |
| `算子对比结果_合并变体.csv`（仓库外） | 115 个变体，含 trans/conj/col 布局 | 对齐 cuSPARSE 的完整矩阵，未来目标 |
| `conf/operators.yaml`（本仓库） | 22 个算子组 → 60 个变体 | 实现细节 + 归属标记 |

三个数不是包含关系，别混：**115** 是完整口径（含 `non`/`trans`/`conj` 与 row/col 布局），
**60** 是本仓库能生成的变体（已登记交付 40 + 保留 20）。

**交付清单是 42 个，已登记、会出报告的是 40 个**，两者的差别说清楚：

| | 个数 | 组成 |
|---|---|---|
| 交付清单 | **42** | xlsx 的 40 + `gather_f16_int` + `scatter_f16_int`。xlsx 漏写了这两个 f16，属于清单的笔误（2026-09-18 与需求方确认），它们**是交付算子**，不是清单外的附加项 |
| 已登记（`delivery_variants`） | **40** | 交付清单 42 − `sddmm_csr_c32_int_non_non_row` − `sddmm_csr_c64_int_non_non_row` |
| 待补 | **2** | 上面两个 sddmm 复数变体：交付清单要，但**还没有复数 SDDMM 内核**（实测 `x dtype must be torch.float32 or torch.float64`）。内核补上后再在 `delivery_variants` 里登记，报告随之变成 42 行 |

所以现在报告里的"40/40"读作**交付清单 42 个中已实现的 40 个全部通过**，不是"交付清单全部通过"。

### `reporting` 字段

`conf/operators.yaml` 每个算子带一个 `reporting`：

```yaml
- id: spmv_csr
  reporting: delivery          # 在交付清单里，进默认报告
- id: spmv_csc
  reporting: retained          # 有实现、ctest/accuracy 有覆盖，但不进默认报告
- id: spsm_csr
  reporting: delivery
  delivery_dtypes: [f32, f64]  # 内核也支持复数，但交付清单只要这两档
- id: sddmm_csr
  delivery_gaps: [c64, c128]   # 交付清单要，但没有内核
```

`retained` **不是"删掉"**：那些算子有实现也有通过的精度测试，从清单里删会让已有的测试
变成没有归属的孤儿。它们照常构建、照常测试，只是被挡在默认报告之外。

`delivery_gaps` 记录"清单要但没实现"的部分。**不把它们写进 `dtypes`** 是有意的：那样会
生成运行期必然失败的变体，读起来像回归，而不是"还没做"。

### 流程

```bash
# 1. 清单 -> 变体表（构建时由 CMake 自动执行，无需手动跑）
python3 tools/gen_variants.py --manifest conf/operators.yaml \
        --out build/ctest/generated/variants.inc

# 2. 跑测试（8 个二进制迭代变体表，而不是各自的硬编码表）
FLAGSPARSE_MATRIX_DIR=/path/to/mtx FLAGSPARSE_BENCH_OUT=./bench \
    ctest --test-dir build -R benchmark

# 3. 交付报告：默认只出 40 行；--all 看全部 60
python3 tools/report.py --bench-dir ./bench --csv delivery.csv

# 4. summary.json + result.html：与 FlagSparse 的 run_flagsparse_pytest.py 同 schema
#    零参数即可，默认读写 capi_results/
python3 tools/write_summary.py              # 同时产出 result.html（--no-html 可跳过）
python3 tools/write_html.py                 # 只重渲染 HTML

# 5. 一致性检查：清单与实现是否漂了（--strict 可进 CI）
python3 tools/check_manifest.py --bench-dir ./bench
```

**加一个算子 = 改 YAML + 重新构建**，测试代码不动。这也是 40 → 115 的路径：
`gen_variants.py` 改成按 CSV 的切分展开即可，但那需要 benchmark 增加 `trans`/`conj`
方向和 col 布局的代码路径，那是实打实的工作量。

### 产物放在哪，为什么和 Python 侧分开

```
FlagSparse/pytest_results/summary.json     25 个算子条目（Python 侧）
c_fs/capi_results/summary.json             40 个变体条目（C API 侧）
c_fs/capi_results/result.html              同上，变体粒度的表格
```

**同名不同粒度**，所以分目录：共用一个目录意味着后写的静默覆盖先写的；而 CI 是按目录
整个上传的，分开才能两份都留住。

`summary.json` 的 `result` **按变体名做键**（`spmv_csr_f32_int_non`），不是按算子聚合。
这不是另创的格式——算子列表注册修改.xlsx 的"新算子列表"本来就是这么命名的，所以 40 行
是从格式里自然落出来的，FlagGems 那套 json/html 结构一个字都不用改。

HTML 没有复用 `run_flagsparse_pytest.py:2994` 的生成器，原因有两条：它吃的是 runner 的
内存 `results` 列表而不是 `summary.json`；而且它的 `HTML_SPEEDUP_DTYPES` 是写死的元组，
**没有 fp64 也没有复数**——这里一半的变体会无处落脚。所以布局照搬（环境表、状态过滤、
列排序），列跟着数据走。

### 变体没测到时会怎样

清单声明了、但 benchmark 没有对应 operand builder 的变体，**会产出一行
`not_implemented_in_test`**，带着缺什么的说明，而不是静默缺席。这一类**单独计数，不算进
`failed`**——测试的缺口不是库的缺陷，混在一起会把库报成坏了。

`check_manifest.py` 会把清单与实现之间的每一处不一致列出来。这个机制建立之前，两者漂了
**34 处**没有任何东西发现。

### 测了、过了，但没有厂商基线：`NoBaseline`

`summary.json` 里 `performance.status` 的取值：

| 状态 | 含义 |
|---|---|
| `Passed` | 有加速比（基线可用、答案通过） |
| **`NoBaseline`** | 内核每一行都跑了、精度都过了，只是厂商库没有这个 dtype/算子可比，算不出加速比。`data.<dtype>.reason` 写着厂商库给的原因，比如 `muSPARSE: Gather dtype unsupported` |
| `Skipped` | 真的没跑出结果（没有 `status: ok` 的行） |
| `NotFound` | 登记了但这次没测到 |

`NoBaseline` 是 2026-09-18 加的。之前这种情况被归成 `Skipped`：MUSA 上 gather/scatter f16 各 30 个矩阵
都有实测耗时，报告却显示"没跑"（`modified/MUSA.md` 第 13 节）。它**不算失败**，也不计入加速比均值。

## 厂商基线与加速比

加速比的分母必须是**用户本来会调的那个库**，否则这个数没有意义。所以基线是厂商自己的
稀疏库，走它的 generic（描述符）API——正是本库 C 头文件所对齐的那一套。

| 后端 | 基线 | 状态 |
|---|---|---|
| CUDA | cuSPARSE | **WIRED**，已在开发机编译并实测 |
| DCU | hipSPARSE | **WIRED**，按公开 API 写就，首次在 DCU 机器上编译才算验证 |
| MACA | 沐曦的稀疏库 | `UNNAMED`——库有，但头路径/库名在本仓库无可考来源 |
| MUSA | 摩尔线程的稀疏库 | `UNNAMED`——同上 |
| 昆仑芯 XPU | 无 | `NONE`——见 [XPU.md](XPU.md)，两条独立阻塞 |
| 昇腾 Ascend | 无通用稀疏 API | `NONE`——可比的数来自 torch_npu/aclnn，形状不同 |

`UNNAMED` 和 `NONE` 是**关于厂商的事实，不是待办**。补一个 `UNNAMED` 槽位只需在
`ctest/baseline/CMakeLists.txt` 里填两个变量、放一个 35 行的前缀文件
(`ctest/baseline/cuda/baseline.cpp` 就是模板)。这里**不猜库名**：猜错会在链接期报一个
既不指明厂商也不指明错误的缺失符号，比没有这个条目更糟。

cuSPARSE 和 hipSPARSE 是同一套 API 换前缀，所以实现体是共享的
`ctest/baseline/_template.inc`，和 `src/adaptor/backend/_template.inc` 同构。

### 加速比只统计精度通过的矩阵

这是硬规则,不是建议：

* 精度判的是 **CPU fp64 oracle**，**不是基线**。厂商内核和我们可能朝同一个方向错，
  互相对上就会被当成通过；
* `accuracy == "pass"` 时才写 `speedup`。`fail` 记原因并跳过，`unchecked`（结果太大、
  读回不划算）同样不写——没校验过的比值不该被平均进去；
* 汇总用**几何平均**，并带 `speedup_matrices` 说明它覆盖了多少个矩阵。这些是比值，
  一个小矩阵上的 40x 会把算术平均整个带偏；而不写覆盖数的话，一个漂亮的总比值会掩盖
  它其实只代表了三分之一的语料。

**任何一条失败路径都不中断扫描**：算子拒绝、算子报错、结果超容差、基线不可用、基线报错，
每一种都产出一行带原因的记录然后继续下一个矩阵。一个在第一个坏矩阵上停下的基准，报的是
它跑完的那部分语料，而不是你给它的语料。

## 一个真实发现：SpMV 的胜负取决于行长偏斜

同一个内核，在不同矩阵上相对 `torch.sparse.mm` 的表现差了 4.5 倍（fp32，本机实测）：

| 矩阵 | 平均行长 | 最长行 | 偏斜 | Triton / torch.sparse |
|---|---|---|---|---|
| ecology1 | 5.0 | 5 | 1.0 | 0.556 |
| cfd2 | 25.0 | 30 | 1.2 | 0.554 |
| cage12 | 15.6 | 33 | 2.1 | 0.656 |
| pkustk13 | 69.7 | 300 | 4.3 | 0.577 |
| mac_econ_fwd500 | 6.2 | 44 | 7.1 | 0.624 |
| **wiki-Talk** | 2.1 | **100022** | **47694** | **2.497** |

规则矩阵（科学计算、网格）上是 0.55–0.66x，幂律图上是 2.5x。**所以"SpMV 快不快"这个
问题没有单一答案，只有语料的答案**——这也正是要扫真实语料、且要连同 `speedup_matrices`
一起报的原因。一个混合语料上的单一几何平均会把 2.5x 的胜和 0.55x 的负抹成一个谁都不认识
的数。

另外注意基线选谁会移动结论：同一个 cfd2 上，裸 `cusparseSpMV` 是 0.0218 ms，而
`torch.sparse.mm` 是 0.0315 ms（1.44x 慢，它把 `(n,1)` 当 SpMM n=1 走，还多一层
dispatch）。FlagSparse Python 侧那套 harness 里名为 `run_cusparse` 的开关调的其实是
`torch.sparse.mm`——**名字是误导的**，读那边的数时要知道。

## 结果怎么读

精度输出每个二进制打一行后端横幅，性能 JSON 的 `env` 带 `backend` + `arch`：

```
[  BACKEND  ] cuda  flagsparse 1000
[   RATIO   ] fp32_n64 max_error_ratio=0.346
```

`max_error_ratio` 是**误差除以容差**，`<= 1` 即通过（规范 §6.3）。fp32 的稀疏累加
是顺序相关的，所以规范 §6.3.1 允许放宽档重试，输出会标成 `PASS(relaxed)` 并同时给出
两个比值 —— 不会把"已知数值性质"伪装成失败，也不会把它藏起来。

性能 JSON 每行带 `status`，以及 `accuracy` / `error_ratio` / `baseline_status` /
`baseline_ms` / `speedup`：

* `ok` —— 测到了；
* `not_supported` —— 这个后端上算子拒绝了这个配置。**记下来而不是丢掉**：某后端没接的
  算子会产出一份满是 `not_supported` 的 JSON，而缺失的文件读起来和通过没区别；
* `skipped_memory` / `skipped_shape` —— 放不下或喂不了，和算子本身无关，单独成类；
* `failed` —— 试了但出错，或者**结果超出容差**（`accuracy: "fail"`，`detail` 里带
  `error_ratio`）。

`baseline_ms` / `speedup` 为 `null` 时，`baseline_status` 和 `baseline_detail` 会说明
是哪一种：本后端没有接厂商基线、厂商库拒绝了这个配置、还是厂商调用出了错。
**编造的基线比没有基线更糟；在没校验过的结果上报加速比更糟。**
