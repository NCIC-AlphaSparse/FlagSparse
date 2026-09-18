# FlagSparse C API

cuSPARSE 对标的稀疏算子 C 接口库。对外是 `libflagsparse.so` + 唯一公共头文件
`include/flagsparse.h`；对内 C++ 调度层做参数校验、句柄/描述符管理，通过
[libtriton_jit](https://github.com/Artlesbol/libtriton_jit) 在运行时编译并启动
Python 侧的 Triton 内核。

架构与接口遵循《FlagSparse 算子库开发规范》，目录结构见规范 §3.2。

---

## 当前状态 —— 先说清楚做完了什么

**这是一个骨架加一条打通的竖切，不是完整移植。**

| 层 | 状态 |
|---|---|
| 公共 C API（`include/flagsparse.h`） | **完整**。52 个函数，覆盖规范 §4 的全部状态码、枚举、句柄、描述符与七类计算 API |
| C++ 调度层（句柄 / 描述符 / 状态映射 / JIT 桥 / 后端 adaptor） | **完整** |
| 构建（CMake、BACKEND 开关、install、Docker） | **完整** |
| 测试框架（gtest + ctest，精度 + 性能，JSON 落盘） | **完整** |
| **算子** | **18 / 22 组打通**：SpMV 全 5 组（`csr`/`coo`/`csc`/`bsr`/`coo_tocsr`）、SpMM 2 组（`csr`/`coo`）、SpSV **全 4 组**（`csr`/`coo`/`sell`/描述符全流程）、SpSM 2 组、`spgemm_csr`、`sddmm_csr`、`gather`/`scatter`、稀疏格式构造。其余 4 组是 SpMM 的性能变体，返回 `NOT_SUPPORTED` |

逐算子状态以 `conf/operators.yaml` 为准（`status: implemented` / `pending`）。

**分母有两个，别混用**：规范 §5.1 的对齐表是 **22 个算子组**，而合并 i32/i64 之后
是 **115 个变体**。上表的 18/22 是**组**。变体级的明细在
`算子对比结果_合并变体.csv` 里，该文件不在本仓库，所以 `operators.yaml` 的
`variants` 字段留空而不是猜——115 也不能从这里的 dtype/format 列表反推，
那些描述的是 Python 侧支持什么，和 CSV 的切分方式不是一回事。

为什么未实现的算子也有入口：规范 §4.4 要求不支持时返回状态码而不是 abort。
这样 **ABI 从第一天起就是完整的**——调用方可以链接最终形态的库，在运行时发现
算子可用性，而不是在链接期。移植一个算子 = 删掉 `src/api/pending.cpp` 里它的桩，
加一个形如 `src/ops/spmv.cpp` 的文件。

已验证（RTX 5090 / CUDA 12.8 / torch 2.9 / Triton 3.6）：

```
导出符号与头文件声明精确 1:1（53 个），0 个 C++ mangled 符号
ctest 全量 6/6 passed
  accuracy.spmv    14/14  CSR / COO / CSC / BSR 四种格式，含 CSC 的三个方向、
                          BSR 的 block_dim 1/2/4/8 与多段网格、COO 两条路线互相印证
  accuracy.sddmm    7/7   k 跨 BLOCK_K 阈值、两种 BLOCK_P 配置、行列主序 × op(A)/op(B)
  accuracy.spsv   11/11  上/下三角 × 单位/非单位对角 × CSR/COO/SELL × 实数/复数；
                          SELL 两条算法路线误差比逐位相同，slice 4/8/32 与不等长行都覆盖
  accuracy.spsm     6/6   含 1500 个右端项（跨多个 RHS tile）与全部布局组合
  accuracy.spgemm   5/5   发现的 nnz 与稠密参考逐个匹配，列有序，fp64 误差为 0
  accuracy.spmm    17/17  CSR + COO 全部严格容差通过：fp32 0.10~0.47、fp64 0~0.018、
                          c64 0.44、c128 0.0091~0.012；两种格式各覆盖行/列主序 ×
                          padded ld × op(B)=B^T × i32/i64，另含 COO 未排序拒收、
                          缺 buffer 拒收、跳过 preprocess 仍正确
  accuracy.gather  10/10  fp32/fp64/c64/c128 × i32/i64，逐元素精确比对
  accuracy.scatter  9/9   含 gather→scatter 往返，以及"未写位置必须不变"
  benchmark.spmv    2/2   density 0.1 达 193 GFLOPS
  benchmark.spmm    5/5   CSR 4096² density 0.1 / n=32 达 1261 GFLOPS，
                          同形状 COO 1768 GFLOPS
```

gather/scatter 是**逐元素精确比对**而不是带容差——它们不做算术，任何容差都只会
掩盖真实缺陷。复数通过实部/虚部交错数组原生支持（Triton 没有复数类型，这是唯一的
表示方式，也是 Python 侧一直以来的做法）。

### 已知环境问题：冷缓存下 scatter 编译失败

在本机（Triton 3.6 / CUDA）上，**Triton 缓存为空时**首次编译 scatter 内核会失败：

```
FLAGSPARSE_STATUS_EXECUTION_FAILED
  -- ModuleNotFoundError: No module named 'mlir'
     triton/experimental/tle/raw/mlir/runtime.py(8)
```

这是 **Triton 自身的打包缺陷**，不是 FlagSparse 的：
`triton/experimental/tle/raw/runtime.py` 把可选的 `tops` 后端包在 `try/except` 里，
却无条件 `import .mlir`，而 `.mlir` 需要的 MLIR Python bindings（`mlir.ir`、
`mlir.passmanager`）这个 build 没有附带。NVIDIA 后端的 `make_llir` 无条件 import
这条链路，所以任何走到那里的编译都会失败。

实测边界（不是推测）：

* 缓存预热后 **10/10 连续通过**，`ctest` 全量绿。
* 冷缓存下 `spmv` 和 `gather` 编译**不受影响**，只有 `scatter` 命中。为什么只有它，
  我没有查清，**不编造解释**。
* 塞一个 stub `mlir` 模块无效——它要的是真的 bindings。
* Python 包自己的 scatter 冷编译是通过的（76 个用例），所以问题不在内核源码。

规避：先跑一次让缓存预热，或安装 MLIR Python bindings。
`flagsparseGetLastErrorString()` 就是为这类问题加的——没有它，调用方只会拿到一个
光秃秃的 `EXECUTION_FAILED`，无从下手。

---

## SpMM：稠密排布不需要第二条代码路径

内核的两个稠密操作数都按**显式 stride** 寻址，于是：

* `FLAGSPARSE_ORDER_ROW` 与 `FLAGSPARSE_ORDER_COL` 是同一个内核，只是 stride 不同；
* `op(B) = B^T` 就是把 B 的两个 stride 对调，**不落地任何转置**；
* `ld` 大于实际列数（padded leading dimension）天然支持。

四者的交叉组合在 `accuracy.spmm` 里逐个跑过，误差比完全一致。所以
`conf/operators.yaml` 里原先记的"列主序是 Python 侧的已知缺口"，**在这条 C API
上不存在**。代价只体现在性能上：4096²/density 0.001/n=32，行主序 0.0130 ms、
列主序 0.0169 ms（+30%），因为列主序下沿 `offs_n` 的读取不再连续。这是实测，不是估计。

`op(A) = A^T` 仍返回 `NOT_SUPPORTED`——它需要先构出转置后的 CSR，那是这个算子还
没有的 prepare 步骤，不做近似。

### ACC_DTYPE：为什么 shim 里出现了两个 `@triton.jit`

`flagsparse_codegen/` 的规矩是**只 re-export、不抄内核**。SpMM 是第一个不能原样
re-export 的算子，原因在桥那一侧：libtriton_jit 的 raw-args 签名只能解析
bool / int / float 三种 constexpr，而 FlagSparse 的内核把 `ACC_DTYPE` 作为
**`tl.dtype` 对象**传入（8 个文件里共 28 个内核这么写）。签名里写 `fp32` 会解析成
`None`，`tl.zeros(dtype=None)` 直接抛异常。

选的办法是在 shim 里加一层薄 `@triton.jit` wrapper：它收一个 bool constexpr，
自己挑 `tl.float64 / tl.float32`，再委托给 Python 包里那个内核。

```python
@triton.jit
def spmm_csr_real(..., ACC_IS_FP64: tl.constexpr, ACCURACY: tl.constexpr, HAS_BETA: tl.constexpr):
    _spmm_csr_real_kernel(..., tl.float64 if ACC_IS_FP64 else tl.float32, ACCURACY, HAS_BETA)
```

没有选的路，以及原因：把 28 个内核的 `ACC_DTYPE` 降级成 bool，等于为了迁就一个
解析器废掉一个精度旋钮（以后想加 tf32 累加器就没位置了）；改 libtriton_jit 不是
我们该动的东西。

**代价是实测出来的，不是假设的**（`nnz/行`固定的合成矩阵，RTX 5090）：

* 输出 `torch.equal` 完全一致；
* Triton 会把 jit→jit 调用内联，wrapper 不出现在生成的 PTX 里。指令上**只有一处
  不同**——被内联的 `if row >= n_rows: return` 由 4 条降为 2 条、谓词取反；其余差异
  是基本块标号重排和 DWARF 内联帧表，不进指令流；
* 三个形状的中位耗时 -1.10% / -0.04% / -1.62%，即**从未变慢**。这几个数只说明
  "没有代价"，不要当加速比引用：单进程、无重复轮次。

内核体仍然只有**一份**。wrapper 里没有任何算术，调优或修 bug 依旧改 Python 包。

### alpha / beta

cuSPARSE 的 `C = alpha*op(A)*op(B) + beta*C` 做成**一次 launch**，办法和 SpMV 一样：
`alpha`/`beta`/`HAS_BETA` 加在 **Python 包的内核**里，不在 C 侧分叉。Python 自己的
调用点传 1/0 且 `HAS_BETA=False`，beta 项在编译期消失，生成的代码与改动前一致
（`tests/pytest` 的 SpMM 相关 327 个用例全绿可证：csr 67 + opt/alpha/csc 209 + coo 51）。复数没有 Triton 类型，所以
alpha/beta 按实部/虚部两个分量传，和操作数本身的交错表示一致。

`HAS_BETA` 必须是 constexpr：cuSPARSE 规定 `beta == 0` 时**完全不读 C**，否则未初始化
的 C 会经 `0 * NaN` 变成 NaN。

---

## SpMM COO：空行是这条路径唯一的真问题

内核体和 CSR 几乎一样（都是一行一个 program 的串行累加），难点只有一个：COO 里
**没有条目的行根本不存在**。Python 侧的 rowrun 把 `seg_starts` 建成"相同 row id 的
连续段"，所以空行压根不会被调度——那条路径先把 C 清零，所以没问题。但 C API 要算
`beta * C`，空行必须也被写到。

解法是给内核加一个 constexpr `SEG_IS_ROW`：

* `False`（Python 侧调用点）＝段是"row 的游程"，row id 从 COO 里读，行为与改动前
  逐字节一致；
* `True`（C API）＝`seg_starts` 是**长度 n_rows+1 的整行偏移数组**，段号就是行号，
  空行照样起一个 program，`beta * C` 自然覆盖到。

注意空行时 `start == end`，此时再去 `tl.load(row_ptr + start)` 会读到**下一行**的
row id——所以 `SEG_IS_ROW=True` 必须同时改掉 row id 的来源，两件事是一件事。

那个偏移数组就是一个 CSR indptr。它通过 **`flagsparseSpMM_bufferSize`** 要，由调用方
持有（CSR 报 0，COO 报 `(rows+1) * 4` 字节）：

```c
size_t buf = 0;
flagsparseSpMM_bufferSize(..., matA, matB, ..., &buf);   /* COO: (rows+1)*4 */
cudaMalloc(&d_buf, buf);
flagsparseSpMM_preprocess(..., d_buf);   /* 建表；COO 未按行排序在这里被拒 */
flagsparseSpMM(..., d_buf);
```

不调 `preprocess` 也对——第一次 `flagsparseSpMM` 会自己建表并记在描述符上。库内部
**不分配任何设备内存**，这是唯一需要 scratch 的算子，所以按 cuSPARSE 的规矩把它交给
调用方，而不是偷偷 malloc。

COO 必须按行有序（cuSPARSE 同样要求）。乱序返回 `INVALID_VALUE` 而不是算出
垃圾——建表时顺带就能发现，代价为零。同一行内的重复 (row, col) 条目会被自然求和。

只接了 rowrun 一条路。atomic 路需要先用一趟单独的 kernel 把 C 乘上 beta，而且按构造
就是非确定性的；alg1 的 bucket 路需要两个 prepare kernel。两者都还没接。

### 一个意外的测量结果：COO 比 CSR 快

同形状、同 JSON（5090，fp32，n=32）：

| 形状 | CSR | COO |
|---|---|---|
| 4096² density 0.1 | 0.0853 ms / 1261 GFLOPS | **0.0608 ms / 1768 GFLOPS** |
| 4096² density 0.001, n=128 | 0.0167 ms / 258 GFLOPS | **0.0116 ms / 369 GFLOPS** |

两个内核结构几乎相同，差的是 **launch 配置**：COO 用的是 Python 包 30 矩阵扫出来的
`BLOCK_NNZ=4`，并按设备 warp size 推 `num_warps`；CSR 的 `BLOCK_NNZ` 来自
warp/factor 启发式（4~32）。所以这条数据要读成"**CSR 的 launch 配置还有空间**"，
而不是"COO 是更快的格式"。没有顺手去调——那是另一件事，需要按矩阵扫过才算数。

---

## 为什么剩下 4 组不是照抄就能完事

Python 包里 `sparse_operations/` 共 39,594 行，其中 `@triton.jit` 内核体
7,139 行（18%），**其余 32,455 行（82%）是编排、校验、路由、回退和基准**。

C++ wrapper 要替换的正是这 82%。内核本身**一行都不用改**——`flagsparse_codegen/`
直接承接 `.py` 文件，`libtriton_jit` 按路径加载。真正的工作量在于把每个算子的
prepare / 路由 / launch 配置逻辑从 Python 搬到 C++。

`spmv_csr` 打通了全部基础设施；`gather`/`scatter` 验证了第二、三个算子确实只是
同一条路径的重复——两者合计只用了 162 行 C++ 调度加 129 行内核，没有改动任何
基础设施。`spmm_csr` 是第一个真正**扩了**基础设施的：它逼出了 ACC_DTYPE 的
wrapper 方案（见上），并把 SpMV 那份 offsets 回读提到 `src/core/csr_meta.cpp`
共用——两个算子各存一份缓存逻辑，迟早会有一份忘了失效。

---

## 构建

```bash
# 从合并仓库根目录初始化 capi/deps/libtriton_jit 子模块。
git submodule update --init --recursive

cmake -S . -B build -G Ninja \
    -DPython_ROOT="$(dirname "$(dirname "$(which python3)")")" \
    -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 平台矩阵

> 每个后端的环境检查、构建和测试方法在 **[docs/](docs/)**，一个平台一份。


测试脚本（精度 + 性能）覆盖 CUDA 加国产加速器。每一格的状态是**配置跑出来的**，不是声称的：

| BACKEND | 平台 | libtriton_jit | FlagSparse adaptor | 状态 |
|---|---|---|---|---|
| `CUDA` | NVIDIA | ✓ | ✓ | **可构建**，本机全量验证 |
| `MUSA` | 摩尔线程 | ✓ | ✓ | **可构建** |
| `MACA` | 沐曦 | ✓ | — | 预留：SDK 位置已填，缺 adaptor.cpp |
| `NPU` | 昇腾 | ✓ | — | 同上（驱动 API 是 ACL，不套 `_template.inc`） |
| `HCU` | 海光 | ✓ | — | 同上（走 HIP，不是两库形状） |
| `GCU` | 燧原 | ✓ | — | 同上，TOPS_ROOT |
| `MLU` | 寒武纪（**通用备用槽**） | ✓ | — | 同上，NEUWARE_HOME |
| `IX` | 天数智芯 | ✓ | — | 同上 |
| `XPU` | 昆仑芯 | **✗** | — | **被依赖卡住**，见下 |

**XPU 单独说**：`deps/libtriton_jit` 支持 CUDA / IX / MUSA / MACA / MLU / NPU / GCU / HCU，**没有 XPU 后端**。它是编译并启动每个内核的那一层，所以这里写 adaptor 也没有东西可跑。配置 `-DBACKEND=XPU` 会在**顶层**就停下并说明这一点——如果让它落到 libtriton_jit，报的是 "Invalid BACKEND: XPU"，点了依赖但没点原因。FlagTree 的 Triton 分支**有** xpu 后端，所以 FlagSparse 的 **Python 侧今天就能跑昆仑芯**（`FLAGSPARSE_BACKEND=xpu`），只有这个 C wrapper 在等 libtriton_jit。

预留槽位不是 TODO 占位：每个槽的 SDK 根目录、库名、头文件都是从 `deps/libtriton_jit/cmake/Backend<NAME>.cmake` 里读出来的（两边找的是同一套 SDK，抄过来就不会漂），所以填一个槽就是**加一个文件**：`src/adaptor/backend/<name>/adaptor.cpp`。驱动 API 是 CUDA 镜像的话，那文件是 `_template.inc` 加一个符号前缀，25 行——看 `backend/musa/adaptor.cpp`。

每个测试二进制会打一行 `[  BACKEND  ] <name>`，所以精度输出和性能 JSON 都能归属到平台；否则同一个二进制在两块芯片上跑出来的行是一样的。

`BACKEND` 可选值见上表。

依赖：CMake ≥ 3.25、C++20 编译器、Ninja、Python 3.10+ 带开发头文件、
`torch>=2.5`、`triton>=3.1,<3.7`、pybind11。**不需要单独的 libtorch C++ 发行版**——
`libtriton_jit/cmake/FindTorch.cmake` 走 `torch.utils.cmake_prefix_path`，
pip 装的 wheel 就够。

构建选项：`FLAGSPARSE_BUILD_TESTS`（默认 ON）、`FLAGSPARSE_BUILD_CLI`（默认 OFF）、
`FLAGSPARSE_INSTALL`（默认 ON）。

### MUSA

`src/adaptor/backend/musa/adaptor.cpp` 是 CUDA 那份加一个符号前缀
（摩尔线程的 driver API 与 CUDA 逐符号镜像），两者共用
`backend/_template.inc`，避免手写两份漂移。需要 MUSA SDK 提供
`$MUSA_HOME/{lib64/libmusart.so, include/musa.h}` 和 `libmusa.so`；缺任一项
CMake 会指名报错而不是在链接期报缺符号。

```bash
cmake -S . -B build -G Ninja -DBACKEND=MUSA -DMUSA_HOME=/usr/local/musa ...
```

---

## 运行测试

```bash
ctest --test-dir build                 # 全部：C API 两层 + Python 算子套件
ctest --test-dir build -R accuracy     # 只跑 C API 精度
ctest --test-dir build -R benchmark    # 只跑 C API 性能
ctest --test-dir build -R pytest       # 只跑 Python 算子套件
ctest --test-dir build -R spmv         # 一个算子的三层一起：accuracy + benchmark + pytest
./build/ctest/accuracy/test_spmv       # 单独跑，含 max_error_ratio 明细
FLAGSPARSE_BENCH_OUT=. ./build/ctest/benchmark/test_spmv   # 性能，JSON 落盘
```

### ctest 里为什么挂着 pytest

`pytest.*` 那些用例跑的是 FlagSparse 的 Python 算子测试，文件留在同一源码树顶层，
这边只是按默认的 `FLAGSPARSE_PYTHON_SRC` 注册进来（`-DFLAGSPARSE_CTEST_PYTEST=OFF` 可关）。

理由不是图省事：这个库 re-export 的就是那些内核。**一个在某加速器上编不出来的内核，
C API 一样用不了**，而从另一个仓库的测试结果里才发现这件事，是发现得太晚。挂进来之后
一条 `ctest` 就覆盖两层、跑在同一块已配置的后端上；`-R <算子>` 会把这个算子的三层
一起选出来。PYTHONPATH 指向的是 `FLAGSPARSE_PYTHON_SRC` 本身，所以测的一定是 C 库
re-export 的那份源码，而不是碰巧装在 site-packages 里的另一个副本。

### 性能 JSON

六个算子共用一套 harness，所以输出是**同一种形状**——跨平台对比时六种格式没法读：

```json
{"operator": "spsv",
 "env": {"backend": "cuda", "arch": "120", "version": 1000},
 "config": {"warmup": 10, "iters": 100, "statistic": "median"},
 "result": [{"name": "n_4096", "status": "ok", "format": "csr", ...,
             "median_ms": 0.034, "gflops": 0.74,
             "baseline_ms": null, "speedup": null}],
 "summary": {"rows": 14, "ok": 14, "not_supported": 0, "failed": 0}}
```

每行都带 `status`：

* `ok` —— 测到了，`median_ms` / `gflops` 有意义；
* `not_supported` —— 这个后端上算子拒绝了这个配置。**记下来而不是丢掉**：某后端没接的
  算子会产出一份满是 `not_supported` 的 JSON，而不是一个缺失的文件——后者读起来和通过
  没区别；
* `failed` —— 试了但出错，状态名和消息保留。

`env` 里的 `backend` + `arch` 是必需的：没有它们，两块芯片上跑出来的行长得一模一样。

精度参考解**恒在 host 上以 fp64 计算**。这不是图省事：参考值算在加速器上等于在测
厂商的稠密库而不是 FlagSparse——在摩尔线程上这一点是硬伤，那里 `torch.sparse`
没有任何可用的 matmul，muDNN 也没有 fp64/复数的 `where` 和 2-D×1-D matmul。

性能 JSON 在没有接厂商基线时，`baseline_ms` 和 `speedup` 写 `null` 而不是 `1.0`——
编造的基线比没有基线更糟。

---

## 使用

```c
#include <flagsparse.h>

flagsparseHandle_t handle;
flagsparseCreate(&handle);

flagsparseSpMatDescr_t A;
flagsparseCreateCsr(&A, rows, cols, nnz,
                    d_rowptr, d_colind, d_values,
                    FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,   /* 或 _64I */
                    FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F);

flagsparseDnVecDescr_t x, y;
flagsparseCreateDnVec(&x, cols, d_x, FLAGSPARSE_R_32F);
flagsparseCreateDnVec(&y, rows, d_y, FLAGSPARSE_R_32F);

const float alpha = 1.0f, beta = 0.0f;
size_t buf = 0;
flagsparseSpMV_bufferSize(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, A, x, &beta, y, FLAGSPARSE_R_32F,
                          FLAGSPARSE_SPMV_ALG_DEFAULT, &buf);
flagsparseSpMV_preprocess(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, A, x, &beta, y, FLAGSPARSE_R_32F,
                          FLAGSPARSE_SPMV_ALG_DEFAULT, NULL);
flagsparseSpMV(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha, A, x, &beta, y, FLAGSPARSE_R_32F,
               FLAGSPARSE_SPMV_ALG_DEFAULT, NULL);
```

从 cuSPARSE 迁移就是把 `cusparse` 换成 `flagsparse`，其余大小写和后缀不变。
索引位宽通过描述符的 `flagsparseIndexType_t` 指定，**不体现在函数名里**，
`INDEX_32I` 和 `INDEX_64I` 都接受（规范 §4.5.1）。

SpMM 换成 `flagsparseCreateDnMat`，多一个 `ld` 和 `flagsparseOrder_t`：

```c
flagsparseDnMatDescr_t B, C;
/* B 是 k x n，C 是 m x n；ld 按 order 解释：ROW 时是行跨度，COL 时是列跨度 */
flagsparseCreateDnMat(&B, k, n, ld_b, d_B, FLAGSPARSE_R_32F, FLAGSPARSE_ORDER_ROW);
flagsparseCreateDnMat(&C, m, n, ld_c, d_C, FLAGSPARSE_R_32F, FLAGSPARSE_ORDER_COL);

flagsparseSpMM(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,   /* op(A)，只支持 non */
               FLAGSPARSE_OPERATION_TRANSPOSE,               /* op(B)，non / trans 都行 */
               &alpha, A, B, &beta, C, FLAGSPARSE_R_32F,
               FLAGSPARSE_SPMM_ALG_DEFAULT, NULL);
```

`op(B) = B^T` 时 B 描述符本身写成 `n x k`，和 cuSPARSE 一致。B 和 C 的 order 可以
各不相同，混着来也对。

`flagsparseSpMV_preprocess` / `flagsparseSpMM_preprocess` 值得单独调用：它把 offsets 回读一次以求出最长行，
SpMV 用它定 kernel 的分段循环上界（**关乎正确性**），SpMM 只用它挑 launch 配置。
不调也能跑（第一次求解会自己算并缓存在描述符上），但那会把一次性的建立成本算进
被计时的那一次。

---

## 加一个算子

1. 在 `flagsparse_codegen/<op>.py` 里 **re-export** Python 包的内核（不要抄：抄一份
   就是第二个真理来源）。三个坑：**不要改名**（libtriton_jit 按 JITFunction 自己的
   `__name__` 命名缓存产物，别名会让 C++ 去要一份从没写过的 metadata）；**不要用
   相对 import**（shim 是按路径加载的，没有包）；内核若有 `tl.dtype` 型 constexpr，
   照 `spmm_csr.py` 加一层 bool constexpr 的 `@triton.jit` wrapper
2. 写 `src/ops/<op>.cpp`，照 `src/ops/spmv.cpp` 的形状：校验 → 路由 → 定 launch
   配置 → 拼 Triton 签名 → `jit::launch`
3. 从 `src/api/pending.cpp` 删掉对应的桩
4. `CMakeLists.txt` 的源文件列表加一行
5. 写 `ctest/accuracy/test_<op>.cpp` 和 `ctest/benchmark/test_<op>.cpp`
   （`ctest/CMakeLists.txt` 按文件存在与否自动挂载，无需改构建）
6. `conf/operators.yaml` 的 `status` 改成 `implemented`

Triton 签名的拼法：指针写 `*fp32:16`，标量写 `fp32`/`i32`，constexpr 直接写字面量，
逗号分隔。复数没有 Triton 类型——按实部/虚部交错数组传分量 dtype，这和 Python 侧
`view_as_real` 的做法一致。

---

## 目录

```
include/flagsparse.h        对外唯一公共头文件
src/api/                    extern "C" 入口：句柄、描述符、未移植算子的 NOT_SUPPORTED
src/core/                   内部类型、dtype 映射、libtriton_jit 桥
src/ops/                    逐算子 C++ 调度
src/adaptor/                厂商运行时抽象，backend/<name>/ 一个后端一份
flagsparse_codegen/         Triton 内核 shim（re-export Python 包，非副本）
deps/libtriton_jit/         子模块
conf/operators.yaml         算子注册表
conf/test_matrix.yaml       测试参数空间与容差
ctest/accuracy/             精度测试
ctest/benchmark/            性能测试（合成形状）
tools/matrix_sweep.cpp      真实矩阵语料扫描：全部算子变体 × 全部矩阵
docker/Dockerfile           构建环境
```
