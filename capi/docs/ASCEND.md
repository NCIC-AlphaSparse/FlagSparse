# Ascend (昇腾)

`BACKEND=NPU` / `FLAGSPARSE_BACKEND=ascend`。**预留槽位**：SDK 位置已填，
缺 `src/adaptor/backend/npu/adaptor.cpp`。

## 现状

```
deps/libtriton_jit   ✓ 有 BackendNPU.cmake（ASCEND_TOOLKIT_HOME）
FlagSparse adaptor   ✗ 未写
Python 层            ✓ 可用，且带两条 Ascend 专用 fallback（见下）
```

## 环境检查
> 这一层离不开可导入的 `flagsparse` 包：内核是 **re-export** 而不是拷贝，所以
> `FLAGSPARSE_PYTHON_SRC` 指错、或机器上有旧副本，症状分别是 `EXECUTION_FAILED`
> 和 `CompilationError`。依赖关系、版本配套和部署清单见
> [README.md#这个库离不开-flagsparse](README.md)。


```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export FLAGSPARSE_BACKEND=ascend
python3 -c "import torch, torch_npu; print(torch.__version__, torch.npu.is_available())"
```

## 填这个槽位要做什么

一个文件，但**不是** `_template.inc` 加前缀：昇腾的 driver API 是 **ACL**，不是 CUDA
镜像的两库形状。要按 `src/adaptor/adaptor.hpp` 的接口用 ACL 实现：设备发现、
`multiprocessor_count` / `max_threads_per_multiprocessor` / `warp_size`、
malloc/free/memcpy/memset/同步。

## 跑测试

```bash
export FLAGSPARSE_BACKEND=ascend

cmake -S . -B build -G Ninja -DBACKEND=NPU \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 精度
ctest --test-dir build -R accuracy --output-on-failure

# 性能：真实矩阵语料。不设 FLAGSPARSE_MATRIX_DIR 会退回合成形状，
# 并把每行标成 corpus=synthetic，不会被误读成真实数据
FLAGSPARSE_MATRIX_DIR=/path/to/mtx FLAGSPARSE_BENCH_OUT=./bench \
    ctest --test-dir build -R benchmark

# 汇总：40+ 变体一张表 / 与 FlagSparse Python 侧同 schema 的 summary.json
python3 tools/report.py --bench-dir ./bench
python3 tools/write_summary.py --bench-dir ./bench --out ./bench

# 算子清单与实现是否一致（CI 可用 --strict）
python3 tools/check_manifest.py --bench-dir ./bench
```

算子清单来自 `conf/operators.yaml`，构建时由 `tools/gen_variants.py` 生成成扫描用的
变体表——**加算子是改 YAML 加重新构建，不动测试代码**。

### 这个后端上预期会看到什么

* **`speedup` 列一定是空的**，`baseline_status` 是 `unavailable`，理由字符串说明
  CANN 没有描述符式稀疏 API。**这是诚实的空，不是没跑完**；
* **典型故障是「编不出来」而不是「算错」** —— 见下一节。所以这里 `not_supported`
  的行会比其他后端多，且多数是 CANN 没能 lower 那个内核形状。

### 跑不起来时按这个顺序查

1. **配置就停下** → adaptor 没写；
2. **某个算子编译失败而不是算错** → 这是昇腾的常态。先查
   `benchmark/benchmark_ascend_probe.py` 的探测结果，它按算子给出 PASS / 编译失败 /
   拒绝三种状态，比在 C API 这层猜快得多；
3. **轮询求解器类内核（SpSM）挂住或编译失败** → 用
   `flagsparse.SPSM_ASCEND_DISPATCH`，CANN 不 lower 自旋在全局内存 ready 标志上的形状；
4. **spmm_coo 编译失败** → 用 `flagsparse.SPMM_COO_ASCEND_DISPATCH`，两条 COO 路线都在
   内核里开 scratchpad，CANN 编不过。

## 基线：CANN 有稀疏算子，但没有描述符 API

这一条和 MACA / MUSA **不是一回事**，别按同一种方式理解：

| | MACA / MUSA | 昇腾 CANN |
|---|---|---|
| 厂商有稀疏实现吗 | 有，且是 cuSPARSE 克隆 | **有**，走 `aclnn`（`aclnnSparse*`）和 torch_npu |
| 缺什么 | 只缺头路径和库名 | 缺的是 **API 形状** |
| 补法 | 填两个 CMake 变量 | 写逐算子适配器 |
| 槽位状态 | `PROBED` | `NONE` |

`ctest/baseline/baseline.hpp` 的八个入口是按 create-matrix / bufferSize / execute
这个三段式描述符模型设计的——因为 cuSPARSE、hipSPARSE、mcSPARSE、muSPARSE 都是那个
形状。CANN 没有这个三段式，它是一个算子一个 `aclnn` 调用。所以前缀表那套移植在这里
用不上：**要一个能比的基线，得按算子逐个绑，这活是真活，只是还没做。**

在做之前，昇腾上的性能行会是 `baseline_status: "unavailable"` 加空 `speedup`，理由
字符串在 `ctest/baseline/CMakeLists.txt` 的 `FLAGSPARSE_ASCEND_BASELINE_WHY` 里。
**这是诚实的空，不是没跑完。**

可比的数今天来自 Python 侧的 `benchmark/benchmark_ascend_probe.py`，它按算子对
torch_npu 取基线——测量形状和这里的 ctest 不同，两边的数不要直接相除。

### 基线文件：`ctest/baseline/ascend/baseline.cpp`

昇腾**有**自己的基线文件，但它**不套 `_template.inc`** —— 其他四家是 35 行前缀表加一份
共享实现，因为它们的稀疏库是 cuSPARSE 克隆。CANN 不是，所以这份文件是个**接缝**：
八个入口各自写明「这一项会绑到哪个 CANN 算子」，在有人真去绑之前返回 `unavailable`
并带上那个名字。

这比路由到 `baseline/none` 有用：`none` 只说「这个后端没有基线」，而接缝说的是
「这个算子会映射到那个调用，只是还没写」——后者可以直接开工。

### 真去绑的时候会撞上的四件事

1. **SpMV 没有对应物**。CANN 没有专门的 SpMV。Python 侧探测用宽度为 1 的稠密操作数去乘，
   那是 **SpMM n=1，不是 SpMV**，两者不可比。拿这个数当 SpMV 基线会系统性地高估我们。
2. **三角求解要先稠密化**。`aclnnTriangularSolve` 吃稠密矩阵，densify 之后**复杂度类都变了**
   —— 这种数不是稀疏求解器的基线，如果非要用，必须在报告里标注清楚。
3. **workspace 的生命周期对不上**。CANN 是每算子一个 `aclnnXxxGetWorkspaceSize`，
   和本接口 `bufferSize` 的契约不是同一种生命周期，不能一一映射。
4. **scatter 在重复下标上的语义不同**。`aclnnIndexPut` 和我们的 scatter 对重复下标的处理
   不一样；`ctest/benchmark/test_scatter.cpp` 里的扫描是**先去重**的，绑基线时必须同样去重，
   否则两边测的不是一回事。

还有一条是环境而不是代码：**aclnn 的稀疏接口在 CANN 版本间变过**，而且算子带 NZ/ND
layout 参数，cuSPARSE 侧没有对应概念。所以这份文件没有在没有 CANN 的机器上硬写完 ——
那样会产出一个哪儿都编不过、却处处误导人的文件。

## 这个平台的重点：内核编不出来的比算错的多

昇腾上的典型故障不是"算错"，而是 **CANN 的 Triton 后端 lower 不出这个内核**。所以
Python 层提供了两条 torch-only 的退路，并且都暴露成公共符号：

| 符号 | 覆盖 | 退路 | 为什么 |
|---|---|---|---|
| `flagsparse.SPSM_ASCEND_DISPATCH` | `spsm_csr` / `spsm_coo` | 逐行 sweep | 轮询求解器自旋在全局内存的 ready 标志上，CANN 不 lower 这个形状 |
| `flagsparse.SPMM_COO_ASCEND_DISPATCH` | `spmm_coo` | 单次 `index_add` | 两条 COO 路线都在内核里开 scratchpad，CANN 编译失败 |

默认由运行时探测自动选，也可以强制：

```bash
FLAGSPARSE_SPSM_ASCEND_DISPATCH=1
FLAGSPARSE_SPMM_COO_ASCEND_DISPATCH=1
```

**强制开关不是调试遗留物**：fallback 是纯 torch，所以这两个开关让它能在**任何**后端上
与 Triton 路径逐位对比 —— 手边没有 NPU 也能验证。改动 fallback 后应当先这样验一遍。
（已在 CUDA 上对比过：SpSM 8 种三角/对角/精度组合、SpMM COO 8 种 dtype×布局组合，
全部吻合到机器精度。）

## 算子能力探测

在一个内核未必编得出来的平台上，"跑没跑通、没通是为什么"比耗时更重要：

```bash
python3 benchmark/benchmark_ascend_probe.py --device 6 --csv-summary probe.csv
```

对**全部 23 个算子 × 20 个矩阵**逐个跑，分成 `PASS` / `MISMATCH` / `TRITON_COMPILE` /
`REJECTED` / `ERROR` / `NO_ADAPTER` 六类。**每个 case 默认清空 Triton 缓存**——不清的话
前一个 case 编出的产物会让后一个的编译失败变成缓存命中，于是编不出来的内核报成 PASS。

那个脚本和它的完整说明在隔壁 checkout：`benchmark/benchmark_ascend_probe.py`、
`docs/ASCEND.md`。

## 一处尚未在真机确认的事

探测脚本靠关键字把异常归类成 `TRITON_COMPILE`，其中 `bishengir` / `ascendc` /
`npucompiler` 是**按昇腾工具链命名猜的**，没在真机上验证过。如果某个关键字对不上，
那类失败会落进 `ERROR` 而不是 `TRITON_COMPILE` —— **分类会偏，但不会丢数据**：
`reason` 字段保留了原始异常文本，可以据此回填关键字。真机上跑第一轮时值得先看这一点。
