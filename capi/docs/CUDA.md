# CUDA (NVIDIA)

`BACKEND=CUDA` / `FLAGSPARSE_BACKEND=cuda`。**唯一一个在本仓库全量验证过的后端**，
所以它同时是参考平台：别的平台出问题时，先确认同一份代码在 CUDA 上是好的。

## 环境检查
> 这一层离不开可导入的 `flagsparse` 包：内核是 **re-export** 而不是拷贝，所以
> `FLAGSPARSE_PYTHON_SRC` 指错、或机器上有旧副本，症状分别是 `EXECUTION_FAILED`
> 和 `CompilationError`。依赖关系、版本配套和部署清单见
> [README.md#这个库离不开-flagsparse](README.md)。


```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
python3 -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
```

已验证组合：**RTX 5090 (sm_120) / CUDA 12.8 / torch 2.9.0 / triton 3.6.0 / Python 3.12**。
要求 CMake ≥ 3.25、C++20、Ninja、pybind11。**不需要单独的 libtorch C++ 发行版** ——
`libtriton_jit/cmake/FindTorch.cmake` 走 `torch.utils.cmake_prefix_path`，pip 装的 wheel 就够。

## 构建

```bash
cmake -S . -B build -G Ninja -DBACKEND=CUDA \
      -DPython_ROOT="$(dirname "$(dirname "$(which python3)")")" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CUDA 分支用 `find_package(CUDAToolkit REQUIRED GLOBAL)`，链 `CUDA::cuda_driver`。
`GLOBAL` 不是可有可无的：`find_package` 导入的 target 作用域限于调用它的目录，而
`ctest/` 要编自己那份 adaptor，拿不到就断在链接期。

## 跑测试

```bash
ctest --test-dir build                       # 两层全跑（33 个用例）
ctest --test-dir build -R accuracy           # C API 精度，8 个算子
ctest --test-dir build -R spsv               # 一个算子的三份一起

# 性能：指向真实矩阵语料，自动对 cuSPARSE 求加速比
FLAGSPARSE_MATRIX_DIR=/path/to/mtx FLAGSPARSE_BENCH_OUT=./bench \
    ctest --test-dir build -R benchmark

# 汇总成一张 40 变体的表，以及与 FlagSparse Python 侧同 schema 的 summary.json
python3 tools/report.py --bench-dir ./bench
python3 tools/write_summary.py --bench-dir ./bench --out ./bench
```

`FLAGSPARSE_MATRIX_DIR` 不设也能跑，退回合成形状并把每行标成 `corpus=synthetic`。

## 基线：cuSPARSE，本平台唯一已验证的一个

CUDA 是六个后端里**唯一**基线被真正编译并实测过的：`ctest/baseline/cuda/baseline.cpp`
对着 cuSPARSE 12.5 的 generic API，八个入口全部跑通。其余后端的基线要么写好未编译
（DCU），要么按厂商命名惯例填了名字等探测（MACA/MUSA），要么没有可绑的 API（昇腾、
昆仑芯）。读跨后端对比时这个差别很重要。

加速比**只统计精度通过的矩阵**，精度判的是 CPU fp64 oracle 而不是 cuSPARSE 自己——
厂商内核和我们可能朝同一方向错，互相对上会被当成通过。

### 一个必须知道的基线陷阱

FlagSparse Python 侧那套 harness 里名为 `run_cusparse` 的开关，**调的其实是
`torch.sparse.mm`**，不是裸 cuSPARSE。两者在同一个 cfd2 上差 1.44 倍：

| | median ms |
|---|---|
| `cusparseSpMV` | 0.0218 |
| `torch.sparse.mm` | 0.0315 |

`torch.sparse.mm` 把 `(n,1)` 当 SpMM n=1 走，还多一层 dispatch。所以两个仓库的加速比
数字**不能直接比**——分母不是同一个东西。

## 已验证的量级（RTX 5090，仅供回归对比，不是标称值）

| 算子 | 配置 | 中位耗时 | 备注 |
|---|---|---|---|
| SpMV CSR | 4096², density 0.1 | 0.0166 ms | 202 GFLOPS |
| SpMM CSR | 4096², density 0.1, n=32 | 0.0853 ms | 1261 GFLOPS |
| SpMM COO | 同上 | **0.0608 ms** | 1768 GFLOPS，比 CSR 快 —— 见下 |
| SpSV CSR | n=8192, density 0.002 | 0.0815 ms | ~2 GFLOPS，延迟受限 |
| SpSM CSR | n=4096, n_rhs=32 | 0.0491 ms | n_rhs 1→32 耗时相同 |
| SpGEMM | 1024², density 0.05 | compute 0.31 ms / **copy 44.4 ms** | 见下 |
| Gather | 2²⁰, 随机索引 | 0.0198 ms | 635 GB/s（顺序 1061） |

## 三个在 CUDA 上量出来、值得记住的结论

**1. SpMM 的 COO 比 CSR 快，而这不是格式的功劳。** 两个内核结构几乎相同，差的是 launch
配置：COO 用的是算子包 30 矩阵扫出来的 `BLOCK_NNZ=4` 并按设备 warp size 推 num_warps，
CSR 的 `BLOCK_NNZ` 来自 warp/factor 启发式。读成"**CSR 的 launch 配置还有空间**"，
不是"COO 是更快的格式"。

**2. SpGEMM 的 `copy` 阶段比另两个阶段慢 30–150 倍**，因为它在**主机上**逐行排序 ——
cuSPARSE 保证有序 CSR，而 fill 内核按 hash 槽位顺序出列。正确但慢，设备端分段排序是
明确的后续项。benchmark 把三个阶段分开报，就是为了让这一项可见而不是被均摊掉。

**3. fp32 的 SDDMM 在长 k 上会走规范 §6.3.1 的放宽档**（strict ratio 可到 2.3，
relaxed 0.023）。那是归约顺序，不是缺陷；输出会如实标成 `PASS(relaxed)`。

## 已知环境问题：冷缓存下 scatter 编译失败

Triton 缓存为空时首次编译 scatter 内核可能失败：

```
FLAGSPARSE_STATUS_EXECUTION_FAILED -- ModuleNotFoundError: No module named 'mlir'
  triton/experimental/tle/raw/mlir/runtime.py(8)
```

这是 **Triton 自身的打包缺陷**：`tle/raw/runtime.py` 把可选的 `tops` 后端包在
try/except 里，却无条件 `import .mlir`，而后者需要这个 build 没附带的 MLIR Python
bindings。实测边界（不是推测）：缓存预热后 10/10 连续通过；冷缓存下 spmv 和 gather
**不受影响**，只有 scatter 命中；塞 stub `mlir` 模块无效。规避：先跑一次预热，
或安装 MLIR Python bindings。
