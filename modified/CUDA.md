# NVIDIA CUDA —— 改动台账

**这个文件的内容注定是空的，那正是它要表达的意思。**

CUDA 是参照路径：共享的那份 Triton 实现就是在 NVIDIA 上验证过的代码，
`src/flagsparse/sparse_operations/` 下**没有任何 `_is_cuda_runtime()` 守卫** ——
不需要，因为所有后端分支都写成"别的后端做什么"，CUDA 是 fallthrough。

所以本文的作用是给其他后端立一条验收线：

## 每一条后端改动都要证明它在 CUDA 上是恒等变换

| 形式 | 怎么证 |
|---|---|
| `if _is_<backend>_runtime():` 新增分支 | 守卫为 False 时控制流逐字不变，跑一遍该算子的 marker |
| 改了某个默认值/profile | CUDA 走的是另一个 key，贴出两个 key 的取值 |
| 按 **dtype** 而不是 backend 分支（如 `_gather_values()`） | CUDA 上也走新路径，必须给出**结果相同**的证据，不能只说"没报错" |
| 测试文件整体换设备抽象 | `_ACCEL is torch.cuda`、`_ACCEL_DEVICE_TYPE == "cuda"`，CPU 上的 fp64 oracle 只会比设备端更准 |

本仓库已有的回归基线（供对比，跑之前先确认自己的基线是什么）：

```
tests/ci        80 passed / 3 skipped
tests/pytest    1613 passed / 3 failed   （3 个是既有失败，与后端改动无关）
```

三个既有 flaky（CUDA 上同样复现，随机输入未固定种子）：
`test_spmv_csc_matches_dense_reference[float32-160-1024]`、
`test_spsv_csr_upper_optimized_route_analysis_workspace_matches_direct[csr_cw_levelschd]`、
`test_spsv_sell_non_unit_rejects_malformed_structure[duplicate_diagonal]`。

## 唯一一类"CUDA 侧改动"

不是后端适配，而是**所有后端共用的东西在 CUDA 上改**：加速器抽象（`_ACCEL`）、
交付变体投影、结果格式、**各后端 baseline 的选择策略**。这类改动要反过来证明它在
**其他后端**上也成立 —— 举证责任在改动者，不在各后端。它们不属于任何一个后端的台账，
写在被改模块自己的注释和 `docs/` 里。

### baseline 策略（2026-09-16 定）

| 后端 | 性能 baseline | 精度参考 |
|---|---|---|
| CUDA | `cupy_cusparse` | PyTorch（+ 厂商列） |
| DCU/ROCm | `hipsparse` | PyTorch（+ 厂商列） |
| MACA | CuPy 装了就用，否则 `torch` | **SciPy (CPU)** |
| MUSA | 无（muSPARSE 在 C API 侧） | **SciPy (CPU)** |
| Ascend | `torch` | **SciPy (CPU)** |
| XPU | `torch` | **SciPy (CPU)** |
| gcu / mlu | 无 | **SciPy (CPU)** |

判据在 `_common._vendor_sparse_library()` 与 `_common._use_scipy_accuracy_reference()`，
由 `tests/ci/test_backend_baseline_policy.py` 钉住。**精度参考换成 SciPy 的理由不是偏好**：
那些平台上的 torch.sparse 本身就是被测对象 —— MACA 在 fp32 CSR 路径上返回非有限值，
MUSA 根本没注册 sparse matmul。参考自己是坏的，会把内核报成错的，那是代价最高的一类误报。

**计时不受影响**：PyTorch baseline 列照旧在加速器上测，只有"被比对的那个值"挪到 CPU。
共享的 SciPy 参考核在 `tests/reference_utils.py`，8 个 benchmark 脚本共用。

---

## 已知性能缺口：SpMM COO 在行长倾斜的矩阵上（2026-09-17 实测）

**不是后端适配问题，是共享内核本身的工作划分问题**，所以记在这里而不是某个后端的台账里。
CUDA 上量到，其他后端走的是同一份内核，同样适用。

### 现象

30 个真实矩阵的交付跑（`pytest_results_cuda_vendorfirst_20260917_145544`）里，
`spmm_coo` 的 f32 只有 0.95x、c32 只有 0.53x，而 f64/c128 有 3.42x/2.89x。

**这个 dtype 差异是假象。** 同一个矩阵上我们的绝对时间几乎不随位宽变化：

```
mip1.mtx     我们:      f32 9.75ms   f64 9.74ms   c64 17.74ms   c128 17.70ms
             cuSPARSE:      0.51         2.59         0.80          6.28
```

位宽翻倍而时间不变 → **不是带宽瓶颈**。变的是分母：cuSPARSE 正常按位宽缩放，在 f64 上
慢了 2.4 倍，于是比值看起来变好。真正的问题是**绝对时间**，f64/c128 的"胜利"只是基线变慢。
计时全在 GPU 上（`ms == gpu_ms`，`process_cpu_ms = 0`），不是主机侧开销。

### 根因：按行划分，被最长的那一行决定

亏损集中在少数矩阵，且与**倾斜度（最长行 ÷ 平均行长）严格单调**：

| 矩阵 | f32 加速比 | 平均行长 | 最长行 | 倾斜度 |
|---|---:|---:|---:|---:|
| wiki-Talk | 0.17x | 2.1 | 100022 | **47694x** |
| Stanford | 0.10x | 8.2 | 38606 | **4706x** |
| mip1 | 0.05x | 155.8 | 66395 | 426x |
| TSOPF_FS_b300_c1 | 0.16x | 150.6 | 13942 | 93x |
| engine | 0.77x | 32.8 | 159 | 5x |
| net150 | 2.05x | 71.7 | 281 | 4x |
| auto | 2.36x | 14.8 | 37 | 3x |
| GL7d14 | **4.08x** | 10.7 | 24 | **2x** |

倾斜度 ≤5 全赢（最高 4x），≥90 全输（最低 0.05x）。`coo_rowrun` 一行给一个 program，
一行占 6.6 万非零时那个 program 独自串行跑完全部，其余早已空转 —— 时间由最长行决定，
与 dtype 无关，正好解释上面那张表。

### 换算法救不了：三条路都试过

mip1.mtx / float32 / dense_cols=32，cuSPARSE 基线 0.498 ms：

| 算法 | 时间 | vs cuSPARSE |
|---|---:|---:|
| `coo_rowrun`（默认） | 9.76 ms | 0.05x |
| `coo_atomic` | **164.92 ms** | 0.003x |
| `spmm_coo_alg1` | 10.08 ms | 0.05x |

**`coo_atomic` 不是候选项，在任何矩阵上都不是** —— 小矩阵 wave.mtx 上它 31.77 ms 对
rowrun 的 0.28 ms，慢 113 倍。它是纯 nnz 并行加原子累加，同一输出行被几万次原子操作
争用而全部串行化。"照搬 cuSPARSE 的原子思路"这条已被实测否定。
`spmm_coo_alg1` 与 rowrun 持平，划分方式没有本质区别。

### 唯一剩下的方向

按 **nnz 均匀切块**重写工作划分，即 `spmv_csr.py` 的 `_spmv_csr_segbin_kernel` 那一套：
按 nnz 均匀切 + 二分定位行 + `associative_scan` 做段内归约。它对行长分布免疫，又不像纯
原子那样争用 —— 恰好落在上面两个失败模式之间。同一批矩阵上 `spmv_csr` 拿到 3.07x，
就是这个划分方式的功劳。

按 `modified/README.md` 的代价阶梯，这属于**第 2 档"同文件内加一个内核变体"**，
和现有的 `coo_rowrun` 并排放，不需要 fork 算子、更不需要往 `backends/` 里放文件。

### 投不投的判断依据

30 个矩阵里只有 **4 个**倾斜度 ≥90（mip1、Stanford、wiki-Talk、TSOPF_FS_b300_c1），
其余 26 个从持平到快 4 倍。要不要做，取决于交付时那几个高倾斜矩阵占多大权重 ——
如果它们在评测集里，f32 的整体加速比会一直被压在 1x 附近。
