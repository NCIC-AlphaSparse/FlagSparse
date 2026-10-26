# 加速比低于 0.8 的变体：现状与优化建议

数据来源：

- 天数 BI-V150：`results_biv_65_20261024_final`（`c814047+dirty`，即合入 main 的天数调优改动），56 个可测变体全部
  Passed，**13 个低于 0.8**。
- H800：`conf/h800_reference.json`（10-14 的 65 变体全量，FlagSparse 对**真实 cuSPARSE**的加速比）。

这些建议没有在天数上实测过，都是从数据推出来的假设，每条都写了怎么验证。动手之前先过一遍文末的
"先查这五项"清单——这个仓库里五个被判为"结构性上限"的算子，最后全是启动配置或 prepare/run 放错位置。

## 先看这一列：H800 自己也不到 0.8

天数这 13 个里有 12 个的分母是 **H800 cuSPARSE 时间按带宽折算**的。所以天数上的比值约等于
"H800 上 FlagSparse 对 cuSPARSE 的比值"再乘天数自身的效率。下表最后一列就是前者：

| 变体 | 天数 | 最差的矩阵（天数） | H800 对 cuSPARSE | 归类 |
|---|---:|---|---:|---|
| `spgemm_csr_f32_int_non_non` | 0.064 | NACA0015 0.031、amazon0601 0.033、roadNet-TX 0.034 | 0.694 | **天数特有**：走 ESC |
| `spmv_coo_c32_int_non` | 0.472 | ASIC_680ks **0.061**、roadNet-TX 0.292 | 0.493 | 通用 + ASIC 路由 |
| `spmv_sell_c32_int_non` | 0.516 | NACA0015 0.353、amazon0601 0.387 | 0.663 | 通用 |
| `spmv_csr_c32_int_non` | 0.606 | roadNet-TX 0.357、auto 0.461 | 0.599 | 通用 |
| `spmv_sell_f16_int_non` | 0.621 | filter3D 0.445、NACA0015 0.483 | 0.531 | 通用 |
| `spmv_sell_f32_int_non` | 0.630 | NACA0015 0.477、roadNet-TX 0.505 | 0.576 | 通用 |
| `spmv_sell_i8i32_int_non` | 0.654 | NACA0015 0.477、GL7d14 0.487 | 0.421 | 通用 |
| `spvv_c32_int_conj` | 0.728 | nnz=1024 0.517（分母是本机 PyTorch） | 0.386 | 通用：固定开销 |
| `spmv_coo_f16f32_int_non` | 0.730 | ASIC_680ks 0.299 | 0.334 | 通用 + 短行 |
| `spmv_coo_f16_int_non` | 0.762 | ASIC_680ks 0.317、filter3D 0.485 | 0.341 | 通用 + 短行 |
| `spmv_coo_i8i32_int_non` | 0.771 | ASIC_680ks 0.278、GL7d14 0.537 | 0.357 | 通用 + 短行 |
| `spmv_coo_f32_int_trans` | 0.776 | ASIC_680ks **0.125**、amazon0601 0.438 | 0.897 | **天数特有**：转置路径 |
| `spmv_csr_f32c32_int_non` | 0.785 | ASIC_680ks 0.467、auto 0.594 | 0.616 | 通用（`timed=call`） |

**结论一：除 spgemm 和 COO 转置外，其余 11 个的问题在 kernel 本身，不在天数。** 它们在 H800 上对真实
cuSPARSE 也只有 0.33–0.66。这意味着可以**先在 CUDA 机器上对着真实 cuSPARSE 调**（本仓库的 CUDA 开发机有
cuSPARSE，单个变体几分钟出数），H800 那一列上去了，天数这一列会跟着上去；在天数上直接调，分母本身是折算值，
信号要弱得多。

**结论二：最差的几乎总是短行矩阵。** 各矩阵的行长分布（`tests/data`）：

| 矩阵 | 行数 | 平均每行 nnz | 最大 | 变异系数 | 空行 |
|---|---:|---:|---:|---:|---:|
| ASIC_680ks | 682,712 | **3.4** | 210 | **1.25** | 0 |
| roadNet-TX | 1,393,383 | **2.8** | 12 | 0.38 | **13,466** |
| NACA0015 | 1,039,183 | 6.0 | 10 | 0.14 | 0 |
| amazon0601 | 403,394 | 8.4 | 10 | 0.33 | 955 |
| GL7d14 | 171,375 | 10.7 | 24 | 0.24 | 6 |
| wave | 156,317 | 13.6 | 44 | 0.24 | 0 |
| auto | 448,695 | 14.8 | 37 | 0.18 | 0 |
| cage12 | 130,228 | 15.6 | 33 | 0.30 | 0 |
| cfd2 | 123,440 | 25.0 | 30 | 0.15 | 0 |
| filter3D | 106,437 | 25.4 | 112 | 0.50 | 0 |

ASIC_680ks（平均 3.4、但有 210 的长行，变异系数 1.25）和 roadNet-TX（平均 2.8、1.3 万个空行）在 COO/CSR 的多个
变体里都是最差的一个；行长 15–25 的 cage12、cfd2、wave 大多已经在 0.8 以上。按行分块的 kernel 在平均 3 个
nnz 的行上，大部分 lane 是空转的。

## 按优先级的建议

> 本节是 2026-10-24 调优前的假设。天数已实测的结果见文末"已验证的调优进展"，下一轮的具体方向见
> "下一轮天数优化方向（2026-10-25）"；三者冲突时以后两节为准。

### P0 `spgemm_csr`：换掉 ESC（预计收益最大）

- 现状：天数的 FlagTree 编不出 TLE 共享内存哈希表，`_spgemm_compute` 让天数走 `_spgemm_esc_compute_safe`
  （全局展开-排序-归约）。每个矩阵比折算基线慢 10–30 倍（NACA0015 135 ms 对 4.1 ms）。
- 建议：MACA 上已经有一条**只用标准 Triton、不依赖 TLE** 的实现（`_maca_triton_spgemm_compute`，
  元数据在 `_spgemm_maca.build_metadata`，`4d5ab61`）。它现在被 `_is_maca_runtime()` 门控住了。
  先在天数上把这两处门控放开到 `_is_maca_runtime() or _is_iluvatar_runtime()`，跑 spgemm 精度再测性能。
- 验证：CUDA 机器上可以用 `FLAGSPARSE_BACKEND=iluvatar` 模拟先跑通精度；性能必须上真机。注意 SpGEMM 的计时
  包含 prepare（见 `flagsparse-benchmark-methodology`），两边对等。

### P1 COO 系列：先查路由，再处理短行

- `spmv_coo_c32_int_non` 的逐矩阵时间和 `spmv_csr_c32_int_non` **几乎一模一样**（GL7d14 都是 0.1104 ms，NACA0015
  0.2908 对 0.2907），说明 COO c32 走的就是 CSR 的 gather 路径——**唯独 ASIC_680ks 例外：COO 0.843 ms，CSR 0.135 ms，
  慢 6 倍**。这正是清单第 5 项（路由）：ASIC 上 COO 进了另一条路径。先打印 ASIC 上 `prepare_spmv_coo` 选中的
  plan（`csr_plan` / `non_csr_plan` / 原子路径），让它和 CSR 走同一条，只修 ASIC 这一个矩阵，这一项的 10 矩阵均值约从 0.47 升到 0.52。
- `spmv_coo_f32_int_trans`（天数特有，H800 上 0.897）：ASIC 0.125、amazon0601 0.438。`spmv_csc` 的转置以前也是这样，
  修法是"CSC(A) 就是 CSR(Aᵀ)，直接委托给 spmv_csr，并把转置缓存在 prepare 里"（`spmv-csc-trans-delegation`）。
  检查天数上 COO 转置是否每次调用都在重建转置、是否走到了带缓存的 transpose plan。
- `spmv_coo_f16 / f16f32 / i8i32`：都经 `mixed_spmx` 把 COO 转成 CSR（已缓存）再按行分块。ASIC 上 0.12 ms 对折算
  0.035 ms。行分块对平均 3.4 nnz 的行太宽：试 `_csr_row_tile` 在平均行长 ≤ 4 时用更窄的 lane（4）、每个 program
  更多行；或者对"平均行长 ≤ 4 且变异系数 > 1"的矩阵改走按 nnz 切分的 kernel（spmv_csr 已有 segbin/nnz-split）。
  这三项在 H800 上也只有 0.33–0.36，**先在 CUDA 上调**。

### P1 SELL 四项：在 CUDA 上先把 H800 那一列调上去

- 现状：H800 上 SELL 对 cuSPARSE 只有 0.42–0.66，天数 0.52–0.65。cage12、wave、cfd2 已经在 0.9 左右，
  NACA0015、filter3D、amazon0601、GL7d14 在 0.45–0.55。
- 天数侧已经试过且回退的（见 `ILUVATAR.md` 改动记录 10-21/10-22）：slot 并行、CSR 压缩后 row-tile、
  CSR 轻量 row-tile、complex slot=8、双 warp。不要重复这些。
- 建议：
  1. 在 CUDA 上按清单第 1–3 项扫 `GROUP`（每个 program 的 slice 数）× `num_warps` × `BLOCK_R`，对象是真实
     cuSPARSE SELL；当前默认 `group = min(8, 128 // block_r)` 是全局常数，不看矩阵。
  2. 打印每个矩阵的填充率 `sell_values / nnz`：NACA0015 行长 6–10 几乎不填充却仍然慢，说明瓶颈不在填充，
     更可能在每个 slice 的串行 slot 循环。
  3. ROCm 上把 SELL 转成行主序 CSR 后用 row-tile 拿到了 1.15x（`_rocm_sell_real_kernel`）；天数试的 row-tile
     更差，差别可能在 tile 参数而不是思路本身，值得用 ROCm 那组参数在天数上 A/B 一次。

### P2 CSR 复数：短行 tile

- `spmv_csr_c32_int_non`：补丁新加的天数 row-tile 复数路径，平均行长 ≤ 4 时用 `ROWS=16, BLOCK_NNZ=8`。
  roadNet-TX（平均 2.8、1.3 万空行）0.357、ASIC 0.513。8 个 lane 处理 2.8 个 nnz，空转过半：试 `BLOCK_NNZ=4`
  搭配 `ROWS=32`（清单第 2、3 项）。
- `spmv_csr_f32c32_int_non`：`timed=call`，每次调用都经过混合精度路径的入口。先确认计时窗口里有没有每次
  都做的准备工作（清单第 4 项）：q4 bench 对 `f16f32` 等也是 `call`，但 cuSPARSE 一侧的描述符是提前建好的。
- **2026-10-24 已在 CUDA（5090）上处理 `spmv_csr_f32c32_int_non`，结论：瓶颈是主机开销，不是 kernel。**
  GPU 时间（CUDA Graph 回放）上，现有 row-tile kernel 在 10 个矩阵里有 9 个已经不慢于 cuSPARSE（cage12 7.9 µs 对
  13.6 µs），但一次调用的墙钟是 42 µs，而且对所有矩阵都差不多——Python 侧开销决定了数字。拆开是：Triton JIT
  分发 16 µs（空 kernel 也要 12 µs）、`fs.<name>` 每次走未缓存的模块 `__getattr__` 3–6 µs、`_check_1d` 里每个张量
  构造 `torch.device` 共 3.6 µs、每次分配 y 3.4 µs、`view_as_real`+`reshape` 4 µs。改动：
  - `mixed_spmx.spmv_csr_mixed` 在纯 CUDA 上改用 `_musa_spmv_launch.launch(..., cuda=True)`（复用 MUSA 那套按
    dtype/对齐/别名/标量值做 key 的已编译 launcher 缓存，直接调 `CompiledKernel.run`，7 µs）；Triton 启动器参数
    对不上时首次 TypeError 就退回 JIT（参数解析在发射之前，不会重复执行）。ROCm/MACA/天数 不走这条路。
  - `flagsparse.__getattr__` 把结果写回模块 globals；`_is_accel_tensor` 在设备类型为 `cuda` 时用 `t.is_cuda`；
    `_finish` 在所有后端都跳过 `out.copy_(out)`（原来只有 MUSA 跳过）。这三条对所有后端、所有算子都生效。
  - q4 bench 的混合精度 CSR `call` 路径传预分配的 `out`，与 cuSPARSE 一侧（以及其余 prepared 变体）对齐。
  - 同进程交替 A/B、7 轮取最小（这台机器负载 120–250，单次数字不可信）：几何平均 0.656 → 0.905；
    `f16f32` 走同一路径 0.715 → 0.921（"old" 已含上面三条通用改动，相对 HEAD 实际更多）。
  - **剩下的是 ASIC_680ks（0.58）**：GPU 上 27 µs 对 cuSPARSE 15.6 µs（后者已接近带宽上限）。85% 的行 ≤4 个 nnz，
    但有 369 行 >32、最长 210，row-tile 每个 tile 按最长行循环。试过"tile 内连续 nnz 段 + 行成员掩码归约"的
    flat 写法（2D/3D 两版），比现在慢 2–10 倍，已放弃。下一步可试 `tl.gather` + 段内前缀和，或者像 MUSA 那样
    用 prepare 期的 row-id 计划走 `_nnz_kernel`（但 `timed=call` 下要看计划能不能合法地放到 prepare）。
  - 天数上：直接发射这条只在 NVIDIA CUDA 开。10-24 结果记录的天数 Triton 是 3.6.0，但它的 iluvatar 后端
    `CompiledKernel.run` 签名没在真机上验证过，所以没开。**而且天数上这一项不是主机瓶颈**（耗时随 nnz 线性，
    cage12 47.6 µs → auto 266.6 µs），主机侧改动在天数上收益很小，见下一条。
  - 天数该怎么调：同一个 kernel 的 f32c32 / f16f32 / i8f32 在 auto 上是 266.6 / 202.3 / 160.3 µs，比例 1.66:1.26:1
    正好等于每个 nnz 的 val+col 字节比 8:6:5，即瓶颈是 val/col 流式读取，只跑到约 200 GB/s（实测带宽 1150 GB/s
    的 17%）；折算分母要求约 34%（与 H800 cuSPARSE 自身的带宽利用率相同）。现行天数 tile 表全部是
    `num_warps=1, BLOCK=8`，每行一次只读 32 B。先用 `FLAGSPARSE_ILUVATAR_MIXED_ROWS/BLOCK/WARPS` 在 auto、
    NACA0015、amazon0601 上扫 BLOCK∈{8,16,32}、ROWS∈{8,16,32}、WARPS∈{1,2,4}，复数单独出一张 tile 表。

### P2 `spvv_c32_int_conj`：固定开销

- nnz 从 1,024 涨到 65,536，FlagSparse 一直是 0.05–0.06 ms，本机 PyTorch 是 0.025–0.07 ms；最大那档已经 1.10x。
  这是每次调用的固定开销（多次 launch、共轭的物化、输出分配或同步），不是带宽。H800 上同样只有 0.386。
- 建议：数一下一次调用实际 launch 了几个 kernel、有没有 `.conj()`/`resolve_conj()` 物化或 `.item()` 同步；
  把共轭折进 kernel 的常量参数里。

## 两个"高"数字也要查（不是低于 0.8，但可能是假的）

- `spmv_coo_f32_int_non` **3.365x**，分母是天数本机 `torch.sparse` 的 COO SpMV。这张卡的 torch.sparse 有静默算错的
  先例（CSR 返回全零），要确认这一列的 PyTorch 结果是校验过的。
- `spmm_csr_f32_int_non_non_row` **14.09x**（10-14 是 8.39x），分母是 CoreX legacy `cusparseScsrmm`，以前就测到
  25 ms 对 H800 cuSPARSE 约 0.15 ms，慢得可疑（可能计时里含 RHS 转置）。

## 先查这五项（每次判"到顶了"之前）

1. **grid 大小**：launch 了多少个 program，和 SM 数比？写死的常数（2、32、"每行一个"）是这个仓库最常见的缺陷。
2. **每个 program 的 tile**：`BLOCK`/`CAP`/lane 数是按数据定的，还是按全局最坏情况定的？
3. **`num_warps` 和 tile 是否匹配**：tile 小于线程数时，缩小 tile 不会更快。
4. **prepare/run 的放置**：每次调用是否在重建厂商基线只建一次的东西？两边的计时窗口是否一致？
5. **路由**：调用到底有没有走到调过的那条路径？上面 COO c32 在 ASIC 上就是这一项。

每改一处，都要重跑对应后端的精度（天数用 `docs/debug/ILUVATAR.md` 命令里的 `-k`），并且用同一个 benchmark、
同一批 `tests/data` 矩阵对比前后。

## 已验证的调优进展（2026-10-24，BI-V150）

本节记录按 1→7 顺序进行的实测。原文前面的“建议”是调优前假设；若与本节结果冲突，以本节实测结论为准。性能比统一按
`speedup = baseline_ms / current_ms` 计算，`>1.0x` 表示快于基线。有厂商基线时使用厂商基线；没有厂商基线时按 H800 cuSPARSE 带宽折算基线计算并报告折算加速比，注明其为折算值。

### 第 1 项：`spmv_coo_c32_int_non` 路由与 gather 参数

- 先对 ASIC_680ks 以 `warmup=10, iters=50` 复测默认路径：`0.8436 ms`，精度 `PASS`。
- A/B `4×128`、`8×64`、`16×16` gather 配置后，ASIC_680ks 分别为 `1.0067 ms`、`0.8715 ms`、`0.8081 ms`。单矩阵最佳配置仅快约 `1.04x`，收益不足以改默认值。
- 将 `16×16` 扩展到 10 个矩阵，精度均 `PASS`。相对旧结果中 H800 cuSPARSE 带宽折算时间的比值范围为 `0.064x–0.483x`，算术平均 `0.278x`；但候选与基线不是同一轮测试，因此仅作诊断数据，不能当作严格 A/B。没有保留 COO 参数变更。

### 第 2 项：`spmv_coo_f32_int_trans` 准备与运行路径

- 检查 `prepare_spmv_coo`：转置 COO 排序、分段信息和 gather plan 在 prepare 阶段建立，prepared 执行复用，没有发现每次 run 重建转置结构的问题。
- ASIC_680ks `warmup=10, iters=50` 默认复测为 `0.5427 ms`，精度 `PASS`。尝试把 Iluvatar gather 参数覆盖扩展到实数转置路径后，`4×128`、`8×64`、`16×16`、`32×8` 分别约 `0.6050`、`0.5433`、`0.5567`、`0.7284 ms`，没有稳定收益，相关实验改动已撤销。
- 旧结果中的 H800 折算基线为 `0.02536 ms`，折算比值约 `0.047x`。这不是同机实测基线；后续应定位该路径与 ASIC 特性的差异，而非重复优化 prepare。

### P1 SELL 参数复测（2026-10-24）

- 在 Iluvatar BI-V150 上按 `warmup=10, iters=50` 对 10 个矩阵、5 个 SELL 变体复测，50/50 精度 `PASS`。下面为各变体相对 H800 cuSPARSE 带宽折算基线的算术平均加速比：`c32 0.516x`、`f16 0.620x`、`f16f32 0.615x`、`f32 0.605x`、`i8i32 0.655x`。
- 对实数变体尝试统一 `group=1, num_warps=1, block_r=32`，仍是 10 矩阵全量复测且 50/50 精度 `PASS`。折算平均比值为 `c32 0.516x`、`f16 0.618x`、`f16f32 0.617x`、`f32 0.625x`、`i8i32 0.624x`。相较同轮默认配置，只有 f32 显示约 `1.044x` 的时间收益；f16 约 `0.994x`、f16f32 约 `1.006x`，i8i32 约 `0.959x`（回退）。收益不一致，统一调度候选已撤销，保留原来的分组设置。
- 下一轮聚焦仅对 Iluvatar f32 变体启用 `group=1/block_r=32`，f16 与 i8i32 沿用当前调度；对照基准需同轮交错测量，以排除运行间波动，再决定是否保留按 dtype 分派。

### 第 3 项：`spvv_c32_int_conj` packed 复数归约

- 原 Iluvatar 实现用一个 Triton kernel 生成实部/虚部 partial，再分别调用 `sum()` 归约，涉及多个 kernel launch。
- 将 CUDA/MACA 已有的 packed real/imag partial + 单个 finalize kernel 路径也用于 Iluvatar；`nnz=1024–65536` 时 `n_parts<=1024`，每个 partial 写入连续的实虚分量，最后一个 kernel 同时归约两部分。
- 精度：`FLAGSPARSE_BACKEND=iluvatar PYTHONPATH=src python3 -m pytest -q tests/pytest/test_q4_variants_accuracy.py -k spvv_c32_int_conj`，`2 passed`。
- 同输入原实现/优化实现 A/B（`warmup=10, iters=50`）测得原实现耗时 `0.0530/0.0551/0.0498/0.0504 ms`，packed 路径耗时 `0.0146/0.0257/0.0259/0.0280 ms`，对应相对原实现加速比 `3.62x/2.14x/1.92x/1.80x`。
- 另以同机 PyTorch 表达式 `(x.conj() * y.index_select(0, indices)).sum()` 为基线复测，四个规模加速比分别为 `3.51x/1.43x/1.68x/2.36x`，算术平均 `2.24x`，均超过 `0.8x`；最大绝对差不超过 `2.2e-5`。这解决了原先低于 `0.8x` 的问题。代码改动位于 `src/flagsparse/sparse_operations/vector_ops.py`。

### 第 4 项：`spmv_csr_f32c32_int_non`（2026-10-24）

- 针对 ASIC_680ks、NACA0015、amazon0601、auto、cfd2 在 Iluvatar BI-V150 上重新测量，默认参数 `warmup=10, iters=50`，精度 `5/5 PASS`。按各矩阵 H800 cuSPARSE 带宽折算基线计算，加速比分别为 `0.391x/0.711x/0.704x/0.595x/0.791x`，算术平均 `0.638x`。
- 扫描 `rows/block = 32/8、64/4、8/16、16/16、8/32`，各配置五矩阵精度均 `PASS`，但都慢于默认，折算加速比算术平均分别为 `0.491x/0.415x/0.482x/0.457x/0.258x`。没有保留 tile 参数改动。
- 对比 Iluvatar 当前 cached kernel（`.cg` 读 CSR 数据/列、`.ca` 读复数向量）和普通缓存策略 kernel，精度均通过。普通 kernel 相对 cached 的加速比在五矩阵上为 `1.106x/0.740x/0.803x/0.701x/0.906x`：只有 ASIC_680ks 受益，另外四个矩阵明显回退，暂不引入矩阵名特判。
- 行长统计显示五矩阵差异明显：ASIC_680ks 平均 `3.41`、P50/P99=`2/16`、最大 `210`、CV=`1.25`；NACA0015 平均 `5.99`、P50/P99=`6/8`、最大 `10`；amazon0601 平均 `8.40`、P50/P99=`10/10`（另有 955 个空行）；auto 平均 `14.77`、P50/P99=`15/22`、最大 `37`；cfd2 平均 `25.02`、P50/P99=`27/30`。因此 ASIC_680ks 的长尾/高度偏斜和其他矩阵的窄分布不应由同一 tile 概括。
- 进一步试验小行组 tile `rows/block=1/4、1/8、2/4、4/4、4/8、8/4`。均为五矩阵 `PASS`，但最好的一组 `8/4` 对默认的逐矩阵时间比（默认/候选）仍只有 `0.66x/0.65x/0.93x/0.84x/0.75x`，折算基线平均加速比 `0.494x`，低于默认的 `0.638x`。单行 CTA `1/4`、`1/8` 极慢，说明减少 CTA 内行间拖尾无法抵偿 CTA 数量膨胀。
- 本轮 kernel 路径探索未找到默认路由的胜者；prepared API 改动仅保留为 opt-in。已有证据不支持简单扩大 tile、warp-per-row 或统一取消 cache modifier；调用侧预分配输出仅节省小部分时间。结果及参数扫描 CSV 均在 `results_csr_f32c32_explore_20261024/`。
- 已将行长统计、分桶排序和 cache 策略选择改成显式 `prepare_spmv_csr_mixed` 计划；`spmv_csr_mixed_prepared`/`flagsparse_spmv_csr(prepared=...)` 的计时只覆盖 run/kernel，预处理不计入。q4 benchmark 的目标变体已切换到该 prepared 方式。
- 原始 mixed call 路径的五矩阵折算加速比为 `0.391x/0.711x/0.704x/0.595x/0.791x`，算术平均 `0.638x`。在同一批输入上切换到 prepared cached run 后为 `0.404x/0.710x/0.704x/0.595x/0.793x`，算术平均 `0.641x`。用同一轮交错 A/B、且两边都预分配 `out`，prepared 相对原始 call 的逐矩阵执行加速比约为 `1.034x/1.000x/1.000x/1.000x/0.999x`，平均约 `1.007x`。
- 强制 `prepared cached` 对全量 10 个矩阵复测（`warmup=10, iters=50`）且精度 `10/10 PASS`，相对 H800 cuSPARSE 带宽折算基线的加速比分别为：ASIC_680ks `0.440x`、GL7d14 `0.970x`、NACA0015 `0.711x`、amazon0601 `0.705x`、auto `0.595x`、cage12 `1.038x`、cfd2 `0.762x`、filter3D `0.832x`、roadNet-TX `0.864x`、wave `0.886x`；算术平均 **`0.780x`**（几何平均 `0.760x`）。
- prepared A/B（同一进程交错、`warmup=10, iters=50`）中，单 kernel 分桶相对 cached 的加速比分别为 `0.527x/0.537x/0.731x/0.425x/0.753x`，精度 `5/5 PASS`，仍明显回退；prepared adaptive 选择为 ASIC_680ks 使用 plain-load、其余四个矩阵使用 cached；ASIC_680ks 的 plain-load 仅有约 `1.040x` kernel 加速，但其余矩阵 plain-load 为 `0.734x/0.802x/0.703x/0.897x`。因此分桶暂不接入默认路由，adaptive 只在 prepared mixed 路径中使用。

#### 复数专用 tile 与 ASIC nnz 均分（2026-10-24）

- 在 ASIC_680ks、auto、amazon0601、NACA0015 上扫描 `ROWS ∈ {8,16,32}`、`BLOCK ∈ {4,8,16,32}`、`WARPS ∈ {1,2,4}`，并比较 cached/plain；候选精度均 `PASS`。`num_stages=1/2/3` 差异很小，未观察到稳定收益。
- 复数路径单独使用 tile 表，平均行长 `<=12` 采用 `16x8x1`，更长的中等行采用 `8x8x1`；`FLAGSPARSE_ILUVATAR_MIXED_ROWS/BLOCK/WARPS` 仍可覆盖。普通 f16/i8 路径不受这张表影响。
- ASIC_680ks 平均行长 `3.41`、最大行长 `210`，自动路由改为在 prepare 阶段生成 `row_ids`/空行计划并使用 nnz 均分 segmented-scan kernel；扫描中 `BLOCK=128, WARPS=1` 最好，直接 kernel 约 `0.082 ms`，集成 prepared 调用多轮中位数约 `0.0890 ms`。计划生成不计入 kernel 计时。
- prepared mixed 路径同时去掉了输出已直接写入 `out` 时的冗余 `out.copy_(y)`。精度回归为 `2 passed`，全量 10 矩阵为 `10/10 PASS`。
- 全量复测使用 `warmup=20, iters=100`，每个矩阵 3 轮取中位数，基线为 H800 cuSPARSE 带宽折算值。相对基线加速比分别为：ASIC_680ks `0.676x`、GL7d14 `0.968x`、NACA0015 `0.711x`、amazon0601 `0.705x`、auto `0.622x`、cage12 `1.052x`、cfd2 `0.793x`、filter3D `0.855x`、roadNet-TX `0.857x`、wave `0.896x`；算术平均 **`0.814x`**，几何平均 **`0.803x`**。
- 相比前一轮 prepared cached 的算术平均 `0.780x`、几何平均 `0.760x`，本轮分别提升约 `0.033x` 和 `0.044x`。结果文件为 `results_csr_f32c32_explore_20261024/prepared_auto_10mat_repeat.csv`；扫描结果在同目录的 `complex_tile_sweep4_stage2.csv`、`complex_tile_target_stages.csv`、`nnz_complex_sweep.csv`。

#### 合入 main 时的修正（2026-10-24，CUDA 机上做）

- 天数补丁把**共用的**实数 tile 表 `_iluvatar_mixed_spmv_tile` 的 12–24 档从 `16×8` 改成了 `8×8`（f16f32 / i8 / f16
  都走这张表，没有实测），与上文"普通 f16/i8 路径不受影响"不符，按 abb5e84 恢复；复数专用表保留。
- prepare 内不再用 TensorPlanCache：分桶、row-id、cache 策略都只在 prepare 里算一次，存在计划对象里。
- MUSA、Ascend、ROCm f16→f16 的 prepared 计划回到各自原有的 call 路径（`route == "call"`，bench 记为 `timed=call`），
  避免 bench 切到 prepared 后绕开 MUSA 调好的路径、在 Ascend 上直接跑 Triton。
- prepared 运行补了 `out.device` 检查；不连续的 `out` 先写临时张量再拷回（原来会写进 `reshape` 出来的副本后丢掉）。
- CUDA 上 prepared 路径也走直接发射；`nnz` 自动路由在 CUDA 上同样启用：5090 上 ASIC_680ks 的 nnz kernel
  11.0 µs，cuSPARSE 15.7 µs，row-tile 27 µs。同进程交替 A/B（负载 500）：call 路径几何平均 0.908 → prepared 1.050，
  ASIC 0.58 → 0.77（剩下的是主机开销）。
- 新增 `tests/pytest/test_spmv_csr_mixed_prepared.py`（5 个路由 × 空行/长行/0 与 1 行/int64/strided out）。

### 第 5–7 项及后续

- 第 5–7 项仍待完成：SELL 变体逐项调参、COO mixed 短行的新路由思路、以及 Iluvatar 兼容的 `spgemm_csr` 实现。
- 本轮保留的 tracked 源码优化还包括既有工作区改动 `src/flagsparse/sparse_operations/spmv_csr.py` 的 Iluvatar complex64 短行路由与 tile 参数。修改前后性能结论应使用对应同机基线和同一计时配置复测。

## 下一轮天数优化方向（2026-10-25，CUDA 机上分析）

> SELL 和 COO 两组的详细计划（瓶颈探针、prepare 时重编码成 SELL-C-σ / nnz 均分、逐项预期和排期）见
> [`ILUVATAR_SELL_COO_PLAN.md`](ILUVATAR_SELL_COO_PLAN.md)；本节的方向 1、2、4、5 在那里展开。

依据：10-24 全量结果 `results_biv_65_20261024_final` 的逐矩阵数据（我们的耗时用 `折算基线 ÷ 加速比` 反推）、
`tests/data` 的行/列长分布、当前代码，以及在 5090 上补测的两组数据（下文标"CUDA 实测"）。上文"已验证的调优进展"
里天数已经试过且回退的做法，这里不再建议。

从数据里看出三类问题，下面的方向都落在其中一类：

1. **长尾结构走进了按行切分的 gather**：按行长排序的行顺序，或一批行里有一条极长的行，整批都要等它。
   ASIC_680ks 的行和列都是"平均 3.4、最长 210"；amazon0601 的行很均匀（最长 10），**列却是幂律分布**
   （平均 8.4、P99 53、最长 2751、变异系数 1.82），所以 amazon 只在转置/共轭方向上慢。
2. **天数的 warp 是 64 宽，有的 kernel tile 只有 32 宽**：一半 lane 常年被 mask 掉（SELL）。
3. **按行的 gather / row-tile 类 kernel 带宽利用率只有 17–25%**（auto 上约 200–240 GB/s，实测带宽 1150 GB/s）；
   SELL 按含填充的存储量算是 324–597 GB/s，是目前天数上利用率最高的 kernel。

| 优先级 | 方向 | 涉及变体（10-24 加速比） | 改动量 | 前置条件 |
|---|---|---|---|---|
| 1 | 转置/共轭方向的长尾走 nnz 均分 | coo_f32_trans 0.776、csr_f32_trans 0.894、coo/csr_c32_conj 的 ASIC/amazon | 小（kernel 现成，实数可直接用） | 无 |
| 2 | COO / CSR 实数混合精度的 ASIC 走 nnz 均分 | coo_f16f32 0.730、coo_f16 0.762、coo_i8i32 0.771 | 小（COO 自带行号） | 无 |
| 3 | SpGEMM 改用 MACA 的无 TLE 实现 | spgemm 0.064 | 小（放开两处门控） | 先过精度 |
| 4 | SELL 让 64 宽 warp 吃满 | sell 五项 0.52–0.65 | 中（加覆盖开关后扫参） | 无 |
| 5 | 复数 gather：先测带宽上限 | csr_c32 0.606、coo_c32 0.522 | 先测量，后决定 | 无 |
| 6 | 原生 f32 CSR 换 row-tile | csr_f32（0.929，不在名单） | 小 | 无 |
| 7 | 主机固定开销（约 22–30 µs） | spvv f32、小矩阵 | 先测量 | Triton 3.6 的启动器签名 |

### 方向 1：转置/共轭方向的长尾改走 nnz 均分（优先做）

- **数据**：同一个矩阵，非转置快、转置慢。ASIC 上 `spmv_coo_f32` 非转置 163 µs，转置 538 µs；CSR 转置 494 µs。
  c32 的共轭（COO 835 µs、CSR 820 µs）和修复前的 COO c32 非转置（843 µs）几乎一样——这三条走的都是按行长
  排序的行顺序；而用自然行顺序的 CSR c32 非转置只要 134.8 µs。amazon 上转置 312 µs，非转置 179 µs。各变体在这两个矩阵上的加速比：
  coo_f32_trans ASIC 0.12 / amazon 0.44，csr_f32_trans 0.17 / 0.46，coo_c32_conj 0.149 / 0.735，csr_c32_conj 0.173 / 0.809。
- **原因**：转置 plan 由 `_spmv_csr_transpose.prepare()` → `make_plan()` 生成。ASIC 的转置结构满足
  `avg <= 4 and maximum > 128`，于是行顺序变成 `argsort(counts)`（按长度排序）。CSR c32 非转置在天数上专门
  去掉了这个排序（`spmv_csr.py` 约第 1126 行，平均行长 ≤4 时把 `plan[3]` 换成 `arange`），它是这几条里唯一快的。
  amazon 平均列长 8.4，不触发排序，但一批 32–64 列里只要有一条长达 2751 的列，整批就要循环几百次。
  天数在"第 2 项"里扫过实数转置的 lanes/batch，没有收益——这和"问题在顺序和拖尾，不在 lane 数"一致。
- **CUDA 实测**（5090，GPU 时间，`op=trans` 的 f32 CSR；nnz 均分 = 在转置结构上跑现有 `_nnz_kernel`）：

  | 矩阵 | cuSPARSE | 当前转置路径 | nnz 均分（BLOCK=256, 1 warp） |
  |---|---:|---:|---:|
  | ASIC_680ks | 21.9 µs | 29.7 µs | **7.8 µs** |
  | amazon0601 | 39.1 µs | 64.4 µs | **27.1 µs** |
  | roadNet-TX | 34.1 µs | 13.6 µs | 13.5 µs |
  | auto | 73.5 µs | 51.7 µs | 56.2 µs（变慢） |

  长尾矩阵快 2.4–3.8 倍，均匀矩阵持平或变慢，所以**只能按结构条件启用**。
- **怎么做**：
  1. 先做一个零成本 A/B：天数上让 `make_plan` 对 ASIC 这类结构也用 `arange`（和 CSR c32 非转置同样的处理），
     测 coo_f32_trans、csr_f32_trans、两个 c32 conj 在 ASIC 上的耗时。
  2. 再做 nnz 均分：`_spmv_csr_transpose.prepare()` 已经算出 `order`（按列排序）和 `source_rows`。在 plan 里
     多存一份 `indices[order]`（排好序的列号，就是 Aᵀ 每个 nnz 的"行号"）和空列列表，run 时调用
     `_nnz_kernel(values[order], source_rows[order], ptr, indices[order], x, y, empty, ...)`。values 的重排已经
     由 gather 的缓存做了（`cache["values"]`），不额外增加每次调用的开销。COO 转置更简单：prepare 里排好序的
     `col_trans` 本身就是行号。
  3. 启用条件不要沿用 `mean <= 4 and max > 128`（会漏掉 amazon），建议
     `max_len >= 128 and max_len >= 32 * mean_len`：10 个矩阵里只命中 ASIC（210 ≥ 109）和 amazon 的列（2751 ≥ 269），
     GL7d14（列最长/平均 4.1）、filter3D、wave、roadNet 都不命中。
  4. 实数 f32 可以直接用现有 kernel；c32 conj 需要第 5 节的"复数 × 复数"扩展，再加共轭标志。
- **预期**：如果天数上的提升倍数接近 f32c32 在 ASIC 上的经验（0.44 → 0.676），ASIC 的转置能从 0.12–0.17 提到
  0.3–0.5，amazon 从 0.44 提到 0.7 以上；coo_f32_trans 这一项的 10 矩阵均值有望过 0.8。

### 方向 2：COO / CSR 实数混合精度在 ASIC 上走 nnz 均分

- **数据**：f16f32、f16、i8i32 这三项 COO 在天数上走 "COO 转 CSR + 和 CSR 混合精度同一个 row-tile kernel"，
  耗时和 CSR 完全一样（ASIC 上都是约 122 µs）；COO 比值更低只是因为 H800 上 cuSPARSE 的 COO 比它自己的 CSR 快
  10–30%。三项里 ASIC 都是最差的（0.28–0.32），CSR 的同类项在 ASIC 上也只有 0.39–0.41。
- **怎么做**：`prepare_spmv_coo` 默认 `sort_by_row=True`，按 (行, 列) 排好序，**行号数组现成**，nnz 均分只需要额外一个空行列表。把
  `spmv_coo_mixed` 天数分支里懒构建的 CSR 转换（现在挂在 `_ROCM_COO_CSR_CACHE` 上）挪进 `prepare_spmv_coo`，
  对满足方向 1 第 3 条结构条件的矩阵走 `_nnz_kernel`。CSR 混合精度则沿用 `prepare_spmv_csr_mixed` 已有的 nnz 计划，
  把 dtype 门控从 f32×c64 放宽到实数 f16/i8。
- ~~坑：非 MUSA 分支的浮点字面量会把 int32 累加提升成 float32~~ ——**2026-10-25 已核实不是问题**：一行和为 32257873
  （大于 2^24 的奇数）的 i8→i32 实测逐位精确，Triton 的 Python 标量是弱类型（详见 `ILUVATAR_SELL_COO_PLAN.md` 6.2）。
- **另外复测一次**：filter3D 上 COO f16→f16 是 99 µs，而 COO f16→f32、CSR f16→f16 都是 49 µs，三者是同一个 kernel。
  先确认能不能复现，能复现再打印实际走的路由。

### 方向 3：SpGEMM 改用 MACA 的无 TLE 实现

- **数据**：天数上 SpGEMM 全走 ESC，耗时几乎都在 count 阶段：55–360 ms，折算基线 1.9–31 ms，慢 10–30 倍。
- **CUDA 实测**（5090，A·A，float32）：MACA 的纯 Triton 实现 `_spgemm_maca.compute`（全局内存 `atomic_cas` 哈希）
  结果正确（相对误差 ≤ 3.5e-7），并且明显快于 ESC：

  | 矩阵 | MACA 实现 | ESC | TLE（天数用不了） |
  |---|---:|---:|---:|
  | cage12 | 4.8 ms | 27.0 ms | 3.2 ms |
  | roadNet-TX | 6.9 ms | 9.3 ms | 2.8 ms |
  | ASIC_680ks | 4.3 ms | 14.0 ms | 3.0 ms |

- **怎么做**：`spgemm_csr.py` 里两处 `_is_maca_runtime()` 门控——`prepare_spgemm_csr` 的 `build_metadata`，以及
  `_spgemm_compute` 开头的 `_maca_triton_spgemm_compute`——先用环境变量开关（例如
  `FLAGSPARSE_ILUVATAR_SPGEMM=maca`）对天数放开，精度通过后再改默认。
- **风险**：依赖全局内存的 `tl.atomic_cas` 和 `atomic_add(sem="relaxed")`；MACA 代码注释写着 C550 上"多 warp 的
  128 元素布局会丢 lane，所以 `_hash_rows` 固定 `num_warps=1`"。天数的 warp 也是 64 宽，保持 `num_warps=1`，
  并且先跑 spgemm 的全部精度用例再看性能。
- **预期**：如果天数上 ESC → MACA 的倍数和 CUDA 上相近（1.3–5.6 倍），0.064 能提到 0.15–0.3。离 0.8 仍远，
  但这是从"换算法"开始，下一步才是在天数上调它的 block 和哈希上限（`limit=4096`）。

### 方向 4：SELL 五项让 64 宽 warp 吃满

- **数据**：按含填充的存储量（`sell_values × (值 + 列号)`）算，f32 SELL 跑到 324–597 GB/s，已经是天数上带宽
  利用率最高的 kernel。cuSPARSE 用的是同一份带填充的数据，读的字节数一样，差距在效率上。填充率：ASIC 3.98 倍、
  filter3D 2.48 倍、roadNet 1.48 倍，其余 1.07–1.30 倍。
- **代码现状**：实数路径在 `slice_size <= 32` 时设 `BLOCK_R=64`、`num_warps=1`，n_slices ≤ 32768 时 `GROUP=1`。
  SLICE 是 32，`r < SLICE` 的掩码让 64 个 lane 里有 32 个一直空着；`GROUP=2` 时 tile 是 2×64=128 个元素，同样
  一半被 mask。复数路径用 `_spmv_sell_complex_kernel`，tile 是 1×32，`num_warps=1`，也是半个 warp；而且实部、
  虚部是 4 次分开的标量读取。
- **已经试过、不要重复**：slot 并行、行×slot tile（32×4、64×4、128×2）、CSR 压缩后 row-tile、complex slot=8、
  双 warp、统一 `group=1, block_r=32`（见 `ILUVATAR.md` 10-21/10-22 和上文 P1 SELL 复测）。
- **没试过的**：
  1. **刚好填满的 tile**：`GROUP=2, BLOCK_R=32, num_warps=1`（两个 slice × 32 行 = 64 lane，每个 lane 一行，没有
     被 mask 的 lane），以及 `GROUP=4, BLOCK_R=32, num_warps=2`。代价是同一个 program 的循环次数取两个 slice
     宽度的较大值，ASIC、filter3D 这两个填充多的矩阵要单独看。旁证：roadNet 有 43544 个 slice，超过 32768
     走 `GROUP=2`，跑到 554 GB/s；NACA0015 有 32475 个 slice，正好低于阈值走 `GROUP=1`，只有 362 GB/s。
     roadNet 的 slice 更窄，这只是线索，不是结论；最便宜的验证是把阈值调低后 A/B。
  2. **复数**：同样用 2 个 slice 填满 warp，并把值和 x 都按 `(…, 2)` 成对读取（8 字节一次），参考
     `_csr_spmv_real_by_complex_cached_kernel` 的写法。
  3. **先加覆盖开关**：现在只有 ROCm 有 `FLAGSPARSE_SELL_ROCM_GROUP/WARPS`。给天数加
     `FLAGSPARSE_SELL_ILUVATAR_GROUP/BLOCK_R/WARPS`，扫参就不用每次改代码。
- **另外**：H800 上 SELL 对 cuSPARSE 也只有 0.42–0.66，说明 kernel 本身偏弱，不只是天数的问题；CUDA 这边可以
  对着真实 cuSPARSE 同步调。

### 方向 5：复数 gather（csr_c32 0.606、coo_c32 0.522）：先测带宽上限

- 除 ASIC 外，COO c32 和 CSR c32 每个矩阵的耗时都一样（auto 都是约 360 µs），走的是同一个已经调得很细的复数
  gather。以 auto 为例，要到 0.8 需要 196 µs，相当于有效带宽从约 240 GB/s 提到约 440 GB/s，调 batch/lanes 补不上。
- **先测**：在天数上用 Triton 写一个纯连续读的 kernel（copy 或 axpby），量出能达到的带宽；再用天数的 profiler 看
  gather kernel 在 auto 上的显存吞吐和 L2 命中率。连续读也只有 300 GB/s 左右，就说明 Triton 在天数上本身到顶，
  在报告里写明；能到 800 GB/s 以上，问题在 gather 的访存方式。
- **结构性改动**：把 `_nnz_kernel` 扩展成"复数 × 复数"（每个 nnz 4 次乘法、实部虚部两个分段扫描、共轭标志），
  COO 直接用自带的行号。每个 nnz 多读 4 字节行号（12 → 16 字节），所以只有效率提高 1.5 倍以上才划算；
  CUDA 上它在规则矩阵上输给 row-tile（f32c32 的 cage12：11.5 µs 对 7.9 µs），天数上要对 10 个矩阵逐个 A/B。
  方向 1 的 c32 conj 也需要这个扩展。
- roadNet 的 c32（0.357 / 0.292）等天数刚加的复数短行路由（平均 ≤3 且最长 ≤16 走 row-tile 32×4）的复测结果。

### 方向 6：原生 f32 CSR 换成 row-tile（不在 0.8 名单里，但绝对收益大）

- 天数上 `spmv_csr_f32_int_non` 走 `legacy_segbin`，auto 上 592 µs；同一矩阵上混合精度的 row-tile kernel 读的数据
  更多（x 是复数），只要 266.6 µs。它现在 0.929，是因为分母是本机 legacy `cusparseScsrmv`（auto 上 512.7 µs，同样很慢）。
- 把天数 f32 非转置路由到 row-tile（或直接复用 `prepare_spmv_csr_mixed` 的实数路径），绝对耗时有望减半，这一项
  对本机厂商基线的比值相应翻倍。

#### 2026-10-26 真机验证

- 先验证 PyTorch CSR：BI-V150 上 `torch.sparse.mm(torch.sparse_csr_tensor(...), x)` 在
  `auto/NACA0015/ASIC_680ks` 等矩阵上返回全零或明显错误结果，不能作为生产路由；本轮只保留其失败记录，
  没有把 PyTorch CSR 的时间写入加速比。
- 新增 Iluvatar `float32` 非转置 CSR 的 row-tile 开关，默认走
  `FLAGSPARSE_ILUVATAR_SPMV_F32_ROUTE=row_tile`，并保留 `=segbin` 回退。默认参数为
  `ROWS=16, LANES=8, WARPS=1, STAGES=1`；显式 `alg` 请求仍不被覆盖。
  （合入 main 时改为在 `prepare_spmv_csr` 里读取这 5 个环境变量并存进 plan，运行阶段不再读环境变量；
  prepare 之后再改环境变量对已有 plan 不生效。模拟 Iluvatar 后端的小矩阵上每次调用主机开销约 40→34 µs。）
- 真机扫描在 `ASIC_680ks/NACA0015/amazon0601/auto` 上均通过精度，最优共同配置为
  `16×8, 1 warp, 1 stage`。全量 10 矩阵 `warmup=20, iters=50` 复测精度 `10/10 PASS`：

  | 矩阵 | row-tile ms | H800 折算基线 ms | 相对基线加速比 |
  |---|---:|---:|---:|
  | ASIC_680ks | 0.143067 | 0.167077 | 1.1678x |
  | GL7d14 | 0.061605 | 0.182708 | 2.9658x |
  | NACA0015 | 0.185692 | 0.232703 | 1.2532x |
  | amazon0601 | 0.107460 | 0.186378 | 1.7344x |
  | auto | 0.252582 | 0.257248 | 1.0185x |
  | cage12 | 0.039748 | 0.169550 | 4.2657x |
  | cfd2 | 0.070357 | 0.144585 | 2.0550x |
  | filter3D | 0.076246 | 0.149654 | 1.9628x |
  | roadNet-TX | 0.220851 | 1.061870 | 4.8081x |
  | wave | 0.043240 | 0.166623 | 3.8535x |

- 算术平均加速比为 **2.5085x**，几何平均为 **2.1818x**；所有误差均在现有 FP32
  正确性门限内。结果文件为 `results_csr_f32_rowtile_default_20261026.csv`，参数扫描为
  `results_csr_f32_rowtile_smoke_20261026.csv`。
- 另外用 `int64/int64` 索引全量复测，精度 `10/10 PASS`；结果文件为
  `results_csr_f32_rowtile_int64_20261026.csv`。强制 complex row-tile 和 COO mixed tile
  的后续 A/B 没有稳定收益，因此没有改变它们的默认路由。

### 2026-10-26 PyTorch 原生聚合探针

- 在 BI-V150 上对低速算子试了 PyTorch `scatter_add_`、`index_add_`、`segment_reduce`，并对
  SELL 预处理布局做了缓存行号后的聚合 A/B。`torch.sparse.mm` 没有重新接入：本机对 CSR/COO
  稀疏 GPU 矩阵仍报 `addmm/addmv requires compiling PyTorch with CUDA 11+`，此前 CSR 还出现过
  全零结果。
- SELL 的 PyTorch 聚合在 10 个矩阵上均慢于现有 SELL kernel；COO mixed、CSR mixed 和
  `spvv_c32_int_conj` 也没有稳定收益。不能用这些路径替换默认 kernel。
- 唯一稳定候选是 `spmv_coo_f32_int_trans`：转置后的输出行平均长度不超过 4 且最大行长超过
  128 时，prepare 阶段缓存 int64 行/列索引，运行阶段用 PyTorch `scatter_add_`。这正好命中
  `ASIC_680ks`，并避开每次调用的 int32→int64 转换；开关为
  `FLAGSPARSE_ILUVATAR_COO_TRANS_TORCH=auto/0/1`，默认 `auto`，其余结构自动回到原 Triton
  转置路径。
- 同进程交替 A/B、`warmup=4/15`、`iters=40` 的 10 矩阵结果见
  `results_coo_f32_trans_torch_ab_20261026.csv`。相对 H800 cuSPARSE 带宽折算基线，算术平均由
  `0.7622x` 提到 `0.7723x`，几何平均由 `0.6420x` 提到 `0.6837x`；ASIC 单项由 `0.1261x`
  提到 `0.2396x`，输出与原路径误差在 FP32 容差内。
- 本轮还修正了 Iluvatar COO f32 非转置误将 `_CooNnzPlan`（复数 kernel）作为默认计划的问题；
  该修正没有把 PyTorch 路径扩展到其他 dtype。路由单测为 `5 passed`。

### 2026-10-26 索引宽度 A/B

- 对 `spmv_coo_f32_int_trans` 在同一进程交替复测了原 Triton 转置路径的 `int32`/`int64` 索引；
  10 个矩阵全部精度一致，`int32` 相对 `int64` 的算术平均加速比为 **`1.446x`**，几何平均为
  **`1.440x`**，逐矩阵范围为 `1.268x–1.668x`。明细见
  `results_coo_f32_trans_index_dtype_ab_20261026.csv`。
- 当前交付数据本来就使用 `int32` 的 row/col/indptr；因此这不是新增默认优化，而是确认不能在
  热路径中把索引无条件 `.long()`。`int64` 仍需保留给 API 兼容和超出 int32 范围的回退。
- PyTorch `scatter_add_` 在 BI-V150 上要求索引为 `int64`（`int32` 会报
  `scatter(): Expected dtype int64 for index`），所以刚加的 PyTorch 长尾路径必须在 prepare
  阶段缓存 `int64` 索引；不能靠“只用 int32”进一步提速。原 Triton 路径仍由
  `FLAGSPARSE_ILUVATAR_COO_TRANS_TORCH=0` 回退。

### 方向 7：主机固定开销（天数上约 22–30 µs）

- spvv 在 nnz 从 1k 到 64k 之间一直是 23–30 µs，SELL i8 在 cage12 上 22 µs，CSR i8 在 cage12 上 25 µs，说明天数上一次
  调用的固定开销在 22–30 µs，小规模用例会被它卡住（spvv f32 有两档是 0.96）。
- **先测**：天数上一个空 Triton kernel 走 JIT 和走 `CompiledKernel.run` 各要多少 µs（5090 上是 12 µs 对 7 µs，
  脚本思路见上文 f32c32 一节）。如果直接发射在天数的 Triton 3.6 上能用，就把 `mixed_spmx._CUDA_DIRECT_LAUNCH`
  的条件扩展到天数（签名对不上会在第一次 TypeError 时自动退回 JIT），再把 spvv 接到同一个 launcher。

### 测量和验收

- 每个方向都要同一进程交替 A/B，每个矩阵至少 3 轮取中位数，报**逐矩阵**数字，不只报均值；结构条件命中了哪些
  矩阵要写出来。
- 新路由要补 pytest：长尾列、空列、空行、int32 大整数和、strided `out`；转置/共轭的 nnz 路由要覆盖 COO 未排序输入。
- 改到 `spmv_csr.py` 的调度函数（`flagsparse_spmv_csr`、`_execute_spmv_route` 等）时，在无 torch 环境下跑一遍
  `tests/ci`：`test_spmv_csr_policy.py` 把这些函数单独抽出来、放进手工列出依赖的桩环境里执行，新引用的名字
  要补进桩环境，否则只在 CI 上报 `NameError`。
