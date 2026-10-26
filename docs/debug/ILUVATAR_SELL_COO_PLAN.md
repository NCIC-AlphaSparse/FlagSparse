# 天数 BI-V150：SpMV SELL / COO 优化计划（2026-10-25）

SELL 五项和 COO 五项是天数上最难的两组：参数已经扫过多轮（见 `ILUVATAR.md` 10-21/10-22 和 `LOW_SPEEDUP.md`
"已验证的调优进展"），单纯调 tile 已经没有空间。本计划按天数的实测数据重新拆解瓶颈，并利用一条新放宽的前提：

> **较重的预处理可以放进 prepare，只统计 run 阶段的 kernel 时间**（和 cuSPARSE 提前建描述符、分配 buffer 对等）。

这意味着可以在 prepare 里把用户给的 SELL / COO **重新编码成更适合天数的内部格式**，运行时只跑一次 kernel。

> **2026-10-25 更新**：第 0 步的探针脚本 `tools/biv_spmv_probe.py` 和 5.1/5.2 的 `prepare_spmv_sell` 已经实现，
> 交付 benchmark 可以用环境变量切到 prepared 路径。天数上要跑什么、怎么判断，直接看**第 9 节**。

数据来源：10-24 全量结果 `results_biv_65_20261024_final`（逐矩阵耗时用 `折算基线 ÷ 加速比` 反推）、`tests/data`
的结构统计、当前代码，以及 2026-10-25 在 5090 上做的原型验证（标"CUDA 原型"，**只证明格式和正确性，不代表天数性能**）。

---

## 1. 现状

| 变体 | 10 矩阵平均 | 最差的矩阵 | 现在走的路径 |
|---|---:|---|---|
| `spmv_sell_f32_int_non` | 0.630 | NACA0015 0.48、filter3D/roadNet/amazon 0.51 | `_spmv_sell_real_kernel` |
| `spmv_sell_f16_int_non` | 0.621 | filter3D 0.45、NACA 0.48、ASIC/amazon 0.50 | 同上 |
| `spmv_sell_i8i32_int_non` | 0.654 | NACA/GL7d14/amazon 0.48–0.49 | 同上 |
| `spmv_sell_c32_int_non` | 0.516 | NACA 0.35、amazon 0.39、GL7d14 0.47 | `_spmv_sell_complex_kernel` |
| `spmv_coo_c32_int_non` | 0.472（天数改 ASIC 后报告 0.522） | roadNet 0.29、auto 0.44、NACA 0.48 | `_spmv_csr_transpose.gather`（复数） |
| `spmv_coo_f16f32_int_non` | 0.730 | ASIC 0.30、auto 0.66、NACA/roadNet 0.67 | COO→CSR + 混合精度 row-tile |
| `spmv_coo_f16_int_non` | 0.762 | ASIC 0.32、filter3D 0.48、NACA 0.69 | 同上 |
| `spmv_coo_i8i32_int_non` | 0.771 | ASIC 0.28、GL7d14 0.54、NACA 0.67 | 同上 |
| `spmv_coo_f32_int_trans` | 0.776 | ASIC 0.12、amazon 0.44、roadNet 0.62 | 转置 gather（`transpose_csr_plan`） |

（SELL 的 f16→f32 也测了，0.623，和 f16 基本一样，下文不单列。）

## 2. 天数上的硬约束和已知数据（做计划的前提）

**硬件和软件**
- warp 64 宽；多轮扫描下来，大多数 kernel 在 `num_warps=1`、小 tile 时最好（f32c32 复数表 16×8×1，SELL 单 warp）。
- CoreX Triton 3.6.0；TLE 共享内存不可用；`torch.sparse` 会静默算错（不能当参考）。
- 计时：`_benchmark_cuda_op` 在天数上用设备 event 包住整个循环。主机比 GPU 慢时，GPU 的空闲间隙也会算进去；
  天数每次调用的固定开销约 22–30 µs（spvv 在 nnz 1k–64k 之间一直是 23–30 µs）。SELL benchmark 用
  `validate=False`，每次调用没有同步。

**现有 kernel 在天数上的带宽利用率**（大矩阵；字节数按该路径实际读的数组算）

| kernel | 有效带宽 | 占实测 1150 GB/s |
|---|---:|---:|
| SELL f32（含填充） | 316–355 GB/s | 27–31% |
| SELL f16 / i8 | 267–295 / 250–285 GB/s | 22–26% |
| CSR 混合精度 row-tile（COO 实数混合精度也走它） | 220–280 GB/s | 19–24% |
| 复数 gather（COO / CSR c32） | 250–330 GB/s | 22–29% |
| nnz 均分 `_nnz_kernel`（f32c32） | auto 约 185 GB/s（431.6 µs）；ASIC 81.8 µs | — |

cage12、wave、cfd2 这几个小矩阵能跑到 450–600 GB/s，很可能整个装进了 L2；**丢分集中在大矩阵**
（NACA0015、auto、amazon0601、GL7d14、ASIC_680ks、roadNet-TX、filter3D）。

**要达到 0.8 需要多少带宽**：按"我们的路径要读的字节数 ÷（折算基线 ÷ 0.8）"算，所有变体、所有矩阵都**不超过
800 GB/s**（实测带宽的 70%）。COO 实数混合精度在大矩阵上只需要 240–335 GB/s（现在 220–280，差 15–25%，
ASIC 除外）；SELL 需要 337–574 GB/s；COO c32 需要 450–650 GB/s，是最难的一项。

**结构统计**（`tests/data`）
- ASIC_680ks：行、列都是平均 3.4、P99 16、最长 210、变异系数 1.25——长尾。
- amazon0601：行很均匀（最长 10），**列是幂律分布**（平均 8.4、P99 53、最长 2751）——只在转置方向长尾。
- SELL（slice=32，不排序）的填充率：ASIC 3.98、filter3D 2.48、roadNet 1.48、NACA/GL7d14 1.30、auto 1.27、wave 1.26、
  amazon 1.19、cfd2 1.08、cage12 1.07。

**天数已经试过、不要重复的**
- SELL：slot 并行、行×slot tile（32×4、64×4、128×2）、CSR 压缩后 row-tile、CSR 轻量 row-tile、complex slot=8、
  双 warp、compiled launcher、循环流水化、统一 `group=1, block_r=32`。
- COO c32：gather 的 batch×lanes（4×128、8×64、16×16），ASIC 上最好只快 1.04 倍；双行串行累加、成对加载、K 归约 tile。
- COO f32 转置：ASIC 上扫 lanes×batch（4×128、8×64、16×16、32×8），无收益；prepare 已缓存转置结构，不是每次重建。
- f32c32（同一类 row-tile kernel）：rows/block 扫描、warp-per-row、cache modifier 开关、单 kernel 分桶，都不如默认。

## 3. 总体思路：prepare 里统一编码成两种内部格式

SELL、COO 非转置、COO 转置在数学上都是"按行（或按 Aᵀ 的行）求和"。既然 prepare 不计时，就不必在运行时迁就
用户给的格式，而是统一编码成两种格式之一，把调优精力集中在两个 kernel 上：

- **格式 A：SELL-C-σ**（规则结构）。在 σ 行的窗口内按行长降序排序，再每 C 行切一个 slice，按列主序存放。
  σ=1024 时 C=32 的填充率降到 1.01–1.07（ASIC 1.17），C=64 降到 1.02–1.11（ASIC 1.39）。y 按原始行号写回（存一个
  `perm` 数组，每行写一次）。
- **格式 B：nnz 均分的分段扫描**（长尾结构）。每个 program 处理固定数量的 nnz，COO 自带行号，转置时用排好序的
  列号当行号，几乎不需要额外元数据。
- **选择规则**（prepare 时按结构算一次）：`最长 ≥ 128 且 最长 ≥ 32 × 平均` 走 B，否则走 A。10 个矩阵里只有 ASIC
  的行/列、amazon 的列命中 B。天数数据支持这个划分：nnz kernel 在 ASIC 上 81.8 µs（row-tile 约 129 µs），在 auto 上
  431.6 µs（row-tile 266.6 µs）。

**这改变了比较的内容，报告里必须写明**：cuSPARSE 读的是用户给的带填充 SELL / 原始 COO；我们读的是 prepare 时
重排过的数据。和已有的 `timed=prepared` 一样，在 CSV 里标出 `layout=sell_c_sigma` / `layout=nnz_balanced`。

## 4. 第 0 步：先在天数上把瓶颈测清楚（半天到一天，后面的取舍都依赖它）

在 NACA0015、auto（大而规则）和 ASIC（长尾）上，用 f32 跑下面几个探针 kernel，每个都同一进程交替测、取 3 轮中位数：

| 探针 | 做什么 | 回答的问题 |
|---|---|---|
| P1 连续读 | Triton 连续读 + 求和，扫 BLOCK、`num_warps` 1/2/4 | Triton 在天数上能跑到的带宽上限 |
| P2 现行 SELL | 现在的 `_spmv_sell_real_kernel` | 基准 |
| P3 去掉 x 的间接访问 | 同一个 kernel，把 `x[col]` 换成 `x[row]` | x 随机读的代价 |
| P4 去掉列号依赖 | 填充位置存 `col=0, val=0`，值的读取不再等列号（`mask` 只看 slot 是否在 slice 宽度内） | 列号→值这条依赖链的代价 |
| P5 占用率 | P2 在 `GROUP`/`num_warps` 不同组合下，program 数为 SM 数的 1/4/16/64 倍 | 是不是并发度不够、延迟没藏住 |
| P6 σ 重编码 | `prepare_spmv_sell` 扫 σ 和 (c, group, warps)，复数加成对读取 | 5.1/5.2 在天数上的实际收益 |
| P7 RCM 重排 | 方阵在 prepare 里按 RCM 重新编号，计时含每次调用的 x 重排 | x 局部性能换回多少（见 5.6） |

再用天数的 profiler 看 P2 在 auto 上的显存吞吐、L2 命中率、achieved occupancy 和访存 stall 占比。

**怎么用结果做决定**

- P1 本身只有 300–400 GB/s → Triton 在天数上就到顶了，SELL/COO 很难再翻倍，后面以减少字节数（σ 排序、压缩）为主，
  并在报告里写明上限。
- P3 明显快于 P2 → 瓶颈在 x 的随机读：σ 窗口取小一些保局部性，看 P7 的 RCM 重排（见 5.6）。
  注意 P3 里的 x 读取**不带掩码**：5090 上带掩码读"每个 lane 固定的地址"在 ASIC 上反而比真实 gather 慢 3 倍，
  会把这个探针的结论弄反；地址总在范围内、填充位置的值已是 0，所以去掉掩码不影响含义。
- P4 明显快于 P2 → 依赖链没被藏住：格式 A 统一按 P4 的方式存填充。
- P5 里 program 越多越快 → 并发度不够：减小 tile、增加 program 数，或者每个 program 同时处理多个 slice。

> **CUDA 原型的对照**：5090 上（kernel 时间，每次清空 L2）P4 的写法没有收益——ASIC 上 98.3 µs 对 84.0 µs，因为
> 原来用 `col >= 0` 遮掉了填充位置的值，少读了字节；σ 排序后填充很少，两种写法差别会变小。天数是否受益，以 P4 为准。

## 5. SELL 计划

### 5.1 σ 排序重编码（确定的收益，先做）

- **为什么**：填充位置的列号一定要读，cuSPARSE 也一样。σ 排序后填充率从 1.07–3.98 降到 1.01–1.17，字节数直接减少。
- **CUDA 原型**（5090，f32，kernel 时间，每次清空 L2；"现行"和"σ 排序后"都用 CUDA 默认配置 `GROUP=4, num_warps=4`）：

  | 矩阵 | cuSPARSE SELL | 现行 kernel | σ=1024 排序后 | 填充率变化 |
  |---|---:|---:|---:|---|
  | ASIC_680ks | 75.8 µs | 84.0 µs | **63.5 µs** | 3.98 → 1.17 |
  | filter3D | 47.1 µs | 49.2 µs | **38.9 µs** | 2.48 → 1.05 |
  | roadNet-TX | 45.1 µs | 43.0 µs | **41.0 µs** | 1.48 → 1.03 |
  | NACA0015 | 59.4 µs | 63.5 µs | 59.4 µs | 1.30 → 1.01 |
  | auto | 88.1 µs | 63.5 µs | 61.4 µs | 1.27 → 1.02 |
  | amazon0601 | 34.8 µs | 32.8 µs | 32.8 µs（持平） | 1.19 → 1.02 |

  结果全部正确。amazon 没有收益，而且比现行 kernel 的最好配置（`GROUP=1`，30.7 µs）慢 7%：σ 太大会打乱 x 的
  局部性，**σ 要扫**（256 / 1024 / 4096）。
- **天数上的预期**（假设耗时和读取字节数成正比，下限按"填充位置的值被跳过"、上限按"填充位置的值也被读"）：

  | 变体 | 现在 | σ 排序后（10 矩阵平均） | 排序后仍然最低的三个矩阵 |
  |---|---:|---:|---|
  | sell_f32 | 0.630 | 0.80–0.96 | NACA 0.55–0.62、amazon 0.56–0.60、roadNet 0.62–0.73 |
  | sell_f16 | 0.621 | 0.82–0.91 | amazon 0.56–0.59、NACA 0.58–0.62、GL7d14 0.66–0.71 |
  | sell_i8i32 | 0.654 | 0.92–0.98 | amazon 0.56–0.57、NACA 0.59–0.62、GL7d14 0.60–0.63 |
  | sell_c32 | 0.516 | 0.63–0.83 | NACA 0.39–0.46、amazon 0.41–0.45、GL7d14 0.51–0.60 |

  平均值主要被 ASIC 和 filter3D 拉高（这两个的填充最多），**NACA、amazon、GL7d14 仍然在 0.55–0.7**，要靠 5.2–5.4。
- **怎么做**：
  1. 新增 `prepare_spmv_sell(values, cols, offsets, shape, slice_size, *, sigma=1024, c=None)`：从用户的 SELL 数组恢复每行的
     元素（有效位是 `cols >= 0`），按 σ 窗口排序后重新切 slice，产出 `vals / cols / slice_offsets / perm`。全部用 torch
     在设备上做（参考 `csr_to_sell` 的向量化写法），prepare 里可以同步。
  2. `flagsparse_spmv_sell(prepared=..., x=..., out=...)` 只做 x/out 检查和一次 kernel 启动。
  3. `tests/test_spmv_sell.py` 的天数分支改为计时 prepared 调用，CSV 增加 `layout` 和 `sigma` 两列。
  4. 正确性：σ 排序后 y 必须按原始行号写回；补"空行在 slice 末尾""行数不是 C 的倍数""只有一行"的测试。

### 5.2 让 64 宽的 warp 吃满

- **代码现状**：实数路径 `slice_size <= 32` 时 `BLOCK_R=64`、`num_warps=1`，slice 数 ≤ 32768 时 `GROUP=1`。SLICE 是 32，
  `r < SLICE` 的掩码让一半 lane 常年空着；`GROUP=2` 时 tile 是 2×64=128 个元素，同样一半被遮掉。
- **重编码后可以直接选 slice 高度**：
  - 方案 a：`C=64, GROUP=1, num_warps=1`。一个 warp 正好一个 slice、每个 lane 一行。代价是填充率比 C=32 略高
    （σ=1024 时 1.02–1.11，ASIC 1.39；σ=8192 时 1.00–1.02，ASIC 1.14）。
  - 方案 b：`C=32, GROUP=2, BLOCK_R=32, num_warps=1`。两个 slice 共用一个 warp；σ 排序后相邻 slice 宽度接近，
    `GROUP` 带来的"按较宽的 slice 循环"代价很小（这正是原来 `GROUP=2` 在高填充矩阵上吃亏的原因）。
- **旁证**：roadNet 有 43544 个 slice，超过 32768 走 `GROUP=2`，跑到 554 GB/s；NACA0015 有 32475 个 slice，正好低于
  阈值走 `GROUP=1`，只有 362 GB/s。roadNet 的 slice 更窄，这只是线索。
- 覆盖开关已加（2026-10-25）：`FLAGSPARSE_SELL_PREP_C / _GROUP / _WARPS / _SIGMA / _PAIRED / _REORDER`，prepare 时读取。
  64 宽 warp 的后端默认取方案 a（`C=64, GROUP=1, num_warps=1`）：5090 上复数用 `GROUP=2` 时 ASIC 从 73.8 µs 变成
  145 µs，同一个 program 里较窄的 slice 要等较宽的那个；方案 b 留给天数用 P6 对比。

### 5.3 复数（sell_c32）

- 现状：`_spmv_sell_complex_kernel`，tile 1×32、`num_warps=1`（半个 warp），实部、虚部、x 的实部、x 的虚部是 4 次分开的
  4 字节读取。
- 改法：在 5.1/5.2 的基础上，值和 x 都按 `(…, 2)` 成对读取（每次 8 字节），参考
  `mixed_spmx._csr_spmv_real_by_complex_cached_kernel` 的写法；累加用两个 fp32 通道。
- c32 是本组最难的一项：即使 σ 排序上限估算为 0.83，NACA/amazon 仍在 0.4 左右。成对读取能不能把效率从 ~300 GB/s
  提到 ~450 GB/s，要看第 0 步 P1/P3 的结果。

### 5.4 窄类型（f16 / i8）的打包读取（看第 0 步结果再做）

- 现象：f16、i8 SELL 的时间没有随字节数同比例下降。NACA 上 f16/f32 = 0.91（字节比 0.75），i8/f32 = 0.86（字节比
  0.625）。说明窄类型每次读取的字节太少（一个 warp 一次只读 64×2 B 或 64×1 B），受指令数或延迟限制。
- 改法：prepare 时按"每个 lane 连续存 K 个 slot"排布（`[slice][j // K][lane][K]`），kernel 用 `(C, K)` 的 tile 一次读
  K 个值：f16 取 K=2 或 4（4/8 字节）、i8 取 K=4 或 8。列号是 int32，也可以按 K 打包。slice 宽度要补到 K 的倍数，
  对平均行长 3 左右的矩阵（ASIC、roadNet）会多出不少填充，K 要按矩阵选。
- 只有当第 0 步显示"指令/延迟受限"而不是"带宽受限"时才值得做。

### 5.6 RCM 重排：在 prepare 里换 x 的局部性（2026-10-25 新增）

- **CUDA 上的发现**（5090，探针 P3/P7，不是天数数据）：把 `x[col]` 换成不随机的读取（P3），NACA0015 快 2.5 倍、auto
  快 3.4 倍、amazon 快 1.8 倍、ASIC 只快 1.07 倍——这几个矩阵的行号顺序很差（NACA 列号与行号之差的中位数 191584，
  auto 35468）。
- **做法**：prepare 时对方阵的对称化结构做逆 Cuthill-McKee（RCM），行、列一起重新编号，每行内部的求和顺序不变
  （结果和不重排逐位一致）；每次调用先把 x 按新顺序排到 plan 自带的缓冲区（多一次启动，算在计时里），y 由 `perm`
  直接写回原始行号。`prepare_spmv_sell(reorder="rcm" | "auto")` 已实现，`auto` 只在中位数缩小 4 倍以上且原来不小于
  4096 时启用（10 个矩阵里只选中 NACA、auto）。
- **CUDA 实测**（交付 benchmark 计时，含 x 重排）：f32 的 NACA 39.8 → 29.6 µs、auto 49.6 → 29.2 µs；f16、i8 类似；
  **复数没有收益**（NACA 0.81 倍、auto 0.93 倍）：复数每个元素要读 8 字节值，x 局部性占比小，多出的 x 重排又更贵。
- **天数上**：x 随机读在缓存更小的卡上通常更贵，NACA 和 auto 又正好是天数最差的两个大矩阵，所以这一项值得优先 A/B
  （第 9 节命令里的 `FLAGSPARSE_SELL_PREP_REORDER=auto`）。它需要 SciPy（库本身不依赖 SciPy，只有这个可选步骤用）。
- 同样的重排以后可以用到 COO、CSR：在 prepare 里把 COO 转成 SELL-C-σ 并重排，直接复用这个 kernel。

### 5.5 列号压缩（可选，收益不覆盖最差的矩阵）

- 每行存一个 int32 基址、每个元素存 uint16 偏移，f32 每个元素从 8 字节降到约 6 字节。按行跨度 < 65536 统计：
  GL7d14、cfd2、filter3D 100%，roadNet 98.6%，cage12 97.9%，wave 94.9%，ASIC 84.5%；**但 NACA 0.7%、auto 20.5%、
  amazon 22.7%**——正好是最差的三个矩阵。不重排列号（需要每次调用重排 x）就用不上，优先级放最后。

## 6. COO 计划

### 6.1 先整理 prepare / run 的分工（所有 COO 混合精度变体）

- 现在 `spmv_coo_mixed` 的天数分支在**运行时**懒构建 COO→CSR（挂在 `_ROCM_COO_CSR_CACHE` 这个 TensorPlanCache 上），
  每次调用还要查缓存、新分配 y、最后 `out.copy_(y)`。
- 改为在 `prepare_spmv_coo` 里一次构建内部格式（第 3 节的 A 或 B），`PreparedCoo` 持有它；运行时直接写进 `out`；
  q4 bench 的 COO prepared 路径传预分配的 `out`。天数上这几项在 GPU 侧受限，单这一步收益不大（只对 cage12、wave 这类
  小矩阵有意义），但它是后面所有改动的地基，也去掉了 TensorPlanCache 的那几条局限。

### 6.2 长尾矩阵（ASIC）走 nnz 均分，COO 的行号现成

- **数据**：三项实数混合精度在 ASIC 上都是最差的（0.28–0.32），耗时约 122 µs；同一个 nnz kernel 在天数上跑 f32c32 的
  ASIC 只要 81.8 µs，而 f32c32 读的字节更多（复数 x）。
- **怎么做**：`prepare_spmv_coo` 默认 `sort_by_row=True`，按 (行, 列) 排好序，行号数组就是 `_nnz_kernel` 的 `ROW`，
  只需额外一个空行列表。选择规则用第 3 节的结构条件。
- **（已核实，不是问题）**：曾担心 `_nnz_kernel` 非 MUSA 分支的浮点字面量 `0.` 会把 int32 累加提升成 float32。
  2026-10-25 在 CUDA 机上用一行和为 32257873（大于 2^24 的奇数）的 i8→i32 COO 实测，改动前后都逐位精确：Triton 的
  Python 标量是弱类型，会跟随张量类型。天数这次把两处字面量统一成 `0`，只是写法一致。
  另外 `NE`、`NNZ` 是 `tl.constexpr`，每个矩阵都会重新编译一次——不影响计时（编译发生在 warmup），但第一次调用很慢，
  可以改成运行时参数。
- **预期**：ASIC 从 0.28–0.32 提到约 0.5，三项均值约 +0.02：i8i32 0.771 → 约 0.79，f16 0.762 → 约 0.785，f16f32 0.730 → 约 0.75。

### 6.3 filter3D 上 COO f16→f16 的异常（先复测）

- COO f16→f16 在 filter3D 上 99 µs、0.48，而 COO f16→f32 和 CSR f16→f16 都是 49 µs，三者是同一个 kernel。
- 先复测；能复现就打印实际路由。如果修好到 f16→f32 的水平（约 0.80），f16 这一项均值再 +0.03，可以过 0.8。

### 6.4 转置 f32（coo_f32_trans）：长尾列走 nnz 均分

- **数据**：ASIC 转置 538 µs（非转置 163 µs），amazon 转置 312 µs（非转置 179 µs）；amazon 的列长最长 2751。
- **原因**：转置 plan 经 `_spmv_csr_transpose.make_plan()`，ASIC 的转置结构触发按长度排序的行顺序；amazon 的一批列里
  只要有一条长列，整批都要等它。天数扫 lanes×batch 无收益，和这个判断一致。
- **CUDA 实测**（5090，GPU 时间，在转置结构上跑现有 `_nnz_kernel`）：ASIC 29.7 → 7.8 µs（cuSPARSE 21.9），amazon
  64.4 → 27.1 µs（cuSPARSE 39.1）；roadNet、auto 持平或变慢，所以必须按结构条件启用。
- **怎么做**：COO 转置时 prepare 已经按列排好序（`col_trans`），它就是 Aᵀ 的行号；值用 `data_trans`。规则矩阵的转置
  （NACA 0.66、roadNet 0.62、auto 0.72）可以在 prepare 里直接构建 Aᵀ 的格式 A，和 SELL 共用 kernel。
- **预期**：如果天数上 nnz kernel 在长尾矩阵上能维持 ASIC 那样的效率（f32c32 81.8 µs 约合 350 GB/s），ASIC 0.12 → 0.4–0.8，
  amazon 0.44 → 0.6–1.0，这一项均值 0.776 → 0.82–0.90。

### 6.5 规则矩阵上的实数混合精度（NACA、auto、amazon、GL7d14 在 0.66–0.74）

- 需要的提升不大：这些矩阵要到 0.8 只需带宽提升 10–25%。
- 但**格式 A 本身不一定比 row-tile 快**：天数上 SELL f16 在 NACA 上 272 GB/s，COO f16 走 row-tile 也有 269–280 GB/s；
  i8 两者都是约 245–250 GB/s。也就是说，在窄类型上 SELL 和 row-tile 的效率差不多，只有 f32 时 SELL 明显更高。
- 所以这里的收益要看第 0 步的结论，以及 5.4 的打包读取能不能用到格式 A 上。方案是：COO 规则矩阵统一走格式 A，
  这样 SELL 做出的任何改进（5.2–5.4）自动对 COO 生效。

### 6.6 复数（coo_c32）：最难

- 天数改过 ASIC 之后报告是 0.522；其余 9 个矩阵在 0.43–0.66，大矩阵需要把带宽从约 250–330 GB/s 提到 450–650 GB/s，
  相当于 1.6–1.8 倍。目前天数上**没有任何一个复数 kernel 达到过这个效率**（SELL c32 在 NACA 上也只有 317 GB/s）。
- 路线：走格式 A + 5.3 的成对读取。按 σ 排序和成对读取的预期，NACA 大约能从 COO 现在的 291 µs 降到 250–280 µs，
  对应 0.48 → 0.5–0.55。
- **诚实的预期**：不出现新的发现的话，这一项均值约 0.6–0.65；要到 0.8 必须先在第 0 步找到天数复数访存的瓶颈。
  如果 P1 显示 Triton 的上限本身就不够，就在报告里写明。

## 7. 排期和验收

| 步骤 | 工作量 | 涉及变体 | 预期 | 验收 |
|---|---|---|---|---|
| 第 0 步：瓶颈探针 | 0.5–1 天 | 全部 | 决定 5.3/5.4/6.5/6.6 怎么做 | 5 个探针在 3 个矩阵上的数据 + profiler 指标 |
| 5.1 σ 排序 + `prepare_spmv_sell` | 1–2 天 | SELL 五项 | 均值 0.80–0.96（c32 0.63–0.83） | 精度全过；同轮交错 A/B；扫 σ |
| 5.2 吃满 warp | 0.5 天 | SELL 五项 | NACA/amazon/GL7d14 提升 | 方案 a/b 各自逐矩阵数据 |
| 6.1 + 6.2 + 6.3 | 1–2 天 | COO 实数混合精度三项 | 约 0.75–0.8，f16 有望过 0.8 | ASIC 走 nnz；逐矩阵不回退 |
| 6.4 转置长尾 | 1 天 | coo_f32_trans（csr 转置、c32 conj 顺带受益） | 0.82–0.90 | ASIC/amazon 逐矩阵；规则矩阵不回退 |
| 5.3 / 5.4 / 6.5 / 6.6 | 视第 0 步 | SELL c32、窄类型、COO c32 | 见各节 | — |

每一步都要：同一进程交替 A/B、每个矩阵至少 3 轮取中位数、报**逐矩阵**数字；精度用 `ILUVATAR.md` 命令里的 `-k`
重跑；改到 `spmv_csr.py` 的调度函数时在无 torch 环境下跑一遍 `tests/ci`（`test_spmv_csr_policy.py` 用桩环境执行这些函数）。

## 8. 风险和注意事项

- **比较口径**：重编码后我们读的数据和 cuSPARSE 不同（更少的填充）。这在"只计 kernel、prepare 不计时"的前提下是
  合理的，但报告里必须写明格式和 σ，不能让人以为是同一份数据上的比较。
- **x 的局部性**：σ 太大打乱 x 的访问顺序（CUDA 原型上 amazon 没有收益，比现行最好配置慢 7%）。σ 必须扫，必要时按矩阵选。
- **内存**：prepare 会复制一份矩阵（值、列号、perm）。超大矩阵上要注意显存占用。
- **用户原地改值**：重编码后的副本看不到用户对原数组的修改。参照 gather 现有做法：记录源张量的 `_version`，运行时发现
  变化就重新填充值（结构不变时只需一次 scatter）；结构变化必须重新 prepare，在文档里写明。
- **y 的散写**：σ 排序后按 `perm` 写回，同一个窗口内的行在 L2 里合并，影响不大；σ 越小越安全。
- **小矩阵和主机开销**：天数上每次调用约 22–30 µs 的固定开销会盖住小矩阵的 kernel 时间。prepared 运行路径要尽量薄
  （不做 `.item()`、不新分配输出）。

## 附：CUDA 原型（只验证格式和正确性）

prepare（全部 torch，设备上执行）：

```python
def prepare_sellcs(data, indices, indptr, n_rows, C=32, sigma=1024):
    L = (indptr[1:] - indptr[:-1]).long()
    win = torch.arange(n_rows, device=data.device) // sigma
    order = torch.argsort(win * (L.max() + 1) + (L.max() - L), stable=True)   # 窗口内按行长降序
    n_slices = (n_rows + C - 1) // C
    Ls = torch.zeros(n_slices * C, dtype=torch.long, device=data.device); Ls[:n_rows] = L[order]
    width = Ls.view(n_slices, C).amax(1)
    off = torch.zeros(n_slices + 1, dtype=torch.long, device=data.device); off[1:] = torch.cumsum(width * C, 0)
    vals = torch.zeros(int(off[-1]), dtype=data.dtype, device=data.device)
    cols = torch.zeros(int(off[-1]), dtype=torch.int32, device=data.device)   # 填充：列 0、值 0
    inv = torch.empty_like(order); inv[order] = torch.arange(n_rows, device=data.device)
    rows = torch.repeat_interleave(torch.arange(n_rows, device=data.device), L, output_size=data.numel())
    p = inv[rows]
    pos = off[p // C] + (torch.arange(data.numel(), device=data.device) - indptr.long()[rows]) * C + p % C
    vals[pos] = data; cols[pos] = indices.to(torch.int32)
    perm = torch.full((n_slices * C,), -1, dtype=torch.int32, device=data.device); perm[:n_rows] = order.to(torch.int32)
    return vals, cols, off.to(torch.int32), perm, n_slices
```

kernel（每个 program 处理 `GROUP` 个 slice，y 按 `perm` 写回原始行号）：

```python
@triton.jit
def sellcs_kernel(y_ptr, v_ptr, c_ptr, off_ptr, perm_ptr, x_ptr, n_slices,
                  C: tl.constexpr, GROUP: tl.constexpr, ACC: tl.constexpr):
    s = tl.program_id(0) * GROUP + tl.arange(0, GROUP)[:, None]
    r = tl.arange(0, C)[None, :]
    smask = s < n_slices
    start = tl.load(off_ptr + s, mask=smask, other=0)
    width = (tl.load(off_ptr + s + 1, mask=smask, other=0) - start) // C
    acc = tl.zeros((GROUP, C), dtype=ACC)
    for j in range(0, tl.max(tl.max(width, 0), 0)):
        m = smask & (j < width)
        pos = start + j * C + r
        col = tl.load(c_ptr + pos, mask=m, other=0)
        v = tl.load(v_ptr + pos, mask=m, other=0).to(ACC)
        acc += v * tl.load(x_ptr + col, mask=m, other=0).to(ACC)
    row = tl.load(perm_ptr + s * C + r, mask=smask, other=-1)
    tl.store(y_ptr + row, acc.to(y_ptr.dtype.element_ty), mask=row >= 0)
```

在 5090 上对 ASIC、NACA、auto、amazon、filter3D、roadNet 六个矩阵（f32）验证结果正确；性能见 5.1。天数上应按 5.2 把
`C` 改成 64，或者保持 32 并设 `GROUP=2`。

## 9. 已实现，交给天数实测（2026-10-25）

### 9.1 `prepare_spmv_sell`（`src/flagsparse/sparse_operations/spmv_sell.py`）

```python
plan = fs.prepare_spmv_sell(values, col_indices, slice_offsets, (m, n), slice_size=32,
                            out_dtype=None, route="auto", sigma=None, c=None, group=None,
                            num_warps=None, paired=None, reorder=None)
y = fs.flagsparse_spmv_sell(prepared=plan, x=x, out=y)   # 或 fs.spmv_sell_prepared(plan, x, out=y)
```

- **route**：`auto` 在 CUDA 上走 `sigma`；天数仅实数走 `sigma`，复数回到原生 `call` 路径；其他后端走 `call`
  （原样交回各后端已有的 SELL 路径）；Ascend 不能用 `sigma`。
- **prepare 做的事**（会同步，不计时）：从用户的 SELL 数组恢复每行元素，在 σ 行窗口内按行长降序排序后重新切 slice；
  `sigma="auto"` 取"从 c 开始翻倍、填充率 ≤ 1.10 的最小窗口"（上限 4096），10 个矩阵的选择：ASIC 4096、filter3D 512、
  amazon/roadNet 256、NACA/auto/GL7d14/wave 128、cage12/cfd2 不排序。可选 RCM 重排（5.6）。
- **kernel**：每个 lane 用自己的行长做掩码——填充位置永远不读，值的读取不等列号，`x` 里的 inf/NaN 也不会透过填充
  的 0 混进结果；结果按 `perm` 写回原始行号；CUDA 上直接发射。
- **默认参数**（显式参数 > 环境变量 > 默认）：

  | 后端 | 实数 (c, group, warps) | 复数 (c, group, warps) | σ | 成对读取 | 重排 |
  |---|---|---|---|---|---|
  | CUDA | 32, 4, 4 | 32, 1, 1 | auto | 关 | none |
  | 天数（64 宽 warp） | 32, 1, 1 | 64, 1, 1（显式 sigma） | 实数 4096 / 复数 auto | 关 | none |
  | MACA / DCU（64 宽 warp） | 32, 1, 1 | 64, 1, 1 | auto | 关 | none |

  （MACA / DCU 的 `route="auto"` 本来就走 `call`，上表只在显式 `route="sigma"` 时生效。）

  环境变量：`FLAGSPARSE_SELL_PREP_SIGMA`（行数或 `auto`）、`_C`、`_GROUP`、`_WARPS`、`_PAIRED`（`1`/`0`）、
  `_REORDER`（`none`/`rcm`/`auto`）。
- **语义**：plan 复制了一份值。用户原地改值（`_version` 变化）下次运行自动刷新；改了列号或 slice 偏移会自动重建；
  inference 张量没有版本计数，改了看不到，需重新 prepare。开了重排的 plan 有自己的 x 缓冲区，不要在多个 stream 上
  并发使用同一个 plan。
- **正确性**：测试覆盖 f32/f64/f16/bf16→f32/i8→i32/i8→f32/c64/c128、空行、长行、行数不是 c 的倍数、strided out、x 含
  inf、原地改值/改结构、RCM、环境变量和错误用法。2026-10-25 本机完整测试结果为 153 passed、16 failed：
  `test_auto_sigma_sorts_only_padded_structures` 已显式指定 `sigma="auto"`，单测复测通过；其余 16 项是 BI-V double/complex128
  strided `copy_` canary 失败，与 SpMV 数值结果无关，需作为已知问题跟进，不能记作全通过。合入 main 时（CUDA 机）这 16 项
  在天数上改为跳过（天数交付本来就用 `-k` 排除 fp64），原因写在测试里。

### 9.2 交付 benchmark 的开关

`tests/test_spmv_sell.py` 读 `FLAGSPARSE_SELL_PREPARED`：不设置时和以前完全一样；设成 `1`/`auto`/`sigma`/`call` 时每个
矩阵先 prepare（不计时），再计时 prepared 调用，CSV 记 `timed=prepared`，并增加 `layout`、`sell_sigma`、`sell_c`、
`sell_group`、`sell_warps`、`sell_paired`、`sell_reorder`、`prepared_values` 列。runner 会把环境变量传给子进程，
所以 `benchmark_biv.py` 直接可用。

### 9.3 CUDA 参考数据（5090，交付 benchmark 计时；不代表天数）

| 矩阵 | 变体 | 原路径 | prepared（默认） | prepared + RCM | cuSPARSE |
|---|---|---:|---:|---:|---:|
| ASIC_680ks | f32 | 32.6 µs | 23.1 µs | — | 60.2 µs |
| ASIC_680ks | c32 | 60.1 µs | 28.5 µs | — | 87.7 µs |
| filter3D | f32 | 23.7 µs | 18.0 µs | — | 38.7 µs |
| filter3D | c32 | 36.6 µs | 21.7 µs | — | 53.4 µs |
| NACA0015 | f32 | 39.8 µs | 41.2 µs | 29.6 µs | 47.7 µs |
| auto | f32 | 49.6 µs | 49.5 µs | 29.2 µs | 86.0 µs |
| cage12 | f32 | 27.6 µs | 17.5 µs | —（auto 不重排） | 6.7 µs |

cage12 这类小矩阵在 CUDA 上受主机开销限制：原路径每次调用约 27 µs，prepared 路径约 15–17 µs。

### 9.4 天数上怎么跑（历史探索流程）

1. **探针**（约 20–40 分钟）：

   ```bash
   setsid timeout -s KILL 7200 env FLAGSPARSE_BACKEND=iluvatar \
     python3 -u tools/biv_spmv_probe.py tests/data \
       --matrices NACA0015,auto,ASIC_680ks,amazon0601,filter3D \
       --dtypes float32,float16,int8,complex64 \
       --csv results_biv_probe_<日期>.csv \
     > results_biv_probe_<日期>.log 2>&1 < /dev/null &
   ```

   每个矩阵、dtype 的结尾都有 `->` 开头的提示行，对应第 4 节的决策表。
2. **SELL 交付 A/B**（三轮，每轮一个新结果目录；只跑 `spmv_sell`）：

   ```bash
   # a) 现状
   python3 benchmark_biv.py --ops spmv_sell --results-dir results_biv_sell_call_<日期>
   # b) prepared 默认（按当前 dtype/backend 默认值选择）
   env FLAGSPARSE_SELL_PREPARED=1 python3 benchmark_biv.py --ops spmv_sell --results-dir results_biv_sell_prep_<日期>
   # c) b) 再加 RCM 自动重排
   env FLAGSPARSE_SELL_PREPARED=1 FLAGSPARSE_SELL_PREP_REORDER=auto \
     python3 benchmark_biv.py --ops spmv_sell --results-dir results_biv_sell_rcm_<日期>
   ```

   每轮之后 `python3 tools/delivery_table.py <结果目录>`。如果探针 P6 里有比默认更好的 (c, group, warps) 或 σ，用对应的
   `FLAGSPARSE_SELL_PREP_C/_GROUP/_WARPS/_SIGMA` 再跑一轮 d)。
3. **怎么判断**：
   - b) 每个矩阵都不慢于 a)：再考虑把天数 benchmark 的默认切到 prepared；复数若 sigma 有回退，应保留 call 路径。
   - c) 在 NACA/auto 上比 b) 快 15% 以上且其余矩阵不回退：天数默认打开 `reorder="auto"`。
   - 复数单独看：CUDA 上 RCM 对复数没有收益，天数如果也一样，复数保持不重排。

### 9.5 天数 BI-V150 实测结论（2026-10-25）

对照目录为 `results_biv_sell_call_20261025/`、`results_biv_sell_prep_20261025/`、
`results_biv_sell_rcm_20261025/`、`results_biv_sell_c32_20261025/`、
`results_biv_sell_c32g2_20261025/`、`results_biv_sell_c32s4096_20261025/`；最终配置完整复测在
`results_biv_sell_auto_final_20261025/spmv_sell/performance.csv`（50 行，5 变体 × 10 矩阵，全部 `PASS`）。
加速比按每矩阵 `speedup_vs_h800_scaled` 求 10 矩阵算术平均；基线是 H800 cuSPARSE 折算到 BI-V150 的时间，并非 BI-V150
同机厂商实测基线。

| 变体 | 最终选择 | 相对 H800 折算基线平均加速比 |
|---|---|---:|
| `spmv_sell_f32_int_non` | prepared sigma，C=32/GROUP=1/WARPS=1/σ=4096 | 0.694x |
| `spmv_sell_f16_int_non` | 同上 | 0.699x |
| `spmv_sell_f16f32_int_non` | 同上 | 0.695x |
| `spmv_sell_i8i32_int_non` | 同上 | 0.700x |
| `spmv_sell_c32_int_non` | `route=auto` 回原生 call | 0.518x |

RCM 和 `GROUP=2` 未带来整体收益；C=32、σ=4096 的实数配置胜出。复数 sigma prepared 在 NACA、amazon、auto 等矩阵明显回退，
所以复数 `auto` 路由恢复原生 call。实数 SELL 相比先前天数路径有所改善，但五项平均加速比仍未达到 0.8x。

Prepared 测试集复测：定向测试 `test_auto_sigma_sorts_only_padded_structures` 通过；完整集为 153 passed、16 failed，16 项均为
BI-V double/complex128 strided `copy_` canary 失败（stride 输出的间隔元素被改写），不是普通 SpMV 结果精度失败；该问题仍需单独跟进。

### 9.6 交给天数 Codex 的 prompt

````text
你在天数智芯 BI-V150 实机上为 FlagSparse 的 SELL SpMV 收集优化数据。仓库在当前目录，main 分支。主要目的是收集数据，不是修代码。

1. git pull --ff-only origin main，确认 tools/biv_spmv_probe.py 存在、python3 -c "import flagsparse as fs; print(fs.prepare_spmv_sell)" 能导入，
   且 python3 -c "import flagsparse; print(flagsparse.__file__)" 指向当前目录的 src/。
2. 完整读一遍 docs/debug/ILUVATAR_SELL_COO_PLAN.md，重点是第 4 节和第 9 节。
3. 按第 9.4 节第 1 步跑探针，跑完把 results_biv_probe_<日期>.log 里每个矩阵/dtype 的表格和 "->" 提示行原样带回。
4. 按第 9.4 节第 2 步跑 a)/b)/c) 三轮（每轮新的 --results-dir），每轮跑完执行 python3 tools/delivery_table.py <结果目录>。
   如果第 3 步的 P6 显示别的 (c, group, warps) 或 sigma 明显更好（快 10% 以上），用对应的 FLAGSPARSE_SELL_PREP_* 环境变量再跑一轮 d)。
5. 不要改 src/ 或 tests/ 下的代码；确实需要改时按 docs/debug/ILUVATAR.md 末尾"实机改动记录"的格式记录。
6. 遇到报错给出完整原文，不要猜根因。

汇报：
1. 探针日志里 5 个矩阵 × 4 个 dtype 的表格和提示行。
2. a)/b)/c)（以及 d)）四轮的 delivery_table.py 输出，以及 spmv_sell 各变体逐矩阵的加速比（各结果目录下 spmv_sell/performance.csv
   的 matrix、variant、triton_ms、sell_sigma、sell_c、sell_group、sell_warps、sell_reorder、status 列）。
3. 环境指纹：torch、triton 版本，设备名，git log -1 --oneline。
````

### 9.7 COO 建议复核与 mixed NNZ A/B（2026-10-25）

计划里 COO→CSR / prepare 缓存的建议在工程上可取，但本身主要是把排序、分桶和索引构建移出重复调用，不应预期显著降低 kernel 时间。短行长尾矩阵改用 NNZ 均分值得实测；结果支持只对结构符合条件的矩阵启用，不能普遍替代 row-tile。复数 COO 的 NNZ 分段扫描仍只是候选方向，本轮没有实现或验证，不能把实数×复数的既有 `_nnz_kernel` 结果直接外推到复数×复数。

在 BI-V150 对 10 个矩阵、三种 mixed COO（`f16f32`、`f16`、`i8i32`）做 plain/NNZ 同进程 A/B，计时用 `triton.testing.do_bench(warmup=30, rep=100)`；每次切换路由重建缓存，准备的行号计划不在重复调用中生成。A/B 数据记录于 `results_coo_mixed_nnz_ab_20261025.csv`。仅 ASIC_680ks 满足平均行长 ≤4 且最长行 >128：

| 算子 | plain | NNZ | NNZ 相对 plain | 10 矩阵平均加速比估算* |
|---|---:|---:|---:|---:|
| `spmv_coo_f16f32_int_non` | 149.78 µs | 115.01 µs | 1.302x | 0.739x（原 0.730x） |
| `spmv_coo_f16_int_non` | 135.05 µs | 106.27 µs | 1.271x | 0.770x（原 0.762x） |
| `spmv_coo_i8i32_int_non` | 143.76 µs | 111.54 µs | 1.289x | 0.779x（原 0.771x） |

*估算只把本轮 ASIC 的配对加速比应用于 10-24 全量测试的 ASIC 项，其余 9 项沿用原值；不是新一轮统一基准的全量平均值。每个矩阵 plain/NNZ 输出均一致；ASIC half 输出的差异最大 `3.81e-6`，i8 输出逐元素完全相等。half 的既有非有限输出在两条路径位置一致，属于输入矩阵/向量产生的溢出，不是 NNZ 路由新增。

根据以上结构性 A/B，`FLAGSPARSE_ILUVATAR_COO_MIXED_NNZ` 默认 `auto`：仅 Iluvatar、平均行长 ≤4 且最长行 >128 时建立 NNZ 行计划并发射 `_nnz_kernel`；设为 `0` 可回退 row-tile。ASIC 的预处理计划由缓存保留，调用期不重复生成。所有其余矩阵的 A/B 比值在约 `0.99–1.01x`，没有证据支持在那里切换。

`spmv_coo_c32_int_non` 另有此前自然行序/row-tile A/B：基于 H800 cuSPARSE 折算基线，10 矩阵算术平均由原始 `0.472x` 提至 `0.522x`（修复 ASIC 行序后 `0.526x`）；ASIC 单项从 `0.061x` 提至约 `0.37x`，但整体仍远低于 `0.8x`。见 `results_coo_c32_rowtile_20261024.csv` 和 `results_coo_c32_adaptive_fixed_20261024.csv`。本轮未复测这项。此前报告里 `spmv_coo_f32_int_trans` 的平均值已更新到约 `0.897x`，因此本轮“四项”按 c32 与上述三项 mixed 计算。

当前结论：短行长尾 NNZ 路由是三项 mixed 的有效 ASIC 局部优化，但不足以令任何一项整体达到 `0.8x`；继续追这三项应转向 CSR row-tile 参数/访存优化（COO 复用该 kernel）。c32 的下一条结构性实验是复数×复数 NNZ 分段 kernel，先做 ASIC、roadNet、auto 的正确性和 A/B，再决定是否推广；现有实数 NNZ 结果不构成它的性能证据。

### 9.8 四项低分 COO 的 tile 扫描与全量复测（2026-10-25）

在 BI-V150 上进一步实测了 `spmv_coo_c32_int_non` 与三项 mixed COO。三项 mixed 扫描了 ASIC、auto、amazon0601、NACA0015 的 `ROWS={8,16,32}`、`BLOCK={4,8,16,32}`；另扫 `num_warps={1,2,4}`、`num_stages={1,2,3}`。复数 c32 在 ASIC 上扫 NNZ kernel 的 block/warps。逐配置原始数据分别见 `results_coo_mixed_tile_sweep_20261025.csv`、`results_coo_mixed_stage_sweep_20261025.csv`、`results_coo_c32_nnz_tile_sweep_20261025.csv`。

扫描结论：mixed kernel 的 `num_warps=1` 稳定优于更多 warps，`num_stages=1/2/3` 没有稳定的大幅收益；不同矩阵的最佳 ROWS/BLOCK 不一致，未找到能稳健替换默认配置的统一 tile，因此不增加按矩阵名特判。复数 NNZ ASIC 扫描中 `BLOCK=128, warps=1` 最佳或接近最佳，与当前默认相同，也不改参数。COO tile 扫描结论仅针对实际扫描的四个矩阵，不代表其它六个矩阵都穷尽搜索。

之后对四项、10 个矩阵做 prepared 路径全量复测，`results_coo_four_low_final_20261025.csv` 共 40 行、全部 `PASS`。逐矩阵加速比使用本轮 `ours_ms` 除法的倒数，即 `results_biv_65_20261024_final/spmv_coo/performance.csv` 中对应 `h800_scaled_ms / ours_ms`，再对 10 个矩阵作算术平均：

| 算子 | 10 矩阵平均加速比 | 当前结论 |
|---|---:|---|
| `spmv_coo_c32_int_non` | 0.542x | 未达 0.8x |
| `spmv_coo_f16f32_int_non` | 0.708x | 未达 0.8x |
| `spmv_coo_f16_int_non` | 0.831x | 达到 0.8x |
| `spmv_coo_i8i32_int_non` | 0.805x | 达到 0.8x |

该结果是本轮 q4 `run_variants` 的 prepared 计时与已有 H800 折算基线组合，不是同一轮完整 delivery runner 生成的认证加速比；本机 Iluvatar 没有可用的 vendor/PyTorch 基线，故本轮 `PASS` 指精度校验通过。mixed NNZ 路由、c32 NNZ 路由在本次复测中均为 `auto`。因此按“算子 10 矩阵平均是否达到 0.8x”衡量，当前仍需继续优化的是 c32 和 f16f32 两项；f16、i8i32 不应再列为低于 0.8 的算子。

定向路由测试 `tests/ci/test_spmv_coo_iluvatar_route.py` 首次运行发现原测试仍断言 ASIC 长尾 c32 走 row-tile，与已启用的 `FLAGSPARSE_ILUVATAR_COO_C32_NNZ=auto` 路由不符；测试现已更新为检查 NNZ 计划及其行号/空行计划。复数×复数 NNZ 路径目前有 ASIC 单矩阵 A/B 和 tile 扫描，不等于全矩阵推广已验证；推广前仍应在其余矩阵做准确性与同轮 A/B。

### 9.9 f16f32 cached 读取与行长排序原型（2026-10-25）

为继续找 f16f32 的访存收益，在 BI-V150 直接对 `_csr_spmv_mixed_kernel`（默认 cache policy）和当前生产使用的 `_csr_spmv_mixed_cached_kernel`（values/columns `.cg`、x `.ca`）做 10 矩阵 A/B。基准通过 `triton.testing.do_bench(warmup=10, rep=40)` 测 kernel 本身，values 缩放后转 half，两个输出逐元素相同。数据见 `results_coo_f16f32_cache_ab_20261025.csv`。cached 路径在 10 个矩阵上均更快，`plain_us / cached_us` 算术平均为 `1.067x`；GL7d14 提升约 `1.238x`、amazon0601 `1.097x`、auto `1.063x`、NACA0015 `1.040x`。生产路径已经是较快的 cached 版本，故不改缓存修饰符。

另做了按 CSR 行长排序的 row-bucket kernel 原型：prepare 生成长度排序的 row id，kernel 按重排行号取 indptr 并写回原行。十个矩阵精度差均为 0，但速度全部变慢，数据见 `results_coo_f16f32_rowbucket_ab_20261025.csv`；如 `base_over_bucket < 1` 表示排序分桶更慢。推测额外 row-id gather 和跨行的非连续 CSR 访问抵消了同一 CTA 行长更接近的好处，因此不接入生产代码。该实验说明简单的全局行长排序不是可行方案；若继续尝试，应使用 prepare 阶段构建真正连续的桶内 CSR 数据布局，并把重排和 scatter 成本留在 prepare/输出路径评估中。

本轮没有发现 c32 的新 tile/caching 收益：已有 ASIC NNZ A/B 只带来局部改善，且 c32 ASIC NNZ `BLOCK=128, warps=1` 已是扫描中的最优附近；十矩阵平均仍是 `0.542x`。目前不应把复数 NNZ kernel 扩展到其余矩阵，除非另有针对性 A/B 证据。f16f32 当前 `0.708x`，row-bucket 原型失败、cached 读取已在用；下一个结构性方向是 prepare 时构造连续的长度分桶 CSR（并和现有 NNZ 路由、row-tile 做同轮全量 A/B），而不是继续重复扫 tile。

### 9.10 复数 NNZ 强制路由全矩阵 A/B（2026-10-25）

为判断复数×复数 NNZ 扫描能否推广，在 BI-V150 对 10 个矩阵直接比较当前结构自适应路由（`FLAGSPARSE_ILUVATAR_COO_C32_NNZ=0`）和强制 NNZ（`=1`），使用 `triton.testing.do_bench(warmup=10, rep=40)`，计 kernel 时间且每个矩阵对照输出最大相对误差不超过 `1.8e-7`。逐矩阵数据见 `results_coo_c32_nnz_force_ab_20261025.csv`。ASIC 由 `156.40` 降至 `124.94 µs`（`1.252x`）；另外九个矩阵全部变慢，`baseline_us / forced_nnz_us` 在 `0.707–0.936x`，十矩阵比值算术平均 `0.867x`。因此将 c32 NNZ 保持在当前 ASIC 长尾条件，不推广到其它矩阵。此 A/B 是 kernel 级同进程诊断数据，不用于替代 9.8 节完整 prepared 复测的平均加速比。

### 9.11 f16f32 连续重排原型复核（2026-10-25）

尝试在 prepare 阶段按行长排序，并将 CSR 的 values/columns 真正按新行序连续重排，再用混合精度 row-tile kernel 执行。首轮结果见 `results_coo_f16f32_repacked_bucket_ab_20261025.csv`，但该原型的输出与原 kernel 不一致（多个矩阵最大误差显著，不能通过正确性校验），且记录的耗时也普遍更慢。由于正确性失败，这批性能数字不作为有效 A/B 结论；原型不接入生产代码，也不能据此断言正确实现的连续重排一定变慢。后续若重做，须先逐行验证重排后的 CSR 行指针、列和值与原矩阵一致，再计时，并把 prepare 成本与 prepared kernel 时间分别记录。

本轮可采信的优化结论仍限于：f16f32 的 cached 读取已在生产路径中，10 矩阵 kernel A/B 平均约 `1.067x`；mixed NNZ 路由在长尾 ASIC 上有效，其他矩阵没有稳定收益；c32 强制 NNZ 仅 ASIC 获益，不能全局推广。四算子全量 prepared 复测中，c32 与 f16f32 的平均加速比分别为 `0.542x` 和 `0.708x`（相对 H800 折算基线；非本机 vendor 基线）。截至本节所述实验，尚无新的生产改动通过正确性与性能双重验证。

### 9.12 f16f32 ASIC NNZ tile 调优（2026-10-25）

针对 mixed NNZ 已命中的 ASIC 长尾矩阵，在 BI-V150 同进程扫描 `_nnz_kernel` 的 `BLOCK={64,128,256,512,1024}`、`num_warps={1,2,4}`。输入为 f16 values、f32 x / accumulator；每个候选相对当前 cached row-tile 输出最大绝对误差均为 `2.38e-7`。`BLOCK=128, warps=2` 最快，为 `89.21 µs`；原 NNZ 配置 `BLOCK=256, warps=2` 为 `100.48 µs`，候选快 `1.126x`。当前 cached row-tile 同轮为 `123.67 µs`，所以候选 NNZ 比 row-tile 快 `1.386x`。参数表见 `results_coo_f16f32_nnz_tile_ab_20261025.csv`。据此只对 Iluvatar COO f16→f32 路径采用 `128/2`，不改其它混合精度 dtype，也不改变路由条件。

随后两轮对 c32 与 f16f32 各 10 个矩阵做 q4 prepared 复测，结果分别在 `results_coo_c32_f16f32_retest_20261025.csv` 和 `results_coo_c32_f16f32_retest2_20261025.csv`，20/20 均 `PASS`。由于两轮间计时波动明显，且本机没有 vendor 基线，不能把与旧 H800 折算基线计算出的平均比值变化当成这次 tile 调整带来的确定收益；这轮调优结论仅为 ASIC 上 NNZ kernel 本身的同轮参数 A/B。c32 仍保持原 ASIC 条件路由，未发现可推广到其它矩阵的安全 kernel 变体。

### 9.13 SELL c32 prepared 与 legacy CSR 交错复测（2026-10-25）

为复核此前 c32 SELL→CSR A/B 的稳定性，在 BI-V150 对 10 个矩阵重新构建 prepared 计划：SELL 固定为 `route="sigma", sigma=4096, C=32, GROUP=1, num_warps=1, paired=false`；CSR 固定为 `alg="legacy_segbin"`。两条路径均在计时前完成 prepare 和首次调用（排除准备与首轮编译），之后每个矩阵各测 5 轮，轮间交替 SELL/CSR 的先后顺序；单次计时使用 `triton.testing.do_bench(warmup=20, rep=35)`。每轮分别记录 kernel 时间，并检查两路输出相互误差及 SELL 对 CPU complex128 参考的误差。逐轮数据见 `results_biv_sell_c32_csr_route_alternating_20261025.csv`。

| 矩阵 | 中位数 SELL/CSR | 较快路径 |
|---|---:|---|
| ASIC_680ks | 1.486x | CSR |
| GL7d14 | 0.926x | SELL |
| NACA0015 | 1.110x | CSR |
| amazon0601 | 0.995x | 基本持平（SELL 略快） |
| auto | 0.961x | SELL |
| cage12 | 0.903x | SELL |
| cfd2 | 0.944x | SELL |
| filter3D | 1.627x | CSR |
| roadNet-TX | 0.972x | SELL |
| wave | 1.075x | CSR |

以每矩阵 5 轮中位数计算，`SELL 时间 / CSR 时间` 的 10 矩阵几何平均为 `1.078x`；CSR 在 4/10 个矩阵更快，SELL 在 5/10 个矩阵更快，amazon0601 近似持平。全部 50 组输出检查通过：相对 CPU 参考误差约 `8.3e-8–2.6e-7`，SELL 与 CSR 输出间误差约 `6.6e-8–2.9e-7`。因此本轮不支持将 c32 全局切换到 CSR；此前 A/B 曾报告的约 `1.28x` 几何平均优势未在本次交错复测中复现，不能据其作为路由切换依据。该比值仅用于本机两种 prepared kernel 的相对比较，不是相对厂商基线的加速比，也不含 prepare 成本。

本轮只新增复测 CSV 与本节记录，没有修改生产 kernel 或默认路由。后续继续从 SELL kernel / 存储布局寻找适用于低于 `0.8x` 变体的优化；CSR 只保留为矩阵结构条件路由的候选，须先有可复现的结构指标收益，并在完整 10 矩阵与精度复测中验证。

合入 main 时的两点补充（2026-10-25，CUDA 机）：

- 本节引用的 `results_biv_sell_c32_csr_route_alternating_20261025.csv` 没有随这次更新一起给过来，表里的数字无法逐轮核对；
  表内 10 个比值的几何平均 `1.078x` 已核算无误（4 个矩阵 CSR 更快、amazon0601 近似持平、其余 5 个 SELL 更快）。
- **SELL 一侧用的不是交付路径**：本节固定 `route="sigma", σ=4096, C=32`，它在 10-25 全量里是 `0.497`（见
  `results_biv_sell_c32s4096_20261025`），而 c32 交付默认走 `route=auto`→`call`（原生复数 kernel）是 `0.518`，比它快约 4%。
  所以这轮是"CSR 对一个比交付更慢的 SELL 配置"。结论（不切 CSR）仍然成立，但要作为切换依据，应把 SELL 一侧换成
  `route="call"` 再测一轮。

### 9.14 P8 探针：gather 聚集度与 y 散写（2026-10-25，待天数实测）

之前 SELL 的 tile、σ、C、GROUP、warps 和 cache modifier 扫描没有稳定赢家；而 RCM 在天数上出现 kernel 回退，尚不清楚是 gather 局部性没有收益，还是 SELL 重排实现本身有额外代价。因此先用合成矩阵分离列访问局部性和 y 写入散布，不据 CUDA 数据预判天数结果。

`tools/biv_spmv_probe.py` 的 P8 不需要 `.mtx`：

- **P8 span**：每行固定相同行数，SELL 填充率为 `1.00`，σ 排序不改变行序；各组复用同一组 values，布局、kernel 和读写字节数相同，只改变列号。列索引限制在对角线附近 `--spans` 个 x 元素内，从局部带宽到全向量。
- **P8 y_scatter**：固定同一份 SELL 计划，分别使用自然顺序与随机 `perm`，隔离输出散写成本。

```bash
setsid timeout -s KILL 7200 env FLAGSPARSE_BACKEND=iluvatar \
  python3 -u tools/biv_spmv_probe.py --matrices none \
    --dtypes float32,float16,int8,complex64 \
    --synthetic-rows 8388608 --rounds 5 \
    --csv results_biv_span_20261025.csv \
  > results_biv_span_20261025.log 2>&1 < /dev/null &
```

`--matrices none` 只执行 P8（跳过 P1 与真实矩阵探针）。本机 BI-V150 L2 为 16 MiB；因此建议用 `--synthetic-rows 8388608`（f32 x 32 MiB、complex64 x 64 MiB），而不是 2,097,152（f32 x 只有 8 MiB）。大规模前先以较小行数 smoke test，确认内存余量及 prepared kernel 正确执行。

5090 上的参考测试曾观察到 f32 聚集/全散列约 `1.72x`、complex64 约 `2.30x`，但这些是 CUDA 参考值，不是 BI-V 实测结论。BI-V 的结果按脚本输出判读：聚集收益 `>=1.15x` 支持继续查 RCM / cache blocking 的实现；`<=1.05x` 则优先转向减少字节数；中间值需扩大轮数和工作集复测。随机 y scatter 若明显变慢，再单独评估连续写临时输出及最终置换的总成本。

#### 9.14.1 BI-V150 P8 实测结果（2026-10-25）

BI-V150 的 L2 为 `16 MiB`，因此使用 `--synthetic-rows 8388608`，使 f32 x 为 `32 MiB`、f16 x 为 `16 MiB`、int8 x 为 `8 MiB`、complex64 x 为 `64 MiB`。首轮脚本曾因 y-scatter 探针复用了同一个 plan 而污染后续 span 组；修正为两个独立 prepared plan 后，以下结果才作为有效数据。

| dtype | band=32 | band=full | 聚集/全散列 | y scatter shuffled/identity |
|---|---:|---:|---:|---:|
| f32 | `892.2 µs` | `10824.2 µs` | `12.13x` | `5.38x` |
| f16 | `765.5 µs` | `4850.6 µs` | `6.34x` | `10.44x` |
| i8→i32 | `661.1 µs` | `1961.9 µs` | `2.97x` | `22.30x` |
| complex64 | `1441.3 µs` | `13244.6 µs` | `9.19x` | `4.71x` |

中间带宽组的共同 knee 在 `band=256`（complex64 为 `band=32`）；因此这张卡上列访问聚集明确有效，不能放弃 RCM / cache blocking。更重要的是，随机 y 写比自然连续写慢 `4.7–22.3x`，验证了此前 RCM 回退主要来自 `perm` 直接散写的判断。下一步优先做“kernel 连续写临时 y，再单独做一次置换回原行序”的 A/B，并把置换成本计入 prepared kernel 总时间；不能只比较临时 y 的 kernel 时间。

CSV：`results_biv_span_20261025_retest.csv`（f32/f16/i8）与 `results_biv_span_complex64_20261025.csv`（complex64）。

#### 9.14.2 连续 y 输出 A/B（2026-10-25）

依据 P8，新增实验开关 `FLAGSPARSE_SELL_CONTIGUOUS_OUTPUT=1`：启用 RCM 的 prepared SELL 时，kernel 先按重排后的行号连续写到 prepared 临时缓冲，再用一次 `index_copy_` 置回原行序；该置换成本包含在计时内。默认值为 `0`，因此不改变交付路径。

| 矩阵 | 开关关闭 | 开关开启 | 开启/关闭 | 输出检查 |
|---|---:|---:|---:|---|
| ASIC_680ks f32 | `202.5 µs` | `214.1 µs` | `0.946x` | PASS |
| NACA0015 f32 | `474.7 µs` | `274.8 µs` | `1.727x` | PASS |
| amazon0601 f32 | `132.0 µs` | `152.0 µs` | `0.869x` | PASS |
| auto f32 | `255.4 µs` | `262.4 µs` | `0.973x` | PASS |

这里的比值是“关闭耗时 / 开启耗时”的加速比；NACA0015 开启后约 `1.73x`，其余三个矩阵没有收益。与当前交付 SELL API 比较时，NACA0015 的 RCM 路径仍为 `0.78x`（`274.8/214.0`），不能宣称算子达标。因此暂不默认开启，只保留给后续按结构条件路由的候选；下一步需要在真实低于 `0.8x` 的 SELL 算子上验证该结构条件是否能改善相对厂商基线，而不是只改善 RCM 子路径。

#### 9.14.3 改为逆置换 gather 后连续写回（2026-10-25）

前一版实验路径用 `index_copy_` 将临时结果散写到最终 `y`，并没有满足“最终输出连续写”的目标。现改为 prepare 阶段生成逆置换 `y_order`，kernel 按重排行连续写 `y_buf`，运行阶段执行 `torch.index_select(y_buf, 0, y_order, out=y)`；因此最终 `y` 是连续写入，且整个操作仍计入 kernel/调用侧计时。新增的 `y_buf`、`y_order` 均在 prepare 阶段分配，不计入运行计时。

该版本 f32 复测：NACA0015 RCM prepared 为 `293.7 µs`，auto 为 `272.2 µs`；两者输出检查均 PASS。complex64 auto 输出检查 PASS，RCM prepared 为 `642.7 µs`。由于复测有波动且连续回写仍包含一次完整输出重排，当前只保留 `FLAGSPARSE_SELL_CONTIGUOUS_OUTPUT=1` 实验开关，不默认启用；是否能提升最终算子，仍须在完整 10 矩阵、各 dtype 和相对厂商基线口径下复测。

#### 9.14.4 10 矩阵基线口径复测（2026-10-25）

为避免只看绝对时间，本轮将 A/B 结果与 `results_biv_sell_call_20261025/spmv_sell/performance.csv` 中的 `h800_scaled_ms` 厂商基线折算值合并计算加速比，定义为 `baseline_ms / measured_ms`。需要区分两条口径：

- **默认交付路径 `P2 sell_api`**：连续输出开关不参与该路径，10 个矩阵几何平均加速比为 f32 `0.616x`、complex64 `0.505x`；开启前后没有算子级变化。
- **实验路径 `P7 prepared RCM`**：只对 9 个方阵有效，非方阵 GL7d14 不执行 RCM。相对同一厂商基线，开启连续输出后的几何平均为 f32 `0.400x`、complex64 `0.259x`；关闭时分别为 `0.432x`、`0.235x`。f32 反而下降，complex64 仅相对 RCM 实验路径改善约 `1.102x`，仍远低于 `0.8x` 目标。

因此，**本轮没有确认 SELL 算子相对厂商基线的性能提升**，也不能把 `FLAGSPARSE_SELL_CONTIGUOUS_OUTPUT=1` 设为默认。完整逐矩阵、逐 dtype 的基线加速比见 `results_biv_sell_contig_speedup_20261025.csv`；原始 A/B 见 `results_biv_sell_contig_ab_off_20261025.csv` 和 `results_biv_sell_contig_ab_on_20261025.csv`。
### 2026-10-26 BI-V150 K-wide SELL/COO 实现与验证结果

基线 HEAD：`494ebee60b780e85d947fd591671ecdea2e7d986`。实现文件为 `src/flagsparse/sparse_operations/spmv_sell.py`、`src/flagsparse/sparse_operations/spmv_coo.py`、`tools/biv_spmv_probe.py`，新增正确性测试为 `tests/pytest/test_spmv_sell_kwide.py`。

A/B 每个矩阵同一进程交错 5 轮、每种配置取中位数，整个 A/B 独立重复两遍；表中为 `old_ms/new_ms (speedup)`，单位 ms。

#### `spmv_coo_f16_int_non`
| matrix | run 1 | run 2 |
|---|---:|---:|
| ASIC_680ks.mtx | 0.0886/0.0879 (1.008x) | 0.0886/0.0880 (1.006x) |
| GL7d14.mtx | 0.0605/0.0448 (1.351x) | 0.0606/0.0448 (1.353x) |
| NACA0015.mtx | 0.1809/0.1809 (1.000x) | 0.1810/0.1809 (1.000x) |
| amazon0601.mtx | 0.0963/0.0836 (1.152x) | 0.0962/0.0835 (1.152x) |
| auto.mtx | 0.2089/0.1880 (1.111x) | 0.2088/0.1880 (1.111x) |
| cage12.mtx | 0.0331/0.0199 (1.665x) | 0.0331/0.0200 (1.658x) |
| cfd2.mtx | 0.0548/0.0305 (1.796x) | 0.0549/0.0305 (1.799x) |
| filter3D.mtx | 0.0538/0.0536 (1.003x) | 0.0538/0.0536 (1.003x) |
| roadNet-TX.mtx | 0.0951/0.0957 (0.994x) | 0.0952/0.0956 (0.996x) |
| wave.mtx | 0.0363/0.0242 (1.499x) | 0.0363/0.0242 (1.498x) |
平均：old `0.090842` ms，new `0.080911` ms，加速 `1.1227x`。

#### `spmv_coo_f16f32_int_non`
| matrix | run 1 | run 2 |
|---|---:|---:|
| ASIC_680ks.mtx | 0.0958/0.0951 (1.008x) | 0.0956/0.0952 (1.004x) |
| GL7d14.mtx | 0.0624/0.0449 (1.391x) | 0.0624/0.0447 (1.396x) |
| NACA0015.mtx | 0.1877/0.1882 (0.998x) | 0.1875/0.1883 (0.996x) |
| amazon0601.mtx | 0.1006/0.0853 (1.180x) | 0.1005/0.0853 (1.178x) |
| auto.mtx | 0.2129/0.1904 (1.118x) | 0.2130/0.1903 (1.119x) |
| cage12.mtx | 0.0354/0.0199 (1.774x) | 0.0354/0.0200 (1.769x) |
| cfd2.mtx | 0.0567/0.0341 (1.665x) | 0.0566/0.0329 (1.721x) |
| filter3D.mtx | 0.0564/0.0561 (1.005x) | 0.0564/0.0563 (1.003x) |
| roadNet-TX.mtx | 0.1119/0.1126 (0.994x) | 0.1119/0.1126 (0.994x) |
| wave.mtx | 0.0379/0.0243 (1.561x) | 0.0379/0.0241 (1.573x) |
平均：old `0.095749` ms，new `0.085027` ms，加速 `1.1261x`。

#### `spmv_coo_i8i32_int_non`
| matrix | run 1 | run 2 |
|---|---:|---:|
| ASIC_680ks.mtx | 0.0936/0.0916 (1.023x) | 0.0937/0.0918 (1.021x) |
| GL7d14.mtx | 0.0625/0.0430 (1.452x) | 0.0624/0.0431 (1.447x) |
| NACA0015.mtx | 0.1844/0.1752 (1.052x) | 0.1844/0.1751 (1.053x) |
| amazon0601.mtx | 0.0912/0.0755 (1.207x) | 0.0914/0.0755 (1.211x) |
| auto.mtx | 0.1715/0.1543 (1.112x) | 0.1717/0.1544 (1.112x) |
| cage12.mtx | 0.0328/0.0188 (1.742x) | 0.0329/0.0188 (1.745x) |
| cfd2.mtx | 0.0500/0.0244 (2.047x) | 0.0500/0.0244 (2.046x) |
| filter3D.mtx | 0.0519/0.0451 (1.151x) | 0.0519/0.0451 (1.151x) |
| roadNet-TX.mtx | 0.1015/0.1003 (1.012x) | 0.1016/0.1002 (1.013x) |
| wave.mtx | 0.0365/0.0224 (1.625x) | 0.0365/0.0225 (1.626x) |
平均：old `0.087617` ms，new `0.075079` ms，加速 `1.1670x`。

#### `spmv_sell_f16_int_non`
| matrix | run 1 | run 2 |
|---|---:|---:|
| ASIC_680ks.mtx | 0.1384/0.0928 (1.491x) | 0.1384/0.0930 (1.488x) |
| GL7d14.mtx | 0.0502/0.0458 (1.098x) | 0.0502/0.0457 (1.099x) |
| NACA0015.mtx | 0.2106/0.1841 (1.144x) | 0.2106/0.1841 (1.144x) |
| amazon0601.mtx | 0.0979/0.0836 (1.172x) | 0.0978/0.0835 (1.171x) |
| auto.mtx | 0.1949/0.1847 (1.055x) | 0.1949/0.1849 (1.054x) |
| cage12.mtx | 0.0222/0.0219 (1.016x) | 0.0222/0.0219 (1.014x) |
| cfd2.mtx | 0.0370/0.0327 (1.133x) | 0.0370/0.0327 (1.131x) |
| filter3D.mtx | 0.0757/0.0564 (1.342x) | 0.0757/0.0564 (1.342x) |
| roadNet-TX.mtx | 0.1131/0.1123 (1.007x) | 0.1130/0.1124 (1.006x) |
| wave.mtx | 0.0311/0.0242 (1.284x) | 0.0312/0.0242 (1.286x) |
平均：old `0.097115` ms，new `0.083867` ms，加速 `1.1580x`。

#### `spmv_sell_f16f32_int_non`
| matrix | run 1 | run 2 |
|---|---:|---:|
| ASIC_680ks.mtx | 0.1402/0.0919 (1.525x) | 0.1401/0.0919 (1.524x) |
| GL7d14.mtx | 0.0504/0.0457 (1.105x) | 0.0504/0.0457 (1.102x) |
| NACA0015.mtx | 0.2133/0.1879 (1.135x) | 0.2132/0.1879 (1.134x) |
| amazon0601.mtx | 0.0991/0.0856 (1.159x) | 0.0991/0.0856 (1.158x) |
| auto.mtx | 0.1969/0.1866 (1.055x) | 0.1970/0.1869 (1.054x) |
| cage12.mtx | 0.0222/0.0219 (1.013x) | 0.0222/0.0218 (1.016x) |
| cfd2.mtx | 0.0386/0.0347 (1.110x) | 0.0386/0.0348 (1.109x) |
| filter3D.mtx | 0.0760/0.0564 (1.348x) | 0.0756/0.0564 (1.340x) |
| roadNet-TX.mtx | 0.1134/0.1127 (1.006x) | 0.1134/0.1129 (1.004x) |
| wave.mtx | 0.0312/0.0243 (1.284x) | 0.0313/0.0243 (1.286x) |
平均：old `0.098107` ms，new `0.084805` ms，加速 `1.1569x`。

#### `spmv_sell_f32_int_non`
| matrix | run 1 | run 2 |
|---|---:|---:|
| ASIC_680ks.mtx | 0.1536/0.0946 (1.625x) | 0.1535/0.0947 (1.620x) |
| GL7d14.mtx | 0.0522/0.0500 (1.045x) | 0.0523/0.0499 (1.048x) |
| NACA0015.mtx | 0.2322/0.2036 (1.140x) | 0.2322/0.2036 (1.141x) |
| amazon0601.mtx | 0.1115/0.0954 (1.169x) | 0.1115/0.0953 (1.169x) |
| auto.mtx | 0.2299/0.2167 (1.061x) | 0.2302/0.2168 (1.062x) |
| cage12.mtx | 0.0296/0.0277 (1.068x) | 0.0296/0.0276 (1.072x) |
| cfd2.mtx | 0.0489/0.0468 (1.044x) | 0.0490/0.0468 (1.046x) |
| filter3D.mtx | 0.1005/0.0733 (1.371x) | 0.1001/0.0734 (1.365x) |
| roadNet-TX.mtx | 0.1138/0.1129 (1.008x) | 0.1138/0.1128 (1.009x) |
| wave.mtx | 0.0492/0.0358 (1.371x) | 0.0491/0.0357 (1.374x) |
平均：old `0.112133` ms，new `0.095681` ms，加速 `1.1720x`。

#### `spmv_sell_i8i32_int_non`
| matrix | run 1 | run 2 |
|---|---:|---:|
| ASIC_680ks.mtx | 0.1347/0.0907 (1.485x) | 0.1343/0.0904 (1.485x) |
| GL7d14.mtx | 0.0478/0.0453 (1.054x) | 0.0479/0.0454 (1.056x) |
| NACA0015.mtx | 0.2007/0.1755 (1.144x) | 0.2008/0.1753 (1.145x) |
| amazon0601.mtx | 0.0872/0.0760 (1.148x) | 0.0872/0.0759 (1.148x) |
| auto.mtx | 0.1614/0.1528 (1.057x) | 0.1614/0.1527 (1.057x) |
| cage12.mtx | 0.0202/0.0210 (0.962x) | 0.0201/0.0209 (0.963x) |
| cfd2.mtx | 0.0258/0.0262 (0.988x) | 0.0259/0.0262 (0.990x) |
| filter3D.mtx | 0.0635/0.0467 (1.362x) | 0.0639/0.0468 (1.365x) |
| roadNet-TX.mtx | 0.1127/0.1122 (1.004x) | 0.1127/0.1122 (1.005x) |
| wave.mtx | 0.0281/0.0232 (1.209x) | 0.0281/0.0232 (1.210x) |
平均：old `0.088215` ms，new `0.076921` ms，加速 `1.1468x`。

#### COO choice 与 prepare

`spmv_coo_f16_int_non`：
ASIC_680ks.mtx: existing / existing；GL7d14.mtx: sell_k2 / sell_k2；NACA0015.mtx: existing / existing；amazon0601.mtx: sell_k4 / sell_k4；auto.mtx: sell_k2 / sell_k2；cage12.mtx: sell_k2 / sell_k2；cfd2.mtx: sell_k2 / sell_k2；filter3D.mtx: existing / existing；roadNet-TX.mtx: existing / existing；wave.mtx: sell_k4 / sell_k4
prepare old 平均 `24.372` ms。
prepare new 平均 `127.735` ms。
ASIC_680ks.mtx: existing / existing；GL7d14.mtx: sell_k2 / sell_k2；NACA0015.mtx: existing / existing；amazon0601.mtx: sell_k4 / sell_k4；auto.mtx: sell_k2 / sell_k2；cage12.mtx: sell_k2 / sell_k2；cfd2.mtx: sell_k2 / sell_k4；filter3D.mtx: existing / existing；roadNet-TX.mtx: existing / existing；wave.mtx: sell_k4 / sell_k4
prepare old 平均 `24.152` ms。
prepare new 平均 `109.258` ms。
ASIC_680ks.mtx: existing / existing；GL7d14.mtx: sell_k2 / sell_k2；NACA0015.mtx: sell_k4 / sell_k4；amazon0601.mtx: sell_k4 / sell_k4；auto.mtx: sell_k2 / sell_k2；cage12.mtx: sell_k2 / sell_k2；cfd2.mtx: sell_k2 / sell_k2；filter3D.mtx: sell_k2 / sell_k2；roadNet-TX.mtx: existing / existing；wave.mtx: sell_k2 / sell_k2
prepare old 平均 `24.069` ms。
prepare new 平均 `138.095` ms。

COO 两遍只有一个 choice 不一致：`spmv_coo_f16f32_int_non/cfd2.mtx` 为 `sell_k2` / `sell_k4`，少于计划第 6 节的 2 个停止阈值。

原始 A/B：`results_biv_kwide_ab_run1_20261026.csv`、`results_biv_kwide_ab_run2_20261026.csv`；投影：`results_biv_kwide_project_20261026.txt`。

#### 全量 65 变体、P10 与异常记录

- 全量输出目录为 `results_biv_65_20261026_kwide_final/`，包含 65 个注册变体、13 个算子目录；`delivery_table.py` 报告为 56 个已配置变体 accuracy/performance `Passed`，9 个 fp64/c64 变体 `NotFound`（未配置）。
- 相对 `results_biv_65_20261024_final` 逐变体、逐矩阵比较可测的 performance CSV 后，除上述 7 个目标变体外，没有平均耗时上升超过 3% 的变体；因此该项为空。目标 7 项的 A/B 平均改善分别为 SELL f32 `1.1720x`、SELL f16 `1.1580x`、SELL f16→f32 `1.1569x`、SELL i8→i32 `1.1468x`、COO f16 `1.1227x`、COO i8→i32 `1.1670x`、COO f16→f32 `1.1261x`（两遍平均口径）。
- P10 输出为 `results_biv_p10_20261026_kwide.csv`，4 个矩阵/3 个数据类型/6 个配置共 84 行，全部 `PASS`。`GROUP=2,K=2,C=32` 在 f32/f16 的 8 个组合和 int8 的 3 个组合中最快；int8 `roadNet-TX.mtx` 由 `C=64,GROUP=1,K=1` 略快。P10 只作为探针数据，不改变生产代码。
- 正确性新增测试 `tests/pytest/test_spmv_sell_kwide.py`：30 passed；`test_spmv_sell_prepared.py`：151 passed、18 skipped；`test_spmv_coo_iluvatar_route.py`：5 passed；q4 SELL/COO：20 passed；complex SELL prepared：8 passed。
- `tests/pytest/test_spmv_sell_out_accuracy.py`：Iluvatar float64 两个用例失败，输出全零；基线 HEAD 复跑同样失败，确认是既有基线/平台问题。
- `tests/pytest/test_spmv_coo_accuracy.py`：Iluvatar float64/complex128 相关 39 个用例失败，输出全零或错误；属于已知 BI-V150 fp64/complex128 平台问题。
- 第一次 65 runner `results_biv_65_20261026_kwide/` 因 complex SELL `route="call"` 错误采用 `k=4`，10 个 complex SELL 行报 `ValueError: k > 1 requires group=1 and real-valued SELL data`；该问题已修复。该次 runner 后续 `spmm_csc` 长时间停滞并手动终止，不作为最终结果。
- `make pre-commit-check` 未完成：pre-commit 初始化网络拉取失败，`CalledProcessError`、`git fetch origin --tags`、`GnuTLS recv error (-110)`。
- 直接工具检查：`black --check` 报以下 4 个文件会被重排：`src/flagsparse/sparse_operations/spmv_sell.py`、`src/flagsparse/sparse_operations/spmv_coo.py`、`tools/biv_spmv_probe.py`、`tests/pytest/test_spmv_sell_kwide.py`；`isort`、`flake8` 命令当前不存在。

#### 合入前审查修复（2026-10-26，CUDA 机）

上面的数据来自 BI-V 侧原始改动包。合入前在 CUDA 机上（`FLAGSPARSE_BACKEND=iluvatar` 模拟 + CUDA 原生）审查，修了 5 处，性能路径的 kernel 和默认配置不变：

1. **未排序 COO 算错**：`prepare_spmv_coo(..., sort_by_row=False)` 收到行未排序的 COO 时，转 SELL 用 bincount 构造 CSR，默认假设行已排序；auto 模式选中 SELL 后 f16 / int8 的 max_err 达 36–38。prepare 时的实测只比速度、不检查结果，所以会选中算错的路径。现在先按行稳定排序再转换。交付测试全用 `sort_by_row=True`，所以没覆盖到。
2. **`FLAGSPARSE_SELL_PREP_K>1` 让复数 SELL 的 prepare 抛错**，`route="auto"` / `"call"` 也一样。第一次 65 runner 就是因为这个失败，当时只修了默认值的情况。现在显式传入的 `k` 仍严格校验；环境变量和默认值只在 K-wide 能运行的地方生效（实数、`group == 1`），其他情况用 1。`route="call"` 显式传 `k > 1` 时报错。
3. **非 Iluvatar 后端显式传 `k > 1` 被静默忽略**：`prepared.k` 显示 4，实际跑的是 K=1。运行时的 Iluvatar 判断已去掉（kernel 是纯 Triton），CUDA 上显式传 `k` 现在真正生效；默认仍是 1，CUDA 的默认行为不变。
4. **COO prepare 冗余**：原来为 2 种输出 dtype × K∈{2,4} 做 4 次完整的重新布局，没选中的计划也全部保留。现在只编码一次，各候选由 `_sell_plan_variant` 共享这份布局，最后只保留选中的那份（`prepared.sell_plans[out_dtype]` 是 `_CooSellPlan`）；`FLAGSPARSE_ILUVATAR_COO_SELL=1` 时不再做实测。5090 模拟下 prepare 从约 14 ms 降到约 7 ms。BI-V 上的耗时需要复测。
   另外，`sort_by_row=False` 时 `data_non` 就是调用方的张量，原路径每次调用都会读到就地修改；SELL 路径现在按版本号检测修改并重新编码数值，语义和原路径一致。
5. **测试**：`tests/pytest/test_spmv_sell_kwide.py` 重写为 78 个用例。新增：随机矩阵（含空行、行宽大于 K）、SELL f16→f32 与 int8→f32、未排序 COO × `sort_by_row`、显式非法 k、环境变量 K 加复数 / group、共享布局、只保留选中计划、强制 K、就地修改语义、非法模式。原来 auto 模式那条断言永远成立，已改为实际检查。SELL 用例在 CUDA 上也会运行，不再整体跳过。用原改动包的代码跑这套测试，Iluvatar 模拟和 CUDA 下都有失败，说明测试能抓到上面这些问题。

验证结果：
- `make format-check`、`lint`、`lint-src`、`pre-commit-check` 全部通过。
- `tests/ci` 476 passed；唯一失败是 `test_installed_wheel`，干净的 HEAD 上同样失败，是本机没装 wheel 的环境问题。
- SELL / COO 的 pytest 在 Iluvatar 模拟和 CUDA 下都通过。

P10 的网格把 GROUP 和 K 绑在一起变（G1K1 / G2K2 / G4K4），没有生产用的 G1K4 作对照。跨文件对比，G2K2 只在 roadNet-TX 上比 K=4 快（f16 82.9 vs 112.3 µs），其他矩阵都更慢，和"roadNet 受 program 数量限制"的判断一致。下一轮要单独扫 GROUP × K。
