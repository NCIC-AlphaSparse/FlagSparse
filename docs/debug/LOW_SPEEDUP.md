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
