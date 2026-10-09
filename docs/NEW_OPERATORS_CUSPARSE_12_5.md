# 新算子清单（对标 cuSPARSE 12.5）

2026-09-27 整理。在已交付的 20 个变体之外，按 cuSPARSE 12.5 的 API 列出 96 个候选变体，
按 5 个国产平台（沐曦 C550、摩尔线程 S5000、海光 BW1000、天数 BI-V150、昇腾 910B）上的实现难度排序。

## 范围

- **不含 f64 和 c64**（c64 = complex128）。`c32`（complex64）保留。
- **不含已交付的 20 个**（`conf/operators.yaml` 的 `delivery_variants`）。
- **收录了原 40 表中剩下的 10 个**：40 表减去已交付的 20 个，再去掉 f64/c64 后剩 10 个，全部收录，备注列标为「40 表」。
- **17 个混合精度**：输入与输出类型不同（如 `f16f32`、`i8i32`），或实矩阵乘复向量（`f32c32`）。

## 命名规则

沿用交付列表的写法：`算子_格式_数据类型_int_opA[_opB][_row|_col]`。

| 片段 | 含义 |
|---|---|
| `int` | int32 索引 |
| `non` / `trans` / `conj` | 不转置 / 转置 / 共轭转置 |
| `row` / `col` | 稠密矩阵行主序 / 列主序 |
| `f16`、`bf16` | 输入输出同类型，按 cuSPARSE 规则用 f32 计算 |
| `f16f32`、`bf16f32` | 输入 f16/bf16，输出和计算都是 f32 |
| `i8i32`、`i8f32` | 输入 i8，输出 i32 / f32 |
| `f32c32` | 矩阵 f32，向量 c32 |
| `_upper` | 上三角（默认下三角） |
| `_alg2`、`_alg3` | `CUSPARSE_SPGEMM_ALG2` / `ALG3`（默认 ALG1） |

## 验证方法

清单中每一项都在 cuSPARSE 12.5 上实际调用过，全部成功：

- 头文件 `CUSPARSE_VER_MAJOR 12` / `MINOR 5`，库为 `libcusparse.so.12.5.9.5`。
- 每个组合都走完整流程：查缓冲区大小 → 分析或预处理（如需要）→ 计算 → 同步 GPU。
- 测试矩阵为 8×8，按 CSR/COO/CSC/BSR/SELL/Blocked-ELL 各构造一份。
- **只验证了接口可用，没有核对计算结果。** 测试矩阵的值全为 0，混合精度组合即使接受了参数，也不能证明结果正确。

### 实测不支持、因此未收录的组合

| API | 不支持 |
|---|---|
| `cusparseSpMV` | BSR 格式（通用 API 不支持） |
| `cusparseSpMM` | BSR 格式下稀疏矩阵转置/共轭；Blocked-ELL 格式下的 i8、c32 和稀疏矩阵转置 |
| `cusparseSDDMM` | CSR 格式下的 bf16（BSR 格式支持） |
| `cusparseSpGEMM` | f16、bf16 |
| `cusparseSpSV` / `cusparseSpSM` | f16；SpSV 不支持 BSR 格式 |
| `cusparseSparseToDense` | BSR、SELL、Blocked-ELL 格式 |

### 接口层面的注意点

- `cusparseSpVV`、`cusparseAxpby` 在 12.5 里**还不是**弃用 API——查 CUDA 12.9 头文件，两者的
  `CUSPARSE_DEPRECATED` 注释明确写着 `// deprecated in CTK 12.8`，即从 12.8 起才标记弃用，12.5
  里正常可用。此前清单误标为「已弃用」，已改为「12.8 起弃用」。
- `csr2csc` 用的 `cusparseCsr2cscEx2` 不属于通用 API，但在 12.5 里。
- `_upper` 不是单独接口：在稀疏矩阵描述符上设上三角属性，再调同一个 `cusparseSpSV`。
- `dense2sparse_bell`：cuSPARSE 转 Blocked-ELL 时，块的列位置可能要求调用方预先给定，只填数值。若是这样，这一项比难度表里写的简单。**已核实**：官方文档 `cusparseDenseToSparse` 一节原文写明支持转换到 CSR/CSC/COO/Blocked-ELL 四种格式，`f16` 在其类型表里，`dense2sparse_bell_f16_int` 成立；块的列位置是否需要预先给定仍未核实。

## 难度怎么算

难度分是估计，用于排先后，不是实测工作量。

**算法本身的基础分，从易到难：** 只搬数据 → 逐元素 / 全局求和 → 需要前缀和的格式转换 →
按行并行的 SpMV/SpMM → 需要原子加的（COO、CSC、转置方向）→ SDDMM 和块格式的块内矩阵乘（`tl.dot`）→
SpGEMM（先算结构、再算数值）→ 三角求解 SpSV/SpSM（行间依赖，线程块之间要等待）。

**加分项：**

| 条件 | 加分 |
|---|---:|
| 稀疏矩阵转置或共轭（依赖方向反过来） | +1.5 ~ +2 |
| i8（整数累加、参考实现、精确比对都要新写） | +1.5 ~ +2 |
| c32（寄存器翻倍；MACA 单线程私有内存上限，MUSA 缺部分复数算子） | +1 |
| f16 且结果要原子累加，或走 `tl.dot` | +1 |
| f16 只读写、用 f32 计算 | +0.5 |
| 读 f16、输出 f32 | +0.3 |
| 稠密矩阵转置或列主序 | +0.5 |

星级：★ < 2.5 ≤ ★★ < 4 ≤ ★★★ < 5.5 ≤ ★★★★ < 7.5 ≤ ★★★★★。

**排序规则：** 计算类按难度排在前面；格式转换整体往后排；bf16 各平台不一定支持，统一放最后。
同时属于 bf16 和格式转换的两项放在全表最末。

**平台共性：** 昇腾 910B 的 Triton 没有共享内存，也没有 `associative_scan`，前缀和、SpGEMM 哈希表等要单独适配，
所有档位在昇腾上都更难一档，表中不逐行重复。三角求解排在最难，主要因为跨线程块等待在国产卡上最容易卡死。

## 清单

### 一、计算类（76），按难度

| # | 变体 | 难度 | cuSPARSE 12.5 API | 类型：输入 → 输出（计算） | 主要难点 | 备注 |
|---:|---|---|---|---|---|---|
| 1 | `scatter_i8_int` | ★ 1.3 | `cusparseScatter` | i8 | 只搬数据 |  |
| 2 | `axpby_f16_int` | ★ 2.0 | `cusparseAxpby` | f16 → f16 (f32) | 逐元素；f16 读写、f32 计算 | 12.8 起弃用 |
| 3 | `spmv_sell_f32_int_non` | ★★ 2.5 | `cusparseSpMV` (SELL_ALG1) | f32 → f32 (f32) | 规则访存，按切片并行 |  |
| 4 | `spmv_csr_f16f32_int_non` | ★★ 2.8 | `cusparseSpMV` | f16 → **f32** (f32) | 按行并行，无原子；f16 读、f32 累加输出 |  |
| 5 | `spvv_f16f32_int_non` | ★★ 2.8 | `cusparseSpVV` | f16 → **f32** (f32) | 全局规约；f16 读、f32 累加输出 | 12.8 起弃用 |
| 6 | `spmv_csr_f16_int_non` | ★★ 3.0 | `cusparseSpMV` | f16 → f16 (f32) | 按行并行，无原子；f16 读写、f32 计算 |  |
| 7 | `spmv_csr_f32c32_int_non` | ★★ 3.0 | `cusparseSpMV` | A f32，x/y **c32** (c32) | 按行并行，无原子；实矩阵 × 复向量 |  |
| 8 | `spmv_sell_f16_int_non` | ★★ 3.0 | `cusparseSpMV` (SELL_ALG1) | f16 → f16 (f32) | 规则访存，按切片并行；f16 读写、f32 计算 |  |
| 9 | `spmm_csr_f16f32_int_non_non_row` | ★★ 3.3 | `cusparseSpMM` | f16 → **f32** (f32) | 按行并行；f16 读、f32 累加输出 |  |
| 10 | `spmm_csr_f16_int_non_non_row` | ★★ 3.5 | `cusparseSpMM` | f16 → f16 (f32) | 按行并行；f16 读写、f32 计算 |  |
| 11 | `spmm_csr_f32_int_non_non_col` | ★★ 3.5 | `cusparseSpMM` | f32 → f32 (f32) | 按行并行；列主序访存 |  |
| 12 | `spmm_csr_f32_int_non_trans_row` | ★★ 3.5 | `cusparseSpMM` | f32 → f32 (f32) | 按行并行；B 转置访存不连续 |  |
| 13 | `spmv_csc_f32_int_non` | ★★ 3.5 | `cusparseSpMV` | f32 → f32 (f32) | 按列散写需原子加 |  |
| 14 | `spmv_csr_c32_int_non` | ★★ 3.5 | `cusparseSpMV` | c32 → c32 (c32) | 按行并行，无原子；复数（寄存器翻倍） | 40 表 |
| 15 | `spmv_sell_c32_int_non` | ★★ 3.5 | `cusparseSpMV` (SELL_ALG1) | c32 → c32 (c32) | 规则访存，按切片并行；复数（寄存器翻倍） |  |
| 16 | `spvv_c32_int_conj` | ★★ 3.5 | `cusparseSpVV` | c32 → c32 (c32) | 全局规约；复数（寄存器翻倍） | 12.8 起弃用 |
| 17 | `spmv_coo_f16f32_int_non` | ★★ 3.8 | `cusparseSpMV` | f16 → **f32** (f32) | 分段规约或原子加；f16 读、f32 累加输出 |  |
| 18 | `spmm_csr_c32_int_non_non_row` | ★★★ 4.0 | `cusparseSpMM` | c32 → c32 (c32) | 按行并行；复数（寄存器翻倍） | 40 表 |
| 19 | `spmv_coo_f32_int_trans` | ★★★ 4.0 | `cusparseSpMV` | f32 → f32 (f32) | 原子散写 |  |
| 20 | `spmv_csr_f32_int_trans` | ★★★ 4.0 | `cusparseSpMV` | f32 → f32 (f32) | 按列散写需原子加 |  |
| 21 | `spmv_csr_i8f32_int_non` | ★★★ 4.0 | `cusparseSpMV` | i8 → **f32** (f32) | 按行并行，无原子；整数累加：参考实现与精确比对全新 |  |
| 22 | `spmv_csr_i8i32_int_non` | ★★★ 4.0 | `cusparseSpMV` | i8 → **i32** (i32) | 按行并行，无原子；整数累加：参考实现与精确比对全新 |  |
| 23 | `spmv_sell_i8i32_int_non` | ★★★ 4.0 | `cusparseSpMV` (SELL_ALG1) | i8 → **i32** (i32) | 规则访存，按切片并行；整数累加：参考实现与精确比对全新 |  |
| 24 | `spvv_i8i32_int_non` | ★★★ 4.0 | `cusparseSpVV` | i8 → **i32** (i32) | 全局规约；整数累加：参考实现与精确比对全新 | 12.8 起弃用 |
| 25 | `spmm_csc_f32_int_non_non_row` | ★★★ 4.5 | `cusparseSpMM` | f32 → f32 (f32) | 按列散写需原子 |  |
| 26 | `spmm_csr_i8i32_int_non_non_row` | ★★★ 4.5 | `cusparseSpMM` | i8 → **i32** (i32) | 按行并行；整数累加：参考实现与精确比对全新 |  |
| 27 | `spmv_coo_c32_int_non` | ★★★ 4.5 | `cusparseSpMV` | c32 → c32 (c32) | 分段规约或原子加；复数（寄存器翻倍） | 40 表 |
| 28 | `spmv_coo_f16_int_non` | ★★★ 4.5 | `cusparseSpMV` | f16 → f16 (f32) | 分段规约或原子加；f16 输出需 f32 缓冲原子累加 |  |
| 29 | `spmv_csc_c32_int_non` | ★★★ 4.5 | `cusparseSpMV` | c32 → c32 (c32) | 按列散写需原子加；复数（寄存器翻倍） |  |
| 30 | `spmv_csc_f16_int_non` | ★★★ 4.5 | `cusparseSpMV` | f16 → f16 (f32) | 按列散写需原子加；f16 输出需 f32 缓冲原子累加 |  |
| 31 | `sddmm_csr_f32_int_non_non_col` | ★★★ 5.0 | `cusparseSDDMM` | f32 → f32 (f32) | 每个非零一次点积，按采样访存；转置/列主序访存 |  |
| 32 | `sddmm_csr_f32_int_non_trans_row` | ★★★ 5.0 | `cusparseSDDMM` | f32 → f32 (f32) | 每个非零一次点积，按采样访存；转置/列主序访存 |  |
| 33 | `sddmm_csr_f32_int_trans_non_row` | ★★★ 5.0 | `cusparseSDDMM` | f32 → f32 (f32) | 每个非零一次点积，按采样访存；转置/列主序访存 |  |
| 34 | `spmm_bell_f32_int_non_non_row` | ★★★ 5.0 | `cusparseSpMM` (BLOCKED_ELL_ALG1) | f32 → f32 (f32) | 面向张量核的块乘（tl.dot） | 无 i8/c32、opA 仅 non |
| 35 | `spmm_bsr_f32_int_non_non_row` | ★★★ 5.0 | `cusparseSpMM` (BSR_ALG1) | f32 → f32 (f32) | 块内矩阵乘（tl.dot） | 仅 opA=non |
| 36 | `spmm_coo_c32_int_non_non_row` | ★★★ 5.0 | `cusparseSpMM` | c32 → c32 (c32) | 分段规约/原子；复数（寄存器翻倍） | 40 表 |
| 37 | `spmm_coo_f16_int_non_non_row` | ★★★ 5.0 | `cusparseSpMM` | f16 → f16 (f32) | 分段规约/原子；f16 输出需 f32 缓冲原子累加 |  |
| 38 | `spmm_csr_f32_int_trans_non_row` | ★★★ 5.0 | `cusparseSpMM` | f32 → f32 (f32) | 按行并行；opA 转置需原子或显式转置 |  |
| 39 | `spmv_coo_c32_int_conj` | ★★★ 5.0 | `cusparseSpMV` | c32 → c32 (c32) | 原子散写；复数（寄存器翻倍） |  |
| 40 | `spmv_coo_i8i32_int_non` | ★★★ 5.0 | `cusparseSpMV` | i8 → **i32** (i32) | 分段规约或原子加；整数累加：参考实现与精确比对全新 |  |
| 41 | `spmv_csr_c32_int_conj` | ★★★ 5.0 | `cusparseSpMV` | c32 → c32 (c32) | 按列散写需原子加；复数（寄存器翻倍） |  |
| 42 | `sddmm_bsr_f32_int_non_non_row` | ★★★★ 5.5 | `cusparseSDDMM` | f32 → f32 (f32) | 块状采样矩阵乘 |  |
| 43 | `sddmm_csr_c32_int_non_non_row` | ★★★★ 5.5 | `cusparseSDDMM` | c32 → c32 (c32) | 每个非零一次点积，按采样访存；复数（寄存器翻倍） | 原 42 表待补项 |
| 44 | `sddmm_csr_f16_int_non_non_row` | ★★★★ 5.5 | `cusparseSDDMM` | f16 → f16 (f32) | 每个非零一次点积，按采样访存；f16 tl.dot（各平台张量核支持不一） | CSR 不支持 bf16 |
| 45 | `spmm_bsr_f32_int_non_trans_row` | ★★★★ 5.5 | `cusparseSpMM` (BSR_ALG1) | f32 → f32 (f32) | 块内矩阵乘（tl.dot）；B 转置访存不连续 | 仅 opA=non |
| 46 | `spmm_coo_i8i32_int_non_non_row` | ★★★★ 5.5 | `cusparseSpMM` | i8 → **i32** (i32) | 分段规约/原子；整数累加：参考实现与精确比对全新 |  |
| 47 | `spmm_csc_c32_int_non_non_row` | ★★★★ 5.5 | `cusparseSpMM` | c32 → c32 (c32) | 按列散写需原子；复数（寄存器翻倍） |  |
| 48 | `spmm_csc_f16_int_non_non_row` | ★★★★ 5.5 | `cusparseSpMM` | f16 → f16 (f32) | 按列散写需原子；f16 输出需 f32 缓冲原子累加 |  |
| 49 | `spmm_bell_f16f32_int_non_non_row` | ★★★★ 5.8 | `cusparseSpMM` (BLOCKED_ELL_ALG1) | f16 → **f32** (f32) | 面向张量核的块乘（tl.dot）；f16 读、f32 累加输出 | 无 i8/c32、opA 仅 non |
| 50 | `spmm_bell_f16_int_non_non_row` | ★★★★ 6.0 | `cusparseSpMM` (BLOCKED_ELL_ALG1) | f16 → f16 (f32) | 面向张量核的块乘（tl.dot）；f16 tl.dot（各平台张量核支持不一） | 无 i8/c32、opA 仅 non |
| 51 | `spmm_bsr_f16_int_non_non_row` | ★★★★ 6.0 | `cusparseSpMM` (BSR_ALG1) | f16 → f16 (f32) | 块内矩阵乘（tl.dot）；f16 tl.dot（各平台张量核支持不一） | 仅 opA=non |
| 52 | `spmm_coo_f32_int_trans_non_row` | ★★★★ 6.0 | `cusparseSpMM` | f32 → f32 (f32) | 分段规约/原子；opA 转置需原子或显式转置 |  |
| 53 | `spmm_csr_c32_int_conj_non_row` | ★★★★ 6.0 | `cusparseSpMM` | c32 → c32 (c32) | 按行并行；opA 转置需原子或显式转置；复数（寄存器翻倍） |  |
| 54 | `sddmm_bsr_f16_int_non_non_row` | ★★★★ 6.5 | `cusparseSDDMM` | f16 → f16 (f32) | 块状采样矩阵乘；f16 tl.dot（各平台张量核支持不一） |  |
| 55 | `spgemm_csr_f32_int_non_non` | ★★★★ 7.0 | `cusparseSpGEMM` (ALG1) | f32 → f32 (f32) | 两阶段（符号+数值），哈希表依赖共享内存 | 40 表 |
| 56 | `spmm_bsr_i8i32_int_non_non_row` | ★★★★ 7.0 | `cusparseSpMM` (BSR_ALG1) | i8 → **i32** (i32) | 块内矩阵乘（tl.dot）；整数累加：参考实现与精确比对全新 | 仅 opA=non |
| 57 | `spsv_csr_f32_int_non` | ★★★★ 7.0 | `cusparseSpSV` | f32 → f32 (f32) | 行间依赖，跨块等待（易卡死） | 40 表 |
| 58 | `spsm_csr_f32_int_non_non_row` | ★★★★★ 7.5 | `cusparseSpSM` | f32 → f32 (f32) | 多右端三角求解，跨块等待 | 40 表 |
| 59 | `spsv_coo_f32_int_non` | ★★★★★ 7.5 | `cusparseSpSV` | f32 → f32 (f32) | 行间依赖，跨块等待（易卡死） | 40 表 |
| 60 | `spsv_csr_f32_int_non_upper` | ★★★★★ 7.5 | `cusparseSpSV` | f32 → f32 (f32) | 行间依赖，跨块等待（易卡死）；上三角 | 上三角 |
| 61 | `spgemm_csr_c32_int_non_non` | ★★★★★ 8.0 | `cusparseSpGEMM` (ALG1) | c32 → c32 (c32) | 两阶段（符号+数值），哈希表依赖共享内存；复数（寄存器翻倍） | 无 f16/bf16 |
| 62 | `spgemm_csr_f32_int_non_non_alg2` | ★★★★★ 8.0 | `cusparseSpGEMM` (ALG2) | f32 → f32 (f32) | 两阶段（符号+数值）+ 限定内存/分块 | 内存受限算法 |
| 63 | `spgemm_csr_f32_int_non_non_alg3` | ★★★★★ 8.0 | `cusparseSpGEMM` (ALG3) | f32 → f32 (f32) | 两阶段（符号+数值）+ 限定内存/分块 | 分块，更省内存 |
| 64 | `spsm_coo_f32_int_non_non_row` | ★★★★★ 8.0 | `cusparseSpSM` | f32 → f32 (f32) | 多右端三角求解，跨块等待 |  |
| 65 | `spsm_csr_f32_int_non_non_col` | ★★★★★ 8.0 | `cusparseSpSM` | f32 → f32 (f32) | 多右端三角求解，跨块等待；B 转置/列主序 |  |
| 66 | `spsm_csr_f32_int_non_trans_row` | ★★★★★ 8.0 | `cusparseSpSM` | f32 → f32 (f32) | 多右端三角求解，跨块等待；B 转置/列主序 |  |
| 67 | `spsv_csr_c32_int_non` | ★★★★★ 8.0 | `cusparseSpSV` | c32 → c32 (c32) | 行间依赖，跨块等待（易卡死）；复数（寄存器翻倍） | 40 表 |
| 68 | `spsv_sell_f32_int_non` | ★★★★★ 8.0 | `cusparseSpSV` | f32 → f32 (f32) | 行间依赖，跨块等待（易卡死） |  |
| 69 | `spsm_csr_c32_int_non_non_row` | ★★★★★ 8.5 | `cusparseSpSM` | c32 → c32 (c32) | 多右端三角求解，跨块等待；复数（寄存器翻倍） |  |
| 70 | `spsv_coo_c32_int_non` | ★★★★★ 8.5 | `cusparseSpSV` | c32 → c32 (c32) | 行间依赖，跨块等待（易卡死）；复数（寄存器翻倍） | 40 表 |
| 71 | `spsv_csr_f32_int_trans` | ★★★★★ 8.5 | `cusparseSpSV` | f32 → f32 (f32) | 行间依赖，跨块等待（易卡死）；转置方向依赖逆序 |  |
| 72 | `spsm_csr_f32_int_trans_non_row` | ★★★★★ 9.0 | `cusparseSpSM` | f32 → f32 (f32) | 多右端三角求解，跨块等待；转置方向依赖逆序 |  |
| 73 | `spsv_coo_f32_int_trans` | ★★★★★ 9.0 | `cusparseSpSV` | f32 → f32 (f32) | 行间依赖，跨块等待（易卡死）；转置方向依赖逆序 |  |
| 74 | `spsv_csr_c32_int_conj` | ★★★★★ 9.5 | `cusparseSpSV` | c32 → c32 (c32) | 行间依赖，跨块等待（易卡死）；转置方向依赖逆序；复数（寄存器翻倍） |  |
| 75 | `spsm_coo_c32_int_conj_non_row` | ★★★★★ 10.5 | `cusparseSpSM` | c32 → c32 (c32) | 多右端三角求解，跨块等待；转置方向依赖逆序；复数（寄存器翻倍） |  |
| 76 | `spsv_sell_c32_int_conj` | ★★★★★ 10.5 | `cusparseSpSV` | c32 → c32 (c32) | 行间依赖，跨块等待（易卡死）；转置方向依赖逆序；复数（寄存器翻倍） |  |

### 二、格式转换（8），往后排

| # | 变体 | 难度 | cuSPARSE 12.5 API | 类型：输入 → 输出（计算） | 主要难点 | 备注 |
|---:|---|---|---|---|---|---|
| 77 | `sparse2dense_csr_f32_int` | ★ 1.5 | `cusparseSparseToDense` | f32 | 清零 + 一次 scatter |  |
| 78 | `sparse2dense_coo_f16_int` | ★ 2.0 | `cusparseSparseToDense` | f16 | 清零 + 一次 scatter |  |
| 79 | `sparse2dense_csc_c32_int` | ★★ 2.5 | `cusparseSparseToDense` | c32 | 清零 + 一次 scatter；复数（寄存器翻倍） |  |
| 80 | `csr2csc_f32_int` | ★★ 3.0 | `cusparseCsr2cscEx2` | f32 | 计数 + 前缀和 + 散写，需原子计数 |  |
| 81 | `dense2sparse_csr_f32_int` | ★★ 3.0 | `cusparseDenseToSparse` | f32 | 计数 + 前缀和 + 写出（依赖 scan） |  |
| 82 | `dense2sparse_csr_f16_int` | ★★ 3.5 | `cusparseDenseToSparse` | f16 | 计数 + 前缀和 + 写出（依赖 scan） |  |
| 83 | `csr2csc_c32_int` | ★★★ 4.0 | `cusparseCsr2cscEx2` | c32 | 计数 + 前缀和 + 散写，需原子计数；复数（寄存器翻倍） |  |
| 84 | `dense2sparse_bell_f16_int` | ★★★ 4.5 | `cusparseDenseToSparse` | f16 | 判断块非零 + 按块排布 |  |

### 三、bf16（12），放最后

| # | 变体 | 难度 | cuSPARSE 12.5 API | 类型：输入 → 输出（计算） | 主要难点 | 备注 |
|---:|---|---|---|---|---|---|
| 85 | `gather_bf16_int` | ★ 1.5 | `cusparseGather` | bf16 | 只搬数据 |  |
| 86 | `scatter_bf16_int` | ★ 1.5 | `cusparseScatter` | bf16 | 只搬数据 |  |
| 87 | `spmv_csr_bf16f32_int_non` | ★★ 2.8 | `cusparseSpMV` | bf16 → **f32** (f32) | 按行并行，无原子；bf16 读、f32 累加输出 |  |
| 88 | `spmv_sell_bf16f32_int_non` | ★★ 2.8 | `cusparseSpMV` (SELL_ALG1) | bf16 → **f32** (f32) | 规则访存，按切片并行；bf16 读、f32 累加输出 |  |
| 89 | `spmv_csr_bf16_int_non` | ★★ 3.0 | `cusparseSpMV` | bf16 → bf16 (f32) | 按行并行，无原子；bf16 读写、f32 计算 |  |
| 90 | `spmm_csr_bf16_int_non_non_row` | ★★ 3.5 | `cusparseSpMM` | bf16 → bf16 (f32) | 按行并行；bf16 读写、f32 计算 |  |
| 91 | `spmv_coo_bf16_int_non` | ★★★ 4.5 | `cusparseSpMV` | bf16 → bf16 (f32) | 分段规约或原子加；bf16 输出需 f32 缓冲原子累加 |  |
| 92 | `spmm_bsr_bf16f32_int_non_non_row` | ★★★★ 5.8 | `cusparseSpMM` (BSR_ALG1) | bf16 → **f32** (f32) | 块内矩阵乘（tl.dot）；bf16 读、f32 累加输出 | 仅 opA=non |
| 93 | `spmm_bell_bf16_int_non_non_row` | ★★★★ 6.0 | `cusparseSpMM` (BLOCKED_ELL_ALG1) | bf16 → bf16 (f32) | 面向张量核的块乘（tl.dot）；bf16 tl.dot（各平台张量核支持不一） | 无 i8/c32、opA 仅 non |
| 94 | `sddmm_bsr_bf16_int_non_non_row` | ★★★★ 6.5 | `cusparseSDDMM` | bf16 → bf16 (f32) | 块状采样矩阵乘；bf16 tl.dot（各平台张量核支持不一） |  |
| 95 | `csr2csc_bf16_int` | ★★ 3.5 | `cusparseCsr2cscEx2` | bf16 | 计数 + 前缀和 + 散写，需原子计数 |  |
| 96 | `dense2sparse_coo_bf16_int` | ★★ 3.5 | `cusparseDenseToSparse` | bf16 | 计数 + 前缀和 + 写出（依赖 scan） |  |
