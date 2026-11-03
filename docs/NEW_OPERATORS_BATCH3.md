# 第三批新算子清单（40 个，对标 cuSPARSE 12.5）

2026-11-01 整理。在已交付的 65 个变体（`conf/operators.yaml` 的 `delivery_variants`）之外，
再列 40 个候选变体。格式沿用 [`NEW_OPERATORS_CUSPARSE_12_5.md`](NEW_OPERATORS_CUSPARSE_12_5.md)（下称"原清单"）。
与原清单不同的是，这里先按**现状**分三档（现有接口已能算对 / 小改 / 新写），每档内部再按难度排序。

## 状态（2026-11-02）

40 个已全部实现，并登记进 `conf/operators.yaml` 的 `delivery_variants`（65 → 105）。下面"清单"里的三档描述的是动手前的状况。

- **精度**：每个变体在 `tests/pytest/test_q4_variants_accuracy.py` 里都有用例，对照 CPU float64 / complex128 的参考结果。gemvi、geam 的边界情况另见 `tests/pytest/test_gemvi_geam_accuracy.py`。
- **基准**：
  - 矩阵类：`tests/q4_variant_bench.py`，由各算子自己的脚本调用。
  - 向量类：`tests/test_vector_ops.py`。
  - gemvi：`tests/test_gemvi.py`，用合成数据。
  - geam：`tests/test_geam_csr.py`。
  - spgemm 复数：由 `tests/test_spgemm.py` 的 CSV 模式追加。
- **cuSPARSE 基线**：gemvi、csrgeam2 和通用 SpGEMM 已加进 `tests/cusparse_generic_baseline.py`。
- **第二档的改法**：
  - 6 个转置混合精度变体：新增散加 kernel，按 `y[col] += v·x[row]` 原子累加，不生成 Aᵀ，也不需要预处理。
  - `spvv`、`spmm_csc`、`sddmm_csr`：增加 `out_dtype` 参数。
  - `spmm_coo`：增加 `op_b` 参数。
  - COO：开放"实矩阵 × 复向量"。
- **第三档的改法**：
  - 新算子 `flagsparse_gemvi`、`flagsparse_geam_csr`。
  - spgemm 复数：在哈希路径上新增 complex64 填充 kernel。原来走 ESC 只有 cuSPARSE 的约 0.1 倍，现在约 0.8 倍。
- **Ascend**：这 40 个都还没有加进 `ASCEND_Q4_VARIANTS`，要等真机验证。

## 范围

- **避开 BSR、Blocked-ELL、SELL 三种格式，也避开 SpSV、SpSM 两类三角求解。**
- **不含 f64 和 c64**（c64 = complex128），与原清单一致。`c32`（complex64）保留。
- **不含已交付的 65 个**，已逐个对照 `delivery_variants` 核对过。
- **不含格式转换**（`sparse2dense`、`dense2sparse`、`csr2csc` 等），全部是计算类算子。
- **收录原清单中尚未交付的 5 个计算类**：#52、#53、#61、#87、#89，备注列标为「原 #N」。
- **17 个混合精度**：输入与输出类型不同（如 `f16f32`、`i8i32`），或者实矩阵乘复向量（`f32c32`）。
- **不收录 `gather_i8_int`**（实测 cuSPARSE 12.5 和 FlagSparse 都能算对，按要求不列入，也不作备选）。
- **int64 索引不单列变体**，见"命名规则"。
- **不引入新的命名片段**：批量（`cusparse*SetStridedBatch`）和 SpGEMM 结构复用（`cusparseSpGEMMreuse_*`）需要新增后缀，暂不收录。

## 命名规则

沿用交付列表的写法：`算子_格式_数据类型_int_opA[_opB][_row|_col]`。原清单已有的片段含义不变，下表只列新增的算子名。

| 片段 | 含义 | 状态 |
|---|---|---|
| `geam` | C = αA + βB，两个稀疏矩阵相加（`cusparse<t>csrgeam2`） | 新算子名 |
| `gemvi` | y = α·op(A)·x + βy，A 稠密、x 稀疏向量（`cusparse<t>gemvi`），不带格式片段，写法同 `spvv` | 新算子名 |

**int32 和 int64 索引统一标 `int`。** 例如传 int64 索引的 SpMV 就是 `spmv_csr_f32_int_non`，与已交付的变体同名，
不算新变体；prepare 也不单独加 int64 验证。现有接口在 CUDA 上传 int64 索引已能算对
（实测 `spmv_csr`、`spmv_coo`、`spmm_csr` 的 f32）。

## 验证方法

### cuSPARSE 12.5 这一侧

本机库为 `libcusparse.so.12.5.9.5`（随 CUDA 12.9 发布，头文件 `CUSPARSE_VER_MAJOR 12` / `MINOR 5`），
GPU 为 RTX 5090。

- 每个组合都走完整流程：查缓冲区大小 → 分析或预处理（如需要）→ 计算 → 同步 GPU。
- **比原清单多了一步：核对计算结果。** 测试矩阵为 8×12，非零值都是小整数（复数时虚部也不为零，i8 和 f16 下都能精确表示），
  结果与 host 端的双精度参考值逐个比对，误差上限 1e-2；实测所有通过项的误差都是 0。
- 共试了 101 个组合（不止本清单的 40 个），97 个正确，4 个不支持（见下表）。清单 40 个和备选表里的组合全部在内，逐个算对。

### FlagSparse 这一侧

用 `src/` 里的公开接口在 RTX 5090 上调用。形状 257×129×33，稀疏矩阵密度 0.15，含一个空行。
与 CPU 端的 float64 / complex128 参考值比对，容差沿用 `tests/pytest/test_q4_variants_accuracy.py` 的 `_TOL`，
整数输出要求完全相等。

- **只在 CUDA 上测了这一个形状。** 第一档"已能算对"的结论不能直接推到其他后端：昇腾等后端有自己的分支，需要逐个上机复核。
- **注意导入路径。** 在仓库根目录直接跑 `python3 -c "import flagsparse"`，导入的是
  `~/.local/lib/python3.12/site-packages/flagsparse` 下的旧安装包，不是 `src/`。用旧包探测时，
  `out_dtype`、`op_b`、`flagsparse_spvv` 都"不存在"，约一半候选会被误判为不支持。探测脚本必须把 `src/` 插在 `sys.path` 最前面。

### 实测不支持、因此未收录的组合

| API | 不支持 |
|---|---|
| `cusparseSpMM` | 实矩阵乘复矩阵（A f32、B/C c32）。`cusparseSpMV` 支持这种组合，SpMM 不支持 |
| `cusparseSpGEMM` | 64 位索引（报错 "only 32-bit indicies are supported"） |
| `cusparseSpMV` | 数据 f32、计算类型 f64 |
| `cusparse<t>gemvi` | `CONJUGATE_TRANSPOSE`（只支持不转置和转置） |

### 接口层面的注意点

- `cusparseSpVV`、`cusparseAxpby`、`cusparse<t>gemvi` 在 12.5 里可用，头文件注释为 `// deprecated in CTK 12.8`。
- `cusparseRot` 在 12.5 头文件里**已经**标了 `CUSPARSE_DEPRECATED`，没有注明版本，所以 `rot_*` 没有收录（实测 f32/f16/c32 都能算对）。
- `cusparse<t>csrgeam2` 在 12.5 里没有弃用标记。它不属于通用 API，只有 S/D/C/Z 四种类型。

## 难度怎么算

规则和星级划分同原清单"难度怎么算"一节。另外，**CSC 转置和 CSR 不转置的访存方式一样**：按行并行，不需要原子加。所以 #4、#7、#10 按 CSR 不转置的基础分计，不再加转置分。

难度分衡量的是**从零实现**的难度。第一档的变体现有代码已经能算对，实际工作量只有登记、测试和基准，与难度分无关。

## 清单

### 一、现有接口已能算对，只需登记（24）

需要做的事：写进 `delivery_variants`、在 `tests/pytest/test_q4_variants_accuracy.py` 加用例、接入基准、补 C API。

| # | 变体 | 难度 | cuSPARSE 12.5 API | 类型：输入 → 输出（计算） | FlagSparse 调用方式 | 备注 |
|---:|---|---|---|---|---|---|
| 1 | `axpby_f32_int` | ★ 1.5 | `cusparseAxpby` | f32 → f32 (f32) | `flagsparse_axpby` | 12.8 起弃用 |
| 2 | `axpby_c32_int` | ★★ 2.5 | `cusparseAxpby` | c32 → c32 (c32) | `flagsparse_axpby` | 12.8 起弃用 |
| 3 | `spvv_f32_int_non` | ★★ 2.5 | `cusparseSpVV` | f32 → f32 (f32) | `flagsparse_spvv(op="non")` | 12.8 起弃用 |
| 4 | `spmv_csc_f32_int_trans` | ★★ 2.5 | `cusparseSpMV` | f32 → f32 (f32) | `flagsparse_spmv_csc(op="trans")` | CSC 转置 = CSR 不转置 |
| 5 | `spmv_csr_bf16f32_int_non` | ★★ 2.8 | `cusparseSpMV` | bf16 → **f32** (f32) | `flagsparse_spmv_csr(out_dtype=f32)` | 原 #87；bf16 各平台不一定支持 |
| 6 | `spmv_csr_bf16_int_non` | ★★ 3.0 | `cusparseSpMV` | bf16 → bf16 (f32) | `flagsparse_spmv_csr` | 原 #89；bf16 各平台不一定支持 |
| 7 | `spmm_csc_f32_int_trans_non_row` | ★★ 3.0 | `cusparseSpMM` | f32 → f32 (f32) | `flagsparse_spmm_csc(op="trans")` | CSC 转置 = CSR 不转置 |
| 8 | `spmm_csr_bf16f32_int_non_non_row` | ★★ 3.3 | `cusparseSpMM` | bf16 → **f32** (f32) | `flagsparse_spmm_csr(out_dtype=f32)` | bf16 各平台不一定支持 |
| 9 | `spvv_c32_int_non` | ★★ 3.5 | `cusparseSpVV` | c32 → c32 (c32) | `flagsparse_spvv(op="non")` | 12.8 起弃用；conj 版已交付 |
| 10 | `spmv_csc_c32_int_conj` | ★★ 3.5 | `cusparseSpMV` | c32 → c32 (c32) | `flagsparse_spmv_csc(op="conj")` | CSC 转置 = CSR 不转置 |
| 11 | `spmv_csc_f16f32_int_non` | ★★ 3.8 | `cusparseSpMV` | f16 → **f32** (f32) | `flagsparse_spmv_csc(out_dtype=f32)` |  |
| 12 | `spmm_csr_f16f32_int_non_non_col` | ★★ 3.8 | `cusparseSpMM` | f16 → **f32** (f32) | `flagsparse_spmm_csr(out_dtype=f32)`，B 列主序 |  |
| 13 | `spmm_coo_f16f32_int_non_non_row` | ★★★ 4.3 | `cusparseSpMM` | f16 → **f32** (f32) | `flagsparse_spmm_coo(out_dtype=f32)` |  |
| 14 | `spmm_csr_i8f32_int_non_non_row` | ★★★ 4.5 | `cusparseSpMM` | i8 → **f32** (f32) | `flagsparse_spmm_csr(out_dtype=f32)` |  |
| 15 | `spmm_coo_f32_int_non_non_col` | ★★★ 4.5 | `cusparseSpMM` | f32 → f32 (f32) | `flagsparse_spmm_coo`，B 列主序 |  |
| 16 | `spmm_csr_c32_int_non_trans_row` | ★★★ 4.5 | `cusparseSpMM` | c32 → c32 (c32) | `flagsparse_spmm_csr(op_b="trans")` | `op_b="conj"` 也已能算对 |
| 17 | `spmv_coo_i8f32_int_non` | ★★★ 5.0 | `cusparseSpMV` | i8 → **f32** (f32) | `flagsparse_spmv_coo(out_dtype=f32)` |  |
| 18 | `spmv_csc_i8i32_int_non` | ★★★ 5.0 | `cusparseSpMV` | i8 → **i32** (i32) | `flagsparse_spmv_csc(out_dtype=i32)` |  |
| 19 | `spmm_csr_i8i32_int_non_non_col` | ★★★ 5.0 | `cusparseSpMM` | i8 → **i32** (i32) | `flagsparse_spmm_csr(out_dtype=i32)`，B 列主序 |  |
| 20 | `sddmm_csr_f32_int_trans_trans_row` | ★★★★ 5.5 | `cusparseSDDMM` | f32 → f32 (f32) | `flagsparse_sddmm_csr(op_a="trans", op_b="trans")` |  |
| 21 | `spmm_coo_f32_int_trans_non_row` | ★★★★ 6.0 | `cusparseSpMM` | f32 → f32 (f32) | `flagsparse_spmm_coo(op="trans")` | 原 #52 |
| 22 | `spmm_csr_c32_int_conj_non_row` | ★★★★ 6.0 | `cusparseSpMM` | c32 → c32 (c32) | `flagsparse_spmm_csr(op="conj")` | 原 #53 |
| 23 | `sddmm_csr_c32_int_non_trans_row` | ★★★★ 6.0 | `cusparseSDDMM` | c32 → c32 (c32) | `flagsparse_sddmm_csr(op_b="trans")` |  |
| 24 | `sddmm_csr_f16_int_non_trans_row` | ★★★★ 6.0 | `cusparseSDDMM` | f16 → f16 (f32) | `flagsparse_sddmm_csr(op_b="trans")` |  |

### 二、小改：补接口参数或转置方向（10）

| # | 变体 | 难度 | cuSPARSE 12.5 API | 类型：输入 → 输出（计算） | 现在卡在哪 | 备注 |
|---:|---|---|---|---|---|---|
| 25 | `spvv_i8f32_int_non` | ★★★ 4.0 | `cusparseSpVV` | i8 → **f32** (f32) | `flagsparse_spvv` 没有 `out_dtype` 参数，i8 输入固定输出 i32 | 12.8 起弃用 |
| 26 | `spmv_csr_f16f32_int_trans` | ★★★ 4.3 | `cusparseSpMV` | f16 → **f32** (f32) | 混合精度路径只支持 `op='non'`（`NotImplementedError`） |  |
| 27 | `spmv_coo_f32c32_int_non` | ★★★ 4.5 | `cusparseSpMV` | A f32，x/y **c32** (c32) | COO 拒绝"f32 矩阵 × c32 向量"；CSR 已有这种 kernel |  |
| 28 | `spmm_coo_f32_int_non_trans_row` | ★★★ 4.5 | `cusparseSpMM` | f32 → f32 (f32) | `flagsparse_spmm_coo` 没有 `op_b` 参数；CSR 已有 |  |
| 29 | `spmm_csc_f16f32_int_non_non_row` | ★★★ 4.8 | `cusparseSpMM` | f16 → **f32** (f32) | `flagsparse_spmm_csc` 没有 `out_dtype` 参数；`mixed_spmx.py` 已有 `prepare_spmm_csc_mixed`，主要是接到公开接口上 |  |
| 30 | `spmm_csr_f16f32_int_trans_non_row` | ★★★ 4.8 | `cusparseSpMM` | f16 → **f32** (f32) | 混合精度路径只支持 `op='non'`；`op_b="trans"` 已能算对 |  |
| 31 | `spmv_csr_f32c32_int_trans` | ★★★ 5.0 | `cusparseSpMV` | A f32，x/y **c32** (c32) | 混合精度路径只支持 `op='non'` |  |
| 32 | `spmv_coo_f16_int_trans` | ★★★ 5.0 | `cusparseSpMV` | f16 → f16 (f32) | 混合精度路径只支持 `op='non'` |  |
| 33 | `spmv_csr_i8i32_int_trans` | ★★★★ 5.5 | `cusparseSpMV` | i8 → **i32** (i32) | 混合精度路径只支持 `op='non'` |  |
| 34 | `sddmm_csr_f16f32_int_non_non_row` | ★★★★ 5.5 | `cusparseSDDMM` | f16 → **f32** (f32) | 传 f32 的 `out` 被拒绝（"out dtype must match the requested output dtype"） | f16 走 `tl.dot` |

### 三、新算子或新 kernel（6）

| # | 变体 | 难度 | cuSPARSE 12.5 API | 类型：输入 → 输出（计算） | 主要难点 | 备注 |
|---:|---|---|---|---|---|---|
| 35 | `gemvi_f32_int_non` | ★★ 2.5 | `cusparseSgemvi` | f32 → f32 (f32) | 按行并行，按稀疏向量的下标取 A 的列再点积；A 为列主序（cuSPARSE 约定） | 12.8 起弃用；需先调 `gemvi_bufferSize` |
| 36 | `gemvi_f32_int_trans` | ★★ 3.0 | `cusparseSgemvi` | f32 → f32 (f32) | 同上，稠密矩阵转置：按输出元素并行，取 A 的若干行 | 12.8 起弃用 |
| 37 | `gemvi_c32_int_non` | ★★ 3.5 | `cusparseCgemvi` | c32 → c32 (c32) | 同 `gemvi_f32_int_non`；复数（寄存器翻倍） | 12.8 起弃用；不支持 conj |
| 38 | `geam_csr_f32_int` | ★★ 3.5 | `cusparseScsrgeam2` | f32 → f32 (f32) | 两阶段：逐行合并计数 + 前缀和，再归并写出 | 不属于通用 API，12.5 未弃用 |
| 39 | `geam_csr_c32_int` | ★★★ 4.5 | `cusparseCcsrgeam2` | c32 → c32 (c32) | 同 `geam_csr_f32_int`；复数（寄存器翻倍） | 可与 f32 版共用 kernel |
| 40 | `spgemm_csr_c32_int_non_non` | ★★★★★ 8.0 | `cusparseSpGEMM` (ALG1) | c32 → c32 (c32) | 两阶段（符号+数值），哈希表依赖共享内存；复数（寄存器翻倍）；现在只接受 f32/f64 | 原 #61 |

## 备选

以下组合都已在 cuSPARSE 12.5 上核实能算对，可以替换上面的任一项。

| 变体 | 现状 | 备注 |
|---|---|---|
| `spmv_csc_i8f32_int_non` | 已能算对 |  |
| `spmv_csr_f16_int_trans` | 已能算对 |  |
| `spmm_coo_i8f32_int_non_non_row` | 已能算对 |  |
| `spmm_csr_i8i32_int_non_trans_row` | 已能算对 |  |
| `spmm_csr_f16f32_int_non_trans_row` | 已能算对 |  |
| `spmm_csc_f32_int_non_non_col` | 已能算对 |  |
| `spmm_csr_c32_int_non_conj_row` | 已能算对 |  |
| `spmm_csr_c32_int_non_non_col` | 已能算对 |  |
| `spmm_coo_c32_int_conj_non_row` | 已能算对 |  |
| `spmm_csr_f32_int_trans_trans_row` | 已能算对 |  |
| `sddmm_csr_c32_int_trans_non_row` | 已能算对 |  |
| `spvv_bf16f32_int_non` | 已能算对 | 12.8 起弃用 |
| `spmv_coo_bf16f32_int_non` | 已能算对 |  |
| `spmv_coo_bf16_int_non` | 已能算对 | 原 #91 |
| `spmm_csr_bf16_int_non_non_row` | 已能算对 | 原 #90 |
| `axpby_bf16_int` | 已能算对 |  |
| `gather_bf16_int` | 已能算对 | 原 #85 |
| `spmv_coo_f16f32_int_trans` | 小改 | 混合精度路径只支持 `op='non'` |
| `spmv_csr_i8f32_int_trans` | 小改 | 混合精度路径只支持 `op='non'` |
| `spmm_csr_i8i32_int_trans_non_row` | 小改 | 混合精度路径只支持 `op='non'` |
| `spmm_csc_i8i32_int_non_non_row` | 小改 | `flagsparse_spmm_csc` 没有 `out_dtype` 参数 |
| `spmv_csc_f32c32_int_non` | 小改 | CSC 拒绝"f32 矩阵 × c32 向量" |
| `spgemm_csr_f32_int_non_non_alg2` / `_alg3` | 需新写 | 原 #62/#63；对 FlagSparse 来说输出与 ALG1 相同，价值偏低 |
