# 摩尔线程 MUSA（S5000）debug

2026-10-06：MUSA C API 已完成 q4 全部 45 变体的 10 个真实 MatrixMarket 矩阵复测。
当前权威数据、精度状态与复现命令见下一节；10 月 5 日合成矩阵结果保留为历史快照。

`FLAGSPARSE_BACKEND=mthreads`。环境、交付复现见 [../MUSA.md](../MUSA.md)。

## 0. 当前入口：MUSA C API

MUSA 的 q4 调试以 C API CTest 为准。重新配置后运行：

```bash
cmake -S capi -B capi/build-musa -G Ninja \
  -DBACKEND=MUSA -DMUSA_HOME=/usr/local/musa \
  -DCMAKE_BUILD_TYPE=Release
cmake --build capi/build-musa -j

# C API 精度
ctest --test-dir capi/build-musa -L capi -R accuracy --output-on-failure

# C API 性能
FLAGSPARSE_MATRIX_DIR=/path/to/mtx \
FLAGSPARSE_BENCH_OUT=./capi/bench-musa \
  ctest --test-dir capi/build-musa -L capi -R benchmark --output-on-failure
```

交付报告需要合并 Python 精度与 muSPARSE 性能时，使用
`run_flagsparse_split_delivery.py`；它不替代上述 C API 全量调试入口。

### 2026-10-06：45 变体、10 个真实矩阵完整复测（当前结果）

本轮在 MTT S5000（MUSA arch 31、muSPARSE 4.3.5）完成。输入固定为
`capi/bench-musa-real-corpus/` 的 10 个 MatrixMarket 文件：ASIC_680ks、GL7d14、
NACA0015、amazon0601、auto、cage12、cfd2、filter3D、roadNet-TX、wave。
该目录特意不含 `tests/data/q4_worker_smoke.mtx`，后者只用于 SpGEMM runner smoke，
不得计入性能平均。

完整性能轮次 **7/7 通过，1078.74 秒**；warmup=10、iters=100，取同步墙钟时间中位数。
45/45 q4 变体都实际进入 C API benchmark，共 464 条 q4 测量行：445 strict pass、
5 pass_relaxed、10 fail、4 unchecked。`pass_relaxed` 和厂商基线未严格通过的行不计入
下表加速比。修复 SpGEMM 空结果行指针后，C API 独立 accuracy 已为 **7/7 测试族通过**；
性能轮次的 SpGEMM 也通过。

加速比定义为 `muSPARSE_ms / FlagSparse_ms`；大于 1 表示 FlagSparse 更快。P/R/F/U 分别为
FlagSparse 严格通过/放宽通过/失败/未校验的行数。加速比只对 FlagSparse 与 muSPARSE **双方
严格通过**的行取算术平均；N/A 表示本轮没有可比的厂商基线，并不表示未执行。

| 变体 | P/R/F/U | 双方严格平均（行数） |
|---|---:|---:|
| `axpby_f16_int` | 1/0/0/0 | N/A |
| `scatter_i8_int` | 10/0/0/0 | N/A |
| `sddmm_csr_c32_int_non_non_row` | 20/0/0/0 | 13.546x (20) |
| `sddmm_csr_f16_int_non_non_row` | 20/0/0/0 | N/A |
| `sddmm_csr_f32_int_non_non_col` | 20/0/0/0 | 35.218x (20) |
| `sddmm_csr_f32_int_non_trans_row` | 20/0/0/0 | 34.687x (20) |
| `sddmm_csr_f32_int_trans_non_row` | 20/0/0/0 | 22.732x (20) |
| `spgemm_csr_f32_int_non_non` | 6/0/0/4 | 0.669x (5) |
| `spmm_coo_c32_int_non_non_row` | 10/0/0/0 | 0.944x (10) |
| `spmm_coo_f16_int_non_non_row` | 9/0/1/0 | N/A |
| `spmm_coo_i8i32_int_non_non_row` | 10/0/0/0 | N/A |
| `spmm_csc_c32_int_non_non_row` | 10/0/0/0 | 3.528x (10) |
| `spmm_csc_f16_int_non_non_row` | 9/0/1/0 | N/A |
| `spmm_csc_f32_int_non_non_row` | 10/0/0/0 | 2.127x (9) |
| `spmm_csr_c32_int_non_non_row` | 10/0/0/0 | 1.071x (10) |
| `spmm_csr_f16_int_non_non_row` | 9/0/1/0 | N/A |
| `spmm_csr_f16f32_int_non_non_row` | 9/0/1/0 | N/A |
| `spmm_csr_f32_int_non_non_col` | 10/0/0/0 | 0.887x (10) |
| `spmm_csr_f32_int_non_trans_row` | 10/0/0/0 | 0.785x (10) |
| `spmm_csr_f32_int_trans_non_row` | 9/1/0/0 | 1.725x (8) |
| `spmm_csr_i8i32_int_non_non_row` | 10/0/0/0 | N/A |
| `spmv_coo_c32_int_conj` | 9/1/0/0 | 0.978x (9) |
| `spmv_coo_c32_int_non` | 10/0/0/0 | 2.585x (10) |
| `spmv_coo_f16_int_non` | 9/0/1/0 | N/A |
| `spmv_coo_f16f32_int_non` | 9/0/1/0 | N/A |
| `spmv_coo_f32_int_trans` | 9/1/0/0 | 0.877x (9) |
| `spmv_coo_i8i32_int_non` | 10/0/0/0 | N/A |
| `spmv_csc_c32_int_non` | 10/0/0/0 | 0.904x (10) |
| `spmv_csc_f16_int_non` | 9/0/1/0 | N/A |
| `spmv_csc_f32_int_non` | 10/0/0/0 | 1.009x (10) |
| `spmv_csr_c32_int_conj` | 9/1/0/0 | 0.905x (9) |
| `spmv_csr_c32_int_non` | 10/0/0/0 | 0.924x (10) |
| `spmv_csr_f16_int_non` | 9/0/1/0 | N/A |
| `spmv_csr_f16f32_int_non` | 9/0/1/0 | N/A |
| `spmv_csr_f32_int_trans` | 9/1/0/0 | 0.977x (9) |
| `spmv_csr_f32c32_int_non` | 10/0/0/0 | N/A |
| `spmv_csr_i8f32_int_non` | 10/0/0/0 | N/A |
| `spmv_csr_i8i32_int_non` | 10/0/0/0 | N/A |
| `spmv_sell_c32_int_non` | 10/0/0/0 | N/A |
| `spmv_sell_f16_int_non` | 9/0/1/0 | N/A |
| `spmv_sell_f32_int_non` | 10/0/0/0 | N/A |
| `spmv_sell_i8i32_int_non` | 10/0/0/0 | N/A |
| `spvv_c32_int_conj` | 1/0/0/0 | 0.686x (1) |
| `spvv_f16f32_int_non` | 1/0/0/0 | N/A |
| `spvv_i8i32_int_non` | 1/0/0/0 | N/A |

此前 8 项 MUSA gather 优化的真实矩阵提升已纳入上表：CSC f32 SpMV 0.376x -> 1.009x、
CSR c32 SpMV 0.142x -> 0.924x、CSR f32 transpose SpMV 0.386x -> 0.977x、CSC c32 SpMV
0.396x -> 0.904x；CSR c32 SpMM 0.203x -> 1.071x、CSC f32 SpMM 0.017x -> 2.127x、CSR f32
transpose SpMM 0.541x -> 1.725x、CSC c32 SpMM 0.017x -> 3.528x。前后均为双方严格通过行的
算术平均；前后独立运行，微小波动属正常。

原始结果在 [最终汇总](../../capi/bench-musa-real-q4-real10-20261006/summary_q4.json)、
[SpMV](../../capi/bench-musa-real-q4-real10-20261006/spmv_benchmark.json)、
[SpMM](../../capi/bench-musa-real-q4-real10-20261006/spmm_benchmark.json) 及同目录的其余
`*_benchmark.json` / `*_accuracy.json`。所有 benchmark JSON 已用 `python -m json.tool`
验证。报告器现将非有限误差写成 JSON `null`，避免此前 `inf` 令
`write_summary_q4.py` 解析失败；失败状态和错误详情仍保留。

复现：

```bash
cmake --build capi/build-musa -j16
export FLAGSPARSE_BACKEND=mthreads MUSA_HOME=/usr/local/musa
export FLAGSPARSE_MATRIX_DIR="$PWD/capi/bench-musa-real-corpus"
export FLAGSPARSE_BENCH_OUT="$PWD/capi/bench-musa-real-q4-retest"
ctest --test-dir capi/build-musa -L capi \
  -R '^benchmark\\.(axpby|spmv|spmm|spvv|spgemm|sddmm|scatter)$' --output-on-failure
python3 capi/tools/write_summary_q4.py \
  --bench-dir "$FLAGSPARSE_BENCH_OUT" --out "$FLAGSPARSE_BENCH_OUT"
```

### 2026-10-05：合成矩阵历史记录

以下命令均在仓库根目录执行。当前本机为 MTT S5000，torch / torch_musa
2.7.1、Triton 3.6.0，环境检查返回 `mthreads musa None`。C API 已在
`capi/build-musa` 配置并编译成功，基线找到 `/usr/local/musa/lib/libmusparse.so`。

```bash
export FLAGSPARSE_BACKEND=mthreads MUSA_HOME=/usr/local/musa
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export FLAGSPARSE_BENCH_OUT="$PWD/capi/bench-musa-local"
mkdir -p "$FLAGSPARSE_BENCH_OUT"

# 45 个变体 × 两个规模，加一项清单检查。
python3 -m pytest tests/pytest/test_q4_variants_accuracy.py -v \
  --record json --output "$FLAGSPARSE_BENCH_OUT/q4_accuracy_result.json"

cmake -S capi -B capi/build-musa -G Ninja \
  -DBACKEND=MUSA -DMUSA_HOME="$MUSA_HOME" \
  -DCMAKE_BUILD_TYPE=Release -DFLAGSPARSE_CTEST_TIMEOUT=3600
cmake --build capi/build-musa -j 16

# q4 涉及的七个 C API 算子族；每个族内部覆盖多个变体。
ctest --test-dir capi/build-musa -L capi \
  -R '^accuracy\.(axpby|spmv|spmm|spvv|spgemm|sddmm|scatter)$' --output-on-failure

# 性能：先用默认合成矩阵检查覆盖；真实语料则设置为 "$PWD/tests/data"。
unset FLAGSPARSE_MATRIX_DIR
ctest --test-dir capi/build-musa -L capi \
  -R '^benchmark\.(axpby|spmv|spmm|spvv|spgemm|sddmm|scatter)$' --output-on-failure
python3 capi/tools/write_summary_q4.py \
  --bench-dir "$FLAGSPARSE_BENCH_OUT" --out "$FLAGSPARSE_BENCH_OUT"
```

输出目录用绝对路径，避免 CTest 切换工作目录后将 JSON 写到意外位置。
`run_flagsparse_split_delivery.py` 是 20 变体交付入口，不能代替这里的 45 变体检查。

本次精度验证：

- Python 精度 **91 passed，50.22 秒**。SpGEMM 测试原先在 MUSA 上展开 CSR
  结果，因缺少 `aten::_to_dense` 而失败；现将算子输出搬到 CPU 再展开比较，
  算子本身仍在 MUSA 执行。
- C API 七个相关精度测试族 **6 passed、1 failed**。SpGEMM 的 `Float32` 等
  四项通过，`EmptyRowsAndEmptyResult` 失败：`empty_a` 的输出行指针未初始化。
  `capi/src/ops/spgemm.cpp` 的 copy 路径在 `C->nnz == 0` 时直接返回，未复制
  行指针；这是独立的 C API 边界问题，不能将本次结果描述成 C API 全通过。
- 原始记录在 `capi/bench-musa-local/python-accuracy.log`、
  `q4_accuracy_result.json` 和 `capi-accuracy.log`。

本次性能验证（2026-10-05，默认合成矩阵）：

- 七个 benchmark 测试族串行运行结束，CTest **7/7 通过，1439.34 秒**。
  预热 10 次、计时 100 次，取包含设备同步的墙钟时间中位数；首次 JIT 与矩阵
  生成在计时之外。合成矩阵为 8192、32768、131072 阶；向量算子使用各自的小用例。
  当前生成器逐元素扫描，且各进程重复生成，是整轮约 24 分钟的主要准备开销。
- **45/45 变体有有效性能行**。`summary_q4.json` 为 8 `Passed`、37 `Measured`：
  前者有可用 muSPARSE 加速比，后者只有自身耗时。此状态只要求至少一行有效，
  **不能解释为 45 个变体的全部用例精度通过**。
- q4 共 **142 行**：111 严格精度通过、7 放宽精度通过、23 精度失败、1 未校验。
  失败分布于 SpMV 的 7 个变体（9 行）和 SpMM 的 7 个变体（14 行）。SDDMM
  五个 q4 变体共 30 行全部通过，但当前测试框架未接入相应 op/order 厂商基线。
- SpGEMM 的最大规模超过检查预算，标为 `unchecked` 且不提供加速比。SpGEMM
  只计 `compute`，不包含 `copy` 中的 MUSA CPU 结果生成，不能视为端到端耗时。
- 产物在 `capi/bench-musa-q4-20261005T154023Z/`：`REPORT.md` 为 45 变体表，
  `q4_rows.csv` 保留包括 K 值在内的逐行参数和失败原因，`q4_variants.csv` 为变体汇总，
  另有原始 `*_benchmark.json`、`summary_q4.json`、`ctest.log`、`ctest-detail.log`
  和 `run_metadata.json`。这是合成性能验证，尚未运行真实 MatrixMarket 语料性能测试。

### 2026-10-05：muSPARSE 基线扩展（仅 MUSA）

`FLAGSPARSE_MUSA_BASELINE_EXTENSIONS` 只在 `BACKEND=MUSA` 的 CTest 构建中定义；
其他后端保留原有 benchmark 分支；共享结构和 Python 测试的影响见下方范围说明。扩展了 13 个此前缺基线的 f32/c32
候选，4 个测试族在默认合成矩阵上 **4/4 通过，839.06 秒**。

- 已获得有效基线的 13 个变体：`spvv_c32_int_conj`；
  `spmv_csc_f32_int_non`、`spmv_csc_c32_int_non`；
  `spmm_csr_f32_int_non_trans_row`、`spmm_csr_c32_int_non_non_row`、
  `spmm_coo_c32_int_non_non_row`、`spmm_csr_f32_int_trans_non_row`；以及
  `spmm_csc_f32_int_non_non_row`、`spmm_csc_c32_int_non_non_row`；以及
  `sddmm_csr_f32_int_non_non_col`、`sddmm_csr_f32_int_non_trans_row`、
  `sddmm_csr_f32_int_trans_non_row`、`sddmm_csr_c32_int_non_non_row`。
- 原生 CSC descriptor 在 muSPARSE setup / buffer-size 阶段返回 status 2；实测后采用
  CSC(A) 与 CSR(A^T) 的等价表示，把 CSC 缓冲区作为 `A^T` 的 CSR 并翻转 opA；这不引入
  格式转换或额外计时。本文件将其说明为 CSC 的 CSR-transpose 回退基线，原始 JSON 尚无独立标记。
- SDDMM 的四个变体在三种矩阵规模、K=16/64 的 24 行均通过 FlagSparse 和 muSPARSE
  的严格精度检查；SpVV c32 也通过。SpMM 的四个新路径可运行；较大两种规模按既有
  规则为 `pass_relaxed`，因为两边都未过严格 CPU oracle。
- 首轮四族结果位于 `capi/bench-musa-q4-baseline-20261005T161300Z/`，包括四个原始
  `*_benchmark.json` 与 `ctest.log`；随后修复 COO descriptor 兼容性后，SpMM 在
  `capi/bench-musa-q4-baseline-20261005T164200Z/` 复跑并通过，后者是 COO SpMM c32
  基线的中间记录。CSC 回退在 `capi/bench-musa-q4-baseline-20261005T205000Z/` 的 SpMV/
  SpMM 重跑中 **2/2 通过，415.51 秒**。这些轮次都只重跑受影响的算子族；其余结果仍以
  上述完整 45 变体产物为准。

### 2026-10-05：历史合并结果与打包说明

本节按算子族选取最后一次有效复跑，未重新执行整套测试。前文首轮统计保留作历史对照。

| 算子族 | 最新原始结果目录（仓库根目录相对路径） |
|---|---|
| SpMV / SpMM | `capi/bench-musa-q4-baseline-20261005T205000Z` |
| SpVV / SDDMM | `capi/bench-musa-q4-baseline-20261005T161300Z` |
| AXPBY / Scatter / SpGEMM | `capi/bench-musa-q4-20261005T154023Z` |

Python 精度为 **91 passed、1 warning，50.22 秒**，覆盖 45 变体的两组规模
`64×48×16`、`257×129×33` 及清单检查。C API 独立精度测试仍为 **6/7 测试族通过**；
SpGEMM 空结果行指针问题尚未修复，基线扩展后没有重跑该独立精度测试。

最新性能结果共 **45 变体、142 行：114 pass、20 pass_relaxed、7 fail、1 unchecked**。
**28 个变体全部行严格通过，5 个变体包含失败行；21 个变体有加速比，24 个没有。**
CTest 通过只表示测试进程通过，不代表其中每行精度通过。

测试使用合成方阵：8192 阶、密度 0.001；32768 阶、密度 0.0005；131072 阶、密度 0.0001。
SpMM 的稠密输出列数为 8；SDDMM 的 K 为 16/64，每变体 6 行；AXPBY/SpVV 使用长度 4、
稀疏 nnz=3 的小用例，各 1 行；Scatter 各 3 行。该历史批次尚无真实 MatrixMarket 语料性能结论。
预热 10 次、测量 100 次，取含同步的墙钟中位数；不包含首次 JIT 和矩阵生成。

下表 P/R/F/U 分别为严格通过、放宽通过、失败、未校验的行数。严格通过指现有测试器的
dtype 对应容差，不代表统一的高精度阈值。误差比例为 `max(|actual-ref|/(atol+rtol*|ref|))`，
不超过 1 为通过；`pass_relaxed` 表示两边均未过严格检查而 FlagSparse 通过放宽检查。
加速比为 `muSPARSE_ms / FlagSparse_ms`，大于 1 表示 FlagSparse 较快。
“已有比值均值”对 JSON 中非空比值作算术平均，**可能包含 muSPARSE 严格精度失败的行**；
“双方严格均值”只保留两边均为 `pass` 的行。括号为参与平均的行数，N/A 不表示没执行算子。

| 变体 | P/R/F/U | 已有比值均值（行数） | 双方严格均值（行数） |
|---|---:|---:|---:|
| `axpby_f16_int` | 1/0/0/0 | N/A | N/A |
| `scatter_i8_int` | 3/0/0/0 | N/A | N/A |
| `sddmm_csr_c32_int_non_non_row` | 6/0/0/0 | 4.022× (6) | 4.022× (6) |
| `sddmm_csr_f16_int_non_non_row` | 6/0/0/0 | N/A | N/A |
| `sddmm_csr_f32_int_non_non_col` | 6/0/0/0 | 18.834× (6) | 18.834× (6) |
| `sddmm_csr_f32_int_non_trans_row` | 6/0/0/0 | 18.298× (6) | 18.298× (6) |
| `sddmm_csr_f32_int_trans_non_row` | 6/0/0/0 | 6.881× (6) | 6.881× (6) |
| `spgemm_csr_f32_int_non_non` | 0/2/0/1 | 1.746× (2) | N/A |
| `spmm_coo_c32_int_non_non_row` | 1/2/0/0 | 0.945× (3) | 0.998× (1) |
| `spmm_coo_f16_int_non_non_row` | 3/0/0/0 | N/A | N/A |
| `spmm_coo_i8i32_int_non_non_row` | 3/0/0/0 | N/A | N/A |
| `spmm_csc_c32_int_non_non_row` | 1/2/0/0 | 0.032× (3) | 0.054× (1) |
| `spmm_csc_f16_int_non_non_row` | 1/0/2/0 | N/A | N/A |
| `spmm_csc_f32_int_non_non_row` | 1/2/0/0 | 0.041× (3) | 0.079× (1) |
| `spmm_csr_c32_int_non_non_row` | 1/2/0/0 | 0.355× (3) | 0.569× (1) |
| `spmm_csr_f16_int_non_non_row` | 3/0/0/0 | N/A | N/A |
| `spmm_csr_f16f32_int_non_non_row` | 3/0/0/0 | N/A | N/A |
| `spmm_csr_f32_int_non_non_col` | 1/2/0/0 | 0.903× (3) | 0.858× (1) |
| `spmm_csr_f32_int_non_trans_row` | 1/2/0/0 | 0.910× (3) | 0.846× (1) |
| `spmm_csr_f32_int_trans_non_row` | 1/2/0/0 | 0.676× (3) | 0.894× (1) |
| `spmm_csr_i8i32_int_non_non_row` | 3/0/0/0 | N/A | N/A |
| `spmv_coo_c32_int_conj` | 2/1/0/0 | 0.907× (3) | 0.884× (1) |
| `spmv_coo_c32_int_non` | 2/0/1/0 | 4.174× (2) | 4.174× (2) |
| `spmv_coo_f16_int_non` | 3/0/0/0 | N/A | N/A |
| `spmv_coo_f16f32_int_non` | 3/0/0/0 | N/A | N/A |
| `spmv_coo_f32_int_trans` | 2/1/0/0 | 0.914× (3) | 0.906× (2) |
| `spmv_coo_i8i32_int_non` | 3/0/0/0 | N/A | N/A |
| `spmv_csc_c32_int_non` | 3/0/0/0 | 0.724× (3) | 0.868× (2) |
| `spmv_csc_f16_int_non` | 2/0/1/0 | N/A | N/A |
| `spmv_csc_f32_int_non` | 3/0/0/0 | 0.786× (3) | 0.953× (2) |
| `spmv_csr_c32_int_conj` | 1/0/2/0 | 0.967× (1) | 0.967× (1) |
| `spmv_csr_c32_int_non` | 2/1/0/0 | 0.497× (3) | 0.514× (2) |
| `spmv_csr_f16_int_non` | 3/0/0/0 | N/A | N/A |
| `spmv_csr_f16f32_int_non` | 3/0/0/0 | N/A | N/A |
| `spmv_csr_f32_int_trans` | 2/1/0/0 | 0.781× (3) | 0.951× (2) |
| `spmv_csr_f32c32_int_non` | 2/0/1/0 | N/A | N/A |
| `spmv_csr_i8f32_int_non` | 3/0/0/0 | N/A | N/A |
| `spmv_csr_i8i32_int_non` | 3/0/0/0 | N/A | N/A |
| `spmv_sell_c32_int_non` | 3/0/0/0 | N/A | N/A |
| `spmv_sell_f16_int_non` | 3/0/0/0 | N/A | N/A |
| `spmv_sell_f32_int_non` | 3/0/0/0 | N/A | N/A |
| `spmv_sell_i8i32_int_non` | 3/0/0/0 | N/A | N/A |
| `spvv_c32_int_conj` | 1/0/0/0 | 1.043× (1) | 1.043× (1) |
| `spvv_f16f32_int_non` | 1/0/0/0 | N/A | N/A |
| `spvv_i8i32_int_non` | 1/0/0/0 | N/A | N/A |

失败行明细（原始误差比例保留在 JSON 中）：

| 变体 | 矩阵 | 严格误差比例 | 放宽误差比例 |
|---|---|---:|---:|
| `spmv_csr_f32c32_int_non` | synthetic_32k_d0.0005 | 1.20418 | 0.0120418 |
| `spmv_coo_c32_int_non` | synthetic_32k_d0.0005 | 1.44216 | 0.0144216 |
| `spmv_csc_f16_int_non` | synthetic_32k_d0.0005 | 1.10425 | 117.34 |
| `spmv_csr_c32_int_conj` | synthetic_32k_d0.0005 | 1.22078 | 0.0122078 |
| `spmv_csr_c32_int_conj` | synthetic_128k_d0.0001 | 1.59704 | 0.0159704 |
| `spmm_csc_f16_int_non_non_row` | synthetic_32k_d0.0005 | 1.03109 | 112.303 |
| `spmm_csc_f16_int_non_non_row` | synthetic_128k_d0.0001 | 1.23457 | 143.603 |

无比值变体的原始基线原因：

| 变体 | 原始 baseline_detail / detail |
|---|---|
| `axpby_f16_int` | no matching cuSPARSE Axpby baseline in harness |
| `scatter_i8_int` | muSPARSE: Scatter dtype unsupported |
| `sddmm_csr_f16_int_non_non_row` | muSPARSE: unsupported SDDMM dtype or operation |
| `spmm_coo_f16_int_non_non_row` | muSPARSE: unsupported SpMM type or operation |
| `spmm_coo_i8i32_int_non_non_row` | mixed-precision SpMM has no matching vendor baseline |
| `spmm_csc_f16_int_non_non_row` | muSPARSE: unsupported SpMM type or operation |
| `spmm_csr_f16_int_non_non_row` | muSPARSE: unsupported SpMM type or operation |
| `spmm_csr_f16f32_int_non_non_row` | mixed-precision SpMM has no matching vendor baseline |
| `spmm_csr_i8i32_int_non_non_row` | mixed-precision SpMM has no matching vendor baseline |
| `spmv_coo_f16_int_non` | muSPARSE: unsupported SpMV type or operation |
| `spmv_coo_f16f32_int_non` | mixed-precision SpMV has no matching vendor baseline |
| `spmv_coo_i8i32_int_non` | mixed-precision SpMV has no matching vendor baseline |
| `spmv_csc_f16_int_non` | muSPARSE: unsupported SpMV type or operation |
| `spmv_csr_f16_int_non` | muSPARSE: unsupported SpMV type or operation |
| `spmv_csr_f16f32_int_non` | mixed-precision SpMV has no matching vendor baseline |
| `spmv_csr_f32c32_int_non` | mixed-precision SpMV has no matching vendor baseline |
| `spmv_csr_i8f32_int_non` | mixed-precision SpMV has no matching vendor baseline |
| `spmv_csr_i8i32_int_non` | mixed-precision SpMV has no matching vendor baseline |
| `spmv_sell_c32_int_non` | no matching cuSPARSE SELL SpMV baseline |
| `spmv_sell_f16_int_non` | no matching cuSPARSE SELL SpMV baseline |
| `spmv_sell_f32_int_non` | no matching cuSPARSE SELL SpMV baseline |
| `spmv_sell_i8i32_int_non` | no matching cuSPARSE SELL SpMV baseline |
| `spvv_f16f32_int_non` | muSPARSE: unsupported SpVV dtype or operation |
| `spvv_i8i32_int_non` | muSPARSE: unsupported SpVV dtype or operation |

已补齐的 13 个基线变体均至少有一行可用比值，但不能据此认为所有规模精度合格。
CSC f32/c32 使用 CSC(A)=CSR(Aᵀ) 加翻转 opA 的 muSPARSE 回退；原始 JSON 没有单独标注回退类型。
原生 CSC 返回 status 2（NOT_IMPLEMENTED）。SpGEMM 最大用例因结果 nnz 超过 2000 万检查预算
而为 unchecked；其计时仅含 compute，不含在 MUSA copy 阶段的 CPU 结果生成，不能当作端到端加速比。

轮次记录：完整七族 1439.34 秒；首次基线四族 839.06 秒；COO 修正后 SpMM 218.29 秒；
最终 CSC 回退 SpMV/SpMM 415.51 秒。`20261005T160900Z` 为中断尝试；
`20261005T164200Z` 的 SpMM 已被 `20261005T205000Z` 替代。各轮结果保留，不混作一次完整运行。

修改范围核对：新增基线分支由 MUSA 宏控制，未修改算子 kernel；但共享 `DeviceCsr` 的
`Format` 字段以及 Python SpGEMM 的 CPU 展开目前没有 MUSA 条件保护。因此当前补丁不能声称
所有共享代码改动仅影响 MUSA；其他后端未在本机回归验证。

归档包含相对本地 `origin/q4`（`031496a`）修改的 9 个文件，以及本次所有 `capi/bench-musa-*`
结果目录（含中断尝试）、已有的两轮 `results_musa_q4_20260928*` 历史结果和可用的 CTest Testing 日志。
归档中的 `ARCHIVE_MANIFEST.txt` 列出文件路径与 SHA-256；不包含编译产物和整个源码仓库。
原始各轮 REPORT/CSV/summary 是该轮快照，最新合并口径以本节及上面的源目录映射为准。

## 1. 这个后端的特点（实测于 torch_musa 2.7.1 / muDNN v3105）

| 能力 | fp32 | fp64 | complex64 | complex128 |
|---|---|---|---|---|
| Triton（含 `atomic_add`、`associative_scan`） | 可用 | 可用 | 可用 | 可用 |
| `where`（三元） | 可用 | 不可用 | 不可用 | 不可用 |
| `sum` | 可用 | 可用 | 不可用 | 不可用 |
| 2 维 × 1 维 `A @ x`（gemv） | 可用 | 不可用 | 不可用 | 不可用 |
| 2 维 × 2 维 `A @ B`（gemm） | 可用 | 可用 | 可用 | 可用 |
| `torch.sparse` 的矩阵乘（CSR 和 COO） | **不可用** | 不可用 | 不可用 | 不可用 |

- **复数高级索引没有 kernel**：`values[order]` 会报 `"IndexMusa" not implemented for 'ComplexFloat'`。
  已统一改走 `_common._gather_values`（按 `view_as_real` 拆成实数索引）。
- 因为 `torch.sparse` 不能做矩阵乘，MUSA 上没有 PyTorch 稀疏基线；精度测试的参考值在 CPU 上算。

## 2. 已知问题

| 问题 | 状态 |
|---|---|
| muDNN 的 gemv 不支持 fp64 / 复数（gemm 支持） | 厂商侧问题，可以作为最小复现报给厂商 |
| `torch.sparse` 无矩阵乘，`_mthreads_vendor_sparse_library()` 默认返回 `None` | 基线列为 N/A 并写明原因 |
| `index_add_` / `scatter_add_` 的复数版本同样可能缺 | spgemm / spsm 只支持实数，目前走不到 |

## 3. q4 变体的风险点

> ⚠️ 2026-10-05：q4 清单已从 42 条改成 45 条（删 3 加 6，见 [README.md](README.md) 第 0 节）。
> 下面这节和第 4 节 2026-09-28/29 的真机结果都是**旧 42 条清单**的内容，表格里的
> `gather_i8_int`、`spmm_bell_f32_int_non_non_row`、`spmm_bsr_f32_int_non_non_row` 现在已经
> 移出 q4 统计范围（MUSA 上测过，精度 PASS，性能部分见第 4 节原有的 Failed/NotFound 记录）。
> 以下风险描述为本轮测试前的历史判断；新增的 6 个变体现已在 MUSA 测试，结果见第 0 节。
> 新增项为 `sddmm_csr_f16/c32_int_non_non_row`、`spmm_csc_c32/f16_int_non_non_row`、
> `spmm_coo_i8i32_int_non_non_row`、`spgemm_csr_f32_int_non_non`。其中 CSC 路径曾因 MUSA 的
> "复数高级索引没有 kernel"限制，而 `spmm_csc` 的 kernel 直接 `tl.atomic_add` 进输出 dtype
> 缓冲区（没有 ACC_DTYPE 累加层），f16 变体新写的 kernel 用了 fp32 累加缓冲区再转回 f16，这条
> 路径现已实测；CSC f16 的较大用例仍有精度失败，详见本轮失败明细。
>
> 另外，这轮 capi（C API 层）在 `capi/src/ops/spmv.cpp` 新增了 `spmv_csr`/`spmv_coo` 的混合精度
> dispatch（int8→int32/float32、fp16→float32），本文件第 0 节描述的 **MUSA C API CTest 入口**
> 从未跑过这部分新代码——如果要给 MUSA 的 C API 结果加上混合精度变体，需要先确认 MUSA 这边的
> `capi/build-musa` 配置能不能编译这次新加的 `capi/flagsparse_codegen/mixed_spmx.py`
> （未提交进仓库前是 `??` 状态的新文件，拉取时确认它在）。

- q4 的复数路径全部拆成实部 / 虚部平面，不用复数索引、复数 `sum`：`spvv_c32_int_conj`、`spmv_sell_c32` 等应当可用，需实测确认。
- **PyTorch 基线**：q4 benchmark 的 PyTorch 列用 `torch.sparse` 做矩阵乘，在 MUSA 上会失败，
  这一列会写 `pytorch_reason`；cuSPARSE 列也为空。于是这些行**没有任何加速比**，只能用
  `tools/baseline_bound.py --vendor-card musa-s5000`（1370 GB/s）按 H800 换算判定。
- 精度测试的参考值在 CPU 上算，不受上面的限制。

## 4. 2026-09-28/29 真机结果（S5000 x1）

本节是 q4 分支首次在 MUSA 真机上的结果，不是 CUDA 模拟结果。环境自检为
`FLAGSPARSE_BACKEND=mthreads`、加速器 `musa`、`fallback=None`；`mthreads-gmi 2.3.2`
报告驱动 `3.3.5-server`。测试开始和结束时 GPU 均无其他进程。

原始产物在 `results_musa_q4_20260928T145913Z/`：

- 精度记录：`q4_accuracy.log`、`q4_accuracy_result.json`；
- 性能汇总：`performance/summary.json`、`performance/summary.csv`；
- 每个父算子的原始性能 CSV、stdout、stderr 位于 `performance/<op>/`。

### 精度

执行命令：

```bash
FLAGSPARSE_BACKEND=mthreads PYTHONPATH=src \
  python3 -u -m pytest tests/pytest/test_q4_variants_accuracy.py -v \
  --record json --output results_musa_q4_20260928T145913Z/q4_accuracy_result.json
```

**85 passed, 1 warning, 68.90 s**。其中 1 项检查清单确有 42 个不重复变体，余下
84 项为每个变体的两个规模（`64x48x16`、`257x129x33`）。因此 **42/42 q4 变体在两个
规模上均与 CPU golden 一致**，包括全部 complex64、fp16 和 int8 路径。唯一 warning 是
PyTorch 的 CSR beta 提示，不影响结果。

### 性能

以下性能命令是历史 Python 侧记录，目的只是记录 FlagSparse 的 MUSA 自身耗时；它不是
当前 MUSA C API 调试入口。MUSA 交付报告的 muSPARSE 对比应使用
`run_flagsparse_split_delivery.py`。命令如下，`tests/data` 中的 10 个 MatrixMarket 矩阵均
传入支持 q4 变体的矩阵算子；预热 5 次、计时 20 次：

```bash
FLAGSPARSE_BACKEND=mthreads PYTHONPATH=src \
  python3 -u run_flagsparse_pytest.py \
  --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_csc,spmm_bsr,spmm_bell,spmm_coo,sddmm_csr \
  --gpus 0 --phase performance --mode quick \
  --results-dir results_musa_q4_20260928T145913Z/performance \
  --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 --timeout 3600
```

汇总的 q4 状态为 **34 Passed、6 Failed、2 NotFound**。`Passed` 的 34 个变体在其
benchmark 覆盖的全部输入上都有有效耗时：

| 算子族 | 通过的 q4 变体 |
|---|---|
| 稀疏向量 | `gather_i8_int`、`scatter_i8_int`、`axpby_f16_int`、`spvv_f16f32_int_non`、`spvv_c32_int_conj`、`spvv_i8i32_int_non` |
| CSR SpMV | `spmv_csr_f16f32_int_non`、`spmv_csr_f16_int_non`、`spmv_csr_f32c32_int_non`、`spmv_csr_c32_int_non`、`spmv_csr_f32_int_trans`、`spmv_csr_i8f32_int_non`、`spmv_csr_i8i32_int_non`、`spmv_csr_c32_int_conj` |
| COO SpMV | `spmv_coo_f16f32_int_non`、`spmv_coo_f32_int_trans`、`spmv_coo_c32_int_non`、`spmv_coo_f16_int_non`、`spmv_coo_c32_int_conj`、`spmv_coo_i8i32_int_non` |
| CSC SpMV | `spmv_csc_f32_int_non`、`spmv_csc_f16_int_non` |
| CSR SpMM | `spmm_csr_f16f32_int_non_non_row`、`spmm_csr_f16_int_non_non_row`、`spmm_csr_f32_int_non_non_col`、`spmm_csr_f32_int_non_trans_row`、`spmm_csr_c32_int_non_non_row`、`spmm_csr_i8i32_int_non_non_row`、`spmm_csr_f32_int_trans_non_row` |
| COO SpMM | `spmm_coo_c32_int_non_non_row`、`spmm_coo_f16_int_non_non_row` |
| CSR SDDMM | `sddmm_csr_f32_int_non_non_col`、`sddmm_csr_f32_int_non_trans_row`、`sddmm_csr_f32_int_trans_non_row` |

异常和聚合限制如下。这里的 `Failed` 不都表示 FlagSparse 内核没有运行，必须结合原始
CSV 判断：

| q4 变体 | 汇总状态 | 原始结果 | 原因 / 处理 |
|---|---|---|---|
| `spmv_sell_f32_int_non` | Failed | 10/10 行 `PASS`，0.0892 / 0.2169 / 4.9845 ms（min / median / max） | 父脚本的非 q4 complex64 行失败，进程退出码为 1，runner 因此把同一父脚本的 q4 行投影为 Failed；该 q4 行本身可用。 |
| `spmv_sell_f16_int_non` | Failed | 10/10 行 `PASS`，0.0844 / 0.2039 / 5.1221 ms | 同上；该 q4 行本身可用。 |
| `spmv_sell_i8i32_int_non` | Failed | 10/10 行 `PASS`，0.0667 / 0.1957 / 4.9451 ms | 同上；该 q4 行本身可用。 |
| `spmv_sell_c32_int_non` | Failed | 无 q4 性能行 | 父脚本的 complex64 行均为 `ERROR`，尚未产出可用的 complex SELL 性能数据；精度两种规模均通过。 |
| `spmv_csc_c32_int_non` | Failed | 10/10 行 `ERROR` | `RuntimeError: "IndexMusa" not implemented for 'ComplexFloat'`；内核耗时前的 complex 高级索引/参考路径仍未完全避开。 |
| `spmm_bell_f32_int_non_non_row` | Failed | 9/10 行 `PASS`，3.5688 / 5.0523 / 14.2175 ms | `ASIC_680ks.mtx` 被 Blocked-ELL 容量保护拒绝：预计存储 398,703,808 个值；其余 9 个矩阵有有效耗时。 |
| `spmm_csc_f32_int_non_non_row` | NotFound | 无 q4 性能行 | benchmark 的 COO `torch.sparse.mm` 参考在 MUSA 抛 `aten::addmm` 未实现，父进程退出前未写入 f32 行。 |
| `spmm_bsr_f32_int_non_non_row` | NotFound | 无 q4 性能行 | 同样依赖 MUSA 未实现的 `torch.sparse.mm` COO 参考，未写入 f32 行。 |

所有 Python 性能行的 PyTorch / cuSPARSE 基线与加速比均为 `N/A`，这是 MUSA 的
`torch.sparse` 矩阵乘未实现导致的预期结果，并非性能为零。需要带 muSPARSE 基线的交付
性能报告仍应使用 `run_flagsparse_split_delivery.py`；它覆盖的是交付的 20 个变体，不覆盖
本节的全部 q4 42 变体。

## 5. 后续待确认

1. 修复或绕过 CSC SpMV complex64 的 `IndexMusa` 高级索引路径，然后重跑
   `spmv_csc_c32_int_non` 的 10 矩阵性能行。
2. 将 CSC / BSR SpMM benchmark 的正确性参考固定为 CPU/SciPy，避免 MUSA 上不可用的
   `torch.sparse.mm` 阻断自身内核计时。
3. 为 SELL complex64 benchmark 采用与精度路径一致的实部 / 虚部平面参考，再单独验证
   `spmv_sell_c32_int_non`。
4. `python3 tools/probe_accel_capabilities.py` 的最新输出（torch_musa 升级后上表可能变化）。
