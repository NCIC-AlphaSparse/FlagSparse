# 摩尔线程 / MUSA（S5000）：65 变体验证任务

`FLAGSPARSE_BACKEND=mthreads`。环境搭建、交付复现见 [`../MUSA.md`](../MUSA.md)。背景见本目录
[`README.md`](README.md)。

## 要做的事

环境按 `../MUSA.md` 第 2 节配好（`FLAGSPARSE_BACKEND=mthreads`、`MUSA_HOME`）后，跑一轮 split：

```bash
setsid timeout -s KILL 43200 python3 -u run_flagsparse_split_delivery.py \
  --mode normal --benchmark-input tests/data --timeout 3600 \
  --h800-reference \
  --results-dir results_musa_split_65_<日期> \
  > results_musa_split_65_<日期>.log 2>&1 < /dev/null &
# 跑完后
python3 tools/delivery_table.py results_musa_split_65_<日期>
```

- 精度那一半调用 `run_flagsparse_pytest.py --phase accuracy --delivery-only`，覆盖全部 65 个变体。
- 性能那一半取 C API 的 `ctest`（对 muSPARSE），**现在也覆盖全部 65 个变体**：`origin/q4` 的 C API 工作
  2026-10-08 已移植进 main（见下面"C API 现在覆盖全部 65 个变体"一节），每个变体按自己的 id 单独测一次。
  muSPARSE 不支持的类型组合（fp16、int8、混合精度、SELL 等）显示 `NoBaseline`——跑通了、精度过了，只是没有
  厂商对照，**这是预期的**，不是失败。
- `--h800-reference`（不带值）：muSPARSE 不支持、标 `NoBaseline` 的变体（2026-10-09 那轮是 26 个），用仓库自带的
  H800 结果按带宽折算出加速比（H800 时间 × 3050/1370），交付表里只显示数字（`--show-baseline` 可看来源）。**muSPARSE 测得到的
  变体不受影响。** 加了这个开关，pytest 那一半会从只跑精度变成**精度 + 性能**（多一轮 Python 性能，耗时会明显变长），
  因为 C API 的规模（SpMM `n=8`、SDDMM `k=16/64`、向量类按矩阵）在 H800 上没有对应数据，而 Python 侧和 H800 参照
  是同一个 benchmark、同样规模。合并时按顺序取：① C API 有 muSPARSE 基线 → C API；② Python 侧有折算加速比 →
  Python（条目里 `source: python`，原 C API 结果留在 `capi_performance`）；③ 否则用 C API 时间对 H800 同变体同矩阵
  折算（只有 SpMV 类能对上）。日志会打印三类各补了多少、还剩哪些。`sddmm_csr_f16` 在 H800 上没有 cuSPARSE 时间，
  用 H800 PyTorch 折算。是估算，不是厂商实测，汇报时要注明；见
  [`../H800_REFERENCE.md`](../H800_REFERENCE.md)。
- 拉到新代码后第一次跑**不要加 `--skip-capi-build`**，确保 C API 是重新编译的。
- `--timeout 3600` 沿用 `../MUSA.md` 的结论（`900` 不够），split 会把它同时传给 CTest。CUDA 上 SpMM 这一族
  单独就要 18 分钟（65 个变体里 17 个是 SpMM，3 种宽度 × 10 个矩阵），MUSA 上 SpMM 只测第一种宽度。
- `--benchmark-input tests/data`：`tests/data` 正好是 10 个交付矩阵（在 git 里）；C API 一侧自己会再筛一遍。
- 外层 `timeout -s KILL` 是整条命令的总限时（内核卡死时 Ctrl-C 送不进去，只能靠 KILL）。65 变体**没有在本机
  实测过总耗时**，到点被杀的话按 `NotFound` 的算子单独补跑，补跑用新的 `--results-dir`。

把 `delivery_table.py` 的完整输出带回来。

## 已知风险点（这个后端特有的，不是猜的）

- **`torch.sparse` 在 MUSA 上完全没有矩阵乘实现**（`../MUSA.md` 4.5 节的实测能力矩阵已经写明）。
  这是四个后端里最明确的一条已知限制：凡是依赖 PyTorch 稀疏矩阵乘做对照/回退的路径都会失败，不止
  q4 新变体，原有 20 个变体的 PyTorch 对照列也受影响。`tests/q4_variant_bench.py` 的 `_time_pytorch`
  helper 专门注释了 "e.g. MUSA registers no sparse matmul, no sparse int8"，理论上会优雅跳过、报
  `pytorch_reason` 而不是崩溃——**这一条需要真机确认，不能只信注释**。
- **复数高级索引没有 kernel**：`values[order]` 这类操作在 MUSA 上报
  `"IndexMusa" not implemented for 'ComplexFloat'`。`_common._gather_values` 已经统一改走
  `view_as_real` 拆实部/虚部规避，但这是否覆盖了全部 45 个新变体的复数路径（`spmv_csr/coo/csc_c32`、
  `spmm_csr/coo/csc_c32`、`spvv_c32_int_conj`、`spmv_sell_c32`、`sddmm_csr_c32`）没有逐一验证过。
- **muDNN 的 gemv 不支持 fp64/复数**（gemm 支持）。这条主要影响原有 fp64 变体，新变体里的 fp64 相关
  路径（如果有间接依赖 gemv 的）也要留意。

## 跑完要确认（2026-10-14 之后这一轮）

1. `missing` 是不是 0；65 个变体精度是不是全 Passed（`spmv_coo_f32_int_non` 不应再是 Skipped）。
2. 日志里 `--h800-reference (musa-s5000): NoBaseline variants filled from the Python half=…, from the C API time=…,
   still without a baseline=…` 这一行原样贴回。按设计"still without"应该是 0；不是 0 就列出是哪些变体、为什么。
3. 交付表里还有没有 `NoBaseline`；`python3 tools/delivery_table.py <结果目录> --show-baseline` 的输出原样贴回（最后一列说明哪些加速比是折算的）。
4. 这轮 pytest 那一半第一次跑性能（加了 `--h800-reference` 的代价），报一下总耗时，以及 Python 性能阶段里有没有
   哪个算子超时或崩溃（`<结果目录>/<算子>/performance_stderr.log`）。
5. 复数变体有没有 `IndexMusa not implemented`；有没有慢几百倍的可疑变体。

把这些结果写回来，不用额外分析，原始数据最有用。

## C API 现在覆盖全部 65 个变体（2026-10-08 在 CUDA 上完成移植）

**来源**：这些 C API 实现本来就有。q4 在 MUSA 上做完并实测过（`origin/q4` 分支，2026-10-06），当时没有走
split，而是直接 `ctest -L capi -R '^benchmark\.(axpby|spmv|spmm|spvv|spgemm|sddmm|scatter)$'`，再用 q4 专门的
`capi/tools/write_summary_q4.py` 出一张 45 变体的表（`summary_q4.json`，不进交付 `summary.json`）。main 一直没合
这部分，所以 main 的 split 性能那一半只有 25 个变体。现在合进来了，split 一次就能出全部 65 个，不再需要
`write_summary_q4.py`。

**移植时改了什么**（都在 CUDA 上验证过）：

- `capi/tools/gen_variants.py` 按仓库根 `conf/operators.yaml` 的 65 项给每个变体生成带 id 的条目；q4 原版读的是
  `q4_variants:` 这个键，main 里已经没有了，原样搬过来会**静默地一行都不产生**。
- 各 benchmark 按 `FLAGSPARSE_BENCH_VARIANTS` 选行、给行打 `variant` 标签；`write_summary.py` 先按标签对应变体
  （`(spmm_csr, f32)` 下有 4 个变体，原来按算子 + 类型对应会对不上）；split 把 65 个 id 传给 CTest。
- `capi/conf/operators.yaml` 按实测补全（SpMV 的 `trans`/`conj_trans`、SpMM 的 f16、`mixed_dtypes`），
  `conf/operators.yaml` 里 65 个变体的 `capi` 全部为 `true`。
- 修了移植带来的一个回归：SDDMM 原有两个交付变体丢了 cuSPARSE 基线（q4 对所有带标签的行都不调基线），现在只
  对转置/列主序变体不调。
- fp16 输入按 Python 侧同样的规则先缩放到 [-1, 1]：`ASIC_680ks` 有值到 1e6，超出 fp16 范围（65504）会变成 inf。
  q4 表里 10 个 fp16 变体各有 1 行失败（9/0/1/0）就是这个原因，现在已经修好。
- **口径变化**：main 原来的 C API 测 `spmm_csr/coo_f32/f64_int_non_non_row` 时 B、C 实际是列主序（`test_spmm.cpp`
  写死了 `ORDER_COL`），名字标的却是 `_row`。现在按名字测行主序，这 4 个数字会变（CUDA 上 `spmm_csr_f32`
  0.392x → 0.686x）。拿新旧报告对比时注意这一点，这不是性能变化，是之前标错了。

**CUDA 上的端到端结果**（`run_flagsparse_split_delivery.py --backend cuda`，RTX 5090，10 个交付矩阵）：65 个变体
精度 65/65 Passed、0 missing；性能 34 个有 cuSPARSE 加速比、31 个 `NoBaseline`；`check_manifest` 0 处不一致；
C API 精度 10/10 测试族通过。唯一的精度瑕疵是 `spmm_csc_f32_int_non_non_row` 在 `cfd2` 上有 2 行误差比
1.03～1.13，略超严格容差（宽松容差下 0.011）。走的是原子累加路径，没有厂商基线可比，按规则拿不到宽松判定，
如实保留，没有改容差。

**只在 MUSA 上编译、CUDA 上测不到的部分**，上机时最该留意：

- `capi/src/ops/spmv.cpp`/`spmm.cpp` 里 `musa_gather_type()` 分支（`capi/src/core/sparse_gather.hpp`）：MUSA 上 CSC、
  转置、复数的 SpMV/SpMM 走这条 gather 路径。q4 当时靠它把这些变体从 0.02x～0.4x 提到 0.9x～3.5x。
- `FLAGSPARSE_MUSA_BASELINE_EXTENSIONS`（只在 `BACKEND=MUSA` 时定义）：muSPARSE 的 CSC 基线（CSC 当成转置后的
  CSR 喂给 muSPARSE，4.3.5 的 CSC 描述符会报错）、SDDMM 带转置/列主序的基线、SpVV 基线。
- 这两部分代码是从 q4 原样搬过来的，移植时没有改动逻辑，只改了变量名（`q4_variant` → `variant_id`）。

**预期结果**：q4 在 MTT S5000（muSPARSE 4.3.5）上的 2026-10-06 实测，45 个新变体、10 个交付矩阵。加速比是
`muSPARSE_ms / FlagSparse_ms`，只对双方严格通过的行取算术平均；P/R/F/U 是严格通过/放宽通过/失败/未校验的行数。
这次重跑应该大致复现这张表；fp16 那 10 行的失败应该消失。

<details>
<summary>q4 的 MUSA 实测表（45 个变体）</summary>

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

来源：`origin/q4` 的 `docs/debug/MUSA.md`（`b058939`）。原始结果目录 `capi/bench-musa-real-q4-real10-20261006/`
没进 git，只在 MUSA 机器上。

</details>

## 交给 Codex 的 prompt

上机时把下面整段原样贴给 Codex（或其他代理），不用改任何内容。

````text
你在 摩尔线程 S5000（MUSA） 实机上复测 FlagSparse 的 65 个交付变体（2026-10-14 之后的这一轮）。仓库在当前目录，main 分支。

背景：这台机器上一轮已经跑过 65 项。之后仓库改了两类东西，这轮就是验证它们：
- 没有基线的变体，现在用仓库自带的 H800 结果按带宽折算出加速比（命令里的 --h800-reference），交付表里只显示数字，
  delivery_table.py --show-baseline 的最后一列说明哪些是折算的。有本机基线的变体不受影响。
- 本后端相关的修复，见 docs/debug/MUSA.md "跑完要确认"一节逐条列出的内容。
主要目的是收集数据，不是修代码。

步骤：
1. git pull --ff-only origin main；git log -1 --oneline 的提交应不早于 d7d1919（"ci again"）。
2. 按 docs/MUSA.md 配好环境，做完它的环境自检。
3. 确认导入的是仓库源码：python3 -c "import flagsparse; print(flagsparse.__file__)"
   输出必须在当前目录的 src/ 下。如果指向 site-packages / dist-packages，先停下来汇报，不要继续。
4. 核对测试矩阵：这轮只用仓库自带的 10 个交付矩阵（conf/operators.yaml 的 delivery_matrices），路径和大小：
   tests/data/ASIC_680ks.mtx  69268748 字节
   tests/data/GL7d14.mtx  26726293 字节
   tests/data/NACA0015.mtx  43158387 字节
   tests/data/amazon0601.mtx  44469712 字节
   tests/data/auto.mtx  44768883 字节
   tests/data/cage12.mtx  61548649 字节
   tests/data/cfd2.mtx  41954205 字节
   tests/data/filter3D.mtx  47472525 字节
   tests/data/roadNet-TX.mtx  27699282 字节
   tests/data/wave.mtx  13325709 字节
   用 ls -l tests/data/*.mtx 核对：恰好这 10 个文件、大小一致。缺文件或大小不对（例如 git lfs 指针、下载不全）
   先停下来汇报，不要换别的矩阵目录。命令里的 --benchmark-input tests/data 指的就是它们。
5. 完整读一遍 docs/debug/MUSA.md，照"要做的事"一节的命令原样跑（已带 --h800-reference，不要去掉）。
   - --results-dir 用新目录（<日期> 填今天），不要复用旧目录，也不要往旧目录里补跑。
   - 命令本身已经用 setsid 放到后台，定期看日志进度，不要中途打断。
   - 不要用 pytest --forked，它在 GPU 上会让所有用例失败。
6. 跑完执行 python3 tools/delivery_table.py <结果目录>。

规矩：
- 不要为了"让它通过"去改 src/ 或 tests/ 下的代码。确实非改不可（比如环境适配）时，每改一个文件，
  都按 docs/debug/MUSA.md 末尾"实机改动记录"一节的格式追加一条记录，并和代码改动放在同一个 commit 里。
- 不要猜根因。遇到报错就给出完整报错原文，以及出错变体在结果目录里那一行的 reason/error 字段。
- 折算出来的加速比是估算，低也照实报，不要为了数字好看换基线或改参数。
- 结果目录整个打包带回（至少 summary.json、summary.csv、各算子的 performance.csv 和 *_stderr.log）。

汇报包含：
1. delivery_table.py 的完整输出，尤其是第一行 "N registered variants, M missing"。
2. 所有非 Passed 的变体、以及加速比为 "-" 的变体：变体名、阶段（精度/性能）、reason 或 error 原文。
3. 逐条回答 docs/debug/MUSA.md "跑完要确认"一节。
4. 环境指纹：torch、triton 版本，设备名，驱动/SDK 版本，git log -1 --oneline，整轮总耗时。
5. docs/debug/MUSA.md "实机改动记录"一节的内容（没改写"无"）。
````

## 实机改动记录

在这台机器上为了跑通而改过的每一个文件都记在这里，跟着 commit 一起推上来。没改就写"无"，不要留空。
只记改动，测试结果写到上面"带回来的东西"/"跑完要确认"里。

格式（一个改动一条，新的追加在最后）：

```
### <日期> <文件路径>::<函数或位置>
- 改了什么：
- 为什么改（附报错原文）：
- 怎么验证的：
- 其他后端是否也需要：是 / 否 / 不确定
```

（暂无）
