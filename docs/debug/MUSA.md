# 摩尔线程 / MUSA（S5000）：65 变体验证任务

`FLAGSPARSE_BACKEND=mthreads`。环境搭建、交付复现见 [`../MUSA.md`](../MUSA.md)。背景见本目录
[`README.md`](README.md)。

## 要做的事

环境按 `../MUSA.md` 第 2 节配好（`FLAGSPARSE_BACKEND=mthreads`、`MUSA_HOME`）后，跑一轮 split：

```bash
# 1. 更新代码：至少要包含 a330b62（见下面"2026-10-14 那轮的两个 NoBaseline"）
git pull --ff-only
git log --oneline -1
git merge-base --is-ancestor a330b62 HEAD && echo "包含 a330b62 修复"

# 2. 环境自检
export PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads MUSA_HOME=/usr/local/musa   # MUSA_HOME 以本机实际安装位置为准
python3 -c "import flagsparse; print(flagsparse.__file__)"   # 必须是 <仓库>/src/flagsparse/__init__.py
python3 -c "from flagsparse.sparse_operations import _common as c; print(c._backend_name(), c._accel_device_type(), c._accel_fallback_reason())"
# 期望：mthreads musa None
ls tests/data/*.mtx | wc -l          # 期望 10
mthreads-gmi                         # 确认没有别的任务在用 GPU

# 3. 全量 split（每次用新的结果目录）
D=results_musa_split_65_<日期>
setsid timeout -s KILL 43200 python3 -u run_flagsparse_split_delivery.py \
  --mode normal --benchmark-input tests/data --timeout 3600 \
  --h800-reference \
  --results-dir $D \
  > $D.log 2>&1 < /dev/null &

# 4. 跑完后
tail -5 $D.log                                    # H800 折算三类各补了多少、还剩哪些
python3 tools/delivery_table.py $D                # 期望 0 missing，性能里没有 NoBaseline
python3 tools/delivery_table.py $D --show-baseline | grep -E "scatter_i8_int|spmm_csc_f16"
# 期望这两行有加速比，来源 h800_scaled
```

- `flagsparse.__file__` 那一步不能省：系统 site-packages 里残留的旧安装会顶替 `src/`，基线悄悄变成 `N/A`。
- `MUSA_HOME` 设错不会报错，muSPARSE 基线同样会悄悄退成 `none`（`../MUSA.md` 第 2 节）。

**2026-10-14 那轮的两个 NoBaseline**（`results_musa_split_65_20261014`，63 Passed + 2 NoBaseline）：
`scatter_i8_int` 和 `spmm_csc_f16_int_non_non_row` muSPARSE 都不支持（C API 报 `Scatter dtype unsupported`、
`unsupported SpMM type or operation`），只能靠 Python 侧的折算补，而那轮 Python 侧这两处都崩了：scatter int8 的
torch 参考被 muDNN 拒绝（`Now muDNN only support float64, float32, half, bfloat16, int32, and int64`，4 行全是
`ERROR`）；spmm_csc 的 `torch.sparse.mm` 参考在 MUSA 上没有实现（`Could not run 'aten::addmm' with arguments from
the 'Sparsemusa' backend`，CSV 为空）。`a330b62`（参考改在 CPU / SciPy 上算）修了这两处，但它比那轮结果晚提交，
所以那份报告里的这两行是旧代码的结果，按上面的命令重跑即可。CUDA 上以 `FLAGSPARSE_BACKEND=mthreads` 模拟已确认
两行都能测出性能；折算这一步只在真机上走得到。

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

### 2026-10-15 优化与复测口径

本轮按用户要求优化代码，验收目标是**每个交付变体各自的平均加速比 > 0.8x**，不是 65 项的整体平均。
矩阵仍为 `tests/data` 的原始 10 个文件，split runner、测试规模、容差和基线优先级不变。
本机 CPU cgroup 配额为 15 核，后续配对测量统一设置
`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`，原始源码也用相同设置复测，
避免默认大线程池触发 CPU 限流。H800 带宽折算仍是估算，不能当作本机厂商实测。
下列定向性能结果只用于调优，未替代最终完整 split；未达标和缺基线的条目必须继续如实报告。

### 2026-10-15 src/flagsparse/sparse_operations/mixed_spmx.py::MUSA SpMV/SpMM 路由
- 改了什么：COO/CSC 转 CSR 使用有界、张量版本感知的缓存；短行 SpMV 调整行块，严重不均衡短行复用 NNZ 分块内核；SpMM 复用行批量 gather；避免输出自拷贝。
- 为什么改（附报错原文）：无报错，原 COO 半精度/整数原子路径及逐行 SpMM 的实测加速比低于 0.8x。只缓存结构和必要的值重排，不缓存计算结果；输入变更失效。
- 怎么验证的：10 矩阵定向复测 `results_musa_opt_20261015_nnz_threads8/`；Python 定向回归 80 passed、3 skipped，含乱序/重复 COO、值/索引/向量修改、无版本推理张量、非连续输出。
- 其他后端是否也需要：否；新路由仅 MUSA。

### 2026-10-15 src/flagsparse/sparse_operations/_spmm_row_gather.py::_kernel/compute
- 改了什么：增加可选累加类型以支持 int8 到 int32；MUSA 使用 8 行、4 个稀疏槽、4 warps。
- 为什么改（附报错原文）：无报错；MUSA 多行批量配置比逐行混合精度内核更快，整数输出必须保持整数累加。
- 怎么验证的：上述 80 项回归与 10 矩阵 SpMM 定向性能复测。
- 其他后端是否也需要：否；默认累加类型和其他后端的配置保持不变。

### 2026-10-15 src/flagsparse/sparse_operations/_spmv_csr_nnz.py::_musa_segmented_add
- 改了什么：MUSA 专用分段扫描的零分支使用整数文字 `0`，避免 int32 被浮点文字隐式提升；通过默认 False 的编译期 MUSA 开关选择，原共享归约函数保留原样。
- 为什么改（附报错原文）：无报错；复用到 MUSA int8/int32 时必须保持精确整数归约。
- 怎么验证的：整数 NNZ 路径包含重复列、空行、输入变更和非连续输出的精确回归。
- 其他后端是否也需要：否；其他后端仍使用原 `_segmented_add` 及浮点零分支。

### 2026-10-15 src/flagsparse/sparse_operations/_spmv_csr_transpose.py::make_plan/gather
- 改了什么：MUSA 调整行块；半精度实数 gather 复用混合精度 CSR 执行器。
- 为什么改（附报错原文）：无报错；原半精度 CSR 标量路径很慢。实验中的全行排序未带来收益，未用于最终混合 SpMV 路径。
- 怎么验证的：80 项 Python 定向回归；含复数转置用例，未修改精度容差。
- 其他后端是否也需要：否；配置与路由均以 MUSA 为条件。

### 2026-10-15 src/flagsparse/sparse_operations/spmv_csr.py::prepare_spmv_csr
- 改了什么：MUSA 半精度默认配置建立 non-gather plan，运行时使用优化的行批量路径。
- 为什么改（附报错原文）：无报错；原 FP16 SpMV 性能明显落后。
- 怎么验证的：半精度交付精度用例及 10 矩阵定向性能测试。
- 其他后端是否也需要：否；显式配置和其他后端保持原行为。

### 2026-10-15 src/flagsparse/sparse_operations/spmv_csc.py::_spmv_csc_via_mixed
- 改了什么：MUSA 半精度非转置调用缓存化的 `spmv_csc_half`。
- 为什么改（附报错原文）：无报错；原路径每次展开列号，抵消了缓存收益。
- 怎么验证的：CSC 值/索引变更回归，10 矩阵定向复测约 4.61x（H800 折算）。
- 其他后端是否也需要：否；原 CUDA/MACA 路由不变。

### 2026-10-15 src/flagsparse/sparse_operations/spmm_csr.py::默认半精度执行
- 改了什么：MUSA 非 accuracy 模式 FP16 复用 row-gather。
- 为什么改（附报错原文）：无报错；原路径逐行/逐槽执行缓慢。
- 怎么验证的：80 项回归及 10 矩阵定向性能复测，FP16 变体约 2.08x（H800 折算）。
- 其他后端是否也需要：否；CUDA 复数路由与其他后端不变。

### 2026-10-15 src/flagsparse/sparse_operations/spmm_coo.py::PreparedCooSpmmRoute
- 改了什么：MUSA FP16 准备 CSR gather 指针，复用现有行批量执行器。
- 为什么改（附报错原文）：无报错；避免原子累加/逐非零元程序的高开销。
- 怎么验证的：COO 精度用例和 10 矩阵定向性能复测，FP16 约 2.19x（H800 折算）。
- 其他后端是否也需要：否。

### 2026-10-15 src/flagsparse/sparse_operations/spmv_sell.py::csr_to_sell/并行执行
- 改了什么：MUSA 复数转换按实虚分量 index_copy；实数使用并行稀疏槽归约，复数使用已有并行内核。
- 为什么改（附报错原文）：原转换报 `"IndexPut" not implemented for 'ComplexFloat'`；串行稀疏槽性能较慢。
- 怎么验证的：10 矩阵、5 种数据/输出组合均 PASS；包含部分 slice、空行和 padding 的回归。
- 其他后端是否也需要：否；复数规避复用 Ascend 的已有方式，其他后端执行路由不变。

### 2026-10-15 src/flagsparse/sparse_operations/gather_scatter.py::_triton_gather_impl
- 改了什么：MUSA FP16 使用原生 index_select；保留输出形状/类型检查，非连续输入先连续化，非连续输出经临时缓冲区复制。
- 为什么改（附报错原文）：无异常；原 Triton 启动约 25us，原生索引约 7--16us。实测原生非连续输出有错误，不能直接传入。
- 怎么验证的：4 个原始 Gather 性能规模均 PASS；重复索引、非连续输入/输出、原地修改、错误形状/类型回归。
- 其他后端是否也需要：否。

### 2026-10-15 capi/flagsparse_codegen/spmv_csr.py::spmv_csr_row_tile_real
- 改了什么：增加 WIDE 开关，小矩阵使用 int32 地址计算，大范围仍保留 int64。
- 为什么改（附报错原文）：无报错；无条件 int64 地址计算增加 MUSA 指令成本。
- 怎么验证的：C API SpMV 精度测试族通过，10 矩阵 FP32 定向实测约 0.824x。
- 其他后端是否也需要：否；默认 WIDE 为 True。

### 2026-10-15 capi/src/ops/spmv.cpp::短行 CSR 与 SELL 启动
- 改了什么：按边界选择 CSR WIDE；MUSA SELL 选择并行稀疏槽内核和对应的 4-warps 配置。
- 为什么改（附报错原文）：无报错；减少地址计算与串行 slot 循环的开销。
- 怎么验证的：C API SpMV 精度测试族通过；10 矩阵定向性能复测。
- 其他后端是否也需要：否；SELL 其他后端沿用原内核和配置。

### 2026-10-15 capi/flagsparse_codegen/spmv_sell.py::并行 SELL 包装
- 改了什么：导出共享复数并行内核，增加 FP32/int32 累加的薄包装。
- 为什么改（附报错原文）：无报错；C API 复用 Python 中同一个实现，避免复制计算逻辑。
- 怎么验证的：C API SpMV 精度测试族通过，覆盖 SELL 实数、半精度、复数、整数。
- 其他后端是否也需要：否；原导出仍保留。

### 2026-10-15 capi/flagsparse_codegen/spmm_csr.py::spmm_csr_batched8_real
- 改了什么：按 WIDE 选择地址宽度，增加可调稀疏槽宽度 K，默认值保持 4。
- 为什么改（附报错原文）：无报错；int64 寻址与固定槽宽限制了 MUSA FP32/FP64 性能。
- 怎么验证的：C API SpMM 精度测试族通过；10 矩阵定向性能测试。
- 其他后端是否也需要：否；该内核只由 MUSA 路由选择。

### 2026-10-15 capi/src/ops/spmm.cpp::run_csr
- 改了什么：MUSA N=8 按累加类型选择行批量和槽宽，使用 4 warps；大地址保留 64 位。
- 为什么改（附报错原文）：无报错；原 batch=4/1 warp 实测吞吐低。
- 怎么验证的：10 矩阵定向实测 FP32 约 0.863x、FP64 约 0.902x，精度测试族通过。
- 其他后端是否也需要：否。

### 2026-10-15 capi/src/ops/spvv.cpp::run
- 改了什么：MUSA nnz<=32 使用 block=32、1 warp，其他情况保持原配置。
- 为什么改（附报错原文）：无报错；小向量不需要 256 个归约槽。
- 怎么验证的：C API SpVV 精度测试族通过；定向性能有波动，尚不能据此认定稳定达到 0.8x。
- 其他后端是否也需要：否。

### 2026-10-15 capi/src/ops/spgemm.cpp::workEstimation/compute/launch_hash
- 改了什么：MUSA 按行产品工作量分桶；复用主机计数存储，合并相邻的计数/溢出标志传输与前缀扫描。
- 为什么改（附报错原文）：无报错；原来全矩阵使用最坏行的哈希容量，短行也承担单行哈希和大表开销。
- 怎么验证的：C API SpGEMM 精度测试族通过；10 矩阵定向复测约 0.825x，存在波动。排序去重实验精度通过但性能退化，已撤回。每次重新计算，不缓存输出计数。
- 其他后端是否也需要：否；分桶与主机缓冲复用均限定 MUSA，其他后端保留原两次传输、溢出检查和前缀扫描流程。

### 2026-10-15 tests/pytest/test_musa_sparse_routes.py::新增回归
- 改了什么：新增 MUSA 格式转换缓存失效、整数精确累加、SELL padding/部分 slice、非连续输出、推理张量和 Gather 边界测试。
- 为什么改（附报错原文）：优化后必须验证输入实时可变，不能只测同一输入重复执行。
- 怎么验证的：连同 q4 SpMV/SpMM 精度用例共 80 passed、3 skipped；跳过的是不在该缓存路由内的 CSC 组合，不是交付变体跳过。
- 其他后端是否也需要：否；该文件显式要求 MUSA，原跨后端测试未删除或放宽。

### 2026-10-15 MUSA-only 范围收紧
- 改了什么：按用户补充要求，`mixed_spmx._finish` 仅在 MUSA 避免输出自拷贝，非 MUSA 恢复原行块选择；`_spmm_row_gather.compute` 的 int32 累加仅在 MUSA 启用；`_spmv_csr_nnz` 的 MUSA 归约通过编译期开关隔离；`capi/src/ops/spgemm.cpp` 的缓冲复用和分桶显式限定 MUSA。
- 为什么改（附报错原文）：无报错；用户要求“修改仅限于musa后端”，不能把共享行为变化带给其他后端。
- 怎么验证的：新增不执行 GPU 内核的路由隔离回归，分别模拟 MUSA/非 MUSA 检查自拷贝、整数累加和行块参数，6 passed；`git diff --check` 通过。此前 80 项 GPU 回归是隔离前结果；隔离后的 Python 回归、C API 重新编译及 SpMV/SpMM/SpGEMM/SpVV 精度复验排在当前 split 完成后运行，日志 `results_musa_scope_20261015.log`，未完成前不视为通过。
- 其他后端是否也需要：否；没有 CUDA/ROCm 等其他后端的真机验证，不声称已验证。
- 测试版本说明：`results_musa_split_65_20261015_opt/` 在隔离修改前启动。修改未调整 MUSA 算法、配置或基线，但源码哈希发生变化；该轮作为隔离前优化结果保留，不标为最终隔离后完整验收。

### 2026-10-16 本轮状态与定向数据

用户确认停止旧整轮后，旧 runner 及排队验证已停止，原目录与日志全部保留。本轮继续限定 MUSA，未修改测试矩阵、规模、容差或基线优先级。核对 `tests/data` 的 10 个 `.mtx` 大小与上文一致；导入路径为当前仓库 `src/flagsparse/__init__.py`，运行环境为 `mthreads musa None`、torch 2.7.1、Triton 3.6.0、MTT S5000。

定向结果不是完整交付验收，所有 H800 折算均仍为估算：

- `results_musa_opt_20261016_complex3d/`：原 10 矩阵上每条精度 PASS；`spmv_csr_f32c32_int_non` 平均从本轮改动前约 0.703x 提升至 0.819x。FP16→FP32 0.779x、FP16 0.754x、int8→FP32 0.820x、int8→int32 0.812x。COO 对应三项约 0.586x、0.685x、0.645x，仍未达标。
- `results_musa_opt_20261016_sell_flat/sell.csv`：5 种值/输出组合、10 矩阵，共 50 条 PASS。按已有同规模 Python 折算基线，SELL FP16 0.609x、FP32 0.695x、int8→int32 0.590x，仍未达标。
- `results_musa_opt_20261016_capi_sell/`：4 个交付 SELL 变体各 10 矩阵，40 条 `status=ok`、精度通过；复数 SELL 按已有 H800 同矩阵折算基线为 0.610x（旧轮约 0.275x），仍未达标。
- 固定最大行宽展开、SELL 行连续布局、成对复数加载、自动 Graph 重放均仅作临时探针，未显示稳定整体收益，没有接入正式源码。SELL→CSR 在 ASIC 上明显改善、其他矩阵有退化，也未接入。
- Python 扩大回归（MUSA 专用回归、q4 SpMV/SpMM、原 CSR 精度文件）已得到 559 passed、1659 skipped、23 deselected；跳过为原有后端/算法适用性规则，未新增或放宽规则。最终补充非连续输出后的结果及完整 split 见后续记录。

**尚未达到“每个变体各自平均 > 0.8x”目标，不能用上述单项或整体平均代替验收。**

### 2026-10-16 src/flagsparse/sparse_operations/mixed_spmx.py::_musa_csr_real_by_complex_kernel/_musa_coo_csr
- 改了什么：实数×复数使用三维实虚分量布局；严重不均衡短行可走复数 NNZ 路径。MUSA COO 从已排序行号用 `searchsorted` 建 CSR 指针，稀疏输入检查随版本感知缓存失效重新执行，RHS/输出仍每次检查。
- 为什么改（附报错原文）：无异常；原实数×复数逐行 kernel 慢。另复现原生 MUSA `bincount` 偶发少计数，4930 个输入得到计数和 4917，导致后续行指针及结果错误。
- 怎么验证的：10 矩阵定向结果见本轮状态；FP32/FP64 实数×复数、跨块、空行、输入修改回归通过，不缓存输出。
- 其他后端是否也需要：否；新 kernel 和预处理均仅 MUSA 路由选择。

### 2026-10-16 src/flagsparse/sparse_operations/_spmv_csr_nnz.py::_nnz_kernel
- 改了什么：增加默认 False 的 `COMPLEX_RHS`，MUSA 分别扫描实虚分量并处理跨块前缀；MUSA 的前缀补偿零分支保持整数类型。
- 为什么改（附报错原文）：无异常；实数×复数需要 NNZ 均衡，int8→int32 不允许跨块补偿隐式转 FP32 而丢失大整数低位。
- 怎么验证的：新增 1101 个 127×127 的精确整数累加，以及 601 个非零元跨块复数、空行和实时修改回归；未开启新参数的其他后端保持原扫描/补偿路径。
- 其他后端是否也需要：否。

### 2026-10-16 src/flagsparse/sparse_operations/_spmv_csr_transpose.py::prepare/gather
- 改了什么：MUSA 转置预处理使用排序后 `searchsorted` 替代 `bincount`；FP16 已按 CSR 排列时提前进入执行器；非连续输出使用临时缓冲区后复制回调用方视图。
- 为什么改（附报错原文）：复现 MUSA `bincount` 少计数；通用处理存在额外调用开销，且临时输出不能丢失调用方指定的非连续输出。
- 怎么验证的：扩大 CSR 精度、q4 SpMV/SpMM 回归通过，补充直接/间接 FP16 gather 的非连续输出回归。
- 其他后端是否也需要：否；其他后端仍使用原预处理和执行路径。

### 2026-10-16 src/flagsparse/sparse_operations/spmv_coo.py::PreparedCoo.__init__
- 改了什么：仅 MUSA 对排序后的转置行号使用 `searchsorted` 构造 CSR 行指针。
- 为什么改（附报错原文）：q4 COO FP32 转置出现 `AssertionError: max error 24 (scale 21.4)`；诊断时行指针末值 4862 而实际 nnz=4930。独立重复计数探针亦复现少计数，不是归约容差问题。
- 怎么验证的：新增 30 次重复 COO 转置回归，原 q4 转置及 CSR 精度回归通过。
- 其他后端是否也需要：否。

### 2026-10-16 src/flagsparse/sparse_operations/spmm_coo.py::PreparedCooSpmmRoute.__init__
- 改了什么：MUSA FP16 的已排序 COO gather 指针也由 `searchsorted` 生成，CUDA 复数仍保持原 `bincount` 路径。
- 为什么改（附报错原文）：同一 MUSA 计数原语用于该路由；避免把少计数风险留在 SpMM 中。
- 怎么验证的：q4 SpMM 与 MUSA 混合输入修改回归通过。
- 其他后端是否也需要：否。

### 2026-10-16 src/flagsparse/sparse_operations/spmv_sell.py::_musa_spmv_sell_kernel/flagsparse_spmv_sell
- 改了什么：MUSA 使用按行并行的二维 ROWS×SLOTS 布局；实数、整数和复数共享索引逻辑，复数累加显式映射 FP32/FP64。
- 为什么改（附报错原文）：旧 SELL 性能低；新路由最初遗漏复数累加类型，回归报 `KeyError: torch.complex64` / `torch.complex128`，现已修正。
- 怎么验证的：部分 slice、空行、padding、FP64/complex128 回归通过；上述 50 条 Python 和 40 条 C API 定向记录通过精度。
- 其他后端是否也需要：否；其他后端内核和配置保持不变。

### 2026-10-16 capi/flagsparse_codegen/spmv_sell.py / capi/src/ops/spmv.cpp::MUSA SELL
- 改了什么：MUSA 专用薄包装导出新的共享二维 SELL kernel，按 32 行/program、4 slots、4 warps 启动；非 MUSA 保留原包装和网格。
- 为什么改（附报错原文）：无报错；复用经过 Python 回归的实现降低旧串行 SELL 开销。
- 怎么验证的：重新构建确认无待编译项，C API SpMV 精度族通过，40 条 SELL 定向记录全部精度通过。
- 其他后端是否也需要：否。

### 2026-10-16 capi/src/ops/spvv.cpp::run / capi/ctest/accuracy/test_spvv.cpp
- 改了什么：MUSA 默认流主机结果依赖有序同步 D2H，省去重复全设备同步；非默认流和其他后端仍显式同步。增加 MUSA 默认流/非默认流、复数输入反复变更的主机结果检查。
- 为什么改（附报错原文）：无报错；重复同步影响小向量耗时。
- 怎么验证的：40 次双流复数结果检查及 C API SpVV 精度族通过；`results_musa_opt_20261015_spvv_sync/` 原 10 矩阵目录口径下的向量 fixture 定向约 1.202x，完整验收仍以新 split 为准。
- 其他后端是否也需要：否。

### 2026-10-16 tests/pytest/test_musa_sparse_routes.py::补充回归
- 改了什么：增加重复 COO 转置、大整数跨块、实数×复数的跨块/输入变更/FP64/部分 tile，以及直接和间接 FP16 gather 非连续输出用例。
- 为什么改（附报错原文）：覆盖本轮预处理及启动配置变更的真实失败和边界条件；不放宽原容差。
- 怎么验证的：扩大 GPU 回归见本轮状态与后续最终验证记录。
- 其他后端是否也需要：否；文件显式限定 MUSA。

### 2026-10-16 tests/pytest/test_spmv_csr_accuracy.py::test_spmv_csr_empty_dtype_ops
- 改了什么：仅 MUSA 复数结果改为对 `view_as_real(actual)` 执行原零值断言。
- 为什么改（附报错原文）：原检查调用 MUSA 不支持的复数 `count_nonzero`，报 `RuntimeError: SetMUTensorDType Unsupported tensor dtype: ComplexFloat` / `ComplexDouble`。
- 怎么验证的：原测试覆盖的所有 MUSA 空矩阵/数据类型/转置组合通过；仍断言非零元素数严格为 0，未跳过测试或改变精度标准。
- 其他后端是否也需要：否；其他后端检查保持原样。

### 2026-10-16 最终启动前验证与阶段性完整复测

- 最终 Python 回归：`561 passed, 1659 skipped, 23 deselected`，15.82 秒；日志 `results_musa_opt_20261016_validation.log`。包含后来补充的直接/间接 FP16 gather 非连续输出检查。
- C API 交付相关精度族：`accuracy.axpby/spmv/spmm/spvv/spgemm/sddmm/gather/scatter`，8/8 passed，44.97 秒。未把非交付 SpSV/SpSM 纳入本轮验收。
- `cmake --build capi/build --parallel 8` 确认 `ninja: no work to do.`；`git diff --check` 通过。
- 新整轮目录：`results_musa_split_65_20261016_opt`，日志 `results_musa_split_65_20261016_opt.log`。使用 `setsid`、12 小时外层限时、原 split runner，`--mode normal --benchmark-input tests/data --timeout 3600 --h800-reference`，并固定三种 CPU 线程池为 8。
- 此整轮用于检查当前阶段代码的 65 项结果，不预先视为所有加速比达标；定向测量中 FP16、COO、SELL 仍低于 0.8x，完整表生成后仍需按每个变体逐项检查。

### 2026-10-16 上述整轮完成后的原始数据复核

`results_musa_split_65_20261016_opt/` 已完成，65 项、0 missing，合并表的精度/性能标签均为 Passed，
但逐行复核仍有 16 个变体平均加速比低于 0.8x，另有 2 条 C API 原始精度失败，**不能按标签认定完全通过**。
Python 普通 CSC SpMM 的 `amazon0601 float32 trans n=32` 也有误差比 2.53 的失败，使 runner 返回 1。
C API 原始文件已保留在该目录的 `capi_raw_snapshot/`。Python CSC 性能族耗时 1212.88 秒；C API 8 个性能族
合计 1097.73 秒。补基线日志原文：

```text
--h800-reference (musa-s5000): NoBaseline variants filled from the Python half=26, from the C API time=0, still without a baseline=0 [] (estimates, marked baseline=h800_scaled)
```

后续整轮显式使用独立的 `--capi-bench-out <新结果目录>/capi`，不再复用共享输出目录。

### 2026-10-16 src/flagsparse/sparse_operations/gather_scatter.py::_triton_scatter_impl
- 改了什么：MUSA FP16/int8 默认配置使用原生 `index_copy_`，显式 block_size 仍走原 Triton 路径；原生索引提升到 int64 并如实记录执行索引类型。保留原输出分配、清零、reset_output 和空输入行为。
- 为什么改（附报错原文）：小向量 Triton 启动开销较大；原生 int32 索引实际报 `RuntimeError: index_copy_(): Expected a long tensor for index, but got Int`，不能假定支持。
- 怎么验证的：`results_musa_opt_20261016_routes_v2/scatter.csv` 8/8 PASS，FP16/int8 对本机 PyTorch 约 0.96x/0.94x；增加非连续输入输出、实时修改、显式调优和非 MUSA 路由隔离回归。后续完整验证见最后记录。
- 其他后端是否也需要：否；原生路由仅 MUSA 默认配置。

### 2026-10-16 src/flagsparse/sparse_operations/spmm_csc.py::FP16 路由和转置 gather
- 改了什么：MUSA 默认非转置 FP16 缓存 CSC 到 CSR 的结构/值重排，使用已有混合 SpMM；原始索引、指针、值、存储或版本变化均失效，不缓存输出。MUSA 未指定调优块参数的 FP32/complex64 转置使用行拥有输出的 gather；支持 auto 和其解析后的 spmm_csc_base，保留其他类型和显式块参数路径。索引回退保留路由标志，元数据反映实际块配置。
- 为什么改（附报错原文）：FP16 每次转 FP32 并原子散射，旧折算均值仅 0.012703x。普通 FP32 转置出现上述误差比 2.53；旧 complex64 转置在 amazon0601 约 1.2 秒/次。
- 怎么验证的：`results_musa_opt_20261016_csc_integrated_v2/` FP16 原 10 矩阵全 PASS，折算均值 3.3125x；`results_musa_opt_20261016_routes_v4/csc_trans.csv` 普通 FP32/complex64、trans/conj、原 10 矩阵共 40 条 PASS。amazon0601 FP32 约 65.82 ms 到 0.917 ms，complex64 约 1233 ms 到 13.32 ms，均为同脚本本身耗时，不是新的厂商基线。
- 其他后端是否也需要：否；新增选择条件显式限定 MUSA，其他后端保持原实现。

### 2026-10-16 src/flagsparse/sparse_operations/mixed_spmx.py::_musa_csr_spmv/_musa_row_tile
- 改了什么：MUSA 网格使用整数上取整，去掉复数路径未用网格；超短实数/整数行采用 256 行、2 slots，大规模中短行采用 128 行、8 slots，小规模中短行仍为 64 行。
- 为什么改（附报错原文）：无异常；减少主机调用开销和超短行 program 数量。
- 怎么验证的：`results_musa_opt_20261016_wide_rows_v1/measurements.json` 配置探针保留首尾原配置作对照；roadNet-TX FP16/int8 约快 10%～15%。原有完整精度标准不变，新增实时存储替换回归。整轮验收仍以新 split 为准。
- 其他后端是否也需要：否；仅修改 MUSA 专用 helper。

### 2026-10-16 src/flagsparse/sparse_operations/spmv_sell.py::MUSA 分发与计划
- 改了什么：MUSA 提前分发，省去通用路径未用的网格/环境变量处理；对 padding 超过有效非零元 2 倍的实数/整数 SELL 压缩为 CSR。版本感知计划复用稀疏校验和连续化结果，RHS 与输出仍每次检查；validate=False 计划升级到 True 时重新校验，输入修改后失效。
- 为什么改（附报错原文）：无异常；ASIC 和 filter3D 的 SELL padding 分别约 3.98 倍、2.48 倍，读取无效槽开销大；重复包装校验影响短 kernel。
- 怎么验证的：原 10 矩阵 50 条精度 PASS，包含非交付 FP16 到 FP32；`results_musa_opt_20261016_routes_v4/sell.csv` 保留全部原始时间（包括 wave FP16 的较慢单次记录），不删慢行。新增高 padding、部分 slice、指针/列/值修改、推理张量、非连续 RHS/out、校验升级和非法索引回归。
- 其他后端是否也需要：否；其他后端的校验和计算分支保持原样。

### 2026-10-16 capi/flagsparse_codegen/sparse_gather.py::sparse_gather_spmv_f32
- 改了什么：复数各 lane 的累加仍为 FP32，仅最终跨 lane 归约提升到 FP64，再转回 FP32 执行 alpha/beta。
- 为什么改（附报错原文）：ASIC 第 405878 行的 3 个乘积包含小项和约正负 74 的抵消；原 FP32 归约丢小项，严格误差比 1.363375。CPU 与 GPU 探针均复现，未修改容差。
- 怎么验证的：`results_musa_opt_20261016_precision_v1/measurements.json`；加宽最后归约后 ASIC 该变体整体严格误差比约 0.0495。C API 新增双分量抵消回归，精度族通过。
- 其他后端是否也需要：否；该模块由 MUSA 专用 sparse_gather 分支使用。

### 2026-10-16 capi/src/core/sparse_gather.hpp::launch_sparse_gather
- 改了什么：间接实数 SpMV 平均行长不足 12 时保留 scatter，较长行采用确定性 gather。
- 为什么改（附报错原文）：cfd2 CSC FP32 的原子加法顺序导致原始严格精度失败；较长行可摊薄间接读取开销。
- 怎么验证的：`results_musa_opt_20261016_capi_precision_v2/` 原 10 矩阵，CSC FP32 10 条可比且精度通过，muSPARSE 平均约 0.974x；复数 CSR 共轭转置 10 条精度通过，muSPARSE 平均约 0.860x。
- 其他后端是否也需要：否；调用方只在 MUSA 选择此执行器。

### 2026-10-16 capi/src/ops/gather_scatter.cpp::gather_max_programs
- 改了什么：仅 MUSA 使用按设备区分的 thread_local program 上限缓存，避免每次查询硬件 SM/warp/线程属性。
- 为什么改（附报错原文）：无异常；小 Gather 的主机属性查询占比高。
- 怎么验证的：上述 C API 定向目录中 FP32/FP64/complex64/complex128 各 10 条精度通过，对 muSPARSE 平均约 0.970x/0.973x/0.966x/0.971x。
- 其他后端是否也需要：否；其他后端仍保留原查询行为。

### 2026-10-16 capi/ctest/accuracy/test_spmv.cpp::MusaComplexGatherCancellation
- 改了什么：MUSA 专用宏下增加来自 ASIC 的三项抵消回归，使用原有 complex64 严格容差，检查两个分量。
- 为什么改（附报错原文）：覆盖上文精度缺陷，防止只通过总体状态标签而漏掉原始失败。
- 怎么验证的：重新构建并运行 accuracy.spmv/gather/scatter，3/3 passed；后续扩大验证见最后记录。
- 其他后端是否也需要：否；新测试显式限定 MUSA。

### 2026-10-16 定向调优中未接入的实验

- `results_musa_opt_20261016_sorted_ell_v1/`：按行长重排的连续布局有局部收益，但多矩阵退化，未接入。
- `results_musa_opt_20261016_coo_cache_pair_v1/`：同进程交替配对表明，拆分 COO 拓扑/值缓存没有稳定收益，部分短行慢约 1%～10%；仅撤销本次实验，保留此前版本感知缓存。
- `results_musa_opt_20261016_loop_v1/`、`results_musa_opt_20261016_load_v1/`：循环展开、软件流水和读取缓存提示没有可靠整体收益，未接入。
- `results_musa_opt_20261016_sell_complex_compact_v1/`：复数 SELL 压缩在 filter3D 有收益，但 ASIC 退化，10 矩阵平均约 0.744x；未接入。稀疏校验计划也仅在 FP16/BF16/FP32/int8 使用，复数保持原校验与 kernel。
- 当前验证过的扩大 Python 回归为 845 passed、1658 skipped，另增 6 项原生 Scatter 调优/后端隔离回归通过。仍未达到每个变体 >0.8x；任何定向均值不能替代新的完整 split 结果。

### 2026-10-16 本次源码固定后的验证与下一轮完整复测

- Python 扩大回归：851 passed、1658 skipped，18.69 秒；`results_musa_opt_20261016_routes_v4/accuracy.xml`。包含 CSR/CSC/SELL、q4、Gather/Scatter 和 MUSA 路由隔离回归；没有降低容差或新增跳过规则。
- C API 交付相关精度族：8/8 passed，39.42 秒；`results_musa_opt_20261016_routes_v4/capi_accuracy.log`。构建确认 `ninja: no work to do.`，`git diff --check` 通过。
- `results_musa_opt_20261016_spmv_final_v1/`：CSR 5 项、COO 3 项各原 10 矩阵，共 80 条 PASS。CSR FP16/FP16到FP32/FP32乘complex64/int8到FP32/int8到int32 的折算均值分别为 0.810x/0.825x/0.842x/0.887x/0.876x；COO 对应三项仍为 0.679x/0.634x/0.691x，未达标。这些 H800 折算仍是估算。
- `results_musa_opt_20261016_routes_v4/scatter_final.csv`：默认 20 次 warmup、200 次迭代，FP16/int8 × 4 个原向量规模，8/8 PASS；对本机 PyTorch 各行约 0.92x～0.97x，不能与厂商或 H800 折算基线混称。
- SELL 最新定向数据仍有低于 0.8x 的类型，所有慢行原样保留。因此下一轮是阶段性完整复测，**不是全部 65 项达标声明**。
- `tests/data` 的原 10 个文件大小再次核对一致，torch 2.7.1、Triton 3.6.0、MTT S5000、驱动 3.3.5-server，HEAD `4d5ab61`；导入当前 `src/`，`mthreads musa None`。

新目录 `results_musa_split_65_20261016_opt2/`，保留本地源码补丁、未跟踪回归测试补丁和代码 SHA256。
命令如下；是否完成及是否达标必须检查该目录的新结果，不能沿用旧表：

```bash
setsid env PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads MUSA_HOME=/usr/local/musa \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  timeout -s KILL 43200 python3 -u run_flagsparse_split_delivery.py \
  --mode normal --benchmark-input tests/data --timeout 3600 --h800-reference \
  --results-dir results_musa_split_65_20261016_opt2 \
  --capi-bench-out results_musa_split_65_20261016_opt2/capi \
  > results_musa_split_65_20261016_opt2.log 2>&1 < /dev/null &
```

### 2026-10-17 上一轮结果与本轮优化范围

`results_musa_split_65_20261016_opt2/` 已完成，65 项、0 missing，合并精度/性能标签均为 Passed，
但实际仍有 8 个变体均值小于 0.8x：COO FP16到FP32/FP16/int8到int32、SELL FP16/int8到int32/complex64、
AXPBY FP16、SpGEMM FP32。继续按用户要求只优化 MUSA 后端；原始 10 矩阵、容差、迭代次数和基线优先级不变。
下述定向数据不是完整验收结论，尤其不能把 H800 折算值当成本机厂商实测。

### 2026-10-17 src/flagsparse/sparse_operations/vector_ops.py::_musa_axpby
- 改了什么：MUSA FP16/BF16 的中小向量用单个融合 kernel 完成缩放与稀疏更新，缓存有界、版本感知的索引反向映射；值和输出每次重新读取。大向量和其他后端保留原路由。
- 为什么改（附报错原文）：无报错；旧轮 AXPBY FP16 均值为 0.723508x，分开的缩放/更新启动开销较高。
- 怎么验证的：原 4 个向量规模全部 PASS，对本机 PyTorch 平均 1.221323x；原始数据 `results_musa_opt_20261017_layout_v1/axpby_public.csv`。新增 FP16/BF16、索引宽度、输入变更、推理张量、空输入和后端隔离回归，均纳入本轮扩大验证。
- 其他后端是否也需要：否。

### 2026-10-17 capi/flagsparse_codegen/spgemm_csr.py::musa_spgemm_scan_*
- 改了什么：增加 MUSA 分块行计数前缀和、块级前缀和及偏移回填 kernel，同时归约总 nnz、最大行 nnz 与溢出标志。
- 为什么改（附报错原文）：无报错；SpGEMM compute 阶段原先回传每行 counts/overflow 并在主机执行前缀和，旧轮均值为 0.715145x。
- 怎么验证的：MUSA C API 重新构建，accuracy.spgemm 通过，含下述跨块边界回归；原 10 矩阵定向记录 `results_musa_opt_20261017_layout_v1/capi_spgemm/`，6 个规则内可比矩阵对本机 muSPARSE 平均 1.076070x。非方阵、输出预算与厂商精度状态按原规则保留。
- 其他后端是否也需要：否；新 kernel 只由 MUSA 分支调用。

### 2026-10-17 capi/src/ops/spgemm.cpp::musa_build_indptr/compute/copy
- 改了什么：MUSA compute 在设备上构建 indptr，只回传 3 个 int64 元数据；copy 阶段仍按原排序流程回传所需行指针。不缓存输出计数，每次重新执行 count 和 scan。其他后端的 counts/overflow 回传与主机前缀和保持原样。
- 为什么改（附报错原文）：无报错；减少 compute 阶段逐行回传及主机扫描成本。
- 怎么验证的：同上 SpGEMM 原 10 矩阵定向性能和 C API 精度测试；完整 split 仍需重新验收。
- 其他后端是否也需要：否。

### 2026-10-17 capi/ctest/accuracy/test_spgemm.cpp::MusaMultiBlockRowPrefix
- 改了什么：增加 MUSA 专用 1024、1025、3079 行的 FP32/FP64 用例，检查 scan 块边界、部分末块和完整结果。
- 为什么改（附报错原文）：无报错；原有小规模精度用例不能覆盖新增前缀和的跨块路径。
- 怎么验证的：accuracy.spgemm 通过，未更改原有精度判定。
- 其他后端是否也需要：否；其他后端不运行新增 MUSA 专用用例。

### 2026-10-17 src/flagsparse/sparse_operations/_musa_spmv_launch.py::launch
- 改了什么：新增最多 64 项的编译启动器 LRU，只缓存 CompiledKernel 的公开 launcher，不保留输入张量或输出。key 包含 kernel、网格、数据类型、设备、地址对齐、非零指针别名、运行时标量、constexpr 和 kernel.debug；每次执行仍解析当前流并运行 Triton profiling hooks。存在 pre-run hook、运行时 debug、instrumentation 或自定义编译阶段 hook 时回退正常 JIT。
- 为什么改（附报错原文）：无报错；MUSA 短行 SpMV 受 JIT 参数绑定及特化处理的主机开销限制，公开 API 配对测量确认启动器复用有收益。
- 怎么验证的：新增 key 隔离、LRU 淘汰、调试/插桩回退、未对齐连续输入/输出、输出独立性、实时 RHS、非默认流及非 MUSA 路由隔离回归；本轮扩大回归全部通过。
- 其他后端是否也需要：否；模块导入不初始化 MUSA 或导入 Triton knobs，只有 MUSA 路由调用 launch。

### 2026-10-17 src/flagsparse/sparse_operations/mixed_spmx.py::_musa_csr_spmv/_musa_row_tile
- 改了什么：MUSA 实数/整数短行 CSR 和 NNZ 路由复用上述启动器；按平均行长、行数与整数累加选择行块和 slots。普通实数乘复数行 kernel 保持原启动方式。
- 为什么改（附报错原文）：无报错；旧轮 COO 三个半精度/整数变体低于 0.8x，格式转换缓存后的重复 JIT 启动仍占比较高。
- 怎么验证的：最终公开 API 定向复测 `results_musa_opt_20261017_dispatch_v2/coo_public_final.csv`，原 10 矩阵共 30 条精度 PASS；FP16到FP32/FP16/int8到int32 的 H800 折算均值分别为 0.821573x/0.897036x/0.902609x。首项裕量较小，不能以定向数据代替完整 split。
- 其他后端是否也需要：否。

### 2026-10-17 src/flagsparse/sparse_operations/spmv_sell.py::MUSA kernel 启动
- 改了什么：MUSA SELL 复用上述编译启动器；保留已有版本感知计划、高 padding 实数压缩及 dtype/规模分块策略。
- 为什么改（附报错原文）：无报错；旧轮 SELL FP16/int8到int32/complex64 均值为 0.735946x/0.718123x/0.787854x，短 kernel 的主机启动成本占比较高。
- 怎么验证的：`results_musa_opt_20261017_dispatch_v2/sell_public.csv` 原 10 矩阵共 30 条精度 PASS；三项 H800 折算均值为 0.969267x/0.961707x/0.893333x。最终重复测量 `sell_public_final.csv` 仍为 30 条 PASS，均值为 0.981482x/0.984545x/0.911817x。
- 其他后端是否也需要：否。

### 2026-10-17 tests/pytest/test_musa_sparse_routes.py::AXPBY 与启动缓存回归
- 改了什么：新增上述 AXPBY、启动缓存特化/回退、公开 COO/SELL 缓冲区及流切换、后端路由隔离用例；保留之前所有回归。
- 为什么改（附报错原文）：无报错；优化不能依赖输入地址不变、复用旧输出或忽略运行时调试设置。
- 怎么验证的：最终扩大 Python 回归 `879 passed, 1658 skipped`，18.79 秒，`results_musa_opt_20261017_dispatch_v2/accuracy_final.xml`；跳过均沿用已有适用性规则，未放宽容差。C API 交付相关精度族 8/8 passed，39.89 秒，`capi_accuracy.log`；`git diff --check` 通过。
- 其他后端是否也需要：否；此回归文件显式要求 MUSA。

### 2026-10-17 未接入的实验

- MUSA mixed COO 块内 segmented atomic 实验没有公开 API 的整体收益，新增实验 kernel 和 wrapper 已撤回；不改变当前 CSR 缓存路线。
- 裸 kernel 启动探针低估了公开 API 的主机开销，最终采用同一公开 API 的原规模配对测量与仓库原 benchmark 校验收益。
- SELL 的额外 tile 实验没有稳定整体收益，保留原配置；性能与精度原始记录见 `results_musa_opt_20261017_layout_v1/` 和 `results_musa_opt_20261017_dispatch_v2/`。

### 2026-10-17 冻结源码与完整复测

- 新结果目录 `results_musa_split_65_20261017_opt3/`，不复用旧轮；包含源码补丁、未跟踪 helper/回归测试补丁和 SHA256。
- 源码导入路径 `/root/gcx/FlagSparse/src/flagsparse/__init__.py`；HEAD `4d5ab61`；torch 2.7.1、torch_musa 2.7.1+5ee0a64、Triton 3.6.0、MTT S5000，驱动 3.3.5-server，mcc 5.1.0，torch.version.musa=40305。
- 再次核对 `tests/data` 原 10 个矩阵名称与大小一致；固定三个 CPU 线程池为 8。不跳过 C API 构建，不改变原 runner 的规模、容差、迭代次数或基线优先级。
- 这是本轮优化后的完整验收启动记录，不是已完成或全部达标声明。必须等新整轮结束后逐变体检查实际平均加速比及原始精度记录。

```bash
setsid env PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads MUSA_HOME=/usr/local/musa \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  timeout -s KILL 43200 python3 -u run_flagsparse_split_delivery.py \
  --mode normal --benchmark-input tests/data --timeout 3600 --h800-reference \
  --results-dir results_musa_split_65_20261017_opt3 \
  --capi-bench-out results_musa_split_65_20261017_opt3/capi \
  > results_musa_split_65_20261017_opt3.log 2>&1 < /dev/null &
```
