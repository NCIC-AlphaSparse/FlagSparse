# 调试手册：交付变体注册表（现为 105 个），上机要做什么

## 2026-11-02：注册表扩到 105 个变体，各后端复测计划

### 这次改了什么

`conf/operators.yaml` 的 `delivery_variants:` 从 65 个扩到 105 个。新增的 40 个是第三批变体，见
[`../NEW_OPERATORS_BATCH3.md`](../NEW_OPERATORS_BATCH3.md)，按文档顺序排在列表末尾。新增的父算子有两个：
`gemvi`、`geam_csr`。另外有几处改动会影响原有算子的行为：

- **新算子**：`flagsparse_gemvi`（稠密矩阵 × 稀疏向量）和 `flagsparse_geam_csr`（C = αA + βB）。两个都是纯 Triton，
  昇腾上走 torch_npu。
- **混合精度路径支持转置**：`mixed_spmx.py` 新增一组原子散加 kernel，按 `y[col] += v·x[row]` 计算 Aᵀx 和 AᵀB。
  f16/i8 的累加类型是 f32/i32，用到 f32 和 int32 的 `tl.atomic_add`。以前这里会抛 `NotImplementedError`。
- **COO 支持"实矩阵 × 复向量"**：f32 矩阵 × c32 向量，同样走散加 kernel。
- **新增参数**：`spvv`、`spmm_csc`、`sddmm_csr` 加了 `out_dtype`；`spmm_coo` 加了 `op_b`。
- **spgemm 支持 complex64**：哈希路径新增 complex64 填充 kernel，用 TLE 共享内存。不用 TLE 的后端（MUSA、
  Iluvatar、MACA）走 ESC，结果对，但很慢（CUDA 上 ESC 只有 cuSPARSE 的约 0.1 倍）。
- **基准**：
  - 新脚本：`tests/test_gemvi.py`（合成数据）、`tests/test_geam_csr.py`。
  - spgemm 的 c32 行：`tests/test_spgemm.py` 只在包含 float32 的那个进程里追加。
  - 其余新变体：由 `tests/q4_variant_bench.py` 追加。
- **新基线只在 CUDA 上有**：gemvi、csrgeam2、通用 SpGEMM 三个 cuSPARSE 基线加在 `tests/cusparse_generic_baseline.py`。
  其他后端照旧走 `skip_reason()`，跳过厂商列。
- **C API 也补齐了 40 个新变体**（只影响走 split 的 MUSA）：新增 `flagsparseGemvi[_bufferSize]` 和
  `flagsparseCsrgeam2_bufferSizeExt` / `flagsparseXcsrgeam2Nnz` / `flagsparseCsrgeam2` 两组接口，SpMV / SpMM 加了
  混合精度转置、CSC 混合、COO `i8f32` / `f32c32`、bf16、c32 trans/conj_trans、CSC / COO 转置等路由，spvv、axpby、
  spgemm 加了新类型。注册表里 105 个都是 `capi: true`。split 的 C API 基准族多了 `gemvi`、`geam`，自动带上。
  gemvi / csrgeam2 的厂商基线只在 CUDA 上接了，其他后端是 `NoBaseline`。
- **H800 参照只覆盖前 65 个**：新增的 40 个在 H800 上没跑过，所以 `--h800-reference` 补不上它们。一个变体如果在
  本机既没有厂商基线、也没有可用的 PyTorch 基线，交付表里就没有加速比，这是预期的。

### CUDA（RTX 5090）上的结果

结果目录是 `pytest_results_105_cuda_20261102`：10 个交付矩阵，warmup 5、iters 20，耗时 21 分钟。

- 105 个变体都在，0 missing。
- 性能 105/105 Passed。
- 精度 104 个 Passed，1 个 Skipped，是 `sddmm_csr_f32_int_non_non_row`。原因是 Iluvatar 专用测试的跳过原因
  "BI-V layout specialization" 没登记，现在已加进 `EXPECTED_ACCURACY_SKIP_REASONS`，按原产物重新投影是 Passed。
- 加速比（几何平均）：原 65 个 0.991，新 40 个 0.778。0.8 倍以下的变体有 40 个，大多是向量类和 CSC，以及
  混合精度的 COO。

C API split（`results_cuda_split_105_20261102_v2`，同样 10 个交付矩阵，约 1 小时）：

- 精度 105/105 Passed，0 missing；性能 46 Passed、54 NoBaseline、5 Failed。NoBaseline 是因为 C API 没给 axpby、
  spvv、CSC、SELL、混合精度、f16 接厂商基线，原 65 个里同类变体也是这样。
- C API 加速比几何平均：原 65 个 0.834（34 个有基线），新 40 个 1.071（12 个有基线）。
- C API SpMV 原来只有 cuSPARSE 的 0.02–0.4 倍，原因是 CUDA 上每行启动一个 program（耗时随行数涨，不随 nnz）。
  现在 CUDA 的实数 CSR/COO 走 row_tile、c32 在 alpha=1/beta=0 时走 Python 的复数 row_tile，10 矩阵几何平均：
  csr f32 1.17、f64 1.14，coo f32 1.00、f64 1.07，c32 csr 0.77、coo 0.70。MUSA 和其他后端的路由没变。
- 5 个 Failed 是 `capi/tools/write_summary.py` 改了规则之后如实报出来的：一个变体只要有一行没过 C API 的精度门槛，
  性能就记 Failed（和 `run_flagsparse_pytest.py` 的规则一致）。以前这一行被直接丢掉，剩下的行照常算
  NoBaseline。这 5 个都没有厂商基线：`spmv_csc_f32_int_trans`、`spmv_csc_c32_int_conj`、`spmv_csr_f32c32_int_trans`、
  `spmm_csc_f32_int_trans_non_row`（ASIC_680ks）和 `spmm_csc_c32_int_non_non_row`（cfd2）。原子散加的严格误差比
  1.0–1.36，放宽后约 0.01，但规范 6.3.1 的放宽要厂商基线来证明矩阵两边都难，没有基线就只能判失败。累加顺序
  不确定，压线的行每轮会变（`spmm_csc_f32_int_non_non_row` 上一轮失败、这一轮通过）。交付精度（pytest）都是 Passed。
- split 在 CUDA 上不能带 `--h800-reference`（会报 "no card known" 退出），这个参数只给非 CUDA 后端用。
- 105 个变体的列表写进了仓库根目录 `Q4算子列表.xlsx` 的 `delivery_105` 工作表（前 65 个已交付，后 40 个新增）。

在 CUDA 上用 `FLAGSPARSE_BACKEND` 模拟各后端的派发，跑 85 个 q4 变体加上 gemvi/geam 的边界用例：

- MACA、MUSA、Iluvatar 都是 216/216。
- DCU 有 10 个失败，报错都是 `row_tile: unverified capabilities for rocm/120`：8 个是以前记过的那 4 个变体，
  另外 2 个是新加的 `spmv_csr_bf16_int_non`。这是 CUDA 卡上模拟不出 gfx 架构造成的，不是代码 bug。

这些都不是真机结果。

### 所有后端都要注意的

1. **`--ops` 写死的命令要加上 `gemvi,geam_csr`**。DCU、MACA、Iluvatar 原来的命令列的是 13 个算子，漏了这两个，
   对应的 5 个变体会变成 `NotFound`。各后端文件里的新命令已经加好了。昇腾的命令和 MUSA 的 split 本来就带
   `--delivery-only`，会自动跟上。
2. **结果目录用新名字**（`results_<后端>_105_<日期>`），不要往 65 变体的旧目录里补跑。
3. **新代码第一次在真机上跑**，最该先看的是：
   - 原子散加 kernel：f32 和 int32 的 `tl.atomic_add`。
   - spgemm c32 在 ESC 路径上用到的复数 torch 运算。
   - 3 个 bf16 变体：`spmv_csr_bf16_int_non`、`spmv_csr_bf16f32_int_non`、`spmm_csr_bf16f32_int_non_non_row`。
4. 跑完把 `python3 tools/delivery_table.py <结果目录> --show-baseline` 的完整输出贴回来。每个后端文件的
   "2026-11-02 复测计划"一节列了这个后端要单独确认的项目。

### 各后端的复测计划

| 后端 | 文件 | 这轮最主要的风险或缺口 |
|---|---|---|
| 海光 DCU | [`DCU.md`](DCU.md) | 新变体都没有厂商基线（hipSPARSE 没接），只有 PyTorch 列；bf16 走 row_tile |
| 沐曦 MACA | [`MACA.md`](MACA.md) | 没有厂商库；spgemm c32 走 ESC；SDDMM 的 PyTorch 参考会被校验拦下 |
| 摩尔 MUSA | [`MUSA.md`](MUSA.md) | split 的性能取自 C API；40 个新变体的 C API 路由是 11-02 新加的，第一次在 MUSA 上跑；C API 没测到的由 Python 那一半补位（要带 `--h800-reference`）；gemvi/geam 没有 muSPARSE 基线 |
| 天数 Iluvatar | [`ILUVATAR.md`](ILUVATAR.md) | bf16 支持未知；spgemm c32 走 ESC；`torch.sparse` 会静默算错 |
| 昇腾 Ascend | [`ASCEND.md`](ASCEND.md) | `ASCEND_Q4_VARIANTS` 没扩：16 个变体没有精度结果，约 34 个没有性能结果；gemvi/geam 没有性能命令 |

## 背景（2026-10-08，CUDA 机器上完成）

`conf/operators.yaml` 的 `delivery_variants:` 从 20 项合并成了 65 项（原 20 个交付变体 + q4 的 45 个
混合精度/int8/转置/布局变体），不再分"交付"和"q4"两部分。一条 runner 命令能跑全部 65 个：

```bash
python3 run_flagsparse_pytest.py \
  --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc,spgemm_csr,sddmm_csr \
  --phase both --benchmark-input tests/data
```

跑完用 `python3 tools/delivery_table.py <结果目录>` 看汇总，第一行就是 `N registered variants, M
missing`，`M` 应该是 0。详细机制（`capi` 字段怎么算的、H800 无基线判定、两个之前的 crash bug）见
`flagsparse-65-unified-registry` 和 `ascend-45-variant-wiring` 两条记忆，这里不重复。

**这一轮全程只在 CUDA（RTX 5090）机器上做的**：65/65 accuracy Passed、65/65 performance Passed、0
missing，真实矩阵（`tests/data` 10 个交付矩阵，warmup 5、iters 20）。**MACA / MUSA / DCU / Iluvatar /
Ascend 这 5 个后端，没有一个在真机上跑过这条命令**——下面每个文件回答"这个后端具体要验证什么、已知
风险点是什么、跑完要带回什么"。昆仑芯（XPU）不在这轮范围内，没有单独文件。

## 分工（和 `docs/<BACKEND>.md` 不重复）

- **[`LOW_SPEEDUP.md`](LOW_SPEEDUP.md)**：加速比低于 0.8 的变体的现状、和 H800 自身比值的对照、按优先级的优化建议（2026-10-24，基于天数结果），天数的实测调优记录，以及下一轮天数优化方向（2026-10-25：转置长尾、COO 混合精度、SpGEMM、SELL 等 7 个方向）。
- **[`ILUVATAR_SELL_COO_PLAN.md`](ILUVATAR_SELL_COO_PLAN.md)**：天数 SpMV SELL / COO 的详细优化计划（2026-10-25）：先用 5 个探针定位瓶颈，再在 prepare 里重编码成 SELL-C-σ 或 nnz 均分格式，含逐项预期、排期和验收。

- `docs/<BACKEND>.md`（原有）：环境怎么搭、交付怎么跑、这台卡的已知限制——背景资料，不随这次改动变化。
- `docs/debug/<BACKEND>.md`（本目录）：**这次"20→65合并"这件事**，该后端还缺什么验证、要上机做什么、
  带回来的数据怎么判断对不对。只管这一件事，不要在这里重复 `docs/<BACKEND>.md` 已经写过的环境搭建内容。
- **交给 Codex 的 prompt 也在这里**：每个文件末尾"交给 Codex 的 prompt"一节，后端名和路径已经填好，
  整段贴给代理就能开工（DCU 要先换掉 `<卡号>`）。
- **改动台账也在这里**：每个文件末尾的"实机改动记录"一节，取代了 2026-10-08 停用的 `modified/` 目录。
  在实机上改了任何文件，都按那一节的格式追加一条，和代码改动放在同一个 commit 里推上来。

## 四个后端共同的任务（MACA / MUSA / DCU / Iluvatar）

这四个后端走的是**同一套机制**：和 CUDA 共用 `test_spmv_csr.py` 等脚本（`FLAGSPARSE_BACKEND` 切换
dispatch），q4 变体的厂商基线测量（`tests/q4_variant_bench.py`）会先查
`tests/cusparse_generic_baseline.py` 的 `skip_reason()`，非 CUDA 后端按设计应该优雅跳过厂商基线、不
崩溃。四个文件内容结构一样，只是各自的已知风险点不同，见各自文件。

**2026-10-08 补充了一轮 CUDA 上的后端模拟验证**（设 `FLAGSPARSE_BACKEND=<metax|mthreads|rocm|
iluvatar>` 环境变量，让 Triton 内核在这台 CUDA 卡上真实按对应后端的分支跑，再对 CPU golden 校验 45
个 q4 变体）：**MACA / MUSA / Iluvatar 三个 91/91 全过**，证明分支调度的逻辑本身（不含性能调优、不
含真实硬件能力）是对的。**DCU(ROCm) 有 4 个变体（`spmv_csr_f16_int_non`/`c32_int_non`/
`f32_int_trans`/`c32_int_conj`）在模拟下报错**，但追到根因是 ROCm 专属的 `row_tile` 算法要求
Triton 驱动真实报出 `backend=="hip"`/`arch` 以 `"gfx"` 开头——这个值绑定真实硅片，CUDA 卡上无论怎么
设环境变量都测不出来，是模拟方法本身的结构性盲区，不是这次合并引入的代码 bug；补测时把能力门槛
改成 gfx936 的已验证值，让 `row_tile` 真实执行，91/91 全过（细节和"如果真机也报这个错该怎么办"见
[`DCU.md`](DCU.md) 对应小节）。四个后端模拟下的全量 `tests/pytest` 里，65 变体涉及的 13 个算子
全部通过；DCU 模拟另有 `spsv_coo`/`spsv_csr` 的复数用例失败，SpSV 不在 65 个变体内、本次也没改动它。**这仍然都不是真机实测**——只是比纯代码审查更强的一
层证据（内核真的执行了，不只是读代码猜）；真机验证的要求不变，见下面各文件。

**共同的验证步骤**：

1. 照 `docs/<BACKEND>.md` 把环境配好、过一遍环境自检。
2. 跑本目录对应文件里给的命令。**不要直接用上面那条通用命令**：各后端文件里的版本在它的基础上补了这张卡
   必需的参数（外层 KILL 限时、`--timeout`、MACA 的 `sddmm_csr=--no-cusparse`、天数挡 fp64 的 `-k` 和
   各脚本的 dtype 限制），都沿用自各自 `docs/<BACKEND>.md` 里已经实机跑通的 20 变体交付命令。
3. `python3 tools/delivery_table.py <结果目录>`，把完整输出（尤其是不是 `0 missing` 这一行）贴回来。
4. 如果有 `FAIL` 或非 0 的 `missing`，把对应变体名和 `<结果目录>/<算子>/performance.csv`（或
   `accuracy_result.json`）里那一行的 `reason`/`error` 字段一起带回来，不要只说"跑挂了"。

## spmv_csr 交付加速比的口径（2026-10-08 修正，影响所有后端）

DCU 实测时发现：runner 给 spmv_csr 跑的是 `--alg compare`，7 种算法各写一行，交付表把它们**算术平均**，
不是调用方实际走的那条路径。修正前的数字：BW1000 报 `0.207x`/`0.236x`（生产路径 `row_tile` 实为
`0.738x`/`0.795x`），CUDA 报 `0.379x`/`0.423x`（生产路径 `legacy_segbin` 重跑为 `0.912x`/`0.827x`）。
runner 现在默认 `--alg auto`，交付投影只认 `alg_requested=auto` 的行，有失败行时变体判 FAIL（以前会被
平均值掩盖）。**修正前各后端交付报告里的 `spmv_csr_f32/f64_int_non` 加速比都不能直接用**，要么重跑，要么
从原始 `performance.csv` 里只取各自生产路径那个算法的行。其余 63 个变体不受影响（CUDA 原始结果按新规则
重新投影，只有这两个变了）。

## C API 现在覆盖全部 65 个变体（2026-10-08）

`origin/q4` 在 MUSA 上做完并实测过的 C API 工作（45 个新变体、muSPARSE 基线扩展、MUSA 专用 gather 路径）已经
移植进 main，CUDA 上端到端跑通：split 一次出全部 65 个，精度 65/65，性能 34 个有 cuSPARSE 加速比、31 个
`NoBaseline`。这对 MUSA 最重要（交付性能取自 C API），细节、MUSA 专属未在 CUDA 上测过的路径、以及 q4 当时的
MUSA 实测表见 [`MUSA.md`](MUSA.md)。另外注意口径变化：`spmm_csr/coo_f32/f64_int_non_non_row` 在 C API 里以前
其实测的是列主序，现在按名字测行主序，数字会变。

## 昇腾（Ascend）的任务不一样，范围和工作量都更大

昇腾这边今天已经做了一部分代码级别的准备工作（`axpby`/`spvv`/`spmv_sell` 的路由、修了一个真实的
`spmv_coo` 共轭 bug、把 45 个变体的内核派发 correctness 在 CUDA 上模拟验证了一遍），但还有明确的、
分优先级的剩余工作需要在真实 NPU 上做，不是简单"跑一遍命令看结果"——见 `docs/debug/ASCEND.md`，
这是本轮剩余工作量最大的一个文件，建议先看。
