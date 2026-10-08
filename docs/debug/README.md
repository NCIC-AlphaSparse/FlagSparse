# 调试手册：65 变体统一注册表，上机要做什么

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

## MUSA 还有一个开发任务

MUSA 的交付性能取自 C API（对 muSPARSE），而 C API 目前只覆盖 65 个变体里的 25 个。`MUSA.md` 的"第二阶段"
一节是让其余 40 个也进 C API 性能统计的开发任务（大部分可以从 `origin/q4` 分支移植），有单独的 prompt。
它改的是所有后端共用的 C API 分发层，推上来之后要在 CUDA 机器上重新编译、跑一遍 `ctest -L capi`。

## 昇腾（Ascend）的任务不一样，范围和工作量都更大

昇腾这边今天已经做了一部分代码级别的准备工作（`axpby`/`spvv`/`spmv_sell` 的路由、修了一个真实的
`spmv_coo` 共轭 bug、把 45 个变体的内核派发 correctness 在 CUDA 上模拟验证了一遍），但还有明确的、
分优先级的剩余工作需要在真实 NPU 上做，不是简单"跑一遍命令看结果"——见 `docs/debug/ASCEND.md`，
这是本轮剩余工作量最大的一个文件，建议先看。
