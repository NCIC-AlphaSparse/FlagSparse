# 昇腾 910B：65 变体剩余工作（实机上做）

`FLAGSPARSE_BACKEND=ascend`。环境搭建、交付复现见 [`../ASCEND.md`](../ASCEND.md)。背景见本目录
[`README.md`](README.md)。

**这个文件和另外四个不一样**：MACA/MUSA/DCU/Iluvatar 是"机制应该已经通了，上机跑一遍确认"；昇腾这边
今天已经做了一部分代码级别的工作，但还有明确分三档、工作量差异很大的剩余任务，不是跑一条命令就能
验完的。`../ASCEND.md`"现在是 65 个变体"一节有同样内容的简版，这里是给要上机干活的人看的任务清单。

## 今天（2026-10-08）在 CUDA 机器上做了什么

1. `axpby`/`spvv`/`spmv_sell` 接上了路由——复用 CUDA 那套脚本（这三个脚本本身 backend-neutral），
   `ASCEND_PERFORMANCE_COMMANDS`/`OP_TEST_CONFIGS` 各加了一行。**没有在真实 NPU 上跑过**。
2. 用 `tools/q4_ascend_dispatch_check.py`（CUDA 上强制 `_is_ascend_runtime()->True`，Triton 启动直接
   报错，逐个跑 45 个变体对 CPU golden）验证了内核派发 correctness，**38/45 OK**。
3. 过程中发现并修了一个真实 bug：`spmv_coo_c32_int_conj` 在模拟昇腾派发下算错了（`spmv_coo.py` 的
   `_resolve_spmv_coo_launch` 跳过共轭的条件没考虑昇腾分支不走 csr_plan），已修复并三方交叉验证过
   （dispatch check、CUDA 真实精度、`tests/ci/test_spmv_coo_ascend_fallback.py` 的 A/B）。**修复本身
   也只在 CUDA 模拟下验证过，没有真实 NPU。**

## 65 个变体的精确分类（不是估计，是跑出来的数据）

| 类别 | 数量 | 状态 |
|---|---:|---|
| 原有变体 | 20 | 2026-09-23 已在 910B4 实机跑过 20/20 PASS，不受今天改动影响 |
| 今天接上路由 | 8 | 路由通了，**从未上过真机** |
| 内核派发对、性能没接 | 30 | dispatch correctness 已模拟验证，但 `benchmark_ascend.py` 不知道这些变体 |
| 没有昇腾分支 | 7 | 模拟派发下走 Triton，CANN 大概率编不出来，**可能直接崩溃，不只是测不出数字** |

<details>
<summary>展开看每一类具体是哪些变体 id</summary>

**今天接上路由（8 个）**：
`axpby_f16_int`、`spvv_c32_int_conj`、`spvv_f16f32_int_non`、`spvv_i8i32_int_non`、
`spmv_sell_c32_int_non`、`spmv_sell_f16_int_non`、`spmv_sell_f32_int_non`、`spmv_sell_i8i32_int_non`

**内核派发对、性能没接（30 个）**：
`scatter_i8_int`、
`spmv_csr_c32_int_conj`、`spmv_csr_c32_int_non`、`spmv_csr_f16_int_non`、`spmv_csr_f16f32_int_non`、
`spmv_csr_f32_int_trans`、`spmv_csr_f32c32_int_non`、`spmv_csr_i8f32_int_non`、`spmv_csr_i8i32_int_non`、
`spmv_coo_c32_int_conj`、`spmv_coo_c32_int_non`、`spmv_coo_f16_int_non`、`spmv_coo_f16f32_int_non`、
`spmv_coo_f32_int_trans`、`spmv_coo_i8i32_int_non`、
`spmm_csr_c32_int_non_non_row`、`spmm_csr_f16_int_non_non_row`、`spmm_csr_f16f32_int_non_non_row`、
`spmm_csr_f32_int_non_non_col`、`spmm_csr_f32_int_non_trans_row`、`spmm_csr_f32_int_trans_non_row`、
`spmm_csr_i8i32_int_non_non_row`、
`spmm_coo_c32_int_non_non_row`、`spmm_coo_f16_int_non_non_row`、`spmm_coo_i8i32_int_non_non_row`、
`sddmm_csr_c32_int_non_non_row`、`sddmm_csr_f16_int_non_non_row`、`sddmm_csr_f32_int_non_non_col`、
`sddmm_csr_f32_int_non_trans_row`、`sddmm_csr_f32_int_trans_non_row`

**没有昇腾分支（7 个）**：
`spmv_csc_f32_int_non`、`spmv_csc_c32_int_non`、`spmv_csc_f16_int_non`、
`spmm_csc_f32_int_non_non_row`、`spmm_csc_c32_int_non_non_row`、`spmm_csc_f16_int_non_non_row`、
`spgemm_csr_f32_int_non_non`

重新生成这张表（比如又修了几个变体之后）：`PYTHONPATH=src python3 tools/q4_ascend_dispatch_check.py`。

</details>

## 上机要做的事，按优先级（工作量从小到大）

### 优先级 1：验证今天接上路由的 8 个变体（预计几分钟）

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export PYTHONPATH="$PWD/src:$PWD" FLAGSPARSE_BACKEND=ascend FLAGSPARSE_ASCEND_VENDOR=torch
python3 run_flagsparse_pytest.py --ops axpby,spvv,spmv_sell --phase both \
  --benchmark-input tests/data --results-dir results_ascend_newops_<日期>
python3 tools/delivery_table.py results_ascend_newops_<日期>
```

预期：8 行全部 accuracy Passed（精度走的是跟 CUDA 一样的通用 pytest 路径，理论上不需要额外适配）；
performance 走没走通不确定——这三个脚本在昇腾上是不是真的能正常起 PyTorch-NPU 计时，今天没有办法
验证。把结果（尤其是不是报错、报了什么错）带回来。

### 优先级 2：给 30 个"派发对但没测性能"的变体接上 runner（预计数天，需要能跑通再改）

`benchmark_ascend.py`/`benchmark_ascend_probe.py` 需要仿照 CUDA 那 7 个脚本
（`test_spmv_csr.py` 等）的 `--q4-variants` 模式，接一条调用 `tests/q4_variant_bench.py` 的路由——但
**`tests/q4_variant_bench.py` 本身默认是 CUDA/CuPy 路径**（`_time_cupy` 直接 `import cupy as cp`），
在接线之前要先确认：

1. 昇腾上有没有可用的厂商稀疏库对照（`../ASCEND.md` 里没有提到有一个类似 cuSPARSE 的东西，大概率没有，
   这 30 个变体的性能列预期只能有 PyTorch-NPU 对照，没有厂商列）；
2. `q4_variant_bench.py` 的 `_time_pytorch`/数据生成部分是不是 backend-neutral 的（理论上是，用的是
   `torch`/`accelerator_device()` 这层抽象，没有硬编码 CUDA，但没有在昇腾上跑过，需要先用一两个变体
   冒烟测试确认）。

这块工作量和风险都明显高于优先级 1，**不要在没有 NPU 实测反馈的情况下一次性接完所有 30 个**——先挑
1-2 个（比如 `spmv_csr_f16_int_non`，dtype 和 op 轴都最简单）跑通，确认整条链路没问题，再铺开。

### 优先级 3：给 `spmv_csc`/`spmm_csc`/`spgemm_csr` 写昇腾分支（预计最久，可能需要厂商配合）

这不是"接个 runner 路由"能解决的，是要给这三个算子写出 CANN 能编译的 kernel 或者 torch_npu 回退实现，
参照 `../ASCEND.md`"Ascend fallback 分发表"一节里 `spsm_csr`/`spmm_coo` 已有的两个回退模式（逐行
sweep、单次 `index_add`）。涉及对 CSC 格式和 SpGEMM 哈希表算法在 NPU 上怎么落地的判断，建议先用
`tools/q4_ascend_dispatch_check.py` 里报错的具体内核名（`_spmv_csc_non_real_kernel`、`_gather`、
`_csc_spmv_mixed_kernel`、`_spmm_csc_non_real_kernel`、`_spmm_csc_non_complex_kernel`、
`_spgemm_hash_count_kernel`）去对应到 `src/flagsparse/sparse_operations/` 里的具体函数，判断能不能
复用已有的 `_is_ascend_runtime()` 分支模式，还是需要全新设计。

## 带回来的东西

不管做到哪个优先级，至少带回：

1. 优先级 1 的 `delivery_table.py` 输出（8 行）。
2. 如果做了优先级 2，每个新接上的变体第一次在真实 NPU 上跑出的 `performance.csv` 原始行（不是口头
   "能跑"，要有实际的 `vendor_ms`/`pytorch_ms`/`status` 字段）。
3. 如果发现 `tools/q4_ascend_dispatch_check.py` 的某个判断在真实 NPU 上不成立（比如它判 OK 但真机
   报错，或者判 TRITON 但其实某个新版本 CANN 能编过），这个工具本身需要更新——它是 CUDA 模拟，不是
   NPU 本身的真值。

## 交给 Codex 的 prompt

上机时把下面整段原样贴给 Codex（或其他代理），不用改任何内容。

````text
你在昇腾 910B 实机上做 FlagSparse 65 变体的剩余工作。仓库在当前目录，main 分支。
git pull --ff-only origin main 之后，完整读一遍 docs/debug/ASCEND.md，按里面的优先级做：

1. 先跑 PYTHONPATH=src python3 tools/q4_ascend_dispatch_check.py。它的结论是在 CUDA 上模拟出来的
   （38/45 OK），本机结果如果不一样（判 OK 但真机报错，或者判 TRITON 但真机能编过），本身就是重要发现。
2. 做优先级 1（8 个变体，命令在文档里），带回 delivery_table.py 的输出。
3. 优先级 2 只挑 spmv_csr_f16_int_non 一个变体试着接通，接通后先停下来汇报，不要一次接完 30 个。
4. 优先级 3 不做。

规矩：
- 不要为了"让它通过"去改 src/ 或 tests/ 下的代码。确实要改（优先级 2 本身就要改 benchmark_ascend.py）
  时，每改一个文件，都按 docs/debug/ASCEND.md 末尾"实机改动记录"一节的格式追加一条记录，并和代码改动
  放在同一个 commit 里。没改就在那一节写"无"。
- 不要猜根因。遇到报错就给出完整报错原文。
- --results-dir 用新目录，不要复用旧目录。不要用 pytest --forked。

汇报包含：
1. q4_ascend_dispatch_check.py 的完整输出。
2. 优先级 1 的 delivery_table.py 输出（8 行），以及非 Passed 变体的 reason/error 原文。
3. 如果做了优先级 2：那个变体第一次在真机上跑出的 performance.csv 原始行。
4. 环境指纹：torch、torch_npu、CANN 版本，设备名。
5. docs/debug/ASCEND.md "实机改动记录"一节的内容。
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
