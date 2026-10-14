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

> **2026-10-11 当前状态**：上表是 10-08 的历史快照。现在 45 个 q4 变体的 Ascend
> dispatch 检查为 `45/45`；q4 benchmark 中除 Blocked-ELL 外的 35 个变体已有显式性能路由，
> CSC 的 6 个变体和本批其余变体已在 NPU 7 的 `wave.mtx` 上通过。原先 7 个无分支算子中的
> CSC 6 个和 SpGEMM 已增加 Ascend-only torch fallback；Blocked-ELL 仍由 capability probe 覆盖。

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

### 先做：65 变体全量重测（带 `--h800-reference`）

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export PYTHONPATH="$PWD/src:$PWD" FLAGSPARSE_BACKEND=ascend FLAGSPARSE_ASCEND_VENDOR=torch
setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --phase both --mode normal --delivery-only --gpus 6,7 --timeout 3600 \
  --benchmark-input /home/matrix --benchmark-warmup 5 --benchmark-iters 20 \
  --h800-reference \
  --results-dir results_ascend_65_<日期> \
  > results_ascend_65_<日期>.log 2>&1 < /dev/null &
# 跑完后
python3 tools/delivery_table.py results_ascend_65_<日期>
```

- 和 `pytest_results_ascend_full_20261011` 那轮是同一条命令，只多了 `--h800-reference`；`--results-dir` 用新目录。
- `--h800-reference`（不带值）：没有 PyTorch-NPU 基线的行（10-11 那轮 25 个）用仓库自带的 H800 结果折算，交付表里标
  `(H800折算)` / `(H800 PyTorch折算)`，有 PyTorch-NPU 基线的行不受影响。卡型自动识别为 `ascend-910b`，日志里应出现
  `ascend-910b`；没有就加 `--vendor-card ascend-910b`。口径见 [`../H800_REFERENCE.md`](../H800_REFERENCE.md)。
- 这轮要确认的三件事：
  1. `spgemm_csr` 的性能现在走 `tests/test_spgemm.py`（交付矩阵、按 dtype 分进程），**第一次在 NPU 上跑**。日志里
     应有 `performance_float32.csv` / `performance_float64.csv`，`spgemm_csr_f32_int_non_non` 应有加速比；
     报错就贴 `spgemm_csr/performance_stderr.log` 原文。
  2. `scatter_i8_int` 应在全量里出现（10-11 那轮是 NotFound，当时 dtype 列表还没 int8；v2 单跑是 0.999x）。
  3. 折算补上的变体预演出来大多是 0.001x–0.06x：这些 q4 路径每次调用 0.3–2.4 秒（`timed=call`）。这是真实性能，
     回传时照实报，并附上 `delivery_table.py` 全表。

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

### 2026-10-10 src/flagsparse/sparse_operations/spmv_sell.py::csr_to_sell
- 改了什么：Ascend complex SELL 转换改为对实部/虚部浮点平面执行 `index_copy_`，再恢复 complex tensor。
- 为什么改（附报错原文）：`RuntimeError: aclnnIndexPutImpl failed, error code is 161002; Tensor self not implemented for DT_COMPLEX64`。
- 怎么验证的：在 910B4 上复现后，用 complex64 NPU tensor 验证浮点平面 `index_copy_` 路径。
- 其他后端是否也需要：否

### 2026-10-10 tests/test_spmv_sell.py::_variant_name/main error row
- 改了什么：正常行和异常行统一写入 `spmv_sell_*` variant id。
- 为什么改（附报错原文）：complex64 异常行的 `variant` 为空，导致交付投影误报 Passed；原始算子错误同上一条。
- 怎么验证的：用已有 Ascend performance.csv 复现了 c32 错误行未被 variant 投影选中的问题。
- 其他后端是否也需要：是

### 2026-10-10 run_flagsparse_pytest.py::Ascend route/delivery performance projection
- 改了什么：Ascend spmv_csr 先追加 `spmv_csr_f16_int_non`；交付投影同时识别 `dtype` 和 `value_dtype`。
- 为什么改（附报错原文）：SELL c32 错误行使用 `value_dtype` 时未被识别；CSR q4 性能此前没有 runner 入口。
- 怎么验证的：新增 value_dtype ERROR 投影回归测试，并在 910B4 上跑单变体 runner。
- 其他后端是否也需要：投影修复需要；Ascend 路由不需要

### 2026-10-10 tests/q4_variant_bench.py::_spmv/run_variants/run_and_append
- 改了什么：Ascend SpMV q4 使用 direct-call 和 `index_add_` PyTorch-NPU 基线，并支持只运行指定 variant。
- 为什么改（附报错原文）：`NotImplementedError: Ascend fallback does not accept prepared SpMV metadata`。
- 怎么验证的：在 910B4 上对 `spmv_csr_f16_int_non`、wave.mtx 做单变体冒烟，得到 FlagSparse/PyTorch-NPU 实际时延和 PASS 状态。
- 其他后端是否也需要：否

### 2026-10-10 benchmark/benchmark_ascend.py::main
- 改了什么：增加 `--q4-variants`，把选定 q4 结果追加到 Ascend CSV。
- 为什么改（附报错原文）：Ascend 专用 benchmark 此前完全不知道 q4 变体，runner 无法生成对应性能行；`tests/` 路径只在导入时临时加入，避免 `AttributeError: module 'pytest' has no attribute 'mark'`。
- 怎么验证的：通过统一 runner 在 910B4 上生成带 variant 的原始 CSV 行。
- 其他后端是否也需要：否

### 2026-10-10 tests/ci/test_delivery_parent_failure.py::test_classic_value_dtype_error_fails_the_matching_variant
- 改了什么：增加 `value_dtype` 错误行不得被汇总为 Passed 的回归测试。
- 为什么改（附报错原文）：`spmv_sell_c32_int_non` 的十个 ERROR 行被 `delivery_table.py` 误报 Passed。
- 怎么验证的：运行定向 pytest。
- 其他后端是否也需要：是

### 2026-10-10 benchmark/benchmark_ascend_accuracy.py::main/_run_q4_variants
- 改了什么：专用 Ascend accuracy artifact 追加 runner 指定的 q4 变体和两个标准 shape。
- 为什么改（附报错原文）：`spmv_csr` 被专用 accuracy probe 路由后只产生旧 dtype 行，`spmv_csr_f16_int_non` 被汇总为没有 accuracy artifact。
- 怎么验证的：统一 runner 的结果中该变体 accuracy 为 Passed，且 artifact 含两个精确 variant nodeid。
- 其他后端是否也需要：否

### 2026-10-10 tests/ci/test_spmv_sell_ascend_conversion.py::test_ascend_complex_csr_to_sell_avoids_complex_index_put
- 改了什么：增加 Ascend complex CSR-to-SELL 转换的回归覆盖。
- 为什么改（附报错原文）：`Tensor self not implemented for DT_COMPLEX64`。
- 怎么验证的：CPU 强制 Ascend 分支检查 SELL values/columns/offsets。
- 其他后端是否也需要：否

### 2026-10-10 tests/ci/test_ascend_reporting.py::test_ascend_csr_bringup_routes_one_q4_variant_to_both_phases
- 改了什么：固定当前只接一个 CSR q4 变体，并检查性能与精度入口均存在。
- 为什么改（附报错原文）：避免未实测时误把其余 29 个变体一次性接入。
- 怎么验证的：运行定向 pytest。
- 其他后端是否也需要：否

### 2026-10-10 run_flagsparse_pytest.py::ASCEND_Q4_VARIANTS spmv_coo
- 改了什么：在已有 Ascend q4 性能入口上新增 `spmv_coo_f16_int_non`，精度仍沿用通用 pytest。
- 为什么改（附报错原文）：该变体此前没有性能 runner 映射，汇总为 `the benchmark recorded no spmv_coo_f16_int_non rows`。
- 怎么验证的：在 910B4 上通过统一 runner 对 wave.mtx 跑 accuracy/performance，原始 CSV 含 FlagSparse/PyTorch-NPU 时延和 PASS 状态。
- 其他后端是否也需要：否

### 2026-10-10 tests/ci/test_ascend_reporting.py::test_ascend_coo_bringup_routes_one_q4_variant_to_performance
- 改了什么：固定 COO 当前只接一个 q4 性能变体，并确认 accuracy 保持通用 pytest 路径。
- 为什么改（附报错原文）：防止后续路由修改丢失该变体或错误切换 accuracy producer。
- 怎么验证的：运行定向 pytest。
- 其他后端是否也需要：否

### 2026-10-10 tests/pytest/test_spmv_coo_accuracy.py::test_spmv_coo_int64_auto_fallback_to_int32
- 改了什么：Ascend 上跳过只验证 Triton int64 到 int32 自动重试的测试。
- 为什么改（附报错原文）：Ascend 的公开 API 使用 `index_add_` 分支，不调用被测试 monkeypatch 的 Triton 内核，原断言报 `assert False`。
- 怎么验证的：在 910B4 上通过统一 runner 运行 `spmv_coo` accuracy。
- 其他后端是否也需要：否

### 2026-10-10 tests/ci/test_spmv_coo_ascend_fallback.py::test_ascend_skips_the_triton_int64_retry_test
- 改了什么：增加 CPU 模拟 Ascend 派发，确认 Triton 专属回退测试会以明确原因跳过。
- 为什么改（附报错原文）：防止后续删除后端限定后再次出现 `assert state["forced_once"]` 失败。
- 怎么验证的：运行定向 pytest。
- 其他后端是否也需要：否

### 2026-10-10 run_flagsparse_pytest.py::ASCEND_Q4_VARIANTS spmv_csr/spmv_coo measured batch
- 改了什么：将真实 NPU 冒烟通过的 10 个变体接入 Ascend q4 accuracy/performance 路由：CSR 7 个、COO 3 个。
- 为什么改（附报错原文）：这些变体此前仍没有性能 runner 行，汇总会报 `the benchmark recorded no ... rows`。
- 怎么验证的：910B4 NPU 7 直接运行 q4 benchmark，10/10 `PASS`；没有修改算子 kernel。
- 其他后端是否也需要：否

### 2026-10-10 tests/ci/test_ascend_reporting.py::test_ascend_*_bringup_routes_measured_q4_variants
- 改了什么：将路由回归从单个变体更新为精确的 CSR 7 个、COO 4 个已测清单。
- 为什么改（附报错原文）：防止批量接线时漏掉变体或意外扩大未实测范围。
- 怎么验证的：运行定向 pytest。
- 其他后端是否也需要：否

### 2026-10-10 src/flagsparse/sparse_operations/spmv_csr.py::_ascend_csr_row_ids
- 改了什么：移除 Ascend CSR `row_ids` 的 data pointer 缓存，按当前 `indptr` 结构重新生成。
- 为什么改（附报错原文）：临时 CSR 张量复用地址后拿到旧 row ids，报 `The size of tensor a (...) must match the size of tensor b (...)`；性能直调通过、专用 accuracy 重复失败。
- 怎么验证的：CPU 回归验证不同 nnz 结构的 row ids；NPU q4 accuracy 复跑受影响的转置/共轭变体。
- 其他后端是否也需要：否

### 2026-10-10 tests/ci/test_spmv_coo_ascend_fallback.py::test_ascend_csr_row_ids_follow_each_current_structure
- 改了什么：增加 CSR Ascend row-id 结构变化回归测试。
- 为什么改（附报错原文）：防止 allocator 地址复用重新引入 stale row ids。
- 怎么验证的：运行定向 pytest。
- 其他后端是否也需要：否

### 2026-10-10 tests/q4_variant_bench.py::_spmm/_sddmm Ascend raw-call routing
- 改了什么：Ascend q4 的 CSR SpMM/SDDMM 直接调用公开 fallback，避免先构造不支持的 Triton prepared kernel。
- 为什么改（附报错原文）：`ModuleNotFoundError("No module named 'shmem'")` 和 `ConvertTritonIRToLinalgIR ... PassManager::run failed`；混合精度路径通过而非混合路径在 prepared 阶段失败。
- 怎么验证的：在 910B4 NPU 7 上重新运行本批 SpMM/SDDMM q4 冒烟。
- 其他后端是否也需要：否

### 2026-10-10 run_flagsparse_pytest.py::ASCEND_Q4_VARIANTS spmm/sddmm measured batch
- 改了什么：接入本批真实 NPU 通过的 10 个变体：CSR SpMM 6 个、COO SpMM 3 个、CSR SDDMM 1 个。
- 为什么改（附报错原文）：此前这些变体没有 q4 runner 行，汇总为 `the benchmark recorded no ... rows`。
- 怎么验证的：910B4 NPU 7 上 10/10 q4 行为 `PASS`；Ascend fallback 改动仅限 Ascend 分支。
- 其他后端是否也需要：否

### 2026-10-11 src/flagsparse/sparse_operations/spmm_csr.py::_ascend_csr_row_ids
- 改了什么：移除 Ascend CSR SpMM fallback 的 data pointer row-id 缓存，按当前 `indptr` 结构重新生成。
- 为什么改（附报错原文）：第二个 q4 accuracy shape 复用旧 row ids，报 `index_add_(): Number of indices (...) should be equal to source.size(dim): (...)`。
- 怎么验证的：CPU 回归覆盖 SpMV/SpMM 两个 helper；NPU 复跑失败的 3 个 SpMM accuracy case。
- 其他后端是否也需要：否

### 2026-10-11 src/flagsparse/sparse_operations/spmv_csc.py::_ascend_spmv_csc_torch
- 改了什么：为 Ascend CSC SpMV 增加 torch_npu `repeat_interleave`/`index_add_` fallback，覆盖 non/trans/conj 和复数分量安全路径；Ascend prepare 不再调用 Triton row-id kernel。
- 为什么改（附报错原文）：`_spmv_csc_non_real_kernel`、`_gather` 和 `_csc_spmv_mixed_kernel` 在当前 CANN lowering 路径失败；prepare 阶段还会触发 `TritonToLinalgIncubated` 的 `_row_ids_kernel` 编译失败。
- 怎么验证的：CPU 模拟 Ascend 覆盖 float32/float64/complex64 与三种 op，定向 CI 10/10 通过。
- 其他后端是否也需要：否

### 2026-10-11 src/flagsparse/sparse_operations/spmm_csc.py::_ascend_spmm_csc_torch
- 改了什么：为 Ascend CSC SpMM 增加 torch_npu index-add fallback，支持实数/复数及 float16 入口的现有 widening；Ascend prepare 使用设备端 CSC 列归属生成。
- 为什么改（附报错原文）：`_spmm_csc_non_real_kernel` 和 `_spmm_csc_non_complex_kernel` 无法在当前 CANN 上 lower。
- 怎么验证的：CPU 模拟 Ascend SpMM correctness 和复数安全 index-add 回归通过。
- 其他后端是否也需要：否

### 2026-10-11 src/flagsparse/sparse_operations/spgemm_csr.py::_ascend_spgemm_compute
- 改了什么：Ascend SpGEMM 强制使用现有纯 torch ESC 路径，绕过 Triton shared-memory hash/count kernels。
- 为什么改（附报错原文）：`_spgemm_hash_count_kernel` 在当前 Ascend CANN lowering 中失败；ESC 仅使用排序、去重和 index-add 张量算子。
- 怎么验证的：CPU 模拟 Ascend CSR SpGEMM 结构和值 correctness 回归通过。
- 其他后端是否也需要：否

### 2026-10-11 src/flagsparse/sparse_operations/sddmm_csr.py::_ascend_csr_row_ids
- 改了什么：Ascend SDDMM fallback 对复数 dense 输入改用 component-safe gather，并按当前 indptr 每次重建 row ids。
- 为什么改（附报错原文）：`sddmm_csr_c32_int_non_non_row` 报 `index_high_dims_op_api ... NPU`，且地址缓存可能复用旧 CSR 结构。
- 怎么验证的：`tools/q4_ascend_dispatch_check.py` 从 44/45 提升到 45/45；新增 CPU 复数回归覆盖。
- 其他后端是否也需要：否

### 2026-10-11 run_flagsparse_pytest.py::ASCEND_Q4_VARIANTS CSC/SpMM/SDDMM batch
- 改了什么：将剩余 CSR/COO SpMV、CSR SpMM、CSR SDDMM 及 CSC 变体接入 Ascend q4 performance runner；benchmark_ascend 增加 CSC command/implementation 标签。
- 为什么改（附报错原文）：这些变体此前没有性能行，或 probe 只能记录能力而无法测量 fallback。
- 怎么验证的：路由、实现标签和 fallback 定向 CI 通过；真实 NPU q4 全量需在 910B 上复跑确认。
- 其他后端是否也需要：否

### 2026-10-11 tests/q4_variant_bench.py::_Matrix.arrays
- 改了什么：CSC complex 值按 component-safe gather 重排，避免 Ascend `aclnnIndex` 直接索引 complex tensor。
- 为什么改（附报错原文）：`Tensor self not implemented for DT_COMPLEX64` 出现在 CSR->CSC 值重排，而不是 CSC fallback kernel。
- 怎么验证的：910B NPU 7 的 `wave.mtx` 上 3 个 SpMV CSC 和 3 个 SpMM CSC 变体全部 `PASS`。
- 其他后端是否也需要：否

### 2026-10-11 tools/q4_ascend_dispatch_check.py / q4 NPU batch
- 改了什么：补齐 SDDMM complex gather 和 CSC complex 重排后的实机路径。
- 为什么改（附报错原文）：dispatch 首轮为 44/45，唯一错误为 SDDMM complex `index_high_dims_op_api`；修复后无需修改非 Ascend kernel。
- 怎么验证的：dispatch check 结果 `45/45 variants take a correct torch path under Ascend`；本批剩余 SpMV COO 2、SpMM CSR 1、SDDMM 4 在 NPU 7 `wave.mtx` 全部 `PASS`。
- 其他后端是否也需要：否

### 2026-10-11 src/flagsparse/sparse_operations/_common.py::_csr_row_ids_torch（合并时补充）
- 改了什么：spmv_csr、spmm_csr、sddmm_csr 的 `_ascend_csr_row_ids` 和 spmv_csc/spmm_csc 的昇腾列号统一改用这个函数：按**张量对象 + `_version` + 行数**缓存 `repeat_interleave` 结果，张量释放时自动移除缓存条目；inference tensor 没有版本号，不缓存。spmm_csr 在 XPU 路径上原来按 `data_ptr()` 缓存，也一并换成这个版本。
- 为什么改（附报错原文）：上面 10-10/10-11 两条记录直接删掉了缓存，修掉了 `index_add_(): Number of indices (...) should be equal to source.size(dim)` 这个地址复用导致的旧结构问题，但每次调用都要重新做一次 `repeat_interleave`，工作量和 SpMV 本身相当。全量结果里 `spmv_csr_f32_int_non` 只有 0.035x（ASIC_680ks fp16 classic：FlagSparse 414 ms，PyTorch-NPU `index_add_` 基线 39 ms）；`spmv_csc_f32_int_non` 只有 0.011x（390 ms 对 4.4 ms）。09-23 那批交付在保留缓存时是 0.95–1.00x。
- 怎么验证的：`tests/ci/test_spmv_coo_ascend_fallback.py::test_csr_row_ids_cache_reuses_only_the_same_unmodified_tensor` 覆盖四种情况：同一张量命中、原地修改后重算、释放后条目被移除、inference tensor 不缓存。drop 自带的结构变化测试仍然通过。**还需要在 910B 上复跑 spmv_csr / spmv_csc，确认性能回到删缓存之前的水平。**
- 其他后端是否也需要：XPU 走同一个函数（原来的按地址缓存有同样的隐患）；CUDA/ROCm/MetaX/MUSA 不走这条路径。

### 2026-10-11 src/flagsparse/sparse_operations/spmm_csc.py::_run_spmm_csc_prepared_with_fallback（合并时补充）
- 改了什么：昇腾分支在需要计时时测出真实的 `compute_ms`，`process_gpu_ms` 为 0.0；`index_fallback_applied` 改为 False，`index_fallback_reason` 为 None。
- 为什么改（附报错原文）：原来返回 `compute_ms=None`，`test_spmm_csc_prepared_path_and_meta` 报 `TypeError: '>=' not supported between instances of 'NoneType' and 'float'`；而且 `index_fallback_applied` 的含义是"从 int64 退回到 int32"，torch_npu 路径没有做这件事，原来标 True 是误报。
- 怎么验证的：CUDA 上 spmm_csc 准确性测试全部通过（这条分支在 CUDA 上不执行）；需要在 910B 上复跑 `-m spmm_csc`。
- 其他后端是否也需要：否

### 2026-10-11 tests/pytest/test_sp{m,}v_csc_accuracy.py::*_int64_*（合并时补充）
- 改了什么：4 个 int64 回退测试在昇腾上跳过，原因和 COO 那条相同。
- 为什么改（附报错原文）：这几个测试故意让 Triton kernel 在 int64 上失败，以验证回退逻辑；昇腾不调用 Triton，所以报 `assert False is True` 和 `Failed: DID NOT RAISE <class 'RuntimeError'>`。
- 怎么验证的：CUDA 上照常执行并通过。
- 其他后端是否也需要：否

### 2026-10-11 tests/ci/test_csc_spgemm_ascend_fallback.py（合并时补充）
- 改了什么：同时替换 `_get_device_backend_info`，改为返回无 GPU 时的结果；按 isort/black 格式化。
- 为什么改（附报错原文）：在有 CUDA 的机器上，CPU 张量会触发 `ValueError: Expected a cuda device, but got: cpu`，9 个用例失败；CI 的 pre-commit 也会修改这个文件。
- 怎么验证的：有 GPU 和 `CUDA_VISIBLE_DEVICES=` 两种环境下都是 11 passed；连续两遍 `make pre-commit-check` 都通过。
- 其他后端是否也需要：否

### 2026-10-14 run_flagsparse_pytest.py::ASCEND_PERFORMANCE_COMMANDS["spgemm_csr"]
- 改了什么：昇腾 `spgemm_csr` 的性能从 `benchmark/benchmark_ascend_probe.py` 改为 `tests/test_spgemm.py`（与其他后端同一脚本），
  并走按 dtype 分进程的 `_run_spgemm_split_dtypes`。
- 为什么改：探测脚本测的是合成规模（`tiny_square`、`square_1k` 等 20 个），`pytest_results_ascend_full_20261011` 里
  `spgemm_csr_f32_int_non_non` 因此没有加速比，也无法和 H800 参照（交付矩阵）对上做折算。
- 怎么验证的：只在 CUDA 机器上确认路由（`tests/ci/test_runner_result_format.py::test_ascend_spgemm_is_measured_on_the_delivery_matrices`）；
  `tests/test_spgemm.py` 在 NPU 上能否跑通**未验证**，昇腾 spgemm 的精度用例是通过的（同一 `flagsparse_spgemm_csr`）。
- 其他后端是否也需要：否。
