# 摩尔线程 / MUSA（S5000）：65 变体验证任务

`FLAGSPARSE_BACKEND=mthreads`。环境搭建、交付复现见 [`../MUSA.md`](../MUSA.md)。背景见本目录
[`README.md`](README.md)。

## 要做的事

环境按 `../MUSA.md` 第 2 节配好（`FLAGSPARSE_BACKEND=mthreads`、`MUSA_HOME`），**按顺序跑两轮，单卡上不要同时跑**。

**第一轮：交付结果，用 `run_flagsparse_split_delivery.py`**（和 `../MUSA.md` 0.5 节 20 变体交付是同一个 runner）：

```bash
setsid timeout -s KILL 43200 python3 -u run_flagsparse_split_delivery.py \
  --mode normal --benchmark-input tests/data --timeout 3600 \
  --results-dir results_musa_split_65_<日期> \
  > results_musa_split_65_<日期>.log 2>&1 < /dev/null &
# 跑完后
python3 tools/delivery_table.py results_musa_split_65_<日期>
```

- 精度那一半调用 `run_flagsparse_pytest.py --phase accuracy --delivery-only`，覆盖全部 65 个变体。
- 性能那一半取 C API 的 `ctest`（对 muSPARSE），只覆盖 C API 交付口径里的 **25 个变体**：原来 20 个，加上
  5 个新变体（`spmm_csr_f32_int_non_non_col`、`spmm_csr_f32_int_non_trans_row`、`sddmm_csr_f32_int_non_non_col`、
  `sddmm_csr_f32_int_non_trans_row`、`sddmm_csr_f32_int_trans_non_row`）。**另外 40 个变体性能显示
  `NOT_CONFIGURED` / `NotFound` 是预期的**——C API 没有实现它们，不是跑挂了。
- 拉到新代码后第一次跑**不要加 `--skip-capi-build`**，确保 C API 是重新编译的。

**第二轮：确认 40 个新变体的性能路径在 MUSA 上能跑通，用通用 runner，只跑性能**：

```bash
setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc,spgemm_csr,sddmm_csr \
  --phase performance --mode normal --gpus 0 --timeout 3600 \
  --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir results_musa_perf_65_<日期> \
  > results_musa_perf_65_<日期>.log 2>&1 < /dev/null &
# 跑完后
python3 tools/delivery_table.py results_musa_perf_65_<日期>
```

- 这一轮只看第一轮性能缺的那 40 个变体：有没有跑出 FlagSparse 自己的耗时、有没有报错。Python 侧在 MUSA
  上没有厂商基线、`torch.sparse` 也没有矩阵乘，**加速比为空是预期的**；精度列是 `NotFound` 也是预期的
  （只跑了性能）。
- 两轮结果要分开放、分开汇报，不要往同一个 `--results-dir` 里跑：`summary.json` 每跑一次就整体重写。
- `--benchmark-input tests/data`：`tests/data` 正好是 10 个交付矩阵（在 git 里）。第二轮没带 `--delivery-only`，
  runner 不会筛矩阵，换成 30 个矩阵的目录就会全跑。第一轮的 C API 一侧自己会筛。
- 外层 `timeout -s KILL` 是整条命令的总限时（内核卡死时 Ctrl-C 送不进去，只能靠 KILL）；`--timeout` 是每个
  算子每个阶段的限时，沿用 `../MUSA.md` 的结论（`900` 不够）。65 变体**没有在本机实测过总耗时**，到点被杀
  的话按 `NotFound` 的算子单独补跑，补跑用新的 `--results-dir`。

把两轮 `delivery_table.py` 的完整输出都带回来。

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

## 跑完要确认

1. 第一轮：65 个变体精度是不是全 Passed；那 25 个变体性能是不是都有 muSPARSE 加速比（原 20 个上次实机都有，
   这次少了就是回归）。
2. 第二轮：40 个新变体是不是都跑出了 FlagSparse 自己的耗时，有没有报错或超时。
3. 所有复数变体（c32）在两轮里是不是真的跑通了，还是报了 `IndexMusa` 之类的错误；以及有没有哪个变体的耗时
   可疑（比如慢几百倍，通常意味着走了某种 fallback 循环，而不是真正的 kernel）。

把这三条的结果写回来，不用额外分析，原始数据最有用。

## 第二阶段：让其余 40 个变体进 C API 性能统计（在 MUSA 上实现）

**先做完上面的两轮验证再开始**，拿到改动前的基线。目标：split 跑完后 65 个变体的性能都不再是
`NOT_CONFIGURED`。有 muSPARSE 基线的给出加速比；muSPARSE 不支持的类型组合（fp16、int8、混合精度很可能
都不支持——cuSPARSE 也没有 fp16 的 SpMV/SDDMM）显示 `NoBaseline`（跑通了、没有厂商对照），**这是可接受
的结果，绝不能伪造基线**。

### 40 个变体现在卡在哪（2026-10-08 在 CUDA 机器上逐个核对过）

| 组 | 数量 | 现状 | 要做的事 |
|---|---:|---|---|
| A | 7 | main 的 C API **已经实现**，只是不在交付统计口径里 | 改清单口径；其中 4 个还要补 benchmark 的轴（见下） |
| B | 29 | main 没有，**`origin/q4` 分支已经实现**，记录是 CUDA 上 ctest 精度通过 | 把 q4 的 C API 改动移植过来，再补 benchmark 行 |
| C | 4 | q4 也没有 | 新写 |

<details>
<summary>每组具体是哪些变体</summary>

- **A（7）**：
  - 已有 benchmark 行，只差口径：`spmv_csr_c32_int_non`、`spmv_coo_c32_int_non`、`spgemm_csr_f32_int_non_non`
  - 还缺 benchmark 轴：`spmm_csr_c32_int_non_non_row`、`spmm_coo_c32_int_non_non_row`（benchmark 只测列主序）、
    `spmv_csc_f32_int_non`、`spmv_csc_c32_int_non`（`test_spmv.cpp` 没有 CSC 构建器）
- **B（29）**：
  - q4 已确认能出 benchmark 行（2）：`scatter_i8_int`、`sddmm_csr_f16_int_non_non_row`
  - 混合精度，分发已实现、缺 benchmark 行（12）：`spmv_csr_f16f32/i8f32/i8i32/f32c32_int_non`、
    `spmv_coo_f16f32/i8i32_int_non`、`spmm_csr_f16f32/i8i32_int_non_non_row`、`spmm_coo_i8i32_int_non_non_row`、
    `spvv_f16f32/i8i32_int_non`、`spmv_sell_i8i32_int_non`
  - 普通路径，已实现、缺 benchmark 行（13）：`axpby_f16_int`、`spmv_sell_f32/f16/c32_int_non`、
    `spvv_c32_int_conj`、`spmv_csr_f32_int_trans`、`spmv_coo_f32_int_trans`、`spmv_csr_c32_int_conj`、
    `spmv_coo_c32_int_conj`、`spmm_csc_f32/c32/f16_int_non_non_row`、`sddmm_csr_c32_int_non_non_row`
  - 已实现，但 benchmark 上 fp16 误差超过容差（2）：`spmv_csr_f16_int_non`、`spmv_coo_f16_int_non`
- **C（4）**：`spmm_csr_f16_int_non_non_row`、`spmm_coo_f16_int_non_non_row`（C API 的 SpMM 清单里没有 f16）、
  `spmv_csc_f16_int_non`、`spmm_csr_f32_int_trans_non_row`（main 的 `spmm.cpp` 对 opA 转置直接返回
  `NOT_SUPPORTED`；q4 有 `capi/flagsparse_codegen/spmm_transpose.py`，可以从它起步）

分组来源：A 由 `conf/operators.yaml` 的 `capi` 字段算出；B、C 来自 `origin/q4` 的
`capi/tools/write_summary_q4.py` 里 `CONFIRMED`、`_MIXED_DISPATCH_VERIFIED`、`_CAPI_DISPATCH_VERIFIED`
三张表，和它 `NOT_YET_COVERED_REASONS` 里每个变体没进统计的原因——动手前值得通读一遍那个文件。

</details>

### 步骤

1. **移植 q4 的 C API 改动，用补丁，不要合并分支。** `origin/q4` 基于旧 main，整分支合并会冲掉 main 之后的
   工作；但它改过的 `capi/` 文件 main 从分叉点（`097a3c0`）之后一个都没动过，所以只拿 `capi/` 是干净的：

   ```bash
   git fetch origin q4
   base=$(git merge-base origin/main origin/q4)
   git diff $base origin/q4 -- capi/ ':!capi/tools/write_summary_q4.py' > /tmp/q4_capi.patch
   git apply --check /tmp/q4_capi.patch && git apply /tmp/q4_capi.patch
   ```

   - 2026-10-08 在 main `263a139` 上 `git apply --check` 通过；C API 代码生成用到的 11 个 Python 内核，main 和
     q4 的签名逐个比对过，完全一致。
   - 不拿 `write_summary_q4.py`：它依赖已经删掉的 `load_q4_variants()`，main 只有一张 65 项的表。
   - **补丁打上后不会报错，但有一处会静默失效，必须改**：q4 的 `capi/tools/gen_variants.py` 从仓库根
     `conf/operators.yaml` 的 `q4_variants:` 读变体（`q4_doc.get("q4_variants", [])`），这个键在 main 里已经合进
     `delivery_variants:`，读出来是空列表，benchmark 会一行新变体都不产生。改成用
     `tools/delivery_variants.py` 的 `load_delivery_variants()`，并给这些行的 scope 标 `delivery`。
2. **让 `capi/tools/write_summary.py` 按变体 id 对行。** 现在的 `variant_name()` 按 (operator, dtype) 对行，
   同一对下只允许一个变体；65 个变体里 `(spmm_csr, f32)` 就有 4 个（`non_non_row/col`、`non_trans_row`、
   `trans_non_row`），会被判成 `__undeclared__`。q4 的 benchmark 已经给行打了 `q4_variant` 标签：先按标签精确
   匹配，没有标签再退回 (operator, dtype)——和 Python 侧 `run_flagsparse_pytest.py` 的
   `_q4_performance_phase` 同一个思路。标签名建议统一改成 `variant`。
3. **补 benchmark 的轴**：`test_spmm.cpp` 加行主序和 opA 转置，`test_spmv.cpp` 加 CSC 构建器，混合精度行按
   变体 id 推出输出类型（q4 的 `Q4_DTYPES` 已经是这么做的）。
4. **改清单口径**：`capi/conf/operators.yaml` 里相关算子（含 q4 新增的 `axpby`/`spvv`/`spmv_sell`/`spmm_csc`，
   q4 标的是 `retained`）改成 `reporting: delivery`，补 `delivery_dtypes`/`ops`。随后同步更新：
   `conf/operators.yaml` 每个变体的 `capi` 字段，以及 `tests/ci/test_delivery_variant_registry.py` 里固定的
   `CAPI_TRUE_VARIANT_IDS` 和"C API 交付口径没有扩大"那条测试——**这两处是故意钉死的，按实际实现情况改，
   不要为了让测试过而改。**
5. **C 组 4 个新写**。
6. **fp16 超容差的 2 个**：不要放宽全局容差。q4 记录的原因是参考值用未量化的 fp64 输入算的；改成用量化成
   fp16 之后的输入来算参考值，容差保持不变。

### 验收

- MUSA 上 `ctest --test-dir capi/build -L capi --output-on-failure` 全过（C API 重新编译，不要 `--skip-capi-build`）。
- 重跑上面第一轮 split，`delivery_table.py` 里 65 个变体的性能**没有一个是 `NOT_CONFIGURED`/`NotFound`**；
  原来 25 个的 muSPARSE 加速比和改动前比没有变差。
- `make format-check lint lint-src` 和 `pytest tests/ci -q` 通过（CI 对 `capi/tools` 跑 `ruff format --check`）。
- **CUDA 没法在 MUSA 机器上验证。** 这些改动动的是 `spmv.cpp`/`spmm.cpp` 等所有后端共用的分发层，推上来后要在
  CUDA 机器上重新编译 C API、跑一遍 `ctest -L capi`，汇报里注明"CUDA 未验证"。

## 交给 Codex 的 prompt

上机时把下面整段原样贴给 Codex（或其他代理）。不用改任何内容。

````text
你在 摩尔线程 S5000（MUSA） 实机上复测 FlagSparse 的 65 个交付变体。仓库在当前目录，main 分支。

背景：65 个变体（原 20 个 + 新合并的 45 个）只在 CUDA 上实测过，本机从没跑过。
任务是在本机跑一遍，把原始结果带回来。主要目的是收集数据，不是修代码。

步骤：
1. git pull --ff-only origin main。
2. 按 docs/MUSA.md 配好环境，做完它的环境自检。
3. 确认导入的是仓库源码：python3 -c "import flagsparse; print(flagsparse.__file__)"
   输出必须在当前目录的 src/ 下。如果指向 site-packages / dist-packages，先停下来汇报，不要继续。
   旧安装包会让基线列全部变成 N/A，看起来像正常结果。
4. 完整读一遍 docs/debug/MUSA.md，照"要做的事"一节的命令原样跑。
   不要自己删减或改写参数，每个参数的来由那一节都写了。
   - --results-dir 用新目录（<日期> 填今天），不要复用旧目录。
   - 命令本身已经用 setsid 放到后台，定期看日志进度，不要中途打断。
   - 不要用 pytest --forked，它在 GPU 上会让所有用例失败。
5. 跑完执行 python3 tools/delivery_table.py <结果目录>。

规矩：
- 不要为了"让它通过"去改 src/ 或 tests/ 下的代码。确实非改不可（比如环境适配）时，每改一个文件，
  都按 docs/debug/MUSA.md 末尾"实机改动记录"一节的格式追加一条记录，并和代码改动放在同一个
  commit 里。没改就在那一节写"无"。
- 不要猜根因。遇到报错就给出完整报错原文，以及出错变体在结果目录里那一行的 reason/error 字段。
- 要按顺序跑两轮（split 交付 + 通用 runner 只跑性能），两轮的 delivery_table.py 输出都要带回来。
  第一轮有 40 个变体性能 NOT_CONFIGURED、第二轮加速比为空，都是预期的。复数变体如果报
  IndexMusa not implemented，列出是哪些变体、在哪一轮。

汇报包含：
1. 两轮各自 delivery_table.py 的完整输出，尤其是第一行 "N registered variants, M missing"。
2. 所有非 Passed 的变体：变体名、阶段（精度/性能）、reason 或 error 原文。
3. 逐条回答 docs/debug/MUSA.md "跑完要确认"一节。
4. 环境指纹：torch、triton 版本，设备名，warp size，驱动/SDK 版本，整轮总耗时。
5. docs/debug/MUSA.md "实机改动记录"一节的内容。
````

### 第二阶段的 prompt

上面两轮验证做完、结果汇报过之后，再把下面这段贴给代理，不用改任何内容。

````text
你在摩尔线程 S5000（MUSA）实机上做 FlagSparse 的 C API 开发：让 65 个交付变体里目前没有 C API 性能数据的
40 个，都能在 run_flagsparse_split_delivery.py 的性能那一半里出数。仓库在当前目录，main 分支。

1. git pull --ff-only origin main，完整读一遍 docs/debug/MUSA.md 的"第二阶段"一节，按里面的步骤 1-6 做。
   那一节已经写清楚了 40 个变体各自卡在哪、哪些可以从 origin/q4 分支移植、移植时哪一处会静默失效。
2. 移植 origin/q4 只用那一节给出的补丁命令，不要 git merge origin/q4，也不要整文件 checkout q4 的文件。
3. 一次做一组，每做完一组就重新编译 C API，跑 ctest --test-dir capi/build -L capi --output-on-failure，
   确认之前通过的用例没有变成失败，再做下一组。
4. 不要伪造厂商基线：muSPARSE 不支持的类型组合就显示 NoBaseline，在汇报里列出是哪些、muSPARSE 返回了什么。
5. 不要放宽容差、不要删测试、不要缩小交付列表来让结果变好看。tests/ci 里钉死的集合按实际实现情况更新，
   并在汇报里说明改了什么、为什么。
6. 每改一个文件，都按 docs/debug/MUSA.md 末尾"实机改动记录"一节的格式追加一条，和代码放在同一个 commit 里。
   分组提交，不要一个大 commit。

验收（全部做到才算完成，做不到的那部分在汇报里如实说明卡在哪）：
- MUSA 上 ctest -L capi 全过。
- 重跑 docs/debug/MUSA.md 第一轮的 split 命令，delivery_table.py 里 65 个变体的性能没有 NOT_CONFIGURED / NotFound；
  原来 25 个的 muSPARSE 加速比和改动前相比没有变差。
- make format-check lint lint-src 和 pytest tests/ci -q 通过。

汇报包含：
1. 改动前后两次 delivery_table.py 的完整输出。
2. 40 个变体逐个的最终状态：Passed（有加速比）/ NoBaseline（附 muSPARSE 返回的状态）/ 仍未完成（附原因）。
3. ctest -L capi 的结果摘要，以及新增了哪些 ctest 用例。
4. docs/debug/MUSA.md "实机改动记录"一节的内容。
5. 明确写一句：这些改动没有在 CUDA 上编译和测试过。
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
