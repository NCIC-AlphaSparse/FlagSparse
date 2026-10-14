# MetaX / MACA（C550）：65 变体验证任务

`FLAGSPARSE_BACKEND=metax`。环境搭建、交付复现见 [`../MACA.md`](../MACA.md)。背景见本目录
[`README.md`](README.md)。

## 要做的事

环境按 `../MACA.md` 第 1 节配好后，先做 `../MACA.md` 0.5 节的开跑前检查（`flagsparse.__file__` 必须指向
本仓库 `src/`；`_backend_name()` 必须是 `metax`），再跑：

```bash
setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc,spgemm_csr,sddmm_csr \
  --phase both --mode normal --gpus 0 --timeout 4500 \
  --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 \
  --op-benchmark-args='sddmm_csr=--no-cusparse' \
  --h800-reference \
  --results-dir results_metax_65_<日期> \
  > results_metax_65_<日期>.log 2>&1 < /dev/null &
# 跑完后
python3 tools/delivery_table.py results_metax_65_<日期>
```

- `--benchmark-input tests/data`：仓库里的 `tests/data` 正好就是 `conf/operators.yaml` 的 10 个交付矩阵（在 git
  里，拉下来就有）。这条命令没带 `--delivery-only`，runner 不会再筛矩阵，给 30 个矩阵的目录就会 30 个全跑，
  汇总出来的加速比也会掺进非交付矩阵，所以不要换成别的目录。
- `--h800-reference`（不带值）：没有任何基线的行（目前是 legacy `spmv_csr` f32/f64、SDDMM 的 q4 变体、`spgemm_csr`），
  用仓库自带的 H800 结果按带宽折算出加速比（`h800_scaled_ms = T_cuSPARSE@H800 × 3050/1440`），`delivery_table.py`
  里这些行后面会标 `(H800折算)`。**有 PyTorch 基线的行不受影响。** 这是估算，不是厂商实测，汇报时要注明；
  `sddmm_csr_f16_int_non_non_row` 在 H800 上没有 cuSPARSE 时间（cuSPARSE SDDMM 不支持 fp16），改用 H800 上
  PyTorch 的时间折算，标 `(H800 PyTorch折算)`（C550 自己的 `sampled_addmm` 结果是错的，7.3 节）。
  口径和覆盖见 [`../H800_REFERENCE.md`](../H800_REFERENCE.md)。
- `sddmm_csr=--no-cusparse`、`--timeout 4500` 沿用 `../MACA.md` 0.5 节的 20 变体交付命令，原因见那一节
  （C550 没有可用厂商稀疏库；SDDMM 的 4 个 K 值不收窄，单阶段耗时长）。

- 外层 `timeout -s KILL` 是整条命令的总限时（内核卡死时 Ctrl-C 送不进去，只能靠 KILL）；`--timeout` 是
  每个算子每个阶段的限时。这两个数是 20 变体时期定的，65 变体多了 6 个父算子，**没有在本机实测过总耗时**，
  到点被杀的话按 `delivery_table.py` 里 `NotFound` 的算子单独补跑（补跑要用新的 `--results-dir`）。
- `--results-dir` 每次都用新目录：`summary.json` 每跑一次就整体重写，往旧目录里补跑一部分会把之前的结果冲掉。

把 `delivery_table.py` 的完整输出带回来，尤其是第一行的 `missing` 数字。

## 已知风险点（这个后端特有的，不是猜的）

- **C550 没有厂商 cuSPARSE/CuPy 基线**（`../MACA.md` 0.5 节已经写了：`CuPy 真装了就用
  cupy_cusparse，否则 torch`，本机实际是 PyTorch）。这意味着 45 个新变体里走 q4 变体测量
  （`tests/q4_variant_bench.py`）的那部分，厂商列大概率全是空，只有 PyTorch 对照——**这是预期行为，
  不是 bug**，跟原有 20 个变体在这台机器上的情况一致。
- **单线程私有内存上限 4 KB**：`tl.static_range` 编译期展开的循环超了会报
  `memory size or pointer value too large to fit in 32 bit`，真正原因在它上一行的 MACA 运行时提示里。
  复数变体（`spmv_csr/coo/csc_c32`、`spmm_csr/coo/csc_c32`、`spvv_c32_int_conj`、`spmv_sell_c32`、
  `spmv_csr_f32c32`）最该先跑，这类问题历史上都出在复数/宽展开路径上。
- **SDDMM 没有可用参考**：`../MACA.md` 7.3 节记录过 `torch.sparse.sampled_addmm` 结果是错的，交付命令
  一直用 `--op-benchmark-args='sddmm_csr=--no-cusparse'` 绕开。这次的新 sddmm 变体（c32、f16，还有
  trans/col 方向）同样会撞上这个限制，SDDMM 的厂商列预期也是空的。

## `spmm_csc` q4 性能汇总污染（已修复）

2026-10-08 的首轮 C550 结果把三个 q4 交付变体
`spmm_csc_f16_int_non_non_row`、`spmm_csc_f32_int_non_non_row`、
`spmm_csc_c32_int_non_non_row` 的性能汇总为 `Failed`，但各变体的 10 条 q4 行实际都是 `PASS`。

真正失败的是同一个父性能进程里的传统行：
`amazon0601.mtx / float32 / int32 / trans / row / spmm_csc_base`，报
`reason=correctness check failed`、`err_vs_pytorch=1.397478580078429`。脚本因这条记录以
`returncode=1` 退出；旧汇总逻辑继承父阶段的 `FAIL`，从而污染了本应通过的 q4 投影。传统行的失败本身仍是
真实失败，修复不会隐藏它。

上游 `982b985` 已修复这个通用 runner 问题。仅当以下条件同时成立时，目标变体才不继承父阶段失败：父阶段
是普通 `exit 1`（不是 timeout 或信号退出）、stderr 无 Python traceback、目标 q4 变体的矩阵行完整、且存在
其他变体的失败行可以解释该退出。目标自身出现 `FAIL` 时仍为失败。这些限制避免异常或半程结果被错误标绿。

复查旧产物时，不能只看 `delivery_table.py` 的三行 `Failed`；应按 `variant` 查看原始 CSV。若目标 q4 行均为
`PASS`，而父阶段是上述可解释的 `exit 1`，则是历史汇总污染，需用包含 `982b985` 的 runner 重跑或重投影。

## 跑完要确认

1. `missing` 是不是 0。
2. 65 行里有没有 `FAIL`（不是"没有基线所以是 N/A"，是真的算错了）。
3. 复数变体和 SDDMM 变体有没有卡死或者 OOM（对应上面两条风险点）。

把这三条的结果写回来，不用额外分析，原始数据最有用。

## 交给 Codex 的 prompt

上机时把下面整段原样贴给 Codex（或其他代理）。不用改任何内容。

````text
你在 MetaX C550（MACA） 实机上复测 FlagSparse 的 65 个交付变体。仓库在当前目录，main 分支。

背景：65 个变体（原 20 个 + 新合并的 45 个）只在 CUDA 上实测过，本机从没跑过。
任务是在本机跑一遍，把原始结果带回来。主要目的是收集数据，不是修代码。

步骤：
1. git pull --ff-only origin main。
2. 按 docs/MACA.md 配好环境，做完它的环境自检。
3. 确认导入的是仓库源码：python3 -c "import flagsparse; print(flagsparse.__file__)"
   输出必须在当前目录的 src/ 下。如果指向 site-packages / dist-packages，先停下来汇报，不要继续。
   旧安装包会让基线列全部变成 N/A，看起来像正常结果。
4. 完整读一遍 docs/debug/MACA.md，照"要做的事"一节的命令原样跑。
   不要自己删减或改写参数，每个参数的来由那一节都写了。
   - --results-dir 用新目录（<日期> 填今天），不要复用旧目录。
   - 命令本身已经用 setsid 放到后台，定期看日志进度，不要中途打断。
   - 不要用 pytest --forked，它在 GPU 上会让所有用例失败。
5. 跑完执行 python3 tools/delivery_table.py <结果目录>。

规矩：
- 不要为了"让它通过"去改 src/ 或 tests/ 下的代码。确实非改不可（比如环境适配）时，每改一个文件，
  都按 docs/debug/MACA.md 末尾"实机改动记录"一节的格式追加一条记录，并和代码改动放在同一个
  commit 里。没改就在那一节写"无"。
- 不要猜根因。遇到报错就给出完整报错原文，以及出错变体在结果目录里那一行的 reason/error 字段。
- 本机没有厂商稀疏库，厂商列为空是预期的。复数变体如果报 memory size or pointer value too large，
  把这行之前的 MACA 运行时提示一起带回来。

汇报包含：
1. delivery_table.py 的完整输出，尤其是第一行 "N registered variants, M missing"。
2. 所有非 Passed 的变体：变体名、阶段（精度/性能）、reason 或 error 原文。
3. 逐条回答 docs/debug/MACA.md "跑完要确认"一节。
4. 环境指纹：torch、triton 版本，设备名，warp size，驱动/SDK 版本，整轮总耗时。
5. docs/debug/MACA.md "实机改动记录"一节的内容。
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

### 2026-10-09 `run_flagsparse_pytest.py`：上游 `982b985`
- 改了什么：为 q4 和传统交付投影增加有条件的父阶段失败归因，保留父状态和解释它的原始失败行。
- 为什么改（附报错原文）：`spmm_csc_base` 的 `amazon0601.mtx` 行报
  `reason=correctness check failed`、`err_vs_pytorch=1.397478580078429`，误伤三个 q4 交付变体。
- 怎么验证的：`python3 -m pytest -q tests/ci/test_delivery_parent_failure.py tests/ci/test_delivery_algorithm_scope.py`，结果 `18 passed`；本机 65 变体复测为 65/65 精度 Passed、65/65 性能 Passed。
- 其他后端是否也需要：是（通用 runner 汇总逻辑，不改变任何后端内核）。

### 2026-10-13 MACA C550 kernel tuning
- 改了什么：`spmv_csr` 的 MACA 专用 gather 计划和复数成对加载内核；`spmv_sell` 的 MACA
  group/warp；复数 scatter 的 block size；`spmm_csc` 复数转置的 `BLOCK_NNZ`。所有分支均由
  `_is_maca_runtime()` 保护，CUDA/ROCm/MUSA/Ascend 路径不变。
- 为什么改：C550 上原 CUDA 参数在部分稳定样本低于 0.8x（CSR、SELL、complex scatter，以及
  `amazon0601` CSC complex trans/conj）。重复测量同时观察到偶发 3--4 ms 运行时尖峰，未修改
  baseline、计时器或汇总逻辑。
- 怎么验证的：CSR q4 80 行全部 `PASS`，最低 `speedup_vs_pytorch=1.007x`；complex scatter
  32 行全部 `PASS`，最低 `0.90x`；CSR/CSC 相关精度与缓存测试共 279 项通过。CSC complex
  trans/conj 10 矩阵、2 索引类型共 40 行全部通过，PyTorch COO 参考下最低约 `20.5x`（本机
  没有可用厂商 CSC baseline）。SELL 复测中位数均高于 0.8x；NACA complex64 有单次尖峰，
  10 次中位数约 `3.7x`，原始日志保存在 `results_metax_opt_20261013/`。
- 其他后端是否也需要：否。

### 2026-10-14 SpGEMM 结果比较与 worker 可靠性
- 改了什么：`tests/test_spgemm.py` 比较 CSR 结果前过滤显式零，避免与 SciPy 丢弃显式零后的结构差异被误报为 `nnz mismatch`；补充独立 matrix-worker CLI 和异常 stderr 归因。新增 `tests/ci/test_spgemm_compare.py` 以及对应的 CI 注册检查。
- 为什么改：MACA 上 SpGEMM 的数学结果可能保留取消相消产生的显式零，而 SciPy 参考会丢弃这些项；原比较器按结构 nnz 直接失败，属于比较器误报，不是 kernel 精度错误。
- 怎么验证的：SpGEMM comparator 回归与相关 CI 检查共 `15 passed`；MACA 10 个测试矩阵的 float32/float64 结果均通过精度检查。
- 其他后端是否也需要：是（比较器和 runner 是通用逻辑；不改变后端 kernel）。

### 2026-10-14 MACA reuse / CSR gather accuracy coverage
- 改了什么：`tests/pytest/test_rocm_sparse_reuse_accuracy.py` 的门控扩展为 ROCm 或 MACA；新增 `tests/pytest/test_spmv_csr_maca_accuracy.py`，覆盖 MACA CSR gather 的 float16/complex64、int32/int64、non/trans/conj、空行和长行，以及 inference/live value 更新；CPU transpose 回归测试固定使用 portable launch 模拟。
- 为什么改：这些测试验证的是共享的 preparation/cache-reuse 合约，原来只在 ROCm 执行会漏掉 MACA 专用 gather 路径；C550 的复数成对加载路径还需要直接精度回归。
- 怎么验证的：MACA reuse 测试 `18 passed`；CSR/CSC 相关精度、缓存和 kernel 回归合计 `279 passed`。
- 其他后端是否也需要：门控改动只扩大到 MACA；portable CPU 测试保持其他后端行为不变。

### 2026-10-14 MACA 性能 baseline 规则
- 改了什么：MACA 性能交付统一使用 PyTorch 作为稀疏 baseline：`FLAGSPARSE_MACA_VENDOR=torch`。CuPy/cuSPARSE 可用时才切换到 `cupy_cusparse`；`none` 只用于明确不需要加速比的诊断运行。
- 为什么改：C550 实机没有可用 CuPy/cuSPARSE。此前 65 算子命令继承了 `FLAGSPARSE_MACA_VENDOR=none`，导致 `spmv_csr_f32_int_non` 和 `spmv_csr_f64_int_non` 的 baseline/加速比显示 `N/A`；这是启动环境造成的，不是 CSR kernel 没有 baseline 实现。
- 怎么验证的：设置 `FLAGSPARSE_MACA_VENDOR=torch` 后，`results_metax_spmv_csr_baseline_20261014/` 中两种 base CSR 变体均有 baseline 且性能 `Passed`；float32 聚合加速比 `295.033x`，float64 为 `284.441x`。规则和完整启动命令同步写在 `docs/MACA.md`。
- 其他后端是否也需要：否；该环境变量和默认选择仅影响 MACA baseline。
