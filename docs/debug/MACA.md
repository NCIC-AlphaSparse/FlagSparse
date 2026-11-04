# MetaX / MACA（C550）：65 变体验证任务

`FLAGSPARSE_BACKEND=metax`。环境搭建、交付复现见 [`../MACA.md`](../MACA.md)。背景见本目录
[`README.md`](README.md)。

## 2026-11-02 复测计划：105 变体

注册表扩到 105 个，改了什么见 [`README.md`](README.md) 的 2026-11-02 一节。2026-11-03 起所有后端用同一行命令（
`run_flagsparse_delivery.py`，说明见 README 的“统一命令”一节）：它自动识别后端，走
`run_flagsparse_pytest.py --delivery-only`，自动带上 `--h800-reference`、`--timeout 4500` 和
`--op-benchmark-args='sddmm_csr=--no-cusparse'`（原因见下面）。原来的 `--ops` 列表换成了 `--delivery-only`，选到
的还是那 15 个父算子，交付收窄也生效了。开跑前照旧做 `../MACA.md` 0.5 节的检查。因机器而异的只有下面的环境准备。

```bash
setsid timeout -s KILL 43200 python3 -u run_flagsparse_delivery.py --gpus 0 > delivery_105.log 2>&1 < /dev/null &   # 物理卡号，逗号分隔；多卡如 --gpus 0,1,2,3，算子按卡轮流分配、并行跑
# 日志第一行是 [delivery] backend=… results=<结果目录>；先用 --dry-run 看一眼解析出的命令也行
# 跑完后交付表已经打在日志末尾，也写进了 <结果目录>/delivery_table.txt
```

**这个后端的风险点（针对新增的 40 个）：**

- **没有厂商列，H800 也补不上**：C550 没有厂商稀疏库，新变体只有 PyTorch 对照。它们在 H800 上没有数据，所以
  `--h800-reference` 只能补前 65 个。
- **SDDMM 的 `sampled_addmm` 结果是错的**：该 PyTorch API 在 C550 上会算错（7.3 节）。2026-11-03
  已为 MACA q4 基准补充取数/相乘/求和路径：API 报错、输出错误或非有限、计时输出校验失败、时延无效时
  回退；备用路径也必须通过独立 CPU 参考校验才发布本机 `pytorch_ms`。原 API 的失败保存在
  `pytorch_sampled_addmm_reason`，实际公式在 `pytorch_formulation`，最终备用路径失败才写 `pytorch_reason`。
  float16 输入先转 float32 计算。该表达式会物化 nnz×K 中间张量，不是融合厂商 SDDMM，不能称为厂商加速比。
  接线前的 `results_metax_105_20261103_1253/` 仍有 4 个新增 SDDMM 无加速比；本次定向结果不覆盖旧产物。
- **spgemm c32 的旧 ESC 慢路径已替换**：2026-11-03 已扩展 MACA 专用纯 Triton
  SpGEMM（`_spgemm_maca.py`）以支持 complex64；复数乘法、排序归约及长行哈希均在
  kernel 内完成。10 矩阵复核平均为 1.640396x（本机 PyTorch），详见本文末尾四项优化记录。
- **4KB 私有内存**：新 kernel 没有用 `tl.static_range` 做大规模展开。gemvi 的 tile 是 64×32，geam 的二分查找用的是
  运行时循环。复数 gemvi 和 geam_c32 要优先看有没有 `memory size or pointer value too large`。

**跑完要确认：**

1. `missing` 是不是 0；105 个变体的精度是不是都 Passed。
2. 复数的新变体有没有编译错误：`gemvi_c32_int_non`、`geam_csr_c32_int`、`spmv_csc_c32_int_conj`、`spmm_csr_c32_int_*`、
   `sddmm_csr_c32_int_non_trans_row`、`spvv_c32_int_non`、`spmv_*_f32c32_*`。
3. 新 40 个里没有加速比的变体逐个列出，附上 `pytorch_reason`。
4. `spgemm_csr_c32_int_non_non` 的耗时（MACA 纯 Triton 路径），以及 `performance.csv` 里是不是只有 10 行 c32。
5. 和 10-15 那轮相比，原 65 个里加速比掉了一半以上的变体。这轮改过混合精度路径和 spmm_csc/sddmm 的公开接口，
   原 65 个也可能受影响。

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
- `--h800-reference`（不带值）：没有任何基线的行（例如 SDDMM 的 q4 变体，或本机 baseline 校验失败的行），
  用仓库自带的 H800 结果按带宽折算出加速比（`h800_scaled_ms = T_cuSPARSE@H800 × 3050/1440`），交付表里只显示数字；
  要看哪些是折算的用 `delivery_table.py --show-baseline`。**有 PyTorch 基线的行不受影响。** 这是估算，不是厂商实测，汇报时要注明；
  `sddmm_csr_f16_int_non_non_row` 在 H800 上没有 cuSPARSE 时间（cuSPARSE SDDMM 不支持 fp16），改用 H800 上
  PyTorch 的时间折算（`--show-baseline` 显示 `h800_pytorch_scaled`；C550 自己的 `sampled_addmm` 结果是错的，7.3 节）。
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

## 跑完要确认（2026-10-14 之后这一轮）

1. `missing` 是不是 0；65 个变体精度是不是全 Passed（`spmv_coo_f32_int_non` 不应再是 Skipped）。
2. 有加速比的变体是不是 65/65。`python3 tools/delivery_table.py <结果目录> --show-baseline` 的输出原样贴回。按 10-13 那轮预演：SDDMM 的 q4 变体、
   `sddmm_csr_f16_int_non_non_row` 为 `h800_pytorch_scaled`（预演 1.595x）。2026-10-15 起 SpGEMM 在
   `FLAGSPARSE_MACA_VENDOR=torch` 下独立测量并校验本机 PyTorch CSR baseline（见 `../MACA.md` 7.3.1）；
   有效时应使用 `pytorch_ms`，不再引用历史 `h800_scaled` 作为本机加速比。
3. `spmv_csr_f32_int_non` / `spmv_csr_f64_int_non` 的加速比和它的基线是什么：`spmv_csr/performance.csv` 里
   `vendor_backend`、`vendor_ms`、`speedup_vs_vendor`、`h800_scaled_ms` 各一行原样贴回（用 PyTorch 基线会是几百倍，
   用 H800 折算预演是 0.68x；这一项待定，先如实带回）。
4. `spgemm_csr` 每一行都不应是 `MATRIX_WORKER_FAILED`（按矩阵隔离的子进程以前在 main 上从没跑通过，这轮第一次能跑）。
5. MACA 专用调参（spmv_csr gather / SELL / 复数 scatter / spmm_csc）合进来后第一次全量跑：和 10-13 那轮比，
   加速比掉了一半以上的变体列出来。

把这些结果写回来，不用额外分析，原始数据最有用。

## 交给 Codex 的 prompt

上机时把下面整段原样贴给 Codex（或其他代理），不用改任何内容。

````text
你在 MetaX C550（MACA） 实机上复测 FlagSparse 的 65 个交付变体（2026-10-14 之后的这一轮）。仓库在当前目录，main 分支。

背景：这台机器上一轮已经跑过 65 项。之后仓库改了两类东西，这轮就是验证它们：
- 没有基线的变体，现在用仓库自带的 H800 结果按带宽折算出加速比（命令里的 --h800-reference），交付表里只显示数字，
  delivery_table.py --show-baseline 的最后一列说明哪些是折算的。有本机基线的变体不受影响。
- 本后端相关的修复，见 docs/debug/MACA.md "跑完要确认"一节逐条列出的内容。
主要目的是收集数据，不是修代码。

步骤：
1. git pull --ff-only origin main；git log -1 --oneline 的提交应不早于 d7d1919（"ci again"）。
2. 按 docs/MACA.md 配好环境，做完它的环境自检。
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
5. 完整读一遍 docs/debug/MACA.md，照"要做的事"一节的命令原样跑（已带 --h800-reference，不要去掉）。
   - --results-dir 用新目录（<日期> 填今天），不要复用旧目录，也不要往旧目录里补跑。
   - 命令本身已经用 setsid 放到后台，定期看日志进度，不要中途打断。
   - 不要用 pytest --forked，它在 GPU 上会让所有用例失败。
6. 跑完执行 python3 tools/delivery_table.py <结果目录>。

规矩：
- 不要为了"让它通过"去改 src/ 或 tests/ 下的代码。确实非改不可（比如环境适配）时，每改一个文件，
  都按 docs/debug/MACA.md 末尾"实机改动记录"一节的格式追加一条记录，并和代码改动放在同一个 commit 里。
- 不要猜根因。遇到报错就给出完整报错原文，以及出错变体在结果目录里那一行的 reason/error 字段。
- 折算出来的加速比是估算，低也照实报，不要为了数字好看换基线或改参数。
- 结果目录整个打包带回（至少 summary.json、summary.csv、各算子的 performance.csv 和 *_stderr.log）。

汇报包含：
1. delivery_table.py 的完整输出，尤其是第一行 "N registered variants, M missing"。
2. 所有非 Passed 的变体、以及加速比为 "-" 的变体：变体名、阶段（精度/性能）、reason 或 error 原文。
3. 逐条回答 docs/debug/MACA.md "跑完要确认"一节。
4. 环境指纹：torch、triton 版本，设备名，驱动/SDK 版本，git log -1 --oneline，整轮总耗时。
5. docs/debug/MACA.md "实机改动记录"一节的内容（没改写"无"）。
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

### 2026-10-15 MACA SDDMM / SpGEMM 调优
- 改了什么：`sddmm_csr.py` 为 MACA complex64 选择 `(BLOCK_P, BLOCK_K, num_warps)=(16,16,2)`，并保留 MACA pair-load 复数 kernel 和非连续输入 contiguous 路径；`spgemm_csr.py` 保留 float32 宽行时的 MACA PyTorch CSR dispatch，并让 dispatch 对缺少 `n_rows` 的轻量 prepared/mock 对象安全回退。改动均限定在 MACA runtime 分支。
- 为什么改：C550 SDDMM 复数宽 tile 的寄存器/地址计算开销较高，实测窄 tile 更快；SpGEMM 的宽行 PyTorch CSR 路径在部分矩阵略快，但目前整体仍未达到 H800 折算 0.8x，不把局部改善当作完成。
- 怎么验证的：`tests/pytest/test_q4_variants_accuracy.py -k 'sddmm_csr'` 为 `10 passed`，`-k 'spgemm_csr'` 为 `2 passed`；`tests/ci/test_spgemm_tle_gate.py tests/ci/test_sddmm_sweep.py` 为 `15 passed`。C550 的 10 矩阵 q4 SDDMM 实测按现有 H800 折算分母计算，转置 float32 算术平均约 `2.17x`，complex64 约 `1.14x`。SpGEMM float32 的 10 矩阵精度均通过；现有 65 算子结果中 SpGEMM H800 折算加速比算术平均约 `0.15x`，仍需继续优化。TLE fill-table A/B 因本机 `_TLE_AVAILABLE=False` 未执行到 kernel，结果作废，不据此改参数。
- 其他后端是否也需要：否；运行时分支只影响 MACA。轻量 mock 兼容只避免非 MACA dispatch 测试因属性缺失崩溃。

### 2026-10-15 `tests/test_spgemm.py`：MACA PyTorch CSR baseline
- 改了什么：在 MACA、`FLAGSPARSE_MACA_VENDOR=torch` 和 SciPy 精度参考模式下，独立计时 `torch.sparse.mm(A_csr, B_csr)` 的 float32/float64 baseline；新增 baseline 状态、格式和误差字段。计时输出必须通过独立 SciPy 校验，错误值、NaN/Inf、异常或无效时延均不发布 `pytorch_ms` 和本机加速比，不污染 FlagSparse 的精度结果。
- 为什么改（附报错原文）：原 vendor 分支返回 `PyTorch CSR SpGEMM baseline is not wired for this runner`，但本机 PyTorch CSR 乘法实际可执行。仅接入性能参考，不修改 FlagSparse 的 prepare、计算和输出写回计时范围。
- 怎么验证的：新增 `tests/ci/test_spgemm_maca_baseline.py`，18 项覆盖 float32/float64、异常和错误输出、非有限值、无效时延、运行时门控及精度/性能隔离；与 dispatch、比较器、FP64 参考、TLE 门控、结果格式回归一起运行，共 `94 passed`。首轮 `results_metax_spgemm_baseline_20261015/` 的精度阶段 `12 passed`，10 矩阵 × 2 dtype 的算子精度及 baseline 校验均 `PASS`。
- 其他后端是否也需要：否；baseline 新路径仅在 MACA 启用，其余后端原行为不变。

### 2026-10-15 `src/flagsparse/sparse_operations/spgemm_csr.py`：MACA FP64 native dispatch
- 改了什么：MACA 宽行 PyTorch CSR dispatch 从 float32 扩展到 float32/float64；A 平均每行至少 10 项，或至少 800000 行且平均每行至少 5.5 项时启用。短行保留原 hash/ESC，其他后端完全保留原 dispatch。
- 为什么改：接入真实 baseline 后，首轮 float64 的 10 矩阵算术平均仅 `0.798037x`；其中 `cfd2`、`filter3D` 的 ESC 加速比为 `0.278583x`、`0.286582x`，而本机 PyTorch CSR 的 FP64 输出通过 SciPy 校验。
- 怎么验证的：新增 `tests/ci/test_spgemm_maca_dispatch.py`，18 项覆盖两种 dtype、阈值边界、空行、非 MACA 路由及 native 路径 live value 更新；连同 baseline 和现有相关回归共 `94 passed`。完整矩阵 A/B 和后续 runner 数据单独记录，不覆盖首轮结果。
- 其他后端是否也需要：否；仅影响 MACA runtime 分支。

### 2026-10-15 `docs/MACA.md`、`docs/debug/MACA.md`：baseline 口径同步
- 改了什么：增加 SpGEMM 本机 PyTorch CSR baseline 的复现命令、CSV 字段和失败策略，更新旧的“SpGEMM 一律无 baseline”说明，并记录上述 MACA dispatch 的 dtype 和阈值。
- 为什么改：本机有效 PyTorch baseline 应优先使用；它与 H800 带宽折算是两种不同评估口径，不能把基线补齐当成 kernel 加速，也不能把 native dispatch 的约 1x 宣称为纯 Triton 性能。
- 怎么验证的：首轮交付投影 `spgemm_csr_f32_int_non_non` 为精度/性能 `Passed`，本机加速比 `1.059200x`，原始数据在 `spgemm_csr/performance.csv`。使用 `tools/baseline_bound.py --all-rows` 另存 `h800_bound_all_rows.csv`，首轮 float32/float64 的 H800 折算算术平均为 `0.283576x` / `0.158081x`，仍未达 `0.8x`。
- 其他后端是否也需要：否；本节描述 MACA 实机行为。

## 2026-10-15 SpGEMM baseline 接线后的受控 A/B

结果保存在 `results_metax_spgemm_baseline_20261015/spgemm_ab.csv` 和
`spgemm_ab_summary.json`，复现脚本为同目录的 `spgemm_ab.py`。该脚本不修改仓库源码；
只在自己的进程中临时切换 `_spgemm_compute`，并确认本机 `_TLE_AVAILABLE=False`，
因此“原路径”明确指本机原先实际执行的 ESC，而不是没有运行过的 TLE kernel。

```bash
env PYTHONPATH=src FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_VENDOR=torch \
  python3 -u results_metax_spgemm_baseline_20261015/spgemm_ab.py --dtypes float32 float64
```

使用 `tests/data` 的 10 个矩阵，warmup=5、iters=20；按矩阵交替测量顺序。
FlagSparse 两条路径均包含 prepare、计算和输出写回；PyTorch baseline 单独计时 CSR mm。
原 ESC 和当前 dispatch 的 20 个输出均通过独立 SciPy 校验，20 个 baseline 全部有效。
下表的数字均为 10 矩阵逐行比值的算术平均，不是总时延之比。

| dtype | 原 ESC / 当前路径 | PyTorch / 原 ESC | PyTorch / 当前路径 |
|---|---:|---:|---:|
| float32 | 1.672425x | 0.792104x | 1.039500x |
| float64 | 1.613562x | 0.795328x | 1.034217x |

本机 PyTorch baseline 下两种 dtype 的平均值均超过 0.8x。宽行路径调用的是同一个
PyTorch CSR 后端，不把约 1x 解释为自研 Triton kernel 已达标；H800 带宽折算独立评估，
不因本机 baseline 补齐而删除或替换历史未达标结论。

## 2026-10-15 SpGEMM 最终版本正式 runner 复测

结果目录：`results_metax_spgemm_baseline_20261015_opt/`；整轮日志为同名 `.log`。
使用 `docs/MACA.md` 7.3.1 的命令，仅将结果目录替换为该新目录；输入为
`tests/data` 的 10 个交付矩阵，warmup=5、iters=20，float32/float64、int32 索引。
本机环境为 MetaX C550，torch `2.10.0+metax3.8.1.0`、triton `3.6.0`，
源码基于 `a330b62` 加本节记录的未提交改动。

- 精度阶段：`12 passed, 4241 deselected`，无失败、无 skipped。
- 性能阶段：20/20 行 `status=PASS`，20/20 行 `pytorch_baseline_status=PASS`，runner 正常退出。
- 正式性能阶段耗时约 650.81 秒；prepare 和输出写回始终包含在 FlagSparse 时延内。
- 单算子运行的交付投影 `spgemm_csr_f32_int_non_non` 为精度/性能 `Passed`、`1.045125x`。
  其他 64 个变体为 `NotFound`，本轮没有复测它们，不能作为新一轮 65 算子全通过的证据。

| dtype | 本机 PyTorch 平均加速比 | H800 带宽折算平均加速比 |
|---|---:|---:|
| float32 | 1.045125x | 0.284665x |
| float64 | 1.033220x | 0.283301x |

平均值均按 10 个矩阵逐行比值做算术平均。本机结果来自 `spgemm_csr/performance.csv`；
另用以下命令对全部 20 行计算 H800 折算，结果保存为 `h800_bound_all_rows.csv`，
不覆盖正式 runner 的本机 baseline 和汇总。

```bash
python3 tools/baseline_bound.py \
  --vendor results_metax_spgemm_baseline_20261015_opt --vendor-card maca-c550 \
  --all-rows --ops spgemm_csr \
  --csv results_metax_spgemm_baseline_20261015_opt/h800_bound_all_rows.csv
```

该工具逐行判定为 1 PASS / 19 FAIL，退出码 1 是 H800 折算未达标，不是算子精度失败。
折算保持原有带宽比例 `3050/1440` 和默认 memory bottleneck 假定，属于估算。
float64 本机平均从首轮 `0.798037x` 提升到 `1.033220x`；但两种 dtype 的 H800 折算
平均仍远低于 0.8x，不能宣称全部性能目标已完成。后续要达该目标需要进一步优化
MACA SpGEMM 实现，不能仅补 baseline 或调整汇总口径。

## 2026-10-15 后续：仅 MACA 的纯 Triton SpGEMM

本节取代上面的 MACA native/PyTorch 计算 dispatch；历史结果保留，不代表当前实现。
用户要求计算使用纯 Triton，baseline 仅用本机 PyTorch，不再评估 H800 折算。

### `src/flagsparse/sparse_operations/_spgemm_maca.py`
- 改了什么：新增不依赖 TLE 的 MACA SpGEMM。Triton 计算逐行展开量和产品前缀，按 64/256/1024/4096 大小桶在行内排序、分段求和；更长行使用全局显存开放寻址哈希。递归前缀扫描生成 int64 CSR 指针，再压紧输出。仅通过 `torch.empty` 分配存储，不调用 PyTorch 稀疏乘法、sort、cumsum、index_add 或旧 ESC；形状分配需要读取两个计数标量，不缓存数值结果。
- 为什么改（附报错原文）：本机没有 TLE；动态 `tl.gather` 编译报 `custom op 'maca.shfl.sync' is unknown`。因此携带产品位置做标量排序，再直接重索引输入，使用反向扫描定位分段边界，避免动态跨 lane gather。C550 128 元素多 warp CAS/add 布局在长行测试出现一半列数值错误，改为 64 元素/单 warp 后，float32/float64 及两种指针类型均通过。没有通过 Torch fallback 隐藏错误。
- 怎么验证的：合成精度、长行强碰撞、空维度、重复列、空 RHS 行、live value、out 复用、跨块递归扫描及 int32 最大列索引测试共 `39 passed`；正式 runner 精度同为 `39 passed`。所有 10 个交付矩阵的最大单行展开量不超过 3340，均走行内排序归并；长行哈希另由强制边界用例验证。
- 其他后端是否也需要：否；模块只从 MACA runtime 分支导入和调用。临时存储按逐行展开上界分配，不使用旧 ESC 的分块回退；非常大的展开量仍可能需要进一步限制峰值显存，不能视为已验证所有超大稀疏矩阵。

### `src/flagsparse/sparse_operations/spgemm_csr.py`
- 改了什么：移除 `_maca_torch_spgemm_compute` 及原宽行阈值；MACA 的真实 `SpGEMMPrepared` 一律调用纯 Triton 实现，prepare 的 metadata 也走 MACA Triton kernel。公共输入校验、contiguous、输出写回和端到端计时口径不变。非 MACA 的 TLE/hash/ESC 分支保持原样。
- 为什么改：不能把与 baseline 相同的 PyTorch CSR 实现作为待优化 kernel；需要以独立实现与 PyTorch 做真实比较。
- 怎么验证的：正式精度 `39 passed`；`tests/ci/test_spgemm_maca_dispatch.py` 验证 MACA/非 MACA、TLE 可用/不可用时路由隔离；相关 baseline、比较器和 FP64 参考回归共 `36 passed`。
- 其他后端是否也需要：否。

### `tests/ci/test_spgemm_maca_dispatch.py`
- 改了什么：将旧 native dispatch 测试改为纯 Triton 路由测试，新增 AST 检查，约束 MACA 计算模块的 Torch 调用只能是存储分配，并确认旧 `_maca_torch_spgemm_compute` 已移除。
- 为什么改：防止后续重新接入 PyTorch sparse 或张量算法而被误报为纯 Triton 性能。
- 怎么验证的：与 baseline、比较器、FP64 参考、TLE 门控回归合计 `36 passed`。
- 其他后端是否也需要：否；只约束 MACA 新路径，同时检查其他后端路由不变。

### `tests/pytest/test_spgemm_maca_accuracy.py`
- 改了什么：新增 MACA 专用纯 Triton 边界精度测试；包括长行/强碰撞、float32/float64、int32/int64 指针、空行/空维度、重复列、空 B 行、prepared 值更新、out 复用、三级递归扫描及最大 int32 列索引。核心测试将 `torch.sparse.mm` 和旧 ESC 替换为抛异常，防止隐式 fallback。
- 为什么改：交付矩阵不能覆盖超过 4096 个产品的行，必须单独验证全局哈希路径和扫描跨块边界。
- 怎么验证的：新增 27 项测试全部通过，连同既有 SpGEMM 精度/reuse/q4 共 `39 passed`。
- 其他后端是否也需要：否；门控为 MACA。

### `docs/MACA.md`、`docs/debug/MACA.md`
- 改了什么：7.3.1 的 SpGEMM runner 命令去掉 `--h800-reference`，说明本机 PyTorch CSR 是唯一性能 baseline；更新纯 Triton 计算路径与验证记录。本次正式结果使用新目录 `results_metax_spgemm_triton_20261015/`。
- 为什么改：按用户要求不做折算，也不混用上轮 native/PyTorch 计算结果；其他算子的历史 H800 文档保留，不借本次任务改变它们的测试口径。
- 怎么验证的：本轮启动命令与文档一致，warmup=5、iters=20，输入为 `tests/data` 的 10 个矩阵；结果在本轮完成后按 dtype 单独记录。
- 其他后端是否也需要：否。

### 纯 Triton 最终实测结果

结果目录为 `results_metax_spgemm_triton_20261015/`，启动命令如下；不带 `--h800-reference`。

```bash
env PYTHONPATH=src FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_VENDOR=torch \
  python3 -u run_flagsparse_pytest.py \
  --ops spgemm_csr --phase both --mode normal --gpus 0 --timeout 2400 \
  --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir results_metax_spgemm_triton_20261015
```

环境：MetaX C550、torch `2.10.0+metax3.8.1.0`、triton `3.6.0`，基于 `a330b62`
加上述未提交改动。精度阶段 `39 passed, 4241 deselected`，无失败或 skipped；性能
20/20 行 `status=PASS`，20/20 行 `pytorch_baseline_status=PASS`，runner 正常退出。
精度阶段约 18.66 秒，性能阶段约 719.99 秒。相关回归和结果格式验证共 `81 passed`。

所有加速比均为本机独立测得的 `pytorch_ms / triton_ms`，CSV 没有 H800 折算列。
两侧输入一致，warmup=5、iters=20；FlagSparse 时延仍包括 prepare、计算和输出写回。

| 矩阵 | float32 | float64 |
|---|---:|---:|
| ASIC_680ks | 2.136395x | 2.092398x |
| GL7d14 | 2.090525x | 2.113907x |
| NACA0015 | 0.998436x | 1.100590x |
| amazon0601 | 2.374381x | 2.389323x |
| auto | 1.895073x | 1.896815x |
| cage12 | 1.424595x | 1.441711x |
| cfd2 | 0.839985x | 0.841748x |
| filter3D | 0.464664x | 0.476849x |
| roadNet-TX | 1.301999x | 1.244056x |
| wave | 2.195355x | 2.224231x |
| 10 矩阵算术平均 | 1.572141x | 1.582163x |

按用户“平均超过 0.8x，不要求每个矩阵都高”的标准，两种 dtype 均达标。
`filter3D` 单矩阵仍低于 0.8x，不能表述为逐矩阵全部达标；后续可继续细分排序桶降低该矩阵
的 4096 桶开销。本轮仅复测 SpGEMM，不代表另外 64 个交付变体重新测过。
原始 20 行数据、baseline 状态和误差见 `spgemm_csr/performance.csv`；不要用本节数字
覆盖上轮 native/PyTorch 计算路径的 CSV。

## 2026-10-15 修改文件交付与复测范围确认

### 另外两项低于 0.8x 的 SDDMM 变体优化

除 SpGEMM 外，本轮还优化了以下两个交付变体。它们都在
`src/flagsparse/sparse_operations/sddmm_csr.py` 中实现，不能因只有一个源码文件就漏记
其中一个变体。优化前数据来自 `results_metax_65_20261014/summary.json` 和
`sddmm_csr/performance.csv`；优化后数值为前述调优阶段的 10 矩阵算术平均记录，
不是新一轮完整 65 变体 runner 的结果。

| 交付变体 | 优化前平均加速比 | 调优阶段平均加速比 | baseline 口径 |
|---|---:|---:|---|
| `sddmm_csr_f32_int_trans_non_row` | 0.662x | 约 2.17x | H800 带宽折算 |
| `sddmm_csr_c32_int_non_non_row` | 0.413x | 约 1.14x | H800 带宽折算 |

#### `sddmm_csr_f32_int_trans_non_row`

- 改了什么：`_apply_sddmm_dense_ops` 在完成 op/布局转换后，仅在 MACA、float32 或
  complex64 且 K 维步长不为 1 时，对对应的 X/Y 调用 `contiguous()`。已经连续的输入
  不复制；同一分支也覆盖列主序及其他转置组合，不为某个矩阵缓存计算结果。
- 为什么改：转置视图使 K 维加载跨大步长，C550 上 nnz 并行 sampled-dot 的访存代价
  很高。先连续化再执行 kernel，可减少后续重复的跨步访问；连续化开销仍保留在调用路径，
  不通过移出计时范围改善数字。
- 怎么验证的：既有 q4 SDDMM 合成精度为 `10 passed`，覆盖转置和布局变体；10 个交付
  矩阵调优阶段的平均折算加速比约 `2.17x`，超过用户按算术平均 `0.8x` 的目标。
- 其他后端是否也需要：否；分支由 `_is_maca_runtime()` 保护，其他后端保留零复制步长路径。

#### `sddmm_csr_c32_int_non_non_row`

- 改了什么：新增 `_sddmm_csr_complex_kernel_maca`，将交错 complex64 的实部/虚部
  成对加载后用 `tl.split` 分离，计算复数乘积并沿 K 求和，保留 alpha/beta 和输入累加
  语义；`_resolve_sddmm_launch_config` 为 MACA complex64 固定选择
  `(BLOCK_P, BLOCK_K, num_warps)=(16,16,2)`。仅 MACA complex64 的默认
  `baseline` kernel 路由到该实现，诊断 kernel 和其他后端路由不变。
- 为什么改：原复数宽 tile 在 MACA lowering 下寄存器和地址计算开销较高；成对加载与
  窄 tile 减轻这些开销，提高可驻留 program 数量。这里的 `baseline` 是内部 kernel
  变体名，不代表调用 PyTorch 计算。
- 怎么验证的：同组 q4 SDDMM 合成精度 `10 passed`；相关 SDDMM sweep 与 SpGEMM
  TLE 门控回归合计 `15 passed`。10 矩阵调优阶段的平均折算加速比约 `1.14x`，超过
  算术平均 `0.8x` 目标，不代表每个矩阵都超过 0.8x。
- 其他后端是否也需要：否；kernel 选择与参数覆盖均只在 MACA runtime 生效。

以上是已有优化和验证记录的补全，本次没有新增 kernel 修改或重新测试。SDDMM 这两项
仍沿用历史 H800 折算评估，不能写成本机 PyTorch 实测加速比；SpGEMM 最新结果则仅
使用本机 PyTorch CSR baseline，两者必须分开汇报。全部当前修改的 65 变体复测仍待执行。

### 压缩包内容与最新复测范围

当前基于 `a330b62b46df3a2048bd14c95650eb4f82698185` 的未提交修改打包为
`flagsparse_maca_changes_20261015.tar.gz`，保留仓库相对路径。包内为当前完整文件，
不是 diff；仅收录下列 9 个源码、测试和文档文件，不包含历史测试结果、诊断脚本、
矩阵数据或已有压缩包。

- `docs/MACA.md`
- `docs/debug/MACA.md`
- `src/flagsparse/sparse_operations/sddmm_csr.py`
- `src/flagsparse/sparse_operations/spgemm_csr.py`
- `src/flagsparse/sparse_operations/_spgemm_maca.py`（新增）
- `tests/test_spgemm.py`
- `tests/ci/test_spgemm_maca_baseline.py`（新增）
- `tests/ci/test_spgemm_maca_dispatch.py`（新增）
- `tests/pytest/test_spgemm_maca_accuracy.py`（新增）

本次交付保留已有 MACA SDDMM 调优、SpGEMM PyTorch baseline 接线，以及最新纯
Triton SpGEMM 实现和回归测试；kernel 和 dispatch 的新增行为限定在 MACA 后端。
本次仅更新文档并打包，没有重新运行测试。

截至此次打包，最新正式复测仍是 `results_metax_spgemm_triton_20261015/`：
SpGEMM 合成精度 `39 passed`，10 矩阵 × float32/float64 的 20 行性能及本机 PyTorch
baseline 均通过，平均加速比分别为 `1.572141x`、`1.582163x`。`39` 是精度用例数，
不是算子数；另外 64 个交付变体未用当前修改重新测试。历史
`results_metax_65_20261014/` 不能作为当前修改的全量 65 算子验证结果。

## 2026-11-03 MACA SDDMM 本机 PyTorch baseline 补齐

### `tests/q4_variant_bench.py`：仅 MACA 的校验失败回退

- 改了什么：新增 `_time_maca_sddmm_pytorch`，只在 `_sddmm` 的 MACA 分支调用。
  优先校验并计时 `torch.sparse.sampled_addmm`；异常、错误结果、NaN/Inf、形状错误、
  计时返回值错误或无效时延，均改试已有的 `_sddmm_reference`（纯 PyTorch 取数、乘法、求和）。
  两条路径都校验计时前的输出和实际计时返回值；最终失败不发布 `pytorch_ms`，不修改
  FlagSparse 自身的精度状态。保留原 sampled API 的失败原因和实际公式标记。
- 为什么改（附报错原文）：完整 105 项轮次中 4 个新增 SDDMM 变体无加速比，原因为
  `PyTorch result off by ... (relative); baseline not used`；另外 5 个旧 SDDMM q4 变体
  也没有本机 PyTorch baseline，仅靠 H800 折算。旧逻辑仅在 API 抛异常时改用取数求和，
  MACA 的 API 返回错误结果却不抛异常，因此没有进入备用路径。
- 计时口径：op/布局的转置视图、dtype 转换、Y 连续化和 CSR descriptor/指针转换仍在计时外，
  与原有备用路径一致；逐行 row ID 构造、取数、乘法、求和及输出分配均在计时内。
  float16 和 float16→float32 均明确使用 float32 计算；complex64 不做共轭乘法。
  没有修改 kernel、FlagSparse 耗时、精度门槛或 H800 规则。
- 怎么验证的：接线前 9 变体 × 10 矩阵独立诊断 90/90 通过，CPU float64/complex128
  参考下最大归一化误差约 `1.4e-7`。接线后以同一 q4 benchmark 的 `run_variants` /
  `append_to_csv` 路径定向验证，结果另存 `results_metax_sddmm_baseline_20261103/`，
  warmup=5、iters=20，最终结果见下表。不把这次定向测试称为新一轮完整 105 项复测。
- 其他后端是否也需要：否；非 MACA 仍走原 `_time_pytorch`，原 API 异常回退和 Ascend
  复数 real-plane 路径保留。SpMV CSR cage12 float64 的历史偶发 NaN 不在本次接线范围内。

### `tests/ci/test_q4_sddmm_maca_baseline.py`

- 改了什么：新增 9 变体 × MACA/非 MACA × 正确/错误/异常的路由和精度隔离测试；覆盖
  NaN/Inf、计时输出损坏、形状错误、无效时延、fallback 自身失败及历史错误字段隔离。
- 为什么改：防止错误 API 被当作有效基线、失败结果发布时延，或者新 MACA 行为影响其他后端。
- 怎么验证的：连同 Ascend complex reference、Iluvatar baseline/validation、SDDMM sweep
  和 runner 结果格式回归合计 `184 passed`。
- 其他后端是否也需要：否；回归测试同时检查非 MACA 行为保持不变。

### 接线后 C550 定向实测结果

`results_metax_sddmm_baseline_20261103/` 中 90/90 行算子精度 `PASS`，90/90 行有有效本机
`pytorch_ms`，全部 `pytorch_formulation=gather/multiply/sum`，最终 `pytorch_reason` 均为空。
原 sampled API 的 90 条失败原因仍保存在 `pytorch_sampled_addmm_reason`。CPU 高精度
参考下 baseline 最大归一化误差约 `1.24e-7`。没有使用 H800 折算。

| 变体 | 10 矩阵 PyTorch / FlagSparse 算术平均 |
|---|---:|
| `sddmm_csr_f32_int_non_non_col` | 20.839693x |
| `sddmm_csr_f32_int_non_trans_row` | 17.861593x |
| `sddmm_csr_f32_int_trans_non_row` | 18.337541x |
| `sddmm_csr_c32_int_non_non_row` | 4.418412x |
| `sddmm_csr_f16_int_non_non_row` | 7.059146x |
| `sddmm_csr_f32_int_trans_trans_row` | 20.882196x |
| `sddmm_csr_c32_int_non_trans_row` | 4.879380x |
| `sddmm_csr_f16_int_non_trans_row` | 23.591153x |
| `sddmm_csr_f16f32_int_non_non_row` | 6.934029x |

以上仅是本机非融合 PyTorch 表达式的性能比值，不能与历史 H800 融合厂商基线的比值
直接比较，也不能将分母变化解释为 kernel 优化。所有平均值均超过 0.8x，但不要求或
宣称每个矩阵都超过 0.8x。输入与计时配置为 10 个交付矩阵、K=32、5 warmup / 20 iters；
不是传统 f32/f64 的 K=32/64/128/256 完整 sweep。

独立合成精度：`pytest -q tests/pytest -m sddmm_csr --mode normal --tb=short` 为
`27 passed, 4 skipped, 5068 deselected`；4 个跳过均为非 MACA 的 BI-V 专用测试。
相关回归为 `184 passed`。数据及日志为 `performance.csv`、`rows.json`、`performance.log`、
`accuracy.log` 和 `regression.log`。定向复现脚本只调用仓库现有 benchmark 接口，不替换
kernel 或修改基线函数；输出必须选新目录，例如：

```bash
env PYTHONPATH="$PWD/src" FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_MODEL=c550 \
  FLAGSPARSE_MACA_VENDOR=torch \
  python3 -u results_metax_sddmm_baseline_20261103/verify_wiring.py \
  results_metax_sddmm_baseline_<新日期>
```

完整 105 项仍使用本文顶部的 `run_flagsparse_delivery.py` 命令。历史
`results_metax_105_20261103_1253/` 和其压缩包未修改，仍如实反映接线前 4 项无加速比的状态。

## 2026-11-03：四个低于 0.8x 变体的 MACA 优化

基于 `39e400a6c8b5a140b9f174c2543ddbaa9d728de0` 的本地改动，保留已完成的
SDDMM baseline 接线。优化前数据来自 `results_metax_105_20261103_pytorch_rerun1/`。
本次只优化四个慢项，kernel/dispatch 的新分支仅在 `_is_maca_runtime()` 时启用。
没有改动 baseline、性能比较器、容差或交付汇总算法。

### 源码改动及原因

- `src/flagsparse/sparse_operations/spmv_csc.py`：MACA 的 float32/complex64
  转置及共轭转置，将 CSC(A) 直接视作 CSR(A.T)，prepare 时建立批量行 gather 计划；
  compute 复用已有 MACA 单 wave/复数组件成对加载 kernel，替代每列一个 program 的
  串行段循环。共轭在 kernel 中执行，避免把数值共轭副本冻结在 prepare 内，原地修改
  values 或输入向量后仍读取当前值。float64/complex128 以及非 MACA 路径沿用原实现。
- `src/flagsparse/sparse_operations/_spmv_csc_maca.py`（新增），以及上述 CSC 文件的
  int8 分支：int8→int32 用缓存的输出行拓扑和纯 Triton int32 gather 归约，消除每次
  调用的 `repeat_interleave` 列号构造及 COO atomic scatter。缓存只保存结构，按
  indices/indptr 的身份、版本、存储及步长失效；无版本的 inference tensor 不缓存。
  数值每次实时读取，先转 int32 再乘加，超过 2**24 的整数和仍精确。支持 int32/int64
  结构索引、非连续输入/输出和时间/meta 返回；int8→float32 继续走原 mixed 路径。
- `src/flagsparse/sparse_operations/_spgemm_maca.py`：在现有 MACA 实数算法上扩展
  complex64 的实/虚部乘法、分段扫描、长行双分量 atomic_add 和结果压紧。仍使用
  原有短行排序、长行全局哈希、递归 CSR 前缀扫描，保留 C550 的 64 元素/单 wave
  CAS 限制。Torch 只分配存储或创建实/虚部视图，不承担稀疏乘法、排序或归约。
- `src/flagsparse/sparse_operations/spgemm_csr.py`：移除 MACA 专用分支排除 complex64
  的条件，使它不再落到通用 ESC；非 MACA 的 TLE/hash/ESC 分派不变。SpGEMM
  端到端计时仍包含 prepare、分配、计算和输出写回。

### 性能结果与口径

同一 C550、`tests/data` 的全部 10 个 `.mtx`，5 warmup / 20 iters，仓库现有 q4
benchmark。基线是通过独立 CPU 参考检查的本机 PyTorch；没有使用 H800 折算。
int8 SpMV 的 PyTorch baseline 仍按原约定使用 float32 计算，本次没有改变。
统计量为 **10 个逐矩阵 `pytorch_ms / ours_ms` 的算术平均值**，不是总时间比或几何平均。

| 变体 | 优化前均值 | 优化后复核均值 | 复核精度 |
| --- | ---: | ---: | --- |
| `spmv_csc_f32_int_trans` | 0.264700x | 2.908067x | 10/10 PASS |
| `spmv_csc_c32_int_conj` | 0.163333x | 2.444874x | 10/10 PASS |
| `spmv_csc_i8i32_int_non` | 0.500280x | 2.356811x | 10/10 PASS |
| `spgemm_csr_c32_int_non_non` | 0.349008x | 1.640396x | 10/10 PASS |

首轮 `results_metax_four_20261103_optimized/` 的四项均值分别为
5.951700x、2.170039x、2.334786x、1.639724x；wave 的 f32 PyTorch 基线以及
cage12 的 c32 FlagSparse 耗时出现毫秒级抖动，因此完整复核了相同 40 组，没有剔除
首轮异常行。复核数据在 `results_metax_four_20261103_confirm/`，表格使用完整复核轮。
两轮均为 40/40 PASS，四项均值均超过 0.8x。复核时 SpGEMM 的 filter3D 仍为
0.502126x；验收按平均值，不宣称所有单矩阵均超过 0.8x。

此次未重新跑完整 105 项，不覆盖原 105 项交付结果；此前该 runner 的传统附加
`spmm_csc / amazon0601 / float32 / conj` 行仍有精度失败，不能将本次定向通过
写成完整 runner 零退出或全仓库无失败。

### 精度、隔离和复现

最终相关合成精度为 `231 passed, 136 deselected`，覆盖 CSC 原有全 dtype/op/index
测试、新增 MACA CSC 边界、SpGEMM 实/复数、q4 相关变体以及 CSR MACA gather 回归。
另有 `61 passed` 的 dispatch、PyTorch baseline、FP64 参考、比较器、TLE gate 和结构缓存
回归。新增/扩展的测试文件为：

- `tests/pytest/test_spmv_csc_maca_accuracy.py`（新增）：空维度、长行、live values、
  共轭视图、精确大整数和、结构修改、inference、跨步输入/输出、时间/meta。
- `tests/ci/test_spmv_csc_maca_dispatch.py`（新增）：18 项验证后端/dtype/op 选择隔离。
- `tests/pytest/test_spgemm_maca_accuracy.py`：扩展 complex64，检查空维度、live
  values、重复列及长行哈希冲突；禁用 Torch sparse 和 ESC，确保运行纯 Triton。
- `tests/ci/test_spgemm_maca_dispatch.py`：加入复数分派检查，仅允许 Torch 存储分配
  和 `view_as_real`，禁止数值计算替代。CUDA/ROCm 等后端未在本机实卡运行。

性能脚本、两轮 `rows.json`/CSV/日志均保留；首轮目录还有 `summary.json`，记录
逐项均值、源码 SHA256 和比较范围。再次运行必须选一个不存在结果的目录：

```bash
env PYTHONPATH="$PWD/src" FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_MODEL=c550 \
  FLAGSPARSE_MACA_VENDOR=torch \
  python3 -u results_metax_four_20261103_optimized/benchmark.py \
  results_metax_four_<新日期>
```

该命令复测本节四个变体，使用与 runner 相同的 `tests/q4_variant_bench.py`
实现和输入口径；完整 105 项仍使用本文顶部的 delivery runner。最终测试日志为
`results_metax_four_20261103_optimized/accuracy_final.log` 和 `regression.log`。

## 2026-11-04：MACA `axpby_c32_int` 融合优化

前一轮完整结果 `results_metax_105_20261103_after_opt/` 为 105/105 精度和性能 Passed、
missing=0、runner exit=0；四项 CSC/SpGEMM 优化均超过 0.8x，唯一低于目标的是
`axpby_c32_int`（0.624039x）。本节优化该项，不修改或覆盖那一轮结果。

### 改动文件和原因

- `src/flagsparse/sparse_operations/_vector_maca.py`（新增）：用输出位置到 sparse
  value 位置的 inverse map，将 `beta*y` 的全向量缩放与稀疏 `alpha*x` 更新融合为
  单个标准 Triton kernel。按相邻实/虚部分量成对读写，支持实数或复数 alpha/beta，
  每个输出只有一个 writer，不使用原子累加。显式屏蔽缺失项的 alpha 贡献，避免
  NaN alpha 污染不存在 sparse 值的输出位置。
- inverse map 通过已有 `TensorPlanCache` 缓存，绑定原始 indices 的对象、版本、
  存储和步长；结构改变即重建，inference 无版本张量不缓存。values 和 y 从不缓存，
  始终读取 live 数据。Torch 只用于存储和实/虚部视图，数值计算在 Triton 内完成。
- `src/flagsparse/sparse_operations/vector_ops.py`：新分支仅用于 MACA complex64
  且 dense length <= 4*1024*1024；避免稀疏度极低、大长度向量的 inverse map 过大，
  单份 map 最大 16 MiB、8 项缓存最多 128 MiB。超过长度上限以及其他 dtype/后端仍用
  原有路径。原有输入/索引校验、in-place 返回语义、时间测量均保留。
- `tests/pytest/test_axpby_maca_accuracy.py`（新增）：49 项 MACA 实机测试，覆盖
  int32/int64、跨步 sparse 输入、实时 values/y、索引原地修改、inference、空向量、
  未排序全密索引、复数系数、NaN alpha 和热缓存后的索引范围校验。
- `tests/ci/test_axpby_maca_dispatch.py`（新增）：8 项验证 MACA/非 MACA、dtype 和
  长度边界隔离；CUDA/ROCm 等后端未进行本机实卡测试。

原实现对 complex64 先启动全向量 scale，再启动 sparse axpy，存在两次 launch、两次
读写稀疏位置以及额外 Python 调度；新实现只启动一个 fused compute kernel。冷缓存
首次建图的成本仍属于调用本身，性能沿用原有 warmup 后 steady-state 口径。

### 性能复测

Axpby 不使用矩阵输入：它的交付性能是 **4 个向量规模**，与 SpMV 的 10 矩阵不同。
沿用 runner 的 5 warmup / 20 iters、本机 PyTorch baseline、CPU complex128 精度参考，
没有改变 `tests/test_vector_ops.py` 的 baseline、容差或计时窗口，也没有 H800 折算。
原 runner 定向精度+性能结果 `results_metax_axpby_20261104_runner/`：

| dense length | nnz | FlagSparse ms | PyTorch ms | 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 32768 | 1024 | 0.031140 | 0.035692 | 1.146206x |
| 131072 | 4096 | 0.036004 | 0.051218 | 1.422543x |
| 524288 | 16384 | 0.036245 | 0.046459 | 1.281804x |
| 1048576 | 65536 | 0.039894 | 0.105858 | 2.653463x |

4 项均为 PASS，逐项加速比算术平均 **1.626004x**，超过 0.8x。f16 和 f32 路径未修改，
该轮平均分别为 1.191988x、1.025935x。prototype 第一轮 complex64 平均为 1.581934x。
首次小规模合成精度为 53 passed，增加范围校验测试后的第二轮 runner 为 55 passed。

第二次原 runner 复核保存在 `results_metax_axpby_20261104_confirm/`，12 个性能用例
全部 PASS，complex64 的 FlagSparse 耗时仍为 0.028831–0.041327 ms，但 PyTorch 在
524288/1048576 两个规模变为 3.278335/0.338852 ms，导致平均抬到 22.718896x。
这个比值是 baseline 抖动，**不当作真实优化幅度**；汇报用上面第一轮 1.626004x。
优化前临时重测 `before.csv` 也在 PyTorch 端出现过 4.178638 ms 抖动，原始行均保留，
没有删掉异常数据或修改原测试汇总。原完整 105 轮的 0.624039x 也含 FlagSparse
131072 规模 2.797442 ms 抖动，不能把前后平均值之比全部解释为 kernel 提升。

最终 `pytest tests/pytest -m 'axpby or spvv' --mode normal` 为
**67 passed, 5129 deselected**；dispatch 和缓存回归为 **11 passed**。两轮定向 runner
均退出 0。此次修改后只重测 Axpby 与相关 SpVV 回归，尚未再次全量复测 105 项，
不能覆盖原 105 项交付表或宣称新一轮完整 105 项全部达标。

复现时使用新目录：

```bash
env PYTHONPATH="$PWD/src" FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_MODEL=c550 \
  FLAGSPARSE_MACA_VENDOR=torch \
  python3 -u run_flagsparse_pytest.py --ops axpby --phase both --mode normal \
  --delivery-only --gpus 0 --timeout 4500 --benchmark-input tests/data \
  --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir results_metax_axpby_<新日期>
```

原始前后 CSV、日志、精度和回归日志在 `results_metax_axpby_20261104_optimized/`。
该目录的 `summarize.py` 可核对原完整 105 轮、前后临时测量、两轮 runner 的所有
complex64 用例，写入 `summary.json` 并记录源码 SHA256，不修改原始性能 CSV。

## 2026-11-04：Axpby 优化后的 105 项全量复测及打包

复测当前本地工作树（HEAD `39e400a6c8b5a140b9f174c2543ddbaa9d728de0`），
使用 `run_flagsparse_delivery.py --gpus 0`，normal 精度、10 个交付矩阵、5 warmup /
20 iters；Axpby/SpVV 沿用向量用例。总限时 43200 秒、单算子单阶段 4500 秒，
沿用本机 PyTorch 优先、缺失时 H800 折算的 runner 口径。

新结果为 `results_metax_105_20261104_final/`，日志为同名 `.log`。后台 workflow
在 runner 退出后自动打包，即使测试失败也保留真实退出码、失败项和已有结果，不
把部分结果写成全量通过。包名为 `flagsparse_metax_105_20261104_final.tar.gz`，
旁边生成 SHA256 校验文件；进度及最终汇总在
`bundle_metax_105_20261104_final/metadata/status.json`。

本次只 fetch，不 pull/merge。比较上游固定为 fetch 后的
`origin/main = 1976a01a67fdaa9f48bfb9e675ccbfc077787635`，比本地 HEAD 新 3 个提交。
包内同时包含相对最新上游的全部差异文件，以及单独的本地修改清单/补丁；上游新增、
本地未同步的 runner/文档/测试差异另行标记，不能当作本地 MACA 优化。打包使用
开跑前文件快照，结束时复核源文件是否保持不变；结果见包内 manifest。

### 本轮按用户要求中止，先交付差异与现有结果

2026-11-04 用户要求先停测试、先打包。已停止本轮 workflow、runner 和测试子进程，
不会再自动测试或生成上述 final 包。本轮 `results_metax_105_20261104_final/` 是
**未完成的部分结果**：完成 gather、scatter、Axpby、SpVV、GEMVI，停止时正在
SpMV CSR 性能阶段；本轮 Axpby complex64 的平均比值为 1.283805x。

改为生成 `flagsparse_metax_changes_results_20261104.tar.gz`，包含：

- 对比 `origin/main=1976a01` 的 25 个差异文件（16 个本地修改/新增，9 个上游
  已更新而本地尚未同步），分开记录本地修改补丁和完整上游比较补丁。
- 上一轮已完成的 `results_metax_105_20261103_after_opt/`（105 项全部精度/性能
  Passed，runner exit=0；**不包含后续 Axpby 优化**），及对应完整日志。
- Axpby 优化的两轮定向 runner、CSC/SpGEMM 四项优化的两轮定向结果、SDDMM
  baseline 接线证据，以及新增 Axpby/SpVV、后端分派和缓存回归日志。
- 本轮中断的部分结果、日志及中止说明。原始 CSV/JSON 不改写，不伪装成完整通过。

包内 `metadata/README.md`、`manifest.json` 和 `results_overview.json` 说明各组结果
对应的代码阶段；源码保留仓库相对路径，并附源码及结果文件 SHA256 清单。
