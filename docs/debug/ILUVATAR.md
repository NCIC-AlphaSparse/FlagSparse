# 天数智芯 Iluvatar（BI-V150）：65 变体验证任务

`FLAGSPARSE_BACKEND=iluvatar`（通常能自动识别）。环境搭建、交付复现见 [`../ILUVATAR.md`](../ILUVATAR.md)，
上机排查记录见 [`../ILUVATAR_DEBUG.md`](../ILUVATAR_DEBUG.md)。背景见本目录 [`README.md`](README.md)。

> SELL / COO 的优化任务（瓶颈探针 `tools/biv_spmv_probe.py`、`prepare_spmv_sell` 的 A/B 命令和交给 Codex 的 prompt）
> 见 [`ILUVATAR_SELL_COO_PLAN.md`](ILUVATAR_SELL_COO_PLAN.md) 第 9 节。

## 要做的事

环境按 `../ILUVATAR.md` 第 1 节配好后（**一定要显式设 `FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse`，
不要用 `torch`**——见下面第一条风险点）：

```bash
setsid timeout -s KILL 43200 \
  env FLAGSPARSE_BACKEND=iluvatar FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse \
  python3 -u run_flagsparse_pytest.py \
    --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc,spgemm_csr,sddmm_csr \
    --phase both --mode normal --gpus 0 --timeout 3600 \
    --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 \
    --h800-reference \
    --pytest-args='-k "test_q4_variant or (((float or half or f16 or f32) and not (double or float64 or bfloat16 or complex64 or complex128)) or (complex64 and (gather or scatter)))"' \
    --op-benchmark-args='gather=--value-dtypes float16,float32,complex64' \
    --op-benchmark-args='scatter=--value-dtypes float16,float32,complex64,int8' \
    --op-benchmark-args='spmv_csr=--dtypes float32 --alg auto' \
    --op-benchmark-args='spmv_coo=--dtypes float32' \
    --op-benchmark-args='spmv_csc=--dtypes float32,complex64' \
    --op-benchmark-args='spmm_csr=--dtypes float16,float32' \
    --op-benchmark-args='spmm_coo=--dtypes float16,float32' \
    --op-benchmark-args='spmm_csc=--dtypes float32,complex64' \
    --op-benchmark-args='sddmm_csr=--dtype float32' \
    --op-benchmark-args='spgemm_csr=--dtypes float32' \
    --results-dir results_iluvatar_65_<日期> \
  > results_iluvatar_65_<日期>.log 2>&1 < /dev/null &
# 跑完后
python3 tools/delivery_table.py results_iluvatar_65_<日期>
```

这条命令是 `../ILUVATAR.md` 第 2 节 20 变体交付命令的扩展，三处改动都在 CUDA 上核对过：

- **`-k` 前面加了 `test_q4_variant or`**。原来的 `-k` 是为了挡掉 fp64（本卡 fp64 H2D 静默返回 0），但它
  同时会把 45 个新变体里的 c32/i8 精度用例筛掉（只剩 54/91）；加上之后 91/91 全保留，fp64/complex128
  用例仍然一个不收。新变体自己不含 fp64，精度对照是 CPU 上算的。
- **`scatter` 加了 `int8`**：`scatter_i8_int` 的性能取自 scatter 脚本自己的 int8 行，原来的
  `float16,float32,complex64` 会让它变成 `NotFound`。
- **新增 `spmv_csc`/`spmm_csc` 的 dtype 限制**：这两个脚本默认会跑 float64/complex128，先跑 fp64 万一
  卡死或崩溃，排在最后追加的新变体行就丢了；它们的交付变体只要 f32/c32/f16，f16 由新变体那一路单独测。
- 原 20 个变体里的 9 个 f64/c64 在本卡上照旧测不了，`delivery_table.py` 里显示 `NotFound` 是预期的；
  `spgemm_csr` 的性能 runner 默认按 dtype 拆开分别跑 float32/float64；命令里的
  `--op-benchmark-args='spgemm_csr=--dtypes float32'` 让它只跑 float32（交付变体只有 f32，fp64 那个子进程在本卡
  只会失败，崩溃时还会把整个 spgemm 性能阶段判成 FAIL）。精度侧的 fp64 用例已被 `-k` 挡掉。
- **`spgemm_csr` 这轮第一次真正能跑**：10-09 那轮 20 行全部是 TLE 编译错误
  （`iluvatar TLE alloc does not support nv_mma_shared_layout=True`），现在天数上改走 ESC（见文末改动记录）。
  跑完确认 `spgemm_csr_f32_int_non_non` 精度 Passed、有加速比；还报 TLE 就说明没拉到新代码。
- **`--h800-reference` 现在能补上 q4 变体**：10-09 那轮命令也带了它，但当时的参照只有 20 个变体，q4 全对不上，
  所以全是 `-`。参照已换成 H800 的 65 变体全量，按那轮数据预演，有加速比的变体 25 → 55（再加 spgemm 就是 56，
  其余 9 个是 fp64）。`sddmm_csr_f16` 用的是 H800 PyTorch 折算，见 [`../H800_REFERENCE.md`](../H800_REFERENCE.md)。

- 外层 `timeout -s KILL` 是整条命令的总限时（内核卡死时 Ctrl-C 送不进去，只能靠 KILL）；`--timeout` 是
  每个算子每个阶段的限时。这两个数是 20 变体时期定的，65 变体多了 6 个父算子，**没有在本机实测过总耗时**，
  到点被杀的话按 `delivery_table.py` 里 `NotFound` 的算子单独补跑（补跑要用新的 `--results-dir`）。
- `--results-dir` 每次都用新目录：`summary.json` 每跑一次就整体重写，往旧目录里补跑一部分会把之前的结果冲掉。

把 `delivery_table.py` 的完整输出带回来，尤其是第一行的 `missing` 数字。

## 已知风险点（这个后端特有的，不是猜的，而且比其他三个后端更多）

- **fp64 的 H2D 拷贝静默返回全 0**（`../ILUVATAR_DEBUG.md` 已定位根因）。45 个新变体不含 f64/c64，
  理论上不受影响，但如果测试过程中任何中间步骤意外提升到 fp64 精度（比如某个参考实现），结果会静默
  错误而不是报错，这是这张卡上最危险的一类问题，运行时多留意。
- **`torch.sparse` 在这张卡上结果是错的，不是不支持**（`../ILUVATAR_DEBUG.md`）。早期一版文档把它
  当作性能基线，已经改成默认 `cupy_cusparse`。45 个新变体走的 `tests/q4_variant_bench.py`
  （`_time_pytorch`）从 2026-10-08 起会先拿 PyTorch 结果和 CPU 参考值比，相对误差超过 1e-2 就不计时、
  在 `pytorch_reason` 里写 `PyTorch result off by ...`——**所以这张卡上 45 个新变体的 PyTorch 列大量
  出现这个 reason 是预期行为**，说明拦截生效了。原有 20 个变体走各自脚本，不经过这道校验，那部分的
  PyTorch 数字仍然要先怀疑。
- **通用 `cusparseSpMV` 在这张卡上会申请约 140 TB 显存**，只能用 CoreX 的 legacy
  `cusparseScsrmv`/`cusparseScsrmm`，且只覆盖 fp32 + int32 + 不转置。45 个新变体里的转置/共轭/
  混合精度/int8 变体大概率没有厂商基线，这是预期行为。

## 跑完要确认（2026-10-14 之后这一轮）

1. `missing` 是不是 0。9 个 f64/c64 变体 `NotFound` 是预期的，其余 56 个精度应 Passed。
2. `spgemm_csr_f32_int_non_non`：精度 Passed、有加速比；日志/CSV 里不应再出现
   `iluvatar TLE alloc does not support nv_mma_shared_layout=True`（出现就说明没拉到新代码）。
3. 有加速比的变体个数（预演 56 = 10-09 预演的 55 + spgemm）。`python3 tools/delivery_table.py <结果目录> --show-baseline` 的输出原样贴回。
4. 任何一行的 PyTorch 列"看起来正常但可疑"（这张卡的 torch.sparse 有静默算错的先例）；int8/混合精度变体精度是否真的通过。

把这些结果写回来，不用额外分析，原始数据最有用。

## 交给 Codex 的 prompt

上机时把下面整段原样贴给 Codex（或其他代理），不用改任何内容。

````text
你在 天数智芯 BI-V150（Iluvatar） 实机上复测 FlagSparse 的 65 个交付变体（2026-10-14 之后的这一轮）。仓库在当前目录，main 分支。

背景：这台机器上一轮已经跑过 65 项。之后仓库改了两类东西，这轮就是验证它们：
- 没有基线的变体，现在用仓库自带的 H800 结果按带宽折算出加速比（命令里的 --h800-reference），交付表里只显示数字，
  delivery_table.py --show-baseline 的最后一列说明哪些是折算的。有本机基线的变体不受影响。
- 本后端相关的修复，见 docs/debug/ILUVATAR.md "跑完要确认"一节逐条列出的内容。
主要目的是收集数据，不是修代码。

步骤：
1. git pull --ff-only origin main；git log -1 --oneline 的提交应不早于 d7d1919（"ci again"）。
2. 按 docs/ILUVATAR.md 配好环境，做完它的环境自检。
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
5. 完整读一遍 docs/debug/ILUVATAR.md，照"要做的事"一节的命令原样跑（已带 --h800-reference，不要去掉）。
   - --results-dir 用新目录（<日期> 填今天），不要复用旧目录，也不要往旧目录里补跑。
   - 命令本身已经用 setsid 放到后台，定期看日志进度，不要中途打断。
   - 不要用 pytest --forked，它在 GPU 上会让所有用例失败。
6. 跑完执行 python3 tools/delivery_table.py <结果目录>。

规矩：
- 不要为了"让它通过"去改 src/ 或 tests/ 下的代码。确实非改不可（比如环境适配）时，每改一个文件，
  都按 docs/debug/ILUVATAR.md 末尾"实机改动记录"一节的格式追加一条记录，并和代码改动放在同一个 commit 里。
- 不要猜根因。遇到报错就给出完整报错原文，以及出错变体在结果目录里那一行的 reason/error 字段。
- 折算出来的加速比是估算，低也照实报，不要为了数字好看换基线或改参数。
- 结果目录整个打包带回（至少 summary.json、summary.csv、各算子的 performance.csv 和 *_stderr.log）。

汇报包含：
1. delivery_table.py 的完整输出，尤其是第一行 "N registered variants, M missing"。
2. 所有非 Passed 的变体、以及加速比为 "-" 的变体：变体名、阶段（精度/性能）、reason 或 error 原文。
3. 逐条回答 docs/debug/ILUVATAR.md "跑完要确认"一节。
4. 环境指纹：torch、triton 版本，设备名，驱动/SDK 版本，git log -1 --oneline，整轮总耗时。
5. docs/debug/ILUVATAR.md "实机改动记录"一节的内容（没改写"无"）。
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

### 2026-10-14 src/flagsparse/sparse_operations/spgemm_csr.py::_spgemm_compute
- 改了什么：天数上 SpGEMM 不再走 TLE 共享内存哈希路径，和 MUSA、XPU 一样走纯 ESC（`tle_hash_safe` 加上 `_is_iluvatar_runtime()`）。
- 为什么改（附报错原文）：天数的 FlagTree 能 import TLE，但编译哈希表时报
  `Memory allocation failed: iluvatar TLE alloc does not support nv_mma_shared_layout=True`，
  `results_iluvatar_65_1009` 里 spgemm 的 20 行（f32/f64 × 10 矩阵）全部 FAIL，精度 2/2 失败。
  （那轮汇总把性能写成 Passed 是 `04135dd` 时 runner 的问题，当前 main 会判为 Failed。）
- 怎么验证的：CUDA 机器上 `FLAGSPARSE_BACKEND=iluvatar` 模拟：哈希路径调用 0 次，结果与稠密乘积一致；
  `tests/pytest -m spgemm_csr` 11 passed；`tests/ci/test_spgemm_tle_gate.py`（旧代码上失败）。**未在 BI-V150 实测**，
  ESC 在天数上的性能也还没有数。
- 其他后端是否也需要：否（MUSA、XPU 早已排除）。

### 2026-10-21 src/flagsparse/sparse_operations/sddmm_csr.py::_run_sddmm_prepared
- 改了什么：仅 BI-V 的 float32、K=32、平均行长 >4，且两个 dense 操作数的 K 维均不连续时，在计算窗口内将输入转为连续布局。每次读取当前输入，不缓存 dense 数值；其余后端和配置保持原路由。
- 为什么改：`sddmm_csr_f32_int_trans_non_row` 的跨步 gather 很慢；同轮 10 矩阵平均耗时 5.9473 ms，打包后的最终路线为 1.1552 ms。短行 ASIC_680ks / roadNet-TX 的打包试验退化，保留零拷贝路径。complex64 扫参没有稳定收益，未保留其试验改动。
- 怎么验证：`tests/data` 的 10 个矩阵、warmup=5、iters=20，q4 CPU 参考全部 PASS。5 个 SDDMM q4 变体共 50 行 PASS，调整短行阈值后额外复测受影响的 f32 转置变体 10/10 PASS。新增 prepared 重复调用、输入原地更新、alpha/beta、out 复用和 int32/int64 行指针精度回归。
- 性能口径：沿用 `results_iluvatar_65_1014_spgemm_f32/sddmm_csr/performance.csv` 的 `h800_scaled_ms`，逐矩阵加速比算术均值约 0.757 → 1.480；7/10 矩阵达到 0.8，cage12/cfd2/wave 仍分别约 0.721/0.480/0.650。这是 H800 带宽折算估计，不是本机厂商基线；尚未重跑全 65 变体 runner。
- 实测数据：`/tmp/biv_sddmm_layout_all_1021_False.csv`（同轮修改前），`/tmp/biv_sddmm_final_trans_1021.csv`（最终路线），`/tmp/biv_sddmm_final_1021.csv`（阈值调整前 50 行）。
- 其他后端是否也需要：否，本次仅验证和启用 BI-V。

### 2026-10-21 BI-V 的 COO SpMM f16 与 SDDMM complex64 后续优化
- 改了什么：`spmm_coo.py::_run_spmm_coo_rowrun_route` 为 BI-V prepared f16 路线使用 8 行、32 列、单 warp 的现有 mixed rows kernel；保留 fp32 稀疏值和重复坐标合并，但 dense B 不再整块转 fp32，kernel 内 fp32 累加并直接写 f16。原子路线和其他后端不变。`sddmm_csr.py::_run_sddmm_prepared` 为 BI-V complex64、K=32 每次打包当前 dense 输入，采用 8/32/1 tile；不缓存计算结果或 dense 数值。
- 为什么改：COO f16 的 B 升精度、C 降精度和单行 grid 开销较高；complex64 SDDMM 的跨步访存和大 tile 寄存器压力共同限制性能。此次未修改基线或计时窗口，打包成本包括在调用时间内。
- 怎么验证：最终源代码按 warmup=5、iters=20 在 `tests/data` 的 10 个矩阵上验证 COO c32/f16/i8 和 SDDMM c32/f32-trans 共 50 行，CPU 参考全部 PASS；新增输入原地修改、alpha/beta、out、空行、重复坐标消去、N=0/17/32/65 和 row/col 布局回归，相关 pytest 28 passed。`py_compile` 和 `git diff --check` 通过。
- 最终加速比（逐矩阵比值算术均值，沿用 10-14 的 H800 带宽折算基线）：COO f16 1.4322，SDDMM c32 0.9693，均超过 0.8。未改路径对照 COO c32 0.6502、COO i8 1.2341；此前优化的 SDDMM f32-trans 1.4525。
- 原始结果：`/tmp/biv_eight_final_spmm_coo.csv`、`/tmp/biv_eight_final_sddmm_csr.csv`；精度日志 `/tmp/biv_eight_accuracy.log`。尚未全量重跑 65 变体。
- 未保留的试验：SELL 行/slot tile（32x4、64x4、128x2）、CSR 紧凑重排、compiled launcher、循环流水化均未达到目标，正式 SELL 路径未进一步修改。COO c32 的双行串行累加约 0.781，成对加载与 K 归约 tile 均未超过 0.8，流水化降至约 0.128，未保留。SDDMM 分离实虚平面、沿 nonzero 维向量化、成对加载均不如最终小 tile。
- 剩余：SELL 的 f32/f16/f16f32/i8i32/c32 五项仍约 0.434/0.408/0.403/0.396/0.393，COO c32 约 0.650，共 6 项未达标；未宣称 8 项全部完成。
- 其他后端是否也需要：否，所有新分派仅 BI-V。

### 2026-10-22 BI-V SELL 后续调优
- 改了什么：`spmv_sell.py` 的 BI-V 实数 SELL 路径在 `slice_size <= 32` 时使用 `BLOCK_R=64`，仍保持单 warp；当 slice 数量很大时合并两个 slice。complex64 路径不改。
- 怎么验证：`tests/test_spmv_sell.py tests/data --dtypes float32,float16,complex64,int8,float16:float32 --slice-size 32 --warmup 5 --iters 20`，10 个矩阵共 50 行全部 PASS；对应日志为 `/tmp/biv_sell_final_1022.log`。
- 结果：相对 `results_iluvatar_65_1014_rerun/spmv_sell/performance.csv` 的 H800 折算基线，10 矩阵逐矩阵加速比平均为 f32 `0.431`、f16 `0.396`、f16→f32 `0.395`、int8→int32 `0.390`、complex64 `0.402`。仍未达到 `0.8x`，因此没有宣称 SELL 完成。
- 未保留的试验：slot 并行、CSR 压缩后 row-tile、CSR 轻量 row-tile、complex slot=8 和双 warp 均低于当前实现，已撤回。

### 2026-10-22 BI-V CSR SpMM 转置 f32 后续调优
- 核对结论：`results_iluvatar_65_1014_spgemm_f32` 是修改前的全量报告，未达标项不止 SELL；后续已单项覆盖 COO f16/c32 和 SDDMM c32/f32-trans。当前继续处理 `spmm_csr_f32_int_trans_non_row`。
- 改了什么：BI-V 的 CSR SpMM f32 accuracy 路径允许使用现有短行批处理 kernel；只对 f32 放宽精度路线限制，其他 accuracy dtype 仍走原标量 kernel。BI-V 转置 CSR 的 materialized route 仍在 prepared 阶段缓存。
- 怎么验证：`tests/test_spmm.py tests/data --dtypes float32 --warmup 5 --iters 20`，70 个 q4 行全部 PASS；`spmm_csr_f32_int_trans_non_row` 的 10 矩阵平均加速比由约 `0.591` 提升到约 `0.696`。
- 当前状态：转置 f32 仍未达到 `0.8x`，后续继续优化该项及其他旧报告中仍低于目标的 CSR/CSC 算子；SELL 五项也仍未完成。

### BI-V mixed-baseline runner

`benchmark_biv.py` is the dedicated 65-variant entry point. It invokes the shared
`run_flagsparse_pytest.py` runner with the BI-V profile and selects a baseline per
matrix row: a correctness-passing local vendor sparse-library measurement first,
then a correctness-passing local PyTorch measurement, and finally the H800
reference scaled to BI-V. The generated performance CSV carries
`biv_baseline_kind`/`biv_baseline_source`; `delivery_table.py --show-baseline`
reports `local_vendor`, `local_pytorch`, `h800_scaled`, or a mixed label.

```bash
python3 -u benchmark_biv.py --results-dir results_biv_65_$(date +%Y%m%d)
```

### 2026-10-24 合入 `flagsparse_changes_and_results_20261024`（在 CUDA 机器上合并时补的两处）
- 合入：天数侧 15 个文件的改动（补丁可干净应用到 `c814047`）、`benchmark_biv.py`、两个天数精度测试。
  `sweep_spmv_real_complex.py`（根目录的调参草稿，写死 `torch.cuda` 与矩阵路径）**没有合入**。
- 修了什么（补丁引入，`FLAGSPARSE_BACKEND=iluvatar` 模拟下 15 个用例失败，干净的 main 上为 0）：
  1. `mixed_spmx.spmv_needs_mixed`：天数 f16 SpMV 一律走 mixed，但 mixed 只支持 `op='non'`，
     `test_spmv_csr_matches_dense_reference[trans|conj-*-float16-*]` 12 个用例报
     `NotImplementedError: mixed-precision/int8 paths support op='non' only`。**这 12 个在 10-24 真机结果的 spmv_csr
     精度里同样是 failed**（交付表看不出，因为不归到任何交付变体）。现在只在 `op='non'`、不转置、未显式指定
     `alg/config/use_opt` 时启用（新增关键字参数 `op`/`transpose`/`explicit_route`，其他调用方不变）。
  2. `spmm_csc.flagsparse_spmm_csc` 的天数 CSC→CSR 快速路径：没有检查 B（报错文字和原路由不一致）、
     绕过了 int64 索引的回退与 `index_fallback_policy="strict"`。现在只在 int32 索引且策略为 `auto` 时启用，
     并先用 `_validate_spmm_csc_B` 检查 B。交付用 int32，不受影响。
- 怎么验证的：CUDA 上 13 个交付算子精度 1787 passed；模拟天数 1136 passed（合并前 15 failed）、MACA 1137、MUSA 1068
  全过；`tests/ci` 468 passed（唯一失败为已知的 `test_installed_wheel`），无 torch 的 CI 模拟、`make pre-commit-check`
  通过。**未在 BI-V150 上重跑。**
- 低于 0.8 的 13 个变体的分析与建议见 [`LOW_SPEEDUP.md`](LOW_SPEEDUP.md)。
- 其他后端是否也需要：否（两处修复都只影响天数分支）。
