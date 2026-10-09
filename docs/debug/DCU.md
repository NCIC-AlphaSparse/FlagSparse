# 海光 DCU（BW1000，ROCm）：65 变体验证任务

`FLAGSPARSE_BACKEND=rocm`（通常能自动识别）。环境搭建、交付复现见 [`../DCU.md`](../DCU.md)。背景见
本目录 [`README.md`](README.md)。

## 要做的事

环境按 `../DCU.md` 第 1、2 节配好后（**务必先做第 3 节的 diagnose，不要直接跑基准**）：

```bash
task_tmp=$(mktemp -d /tmp/flagsparse-dcu-65.XXXXXX)
export TMPDIR="$task_tmp" TMP="$task_tmp" TEMP="$task_tmp"   # Slurm 残留的 TMPDIR 会让 clang 编译失败，见 ../DCU.md 第 0 节
export HIP_VISIBLE_DEVICES=<卡号>                              # runner 用 CUDA_VISIBLE_DEVICES 隔离设备，ROCm 上会打乱随机种子
setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc,spgemm_csr,sddmm_csr \
  --phase both --mode normal --gpus 0 --timeout 3600 \
  --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 \
  --op-benchmark-args 'spmv_csr=--alg auto' \
  --results-dir results_rocm_65_<日期> \
  > results_rocm_65_<日期>.log 2>&1 < /dev/null &
# 跑完后
python3 tools/delivery_table.py results_rocm_65_<日期>
```

- `--benchmark-input tests/data`：仓库里的 `tests/data` 正好就是 `conf/operators.yaml` 的 10 个交付矩阵（在 git
  里，拉下来就有）。这条命令没带 `--delivery-only`，runner 不会再筛矩阵，给 30 个矩阵的目录就会 30 个全跑，
  汇总出来的加速比也会掺进非交付矩阵，所以不要换成别的目录。
- `--timeout 3600`、外层 KILL 限时沿用 `../DCU.md` 的 20 变体交付命令。

- 外层 `timeout -s KILL` 是整条命令的总限时（内核卡死时 Ctrl-C 送不进去，只能靠 KILL）；`--timeout` 是
  每个算子每个阶段的限时。这两个数沿用 20 变体时期的设置；2026-10-08 的 65 变体实测耗时为 `40m11s`，
  在总限时内完成。若后续到点被杀，按 `delivery_table.py` 里 `NotFound` 的算子单独补跑（补跑要用新的
  `--results-dir`）。
- `--results-dir` 每次都用新目录：`summary.json` 每跑一次就整体重写，往旧目录里补跑一部分会把之前的结果冲掉。
- ~~`tests/test_spmv_csr.py` 的默认算法是 `compare`……必须保留 `--op-benchmark-args 'spmv_csr=--alg auto'`~~
  **已在 runner 里修掉（2026-10-08）**：`run_flagsparse_pytest.py` 的 spmv_csr 性能命令改成了 `--alg auto`，交付
  投影也只认 `alg_requested=auto` 的行，有失败行时变体直接判 FAIL。上面命令里的
  `--op-benchmark-args 'spmv_csr=--alg auto'` 现在是多余的，留着无害。已经生成的首轮原始结果不要重写或删除，
  应注明其统计口径。

把 `delivery_table.py` 的完整输出带回来，尤其是第一行的 `missing` 数字。

## 已知风险点（这个后端特有的，不是猜的）

- **hipSPARSE 只支持不转置**（`../DCU.md` 第 4 节）。2026-10-08 实测中有 9 个转置/共轭变体没有
  厂商基线，但 PyTorch 对照均有效，详见下方结果。`tests/cusparse_generic_baseline.py`（给 q4 变体用的
  ctypes cuSPARSE 绑定）本身只接 CUDA，在 DCU 上会由 `skip_reason()` 判空，不会尝试调用 hipSPARSE。
- **int8 变体仍应单独核对实际用例数**：`scatter_i8`、`spmv_csr_i8i32`、`spmv_csr_i8f32`、
  `spmm_csr_i8i32`、`spmv_coo_i8i32`、`spmv_sell_i8i32`、`spvv_i8i32`。2026-10-08 首次实测均通过，
  具体结果见下方；不能仅凭没有报错或没有厂商基线就判定通过。
- **不要把 `spmv_csr --alg compare` 的合并值当成生产路径性能**。2026-10-08 首轮结果中的 FP32
  `0.207x` 和 FP64 `0.236x` 各自混合了 10 个矩阵、7 种算法；同一份 CSV 中单独筛选 `row_tile` 后，
  FP32 为 `0.738x`（9 个可比矩阵），FP64 为 `0.795x`（10 个矩阵）。runner 已改为默认 `--alg auto`
  （见"要做的事"一节），新跑次的交付数字就是生产路径。
- **如果 `spmv_csr_f16_int_non`/`c32_int_non`/`f32_int_trans`/`c32_int_conj` 这几个变体精度报
  `NotImplementedError: CSR SpMV row_tile: unverified capabilities for rocm/<arch>`，这是一个需要
  立刻上报、不要自己绕过去的信号**。原因：`auto` 模式下 ROCm 的 CSR SpMV 固定选
  `row_tile`（`spmv_csr.py::_configure_spmv_route` 里 `"row_tile" if _is_rocm_runtime() else
  "legacy_" + backend`，这是 gfx936 调优过的生产路径，不是新代码临时选择），而 `row_tile` 要求
  `_spmv_backend_caps()` 算出 `verified=True`——这个值只在 `triton.runtime.driver.active
  .get_current_target()` 真的报 `backend=="hip"` 且 `arch` 以 `"gfx"` 开头时才成立
  （`spmv_csr.py:_spmv_backend_caps`）。这条在 CUDA 机器上**不可能**模拟到（Triton 驱动查的是真实
  硅片，设什么环境变量都测不出 `hip`/`gfx`），所以这 4 个变体在单纯设环境变量的 CUDA 模拟里全部挂在
  这一步。之后补测过：把 `_spmv_backend_caps()` 改成返回 gfx936 的已验证能力，让 `row_tile` 带着
  ROCm 调优参数真实跑起来，45 个新变体 91/91 全过——逻辑本身没问题，但前提是真实 DCU 上 Triton-ROCm 驱动能正确报出
  `hip`/`gfx*`。**如果真机上也报这个 `NotImplementedError`，说明 Triton-ROCm 驱动没有正确识别硬件，
  这是真 bug（或者环境没装对），务必带回具体的 `arch`/`target` 字符串。**

## 2026-10-08 BW1000 实测结果

测试基于 `main` 的 `263a139`（`debug leads`），矩阵目录为 `tests/data`，其中有 10 个矩阵。结果目录为
`results_rocm_65_20261008`，后台日志为 `results_rocm_65_20261008.log`，整轮耗时 `40m11s`。本轮使用的
环境变量和 runner 命令如下；这是保留用于追溯的**首轮原始命令**，其中尚未加入后文说明的 SpMV CSR
`--alg auto` 修正：

```bash
task_tmp=$(mktemp -d /tmp/flagsparse-dcu-65.XXXXXX)
export TMPDIR="$task_tmp" TMP="$task_tmp" TEMP="$task_tmp"
export HIP_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD/src"
export FLAGSPARSE_BACKEND=rocm
setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc,spgemm_csr,sddmm_csr \
  --phase both --mode normal --gpus 0 --timeout 3600 \
  --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir results_rocm_65_20261008 \
  > results_rocm_65_20261008.log 2>&1 < /dev/null &
```

环境指纹：

- 仓库源码：`/public/home/guochengxin/gcx/FlagSparse/src/flagsparse/__init__.py`
- PyTorch `2.4.1`，Triton `3.6.0`
- 设备：`BW1000 64G`
- Triton target：`hip gfx936 64`（backend、arch、warp size）
- PyTorch HIP SDK `6.1.25065`，`hipconfig` `6.1.25085`，DTK `25.04`
- hip-python `7.2.2.562.43`
- 驱动 `6.3.31-V1.5.6`，hy-smi `1.24.1`

环境自检和 hipSPARSE diagnose 均通过，没有为了跑通而修改 `src/` 或 `tests/`。运行
`python3 tools/delivery_table.py results_rocm_65_20261008` 的完整输出如下：

```text
results_rocm_65_20261008/summary.json: 65 registered variants, 0 missing; accuracy {'Passed': 65}; performance {'Passed': 65}
  #  variant                          accuracy    performance   speedup
  1  gather_f16_int                   Passed      Passed        2.334x
  2  gather_f32_int                   Passed      Passed        1.260x
  3  gather_f64_int                   Passed      Passed        1.238x
  4  gather_c32_int                   Passed      Passed        1.153x
  5  gather_c64_int                   Passed      Passed        1.198x
  6  scatter_f16_int                  Passed      Passed        1.835x
  7  scatter_f32_int                  Passed      Passed        1.753x
  8  scatter_f64_int                  Passed      Passed        1.414x
  9  scatter_c32_int                  Passed      Passed        1.379x
 10  scatter_c64_int                  Passed      Passed        1.103x
 11  spmv_csr_f32_int_non             Passed      Passed        0.207x
 12  spmv_csr_f64_int_non             Passed      Passed        0.236x
 13  spmv_coo_f32_int_non             Passed      Passed        0.890x
 14  spmv_coo_f64_int_non             Passed      Passed        0.839x
 15  spmm_csr_f32_int_non_non_row     Passed      Passed        0.881x
 16  spmm_csr_f64_int_non_non_row     Passed      Passed        0.936x
 17  spmm_coo_f32_int_non_non_row     Passed      Passed        1.453x
 18  spmm_coo_f64_int_non_non_row     Passed      Passed        1.473x
 19  sddmm_csr_f32_int_non_non_row    Passed      Passed        2.155x
 20  sddmm_csr_f64_int_non_non_row    Passed      Passed        2.889x
 21  scatter_i8_int                   Passed      Passed        1.858x
 22  axpby_f16_int                    Passed      Passed        0.813x
 23  spmv_sell_f32_int_non            Passed      Passed        0.706x
 24  spmv_csr_f16f32_int_non          Passed      Passed        0.910x
 25  spvv_f16f32_int_non              Passed      Passed        0.833x
 26  spmv_csr_f16_int_non             Passed      Passed        0.796x
 27  spmv_csr_f32c32_int_non          Passed      Passed        0.877x
 28  spmv_sell_f16_int_non            Passed      Passed        0.744x
 29  spmm_csr_f16f32_int_non_non_row  Passed      Passed        2.164x
 30  spmm_csr_f16_int_non_non_row     Passed      Passed        2.048x
 31  spmm_csr_f32_int_non_non_col     Passed      Passed        4.892x
 32  spmm_csr_f32_int_non_trans_row   Passed      Passed        4.891x
 33  spmv_csc_f32_int_non             Passed      Passed        0.632x
 34  spmv_csr_c32_int_non             Passed      Passed        0.738x
 35  spmv_sell_c32_int_non            Passed      Passed        0.806x
 36  spvv_c32_int_conj                Passed      Passed        1.069x
 37  spmv_coo_f16f32_int_non          Passed      Passed        0.354x
 38  spmm_csr_c32_int_non_non_row     Passed      Passed        2.104x
 39  spmv_coo_f32_int_trans           Passed      Passed        0.915x
 40  spmv_csr_f32_int_trans           Passed      Passed        0.890x
 41  spmv_csr_i8f32_int_non           Passed      Passed        0.927x
 42  spmv_csr_i8i32_int_non           Passed      Passed        0.930x
 43  spmv_sell_i8i32_int_non          Passed      Passed        0.768x
 44  spvv_i8i32_int_non               Passed      Passed        0.796x
 45  spmm_csc_f32_int_non_non_row     Passed      Passed        1.481x
 46  spmm_csr_i8i32_int_non_non_row   Passed      Passed        2.246x
 47  spmv_coo_c32_int_non             Passed      Passed        0.728x
 48  spmv_coo_f16_int_non             Passed      Passed        0.340x
 49  spmv_csc_c32_int_non             Passed      Passed        0.386x
 50  spmv_csc_f16_int_non             Passed      Passed        0.200x
 51  sddmm_csr_f32_int_non_non_col    Passed      Passed        2.874x
 52  sddmm_csr_f32_int_non_trans_row  Passed      Passed        6.899x
 53  sddmm_csr_f32_int_trans_non_row  Passed      Passed        1.357x
 54  spmm_coo_c32_int_non_non_row     Passed      Passed        2.004x
 55  spmm_coo_f16_int_non_non_row     Passed      Passed        3.923x
 56  spmm_csr_f32_int_trans_non_row   Passed      Passed        1.012x
 57  spmv_coo_c32_int_conj            Passed      Passed        0.766x
 58  spmv_coo_i8i32_int_non           Passed      Passed        0.643x
 59  spmv_csr_c32_int_conj            Passed      Passed        0.822x
 60  spmm_coo_i8i32_int_non_non_row   Passed      Passed        3.629x
 61  spmm_csc_c32_int_non_non_row     Passed      Passed        0.569x
 62  sddmm_csr_c32_int_non_non_row    Passed      Passed        1.241x
 63  sddmm_csr_f16_int_non_non_row    Passed      Passed        2.980x
 64  spmm_csc_f16_int_non_non_row     Passed      Passed        1.303x
 65  spgemm_csr_f32_int_non_non       Passed      Passed        0.478x
```

### 交付口径问题：SpMV CSR 默认测了全部算法

首轮 runner 实际启动的 SpMV CSR 性能命令包含：

```text
tests/test_spmv_csr.py tests/data --alg compare --warmup 5 --iters 20
```

因为这条 65 变体命令使用显式 `--ops`，runner 不会启用仅在 `--delivery-only` 下生效的 benchmark
收窄参数；而 `delivery_table.py` 按 dtype/index/op 筛选变体时不再按算法筛选。因此
`spmv_csr_f32_int_non` 和 `spmv_csr_f64_int_non` 各自匹配 `10 个矩阵 x 7 种算法 = 70` 行，算法包括
`legacy_rowpar`、`legacy_segbin`、`legacy_bucket_vector`、`row_tile`、`row_vector`、`row_split_reduce` 和
`row_adaptive_split`。`delivery_table.py` 显示的 FP32 `0.207x`、FP64 `0.236x` 是所有算法的算术平均，
不是 ROCm 生产路径 `auto -> row_tile` 的加速比。

从同一份首轮 CSV 只筛选 `row_tile`，结果为：

| dtype | 可比矩阵数 | 算术平均加速比 | 备注 |
| --- | ---: | ---: | --- |
| FP32 | 9 | `0.738x` | `auto.mtx` 精度检查失败，厂商基线通过，因此该行加速比为 N/A |
| FP64 | 10 | `0.795x` | 10 个矩阵均可比 |

后续交付复测应使用“要做的事”一节已修正的命令，即追加
`--op-benchmark-args 'spmv_csr=--alg auto'`。首轮目录和上面的完整输出保留为原始数据，不应据此声称
生产 `row_tile` 路径只有 `0.207x`/`0.236x`。首轮整体 performance 显示 `Passed` 也不代表每个匹配行都
通过；上面的 FP32 `auto.mtx` 失败行正是被多算法聚合掩盖的例子。

### 跑完确认结果

1. `missing = 0`，65 个变体的精度和性能阶段均为 `Passed`。
2. 8 个 int8 变体均实际执行了 2 个精度用例，并通过 `2/2`：`scatter_i8_int`、
   `spmm_coo_i8i32_int_non_non_row`、`spmm_csr_i8i32_int_non_non_row`、`spmv_coo_i8i32_int_non`、
   `spmv_csr_i8f32_int_non`、`spmv_csr_i8i32_int_non`、`spmv_sell_i8i32_int_non`、
   `spvv_i8i32_int_non`。
3. 9 个转置/共轭变体没有厂商基线但有有效 PyTorch 对照：8 个矩阵变体均为厂商 `0/10`、PyTorch
   `10/10`，`spvv_c32_int_conj` 为厂商 `0/4`、PyTorch `4/4`。典型 reason 为
   `no hipSPARSE baseline wired for this operator (ROCm)`。这 9 个变体是
   `sddmm_csr_f32_int_non_trans_row`、`sddmm_csr_f32_int_trans_non_row`、
   `spmm_csr_f32_int_non_trans_row`、`spmm_csr_f32_int_trans_non_row`、`spmv_coo_c32_int_conj`、
   `spmv_coo_f32_int_trans`、`spmv_csr_c32_int_conj`、`spmv_csr_f32_int_trans`、
   `spvv_c32_int_conj`。
4. 4 个 capability 风险变体 `spmv_csr_f16_int_non`、`spmv_csr_c32_int_non`、
   `spmv_csr_f32_int_trans`、`spmv_csr_c32_int_conj` 均通过精度 `2/2`，没有出现
   `unverified capabilities for rocm/...`；Triton target 输出为 `hip gfx936 64`。

## 跑完要确认

1. `missing` 是不是 0。
2. int8 变体的精度是不是真的通过了（不是因为没有基线列、看起来"没报错"就当作过了）。
3. 转置/共轭变体性能列是不是如预期全是 N/A（厂商列）+ 有效数字（PyTorch 列），而不是报错或空值。

把这三条的结果写回来，不用额外分析，原始数据最有用。

## 交给 Codex 的 prompt

上机时把下面整段原样贴给 Codex（或其他代理）。**贴之前把 `<卡号>` 换成本机实际要用的卡。**

````text
你在 海光 DCU（ROCm） 实机上复测 FlagSparse 的 65 个交付变体。仓库在当前目录，main 分支。

背景：65 个变体（原 20 个 + 新合并的 45 个）只在 CUDA 上实测过，本机从没跑过。
任务是在本机跑一遍，把原始结果带回来。主要目的是收集数据，不是修代码。

步骤：
1. git pull --ff-only origin main。
2. 按 docs/DCU.md 配好环境，做完它的环境自检。
3. 确认导入的是仓库源码：python3 -c "import flagsparse; print(flagsparse.__file__)"
   输出必须在当前目录的 src/ 下。如果指向 site-packages / dist-packages，先停下来汇报，不要继续。
   旧安装包会让基线列全部变成 N/A，看起来像正常结果。
4. 完整读一遍 docs/debug/DCU.md，照"要做的事"一节的命令原样跑。
   命令里的 HIP_VISIBLE_DEVICES 填 <卡号>。
   不要自己删减或改写参数，每个参数的来由那一节都写了。
   - --results-dir 用新目录（<日期> 填今天），不要复用旧目录。
   - 命令本身已经用 setsid 放到后台，定期看日志进度，不要中途打断。
   - 不要用 pytest --forked，它在 GPU 上会让所有用例失败。
5. 跑完执行 python3 tools/delivery_table.py <结果目录>。

规矩：
- 不要为了"让它通过"去改 src/ 或 tests/ 下的代码。确实非改不可（比如环境适配）时，每改一个文件，
  都按 docs/debug/DCU.md 末尾"实机改动记录"一节的格式追加一条记录，并和代码改动放在同一个
  commit 里。没改就在那一节写"无"。
- 不要猜根因。遇到报错就给出完整报错原文，以及出错变体在结果目录里那一行的 reason/error 字段。
- 如果 spmv_csr_f16_int_non / c32_int_non / f32_int_trans / c32_int_conj 报
  unverified capabilities for rocm/...，这是真问题，要带回这行命令的输出：
  python3 -c "import triton; t=triton.runtime.driver.active.get_current_target(); print(t.backend, t.arch, t.warp_size)"

汇报包含：
1. delivery_table.py 的完整输出，尤其是第一行 "N registered variants, M missing"。
2. 所有非 Passed 的变体：变体名、阶段（精度/性能）、reason 或 error 原文。
3. 逐条回答 docs/debug/DCU.md "跑完要确认"一节。
4. 环境指纹：torch、triton 版本，设备名，warp size，驱动/SDK 版本，整轮总耗时。
5. docs/debug/DCU.md "实机改动记录"一节的内容。
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

无。
