# 天数 BI-V150：K-wide SELL-C-σ 与 COO 转 SELL 实现计划（2026-10-26）

> 依据：P9 探针（`results_biv_p9_20261026.*`，778 行、0 FAIL，见 [`ILUVATAR_P9_RUNBOOK.md`](ILUVATAR_P9_RUNBOOK.md)）
> 和 10-25 的 prepared 复测（`results_coo_four_low_final_20261025.csv` 等）。
> 加速比口径：10 个矩阵的算术平均，分母为 `results_biv_65_20261024_final` / `results_biv_sell_auto_final_20261025`
> 里的 `h800_scaled_ms`，和交付报告一致。

## 0. 范围和预期

这一轮实现 7 个变体，预期值是从 P9 数据推算的估计：

| 变体 | 现在 | 本轮预期 | 做法 |
|---|---:|---:|---|
| `spmv_sell_f32_int_non` | 0.694 | ≈0.84 | K-wide，K=4 |
| `spmv_sell_f16_int_non` | 0.699 | ≈0.83 | 同上 |
| `spmv_sell_f16f32_int_non` | 0.695 | ≈0.82 | 同上 |
| `spmv_sell_i8i32_int_non` | 0.700 | ≈0.80 | 同上 |
| `spmv_coo_f16_int_non` | 0.831 | ≈0.97 | prepare 时实测选择：现有路径 / SELL K=2 / SELL K=4 |
| `spmv_coo_i8i32_int_non` | 0.805 | ≈0.95–0.97 | 同上 |
| `spmv_coo_f16f32_int_non` | 0.708 | ≈0.86 | 同上 |

**本轮预计没有一项能到 1.0。** 要到 1.0 还需要阶段 2（减少 program 数）和 16 位列偏移，这两项这轮只测、不进生产代码。

**不做**：
- 复数三项（SELL / CSR / COO c32）。P9 投影只能 +0.01–0.08，K-wide 对它们无效。
- `CSR f32c32`、`COO f32 trans`、conj 变体。P9b 测不到这几个。
- CSR f16 / f16f32 / i8i32。现在已经 ≥0.92，而且 CSR 的 mixed 路径按 `call` 计时，没有 prepare 的挂载点。
- SpGEMM。
- 按矩阵名选路由；检测"数值全相同"后跳过读数值；继续扫 tile / σ / warps。

## 1. P9 给出的依据（只列和实现有关的）

- **交付的实数 SELL 就是 K=1 的 kernel**：P9 里 c=32 API 行的时间，和 10-25 交付的 SELL 时间相差约 2% 以内。所以 K-wide 的收益能直接转成交付收益。
- **K-wide 的收益来源**：一个 program 里同时在途的"读列号 → 读 x"依赖链变多了，不是步宽本身（P9a：循环读 32 宽的步也能到 665 GB/s）。
  ASIC_680ks、filter3D、wave、amazon0601、NACA0015 上是 1.15–1.95x，auto、roadNet-TX、cfd2 上是 1.00–1.06x。
- **固定 c=32、K=4 就能拿到 oracle（每个矩阵各选最好配置）的绝大部分**，SELL 实数四项在任何矩阵上都不会变慢：

  | 变体 | K=1 | K=2 | K=4 | oracle |
  |---|---:|---:|---:|---:|
  | SELL f32 | 0.696 | 0.781 | **0.841** | 0.848 |
  | SELL f16 | 0.700 | 0.797 | **0.825** | 0.861 |
  | SELL f16f32 | 0.696 | 0.789 | **0.817** | 0.848 |
  | SELL i8i32 | 0.702 | 0.794 | **0.799** | 0.867 |

- **COO 不能无条件转 SELL**：转 SELL（c=32、K=4）后，ASIC_680ks 0.82–0.86x、roadNet-TX 0.62–0.71x、NACA0015 0.91–0.95x、filter3D 0.84–0.93x，
  比现有路径（短行长尾走 NNZ 路由，其余走 CSR row-tile）慢。按行长统计分不开输家和赢家：
  filter3D（25.4 nnz/行）输、cfd2（25.0）赢，NACA（6.0）输、amazon（8.4）赢。
  所以 COO 在 prepare 时**实测**选择，和 cuSPARSE 的 analysis 阶段一样。不按矩阵名选。
- **调度下限**：BI-V 上每个 program 至少耗时 2.50 ns。roadNet-TX 在 c=32 时有 43.5K 个 slice，下限 109 µs，实测 112–114 µs，和 dtype 无关。
  这是阶段 2 的问题，本轮不处理。

## 2. 阶段 1a：K-wide kernel 进 `PreparedSellSpmv`

文件：`src/flagsparse/sparse_operations/spmv_sell.py`。

1. **新增 kernel `_spmv_sell_sigma_wide_kernel`**：从 `tools/biv_spmv_probe.py` 的 `_probe_sell_wide_kernel` 移植实数分支。这个 kernel 在 BI-V 上 0 FAIL。
   - 参数顺序和指针与 `_spmv_sell_sigma_kernel` 的启动点一致（`spmv_sell_prepared` 里的 `tensors` 元组，大约在第 1145 行）。
     尤其行号要用和现有启动相同的 perm 数组（RCM 时是 `kernel_perm` / `y_buf` 那一套），不要照搬探针里的 `perm`。
   - 只支持 `GROUP == 1`、实数。复数不动：Iluvatar 上复数的 `route="auto"` 本来就回到原生路径。
   - 不要改现有的 `_spmv_sell_sigma_kernel`，CUDA 和其他后端继续用它。
2. **`prepare_spmv_sell(..., k=None)`**，加环境变量 `FLAGSPARSE_SELL_PREP_K`，走现有的 `_sell_prep_option`。
   - `k` 必须是 [1, 16] 内的 2 的幂。`k > 1` 时如果 `group > 1` 或数据是复数，抛 `ValueError`（明确报错，不要静默降级）。
   - `PreparedSellSpmv.__slots__` 和 prepare 里那串 `setattr(p, name, None)` 都要加上 `k`。
   - `k == 1` 时仍启动原 kernel，行为和现在完全一致。
3. **默认值**：只有 Iluvatar 的实数数据默认 `k=4`，其他一律 `k=1`。
   - 用模块级常量来判断，例如 `_SELL_PREP_K_DEFAULT = 4 if _is_iluvatar_runtime() else 1`。
     **不要**复用 `_SELL_PREP_WARP64`：它还包含 MACA 和 ROCm，这两个后端没测过。
   - 也不要每次调用都执行 `_backend_name()`：以前这样做让 spmv_csc 慢了 4.6%。
4. 文档字符串里补一句 `k` 的含义：每一步读 K 个 slice 列，读的 `[K, C]` 块在内存里连续。

## 3. 阶段 1b：COO f16 / i8i32 / f16f32 prepare 时转 SELL

现在的路径是 `prepare_spmv_coo` → `flagsparse_spmv_coo(prepared=...)`。第二步在调用时判断 `spmv_needs_mixed`，然后进入
`mixed_spmx.spmv_coo_mixed`，在 Iluvatar 分支里经 `_ROCM_COO_CSR_CACHE` 走 CSR row-tile 或 NNZ kernel。

1. **在 prepare 里构建**：在 `prepare_spmv_coo` 里，当后端是 Iluvatar、`op == non`、数据 dtype 是 float16 或 int8 时：
   - 由已排序的 COO 得到 CSR（`ptr`）；
   - 调 `fs.csr_to_sell(..., slice_size=32)`，再调 `prepare_spmv_sell(..., route="sigma", c=32, group=1, num_warps=1, k=K)`；
   - 计划存在 `PreparedCoo` 新增的 slot 上，例如 `sell_plans`，按 out dtype 区分。f16 数据可能输出 f16 或 f32，int8 可能输出 int32。
   - 布局可以共用、按 out dtype 分别缓存常量，也可以每个 out dtype 一份计划；任选一种，在 PR 说明里写清楚。
   - **所有布局构建和第 2 点的实测都放在 `prepare_spmv_coo` 里，计时的调用里不能有。** 不要再新增 `TensorPlanCache` 的用法。
2. **prepare 时实测选择**：候选是"现有路径"、"SELL K=2"、"SELL K=4"。
   - 每个候选在 prepare 里先预热 2 次，再用 device event 计时 5 次，取中位数。
   - 只有 SELL 比现有路径快 5% 以上才选 SELL；两个 K 相差不到 3% 时选 K 较小的那个。这两个裕量是为了防止重复运行时选择来回跳。
   - x 用和真实 x 同 dtype 的全 1 向量；y 用临时 buffer。
   - 选择结果存在 prepared 对象上，例如 `p.sell_choice = "existing" | "sell_k2" | "sell_k4"`，再存三个候选的时间，方便日志和复查。
3. **环境变量 `FLAGSPARSE_ILUVATAR_COO_SELL=auto|0|1`**：
   - `auto` 是实测选择，为默认值；
   - `0` 永远走现有路径（A/B 的对照组）；
   - `1` 永远走 SELL，K 取 `FLAGSPARSE_SELL_PREP_K`，没设时为 4。
   - 取其他值时，像 `FLAGSPARSE_ILUVATAR_COO_C32_NNZ` 那样抛 `ValueError`。
4. **调用路径**：`flagsparse_spmv_coo(prepared=...)` 在 mixed 分支里，如果 `prepared.sell_plans` 里有对应 out dtype 的计划，
   就直接调 `spmv_sell_prepared(plan, x, out=out)`；否则走原路径。非 prepared 的直接调用保持不变。
5. **数据语义**：和 `PreparedCoo` 一致，prepare 时取一份快照。用户之后修改原始 COO 数据，结果是否跟着变，要和现在的 `PreparedCoo` 行为相同。
   这一点写进测试。

## 4. 阶段 2：只测、不进生产代码

1. **P10 探针**：在 `tools/biv_spmv_probe.py` 里加一组 GROUP × K 的 kernel，即一个 program 处理 GROUP 个 slice、每个 slice 每步读 K 列。
   对比 c=32 / c=64，测试矩阵用 roadNet-TX、ASIC_680ks、NACA0015、amazon0601，dtype 用 f32、f16、int8。
   目标是确认在 roadNet 上能不能突破每个 program 2.5 ns 的下限，并且在其他矩阵上不变慢。
2. 只带回数据和结论。是否把它加进第 3 节的实测候选，等数据回来后在 CUDA 机这边决定。

## 5. 验证

所有命令都带 `FLAGSPARSE_BACKEND=iluvatar`。不要用 `pytest --forked`：GPU 上下文在 fork 之后失效，会让所有 GPU 测试失败。按文件逐个跑。

1. **正确性，新增测试**（放在 `tests/pytest/` 下）：
   - `prepare_spmv_sell` 在 `k ∈ {1, 2, 4, 8, 16}` 下，覆盖 f32、f16、f16→f32、int8→int32。
     矩阵要包含：空行、行数不是 c 的倍数、slice 宽度不是 K 的倍数、只有一个 slice、nnz=0。int8 要求完全相等。
   - COO 在 `FLAGSPARSE_ILUVATAR_COO_SELL=0`、`1`、`auto` 三种设置下，结果都和参考一致。
   - `k > 1` 配合 `group > 1`，以及 `k > 1` 配合复数，都应抛 `ValueError`。
2. **已有测试全部通过**：
   `tests/pytest/test_spmv_sell_prepared.py`、`test_spmv_sell_out_accuracy.py`、`test_spmv_coo_accuracy.py`、
   `test_q4_variants_accuracy.py -k "spmv_sell or spmv_coo"`、`tests/ci/test_spmv_coo_iluvatar_route.py`。
   如果路由断言因为新路径需要修改，要在说明里写清楚改了什么、为什么改。
3. **`make pre-commit-check`** 全部通过（isort + black + flake8）。
4. **性能 A/B**：7 个变体 × 10 个矩阵，计时方式和 `tests/q4_variant_bench.py` 一样（prepare 不计时，`out` 预先分配）。
   - 旧路径是 `FLAGSPARSE_SELL_PREP_K=1 FLAGSPARSE_ILUVATAR_COO_SELL=0`，新路径是默认设置。
   - 新旧在同一进程里交错测，至少 5 轮，每个配置取中位数。
   - 输出 CSV 要包含 `variant,matrix,ours_ms,status` 和 `choice` 这几列，这样 `tools/biv_p9_project.py --current` 可以直接读。
   - 再记下每个矩阵的 prepare 耗时，新旧各一份。
   - 整个 A/B **跑两遍**，对比两遍 COO 的 `choice` 是否一致。
5. **全量**：用 65 变体的交付 runner 输出到一个**新目录**，例如 `results_biv_65_20261027_kwide`。
   不要往已有的结果目录里补跑子集：summary 会被重新投影到全部 65 个变体上，补跑子集会弄乱结果。
   然后逐个对比 65 个变体相对 `results_biv_65_20261024_final` 的变化。除了这 7 个变体，其他变体平均下降超过 3% 的都要列出来。

## 6. 停下来报告的条件

- 任何正确性 FAIL：先修，修不好就撤掉对应那部分，不要带着 FAIL 交付。
- SELL 实数四项在 A/B 里相对 K=1 的平均提速不到 1.08x（P9 预测 1.15–1.2x）：报告数据，不要自己调参。
- 两遍 A/B 里 COO 的 `choice` 有 2 个以上矩阵不一致：报告，不要擅自改裕量。
- 改动影响了 Iluvatar 以外的默认行为：说明哪里有问题，并撤回。

## 7. 规则和交付

- 基于做 P9 时的代码（记下 `git rev-parse HEAD`），改动尽量小。不要回退、格式化或"顺手清理"无关代码。
  以前的外部改动包回退过上游的修复，这次 diff 里只能出现本计划涉及的内容。
- 不要 commit、不要 push。按 `flagsparse_changes_and_results_20261026` 的格式打包，内容包括：
  `MANIFEST.txt`（基线 commit、改动文件列表、生成时间）、`patches/`（`git diff` 和新增文件）、`modified_files/`、`results/`。
- 在 `docs/debug/ILUVATAR_SELL_COO_PLAN.md` 末尾新增一节，记录 A/B 表格（逐矩阵数据和平均值）、COO 每个矩阵的 `choice`、prepare 耗时、全量对比和 P10 结论。
