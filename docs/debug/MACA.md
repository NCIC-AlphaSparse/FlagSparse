# 沐曦 MACA（C550）debug

2026-10-06：CSR 默认转置 prepared gather、复数 SELL 调度及 SpGEMM matrix-worker 参数修复已实现，
MACA 真机复测待完成。历史性能数字未替换，见 [CUDA 手册](CUDA.md)。

`FLAGSPARSE_BACKEND=metax`（设置了 `MACA_PATH` 时能自动识别）。环境、交付复现见 [../MACA.md](../MACA.md)。

当前 debug 入口使用 Python runner：
`python3 tools/run_backend_tests.py --backend maca --phase both --mode normal`。
MACA 的 C API adaptor 尚未可构建，调试时不要用 CTest 作为后端结论。

## 1. 这个后端的特点

- 以 CUDA 兼容栈的方式接入（`torch.version.cuda` 显示 11.6），Triton kernel 与 CUDA 相同。
- 实测指纹：设备名 `MetaX C550`，**warp 大小 64**，104 个 MP，每 MP 2048 线程。
- **单线程私有内存上限 4 KB**：编译期展开的循环（`tl.static_range`）越长越占私有内存，超了会报
  `memory size or pointer value too large to fit in 32 bit`，真正原因在它上一行的 MACA 运行时提示里。
- 要用沐曦自己的 triton（`3.6.0+metax3.8.1.0`），FlagOS 的 flagtree 在这台 glibc 2.31 的机器上加载不了；
  代价是没有 `triton.experimental.tle`，`alpha_spmm_alg1` 不可用。
- SDK 自带 **mcSPARSE**，目前没接成基线（可以接）。

## 2. 已知问题

| 问题 | 状态 |
|---|---|
| SpSV `_spsv_csr_cw_kernel`：下三角即使没有依赖也非法访存；上三角有依赖时卡死 | 未解决；只影响单位对角走 cw 路由的情况 |
| 复数 SpMM COO 因 `BLOCK_NNZ=256` 展开超出 4 KB 私有内存 | 已修：MACA + 复数时 `BLOCK_NNZ` 限为 4 |
| spgemm_csr 在某个容器上 5 次里失败 3 次，换容器后 800+ 次全过 | 视为容器 / 硬件状态问题，复现时记录容器 id |
| 卡上的代码（`/root/gcx/FlagSparse`）和本地已经不同步 | 看报错前先看机器上那份源码 |
| q4 `gather_i8_int` 没有性能行 | runner 的 `DELIVERY_BENCHMARK_ARGS["gather"]` 限定的 `--value-dtypes` 漏了 `int8`；`tests/test_gather.py` 本身支持 int8，且 q4 精度已通过。q4 对 gather 复用普通（不带 `variant`）性能行，因此汇总为 `NOT_CONFIGURED`。应把 `int8` 加入该 runner 配置后重跑。 |
| q4 `spmm_bell_f32_int_non_non_row` 在 `ASIC_680ks` 不能测 | q4 路径以 `block_dim=2` 转 Blocked-ELL 时，`398,703,808` 个 float32 values 超过专用的 1 GiB values 上限，转换在 FlagSparse、PyTorch 和厂商 baseline 之前报 `MemoryError`。其余 9 个矩阵通过。见第 4.2 节。 |

## 3. q4 变体的风险点

> ⚠️ 2026-10-05：q4 清单已从 42 条改成 45 条（删 3 加 6，见 [README.md](README.md) 第 0 节）。
> 下面这节和第 4 节 2026-09-28 的实测表格都是**旧 42 条清单**的内容，表格里的
> `gather_i8_int`、`spmm_bell_f32_int_non_non_row`、`spmm_bsr_f32_int_non_non_row`
> 现在已经移出 q4 统计范围（继续保留实现，MACA 上测过且 PASS）。新增的 6 个变体
> （`sddmm_csr_f16/c32_int_non_non_row`、`spmm_csc_c32/f16_int_non_non_row`、
> `spmm_coo_i8i32_int_non_non_row`、`spgemm_csr_f32_int_non_non`）已于 2026-10-05 在
> MACA C550 上完成 Python 精度验证（均为 2/2 PASS）；本节旧表仍不含它们的性能结果。

- 新代码的循环都是运行时 `range`，没有依赖私有内存的长展开；复数路径最该验证：
  `spmv_csr/coo/csc_c32`、`spmm_csr/coo_c32`、`spvv_c32_int_conj`、`spmv_sell_c32`、`spmv_csr_f32c32`。
- `spmv_csr` 的混合精度 kernel 每个程序处理 `ROWS x BLOCK` 个元素（共 512 个），按 4 个 warp 调；warp 大小是 64 时每个 warp 分到的元素数不同，需要实测性能。
- 当前 Python runner 没有 mcSPARSE binding，因而本节性能均以 PyTorch 为基线；SDK 实际已有
  `libmcsparse.so` 及 `<mcsparse/mcsparse.h>`，但尚未接入。接入后可直接比厂商库；当前可先用
  `--vendor-card maca-c550`（1440 GB/s）作带宽上限判定。

### 3.1 当前 45 条 q4 精度结果（2026-10-05）

在 MetaX C550 的 MACA Python 3.12 环境，显式设置
`FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_MODEL=c550 FLAGSPARSE_MACA_VENDOR=torch` 后执行：

```bash
PYTHONPATH=src python3 -m pytest tests/pytest/test_q4_variants_accuracy.py -v
```

结果为 **91 passed, 2 warnings in 23.15s**：45 个当前 q4 变体在两个 shape 上均通过（90 个
数值用例），另有 1 个“清单恰有 45 条且名称不重复”的断言通过。新增的
`sddmm_csr_f16/c32`、`spmm_csc_c32/f16`、`spmm_coo_i8i32` 和 `spgemm_csr_f32` 6 条均已覆盖。
warning 是缺少 `flash_attn` 与 PyTorch CSR beta 提示，不影响精度判定。性能结果尚不写入本节的
第 4 节旧 42 条历史表；当前 45 条性能结果见下一节。

### 3.2 当前 45 条 q4 性能结果（2026-10-05）

使用 `tests/data` 的 10 个 MatrixMarket 矩阵（向量算子为 4 个合成规模）、5 次预热、20 次计时：

```bash
PYTHONPATH=src FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_MODEL=c550 \
FLAGSPARSE_MACA_VENDOR=torch python3 -u tools/run_backend_tests.py \
  --backend maca --phase benchmark --mode normal --gpus 0 \
  --ops scatter,axpby,spmv_sell,spmv_csr,spvv,spmm_csr,spmv_csc,spmv_coo,spmm_csc,sddmm_csr,spmm_coo,spgemm_csr \
  --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 \
  --op-benchmark-args='sddmm_csr=--no-cusparse' --timeout 4500 \
  --results-dir results_metax_q4_45_perf_20261005
```

12 个父算子串行耗时合计 **1h50m45s**；最慢的是 `sddmm_csr`（26m51s）、`spmm_csc`
（24m40s）和 `spmm_csr`（22m02s）。下表的加速比是逐矩阵的 FlagSparse / PyTorch 比值的算术均值，
不是 mcSPARSE 比值；SDDMM 的 PyTorch 基线是未融合的 gather/multiply/sum，不能据此推断厂商 SDDMM
加速比。

| 变体 | 精度 | 性能 | PyTorch 加速比 |
|---|---:|---:|---:|
| `scatter_i8_int` | 2/2 PASS | PASS | 2.186x |
| `axpby_f16_int` | 2/2 PASS | PASS | 1.198x |
| `spmv_sell_f32_int_non` | 2/2 PASS | PASS | 1.270x |
| `spmv_csr_f16f32_int_non` | 2/2 PASS | PASS | 4.002x |
| `spvv_f16f32_int_non` | 2/2 PASS | PASS | 1.257x |
| `spmv_csr_f16_int_non` | 2/2 PASS | PASS | 1.532x |
| `spmv_csr_f32c32_int_non` | 2/2 PASS | PASS | 2.842x |
| `spmv_sell_f16_int_non` | 2/2 PASS | PASS | 2.104x |
| `spmm_csr_f16f32_int_non_non_row` | 2/2 PASS | PASS | 59.914x |
| `spmm_csr_f16_int_non_non_row` | 2/2 PASS | PASS | 24.860x |
| `spmm_csr_f32_int_non_non_col` | 2/2 PASS | PASS | 18.155x |
| `spmm_csr_f32_int_non_trans_row` | 2/2 PASS | PASS | 18.007x |
| `spmv_csc_f32_int_non` | 2/2 PASS | PASS | 3.536x |
| `spmv_csr_c32_int_non` | 2/2 PASS | PASS | 1.605x |
| `spmv_sell_c32_int_non` | 2/2 PASS | PASS | 0.592x |
| `spvv_c32_int_conj` | 2/2 PASS | PASS | 0.808x |
| `spmv_coo_f16f32_int_non` | 2/2 PASS | PASS | 3.064x |
| `spmm_csr_c32_int_non_non_row` | 2/2 PASS | PASS | 14.268x |
| `spmv_coo_f32_int_trans` | 2/2 PASS | PASS | 0.691x |
| `spmv_csr_f32_int_trans` | 2/2 PASS | PASS | 0.089x |
| `spmv_csr_i8f32_int_non` | 2/2 PASS | PASS | 4.061x |
| `spmv_csr_i8i32_int_non` | 2/2 PASS | PASS | 3.811x |
| `spmv_sell_i8i32_int_non` | 2/2 PASS | PASS | 1.316x |
| `spvv_i8i32_int_non` | 2/2 PASS | PASS | 1.849x |
| `spmm_csc_f32_int_non_non_row` | 2/2 PASS | PASS | 81.381x |
| `spmm_csr_i8i32_int_non_non_row` | 2/2 PASS | PASS | 58.599x |
| `spmv_coo_c32_int_non` | 2/2 PASS | PASS | 0.628x |
| `spmv_coo_f16_int_non` | 2/2 PASS | PASS | 2.621x |
| `spmv_csc_c32_int_non` | 2/2 PASS | PASS | 1.299x |
| `spmv_csc_f16_int_non` | 2/2 PASS | PASS | 0.502x |
| `sddmm_csr_f32_int_non_non_col` | 2/2 PASS | PASS | 597.426x |
| `sddmm_csr_f32_int_non_trans_row` | 2/2 PASS | PASS | 843.804x |
| `sddmm_csr_f32_int_trans_non_row` | 2/2 PASS | PASS | 190.029x |
| `spmm_coo_c32_int_non_non_row` | 2/2 PASS | PASS | 25.837x |
| `spmm_coo_f16_int_non_non_row` | 2/2 PASS | PASS | 99.646x |
| `spmm_csr_f32_int_trans_non_row` | 2/2 PASS | PASS | 13.336x |
| `spmv_coo_c32_int_conj` | 2/2 PASS | PASS | 0.574x |
| `spmv_coo_i8i32_int_non` | 2/2 PASS | PASS | 3.242x |
| `spmv_csr_c32_int_conj` | 2/2 PASS | PASS | 0.075x |
| `spmm_coo_i8i32_int_non_non_row` | 2/2 PASS | PASS | 20.605x |
| `spmm_csc_c32_int_non_non_row` | 2/2 PASS | PASS | 20.232x |
| `sddmm_csr_c32_int_non_non_row` | 2/2 PASS | PASS | 79.960x |
| `sddmm_csr_f16_int_non_non_row` | 2/2 PASS | PASS | N/A |
| `spmm_csc_f16_int_non_non_row` | 2/2 PASS | PASS | N/A |
| `spgemm_csr_f32_int_non_non` | 2/2 PASS | **ERROR** | N/A |

**原始结果复查（不要只看 runner 的 `summary.csv`）：**42 条有有效 PyTorch 加速比；
`sddmm_csr_f16_int_non_non_row` 因 PyTorch `sampled_addmm` 不支持 `Half` 而无基线；
`spmm_csc_f16_int_non_non_row` 的 FlagSparse 实现正确但会 upcast 到 fp32，MACA PyTorch CSC fp16
同样不支持，故无可比基线。`spgemm_csr_f32_int_non_non` 的 10 个原始性能行均为 ERROR：
`test_spgemm.py` 不识别它给 worker 传入的 `--_matrix-worker` 参数。当前 runner 因父进程退出码为 0，
误将该 q4 汇总行标成 Passed；该加速比必须视为**未测**，不能用于报告。

## 4. q4 实测结果（2026-09-28，旧 42 条清单，新增 6 个变体未测）

**环境**：MetaX C550，MACA SDK 3.8.2.6，torch `2.10.0+metax3.8.1.0`，
triton `3.6.0+metax3.8.1.0`，Python 3.12；显式设置
`FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_MODEL=c550 FLAGSPARSE_MACA_VENDOR=torch`。

**精度**：

```bash
PYTHONPATH=src FLAGSPARSE_BACKEND=metax \
  python3 -m pytest tests/pytest/test_q4_variants_accuracy.py -v
```

结果为 **85 passed, 2 warnings in 40.93s**：42 个变体各两个形状全部通过，另有一个清单完整性断言；
复数、混合精度和 int8 均通过。warning 是缺少 `flash_attn` 及 PyTorch CSR beta 提示，与 q4 无关。

**性能**：使用 `tests/data` 的 10 个 MatrixMarket 矩阵（向量算子使用 4 个合成规模）、5 次预热、20 次计时。
没有可用的 cuSPARSE/mcSPARSE baseline，表中的加速比均为 FlagSparse 相对 PyTorch 的算术平均。
SDDMM 的 PyTorch 侧是独立的 `sum(X[row] * Y[col])` 参考表达式，包含 gather、乘法和归约，不能当作厂商
SDDMM 加速比。`tools/baseline_bound.py --vendor-card maca-c550` 还需要同配置 H800 结果，本次没有该参考结果，
因此没有给出带宽上限判定。

结果由以下三次运行合并：

- `results_metax_q4_perf_scoped_20260928/`：向量、SpMV、CSR/CSC SpMM；
- `results_metax_q4_perf_background_20260928/`：BSR/BELL SpMM、SDDMM；
- `results_metax_q4_perf_coo_retry_20260928/`：修正 CLI 参数后的 COO SpMM。

性能 `PASS` 表示该行已和脚本的参考结果比对通过；精度列的 `2/2 PASS` 来自前述 pytest。BELL 的 9 个有效
矩阵才纳入平均加速比。

| # | 变体 | 精度 | 性能 | PyTorch 加速比 |
|---:|---|---|---|---:|
| 1 | `gather_i8_int` | 2/2 PASS | NOT_CONFIGURED | - |
| 2 | `scatter_i8_int` | 2/2 PASS | 4/4 PASS | 2.192x |
| 3 | `axpby_f16_int` | 2/2 PASS | 4/4 PASS | 0.976x |
| 4 | `spmv_sell_f32_int_non` | 2/2 PASS | 10/10 PASS | 1.389x |
| 5 | `spmv_csr_f16f32_int_non` | 2/2 PASS | 10/10 PASS | 3.989x |
| 6 | `spvv_f16f32_int_non` | 2/2 PASS | 4/4 PASS | 1.552x |
| 7 | `spmv_csr_f16_int_non` | 2/2 PASS | 10/10 PASS | 1.527x |
| 8 | `spmv_csr_f32c32_int_non` | 2/2 PASS | 10/10 PASS | 3.395x |
| 9 | `spmv_sell_f16_int_non` | 2/2 PASS | 10/10 PASS | 7.494x |
| 10 | `spmm_csr_f16f32_int_non_non_row` | 2/2 PASS | 10/10 PASS | 60.774x |
| 11 | `spmm_csr_f16_int_non_non_row` | 2/2 PASS | 10/10 PASS | 24.943x |
| 12 | `spmm_csr_f32_int_non_non_col` | 2/2 PASS | 10/10 PASS | 18.089x |
| 13 | `spmm_csr_f32_int_non_trans_row` | 2/2 PASS | 10/10 PASS | 17.912x |
| 14 | `spmv_csc_f32_int_non` | 2/2 PASS | 10/10 PASS | 3.518x |
| 15 | `spmv_csr_c32_int_non` | 2/2 PASS | 10/10 PASS | 1.562x |
| 16 | `spmv_sell_c32_int_non` | 2/2 PASS | 10/10 PASS | 0.783x |
| 17 | `spvv_c32_int_conj` | 2/2 PASS | 4/4 PASS | 0.556x |
| 18 | `spmv_coo_f16f32_int_non` | 2/2 PASS | 10/10 PASS | 3.042x |
| 19 | `spmm_csr_c32_int_non_non_row` | 2/2 PASS | 10/10 PASS | 14.277x |
| 20 | `spmv_coo_f32_int_trans` | 2/2 PASS | 10/10 PASS | 0.702x |
| 21 | `spmv_csr_f32_int_trans` | 2/2 PASS | 10/10 PASS | 0.085x |
| 22 | `spmv_csr_i8f32_int_non` | 2/2 PASS | 10/10 PASS | 3.992x |
| 23 | `spmv_csr_i8i32_int_non` | 2/2 PASS | 10/10 PASS | 4.022x |
| 24 | `spmv_sell_i8i32_int_non` | 2/2 PASS | 10/10 PASS | 1.088x |
| 25 | `spvv_i8i32_int_non` | 2/2 PASS | 4/4 PASS | 1.709x |
| 26 | `spmm_csc_f32_int_non_non_row` | 2/2 PASS | 10/10 PASS | 81.518x |
| 27 | `spmm_csr_i8i32_int_non_non_row` | 2/2 PASS | 10/10 PASS | 58.775x |
| 28 | `spmv_coo_c32_int_non` | 2/2 PASS | 10/10 PASS | 0.592x |
| 29 | `spmv_coo_f16_int_non` | 2/2 PASS | 10/10 PASS | 2.110x |
| 30 | `spmv_csc_c32_int_non` | 2/2 PASS | 10/10 PASS | 1.447x |
| 31 | `spmv_csc_f16_int_non` | 2/2 PASS | 10/10 PASS | 4.294x |
| 32 | `sddmm_csr_f32_int_non_non_col` | 2/2 PASS | 10/10 PASS | 599.078x |
| 33 | `sddmm_csr_f32_int_non_trans_row` | 2/2 PASS | 10/10 PASS | 847.620x |
| 34 | `sddmm_csr_f32_int_trans_non_row` | 2/2 PASS | 10/10 PASS | 194.917x |
| 35 | `spmm_bell_f32_int_non_non_row` | 2/2 PASS | 9/10 PASS, 1 ERROR | 7.984x |
| 36 | `spmm_bsr_f32_int_non_non_row` | 2/2 PASS | 10/10 PASS | 9.054x |
| 37 | `spmm_coo_c32_int_non_non_row` | 2/2 PASS | 10/10 PASS | 25.846x |
| 38 | `spmm_coo_f16_int_non_non_row` | 2/2 PASS | 10/10 PASS | 100.844x |
| 39 | `spmm_csr_f32_int_trans_non_row` | 2/2 PASS | 10/10 PASS | 13.446x |
| 40 | `spmv_coo_c32_int_conj` | 2/2 PASS | 10/10 PASS | 0.517x |
| 41 | `spmv_coo_i8i32_int_non` | 2/2 PASS | 10/10 PASS | 3.251x |
| 42 | `spmv_csr_c32_int_conj` | 2/2 PASS | 10/10 PASS | 0.074x |

性能上最需要后续调优的是 CSR transpose / complex-conj SpMV（分别为 0.085x 与 0.074x）；结果正确，
但均明显慢于 PyTorch。COO complex / conj SpMV、complex SELL SpMV 和 complex-conj SpVV 也低于 1x。

### 4.1 `gather_i8_int` 性能遗漏

这不是 int8 gather 的算子缺陷：q4 精度在两种形状均通过，`tests/test_gather.py` 的默认 dtype 列表包含
`int8`。遗漏发生在 runner 的交付收窄参数：它把 gather 固定为
`float16,float32,float64,complex64,complex128`，没有保留 int8。q4 汇总对这种不带 `variant` 标签的普通
性能行按 dtype 映射，因而找不到 `gather_i8_int` 的行。

修复 runner 配置后应单独重跑 gather：

```bash
PYTHONPATH=src FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_MODEL=c550 \
FLAGSPARSE_MACA_VENDOR=torch python3 run_flagsparse_pytest.py \
  --phase performance --delivery-only --gpus 0 --ops gather \
  --benchmark-warmup 5 --benchmark-iters 20 \
  --op-benchmark-args='gather=--value-dtypes int8' \
  --results-dir results_metax_q4_gather_i8
```

### 4.2 `spmm_bell` 的 `ASIC_680ks` 容量保护

q4 BELL 使用 `block_dim=2`。`ASIC_680ks`（`682712 x 682712`、2,329,176 nnz）每个 block row 的最大
block 数为 292，生成 398,703,808 个 float32 slots：仅 data 为约 1.485 GiB；99,675,952 个 int32 block
index 约 0.371 GiB，data + index 为约 1.857 GiB。带 32 列的 B 和输出 C 各约 83.3 MiB，主要常驻张量约
2.02 GiB。

q4 的 `_BELL_MAX_STORED = 1 << 28` 只按 values 数量限制（1 GiB float32 data），所以在分配前报：

```text
MemoryError: Blocked-ELL would store 398703808 values
```

该检查发生在 FlagSparse、PyTorch CSR baseline 和厂商 BELL baseline 的准备之前。当前 C550 没有接入
mcSPARSE/cuSPARSE BELL baseline；即使接入，同格式 baseline 也需要这份 BELL 数据。

通用 `tests/test_spmm_bell.py` 自 2026-08-27 起有独立的 2 GiB **data + index** 保护，并已实际通过
`ASIC_680ks` 的普通 float32 BELL 行。C550 有 64 GiB 显存，因此对该矩阵，2 GiB data + index 阈值足够；
q4 的限制应改为按 data + index 的实际字节数判断，不能只把 values 个数阈值翻倍。此结论尚未改源码。

## 5. 后续检查

1. 修复 `gather` 的 runner dtype 列表并补跑 `gather_i8_int` 性能。
2. 按 data + index 估算替换 q4 BELL 的 values-only 容量保护，补跑 `ASIC_680ks`；同时保留转换过程的 OOM
   保护和实际峰值显存记录。
3. 有同配置 H800 结果后运行 `tools/baseline_bound.py <H800结果目录> --vendor <本次结果目录> --vendor-card maca-c550`。
4. 如出现私有内存报错，记录 MACA 运行时的 kernel 请求 / 系统上限原文。
