# 海光 DCU（BW1000，gfx936）debug

`FLAGSPARSE_BACKEND=rocm`（通常能自动识别：`torch.version.hip` 不为空）。环境、交付复现见 [../DCU.md](../DCU.md)。

当前 debug 入口使用 Python runner：
`python3 tools/run_backend_tests.py --backend rocm --phase both --mode normal`。
该后端没有可运行的 C API adaptor，不要用 `capi/` 的 CTest 作为 DCU 结论。

## 1. 这个后端的特点

- Triton kernel 与 CUDA 相同；差别主要在**对比基线**：用 hipSPARSE（通过 `hip-python`），每个算子由
  `_<op>_sparse_ref_backend()` 选 `hipsparse` / `cupy_cusparse` / `None`，`None` 时在 `*_reason` 列写原因。
- hipSPARSE 的 SpMM 只支持不转置：CSR SpMM 的 `trans` / `conj` 没有基线（报跳过原因，不是失败）。
- gfx936 上 SpMV CSR 走 ROCm 专用路由（`row_tile`，fp32 本地累加，`max_row_nnz <= 8192` 时才用 fp32 累加）。
- warp 大小 64。

## 2. 已知问题

| 问题 | 状态 |
|---|---|
| `import flagsparse` 跑到系统里旧的已安装包，基线列无故变 N/A | 2026-09-28 已检查：DCU 机未安装 `flagsparse` wheel；加 `PYTHONPATH=src` 后导入当前仓库。以后换环境仍需先查（见 [README](README.md) 第 2 节） |
| runner 用 `CUDA_VISIBLE_DEVICES` 隔离设备，ROCm 上会打乱 torch 的随机种子 | 在 DCU 上额外 `export HIP_VISIBLE_DEVICES=<卡号>` |
| Slurm 注入的 `TMPDIR` / `TMP` / `TEMP` 指向已删除的 scratch 目录，所有未缓存 Triton kernel 都报 `clang-18: unable to make temporary file` | 跑测试前用 `mktemp -d /tmp/flagsparse-dcu.XXXXXX` 创建目录，并把三个变量都指向它 |
| spmv_csr 性能约 0.27×（f32 0.276 / f64 0.260，对比 hipSPARSE） | 未解决，和精度修复无关；先查 `row_tile` 路由的启动配置 |
| DCU 侧的代码包是按旧版本改的，整文件覆盖会回退新代码（曾把 iluvatar 改回 mlu） | 合并时按段取，不要整文件复制 |
| DCU 那边 `spmm_csr.py` 里的两个缺陷（`row_ids` 多余分配、短行路由忽略 `accuracy`） | 本仓库已修；DCU 那边每次发来的包都还带着，不要合回来 |
| 2026-09-27：ROCm 上 int64→int32 索引压缩导致精度测试失败 | 已删除；代价是 int64 输入在 ROCm 上用不了 opt 路径 |

## 3. q4 42 个变体的风险点

- 与 CUDA 同一套 Triton 代码，CUDA 上全过。重点确认：
  - **int8**（`spmv_*_i8i32`、`spmv_csr_i8f32`、`spmm_csr_i8i32`、`spvv_i8i32`、`gather/scatter_i8`）：int8 读、int32 累加，
    以及 int32 的 `tl.atomic_add`（`spmv_coo_i8i32` 用到）；
  - **float16**：新代码不做 f16 原子加，但读 f16、在 f32 里算；
  - **SELL SpMV**（`spmv_sell_*`）：全新 kernel，按切片起程序，切片内行数 = `slice_size`（默认 32）。
- **基线**：q4 新接的 cuSPARSE 基线（`tests/cusparse_generic_baseline.py`）只在 CUDA 可用；
  DCU 上这些行的 cuSPARSE 列为空并写明原因，runner 会改用 PyTorch 加速比，再用 `tools/baseline_bound.py --vendor-card dcu-bw1000` 判定。
  如果需要 hipSPARSE 基线，可以按 `cusparse_generic_baseline.py` 的结构用 `hip-python` 补（hipSPARSE 的 SpVV / Axpby / SELL SpMV 接口名一一对应）。

## 4. 2026-09-28 BW1000 实测结果

测试提交为 `8492853`（工作区只新增结果目录，测试过程中没有修改源码或测试文件）。

### 4.1 环境确认

| 项目 | 实测值 |
|---|---|
| 设备 | `BW200, UBB BW1000`，1 张卡 |
| PyTorch | `2.4.1` |
| HIP | `6.1.25065` |
| Triton | `3.6.0` |
| 后端识别 | `backend=rocm`，`device_type=cuda`，`is_rocm=True` |
| hipSPARSE | 可用，`_hipsparse_unavailable_reason() is None` |
| FlagSparse 导入路径 | `<repo>/src/flagsparse/__init__.py` |
| 已安装 wheel | `python3 -m pip show flagsparse` 返回 `Package(s) not found` |

有效运行前需修正临时目录，并同时设置 HIP/CUDA 卡号：

```bash
task_tmp_dir=$(mktemp -d /tmp/flagsparse-q4-dcu.XXXXXX)
export TMPDIR="$task_tmp_dir" TMP="$task_tmp_dir" TEMP="$task_tmp_dir"
export PYTHONPATH="$PWD/src"
export FLAGSPARSE_BACKEND=rocm
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
```

未修正时的首轮结果是 `10 passed, 75 failed`，75 个失败的共同原因都是 DTK `clang-18`
无法在不存在的 Slurm scratch 目录创建临时文件，并非算子错误。修正后重新编译并运行，结果如下。

### 4.2 42 个变体精度和性能

精度命令：

```bash
CUDA_LAUNCH_BLOCKING=1 python3 -m pytest \
  tests/pytest/test_q4_variants_accuracy.py -v --tb=short
```

结果为 `85 passed, 1 warning in 46.35s`：1 个清单检查，加上 42 个变体各 2 个规模，
即 **84/84 个计算用例通过**。表中的加速比是 runner 对有效 case 的算术平均，定义为
`baseline_time / FlagSparse_time`，大于 1 表示 FlagSparse 更快；优先使用可用的厂商基线，
没有 q4 厂商基线时使用 PyTorch。

| # | 变体 | 精度 | 平均加速比 |
|---:|---|---:|---:|
| 1 | `gather_i8_int` | 2/2 PASS | **2.752x** |
| 2 | `scatter_i8_int` | 2/2 PASS | **1.849x** |
| 3 | `axpby_f16_int` | 2/2 PASS | 0.502x |
| 4 | `spmv_sell_f32_int_non` | 2/2 PASS | 0.350x |
| 5 | `spmv_csr_f16f32_int_non` | 2/2 PASS | 0.985x |
| 6 | `spvv_f16f32_int_non` | 2/2 PASS | 0.610x |
| 7 | `spmv_csr_f16_int_non` | 2/2 PASS | 0.866x |
| 8 | `spmv_csr_f32c32_int_non` | 2/2 PASS | 0.938x |
| 9 | `spmv_sell_f16_int_non` | 2/2 PASS | 0.359x |
| 10 | `spmm_csr_f16f32_int_non_non_row` | 2/2 PASS | **2.180x** |
| 11 | `spmm_csr_f16_int_non_non_row` | 2/2 PASS | **2.059x** |
| 12 | `spmm_csr_f32_int_non_non_col` | 2/2 PASS | **4.878x** |
| 13 | `spmm_csr_f32_int_non_trans_row` | 2/2 PASS | **4.765x** |
| 14 | `spmv_csc_f32_int_non` | 2/2 PASS | 0.642x |
| 15 | `spmv_csr_c32_int_non` | 2/2 PASS | 0.797x |
| 16 | `spmv_sell_c32_int_non` | 2/2 PASS | 0.220x |
| 17 | `spvv_c32_int_conj` | 2/2 PASS | 0.370x |
| 18 | `spmv_coo_f16f32_int_non` | 2/2 PASS | 0.358x |
| 19 | `spmm_csr_c32_int_non_non_row` | 2/2 PASS | **2.112x** |
| 20 | `spmv_coo_f32_int_trans` | 2/2 PASS | 0.151x |
| 21 | `spmv_csr_f32_int_trans` | 2/2 PASS | 0.035x |
| 22 | `spmv_csr_i8f32_int_non` | 2/2 PASS | **1.008x** |
| 23 | `spmv_csr_i8i32_int_non` | 2/2 PASS | **1.021x** |
| 24 | `spmv_sell_i8i32_int_non` | 2/2 PASS | 0.374x |
| 25 | `spvv_i8i32_int_non` | 2/2 PASS | 0.679x |
| 26 | `spmm_csc_f32_int_non_non_row` | 2/2 PASS | **4.748x** [1] |
| 27 | `spmm_csr_i8i32_int_non_non_row` | 2/2 PASS | **2.263x** |
| 28 | `spmv_coo_c32_int_non` | 2/2 PASS | 0.124x |
| 29 | `spmv_coo_f16_int_non` | 2/2 PASS | 0.341x |
| 30 | `spmv_csc_c32_int_non` | 2/2 PASS | 0.380x |
| 31 | `spmv_csc_f16_int_non` | 2/2 PASS | 0.202x |
| 32 | `sddmm_csr_f32_int_non_non_col` | 2/2 PASS | **2.850x** |
| 33 | `sddmm_csr_f32_int_non_trans_row` | 2/2 PASS | **6.920x** |
| 34 | `sddmm_csr_f32_int_trans_non_row` | 2/2 PASS | **1.348x** |
| 35 | `spmm_bell_f32_int_non_non_row` | 2/2 PASS | 0.528x [2] |
| 36 | `spmm_bsr_f32_int_non_non_row` | 2/2 PASS | 0.249x |
| 37 | `spmm_coo_c32_int_non_non_row` | 2/2 PASS | **1.985x** |
| 38 | `spmm_coo_f16_int_non_non_row` | 2/2 PASS | **3.951x** |
| 39 | `spmm_csr_f32_int_trans_non_row` | 2/2 PASS | **1.005x** |
| 40 | `spmv_coo_c32_int_conj` | 2/2 PASS | 0.156x |
| 41 | `spmv_coo_i8i32_int_non` | 2/2 PASS | 0.680x |
| 42 | `spmv_csr_c32_int_conj` | 2/2 PASS | 0.040x |

42 个变体中有 17 个平均加速比大于 1。需要注意算术平均会隐藏矩阵间的波动；做性能回归时应查看
各算子目录的 `performance.csv`，不能只看上表均值。

[1] `spmm_csc_f32_int_non_non_row` 自身的 10 个 f32/int32/non 性能 case 全部通过；但父算子的
完整 sweep 另有 4 个非 q4 complex case 精度失败，因此 `summary.json` 中该变体继承了父阶段的
`performance=Failed`。失败组合见 4.4 节。

[2] BELL 的 0.528x 是 9 个有效矩阵的平均值。`ASIC_680ks.mtx` 转成 Blocked-ELL 后需要存储
398,703,808 个值，保护检查抛出 `MemoryError`，该矩阵没有计入加速比。

### 4.3 runner 命令和结果目录

前 11 类算子使用文档给出的完整 runner 命令：

```bash
python3 run_flagsparse_pytest.py \
  --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,\
spmm_csr,spmm_coo,spmm_csc,spmm_bsr,spmm_bell,sddmm_csr \
  --gpus 0 --results-dir results_rocm_2026-09-28_q4 \
  --benchmark-input tests/data
```

长时间运行在 `spmm_bsr` 阶段被外部中断。为避免覆盖已有 `summary.json`，BSR/BELL/SDDMM 在新目录
续跑，并把 BSR/BELL 限定到 q4 变体实际需要的 `float32,int32,non` 轴：

```bash
python3 run_flagsparse_pytest.py \
  --ops spmm_bsr,spmm_bell,sddmm_csr --gpus 0 \
  --results-dir results_rocm_2026-09-28_q4_tail_scoped \
  --benchmark-input tests/data \
  --op-benchmark-args 'spmm_bsr=--dtypes float32 --index-dtypes int32 --ops non' \
  --op-benchmark-args 'spmm_bell=--dtypes float32 --index-dtypes int32 --ops non'
```

结果目录：

- `results_rocm_2026-09-28_q4/`：gather 到 spmm_csc 的完整日志、JSON 和 CSV；
- `results_rocm_2026-09-28_q4_tail_scoped/`：spmm_bsr、spmm_bell、sddmm_csr 的 q4 轴结果；
- `spmm_bsr`、`spmm_bell`、`sddmm_csr` 的准确性阶段均通过；BSR 和 SDDMM 性能阶段通过；
  BELL 常规性能行通过，但 q4 附加行有上述 `ASIC_680ks.mtx` 内存保护错误。

### 4.4 runner 发现的非 q4 问题

这些问题不影响“42 个变体精度全过”的结论，但应在后续 DCU 调试中单独处理。

1. `spmm_csc` 完整性能 sweep 的 240 行里有 4 行 correctness check 失败：

   | 矩阵 | dtype | index | op | `err_vs_ref` |
   |---|---|---|---|---:|
   | `ASIC_680ks.mtx` | complex64 | int32 | conj | 398315.519 |
   | `roadNet-TX.mtx` | complex64 | int32 | non | 4414.752 |
   | `auto.mtx` | complex128 | int64 | conj | 2398469.749 |
   | `cfd2.mtx` | complex128 | int64 | trans | 1312922.383 |

   q4 的 `spmm_csc_f32_int_non_non_row` 不在这些失败组合中，其 10 行性能数据均为 PASS。

2. BELL 的 hipSPARSE 基线不可用，原因是当前 `hip-python` wrapper 的
   `hipsparseCreateBlockedEll` 只接受 9 个位置参数，而调用传了 10 个。常规 BELL 行因此记录
   `vendor_ms=N/A`；q4 行使用 PyTorch 回退计算加速比。

3. `tools/probe_accel_capabilities.py` 已确认分配、H2D/D2H、逐元素运算和归约在
   f32/f64/c64/c128 上正常；后续某个隔离探针超过 90 秒没有返回，手工中止。正式的 84 个 q4
   计算用例和 runner 不受影响。
