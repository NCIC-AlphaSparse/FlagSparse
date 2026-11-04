# H800 参考结果（随仓库分发）

> 本文件由 `tools/h800_reference.py` 从 H800 的一轮 runner 结果生成，**不要手改**；数据在 [`conf/h800_reference.json`](../conf/h800_reference.json)。

## 来源

- 设备：NVIDIA H800 × 8
- 原始结果目录：`results_cuda_105_20261103_1556`（被 git 忽略，不随仓库）
- FlagSparse commit：`bc586355e027+dirty`；torch 2.8.0a0+5228986c39.nv25.05；triton 3.6.0
- 生成日期：2026-11-04

## 用途

**只用于没有任何基线的行**：目标卡上有厂商库（hipSPARSE、muSPARSE……）或 PyTorch 时间的行一律用实测基线，不会被折算覆盖。没有厂商 / PyTorch 基线的行，用这里的 **H800 上 cuSPARSE 的时间**做基线：

```
h800_scaled_ms = T_cuSPARSE@H800 × P_H800 / P_目标卡      (P = 实测显存带宽)
speedup        = h800_scaled_ms / T_FlagSparse@目标卡     (1.0 = 与折算后的 cuSPARSE 一样快，≥0.8 合格)
```

```bash
# 不用再找目录：不带值即用本仓库自带的文件
python3 run_flagsparse_pytest.py --h800-reference ...        # 卡型按后端/设备名自动识别
python3 run_flagsparse_split_delivery.py --h800-reference ...  # C API 拆分交付（MUSA）：补 NoBaseline 变体
python3 tools/baseline_bound.py --vendor <目标卡结果目录>    # 已跑完的结果目录事后查看
```

这是**估算**，不是厂商实测：H800 与目标卡的访存效率、缓存、launch 开销并不成比例。折算出的加速比在 summary 里带 `baseline: h800_scaled`；`tools/delivery_table.py` 只显示数字，加 `--show-baseline` 才显示来源。

H800 上某个算例**没有** cuSPARSE 时间时（cuSPARSE SDDMM 不支持 fp16），退一步用同一行 H800 PyTorch 的时间（PyTorch 结果须通过校验），标 `baseline: h800_pytorch_scaled`；cuSPARSE 跑了但结果不对的行不退。FlagSparse 自己在 H800 上的时间永远不当基线。

C API 拆分交付（MUSA）：带 `--h800-reference` 时 pytest 那一半也跑性能（与本参照同一 benchmark、同一规模），NoBaseline 变体优先取它；取不到再用 C API 时间按**变体 + 矩阵名**匹配（C API 的 SpMM `n=8`、SDDMM `k=16/64`、按矩阵测的向量类在这里没有对应规模，只有 SpMV 类能对上）。

## 覆盖情况与限制

- 只有带 cuSPARSE（或 vendor）时间的行才能做基线；下表“可作基线的行”就是这个数。
  为 0 的算子（当时只有 CuPy 基线、没有对应格式的库）**没有折算基线**，行会保持 `N/A`。
- FlagSparse commit 带 `+dirty` 表示那轮 H800 是在有未提交改动的工作树上跑的。
- “来源”一列不是本轮的算子，是从上一份参考文件带过来的（新一轮没测它们），数据来自那一轮。

## 各算子收录情况

| 算子 | 行数 | 可作基线的行 | 来源 | 保留的时间 / 加速比列 |
|---|---:|---:|---|---|
| `axpby` | 12 | 12 | 本轮 | `triton_ms`, `cusparse_ms`, `pytorch_ms`, `triton_speedup_vs_cusparse`, `triton_speedup_vs_pytorch`, `cupy_ms`, `triton_speedup_vs_cupy` |
| `gather` | 20 | 20 | 本轮 | `triton_ms`, `pytorch_ms`, `cusparse_ms`, `triton_speedup_vs_pytorch`, `triton_speedup_vs_cusparse` |
| `geam_csr` | 20 | 20 | 本轮 | `triton_ms`, `cusparse_ms`, `pytorch_ms`, `triton_speedup_vs_cusparse`, `triton_speedup_vs_pytorch` |
| `gemvi` | 12 | 12 | 本轮 | `triton_ms`, `cusparse_ms`, `pytorch_ms`, `triton_speedup_vs_cusparse`, `triton_speedup_vs_pytorch` |
| `scatter` | 24 | 24 | 本轮 | `triton_ms`, `pytorch_ms`, `cusparse_ms`, `triton_speedup_vs_pytorch`, `triton_speedup_vs_cusparse` |
| `sddmm_csr` | 170 | 170 | 本轮 | `triton_ms`, `pytorch_api_ms`, `triton_speedup_vs_pytorch_api`, `triton_speedup_vs_pytorch`, `cusparse_ms`, `pytorch_ms`, `triton_speedup_vs_cusparse` |
| `spgemm_csr` | 30 | 30 | 本轮 | `triton_ms`, `cusparse_ms`, `pytorch_ms`, `triton_speedup_vs_cusparse`, `triton_speedup_vs_pytorch` |
| `spmm_bell` | 240 | 0 | `pytest_results_sparse_202609221038` (`47a441e6c810+dirty`) | `ms`, `vendor_ms`, `torch_bell_ms`, `cupy_bell_ms` |
| `spmm_bsr` | 513 | 0 | `pytest_results_sparse_202609221038` (`47a441e6c810+dirty`) | `ms`, `torch_ms`, `cusparse_ms`, `torch_vs_alg_speedup`, `cusparse_vs_alg_speedup`, `scipy_vs_alg_speedup` |
| `spmm_coo` | 90 | 90 | 本轮 | `ms`, `torch_ms`, `cusparse_ms`, `triton_speedup_vs_pytorch`, `cusparse_vs_alg_speedup` |
| `spmm_csc` | 290 | 130 | 本轮 | `ms`, `pytorch_ms`, `triton_speedup_vs_pytorch`, `cusparse_ms`, `cusparse_vs_alg_speedup` |
| `spmm_csr` | 160 | 160 | 本轮 | `triton_ms`, `cusparse_ms`, `pytorch_ms`, `triton_speedup_vs_cusparse`, `triton_speedup_vs_pytorch` |
| `spmv_bsr` | 960 | 0 | `pytest_results_sparse_202609221038` (`47a441e6c810+dirty`) | `bsr_ms`, `pytorch_ms`, `bsr_speedup_vs_pytorch`, `pytorch_padded_ms`, `cusparse_ms` |
| `spmv_coo` | 110 | 109 | 本轮 | `base_ms`, `opt_ms`, `cusparse_ms`, `pytorch_ms`, `opt_speedup_vs_cusparse`, `opt_speedup_vs_pytorch` |
| `spmv_csc` | 310 | 310 | 本轮 | `csc_ms`, `pytorch_ms`, `csc_speedup_vs_pytorch`, `cusparse_ms`, `csc_speedup_vs_cusparse` |
| `spmv_csr` | 150 | 150 | 本轮 | `ms`, `vendor_ms`, `speedup_vs_vendor`, `pytorch_ms`, `triton_speedup_vs_pytorch` |
| `spmv_sell` | 50 | 50 | 本轮 | `triton_ms`, `cusparse_ms`, `pytorch_ms`, `cupy_ms`, `triton_speedup_vs_cupy`, `triton_speedup_vs_cusparse`, `triton_speedup_vs_pytorch` |
| `spsm_coo` | 120 | 92 | `pytest_results_sparse_202609221038` (`47a441e6c810+dirty`) | `FlagSparse_ms`, `cuSPARSE_ms`, `hipSPARSE_ms`, `FlagSparse_vs_vendor_speedup` |
| `spsv_coo` | 720 | 600 | `pytest_results_sparse_202609221038` (`47a441e6c810+dirty`) | `FlagSparse_ms`, `CuPy/cuSPARSE_ms`, `PyTorch_ms`, `FlagSparse_vs_CuPy/cuSPARSE_speedup`, `FlagSparse_vs_PyTorch_speedup` |
| `spsv_csr` | 720 | 600 | `pytest_results_sparse_202609221038` (`47a441e6c810+dirty`) | `FlagSparse_ms`, `CuPy/cuSPARSE_ms`, `PyTorch_ms`, `FlagSparse_vs_CuPy/cuSPARSE_speedup`, `FlagSparse_vs_PyTorch_speedup` |
| `spsv_sell` | 240 | 200 | `pytest_results_sparse_202609221038` (`47a441e6c810+dirty`) | `FlagSparse_ms`, `cuSPARSE_ms`, `PyTorch_ms`, `FlagSparse_vs_cuSPARSE_speedup`, `FlagSparse_vs_PyTorch_speedup` |
| `spvv` | 24 | 24 | 本轮 | `triton_ms`, `cusparse_ms`, `pytorch_ms`, `triton_speedup_vs_cusparse`, `triton_speedup_vs_pytorch`, `cupy_ms`, `triton_speedup_vs_cupy` |

## 各交付变体能否折算

按变体统计本轮 H800 上带厂商库时间的矩阵数；为 0 的变体只能退到 H800 PyTorch 的时间（见上）。

| 变体 | 可作基线的矩阵 / 总矩阵 |
|---|---|
| `axpby_c32_int` | 4 / 4 |
| `axpby_f16_int` | 4 / 4 |
| `axpby_f32_int` | 4 / 4 |
| `gather complex128 (legacy)` | 4 / 4 |
| `gather complex64 (legacy)` | 4 / 4 |
| `gather float16 (legacy)` | 4 / 4 |
| `gather float32 (legacy)` | 4 / 4 |
| `gather float64 (legacy)` | 4 / 4 |
| `geam_csr_c32_int` | 10 / 10 |
| `geam_csr_f32_int` | 10 / 10 |
| `gemvi_c32_int_non` | 4 / 4 |
| `gemvi_f32_int_non` | 4 / 4 |
| `gemvi_f32_int_trans` | 4 / 4 |
| `scatter complex128 (legacy)` | 4 / 4 |
| `scatter complex64 (legacy)` | 4 / 4 |
| `scatter float16 (legacy)` | 4 / 4 |
| `scatter float32 (legacy)` | 4 / 4 |
| `scatter float64 (legacy)` | 4 / 4 |
| `scatter int8 (legacy)` | 4 / 4 |
| `sddmm_csr float32 (legacy)` | 40 / 40 |
| `sddmm_csr float64 (legacy)` | 40 / 40 |
| `sddmm_csr_c32_int_non_non_row` | 10 / 10 |
| `sddmm_csr_c32_int_non_trans_row` | 10 / 10 |
| `sddmm_csr_f16_int_non_non_row` | 10 / 10 |
| `sddmm_csr_f16_int_non_trans_row` | 10 / 10 |
| `sddmm_csr_f16f32_int_non_non_row` | 10 / 10 |
| `sddmm_csr_f32_int_non_non_col` | 10 / 10 |
| `sddmm_csr_f32_int_non_trans_row` | 10 / 10 |
| `sddmm_csr_f32_int_trans_non_row` | 10 / 10 |
| `sddmm_csr_f32_int_trans_trans_row` | 10 / 10 |
| `spgemm_csr float32 (legacy)` | 10 / 10 |
| `spgemm_csr float64 (legacy)` | 10 / 10 |
| `spgemm_csr_c32_int_non_non` | 10 / 10 |
| `spmm_coo float32 (legacy)` | 10 / 10 |
| `spmm_coo float64 (legacy)` | 10 / 10 |
| `spmm_coo_c32_int_non_non_row` | 10 / 10 |
| `spmm_coo_f16_int_non_non_row` | 10 / 10 |
| `spmm_coo_f16f32_int_non_non_row` | 10 / 10 |
| `spmm_coo_f32_int_non_non_col` | 10 / 10 |
| `spmm_coo_f32_int_non_trans_row` | 10 / 10 |
| `spmm_coo_f32_int_trans_non_row` | 10 / 10 |
| `spmm_coo_i8i32_int_non_non_row` | 10 / 10 |
| `spmm_csc complex128 (legacy)` | 20 / 60 |
| `spmm_csc complex64 (legacy)` | 20 / 60 |
| `spmm_csc float32 (legacy)` | 20 / 60 |
| `spmm_csc float64 (legacy)` | 20 / 60 |
| `spmm_csc_c32_int_non_non_row` | 10 / 10 |
| `spmm_csc_f16_int_non_non_row` | 10 / 10 |
| `spmm_csc_f16f32_int_non_non_row` | 10 / 10 |
| `spmm_csc_f32_int_non_non_row` | 10 / 10 |
| `spmm_csc_f32_int_trans_non_row` | 10 / 10 |
| `spmm_csr float32 (legacy)` | 10 / 10 |
| `spmm_csr float64 (legacy)` | 10 / 10 |
| `spmm_csr_bf16f32_int_non_non_row` | 10 / 10 |
| `spmm_csr_c32_int_conj_non_row` | 10 / 10 |
| `spmm_csr_c32_int_non_non_row` | 10 / 10 |
| `spmm_csr_c32_int_non_trans_row` | 10 / 10 |
| `spmm_csr_f16_int_non_non_row` | 10 / 10 |
| `spmm_csr_f16f32_int_non_non_col` | 10 / 10 |
| `spmm_csr_f16f32_int_non_non_row` | 10 / 10 |
| `spmm_csr_f16f32_int_trans_non_row` | 10 / 10 |
| `spmm_csr_f32_int_non_non_col` | 10 / 10 |
| `spmm_csr_f32_int_non_trans_row` | 10 / 10 |
| `spmm_csr_f32_int_trans_non_row` | 10 / 10 |
| `spmm_csr_i8f32_int_non_non_row` | 10 / 10 |
| `spmm_csr_i8i32_int_non_non_col` | 10 / 10 |
| `spmm_csr_i8i32_int_non_non_row` | 10 / 10 |
| `spmv_coo float32 (legacy)` | 10 / 10 |
| `spmv_coo float64 (legacy)` | 10 / 10 |
| `spmv_coo_c32_int_conj` | 10 / 10 |
| `spmv_coo_c32_int_non` | 10 / 10 |
| `spmv_coo_f16_int_non` | 10 / 10 |
| `spmv_coo_f16_int_trans` | 9 / 10 |
| `spmv_coo_f16f32_int_non` | 10 / 10 |
| `spmv_coo_f32_int_trans` | 10 / 10 |
| `spmv_coo_f32c32_int_non` | 10 / 10 |
| `spmv_coo_i8f32_int_non` | 10 / 10 |
| `spmv_coo_i8i32_int_non` | 10 / 10 |
| `spmv_csc complex128 (legacy)` | 60 / 60 |
| `spmv_csc complex64 (legacy)` | 60 / 60 |
| `spmv_csc float32 (legacy)` | 60 / 60 |
| `spmv_csc float64 (legacy)` | 60 / 60 |
| `spmv_csc_c32_int_conj` | 10 / 10 |
| `spmv_csc_c32_int_non` | 10 / 10 |
| `spmv_csc_f16_int_non` | 10 / 10 |
| `spmv_csc_f16f32_int_non` | 10 / 10 |
| `spmv_csc_f32_int_non` | 10 / 10 |
| `spmv_csc_f32_int_trans` | 10 / 10 |
| `spmv_csc_i8i32_int_non` | 10 / 10 |
| `spmv_csr float32 (legacy)` | 10 / 10 |
| `spmv_csr float64 (legacy)` | 10 / 10 |
| `spmv_csr_bf16_int_non` | 10 / 10 |
| `spmv_csr_bf16f32_int_non` | 10 / 10 |
| `spmv_csr_c32_int_conj` | 10 / 10 |
| `spmv_csr_c32_int_non` | 10 / 10 |
| `spmv_csr_f16_int_non` | 10 / 10 |
| `spmv_csr_f16f32_int_non` | 10 / 10 |
| `spmv_csr_f16f32_int_trans` | 10 / 10 |
| `spmv_csr_f32_int_trans` | 10 / 10 |
| `spmv_csr_f32c32_int_non` | 10 / 10 |
| `spmv_csr_f32c32_int_trans` | 10 / 10 |
| `spmv_csr_i8f32_int_non` | 10 / 10 |
| `spmv_csr_i8i32_int_non` | 10 / 10 |
| `spmv_csr_i8i32_int_trans` | 10 / 10 |
| `spmv_sell_c32_int_non` | 10 / 10 |
| `spmv_sell_f16_int_non` | 10 / 10 |
| `spmv_sell_f16f32_int_non` | 10 / 10 |
| `spmv_sell_f32_int_non` | 10 / 10 |
| `spmv_sell_i8i32_int_non` | 10 / 10 |
| `spvv_c32_int_conj` | 4 / 4 |
| `spvv_c32_int_non` | 4 / 4 |
| `spvv_f16f32_int_non` | 4 / 4 |
| `spvv_f32_int_non` | 4 / 4 |
| `spvv_i8f32_int_non` | 4 / 4 |
| `spvv_i8i32_int_non` | 4 / 4 |

## H800 上 105 个交付变体的结果

| 变体 | 精度 | 性能 | 加速比（H800 vs cuSPARSE/PyTorch，runner 口径） |
|---|---|---|---|
| `axpby_c32_int` | Passed (2/2) | Passed | complex64: 0.206 |
| `axpby_f16_int` | Passed (2/2) | Passed | fp16: 0.408 |
| `axpby_f32_int` | Passed (2/2) | Passed | fp32: 0.292 |
| `gather_c32_int` | Passed (2/2) | Passed | complex64: 1.478 |
| `gather_c64_int` | Passed (3/3) | Passed | complex128: 1.232 |
| `gather_f16_int` | Passed (2/2) | Passed | fp16: 1.132 |
| `gather_f32_int` | Passed (2/2) | Passed | fp32: 1.053 |
| `gather_f64_int` | Passed (2/2) | Passed | fp64: 1.471 |
| `geam_csr_c32_int` | Passed (2/2) | Passed | complex64: 1.553 |
| `geam_csr_f32_int` | Passed (2/2) | Passed | fp32: 1.543 |
| `gemvi_c32_int_non` | Passed (2/2) | Passed | complex64: 0.817 |
| `gemvi_f32_int_non` | Passed (2/2) | Passed | fp32: 0.791 |
| `gemvi_f32_int_trans` | Passed (2/2) | Passed | fp32: 0.997 |
| `scatter_c32_int` | Passed (4/4) | Passed | complex64: 0.758 |
| `scatter_c64_int` | Passed (4/4) | Passed | complex128: 0.822 |
| `scatter_f16_int` | Passed (4/4) | Passed | fp16: 0.808 |
| `scatter_f32_int` | Passed (4/4) | Passed | fp32: 0.991 |
| `scatter_f64_int` | Passed (4/4) | Passed | fp64: 0.981 |
| `scatter_i8_int` | Passed (2/2) | Passed | int8: 1.607 |
| `sddmm_csr_c32_int_non_non_row` | Passed (2/2) | Passed | complex64: 1.188 |
| `sddmm_csr_c32_int_non_trans_row` | Passed (2/2) | Passed | complex64: 0.402 |
| `sddmm_csr_f16_int_non_non_row` | Passed (2/2) | Passed | fp16: 2.068 |
| `sddmm_csr_f16_int_non_trans_row` | Passed (2/2) | Passed | fp16: 2.634 |
| `sddmm_csr_f16f32_int_non_non_row` | Passed (2/2) | Passed | fp16: 2.131 |
| `sddmm_csr_f32_int_non_non_col` | Passed (2/2) | Passed | fp32: 2.447 |
| `sddmm_csr_f32_int_non_non_row` | Passed (4/6) | Passed | fp32: 1.473 |
| `sddmm_csr_f32_int_non_trans_row` | Passed (2/2) | Passed | fp32: 2.152 |
| `sddmm_csr_f32_int_trans_non_row` | Passed (2/2) | Passed | fp32: 1.537 |
| `sddmm_csr_f32_int_trans_trans_row` | Passed (2/2) | Passed | fp32: 2.449 |
| `sddmm_csr_f64_int_non_non_row` | Passed (4/4) | Passed | fp64: 1.151 |
| `spgemm_csr_c32_int_non_non` | Passed (2/2) | Passed | complex64: 0.759 |
| `spgemm_csr_f32_int_non_non` | Passed (2/2) | Passed | fp32: 0.695 |
| `spmm_coo_c32_int_non_non_row` | Passed (2/2) | Passed | complex64: 0.354 |
| `spmm_coo_f16_int_non_non_row` | Passed (2/2) | Passed | fp16: 0.908 |
| `spmm_coo_f16f32_int_non_non_row` | Passed (2/2) | Passed | fp16: 0.621 |
| `spmm_coo_f32_int_non_non_col` | Passed (2/2) | Passed | fp32: 1.012 |
| `spmm_coo_f32_int_non_non_row` | Passed (2/2) | Passed | fp32: 1.170 |
| `spmm_coo_f32_int_non_trans_row` | Passed (2/2) | Passed | fp32: 1.407 |
| `spmm_coo_f32_int_trans_non_row` | Passed (2/2) | Passed | fp32: 0.607 |
| `spmm_coo_f64_int_non_non_row` | Passed (2/2) | Passed | fp64: 4.483 |
| `spmm_coo_i8i32_int_non_non_row` | Passed (2/2) | Passed | int8: 0.210 |
| `spmm_csc_c32_int_non_non_row` | Passed (2/2) | Passed | complex64: 0.182 |
| `spmm_csc_f16_int_non_non_row` | Passed (2/2) | Passed | fp16: 0.537 |
| `spmm_csc_f16f32_int_non_non_row` | Passed (2/2) | Passed | fp16: 0.375 |
| `spmm_csc_f32_int_non_non_row` | Passed (2/2) | Passed | fp32: 0.401 |
| `spmm_csc_f32_int_trans_non_row` | Passed (2/2) | Passed | fp32: 0.439 |
| `spmm_csr_bf16f32_int_non_non_row` | Passed (2/2) | Passed | bf16: 1.169 |
| `spmm_csr_c32_int_conj_non_row` | Passed (2/2) | Passed | complex64: 0.743 |
| `spmm_csr_c32_int_non_non_row` | Passed (2/2) | Passed | complex64: 0.330 |
| `spmm_csr_c32_int_non_trans_row` | Passed (2/2) | Passed | complex64: 0.890 |
| `spmm_csr_f16_int_non_non_row` | Passed (2/2) | Passed | fp16: 2.313 |
| `spmm_csr_f16f32_int_non_non_col` | Passed (2/2) | Passed | fp16: 0.572 |
| `spmm_csr_f16f32_int_non_non_row` | Passed (2/2) | Passed | fp16: 1.173 |
| `spmm_csr_f16f32_int_trans_non_row` | Passed (2/2) | Passed | fp16: 1.283 |
| `spmm_csr_f32_int_non_non_col` | Passed (2/2) | Passed | fp32: 3.274 |
| `spmm_csr_f32_int_non_non_row` | Passed (4/4) | Passed | fp32: 3.605 |
| `spmm_csr_f32_int_non_trans_row` | Passed (2/2) | Passed | fp32: 4.569 |
| `spmm_csr_f32_int_trans_non_row` | Passed (2/2) | Passed | fp32: 1.758 |
| `spmm_csr_f64_int_non_non_row` | Passed (4/4) | Passed | fp64: 3.352 |
| `spmm_csr_i8f32_int_non_non_row` | Passed (2/2) | Passed | int8: 1.163 |
| `spmm_csr_i8i32_int_non_non_col` | Passed (2/2) | Passed | int8: 0.571 |
| `spmm_csr_i8i32_int_non_non_row` | Passed (2/2) | Passed | int8: 1.223 |
| `spmv_coo_c32_int_conj` | Passed (2/2) | Passed | complex64: 1.110 |
| `spmv_coo_c32_int_non` | Passed (2/2) | Passed | complex64: 0.417 |
| `spmv_coo_f16_int_non` | Passed (2/2) | Passed | fp16: 0.352 |
| `spmv_coo_f16_int_trans` | Passed (2/2) | Passed | fp16: 2.493 |
| `spmv_coo_f16f32_int_non` | Passed (2/2) | Passed | fp16: 0.373 |
| `spmv_coo_f32_int_non` | Passed (5/5) | Passed | fp32: 1.228 |
| `spmv_coo_f32_int_trans` | Passed (2/2) | Passed | fp32: 0.929 |
| `spmv_coo_f32c32_int_non` | Passed (2/2) | Passed | fp32: 0.358 |
| `spmv_coo_f64_int_non` | Passed (3/3) | Passed | fp64: 1.310 |
| `spmv_coo_i8f32_int_non` | Passed (2/2) | Passed | int8: 0.342 |
| `spmv_coo_i8i32_int_non` | Passed (2/2) | Passed | int8: 0.363 |
| `spmv_csc_c32_int_conj` | Passed (2/2) | Passed | complex64: 0.265 |
| `spmv_csc_c32_int_non` | Passed (2/2) | Passed | complex64: 1.365 |
| `spmv_csc_f16_int_non` | Passed (2/2) | Passed | fp16: 2.689 |
| `spmv_csc_f16f32_int_non` | Passed (2/2) | Passed | fp16: 0.857 |
| `spmv_csc_f32_int_non` | Passed (2/2) | Passed | fp32: 0.976 |
| `spmv_csc_f32_int_trans` | Passed (2/2) | Passed | fp32: 0.244 |
| `spmv_csc_i8i32_int_non` | Passed (2/2) | Passed | int8: 0.266 |
| `spmv_csr_bf16_int_non` | Passed (2/2) | Passed | bf16: 0.280 |
| `spmv_csr_bf16f32_int_non` | Passed (2/2) | Passed | bf16: 0.953 |
| `spmv_csr_c32_int_conj` | Passed (2/2) | Passed | complex64: 1.143 |
| `spmv_csr_c32_int_non` | Passed (2/2) | Passed | complex64: 0.481 |
| `spmv_csr_f16_int_non` | Passed (2/2) | Passed | fp16: 0.538 |
| `spmv_csr_f16f32_int_non` | Passed (2/2) | Passed | fp16: 0.921 |
| `spmv_csr_f16f32_int_trans` | Passed (2/2) | Passed | fp16: 0.737 |
| `spmv_csr_f32_int_non` | Passed (59/59) | Passed | fp32: 0.607 |
| `spmv_csr_f32_int_trans` | Passed (2/2) | Passed | fp32: 0.915 |
| `spmv_csr_f32c32_int_non` | Passed (2/2) | Passed | fp32: 1.027 |
| `spmv_csr_f32c32_int_trans` | Passed (2/2) | Passed | fp32: 0.852 |
| `spmv_csr_f64_int_non` | Passed (55/55) | Passed | fp64: 0.675 |
| `spmv_csr_i8f32_int_non` | Passed (2/2) | Passed | int8: 0.978 |
| `spmv_csr_i8i32_int_non` | Passed (2/2) | Passed | int8: 0.966 |
| `spmv_csr_i8i32_int_trans` | Passed (2/2) | Passed | int8: 0.755 |
| `spmv_sell_c32_int_non` | Passed (2/2) | Passed | complex64: 0.619 |
| `spmv_sell_f16_int_non` | Passed (2/2) | Passed | fp16: 0.489 |
| `spmv_sell_f32_int_non` | Passed (2/2) | Passed | fp32: 0.542 |
| `spmv_sell_i8i32_int_non` | Passed (2/2) | Passed | int8: 0.416 |
| `spvv_c32_int_conj` | Passed (2/2) | Passed | complex64: 0.429 |
| `spvv_c32_int_non` | Passed (2/2) | Passed | complex64: 0.396 |
| `spvv_f16f32_int_non` | Passed (2/2) | Passed | fp16: 0.474 |
| `spvv_f32_int_non` | Passed (2/2) | Passed | fp32: 0.544 |
| `spvv_i8f32_int_non` | Passed (2/2) | Passed | int8: 0.492 |
| `spvv_i8i32_int_non` | Passed (2/2) | Passed | int8: 0.487 |

## 重新生成

```bash
python3 tools/h800_reference.py <H800 结果目录>
```

只有在 H800 上重跑、且新结果确实要取代这份参考时才重新生成；生成后 diff 应当只有数值变化。
