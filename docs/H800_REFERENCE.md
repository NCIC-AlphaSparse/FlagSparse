# H800 参考结果（随仓库分发）

> 本文件由 `tools/h800_reference.py` 从 H800 的一轮 runner 结果生成，**不要手改**；数据在 [`conf/h800_reference.json`](../conf/h800_reference.json)。

## 来源

- 设备：NVIDIA H800 × 8
- 原始结果目录：`pytest_results_sparse_202609221038`（被 git 忽略，不随仓库）
- FlagSparse commit：`47a441e6c810+dirty`；torch 2.8.0a0+5228986c39.nv25.05；triton 3.6.0
- 生成日期：2026-09-28

## 用途

没有厂商 / PyTorch 基线的后端，用这里的 **H800 上 cuSPARSE 的时间**做基线：

```
h800_scaled_ms = T_cuSPARSE@H800 × P_H800 / P_目标卡      (P = 实测显存带宽)
speedup        = h800_scaled_ms / T_FlagSparse@目标卡     (1.0 = 与折算后的 cuSPARSE 一样快，≥0.8 合格)
```

```bash
# 不用再找目录：不带值即用本仓库自带的文件
python3 run_flagsparse_pytest.py --h800-reference --vendor-card iluvatar-biv150 ...
python3 tools/baseline_bound.py --vendor <目标卡结果目录> --vendor-card iluvatar-biv150
```

这是**估算**，不是厂商实测：H800 与目标卡的访存效率、缓存、launch 开销并不成比例。

## 覆盖情况与限制

- 只有带 cuSPARSE（或 vendor）时间的行才能做基线；下表“可作基线的行”就是这个数。
  为 0 的算子（当时只有 CuPy 基线、没有对应格式的库）**没有折算基线**，行会保持 `N/A`。
- FlagSparse commit 带 `+dirty` 表示那轮 H800 是在有未提交改动的工作树上跑的。

## 各算子收录情况

| 算子 | 行数 | 可作基线的行 | 保留的时间 / 加速比列 |
|---|---:|---:|---|
| `gather` | 48 | 40 | `triton_ms`, `pytorch_ms`, `cusparse_ms`, `triton_speedup_vs_pytorch`, `triton_speedup_vs_cusparse` |
| `scatter` | 40 | 40 | `triton_ms`, `pytorch_ms`, `cusparse_ms`, `triton_speedup_vs_pytorch`, `triton_speedup_vs_cusparse` |
| `sddmm_csr` | 240 | 240 | `triton_ms`, `pytorch_api_ms`, `triton_speedup_vs_pytorch_api`, `triton_speedup_vs_pytorch`, `cusparse_ms`, `pytorch_ms`, `triton_speedup_vs_cusparse` |
| `spgemm_csr` | 60 | 58 | `triton_ms`, `cusparse_ms`, `pytorch_ms`, `triton_speedup_vs_cusparse`, `triton_speedup_vs_pytorch` |
| `spmm_bell` | 240 | 0 | `ms`, `vendor_ms`, `torch_bell_ms`, `cupy_bell_ms` |
| `spmm_bsr` | 513 | 0 | `ms`, `torch_ms`, `cusparse_ms`, `torch_vs_alg_speedup`, `cusparse_vs_alg_speedup`, `scipy_vs_alg_speedup` |
| `spmm_coo` | 240 | 240 | `ms`, `torch_ms`, `cusparse_ms`, `triton_speedup_vs_pytorch`, `cusparse_vs_alg_speedup` |
| `spmm_csc` | 720 | 240 | `ms`, `pytorch_ms`, `triton_speedup_vs_pytorch`, `cusparse_ms`, `cusparse_vs_alg_speedup` |
| `spmm_csr` | 720 | 720 | `triton_ms`, `cusparse_ms`, `pytorch_ms`, `triton_speedup_vs_cusparse`, `triton_speedup_vs_pytorch` |
| `spmv_bsr` | 960 | 0 | `bsr_ms`, `pytorch_ms`, `bsr_speedup_vs_pytorch`, `pytorch_padded_ms`, `cusparse_ms` |
| `spmv_coo` | 720 | 720 | `base_ms`, `opt_ms`, `cusparse_ms`, `pytorch_ms`, `opt_speedup_vs_cusparse`, `opt_speedup_vs_pytorch` |
| `spmv_csc` | 720 | 720 | `csc_ms`, `pytorch_ms`, `csc_speedup_vs_pytorch`, `cusparse_ms` |
| `spmv_csr` | 780 | 780 | `ms`, `vendor_ms`, `speedup_vs_vendor` |
| `spsm_coo` | 120 | 92 | `FlagSparse_ms`, `cuSPARSE_ms`, `hipSPARSE_ms`, `FlagSparse_vs_vendor_speedup` |
| `spsv_coo` | 720 | 600 | `FlagSparse_ms`, `CuPy/cuSPARSE_ms`, `PyTorch_ms`, `FlagSparse_vs_CuPy/cuSPARSE_speedup`, `FlagSparse_vs_PyTorch_speedup` |
| `spsv_csr` | 720 | 600 | `FlagSparse_ms`, `CuPy/cuSPARSE_ms`, `PyTorch_ms`, `FlagSparse_vs_CuPy/cuSPARSE_speedup`, `FlagSparse_vs_PyTorch_speedup` |
| `spsv_sell` | 240 | 200 | `FlagSparse_ms`, `cuSPARSE_ms`, `PyTorch_ms`, `FlagSparse_vs_cuSPARSE_speedup`, `FlagSparse_vs_PyTorch_speedup` |

## H800 上 20 个交付变体的结果

| 变体 | 精度 | 性能 | 加速比（H800 vs cuSPARSE/PyTorch，runner 口径） |
|---|---|---|---|
| `gather_c32_int` | Passed (1/1) | Passed | complex64: 1.464 |
| `gather_c64_int` | Passed (2/2) | Passed | complex128: 1.215 |
| `gather_f16_int` | Passed (1/1) | Passed | fp16: 1.137 |
| `gather_f32_int` | Passed (1/1) | Passed | fp32: 1.055 |
| `gather_f64_int` | Passed (1/1) | Passed | fp64: 1.469 |
| `scatter_c32_int` | Passed (2/2) | Passed | complex64: 0.760 |
| `scatter_c64_int` | Passed (2/2) | Passed | complex128: 0.814 |
| `scatter_f16_int` | Passed (2/2) | Passed | fp16: 0.807 |
| `scatter_f32_int` | Passed (2/2) | Passed | fp32: 0.982 |
| `scatter_f64_int` | Passed (2/2) | Passed | fp64: 0.978 |
| `sddmm_csr_f32_int_non_non_row` | Passed (2/2) | Passed | fp32: 3.284 |
| `sddmm_csr_f64_int_non_non_row` | Passed (2/2) | Passed | fp64: 1.597 |
| `spmm_coo_f32_int_non_non_row` | Passed (1/1) | Passed | fp32: 0.763 |
| `spmm_coo_f64_int_non_non_row` | Passed (1/1) | Passed | fp64: 3.914 |
| `spmm_csr_f32_int_non_non_row` | Passed (2/2) | Passed | fp32: 1.133 |
| `spmm_csr_f64_int_non_non_row` | Passed (2/2) | Passed | fp64: 1.381 |
| `spmv_coo_f32_int_non` | Passed (1/1) | Passed | fp32: 0.912 |
| `spmv_coo_f64_int_non` | Passed (1/1) | Passed | fp64: 1.077 |
| `spmv_csr_f32_int_non` | Passed (57/57) | Passed | fp32: 0.314 |
| `spmv_csr_f64_int_non` | Passed (53/53) | Passed | fp64: 0.405 |

## 重新生成

```bash
python3 tools/h800_reference.py <H800 结果目录>
```

只有在 H800 上重跑、且新结果确实要取代这份参考时才重新生成；生成后 diff 应当只有数值变化。
