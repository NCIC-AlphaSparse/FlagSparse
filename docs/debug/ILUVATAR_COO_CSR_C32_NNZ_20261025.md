# BI-V150 COO/CSR c32 NNZ 路由（2026-10-25）

## 结论

本轮把 COO 已有的 NNZ 均分复数 kernel 扩展到 CSR c32。CSR 不再为
`平均行长 <= 4` 且 `最长行 > 128` 的长尾矩阵强行使用 row-gather；这类矩阵
在 prepare 阶段生成 `row_ids`，运行阶段使用 `_complex_nnz_kernel`。

默认只命中结构条件，不按矩阵名特判：

```text
mean_nnz <= 4 and max_row_nnz > 128
```

本机 10 矩阵结果显示，条件只命中 `ASIC_680ks`。CSR c32 的平均加速比从
`0.6290x` 提升到 `0.6478x`，几何平均从 `0.6204x` 提升到 `0.6401x`；
ASIC 单项从 `0.5128x` 提升到 `0.7036x`。仍未达到 `0.8x`，因此不能宣称
该算子达标。

COO 当前自动 NNZ 路由的复测平均加速比为 `0.5410x`，几何平均为 `0.5369x`；
ASIC 为 `0.5226x`。COO 本轮没有新增默认路由，只做了同口径复测。

## 实现

- `src/flagsparse/sparse_operations/spmv_csr.py` 在 `prepare_spmv_csr` 中为符合条件的
  CSR c32 计划建立 `row_ids` 和空行列表，分桶/重复行号不计入 prepared kernel 计时。
- 运行阶段复用 `src/flagsparse/sparse_operations/_spmv_csr_nnz.py` 的
  `_complex_nnz_kernel`，连续处理 NNZ 并按 row id 做分段扫描。
- 新开关为 `FLAGSPARSE_SPMV_CSR_COMPLEX_NNZ={auto,0,1}`；默认是 `auto`。
  `0` 禁用，`1` 仅用于实验强制开启，不能作为默认配置。
- `FLAGSPARSE_SPMV_CSR_COMPLEX_ROUTE=row_tile|segbin` 会绕过这条 NNZ 路径，保留
  原有显式路由语义。
- 可用 `FLAGSPARSE_SPMV_CSR_COMPLEX_NNZ_BLOCK` 和
  `FLAGSPARSE_SPMV_CSR_COMPLEX_NNZ_WARPS` 做真机调参；默认分别为 `128` 和 `1`。

## 真机 A/B

测试设备为 BI-V150，`complex64`、`int32/int32`、`op=non`，prepared 运行，
`warmup=5`、`iters=30`，未启用厂商基线。精度全部 PASS。

### CSR c32

| 矩阵 | NNZ 关闭 | 关闭加速比 | 自动路由 | 自动加速比 |
|---|---:|---:|---:|---:|
| ASIC_680ks | 0.134749 ms | 0.5128x | 0.098204 ms | **0.7036x** |
| GL7d14 | 0.110264 ms | 0.6445x | 0.110154 ms | 0.6451x |
| NACA0015 | 0.291137 ms | 0.5377x | 0.291812 ms | 0.5365x |
| amazon0601 | 0.151014 ms | 0.6076x | 0.150908 ms | 0.6080x |
| auto | 0.360046 ms | 0.4615x | 0.359558 ms | 0.4621x |
| cage12 | 0.068294 ms | 0.7613x | 0.068557 ms | 0.7584x |
| cfd2 | 0.081710 ms | 0.7741x | 0.081733 ms | 0.7739x |
| filter3D | 0.091241 ms | 0.6994x | 0.091223 ms | 0.6995x |
| roadNet-TX | 0.210323 ms | 0.5730x | 0.210794 ms | 0.5717x |
| wave | 0.074396 ms | 0.7183x | 0.074288 ms | 0.7194x |
| **算术平均** | — | **0.6290x** | — | **0.6478x** |
| **几何平均** | — | **0.6204x** | — | **0.6401x** |

强制 `FLAGSPARSE_SPMV_CSR_COMPLEX_NNZ=1` 的全量 A/B 为 `0.1904 ms` 算术平均，
明显慢于自动路由的 `0.1537 ms`；因此只保留结构化 `auto` 条件。

### COO c32 对照

COO 使用已有的 `FLAGSPARSE_ILUVATAR_COO_C32_NNZ=auto`，同样的 10 矩阵 prepared
复测结果为：算术平均 `0.5410x`、几何平均 `0.5369x`。关闭 COO NNZ 时分别为
`0.5259x` 和 `0.5191x`；ASIC 从 `0.3732x` 提升到 `0.5226x`。

## 结果文件

- `results_csr_c32_nnz_off_full_20261025.csv`
- `results_csr_c32_nnz_auto_full_20261025.csv`
- `results_coo_c32_nnz_off_full_20261025.csv`
- `results_coo_c32_nnz_auto_full_20261025.csv`
- `results_coo_csr_c32_nnz_auto_speedup_20261025.csv`
- `results_csr_mixed_nnz_prepared_ab_20261025.csv`
- `results_csr_mixed_nnz_prepared_speedup_20261025.csv`

相对基线均使用 `results_csr_f32c32_explore_20261024/spmv_csr_c32_retest_10mat_with_speedup.csv`
和 `results_biv_65_20261024_final/spmv_coo/performance.csv` 中的 H800 折算基线；
不是把 COO 和 CSR 互相作为分母。

## 混合精度 CSR 扩展

同一结构条件也接入了 prepared 混合 CSR 的实数 NNZ kernel，覆盖
`spmv_csr_f16f32_int_non`、`spmv_csr_f16_int_non` 和
`spmv_csr_i8i32_int_non`。新开关为
`FLAGSPARSE_ILUVATAR_CSR_MIXED_NNZ={auto,0,1}`，默认 `auto`；调用侧仍需使用
`prepare_spmv_csr_mixed`，prepare 生成的 row-id 计划不计入 kernel 时间。

10 个矩阵、每项 `warmup=5`、`iters=30` 的 prepared A/B 如下：

| 算子 | 算术平均加速比 | 几何平均加速比 | ASIC 加速比 | ASIC off→auto |
|---|---:|---:|---:|---:|
| `spmv_csr_f16f32_int_non` | **0.6781x** | 0.6717x | 0.5148x | 1.79x |
| `spmv_csr_f16_int_non` | **0.7086x** | 0.7017x | 0.5684x | 1.73x |
| `spmv_csr_i8i32_int_non` | **0.7174x** | 0.7095x | 0.5536x | 1.67x |

自动路由在其余 9 个矩阵保持 `cached`，结果与关闭 NNZ 的差异在噪声范围内；
三项输出均与关闭 NNZ 的 prepared 结果通过对比。混合精度的 q4 call 路径没有被
这次改动改变，仍按原有 call 计时口径。

## 下一步

CSR/COO c32 的 tile 扫描已没有稳定通用赢家；混合精度 CSR 的 prepared NNZ 路径
也已完成首轮验证。下一步应把相同路由与计时口径应用到 COO 的 prepared mixed
入口，或继续处理 `auto/NACA0015` 的 x gather 局部性；不能把强制 NNZ 推广到平均
行长大于 4 的矩阵。
