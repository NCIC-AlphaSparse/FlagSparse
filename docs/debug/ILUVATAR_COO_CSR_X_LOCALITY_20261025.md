# Iluvatar CSR/COO c32 x-gather 局部性实验（2026-10-25）

## 结论

本轮针对 `auto` 和 `NACA0015` 的 `spmv_csr_c32_int_non`、
`spmv_coo_c32_int_non` 尝试了只重排列号、以及 RCM 行列联合重排两条路径。
BI-V150 真机结果均未优于当前 batched gather，因此**没有把实验代码保留到生产路径**，
也没有改变本轮开始前的默认算子实现。

这里的加速比统一是相对 H800 cuSPARSE 折算基线，不是 CSR 与 COO 互相比：

| 算子/矩阵 | 当前时间（ms） | H800 折算基线（ms） | 当前加速比 |
|---|---:|---:|---:|
| `spmv_csr_c32_int_non` / `auto` | 0.46246 | 0.16615 | 0.3593x |
| `spmv_csr_c32_int_non` / `NACA0015` | 0.39029 | 0.15656 | 0.4006x |
| `spmv_coo_c32_int_non` / `auto` | 0.51828 | 0.15658 | 0.3021x |
| `spmv_coo_c32_int_non` / `NACA0015` | 0.45403 | 0.13819 | 0.3043x |

COO 的 H800 折算分母来自 `results_biv_65_20261024_final/spmv_coo/performance.csv`；
之前 10 矩阵 COO c32 自动路由的算术平均仍为 `0.5410x`、几何平均 `0.5369x`，
见 `docs/debug/ILUVATAR_COO_CSR_C32_NNZ_20261025.md`。

## A/B 方法

- 设备：Iluvatar BI-V150 OAM；值类型 `complex64`；`op=non`；prepared 运行。
- 计时：预热 4 次，随后 7 次，取中位数；首轮 Triton 编译不计入复测样本。
- 结构预处理在 prepare 阶段执行；对每次运行的 `x` 置换单独放在计时区间外，
  kernel 计时只覆盖 gather。由于 x 置换本身不能在没有 x 的 prepare 接口中完成，
  prepare 只保存映射，不能把实际数据复制提前到 prepare。
- 精度：重排方案连续运行结果一致，未发现数值异常；由于方案未采纳，未增加正式精度回归用例。

### 1. 只重排列号

在 prepare 中生成 RCM 的 `old column -> new column` 映射，运行前把 `x` 按
`new -> old` 置换，CSR/COO 的输出行顺序保持不变，避免 y scatter。

| 矩阵 | 原始 gather（ms） | RCM 列号（ms） | RCM/原始 |
|---|---:|---:|---:|
| `auto` | 约 0.452 | 约 0.565 | 1.25x（回退） |
| `NACA0015` | 约 0.453 | 约 0.570 | 1.26x（回退） |

RCM 虽然降低了 CPU 侧按行列跨度指标，但在 BI-V150 的当前 gather kernel 中，
只改列标签没有带来对应的 GPU 局部性收益，反而增加了 kernel 访问/调度成本。

### 2. RCM 行列联合重排、原位写回 y

进一步把 CSR 行内数据按 RCM 行顺序重排，kernel 仍用原始行号直接写 y，试图避免
SELL 路径中的随机 y 回写。

| 矩阵 | 原始 gather（ms） | RCM 行列（ms） | RCM/原始 |
|---|---:|---:|---:|
| `auto` | 约 0.566 | 约 0.608 | 1.07x（回退） |
| `NACA0015` | 约 0.453 | 约 1.492 | 3.29x（回退） |

该路径还需要额外的行内值布局和 kernel 分支，收益不足以抵消开销。

## 采纳决定

- 未修改默认 `spmv_csr_c32_int_non` 或 `spmv_coo_c32_int_non` 路由。
- 未新增 x permutation、RCM 列映射或 y 回写代码；本轮结束后工作区只保留本轮
  之前已有的 CSR/COO NNZ、SELL 和 probe 改动。
- x-gather 的下一条可行方向不是单独重排 x，而是设计真正的 cache-blocked gather
  kernel：在 kernel 内按列块组织多个输出行，并证明块内 x 缓存复用能覆盖计划/临时
  缓冲开销；仍需先在 BI-V150 上做真机 A/B，不能仅凭 RCM 的图带宽指标宣称达标。
