# 华为昇腾（910B）debug

`FLAGSPARSE_BACKEND=ascend`。环境、交付复现见 [../ASCEND.md](../ASCEND.md)。

当前 debug 入口使用 Python runner：
`python3 tools/run_backend_tests.py --backend ascend --phase both --mode normal`。
Ascend 的 C API adaptor 尚未可构建，不要用 CTest 结果判断该后端算子。

## 1. 这个后端的特点

- 昇腾的 Triton 缺 shmem 扩展，也降不下 `associative_scan`，FlagSparse 的大多数 Triton kernel 都编译不了。
  所以已有的 7 个交付算子**全部走 torch_npu 回退**（`index_select` / `index_add_` 等）。
- 这意味着昇腾报告里的加速比，是 torch_npu 回退和 PyTorch-NPU 基线比，**是同一个库自己和自己比**，
  CSV 里有标注，汇报时要说明。
- 交付测试用卡是 **910B**。带宽按公开值 1600 GB/s（`--vendor-card ascend-910b`）；
  3070 GB/s 是 Atlas 800T A3（910C）的实测值，不是测试用卡，不要用。

## 2. q4 42 个变体：在 CUDA 上模拟昇腾分支的结果

复现：`PYTHONPATH=src python3 tools/q4_ascend_dispatch_check.py`（CUDA 机器上跑，强制所有模块走昇腾分支、禁止 Triton 启动）。

**36 / 42 走 torch 路径且结果正确。** 其余 6 个：

| 变体 | 情况 | 原因 | 建议 |
|---|---|---|---|
| `spmv_csc_f32_int_non`、`spmv_csc_c32_int_non` | 仍调用 Triton（`_spmv_csc_non_*_kernel`） | `spmv_csc` 没有昇腾分支（原有算子） | 上机确认能否编译；不能就补 torch_npu 路径（CSC 的 op=non 相当于按列 `index_add_`） |
| `spmm_csc_f32_int_non_non_row` | 仍调用 Triton | `spmm_csc` 没有昇腾分支 | 同上 |
| `spmm_bsr_f32_int_non_non_row` | 仍调用 Triton | `spmm_bsr` 没有昇腾分支 | 同上 |
| `spmm_bell_f32_int_non_non_row` | 仍调用 Triton | `spmm_bell` 没有昇腾分支 | 同上 |
| `spmv_csr_c32_int_conj` | **结果错误**（误差 20.3） | `flagsparse_spmv_csr` 原有的昇腾回退（`spmv_csr.py`，2026-09-11 提交）把共轭转置当普通转置算，漏了 `conj` | 回退里对 conj 先取 `data.conj()`；`spmm_csr` 的昇腾回退有同样问题（影响 `spmm_csr_c32_int_conj_non_row`，清单第 54 个） |

模拟**测不到**的：torch_npu 是否真的支持这些 torch 操作。上机要特别留意：
- int8 的 `index_add_`、`index_select`（`spmv_*_i8i32`、`spmm_csr_i8i32`、`spvv_i8i32`）；
- float16 的 `index_add_`（f16 各变体）；
- 复数：新代码已按实部 / 虚部拆开，只依赖实数 `index_add_`。

## 3. runner 在昇腾上的两个缺口

- **精度**：gather、scatter、spmv_csr、spmm_csr、sddmm_csr 这 5 个算子在昇腾上的精度走
  `benchmark/benchmark_ascend_accuracy.py`，不跑 pytest，**所以这 5 个算子下的 20 个 q4 变体，runner 在昇腾上测不到**。
  上机时直接跑：
  ```bash
  FLAGSPARSE_BACKEND=ascend PYTHONPATH=src python3 -m pytest tests/pytest/test_q4_variants_accuracy.py -v
  ```
- **性能**：已有算子在昇腾上用昇腾专用 benchmark 脚本，不带 `--q4-variants`，所以产出不了这 29 个变体的性能行。
  3 个新算子（axpby、spvv、spmv_sell）在昇腾上用的是和 CUDA 相同的脚本，可以出数据。

## 4. 已知的其他问题

- gather / scatter 的昇腾补丁里有**死代码**（第二个 `_is_ascend_runtime()` 分支已无条件返回，第三个永远走不到）。
- **计时不一致**：scatter 的快速路径在 `start_time` 前没有 `synchronize()`，gather 有。修了会改变昇腾报出的数字，需要负责人拍板。
- SDDMM 的昇腾回退旧版会物化 `n_rows x n_cols` 稠密矩阵；现在按 262144 个非零分块计算，大矩阵不再 OOM。

## 5. 待确认（上机后回报）

1. `tools/q4_ascend_dispatch_check.py` 列出的 5 个 Triton 变体，在 910B 上能否编译、结果是否正确。
2. int8 / float16 的 `index_add_`、`index_select` 在 torch_npu 上是否可用、结果是否正确。
3. 第 3 节那条 pytest 命令的完整结果。
