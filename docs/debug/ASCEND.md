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

## 2. q4 45 个变体：在 CUDA 上模拟昇腾分支的结果

复现：`PYTHONPATH=src python3 tools/q4_ascend_dispatch_check.py`（CUDA 机器上跑，强制所有模块走昇腾分支、禁止 Triton 启动）。

> ⚠️ 2026-10-05 更新：q4 清单从 42 条改成了 45 条（删 3 加 6，见 [README.md](README.md) 第 0 节）。
> 下面是按新清单重跑的结果；旧文档里记的"36/42、1 个结果错误"是旧清单的数字，已经过时。
> `spmv_csr_c32_int_conj` 当时发现的 conj 漏算问题**已经修了**（`_common.py` 新增
> `_index_add_values`/`_gather_values`，`spmv_csr.py`/`spmm_csr.py` 的昇腾回退都改成
> `data.conj() if op_code == SPMV_OP_CONJ_TRANS else data`），这次重跑路由和取值都是对的
> （用"在 CUDA 上 monkeypatch `_is_ascend_runtime()`、拿 CPU fp64 oracle 对照"的方式验证过，
> 不是真机，但比单纯看路由更接近"结果对不对"）。`spmm_bsr`/`spmm_bell` 不再统计在内（已移出
> q4 范围，但它们本来就没有昇腾分支，这个缺口依然存在，只是不再算 q4 的账）。

**39 / 45 走 torch 路径。** 其余 6 个仍会调用 Triton（在 910B 上会直接编译失败）：

| 变体 | 情况 | 原因 | 建议 |
|---|---|---|---|
| `spmv_csc_f32_int_non`、`spmv_csc_c32_int_non` | 仍调用 Triton（`_spmv_csc_non_*_kernel`） | `spmv_csc` 没有昇腾分支（原有算子） | 上机确认能否编译；不能就补 torch_npu 路径（CSC 的 op=non 相当于按列 `index_add_`） |
| `spmm_csc_f32_int_non_non_row`、`spmm_csc_c32_int_non_non_row`、`spmm_csc_f16_int_non_non_row` | 仍调用 Triton | `spmm_csc` 没有昇腾分支；后两个（c32/f16）是这次 Q4list 同步新增的变体，一上来就带着这个老缺口 | 同上 |
| `spgemm_csr_f32_int_non_non` | 仍调用 Triton | **这次新发现的缺口**：`spgemm_csr` 没有昇腾分支，这是 Q4list 同步新增的变体，之前的 36/42 统计里没有这一项，没人专门记录过 | 需要设计 torch_npu 回退（或确认 `torch.sparse` 的 CSR×CSR 乘法在 NPU 上是否可用），目前完全没有人做这件事 |

`spmm_bsr_f32_int_non_non_row`、`spmm_bell_f32_int_non_non_row` 这两个老缺口（同样没有昇腾分支）
已经不在 q4 的 45 条统计范围内了，但代码层面的问题没有变化，仍然会在 910B 上编译失败。

模拟**测不到**的：torch_npu 是否真的支持这些 torch 操作，以及新增 6 个变体里走 torch 路径的那些
（`sddmm_csr_f16/c32_int_non_non_row`、`spmm_coo_i8i32_int_non_non_row`）在真机上数值是否正确——
这 3 个的 Ascend fallback 代码已经写了（`mixed_spmx.py`/`sddmm_csr.py` 的改动），路由上能走到
torch，但**从未在 910B 真机上跑过**。上机要特别留意：
- int8 的 `index_add_`、`index_select`（`spmv_*_i8i32`、`spmm_csr_i8i32`、`spvv_i8i32`，以及新增的
  `spmm_coo_i8i32`）；
- float16 的 `index_add_`（f16 各变体，以及新增的 `sddmm_csr_f16`）；
- 复数：新代码已按实部 / 虚部拆开，只依赖实数 `index_add_`（新增的 `sddmm_csr_c32` 同样是这个套路）。

## 3. runner 在昇腾上的两个缺口

> ⚠️ 下面两个数字按新的 45 条清单重新数过（2026-10-05）：gather/scatter/spmv_csr/spmm_csr/
> sddmm_csr 这 5 个算子现在合计 **21** 个 q4 变体（`sddmm_csr` 因新增 f16/c32 从 3 个变成 5 个，
> 其余 4 个算子不变）；`axpby`/`spvv`/`spmv_sell` 合计 **8** 个（没有变化，这次 Q4list 同步没碰
> 这 3 个算子）。旧文档写的"20 个"/"29 个"是旧 42 条清单下的数字，"29" 具体怎么来的已经核对不上
> （42 减 8 应该是 34，不是 29），不要再引用旧数字，以这次重新统计的为准。

- **精度**：gather、scatter、spmv_csr、spmm_csr、sddmm_csr 这 5 个算子在昇腾上的精度走
  `benchmark/benchmark_ascend_accuracy.py`，不跑 pytest，**所以这 5 个算子下的 21 个 q4 变体
  （包括新增的 `sddmm_csr_f16_int_non_non_row`、`sddmm_csr_c32_int_non_non_row`），runner 在昇腾
  上测不到**。上机时直接跑：
  ```bash
  FLAGSPARSE_BACKEND=ascend PYTHONPATH=src python3 -m pytest tests/pytest/test_q4_variants_accuracy.py -v
  ```
- **性能**：已有算子在昇腾上用昇腾专用 benchmark 脚本，不带 `--q4-variants`，所以产出不了剩下
  37 个变体（45 减去 axpby/spvv/spmv_sell 的 8 个）的性能行。这 3 个新算子在昇腾上用的是和
  CUDA 相同的脚本，可以出数据；但新增的 6 个变体里属于其他算子族的那 5 个
  （`sddmm_csr_f16/c32`、`spmm_csc_c32/f16`、`spmm_coo_i8i32`，`spgemm_csr_f32` 另有第 2 节的
  Triton 缺口问题）同样拿不到性能数据。

## 4. 已知的其他问题

- gather / scatter 的昇腾补丁里有**死代码**（第二个 `_is_ascend_runtime()` 分支已无条件返回，第三个永远走不到）。
- **计时不一致**：scatter 的快速路径在 `start_time` 前没有 `synchronize()`，gather 有。修了会改变昇腾报出的数字，需要负责人拍板。
- SDDMM 的昇腾回退旧版会物化 `n_rows x n_cols` 稠密矩阵；现在按 262144 个非零分块计算，大矩阵不再 OOM。

## 5. 待确认（上机后回报）

1. `tools/q4_ascend_dispatch_check.py` 列出的 6 个 Triton 变体（含新发现的 `spgemm_csr_f32_int_non_non`），在 910B 上能否编译、结果是否正确。
2. int8 / float16 的 `index_add_`、`index_select` 在 torch_npu 上是否可用、结果是否正确——**包括新增的
   `sddmm_csr_f16_int_non_non_row`、`spmm_coo_i8i32_int_non_non_row`**，这两个在 CUDA 上验证过，
   从未在 910B 上跑过。
3. 新增的 `sddmm_csr_c32_int_non_non_row`（复数 real/imag 拆分）在 torch_npu 上是否可用。
4. 第 3 节那条 pytest 命令的完整结果（现在应该是 45 个变体的精度，不是 42 个）。
