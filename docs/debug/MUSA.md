# 摩尔线程 MUSA（S5000）debug

`FLAGSPARSE_BACKEND=mthreads`。环境、交付复现见 [../MUSA.md](../MUSA.md)。

## 1. 这个后端的特点（实测于 torch_musa 2.7.1 / muDNN v3105）

| 能力 | fp32 | fp64 | complex64 | complex128 |
|---|---|---|---|---|
| Triton（含 `atomic_add`、`associative_scan`） | 可用 | 可用 | 可用 | 可用 |
| `where`（三元） | 可用 | 不可用 | 不可用 | 不可用 |
| `sum` | 可用 | 可用 | 不可用 | 不可用 |
| 2 维 × 1 维 `A @ x`（gemv） | 可用 | 不可用 | 不可用 | 不可用 |
| 2 维 × 2 维 `A @ B`（gemm） | 可用 | 可用 | 可用 | 可用 |
| `torch.sparse` 的矩阵乘（CSR 和 COO） | **不可用** | 不可用 | 不可用 | 不可用 |

- **复数高级索引没有 kernel**：`values[order]` 会报 `"IndexMusa" not implemented for 'ComplexFloat'`。
  已统一改走 `_common._gather_values`（按 `view_as_real` 拆成实数索引）。
- 因为 `torch.sparse` 不能做矩阵乘，MUSA 上没有 PyTorch 稀疏基线；精度测试的参考值在 CPU 上算。

## 2. 已知问题

| 问题 | 状态 |
|---|---|
| muDNN 的 gemv 不支持 fp64 / 复数（gemm 支持） | 厂商侧问题，可以作为最小复现报给厂商 |
| `torch.sparse` 无矩阵乘，`_mthreads_vendor_sparse_library()` 默认返回 `None` | 基线列为 N/A 并写明原因 |
| `index_add_` / `scatter_add_` 的复数版本同样可能缺 | spgemm / spsm 只支持实数，目前走不到 |

## 3. q4 42 个变体的风险点

- q4 的复数路径全部拆成实部 / 虚部平面，不用复数索引、复数 `sum`：`spvv_c32_int_conj`、`spmv_sell_c32` 等应当可用，需实测确认。
- **PyTorch 基线**：q4 benchmark 的 PyTorch 列用 `torch.sparse` 做矩阵乘，在 MUSA 上会失败，
  这一列会写 `pytorch_reason`；cuSPARSE 列也为空。于是这些行**没有任何加速比**，只能用
  `tools/baseline_bound.py --vendor-card musa-s5000`（1370 GB/s）按 H800 换算判定。
- 精度测试的参考值在 CPU 上算，不受上面的限制。

## 4. 2026-09-28/29 真机结果（S5000 x1）

本节是 q4 分支首次在 MUSA 真机上的结果，不是 CUDA 模拟结果。环境自检为
`FLAGSPARSE_BACKEND=mthreads`、加速器 `musa`、`fallback=None`；`mthreads-gmi 2.3.2`
报告驱动 `3.3.5-server`。测试开始和结束时 GPU 均无其他进程。

原始产物在 `results_musa_q4_20260928T145913Z/`：

- 精度记录：`q4_accuracy.log`、`q4_accuracy_result.json`；
- 性能汇总：`performance/summary.json`、`performance/summary.csv`；
- 每个父算子的原始性能 CSV、stdout、stderr 位于 `performance/<op>/`。

### 精度

执行命令：

```bash
FLAGSPARSE_BACKEND=mthreads PYTHONPATH=src \
  python3 -u -m pytest tests/pytest/test_q4_variants_accuracy.py -v \
  --record json --output results_musa_q4_20260928T145913Z/q4_accuracy_result.json
```

**85 passed, 1 warning, 68.90 s**。其中 1 项检查清单确有 42 个不重复变体，余下
84 项为每个变体的两个规模（`64x48x16`、`257x129x33`）。因此 **42/42 q4 变体在两个
规模上均与 CPU golden 一致**，包括全部 complex64、fp16 和 int8 路径。唯一 warning 是
PyTorch 的 CSR beta 提示，不影响结果。

### 性能

性能阶段使用 Python benchmark，目的只是记录 FlagSparse 的 MUSA 自身耗时；它不是
MUSA 交付报告的 muSPARSE 对比。命令如下，`tests/data` 中的 10 个 MatrixMarket 矩阵均
传入支持 q4 变体的矩阵算子；预热 5 次、计时 20 次：

```bash
FLAGSPARSE_BACKEND=mthreads PYTHONPATH=src \
  python3 -u run_flagsparse_pytest.py \
  --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_csc,spmm_bsr,spmm_bell,spmm_coo,sddmm_csr \
  --gpus 0 --phase performance --mode quick \
  --results-dir results_musa_q4_20260928T145913Z/performance \
  --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 --timeout 3600
```

汇总的 q4 状态为 **34 Passed、6 Failed、2 NotFound**。`Passed` 的 34 个变体在其
benchmark 覆盖的全部输入上都有有效耗时：

| 算子族 | 通过的 q4 变体 |
|---|---|
| 稀疏向量 | `gather_i8_int`、`scatter_i8_int`、`axpby_f16_int`、`spvv_f16f32_int_non`、`spvv_c32_int_conj`、`spvv_i8i32_int_non` |
| CSR SpMV | `spmv_csr_f16f32_int_non`、`spmv_csr_f16_int_non`、`spmv_csr_f32c32_int_non`、`spmv_csr_c32_int_non`、`spmv_csr_f32_int_trans`、`spmv_csr_i8f32_int_non`、`spmv_csr_i8i32_int_non`、`spmv_csr_c32_int_conj` |
| COO SpMV | `spmv_coo_f16f32_int_non`、`spmv_coo_f32_int_trans`、`spmv_coo_c32_int_non`、`spmv_coo_f16_int_non`、`spmv_coo_c32_int_conj`、`spmv_coo_i8i32_int_non` |
| CSC SpMV | `spmv_csc_f32_int_non`、`spmv_csc_f16_int_non` |
| CSR SpMM | `spmm_csr_f16f32_int_non_non_row`、`spmm_csr_f16_int_non_non_row`、`spmm_csr_f32_int_non_non_col`、`spmm_csr_f32_int_non_trans_row`、`spmm_csr_c32_int_non_non_row`、`spmm_csr_i8i32_int_non_non_row`、`spmm_csr_f32_int_trans_non_row` |
| COO SpMM | `spmm_coo_c32_int_non_non_row`、`spmm_coo_f16_int_non_non_row` |
| CSR SDDMM | `sddmm_csr_f32_int_non_non_col`、`sddmm_csr_f32_int_non_trans_row`、`sddmm_csr_f32_int_trans_non_row` |

异常和聚合限制如下。这里的 `Failed` 不都表示 FlagSparse 内核没有运行，必须结合原始
CSV 判断：

| q4 变体 | 汇总状态 | 原始结果 | 原因 / 处理 |
|---|---|---|---|
| `spmv_sell_f32_int_non` | Failed | 10/10 行 `PASS`，0.0892 / 0.2169 / 4.9845 ms（min / median / max） | 父脚本的非 q4 complex64 行失败，进程退出码为 1，runner 因此把同一父脚本的 q4 行投影为 Failed；该 q4 行本身可用。 |
| `spmv_sell_f16_int_non` | Failed | 10/10 行 `PASS`，0.0844 / 0.2039 / 5.1221 ms | 同上；该 q4 行本身可用。 |
| `spmv_sell_i8i32_int_non` | Failed | 10/10 行 `PASS`，0.0667 / 0.1957 / 4.9451 ms | 同上；该 q4 行本身可用。 |
| `spmv_sell_c32_int_non` | Failed | 无 q4 性能行 | 父脚本的 complex64 行均为 `ERROR`，尚未产出可用的 complex SELL 性能数据；精度两种规模均通过。 |
| `spmv_csc_c32_int_non` | Failed | 10/10 行 `ERROR` | `RuntimeError: "IndexMusa" not implemented for 'ComplexFloat'`；内核耗时前的 complex 高级索引/参考路径仍未完全避开。 |
| `spmm_bell_f32_int_non_non_row` | Failed | 9/10 行 `PASS`，3.5688 / 5.0523 / 14.2175 ms | `ASIC_680ks.mtx` 被 Blocked-ELL 容量保护拒绝：预计存储 398,703,808 个值；其余 9 个矩阵有有效耗时。 |
| `spmm_csc_f32_int_non_non_row` | NotFound | 无 q4 性能行 | benchmark 的 COO `torch.sparse.mm` 参考在 MUSA 抛 `aten::addmm` 未实现，父进程退出前未写入 f32 行。 |
| `spmm_bsr_f32_int_non_non_row` | NotFound | 无 q4 性能行 | 同样依赖 MUSA 未实现的 `torch.sparse.mm` COO 参考，未写入 f32 行。 |

所有 Python 性能行的 PyTorch / cuSPARSE 基线与加速比均为 `N/A`，这是 MUSA 的
`torch.sparse` 矩阵乘未实现导致的预期结果，并非性能为零。需要带 muSPARSE 基线的交付
性能报告仍应使用 `run_flagsparse_split_delivery.py`；它覆盖的是交付的 20 个变体，不覆盖
本节的全部 q4 42 变体。

## 5. 后续待确认

1. 修复或绕过 CSC SpMV complex64 的 `IndexMusa` 高级索引路径，然后重跑
   `spmv_csc_c32_int_non` 的 10 矩阵性能行。
2. 将 CSC / BSR SpMM benchmark 的正确性参考固定为 CPU/SciPy，避免 MUSA 上不可用的
   `torch.sparse.mm` 阻断自身内核计时。
3. 为 SELL complex64 benchmark 采用与精度路径一致的实部 / 虚部平面参考，再单独验证
   `spmv_sell_c32_int_non`。
4. `python3 tools/probe_accel_capabilities.py` 的最新输出（torch_musa 升级后上表可能变化）。
