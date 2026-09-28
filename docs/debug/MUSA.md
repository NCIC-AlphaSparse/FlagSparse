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

## 4. 待确认（上机后回报）

1. `tests/pytest/test_q4_variants_accuracy.py` 全部结果，重点是复数变体。
2. `python3 tools/probe_accel_capabilities.py` 的最新输出（torch_musa 升级后上表可能变化）。
3. runner 一轮完整结果目录。
