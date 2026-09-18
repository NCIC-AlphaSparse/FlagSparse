# 海光 DCU / ROCm —— 改动台账

**实机环境**：gfx936（2026-08 实测）。
**基于版本**：上游 FlagSparse。
**最近回传**：2026-08。

跑法、诊断优先的排查顺序、已知限制见 [`docs/DCU.md`](../docs/DCU.md)；
C API 那层见 [`capi/docs/DCU.md`](../capi/docs/DCU.md)。

---

## 1. 库内的后端分支（11 个文件，共 37 处，是各后端里最多的）

| 文件 | 处数 | 改动性质 |
|---|---:|---|
| `sparse_operations/spsv.py` | 19 | 路由与 knob：`solve_kind == "csr_cw" and _is_rocm_runtime()` → `worker_count_use = 1` 等 |
| `sparse_operations/spmm_csr.py` | 4 | 启动参数里的 `"is_hip"` 标志、launch overrides |
| `sparse_operations/gather_scatter.py` | 4 | 厂商基线（hipSPARSE）分发 |
| `sparse_operations/spmv_csr.py` | 2 | **`_spmv_csr_default_backend()` 返回 `rowpar`** |
| `sparse_operations/spgemm_csr.py` | 2 | 基线分发 |
| `spsm.py` / `spmv_bsr.py` / `spmm_coo.py` / `spmm_bsr.py` / `spmm_bell.py` / `sddmm_csr.py` | 各 1 | 厂商基线可用性判断 |

**SpMV CSR 是唯一按后端切换内核实现的算子**，其余算子两个后端共用内核体：

| 运行时 | 默认内核 | 策略 |
|---|---|---|
| CUDA | `_spmv_csr_segbin_kernel` | 按 nnz 均匀切块 + 二分定位行 + `associative_scan` |
| **DCU/ROCm** | `_spmv_csr_real_kernel` | **一行一 program + 行内分段循环** |

两条路都是后端中立的通用 Triton 代码，两个后端都能跑，**切换只是默认值不同**
（`FLAGSPARSE_SPMV_CSR_KERNEL=segbin|rowpar` 可强制）。segbin 未在 DCU 上调过参
（`BLOCK` 固定 256），rowpar 未在 CUDA 上调过参。

`use_opt=True` 的 bucket 路径另有一套设备属性调优：HIP 上换用
`_SPMV_OPT_BUCKET_CONFIGS_HIP*` 分档，`num_warps` 上限压到 8、`block_size` 上限压到 512。
**CUDA 上实测为恒等变换。**

## 2. 已定位但**不在本仓库修复范围**的两个问题

**SpGEMM 在超大矩阵上触发 rocSPARSE 的显存非法访问（VMFault）。**
`mip1.mtx`（66463²、nnz≈1035 万、A_EQUALS_B 自乘）跑到参考实现阶段进程被 `SIGABRT` 打死：

```
Invalid address access ... >>>>>>>> KERNEL VMFault !!!! <<<<<<
kernel name: _ZL23csrgemm_fill_wf_per_row...
```

故障内核是 **rocSPARSE 内部的 SpGEMM 填充内核，不是 FlagSparse 的 Triton 内核**，
排查时不要往 Triton 侧找。GPU 页错误是 `SIGABRT`，Python 的 `except BaseException` 拦不住。

已做的**规避**（这部分是本仓库的改动）：`tests/test_spgemm.py` 改成逐矩阵 flush + fsync
写 CSV，崩溃前完成的矩阵结果得以保留；同时写 `<csv>.inflight.json` 记录正在处理的项，
正常跑完才删除 —— 崩溃后 `last_completed` 的**下一个**矩阵即为触发者。

**SpSV / SpSM 在 DCU 上 GPU 内核死锁。** Python 层正常返回，hang 在同步调用，16×16 跑 15
分钟也不结束。根因是内核里跨 program 的裸自旋等待：

```python
while ready == 0:
    ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
```

消费者 program 占住 CU，生产者排不进去，flag 永远不会被置位。
**`worker_count_use = 1` 那个串行保护并没有真正规避掉它。** 这是内核层固有问题，与
hipSPARSE 参考层无关。影响 `tests/pytest` 1836 个用例中的 851 个，跑套件时需 `--ignore`
掉那四个文件。

> 同一故障模式在 MetaX C550 上也出现（见 [`MACA.md`](MACA.md) §3 第 2 条）。
> 两块不同硬件落在同一个自旋结构上 —— 指向**内核写法**，不是某家的驱动。

## 3. 已知限制（不是 bug，不用查）

- **hipSPARSE 的 SpMM 入口只支持非转置**，CSR SpMM 的 `op=trans` / `op=conj` 直接跳过并给出
  原因。COO SpMM 不受影响（op 在调用前已物化）。
- **fp16 / bf16 没有厂商基线**：CuPy 和 hipSPARSE 的稀疏矩阵都不支持，两个后端上这一列
  都是 `N/A`，回落 `torch.sparse` 参考。CUDA 上就是如此。
