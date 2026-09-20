# 海光 DCU / ROCm —— 改动台账

**实机环境**：BW200, ROCm torch `2.4.1+das.opt2.dtk2504` (`torch.version.hip=6.1.25065`), `hip-python=7.2.2.562.43`, SciPy 1.15.2（2026-09-19 实测）；此前 gfx936 结论见各节。
**基于版本**：`e7d96b12566dbce90b4c99fbb3883b31bd5d9158`。
**最近回传**：2026-09-19：修复 hip-python 7.x 的 hipSPARSE 性能基线 descriptor 签名兼容，并将 SpGEMM f32/f64 性能 sweep 隔离；40 变体精度+30 矩阵性能交付任务在后台运行中（总限时 3 小时）。

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

## 4. 2026-09 BW200：hip-python 7.x descriptor 返回签名

| 文件 | 锚点 | 改动 | 为什么 | 其他后端 |
|---|---|---|---|---|
| `src/flagsparse/sparse_operations/_common.py` | `_hipsparse_create_{csr,csc,coo}_descriptor`；`_prepare_spmv_{ref,coo}_hipsparse` | 先尝试 hip-python 7.x 的直接返回 descriptor 签名，再回落原有 out-parameter 签名；调用方使用返回的 descriptor。 | 7.x 的 `hipsparseCreateCsr` 只接受 10 个参数，旧调用传入 out-parameter 后报 `TypeError`，性能基线静默显示 `N/A`。 | 仅 ROCm/hipSPARSE 路径；旧绑定仍走原签名。 |
| `src/flagsparse/sparse_operations/spmm_csr.py` | `_prepare_spmm_ref_hipsparse` | 保存直接返回的 CSR/CSC sparse descriptor。 | 否则 SpMM CSR 将空的旧 out-parameter 对象传给 `hipsparseSpMM_bufferSize`，返回 `INVALID_VALUE`。 | 仅 hipSPARSE 路径。 |
| `src/flagsparse/sparse_operations/spmm_coo.py` | `_prepare_spmm_coo_ref_hipsparse` | 保存直接返回的 COO sparse descriptor。 | hip-python 7.x 的 `hipsparseCreateCoo` 不再通过 out-parameter 写回。 | 仅 hipSPARSE 路径。 |
| `src/flagsparse/sparse_operations/sddmm_csr.py` | `_prepare_sddmm_csr_ref_hipsparse` | 保存直接返回的 CSR sparse descriptor。 | 防止后续 SDDMM 调用使用空的旧 descriptor。 | 仅 hipSPARSE 路径；本轮未实测 SDDMM。 |
| `src/flagsparse/sparse_operations/spsv.py` | `_prepare_spsv_csr_ref_hipsparse` | 保存直接返回的 CSR sparse descriptor。 | 防止后续 SpSV 属性设置/调用使用空的旧 descriptor。 | 仅 hipSPARSE 路径；本轮未实测 SpSV，且 DCU Triton SpSV 已知可能死锁。 |
| `run_flagsparse_pytest.py` | `_run_spgemm_split_dtypes`；`run_performance` | SpGEMM f32/f64 以独立子进程、独立 CSV 执行，最后合并为原有 `performance.csv`。 | float32 上 rocSPARSE `SIGABRT` 会中断原本同进程内的 f64 sweep，交付报告只能标 f64 为 `NotFound`。 | 仅 `spgemm_csr` 的通用 benchmark 后端；其他算子和专用后端不变。 |
| `tests/test_spgemm.py` | `run_all_dtypes_export_csv`；CLI `--dtypes` | 允许 CSV sweep 只运行指定的 float32 或 float64。 | 供 runner 每个 dtype 启动单独进程。 | 无后端专属行为；未指定参数时仍运行 f32+f64。 |
| `tests/ci/test_runner_result_format.py` | `test_spgemm_performance_isolates_dtype_crashes` | 验证 f32 子进程异常退出后仍执行并保留 f64 结果。 | 防止回归到单一 sweep 导致 f64 被崩溃吞掉。 | 无 GPU、跨后端策略测试。 |

**实测**：`diagnose_hipsparse_ref.py` 的 SpMV CSR/COO、SpMM CSR/COO、gather、scatter 全部到执行和销毁完成；真实 `c8_mat11.mtx` 上 SpMV CSV 已写入 hipSPARSE `0.3933 ms`，SpMM CSR CSV 已写入 hipSPARSE `0.2557 ms`。同一 SpMM 输入的 hipSPARSE 与 PyTorch 输出连续 3 次 `max_abs=0`。SpGEMM 隔离逻辑已由无 GPU 单测覆盖；现有后台全量任务在该改动前已完成 SpGEMM，需后续单独重跑 SpGEMM 才能补 f64 结果。

## 5. 合并记录（2026-09-19，合入 `e7d96b1` 之上）

回传了 3 个文件（`run_flagsparse_pytest.py`、`tests/test_spgemm.py`、`tests/ci/test_runner_result_format.py`），
全部合入，基线与仓库一致（三个文件与 `e7d96b1` 逐字节相同，不存在旧副本覆盖）。

| 合并时的改动 | 说明 |
|---|---|
| `run_flagsparse_pytest.py`：拆分判据放宽 | 原为 `backend in GENERIC_BENCHMARK_BACKENDS`，未导出 `FLAGSPARSE_BACKEND` 时（CUDA 上常见）会退回单进程；改为 `not backend or backend in ...` |
| `tests/ci/test_runner_result_format.py`：格式 | docstring 与内嵌函数间缺一个空行，`ruff format --check` 拒绝，`make ci` 在 format-check 即停（见 `prompt.md` 第 6 节） |

**CUDA 实测**（RTX 5090，`--ops spgemm_csr --phase performance --delivery-only`）：起了 2 个子进程，
产出 `performance_float32.csv` / `performance_float64.csv`，合并后的 `performance.csv` 两种 dtype 各 3 行，
`spgemm_csr_f32/f64` 两个变体都报出加速比。新增的 CI 用例在改动前的 runner 上报 `AttributeError`，
说明它确实能挡住回归。`make ci` 全链通过：103 passed / 3 skipped。

**第二批（同日）**：hipSPARSE 7.x descriptor 兼容的 5 个文件已回传并合入（`_common.py`、
`spmm_csr.py`、`spmm_coo.py`、`sddmm_csr.py`、`spsv.py`），基线同样与 `e7d96b1` 一致。
`_common.py` 的三个 `_hipsparse_create_*_descriptor` 先按 7.x 的"直接返回 descriptor"签名调用，
`TypeError` 时回落旧的 out-parameter 形式；6 个调用点都改成"返回值非 None 才替换 descriptor"。

**为什么这对旧绑定是安全的**：`_hip_check_result()` 只在返回值是二元组时给出 payload，旧绑定返回
纯状态码，payload 为 `None`，调用点因此保留自己创建的 descriptor——不会把 descriptor 换成状态码。
`spgemm_csr.py` 的 `_hipsparse_create_csr_descriptor_from_tensors()` 结尾本来就是 `... or spmat`，
两种绑定都正确，不需要改；全仓库 6 个调用点已全部覆盖，无遗漏。

**合并时新增**：`tests/ci/test_hipsparse_descriptor_bindings.py`——用假的 hip-python 模块分别模拟
7.x 和旧绑定，断言前者拿到 descriptor 且不写 out-parameter、后者返回 `None` 且写 out-parameter。
无需 GPU 或 hip-python；在合并前的代码上会以 `TypeError: expected 10 arguments, got 11` 失败。
`make ci` 全链通过：109 passed / 3 skipped。

**本机能验到哪**：CUDA 机器上 `hipsparse is None`，这些分支不会执行，所以只验证了签名兼容逻辑
（假模块）和非 ROCm 路径无回归。**7.x 上的真实加速比数字仍要 DCU 实机复核**。

**另记**：`commands` / `failed_dtypes` / `timed_out_dtypes` 只进 `summary_flat.json`，
不进逐算子的 `performance_detail.json`（后者字段是 `_phase_detail_for_file()` 的白名单）。
崩溃时想知道是哪个 dtype 挂的，看 `summary_flat.json` 或 `performance_stderr.log`（按 dtype 分段）。
