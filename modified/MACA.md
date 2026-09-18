# 沐曦 MetaX / MACA 本地源码改动复现台账

**实机环境**：MetaX C550（`warp_size=64`、104 MP、64 GB），MACA SDK 3.8.2.6，
torch `2.10.0+metax3.8.1.0`，Triton `3.6.0+metax3.8.1.0`，Python 3.12。
**基线版本**：`87b0aff`（`2026-09-18` 拉取的 `origin/main`）。行号均以该版本为准；
移植时以“锚点”定位，不能只依赖行号。
**本轮状态**：本轮没有保留新的源代码行为改动。下面逐项记录当前本机保留的 MetaX 分支和
测试改动，以便在另一份上游副本中复现。曾短暂试验的 SpSV CPU SciPy fallback 已撤销，
不属于可移植代码。

运行方法、已测结果和未解决缺陷见 [`docs/MACA.md`](../docs/MACA.md)；C API 那层见
[`capi/docs/MACA.md`](../capi/docs/MACA.md)。

---

## 1. 源码修改总表

| 文件 | 当前行号 / 锚点 | 修改 | 触发现象与理由 | 其他后端 |
| --- | --- | --- | --- | --- |
| `src/flagsparse/sparse_operations/spsv.py` | 5997-6023，`_MACA_SPSV_PROFILES` / `_maca_spsv_knob()` / `_spsv_route_warp_size()` | C550 profile 提供 `ALG3/ALG4` 的 `warp_size=64`，并将其接入共有 SpSV 路由。 | 实机 `torch.cuda.get_device_properties(0).warp_size == 64`；CUDA 默认 32 会让 warp-sized launch 参数错误。 | `_is_maca_runtime()` 外恒等于原来的 32。 |
| 同上 | 2014-2017，`_spsv_auto_route_non_trans()` | `enable_advanced_auto` 可由 C550 profile 禁用时强制 `csr_cw`。 | 允许将 C550 从尚未验证的高级自动路由退回保守 ALG1，以单 profile knob 复现 A/B。当前值为 `True`，不改变默认选择。 | 仅 MetaX guard 生效。 |
| 同上 | 6219-6234，`_spsv_smblk_use_persistent()` | 允许 C550 profile 决定 ALG4 是否走 persistent worker。 | 保留 `FLAGSPARSE_SPSV_SMBLK_KERNEL=rowprog|persistent` 覆盖，便于单独测量；当前 profile 为 `False`。 | 非 MetaX 仍返回原逻辑。 |
| 同上 | 7316-7322，`_execute_spsv_csr_plan()` 内 worker 数选择 | 仅在 profile 的 `cw_serial=True` 时将 C550 `csr_cw` 设为单 worker。 | 用于规避 ready-flag polling 的试验开关；当前 `False`，不隐式改变执行。 | ROCm 保持独立 serial 分支；其他后端不变。 |
| `src/flagsparse/sparse_operations/spmv_csr.py` | 978-1019，`_MACA_SPMV_PROFILES` / `_maca_spmv_knob()` / `_spmv_csr_default_backend()` | 按型号选择 C550 默认 CSR SpMV family：`segbin`。 | MACA 的 CUDA 兼容路径从 CUDA `segbin` 起步；与 DCU 的 `rowpar` 分开，避免把 gfx936 调优值误用于 C550。 | `FLAGSPARSE_SPMV_CSR_KERNEL` 仍可覆盖；非 MetaX 原路径不变。 |
| `src/flagsparse/sparse_operations/spmm_coo.py` | 1188-1190、1234-1249，`_MACA_SPMM_COO_COMPLEX_BLOCK_NNZ` / `_resolve_spmm_coo_launch_config()` | MetaX complex COO SpMM 将 `BLOCK_NNZ` 上限钳为 4。 | `tl.static_range` 对 `BLOCK_NNZ` 展开；C550 4 KB/thread 私有内存上限下 complex、256 会申请约 8 KB 并被拒绝。实测 4 可启动、通过正确性，并是 30 矩阵扫描最优值。 | 条件同时要求 MetaX、complex dtype、调用值大于 4；实数及所有其他后端不变。 |
| `src/flagsparse/sparse_operations/spmm_csr.py` | 589-610、1895-1914、3035-3052 | 三个 launch-property 标准化函数将 `is_maca` 传给调优层，并通过 `_spmm_is_maca_device()` 暴露。 | MACA 不能只依赖 `torch.version.hip` 区分，必须让调优层感知 MetaX runtime。 | 新字段仅供 MetaX 分支读取；非 MetaX 与原 device property 相同。 |
| 同上 | 5171-5191，`_pytorch_sparse_reference()` 路径 | MetaX 上构造 PyTorch 稀疏 reference 时优先走 `_pytorch_sparse_mm()`，CSR 失败时回退 COO。 | MACA fp32 CSR + int64 index 会产生非有限结果；参考结果不可靠会把正确内核误判失败。 | `_is_maca_runtime()` 守卫；其余后端维持原 CSR 构造和 fallback。 |

### 1.1 SpSV profile 的复现

文件：`src/flagsparse/sparse_operations/spsv.py`。

```python
_MACA_SPSV_PROFILES = {
    "c550": {
        "smblk_persistent": False,
        "enable_advanced_auto": True,
        "alg3_warp_size": 64,
        "alg4_warp_size": 64,
        "cw_serial": False,
    },
}
```

复现步骤：

1. 从 `_common` 导入并使用 `_is_maca_runtime()`、`_maca_device_model()`；不得直接以
   `torch.version.cuda` 判断，因为 MACA 兼容 CUDA。
2. 为 `c550` 配置加入上段 profile，未知型号回落 `_MACA_SPSV_DEFAULT_PROFILE`。
3. 在 ALG3/ALG4 warp 参数处调用 `_spsv_route_warp_size(kind)`，而不是写死 64。
4. 保留两个环境变量 A/B 入口：`FLAGSPARSE_SPSV_SMBLK_KERNEL` 和已有路由开关。

**验证依据**：仅 `warp_size=64` 是 C550 设备指纹实测值；`smblk_persistent`、
`enable_advanced_auto`、`cw_serial` 是 CUDA 起点，不得在没有重新测量时宣传为最优值。

**已知限制**：`csr_cw` 在 C550 上 lower 会非法访存、upper 有依赖时可死锁。profile 本身没有
修复该内核，三角类 benchmark 必须用 `timeout -s KILL`。设备 kernel 未可靠运行的变体应保持
`NotFound` / `Timeout` / `Error` 的真实状态，不能用 CPU 求解替代。

### 1.2 SpMV CSR profile 的复现

文件：`src/flagsparse/sparse_operations/spmv_csr.py`，锚点
`_spmv_csr_default_backend()`（当前 997-1019 行）。

```python
if _is_rocm_runtime():
    return "rowpar"
if _is_maca_runtime():
    return _maca_spmv_knob("csr_kernel")
```

`_MACA_SPMV_PROFILES["c550"]["csr_kernel"] = "segbin"`。复现时先确认
`FLAGSPARSE_SPMV_CSR_KERNEL` 为空；该环境变量优先级更高，适合在 30 个真实矩阵上测
`segbin` 与 `rowpar`，不能用单个小矩阵替代调优结论。

### 1.3 Complex COO SpMM private-memory 限制的复现

文件：`src/flagsparse/sparse_operations/spmm_coo.py`，锚点
`_resolve_spmm_coo_launch_config()`（当前 1193-1252 行）。

```python
if (
    _is_maca_runtime()
    and value_dtype is not None
    and _is_complex_dtype(value_dtype)
    and block_nnz > _MACA_SPMM_COO_COMPLEX_BLOCK_NNZ
):
    block_nnz = _MACA_SPMM_COO_COMPLEX_BLOCK_NNZ
```

触发条件是 complex dtype 且 caller 请求的 `block_nnz > 4`。不能把公开函数默认参数盲改成
4：当前实现仅在 C550 complex 情况裁剪，避免改变其他后端和实数任务。诊断报错为
`memory size or pointer value too large to fit in 32 bit`；如需提升上限，需要平台侧修改
`metax.ko` 的 `pri_mem_sz`，那是系统变更，不属于本仓库源码修改。

### 1.4 SpMM CSR reference 与 launch metadata 的复现

文件：`src/flagsparse/sparse_operations/spmm_csr.py`。

三处 property normalizer（当前 589、1895、3035 行）都应添加同一种字段：

```python
"is_hip": _is_rocm_runtime(),
"is_maca": _is_maca_runtime(),
```

不要从 `torch.version.hip` 派生 `is_maca`，否则 `FLAGSPARSE_BACKEND=metax` 的显式覆盖不能传到
调优层。reference 路径（当前 5171-5191 行）必须将 C550 fp32 + int64 CSR 异常隔离在
`_is_maca_runtime()` 内：先调用共享 `_pytorch_sparse_mm()` 选择 int32 CSR 或 COO，再仅在异常时
构造 COO 重试。这里是**精度参考路径**，不是 FlagSparse Triton kernel 的执行 fallback。

---

## 2. 测试与 benchmark 修改

| 文件 | 当前行号 / 锚点 | 修改 | 原因 | 其他后端 |
| --- | --- | --- | --- | --- |
| `tests/test_spgemm.py` | 788、1539-1555、2360 | 传入 `isolate_matrices=_is_maca_runtime()`；开启时每个 MatrixMarket 文件进入独立子进程。 | SpGEMM 某个矩阵 fault 后会污染 MACA runtime，后续矩阵结果不能信任。 | MetaX 才隔离；其他后端仍在同一进程批量执行。 |
| `tests/test_spmv.py` | 314-340、351-370 | C550 的 PyTorch baseline/reference 通过共享 helper 选择 int32 CSR 或 COO，并在计时前实体化 transpose operand。 | MACA fp32 CSR + int64 index 不稳定，避免 PyTorch reference 产生非有限值或因格式不同而测错对象。 | 守卫为 `_is_maca_runtime()`。 |
| `tests/pytest/test_spsv_csr_accuracy.py` | 45、184-201、1450、1484、1589、1625 | CuPy 可用时保留 cuSPARSE oracle；非 CUDA（含 MetaX）时使用 SciPy triangular solve oracle。 | CuPy 只服务 NVIDIA cuSPARSE；此前对应专项测试在非 CUDA 平台被跳过。 | 这是一项跨后端测试修复；CUDA 有 CuPy 时输出不变，非 CUDA 新增 CPU reference。 |

> **归属订正（合并时加，2026-09-18）**：上表 `tests/pytest/test_spsv_csr_accuracy.py` 那一行不是
> MACA 这一轮的改动，是 MUSA 回传、已随 `ad474d0` 合入的（见 `modified/MUSA.md` 第 8 节）。
> 内容描述无误，C550 上直接受益，移植时不需要重复做。

### 2.1 SpSV 精度 oracle 的边界

`_cusparse_or_scipy_ref_spsv()` 在当前 184-201 行。其含义是：

```python
x = flagsparse_spsv_csr(data, indices, indptr, b, shape)  # 输入张量所在设备执行
x_ref = scipy_triangular_solve(A, b, ...)     # 仅 CPU 参考值
assert torch.allclose(x.to(x_ref.device), x_ref, ...)
```

这不是将 `flagsparse_spsv_csr` 改写为 SciPy。C550 SpSV 若崩溃/挂死，测试应按真实错误或 timeout
记录；不能由 oracle 的存在推导“SpSV 已通过”。`scipy_triangular_solve()` 位于
`tests/pytest/accuracy_utils.py:217-235`，且输入/输出均在 CPU。

### 2.2 SpGEMM 独立进程复现

在 `tests/test_spgemm.py` 保持 worker 协议不变，只把
`isolate_matrices=ast_common._is_maca_runtime()` 传给两处 batch 入口（当前 788、2360 行）。
当该 flag 为真，`run_all_mtx()` 在当前 1539-1555 行调用 `_run_matrix_worker_subprocess()`；不要
对整套 benchmark 无条件增加 subprocess，否则会改变其他后端的时间和错误边界。

---

## 3. 已撤销、不得复现的改动

本轮曾在 `src/flagsparse/sparse_operations/spsv.py` 的
`_execute_spsv_csr_plan()` 前加入 `_maca_spsv_cpu_fallback()`：将 CSR 和 RHS 拷到 CPU，调用
`scipy.sparse.linalg.spsolve_triangular()`，再把结果传回 C550。该改动已经**完全删除**，原因如下：

1. 它将被测 GPU 算子替换成 CPU 计算，不能反映 C550 kernel 正确性；
2. 其耗时包含 CPU solve 与 host/device copy，不能作为 GPU 性能；
3. C550 没有厂商 baseline 时，只有“设备算子真实运行并通过”的性能行才可标记
   `NoBaseline`；设备 kernel 崩溃/挂死或未测时不能借 CPU fallback 填补。

因此移植本台账时，**不要**添加 `_maca_spsv_cpu_fallback()`，也不要为它修改
`test_spsv_csr_transpose_public_solve_uses_transpose_kernel()` 的 kernel-route 断言。

---

## 4. 复现前后的检查

```bash
export PYTHONPATH="$PWD/src"
export FLAGSPARSE_BACKEND=metax

python3 - <<'PY'
from flagsparse.sparse_operations import _common, spmv_csr, spsv
print(_common._backend_name())                 # metax
print(_common._maca_device_model())            # c550
print(spmv_csr._spmv_csr_default_backend())    # segbin
print(spsv._spsv_route_warp_size("alg3"))     # 64
print(spsv._spsv_route_warp_size("alg4"))     # 64
PY

python3 -m compileall -q src tests
git diff --check
```

性能交付必须显式指定 `/root/gcx/matrix`、warmup=5、iters=20。三角类启动命令必须外套
`timeout -s KILL`；本机没有可信的 C550 SpSV device 结果前，不应报告其 GPU 性能。

---

## 5. 30 矩阵性能 timeout 与交付范围收敛（已完成）

**状态：已完成（2026-09-18）。** 以下方案基于
`pytest_results_metax_delivery_perf_w5_i20_int32_non/` 中已中止日志制定，并已完成于
`pytest_results_metax_delivery_perf_remaining_w5_i20/`。命令仍可用于重新测量。

`run_flagsparse_pytest.py --timeout` 是**每个父算子每个阶段**的超时。此前设为 900 秒：

| 父算子 | 默认 CSV sweep | 900 秒实际进度 | 推断 |
| --- | --- | --- | --- |
| `sddmm_csr` | 2 dtype × 4 K × 30 = 240 组 | 59/240 | 不是单矩阵挂死；保留默认 K sweep 预计至少 3660 秒 |
| `spmm_csr` | 4 dtype × 2 index × 3 op × 30 = 720 组 | 252/720 | 默认 sweep 包含不属于 40 变体的 int64、transpose、conjugate 组合；预计至少 2570 秒 |

40 个交付变体只需要 `int32` 和 `non`。下一轮应收窄这两项，而不是仅将 timeout 从 900
秒盲目提高：

```bash
setsid env PYTHONPATH="$PWD/src" FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_VENDOR=none \
  timeout -s KILL 3600 python3 -u run_flagsparse_pytest.py \
  --phase performance --mode normal --delivery-only --gpus 0 --timeout 1200 \
  --ops spmm_csr,sddmm_csr \
  --benchmark-input /root/gcx/matrix --benchmark-warmup 5 --benchmark-iters 20 \
  --benchmark-args=--no-cusparse \
  --op-benchmark-args='spmm_csr=--dtypes float32,float64,complex64,complex128 --index-dtypes int32 --ops non' \
  --op-benchmark-args='sddmm_csr=--dtype float32,float64 --index-dtype int32 --k 64' \
  --results-dir pytest_results_metax_delivery_perf_remaining_w5_i20 \
  > pytest_results_metax_delivery_perf_remaining_w5_i20/runner.log 2>&1 < /dev/null &
echo $! > pytest_results_metax_delivery_perf_remaining_w5_i20/runner.pid
```

实际结果：`spmm_csr` 生成 120 条 CSV 记录（4 dtype × int32 × non × 30 矩阵），
`sddmm_csr` 生成 60 条（2 dtype × int32 × K=64 × 30 矩阵）；两项 parent performance 均为
`Passed`，且没有触发 1200 秒父任务超时或 3600 秒外层超时。`setsid` 与重定向保证退出
Codex/终端后进程继续运行。

| 交付变体 | 覆盖矩阵 | CSV `base/gems` 均值 |
| --- | ---: | ---: |
| `sddmm_csr_f32_int_non_non_row` | 30 | 7.45x |
| `sddmm_csr_f64_int_non_non_row` | 30 | 5.92x |
| `spmm_csr_f32_int_non_non_row` | 30 | 20.88x |
| `spmm_csr_f64_int_non_non_row` | 30 | 13.81x |
| `spmm_csr_c32_int_non_non_row` | 30 | 10.24x |
| `spmm_csr_c64_int_non_non_row` | 30 | 5.14x |

以上数值来自 `--no-cusparse` 的 CSV `base/gems` 字段，不是相对厂商 sparse library 的
speedup。完整结果见 `pytest_results_metax_delivery_perf_remaining_w5_i20/summary.json`。

若需要保留默认完整 CSV sweep，必须单独跑，且改用：`sddmm_csr --timeout 4500`、
`spmm_csr --timeout 3600`，外层 `timeout -s KILL` 应大于相应单项 timeout。不要在一个
7200 秒的总外层命令中串行执行两者，否则外层会先终止后一个任务。
