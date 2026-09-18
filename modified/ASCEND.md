# 昇腾 Ascend —— 改动台账

**实机环境**：Ascend 910B4-1 / CANN 9.0.0 / torch 2.9.0+cpu + torch-npu 2.9.0.post2（只用 NPU 6、7）。
**代码基线**：当前 checkout `87b0affbe6f9a3b352e84db31dd7297dc0cbac8e`。
**实机改动**：当前工作树相对该基线的源代码 diff 只有
`run_flagsparse_pytest.py`、`benchmark/benchmark_ascend.py`、
`src/flagsparse/sparse_operations/sddmm_csr.py` 和 `src/flagsparse/sparse_operations/spsv.py`；本文件同时记录
checkout 中已经存在的其他 Ascend 适配，便于在其他机器完整复现。

跑法见 [`docs/ASCEND.md`](../docs/ASCEND.md)；C API 那层见
[`capi/docs/ASCEND.md`](../capi/docs/ASCEND.md)。

---

## 1. 库内的后端分支（6 个文件，共 17 处）

| 文件 | 处数 | 锚点 | 改动 | 为什么 |
|---|---:|---|---|---|
| `sparse_operations/gather_scatter.py` | 7 | `_is_ascend_runtime()` 守卫的 fallback | gather/scatter 走 torch_npu 路径 | CANN 侧 lower 不出对应内核形状 |
| `sparse_operations/spmv_csr.py` | 3 | `_spmv_csr_default_backend()` 等 | 起点 `segbin`；**`tl.associative_scan` 在 Ascend 上 lower 不了**，相关路径退回 torch_npu | 与摩尔线程明确不同：MUSA 的 associative_scan 是通的，不需要这个回退 |
| `sparse_operations/spmm_csr.py` | 3 | 厂商基线选择 + dispatch | `vendor in ("ops_sparse","torch") and _is_ascend_runtime()` | |
| `sparse_operations/sddmm_csr.py` | 2 | 同上 | | |
| `sparse_operations/spsm.py` | 1 | **`SPSM_ASCEND_DISPATCH`** | `spsm_csr` / `spsm_coo` 的 Ascend NPU fallback 分发表 | CANN lower 不出那两种内核形状 |
| `sparse_operations/spmm_coo.py` | 1 | **`SPMM_COO_ASCEND_DISPATCH`** | `spmm_coo` 走 torch_npu，绕开 shmem 编译问题 | |

两张分发表都从 `sparse_operations/__init__.py` 和 `flagsparse/__init__.py` 导出
（`__all__` 里各两条）。

> **导出这件事要确认，不要假设。** 这一轮的 drop 曾用了未导出的 `_IS_ASCEND_RUNTIME`，
> 留下一个只在特定分支才触发的 NameError。`from ._common import *` 只解析 `__all__` 里的名字。

## 1.1 2026-09-17 实机修复

| 文件 | 锚点 | 改动 | 为什么 | 其他后端 |
|---|---|---|---|---|
| `run_flagsparse_pytest.py` | `_base_env()` / `_subprocess_device_id()` | Ascend 子进程设置 `ASCEND_RT_VISIBLE_DEVICES=<物理卡>`，并向专用 benchmark 传逻辑设备 `0` | `CUDA_VISIBLE_DEVICES` 对 torch_npu 无效；实测 pytest 默认 `npu` 会落到物理 NPU 0，即使 runner 分配的是 6 或 7 | `FLAGSPARSE_BACKEND=ascend` 守卫；其他后端保持原设备号和环境变量 |
| `src/flagsparse/sparse_operations/spsv.py` | `_spsv_ascend_row_sweep()` / `_execute_spsv_csr_plan()` | CSR/COO SpSV 在 Ascend 上改为逐行 torch_npu 非转置求解；转置显式拒绝 | 910B Triton lower `tl.atomic_add` 失败后编译器 SIGSEGV；实测 float32 CSR、COO 最小用例通过且不再崩溃，未验证转置不能返回错误数值 | `_is_ascend_runtime()` 守卫；其他后端保留原 Triton plan |

## 1.2 源代码逐段复现说明（相对 `87b0aff`）

以下行号以当前 checkout 为准；文件后续发生编辑时，应以函数名锚点为准。

### `run_flagsparse_pytest.py`

1. **物理卡隔离**（当前行 1442--1455，`_base_env`）。保留原有
   `CUDA_VISIBLE_DEVICES=<gpu_id>`，并在 `FLAGSPARSE_BACKEND=ascend` 时增加：

   ```python
   env["ASCEND_RT_VISIBLE_DEVICES"] = str(gpu_id)
   ```

   `torch_npu` 不读取 CUDA 的可见设备变量；不加此行时，默认
   `torch.device("npu")` 可能实际使用物理 NPU 0。这里传入的是物理卡号（6 或 7），
   子进程随后只看到这一张卡，并将它重编号为逻辑设备 0。条件分支只对 Ascend 生效，
   其他后端的环境不变。

2. **逻辑设备重编号**（当前行 1458--1462，`_subprocess_device_id`）：Ascend 返回
   `0`，其他后端返回原始 `gpu_id`。因此 `ASCEND_RT_VISIBLE_DEVICES=6` 时，子进程命令
   必须是 `--device 0`，不能继续传 `--device 6`。

3. **精度命令**（当前行 1555--1563，`run_accuracy` Ascend 分支）：
   `benchmark/benchmark_ascend_accuracy.py` 的 `--device` 改为
   `_subprocess_device_id(gpu_id)`，保证精度子进程与环境掩码一致。

4. **性能命令**（当前行 2341--2350，`run_performance`）：传给
   `render_performance_command` 的 `device` 同样改为
   `_subprocess_device_id(gpu_id)`，保证性能子进程不会访问物理 NPU 0。

5. **真实矩阵参数链路**（当前行 617--637，`ASCEND_PERFORMANCE_COMMANDS`）：五个
   Ascend 直接 benchmark 模板均增加：

   ```python
   "--input",
   "{input}",
   ```

   `render_performance_command()` 已经会用 runner 的 `benchmark_input` 替换 `{input}`。
   因此 `--benchmark-input /home/matrix` 最终变为
   `benchmark_ascend.py --input /home/matrix`；没有这两行时，runner 虽然解析目录，
   Ascend benchmark 实际仍使用默认合成 shape。该模板仅覆盖
   `ASCEND_BASELINE_OPS` 的 gather、scatter、SpMV CSR、SpMM CSR、SDDMM CSR；其余
   Ascend 算子仍按 capability probe 路由，不能伪称为真实矩阵性能测试。

6. **逐矩阵性能隔离与 Timeout 投影**（当前行约 1692--1695、2321--2345、
   2476--2495、2836--2858）：`ASCEND_PER_MATRIX_PERFORMANCE_OPS` 包含
   `spmm_csr` 和 `sddmm_csr`。输入为目录时，每个 `.mtx` 在独立子进程运行，同一
   矩阵内仍会分别尝试 FlagSparse 与 PyTorch-NPU baseline；baseline 报错时 CSV 保留
   它的状态和 FlagSparse 时间，`speedup` 留空。单个矩阵超时只记录在
   `timed_out_matrices`，已完成矩阵的行会合并保留。每矩阵命令也用
   `_subprocess_device_id(gpu_id)`，所以受 `ASCEND_RT_VISIBLE_DEVICES` 掩码后仍传
   `--device 0`。

   交付投影 `_delivery_performance_phase()` 以前在没有 dtype 行时一律变成
   `NOT_CONFIGURED`/NF；现在父任务为 `TIMEOUT` 时保留 `Timeout`，从而区分“没有跑”与
   “实际跑过但限时结束”。这个变化不影响普通没有对应 dtype 行的 NF 语义。

   直接 benchmark 的 dtype 列表是 `float16,float32,float64`；bf16 不是交付变体，
   因此不额外运行或投影它。已在该改动前启动的旧后台任务仍可能携带 bf16 参数，不影响
   f16/f32/f64 的结果解释。

### `src/flagsparse/sparse_operations/spsv.py`

1. **新增 `_spsv_ascend_row_sweep`**（当前行 7215--7265）。该函数是 CANN 无法
   lower 原 Triton CW atomic SpSV 时的 torch-NPU fallback：

   - `indptr64` 转为 host offsets，逐行取 CSR 区间；
   - lower 按正序、upper 按逆序遍历；
   - 用 `cols < row` 或 `cols > row` 找已求解的非对角项，计算
     `rhs[row] - sum(vals * x[cols])`；
   - unit diagonal 直接写回，否则除以对角值；无对角值时写入零；
   - 保持目标 dtype，校验 `out` 的 shape/dtype 后执行 `copy_`；
   - `return_time=True` 时在计时前后调用 `_ACCEL.synchronize()`，返回毫秒数。

   这是正确性 fallback，不是高吞吐实现；三角依赖导致它按行执行。

2. **在执行入口启用 fallback**（当前行 7317--7335，`_execute_spsv_csr_plan`）。
   原有的计算 dtype 转换和 alpha 缩放完成后，若 `_is_ascend_runtime()` 为真则：

   - `trans_mode != "N"` 立即抛出
     `NotImplementedError("Ascend SpSV fallback currently supports non-transpose solves only")`；
   - 非转置路径调用 `_spsv_ascend_row_sweep(...)`；
   - 非 Ascend 路径完全保留原来的 stream/Triton plan 和 transpose kernel。

### `benchmark/benchmark_ascend.py`

- **位置与输入模型**：`Case.matrix_path`（当前 31--38 行）保存来源文件；
  `_make_csr`（当前 64--95 行）用 `scipy.io.mmread` 读取 MatrixMarket、转 CSR、合并
  重复项并按当前 dtype 转换。未传输入时保留原本确定性随机 CSR，保证旧调用兼容。
- **选择性执行**：`run(..., op=...)`（当前 166--283 行）只为指定的算子创建 dense
  输入、SciPy 参考和计时闭包，避免 runner 每执行一个 `--op` 却重复运行五个算子。
- **CSV/JSON 身份信息**：`record`（当前 213--239 行）随每个结果写入 `matrix` 和
  `shape=<m>x<n>;nnz=<nnz>`；CSV 序列化（当前 332--359 行）也写入 `matrix` 列。
- **CLI 和遍历**：`--input`（当前 297--316 行）接受一个 `.mtx` 或目录，目录以排序后的
  `.mtx` 列表逐个运行；读取不到文件立即 argparse error，不静默回退。
- **baseline/参考**：FlagSparse 与 PyTorch-NPU 都在同一 NPU、同一输入和同一
  warmup/iters 下计时；SciPy/NumPy 只负责 CPU 精度参考，不参与性能计时。
- **大矩阵安全性**：SDDMM 的 SciPy 参考和 torch-NPU baseline 都改为按 CSR 非零元
  分块采样（chunk=262144），不再构造百万行乘百万列的稠密中间矩阵。
- **理由**：原 benchmark 只接受 `--m/--n/--nnz`，runner 虽然解析了矩阵目录却没有
  传给该脚本；这会造成结果目录看似真实矩阵、实际仍是合成 case。

### `src/flagsparse/sparse_operations/sddmm_csr.py`

- **位置**：Ascend 分支 `flagsparse_sddmm_csr` 当前 835--875 行。
- **改动**：保留 `_is_ascend_runtime()` 守卫和原有 alpha/beta/out/计时语义，将
  `torch.matmul(x, y.T)` 的完整稠密矩阵替换为：

  ```python
  vals = torch.empty((indices.numel(),), device=x.device, dtype=x.dtype)
  for begin in range(0, int(indices.numel()), 262144):
      end = min(begin + 262144, int(indices.numel()))
      vals[begin:end] = torch.sum(x[row_ids[begin:end]] * y[cols[begin:end]], dim=1)
  ```

  随后仍执行 `vals * alpha`、可选 `beta * data`、`out.copy_`，且计时前后仍同步 NPU。
- **理由**：真实矩阵包含百万行/列；完整 dense product 会直接耗尽 NPU 内存，分块采样
  保持 `out[p] = dot(x[row(p)], y[col(p)])` 语义并限制临时内存。非 Ascend 后端继续走
  原 Triton 路径。

### `benchmark/benchmark_ascend_accuracy.py`

- **位置**：当前 8--33 行，CLI `--dtypes`。
- **改动**：默认 dtype 从 `float32,float64` 改为 `float16,float32,float64`；没有增加
  bf16，因为它不是 40 条 Ascend 交付变体。每个 dtype 的 FlagSparse 输出仍与 CPU
  SciPy/NumPy 参考比较，float16 使用既有的 `2e-2` 容差。
- **理由**：f16 是交付变体但原脚本从未生成 f16 精度条目，导致结果显示 NF 而非真实
  精度状态。

## 1.3 当前 checkout 中已经存在、复现时不能漏掉的 Ascend 分支

这些不是当前 `git diff` 的新增行，但完整复现 Ascend 时必须保留：

| 文件和锚点 | 作用 |
|---|---|
| `gather_scatter.py:1322--1469` | Ascend gather/scatter 使用 torch-NPU 路径 |
| `spmv_csr.py:997--1017,1271` | 避开 Ascend 不支持的 `tl.associative_scan` |
| `spmm_csr.py:4276,4729,5271` | Ascend CSR SpMM dispatch/fallback |
| `sddmm_csr.py:83,835` | Ascend SDDMM 使用 torch-NPU 坐标采样 fallback（当前为分块实现） |
| `spsm.py:1270--1353` | `SPSM_ASCEND_DISPATCH`，SpSM CSR/COO fallback |
| `spmm_coo.py:1578--1650` | `SPMM_COO_ASCEND_DISPATCH`，COO `index_add` fallback |
| `sparse_operations/__init__.py:79,163,247,267` 与 `flagsparse/__init__.py:117--118,287--288` | 导出两个 dispatch 表 |

可用下列环境变量强制验证两条 fallback：

```bash
export FLAGSPARSE_SPSM_ASCEND_DISPATCH=1
export FLAGSPARSE_SPMM_COO_ASCEND_DISPATCH=1
```

## 2. runner 与 benchmark

| 文件 | 锚点 | 改动 | 为什么 |
|---|---|---|---|
| `run_flagsparse_pytest.py` | `ASCEND_BASELINE_OPS` | 五个算子（gather / scatter / spmv_csr / spmm_csr / sddmm_csr）走 `benchmark_ascend.py`，有厂商基线 | 只有这五个在 CANN 上有可用参照 |
| `run_flagsparse_pytest.py` | `ASCEND_ACCURACY_OPS` | **由 `ASCEND_BASELINE_OPS` 派生**，不重复写第二份名单 | 回传的版本里带了第二份硬编码副本，两份一旦改动就会漂移 |
| `run_flagsparse_pytest.py` | `ASCEND_PROBE_OPS` | 其余 17 个算子走能力探测 | 在可能 lower 不出内核的平台上，PASS/REJECTED/TRITON_COMPILE 才是有意义的测量 |
| `run_flagsparse_pytest.py` | `run_accuracy()` 的 Ascend 分支 | 这五个算子的精度改由 `benchmark/benchmark_ascend_accuracy.py` 跑，结果按原 `accuracy_result.json` 格式解析汇总 | |
| `benchmark/benchmark_ascend_probe.py` | 整文件 | 20 个矩阵、缓存清理、BSR/SELL/SpSV/SpSM 输入适配，区分 PASS / TRITON_COMPILE / REJECTED / ERROR / NO_ADAPTER | **每个用例前清空 Triton 缓存**，否则上一个用例的产物会让编译失败看起来像通过 |

> 这个探测脚本**名字带 ascend，实现是后端中立的**（通过 `_resolve_accel()` 跟随当前后端），
> 昆仑芯和后续 probe-only 后端复用同一份。昆仑芯那边提出把它改名
> `backend_capability_probe.py`，但文件没发过来，见 [`XPU.md`](XPU.md) §3。

## 3. 40 个交付变体的精度与性能记录

**数据来源**：`pytest_results_ascend_delivery_30matrix_67_torch/summary.json`，测试完成于
2026-09-17；PyTorch-NPU 为 baseline（`FLAGSPARSE_ASCEND_VENDOR=torch`）。这次运行的
直接 benchmark 实际只覆盖内置合成 case，**没有把 `/home/matrix` 的 30 个 `.mtx` 传到
Ascend benchmark 命令中**；因此下表的加速比只能用于当前 smoke/capability 记录，不能
报告为“30 个真实矩阵”的加速比。

精度列的 `P/F/T` 分别是通过/失败/总用例数；`NF` 是该交付变体没有对应精度用例。
性能的 `x.xx×` 是 `pytorch_ms / flagsparse_ms`，大于 1 表示 FlagSparse 更快；`Probe`
只有运行和参考结果，没有 PyTorch-NPU 性能基线；`NF` 是没有性能任务；`Error` 不存在
有效加速比。

| # | 交付变体 | 精度 P/F/T | 性能结果（相对 PyTorch-NPU） |
|---:|---|---|---|
| 1 | `gather_c32_int` | NF | NF |
| 2 | `gather_c64_int` | NF | NF |
| 3 | `gather_f16_int` | NF | Passed，0.874× |
| 4 | `gather_f32_int` | 1/0/1 | Passed，0.869× |
| 5 | `gather_f64_int` | 1/0/1 | Passed，0.891× |
| 6 | `scatter_c32_int` | NF | NF |
| 7 | `scatter_c64_int` | NF | NF |
| 8 | `scatter_f16_int` | NF | Passed，0.894× |
| 9 | `scatter_f32_int` | 1/0/1 | Passed，0.918× |
| 10 | `scatter_f64_int` | 1/0/1 | Passed，0.784× |
| 11 | `sddmm_csr_f32_int_non_non_row` | 1/0/1 | Passed，1.308× |
| 12 | `sddmm_csr_f64_int_non_non_row` | 0/1/1 | Error（NPU `DT_DOUBLE` matmul 不支持；无有效 speedup） |
| 13 | `spgemm_csr_f32_int_non_non` | 4/0/4 | Probe：20/20 PASS，无性能 baseline |
| 14 | `spgemm_csr_f64_int_non_non` | 4/0/4 | NF |
| 15 | `spmm_coo_c32_int_non_non_row` | 0/12/12 | NF |
| 16 | `spmm_coo_c64_int_non_non_row` | 0/12/12 | NF |
| 17 | `spmm_coo_f32_int_non_non_row` | 12/0/12 | Error：20/20（float/double 输入适配不一致） |
| 18 | `spmm_coo_f64_int_non_non_row` | 0/12/12 | NF |
| 19 | `spmm_csr_c32_int_non_non_row` | NF | NF |
| 20 | `spmm_csr_c64_int_non_non_row` | NF | NF |
| 21 | `spmm_csr_f32_int_non_non_row` | 1/0/1 | Passed，0.958× |
| 22 | `spmm_csr_f64_int_non_non_row` | 1/0/1 | Passed，1.000× |
| 23 | `spmv_coo_c32_int_non` | 18/0/18 | NF |
| 24 | `spmv_coo_c64_int_non` | 0/18/18 | NF |
| 25 | `spmv_coo_f32_int_non` | 18/0/18 | Error：20/20（float/double 输入适配不一致） |
| 26 | `spmv_coo_f64_int_non` | 0/18/18 | NF |
| 27 | `spmv_csr_c32_int_non` | NF | NF |
| 28 | `spmv_csr_c64_int_non` | NF | NF |
| 29 | `spmv_csr_f32_int_non` | 1/0/1 | Passed，0.927× |
| 30 | `spmv_csr_f64_int_non` | 1/0/1 | Passed，0.957× |
| 31 | `spsm_csr_f32_int_non_non_row` | 8/0/8 | Probe：20/20 PASS，无性能 baseline |
| 32 | `spsm_csr_f64_int_non_non_row` | 0/8/8 | NF |
| 33 | `spsv_coo_c32_int_non` | 0/55/55 | NF |
| 34 | `spsv_coo_c64_int_non` | 0/59/59 | NF |
| 35 | `spsv_coo_f32_int_non` | 16/32/48 | Probe：20/20 PASS，无性能 baseline |
| 36 | `spsv_coo_f64_int_non` | 16/32/48 | NF |
| 37 | `spsv_csr_c32_int_non` | 0/58/78 | NF |
| 38 | `spsv_csr_c64_int_non` | 0/62/82 | NF |
| 39 | `spsv_csr_f32_int_non` | 18/32/70 | Probe：20/20 PASS，无性能 baseline |
| 40 | `spsv_csr_f64_int_non` | 20/34/74 | NF |

注意：`summary.json` 会把 probe 的空 `speedup` 归一成 `0.0`，不代表 0× 性能；本表
已按原始 `performance.csv` 还原为“无性能 baseline”。同理，SDDMM float64 的汇总状态
曾显示 Passed/0.0，但原始 CSV 两边都报 `DT_DOUBLE` 不支持，故按 Error 记录。

## 4. 仍开着

- `benchmark_ascend_probe.py` **未进上游**，目前只在本仓库。
- 探测脚本只探能力不影响算子执行，所以它的改动不必逐次回传（已与维护者确认）。

## 5. 从干净 checkout 复现

```bash
git clone <repository-url>
cd flpagsparse_cwrapper
git checkout 87b0affbe6f9a3b352e84db31dd7297dc0cbac8e
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export PYTHONPATH="$PWD/src:$PWD"
export FLAGSPARSE_BACKEND=ascend
export FLAGSPARSE_ASCEND_VENDOR=torch
```

将 §1.2 的两个源文件改动移植后，先做静态检查：

```bash
git diff --check
python3 -m py_compile run_flagsparse_pytest.py \
  src/flagsparse/sparse_operations/spsv.py
```

设备隔离的关键规则是：runner 接收物理卡 `6,7`，每个子进程设置
`ASCEND_RT_VISIBLE_DEVICES=<6 或 7>`，命令行设备号则传逻辑 `0`。不要在已经设置该
掩码的子进程中继续传 `--device 6`。

长测试按约定使用独立 session，避免终端退出杀掉 NPU 进程：

```bash
setsid timeout -k 15s -s KILL 7200s \
  env PYTHONPATH="$PWD/src:$PWD" \
  FLAGSPARSE_BACKEND=ascend FLAGSPARSE_ASCEND_VENDOR=torch \
  python3 -u run_flagsparse_pytest.py \
  --phase both --mode normal --delivery-only --gpus 6,7 \
  --timeout 900 --results-dir pytest_results_ascend \
  > pytest_results_ascend.log 2>&1 < /dev/null &
```

`performance.csv` 中的 `triton_ms`、`pytorch_ms`、`speedup`、`max_abs_err` 才是
性能/精度字段；若要声称“30 个真实矩阵”，还必须检查实际子命令带有 MatrixMarket
输入目录，不能只看结果目录名称。

## 6. 已知边界

- Ascend SpSV fallback 当前只支持非转置（`trans_mode == "N"`）；转置/共轭转置会
  显式 `NotImplementedError`。
- 部分 float64/complex 路径受 torch-npu/CANN 算子能力限制，应单独记录为 dtype/API
  限制，不得当成性能结果。
- `ASCEND_RT_VISIBLE_DEVICES` 是 Ascend 的物理卡隔离变量，不能用
  `CUDA_VISIBLE_DEVICES` 替代。
- 实机测试只允许 NPU 6、7；复现前用 `npu-smi info` 确认这两张卡没有其他任务。

## 7. 合并记录（2026-09-18，合入 `86a09cd` 之上）

本台账只有文字描述、没有补丁，且各节对改了几个文件的说法不一致（开头说 4 个，§1.2 列了 5 个，
§5 说 2 个）。**只合入了有完整代码、且不依赖其他未合改动的 3 处**，已于 2026-09-18 在 910B4（NPU 6）上验证通过，结果见本节末「NPU 实机验证」：

| 台账位置 | 合入内容 | 合并时的差异 |
|---|---|---|
| §1.2 runner 1–4 | `_base_env()` 在 Ascend 上加 `ASCEND_RT_VISIBLE_DEVICES=<物理卡>`；新增 `_subprocess_device_id()`，精度/性能子进程传逻辑设备 0 | 与 XPU 合并时加入的 `script_device` 合成同一个函数（XPU 也返回 0） |
| §1.2 `sddmm_csr.py` | Ascend 分支由整块 `torch.matmul(x, y.T)` 改为按非零元分块采样 | 分块大小提成常量 `_ASCEND_SDDMM_CHUNK_NNZ = 262144`；在 CUDA 上强制走该分支、分块取 7/1000/262144，与 Triton 结果一致（f32 3.8e-6，f64 1e-14） |
| §1.2 `benchmark_ascend_accuracy.py` | 默认 `--dtypes` 加 `float16` | 无 |

**第二批（2026-09-18，Ascend 回传了 3 个完整文件：runner、`sddmm_csr.py`、`benchmark_ascend_accuracy.py`）**：
与 `87b0aff` 逐行对比后，`sddmm_csr.py`、`benchmark_ascend_accuracy.py` 与第一批合入的内容等价；runner 里
新合入两处：

| 改动 | 说明 |
|---|---|
| `ASCEND_PERFORMANCE_COMMANDS` 去掉 `bfloat16` | bf16 不是交付 dtype |
| `_delivery_performance_phase()`：父任务 `TIMEOUT` 且没有该 dtype 的行时保留 `Timeout` | 合进 `86a09cd` 重写后的函数；新增 CI 用例 |

**第三批（2026-09-18，Ascend 回传了 `spsv.py` 和 `benchmark/benchmark_ascend.py`）**：本台账描述的改动至此全部合入。

| 文件 | 合入内容 | 合并时的差异 / 验证 |
|---|---|---|
| `benchmark/benchmark_ascend.py` | `.mtx` 输入（`--input` 接受文件或目录）、`run(op=...)` 只跑指定算子、CSV 加 `matrix` 列、SDDMM 的 SciPy 参考和 PyTorch-NPU 基线都分块 | 原样合入。本机只验证了 CLI；NPU 上未跑 |
| `src/flagsparse/sparse_operations/spsv.py` | `_spsv_ascend_row_sweep()` 逐行求解；`_execute_spsv_csr_plan()` 在 Ascend 上分流，转置显式 `NotImplementedError` | **改了一处**：对角元绝对值小于 `_spsv_diag_eps_for_dtype()` 时按 1 除，与 Triton 内核的 `diag_safe` 一致（原版直接除，会得到极大值）。缺对角元写 0 与 Triton 内核一致，保留（前面"应当报错"的说法收回）。在 CUDA 上强制走该分支，与 Triton 结果对比：f32 ≤ 6e-5、f64 ≤ 3e-13，上/下三角、单位/非单位对角都一致 |
| `run_flagsparse_pytest.py` | `ASCEND_PERFORMANCE_COMMANDS` 加 `--input {input}`；`ASCEND_PER_MATRIX_PERFORMANCE_OPS = {spmm_csr, sddmm_csr}`，与 XPU 的逐矩阵条件并列 | 新增 CI 用例：5 个 Ascend 模板都带 `--input {input}`、不含 bf16，且 `benchmark_ascend.py` 确实认 `--input` |

**注意**：SpSV 回退是逐行的 torch 循环，每行都有 host 同步，是正确性兜底，不是高吞吐实现；在百万行的
真实矩阵上会非常慢。Ascend 的 SpSV 性能走探测脚本，不出加速比，报告时不要把它的耗时当成内核性能。

### NPU 实机验证（2026-09-18，910B4，NPU 6，代码 `7ec9add`）

| 项 | 命令要点 | 结果 |
|---|---|---|
| 设备隔离：pytest 路径 | `--ops spmv_coo --phase accuracy --gpus 6` | 子进程（PID 大于 runner）出现在 **NPU 6**，NPU 0 无新进程。精度 41 通过 / 36 失败，失败全是 float64、complex128 各 18 个，与 §3 记录一致（NPU 的 double 限制） |
| 设备隔离：专用脚本路径 | `--ops gather --phase both --gpus 6` | 精度、性能子进程命令都是 `--device 0`，两阶段 Passed |
| SDDMM 分块：精度 | `benchmark_ascend_accuracy.py --op sddmm_csr` | float16 / float32 / **float64** 全部 passed |
| SDDMM 分块：大矩阵 | 100 万 × 100 万、nnz = 500 万、K = 64 | 不再 OOM（旧写法要 ~4 TB 稠密矩阵），2.55 s，1 万个抽样最大误差 3.8e-6 |
| f16 精度脚本 | `benchmark_ascend_accuracy.py --op gather/scatter` | 两个算子的 float16 / float32 / float64 全部 passed |
| f16 交付投影 | `--ops gather,scatter --delivery-only --phase accuracy` | `gather_f16_int`、`scatter_f16_int` 均为 **Passed**（此前 NF） |

**顺带解决的问题**：§3 里 `sddmm_csr_f64_int_non_non_row` 的精度 0/1/1、性能 Error（`DT_DOUBLE` matmul 不支持）。
旧实现先算 `torch.matmul(x, y.T)`，NPU 的 matmul 不支持 double；分块实现只用逐元素乘和求和，不再调用
matmul，float64 精度已通过。性能是否也不再报错，要等下一轮交付测试确认。
