# Ascend 910B 测试指南

本文说明 FlagSparse 在 Ascend 910B/CANN/torch_npu 环境中的验证和性能测试流程。Ascend
路径与 CUDA、ROCm、MetaX、MUSA 分开分发；本文设置不会改变其他后端。C API 那一层见
`capi/docs/ASCEND.md`；各后端文档的对应关系见 [README.md](README.md)。

## 环境检查

性能测试只使用 NPU 6、7 号卡：

```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export PYTHONPATH=$PWD/src
export FLAGSPARSE_BACKEND=ascend
export FLAGSPARSE_ASCEND_VENDOR=torch      # 默认值；ops_sparse 只用于厂商 A/B

python3 - <<'PY'
import importlib.metadata as md
import torch
print("torch:", md.version("torch"))
print("torch_npu:", md.version("torch-npu"))
print("torch.npu:", hasattr(torch, "npu"), torch.npu.is_available())
PY
```

**性能 baseline 现在默认就是 PyTorch**（`FLAGSPARSE_ASCEND_VENDOR=torch`），不再是
"ops-sparse 优先、缺失才回落"——这样昇腾这一列和其余非 CUDA 后端量的是同一个参照。
`ops_sparse` Python bridge 和 `libaclsparse.so` 仍然可选，装了就能用
`FLAGSPARSE_ASCEND_VENDOR=ops_sparse` 选回去做一次厂商 A/B。

**精度参考是 CPU 上的 SciPy**（见仓库根 `README.md` 的 `FLAGSPARSE_ACCURACY_REFERENCE`）：
昇腾上的 torch.sparse 本身就是被测对象而不是参考。

## 交付复现：40 个变体 × 30 个矩阵（精度 + 性能）

只能用 NPU 6、7。先 `npu-smi info` 确认这两张卡上没有别的任务。

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export PYTHONPATH="$PWD/src:$PWD" FLAGSPARSE_BACKEND=ascend FLAGSPARSE_ASCEND_VENDOR=torch
python3 -c "from flagsparse.sparse_operations import _common as c; print(c._backend_name(), c._accel_fallback_reason())"
# 期望：ascend None

setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --phase both --mode normal --delivery-only --gpus 6,7 --timeout 3600 \
  --benchmark-input /home/matrix --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir pytest_results_ascend_delivery \
  > pytest_results_ascend_delivery.log 2>&1 < /dev/null &

python3 tools/delivery_table.py pytest_results_ascend_delivery   # 跑完后：40 行结果，缺变体时退出码为 1
```

runner 在 Ascend 上自动做三件事，不需要手动处理：

- **卡隔离**：每个子进程设 `ASCEND_RT_VISIBLE_DEVICES=<6 或 7>`、命令行传逻辑设备 `--device 0`
  （torch_npu 不认 `CUDA_VISIBLE_DEVICES`，不设的话默认会落到物理 NPU 0）；
- **路由**：gather / scatter / spmv_csr / spmm_csr / sddmm_csr 的性能走 `benchmark/benchmark_ascend.py`，
  对 PyTorch-NPU 计时，并通过 `--input` 拿到 `--benchmark-input` 的 30 个 `.mtx`；spmm_csr、sddmm_csr
  每个矩阵单独一个子进程（`--timeout` 是每个矩阵的上限）。其余 6 个父算子走能力探测
  `benchmark/benchmark_ascend_probe.py`，**只有能不能跑、没有加速比**；
- **精度**：上面 5 个算子用 `benchmark/benchmark_ascend_accuracy.py`（每个 dtype 一个合成用例，对 SciPy），
  其余走 `tests/pytest`，参考同样是 CPU 上的 SciPy。

**跑完先核对真实矩阵确实传进去了**：`pytest_results_ascend_delivery/spmm_csr/performance.csv` 的 `matrix`
列应当是 30 个 `.mtx` 文件名，而不是 `synthetic`。

**预期会看到的非 Passed**（2026-09-17/18 在 910B4 上实测，`modified/ASCEND.md` 第 3、7 节）：

- float64 / complex128 的不少变体失败：NPU 对 double 的 matmul 等算子不支持（`DT_DOUBLE`）。
  SDDMM 分块改写后 `sddmm_csr_f64` 精度已通过；
- SpSV 在 Ascend 上是逐行求解的正确性兜底（`_spsv_ascend_row_sweep`），只支持非转置，转置/共轭用例报
  `NotImplementedError`；它在百万行矩阵上很慢，而且性能走探测脚本，**不要**把它的耗时当成内核性能；
- 走探测脚本的 6 个父算子，性能状态可以是 Passed，但没有加速比，`delivery_table.py` 里显示为 `-`。

**`86a09cd`（2026-09-18）之前跑出的性能结果作废**，原因见 `prompt.md` 第 2 节；2026-09-17 那一轮的
Ascend benchmark 实际用的是合成矩阵（`modified/ASCEND.md` 第 3 节），也不能当作 30 矩阵结果。

## Ascend fallback 分发表

有两个内核在 CANN 上编不出来，各自有一条 torch-only 的退路，并且都暴露成公共符号：

| 符号 | 覆盖 | 退路 | 为什么 |
|---|---|---|---|
| `flagsparse.SPSM_ASCEND_DISPATCH` | `spsm_csr` / `spsm_coo` | 逐行 sweep | 轮询求解器自旋在全局内存的 ready 标志上，CANN 不 lower 这个形状 |
| `flagsparse.SPMM_COO_ASCEND_DISPATCH` | `spmm_coo` | 单次 `index_add` | 两条 COO 路线都在内核里开 scratchpad，CANN 编译失败 |

两者默认由运行时探测自动选择，也可以用环境变量强制：

```bash
FLAGSPARSE_SPSM_ASCEND_DISPATCH=1      # 强制走 SpSM fallback
FLAGSPARSE_SPMM_COO_ASCEND_DISPATCH=1  # 强制走 SpMM COO fallback
```

**强制开关不是调试遗留物**：fallback 的实现是纯 torch，所以这两个开关让它能在任何后端上
与 Triton 路径逐位对比验证——包括手边没有 NPU 的机器。改动 fallback 后应当先这样验证。

SpSM 那条退路是**逐行**的（三角求解天生有依赖链），所以它是正确性兜底而不是快路；它在 RHS
维度上是向量化的，代价是 O(n_rows) 次 launch 而不是 O(n_rows × n_rhs)。SpMM COO 那条没有
依赖链，是一次 scatter-add，所以它本身就是条像样的路径。

## 算子能力探测（benchmark_ascend_probe.py）

在一个内核未必能被 CANN 编译出来的后端上，"跑没跑通、没通是为什么"比耗时更重要。
探测脚本对**全部 23 个算子** × **20 个矩阵**逐个跑一遍，把结果分成五类：

| 状态 | 含义 |
|---|---|
| `PASS` | 跑通，且与 CPU fp64 参考一致 |
| `MISMATCH` | 跑通但数值不对——是错答案，不是缺功能 |
| `TRITON_COMPILE` | Triton 后端无法 lower 这个内核。**这才是 Ascend 的典型故事**：内核本身没问题，CANN 编不出来 |
| `REJECTED` | 算子自己拒绝了输入（dtype/布局/形状不支持）——是主动划的边界，不是缺陷 |
| `ERROR` | 其余情况，单独分类以免与编译失败混淆 |

```bash
export FLAGSPARSE_BACKEND=ascend
python3 benchmark/benchmark_ascend_probe.py \
  --device 6 --dtype float32 --csv-summary probe.csv

# 只看某几个算子 / 某几个矩阵
python3 benchmark/benchmark_ascend_probe.py --op spsm_csr --op spmm_coo \
  --matrix square_1k --matrix square_4k
```

**每个 case 默认清空 Triton 缓存再跑**。这不是洁癖：前一个 case 编出来的产物会让后一个
case 的编译失败变成缓存命中，于是一个编不出来的内核报成 PASS。`--keep-cache` 可以关掉
（快很多），但关掉之后这份结果就不能用来判断"能不能编译"。

矩阵是按 seed 生成的合成矩阵，覆盖尺寸、长宽比、密度，以及退化端（单行、单列、全零），
所以结果可复现，不需要随仓库分发矩阵集。BSR/Blocked-ELL/SELL/SpSV/SpSM 的输入由脚本
按各自格式真实构造（分块、切片、三角且对角占优），不是硬凑的——否则算子拒绝一个畸形输入
会被记成 Ascend 的限制。

## 直接 benchmark（排查用）

`benchmark/benchmark_ascend.py` 覆盖 gather、scatter、spmv_csr、spmm_csr、sddmm_csr。单独调用时直接传
物理卡号（没有 runner 设的可见设备掩码）：

```bash
# 一个真实矩阵、一个算子
python3 benchmark/benchmark_ascend.py --device 6 --op spmm_csr \
  --input /home/matrix/cage12.mtx --dtypes float16,float32,float64 \
  --dense-cols 64 --warmup 5 --iters 20 --csv-summary /tmp/asc_spmm.csv

# 不给 --input 时用内置合成 case（--m/--n/--nnz）
python3 benchmark/benchmark_ascend.py --device 6 --m 4096 --n 4096 --nnz 131072 \
  --dense-cols 64 --warmup 5 --iters 20
```

`--input` 可以是一个 `.mtx` 或一个目录（目录下每个 `.mtx` 各跑一遍）。`--dtypes` 接受 `float16`、`bfloat16`、
`float32`、`float64`；交付只需要 `float16,float32,float64`，bf16 不用跑。输出包含 FlagSparse 和 PyTorch-NPU 的
mean/median/min/p95 延迟，以及对 SciPy（CPU）的最大绝对误差；CSV 里的 `matrix` 列写明用的是哪个矩阵。

## 统一 runner

交付测试的命令见上面"交付复现"一节。要单独跑几个算子时，同样的环境变量加 `--ops` 即可：

```bash
python3 run_flagsparse_pytest.py --ops spmm_csr,sddmm_csr --phase both --mode normal \
  --gpus 6 --timeout 3600 --benchmark-input /home/matrix \
  --benchmark-warmup 5 --benchmark-iters 20 --results-dir pytest_results_ascend_spmm_sddmm
```

每个算子的 `performance.csv` 记录 `triton_ms`（FlagSparse 的 Ascend 路径）、`pytorch_ms`、`speedup`、
`max_abs_err` 和 `matrix`；根目录生成 `summary.json`、`summary.csv`、`summary_flat.json` 和 `result.html`。

## Ascend 实现策略

910B 当前 Triton lowering 对部分 CSR/gather kernel 依赖不完整，且 PyTorch SparseCSR
`addmm` 在该后端不可用。因此 Ascend 分支使用等价的 torch_npu 原生 fallback：

- SpMV/SpMM：CSR row-id 缓存 + `index_add_`
- SDDMM：按 CSR 非零元分块做 `sum(x[row] * y[col])`（每块 262144 个非零元），不构造稠密乘积 ——
  百万行的真实矩阵上稠密乘积会耗尽显存，且 NPU 的 matmul 不支持 double
- SpSV：逐行求解（`_spsv_ascend_row_sweep`），只支持非转置
- Gather：NPU `torch.gather`
- Scatter：NPU `index_copy_`

这些 fallback 只在 Ascend 条件分支生效，其他后端保留原 Triton/vendor 路径。

## 已知限制

- 输出中的 baseline 默认就是 PyTorch-NPU 而非 ops-sparse，这是**策略**不是缺失；
  要 ops-sparse 得显式 `FLAGSPARSE_ASCEND_VENDOR=ops_sparse`，且它确实装了。
- 测试前确认 6、7 号卡没有其他任务：`npu-smi info`。
- 若 runner 显示 `Skipped`，检查是否导出了 `FLAGSPARSE_BACKEND=ascend`；没有该变量时
  runner 按 CUDA/其他后端流程执行。
- 若 `torch.npu` 不可用，重新 source CANN `set_env.sh` 并检查 torch/torch_npu 版本匹配。
