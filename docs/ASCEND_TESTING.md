# Ascend 910B 测试指南

本文说明 FlagSparse 在 Ascend 910B/CANN/torch_npu 环境中的验证和性能测试流程。Ascend
路径与 CUDA、ROCm、MetaX、MUSA 分开分发；本文设置不会改变其他后端。

## 环境检查

性能测试只使用 NPU 6、7 号卡：

```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export PYTHONPATH=$PWD/src
export FLAGSPARSE_BACKEND=ascend
export FLAGSPARSE_ASCEND_VENDOR=ops_sparse

python3 - <<'PY'
import importlib.metadata as md
import torch
print("torch:", md.version("torch"))
print("torch_npu:", md.version("torch-npu"))
print("torch.npu:", hasattr(torch, "npu"), torch.npu.is_available())
PY
```

`ops_sparse` Python bridge 和 `libaclsparse.so` 是可选的；缺失时 baseline 明确使用
PyTorch-NPU。

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

## 直接 benchmark

当前 Ascend benchmark 覆盖 `gather`、`scatter`、`spmv_csr`、`spmm_csr`、`sddmm_csr`：

```bash
python3 benchmark/benchmark_ascend.py \
  --device 6 --m 4096 --n 4096 --nnz 131072 \
  --dense-cols 64 --warmup 5 --iters 20
```

默认只测试 `float32`。需要多 dtype 时，可直接通过统一 runner 的
`--benchmark-args` 透传 `--dtypes`（逗号分隔）；每个 dtype 会写入独立的 CSV 行：

```bash
PYTHONPATH=src python3 -u run_flagsparse_pytest.py \
  --phase performance --ops gather,scatter,spmv_csr,spmm_csr,sddmm_csr \
  --gpus 6,7 --benchmark-warmup 5 --benchmark-iters 20 \
  --benchmark-args="--dtypes float16,bfloat16,float32,float64" \
  --results-dir pytest_results_ascend_dtypes
```

当前 benchmark 接受 `float16`、`bfloat16`、`float32`、`float64`；具体算子是否能在
910B/CANN 上执行仍以对应 CSV 行的状态和错误字段为准。

7 号卡使用相同命令，将 `--device 6` 改为 `--device 7`。输出包含 FlagSparse 和
PyTorch-NPU 的 mean/median/min/p95 延迟，以及 SciPy CPU 最大绝对误差。

## 统一 runner

Ascend 模式下，`run_flagsparse_pytest.py` 对上述五个算子改用
`benchmark/benchmark_ascend.py`，并将 `--gpus` 的卡号传给 `--device`。推荐后台运行：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export PYTHONPATH=$PWD/src
export FLAGSPARSE_BACKEND=ascend
export FLAGSPARSE_ASCEND_VENDOR=ops_sparse

run_id=$(date -u +%Y%m%dT%H%M%SZ)
setsid python3 -u run_flagsparse_pytest.py \
  --ops gather,scatter,spmv_csr,spmm_csr,sddmm_csr \
  --phase performance --gpus 6,7 \
  --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir "pytest_results_ascend_${run_id}" \
  > "pytest_ascend_${run_id}.log" 2>&1 < /dev/null &
```

查看进度：

```bash
tail -f pytest_ascend_<时间戳>.log
```

每个算子的 `performance.csv` 记录 `triton_ms`（Ascend FlagSparse 路径）、`pytorch_ms`、
`speedup`、`max_abs_err`；根目录生成 `summary.json`、`summary.csv`、`summary_flat.json`
和 `result.html`。

## Ascend 实现策略

910B 当前 Triton lowering 对部分 CSR/gather kernel 依赖不完整，且 PyTorch SparseCSR
`addmm` 在该后端不可用。因此 Ascend 分支使用等价的 torch_npu 原生 fallback：

- SpMV/SpMM：CSR row-id 缓存 + `index_add_`
- SDDMM：NPU dense matmul 后按 CSR 坐标采样
- Gather：NPU `torch.gather`
- Scatter：NPU `index_copy_`

这些 fallback 只在 Ascend 条件分支生效，其他后端保留原 Triton/vendor 路径。

## 已知限制

- 未安装 `ops_sparse`/`libaclsparse.so` 时，输出中的 baseline 是 PyTorch-NPU，不是
  ops-sparse。
- 测试前确认 6、7 号卡没有其他任务：`npu-smi info`。
- 若 runner 显示 `Skipped`，检查是否导出了 `FLAGSPARSE_BACKEND=ascend`；没有该变量时
  runner 按 CUDA/其他后端流程执行。
- 若 `torch.npu` 不可用，重新 source CANN `set_env.sh` 并检查 torch/torch_npu 版本匹配。
