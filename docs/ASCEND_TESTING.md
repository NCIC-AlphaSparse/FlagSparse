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
