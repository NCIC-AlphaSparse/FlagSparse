# CSR SpMV dtype, op and timing contract

## Algorithms and types

All seven registered algorithms implement `non`, `trans`, and `conj`, with
float16, bfloat16, float32, float64, complex64 and complex128 inputs. `conj` means
`A.conj().T @ x`, not conjugation of x. Output dtype matches the input dtype.

The four new algorithms retain CUDA/ROCm profiles only. Other backends keep their
existing paths; an explicit unsupported new algorithm must raise. Support
declarations are not hardware validation. The dtype/op expansion is **unverified
on hardware**; no operator, interpreter, pytest or GPU compilation was run in the
development environment for this expansion.

| Algorithm | FP16/BF16 accumulation | FP32 | FP64 | Complex64/128 components |
| --- | --- | --- | --- | --- |
| row_tile, row_vector, row_split_reduce, row_adaptive_split | FP32 | FP64 | FP64 | FP32 / FP64 |
| legacy_rowpar | Existing FP32 | Existing FP64 | FP64 | Existing FP32 / FP64 |
| legacy_segbin, legacy_bucket_vector | FP32 | Existing FP32 | FP64 | FP32 / FP64 |

The new row kernels use separate real/imaginary accumulators and no floating-point
atomics. Split partials and every reduction level retain the selected component
precision. Legacy segbin retains its atomics. The bucket algorithm retains its own
bucket/batched-row scheduling. It requires int32 column indices; the other six
algorithms accept int32/int64 columns. All accept int32/int64 row pointers.

## Invocation and timing

Prepared objects retain the **original CSR and original shape** for every op.
For `(m, n)`, non takes an n-vector and returns an m-vector; trans/conj take an
m-vector and return an n-vector. Prepared alg/config/op remain fixed. A runtime
transpose creates invocation-local CSR arrays by stable column sorting, retaining
duplicates and explicit zeros. Conjugation is applied to reordered matrix values.
Classification, buckets and segments use the resulting row lengths. Neither the
transpose nor the execution plan is retained across invocations.

**Always `ms = process_cpu_ms + gpu_ms`, with or without `--timing`.**

- `gpu_ms` measures the complete run on the actual stream: CSR transpose, plans,
  necessary conversions, output initialization, multiplication and all reductions.
- `process_gpu_ms` diagnoses transpose construction and bucket/segment plans.
- `compute_ms` diagnoses arithmetic, its conversions, initialization and reductions.
- `process_cpu_ms` excludes Python configuration and GPU waits; this implementation
  uses GPU plan construction and reports zero CPU algorithm processing.
- Phase diagnostics run separately. Their sum never replaces complete run latency.
- Input loading, baseline descriptors, reference construction and warmup are outside
  native run timing. The baseline receives original CSR with the requested op.

This changes historical legacy trans/conj timing, which performed CSR transpose
inside prepare. Do not compare those older timings directly with implementation
version 2. Nontranspose defaults and existing precision policies remain unchanged.

Metadata includes `compute_dtype`, `component_dtype`, `transpose_strategy`, requested
and execution index types, backend/profile, implementation version and any safe
index compatibility fallback. Errors cannot silently select another algorithm.

## Compute-node validation

```bash
python tests/test_spmv_csr.py --synthetic --alg compare --dtypes all --ops all --index-dtypes int32,int64 --indptr-dtypes int32,int64 --timing --csv-csr synthetic.csv
python tests/test_spmv_csr.py ../matrix --alg compare --dtypes all --ops all --index-dtypes int32,int64 --indptr-dtypes int32,int64 --warmup 10 --iters 50 --timing --csv-csr ./results/spmv_csr_all_dtype_ops.csv
python -m pytest tests/pytest/test_spmv_csr_accuracy.py -m spmv_csr
FLAGSPARSE_SPMV_CSR_MTX_DIR=../matrix python -m pytest tests/pytest/test_spmv_csr_accuracy.py -k external_matrix_regressions
```

Row-level results are recorded in the `status` column of the CSV, and the script
prints a one-line count of FAIL rows at the end. Like the other benchmark scripts it
exits 0 whether or not rows FAIL, because the runner treats a non-zero exit as a failed
phase and every delivery row of the operator inherits that. `--fail-fast` is the
explicit way to stop, and exit non-zero, at the first failing row.

`compare` selects seven algorithms with int32 columns and six with int64 columns
on a capable CUDA/ROCm device, for each requested dtype/op. It reports exclusions;
prepare still checks actual device capability. `auto` continues to select the old
default. Without `--indptr-dtypes`, row pointers follow the column index dtype.

Accuracy cases include all six dtypes, all ops, mixed indices, rectangular/empty
matrices, duplicate unsorted columns, cancellation, multilevel reduction and short
original rows becoming long transposed rows. A rebuild-count test checks that
prepare performs no transpose and that complete/diagnostic runs rebuild separately.
The external suite covers the original 30 matrices, including complex failures.

Keep the shared tolerance unchanged. Native FP32 complex components can fail on
cancellation-heavy inputs; preserve FAIL, error magnitude and absent usable speedup
rather than silently promoting precision. Acceptance requires actual CUDA/ROCm
correctness and performance evidence, recorded separately for each platform.

## 中文说明

七个算法补齐六种 dtype 和 non/trans/conj；复数按实虚部分离计算，complex64 使用
FP32 分量，complex128 使用 FP64 分量。legacy_bucket_vector 仍仅支持 int32 列索引。
prepared 保存原始 CSR；每次 run 重建转置 CSR，按新行长分类，不缓存跨调用计划。
转置和执行计划构建计入 process_gpu_ms，完整调用 gpu_ms 包含全部必要工作。
有无 --timing 均使用 ms=process_cpu_ms+gpu_ms，不能用分段之和替代完整耗时。
本次仅进行源码和静态检查；实机准确性、性能及旧失败矩阵的复测尚待计算节点完成。
