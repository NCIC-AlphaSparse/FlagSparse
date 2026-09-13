# CSR SpMV algorithms and measurement contract / 算法与计时契约

The four new algorithms are implemented as explicit native CSR routes. The default
selection is unchanged. CUDA and ROCm have conservative launch profiles; **GPU
correctness and performance validation remain pending**. Offline compilation has
passed as described below. The static
support matrix therefore labels the new routes `UNVERIFIED`. Other backends retain
the existing routes and reject explicit requests for the new algorithms.

四个新算法均返回完整输出，不缓存跨调用执行计划，不接入调优器。当前代码提供
CUDA/ROCm 保守 profile；离线编译已通过，实机准确性和性能尚未验证，不能据此宣称提速。

## Algorithms

| Name | Native CSR computation | Per-invocation process |
| --- | --- | --- |
| `row_tile` | A `[rows_per_program, lanes_per_row]` tile; accumulate lanes and reduce each row. The tile loops to its local maximum length. Empty/tail rows are masked. | None |
| `row_vector` | One program per row, looping to that row's actual length, then one lane reduction. | None |
| `row_split_reduce` | Independent segments produce FP64 partials; bounded fan-in reductions repeatedly reduce each row to one sum. No floating-point atomics. | GPU counts, prefix sums, segment descriptors and reduction-level prefixes |
| `row_adaptive_split` | Short rows use `row_tile`, medium rows `row_vector`, long rows segmented reduction. Row ownership is disjoint. | GPU membership, prefix compaction and long-row segment plan |

Both basic row kernels have contiguous-row and row-list compile specializations.
Standalone calls use contiguous rows; adaptive calls use compact row lists.
`row_split_reduce` segments every nonempty row, including short rows. It is a complete
algorithm rather than a long-row-only helper.

GPU classification writes counts and membership in one pass. Tensor prefix sums
provide stable row offsets; GPU descriptors use a binary search over row segment
prefixes (one search per segment, not per nonzero). A partial slot is its descriptor
index. Every reduction layer consumes at most `reduce_block_size` partials per group.
Integer offsets, segment counts and address arithmetic remain 64 bit. Host reads of
counts allocate exact buffers inside the complete run; these waits are not CPU
algorithm time. All numeric reduction layers and final output stores belong to
compute, while construction of reduction prefixes belongs to process.

FP32 and FP64 inputs both use FP64 multiplication and accumulation, with output cast
to input dtype. The new kernels disable FP fusion. Existing `legacy_rowpar`,
`legacy_segbin`, and `legacy_bucket_vector` keep their original numerical strategies;
in particular, old FP32 segbin/bucket routes are not FP64 algorithms.

## API and configuration

```python
import flagsparse as fs

fs.list_spmv_csr_algorithms(op="non", dtype=data.dtype, backend="rocm")
fs.get_spmv_csr_algorithm_spec("row_adaptive_split")

prepared = fs.prepare_spmv_csr(
    data, indices, indptr, shape, op="non", alg="row_adaptive_split",
    config={"short_row_threshold": 32, "split_row_threshold": 1024},
)
y = fs.flagsparse_spmv_csr_run(prepared, x)
y, ms, meta = fs.flagsparse_spmv_csr_run(
    prepared, x, out=output, timing=True, return_time=True, return_meta=True,
)
# High-level calls accept the same algorithm/configuration selection.
y = fs.flagsparse_spmv_csr(data, indices, indptr, x, shape, alg="row_vector")
```

`auto` is a deterministic selector for the existing backend default, with existing
legacy environment overrides. `compare` exists only in the benchmark CLI. Prepared
objects pin resolved `alg`, `config`, and `op`; mismatched explicit run options raise.
Output must be contiguous, match dtype/device/shape, and share no storage with inputs.
Explicit `use_opt=False/True` selects the old base/bucket routes and conflicts with
inconsistent new options. For compatibility, an auto-prepared object may still be
used with historical `use_opt` calls. Explicit unsupported concrete routes raise.

The pure Python registry in `_spmv_csr_config.py` has no Torch/Triton imports. Its
backend vocabulary is `cuda/rocm/metax/mthreads/ascend`, separately from the actual
Triton target. New routes require known FP64, 64-bit-index and reduction capability.
Profile priority is explicit overrides, architecture overrides, backend conservative
defaults, then a resource-limited conservative candidate. Invalid built-in candidates
are discarded with reasons in `config_rejections`; invalid explicit settings raise.
Architecture overrides are currently empty: no unmeasured architecture tuning is
presented as an optimum.

| Configuration section | Initial values on CUDA / ROCm |
| --- | --- |
| `row_tile` | `lanes_per_row=8`, `num_warps=2`, `rows_per_program=2*subgroup_width/8` |
| `row_vector` | `block_nnz=128`, `num_warps=2` |
| `row_split_reduce` | `segment_nnz=1024`, `block_nnz=128`, `num_warps=2`, `reduce_block_size=256`, `reduce_num_warps=4` |
| `process` | `block_size=256`, `num_warps=4` for classification, compaction and descriptors; final stores use the same block size with `reduce_num_warps` |
| Each loop | `loop_num_stages=1` |
| Adaptive thresholds | `short_row_threshold=32`, `split_row_threshold=1024` |

The nested config accepts partial overrides and rejects unknown keys. Both column
indices and row pointers independently support int32/int64. A same-algorithm int32
retry occurs only for a recognizable index-compatibility error and only after range
validation. OOM, invalid memory access, and arbitrary launch failures are never
treated as index compatibility. Metadata preserves requested and actual index types,
selected algorithm, numerical dtype, configuration, target, architecture and retry
reason. Profiling and launches use the input device's current stream.

## Timing and baseline

**Always `ms = process_cpu_ms + gpu_ms`, including with `--timing`.** This explicit
CSR contract overrides the older SOP convention of summing phase measurements.

完整调用使用不插入分段事件的 run 测量；`--timing` 另行运行相同输入、算法和配置，
取得 `process_gpu_ms/compute_ms`。不能用分段之和替换 `gpu_ms` 或加速比的分母。

- `gpu_ms`: actual-stream event measurement of the entire invocation, including
  classification, descriptors, host submission gaps/count waits, allocations,
  numeric computation, output initialization and every reduction layer.
- `process_cpu_ms`: CPU construction of kernel-consumed execution data only. It is
  zero in the new GPU plan implementation. Launch dictionaries and device waits do
  not count as CPU algorithm work.
- `process_gpu_ms` / `compute_ms`: independently measured diagnostics. A run with
  diagnostics executes again; separate sums need not equal the complete event time.
- MatrixMarket loading, CSR construction/input casts, reference and vendor
  descriptor construction, compilation and warmup are outside benchmark timing.
- hipSPARSE handles are bound to the measured Torch stream. CuPy CSR nontranspose
  uses an external view of that stream. A CuPy transpose that changes storage format
  is reported as unavailable rather than used as a CSR baseline.
- The CPU FP64/complex128 scatter reference is correctness-only. The vendor output
  is independently checked. Failed correctness rows have no usable speedup/ranking.

## Reproduce and validate

```bash
python -m pytest tests/ci/test_spmv_csr_policy.py
TRITON_INTERPRET=1 python -m pytest tests/ci/test_spmv_csr_interpreter.py
python tools/ci/check_spmv_csr_compile.py --backend cuda --arch 80
python tools/ci/check_spmv_csr_compile.py --backend hip --arch gfx90a --output build/spmv_csr_compile_hip.json
python -m pytest tests/pytest/test_spmv_csr_accuracy.py -m spmv_csr --mode quick
python tests/test_spmv_csr.py --synthetic --alg compare --timing --csv-csr synthetic.csv
python tests/test_spmv_csr.py /path/to/matrices --alg compare --dtypes float32,float64 --index-dtypes int32,int64 --indptr-dtypes int32,int64 --timing --warmup 5 --iters 20 --csv-csr matrices.csv
FLAGSPARSE_SPMV_CSR_MTX_DIR=/path/to/matrices python -m pytest tests/pytest/test_spmv_csr_accuracy.py -k external_matrix_regressions
python run_flagsparse_performance.py --ops spmv_csr --benchmark-input matrix --benchmark-args "--alg compare --timing"
```

The compatibility flags `--dtype`, `--index-dtype`, `--csv`, and `--no-cusparse`
remain aliases. `--no-vendor` disables the vendor measurement. Default benchmark
dtypes are FP32/FP64; `--dtypes all --ops all` exercises the legacy support surface.
`--config` accepts JSON. `tests/test_spmv.py` and `benchmark/benchmark_spmv.py` forward
to the single CSR benchmark entry. Accuracy cases live only under `tests/pytest`.
The GPU benchmark suite runs `compare` with timing and emits `spmv_csr.csv`.

CSV is flushed per matrix × dtype × column-index dtype × row-pointer dtype × op ×
algorithm. It contains software/device/commit identity, actual parameters, complete
latencies and optional phase diagnostics. `+dirty` in commit identity requires
retaining the working-tree patch alongside results for exact reproduction.

`tests/data/spmv_csr_regressions.json` records the original 30 matrix names and six
FP32 failures: `2cubes_sphere`, `CurlCurl_1`, `c8_mat11`, `engine`, `smt`, and
`water_tank`. External tests require those files when the regression directory is
enabled; they do not relax shared tolerances. Short-row focus is `roadNet-TX`,
`ecology1`, `NACA0015`; long-tail focus is `wiki-Talk`, `Stanford`, `mip1`.

The acceptance decision must use complete invocation latency, particularly for
adaptive/split algorithms whose per-call plan can dominate computation. No fixed
speedup over hipSPARSE is promised and this change does not switch the default.

## Local validation record (2026-09-12)

- Python 3.12.14, PyTorch 2.14.0+cpu, Triton 3.6.0 (`triton-windows` 3.6.0.post26).
- CUDA sm80: all 53 compiled specializations produced binaries offline.
- HIP gfx90a: all 53 compiled specializations produced binaries offline.
- 42 CPU policy, public API, registry, runner and documentation checks passed.
- 144 Triton interpreter tests passed. They execute production row kernels, plan construction,
  descriptors and multilevel reductions on CPU; public orchestration tests mock
  only device capability/acceptance and events. These are not GPU timings.
- 502 GPU accuracy cases collect from the single `tests/pytest/test_spmv_csr_accuracy.py`
  module; the forwarding benchmark entry points collect no duplicate cases.
- No CUDA/ROCm GPU accuracy run or 30-matrix performance rerun was performed in
  this environment. Other platforms remain unverified. The CPU interpreter emits
  an upstream NumPy scalar-conversion deprecation warning; no tolerance was relaxed.

Re-run the documented GPU commands on each target machine and retain the CSVs and
working-tree patch before declaring the hardware acceptance criteria complete.
