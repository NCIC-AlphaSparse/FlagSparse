# CUDA（RTX 5090）Q4 调试、优化与复测

更新：2026-10-06。本文是 CUDA Q4 的统一入口，合并了此前 6 份优化/补测报告；
已被后续结果替代的初轮耗时表不再作为当前结论，原始数据仍保留。
DCU/MACA/MUSA 等后端的真机结论见各自手册，不能用 CUDA 结果替代。

## 1. 当前结论

- 45/45 个变体都有 CuPy 加速比：43 项为 2026-10-06 的分批复测，
  CSC f32/c32 SpMM 两项使用 2026-10-02 历史结果。这不是 45 项同轮重跑。
- 按 CuPy 优先口径，44/45 个变体的平均加速比 ≥0.8。
  **SpGEMM f32 最新值 0.789×，尚未达标**，相对 PyTorch 为 0.799×。
- 每个算子独立计算多个案例的 `baseline_ms / ours_ms` 算术平均，
  不跨算子平均，也不是累计耗时之比。部分单矩阵仍低于 0.8。
- 最新矩阵案例精度全部通过。下表的 PyTorch 空值仅表示历史 CSC 两项没有该性能记录。
- 微秒级耗时受设备状态和启动开销影响；CSC f16 SpMV 0.846×、COO int8 SpMM
  0.889×、scatter 0.904× 等接近门槛的项应关注复测波动。

## 2. 45 项平均加速比

| # | 变体 | CuPy 平均 | PyTorch 平均 | 案例数 | 来源日期 |
|---:|---|---:|---:|---:|---|
| 1 | scatter_i8_int | 0.904 | 1.883 | 4 | 2026-10-06 |
| 2 | axpby_f16_int | 3.506 | 0.991 | 4 | 2026-10-06 |
| 3 | spmv_sell_f32_int_non | 1.391 | 0.941 | 10 | 2026-10-06 |
| 4 | spmv_csr_f16f32_int_non | 1.425 | 0.921 | 10 | 2026-10-06 |
| 5 | spvv_f16f32_int_non | 2.180 | 1.308 | 4 | 2026-10-06 |
| 6 | spmv_csr_f16_int_non | 1.295 | 0.832 | 10 | 2026-10-06 |
| 7 | spmv_csr_f32c32_int_non | 1.189 | 0.881 | 10 | 2026-10-06 |
| 8 | spmv_sell_f16_int_non | 1.377 | 0.904 | 10 | 2026-10-06 |
| 9 | spmm_csr_f16f32_int_non_non_row | 3.336 | 1.358 | 10 | 2026-10-06 |
| 10 | spmm_csr_f16_int_non_non_row | 7.480 | 3.011 | 10 | 2026-10-06 |
| 11 | spmm_csr_f32_int_non_non_col | 4.022 | 6.023 | 10 | 2026-10-06 |
| 12 | spmm_csr_f32_int_non_trans_row | 4.016 | 6.004 | 10 | 2026-10-06 |
| 13 | spmv_csc_f32_int_non | 1.272 | 0.758 | 10 | 2026-10-06 |
| 14 | spmv_csr_c32_int_non | 1.124 | 0.769 | 10 | 2026-10-06 |
| 15 | spmv_sell_c32_int_non | 1.148 | 0.871 | 10 | 2026-10-06 |
| 16 | spvv_c32_int_conj | 1.733 | 0.938 | 4 | 2026-10-06 |
| 17 | spmv_coo_f16f32_int_non | 1.184 | 0.722 | 10 | 2026-10-06 |
| 18 | spmm_csr_c32_int_non_non_row | 1.646 | 0.770 | 10 | 2026-10-06 |
| 19 | spmv_coo_f32_int_trans | 1.247 | 0.934 | 10 | 2026-10-06 |
| 20 | spmv_csr_f32_int_trans | 1.164 | 0.895 | 10 | 2026-10-06 |
| 21 | spmv_csr_i8f32_int_non | 1.414 | 0.935 | 10 | 2026-10-06 |
| 22 | spmv_csr_i8i32_int_non | 1.390 | 0.900 | 10 | 2026-10-06 |
| 23 | spmv_sell_i8i32_int_non | 1.385 | 0.911 | 10 | 2026-10-06 |
| 24 | spvv_i8i32_int_non | 1.481 | 1.401 | 4 | 2026-10-06 |
| 25 | spmm_csc_f32_int_non_non_row | 3.691 | — | 10 | 2026-10-02 |
| 26 | spmm_csr_i8i32_int_non_non_row | 3.409 | 1.392 | 10 | 2026-10-06 |
| 27 | spmv_coo_c32_int_non | 1.166 | 0.855 | 10 | 2026-10-06 |
| 28 | spmv_coo_f16_int_non | 0.937 | 0.603 | 10 | 2026-10-06 |
| 29 | spmv_csc_c32_int_non | 1.240 | 0.870 | 10 | 2026-10-06 |
| 30 | spmv_csc_f16_int_non | 0.846 | 0.591 | 10 | 2026-10-06 |
| 31 | sddmm_csr_f32_int_non_non_col | 29.846 | 3.552 | 10 | 2026-10-06 |
| 32 | sddmm_csr_f32_int_non_trans_row | 36.491 | 1.967 | 10 | 2026-10-06 |
| 33 | sddmm_csr_f32_int_trans_non_row | 11.152 | 1.982 | 10 | 2026-10-06 |
| 34 | spmm_coo_c32_int_non_non_row | 1.648 | 0.774 | 10 | 2026-10-06 |
| 35 | spmm_coo_f16_int_non_non_row | 2.466 | 1.008 | 10 | 2026-10-06 |
| 36 | spmm_csr_f32_int_trans_non_row | 4.395 | 1.904 | 10 | 2026-10-06 |
| 37 | spmv_coo_c32_int_conj | 0.991 | 0.930 | 10 | 2026-10-06 |
| 38 | spmv_coo_i8i32_int_non | 1.166 | 0.729 | 10 | 2026-10-06 |
| 39 | spmv_csr_c32_int_conj | 0.956 | 0.881 | 10 | 2026-10-06 |
| 40 | spmm_coo_i8i32_int_non_non_row | 0.889 | 0.386 | 10 | 2026-10-06 |
| 41 | spmm_csc_c32_int_non_non_row | 1.156 | — | 10 | 2026-10-02 |
| 42 | sddmm_csr_c32_int_non_non_row | 8.976 | 1.269 | 10 | 2026-10-06 |
| 43 | sddmm_csr_f16_int_non_non_row | 14.983 | 2.795 | 10 | 2026-10-06 |
| 44 | spmm_csc_f16_int_non_non_row | 1.035 | 0.423 | 10 | 2026-10-06 |
| 45 | spgemm_csr_f32_int_non_non | 0.789 | 0.799 | 10 | 2026-10-06 |

机器可读 [JSON](Q4_CUDA_45_SPEEDUPS_20261006.json)、
[CSV](Q4_CUDA_45_SPEEDUPS_20261006.csv) 保留各项原始来源路径、日期、案例数、
CuPy 单案例低于 0.8 的数量和未四舍五入的均值。

## 3. 测试规模与基线

设备：RTX 5090；PyTorch 2.9.0+cu128、Triton 3.6.0、CuPy 13.3.0。
使用历史 10 个 MatrixMarket 文件及 runner 的系数转换/缩放：
ASIC_680ks、GL7d14、NACA0015、amazon0601、auto、cage12、cfd2、
filter3D、roadNet-TX、wave。SpMM N=32，SDDMM K=32，SELL slice=32。
向量算子保留 4 个规模：32768:1024、131072:4096、524288:16384、1048576:65536。

- warmup=5、iters=20。矩阵、Axpby、SpVV 沿用同步 wall-clock 和原有 prepared/call 边界；
  prepare、JIT 冷启动不计入稳态。混合 COO、CSC half 等原本完整调用计时的路径不改为预处理计时。
  各矩阵独立调用 runner 并重置原种子 2026，不保证与历史整批运行的随机输入逐元素相同。
- scatter 沿用原报告 CUDA Graph：batch=100、warmup=5、repeats=20。
  三方都使用相同图计时；CuPy 用 ExternalStream 绑定捕获流。
  唯一 int32 索引、reset_output=True，清零和写入每次都计时。
- CuPy SpMV/SpMM 是等价 CSR `A @ B`；格式/转置/共轭准备在计时外。
  SpMM RHS 在计时外转换为 Fortran 布局，避免重复转换。
  SELL 对比等价 CSR 运算，不是 CuPy SELL kernel。
- CuPy sparse half/int8 不支持相同类型组合，使用相同量化后输入转 f32 计算，转换在计时外；
  Triton 仍按要求输出 half/int32。实数矩阵×复数向量提升到 complex64，不能丢弃虚部。
  原始行记录 `cupy_compute_dtype`，整数结果验证精确相等。
- **CuPy SDDMM 是分块 gather/product/sum，不是原生 sparse SDDMM。**
  每块最多 262144 nnz，gather、乘法、归约和输出分配全部计时，复数乘法不共轭。
  PyTorch 列使用原生 `sampled_addmm(beta=0)`；half 基线转 f32。
  较大的 CuPy 加速比不表示相对原生 cuSPARSE 内核有同等提升。
- CuPy SpVV half 使用 f32 gather/dot，c32 conj 使用 gather/vdot，
  int8 使用 int32 gather/product/sum，避免 int8 dot 溢出。
  Axpby 使用 dense 缩放和唯一索引更新，half 输出、f32 索引乘加，
  三方独立输出；先检查一次更新的精度，再原地循环计时。
- SpGEMM 为 float32/int32、input-mode=auto：
  GL7d14 使用 A@A.T，其余使用 A@A；不启用 adaptive loops。
  沿用该脚本 prepared topology/数值计算边界。
  其 CSV 的 `cusparse_ms` 实际来自 CuPy CSR `A @ B`，不是直接 cuSPARSE 调用。
- CuPy 结果先检查 CPU oracle 或原脚本的输出对比，再纳入加速比。
  Python/Triton 和 C API 是不同路径，不混用时间。benchmark 与精度测试须串行、独占 GPU。

## 4. 实现要点

- CSR/COO f32 transpose、c32 conj SpMV：prepare 建立转置拓扑，分组 gather 替代逐次
  转置重建/atomic scatter。值排列按 Torch version counter 刷新，inference tensor
  直接间接读取实时 values；外部绕过 version counter 的更新后须重新 prepare。
- CSR non f16/c32、COO non c32 SpMV 使用 row-owned batched gather；
  显式算法/配置保持原路由，metadata 反映实际 implementation/compute type。
- CSC c32 SpMV 使用准备好的行拓扑，去掉复数 atomic scatter 的额外清零/copy。
  CSC f16 SpMV 分组直接读取列指针，去掉每次 repeat_interleave；f32 累加后转换输出。
- CSR f32/f16 SpMM 的 CUDA 短行、width=32 采用 multi-row kernel，
  half 乘法/累加为 f32，直接写 half，避免 B/C 中间升降精度缓冲。
- CSR/COO c32 SpMM width=32 批量 gather；COO 保留高精度 canonical 数据，
  throughput kernel 使用 f32 分量。BaseAccuracy 和其他后端保持原路由。
- COO int8→int32 SpMM 在 CUDA 每个 program 批量处理 16 个非零元素，支持无序 COO、
  重复坐标及 B/C strides，保留精确整数 atomic 累加和完整调用计时。
  CuPy 平均由 0.329× 升至 0.889×；其他后端保留原实现。
- SELL 保持调用者的 slice_size，合并实数 slice 调度，优化复数 warp/slot 归约，
  移除多余 reshape；不能修改 slice_size 来重新解释输入。
- c32 SpVV 打包实部/虚部 partial，将最终归约融合为一个 Triton kernel，
  单 block 直接写结果；非 CUDA 保留旧路径。
- SpGEMM runner 修复 matrix-worker 分派和无效参数，使子进程能完成计算。
  **性能仍未达标**，不把 runner 修复当成性能优化完成。

### C API 独立记录

CSR f32 transpose SpMM 在调用者提供 workspace 时，preprocess 保存转置指针、
nnz 排列和源行，CUDA gather 融合 alpha/beta，不缓存数值。
拓扑构建含 CPU readback，不计入稳态；空 workspace 使用原 atomic 路由。
不均匀行 fallback 按 nnz 分块减少空程序；opB、行/列布局及 alpha/beta 保持不变。

历史 C API 10 矩阵精度通过，25 项 SpMM 精度回归通过。
其初轮 CuPy 平均 1.257× / PyTorch 0.486× 的基线口径早于本轮 RHS 布局转换移出计时，
仅作历史记录，不混入上面的 Python/Triton 45 项验收。
4×4 worker smoke 只验证分派，不作为性能提升证据。

## 5. 数据来源与验证

| 数据批次 | 变体 / 精度案例 | 有效数据 |
|---|---|---|
| 低速优化及回归目标 | 22 项 / 214 PASS | [JSON](results_cuda_q4_allslow_20261006_final/results.json)、[CSV](results_cuda_q4_allslow_20261006_final/performance.csv) |
| 原历史缺失变体 | 4 项 / 40 PASS | [JSON](results_cuda_q4_missing4_20261006_final/results.json)、[CSV](results_cuda_q4_missing4_20261006_final/performance.csv) |
| 缺失 CuPy 基线 | 16 项 / 136 PASS | [JSON](results_cuda_q4_missing16_20261006_complete/results.json)、[CSV](results_cuda_q4_missing16_20261006_complete/performance.csv) |
| SpGEMM 额外复测 | 1 项 / 10 PASS | [CSV](results_cuda_q4_spgemm_20261006/performance.csv) |
| CSC f32/c32 SpMM 历史 | 2 项 / 各 10 矩阵 | [CSV](../../pytest_results_cuda_q4_20261002_170944/spmm_csc/performance.csv) |
| C API 历史独立复测 | 不计入 45 项 | [JSON](results_cuda_q4_baselines_20261006_final/results.json) |
| C API worker smoke | 不作为性能结果 | [JSON](results_cuda_q4_opt_20261006/spmm_benchmark.json) |

不同阶段的精度回归有重叠，不能直接把用例数相加：
22 项阶段 977 passed；4 项阶段 535 passed；补齐 CuPy 阶段 319 passed。
后者包含 6 个新增基线验证用例，4 项阶段包含 90 个 mixed COO 边界用例。
历史目录被 .gitignore 排除，检索时须使用 `rg --files --no-ignore`。

`initial`、`gather`、`v1/v2/v3`、`after_v1/after_v2`、`missing17_vectors` 等目录为中间实验，
不能作为最终验收；`missing16_final` 是命名早于类型修复的中间结果，
有效数据必须使用 `missing16_complete`。原始数据及其他后端归档未删除。

## 6. 复现与待办

重跑时使用新输出目录，避免覆盖已归档的验收结果。

```bash
# 默认 22 项优化目标
PYTHONPATH=src:. python -u tools/benchmark_q4_baselines.py \
  --output results_cuda_q4_retest_22

# 同规模单项复测；--variants 可传逗号分隔的完整变体名
PYTHONPATH=src:. python -u tools/benchmark_q4_baselines.py \
  --variants spmm_coo_i8i32_int_non_non_row \
  --output results_cuda_q4_retest_coo_i8

# 最新基线与 Q4 精度回归
PYTHONPATH=src:. python -m pytest -o addopts='' \
  tests/pytest/test_q4_cupy_baselines.py tests/pytest/test_q4_variants_accuracy.py \
  tests/pytest/test_q4_perf_routes.py -q

# C API 独立验证
cmake --build capi/build -j4
capi/build/ctest/accuracy/test_spmm
```

SpGEMM 按相同 10 文件重新测试：

```bash
PYTHONPATH=src:. python -u tests/test_spgemm.py \
  tests/data/ASIC_680ks.mtx tests/data/GL7d14.mtx tests/data/NACA0015.mtx \
  tests/data/amazon0601.mtx tests/data/auto.mtx tests/data/cage12.mtx \
  tests/data/cfd2.mtx tests/data/filter3D.mtx tests/data/roadNet-TX.mtx tests/data/wave.mtx \
  --dtype float32 --dtypes float32 --index-dtype int32 --warmup 5 --iters 20 \
  --input-mode auto --csv results_cuda_q4_retest_spgemm/performance.csv
```

待办：优化 SpGEMM 至逐算子 CuPy 平均 ≥0.8；按相同口径更新两项历史 CSC SpMM；
DCU/MACA/MUSA 等后端需在对应设备复测，不能用 CUDA 通过结果宣称跨后端达标。
