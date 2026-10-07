# 天数智芯 BI-V150 debug

`FLAGSPARSE_BACKEND=iluvatar`（通常能自动识别）。环境、交付复现见 [../ILUVATAR.md](../ILUVATAR.md)，
上机排查记录见 [../ILUVATAR_DEBUG.md](../ILUVATAR_DEBUG.md)。

当前 debug 入口使用 Python runner：
`python3 tools/run_backend_tests.py --backend iluvatar --phase both --mode normal`。
Iluvatar 的 C API `IX` profile 尚未可构建，调试时不要运行 C API CTest。

## 1. 这个后端的特点

- 以 CUDA 兼容栈接入（CoreX 4.4.0，`torch.version.cuda` 显示 10.2）。
- 实测指纹：设备名 `Iluvatar BI-V150 OAM`，**warp 大小 64**，16 个 MP，
  Triton 目标 `GPUTarget(backend='corex', arch=71, warp_size=64)`。
- 因为目标不是 `cuda`/`hip`，`spmv_csr` 只能用三条 legacy 路由；新路由会直接报不支持。
- 厂商基线是 CoreX 的**旧版** `cusparseScsrmv` / `cusparseScsrmm`（通过 CuPy 调用），
  只支持 fp32 + int32 + 不转置；通用 `cusparseSpMV` 在这张卡上会申请约 140 TB 显存，不可用。

## 2. 已知问题

| 问题 | 状态 |
|---|---|
| fp64 的 H2D 拷贝静默返回全 0（fp64 各类报错的共同根因） | 已定位；q4 不含 f64 / c64，理论上不受影响 |
| `torch.sparse` 在这张卡上结果是错的 | 不要用它当参考；精度参考值在 CPU 上算 |
| 容器里 torch 没注册到解释器，子进程会丢 torch | 用 `corex.pth` 指向厂商 dist-packages；`python3 -I -c "import torch"` 验证 |
| CoreX 版本和宿主驱动必须一致（4.5.0 镜像配 4.4.0 驱动会让 `cudaMalloc` 返回 801） | 以 `ixsmi` 能显示 CUDA 版本为准 |

## 3. q4 变体的风险点

> ⚠️ 2026-10-05：q4 清单已从 42 条改成 45 条（删 3 加 6，见 [README.md](README.md) 第 0 节）。
> 下面这节写的是旧 42 条清单的风险点。新增的 6 个变体（`sddmm_csr_f16/c32_int_non_non_row`、
> `spmm_csc_c32/f16_int_non_non_row`、`spmm_coo_i8i32_int_non_non_row`、
> `spgemm_csr_f32_int_non_non`）**在天数 BI-V150 上一次都没跑过**，只在 CUDA 验证过——这 6 个
> 里有 f16/i8i32/c32，和下面已经点出的"int8/f16/复数都是未知数"是同一类风险，上机时应该和旧
> 42 条一起测，不要漏掉。另外 capi 侧（spvv/axpby/spmv_sell/spmm_csc 新算子组、spmv/spmm 的
> transpose 支持）这段时间进展很快，但天数的 C API `IX` profile 尚未可构建（见本文件开头），
> 这部分和 capi 完全无关，不受影响。

- **int8 是最大的未知数**：这张卡的 Triton 对 int8 读、int32 累加、int32 原子加支持到什么程度没有测过。
  涉及 `scatter_i8_int`（`gather_i8_int` 已移出 q4 范围，见上面第 0 条）、`spmv_csr_i8i32`、
  `spmv_csr_i8f32`、`spmv_coo_i8i32`、`spmv_sell_i8i32`、`spmm_csr_i8i32`、`spvv_i8i32`，以及
  新增的 `spmm_coo_i8i32_int_non_non_row`，共 8 个。
- **f16**：`spmv_csr` 的 f16 在这张卡上实测通过过；其余 f16 变体待测。
- **复数**：c32 各变体没在这张卡上验证过。
- **基线**：CoreX 旧接口只覆盖 fp32 + 不转置，所以 q4 的绝大多数变体在这里没有厂商基线，
  用 `tools/baseline_bound.py --vendor-card iluvatar-biv150`（1150 GB/s）判定。
- 首轮固定 `FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse`（**不要用 `torch`**：BI-V150 上 PyTorch 稀疏静默返回全零），一次只变一个变量。

## 4. 2026-10-06：45 个 q4 变体复测

本轮使用当前 checkout（`q4` 分支）和 BI-V150 OAM，环境为 CoreX 4.4.0、Triton
3.6.0、PyTorch 2.7.1、`warp_size=64`。精度参考固定为 CPU SciPy；性能基线固定为
`FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse`，没有把 BI-V150 上错误的
`torch.sparse` 结果当作基线。

### 4.1 精度

命令：

```bash
FLAGSPARSE_BACKEND=iluvatar \
FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse \
PYTHONPATH="$PWD/src" \
python3 -m pytest -q tests/pytest/test_q4_variants_accuracy.py
```

结果：**89 passed, 2 failed, 5 warnings, 22.43 s**。即 45 个变体的两个 shape
用例共 90 个，**44/45 个变体通过**；失败只属于新增的
`spgemm_csr_f32_int_non_non`，两个 shape 都在 Triton 编译阶段失败，错误为
`iluvatar TLE alloc does not support nv_mma_shared_layout=True`，不是数值误差。
其余 44 个变体（包括全部 int8、f16、c32、转置和 conj 组合）均通过。

### 4.2 真实性能

首轮 runner 使用 5 次 warmup、20 次计时，输入为 `tests/data`；该目录中用于真实
矩阵统计的是以下 10 个 `.mtx` 文件，排除了仅用于冒烟的 `q4_worker_smoke.mtx`：
`ASIC_680ks`、`GL7d14`、`NACA0015`、`amazon0601`、`auto`、`cage12`、`cfd2`、
`filter3D`、`roadNet-TX`、`wave`。首轮完整结果和日志：

```text
results_iluvatar_q4_45_20261006_perf/
iluvatar_q4_45_20261006.performance.log
```

首轮命令（父算子 sweep）：

```bash
FLAGSPARSE_BACKEND=iluvatar FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse \
PYTHONPATH="$PWD/src" python3 run_flagsparse_pytest.py \
  --phase performance --mode normal \
  --ops scatter,axpby,spmv_sell,spmv_csr,spvv,spmm_csr,spmv_csc,spmv_coo,spmm_csc,sddmm_csr,spmm_coo,spgemm_csr \
  --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 \
  --timeout 3600 --results-dir results_iluvatar_q4_45_20261006_perf
```

由于 `spmv_csr`/`spmv_coo`/`spmv_csc` 的父级 full dtype sweep 会先遇到 BI-V150
的 fp64/厂商基线限制，q4 专用行又补跑了一次，结果保存在
`results_iluvatar_q4_45_20261006_perf/q4_direct_missing.json`。最终按变体汇总如下：

| 范围 | PASS | FAIL | 无数据 | 说明 |
|---|---:|---:|---:|---|
| 45 个 q4 变体性能 | 43 | 1 | 1 | `spmm_csc_f32_int_non_non_row` 的父级 sweep 有 fp64 abort；`spmm_csc_c32_int_non_non_row` 的 q4 性能路由漏注册 |
| 45 个 q4 变体精度 | 44 | 1 | 0 | 失败为 `spgemm_csr_f32_int_non_non` 编译限制 |

有性能数据的 43 个变体中，所有实际 q4 行的状态均为 `PASS`；没有把父级因 fp64
而失败的汇总状态直接投影成 q4 失败。性能 CSV 的逐矩阵原始数据在各算子子目录的
`performance.csv`，专用补跑的逐矩阵 JSON 在上面的 `q4_direct_missing.json`。

### 4.3 本轮结论和限制

- int8、f16、c32、转置、conj 组合均已在真实矩阵上跑通精度；性能可用的这些组合也均为 PASS。
- BI-V150 的 fp64 H2D 静默归零、旧版 cuSPARSE 仅支持 fp32/int32/non，以及 Triton
  TLE 共享内存布局限制仍然存在；因此父级 full sweep 的 FAIL 不能直接等同于 q4 失败。
- `spmm_csc_f32_int_non_non_row` 的 f32 真实矩阵行本身通过；当前 FAIL 是父级 full sweep
  被 fp64 `AtomicLoadFAdd` 编译 abort 连带标记，不是 f32 correctness 失败。单独复测时应
  限定 `float32,int32,op=non,layout=row`，避开 BI-V150 的 fp64 路径。
- `spgemm_csr_f32_int_non_non` 当前应标记为 **精度不可用/待 CoreX 编译器修复**；
  `spmm_csc_c32_int_non_non_row` 在统一 q4 runner 中仍应标记为 **性能未配置**，不要填入
  虚假的延迟或加速比；但下面的独立 c32 路径复测已证明内核本身可运行。
  这不是 c32 精度或 CoreX 内核失败：精度用例已通过，但
  `tests/q4_variant_bench.py` 的 `VARIANTS["spmm_csc"]` 当前只注册了
  `spmm_csc_f16_int_non_non_row`，没有注册 c32 变体，所以性能 CSV 中没有 c32 行，
  runner 才报告 `the benchmark recorded no c32 rows`。

### 4.4 `spmm_csc_f32_int_non_non_row` 单独复测

为排除父级 fp64 sweep 的影响，单独执行了：

```bash
FLAGSPARSE_BACKEND=iluvatar FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse \
PYTHONPATH="$PWD/src" python3 tests/test_spmm_csc.py tests/data \
  --csv-csc results_iluvatar_spmm_csc_f32_20261007/performance.csv \
  --dtypes float32 --index-dtypes int32 --ops non --layout row \
  --warmup 5 --iters 20 --timing
```

结果为 **10/10 个真实矩阵 PASS**（另有 1 个 `q4_worker_smoke.mtx` PASS），没有
fp64 编译错误。10 个真实矩阵的 FlagSparse `ms` 平均 **1.2114 ms**，最小
`wave.mtx=0.4897 ms`，最大 `NACA0015.mtx=3.7253 ms`。本次 CuPy/cuSPARSE CSC
基线仍不可用，因此没有合法的加速比：PyTorch 原生 CSC SpMM 不支持，CuPy/cuSPARSE
CSC 在当前 CoreX 环境也未成功接入，CSV 中的 `pytorch_ms` 和 `cusparse_ms` 均为空。
不能用 correctness reference 的 COO 转换耗时冒充 CSC 基线。CSV、stdout、stderr 在
`results_iluvatar_spmm_csc_f32_20261007/`。

### 4.5 参数问题

- 原来的父级命令使用 `--dtypes float16,float32,float64,complex64,complex128`
  和 `--ops non,trans,conj`，这是完整能力扫描参数，不适合直接作为 BI-V150 的
  q4 f32 结论。它会在已经完成 f32 行之后继续进入 fp64 路径，并触发
  `AtomicLoadFAdd<f64>` 编译 abort；runner 因父进程退出码 `-6` 将 q4 继承状态记为 FAIL。
- q4 变体的实际参数契约是 `dtype=float32`、`index_dtype=int32`、`op=non`、
  `layout=row`。单项验证应使用 `--dtypes float32 --index-dtypes int32 --ops non
  --layout row`，并保留 `--warmup 5 --iters 20 --timing`，不要把 fp64、trans、conj
  混进这个结论。
- `FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse` 只指定“尝试使用 CuPy/cuSPARSE 基线”，
  不是强制生成基线；当前 CoreX CSC 路径探测结果仍是 unavailable，所以
  `pytorch_ms`/`cusparse_ms` 为空，不能计算 speedup。
- `tests/data` 中还包含 `q4_worker_smoke.mtx`。它用于 worker 冒烟，不属于真实性能
  统计；本轮真实矩阵结论只计 10 个 `.mtx` 文件。

### 4.6 `spmm_csc_c32_int_non_non_row` 单独复测

由于 q4 性能变体表漏注册 c32，使用 CSC 性能脚本直接按该变体契约测量：

```bash
FLAGSPARSE_BACKEND=iluvatar FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse \
PYTHONPATH="$PWD/src" python3 tests/test_spmm_csc.py tests/data \
  --csv-csc results_iluvatar_spmm_csc_c32_20261007/performance.csv \
  --dtypes complex64 --index-dtypes int32 --ops non --layout row \
  --warmup 5 --iters 20 --timing
```

结果为 **10/10 个真实矩阵 PASS**（另有 1 个 `q4_worker_smoke.mtx` PASS），没有
LLVM/Triton 编译错误或 correctness 错误。FlagSparse `ms` 平均 **6.5211 ms**，
最小 `GL7d14.mtx=3.2344 ms`，最大 `NACA0015.mtx=13.4651 ms`。本次仍没有合法
加速比：PyTorch 原生 CSC SpMM 不支持，CuPy/cuSPARSE CSC 基线也不可用，CSV 中
`pytorch_ms`/`cusparse_ms` 为空。结果在
`results_iluvatar_spmm_csc_c32_20261007/`。

这说明：`spmm_csc_c32_int_non_non_row` 是“统一 q4 runner 未配置”，不是“CoreX
无法运行”。后续若要纳入 45 项统一性能统计，需要先在
`tests/q4_variant_bench.py` 的 `VARIANTS["spmm_csc"]` 注册该变体，再由 runner
生成带 `variant` 标签的性能行。

### 4.7 逐变体明细

下表的“性能样本”只数 10 个真实矩阵；`spvv`/`axpby` 是脚本定义的 4 个合成规模。
加速比是可获得 PyTorch/CuPy 基线时的逐样本平均值，`<1` 表示 FlagSparse 较慢，`—`
表示该变体没有可靠的可比基线。`继承`表示 runner 使用父算子的未标记行投影到该 q4
变体；这类行的状态仍以实际性能脚本的 correctness check 为准。

| # | q4 变体 | 精度 | 性能 | 性能样本 | 平均加速比 | 备注 |
|---:|---|---|---|---:|---:|---|
| 1 | `scatter_i8_int` | 2/2 PASS | PASS（继承） | 48 | — | 父脚本未写 variant 标签 |
| 2 | `axpby_f16_int` | 2/2 PASS | PASS | 4 合成 | 0.823x | PyTorch 基线 |
| 3 | `spmv_sell_f32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 4 | `spmv_csr_f16f32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 5 | `spvv_f16f32_int_non` | 2/2 PASS | PASS | 4 合成 | 1.380x | PyTorch 基线 |
| 6 | `spmv_csr_f16_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 7 | `spmv_csr_f32c32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 8 | `spmv_sell_f16_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 9 | `spmm_csr_f16f32_int_non_non_row` | 2/2 PASS | PASS | 10 | 0.496x | PyTorch 基线 |
| 10 | `spmm_csr_f16_int_non_non_row` | 2/2 PASS | PASS | 10 | 0.285x | PyTorch 基线 |
| 11 | `spmm_csr_f32_int_non_non_col` | 2/2 PASS | PASS | 10 | 0.314x | PyTorch 基线 |
| 12 | `spmm_csr_f32_int_non_trans_row` | 2/2 PASS | PASS | 10 | 0.315x | PyTorch 基线 |
| 13 | `spmv_csc_f32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 14 | `spmv_csr_c32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 15 | `spmv_sell_c32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 16 | `spvv_c32_int_conj` | 2/2 PASS | PASS | 4 合成 | 0.698x | PyTorch 基线 |
| 17 | `spmv_coo_f16f32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 18 | `spmm_csr_c32_int_non_non_row` | 2/2 PASS | PASS | 10 | 1.127x | PyTorch 基线 |
| 19 | `spmv_coo_f32_int_trans` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 20 | `spmv_csr_f32_int_trans` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 21 | `spmv_csr_i8f32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 22 | `spmv_csr_i8i32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 23 | `spmv_sell_i8i32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 24 | `spvv_i8i32_int_non` | 2/2 PASS | PASS | 4 合成 | 1.422x | PyTorch 基线 |
| 25 | `spmm_csc_f32_int_non_non_row` | 2/2 PASS | PASS（单项）/FAIL（父扫） | 10 | — | 单项 f32 全部 PASS；父级跑 fp64 时触发 `AtomicLoadFAdd<f64>` 编译 abort，退出码 `-6` |
| 26 | `spmm_csr_i8i32_int_non_non_row` | 2/2 PASS | PASS | 10 | 0.452x | PyTorch 基线 |
| 27 | `spmv_coo_c32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 28 | `spmv_coo_f16_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 29 | `spmv_csc_c32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 30 | `spmv_csc_f16_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 31 | `sddmm_csr_f32_int_non_non_col` | 2/2 PASS | PASS | 10 | — | 无可靠 PyTorch 稀疏基线 |
| 32 | `sddmm_csr_f32_int_non_trans_row` | 2/2 PASS | PASS | 10 | — | 无可靠 PyTorch 稀疏基线 |
| 33 | `sddmm_csr_f32_int_trans_non_row` | 2/2 PASS | PASS | 10 | — | 无可靠 PyTorch 稀疏基线 |
| 34 | `spmm_coo_c32_int_non_non_row` | 2/2 PASS | PASS | 10 | 2.855x | PyTorch 基线 |
| 35 | `spmm_coo_f16_int_non_non_row` | 2/2 PASS | PASS | 10 | 0.477x | PyTorch 基线 |
| 36 | `spmm_csr_f32_int_trans_non_row` | 2/2 PASS | PASS | 10 | 0.056x | PyTorch 基线 |
| 37 | `spmv_coo_c32_int_conj` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 38 | `spmv_coo_i8i32_int_non` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 39 | `spmv_csr_c32_int_conj` | 2/2 PASS | PASS | 10 | — | 无可用厂商基线 |
| 40 | `spmm_coo_i8i32_int_non_non_row` | 2/2 PASS | PASS | 10 | 0.085x | PyTorch 基线 |
| 41 | `spmm_csc_c32_int_non_non_row` | 2/2 PASS | 无数据 | 0 | — | 当前脚本没有 c32 CSC 性能路径 |
| 42 | `sddmm_csr_c32_int_non_non_row` | 2/2 PASS | PASS | 10 | — | 无可靠 PyTorch 稀疏基线 |
| 43 | `sddmm_csr_f16_int_non_non_row` | 2/2 PASS | PASS | 10 | — | 无可靠 PyTorch 稀疏基线 |
| 44 | `spmm_csc_f16_int_non_non_row` | 2/2 PASS | PASS | 10 | 0.246x | PyTorch 基线 |
| 45 | `spgemm_csr_f32_int_non_non` | 0/2 FAIL | PASS（继承） | 22 | — | 精度编译失败：TLE 不支持 `nv_mma_shared_layout=True` |
