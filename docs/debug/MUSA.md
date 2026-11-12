# 摩尔线程 / MUSA（S5000）：65 变体验证任务

`FLAGSPARSE_BACKEND=mthreads`。环境搭建、交付复现见 [`../MUSA.md`](../MUSA.md)。背景见本目录
[`README.md`](README.md)。

**MUSA 实现约束（用户于 2026-10-18 明确）**：算子计算内核必须使用 Triton，允许使用当前后端支持且经实测验证的 TLE 特性。禁止以手写 MUSA C++/`.mu` 内核、mcc 编译的计算库或 PyTorch 原生索引替代被测算子的计算内核。C API 主机封装、设备内存/流管理和厂商对照基线不属于计算内核；其他后端保持原实现。本文后半部保留的原生实验为撤回历史，不能用于全 Triton 验收。

## 2026-11-04：H800 参照扩到 105 变体，重跑补 17 个 NoBaseline；3 条 Python 慢路径待上卡排查

**为什么 11-03 整轮还剩 17 个 NoBaseline。** 那轮已经带了 `--h800-reference`，但仓库自带的
`conf/h800_reference.json` 当时只有 10-14 那轮 H800 的 65 个变体，新 40 个变体在参照里查不到，折算无从谈起。
2026-11-04 用 H800 105 变体全量（`results_cuda_105_20261103_1556`，105/105 精度、性能全过）补了参照：
**原 65 个变体保持 10-14 那轮的数据一行不动**，只补新 40 个（`tools/h800_reference.py --keep-variants-from`）。
其中 `axpby_f32_int`、`axpby_c32_int`、`spvv_f32_int_non` 10-14 那轮已经测过，也保留旧行，只补了变体汇总。
这 17 个变体在 H800 上都有 10 个矩阵的 cuSPARSE 时间，只有 `spmv_coo_f16_int_trans` 的 amazon0601
那一行按规则退用 H800 PyTorch 时间。命令不变，更新代码后照上面"2026-11-02 复测计划"的那一行重跑即可。开跑前先确认参照是新版：

```bash
python3 -c "import json; r=json.load(open('conf/h800_reference.json'))['reference']; print(r['source_dir'], r['supplement']['source_dir'], len(r['supplement']['variants']) + len(r['supplement']['summary_only']))"
# 期望：results_65_cuda_1014 results_cuda_105_20261103_1556 40
```

**重跑后应该看到的数（在 CUDA 机器上用 11-03 整轮的原始 CSV 离线预演，不是实测）。** 补位顺序是
C API 厂商 > Python 那一半对 H800 折算 > C API 时间对 H800 折算。重跑后 Python 那一半会给全部 17 个算出折算
加速比，所以交付表上的数取 Python 那一列。C API 那一列只是参考：SpMM 的 C API 规模是 n=8，H800 是 N=32，对不上，所以是 `-`。

| 变体 | Python 折算 | C API 折算 |
|---|---:|---:|
| spmm_coo_f16f32_int_non_non_row | 1.788 | - |
| spmm_csc_f16f32_int_non_non_row | **0.008** | - |
| spmm_csr_bf16f32_int_non_non_row | 2.014 | - |
| spmm_csr_f16f32_int_non_non_col | 0.289 | - |
| spmm_csr_f16f32_int_trans_non_row | 0.918 | - |
| spmm_csr_i8f32_int_non_non_row | 2.329 | - |
| spmm_csr_i8i32_int_non_non_col | 0.279 | - |
| spmv_coo_f16_int_trans | 6.425 | 0.213 |
| spmv_coo_f32c32_int_non | 0.196 | 0.182 |
| spmv_coo_i8f32_int_non | 0.826 | 0.021 |
| spmv_csc_f16f32_int_non | 1.725 | 0.808 |
| spmv_csc_i8i32_int_non | **0.027** | 0.902 |
| spmv_csr_bf16_int_non | **0.014** | 0.564 |
| spmv_csr_bf16f32_int_non | 0.958 | 0.550 |
| spmv_csr_f16f32_int_trans | 1.051 | 0.854 |
| spmv_csr_f32c32_int_trans | 1.175 | 1.028 |
| spmv_csr_i8i32_int_trans | 1.197 | 0.987 |

这些是**估算**，分母是 H800 cuSPARSE 按带宽 3050/1370 折算后的时间，不是 muSPARSE。
`spmv_coo_f16_int_trans` 的 6.4 是真实结果：cuSPARSE 的 COO 转置在 H800 上本来就慢（0.10–0.31 ms），逐矩阵 0.235–11.5。

### 三条慢路径（加粗的 3 行）

三项都是 MUSA 上 Python 侧比同输入的相邻变体慢 50–300 倍，不是 H800 基线的问题。在 RTX 5090 上
用 `FLAGSPARSE_BACKEND=mthreads` 模拟 MUSA 路由，只有第 1 条能复现出缓存失效，第 2、3 条复现不出来，说明要在 S5000 上看。
都是 Triton 路由选择问题，修法沿用已有的 Triton 路径，不违反文首的全 Triton 约束。

**1. `spmv_csc_i8i32_int_non`（2.1–8.7 ms；C API 同变体 0.902）：每次调用都重做 COO→CSR，已在模拟上确认。**
`spmv_csc.py::_spmv_csc_via_mixed` 里 int8 不进 fp16 那条带缓存的 `spmv_csc_half`，落到最后的 `else`。
那里**每次调用**用 `torch.repeat_interleave` 新建 `col_ids`，再交给 `mixed_spmx.spmv_coo_mixed`。MUSA 分支的
`_musa_coo_csr` 以 `(data, row, col)` 的张量身份为缓存键，`col_ids` 每次都是新张量，所以每次都 miss，
在计时窗口里整份 `torch.argsort` 一遍。模拟上连续 5 次调用 = 5 次 argsort。
修法方向：让 MUSA 的 int8 和 fp16 一样，以 `(data, indices, indptr)` 为键缓存 CSC→CSR，即复用 `spmv_csc_half`
里 `_MUSA_CSC_CSR_CACHE` 那段，再走 `spmv_csr_mixed`。修完检查：连续调用只在第一次 argsort；精度回归是
`tests/pytest/test_spmv_csc_accuracy.py`。

**2. `spmm_csc_f16f32_int_non_non_row`（37–134 ms）：落进 fp32 非转置 CSC 通用路径，这条路径在 MUSA 上本身就慢。**
`spmm_csc.py::flagsparse_spmm_csc` 开头的 fp16 分支里，MUSA 专用快速路径（`_MUSA_CSC_HALF_CACHE` 缓存
CSC→CSR，再调 `mixed_spmx.spmm_csr_mixed`）要求 `result_dtype == torch.float16`。f16→f32 不满足，
退到"整体转 fp32 再调一遍 `flagsparse_spmm_csc`"的通用路径。11-03 整轮原始 CSV 里同一路径的另外两个变体也一样慢：

| 矩阵 | f16（MUSA 快速路径） | f32 非转置 | f16f32 | c32 非转置 | f32 转置 |
|---|---:|---:|---:|---:|---:|
| cage12 | 0.12 ms | 41.5 ms | 41.5 ms | 90.1 ms | 0.86 ms |
| NACA0015 | 0.58 ms | 128.6 ms | 128.7 ms | 288.0 ms | 1.85 ms |

f32/c32 的交付数取的是 C API（muSPARSE 基线），所以表上看不出来，但 Python 侧这条路径和转置方向差了约 70 倍。
排查顺序：
- 先在卡上打印 f32 非转置调用实际选中的 `alg` 和 kernel，`fs.prepare_spmm_csc_route(...)` 返回的 prepared 里有 `.alg`。
  看是不是落到了逐 nnz atomic 的那条。
- 再把 MUSA 快速路径的条件放宽到 `result_dtype in (float16, float32)`。`spmm_csr_mixed` 本身支持 fp32 输出，
  要传 `out_dtype`，改完对 f16f32 复测。
- f32/c32 非转置的根因单独记一条，不要只修 f16f32。

**3. `spmv_csr_bf16_int_non`（2.3–8.5 ms；同样是 bf16 输入的 `spmv_csr_bf16f32_int_non` 只要 0.03–0.15 ms）：bf16 没进 row gather。**
`spmv_csr.py::prepare_spmv_csr` 里 MUSA 只对 `torch.float16` 建 row gather plan
（`_is_mthreads_runtime() and data.dtype == torch.float16`），bf16 落到默认的 `segbin` kernel（`_spmv_csr_default_backend`
对 MUSA 返回 `"segbin"`）。bf16f32 走的是 `mixed_spmx` 的另一条路径，所以快。5090 上 bf16 segbin 只有 0.06 ms，
慢是 S5000 上 bf16 segbin 的问题。
排查顺序：
- 在卡上对同一矩阵比较 bf16 的 `alg="segbin"` / `"rowpar"`，以及把条件改成 `data.dtype in (torch.float16, torch.bfloat16)`
  之后的 row gather，三者的时间。
- 选最快的一条，然后用 `tests/pytest/test_spmv_csr_accuracy.py` 和 `tests/pytest/test_musa_sparse_routes.py` 回归。

**单变体复测**（每条修完先在 2 个矩阵上看时间，确认快了再跑整轮）：

```bash
FLAGSPARSE_BACKEND=mthreads PYTHONPATH=src python3 - <<'EOF'
import sys; sys.path.insert(0, "tests")
import q4_variant_bench as q
paths = q.matrix_paths(["tests/data/cage12.mtx", "tests/data/NACA0015.mtx"])
for op, names in [("spmv_csc", ["spmv_csc_i8i32_int_non", "spmv_csc_f16f32_int_non"]),
                  ("spmm_csc", ["spmm_csc_f16f32_int_non_non_row", "spmm_csc_f16_int_non_non_row"]),
                  ("spmv_csr", ["spmv_csr_bf16_int_non", "spmv_csr_bf16f32_int_non"])]:
    for r in q.run_variants(op, paths, 5, 20, variant_names=names):
        print(r["variant"], r["matrix"], "%.4f ms" % r["ours_ms"], r["status"])
EOF
# 修好的标准：每对里慢的那个和快的那个在同一量级（>0.8 由整轮的 H800 折算判定）
```

## 2026-11-03：本机 muSPARSE 基线接入

`capi/ctest/baseline/musa/baseline.cpp` 新接入 6 个直接厂商基线变体：

| 变体 | 接口 | 对旧 0712 轮的意义 |
|---|---|---|
| gemvi_c32_int_non | musparseCgemvi | 补 NoBaseline |
| geam_csr_f32_int / geam_csr_c32_int | musparseXcsrgeam_nnz + musparseScsrgeam/Ccsrgeam | 补 2 个 NoBaseline |
| gemvi_f32_int_non | musparseSgemvi | 替代原 PyTorch 对照 |
| axpby_f32_int / axpby_c32_int | musparseAxpby | 替代原 PyTorch / H800 折算对照 |

GEAM 使用本机 legacy **csrgeam** ABI，结构 nnz 和数值计算都包含在每次计时中，
容量不足在写 columns/values 前返回失败。Axpby 双方每次调用前恢复初始 y 并同步，恢复不计时；
基线返回的结果仍对应原始 alpha/beta 和完整 dense y 的语义。
GEMVI transpose 仍明确返回不支持，不能因 bufferSize 成功而宣称存在计算基线。

另两条路线显式准备 CSR：

- `spmm_coo_f32_int_trans_non_row`：主机将 COO(A) 转为 CSR(Aᵀ)，厂商执行 non-transpose SpMM。
- `spmv_csc_c32_int_conj`：CSC(A) 结构直接作为 CSR(Aᵀ)，主机共轭当前值生成 CSR(Aᴴ)，厂商执行 non-transpose SpMV。

这两条不是原格式原生支持。每次 baseline 调用重新读取当前输入，不跨调用缓存；
下载、主机转换、上传、同步的时间单独写入 `baseline_prepare_ms`，不包含在 `baseline_ms` 或加速比中。
原始 JSON 和 summary 的每个 shape 均保留 `baseline_method`、`baseline_timing_scope`、`baseline_prepare_ms`。
若用于端到端对比，需要另计转换成本，不能将本次加速比描述为端到端加速比。
所有厂商名称通过运行时版本查询输出，本机为 **muSPARSE 1.5.1**；4.3.5 是 SDK 版本。

定向复测输出：`results_musa_baseline_integration_20261103/`，8 个变体共 50 条记录。
GEAM 和两条 CSR 准备路线使用 `tests/data` 原 10 个矩阵、原始数值；GEMVI 使用原 4 个合成规模，Axpby 使用原向量。
50/50 基线调用成功，48 条严格通过，ASIC_680ks 上 SpMM / CSC conj 的 2 条为 `pass_relaxed`：
双方严格误差比均约 1.363375，FlagSparse relaxed 比约 0.013634，按原有规范 6.3.1 判定，未改容差。
定向结果的算术平均加速比：

| 变体 | 加速比 |
|---|---:|
| gemvi_f32_int_non | 0.609 |
| gemvi_c32_int_non | 0.467 |
| geam_csr_f32_int | 2.525 |
| geam_csr_c32_int | 2.395 |
| axpby_f32_int | 0.956 |
| axpby_c32_int | 0.958 |
| spmm_coo_f32_int_trans_non_row | 0.207 |
| spmv_csc_c32_int_conj | 0.166 |

性能 `Passed` 只表示按框架精度规则可报告加速比，不能理解为已达到 0.8。
原 19 个 NoBaseline 中可补的 3 个已在本次定向复测通过，其余 16 个窄类型/混合精度不支持的事实不变；
`spmv_csr_f32c32_int_trans` 也仍无直接厂商基线。
这是接入定向验证，**没有重跑或改写旧 105 项整轮结果**。现有 split / 统一 runner 重建后会自动使用这些基线，
厂商优先级、Python/H800 补位规则未变。

修改范围只有测试侧 MUSA 基线、MUSA 宏保护的测试封装、summary 元数据透传、回归测试和本节文档；
没有修改 `src/` 或 `capi/src/` 中的生产 kernel。新增 `accuracy.musa_baseline` 仅在 MUSA 构建注册。

## 2026-11-02 复测计划：105 变体

注册表扩到 105 个，改了什么见 [`README.md`](README.md) 的 2026-11-02 一节。2026-11-03 起所有后端用同一行命令（
`run_flagsparse_delivery.py`，说明见 README 的“统一命令”一节）：它自动识别后端，识别到 MUSA 后走
`run_flagsparse_split_delivery.py --backend mthreads`，带 `--h800-reference`，和原来的 split 命令参数一样。因机
器而异的只有下面的环境准备。

```bash
export MUSA_HOME=<MUSA 工具链目录>   # C API 构建要用，见 ../MUSA.md 第 2 节
setsid timeout -s KILL 43200 python3 -u run_flagsparse_delivery.py --gpus 0 > delivery_105.log 2>&1 < /dev/null &   # 多卡如 --gpus 0,1：pytest 那一半按卡分配算子，C API 那一半用第一张卡
# 日志第一行是 [delivery] backend=… results=<结果目录>；先用 --dry-run 看一眼解析出的命令也行
# 跑完后交付表已经打在日志末尾，也写进了 <结果目录>/delivery_table.txt
```

**C API 没测到的变体由 Python 那一半补位（2026-11-02 起）：**

- split 的性能原本只取 C API 的结果，Python 那一半只补 `NoBaseline`。C API 没实现或没测到的变体会被
  `capi/tools/write_summary.py` 记成 `NotFound`。新 40 个变体一开始有 24 个 C API 没实现，2026-11-02 已在 C API
  里全部补上（见下面"C API 现在覆盖全部 105 个变体"），所以这条补位现在只是兜底：MUSA 上某个新路由报
  `not_supported` 或者没出行时才会用到。
- 现在 `run_flagsparse_split_delivery.py::fill_unmeasured_from_python` 会先处理这类变体：C API 一行都没有
  （`NotFound`），或者行全是 `not_supported`（`Skipped`、没有任何耗时明细），就改用 Python 那一半的性能结果。
  不管有没有加速比、是不是 FAIL，只要 Python 真的测了就用。来源标成 `source: python`，原来的 C API 条目留在
  `capi_performance`。C API 测到了的变体完全不动。
- Python 那一半只有在带 `--h800-reference` 时才跑性能，所以上面的命令里这个参数不能省。省掉的话，这些变体
  仍然是 `NotFound`。
- 日志里会多一行 `variants the C API did not measure: filled from the Python half=…, still unmeasured=…`。
- 补上的数字来自 Python 侧的基准，规模和 C API 不同。新 40 个在 H800 上没有数据，MUSA 上 `torch.sparse` 又没有
  矩阵乘，所以这些变体大多是"性能 Passed、没有加速比"。这是实情，照实报。

**C API 现在覆盖全部 105 个变体（2026-11-02，CUDA 上完成）：**

- 40 个新变体全部在 C API 里有了路由，`capi: true`，`CAPI_FALSE_IDS` 为空。新增两个公开接口族：
  `flagsparseGemvi[_bufferSize]`（对应 `cusparse<t>gemvi`）和 `flagsparseCsrgeam2_bufferSizeExt` /
  `flagsparseXcsrgeam2Nnz` / `flagsparseCsrgeam2`（对应 `cusparse<t>csrgeam2`，去掉了 `cusparseMatDescr_t` 参数），
  对应的 ctest 族是 `gemvi`、`geam`，split 会自动带上。
- 在 CUDA（RTX 5090）上都测过、精度通过。**MUSA 上一个都没跑过**，下面这些是新代码：
  - SpMV 的混合精度转置、CSC 混合、COO 的 `i8f32` / `f32c32`：Triton 原子散加（`mixed_spmx.py` 的
    `_csr_spmv_trans_atomic_kernel` 等），f16/bf16 输出先写 f32 scratch 再转回。
  - SpMM：CSC 转置当作 Aᵀ 的 CSR 走 `run_csr`；COO f32 转置在 CUDA 走转置拓扑 gather，**其他后端走原子散加**；
    c32 的 `trans` / `conj_trans` 在 preprocess 里建 Aᵀ 的 CSR，再用 `_permute_values_kernel` 搬值（共轭时取反虚部）；
    混合精度 CSR 转置 / CSC 用 `_csr_spmm_trans_atomic_kernel`。
  - **MUSA 的 gather 判定改过**：`indirect` 从只看 CSC 改成 `(CSC && !trans) || (CSR && trans)`，原因是 CSC 转置现在
    走 CSR 视图。这改动只在 CUDA 上验证了编译和其他后端路径，MUSA 上要重点确认 CSC / CSR 转置的 SpMM 精度。
  - gemvi / geam / spvv f32、i8f32 / axpby f32、c32 / spgemm c32 都是新 kernel 或新分支。
- **基线**：gemvi 和 csrgeam2 是 cuSPARSE 的 legacy 接口，只在 `ctest/baseline/cuda/baseline.cpp` 接了
  （`FS_SP_LEGACY_GEMVI_CSRGEAM2`）。MUSA 的基线文件里是桩，返回 "legacy ... not wired"，所以这 5 个变体在 MUSA 上
  是 `NoBaseline`，由 Python 那一半补基线。muSPARSE 有没有这两个接口没查过。
- C API 的 spgemm 在 compute 里只在设备上算结构，值在 `copy()` 里在主机上算，计时不含值。c32 也是这样。

**这个后端的风险点（针对新增的 40 个）：**

- **全 Triton 约束**：
  - gemvi、geam 和混合精度转置用的都是 Triton kernel。MUSA 上 COO 的转置和"实 × 复"不走 `_musa_coo_csr`，改走通用的
    Triton 原子散加。
  - spgemm c32 和 f32 一样走 ESC（MUSA 不在 `tle_hash_safe` 里），ESC 用的是 torch 的排序和归约。这一点是否满足 10-18 的
    约束，和 f32 的 spgemm 一样，需要你判断。
- **`tl.atomic_add`**：散加 kernel 在 MUSA 上用到 f32 和 int32 的原子加，以前 MUSA 的混合精度路径没用过，属于新的风险点。
- **`torch.sparse` 没有矩阵乘**：geam（CSR 加法）、spgemm（CSR 乘法）的 PyTorch 基线在 MUSA 上会失败，写进
  `pytorch_reason`。gemvi 的 PyTorch 基线是稠密运算，不受影响。
- **复数索引**：geam_c32、gemvi_c32 的 kernel 都通过 `view_as_real` 访问数据（spgemm 的 ESC 当时漏了，见下面 11-03 一节）。q4 基准构造 Aᵀ 时用的是
  `_common._gather_values`，不会碰到 `IndexMusa` 对 ComplexFloat 的限制。

**跑完要确认：**

1. 精度那一半：105 个变体的精度是不是都 Passed（精度由 pytest 产出，不受上面 C API 缺口的影响）。
2. 日志里 `NoBaseline variants filled from the Python half=…, from the C API time=…, still without a baseline=…` 这一行
   原样贴回。
3. 日志里 `variants the C API did not measure: filled from the Python half=…, still unmeasured=…` 这一行原样贴回。
   按设计 `still unmeasured` 应该是 0，不是 0 就列出是哪些变体。
4. 40 个新变体在 C API 里各自是什么状态（Passed、NoBaseline 还是由 Python 补位）。合并后的
   `summary_split.json` 里看 `capi_performance` 字段就知道。有 `not_supported` 或精度失败的，贴 C API 的 detail。
5. 散加 kernel 和 spgemm c32 有没有报错或卡死。

**2026-11-03 那一轮（`results_mthreads_105_20261103_0509`，HEAD c8993be）的结果和修复：**

精度 104/105，性能 56 Passed、39 Failed、10 NoBaseline。39 个 Failed 里 SpMV 占 35 个，它们是一个基线 bug 连带出来的，
不是 SpMV 本身坏了：

- **muSPARSE 的 CSC conj 基线写越界**（已修，`capi/ctest/baseline/musa/baseline.cpp::csc_as_csr_operation`）。CSC(A)
  是按 CSR(Aᵀ)=M 交给 muSPARSE 的，非转置和转置互换没问题，但 conj 原样传了下去，于是算的是 Mᴴx = conj(A)x。
  - 方阵上只是算错：ASIC_680ks 上的误差比是 6.9e7，结果这个"厂商也没过"让我们拿到了一个不该给的 pass_relaxed。
  - 长方矩阵 GL7d14（171375×47271）上，它往长度 47271 的 y 里写了 171375 个值，越界写坏了显存，从这一行起整个
    SpMV 进程后面的行几乎都错（误差比约 1/tol，"INVALID_VALUE"）。
  - 10-19 那轮没有 `spmv_csc_c32_int_conj`，所以当时没触发。
  - 现在 complex 的 CSC conj 直接拒绝并写明原因（通用 API 表达不了不带转置的共轭）；实数的 conj 按 trans 处理。
- **spgemm c32 精度 Failed**（已修，`spgemm_csr.py` 的 ESC）：torch_musa 没有 complex 的索引 kernel，报
  `"IndexMusa" not implemented for 'ComplexFloat'`。上面"复数索引"那条只查了 geam/gemvi，漏了 ESC。现在 ESC
  里复数的取值、乘法和重复项累加都在 `view_as_real` 视图上做。在 CUDA 上和 SciPy 对照过，complex64、f32、f64 都正确，
  MUSA 上还没跑过。
- 修完以后还剩下的（需要在 MUSA 上复测确认）：
  - `spmm_coo_f32_int_trans_non_row`：ASIC_680ks 上误差比 1.36，没有 muSPARSE 基线，判不了放宽。
  - `spmm_csr_c32_int_conj_non_row`：ASIC_680ks 上 1.36，muSPARSE 是 0.07，属于我们自己的精度差距。
  - 10 个 NoBaseline：混合精度 SpMM 7 个、gemvi_c32、geam f32/c32。

## 要做的事

环境按 `../MUSA.md` 第 2 节配好（`FLAGSPARSE_BACKEND=mthreads`、`MUSA_HOME`）后，跑一轮 split：

```bash
# 1. 更新代码：至少要包含 a330b62（见下面"2026-10-14 那轮的两个 NoBaseline"）
git pull --ff-only
git log --oneline -1
git merge-base --is-ancestor a330b62 HEAD && echo "包含 a330b62 修复"

# 2. 环境自检
export PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads MUSA_HOME=/usr/local/musa   # MUSA_HOME 以本机实际安装位置为准
python3 -c "import flagsparse; print(flagsparse.__file__)"   # 必须是 <仓库>/src/flagsparse/__init__.py
python3 -c "from flagsparse.sparse_operations import _common as c; print(c._backend_name(), c._accel_device_type(), c._accel_fallback_reason())"
# 期望：mthreads musa None
ls tests/data/*.mtx | wc -l          # 期望 10
mthreads-gmi                         # 确认没有别的任务在用 GPU

# 3. 全量 split（每次用新的结果目录）
D=results_musa_split_65_<日期>
setsid timeout -s KILL 43200 python3 -u run_flagsparse_split_delivery.py \
  --mode normal --benchmark-input tests/data --timeout 3600 \
  --h800-reference \
  --results-dir $D \
  > $D.log 2>&1 < /dev/null &

# 4. 跑完后
tail -5 $D.log                                    # H800 折算三类各补了多少、还剩哪些
python3 tools/delivery_table.py $D                # 期望 0 missing，性能里没有 NoBaseline
python3 tools/delivery_table.py $D --show-baseline | grep -E "scatter_i8_int|spmm_csc_f16"
# 期望这两行有加速比，来源 h800_scaled
```

- `flagsparse.__file__` 那一步不能省：系统 site-packages 里残留的旧安装会顶替 `src/`，基线悄悄变成 `N/A`。
- `MUSA_HOME` 设错不会报错，muSPARSE 基线同样会悄悄退成 `none`（`../MUSA.md` 第 2 节）。

**2026-10-14 那轮的两个 NoBaseline**（`results_musa_split_65_20261014`，63 Passed + 2 NoBaseline）：
`scatter_i8_int` 和 `spmm_csc_f16_int_non_non_row` muSPARSE 都不支持（C API 报 `Scatter dtype unsupported`、
`unsupported SpMM type or operation`），只能靠 Python 侧的折算补，而那轮 Python 侧这两处都崩了：scatter int8 的
torch 参考被 muDNN 拒绝（`Now muDNN only support float64, float32, half, bfloat16, int32, and int64`，4 行全是
`ERROR`）；spmm_csc 的 `torch.sparse.mm` 参考在 MUSA 上没有实现（`Could not run 'aten::addmm' with arguments from
the 'Sparsemusa' backend`，CSV 为空）。`a330b62`（参考改在 CPU / SciPy 上算）修了这两处，但它比那轮结果晚提交，
所以那份报告里的这两行是旧代码的结果，按上面的命令重跑即可。CUDA 上以 `FLAGSPARSE_BACKEND=mthreads` 模拟已确认
两行都能测出性能；折算这一步只在真机上走得到。

- 精度那一半调用 `run_flagsparse_pytest.py --phase accuracy --delivery-only`，覆盖全部 65 个变体。
- 性能那一半取 C API 的 `ctest`（对 muSPARSE），**现在也覆盖全部 65 个变体**：`origin/q4` 的 C API 工作
  2026-10-08 已移植进 main（见下面"C API 现在覆盖全部 65 个变体"一节），每个变体按自己的 id 单独测一次。
  muSPARSE 不支持的类型组合（fp16、int8、混合精度、SELL 等）显示 `NoBaseline`——跑通了、精度过了，只是没有
  厂商对照，**这是预期的**，不是失败。
- `--h800-reference`（不带值）：muSPARSE 不支持、标 `NoBaseline` 的变体（2026-10-09 那轮是 26 个），用仓库自带的
  H800 结果按带宽折算出加速比（H800 时间 × 3050/1370），交付表里只显示数字（`--show-baseline` 可看来源）。**muSPARSE 测得到的
  变体不受影响。** 加了这个开关，pytest 那一半会从只跑精度变成**精度 + 性能**（多一轮 Python 性能，耗时会明显变长），
  因为 C API 的规模（SpMM `n=8`、SDDMM `k=16/64`、向量类按矩阵）在 H800 上没有对应数据，而 Python 侧和 H800 参照
  是同一个 benchmark、同样规模。合并时按顺序取：① C API 有 muSPARSE 基线 → C API；② Python 侧有折算加速比 →
  Python（条目里 `source: python`，原 C API 结果留在 `capi_performance`）；③ 否则用 C API 时间对 H800 同变体同矩阵
  折算（只有 SpMV 类能对上）。日志会打印三类各补了多少、还剩哪些。`sddmm_csr_f16` 在 H800 上没有 cuSPARSE 时间，
  用 H800 PyTorch 折算。是估算，不是厂商实测，汇报时要注明；见
  [`../H800_REFERENCE.md`](../H800_REFERENCE.md)。
- 拉到新代码后第一次跑**不要加 `--skip-capi-build`**，确保 C API 是重新编译的。
- `--timeout 3600` 沿用 `../MUSA.md` 的结论（`900` 不够），split 会把它同时传给 CTest。CUDA 上 SpMM 这一族
  单独就要 18 分钟（65 个变体里 17 个是 SpMM，3 种宽度 × 10 个矩阵），MUSA 上 SpMM 只测第一种宽度。
- `--benchmark-input tests/data`：`tests/data` 正好是 10 个交付矩阵（在 git 里）；C API 一侧自己会再筛一遍。
- 外层 `timeout -s KILL` 是整条命令的总限时（内核卡死时 Ctrl-C 送不进去，只能靠 KILL）。65 变体**没有在本机
  实测过总耗时**，到点被杀的话按 `NotFound` 的算子单独补跑，补跑用新的 `--results-dir`。

把 `delivery_table.py` 的完整输出带回来。

## 已知风险点（这个后端特有的，不是猜的）

- **`torch.sparse` 在 MUSA 上完全没有矩阵乘实现**（`../MUSA.md` 4.5 节的实测能力矩阵已经写明）。
  这是四个后端里最明确的一条已知限制：凡是依赖 PyTorch 稀疏矩阵乘做对照/回退的路径都会失败，不止
  q4 新变体，原有 20 个变体的 PyTorch 对照列也受影响。`tests/q4_variant_bench.py` 的 `_time_pytorch`
  helper 专门注释了 "e.g. MUSA registers no sparse matmul, no sparse int8"，理论上会优雅跳过、报
  `pytorch_reason` 而不是崩溃——**这一条需要真机确认，不能只信注释**。
- **复数高级索引没有 kernel**：`values[order]` 这类操作在 MUSA 上报
  `"IndexMusa" not implemented for 'ComplexFloat'`。`_common._gather_values` 已经统一改走
  `view_as_real` 拆实部/虚部规避，但这是否覆盖了全部 45 个新变体的复数路径（`spmv_csr/coo/csc_c32`、
  `spmm_csr/coo/csc_c32`、`spvv_c32_int_conj`、`spmv_sell_c32`、`sddmm_csr_c32`）没有逐一验证过。
- **muDNN 的 gemv 不支持 fp64/复数**（gemm 支持）。这条主要影响原有 fp64 变体，新变体里的 fp64 相关
  路径（如果有间接依赖 gemv 的）也要留意。

## 跑完要确认（2026-10-14 之后这一轮）

1. `missing` 是不是 0；65 个变体精度是不是全 Passed（`spmv_coo_f32_int_non` 不应再是 Skipped）。
2. 日志里 `--h800-reference (musa-s5000): NoBaseline variants filled from the Python half=…, from the C API time=…,
   still without a baseline=…` 这一行原样贴回。按设计"still without"应该是 0；不是 0 就列出是哪些变体、为什么。
3. 交付表里还有没有 `NoBaseline`；`python3 tools/delivery_table.py <结果目录> --show-baseline` 的输出原样贴回（最后一列说明哪些加速比是折算的）。
4. 这轮 pytest 那一半第一次跑性能（加了 `--h800-reference` 的代价），报一下总耗时，以及 Python 性能阶段里有没有
   哪个算子超时或崩溃（`<结果目录>/<算子>/performance_stderr.log`）。
5. 复数变体有没有 `IndexMusa not implemented`；有没有慢几百倍的可疑变体。

把这些结果写回来，不用额外分析，原始数据最有用。

## C API 现在覆盖全部 65 个变体（2026-10-08 在 CUDA 上完成移植）

**来源**：这些 C API 实现本来就有。q4 在 MUSA 上做完并实测过（`origin/q4` 分支，2026-10-06），当时没有走
split，而是直接 `ctest -L capi -R '^benchmark\.(axpby|spmv|spmm|spvv|spgemm|sddmm|scatter)$'`，再用 q4 专门的
`capi/tools/write_summary_q4.py` 出一张 45 变体的表（`summary_q4.json`，不进交付 `summary.json`）。main 一直没合
这部分，所以 main 的 split 性能那一半只有 25 个变体。现在合进来了，split 一次就能出全部 65 个，不再需要
`write_summary_q4.py`。

**移植时改了什么**（都在 CUDA 上验证过）：

- `capi/tools/gen_variants.py` 按仓库根 `conf/operators.yaml` 的 65 项给每个变体生成带 id 的条目；q4 原版读的是
  `q4_variants:` 这个键，main 里已经没有了，原样搬过来会**静默地一行都不产生**。
- 各 benchmark 按 `FLAGSPARSE_BENCH_VARIANTS` 选行、给行打 `variant` 标签；`write_summary.py` 先按标签对应变体
  （`(spmm_csr, f32)` 下有 4 个变体，原来按算子 + 类型对应会对不上）；split 把 65 个 id 传给 CTest。
- `capi/conf/operators.yaml` 按实测补全（SpMV 的 `trans`/`conj_trans`、SpMM 的 f16、`mixed_dtypes`），
  `conf/operators.yaml` 里 65 个变体的 `capi` 全部为 `true`。
- 修了移植带来的一个回归：SDDMM 原有两个交付变体丢了 cuSPARSE 基线（q4 对所有带标签的行都不调基线），现在只
  对转置/列主序变体不调。
- fp16 输入按 Python 侧同样的规则先缩放到 [-1, 1]：`ASIC_680ks` 有值到 1e6，超出 fp16 范围（65504）会变成 inf。
  q4 表里 10 个 fp16 变体各有 1 行失败（9/0/1/0）就是这个原因，现在已经修好。
- **口径变化**：main 原来的 C API 测 `spmm_csr/coo_f32/f64_int_non_non_row` 时 B、C 实际是列主序（`test_spmm.cpp`
  写死了 `ORDER_COL`），名字标的却是 `_row`。现在按名字测行主序，这 4 个数字会变（CUDA 上 `spmm_csr_f32`
  0.392x → 0.686x）。拿新旧报告对比时注意这一点，这不是性能变化，是之前标错了。

**CUDA 上的端到端结果**（`run_flagsparse_split_delivery.py --backend cuda`，RTX 5090，10 个交付矩阵）：65 个变体
精度 65/65 Passed、0 missing；性能 34 个有 cuSPARSE 加速比、31 个 `NoBaseline`；`check_manifest` 0 处不一致；
C API 精度 10/10 测试族通过。唯一的精度瑕疵是 `spmm_csc_f32_int_non_non_row` 在 `cfd2` 上有 2 行误差比
1.03～1.13，略超严格容差（宽松容差下 0.011）。走的是原子累加路径，没有厂商基线可比，按规则拿不到宽松判定，
如实保留，没有改容差。

**只在 MUSA 上编译、CUDA 上测不到的部分**，上机时最该留意：

- `capi/src/ops/spmv.cpp`/`spmm.cpp` 里 `musa_gather_type()` 分支（`capi/src/core/sparse_gather.hpp`）：MUSA 上 CSC、
  转置、复数的 SpMV/SpMM 走这条 gather 路径。q4 当时靠它把这些变体从 0.02x～0.4x 提到 0.9x～3.5x。
- `FLAGSPARSE_MUSA_BASELINE_EXTENSIONS`（只在 `BACKEND=MUSA` 时定义）：muSPARSE 的 CSC 基线（CSC 当成转置后的
  CSR 喂给 muSPARSE，4.3.5 的 CSC 描述符会报错）、SDDMM 带转置/列主序的基线、SpVV 基线。
- 这两部分代码是从 q4 原样搬过来的，移植时没有改动逻辑，只改了变量名（`q4_variant` → `variant_id`）。

**预期结果**：q4 在 MTT S5000（muSPARSE 4.3.5）上的 2026-10-06 实测，45 个新变体、10 个交付矩阵。加速比是
`muSPARSE_ms / FlagSparse_ms`，只对双方严格通过的行取算术平均；P/R/F/U 是严格通过/放宽通过/失败/未校验的行数。
这次重跑应该大致复现这张表；fp16 那 10 行的失败应该消失。

<details>
<summary>q4 的 MUSA 实测表（45 个变体）</summary>

| 变体 | P/R/F/U | 双方严格平均（行数） |
|---|---:|---:|
| `axpby_f16_int` | 1/0/0/0 | N/A |
| `scatter_i8_int` | 10/0/0/0 | N/A |
| `sddmm_csr_c32_int_non_non_row` | 20/0/0/0 | 13.546x (20) |
| `sddmm_csr_f16_int_non_non_row` | 20/0/0/0 | N/A |
| `sddmm_csr_f32_int_non_non_col` | 20/0/0/0 | 35.218x (20) |
| `sddmm_csr_f32_int_non_trans_row` | 20/0/0/0 | 34.687x (20) |
| `sddmm_csr_f32_int_trans_non_row` | 20/0/0/0 | 22.732x (20) |
| `spgemm_csr_f32_int_non_non` | 6/0/0/4 | 0.669x (5) |
| `spmm_coo_c32_int_non_non_row` | 10/0/0/0 | 0.944x (10) |
| `spmm_coo_f16_int_non_non_row` | 9/0/1/0 | N/A |
| `spmm_coo_i8i32_int_non_non_row` | 10/0/0/0 | N/A |
| `spmm_csc_c32_int_non_non_row` | 10/0/0/0 | 3.528x (10) |
| `spmm_csc_f16_int_non_non_row` | 9/0/1/0 | N/A |
| `spmm_csc_f32_int_non_non_row` | 10/0/0/0 | 2.127x (9) |
| `spmm_csr_c32_int_non_non_row` | 10/0/0/0 | 1.071x (10) |
| `spmm_csr_f16_int_non_non_row` | 9/0/1/0 | N/A |
| `spmm_csr_f16f32_int_non_non_row` | 9/0/1/0 | N/A |
| `spmm_csr_f32_int_non_non_col` | 10/0/0/0 | 0.887x (10) |
| `spmm_csr_f32_int_non_trans_row` | 10/0/0/0 | 0.785x (10) |
| `spmm_csr_f32_int_trans_non_row` | 9/1/0/0 | 1.725x (8) |
| `spmm_csr_i8i32_int_non_non_row` | 10/0/0/0 | N/A |
| `spmv_coo_c32_int_conj` | 9/1/0/0 | 0.978x (9) |
| `spmv_coo_c32_int_non` | 10/0/0/0 | 2.585x (10) |
| `spmv_coo_f16_int_non` | 9/0/1/0 | N/A |
| `spmv_coo_f16f32_int_non` | 9/0/1/0 | N/A |
| `spmv_coo_f32_int_trans` | 9/1/0/0 | 0.877x (9) |
| `spmv_coo_i8i32_int_non` | 10/0/0/0 | N/A |
| `spmv_csc_c32_int_non` | 10/0/0/0 | 0.904x (10) |
| `spmv_csc_f16_int_non` | 9/0/1/0 | N/A |
| `spmv_csc_f32_int_non` | 10/0/0/0 | 1.009x (10) |
| `spmv_csr_c32_int_conj` | 9/1/0/0 | 0.905x (9) |
| `spmv_csr_c32_int_non` | 10/0/0/0 | 0.924x (10) |
| `spmv_csr_f16_int_non` | 9/0/1/0 | N/A |
| `spmv_csr_f16f32_int_non` | 9/0/1/0 | N/A |
| `spmv_csr_f32_int_trans` | 9/1/0/0 | 0.977x (9) |
| `spmv_csr_f32c32_int_non` | 10/0/0/0 | N/A |
| `spmv_csr_i8f32_int_non` | 10/0/0/0 | N/A |
| `spmv_csr_i8i32_int_non` | 10/0/0/0 | N/A |
| `spmv_sell_c32_int_non` | 10/0/0/0 | N/A |
| `spmv_sell_f16_int_non` | 9/0/1/0 | N/A |
| `spmv_sell_f32_int_non` | 10/0/0/0 | N/A |
| `spmv_sell_i8i32_int_non` | 10/0/0/0 | N/A |
| `spvv_c32_int_conj` | 1/0/0/0 | 0.686x (1) |
| `spvv_f16f32_int_non` | 1/0/0/0 | N/A |
| `spvv_i8i32_int_non` | 1/0/0/0 | N/A |

来源：`origin/q4` 的 `docs/debug/MUSA.md`（`b058939`）。原始结果目录 `capi/bench-musa-real-q4-real10-20261006/`
没进 git，只在 MUSA 机器上。

</details>

## 交给 Codex 的 prompt

上机时把下面整段原样贴给 Codex（或其他代理），不用改任何内容。

````text
你在 摩尔线程 S5000（MUSA） 实机上复测 FlagSparse 的 65 个交付变体（2026-10-14 之后的这一轮）。仓库在当前目录，main 分支。

背景：这台机器上一轮已经跑过 65 项。之后仓库改了两类东西，这轮就是验证它们：
- 没有基线的变体，现在用仓库自带的 H800 结果按带宽折算出加速比（命令里的 --h800-reference），交付表里只显示数字，
  delivery_table.py --show-baseline 的最后一列说明哪些是折算的。有本机基线的变体不受影响。
- 本后端相关的修复，见 docs/debug/MUSA.md "跑完要确认"一节逐条列出的内容。
主要目的是收集数据，不是修代码。

步骤：
1. git pull --ff-only origin main；git log -1 --oneline 的提交应不早于 d7d1919（"ci again"）。
2. 按 docs/MUSA.md 配好环境，做完它的环境自检。
3. 确认导入的是仓库源码：python3 -c "import flagsparse; print(flagsparse.__file__)"
   输出必须在当前目录的 src/ 下。如果指向 site-packages / dist-packages，先停下来汇报，不要继续。
4. 核对测试矩阵：这轮只用仓库自带的 10 个交付矩阵（conf/operators.yaml 的 delivery_matrices），路径和大小：
   tests/data/ASIC_680ks.mtx  69268748 字节
   tests/data/GL7d14.mtx  26726293 字节
   tests/data/NACA0015.mtx  43158387 字节
   tests/data/amazon0601.mtx  44469712 字节
   tests/data/auto.mtx  44768883 字节
   tests/data/cage12.mtx  61548649 字节
   tests/data/cfd2.mtx  41954205 字节
   tests/data/filter3D.mtx  47472525 字节
   tests/data/roadNet-TX.mtx  27699282 字节
   tests/data/wave.mtx  13325709 字节
   用 ls -l tests/data/*.mtx 核对：恰好这 10 个文件、大小一致。缺文件或大小不对（例如 git lfs 指针、下载不全）
   先停下来汇报，不要换别的矩阵目录。命令里的 --benchmark-input tests/data 指的就是它们。
5. 完整读一遍 docs/debug/MUSA.md，照"要做的事"一节的命令原样跑（已带 --h800-reference，不要去掉）。
   - --results-dir 用新目录（<日期> 填今天），不要复用旧目录，也不要往旧目录里补跑。
   - 命令本身已经用 setsid 放到后台，定期看日志进度，不要中途打断。
   - 不要用 pytest --forked，它在 GPU 上会让所有用例失败。
6. 跑完执行 python3 tools/delivery_table.py <结果目录>。

规矩：
- 不要为了"让它通过"去改 src/ 或 tests/ 下的代码。确实非改不可（比如环境适配）时，每改一个文件，
  都按 docs/debug/MUSA.md 末尾"实机改动记录"一节的格式追加一条记录，并和代码改动放在同一个 commit 里。
- 不要猜根因。遇到报错就给出完整报错原文，以及出错变体在结果目录里那一行的 reason/error 字段。
- 折算出来的加速比是估算，低也照实报，不要为了数字好看换基线或改参数。
- 结果目录整个打包带回（至少 summary.json、summary.csv、各算子的 performance.csv 和 *_stderr.log）。

汇报包含：
1. delivery_table.py 的完整输出，尤其是第一行 "N registered variants, M missing"。
2. 所有非 Passed 的变体、以及加速比为 "-" 的变体：变体名、阶段（精度/性能）、reason 或 error 原文。
3. 逐条回答 docs/debug/MUSA.md "跑完要确认"一节。
4. 环境指纹：torch、triton 版本，设备名，驱动/SDK 版本，git log -1 --oneline，整轮总耗时。
5. docs/debug/MUSA.md "实机改动记录"一节的内容（没改写"无"）。
````

## 实机改动记录

在这台机器上为了跑通而改过的每一个文件都记在这里，跟着 commit 一起推上来。没改就写"无"，不要留空。
只记改动，测试结果写到上面"带回来的东西"/"跑完要确认"里。

格式（一个改动一条，新的追加在最后）：

```
### <日期> <文件路径>::<函数或位置>
- 改了什么：
- 为什么改（附报错原文）：
- 怎么验证的：
- 其他后端是否也需要：是 / 否 / 不确定
```

### 2026-10-15 优化与复测口径

本轮按用户要求优化代码，验收目标是**每个交付变体各自的平均加速比 > 0.8x**，不是 65 项的整体平均。
矩阵仍为 `tests/data` 的原始 10 个文件，split runner、测试规模、容差和基线优先级不变。
本机 CPU cgroup 配额为 15 核，后续配对测量统一设置
`OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`，原始源码也用相同设置复测，
避免默认大线程池触发 CPU 限流。H800 带宽折算仍是估算，不能当作本机厂商实测。
下列定向性能结果只用于调优，未替代最终完整 split；未达标和缺基线的条目必须继续如实报告。

### 2026-10-15 src/flagsparse/sparse_operations/mixed_spmx.py::MUSA SpMV/SpMM 路由
- 改了什么：COO/CSC 转 CSR 使用有界、张量版本感知的缓存；短行 SpMV 调整行块，严重不均衡短行复用 NNZ 分块内核；SpMM 复用行批量 gather；避免输出自拷贝。
- 为什么改（附报错原文）：无报错，原 COO 半精度/整数原子路径及逐行 SpMM 的实测加速比低于 0.8x。只缓存结构和必要的值重排，不缓存计算结果；输入变更失效。
- 怎么验证的：10 矩阵定向复测 `results_musa_opt_20261015_nnz_threads8/`；Python 定向回归 80 passed、3 skipped，含乱序/重复 COO、值/索引/向量修改、无版本推理张量、非连续输出。
- 其他后端是否也需要：否；新路由仅 MUSA。

### 2026-10-15 src/flagsparse/sparse_operations/_spmm_row_gather.py::_kernel/compute
- 改了什么：增加可选累加类型以支持 int8 到 int32；MUSA 使用 8 行、4 个稀疏槽、4 warps。
- 为什么改（附报错原文）：无报错；MUSA 多行批量配置比逐行混合精度内核更快，整数输出必须保持整数累加。
- 怎么验证的：上述 80 项回归与 10 矩阵 SpMM 定向性能复测。
- 其他后端是否也需要：否；默认累加类型和其他后端的配置保持不变。

### 2026-10-15 src/flagsparse/sparse_operations/_spmv_csr_nnz.py::_musa_segmented_add
- 改了什么：MUSA 专用分段扫描的零分支使用整数文字 `0`，避免 int32 被浮点文字隐式提升；通过默认 False 的编译期 MUSA 开关选择，原共享归约函数保留原样。
- 为什么改（附报错原文）：无报错；复用到 MUSA int8/int32 时必须保持精确整数归约。
- 怎么验证的：整数 NNZ 路径包含重复列、空行、输入变更和非连续输出的精确回归。
- 其他后端是否也需要：否；其他后端仍使用原 `_segmented_add` 及浮点零分支。

### 2026-10-15 src/flagsparse/sparse_operations/_spmv_csr_transpose.py::make_plan/gather
- 改了什么：MUSA 调整行块；半精度实数 gather 复用混合精度 CSR 执行器。
- 为什么改（附报错原文）：无报错；原半精度 CSR 标量路径很慢。实验中的全行排序未带来收益，未用于最终混合 SpMV 路径。
- 怎么验证的：80 项 Python 定向回归；含复数转置用例，未修改精度容差。
- 其他后端是否也需要：否；配置与路由均以 MUSA 为条件。

### 2026-10-15 src/flagsparse/sparse_operations/spmv_csr.py::prepare_spmv_csr
- 改了什么：MUSA 半精度默认配置建立 non-gather plan，运行时使用优化的行批量路径。
- 为什么改（附报错原文）：无报错；原 FP16 SpMV 性能明显落后。
- 怎么验证的：半精度交付精度用例及 10 矩阵定向性能测试。
- 其他后端是否也需要：否；显式配置和其他后端保持原行为。

### 2026-10-15 src/flagsparse/sparse_operations/spmv_csc.py::_spmv_csc_via_mixed
- 改了什么：MUSA 半精度非转置调用缓存化的 `spmv_csc_half`。
- 为什么改（附报错原文）：无报错；原路径每次展开列号，抵消了缓存收益。
- 怎么验证的：CSC 值/索引变更回归，10 矩阵定向复测约 4.61x（H800 折算）。
- 其他后端是否也需要：否；原 CUDA/MACA 路由不变。

### 2026-10-15 src/flagsparse/sparse_operations/spmm_csr.py::默认半精度执行
- 改了什么：MUSA 非 accuracy 模式 FP16 复用 row-gather。
- 为什么改（附报错原文）：无报错；原路径逐行/逐槽执行缓慢。
- 怎么验证的：80 项回归及 10 矩阵定向性能复测，FP16 变体约 2.08x（H800 折算）。
- 其他后端是否也需要：否；CUDA 复数路由与其他后端不变。

### 2026-10-15 src/flagsparse/sparse_operations/spmm_coo.py::PreparedCooSpmmRoute
- 改了什么：MUSA FP16 准备 CSR gather 指针，复用现有行批量执行器。
- 为什么改（附报错原文）：无报错；避免原子累加/逐非零元程序的高开销。
- 怎么验证的：COO 精度用例和 10 矩阵定向性能复测，FP16 约 2.19x（H800 折算）。
- 其他后端是否也需要：否。

### 2026-10-15 src/flagsparse/sparse_operations/spmv_sell.py::csr_to_sell/并行执行
- 改了什么：MUSA 复数转换按实虚分量 index_copy；实数使用并行稀疏槽归约，复数使用已有并行内核。
- 为什么改（附报错原文）：原转换报 `"IndexPut" not implemented for 'ComplexFloat'`；串行稀疏槽性能较慢。
- 怎么验证的：10 矩阵、5 种数据/输出组合均 PASS；包含部分 slice、空行和 padding 的回归。
- 其他后端是否也需要：否；复数规避复用 Ascend 的已有方式，其他后端执行路由不变。

### 2026-10-15 src/flagsparse/sparse_operations/gather_scatter.py::_triton_gather_impl
- 改了什么：MUSA FP16 使用原生 index_select；保留输出形状/类型检查，非连续输入先连续化，非连续输出经临时缓冲区复制。
- 为什么改（附报错原文）：无异常；原 Triton 启动约 25us，原生索引约 7--16us。实测原生非连续输出有错误，不能直接传入。
- 怎么验证的：4 个原始 Gather 性能规模均 PASS；重复索引、非连续输入/输出、原地修改、错误形状/类型回归。
- 其他后端是否也需要：否。

### 2026-10-15 capi/flagsparse_codegen/spmv_csr.py::spmv_csr_row_tile_real
- 改了什么：增加 WIDE 开关，小矩阵使用 int32 地址计算，大范围仍保留 int64。
- 为什么改（附报错原文）：无报错；无条件 int64 地址计算增加 MUSA 指令成本。
- 怎么验证的：C API SpMV 精度测试族通过，10 矩阵 FP32 定向实测约 0.824x。
- 其他后端是否也需要：否；默认 WIDE 为 True。

### 2026-10-15 capi/src/ops/spmv.cpp::短行 CSR 与 SELL 启动
- 改了什么：按边界选择 CSR WIDE；MUSA SELL 选择并行稀疏槽内核和对应的 4-warps 配置。
- 为什么改（附报错原文）：无报错；减少地址计算与串行 slot 循环的开销。
- 怎么验证的：C API SpMV 精度测试族通过；10 矩阵定向性能复测。
- 其他后端是否也需要：否；SELL 其他后端沿用原内核和配置。

### 2026-10-15 capi/flagsparse_codegen/spmv_sell.py::并行 SELL 包装
- 改了什么：导出共享复数并行内核，增加 FP32/int32 累加的薄包装。
- 为什么改（附报错原文）：无报错；C API 复用 Python 中同一个实现，避免复制计算逻辑。
- 怎么验证的：C API SpMV 精度测试族通过，覆盖 SELL 实数、半精度、复数、整数。
- 其他后端是否也需要：否；原导出仍保留。

### 2026-10-15 capi/flagsparse_codegen/spmm_csr.py::spmm_csr_batched8_real
- 改了什么：按 WIDE 选择地址宽度，增加可调稀疏槽宽度 K，默认值保持 4。
- 为什么改（附报错原文）：无报错；int64 寻址与固定槽宽限制了 MUSA FP32/FP64 性能。
- 怎么验证的：C API SpMM 精度测试族通过；10 矩阵定向性能测试。
- 其他后端是否也需要：否；该内核只由 MUSA 路由选择。

### 2026-10-15 capi/src/ops/spmm.cpp::run_csr
- 改了什么：MUSA N=8 按累加类型选择行批量和槽宽，使用 4 warps；大地址保留 64 位。
- 为什么改（附报错原文）：无报错；原 batch=4/1 warp 实测吞吐低。
- 怎么验证的：10 矩阵定向实测 FP32 约 0.863x、FP64 约 0.902x，精度测试族通过。
- 其他后端是否也需要：否。

### 2026-10-15 capi/src/ops/spvv.cpp::run
- 改了什么：MUSA nnz<=32 使用 block=32、1 warp，其他情况保持原配置。
- 为什么改（附报错原文）：无报错；小向量不需要 256 个归约槽。
- 怎么验证的：C API SpVV 精度测试族通过；定向性能有波动，尚不能据此认定稳定达到 0.8x。
- 其他后端是否也需要：否。

### 2026-10-15 capi/src/ops/spgemm.cpp::workEstimation/compute/launch_hash
- 改了什么：MUSA 按行产品工作量分桶；复用主机计数存储，合并相邻的计数/溢出标志传输与前缀扫描。
- 为什么改（附报错原文）：无报错；原来全矩阵使用最坏行的哈希容量，短行也承担单行哈希和大表开销。
- 怎么验证的：C API SpGEMM 精度测试族通过；10 矩阵定向复测约 0.825x，存在波动。排序去重实验精度通过但性能退化，已撤回。每次重新计算，不缓存输出计数。
- 其他后端是否也需要：否；分桶与主机缓冲复用均限定 MUSA，其他后端保留原两次传输、溢出检查和前缀扫描流程。

### 2026-10-15 tests/pytest/test_musa_sparse_routes.py::新增回归
- 改了什么：新增 MUSA 格式转换缓存失效、整数精确累加、SELL padding/部分 slice、非连续输出、推理张量和 Gather 边界测试。
- 为什么改（附报错原文）：优化后必须验证输入实时可变，不能只测同一输入重复执行。
- 怎么验证的：连同 q4 SpMV/SpMM 精度用例共 80 passed、3 skipped；跳过的是不在该缓存路由内的 CSC 组合，不是交付变体跳过。
- 其他后端是否也需要：否；该文件显式要求 MUSA，原跨后端测试未删除或放宽。

### 2026-10-15 MUSA-only 范围收紧
- 改了什么：按用户补充要求，`mixed_spmx._finish` 仅在 MUSA 避免输出自拷贝，非 MUSA 恢复原行块选择；`_spmm_row_gather.compute` 的 int32 累加仅在 MUSA 启用；`_spmv_csr_nnz` 的 MUSA 归约通过编译期开关隔离；`capi/src/ops/spgemm.cpp` 的缓冲复用和分桶显式限定 MUSA。
- 为什么改（附报错原文）：无报错；用户要求“修改仅限于musa后端”，不能把共享行为变化带给其他后端。
- 怎么验证的：新增不执行 GPU 内核的路由隔离回归，分别模拟 MUSA/非 MUSA 检查自拷贝、整数累加和行块参数，6 passed；`git diff --check` 通过。此前 80 项 GPU 回归是隔离前结果；隔离后的 Python 回归、C API 重新编译及 SpMV/SpMM/SpGEMM/SpVV 精度复验排在当前 split 完成后运行，日志 `results_musa_scope_20261015.log`，未完成前不视为通过。
- 其他后端是否也需要：否；没有 CUDA/ROCm 等其他后端的真机验证，不声称已验证。
- 测试版本说明：`results_musa_split_65_20261015_opt/` 在隔离修改前启动。修改未调整 MUSA 算法、配置或基线，但源码哈希发生变化；该轮作为隔离前优化结果保留，不标为最终隔离后完整验收。

### 2026-10-16 本轮状态与定向数据

用户确认停止旧整轮后，旧 runner 及排队验证已停止，原目录与日志全部保留。本轮继续限定 MUSA，未修改测试矩阵、规模、容差或基线优先级。核对 `tests/data` 的 10 个 `.mtx` 大小与上文一致；导入路径为当前仓库 `src/flagsparse/__init__.py`，运行环境为 `mthreads musa None`、torch 2.7.1、Triton 3.6.0、MTT S5000。

定向结果不是完整交付验收，所有 H800 折算均仍为估算：

- `results_musa_opt_20261016_complex3d/`：原 10 矩阵上每条精度 PASS；`spmv_csr_f32c32_int_non` 平均从本轮改动前约 0.703x 提升至 0.819x。FP16→FP32 0.779x、FP16 0.754x、int8→FP32 0.820x、int8→int32 0.812x。COO 对应三项约 0.586x、0.685x、0.645x，仍未达标。
- `results_musa_opt_20261016_sell_flat/sell.csv`：5 种值/输出组合、10 矩阵，共 50 条 PASS。按已有同规模 Python 折算基线，SELL FP16 0.609x、FP32 0.695x、int8→int32 0.590x，仍未达标。
- `results_musa_opt_20261016_capi_sell/`：4 个交付 SELL 变体各 10 矩阵，40 条 `status=ok`、精度通过；复数 SELL 按已有 H800 同矩阵折算基线为 0.610x（旧轮约 0.275x），仍未达标。
- 固定最大行宽展开、SELL 行连续布局、成对复数加载、自动 Graph 重放均仅作临时探针，未显示稳定整体收益，没有接入正式源码。SELL→CSR 在 ASIC 上明显改善、其他矩阵有退化，也未接入。
- Python 扩大回归（MUSA 专用回归、q4 SpMV/SpMM、原 CSR 精度文件）已得到 559 passed、1659 skipped、23 deselected；跳过为原有后端/算法适用性规则，未新增或放宽规则。最终补充非连续输出后的结果及完整 split 见后续记录。

**尚未达到“每个变体各自平均 > 0.8x”目标，不能用上述单项或整体平均代替验收。**

### 2026-10-16 src/flagsparse/sparse_operations/mixed_spmx.py::_musa_csr_real_by_complex_kernel/_musa_coo_csr
- 改了什么：实数×复数使用三维实虚分量布局；严重不均衡短行可走复数 NNZ 路径。MUSA COO 从已排序行号用 `searchsorted` 建 CSR 指针，稀疏输入检查随版本感知缓存失效重新执行，RHS/输出仍每次检查。
- 为什么改（附报错原文）：无异常；原实数×复数逐行 kernel 慢。另复现原生 MUSA `bincount` 偶发少计数，4930 个输入得到计数和 4917，导致后续行指针及结果错误。
- 怎么验证的：10 矩阵定向结果见本轮状态；FP32/FP64 实数×复数、跨块、空行、输入修改回归通过，不缓存输出。
- 其他后端是否也需要：否；新 kernel 和预处理均仅 MUSA 路由选择。

### 2026-10-16 src/flagsparse/sparse_operations/_spmv_csr_nnz.py::_nnz_kernel
- 改了什么：增加默认 False 的 `COMPLEX_RHS`，MUSA 分别扫描实虚分量并处理跨块前缀；MUSA 的前缀补偿零分支保持整数类型。
- 为什么改（附报错原文）：无异常；实数×复数需要 NNZ 均衡，int8→int32 不允许跨块补偿隐式转 FP32 而丢失大整数低位。
- 怎么验证的：新增 1101 个 127×127 的精确整数累加，以及 601 个非零元跨块复数、空行和实时修改回归；未开启新参数的其他后端保持原扫描/补偿路径。
- 其他后端是否也需要：否。

### 2026-10-16 src/flagsparse/sparse_operations/_spmv_csr_transpose.py::prepare/gather
- 改了什么：MUSA 转置预处理使用排序后 `searchsorted` 替代 `bincount`；FP16 已按 CSR 排列时提前进入执行器；非连续输出使用临时缓冲区后复制回调用方视图。
- 为什么改（附报错原文）：复现 MUSA `bincount` 少计数；通用处理存在额外调用开销，且临时输出不能丢失调用方指定的非连续输出。
- 怎么验证的：扩大 CSR 精度、q4 SpMV/SpMM 回归通过，补充直接/间接 FP16 gather 的非连续输出回归。
- 其他后端是否也需要：否；其他后端仍使用原预处理和执行路径。

### 2026-10-16 src/flagsparse/sparse_operations/spmv_coo.py::PreparedCoo.__init__
- 改了什么：仅 MUSA 对排序后的转置行号使用 `searchsorted` 构造 CSR 行指针。
- 为什么改（附报错原文）：q4 COO FP32 转置出现 `AssertionError: max error 24 (scale 21.4)`；诊断时行指针末值 4862 而实际 nnz=4930。独立重复计数探针亦复现少计数，不是归约容差问题。
- 怎么验证的：新增 30 次重复 COO 转置回归，原 q4 转置及 CSR 精度回归通过。
- 其他后端是否也需要：否。

### 2026-10-16 src/flagsparse/sparse_operations/spmm_coo.py::PreparedCooSpmmRoute.__init__
- 改了什么：MUSA FP16 的已排序 COO gather 指针也由 `searchsorted` 生成，CUDA 复数仍保持原 `bincount` 路径。
- 为什么改（附报错原文）：同一 MUSA 计数原语用于该路由；避免把少计数风险留在 SpMM 中。
- 怎么验证的：q4 SpMM 与 MUSA 混合输入修改回归通过。
- 其他后端是否也需要：否。

### 2026-10-16 src/flagsparse/sparse_operations/spmv_sell.py::_musa_spmv_sell_kernel/flagsparse_spmv_sell
- 改了什么：MUSA 使用按行并行的二维 ROWS×SLOTS 布局；实数、整数和复数共享索引逻辑，复数累加显式映射 FP32/FP64。
- 为什么改（附报错原文）：旧 SELL 性能低；新路由最初遗漏复数累加类型，回归报 `KeyError: torch.complex64` / `torch.complex128`，现已修正。
- 怎么验证的：部分 slice、空行、padding、FP64/complex128 回归通过；上述 50 条 Python 和 40 条 C API 定向记录通过精度。
- 其他后端是否也需要：否；其他后端内核和配置保持不变。

### 2026-10-16 capi/flagsparse_codegen/spmv_sell.py / capi/src/ops/spmv.cpp::MUSA SELL
- 改了什么：MUSA 专用薄包装导出新的共享二维 SELL kernel，按 32 行/program、4 slots、4 warps 启动；非 MUSA 保留原包装和网格。
- 为什么改（附报错原文）：无报错；复用经过 Python 回归的实现降低旧串行 SELL 开销。
- 怎么验证的：重新构建确认无待编译项，C API SpMV 精度族通过，40 条 SELL 定向记录全部精度通过。
- 其他后端是否也需要：否。

### 2026-10-16 capi/src/ops/spvv.cpp::run / capi/ctest/accuracy/test_spvv.cpp
- 改了什么：MUSA 默认流主机结果依赖有序同步 D2H，省去重复全设备同步；非默认流和其他后端仍显式同步。增加 MUSA 默认流/非默认流、复数输入反复变更的主机结果检查。
- 为什么改（附报错原文）：无报错；重复同步影响小向量耗时。
- 怎么验证的：40 次双流复数结果检查及 C API SpVV 精度族通过；`results_musa_opt_20261015_spvv_sync/` 原 10 矩阵目录口径下的向量 fixture 定向约 1.202x，完整验收仍以新 split 为准。
- 其他后端是否也需要：否。

### 2026-10-16 tests/pytest/test_musa_sparse_routes.py::补充回归
- 改了什么：增加重复 COO 转置、大整数跨块、实数×复数的跨块/输入变更/FP64/部分 tile，以及直接和间接 FP16 gather 非连续输出用例。
- 为什么改（附报错原文）：覆盖本轮预处理及启动配置变更的真实失败和边界条件；不放宽原容差。
- 怎么验证的：扩大 GPU 回归见本轮状态与后续最终验证记录。
- 其他后端是否也需要：否；文件显式限定 MUSA。

### 2026-10-16 tests/pytest/test_spmv_csr_accuracy.py::test_spmv_csr_empty_dtype_ops
- 改了什么：仅 MUSA 复数结果改为对 `view_as_real(actual)` 执行原零值断言。
- 为什么改（附报错原文）：原检查调用 MUSA 不支持的复数 `count_nonzero`，报 `RuntimeError: SetMUTensorDType Unsupported tensor dtype: ComplexFloat` / `ComplexDouble`。
- 怎么验证的：原测试覆盖的所有 MUSA 空矩阵/数据类型/转置组合通过；仍断言非零元素数严格为 0，未跳过测试或改变精度标准。
- 其他后端是否也需要：否；其他后端检查保持原样。

### 2026-10-16 最终启动前验证与阶段性完整复测

- 最终 Python 回归：`561 passed, 1659 skipped, 23 deselected`，15.82 秒；日志 `results_musa_opt_20261016_validation.log`。包含后来补充的直接/间接 FP16 gather 非连续输出检查。
- C API 交付相关精度族：`accuracy.axpby/spmv/spmm/spvv/spgemm/sddmm/gather/scatter`，8/8 passed，44.97 秒。未把非交付 SpSV/SpSM 纳入本轮验收。
- `cmake --build capi/build --parallel 8` 确认 `ninja: no work to do.`；`git diff --check` 通过。
- 新整轮目录：`results_musa_split_65_20261016_opt`，日志 `results_musa_split_65_20261016_opt.log`。使用 `setsid`、12 小时外层限时、原 split runner，`--mode normal --benchmark-input tests/data --timeout 3600 --h800-reference`，并固定三种 CPU 线程池为 8。
- 此整轮用于检查当前阶段代码的 65 项结果，不预先视为所有加速比达标；定向测量中 FP16、COO、SELL 仍低于 0.8x，完整表生成后仍需按每个变体逐项检查。

### 2026-10-16 上述整轮完成后的原始数据复核

`results_musa_split_65_20261016_opt/` 已完成，65 项、0 missing，合并表的精度/性能标签均为 Passed，
但逐行复核仍有 16 个变体平均加速比低于 0.8x，另有 2 条 C API 原始精度失败，**不能按标签认定完全通过**。
Python 普通 CSC SpMM 的 `amazon0601 float32 trans n=32` 也有误差比 2.53 的失败，使 runner 返回 1。
C API 原始文件已保留在该目录的 `capi_raw_snapshot/`。Python CSC 性能族耗时 1212.88 秒；C API 8 个性能族
合计 1097.73 秒。补基线日志原文：

```text
--h800-reference (musa-s5000): NoBaseline variants filled from the Python half=26, from the C API time=0, still without a baseline=0 [] (estimates, marked baseline=h800_scaled)
```

后续整轮显式使用独立的 `--capi-bench-out <新结果目录>/capi`，不再复用共享输出目录。

### 2026-10-16 src/flagsparse/sparse_operations/gather_scatter.py::_triton_scatter_impl
- 改了什么：MUSA FP16/int8 默认配置使用原生 `index_copy_`，显式 block_size 仍走原 Triton 路径；原生索引提升到 int64 并如实记录执行索引类型。保留原输出分配、清零、reset_output 和空输入行为。
- 为什么改（附报错原文）：小向量 Triton 启动开销较大；原生 int32 索引实际报 `RuntimeError: index_copy_(): Expected a long tensor for index, but got Int`，不能假定支持。
- 怎么验证的：`results_musa_opt_20261016_routes_v2/scatter.csv` 8/8 PASS，FP16/int8 对本机 PyTorch 约 0.96x/0.94x；增加非连续输入输出、实时修改、显式调优和非 MUSA 路由隔离回归。后续完整验证见最后记录。
- 其他后端是否也需要：否；原生路由仅 MUSA 默认配置。

### 2026-10-16 src/flagsparse/sparse_operations/spmm_csc.py::FP16 路由和转置 gather
- 改了什么：MUSA 默认非转置 FP16 缓存 CSC 到 CSR 的结构/值重排，使用已有混合 SpMM；原始索引、指针、值、存储或版本变化均失效，不缓存输出。MUSA 未指定调优块参数的 FP32/complex64 转置使用行拥有输出的 gather；支持 auto 和其解析后的 spmm_csc_base，保留其他类型和显式块参数路径。索引回退保留路由标志，元数据反映实际块配置。
- 为什么改（附报错原文）：FP16 每次转 FP32 并原子散射，旧折算均值仅 0.012703x。普通 FP32 转置出现上述误差比 2.53；旧 complex64 转置在 amazon0601 约 1.2 秒/次。
- 怎么验证的：`results_musa_opt_20261016_csc_integrated_v2/` FP16 原 10 矩阵全 PASS，折算均值 3.3125x；`results_musa_opt_20261016_routes_v4/csc_trans.csv` 普通 FP32/complex64、trans/conj、原 10 矩阵共 40 条 PASS。amazon0601 FP32 约 65.82 ms 到 0.917 ms，complex64 约 1233 ms 到 13.32 ms，均为同脚本本身耗时，不是新的厂商基线。
- 其他后端是否也需要：否；新增选择条件显式限定 MUSA，其他后端保持原实现。

### 2026-10-16 src/flagsparse/sparse_operations/mixed_spmx.py::_musa_csr_spmv/_musa_row_tile
- 改了什么：MUSA 网格使用整数上取整，去掉复数路径未用网格；超短实数/整数行采用 256 行、2 slots，大规模中短行采用 128 行、8 slots，小规模中短行仍为 64 行。
- 为什么改（附报错原文）：无异常；减少主机调用开销和超短行 program 数量。
- 怎么验证的：`results_musa_opt_20261016_wide_rows_v1/measurements.json` 配置探针保留首尾原配置作对照；roadNet-TX FP16/int8 约快 10%～15%。原有完整精度标准不变，新增实时存储替换回归。整轮验收仍以新 split 为准。
- 其他后端是否也需要：否；仅修改 MUSA 专用 helper。

### 2026-10-16 src/flagsparse/sparse_operations/spmv_sell.py::MUSA 分发与计划
- 改了什么：MUSA 提前分发，省去通用路径未用的网格/环境变量处理；对 padding 超过有效非零元 2 倍的实数/整数 SELL 压缩为 CSR。版本感知计划复用稀疏校验和连续化结果，RHS 与输出仍每次检查；validate=False 计划升级到 True 时重新校验，输入修改后失效。
- 为什么改（附报错原文）：无异常；ASIC 和 filter3D 的 SELL padding 分别约 3.98 倍、2.48 倍，读取无效槽开销大；重复包装校验影响短 kernel。
- 怎么验证的：原 10 矩阵 50 条精度 PASS，包含非交付 FP16 到 FP32；`results_musa_opt_20261016_routes_v4/sell.csv` 保留全部原始时间（包括 wave FP16 的较慢单次记录），不删慢行。新增高 padding、部分 slice、指针/列/值修改、推理张量、非连续 RHS/out、校验升级和非法索引回归。
- 其他后端是否也需要：否；其他后端的校验和计算分支保持原样。

### 2026-10-16 capi/flagsparse_codegen/sparse_gather.py::sparse_gather_spmv_f32
- 改了什么：复数各 lane 的累加仍为 FP32，仅最终跨 lane 归约提升到 FP64，再转回 FP32 执行 alpha/beta。
- 为什么改（附报错原文）：ASIC 第 405878 行的 3 个乘积包含小项和约正负 74 的抵消；原 FP32 归约丢小项，严格误差比 1.363375。CPU 与 GPU 探针均复现，未修改容差。
- 怎么验证的：`results_musa_opt_20261016_precision_v1/measurements.json`；加宽最后归约后 ASIC 该变体整体严格误差比约 0.0495。C API 新增双分量抵消回归，精度族通过。
- 其他后端是否也需要：否；该模块由 MUSA 专用 sparse_gather 分支使用。

### 2026-10-16 capi/src/core/sparse_gather.hpp::launch_sparse_gather
- 改了什么：间接实数 SpMV 平均行长不足 12 时保留 scatter，较长行采用确定性 gather。
- 为什么改（附报错原文）：cfd2 CSC FP32 的原子加法顺序导致原始严格精度失败；较长行可摊薄间接读取开销。
- 怎么验证的：`results_musa_opt_20261016_capi_precision_v2/` 原 10 矩阵，CSC FP32 10 条可比且精度通过，muSPARSE 平均约 0.974x；复数 CSR 共轭转置 10 条精度通过，muSPARSE 平均约 0.860x。
- 其他后端是否也需要：否；调用方只在 MUSA 选择此执行器。

### 2026-10-16 capi/src/ops/gather_scatter.cpp::gather_max_programs
- 改了什么：仅 MUSA 使用按设备区分的 thread_local program 上限缓存，避免每次查询硬件 SM/warp/线程属性。
- 为什么改（附报错原文）：无异常；小 Gather 的主机属性查询占比高。
- 怎么验证的：上述 C API 定向目录中 FP32/FP64/complex64/complex128 各 10 条精度通过，对 muSPARSE 平均约 0.970x/0.973x/0.966x/0.971x。
- 其他后端是否也需要：否；其他后端仍保留原查询行为。

### 2026-10-16 capi/ctest/accuracy/test_spmv.cpp::MusaComplexGatherCancellation
- 改了什么：MUSA 专用宏下增加来自 ASIC 的三项抵消回归，使用原有 complex64 严格容差，检查两个分量。
- 为什么改（附报错原文）：覆盖上文精度缺陷，防止只通过总体状态标签而漏掉原始失败。
- 怎么验证的：重新构建并运行 accuracy.spmv/gather/scatter，3/3 passed；后续扩大验证见最后记录。
- 其他后端是否也需要：否；新测试显式限定 MUSA。

### 2026-10-16 定向调优中未接入的实验

- `results_musa_opt_20261016_sorted_ell_v1/`：按行长重排的连续布局有局部收益，但多矩阵退化，未接入。
- `results_musa_opt_20261016_coo_cache_pair_v1/`：同进程交替配对表明，拆分 COO 拓扑/值缓存没有稳定收益，部分短行慢约 1%～10%；仅撤销本次实验，保留此前版本感知缓存。
- `results_musa_opt_20261016_loop_v1/`、`results_musa_opt_20261016_load_v1/`：循环展开、软件流水和读取缓存提示没有可靠整体收益，未接入。
- `results_musa_opt_20261016_sell_complex_compact_v1/`：复数 SELL 压缩在 filter3D 有收益，但 ASIC 退化，10 矩阵平均约 0.744x；未接入。稀疏校验计划也仅在 FP16/BF16/FP32/int8 使用，复数保持原校验与 kernel。
- 当前验证过的扩大 Python 回归为 845 passed、1658 skipped，另增 6 项原生 Scatter 调优/后端隔离回归通过。仍未达到每个变体 >0.8x；任何定向均值不能替代新的完整 split 结果。

### 2026-10-16 本次源码固定后的验证与下一轮完整复测

- Python 扩大回归：851 passed、1658 skipped，18.69 秒；`results_musa_opt_20261016_routes_v4/accuracy.xml`。包含 CSR/CSC/SELL、q4、Gather/Scatter 和 MUSA 路由隔离回归；没有降低容差或新增跳过规则。
- C API 交付相关精度族：8/8 passed，39.42 秒；`results_musa_opt_20261016_routes_v4/capi_accuracy.log`。构建确认 `ninja: no work to do.`，`git diff --check` 通过。
- `results_musa_opt_20261016_spmv_final_v1/`：CSR 5 项、COO 3 项各原 10 矩阵，共 80 条 PASS。CSR FP16/FP16到FP32/FP32乘complex64/int8到FP32/int8到int32 的折算均值分别为 0.810x/0.825x/0.842x/0.887x/0.876x；COO 对应三项仍为 0.679x/0.634x/0.691x，未达标。这些 H800 折算仍是估算。
- `results_musa_opt_20261016_routes_v4/scatter_final.csv`：默认 20 次 warmup、200 次迭代，FP16/int8 × 4 个原向量规模，8/8 PASS；对本机 PyTorch 各行约 0.92x～0.97x，不能与厂商或 H800 折算基线混称。
- SELL 最新定向数据仍有低于 0.8x 的类型，所有慢行原样保留。因此下一轮是阶段性完整复测，**不是全部 65 项达标声明**。
- `tests/data` 的原 10 个文件大小再次核对一致，torch 2.7.1、Triton 3.6.0、MTT S5000、驱动 3.3.5-server，HEAD `4d5ab61`；导入当前 `src/`，`mthreads musa None`。

新目录 `results_musa_split_65_20261016_opt2/`，保留本地源码补丁、未跟踪回归测试补丁和代码 SHA256。
命令如下；是否完成及是否达标必须检查该目录的新结果，不能沿用旧表：

```bash
setsid env PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads MUSA_HOME=/usr/local/musa \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  timeout -s KILL 43200 python3 -u run_flagsparse_split_delivery.py \
  --mode normal --benchmark-input tests/data --timeout 3600 --h800-reference \
  --results-dir results_musa_split_65_20261016_opt2 \
  --capi-bench-out results_musa_split_65_20261016_opt2/capi \
  > results_musa_split_65_20261016_opt2.log 2>&1 < /dev/null &
```

### 2026-10-17 上一轮结果与本轮优化范围

`results_musa_split_65_20261016_opt2/` 已完成，65 项、0 missing，合并精度/性能标签均为 Passed，
但实际仍有 8 个变体均值小于 0.8x：COO FP16到FP32/FP16/int8到int32、SELL FP16/int8到int32/complex64、
AXPBY FP16、SpGEMM FP32。继续按用户要求只优化 MUSA 后端；原始 10 矩阵、容差、迭代次数和基线优先级不变。
下述定向数据不是完整验收结论，尤其不能把 H800 折算值当成本机厂商实测。

### 2026-10-17 src/flagsparse/sparse_operations/vector_ops.py::_musa_axpby
- 改了什么：MUSA FP16/BF16 的中小向量用单个融合 kernel 完成缩放与稀疏更新，缓存有界、版本感知的索引反向映射；值和输出每次重新读取。大向量和其他后端保留原路由。
- 为什么改（附报错原文）：无报错；旧轮 AXPBY FP16 均值为 0.723508x，分开的缩放/更新启动开销较高。
- 怎么验证的：原 4 个向量规模全部 PASS，对本机 PyTorch 平均 1.221323x；原始数据 `results_musa_opt_20261017_layout_v1/axpby_public.csv`。新增 FP16/BF16、索引宽度、输入变更、推理张量、空输入和后端隔离回归，均纳入本轮扩大验证。
- 其他后端是否也需要：否。

### 2026-10-17 capi/flagsparse_codegen/spgemm_csr.py::musa_spgemm_scan_*
- 改了什么：增加 MUSA 分块行计数前缀和、块级前缀和及偏移回填 kernel，同时归约总 nnz、最大行 nnz 与溢出标志。
- 为什么改（附报错原文）：无报错；SpGEMM compute 阶段原先回传每行 counts/overflow 并在主机执行前缀和，旧轮均值为 0.715145x。
- 怎么验证的：MUSA C API 重新构建，accuracy.spgemm 通过，含下述跨块边界回归；原 10 矩阵定向记录 `results_musa_opt_20261017_layout_v1/capi_spgemm/`，6 个规则内可比矩阵对本机 muSPARSE 平均 1.076070x。非方阵、输出预算与厂商精度状态按原规则保留。
- 其他后端是否也需要：否；新 kernel 只由 MUSA 分支调用。

### 2026-10-17 capi/src/ops/spgemm.cpp::musa_build_indptr/compute/copy
- 改了什么：MUSA compute 在设备上构建 indptr，只回传 3 个 int64 元数据；copy 阶段仍按原排序流程回传所需行指针。不缓存输出计数，每次重新执行 count 和 scan。其他后端的 counts/overflow 回传与主机前缀和保持原样。
- 为什么改（附报错原文）：无报错；减少 compute 阶段逐行回传及主机扫描成本。
- 怎么验证的：同上 SpGEMM 原 10 矩阵定向性能和 C API 精度测试；完整 split 仍需重新验收。
- 其他后端是否也需要：否。

### 2026-10-17 capi/ctest/accuracy/test_spgemm.cpp::MusaMultiBlockRowPrefix
- 改了什么：增加 MUSA 专用 1024、1025、3079 行的 FP32/FP64 用例，检查 scan 块边界、部分末块和完整结果。
- 为什么改（附报错原文）：无报错；原有小规模精度用例不能覆盖新增前缀和的跨块路径。
- 怎么验证的：accuracy.spgemm 通过，未更改原有精度判定。
- 其他后端是否也需要：否；其他后端不运行新增 MUSA 专用用例。

### 2026-10-17 src/flagsparse/sparse_operations/_musa_spmv_launch.py::launch
- 改了什么：新增最多 64 项的编译启动器 LRU，只缓存 CompiledKernel 的公开 launcher，不保留输入张量或输出。key 包含 kernel、网格、数据类型、设备、地址对齐、非零指针别名、运行时标量、constexpr 和 kernel.debug；每次执行仍解析当前流并运行 Triton profiling hooks。存在 pre-run hook、运行时 debug、instrumentation 或自定义编译阶段 hook 时回退正常 JIT。
- 为什么改（附报错原文）：无报错；MUSA 短行 SpMV 受 JIT 参数绑定及特化处理的主机开销限制，公开 API 配对测量确认启动器复用有收益。
- 怎么验证的：新增 key 隔离、LRU 淘汰、调试/插桩回退、未对齐连续输入/输出、输出独立性、实时 RHS、非默认流及非 MUSA 路由隔离回归；本轮扩大回归全部通过。
- 其他后端是否也需要：否；模块导入不初始化 MUSA 或导入 Triton knobs，只有 MUSA 路由调用 launch。

### 2026-10-17 src/flagsparse/sparse_operations/mixed_spmx.py::_musa_csr_spmv/_musa_row_tile
- 改了什么：MUSA 实数/整数短行 CSR 和 NNZ 路由复用上述启动器；按平均行长、行数与整数累加选择行块和 slots。普通实数乘复数行 kernel 保持原启动方式。
- 为什么改（附报错原文）：无报错；旧轮 COO 三个半精度/整数变体低于 0.8x，格式转换缓存后的重复 JIT 启动仍占比较高。
- 怎么验证的：最终公开 API 定向复测 `results_musa_opt_20261017_dispatch_v2/coo_public_final.csv`，原 10 矩阵共 30 条精度 PASS；FP16到FP32/FP16/int8到int32 的 H800 折算均值分别为 0.821573x/0.897036x/0.902609x。首项裕量较小，不能以定向数据代替完整 split。
- 其他后端是否也需要：否。

### 2026-10-17 src/flagsparse/sparse_operations/spmv_sell.py::MUSA kernel 启动
- 改了什么：MUSA SELL 复用上述编译启动器；保留已有版本感知计划、高 padding 实数压缩及 dtype/规模分块策略。
- 为什么改（附报错原文）：无报错；旧轮 SELL FP16/int8到int32/complex64 均值为 0.735946x/0.718123x/0.787854x，短 kernel 的主机启动成本占比较高。
- 怎么验证的：`results_musa_opt_20261017_dispatch_v2/sell_public.csv` 原 10 矩阵共 30 条精度 PASS；三项 H800 折算均值为 0.969267x/0.961707x/0.893333x。最终重复测量 `sell_public_final.csv` 仍为 30 条 PASS，均值为 0.981482x/0.984545x/0.911817x。
- 其他后端是否也需要：否。

### 2026-10-17 tests/pytest/test_musa_sparse_routes.py::AXPBY 与启动缓存回归
- 改了什么：新增上述 AXPBY、启动缓存特化/回退、公开 COO/SELL 缓冲区及流切换、后端路由隔离用例；保留之前所有回归。
- 为什么改（附报错原文）：无报错；优化不能依赖输入地址不变、复用旧输出或忽略运行时调试设置。
- 怎么验证的：最终扩大 Python 回归 `879 passed, 1658 skipped`，18.79 秒，`results_musa_opt_20261017_dispatch_v2/accuracy_final.xml`；跳过均沿用已有适用性规则，未放宽容差。C API 交付相关精度族 8/8 passed，39.89 秒，`capi_accuracy.log`；`git diff --check` 通过。
- 其他后端是否也需要：否；此回归文件显式要求 MUSA。

### 2026-10-17 未接入的实验

- MUSA mixed COO 块内 segmented atomic 实验没有公开 API 的整体收益，新增实验 kernel 和 wrapper 已撤回；不改变当前 CSR 缓存路线。
- 裸 kernel 启动探针低估了公开 API 的主机开销，最终采用同一公开 API 的原规模配对测量与仓库原 benchmark 校验收益。
- SELL 的额外 tile 实验没有稳定整体收益，保留原配置；性能与精度原始记录见 `results_musa_opt_20261017_layout_v1/` 和 `results_musa_opt_20261017_dispatch_v2/`。

### 2026-10-17 冻结源码与完整复测

- 新结果目录 `results_musa_split_65_20261017_opt3/`，不复用旧轮；包含源码补丁、未跟踪 helper/回归测试补丁和 SHA256。
- 源码导入路径 `/root/gcx/FlagSparse/src/flagsparse/__init__.py`；HEAD `4d5ab61`；torch 2.7.1、torch_musa 2.7.1+5ee0a64、Triton 3.6.0、MTT S5000，驱动 3.3.5-server，mcc 5.1.0，torch.version.musa=40305。
- 再次核对 `tests/data` 原 10 个矩阵名称与大小一致；固定三个 CPU 线程池为 8。不跳过 C API 构建，不改变原 runner 的规模、容差、迭代次数或基线优先级。
- 这是本轮优化后的完整验收启动记录，不是已完成或全部达标声明。必须等新整轮结束后逐变体检查实际平均加速比及原始精度记录。

```bash
setsid env PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads MUSA_HOME=/usr/local/musa \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  timeout -s KILL 43200 python3 -u run_flagsparse_split_delivery.py \
  --mode normal --benchmark-input tests/data --timeout 3600 --h800-reference \
  --results-dir results_musa_split_65_20261017_opt3 \
  --capi-bench-out results_musa_split_65_20261017_opt3/capi \
  > results_musa_split_65_20261017_opt3.log 2>&1 < /dev/null &
```

### 2026-10-18 上游同步后的复测与优化口径

> 此节及下方原生实现记录为已撤回的实验历史，不代表当前全 Triton 实现。
> 用户随后明确禁止原生 MUSA 内核；原生定向结果与 opt4 均不用于全 Triton 验收。

- 基准提交为 `3cea4bc3615655dabcf676850de7d770d7da795a`。此前优化已被上游吸收；旧 stash、完整轮结果和打包文件均保留。本轮开始先重建 C API，恢复源码与二进制一致。
- 最近完成的完整轮仍是 `results_musa_split_65_20261017_opt3/`，65/65 精度 Passed、0 missing，但下表 7 个变体未过 0.8x。这里继续按用户要求优化**每个变体各自的矩阵平均值**，不是整个交付集的平均值，也不要求每个矩阵单独大于 0.8x。
- 性能输入仍为 `tests/data` 的原 10 个矩阵，名称与字节数再次核对一致。固定三个 CPU 线程池为 8，GPU 测量串行；未改 runner、测试规模、迭代次数、容差、跳过规则或基线优先级。
- 最终源码重复测量如下。COO 三项取原 q4 benchmark（5 warmup / 20 iters）的公开 API 耗时，对照旧完整轮同变体同矩阵的 `h800_scaled_ms`；**属于 H800 带宽折算估算，不是本机厂商实测**。Gather/Scatter 取原 C API benchmark（10 warmup / 100 iters）的本机 muSPARSE 基线。

| 变体 | 旧完整轮 | 最终定向复测 | 矩阵精度 | 基线 |
|---|---:|---:|---|---|
| `spmv_coo_f16f32_int_non` | 0.726342x | 0.884297x | 10/10 PASS | H800 带宽折算 |
| `spmv_coo_f16_int_non` | 0.790823x | 0.936304x | 10/10 PASS | H800 带宽折算 |
| `spmv_coo_i8i32_int_non` | 0.772409x | 0.944477x | 10/10 PASS | H800 带宽折算 |
| `gather_f32_int` | 0.794171x | 0.987362x | 10/10 pass | 本机 muSPARSE |
| `scatter_f64_int` | 0.794720x | 0.996308x | 10/10 pass | 本机 muSPARSE |
| `scatter_c32_int` | 0.791277x | 0.995981x | 10/10 pass | 本机 muSPARSE |
| `scatter_c64_int` | 0.795244x | 0.997719x | 10/10 pass | 本机 muSPARSE |

原始数据：`results_musa_opt_20261018_followup_v1/coo_native_final.csv`、`capi_native_final/*_benchmark.json`；COO 全部 6 个 q4 变体共 60 条均 PASS。早期探针和集成中间轮完整保留，不能选取最好一次替代最终数据。

### 2026-10-18 src/flagsparse/sparse_operations/_musa_spmv_native.py::try_spmv/_load
- 改了什么：增加可选 MUSA 原生行子组 SpMV，FP16 在 FP32 中累加，int8 按请求在 int32/FP32 中累加；只在实测过的 mp_31 架构使用。按平均行长选择 2/8 lanes，平均行长 (3,4] 的不规则短行保留原 NNZ 路线，不按矩阵文件名选路。每次分配或使用调用方输出，实时读取 RHS；只缓存编译代码，不缓存结果。首次通过 mcc 编译到 Triton 原子代码缓存；编译器缺失或构建失败回退 Triton。每次解析当前流，原生包装保留/恢复当前设备；调试、插桩或启动钩子存在时回退正常 Triton 路径。
- 为什么改（附报错原文）：无算子错误；COO FP16 到 FP32 在旧完整轮和拉取后的原规模复测中持续约 0.73x，单纯调整 tile/启动器不足以达标。原生行子组减少动态行块归约及启动开销，配对测量确认有收益。集成时曾把 Triton 空 HookChain 误判为存在钩子，现按其 calls 判断；中间轮原始记录保留。
- 怎么验证的：原 10 矩阵公开 API 配对探针、原 q4 benchmark 两次正式入口复测；最终三项均值见上表。新增回归明确拒绝意外回退，并检查部分块、空行、非连续输入输出、值/索引/存储修改、推理张量及调试/钩子/编译器不可用回退；原非默认流回归同样通过。
- 其他后端是否也需要：否；模块仅在 MUSA COO 分支惰性导入。

### 2026-10-18 src/flagsparse/sparse_operations/mixed_spmx.py::spmv_coo_mixed
- 改了什么：仅 MUSA COO 路由先尝试上述原生内核，不适用时继续调用原 `_musa_csr_spmv`；保留原有版本/存储感知 COO 转 CSR 缓存、输出和计时语义。CSR、CSC、SpMM 和非 MUSA 路由不变。
- 为什么改（附报错原文）：无报错；接入经实测确认有收益的 MUSA COO 实现。
- 怎么验证的：最终 COO 60 条 q4 精度全部 PASS；扩大 Python 回归 1199 passed、1678 skipped（沿用原适用性规则，20.12 秒），日志和 JUnit 为 `native_accuracy_final.log/.xml`。
- 其他后端是否也需要：否。

### 2026-10-18 capi/src/adaptor/backend/musa/gather_scatter.mu 与 gather_scatter.hpp
- 改了什么：新增 MUSA 原生按索引逐元素复制，支持 1/2/4/8/16 字节值和 int32/int64 索引；按原始位模式复制实数/复数，无预处理或结果缓存。每次使用 handle 指定的流，网格步进覆盖全部输入。
- 为什么改（附报错原文）：无报错；原 Scatter 三项约 0.79x，短向量重复 JIT 调度的主机开销较高。
- 怎么验证的：原 C API Gather/Scatter 精度族通过；新增逐字节回归包含特殊浮点位模式、未写位置保留、输入/索引更新、部分块、两种索引宽度、默认/非默认流。原 10 矩阵两轮定向测量 40 条各自精度通过，最终均值见上表。
- 其他后端是否也需要：否。

### 2026-10-18 capi/src/adaptor/CMakeLists.txt::MUSA 原生复制构建
- 改了什么：只在 MUSA 分支且找到 mcc 时生成 mp_31 原生复制目标并链接到库；未找到 mcc 或其他架构继续使用原 Triton 路径，其他后端构建不变。
- 为什么改（附报错原文）：接入上述原生内核。初版构建规则曾报 `missing and no known rule to make it`，已增加明确 custom target 依赖；原始构建日志保留。
- 怎么验证的：C API 重新构建通过，基线仍为 `/usr/local/musa/lib/libmusparse.so`；8 个交付相关精度测试族全部通过。
- 其他后端是否也需要：否。

### 2026-10-18 capi/src/ops/gather_scatter.cpp::flagsparseGather/flagsparseScatter
- 改了什么：在 MUSA 原生目标存在且架构为 31 时调用原生复制；保留原 descriptor 校验、零输入语义和其他后端路径。
- 为什么改（附报错原文）：无报错；减少短向量重复 JIT 调度开销，保证已有相邻 0.8x 的变体有实测余量。
- 怎么验证的：原 benchmark 原 10 矩阵，Gather/Scatter 四项最终 0.987362x/0.996308x/0.995981x/0.997719x，40 条精度全部 pass；C API 精度族和新增流回归通过。
- 其他后端是否也需要：否。

### 2026-10-18 tests/pytest/test_musa_sparse_routes.py::原生 COO 回归
- 改了什么：增加原生路径实际执行、不同平均行长、部分块/空行、四种输出组合、输入/存储修改、推理张量和调试/钩子/编译器缺失回退用例；保留所有原用例和精度容差。
- 为什么改（附报错原文）：无报错；防止原生集成静默回退、复用旧结果或忽略运行时钩子。
- 怎么验证的：纳入上述扩大 Python 回归，1199 passed、1678 skipped。另跑后端/基线契约检查为 28 passed、1 failed；失败原文 `AssertionError: mthreads: no fallback reason on a CUDA box`，对应测试明确假定 CUDA 机器且 MUSA 插件缺失，当前是真实 MUSA 环境。未修改测试或后端探测以让它通过，原日志 `native_backend_contract.log` 保留。
- 其他后端是否也需要：否；本文件显式要求 MUSA。

### 2026-10-18 capi/ctest/accuracy/test_scatter.cpp::MusaLiveBitCopiesOnDefaultAndNondefaultStreams
- 改了什么：仅 MUSA 增加 Gather 到 Scatter 的原始位复制回归，覆盖 6 种值类型、两种索引宽度、默认/非默认流和多次修改；保留原测试。
- 为什么改（附报错原文）：无算子错误；原有测试不能覆盖原生复制的非默认流、输入实时性及特殊浮点位模式。新增测试初版误用辅助函数名，已修正，构建日志保留。
- 怎么验证的：C API 8 个交付相关精度族全部通过（29.30 秒），加上 Gather/Scatter benchmark 共 10/10 passed（42.15 秒），见 `capi_native_final.log`。
- 其他后端是否也需要：否。

### 2026-10-18 冻结与新完整轮

- 上述定向结果不覆盖旧完整轮，也不代表 65 项已经全部达标。新完整轮使用 `results_musa_split_65_20261018_opt4/`；该目录保存冻结补丁、源码 SHA256 与运行信息。
- 不跳过 C API 构建，沿用原 runner 和原 10 矩阵。命令如下；实际进程和启动时间以结果目录内记录及日志为准。

```bash
setsid env PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads MUSA_HOME=/usr/local/musa \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  timeout -s KILL 43200 python3 -u run_flagsparse_split_delivery.py \
  --mode normal --benchmark-input tests/data --timeout 3600 --h800-reference \
  --results-dir results_musa_split_65_20261018_opt4 \
  --capi-bench-out results_musa_split_65_20261018_opt4/capi \
  > results_musa_split_65_20261018_opt4.log 2>&1 < /dev/null &
```

### 2026-10-18 撤回原生 MUSA 实现，恢复全 Triton 约束

- 用户要求：计算内核必须保持全 Triton，允许使用 TLE，不允许原生 MUSA 实现。
- opt4 在 C API SDDMM 性能阶段停止；进程组 1776109 已退出，原日志、结果和冻结补丁全部保留。该轮包含原生实现，不能计作全 Triton 结果。
- 移除 `_musa_spmv_native.py`、`gather_scatter.mu/.hpp` 及 Python/CMake/C API 原生接入；可恢复副本见 `results_musa_opt_20261018_triton_v1/withdrawn_native/`。COO 混合精度 SpMV 恢复原 Triton 路径。
- 同时移除此前 MUSA Python FP16 Gather 和 FP16/int8 Scatter 的 PyTorch 原生索引分支，改为支持输入/索引/输出 stride 的 `@triton.jit` 索引复制和清零 kernel，保留其他后端的原路由。
- 将原生专属回归调整为断言 Triton 路由，并禁止 `index_select`/`gather`/`index_copy_` 调用；保留输入修改、非连续输出和流的精度覆盖。
- C API 已重新构建；构建规则和库符号核对不再包含原生复制目标。8 个交付精度族全部通过（38.07 秒），日志 `results_musa_opt_20261018_triton_v1/capi_accuracy.log`。
- Python 路由及 Gather/Scatter 精度回归 188 passed、2 skipped；扩大 SpMV/SpMM/Gather/Scatter 回归 1088 passed、1658 skipped（20.12 秒），日志/JUnit 为 `accuracy_final.log/.xml`。初轮 4 个 CPU mock 因未隔离 Triton 清零函数而失败，修正测试隔离后通过；初轮日志保留。最终源码精度复测记录另行追加。
- 原 10 矩阵纯 Triton C API 定向性能 `capi_before_v2/*_benchmark.json`：Gather FP32 0.966756x，Scatter FP64/complex64/complex128 分别 0.985301x/0.979894x/0.982658x，40 条精度全部 pass；本机 muSPARSE 对照，未修改 benchmark。首次定向运行未提前创建输出子目录，CTest 通过但未产生 JSON，不用作性能证据；重跑创建目录后的数据完整保留。
- COO 的归约布局/warp 配置探针未确认稳定收益；未将其集成入生产路径。临时的可调 launch 参数也撤回，保持此前上游 Triton launcher 不变。TLE 安装及 MUSA 特性白名单已检查，此轮未启用未经验证的 TLE kernel。
- 尚未重新完成全 Triton 的完整 65 项性能验收；COO 等未达标项仍需优化，不能沿用含原生实现的数字声明通过。
- 最终源码复测为 **1087 passed、1658 skipped**（18.85 秒），见 `accuracy_restored.log/.xml`；相比上面的临时探针版本少 1 个仅测试可调 launcher 编译参数的用例，因为对应生产改动已撤回，并非减少算子精度覆盖。原有 Triton launcher 与上游一致。当前未启动新的完整后台轮。

### 2026-10-18 src/flagsparse/sparse_operations/gather_scatter.py::MUSA Triton 索引复制
- 改了什么：MUSA 的 Gather/Scatter 统一调用 `@triton.jit` 索引复制，清零也使用 Triton；显式传入值/索引/输出 stride，并通过实部视图覆盖复数。移除原有 MUSA FP16 `index_select` 和 FP16/int8 `index_copy_` 分支。
- 为什么改（附报错原文）：无算子报错；落实用户的全 Triton 约束，并保留原生分支此前提供的非连续输入/输出语义。
- 怎么验证的：上述扩大 Python 回归；新增测试禁止 PyTorch 原生索引调用，并验证公开 API、实时输入、非连续输出、int32/int64 索引。
- 其他后端是否也需要：否；新路径严格限于 MUSA，厂商性能基线不改。

### 2026-10-18 全 Triton 新完整轮后台启动

- 按用户要求于 2026-10-18 00:10:30 UTC 用 setsid 启动完整 65 项精度与性能复测。新目录为 `results_musa_split_65_20261018_triton1/`，日志为 `results_musa_split_65_20261018_triton1.log`；不复用含原生实现的 opt4。
- 启动后核验 timeout PID/PGID/SID 均为 1914848，PPID=1；split runner PID=1914850，Python 阶段 PID=1914851，日志已进入 delivery-only 测试。此记录更新此前“当前未启动新的完整后台轮”的状态，不代表整轮已完成。
- HEAD `3cea4bc3615655dabcf676850de7d770d7da795a`；源码导入和 `mthreads musa None` 指纹核对通过；torch 2.7.1、torch_musa 2.7.1+5ee0a64、Triton 3.6.0、MTT S5000、驱动 3.3.5-server。
- 性能输入仍为 `tests/data` 的原 10 矩阵，名称和字节数一致；固定 OMP/MKL/OPENBLAS 线程池为 8，CTest 串行，启动前设备空闲。没有修改实现、runner、测试规模、容差、迭代次数或基线优先级。
- 沿用 `--mode normal --benchmark-input tests/data --timeout 3600 --h800-reference`，不跳过 C API 构建，显式 `--capi-bench-out results_musa_split_65_20261018_triton1/capi`；12 小时仅为外层限时。完整命令、PID 与源码/矩阵指纹见本轮 `RUN_INFO.md`、`runner.pid`、`source.patch` 和 `input_source.sha256`。
- 完成后检查 `summary_split.json` 和 `tools/delivery_table.py --show-baseline`；H800 折算仍属估算。尚不能声明 65 项全部精度通过或每项平均加速比 >0.8x。

### 2026-10-19 全 Triton 完整轮结果及继续优化

- 最近完成轮为 `results_musa_split_65_20261019_triton2/`：65/65 精度 Passed、65/65 性能测试 Passed、0 missing；按每个变体的矩阵平均加速比检查为 64/65 超过 0.8x，剩余 `spmv_coo_f16f32_int_non` 为 0.760595x。该变体采用 H800 带宽折算估算，不能当作本机 muSPARSE 实测。
- 本轮继续使用 `tests/data` 原 10 个矩阵、原 q4 benchmark 的 5 warmup / 20 iters 和原 split runner；未改规模、容差、基线或迭代次数。旧轮、归档和全部探索数据保留。

### 2026-10-19 src/flagsparse/sparse_operations/mixed_spmx.py::MUSA FP16 到 FP32 COO SpMV
- 改了什么：在 MUSA FP16 到 FP32 COO 路由新增 Triton 固定最大行长循环，避免每个 program 动态归约循环上界；仅在最大行长不超过 128 且平均行长大于 4 时启用。其他形状沿用原 NNZ/动态行块 Triton kernel。版本/存储感知 COO 缓存附带绑定的 Triton 启动特化，校验 RHS 类型、设备、对齐和指针别名，每次读取当前 RHS/输出地址及当前流，不缓存输出。COO SpMM 保留原三项缓存接口。
- 为什么改（附报错原文）：无算子报错；上一完整轮该变体为 0.760595x，未达到用户要求。探针定位到动态行长归约和重复启动特化开销；行长排序、额外布局/warp 探针没有稳定整体收益，未接入生产路径。
- 怎么验证的：原 q4 公开 API 六个 COO 变体共 60 条精度全部 PASS；目标变体两次定向十矩阵平均为 0.810783x、0.806586x（同一旧轮 H800 折算分母），记录 `results_musa_opt_20261019_triton_v3/coo_bound_public.csv` 和 `coo_bound_repeat.csv`。最终回归 194 passed、2 skipped，见 `accuracy_final.log/.xml`。启动器缓存键补齐 RHS dtype 后由完整轮再验收，不能以定向结果声明 65/65 性能达标。
- 其他后端是否也需要：否；新增调用严格位于 MUSA 分支。`audit_scope.py` 确认共享 kernel 定义、非 MUSA COO 路由及 C API 计算/构建代码与上游一致；非 MUSA 索引复制 CPU mock 对照为 96 passed、1 deselected，不代表其他加速器实机性能验证。

### 2026-10-19 tests/pytest/test_musa_sparse_routes.py::绑定启动器实时性回归
- 改了什么：增加固定行长、NNZ、短行三条路线与连续/非连续输出的六个用例，覆盖默认/非默认流、RHS 修改与对齐变化、输出独立性、值/索引修改、替换存储及预热后类型校验。
- 为什么改（附报错原文）：无算子报错；防止绑定启动器复用错误指针、旧结果、旧流或绕过参数校验。
- 怎么验证的：与 Gather/Scatter 精度回归一起为 194 passed、2 skipped；既有乱序/非连续 COO 和推理张量回归保留并通过。
- 其他后端是否也需要：否；测试要求实机 MUSA。

### 2026-10-19 全 Triton 第三轮完整复测

- 新目录 `results_musa_split_65_20261019_triton3/`，显式将 C API 原始数据存入该目录下 `capi/`，避免覆盖上一轮原始数据。仍按 setsid 在后台启动，不跳过精度、性能或 C API 构建。源码补丁、SHA256、矩阵指纹和命令保存在结果目录。
- 此处是待验收记录：完整轮约需 50 分钟，完成后再检查 65 项精度、missing、基线来源和各变体平均加速比；不能提前声明全部达标。

### 2026-11-04 三条 Python 慢路径实机检查

- 起始提交 `e447108`；从当前仓库 `src/` 导入，后端报告 `mthreads musa None`，未使用 CUDA 模拟。环境为 torch 2.7.1、Triton 3.6.0、MTT S5000、驱动 3.3.5-server。
- `conf/h800_reference.json` 已确认来源为 `results_cuda_105_20261103_1556`，105 个变体；`tests/data` 原 10 个矩阵的名称、字节数与本文记录一致。H800 带宽折算仍为估算，不是本机 muSPARSE 实测。
- **口径提示（合入时补）**：这是 `f7f04f1` 之前的整体 105 参照。现行参照原 65 个变体保持 10-14 的数据、只从 11-03 补新 40 个（见文首）。
  这轮交付表有 35 个变体用 H800 折算，其中 16 个属于新 40 个，不受影响；**19 个属于原 65 个**。已用本轮原始 CSV 离线重算
  （同样的行、同样的补位条件；先用旧参照复现，19 个与交付表完全一致，再换现行参照）：18 个变化在 −4%～+1% 之间，
  **只有 `spmv_coo_f16_int_non` 跨线：0.822 → 0.788**。原因是 10-14 那轮 H800 cuSPARSE 的 COO SpMV 快 5–9%（0.012–0.05 ms 的短调用），
  同族 `spmv_coo_f16f32_int_non` 0.844 → 0.809、`spmv_coo_i8i32_int_non` 0.896 → 0.867 也降了但仍过线。其余：
  SpMM 6 个 1.80–3.31（±1%）、`spmv_csc_f16_int_non` 5.75、`spmv_csr_*` 5 个 0.85–1.12、`spmv_sell_*` 4 个 0.84–1.01（−1.3%～−0.2%）。
  这是离线预演，不是新口径下的实测；下次全量重跑后以实测为准。
- 定向原始记录在 `results_musa_routes_20261104/`。先测 cage12、NACA0015，再按原 q4 的 5 warmup / 20 iters 跑全部 10 个矩阵；未改容差、规模、基线、迭代次数或跳过规则。

### 2026-11-04 src/flagsparse/sparse_operations/mixed_spmx.py::_musa_csc_spmv

- 改了什么：提取 MUSA CSC→CSR 版本/存储感知缓存供 FP16 和 int8 共用；缓存准备后的稀疏输入，实时读取 RHS 和输出，不缓存运算结果。计时请求覆盖当前调用的准备和计算。
- 为什么改（附报错原文）：无算子异常；连续 5 次 int8 CSC 调用原先执行 5 次整矩阵 argsort，修复后仅首次执行一次。文档指出的缓存失效已在 S5000 复现。
- 怎么验证的：`before_routes.json`、`after_routes.json`、`final_10_routes.json`；新增两种索引和两种输出类型的回归，覆盖 RHS、值、行索引、列指针修改、推理张量及跨步输出。
- 其他后端是否也需要：否；仅由 MUSA 分支调用，非 MUSA 的 FP16 路线不变。

### 2026-11-04 src/flagsparse/sparse_operations/spmv_csc.py::_spmv_csc_via_mixed

- 改了什么：MUSA 非转置 int8 CSC 直接调用上述缓存路线，避免每次新建 col_ids 使 COO 缓存失效；后续计算仍用原 Triton CSR kernel。
- 为什么改（附报错原文）：无异常；cage12 / NACA0015 原公开 API 为 2.173 / 8.017 ms。
- 怎么验证的：修复后两矩阵为 0.0397 / 0.1166 ms；最终 10 矩阵该变体全部 PASS，均值 0.060724 ms。记录 `before_q4.json`、`after_q4.json`、`final_10_q4.json`。
- 其他后端是否也需要：否；新增入口严格受 `_is_mthreads_runtime()` 保护。

### 2026-11-04 src/flagsparse/sparse_operations/spmm_csc.py::flagsparse_spmm_csc

- 改了什么：MUSA 默认非转置 CSC 快速路线支持 FP16→FP32，并扩展到 FP32、complex64；缓存 CSC→CSR 后使用原 Triton CSR SpMM，显式传输出类型，复数重排使用既有 `_gather_values`。新覆盖类型仅对 int32、auto 索引策略启用；int64、strict、显式调优、prepared、timing、return_meta 保留原路线，原 FP16 输出快速路线保留。
- 为什么改（附报错原文）：FP16→FP32 原先转成 FP32 后落入通用 CSC 路线。另单独确认 FP32/complex64 的根因为 `spmm_csc_base` 逐非零元 atomic scatter（`_spmm_csc_non_real_kernel` / complex kernel）；cage12 / NACA0015 的 FP32 为 39.247 / 122.392 ms，complex64 为 83.247 / 269.961 ms。扩展回归首次发现 `Failed: DID NOT RAISE <class 'RuntimeError'>`，已通过限制新快速路线保留 int64 自动重试与 strict 行为。
- 怎么验证的：FP16→FP32 两矩阵从 39.506 / 122.530 ms 降至 0.1247 / 0.5948 ms；FP32 降至 0.1845 / 0.8615 ms，complex64 降至 0.6909 / 2.5716 ms。最终十矩阵四个 SpMM 变体共 40/40 PASS；两项原有 int64 回退回归已通过。跨步 B/out、虚部、输入修改及推理张量回归通过。
- 其他后端是否也需要：否；仅扩展 MUSA 分支，其他后端 AST 与起始提交一致。return_meta 仍测原路线，不能用其耗时代表快速路线。

### 2026-11-04 src/flagsparse/sparse_operations/spmv_csr.py::prepare_spmv_csr

- 改了什么：仅 MUSA 的 BF16、默认 auto、非转置且无显式配置时建立既有 Triton row gather plan；显式算法选择不变。
- 为什么改（附报错原文）：无异常；S5000 上 cage12 / NACA0015 的 BF16 `legacy_segbin` 为 2.339 / 8.042 ms，`legacy_rowpar` 为 0.325 / 2.464 ms，gather 为 0.0593 / 0.151 ms，选择 gather。当前 API 的显式名称为 `legacy_segbin` / `legacy_rowpar`，不是文首排查示例中的简写。
- 怎么验证的：公开 auto API 从 2.278 / 8.728 ms 降至 0.0556 / 0.1507 ms；最终原十矩阵全部 PASS，均值 0.093168 ms。显式算法不建立 gather plan，准备后修改值仍即时生效。
- 其他后端是否也需要：否；BF16 条件只对 MUSA 扩展。

### 2026-11-04 tests/pytest/test_musa_sparse_routes.py 与验证结果

- 改了什么：新增 11 项 MUSA 实机回归，覆盖缓存复用、输入变更、推理张量、int32/int64、输出类型、跨步张量、复数虚部和显式算法。
- 为什么改（附报错原文）：防止优化依赖旧输入或破坏索引策略。原始失败日志保留，不以过滤/跳过处理失败。
- 怎么验证的：最终扩大回归 `822 passed, 1658 skipped`（19.58 秒），见 `regression_final.log/.xml`；跳过沿用原适用性规则。最终 8 个定向变体 × 原 10 矩阵为 80/80 PASS，见 `final_10_q4.json`。基线/后端分派 CI 为 46 passed，见 `ci_final.log/.xml`。`audit_scope.py` 四文件全部通过：移除 MUSA 分支后的 AST 与起始提交一致，已有 Triton kernel 定义未改；`git diff --check` 通过。
- 其他后端是否也需要：否；回归文件仅在 MUSA 实机运行。本次仅改主机路由和准备缓存，计算复用已有 Triton kernel，未增加原生 MUSA 计算实现。

### 2026-11-04 完整 105 变体后台复测安排

- 新目录 `results_mthreads_105_20261104_035312_routes/`，日志 `results_mthreads_105_20261104_035312_routes.log`；使用 setsid、GPU 0、原 `tests/data`、新版 H800 105 参照和原 split runner。不跳过精度、性能或 C API 构建，C API 原始结果独立存入新目录的 `capi_raw/`。
- 启动器为新目录中的 `launch.py`，保留源码快照/哈希、矩阵指纹、命令和起止状态；旧结果与归档保留。预计约 1 小时（上一完整轮约 62 分钟）。
- 此处是复测安排，不是整轮完成声明。结束后须按本文 11-02 清单检查 105 项精度、两个补位统计、40 个新变体的 C API 状态、卡死/异常和逐变体平均加速比；不得将定向结果当作全部 >0.8x 的结论。


### 2026-11-04 十个低于 0.8x 的 MUSA 变体优化

- 用户授权优化上一完整轮低于 0.8x 的十项；起始源码为 `6bb11c5`，优化前性能来自 `results_mthreads_105_20261104_072430_upstream/`（测试提交 `17448bd`）。本次计算保持 Triton，新增路由只在 MUSA 启用。没有修改 benchmark、容差、迭代、基线优先级或矩阵。
- 最终定向测试为 `results_musa_opt_20261104_low10/final_targets.json` 和 `FINAL_TARGETS.md`：十项逐变体算术平均均超过 0.8x；矩阵算子各用 `tests/data` 原十矩阵，GEMVI 各用原四个合成规模，共 88 条精度全部通过，其中三条按原 spec 6.3.1 为 PASS(relaxed)。这些是定向结果，不能代替 105 项完整验收。
- 六项 C API 使用同轮本机 muSPARSE；四项 Python 沿用上一完整轮同一逐矩阵 H800 带宽折算分母，仍是估算。当前 H800 参照 SHA256 与上一轮一致，旧 65 项来源 10-14，新增 40 项来源 11-03。

| 变体 | 优化前 | 最终定向 | 基线 |
|---|---:|---:|---|
| `gemvi_c32_int_non` | 0.461463x | 1.056042x | 本机 muSPARSE |
| `gemvi_f32_int_non` | 0.580628x | 1.798935x | 本机 muSPARSE |
| `spmm_coo_f32_int_trans_non_row` | 0.206547x | 1.007306x | 本机 muSPARSE |
| `spmm_csc_f32_int_trans_non_row` | 0.783727x | 1.334679x | 本机 muSPARSE |
| `spmm_csr_f16f32_int_non_non_col` | 0.288809x | 3.350625x | H800 折算估算 |
| `spmm_csr_i8i32_int_non_non_col` | 0.279965x | 2.805150x | H800 折算估算 |
| `spmv_coo_f32c32_int_non` | 0.200963x | 0.826824x | H800 折算估算 |
| `spmv_csc_c32_int_conj` | 0.168668x | 0.955029x | 本机 muSPARSE |
| `spmv_csc_f32_int_trans` | 0.190422x | 0.943172x | 本机 muSPARSE |
| `spmv_csr_bf16_int_non` | 0.681618x | 0.962396x | H800 折算估算 |

### 2026-11-04 src/flagsparse/sparse_operations/mixed_spmx.py::MUSA 实数乘复数 COO SpMV

- 改了什么：FP32×complex64 非转置 COO 复用版本/存储感知 COO→CSR 缓存；固定最大行长 Triton kernel 支持复数 RHS，按平均行长选择 tile；复用绑定启动器并保持实时 RHS、输出地址、流和缓存失效。长行/极不均匀行沿用现有 Triton 路线。MUSA CSR 复数启动也复用已有编译启动器。
- 为什么改（附报错原文）：无算子异常；原变体为 0.200963x。泛用原子散射、重复启动准备和行内归约成本较高，定向结果从约 0.66x、0.73x、0.79x 提升到最终表中数值。
- 怎么验证的：原十矩阵全部 PASS；默认/非默认流、RHS/值/索引修改、替换存储、输出别名检查和 inference tensors 回归通过。所有探索日志保留在上述目录。
- 其他后端是否也需要：否。`audit_scope.py` 移除 MUSA 分支后的 AST 与起始提交一致；已有共享 Triton kernel 定义不变。

### 2026-11-04 src/flagsparse/sparse_operations/_spmv_csr_transpose.py::MUSA BF16 gather

- 改了什么：已有 MUSA FP16 row gather 快速路由扩展至 BF16，沿用 `_musa_csr_spmv`，保持 FP32 累加及输出 dtype/stride 处理。
- 为什么改（附报错原文）：无异常；原 BF16 CSR 非转置变体为 0.681618x。
- 怎么验证的：原十矩阵全部 PASS；BF16/FP16 的直接/间接拓扑及跨步输出回归通过。
- 其他后端是否也需要：否；条件明确要求 `_is_mthreads_runtime()`，非 MUSA AST 不变。

### 2026-11-04 src/flagsparse/sparse_operations/_spmm_row_gather.py::MUSA 列主序 RHS

- 改了什么：仅 MUSA FP16/int8 的列主序 RHS，新增 Triton 布局复制后调用既有 row gather；每次读取当前 RHS，复制时间计入调用时间，地址计算使用 int64 防止大张量偏移溢出。
- 为什么改（附报错原文）：无异常；原 f16f32/int8i32 列主序 SpMM 分别为 0.288809x/0.279965x，重复非连续读取 RHS 成本高。
- 怎么验证的：两变体各原十矩阵全部 PASS；padded 列主序 RHS、实时修改、int32 精确输出、跨步输出的四个回归通过，见 `pack_accuracy_final.log/.xml`。
- 其他后端是否也需要：否；新复制调用严格在 MUSA 分支，已有计算 kernel 不变。

### 2026-11-04 capi/src/ops/spmv.cpp::CSC 转置直接 gather

- 改了什么：在 `musa_gather_type` 分支中，零基 CSC 转置/共轭转置直接以 CSR(A^T) 视图调用既有 Triton gather，复用原共轭和 alpha/beta 语义。
- 为什么改（附报错原文）：无异常；CSC FP32 转置和 complex64 共轭原均值为 0.190422x/0.168668x。
- 怎么验证的：两变体原十矩阵及 C API SpMV 34 个精度用例通过；非零虚部、共轭、非默认 alpha/beta、空列、重复索引和实时值/RHS 更新由新增用例覆盖。
- 其他后端是否也需要：否；新增判断位于既有 MUSA guard 内。

### 2026-11-04 capi/src/ops/spmm.cpp::COO/CSC 转置路由

- 改了什么：MUSA FP32 COO 转置准备紧凑 gather 拓扑，仅缓存索引，每次读取实时值/B/C；相应 bufferSize 和预处理仅在 MUSA 扩展。MUSA CSC 转置的八列输出直接使用其压缩布局。
- 为什么改（附报错原文）：无异常；原 COO/CSC 转置为 0.206547x/0.783727x。
- 怎么验证的：两个变体各原十矩阵全部按原规则通过；C API SpMM 28 个精度用例通过，含新 scratch、实时值/RHS、padded 布局和 COO alpha/beta。CSC 原公开 API 的 alpha=1、beta=0 限制保留并显式测试。
- 其他后端是否也需要：否；原 CUDA COO 转置路径保留，MUSA 专用准备/计算在独立条件中提前返回。

### 2026-11-04 capi/src/core/sparse_gather.hpp::MUSA 八列布局特化

- 改了什么：仅 MUSA、实数 FP32、N=8、B/C 步长均为 (8,1) 时调度新的行主序 Triton kernel；按平均行长选择 tile，平均行长大于 24 时保留串行 FMA 控制抵消误差。
- 为什么改（附报错原文）：首轮 K=4 长行归约在 cfd2 失败：`result outside tolerance: strict ratio 1.096677; relaxed would pass (0.010967) but muSPARSE 1.5.1 passed strict, so spec 6.3.1 does not apply`。已改归约方式，未放宽容差或删除案例；失败日志保留在 `capi_spmm_row_v1/`。
- 怎么验证的：`capi_v3_spmm/` 及 `final_spmm/` 原十矩阵均无失败；小/中/长行和 padded 布局回归通过。
- 其他后端是否也需要：否；布局特化显式检查 backend=musa，其他分支参数和 kernel 不变。

### 2026-11-04 capi/flagsparse_codegen/sparse_gather.py::sparse_gather_spmm_row_f32

- 改了什么：新增 MUSA 专用 Triton kernel，稀疏 lanes 在循环中独立累加、末尾归约，八个连续 dense 列按已验证布局访问；支持直接/间接拓扑及 alpha/beta，宽索引使用 int64 行地址。
- 为什么改（附报错原文）：无编译异常；普通动态步长影响连续读取与归约效率。cfd2 归约误差及修正见上一记录。
- 怎么验证的：原 20 条 SpMM 性能/精度案例与 C API 精度回归；新增符号只由上述 MUSA 条件调度。
- 其他后端是否也需要：否；文件中原有全部 Triton kernel AST 不变。

### 2026-11-04 capi/src/ops/gemvi.cpp::MUSA FP32/complex64 非转置计划

- 改了什么：仅 MUSA 缓存设备 SM 数；FP32 组件的非转置 GEMVI 使用专用行连续布局计划、分块和 split，真实满足 16 整除条件的整数参数才加入 JIT 对齐提示。其他类型/转置/后端保持原计划和 kernel。
- 为什么改（附报错原文）：无异常；原 FP32/complex64 均值为 0.580628x/0.461463x。布局探针收益最初未传到 C API，定位为动态参数缺少对齐信息；补齐受实际值约束的提示后正式实测达标。
- 怎么验证的：原四个合成规模各全部 PASS；C API GEMVI 五个用例通过，包含所有类型/方向、非零虚部 alpha/beta、padded lda、split 归约和 beta=0 不读输出。
- 其他后端是否也需要：否；计划选择与签名提示仅由 `musa_col` 启用，SM 缓存也检查 backend=musa。

### 2026-11-04 capi/flagsparse_codegen/gemvi.py::gemvi_musa_col

- 改了什么：新增 Triton kernel，将连续输出行置于最后一维，每个稀疏 lane 累加后一次归约；复数完整计算实/虚部及 alpha/beta，split 最终归约复用原 kernel。
- 为什么改（附报错原文）：无异常；减少原布局下跨 lane 归约与非连续访存。
- 怎么验证的：上述原四规模及五项 C API 精度用例；`final_gemvi/` 保存原始性能和精度报告。
- 其他后端是否也需要：否；原三个导出 kernel AST 不变，新 kernel 只有 MUSA 非转置 FP32 组件调用。

### 2026-11-04 tests/pytest/test_musa_sparse_routes.py::新增路径回归

- 改了什么：既有 FP16 gather/绑定 COO 启动回归扩展到 BF16、FP32×complex64；增加 inference tensors、缓存失效、流、实时数据和列主序 SpMM 的跨步输出检查。
- 为什么改（附报错原文）：无算子异常；防止优化缓存旧值、旧指针、旧流或改变输出语义。
- 怎么验证的：扩大回归 `1126 passed, 1694 skipped`（26.14 秒）；随后补充推理张量回归 `12 passed`，最终布局回归 `4 passed`。跳过规则不改；见 `expanded_accuracy`、`inference_accuracy`、`pack_accuracy_final` 日志与 JUnit。
- 其他后端是否也需要：否；该文件已有 MUSA 实机条件。

### 2026-11-04 capi/ctest/accuracy/test_spmv.cpp::CSC 共轭实时输入回归

- 改了什么：增加 CSC 直接共轭、非零虚部、非默认 alpha/beta、空列和实时值/RHS 更新用例。
- 为什么改（附报错原文）：无算子异常；定向性能语料复数值的虚部为零，需独立覆盖完整复数语义。
- 怎么验证的：SpMV 34 个精度用例全部通过，日志 `capi_accuracy_final.log` 和 `capi_accuracy_LastTest.log`。
- 其他后端是否也需要：否；新增用例仅 MUSA 执行。

### 2026-11-04 capi/ctest/accuracy/test_spmm.cpp::转置 gather 实时输入回归

- 改了什么：增加 COO/CSC、多个行长、packed/padded 布局、值/RHS 更新和 scratch 切换测试；COO 测非默认 alpha/beta，CSC 验证原限制。
- 为什么改（附报错原文）：初版新增用例错误地要求 CSC 非默认 alpha/beta 成功，实测 `Which is: 10`（NOT_SUPPORTED）；确认原接口合同后修正新增测试，生产接口限制未改。首次编译缺少 adaptor 命名空间/头文件的问题也已修正，初轮记录保留。
- 怎么验证的：最终 SpMM 28 个精度用例全部通过；CTest 初次过滤误用下划线未选中测试，改用 `^accuracy\.(spmv|spmm|gemvi)$` 后实际执行，最终三族 67 用例全部通过（16.25 秒）。
- 其他后端是否也需要：否；新增 SpMM 用例仅 MUSA 执行。

### 2026-11-04 capi/ctest/accuracy/test_gemvi.cpp::复数 split 与 padded lda

- 改了什么：增加 complex64 非转置 split、padded lda 及非零虚部 alpha/beta 的 CPU complex128 对照。
- 为什么改（附报错原文）：无异常；原 split 用例的复数方向为转置，不能覆盖新的 MUSA 非转置 kernel。
- 怎么验证的：GEMVI 五个用例全部通过；原测试规模、容差和基线不变。
- 其他后端是否也需要：计算实现不需要变更；新增语义回归可在支持该 API 的其他后端执行。

### 2026-11-04 范围检查与完整 105 项验收安排

- `scope_python.json`：三份 Python 文件移除 MUSA 分支后的 AST 与 `6bb11c5` 相同；五份 Python/codegen 文件原有共享 Triton kernel 不变。`scope_capi_review.json` 逐项记录四份 C++ 文件的 MUSA 条件及原非 MUSA 参数/路径保留情况。`git diff --check` 通过；未在其他加速器上实测。
- 本轮未改 runner、benchmark、数据、注册表、厂商基线或 libtriton_jit。全部新计算为 Triton；没有加入 MUSA 原生计算实现或用 PyTorch 索引替代计算。
- 上一完整轮除十个低于 0.8x 的变体外，另有 `spmv_csr_f32c32_int_trans` 在 ASIC_680ks 的 C API 精度门槛失败，strict ratio 为 1.363373，且没有本机混合精度基线可触发 relaxed 规则。本次没有改该 scatter 路线，完整验收须单独报告它，不能声称 105 项全部通过。
- 新完整轮准备使用 `results_mthreads_105_20261104_152855_low10/`，日志 `results_mthreads_105_20261104_152855_low10.log`；以 setsid 启动原 split runner，105 项精度/性能、原十矩阵、原四 GEMVI 规模、现行 H800 参照、串行 C API、线程池 8。预计约 1 小时；实际 PID、启动/结束时间、源码/矩阵 SHA256、命令和状态以该目录的 `RUN_INFO.json` 为准。
- 本条是启动安排，不是完整结果声明。结束后由 `post_audit.py` 检查 missing、精度/性能状态、H800 补位及逐变体平均加速比。旧轮、旧压缩包和全部探针保留。

### 2026-11-05 SpVV FP16→FP32 定向优化与 CSR 混合复数转置误差定位

- 用户本次要求：继续优化 `spvv_f16f32_int_non`，查明 `spmv_csr_f32c32_int_trans` 精度失败原因。起始提交 `52103191ed425f23f112f63ad7e682b5e03bcd8b`；完整证据保存在 `results_musa_opt_20261105_spvv_precision/`。未拉取或覆盖旧结果，未改 C API、benchmark、容差、矩阵、baseline 或 runner。
- 上一完整 105 项结果 `results_mthreads_105_20261104_152855_low10/` 已结束：105 项 pytest 精度 Passed，性能 104 Passed / 1 Failed，missing=0；103 项平均加速比 >0.8x，另两项为 SpVV 0.760891x 与下面的 CSR 精度门槛失败。本次是定向复测，没有重跑或改写该整轮结果。
- 原四个 SpVV 规模、seed=0、warmup=5、iters=20；基线是本次运行本机 PyTorch gather/product/sum，不是 H800 折算。正式 `spvv_final_all.csv` 的 24 条 SpVV 记录全部 PASS，其中 FP16 四例平均 1.345778x；两轮独立补跑均值 1.323435x / 1.223940x，四例均 >0.8x。保存首次优化前控制组 `spvv_before.csv`，其均值 0.828783x，体现与上一完整轮的计时波动，不能把不同轮 PyTorch 分母当成固定值。

| dense / nnz | 上一完整轮耗时 ms | 本次最终耗时 ms | 本次 PyTorch ms | 本次加速比 |
|---|---:|---:|---:|---:|
| 32768 / 1024 | 0.054723 | 0.024792 | 0.035625 | 1.436975x |
| 131072 / 4096 | 0.055078 | 0.023262 | 0.024154 | 1.038359x |
| 524288 / 16384 | 0.051871 | 0.023373 | 0.027209 | 1.164142x |
| 1048576 / 65536 | 0.051844 | 0.032625 | 0.056886 | 1.743637x |

### 2026-11-05 src/flagsparse/sparse_operations/_vector_musa.py::spvv_half

- 改了什么：新增 MUSA 专用 Triton FP16 输入、FP32 乘法/累加/输出 SpVV。nnz≤16384 一次归约，较长输入用分块加最终归约；最终归约 tile 最多 1024 并循环读取，避免超长输入产生无限增大的编译 tile。复用已有有界编译启动器，每次读取实时输入、输出地址和当前流；结果和临时区独立分配，无原子加法、无计算结果缓存。跨步索引仅在此路径连续化。
- 为什么改（附报错原文）：原实现总要分配 partial 并调用 PyTorch sum；启动和调度耗时占比高。新增跨步回归首次发现 `AssertionError: Scalars are not close! Expected -110.6341971308284 but got -30.58177947998047.`，原因是 int32 跨步索引未经连续化；已修复，首轮失败日志 `spvv_regression.log` 保留。
- 怎么验证的：原四例全部 PASS；新 MUSA 回归 90 passed，覆盖空输入、边界、重复/乱序索引、int32/int64、跨步和实时修改、输出生命周期、非法索引、return_time、确定性、双流及 1048577 项的最终归约循环；原 SpVV 精度用例 12 passed，共 102 passed。调参记录 `spvv_tuning.json`；未选用的单 CTA 65536 方案实测慢，未保留。正式 24 条 benchmark 全 PASS，结果见上表。
- 其他后端是否也需要：否；仅下面的 MUSA FP16 guard 导入此模块，计算全部为 Triton，没有原生 MUSA 实现或 PyTorch 索引计算。

### 2026-11-05 src/flagsparse/sparse_operations/vector_ops.py::flagsparse_spvv

- 改了什么：在现有校验、连续化、输出类型判断之后，仅 `_is_mthreads_runtime() and values.dtype == torch.float16` 调用上述 helper。原 op、out_dtype、validate、return_time 和错误语义保留。
- 为什么改（附报错原文）：无额外报错；将 FP16 归约优化限制在 MUSA，其他 dtype/后端保持原路线。
- 怎么验证的：102 项精度/回归、24 条原 SpVV benchmark 全通过；`audit_scope.py` 确认禁用 MUSA 后整个文件 AST 与起始提交相同，所有已有共享 Triton kernel AST 相同。`scope_audit.json` 保存源码 SHA256。
- 其他后端是否也需要：否；未在其他加速器上实测，非 MUSA 路径通过 AST 与分派 CI 验证不变。

### 2026-11-05 tests/pytest/test_spvv_musa_accuracy.py

- 改了什么：新增 90 项仅 MUSA 适用的半精度 SpVV 回归，用实际 FP16 输入的 CPU FP64 结果作参考，并检查结果形状和 FP32 dtype。
- 为什么改（附报错原文）：防止快速启动、单/两级归约遗漏边界或读取旧指针；上述跨步 int32 回归真实检出了缺陷，已在 helper 修复，未放宽其容差或删用例。
- 怎么验证的：最终 `spvv_regression_final.log/.xml` 中本文件 90 passed；与原 SpVV 12 项合计 102 passed，159 deselected 是命令按 spvv 选择其他测试，不是更改适用性规则。
- 其他后端是否也需要：否；模块级条件仅在 MUSA 实机运行。

### 2026-11-05 tests/ci/test_spvv_musa_dispatch.py

- 改了什么：新增 24 项 CPU mock 分派测试，覆盖 MUSA 开关、六种 dtype 和 non/conj，确认专用路径只接 FP16。
- 为什么改（附报错原文）：无异常；防止导入或调度新 helper 意外影响其他后端、bf16、实数/复数或整数路线。
- 怎么验证的：与 MACA 既有分派 CI 共 68 passed，见 `spvv_dispatch_ci.log/.xml`。
- 其他后端是否也需要：其他后端执行本 CI 也可验证其路径未改变；不需要计算实现改动。

### 2026-11-05 tests/ci/test_spvv_maca_dispatch.py

- 改了什么：MACA 后端/dtype mock 测试显式禁用 MUSA predicate，仅增加一行，防止在 MUSA 主机运行该 CPU mock 时混入实际后端状态。
- 为什么改（附报错原文）：无新增实际算子报错；该测试已有其他后端 predicates 的隔离，补齐新分支即可保持模拟语义。
- 怎么验证的：原 44 项 MACA CI 与新 24 项合计 68 passed，未修改断言或删除原有组合。
- 其他后端是否也需要：测试隔离修正可跨平台使用；MACA 及其他后端 kernel 未改。

### 2026-11-05 spmv_csr_f32c32_int_trans 精度根因（生产实现未改）

- 原失败全文：`result outside tolerance: strict ratio 1.363373; relaxed would pass (0.013634) but there is no vendor baseline to confirm the matrix is hard for both (spec 6.3.1 requires it)`；基线原文 `muSPARSE: mixed-precision SpMV baseline is not wired`。失败位于性能阶段的 C API benchmark 精度门槛，不能与已通过的两项小规模 pytest 精度混为一谈。
- 复现直接调用当前 `capi/build/libflagsparse.so`，矩阵由原 `capi/ctest/mmio.hpp` 解析器导出，RHS 逐项为原 `dense_pattern`，FP32 矩阵、complex64 输入/输出、alpha=1、beta=0。8 次运行中 3 次失败、5 次严格通过，失败时均为输出列 405878（0-based）。所有复现文件和真实输出保存在该目录，不改测试逻辑。
- 该列三项实际 FP32 乘积为 `0.11203476786613464`、`-74.05411529541016`、`+74.05411529541016`，来源行 405878、405880、406216。FP64 参考为 `0.11203476414084435`，按原规则转为 FP32 后是 `0.11203476786613464`。先把小数加进大数再抵消得到 `0.11203765869140625`，绝对误差 `2.8908252716064453e-6`，允许误差约 `2.1203476786613465e-6`，比值正好 `1.3633732338799878`。六种 FP32 累加顺序中四种重现该失败，两种先抵消大项得到准确结果。
- 根因是 `_csr_spmv_trans_real_by_complex_kernel` 的无序 FP32 `tl.atomic_add` 与灾难性抵消，不是复数虚部、错误索引或参考未按输入类型量化。原 benchmark 已将矩阵/RHS 量化后用 FP64 求参考；把 FP32 乘积用 FP64 汇总后整矩阵最大 strict ratio 仅 0.050558，证据排除乘法舍入为主要原因。
- 同轮 ASIC 的普通 FP32 CSR 转置也出现 ratio 1.363375；因为 muSPARSE 同样未过 strict，可按原 spec 6.3.1 记为 PASS(relaxed)。混合实数×复数 baseline 未接入，不能套用该豁免。已有 complex64 共轭转置走不同 gather 路线，同轮 strict ratio 0.049509，并非所有复数都触发这条 atomic 路线。
- 修复方向已用独立诊断 Triton kernel 验证：按输出列建立 gather 拓扑，用 FP64 累加后一次转 FP32；在原 ASIC 全量输入及非零虚部输入各重复三次，实/虚 strict ratio 均 0，结果逐位一致。此探针仅为根因验证，不替换生产实现，也不作为性能/交付通过结果。正式修复还需接入 MUSA C API 的 scratch/preprocess、完整复数/实时输入语义及原十矩阵验证；本次按要求完成定位，原生产失败状态保留。
- `git diff --check` 与范围审计通过；本次六个源码/测试/文档文件一起提交，旧结果、旧归档保留。

### 2026-11-05 修复 spmv_csr_f32c32_int_trans 的 MUSA C API 精度

- 用户进一步授权正式修复。起始提交 `8b5e7add9aa25387cf1610e21e974ebdd843c4c4`；新证据目录 `results_musa_fix_20261105_csr_complex_trans/`，旧轮和定位探针均保留。此节是正式生产接入，上一节“生产实现未改”描述的是前一提交的定位阶段。
- 原 `tests/data` 十矩阵 SHA256/大小与上一完整轮 `RUN_INFO.json` 一致；未改原 benchmark、容差、基线、runner、Python kernel 或索引策略。此次是定向验证，不是新的完整 105 项验收。
- C API 原十矩阵的该变体 **10/10 strict pass，strict ratio 全为 0**；三个相邻变体也通过，共 40 条精度记录、0 failed。原 ASIC 失败输入直接调用生产 libflagsparse.so 连跑 8 次，strict ratio 全为 0。记录见 `capi_target/spmv_benchmark.json`、`spmv_accuracy.json`、`precision_after.json`。
- 目标 C API 十矩阵平均耗时 0.231383 ms，ASIC 为 0.308179 ms；厂商仍返回 `muSPARSE: mixed-precision SpMV baseline is not wired`，因此 C API 投影是 NoBaseline，不伪装成本机厂商加速比。`target_split_projection.json` 仅对该目标调用原 `apply_h800_scaled_baseline`，按现有优先级采用本次 Python 对同一 harness H800 的折算参照，性能 Passed、均值 **1.173173x（估算）**。完整 105 项尚未重跑，不能由此宣称整轮全部通过。
- 原 Python CSR 流程为 `--ops spmv_csr --delivery-only --mode normal --phase both --gpus 0 --benchmark-input tests/data --h800-reference`，精度 `487 passed, 1680 skipped`（原适用性跳过），性能 CSV 150/150 PASS；目标 pytest 2/2 passed、目标原十矩阵 10/10 PASS。原 C API SpMV 回归最终 37/37 passed，含新增非默认流。

### 2026-11-05 capi/src/ops/spmv.cpp::MUSA 实数乘复数 CSR 转置 gather

- 改了什么：仅 `backend=musa && CSR scatter request && real_by_complex` 分支报告 gather 工作区、支持显式 preprocess 或首次计算惰性准备，并调用新 Triton kernel。复用已有 `prepare_sparse_gather` 的输出列指针、原值位置和源行拓扑；工作区不缓存值或 RHS。工作区切换、显式重新预处理、`flagsparseCsrSetPointers` 后按既有描述符失效规则重建。非空输出传空工作区返回 INVALID_VALUE，并说明应使用 bufferSize；原 alpha=1、beta=0 限制及其他索引限制保留。
- 为什么改（附报错原文）：原 `result outside tolerance: strict ratio 1.363373; relaxed would pass (0.013634) but there is no vendor baseline to confirm the matrix is hard for both (spec 6.3.1 requires it)`。无序 FP32 atomic 累加在 ASIC 发生大项抵消，已由前一节精确定位；正式路径改为每个输出列单独拥有归约。
- 怎么验证的：生产库 8 次 ASIC 复现、原十矩阵、40 条相邻变体记录、37 项 C API 回归全部通过；调用方值/RHS 原地更新、索引重预处理、SetPointers、双工作区和非默认流均覆盖。C API 返回的工作区大小变大，调用方须按 bufferSize 分配并传入；原 C API benchmark 已按此契约执行，无需修改。
- 其他后端是否也需要：否；`audit_scope.py` 去除三个 MUSA 保护分支及两个新 helper 后，C++ 非 MUSA 代码与起始提交逐字一致；preprocess 的原校验返回按等价分支还原。未在其他加速器实测。

### 2026-11-05 capi/flagsparse_codegen/sparse_gather.py::musa_csr_spmv_trans_real_complex_f64

- 改了什么：新增 MUSA 专用 Triton 实数矩阵乘复数向量的转置 gather；实、虚部分别使用 FP64 乘法/累加与固定归约，最后各转 FP32 写回 complex64。支持空列、重复项、部分 tile、长列循环以及拓扑宽地址；不使用 atomic，不读取旧输出，不加入原生 MUSA 或 PyTorch 计算。原有 kernel 定义不变。
- 为什么改（附报错原文）：上述 ASIC strict ratio 1.363373；FP32 原子加法先合并小项再抵消大项时丢失精度，必须在归约过程中保留小项。
- 怎么验证的：十矩阵 strict ratio 均为 0；非零虚部、trans/conj（实数 A 两者等价）、int32/int64 描述符、重复列、长列、空输入、重复执行和非默认流回归通过。运行配置只依平均列长选 R/K，不依矩阵名选路。既有 codegen AST 与起始提交完全相同。
- 其他后端是否也需要：否；新增符号仅由上面的 MUSA helper 调用，其他后端仍走原 atomic 路径。

### 2026-11-05 capi/ctest/accuracy/test_spmv.cpp::MUSA 转置归约回归

- 改了什么：既有 RealFloat32ByComplex64ScatterRoutes 仅对 MUSA CSR 转置接受新的非零 bufferSize 并传入工作区，保留所有数值断言和其他后端的零工作区断言。新增三个 MUSA 回归，覆盖 ASIC 的三项抵消、非零虚部、重复列、长列、空列/空行/空输出、int32/int64、显式/惰性预处理、值/RHS 更新、索引重新预处理、SetPointers 失效、工作区切换、beta=0 覆盖 NaN、逐位重复性及非默认流。保留非默认 alpha 的 NOT_SUPPORTED 断言。
- 为什么改（附报错原文）：原严格误差失败见上；避免只修 corpus 的零虚部输入，却破坏复数或准备计划语义。测试使用原默认容差，无放宽或删除失败案例。
- 怎么验证的：最终 `capi_spmv_accuracy_final.log/.xml` 为 37 passed；早期 36 项日志保留。使用 `muStreamCreate` 仅创建测试流，所有算术仍由 Triton 执行。
- 其他后端是否也需要：生产实现不需要改；新增用例仅 MUSA 运行，MUSA driver header 和流用例均由既有编译宏保护。

### 2026-11-05 定向验证过程与范围审计

- 定向验证通过 setsid 串行启动，避免 GPU 性能测试重叠；`launch_target.py` 的前两个阶段成功。首次 Python 阶段误传该入口不支持的 `--backend mthreads`，报 `error: unrecognized arguments: --backend mthreads`，未开始算子测试；原失败启动记录 `RUN_INFO.json` 和 `python_csr.log` 保留。移除该参数、通过原 `FLAGSPARSE_BACKEND=mthreads` 环境配置，用 setsid 在全新 `python_csr_final/` 目录重跑成功；没有修改 runner 或掩盖失败日志。最终结论以该目录实际精度/性能 exit_code 和最终 C API JUnit 为准。
- `scope_audit.json` 确认非 MUSA C++ 路径和全部已有 codegen AST 不变；新计算全部为 Triton。`git diff --check` 通过。本次三份代码/测试文件及本记录一起提交，未推送或重写旧结果。

### 2026-11-05 完整 105 项结果与三项后续优化

- 起始提交为 `7a18b6f`，代码树与当轮上游 `5bad1a0` 一致。完整轮 `results_mthreads_105_20261105_032658_upstream/` 于 UTC 03:27:13–04:27:34 完成，105 项精度和性能均 Passed，missing=0；性能 Passed 表示可报告结果，不表示达到 0.8x。仍低于 0.8x 的三项是 SELL complex64 0.792618x、SpVV int8→FP32 0.794818x、SpVV int8→int32 0.788876x。
- 用户授权仅优化上述三项。本轮证据目录 `results_musa_opt_20261105_low3/`，未再次拉取代码。原矩阵、注册表、baseline、runner、容差、warmup=5/iters=20 和基线优先级均不变；未改 C API 或已有共享 kernel。生产改动只在 MUSA 路径生效，全部计算保持 Triton。
- 原 `run_flagsparse_pytest.py --ops spvv,spmv_sell --delivery-only --mode normal --phase both --gpus 0 --timeout 3600 --benchmark-input tests/data --h800-reference` 定向结果在 `python_targets_final/`：SpVV 精度 220 passed、178 skipped（原后端适用性规则），SELL 精度 12 passed，两个性能阶段共 74/74 原始记录 PASS。逐变体平均：SELL complex64 **0.867253x（H800 带宽折算）**，SpVV int8→FP32 **1.251736x**、int8→int32 **1.289654x**（同轮本机 PyTorch）。这些是定向结果，不是新的完整 105 项验收。
- 同口径独立测量中 SELL 首轮为 0.866142x；SpVV 首轮为 1.221583x / 1.226363x。优化前控制组与上一完整轮有计时波动，特别是 roadNet-TX 由约 0.163 ms 变为约 0.084 ms；不能把这部分变化全部归因于优化。新旧同轮对照、全部探索和失败记录均保留，未挑选或覆盖旧数据。
- 新完整轮使用 `results_mthreads_105_20261105_053710_low3/`，以 setsid、原 split runner、GPU 0、线程池 8、原十矩阵和现行基线启动，并重新执行 C API 构建。实际启动/结束时间和退出码以 `RUN_INFO.json` 为准；完整轮结束前不宣称全部 105 项达到 0.8x。

### 2026-11-05 src/flagsparse/sparse_operations/_vector_musa.py::SpVV int8 归约

- 改了什么：新增独立的 int8 Triton 分块和最终归约，nnz≤16384 单 kernel 直接写独立结果，长输入两级归约；最终 tile 有界并循环处理。int32 输出全程 int32 累加，FP32 输出全程 FP32 累加，避免 FP32 路线先发生 int32 溢出；跨步索引连续化。已有 FP16 kernel 不变，不缓存输出或输入值。
- 为什么改（附报错原文）：无生产算子异常；原两条 int8 路线需调度通用 kernel 和 PyTorch sum，短向量的约 43–46 us 调用成本使均值略低于 0.8x。
- 怎么验证的：上述原四个合成规模及完整 SpVV 24 条性能记录通过；新增 int8 边界、空输入、重复/乱序索引、int32/int64 索引、跨步、实时更新、独立结果、超过 int32 范围的 FP32 累加、int32 回绕、确定性和非默认流回归通过。原 FP16 回归不变。
- 其他后端是否也需要：否；仅 MUSA int8 guard 调用，新增两个计算 kernel 都是 Triton，没有原生 MUSA 实现。

### 2026-11-05 src/flagsparse/sparse_operations/vector_ops.py::MUSA int8 分派

- 改了什么：仅 `_is_mthreads_runtime() and values.dtype == torch.int8` 调用新归约 helper；保留现有校验、输入连续化、compute/out_dtype、op 和 return_time 语义。
- 为什么改（附报错原文）：无生产异常；将 int8 性能修正隔离到 MUSA，其他后端和 dtype 保持原路线。
- 怎么验证的：SpVV GPU/分派测试合计 284 passed；原 runner 定向精度全部通过。`audit_scope.py` 验证禁用 MUSA 后 AST 与起始提交相同、原有共享 Triton kernel AST 不变。
- 其他后端是否也需要：否；其他加速器未实测，非 MUSA 路径通过 AST 和 CPU mock 检查。

### 2026-11-05 src/flagsparse/sparse_operations/spmv_sell.py::MUSA complex64 slice 调度

- 改了什么：仅 MUSA、complex64、slice_size=32、非极短行配置复用已有 `_spmv_sell_complex_parallel_kernel`，使用标量 slice 边界、连续行 lanes 和 4 个稀疏槽；极短行保留原 128 行分组，其他 dtype 和 slice 大小保留原路径。没有改任何已有 kernel 定义、缓存或值更新语义。
- 为什么改（附报错原文）：无生产异常；原通用按行分组 kernel 的向量边界/归约及访存开销偏高。成对加载、64 位打包及压缩拓扑探针未显示稳定整体收益，未接入。
- 怎么验证的：原十矩阵及其他 SELL dtype 共 50/50 性能记录 PASS；SELL/分派回归 47 passed，覆盖非零虚部、重复列、空行、NaN padding、最后部分 slice、跨步输入/输出、实时值/列更新、存储替换和非默认流。范围审计确认非 MUSA AST 和全部原有 kernel 不变。
- 其他后端是否也需要：否；新选择位于原 `if musa` 内，其他后端仍按原调度运行。

### 2026-11-05 tests/pytest/test_spvv_musa_accuracy.py::int8 回归

- 改了什么：新增 118 项仅 MUSA 的 int8 精度/语义回归，以 CPU int64 数学结果为参考；int32 按原输出规则截断，FP32 不经 int32 中间累加。保留原 90 项 FP16 回归和容差。
- 为什么改（附报错原文）：无新增生产异常；避免单/两级归约仅对交付小输入正确，却破坏长输入、跨步、流或溢出语义。
- 怎么验证的：连同原 FP16 和两个后端分派文件为 284 passed，日志 `spvv_regression.log/.xml`；原 runner SpVV 的 220 项精度通过。
- 其他后端是否也需要：否；新增 GPU 用例仅在 MUSA 运行。

### 2026-11-05 tests/ci/test_spvv_musa_dispatch.py::int8 分派范围

- 改了什么：保留原 dtype/non/conj 组合，并增加 int8 默认、显式 int32、显式 FP32 输出的 MUSA/非 MUSA 分派断言。
- 为什么改（附报错原文）：无算子异常；新 helper 必须只接 MUSA int8，不能误入 MACA 或其他 dtype。
- 怎么验证的：本文件 32 passed，既有 MACA 分派 44 passed；断言仍检查实际选择的路径。
- 其他后端是否也需要：无需生产修改；mock 可在其他后端运行。

### 2026-11-05 tests/pytest/test_musa_sparse_routes.py::复数 SELL slice 边界

- 改了什么：新增四项 MUSA complex64、C=32 的部分 slice/实时数据回归，包含重复列、NaN padding、跨步存储、输出哨兵和非默认流。
- 为什么改（附报错原文）：新增 fixture 初轮将 NaN padding 的列改为合法列，却未同步设置有限值，报 `Greatest absolute difference: nan at index (0,)`；已在新 fixture 的该槽设为 `2+3j` 后再转为合法列。没有修改生产实现或放宽断言，初轮失败日志保留。
- 怎么验证的：修正后原 SELL 回归和本文件新用例全部通过，见 `sell_regression_final.log/.xml`（连同分派 47 passed；124 deselected 是命令筛选其他算子）。
- 其他后端是否也需要：否；本文件原已限定 MUSA。

### 2026-11-05 tests/ci/test_spmv_sell_musa_dispatch.py::SELL 分派隔离

- 改了什么：新增 24 项 CPU mock，交叉检查 MUSA 开关、complex64/complex128/FP32、C=16/32、短/普通行宽。
- 为什么改（附报错原文）：无算子异常；确保只选择已验证的 MUSA complex64 C=32 路径。
- 怎么验证的：24 passed，属于上述 SELL 合计 47 passed；`git diff --check` 与 `scope_audit.json` 通过。
- 其他后端是否也需要：不改其他后端实现，测试可跨平台运行。


### 2026-11-10 MUSA 剩余 8 项优化：验证条件和阶段结果

本轮依据 `results_mthreads_105_20261110_083219_upstream_d52ac62_musa_opt_unbound_defaultthreads/summary_split.json`，该轮 105/105 精度、性能状态 Passed，8 项算子均值低于 0.8。保持原 10 个 tests/data 矩阵、基线优先级、全部慢行、精度容差和计时参数；不绑核，OMP_NUM_THREADS、MKL_NUM_THREADS、OPENBLAS_NUM_THREADS 均不设置。Python 默认 255/128 线程，未修改线程数；H800 折算项仍为估算，不改称本机厂商基线。

### 2026-11-10 capi/src/ops/spmv.cpp::MUSA COO FP32 转置

- 改了什么：仅 backend=musa、COO、FP32、转置/共轭转置走三个 Triton kernel，清零 FP64 工作区、按 nnz 分块累加 FP64 乘积、融合 alpha/beta 转 FP32 写回。bufferSize 报告输出长度乘 sizeof(double)，预处理不缓存值和拓扑，计算读取实时 row/col/value/x；每次从 handle 获取当前流。空输出无需工作区。撤回上一轮间接输出行 gather，因此 capi/src/core/sparse_gather.hpp 恢复到 HEAD。
- 为什么改（附报错原文）：无异常；上一轮均值 0.646625，列长度不均及间接读取使行归约较慢。FP32 原子累加不能保留抵消项，改为 FP64 乘法/原子累加后只转换一次。
- 怎么验证的：38/38 C API SpMV 精度通过，含 int32/int64、长列、alpha/beta、beta=0 时 NaN 输出、输入/拓扑更新及非默认流。原 10 矩阵官方定向测试 10/10 精度通过，平均 0.933926，原 warmup=10/iters=100/median 不变。证据 `results_musa_opt_20261110_low8/capi_atomic64_v1/`、`capi_final_accuracy.log`。
- 其他后端是否也需要：否；运行时 predicate 显式要求 musa，其他后端保持原分派。

### 2026-11-10 capi/flagsparse_codegen/sparse_gather.py::COO FP64 三阶段内核

- 改了什么：新增 musa_coo_transpose_zero/scatter_f64/finish，全 Triton，nnz tile=128、向量 tile=512，地址提升为 int64。beta=0 不读取旧输出，累加工作区不跨调用复用数值。
- 为什么改（附报错原文）：同上一条；替代上一轮未达标的 COO 输出行 gather。已有 CSR 实乘复数 FP64 gather 等内核保持不变。
- 怎么验证的：同上一条 C API 精度和官方 10 矩阵定向结果；其他精度内核继续由原测试验证。
- 其他后端是否也需要：否；新增符号只被 MUSA predicate 调用。

### 2026-11-10 src/flagsparse/sparse_operations/mixed_spmx.py::分组 SELL、短行布局和成对加载

- 改了什么：保留已有输入版本/存储/元数据失效和实时 RHS/流/输出语义；新增 MUSA 专用 slot-major 分组 SELL kernel，在低 padding、中等行长的大规模 int8/实乘复数路径使用。规则极短行选固定行上界归约。FP16 规则短行在支持 TLE 时使用显式 blocked layout，模块仅在 MUSA 导入，缺 TLE 保留原路径。FP16 中小矩阵可用 Triton 打包精确拓宽值和 int32 列号，每项 64 位加载；所有值打包由输入缓存版本失效重建，不缓存结果。前轮改动也保留：CSR 输入校验复用、COO helper 直接调用、CSR 转置实乘复数 FP64 归约。
- 为什么改（附报错原文）：无算子异常；8 项均值未达 0.8，短行的访存布局、RHS 间接加载与主机调用开销限制性能。默认线程下 CPU 限额停顿仍被保留，不通过修改测试规避。
- 怎么验证的：多组扫参及被拒候选全部保存在 `results_musa_opt_20261110_low8/`。定向 final_rows.json 保留 60 条原始计时，全部精度 PASS；FP16、实乘复数、COO i8f32/i8i32、CSR i8i32 均值分别为 0.920779、0.859470、0.806942、0.891465、1.162907。FP16->FP32 本轮仍有一次约 4.8ms 停顿，该次均值 0.745844，尚不能据此宣称 8 项全达标。完整验证结果以后续独立结果目录为准。
- 其他后端是否也需要：否；新 kernel 和选择仅由 MUSA 专用 helper 触达，现有其他后端算法保持原实现。

### 2026-11-10 src/flagsparse/sparse_operations/spmv_sell.py::规则 FP16 SELL

- 改了什么：仅 MUSA FP16、大规模、平均及最大 slice 宽度满足短行条件时，准备 CSR 视图以调用上述 TLE 路径；保留高 padding 压缩、版本失效和输出存储语义。前轮 MUSA FP32 的 32 行 tile 复用原 wide kernel 改动保留。
- 为什么改（附报错原文）：无异常；SELL FP16 上一轮均值 0.691629，规则短行可减少冗余 slice 访存。
- 怎么验证的：原 10 矩阵 SELL 定向精度全过；TLE/成对加载集成后的最终定向和完整 105 项证据见本轮结果目录。
- 其他后端是否也需要：否；_musa_sell_csr_plan 只在 MUSA 分支调用，TLE 导入受 MUSA 开关保护。

### 2026-11-10 tests/pytest/test_musa_sparse_routes.py::新增路径语义回归

- 改了什么：增加 grouped SELL 的尾部、不等行长、int32/FP32/complex64 输出及实时输入检查；大规模规则 FP16 的 COO/CSR/SELL 公共 API 检查覆盖部分 tile、不同输出、持有历史结果及输入更新；成对加载检查值/列/RHS 修改和 .data 存储替换。测试只在 MUSA 实机运行，原断言和容差不放宽。
- 为什么改（附报错原文）：防止新拓扑和值打包造成旧值、旧存储或覆盖历史输出。初版测试误改 prepared COO 之外的原 values，已改为修改 prepared.data_non；成对加载原 FP16 CSR 未走 mixed 路径，改用实际支持此路径的 COO 公共入口验证。失败日志完整保留。
- 怎么验证的：修正测试后的 TLE 专项 6 passed、成对加载专项 2 passed；全套最终结果保存 python_verified_regression 等后续日志，不将中途失败记作通过。
- 其他后端是否也需要：否；原文件具有显式 MUSA 实机条件。

### 2026-11-10 capi/ctest/accuracy/test_spmv.cpp::COO 转置回归

- 改了什么：保留新增的 COO 转置抵消/长列/实时数据/alpha-beta/int32-int64 测试，并在 MUSA 非默认流检查中运行两种索引 COO 转置。
- 为什么改（附报错原文）：无算子异常；FP64 原子归约需要验证抵消项、空输出及实时流语义。
- 怎么验证的：重新构建后 38/38 SpMV 精度通过，未删除或放宽原断言。
- 其他后端是否也需要：否；非默认流扩展受 MUSA 宏保护，其他新增用例显式检查后端。


本轮最终精度记录（2026-11-10）：`results_musa_opt_20261110_low8/python_complete_regression.log` 为 **346 passed、4 skipped**；`capi_complete_accuracy.log` 为 **39/39 passed**，含新增 COO 空矩阵/零输出/beta 检查。`final_half_rows.json` 的追加单变体复测仍保留 amazon0601 的约 4.45ms 停顿，均值 **0.693244**；不替换或取多轮最好值。最终达标数等待后续完整 105 项实测，不能由定向结果拼出 8/8 达标。

### 2026-11-11 剩余 8 项：不绑核、默认线程下继续调优

上一完整轮 `results_mthreads_105_20261110_135953_upstream_d52ac62_musa_opt_unbound_defaultthreads/` 为 105/105 精度 Passed、97/105 均值大于 0.8。8 项为 scatter FP16、COO FP16/FP16→FP32/FP32×complex64、CSR BF16/BF16→FP32、SELL FP16/int8→int32。本轮仍使用原 tests/data 十矩阵、原基线优先级/容差/计时参数，不绑核，不设置 OMP_NUM_THREADS、MKL_NUM_THREADS、OPENBLAS_NUM_THREADS，也不调用线程数设置接口。

### 2026-11-11 src/flagsparse/sparse_operations/mixed_spmx.py::BF16 路由与实乘复数融合扫描

- 改了什么：MUSA 半精度 TLE、成对加载和低 padding SELL 选择扩展到 BF16，转换为 FP32 累加，保持原输出 dtype。MUSA FP32 实乘 complex64 的 NNZ 路径新增 Triton 三元分段扫描，共用行号比较并以 64 位加载完整复数 RHS；保留原输出行唯一所有权、跨块前缀重算和空行清零。值、索引、RHS、输出、当前流和原缓存失效语义不变，未缓存结果。
- 为什么改（附报错原文）：无生产异常；BF16 之前未复用已验证的半精度路线，复数 NNZ 原路径分别扫描实部和虚部。原 ASIC 探针约 58.9 us，融合扫描加完整 RHS 加载候选约 53.7 us；此为定向探针，不代表完整变体均值。
- 怎么验证的：`results_musa_opt_20261111_low8/final_v1_rows.json` 的 50/50 原矩阵精度通过，BF16/BF16→FP32/COO FP16 均值 0.845239/0.851785/0.811480；COO FP16→FP32/实乘复数仍有慢样本，均值 0.600191/0.588707，不能宣称全达标。融合扫描随后执行跨块、部分 tile、空行、随机非零虚部、int32/int64、跨步输出、输入更新和非默认流检查，并重测原十矩阵；最终日志见本轮 `regression_final.log` 和 `complex_final_*`。
- 其他后端是否也需要：否；本轮只改 `_musa_coo_half_spmv` 及新增 MUSA kernel。原 kernel 定义 AST 不变；选择只由 MUSA helper 调用。

### 2026-11-11 src/flagsparse/sparse_operations/gather_scatter.py::scatter launcher 复用

- 改了什么：在已有索引版本缓存中复用 Triton launcher，按 dtype、device、对齐、alias、stride 和 debug 状态区分，实时读取当前指针和流；保留 profiling/debug 回退、推理张量不缓存、重复索引最后一个值胜出的规则。缓存仅保存拓扑和调度，最多 16 个 launcher。
- 为什么改（附报错原文）：无生产异常；短 kernel 的重复 JIT 分派增加主机时间。
- 怎么验证的：原 4 个合成规模全部精度 PASS，`scatter_official.csv` 保留最大规模约 0.971 ms 慢样本；正常样本约 13–15 us。新增跨步、重复索引、存储替换、非默认流和 inference 张量回归；未修改原计时方法。
- 其他后端是否也需要：否；只改 `_musa_scatter_dense`。原 Triton kernel 定义及其他后端调用不变。

### 2026-11-11 src/flagsparse/sparse_operations/spmv_sell.py::int8 条目打包

- 改了什么：MUSA、C=32、列号范围允许的中等行宽 int8 SELL，将 24 位列号和原 signed int8 值由 Triton 无损打包到单个 32 位条目，低 24 位为 col+1，零表示 padding；新 Triton 计算 kernel 按原 int32/FP32 累加语义读取，支持部分 slice 和跨步输出。原版本/存储/元数据失效规则负责重建打包数据，输出不缓存；launcher 支持四个或五个张量参数。
- 为什么改（附报错原文）：无生产异常；减少独立值/列数组的访存，amazon/auto 探针约 55→48 us、80→74 us。未将探针最好值替换正式结果。
- 怎么验证的：原十矩阵 FP16/int8 共 20/20 精度 PASS；`sell_pairs.csv` 按上一完整轮固定基线的均值为 0.665824/0.761067，慢样本全部保留，尚未达标。新增 -128/127、padding、部分 slice、跨步输出、输入/存储更新和 return_time 回归。`scope_audit.json` 检查去掉 MUSA 分支后的公共入口 AST 与本轮开始时一致。
- 其他后端是否也需要：否；打包和新 kernel 仅在原 `if musa` 分支调用，原有 kernel 定义不变。

### 2026-11-11 tests/pytest/test_musa_sparse_routes.py::本轮语义检查

- 改了什么：扩展 BF16 TLE/成对加载/SELL 准备测试，增加 scatter launcher、SELL int8 打包和实乘复数融合扫描回归，保留原精度容差及断言。
- 为什么改（附报错原文）：防止打包缓存和 launcher 复用造成旧值或旧存储。新增复数随机测试初轮在设备执行哨兵比较报 `SetMUTensorDType Unsupported tensor dtype: ComplexFloat`，已将该比较放到 CPU；不改生产逻辑或容差，失败日志保留。
- 怎么验证的：首阶段 `regression_v1.log` 为 363 passed、4 skipped；新增融合扫描旧语义回归 14 passed。最终更完整记录见 `regression_final.log`。所有新增实机用例受本文件 MUSA 条件限制。
- 其他后端是否也需要：否；非 MUSA 分派同时通过静态范围审计及现有 CPU mock 回归验证。

实验留档：压缩半精度列号、TLE 实乘复数布局/eviction、64 位值列对、sigma 行长重排、persistent、长短行分流和完全展开未见稳定整体收益，未接入。首次融合扫描实验有循环变量形状冲突，v2 使用独立前缀变量修正；原始失败日志保留。`bf16_launch` 与首轮 TLE 探针曾短暂重叠，相关计时仅作诊断，最终定向和完整轮保持串行。所有完整验收以新独立目录的 summary_split.json 为准，不拼接多轮最好值。

### 2026-11-11 tests/ci/test_spmv_sell_musa_dispatch.py::补齐已有 FP32 wide 路由 mock

- 改了什么：mock 包含 `_spmv_sell_wide_kernel`，并断言 MUSA FP32、C=32、普通宽度选择该路径；原 complex64/C=32 专用路径及非 MUSA 断言保留。
- 为什么改（附报错原文）：扩大验证时旧 mock 报 `AttributeError: 'JITFunction' object has no attribute 'name'`；其遗漏前轮已接入的 FP32 wide kernel。与本轮 int8 打包和复数融合无关，未为此改生产逻辑。
- 怎么验证的：`sell_dispatch_final.log` 为 24 passed；首轮扩大检查日志 `regression_final.log` 保留 420 passed、4 skipped、1 failed，其中该失败已单独修复复验。实机两文件为 365 passed、4 skipped，SpVV 分派 32 passed。
- 其他后端是否也需要：不改任何后端生产实现；该 mock 可跨平台运行。

补充（本轮冻结前）：FP16 SELL 中小矩阵、实际平均行长 >12、最大 slice 宽度 ≤128 时，MUSA compact CSR 计划进一步复用既有成对加载 kernel。原十矩阵探针对比 cfd2 约 37.8→31.1 us、wave 约 32.9→31.1 us；接入后新增跨步输出/非默认流/值与索引存储替换测试，相关合计 14 passed（`sell_half_pairs_regression.log`）。`sell_half_final.csv` 为原十矩阵 10/10 精度 PASS、固定原基线均值 **0.840969**，保留 roadNet-TX 约 4.86ms 慢样本。scatter 官方 4 个规模按同一轮固定基线均值 **1.147117**，同样保留全部慢样本。上述均为独立定向结果，不能替代完整 105 项达标数。

本轮完整复测由 `results_musa_opt_20261111_low8/launch_full.py` 启动：`setsid timeout -s KILL 43200 /usr/bin/python`，官方 `run_flagsparse_delivery.py --gpus 0 --results-dir ...`；实际独立目录见 `artifacts/latest_105_retest_directory.txt` 和其 RUN_INFO.json。开始后冻结代码，原十矩阵、105 变体、基线/容差和计时设置不变，不同时运行其他 GPU 实验。启动、完成状态及最终未达标列表以该轮 RUN_INFO.json、summary_split.json、completion_audit.json 为准。

最终串行复数十矩阵定向：`complex_serial_rows.json` 为 10/10 精度 PASS，均值 **0.820636**；这轮无其他 GPU 实验或 CPU mock 并行。之前 `complex_final` 的 0.767139 因曾与短时 CPU mock 并行，只作诊断且日志保留。冻结前按各自最后有效定向统计，8 项中 6 项 >0.8，COO FP16→FP32（0.600191）和 int8 SELL（0.761067）仍低于阈值；不能由这些分项测试宣称完整轮达标。

### 2026-11-11 默认线程下的稳定性排查（完整轮 021131 后）

完整轮 `results_mthreads_105_20261111_021131_upstream_d52ac62_musa_opt_unbound_defaultthreads/` 历时 2:57:53；105 项 accuracy 汇总为 Passed，但 C API `spmv_csr_f32_int_trans` 的 ASIC 原始精度记录失败（strict ratio 1.363375，厂商通过 strict），使 performance 为 104 Passed/1 Failed。不能把汇总 accuracy 105 Passed 解释成 C API 全通过。其余性能 9 项低于 0.8，95 项有效 Passed 且 >0.8。原 8 项中的 scatter、复数 COO 和两条 BF16 已超过阈值，同时其他相邻变体出现慢样本。

新证据目录：`results_musa_opt_20261111_stability/`。保持原 10 矩阵、固定完整轮基线、容差和 wallclock warmup=5/iters=20，不绑核，不设置 OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS。CPU 配额 `1500000 100000`（相当于 15 核），PyTorch 默认 255/128 线程。本轮 `diagnose_timing.json` 的原计时器未改，20 条记录中 4 条 >1ms（3.67–4.46ms）均伴随 cgroup 限流计数增长；计数范围包括预热，不能单凭此确定每个慢样本的唯一原因。GPU kernel 提速无法保证消除主机被调度暂停。

显式环境候选 `OMP_WAIT_POLICY=PASSIVE` 仅让 OpenMP 空闲线程等待时休眠，不改变线程数，不绑核；生产代码和官方 runner 默认均未设置该变量。三组预先约定、交替顺序、独立进程的默认/PASSIVE 对照，原矩阵和原计时不变，全部 120 条精度 PASS：COO FP16→FP32 默认 0.792071/0.772826/0.733021，PASSIVE 0.890547/0.884785/0.887093；COO int8→int32 默认 0.966848/0.903816/0.752937，PASSIVE 0.995565/0.994515/1.003872。全部样本保留，不挑最好一轮；它是不同环境的对照，不能冒充原默认等待策略下的验收。原始记录在 `wait_*`，摘要 `wait_policy_summary.json`。尚须全部 9 项和完整 105 项验证，有限轮次的稳定不代表绝对不会波动。

### 2026-11-11 src/flagsparse/sparse_operations/spmv_sell.py::TLE int8 布局和元数据上界

- 改了什么：新增 MUSA Triton/TLE `_musa_sell_i8_tle`，使用版本检查保护的最大 slice 宽度和显式 row/slot 布局计算无损 packed int8 数据，保留 int32/FP32 累加。允许短行和至多 32 padded 槽/行打包；长行仍保留原 packed kernel。仅原 `if musa` 分支选择新路径，不改现有共享 kernel 定义。准备 paired/TLE 路由时，上界由阻塞复制的 offsets 元数据在主机计算，不复制矩阵值、向量或结果。
- 为什么改（附报错原文）：正常 int8 SELL 探针 GL7d14/amazon/cfd2 约 26.6→22.7、52.8→47.0、32.5→24.8 us。完整回归还暴露旧 FP16 路由 `KeyError: 'csr_launches'`：数值 fallback 正确，但设备临时差分 max 在非默认流偶发返回 16843009，原 offsets=[0,544,1088,1632]，正确上界为 544，导致快速路径被拒。单独将 max 结果复制 CPU 和替换 row-id 生成均未解决，已撤回；从原 offsets 的阻塞主机拷贝计算冷元数据后，原 4 种 FP16 输出组合连续 10 组（120 次路径检查）全部选择成对加载。没有通过放松路线断言绕过问题。
- 怎么验证的：`sell_tle_regression.log` 13 passed；非默认流复现及失败候选日志完整保留，最终 `reproduce_half_plan_host_bound.log` 为 10 组通过、`reproduce_half_plan.json` 为 120/120 路径选择。扩大精度与原十矩阵测量见本轮后续 `regression_host_bound.log`、`all9_*`，所有慢样本保留。首次 TLE 动态循环触发 SliceEncoding 编译异常，候选改用缓存失效保护的固定宽度，未修改后端编译器。
- 其他后端是否也需要：否；原共享 Triton kernel 定义 AST 和去掉 MUSA 分支后的 SELL 公共入口 AST 与本轮起点一致，证据 `scope_audit.json`。

### 2026-11-11 src/flagsparse/sparse_operations/mixed_spmx.py::CSR/COO int8 复用打包 SELL

- 改了什么：MUSA 低 padding int8 CSR/COO 计划将 SELL 值和列号打包后复用上述 TLE kernel（mean≤12），较长行使用既有 packed grouped kernel。原源数据/拓扑版本、存储和元数据签名负责失效，不缓存数值输出。行长上界同样由原 ptr 元数据的阻塞主机拷贝计算，防止非默认流临时 max 返回错误上界。
- 为什么改（附报错原文）：无新增生产异常；amazon 探针约 54.3→46.7 us，auto 约 84.4→73.0 us，为 int8 COO/CSR 留出更大余量。原半精度元数据异常见上一条，不修改其他后端归约实现。
- 怎么验证的：新增 4 项公开 CSR API、262145 行、6/13 槽、int32/FP32 输出，覆盖重复列、signed int8 极值、末尾 tile、非默认流和输入/存储更新；首轮 4 项数值全通过。最终与既有回归共同执行，见 `regression_host_bound.log`。原十矩阵及三轮显式环境对照单独留档，不将 probe 最好值替换正式记录。
- 其他后端是否也需要：否；只改 `_musa_coo_half_spmv`，所有新增选择都位于 MUSA helper；其他后端 kernel/分派不变。

### 2026-11-11 tests/pytest/test_musa_sparse_routes.py::打包 CSR 与路由诊断

- 改了什么：新增上述 4 项大行数 int8 CSR/COO 共用路径回归；既有 FP16 SELL 成对路径断言增加失败时的 offsets/有效条目数信息，精度和路径选择断言均保留。
- 为什么改（附报错原文）：保证优化仍读实时输入并处理末尾 tile；明确旧 FP16 偶发 fallback 的元数据错误，不因数值正确就忽略性能路径退化。
- 怎么验证的：原始失败、复现实验、最后通过的完整集合分别保存在本轮不同日志中，不覆盖失败记录。线程数和 affinity 从未在测试内修改。
- 其他后端是否也需要：否；测试文件有 MUSA 实机条件，非 MUSA 隔离另外通过 CPU mock 和 AST 检查。

本轮冻结结果：`regression_host_bound.log` 为 **397 passed、4 skipped**；非默认流成对计划 120/120 次选择正确。`all9_default_1_*` 原等待策略 90/90 精度通过、5/9 >0.8、13 条 >1ms。相同冻结代码在 `all9_passive_1/2/3_*` 三个独立进程中 **270/270 精度通过、9/9 每轮 >0.8、0 条 >1ms**。各变体三轮最小均值：COO FP16 0.964749、FP16→FP32 0.823719、i8→FP32 0.992841、i8→int32 1.029266；CSR FP16 1.110560、i8→int32 1.266773；SELL FP16 0.966960、FP32 1.034718、i8→int32 1.002303（精确值以 `all9_comparison_summary.json` 为准）。全部原始记录保留，不挑最好一轮或丢弃慢样本。最低项只有约 3% 阈值余量，三轮通过不构成绝对稳定保证。

随后使用本轮 `launch_full.py` 的独立 `*_waitpassive` 目录后台 setsid 跑官方 runner 全 105 项，显式记录 OMP_WAIT_POLICY=PASSIVE、三个线程数变量未设置和不绑核；这是等待策略改变后的新配置，不能覆盖旧默认配置结果。代码、基线、容差及计时器冻结；当前原 C API CSR FP32 转置的 ASIC 精度问题未修复，完整结果会继续如实保留。实际目录、开始时间和完成状态见 `artifacts/latest_105_retest_directory.txt` 及新轮 RUN_INFO.json。

### 2026-11-11 CSR FP32 转置严格精度修复与三轮验证

用户要求复查 `spmv_csr_f32_int_trans` 能否严格通过。证据目录为
`results_musa_fix_20261111_csr_fp32_strict/`。开始时保存了原工作区文件，不覆盖前轮未提交修改。
原 ASIC 输入直接调用生产库连续 8 次，严格误差比均为 1.363373，失败列为 405878（0-based）。
前一整轮 `064012_*_waitpassive` 的通过属于文档 6.3.1 的 `pass_relaxed`，不等于严格修复。
复查还发现该整轮 SpMV 原始记录包含 COO complex64 conj 和 CSC FP32 trans 两条其他放宽项；
此前“104 项严格通过、1 项放宽”的口头计数不准确，不能从交付汇总的 Passed 推断全部严格通过。

#### capi/src/ops/spmv.cpp::MUSA CSR FP32 transpose

- 改了什么：增加只匹配 MUSA、CSR、FP32、trans/conj 的分支。复用原预处理拓扑的 native source row IDs，在原拓扑后追加 `n_cols * sizeof(double)` 工作区。共享已有 COO 的 Triton zero / FP64 scatter / finish 调用器，每次清零、实时读取矩阵和 RHS、FP64 乘法/原子累加，最后融合 alpha/beta 并一次转回 FP32；beta=0 不读旧输出。原 COO 包装传入相同参数，行为保持等价。调用方须按新的 bufferSize 分配工作区；不缓存计算结果。
- 为什么改（附报错原文）：原严格误差比 1.363373；新增大数抵消用例在旧代码报 `actual: 12413.926874212006 vs 1`。旧短列 FP32 scatter 和长列 FP32 gather 都可能丢失小项。全量 FP64 gather 十矩阵严格通过但仅 0.708952x，混合宽 scatter / gather 为 0.911157x，均作为未采用实验保留；统一 FP64 scatter 获得更好余量，不按矩阵名或参考输出选择路径。
- 怎么验证的：最终 ASIC 生产库 8 次重放相对量化 FP64 参考误差比全为 0；官方原十矩阵的未修改 benchmark 三个独立进程每轮 10/10 strict pass，ASIC strict ratio 0.00922928，整组最大 0.119437，均小于 1，放宽使用次数为 0。同轮本机 muSPARSE 平均加速比依次为 1.076152、1.0568982、1.0572383。原 C API warmup=10、iters=100、median 计时不变。
- 其他后端是否也需要：否；`scope_audit.json` 删除新增 MUSA 分支并还原等价 COO helper 后，非 MUSA C++ 与起始工作区逐字相同。全部已有 Triton kernel 与本轮开始时逐字相同，本次只扩展其 MUSA C API 使用范围；没有引入 MUSA 原生计算、PyTorch 计算或改动 baseline/容差。

#### capi/ctest/accuracy/test_spmv.cpp::MUSA CSR FP32 严格回归

- 改了什么：新增 CSR FP32 转置大数相消和空输入用例，扩展既有 MUSA 非默认流用例。覆盖 int32/int64、trans/conj、重复列、长列、空列/空行/空输出、alpha/beta、beta=0 覆盖 NaN、值/RHS 实时更新、索引显式重预处理、SetPointers 失效与双工作区切换；严格调用原 default_tolerance。读取非默认流结果前按既有回归模式 dev_sync。
- 为什么改（附报错原文）：新用例在旧生产实现严格失败，见 `regression_before.log`；初版新测试遗漏非默认流下载前同步而得到 inf，已补同步，保留该中间失败日志，不修改数值断言。
- 怎么验证的：最终 `regression_final.log/.xml` 为 41/41 passed。第一次正式 benchmark 没有预建输出目录，只执行了测试而没有输出 JSON，保留 `benchmark_v1.log`，不能将其作为交付记录；重跑到新目录保存所有数据。最终三轮目录为 `final_trial_1/2/3`，不选最好一轮、不删除矩阵或慢样本。
- 其他后端是否也需要：否；新增用例仅在 MUSA 实机运行，流用例仍由既有 MUSA 编译宏保护，其他后端原断言不变。

最终验证环境仍为 `OMP_WAIT_POLICY=PASSIVE`，OMP/MKL/OPENBLAS 三个线程数变量均不设置，不绑核。
`VALIDATION_INFO.json` 确认验证期间全部已跟踪源码及原十矩阵未变化。第一轮额外执行官方
SpMV benchmark 未过滤变体集合：570 条实际执行记录无失败（568 strict pass，另外 2 条其他变体
pass_relaxed），另有 40 条已声明但无测试路径的原记录；这不等于全部严格通过或完整 105 项复测。
本次只确认目标 CSR FP32 转置三轮严格通过，未重新执行完整 105 项，前次压缩包也仍对应修复前完整轮。

### 2026-11-11 另外两条 SpMV 放宽项的严格修复

用户要求先让 `spmv_coo_c32_int_conj` 和 `spmv_csc_f32_int_trans` 尝试严格通过，再重新打包代码。
本轮证据保存在 `results_musa_fix_20261111_remaining_strict/`；保留上一轮 CSR FP32 严格修复。
两项目标在原 ASIC 记录中 strict ratio 均为 1.36338。未改容差、原矩阵、baseline 优先级、计时器或 runner。

#### capi/src/ops/spmv.cpp::MUSA complex COO / FP32 CSC transpose

- 改了什么：MUSA COO 转置的宽累加分支扩展到 complex64，外部工作区为每个输出的两个 FP64 分量；实时读取坐标/值/RHS，先清零，再做宽复数 scatter，最后融合完整复数 alpha/beta。转置和共轭转置分别传入 CONJ，beta=0 不读旧输出。MUSA CSC FP32 转置直接复用现有 `spmv_csr_row_tile_real`，按 CSC 列作为输出行、ACC_IS_FP64=True，乘法和全程累加均为 FP64；不改变格式、不新增工作区或结果缓存。
- 为什么改（附报错原文）：两个原 ASIC case 均 strict ratio 1.36338，分别来自 FP32 原子累加和 FP32 列内局部和的大项抵消。新增 CSC 抵消回归在旧代码报 `actual: 12413.926874212006 vs 1`；初次 COO 回归还暴露旧 preprocess 对乱序行坐标返回 INVALID_VALUE，原失败日志保留，不能将这条前置失败说成数值复现。
- 怎么验证的：原十矩阵三个独立进程两项目标各 30/30 strict pass，CSR FP32 相邻目标也为 30/30 strict pass；本轮正式结果均未使用 relaxed。COO complex64 三轮均值 0.9124716、0.9077780、0.9099741，CSC FP32 为 0.9034054、0.9068323、0.9052082；CSR FP32 为 1.0601923、1.0681209、1.0647790。均为同轮本机 muSPARSE、原 warmup=10/iters=100/median。两个 ASIC 原始输入直接调用生产库各 8 次，相对量化 FP64 参考 strict ratio 均为 0；官方 benchmark 相对原参考的 ASIC ratio 均为 0.00922928，整组最大 0.119437，门槛均为 1。
- 其他后端是否也需要：否；新分支均要求 backend=musa 且严格匹配格式、dtype、方向。`scope_audit.json` 还原新增 MUSA 分支后，C++ 与本轮起始工作区逐字相同；前轮 CSR 严格修复和已有实数 COO 路径保持等价。调用方对 complex COO 须按新 bufferSize 分配工作区，现有官方 harness 已遵守。

#### capi/flagsparse_codegen/sparse_gather.py::complex FP64 scatter/finish

- 改了什么：新增 `musa_coo_transpose_scatter_c32_f64` 和 `musa_coo_transpose_finish_c32_f64` 两个 Triton kernel，复数乘法的四个乘积及实虚累加都使用 FP64，CONJ 作用于矩阵虚部；alpha/beta 在宽精度完成后各分量只转一次 FP32。复用原 FP64 zero kernel，全部已有 kernel 定义保持不变。
- 为什么改（附报错原文）：原复杂 COO scatter 将乘积和输出都保留在 FP32，ASIC 的 strict ratio 为 1.36338；只把最终转换改宽无法恢复此前被吞掉的小项。
- 怎么验证的：正式十矩阵三轮、生产库重放、非零虚部、完整复数 alpha/beta、重复/乱序坐标和非默认流全部通过；AST 对比确认只新增这两个函数，没有改共享 kernel 或 Python 算子。无手写 MUSA kernel、mcc 计算库或 PyTorch 原生计算路径。
- 其他后端是否也需要：否；新增符号只由上述 MUSA C API 分支引用；CSC 复用的 CSR row-tile kernel 源码本轮未修改。

#### capi/ctest/accuracy/test_spmv.cpp::两条严格路径回归

- 改了什么：新增 COO complex64 和 CSC FP32 抵消回归、两种格式空输入回归及非默认流回归。覆盖 int32/int64、trans/conj、真实非零虚部、复数 alpha/beta、长列、多 NNZ 块、重复/乱序坐标、部分 tile、空列/空行/空输出、beta=0 时 NaN 旧输出，以及值/RHS/坐标实时修改。所有精度断言使用原 default_tolerance，未加入放宽或替换预期值逻辑。
- 为什么改（附报错原文）：保存原两个 case 的严格失败证据及上述新增回归失败，避免仅因正式 benchmark 的虚部为零而漏掉复数计算语义；同时确保精度修复没有破坏异步流和实时输入语义。
- 怎么验证的：`regression_final.log/.xml` 为 45/45 passed。第一正式轮运行未过滤的官方 SpMV benchmark：610 条记录中 570 条实际执行全部 strict pass、0 failed、0 pass_relaxed；另 40 条为原有已声明但无测试路径记录，不算执行通过。三轮目标共 90/90 strict pass，完整原始 JSON 均保留。
- 其他后端是否也需要：否；新用例仅 MUSA 执行，流用例使用已有编译宏保护，其他后端断言和原有所有容差保持不变。

环境继续为 `OMP_WAIT_POLICY=PASSIVE`，三个线程数变量不设置，不绑核。验证期间全部已跟踪源码及
原十矩阵未变化，见 `VALIDATION_INFO.json`；文档在验证完成后追加。本轮是 SpMV 完整集合与三项目标
的定向复测，没有重新跑完整 105 项。重新打包的代码与新严格验证匹配，不能把旧 105 项整轮结果改写为修复后验收。
