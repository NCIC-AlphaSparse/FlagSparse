# Q4 性能优化目标清单（给 codex，2026-10-06）

## 现状：下面绝大多数条目已经在 CUDA 上做完了

最新实现、45 项加速比、基线口径及复现统一见
[CUDA 调试与复测手册](debug/CUDA.md)（取代了已删除的 `docs/Q4_CAPI_HANDOFF.md`）。
**45 个变体里 44 个现在相对 CuPy 的平均加速比 ≥0.8x；只有 1 个还没达标：
`spgemm_csr_f32_int_non_non`（0.789x）**——这是目前唯一打开的性能待办，细节见本文档
最后"当前唯一待办：SpGEMM"一节。下面每一节我都标了 ✅ 已做完 / ⏳ 还要做，对照
`CUDA.md` 的新数字核对过；原始的分析思路保留，供排查同类问题时参考。

**跨后端复测仍然是开着的**：这些优化改的是 CUDA 和共享的 Python/Triton 源码，
`CUDA.md` 自己写得很清楚——"DCU/MACA/MUSA 等后端需在对应设备复测，不能用 CUDA 通过
结果宣称跨后端达标"。下面表里的 DCU/MACA 数字都是 2026-10-05 旧代码的真机结果，**还
没有用新代码重新跑过**，不代表这次优化对它们也生效了。

---

写这份文档时，q4 的 45 个变体在 CUDA/DCU/MACA/MUSA 四个后端上**精度全部通过**（capi 侧
dispatch 覆盖也是 45/45）。精度不是问题，这份文档只谈
**性能**——哪些变体明显比该比的对象慢，以及已知的几类结构性原因。

**不是这份文档要做的事**：不涉及精度 bug（MUSA 上发现的几个真实精度失败、MACA 的
spgemm runner bug 另见下文"容易踩的坑"一节，但那是"数据不可信"不是"变多慢"）；不涉及
新算子组/新 dtype 覆盖（那部分内容已并入 `docs/debug/CUDA.md`）。

## 数据来源，按可信度排

0. **`docs/debug/CUDA.md`**（2026-10-06，这次优化之后的最新 CUDA 数字，Python/Triton
   对 CuPy/PyTorch）——**最新的 CUDA 状态以这份为准**，本文档里标"✅ CUDA 新数值"的都
   引自这里。
1. **`docs/debug/DCU.md` 第 5 节**（2026-10-05，BW1000 64G 真机，45 个变体逐行聚合的
   `triton_speedup_vs_pytorch`，不是父脚本的粗粒度平均）——本文档"旧数值"大部分引自
   这里，是这次 CUDA 优化**之前**的代码跑出来的，还没用新代码复测。
2. **`docs/debug/MACA.md` 第 3.2 节**（2026-10-05，C550 真机，同样逐变体聚合，同样是
   优化前的旧代码）。
3. **旧的 `capi/capi_results/summary_q4.json`**（CUDA，C API 层，对 cuSPARSE 的
   `avg_speedup_vs_vendor`）——**这是另一条独立的 dispatch 路径**（C++ `capi/src/ops/
   *.cpp`），不是 Python/Triton 路径，两者对同一个"变体"可能性能表现完全不同。这次
   CUDA 优化之后，Python/Triton 路径的新数字已经在 `CUDA.md` 里，但 capi 这条路径的
   `summary_q4.json` 没有重新生成，仍是旧数字，别拿旧 capi 数字当成"优化后还是这样"。
4. **`docs/debug/MUSA.md`**——有加速比的变体不多（muSPARSE 基线覆盖有限），且小部分变体
   在大矩阵上精度失败，这些变体的加速比先别当真，等精度修完再看性能。

跑 `python3 capi/tools/write_summary_q4.py` 或读 `docs/debug/{CUDA,DCU,MACA,MUSA}.md` 能
拿到比这份文档更新的数字——这里的表是多次快照拼出来的，不会随代码变化自动更新。

## 已知的两个"结构性慢"模式（优先级最高，跨后端、跨维度地慢）

### 1. CSR SpMV 的 opA=TRANSPOSE / opA=CONJUGATE_TRANSPOSE 走 atomic-scatter 路由

✅ **CUDA 上已做完**（`docs/debug/CUDA.md` 第 4 节："prepare 建立转置拓扑，分组 gather
替代逐次转置重建/atomic scatter"，对应新文件 `src/flagsparse/sparse_operations/
_spmv_csr_transpose.py`）。⏳ **DCU/MACA 还没用新代码复测**——下表左两列是旧代码在真机
上的数字，右两列是这次优化后 CUDA 的新数字，差距看完就知道这次改动有多关键：

`spmv_csr_f32_int_trans`、`spmv_csr_c32_int_conj`（以及对应的 `spmv_coo_f32_int_trans`、
`spmv_coo_c32_int_conj`）之前在 **Python/Triton 路径**（`src/flagsparse/sparse_operations/
spmv_csr.py`/`spmv_coo.py` 的 transpose/conj 分支）上，在所有测过的后端都明显 <1x：

| 变体 | DCU 旧 (Python) | MACA 旧 (Python) | CUDA 旧 (capi C++) | CUDA 新 (Python，CuPy) |
|---|---:|---:|---:|---:|
| `spmv_csr_f32_int_trans` | 0.034x | 0.089x | 0.667x | **1.164x** |
| `spmv_csr_c32_int_conj` | 0.039x | 0.075x | 0.813x | **0.956x** |
| `spmv_coo_f32_int_trans` | - | 0.691x | 0.629x | **1.247x** |
| `spmv_coo_c32_int_conj` | - | 0.574x | 0.759x | **0.991x** |

`spmm_csr_f32_int_trans_non_row` 同理：旧 capi/CUDA 0.298x → 新 CuPy **4.395x**（PyTorch
1.904x）。C API（capi）侧也单独做了对应优化（见 `CUDA.md` 的"C API 独立记录"一节：
preprocess 阶段保存转置指针/nnz 排列，CUDA gather 融合 alpha/beta），但那是独立的一套
验证，不计入上面 Python/Triton 的 45 项。

**当时的关键线索（已验证对，留作记录）**：同一个逻辑操作，capi（C++ 手写的
atomic-scatter dispatch）在 CUDA 上只慢到 0.6-0.8x，但 Python/Triton 路径在 DCU/MACA 上
慢到 0.03-0.09x——差了一个数量级，说明问题不是"transpose 本身只能这么慢"，是 Python 侧
的 Triton kernel 调度/访存模式有改进空间。这次的修法印证了这个判断：换成 prepare 阶段建
转置拓扑 + 分组 gather，CUDA 上直接从 0.03-0.3x 跳到 1x 以上。

⏳ **DCU/MACA 的待办**：这个 fix 改的是共享 Python 源码（`_spmv_csr_transpose.py` 等），
理论上在 DCU/MACA 上也会有同等量级的提升，但**还没有人在那两台机器上用新代码重新跑一遍
`spmv_csr_f32_int_trans`/`spmv_csr_c32_int_conj`/`spmm_csr_f32_int_trans_non_row` 来确认**。
这是现在最值得先去验证的一项，因为理论上收益最大、代码已经写好了，只差上机跑一遍。

### 2. DCU 上 SELL SpMV 普遍偏慢，但 MACA 上不是——很可能是 DCU 特定的 tuning 问题

✅ **kernel 本身在 CUDA 上已经清理过**（`CUDA.md` 第 4 节："保持调用者的 slice_size，
合并实数 slice 调度，优化复数 warp/slot 归约，移除多余 reshape；不能修改 slice_size 来
重新解释输入"——新数字：CUDA 上 f32/f16/c32/i8i32 四个全部 ≥1.1x，c32 从明显 <1x 变成
1.148x）。**但这条修的是通用调度效率，明确没有碰 slice_size 本身**，所以下面 DCU 专属
的疑点依然是 ⏳ 开着的：

`spmv_sell_*` 四个变体之前在 **DCU** 上全部 <0.4x：

| 变体 | DCU 旧 | MACA 旧 | CUDA 新 (CuPy) |
|---|---:|---:|---:|
| `spmv_sell_f32_int_non` | 0.354x | 1.270x | **1.391x** |
| `spmv_sell_f16_int_non` | 0.363x | 2.104x | **1.377x** |
| `spmv_sell_c32_int_non` | 0.212x | 0.592x | **1.148x** |
| `spmv_sell_i8i32_int_non` | 0.376x | 1.316x | **1.385x** |

**这不是"SELL kernel 本身写得差"**——MACA 上 f32/f16/i8i32 三个本来就 >1x，只有 c32 在
DCU/MACA 两边都 <1x；这次 CUDA 上的调度优化让 c32 也追上来了（1.148x），**复数 SELL 的
调度问题看来确实是真的 kernel 问题，这次顺手修了**。DCU 偏慢更可能是 `docs/debug/DCU.md`
第 1 节记的那条已知特点："gfx936 上 SpMV CSR 走 ROCm 专用路由（row_tile，fp32 本地累
加）"——SELL 的 `slice_size`（默认 32）和 DCU 的 warp size（64）不匹配，可能每个 warp
有一半线程闲着。**这次的 CUDA 修复明确没有改 slice_size（"不能修改 slice_size 来重新
解释输入"），所以 ⏳ DCU 这个 slice_size/warp_size 不匹配的疑点仍然没人验证过**——需要
DCU 机器才能动手调，建议先用新代码在 DCU 上重新跑一遍看这四个变体是否还是 <0.4x，再决
定要不要单独做 DCU 专属的 slice_size 调整。

## 其他 <1x 的点（优先级较低，多数有具体原因，不是未知缺陷）

| 变体 | DCU/MACA 旧数值 | CUDA 新数值 (CuPy) | 状态 / 备注 |
|---|---|---:|---|
| `axpby_f16_int` | DCU 0.460x | **3.506x** | ✅ CUDA 已做完；⏳ DCU/MACA 待复测。单个标量向量操作，之前可能是 kernel launch overhead 主导 |
| `spvv_c32_int_conj` | DCU 0.360x，MACA 0.808x | **1.733x** | ✅ CUDA 已做完（c32 SpVV 实部/虚部归约融合成一个 kernel，见上面第 1 节）；⏳ DCU/MACA 待复测 |
| `spvv_f16f32_int_non` | DCU 0.582x | **2.180x** | ✅ CUDA 已做完；⏳ DCU/MACA 待复测 |
| `spmm_coo_i8i32_int_non_non_row` | DCU 0.350x | **0.889x** | ✅ CUDA 已做完（`CUDA.md`："COO int8→int32 SpMM 在 CUDA 每个 program 批量处理 16 个非零元素...CuPy 平均由 0.329× 升至 0.889×；**其他后端保留原实现**"——这句话是明确写的，DCU/MACA 没有同步改，⏳ 待移植+复测） |
| `spgemm_csr_f32_int_non_non` | DCU 0.245x（hipSPARSE 基线 0.48x） | 0.789x（capi/旧 CUDA 曾是 1.797x，口径不同不能直接比） | ⏳ **唯一还没做完的**，见下一节 |

## 容易踩的坑（写文档/看数据时的检查清单）

- **"没有厂商基线"≠"性能差"。** 很多变体（`spmv_coo`/`spmv_csc`/`spmm_csc` 全系列、
  `sddmm_csr_f16`、所有 i8/f16 混合精度变体）在 PyTorch 或 cuSPARSE/muSPARSE/hipSPARSE 里
  根本没有对应实现（`torch.sparse` 覆盖不到，厂商库不支持这个 dtype/format 组合），所以
  报表上是"-"或"N/A"，**这是"测不出"，不是"跑得差"**，不要误当成优化目标。
- **MACA 的 `spgemm_csr_f32_int_non_non` 性能行实际没跑通**：10 行原始输出全是 `ERROR`
  （`test_spgemm.py` 不认它给 worker 传的 `--_matrix-worker` 参数），但父进程退出码是 0，
  `run_flagsparse_pytest.py` 的 runner 误把这行汇总成了 `Passed`。`docs/debug/MACA.md` 第
  3.2 节明确写了"该加速比必须视为未测"。✅ **这个 runner bug 本身已经在 CUDA 这边修了**
  （`CUDA.md` 第 4 节："SpGEMM runner 修复 matrix-worker 分派和无效参数，能让子进程完成
  计算"——`test_spgemm.py` 是跨后端共享脚本，这个修复大概率对 MACA 同样生效）。⏳ 但
  **MACA 还没有用修完的 runner 重新跑一遍，拿到真实的 spgemm 性能数字**——这是确认
  `spgemm_csr_f32_int_non_non` 到底要不要继续优化之前，MACA 侧必须先做的一步。
- **MUSA 的几个变体当前精度就没过**（`spmv_csr_f32c32_int_non`、`spmv_coo_c32_int_non`、
  `spmv_csc_f16_int_non`、`spmv_csr_c32_int_conj`、`spmm_csc_f16_int_non_non_row`，在大矩阵
  规模上，见 `docs/debug/MUSA.md` 的失败行明细表）——这些变体的 MUSA 加速比**先别用来做
  性能判断**，精度都没稳定通过，速度数字没有意义。
- **DCU/MACA 的 Python 路径和 CUDA 的 capi (C++) 路径是两套独立实现**，同一个变体名在两
  边的性能可能差一个数量级（见上面 transpose 那组对比）。改哪个要先确认清楚，别改了
  Python 侧的 kernel 却以为也修了 capi 的 dispatch（反之亦然）。
- 这份文档里引用的"加速比"定义不统一：DCU/MACA 的 Python 数字是对 **PyTorch**；CUDA 的
  capi 数字是对 **cuSPARSE**；两者不能直接比较谁的分子更高，只能看同一个来源内部的相对
  高低。

## 当前唯一待办：SpGEMM（CUDA 上也没达标）

`spgemm_csr_f32_int_non_non` 是 45 个变体里唯一一个在 CUDA 上优化完一轮之后**仍然
<0.8x**（CuPy 0.789x，PyTorch 0.799x）。`CUDA.md` 明确写了"性能仍未达标，不把 runner
修复当成性能优化完成"——也就是说 runner 的 `--_matrix-worker` bug 修好之后，拿到的是真
实数据，这个真实数据就是 0.789x，不是测不出来，是真的还没够快。复现命令见
`docs/debug/CUDA.md` 第 6 节的 SpGEMM 复测命令（10 个 MatrixMarket 文件，float32/int32，
`--input-mode auto`）。这是接下来唯一需要"真的去优化算法/kernel"的一项。

## 跨后端复测清单（不是新代码，是确认这次 CUDA 修复在其他后端是否同样生效）

按预期收益从高到低排：

1. **transpose/conj SpMV**（第 1 节）：DCU 上旧数字是 0.03-0.09x，CUDA 修完后跳到
   1.0-1.2x，差了一个数量级——**预期收益最大**，优先在 DCU/MACA 上用新代码复测
   `spmv_csr_f32_int_trans`、`spmv_csr_c32_int_conj`、`spmv_coo_f32_int_trans`、
   `spmv_coo_c32_int_conj`、`spmm_csr_f32_int_trans_non_row`。
2. **MACA 的 spgemm runner bug**：确认是不是真的被 CUDA 那次 runner 修复一起解决了，拿到
   MACA 上 `spgemm_csr_f32_int_non_non` 的真实数字（之前一直是假 `Passed`）。
3. **复数 SELL SpMV**：DCU/MACA 上 c32 变体之前明显 <1x，CUDA 上调度优化后变成 1.148x，
   复测 `spmv_sell_c32_int_non`。
4. **COO int8→int32 SpMM**：`CUDA.md` 明确写了"其他后端保留原实现"，这个优化**没有自动
   带到 DCU/MACA**，需要先确认要不要移植这套"每个 program 批量处理 16 个非零元素"的改
   法过去，而不只是"重新跑一遍等它自动变快"。
5. **DCU 专属**：`spmv_sell` 的 `slice_size`（默认 32）和 DCU warp size（64）是否匹配——
   这次 CUDA 修复明确没有碰 slice_size，需要 DCU 机器单独验证、单独调。

每做完一项，用对应后端文档里已经给出的命令重新跑一遍该变体的 benchmark，更新
`docs/debug/{DCU,MACA,MUSA}.md` 里的数字，不要凭感觉估计有没有提升。
