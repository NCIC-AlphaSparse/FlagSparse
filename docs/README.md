# 后端文档索引

三处文档，各回答一个问题：`docs/<BACKEND>.md` 怎么在那台机器上跑、
`capi/docs/<BACKEND>.md` C API 那一层、[`modified/<BACKEND>.md`](../modified/README.md)
这轮在实机上改了哪些文件（改动台账，省得后端文件来回传）。

**一个后端一个文件。** Python 侧在 `docs/`，C API 侧在 `capi/docs/`，同名对应：
`docs/MUSA.md` 讲怎么在摩尔线程上跑 Python/Triton 这套，`capi/docs/MUSA.md` 讲 C API
那一层。跨引用都按这个名字走。

| 后端名（`FLAGSPARSE_BACKEND`） | 厂商 / 平台 | Python 侧 | C API 侧 |
|---|---|---|---|
| `cuda` | NVIDIA | 见仓库根 `README.md` | [`capi/docs/CUDA.md`](../capi/docs/CUDA.md) |
| `rocm` | 海光 DCU / ROCm | [`DCU.md`](DCU.md) | [`capi/docs/DCU.md`](../capi/docs/DCU.md) |
| `metax` | 沐曦 MetaX / MACA | [`MACA.md`](MACA.md) | [`capi/docs/MACA.md`](../capi/docs/MACA.md) |
| `mthreads` | 摩尔线程 / MUSA | [`MUSA.md`](MUSA.md) | [`capi/docs/MUSA.md`](../capi/docs/MUSA.md) |
| `ascend` | 昇腾 / CANN | [`ASCEND.md`](ASCEND.md) | [`capi/docs/ASCEND.md`](../capi/docs/ASCEND.md) |
| `xpu` | 昆仑芯 | [`XPU.md`](XPU.md) | [`capi/docs/XPU.md`](../capi/docs/XPU.md) |
| `gcu` | 燧原 | — 尚无 | — 尚无 |
| `mlu` | 寒武纪 | — 尚无 | — 尚无 |

最后两行是**已注册但没有任何文档**的槽位：`_BACKEND_NAMES` 和 `_dispatch.py` 的
`_IMPLEMENTATION_BACKENDS` 都是 8 条，`backends/gcu`、`backends/mlu` 目录已建好但代码留空，
走共享实现。没有覆盖不等于不能跑，只是没人在真机上验过，也就没有可写的文档。

**`XPU.md` 的实测最少**：DCU / MACA / MUSA / ASCEND 都有完整的实机验证结果；XPU 在 P800 上有
2026-09-17/18 的有限实测和一轮 40 变体精度，性能还没有可用的交付数据，文件顶部标了出来。

`cuda` 没有单独文件是因为它是参照路径 —— 环境、跑法、算子清单都在仓库根的
`README.md` / `README_cn.md` 里，后端文档只记"与 CUDA 不同的地方"。

> **从 `flpagsparse_cwrapper` 过来的？** 那个仓库的全部内容已于 2026-09-18 合并进本仓库，
> 之后的工作都在这里做。背景、切换步骤和当初的合并方式见 [`REPO_MIGRATION.md`](REPO_MIGRATION.md)。

## 交付复现入口（40 个变体 × 30 个矩阵）

通用命令、参数说明和各后端差异的总表在仓库根 [`README_cn.md`](../README_cn.md) 的"复现交付测试"一节
（英文版 [`README.md`](../README.md) "Reproducing the delivery run"）。各后端的确切命令、环境自检和预期会看到的
非 Passed 项：

| 后端 | 章节 | 与通用命令的主要差别 |
|---|---|---|
| CUDA | 仓库根 README 的那一节 | 无 |
| DCU | [`DCU.md`](DCU.md) 0.5 节 | SpSV/SpSM 可能死锁（记为 `Timeout`），外层超时给足 |
| MACA | [`MACA.md`](MACA.md) 0.5 节 | `--no-cusparse`，`--timeout 4500`（SDDMM 的 K sweep） |
| MUSA | [`MUSA.md`](MUSA.md) 0.5 节 | 改用 `run_flagsparse_split_delivery.py`，性能取自 C API（muSPARSE） |
| Ascend | [`ASCEND.md`](ASCEND.md) "交付复现" | 只用 `--gpus 6,7`；5 个算子对 PyTorch-NPU，其余只做能力探测 |
| XPU | [`XPU.md`](XPU.md) 1.5 节 | `FLAGTREE_BACKEND=xpu` 等环境变量；5 个算子对 PyTorch-XPU，其余只做能力探测 |

跑完统一用 `python3 tools/delivery_table.py <结果目录>` 看结果。

## 每个文件里有什么

| 文件 | 内容 |
|---|---|
| [`DCU.md`](DCU.md) | ROCm 环境、诊断优先的排查顺序、hipSPARSE 基线覆盖范围、SpMV 走 rowpar 的原因、已知限制 |
| [`MACA.md`](MACA.md) | C550 bring-up（FlagTree metax 后端、环境指纹）、921 用例的验证范围、SpSV 的两个缺陷、SpMM COO 复数的 4 KB 私有内存上限、调优 A/B |
| [`MUSA.md`](MUSA.md) | 独立设备类型 `musa` 与兼容后端的区别、`_ACCEL` 抽象、实测能力矩阵（muDNN 的 gemv 缺口）、normal 回归结果、已解决问题的复现记录 |
| [`ASCEND.md`](ASCEND.md) | 910B 环境检查、Ascend fallback 分发表、算子能力探测、已知限制 |
| [`XPU.md`](XPU.md) | 昆仑芯：插件探测为什么不能只看 `torch.xpu`、`torch_xmlir` 的 CUDA-shim 路径、5 个算子对 PyTorch-XPU 计时其余能力探测、为什么没有厂商基线、首次上机顺序 |
