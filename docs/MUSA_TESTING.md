# 摩尔线程 MUSA 测试指南

本文说明 FlagSparse 在摩尔线程（Moore Threads / MUSA）环境中的运行方式。MUSA 与
CUDA、ROCm、MetaX、Ascend 分开分发，本文的设置不会改变其他后端的行为。

> **验证范围说明。** 本文的分发逻辑、环境变量和基线选择，已在一台 **CUDA 机器**上用
> `FLAGSPARSE_BACKEND=mthreads` 覆盖验证过：8 个算子的精度套件全部通过
> （spmv_csr 77、spmv_coo 77、spmm_csr 59、spmm_coo 51、sddmm_csr 9、gather 26、
> scatter 50、spsv_csr 341），端到端基准也确认了基线自动切到 PyTorch、厂商列为 `N/A`、
> 加速比按 `FS/PT` 计算。
>
> **但这台机器没有安装 `torch_musa`**，所以 `_accel_device_type()` 实际是 `cuda` 而非
> `musa`——验证到的是 **mthreads 的分发决策 + 回退路径**，不是真正的 MUSA 设备路径。
> 「MUSA 设备上的张量能否被接受」「内核在摩尔线程硬件上能否编译运行」「数值结果是否
> 正确」这三项**未经验证**。下文用 **【待验证】** 标出需要真机确认的部分。

## 1. MUSA 与 CUDA 兼容后端的区别

这一点决定了后面所有设置，值得先说清楚。

CUDA、ROCm（DCU）、MetaX（MACA）三者在 PyTorch 里**都表现为 `torch.cuda`**：
`torch.cuda.is_available()` 为真、张量的 `.is_cuda` 为真、`torch.device("cuda")` 有效。
所以它们共用同一套调用路径。

**MUSA 不是这样。** 它通过一个 out-of-tree 扩展（`torch_musa`）提供**独立的设备类型**
`musa`，因此：

- `torch.cuda.*` 不适用，要用 `torch.musa.*`
- `Tensor.is_cuda` 为假，判断设备要看 `.device.type == "musa"`
- `torch.device("cuda")` 不是有效目标，要用 `torch.device("musa")`

仓库用 `_ACCEL` 这一层抽象掉了这个差异（`src/flagsparse/sparse_operations/_common.py`
的 `_resolve_accel`）：它把 torch 子模块和设备类型**成对**返回，`torch.cuda`/`"cuda"`、
`torch.musa`/`"musa"`、`torch.npu`/`"npu"`。成对返回是必要的——只换模块不换设备类型，
会让 `_is_accel_tensor()` 拒绝掉每一个张量。

Ascend（`torch.npu`）与 MUSA 属于同一类。

## 2. 环境准备

```bash
export PYTHONPATH=$PWD/src
export FLAGSPARSE_BACKEND=mthreads
```

`FLAGSPARSE_BACKEND` 的合法值是 `cuda`、`rocm`、`metax`、`mthreads`、`ascend`
（`_BACKEND_NAMES`），写错会直接抛 `ValueError` 而不是静默回退。

注意后端名是 **`mthreads`**，不是 `musa`——`musa` 是设备类型字符串，两者不要混。

不设这个变量也可以：`_detect_mthreads_runtime()` 会自动探测 `torch.musa` 是否存在且
`torch.musa.is_available()`。显式设置的好处是意图明确，且探测失败时能立刻发现。

### 自检

```bash
python3 - <<'PY'
import importlib.metadata as md
import torch
print("torch:", md.version("torch"))
try:
    print("torch_musa:", md.version("torch_musa"))
except Exception as exc:
    print("torch_musa: 未安装 -", exc)
print("torch.musa 存在:", hasattr(torch, "musa"))
if hasattr(torch, "musa"):
    print("torch.musa.is_available():", torch.musa.is_available())
    print("设备数:", torch.musa.device_count())
PY
```

再确认 FlagSparse 侧的分发：

```bash
python3 - <<'PY'
import flagsparse.sparse_operations._common as C
print("backend            :", C._backend_name())
print("is_mthreads_runtime:", C._is_mthreads_runtime())
print("accel device type  :", C._accel_device_type())
print("vendor sparse lib  :", C._vendor_sparse_library())
print("fallback reason    :", C._accel_fallback_reason())
PY
```

**期望输出**：`backend = mthreads`、`accel device type = musa`、
`vendor sparse lib = torch`、`fallback reason = None`。

已在摩尔线程真机确认（torch 2.7.1 / torch_musa 2.7.1，1 张卡）：

```
backend            : mthreads
is_mthreads_runtime: True
accel device type  : musa
vendor sparse lib  : torch
fallback reason    : None
```

如果 `fallback reason` 出现下面这句，说明 `torch_musa` 没装好：

```
backend 'mthreads' selected but torch.musa is unavailable (torch_musa not installed?);
falling back to torch.cuda
```

此时 `_accel_device_type()` 会是 `cuda` 而不是 `musa`——**整套东西会退回在 CUDA 语义下
运行**，跑出来的不是 MUSA 的数。这个回退是故意做成成对的（见第 1 节），不会产生
"模块是 torch.cuda 但设备类型声称 musa" 这种自相矛盾的状态，但结果也就不是 MUSA 的了。
务必在正式跑之前确认这一行是 `None`。

## 3. 性能基线

MUSA 上**默认没有厂商基线**（`_mthreads_vendor_sparse_library()` 返回 `None`），
厂商列报 `N/A` 并给出原因，FlagSparse 自己的耗时照常采集。

这一点在 2026-09-11 改过。此前默认是 `torch`，依据是"`torch.sparse` 在 MUSA 上能跑
且能给出真实参照"——**这句话在 MTT S5000 上实测为假**（torch 2.7.1 / torch_musa 2.7.1）：

```
CSR: NotImplementedError: Could not run 'aten::empty.memory_format'
     with arguments from the 'SparseCsrmusa' backend
COO: NotImplementedError: Could not run 'aten::addmm'
     with arguments from the 'Sparsemusa' backend
```

**四个 dtype 全挂，float32 也挂**，与 dtype 无关。注意 `torch.sparse_csr_tensor()`
**构造是成功的**，只有乘法缺失——这正是这个缺口能通过 review 的原因，光看"能不能建出
稀疏张量"是看不出来的。

可以用 `FLAGSPARSE_MTHREADS_VENDOR` 覆盖：

| 取值 | 含义 |
|---|---|
| `none` | 默认。不使用厂商基线，相关列为 `N/A` |
| `torch` | 用 `torch.sparse` 作基线——**当前真机上会报错**，保留是为了将来 torch_musa 补齐算子后可以用实测把默认翻回去，而不是靠假设 |
| `musparse` | 声明使用 muSPARSE **【待验证】**——仓库里没有对应的 binding 实现，设置后能否真正生效需要在真机确认 |

其他取值会抛 `ValueError`。

CuPy/cuSPARSE 在 MUSA 上**不适用**，相关列会给出明确原因而不是伪造数字：

```
CuPy/cuSPARSE is not applicable on the mthreads backend (baseline: torch)
```

## 4. 运行测试

runner 本身没有 MUSA 特判，用通用命令即可：

```bash
export PYTHONPATH=$PWD/src
export FLAGSPARSE_BACKEND=mthreads

# 精度
python run_flagsparse_accuracy.py --mode quick --gpus 0

# 性能
python run_flagsparse_performance.py --ops spmv_csr,spmm_csr \
  --benchmark-input matrix --benchmark-warmup 5 --benchmark-iters 20

# 两阶段一起
python run_flagsparse_pytest.py --phase both --mode quick --gpus 0 \
  --benchmark-input matrix --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir pytest_results_musa
```

单个算子脚本也可以直接跑：

```bash
python tests/test_spmv.py <目录或文件.mtx> --warmup 5 --iters 20
python tests/test_spmm.py <目录/> --csv out.csv
```

**算子清单和超时值需要在真机上按实际情况调整【待验证】。** 参考 MetaX 的经验：
C550 上 SpSV/SpSM 的内核会卡住、SpMM BELL 单矩阵耗时过长，都从默认清单里排除了。
MUSA 上哪些算子需要同样处理，只能实测后确定。

### pytest 精度套件（tests/pytest）

这套用例原本对 CUDA 是硬编码的，在 MUSA 上会连续踩两个坑，**两个都已经修好**，
但都是"在 CUDA 机器上看不出来"的那一类，所以记在这里：

1. **全部 skip。** 每个文件顶上是
   `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), ...)`。
   MUSA 上 `torch.cuda.is_available()` 为假，于是 2056 个用例全部静默跳过，
   输出只有 `341 skipped`，看起来像"跑过了"。
   现在换成 `accuracy_utils.accelerator_available()`，它探 `_ACCEL.is_available()`。

2. **全部 fail。** 守卫放行之后，用例体里的 `torch.device("cuda")` 会抛
   `NotImplementedError: Could not run 'aten::empty.memory_format' with arguments
   from the 'CUDA' backend`。148 处设备字面量现在统一走
   `accuracy_utils.accelerator_device()` / `ACCELERATOR_DEVICE_TYPE`，取自
   `_common._ACCEL_DEVICE_TYPE`。

3. **参考计算本身跑在加速器上。** 前两个坑填完之后，`test_spmv_csr_accuracy.py`
   仍然 74 failed / 3 passed，而且**一个都没进到 FlagSparse 内核**——全死在测试自己的
   数据生成和 dense 参考上：`_random_csr_mn` 的 `torch.where`（fp64/复数无 muDNN 内核）、
   以及 `ref_mat.to(ref_dtype) @ x.to(ref_dtype)`（`_reference_dtype` 把 fp32 升到 fp64，
   撞上上面那个 gemv 缺口）。
   现在数据生成和 golden reference 统一走 `accuracy_utils.golden_device()`（恒为 CPU），
   只有真正交给算子的张量才 `.to(accelerator_device())`。这本来就是 `accuracy_utils`
   模块 docstring 写的 policy（"compare against a CPU float64 golden reference"），
   只是这些文件没照做。

三处在 CUDA/ROCm/MACA 上都是恒等变换（`_ACCEL is torch.cuda`、
`_ACCEL_DEVICE_TYPE == "cuda"`；CPU 上的 fp64 oracle 只会比设备端更准），
所以对其他后端零影响。

```bash
python -m pytest tests/pytest -q      # MUSA 上应当真的跑起来，不再是 skipped
```

> 已知 flaky（与 MUSA 无关，CUDA 上同样复现）：`test_spmv_csc_matches_dense_reference`
> 的 `float32-160-1024` 参数、`test_spsv_csr_upper_optimized_route_analysis_workspace_matches_direct[csr_cw_levelschd]`、
> `test_spsv_sell_non_unit_rejects_malformed_structure[duplicate_diagonal]`。
> 随机输入未固定种子，单跑会过。

## 4.5 实测能力矩阵（MTT S5000，2026-09-11）

用 `tools/probe_accel_capabilities.py` 在真机上逐层测出来的。这张表的用处是**在读任何
失败之前先知道它属于哪一层**——之前 `tests/pytest/test_spmv_csr_accuracy.py` 的 74 个
失败全部发生在测试自己的参考计算里，一个都没进到 FlagSparse 内核。

```bash
export PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads
python tools/probe_accel_capabilities.py --json musa_caps.json
```

环境：torch 2.7.1 / torch_musa 2.7.1 / Triton 3.6.0 / muDNN v3105 / MTT S5000 ×1。

| | fp32 | fp64 | c64 | c128 |
|---|---|---|---|---|
| 分配、H2D/D2H、`torch.complex` 构造 | ok | ok | ok | ok |
| `add` / `mul`、`view_as_real/complex` | ok | ok | ok | ok |
| `nonzero` / `bincount` / `cumsum`、`allclose`、dtype 升级 | ok | ok | ok | ok |
| `where`（Ternary） | ok | **FAIL** | **FAIL** | **FAIL** |
| `sum` | ok | ok | **FAIL** | **FAIL** |
| `A @ x`（2-D × 1-D，gemv） | ok | **FAIL** | **FAIL** | **FAIL** |
| `A @ B`（2-D × 2-D，gemm） | ok | ok | ok | ok |
| Triton：load/store、where+sum、atomic_add、**associative_scan** | ok | ok | ok | ok |
| `torch.sparse` matmul（CSR / COO） | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| **FlagSparse `spmv_csr`（参考值在 CPU）** | **ok** | **ok** | **ok** | **ok** |

三条结论：

**1. Triton 是健康的，四个 dtype 全通。** 包括 `tl.associative_scan`——Ascend 的 Triton
lower 不了它，所以那边 `spmv_csr` 退回了 torch_npu；摩尔线程不需要这个回退。
`routing/spmv_csr uses triton` 也确认了算子真的走 Triton（`kernel=segbin`），没有静默
回退到 torch，所以上面那行 `ok` 是内核的功劳而不是回退路径的。

**2. muDNN 的 gemv 缺口 —— 值得报给厂商。** 同样的数学、同样的 dtype，只把 `x` 从
`(N,)` 改成 `(N,1)` 就从 FAIL 变 ok：

```
torch/matvec (@)         fp32 ok   fp64 FAIL   c64 FAIL   c128 FAIL
torch/matvec as 2D (@)   fp32 ok   fp64 ok     c64 ok     c128 ok
```

说明 torch_musa 的 2-D×1-D 走 muDNN 的 gemv（缺 fp64/复数），而 2-D×2-D 走了另一条有
这些类型的路径。这是一个最小可复现的 dispatch 缺口，不是"MUSA 不支持 fp64"。
同理"MUSA 不支持复数"也是过度概括——复数张量存得下、`add`/`mul`/`view_as_real` 都正常，
muDNN 只在 ternary / reduce / gemv 上拒绝。

**3. `torch.sparse` 在 MUSA 上没有可用的 matmul**，见第 3 节。

## 5. 算子层面的 MUSA 适配现状

目前 MUSA **没有专用内核**，走的是 CUDA 内核加通用分发。代码里与 MUSA 相关的分支只有
三处：

| 位置 | 内容 |
|---|---|
| `_common.py` | 运行时探测、`_ACCEL` 抽象、厂商基线选择、CuPy 不适用的说明 |
| `spmv_csr.py:978` | CSR SpMV 的内核选择返回 `segbin`，注释写明"和 Ascend 一样从 CUDA 内核起步" |
| `tests/test_spsv.py:94` | SpSV 只暴露 `{1: "csr_cw"}` 这一种 solve kind，与 CUDA 的多路由不同 |

也就是说，**MUSA 现在能跑，但没有针对摩尔线程硬件调过参**。要调优的话，正确做法是在
Python 的 launch 选择层加 `_is_mthreads_runtime()` 保护的配置（`BLOCK_N`、`num_warps`、
`num_stages` 之类），**不要 fork 内核实现**——`_spmm_rocm_launch_overrides` 那类函数是
天然的挂载点，目前只在 ROCm 返回配置。

## 6. 排查

**`fallback reason` 不是 `None`** —— `torch_musa` 没装或版本不匹配。这是最需要先排除的
问题，因为它会让整轮测试静默地跑在 CUDA 语义下。

**某个算子报张量不在加速器上** —— 检查 `_accel_device_type()`。MUSA 下张量必须在
`musa` 设备上，用 `torch.device("cuda")` 建的张量会被 `_is_accel_tensor()` 拒绝。

**厂商基线列全是 N/A** —— 正常。MUSA 默认用 PyTorch 基线，CuPy/cuSPARSE 列本来就
不适用，看 `reason` 字段的说明。

**想确认某一列的加速比是拿什么算的** —— 看 `run_flagsparse_pytest.py` 的
`PERFORMANCE_SPEEDUP_SCHEMAS`，它按"首个非空匹配"决定上报指标。

## 相关文档

- [DCU_TESTING.md](DCU_TESTING.md) —— ROCm/DCU
- [METAX_TESTING.md](METAX_TESTING.md) —— MetaX/MACA C550
- [ASCEND_TESTING.md](ASCEND_TESTING.md) —— Ascend 910B（与 MUSA 同属独立设备类型）
