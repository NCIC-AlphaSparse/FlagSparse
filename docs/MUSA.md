# 摩尔线程 MUSA

Python 侧的 MUSA 后端：与兼容后端的区别、环境、跑法、实测能力矩阵，以及调试记录。
C API 那一层见 `capi/docs/MUSA.md`；各后端文档的对应关系见 [README.md](README.md)。

验证环境：MTT S5000 ×1 / torch 2.7.1 / torch_musa 2.7.1 / muDNN v3105 / Triton 3.6.0。

---

## 0. 开工前先确认这一行

```bash
export PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads
python -c "import flagsparse.sparse_operations._common as C; print(C._backend_name(), C._accel_device_type(), C._accel_fallback_reason())"
# 必须是: mthreads musa None
```

`fallback reason` 不是 `None` 说明 `torch_musa` 没装好，整轮会**静默地跑在 CUDA 语义下**，
跑出来的不是 MUSA 的数。这一行不对，下面全部作废。

---

## 0.5 交付复现：40 个变体 × 30 个矩阵（精度 + 性能）

**MUSA 用 `run_flagsparse_split_delivery.py`，不用 `run_flagsparse_pytest.py --phase both`。**
Python 侧在 MUSA 上没有厂商稀疏库（第 3 节），性能阶段只有 FlagSparse 自己的耗时、没有加速比；
muSPARSE 基线在 C API 侧。这个 runner 精度取 pytest（SciPy 参考），性能取 C API 的 `ctest -R benchmark`
（对 muSPARSE），合成一份 `summary_split.json`。

**环境**：

```bash
export PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads MUSA_HOME=/usr/local/musa
python3 -c "from flagsparse.sparse_operations import _common as c; print(c._backend_name(), c._accel_device_type(), c._accel_fallback_reason())"
# 期望：mthreads musa None
mthreads-gmi                          # 单卡；跑之前确认没有别的任务在用 GPU
```

**命令**：

```bash
setsid timeout -s KILL 43200 python3 -u run_flagsparse_split_delivery.py \
  --mode normal --benchmark-input /root/gcx/matrix --timeout 3600 \
  --results-dir pytest_results_mthreads_split \
  > pytest_results_mthreads_split.log 2>&1 < /dev/null &
```

`--timeout` 同时传给 pytest 的每个算子阶段和 C API 的 CTest（`-DFLAGSPARSE_CTEST_TIMEOUT`）。
首次运行会从零构建 C API（`capi/build`），配置日志里应当出现
`ctest baseline: MUSA -> /usr/local/musa/lib/libmusparse.so`；已经构建过可加 `--skip-capi-build`。

**参考**：

| | 本后端 |
|---|---|
| 性能 baseline | **muSPARSE**（C API 侧，`capi/docs/MUSA.md`）；Python 侧没有 |
| 精度参考 | **CPU 上的 SciPy** —— MUSA 上 `torch.sparse` 能建 CSR/COO 张量但**没注册 sparse matmul** |

**预期会看到的非 Passed**（2026-09-18 在 MTT S5000 上实测，`modified/MUSA.md` 第 13 节）：

- 精度 40/40 Passed；
- 性能 `gather_f16_int`、`scatter_f16_int` 为 **`NoBaseline`**：内核跑通、精度通过，只是 muSPARSE 不支持
  fp16 的 gather/scatter；
- 性能 `spgemm_csr_*`、`spsv_*` 为 `NotFound`（C API benchmark 中 Triton 崩溃），`spsm_csr_*` 为
  `NotFound`（3600 秒超时）—— 两类原因不同，崩溃日志还在等 MUSA 那边提供。

跑完用同一个工具看 40 行结果（缺变体时退出码为 1），回传时直接贴它的输出：

```bash
python3 tools/delivery_table.py pytest_results_mthreads_split              # 加 --markdown 输出 Markdown 表
```

**`86a09cd`（2026-09-18）之前跑出的性能结果作废**，要用当前 runner 重跑，原因见 `prompt.md` 第 2 节。
参数为什么都不能省、各状态的含义，见仓库根 `README_cn.md` 的"复现交付测试"一节。

只想看 Python 侧（没有加速比）时，用通用命令 `run_flagsparse_pytest.py --phase both --mode normal --delivery-only ...`。

```bash
# 在任意后端上强制切换精度参考，用于验证另一条路径
export FLAGSPARSE_ACCURACY_REFERENCE=auto    # auto（默认）| scipy | torch
```

---

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

仓库用 `_ACCEL` 这一层抽象掉了这个差异（`_common.py` 的 `_resolve_accel`）：它把 torch
子模块和设备类型**成对**返回，`torch.cuda`/`"cuda"`、`torch.musa`/`"musa"`、
`torch.npu`/`"npu"`。成对返回是必要的 —— 只换模块不换设备类型，会让
`_is_accel_tensor()` 拒绝掉每一个张量。

Ascend（`torch.npu`）与 MUSA 属于同一类。`tests/` 下直接跑的 benchmark 脚本用
`tests/benchmark_utils.py` 拿同一对名字（`ACCEL` / `accelerator_device()`），
pytest 套件用 `tests/pytest/accuracy_utils.py`。

---

## 2. 环境准备

```bash
export PYTHONPATH=$PWD/src
export FLAGSPARSE_BACKEND=mthreads
```

后端名是 **`mthreads`**，不是 `musa` —— `musa` 是设备类型字符串，两者不要混。
`FLAGSPARSE_BACKEND` 写错会直接抛 `ValueError` 而不是静默回退。

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
`vendor sparse lib = torch`、`fallback reason = None`。已在真机确认。

如果 `fallback reason` 出现下面这句，说明 `torch_musa` 没装好：

```
backend 'mthreads' selected but torch.musa is unavailable (torch_musa not installed?);
falling back to torch.cuda
```

此时 `_accel_device_type()` 会是 `cuda` 而不是 `musa`。这个回退是故意做成成对的
（见第 1 节），不会产生"模块是 torch.cuda 但设备类型声称 musa"这种自相矛盾的状态，
但结果也就不是 MUSA 的了。

---

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
**构造是成功的**，只有乘法缺失 —— 这正是这个缺口能通过 review 的原因，光看"能不能建出
稀疏张量"是看不出来的。

可以用 `FLAGSPARSE_MTHREADS_VENDOR` 覆盖：

| 取值 | 含义 |
|---|---|
| `none` | 默认。不使用厂商基线，相关列为 `N/A` |
| `torch` | 用 `torch.sparse` 作基线 —— **当前真机上会报错**，保留是为了将来 torch_musa 补齐算子后可以用实测把默认翻回去，而不是靠假设 |
| `musparse` | 声明使用 muSPARSE **【Python 侧未接】** —— 仓库里没有对应的 binding，全仓搜不到 `import musparse`，设成它只会返回字符串，基线列仍是 `N/A`。C API 侧的 muSPARSE 基线是另一回事，已接上，见 `capi/docs/MUSA.md` |

其他取值会抛 `ValueError`。

CuPy/cuSPARSE 在 MUSA 上**不适用**，相关列会给出明确原因而不是伪造数字：

```
CuPy/cuSPARSE is not applicable on the mthreads backend (baseline: torch)
```

---

## 4. 运行测试

runner 本身没有 MUSA 特判。**交付用的命令是 0.5 节那条**（`--delivery-only --mode normal`），
下面这组 `--mode quick` 的**只用于冒烟测试**——quick 会漏跑约四成用例，而且恰好跳过两个历史
问题用例（见 `prompt.md` 第 3 节），照抄它产出的"验收"结果不算数：

```bash
export PYTHONPATH=$PWD/src
export FLAGSPARSE_BACKEND=mthreads

# 冒烟：只确认链路能通
python run_flagsparse_pytest.py --phase accuracy --mode quick --gpus 0 \
  --results-dir pytest_results_musa_smoke
```

交付跑之前要知道的三条（2026-09-17 在 MTT S5000 上实测）：

* **`--timeout 900` 不够。** 它是每个算子每个阶段的上限；30 个真实矩阵的完整 dtype 网格上，
  spgemm / sddmm / spsv 的性能阶段会被 SIGKILL，结果要么 0 行、要么最后一个 dtype 还没轮到。
  用 `--timeout 3600`。
* **单卡，顺序跑。** 本机只有一张 MTT S5000，runner 一个算子接一个算子跑，不是并行；
  **同一时刻不要再起任何碰 GPU 的任务**——精度结果不受争用影响，但并发期间的计时数字作废。
  起手前先 `mthreads-gmi` 确认卡数和占用。
* **总时长 ≈ 各算子组耗时之和**，按第 6 节的逐组耗时估，别按单个算子估。

**性能想要加速比，走 `run_flagsparse_split_delivery.py`**：Python 侧 MUSA 没有厂商稀疏库
（第 3 节），`--phase performance` 的分母恒为空；C API 侧有 muSPARSE 基线，这个 runner 精度取
pytest、性能取 C API，合成一份 `summary_split.json`。见 `modified/MUSA.md` 第 12 节。

单个算子脚本也可以直接跑：

```bash
python tests/test_spmv.py <目录或文件.mtx> --warmup 5 --iters 20
python tests/test_spmm.py <目录/> --csv out.csv
```

### pytest 精度套件（tests/pytest）

这套用例原本对 CUDA 是硬编码的，在 MUSA 上会连续踩三个坑，**都已经修好**，
但都是"在 CUDA 机器上看不出来"的那一类，所以记在这里：

1. **全部 skip。** 每个文件顶上是
   `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), ...)`。
   MUSA 上 `torch.cuda.is_available()` 为假，于是 2056 个用例全部静默跳过，
   输出只有 `341 skipped`，看起来像"跑过了"。
   现在换成 `accuracy_utils.accelerator_available()`，它探 `_ACCEL.is_available()`。

2. **全部 fail。** 守卫放行之后，用例体里的 `torch.device("cuda")` 会抛
   `NotImplementedError: Could not run 'aten::empty.memory_format' with arguments
   from the 'CUDA' backend`。148 处设备字面量现在统一走
   `accuracy_utils.accelerator_device()` / `ACCELERATOR_DEVICE_TYPE`。

3. **参考计算本身跑在加速器上。** 前两个坑填完之后，`test_spmv_csr_accuracy.py`
   仍然 74 failed / 3 passed，而且**一个都没进到 FlagSparse 内核** —— 全死在测试自己的
   数据生成和 dense 参考上：`_random_csr_mn` 的 `torch.where`（fp64/复数无 muDNN 内核）、
   以及 `ref_mat.to(ref_dtype) @ x.to(ref_dtype)`（`_reference_dtype` 把 fp32 升到 fp64，
   撞上 gemv 缺口，见 4.5）。
   现在数据生成和 golden reference 统一走 `accuracy_utils.golden_device()`（恒为 CPU），
   只有真正交给算子的张量才 `.to(accelerator_device())`。

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

### 4.5 实测能力矩阵（MTT S5000，2026-09-11）

用 `tools/probe_accel_capabilities.py` 在真机上逐层测出来的。这张表的用处是**在读任何
失败之前先知道它属于哪一层** —— 之前 `test_spmv_csr_accuracy.py` 的 74 个失败全部发生在
测试自己的参考计算里，一个都没进到 FlagSparse 内核。

```bash
export PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads
python tools/probe_accel_capabilities.py --json musa_caps.json
```

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

**1. Triton 是健康的，四个 dtype 全通。** 包括 `tl.associative_scan` —— Ascend 的 Triton
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
同理"MUSA 不支持复数"也是过度概括 —— 复数张量存得下、`add`/`mul`/`view_as_real` 都正常，
muDNN 只在 ternary / reduce / gemv 上拒绝。

**3. `torch.sparse` 在 MUSA 上没有可用的 matmul**，见第 3 节。

补充一条同源的：**复数的 gather 方向全挂、scatter 方向正常** ——
`a[order]` / `index_select` / `a[rows, cols]` 复数 FAIL，`index_copy_` 复数 ok。
库里已用 `_common._gather_values()` 绕开（改用 `view_as_real` 后 gather），
SpGEMM / SpMM / SpMV / SpSM / SpSV 里所有 `data[order]` 路径都接入了它。
**分支按 dtype 不按 backend**，所以 CUDA 上实数走原路径。

---

## 5. 算子层面的 MUSA 适配现状

MUSA **没有独立的内核实现**，走的是共享内核加通用分发。代码里按 MUSA 分叉的地方是
可枚举的：

| 位置 | 内容 |
|---|---|
| `_common.py` | 运行时探测、`_ACCEL` 抽象、厂商基线选择、CuPy 不适用的说明、`_gather_values()` |
| `spmv_csr.py:1017` | CSR SpMV 的内核选择返回 `segbin`，"和 Ascend 一样从 CUDA 内核起步" |
| `spgemm_csr.py:1148` | `tle_hash_safe = not _is_mthreads_runtime()`，MUSA 上禁用 TLE hash 路径（见 7.1） |
| `spsv.py:5479` | SELL 非转置 ALG2 的实际 launch 降级到 ALG1（见 7.2） |
| `tests/test_spsv.py:94` | SpSV 只暴露 `{1: "csr_cw"}` 这一种 solve kind，与 CUDA 的多路由不同 |

也就是说，**MUSA 现在能跑，但没有针对摩尔线程硬件调过参**。要调优的话，正确做法是在
Python 的 launch 选择层加 `_is_mthreads_runtime()` 保护的配置（`BLOCK_N`、`num_warps`、
`num_stages` 之类），**不要 fork 内核实现** —— `_spmm_rocm_launch_overrides` 那类函数是
天然的挂载点，目前只在 ROCm 返回配置。

---

## 6. 当前测试状态（2026-09-13，`--mode normal`）

```bash
PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads \
python -u run_flagsparse_pytest.py \
  --phase accuracy --mode normal --gpus 0 --timeout 900 \
  --results-dir pytest_results_musa_accuracy_normal_after_spgemm_fix
```

| 状态 | 算子组 | 用例 |
|---|---:|---:|
| Passed | 15 | **1564 passed / 0 failed** |
| Skipped | 2 | 134 skipped |
| TIMEOUT | 1 | `spsv_sell`，900.37 s —— **已于本轮之后解决，见 7.2** |

通过的 15 组：`gather`、`scatter`、`sddmm_csr`、`spgemm_csr`、`spmm_bsr`、`spmm_coo`、
`spmm_csc`、`spmm_csr`、`spmv_bsr`、`spmv_coo`、`spmv_csc`、`spmv_csr`、`spsm_coo`、
`spsm_csr`、`spsv_coo`。逐组：

| 算子组 | 状态 | passed | skipped | 耗时 |
|---|---|---:|---:|---:|
| `spmv_bsr` | Passed | 271 | 0 | 16.5 s |
| `spsv_coo` | Passed | 219 | 0 | 46.6 s |
| `spmm_csc` | Passed | 153 | 0 | 15.0 s |
| `spmm_bsr` | Passed | 151 | 0 | 14.2 s |
| `spmv_csc` | Passed | 79 | 0 | 27.7 s |
| `spmv_coo` / `spmv_csr` | Passed | 77 / 77 | 0 | 26.3 / 27.1 s |
| `spmm_csr` | Passed | 59 | 0 | 31.8 s |
| `spmm_coo` | Passed | 51 | 0 | 651.9 s（见 7.4，这个数不可信） |
| `scatter` | Passed | 50 | 0 | 13.4 s |
| `spsm_csr` / `spsm_coo` | Passed | 36 / 36 | 0 | 20.6 / 21.1 s |
| `gather` | Passed | 26 | 0 | 10.2 s |
| `spgemm_csr` | Passed | 9 | 0 | **16.2 s**（此前 TIMEOUT 300 s） |
| `sddmm_csr` | Passed | 9 | 0 | 15.1 s |
| `spsv_csr` | Skipped | 261 | 80 | 53.5 s |
| `spmm_bell` | Skipped | 0 | 54 | 5.3 s |
| **`spsv_sell`** | **TIMEOUT** | 0 | 0 | **900.4 s** |

**`spsv_csr` 的 80 个 skip** ~~是 CuPy/cuSPARSE 比对用例，在 MUSA 上本就不适用，属正常~~
—— **这个结论已被推翻**（2026-09-17）。`accuracy_utils.scipy_triangular_solve()` 早就写好了、
docstring 明写"for MUSA accuracy tests"，只是从没接到这 4 组比对用例上；接上 SciPy fallback
之后，80 个用例从 skipped 变成真正在比对，`spsv_csr` 为 **341 passed / 0 skipped**。

> **教训**：见到 `cp is None → skip` 这类硬编码，先查 `accuracy_utils.py` 有没有现成但没接线的
> fallback，不要直接认定"这个后端没法测"。这条"属正常"写于 2026-09-13，当时是对的，但没人在它
> 过时的时候回来改——它是本文档自己的一个"没跑过就写没跑过"的反例。
不要与更早的 quick 那轮（1212 passed / 105 skipped）混用 —— quick 只跑每个 marker 的子集。

---

## 7. 已解决的问题（保留记录）

### 7.1 `spgemm_csr` 卡死

当时的现象：quick 下四个 dtype×索引宽度组合全挂，唯一通过的用例**不跑内核**（只验证
dtype 拒绝）；kill 前 **stderr 为空**，符合"卡在驱动里"。CUDA 上同一个 marker 4.25 s
跑完 5 个。

根因收敛到 **TLE 的 shared-memory hash/CAS 算法路径**：内核能提交，但任意同步或主机
读取会在 MTT S5000 上阻塞。

**不是通用的 Triton 能力缺失** —— 独立原语探针（`atomic_cas` 循环、跨 lane `while`
归约、`atomic_add`、`associative_scan`）全部通过。所以问题在算法对这些原语的组合用法，
不在原语本身。这个"原语全过、组合卡死"的形状值得记住 —— 它把排查方向从"摩尔线程的
Triton 缺能力"掉转成"我们的算法在这块卡上踩到了别的东西"，两者的下一步完全不同。

修复（`spgemm_csr.py`）只在 MUSA 上禁用该优化路径，改走已验证的 ESC 路径：

```python
tle_hash_safe = not _is_mthreads_runtime()
if tle_hash_safe and _TLE_AVAILABLE and prepared.a_data.dtype in (torch.float32, torch.float64):
```

验证：最小旧复现 1 passed，`spgemm_csr` quick 5 passed、normal 9 passed；CUDA 侧确认为
恒等变换，`pytest -m spgemm_csr` 9 passed。

> `_is_mthreads_runtime` 在 `_common.__all__` 里，`from ._common import *` 能解析到 ——
> 这一点值得确认而不是假设：Ascend 那轮的 drop 用了未导出的 `_IS_ASCEND_RUNTIME`，
> 留下一个只在特定分支才触发的 NameError。

### 7.2 `spsv_sell` 超时

**不是"整体测试太慢"，是 SELL ALG2 内核偶发失去前进性。**

- 内核：`_spsv_sell_slice_kernel_alg2`
- 机制：持久化 worker + ready flag 轮询 + acquire/release atomic
- dump 报错：`MUSA_ERROR_LAUNCH_TIMEOUT`
- 卡死位置固定：**n=64 / slice=8 / ALG2，首次 solve 阶段**

修复在 `spsv.py` —— MUSA 上把**非转置 ALG2 的实际 launch 降级到已验证的 ALG1**，
descriptor/API 仍保留用户请求的 `alg_num=2`，CUDA / ROCm / MACA 不变：

```python
effective_alg_num = (
    SPSV_SELL_ALG1
    if _is_mthreads_runtime() and alg_num == SPSV_SELL_ALG2
    else alg_num
)
```

验证：相同输入重复 **200 轮、每轮两次 solve** 全过；`spsv_sell` normal
**276 passed / 11 skipped / 14.93 s**，不再 timeout。CUDA 侧确认恒等变换，
`spsv_sell + spsv_csr + spsv_coo` 846 passed（1 个上游既有失败 `duplicate_diagonal`，
与本改动无关）。

### 7.3 `spmm_bell` 的整体 skip

此前 54/54 被跳过，理由是测试里的 CUDA-only guard（"BELL SpMM currently requires CUDA
tensors"）—— 没说是哪一步要求的，属于一刀切屏蔽而不是验证过的排除。
**现在是真的跑通了，不是继续跳过。**

算子侧（`spmm_bell.py`）漏掉了其他算子早就做过的加速器抽象：

- 输入检查 `data.device.type != "cuda"` → `_is_accel_tensor()`
- `torch.cuda.Event` / `torch.cuda.synchronize()` → `_ACCEL.Event` / `_ACCEL.synchronize()`
  （timing 与 meta 路径，共 3 处）

测试侧（`tests/pytest/test_spmm_bell_accuracy.py`）：移除 MUSA 整体 skip，改用
accelerator device，MUSA 上走 SciPy CPU 参考，比较前把 CPU 参考搬到输出所在 device。

MUSA：**54 passed / 0 failed / 0 skipped**，覆盖 float32 / float64 / complex64 /
complex128 × int32 / int64 × block_dim 2/4，含 prepared/meta 路径、空槽、非法长度、
保留的 transpose/conj 行为。CUDA 侧同样 54 passed。

**一刀切的 skip 不是"算子通过"，它掩盖的是一个真实的适配缺口。**

### 7.4 更正：`spmm_coo` 的"24 倍慢"不成立

**此前基于单次 `summary.csv` 的 651.9 s / 51 用例算出 12.78 s/用例，断言它比
`spmm_csr` 慢 24 倍、并推测慢和偶发卡死"大概率同一根因"。这个结论是错的，已撤回。**

同一条 runner 命令重跑：

```
spmm_coo   51 passed / 13.32 s
spmm_csr   59 passed / 13.34 s
```

两者基本持平。那 651.9 s 不是稳定性能数据，应是 MUSA runtime 异常或冷启动状态导致的
累计耗时。

方法上的教训：**把单次计时当成了算子的特征量**。判定"非确定性"要靠重复计数，判定
"性能离群"同样要 —— 一个数据点不足以支撑任何一边的结论。

---

## 8. 仍需处理

**`spmm_coo`：复数共轭用例仍有稳定性风险，但不是性能问题。**

`conj-int64-complex128-8-4-16`：最近 3 次独立运行全过，但此前的重复测试里出现过一次
`timeout -s KILL`，**因此不宣告彻底修复**。`spmm_coo.py` 未作改动。

同一个 op × dtype 组合在 CUDA 上也观察到过卡死（`conj-int32-complex64`），两块不同硬件
落在同一类用例上，指向 **conj 复数路径本身**而不是某块卡的驱动。

收窄方向：`spmm_coo.py` 的复数共轭物化、排序、以及 row-run / atomic 这几条 Triton 路径。
先用 `tools/bisect_hang.py --marker spmm_coo` 拿到逐用例耗时，看是所有用例都慢还是
某几个特化拖的。计数用例：

```bash
for i in $(seq 1 5); do
  timeout -s KILL 90 python -m pytest \
    'tests/pytest/test_spmm_coo_accuracy.py::test_spmm_coo_matches_dense_reference[conj-int64-complex128-8-4-16]' \
    --mode quick -q -o addopts= >/dev/null 2>&1
  echo "run $i -> exit $?"   # 137 或 -9 = 被 KILL，即卡死
done
```

**`spsv_sell:404` 的 skip 理由含糊** —— `MUSA SELL validation differs from CUDA`，
没说差在哪，值得回头看。同组里 `torch_musa does not implement isnan for complex tensors`
那部分是已知的 torch_musa 能力缺口，窄而具体，可接受。

**Python 侧的 muSPARSE 基线未接**，见第 3 节。

---

## 9. 工具

| 工具 | 用途 |
|---|---|
| `tools/probe_accel_capabilities.py` | 分层能力探针。默认隔离模式，一项一个子进程 + 硬超时，抓子进程 stderr |
| `tools/bisect_hang.py` | 把超时的 marker 收到具体卡死的用例。三阶段：collect → whole → 逐个独立进程 |
| `tools/diagnose_musa.sh` | 上面两个串起来跑一遍，输出打包 |

```bash
python tools/bisect_hang.py --marker <marker> --timeout 60 --json hang.json
python tools/bisect_hang.py --marker <marker> --stop-after-first-hang   # 只要最小复现
```

**必须用探针的默认隔离模式**：某项真卡死时只占一行 `HANG`，不会拖垮整个探针。

---

## 10. 踩过的坑，别重踩

**一律 `timeout -s KILL`。** Ctrl-C 打不断卡死的 GPU 内核 —— 进程阻塞在驱动里，
软超时留下僵尸，下一个进程排在它后面。

**一个配置一个进程。** 内核 fault 之后厂商 runtime 被污染，同进程里后续结果全是垃圾。

**照字面读厂商报错，先看 stderr。** muDNN 真正的诊断打在 stderr，Python 异常只说
`MmCall failed`。MetaX 那次真正的原因（每线程 8 KB 私有内存超 4 KB 上限）明写在上一行。

**"没输出"不等于"通过"。** 真实发生过三次：
- 探针拆成两个文件，只 scp 了主文件 → 整组 16 个 FAIL，和硬件结论长得一模一样
- `pytest.ini` 有 `addopts = -v`，`--collect-only` 打树形不打 node id → 解析到 0 个，
  表现为"没有测试"（已用 `-o addopts=` 修掉）
- `grep | sed || echo` 在无匹配时 sed 仍 exit 0，缺失表现为什么都不打印

**`timeout -s KILL` 的返回码，shell 看到 137，Python 的 subprocess 看到 -9。**
按 shell 语义写判断会把每个卡死标成普通失败，整个诊断方向反掉。`-11`(SIGSEGV) 是崩溃
不是卡死，要排除在外。

**确认通过的测试到底跑的是哪条路径。** SpSV 的 unit/non-unit 走两个完全不同的内核；
`spgemm_csr` 唯一通过的那个用例根本不跑内核。

**诊断代码本身也是嫌疑人。** 先用测试自己的算子复现，再加仪器，且只用
`isnan().sum()` / `max()` 这类廉价原语 —— 在不成熟的 runtime 上，花哨的张量操作本身
可能是坏的。

**别从"剩余失败数"估"剩余缺陷数"。** 每修好一层才露出下一层，四层长得完全不一样：
skip guard → 148 处设备字面量 → 测试自己的 oracle → 库自己的重排。
74 个失败里藏着两个独立的 bug。

---

## 11. 测试侧的两套做法并存

改文件前先确认在改哪一套：

| 文件 | 做法 |
|---|---|
| `test_spmv_csr` / `test_spmv_coo` / `test_spmv_csc` / `test_gather_scatter` | `golden_device()`，无后端门控，参考恒在 CPU |
| 其余 accuracy 文件 | `is_mthreads_backend()` 门控 + SciPy 参考 |

`accuracy_utils.py` 两套共用，是超集（`golden_device` 和 `scipy_*` 都有）。

`spmv_coo` / `spmv_csc` 的 fp32 与 complex64 容差放宽到 `1e-3` 以容纳不同的累加顺序；
fp64 / complex128 维持严格。
