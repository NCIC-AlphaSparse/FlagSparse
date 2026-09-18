# 昆仑芯 XPU

Python 侧的昆仑芯后端：槽位已就位，实机（P800）脚本已回传并合入。本文写的是怎么跑、
以及哪些结论现在还不能下。改了哪些文件见 [`modified/XPU.md`](../modified/XPU.md)。
C API 那一层见
`capi/docs/XPU.md`（结论是 C API 目前在昆仑芯上跑不起来，可达的只有 Python 侧）；
各后端文档的对应关系见 [README.md](README.md)。

> **状态：已有有限实测，不是完整交付轮。** 2026-09-17 在 P800 的 `torch_xmlir` CUDA-shim
> 路径上，当前仓库的 `gather` normal 精度中 f16/f32/c32 通过、f64/c64 因厂商 eager
> 降精度失败；`spmv_csr` fp32 小矩阵独立参考通过（最大绝对误差 `1.43e-6`）。上层仓库的
> `results/xpu_variants_40.csv` 保存了 40 个变体的历史状态。2026-09-18 跑过一轮 `--delivery-only`
> 的 40 变体精度（逐变体结果见 `modified/XPU.md` 第 8 节）；那一轮的性能数字出自 `86a09cd` 之前的 runner，
> 不能用，要按 1.5 节重跑。下面未标明实测的内容不能当作通过率结论。

---

## 1. 探测：插件必须真的装了，`torch.xpu` 存在不算数

后端名是 **`xpu`**，注册在 `_common.py` 的 `_BACKEND_SPECS`：

| 字段 | 值 |
|---|---|
| 厂商 | Kunlunxin XPU |
| torch 命名空间 | `torch_xmlir` 构建走 `torch.cuda` shim；`torch_xpu` 构建才走原生 `torch.xpu` |
| 插件模块 | `torch_xmlir`、`torch_xpu`（任一可 import 即可） |
| 设备名 token | `kunlun`、`xpu` |

**探测要求插件模块可 import，光有 `torch.xpu` 命名空间不认。** 这不是多余的谨慎：
上游 PyTorch 给 Intel GPU 也装了 `torch.xpu` 命名空间，只看命名空间会把一台 Intel 机器
当成昆仑芯 —— 要么在 Intel 硅片上跑却报告昆仑芯，要么在有命名空间没设备的机器上于
第一次分配时报一个既不指明厂商也不指明原因的错。逻辑在 `_vendor_plugin_present()`。

`FLAGSPARSE_BACKEND=xpu` 的显式指定**优先于探测**，这是让没法探测的机器也能进到这个
槽位的办法。

### 自检

```bash
export PYTHONPATH=$PWD/src
export FLAGSPARSE_BACKEND=xpu

python3 - <<'PY'
import importlib, torch
for m in ("torch_xmlir", "torch_xpu"):
    try:
        importlib.import_module(m); print(f"{m}: ok")
    except Exception as exc:
        print(f"{m}: 不可用 - {exc}")
print("torch.xpu 命名空间:", hasattr(torch, "xpu"))
if hasattr(torch, "xpu"):
    print("torch.xpu.is_available():", torch.xpu.is_available())
PY

python3 - <<'PY'
import flagsparse.sparse_operations._common as C
print("backend          :", C._backend_name())        # 期望 xpu
print("is_xpu_runtime   :", C._is_xpu_runtime())      # 期望 True
print("accel device type:", C._accel_device_type())   # 期望 xpu
print("fallback reason  :", C._accel_fallback_reason())
PY
```

`torch_xmlir` 是 FlagTree 的 CUDA-shim 构建：它将 `torch.cuda` 调用路由到昆仑芯，
`torch.xpu` 仍是上游 PyTorch 的不支持 stub，`torch.xpu.is_available()` 可为 `False` 且
`torch.device("xpu")` 会报 `Torch not compiled with XPU enabled`。这是该构建的预期行为，
不是 CUDA 回退；此时自检的正确结果是 `backend=xpu`、`accel device type=cuda`、
`fallback reason=None`。同时设置 `FLAGTREE_BACKEND=xpu`、`TRITON_BACKEND=xpu` 和
`XPU_EVENT_KL3_ENABLE=1`，并使用厂商运行时要求的 `LD_LIBRARY_PATH`。2026-09-17 已以该
路径在 P800 的设备 1 跑通 `gather float32`（256x256、nnz=4096）。

---

## 1.5 交付复现：40 个变体 × 30 个矩阵（精度 + 性能）

**环境**（第 1 节自检通过后）：

```bash
export PYTHONPATH=$PWD/src
export FLAGSPARSE_BACKEND=xpu FLAGTREE_BACKEND=xpu TRITON_BACKEND=xpu XPU_EVENT_KL3_ENABLE=1
# 另按厂商运行时要求设置 LD_LIBRARY_PATH
python3 -c "from flagsparse.sparse_operations import _common as c; print(c._backend_name(), c._accel_device_type(), c._accel_fallback_reason())"
# 期望：xpu cuda None（torch_xmlir 通过 torch.cuda 运行，device type 为 cuda 是正确的）
```

**命令**：

```bash
setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --phase both --mode normal --delivery-only --gpus <卡号> --timeout 3600 \
  --benchmark-input <30 个 .mtx 所在目录> --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir pytest_results_xpu_delivery \
  > pytest_results_xpu_delivery.log 2>&1 < /dev/null &

python3 tools/delivery_table.py pytest_results_xpu_delivery   # 跑完后：40 行结果，缺变体时退出码为 1
```

runner 在 XPU 上自动处理：子进程 `CUDA_VISIBLE_DEVICES=<卡号>`、命令行传逻辑设备 `--device 0`；精度阶段
**强制**用 CPU 上的 SciPy 参考（`FLAGSPARSE_ACCURACY_REFERENCE=scipy`）；gather / scatter / spmv_csr /
spmm_csr / sddmm_csr 的性能走 `benchmark/benchmark_xpu.py`，每个矩阵一个子进程，对等价的 PyTorch-XPU
表达式计时；其余 6 个父算子走能力探测，只有能不能跑、没有加速比。

**参考**：

| | 本后端 |
|---|---|
| 性能 baseline | `torch`（PyTorch-XPU 表达式）—— XDNN 是固定算子集不是描述符 API，没有可绑的厂商稀疏库 |
| 精度参考 | **CPU 上的 SciPy**（runner 强制） |

`benchmark_xpu.py` 自己也会先核对输出再计时：超出容差的矩阵记为 `MISMATCH`，不计入加速比。这是性能脚本的
门禁，不是交付的精度结果。

**预期会看到的非 Passed**（2026-09-18 在 P800 上实测，`modified/XPU.md` 第 8 节）：40 个变体里只有 gather /
scatter 的 f16、f32、c32 和 spmv_csr 部分用例通过精度，其余多为 Triton XPU lowering 资源不足（`uni_sram`）、
legalization 失败或找不到可执行函数。这是平台当前的真实状态，如实回报即可。

**`86a09cd`（2026-09-18）之前跑出的性能结果作废**，原因见 `prompt.md` 第 2 节。

---

## 2. 内核：没有 XPU 专用实现，走共享那一份

`src/flagsparse/sparse_operations/` 仍以共享实现为主；当前仅有 `_common.py` 的运行时解析和
`spmv_csr.py` 的兼容分支使用 `_is_xpu_runtime()`。`backends/xpu/` 目录仍只有 `__init__.py`，
没有一份按后端复制的算子实现。

这是有意的：`_dispatch.py` 的覆盖层默认为空，**只有真正分叉的文件才会出现在里面**。
真在昆仑芯上要改，优先级从低到高是——加 profile 表项（launch 参数、`num_warps` 之类）
→ 同文件内加内核变体 → fallback 分发表 → 最后才是往 `backends/xpu/` 放整份文件。

所以昆仑芯上能不能跑，取决于 **FlagTree 的 xpu target 能不能把这些 Triton 内核编出来**，
而不是取决于本仓库有没有 XPU 代码。

---

## 3. 跑测试

### 按后端 suite 跑

`tests/backends/xpu/suite.json` 已就位，`tools/run_backend_tests.py` 会据此设好
`FLAGSPARSE_BACKEND`：

```bash
python tools/run_backend_tests.py --backend xpu --phase accuracy --mode quick
```

精度用例与其他后端**共用** `tests/pytest`，不另起一套 —— 六份同样的 oracle 各自漂移
是这个仓库明确要避免的。

### 性能阶段：五个算子走自己的脚本，其余走能力探测

```python
PROBE_ONLY_BACKENDS: tuple[str, ...] = ("gcu", "mlu")     # xpu 已经不在里面
XPU_BASELINE_OPS = ("gather", "scatter", "spmv_csr", "spmm_csr", "sddmm_csr")
```

这五个走 `benchmark/benchmark_xpu.py`，以等价的 PyTorch-XPU tensor expression 为 baseline，
先核对输出再分别计时，报告 `triton_ms`、`pytorch_ms`、`max_abs_err` 和
`triton_speedup_vs_pytorch`（也保留兼容列 `speedup`）。其余算子仍走**能力探测**，按算子报
`PASS` / `REJECTED` / `TRITON_COMPILE` / `ERROR` / `NO_ADAPTER`。在一个内核可能根本
lower 不出来的平台上，这才是有意义的测量 —— 替代方案是一个空的性能阶段，而空阶段
读起来和通过一模一样。

> **这是 PyTorch baseline，不是厂商 sparse-library baseline。** XDNN 没有可替代 cuSPARSE
> 描述符 API 的通用稀疏接口；因此 `speedup` 表示 FlagSparse 相对 PyTorch-XPU expression 的
> 比值。输出超出 dtype 容差时该矩阵为 `MISMATCH`，不会参与平均加速比。

探测脚本和 `benchmark_xpu.py` **无论内核跑通、被拒还是编译失败都 exit 0**，结论在 CSV 行
里。runner 因此用 `_capability_probe_status()` 折叠行状态（全同取之、混合取 `MIXED`、
无行取 `NO_TESTS`）而不是看退出码 —— 看退出码会把每一行都报成 pass。

交付测试的完整命令见 **1.5 节**（`--mode normal --delivery-only`，带 `--timeout` 和 30 个矩阵）。只想确认链路
能通时，可以先跑一次冒烟（结果不能当交付数据）：

```bash
python run_flagsparse_pytest.py --phase both --mode quick --delivery-only --gpus 0 \
  --timeout 900 --results-dir pytest_results_xpu_smoke
```

> 探测脚本叫 `benchmark/benchmark_ascend_probe.py`，**名字有历史包袱，实现是后端中立的**
> —— 它通过 `_resolve_accel()` 跟随当前后端，不绑昇腾。每个用例都清空 Triton 缓存后再跑，
> 因为上一个用例留下的缓存产物会让编译失败看起来像通过。

---

### 单独跑那个性能脚本（排查时最有用）

runner 会自动调它；给目录时，XPU baseline 的每个矩阵都会由 runner 以 `--matrix` 在独立
进程中执行，因此一张卡住的矩阵不会吞掉同一算子的其余结果。runner 已把 `--gpus 4` 映射为
子进程的 `CUDA_VISIBLE_DEVICES=4` 和逻辑 `--device 0`；单独调用脚本时则传物理设备号。定位问题时单独跑更快：

```bash
export PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=xpu
python3 benchmark/benchmark_xpu.py --op spmv_csr --device 0 \
  --csv-summary /tmp/xpu_spmv.csv --warmup 5 --iters 20
# --matrix <file.mtx> 或 --matrix-dir <目录> 用真实矩阵；不给就用合成输入
```

CSV 的 `status` 可以是 `PASS`、`MISMATCH` 或 `ERROR`；`MISMATCH` 会带误差与容差说明且不计性能。
`benchmark_xpu_variants.py` 是按清单跑变体的那一层，接受 `--manifest` 与 `--matrix-dir`。

## 4. 没有厂商基线，而且不是"等一个库名"

昆仑芯的数学库是 **XDNN**，提供的是**固定的稀疏算子**，不是描述符式的 generic API。
因此没有 XDNN sparse-library 的可比速度；当前 `speedup` 是相对 PyTorch-XPU expression 的速度比，
不是 XDNN 的结果。

这和 MACA / MUSA 的情况**不是一回事**，不要混：

| | 昆仑芯 XPU | MACA / MUSA |
|---|---|---|
| 厂商有稀疏库吗 | 有 XDNN，但不是 generic API | 有，且是 cuSPARSE 克隆 |
| 缺什么 | 缺的是 **API 形状**，补不了 | 只缺头路径和库名 |
| 怎么解 | 需要厂商出 generic API，或本仓库为 XDNN 写逐算子适配 | 填两个 CMake 变量 |

这是**诚实的空**，不是没做完。详见 `capi/docs/XPU.md`。

---

## 5. C API 在昆仑芯上跑不起来

两条独立阻塞，补上其中一条不解锁另一条：

1. **`deps/libtriton_jit` 的后端清单里没有 XPU**（有 CUDA / GCU / HCU / MACA / MLU /
   MUSA / NPU）。这个桥负责编译和启动每一个内核，所以写一个 adaptor 不会有帮助 ——
   adaptor 解决的是 runtime 调用，不是内核从哪来；
2. **没有 cuSPARSE 形状的稀疏库可做基线**（见第 4 节）。

`tests/ci/test_backend_test_profiles.py` 里 `xpu` 的 `capi_buildable` 因此是 `False`，
这是被断言住的状态而不是待办。要在昆仑芯上测算子，今天的入口就是本文第 3 节的 Python 侧。

---

## 5.5 排查

| 现象 | 首先检查 |
|---|---|
| `backend` 不是 `xpu` / `fallback reason` 非空 | 厂商插件（`torch_xmlir` / `torch_xpu`）是否真的能 import。`torch_xmlir` 路径的 `accel device type=cuda` 是预期；只有 `torch.xpu` 命名空间不算 —— 上游 PyTorch 给 Intel GPU 也装它，认了就会把 Intel 卡当昆仑芯 |
| `fallback reason` 不是 `None` | 同上；不处理的话整轮会**静默跑在 CUDA 语义下**，跑出来的不是这台机器的数 |
| 某算子报 `TRITON_COMPILE` | 内核在 FlagTree 的 xpu target 上 lower 不出来。先用最小 Triton kernel 确认后端本身可用（参考 `MACA.md` 2.1），再看是哪个算子 |
| 某算子报 `REJECTED` | 算子自己拒绝了输入（dtype/layout/shape），是**有意的限制**不是缺陷，看 `reason` |
| 某算子报 `NO_ADAPTER` | 探测脚本没有这个算子的输入配方，是脚本的缺口，按 `benchmark_ascend_probe.py` 里已有的算子照着加 |
| 性能列为空 | 若为 `MISMATCH`，先修精度；若为 `ERROR`，先看 reason。`PASS` 时应有相对 PyTorch expression 的加速比，见第 3、4 节 |
| 精度结论可疑 | 精度参考是 CPU 上的 SciPy（见 1.5 节）。想对照 torch 的结论：`FLAGSPARSE_ACCURACY_REFERENCE=torch` 再跑一次 |
| ctest 配置在 XPU 处停下 | 预期行为，见第 5 节。等 `deps/libtriton_jit` 出现 `cmake/BackendXPU.cmake` |

跨后端通用的几条（`timeout -s KILL`、一个配置一个进程、先看 stderr、"没输出"不等于"通过"）
展开见 `MUSA.md` 第 10 节。

## 6. 首次上机时按这个顺序走

1. 第 1 节的自检 —— `backend=xpu` 且 `fallback reason=None` 再往下；`torch_xmlir` 路径的 `accel device type=cuda` 正确；
2. 确认 FlagTree 的 xpu target 能编译执行一个**最小 Triton kernel**（写成真实 .py 文件，
   不要用 stdin，Triton 要读源码；参考 `MACA.md` 2.1 那段）；
3. 单算子精度：先 `gather` / `scatter`（最简单、无需矩阵文件），再 `spmv_csr`；
4. 全量精度 `--mode normal`（`quick` 会砍掉约四成覆盖，且恰好跳过历史上出过问题的两个
   用例，见 `MACA.md` 7.1）；
5. 性能阶段的能力探测，拿到逐算子的 PASS / TRITON_COMPILE 分布；
6. 把第 1 节自检的实际输出、以及探测结果回填到本文 —— 那时才能把顶部的"未验证"去掉。

排查时通用的几条（跨后端适用）：**一律 `timeout -s KILL`**、**一个配置一个进程**、
**先看 stderr 的厂商诊断**、**"没输出"不等于"通过"**，展开见 `MUSA.md` 第 10 节。
