# 昆仑芯 XPU —— 改动台账

**实机环境**：P800 / XPU runtime（`benchmark_xpu_variants.py` 的 `MEASURED` 表注明逐条实跑过）。
**基于版本**：一份**早于 2026-09-16 交付变体改造**的 `run_flagsparse_pytest.py` 副本。
**最近回传**：2026-09-17，`_common.py` 的 `torch_xmlir` CUDA-shim 识别、对应 CI 契约测试和本文档；此前 2026-09-16 17:22 回传 3 个文件，2 个脚本逐字节合入，runner **逐 hunk 移植**（原因见下）。

跑法见 [`docs/XPU.md`](../docs/XPU.md)；C API 那层（目前跑不起来）见
[`capi/docs/XPU.md`](../capi/docs/XPU.md)。

---

## 0. 本轮 `docs/XPU.md` 的一致性修订（完整追加改动）

原文还有六处沿用“baseline 不执行”的旧描述；以下是该文件的全部追加修改，修改后行号为
93-95、105-108、181、187-188、214、220、232。理由是避免将
`torch_xmlir` 的有效 shim、`MISMATCH` 和 PyTorch expression 速度比误解为回退或空性能。

```diff
@@
-注意与性能路径区分：五个交付算子的性能走 `benchmark/benchmark_xpu.py`，它只给
-时延和「能否执行」，不给加速比（见第 3 节）。
+注意与精度 suite 区分：五个交付算子的性能走 `benchmark/benchmark_xpu.py`，它以
+PyTorch-XPU expression 作精度门禁和性能 baseline；它不是 SciPy 精度 oracle，也不是 XDNN
+厂商稀疏库基线（见第 3 节）。
@@
-`src/flagsparse/sparse_operations/` 下的算子文件里**没有任何 `_is_xpu_runtime()` 分支**
-（可以自己 grep 确认）。`backends/xpu/` 目录已建好但只有 `__init__.py`，算子代码留空，
-因此解析到共享实现。
+`src/flagsparse/sparse_operations/` 仍以共享实现为主；当前仅有 `_common.py` 的运行时解析和
+`spmv_csr.py` 的兼容分支使用 `_is_xpu_runtime()`。`backends/xpu/` 目录仍只有 `__init__.py`，
+没有一份按后端复制的算子实现。
@@
-CSV 的 `status` 只有两种：`PASS`（跑通了）或 `ERROR`（带 `reason`）。
+CSV 的 `status` 可以是 `PASS`、`MISMATCH` 或 `ERROR`；`MISMATCH` 会带误差与容差说明且不计性能。
@@
-所以 speedup 列会是空的，`reason` 字段里写明原因。
+因此没有 XDNN sparse-library 的可比速度；当前 `speedup` 是相对 PyTorch-XPU expression 的速度比，
+不是 XDNN 的结果。
@@
-| `backend` 不是 `xpu` / `accel device type` 不是 `xpu` | 厂商插件（`torch_xmlir` / `torch_xpu`）是否真的能 import。**只有 `torch.xpu` 命名空间不算** —— 上游 PyTorch 给 Intel GPU 也装它，认了就会把 Intel 卡当昆仑芯 |
+| `backend` 不是 `xpu` / `fallback reason` 非空 | 厂商插件（`torch_xmlir` / `torch_xpu`）是否真的能 import。`torch_xmlir` 路径的 `accel device type=cuda` 是预期；只有 `torch.xpu` 命名空间不算 —— 上游 PyTorch 给 Intel GPU 也装它，认了就会把 Intel 卡当昆仑芯 |
@@
-| 性能列有数但没有加速比 | 预期行为：没有厂商稀疏库可比，见第 4 节 |
+| 性能列为空 | 若为 `MISMATCH`，先修精度；若为 `ERROR`，先看 reason。`PASS` 时应有相对 PyTorch expression 的加速比，见第 3、4 节 |
@@
-1. 第 1 节的自检 —— `backend=xpu` 且 `accel device type=xpu` 再往下；
+1. 第 1 节的自检 —— `backend=xpu` 且 `fallback reason=None` 再往下；`torch_xmlir` 路径的 `accel device type=cuda` 正确；
```


## 1. ⚠️ runner 不能整份覆盖

回传的 `run_flagsparse_pytest.py` 与仓库版差 **+148 / −230 行**。多出来的是 XPU 路由；
**少掉的 230 行里有 137 行是当天刚做完的交付变体投影**：

```
_DELIVERY_DTYPE_TOKENS / _DELIVERY_PERF_DTYPES
_delivery_not_configured_phase()  _delivery_accuracy_phase()
_delivery_performance_phase()     _delivery_results()
summarize_accuracy_cases()
from tools.delivery_variants import load_delivery_variants
write_summary() 里 ordered = _delivery_results(results)
```

整份覆盖的结果是 `summary.json` 从 **40 个交付变体**退回按算子名出结果，C API 与 Python
两侧的清单统一当场失效。所以只摘了下面第 2 节那几项移植，其余保留仓库版。

**下次回传 runner 请只写改了哪几个函数**，或者先 pull 一次再改。

## 2. 已移植的改动（都在 `run_flagsparse_pytest.py`）

| 锚点 | 改动 | 为什么 | 其他后端 |
|---|---|---|---|
| 模块顶部 | `csv.field_size_limit(sys.maxsize)`，`OverflowError` 时退到 `2**31-1` | XPU 编译诊断里可能带 MLIR reproducer，超过 csv 模块 128 KiB 默认上限。**它是执行结果的一部分，留不住就会把 TRITON_COMPILE 行变成 PASS** | 无影响 |
| `STATUS_TO_FLAGGEMS` | 新增 `REJECTED→Skipped`、`TRITON_COMPILE→Failed`、`MISMATCH→Failed`、`ERROR→Error`、`MIXED→Error` | 探测结论要能进 FlagGems 格式。后端拒绝的内核是 skip，编译不出来的是 fail，**两者都不能读成 pass** | 纯新增键 |
| `PROBE_ONLY_BACKENDS` | `("xpu","gcu","mlu")` → `("gcu","mlu")` | XPU 有了自己的脚本 | gcu/mlu 不变 |
| `XPU_BASELINE_OPS`（新增） | `gather`/`scatter`/`spmv_csr`/`spmm_csr`/`sddmm_csr` 五个走 `benchmark_xpu.py`，其余留在探测路径 | XPU SDK 没有 cuSPARSE 形状的 generic 稀疏 API | 仅 `backend == "xpu"` 分支 |
| `_xpu_baseline_command()`（新增） | 渲染 `benchmark/benchmark_xpu.py --op/--device/--csv-summary/--warmup/--iters` | | |
| `run_performance()` | 加 `elif backend == "xpu":` 路由 | | 其他分支不变 |
| `_capability_probe_status()`（新增） | 把 CSV 行里的状态折叠成一个：全同取之，混合取 `MIXED`，无行取 `NO_TESTS` | **探测脚本无论内核跑通、被拒还是编译失败都 exit 0**，verdict 在行里。拿退出码当结论会把每一行都报成 pass | 只在探测/XPU 脚本上启用 |
| `write_benchmark_json_from_csv()` | 加 `status=` / `test_case=` 透传（`benchmark_json_from_csv` 早就有这两个参数） | 上面那条要用 | 默认值不变 |
| `load_operator_catalog()` | 除 `ops:` 外支持 `operators:` 列表，提取 `id`/`labels`/`reporting`/`delivery_dtypes`/`dtypes` | 能直接读 C API 的交付清单 `capi/conf/operators.yaml` | `conf/operators.yaml` 走原分支 |
| `_parse_operator_catalog_fallback()` | 认 `reporting:` 字段 | 没装 yaml 时的退化解析要一致 | |
| `read_ops(delivery_only=...)` + `--delivery-only` | 只选 `reporting: delivery` 的条目 | | 默认 False |

实测（本机，无 XPU 硬件，只验路由与解析）：`--delivery-only` 从 `capi/conf/operators.yaml`
的 22 条里选出 **11** 条 delivery，不带该参数是 15 条；`conf/operators.yaml` 仍解析出 25 条；
五个 baseline 算子路由到 `benchmark_xpu.py`，其余到探测脚本；状态折叠四种情况全部符合预期。

## 3. 未移植 / 缺口

| 项 | 情况 |
|---|---|
| **`benchmark/backend_capability_probe.py`** | 回传的 runner 把探测脚本指向这个**后端中立的新名字**，并改用 `--mode quick` 取代 `--warmup/--iters`。**这个文件不在回传里**，仓库中也没有。因此 `_probe_command()` 仍指向现有的 `benchmark/benchmark_ascend_probe.py`（它没有 `--mode` 参数）。**要么把新脚本发过来，要么确认沿用旧的** |
| 大量括号/换行重排 | 回传版本未经 `ruff format`，diff 里相当一部分是这类噪声，未采纳 |

## 4. 新增脚本（逐字节合入；后续已在本地补齐 PyTorch baseline）

| 文件 | 内容 |
|---|---|
| `benchmark/benchmark_xpu.py` | 在 XPU 上构造并执行 FlagSparse 内核并计时。CSV 列：`dtype,shape,triton_ms,pytorch_ms,speedup,max_abs_err,status,baseline,reason` |
| `benchmark/benchmark_xpu_variants.py` | 按清单跑变体，带 `MEASURED` 表（注明每条都在 P800 上单独启动过），支持 `--manifest` / `--matrix-dir` |

> **本段初始结论已被第 7 节的本地改动替代。** `benchmark_xpu.py` 现在实际执行
> `baseline()`：候选与 PyTorch-XPU expression 先作 host 侧容差核对，再分别计时；CSV 会写入
> `pytorch_ms`、`speedup`、`triton_speedup_vs_pytorch`、`max_abs_err` 和 `baseline=pytorch`。
> 不满足容差的行是 `MISMATCH`，不写性能数。因此这里的 baseline 仍是 PyTorch expression，
> **不是** XDNN/vendor sparse-library baseline。

## 5. 库内改动：初始回传为零；2026-09-17 起有 XPU 兼容改动

初始回传时 `src/flagsparse/` 下没有 XPU 专用逻辑、`backends/xpu/` 只有 `__init__.py`；这已经
不再是当前状态。第 6 节记录了 `_common.py` 的 CUDA-shim 解析和 `spmv_csr.py` 的 XPU
compatibility 分支。其余算子仍走共享实现，是否能运行仍取决于 FlagTree XPU target 的 lowering。

## 6. 2026-09-17 `torch_xmlir` CUDA-shim 适配

**实机环境指纹**：P800 / `torch 2.9.0+cu129` / FlagTree-Triton 3.6.0 / `torch_xmlir`；
`torch.xpu.is_available() == False` 但 `torch.cuda.is_available() == True`。
**基于版本**：`54a3d8d1fa731b4cf3855e1644b530d87b37e38c`（`ci refines`）。
**本轮回传**：5 个文件；当前仓库的设备 1 跑通 `gather float32`
（256x256、nnz=4096，101.05 ms），设备 2 跑通 `spmv_csr float32`
（256x256、nnz=4096，0.743 ms，独立参考最大绝对误差 `1.43e-6`）。
`gather` normal 精度中 f16/f32/c32 通过，f64/c64 因厂商 eager 降精度失败；没有运行完整 40 变体交付轮。

| 文件 | 锚点 | 改动 | 为什么 | 其他后端 |
|---|---|---|---|---|
| `src/flagsparse/sparse_operations/_common.py` | `_vendor_accel_available()` | 在原生厂商命名空间路径上，将 `is_available()` 纳入 `_resolve_accel()` 和 fallback 判定。 | 插件可导入不等于设备可分配；原生 `torch_xpu`、MUSA、NPU 等路径不能因为空命名空间而声称已就绪。 | 注册表驱动；只影响有独立命名空间但设备不可用的后端。`torch_xmlir` shim 由下一行优先处理。 |
| `src/flagsparse/sparse_operations/_common.py` | `_xpu_cuda_shim_present()`、`_resolve_accel()`、`_accel_fallback_reason()` | `torch_xmlir` 存在时将 XPU 映射到 `torch.cuda` CUDA shim，并以 shim 的 `torch.cuda.is_available()` 判断就绪；原生 `torch_xpu` 路径保留。 | 上层 P800 实跑脚本和历史 40 变体 CSV 都通过 `cuda:<id>` 执行；将 `torch.xpu.is_available()==False` 判为回退会拒绝真实 XPU。 | 仅 `backend == "xpu"` 且 `torch_xmlir` 可导入时改变；CUDA、ROCm、MACA 与原生 `torch_xpu` 不变。 |
| `src/flagsparse/sparse_operations/_common.py` | `__all__` | 导出 `_is_xpu_runtime()`。 | `spmv_csr.py` 通过 `from ._common import *` 使用该守卫；漏导出会在实际 XPU 分支触发 `NameError`。 | 纯导出，无运行时行为变化。 |
| `src/flagsparse/sparse_operations/spmv_csr.py` | `_spmv_csr_real_kernel()`、`_spmv_csr_default_backend()`、`_triton_spmv_csr_impl_rowpar()` | XPU 默认选 `rowpar`；仅以 `XPU_COMPAT` constexpr 移除 rowpar 内的 `tl.where` 类型合流。 | 当前版误选 `segbin`，在 `tt.scan` encoding 处编译失败；恢复 rowpar 后，XPU fp32 在 256x256、nnz=4096 上通过独立参考（最大误差 `1.43e-6`）。 | 仅 `_is_xpu_runtime()` 为真时改变 kernel 选择和 Triton 分支；其他后端保留 `segbin` 与原 `tl.where`。 |
| `tests/ci/test_backend_baseline_policy.py` | `test_fallback_reason_requires_an_available_vendor_device()` | 模拟插件存在、原生 XPU 命名空间不可用且没有 `torch_xmlir` shim 时的 fallback。 | 锁住第一行的“插件存在不等于设备可用”契约，避免静默使用 CUDA 却报告独立后端。 | 纯模拟，无设备依赖；覆盖所有原生独立命名空间的共同语义。 |
| `tests/ci/test_backend_baseline_policy.py` | `test_torch_xmlir_xpu_uses_the_cuda_shim()` | 模拟 `torch_xmlir` 就绪时的 `torch.cuda` 解析和 `fallback=None`。 | 该路径与上一行不同；必须防止以后将 P800 的可运行 shim 再误报为 CUDA fallback。 | 纯 XPU 模拟，对其他后端无影响。 |
| `docs/XPU.md` | 顶部状态、第 1 节运行时表和“自检”后 | 将“没有实测通过率”改为有限实测，记录 CUDA-shim、所需环境变量、正确判据和 gather/SpMV 实测；同时保留 f64/c64 降精度失败与未完成全量交付的事实。 | 旧文档把 `device_type=cuda` 当成必然回退，与 P800 的厂商兼容层矛盾，容易拒绝真实结果或误报通过率。 | 文档，无运行时影响。 |
| `modified/XPU.md` | 本节顶部环境/基线/回传字段及本表 | 记录本轮五个文件、实机命令结果、失败限制及每项修改理由。 | 台账是跨后端合并的唯一回传载体；若不写全，后续移植会遗漏 `_common.__all__` 或误把 shim 当普通 CUDA fallback。 | 文档，无运行时影响。 |

### 完整修改代码块

以下是本轮源代码、测试和运行文档的**全部修改 hunk**。每个文件的修改理由及其他后端
影响已写在上表，不另作省略或重述。

`src/flagsparse/sparse_operations/_common.py`：理由是区分“原生独立设备不可用”的真回退与
`torch_xmlir` 的有效 CUDA-shim；后者在 P800 上实际运行 XPU。

```diff
@@
+def _vendor_accel_available(spec):
+    """Whether a vendor namespace can actually allocate work on this host."""
+    if not spec.torch_namespace:
+        return True
+    namespace = getattr(torch, spec.torch_namespace, None)
+    if namespace is None:
+        return False
+    is_available = getattr(namespace, "is_available", None)
+    if is_available is None:
+        return True
+    try:
+        return bool(is_available())
+    except Exception:
+        # A few vendor extensions do not make this probe safe before their
+        # runtime is initialized; preserve their historical allocation path.
+        return True
+
+
+def _xpu_cuda_shim_present(spec):
+    """Whether FlagTree's Kunlunxin plugin presents XPU through torch.cuda."""
+    if spec is None or spec.name != "xpu":
+        return False
+    try:
+        importlib.import_module("torch_xmlir")
+        return True
+    except Exception:
+        return False
@@
     "_is_maca_runtime",
     "_is_mthreads_runtime",
     "_is_ascend_runtime",
+    "_is_xpu_runtime",
     "_resolve_accel",
@@
     name = _backend_name()
     spec = _BACKEND_SPEC_BY_NAME.get(name)
+    if _xpu_cuda_shim_present(spec):
+        # FlagTree's Kunlunxin stack is torch_xmlir: it rewrites torch.cuda
+        # calls onto XPU. Native torch.xpu is an unsupported stub in this build.
+        return torch.cuda, "cuda"
     # cuda / rocm / metax carry no namespace of their own: they all answer to
     # torch.cuda, which is why the registry leaves torch_namespace unset for them.
-    if spec is not None and spec.torch_namespace and _vendor_plugin_present(spec):
+    if (
+        spec is not None
+        and spec.torch_namespace
+        and _vendor_plugin_present(spec)
+        and _vendor_accel_available(spec)
+    ):
@@
     if spec.torch_namespace == "cuda":
         return None
+    if _xpu_cuda_shim_present(spec):
+        try:
+            if torch.cuda.is_available():
+                return None
+        except Exception:
+            pass
+        return (
+            "backend 'xpu' selected but torch_xmlir's CUDA shim reports no "
+            "available device; falling back to torch.cuda"
+        )
@@
     has_namespace = getattr(torch, spec.torch_namespace, None) is not None
     has_plugin = _vendor_plugin_present(spec) if spec.plugin_modules else True
-    if has_namespace and has_plugin:
+    has_device = _vendor_accel_available(spec) if has_namespace else False
+    if has_namespace and has_plugin and has_device:
         return None
@@
-    else:
+    elif not has_plugin:
         missing = (
             f"torch.{spec.torch_namespace} exists but {plugins} does not import, "
             "so the namespace belongs to some other vendor"
         )
+    else:
+        missing = f"torch.{spec.torch_namespace} reports no available device"
```

`src/flagsparse/sparse_operations/spmv_csr.py`：理由是 XPU lowering 无法编译 `segbin` 的
`associative_scan`，且 `rowpar` 中的 `tl.where` 会触发 XPU scalar/tensor type join 错误。

```diff
@@
     BLOCK_NNZ: tl.constexpr,
     MAX_SEGMENTS: tl.constexpr,
     HAS_BETA: tl.constexpr,
+    XPU_COMPAT: tl.constexpr,
 ):
@@
-        part = tl.where(mask, a * x_vals, 0.0)
+        if XPU_COMPAT:
+            # Masked loads already contribute zero. XPU's UnrollControl pass
+            # rejects the scalar/tensor type join emitted by tl.where here.
+            part = a * x_vals
+        else:
+            part = tl.where(mask, a * x_vals, 0.0)
         acc = acc + tl.sum(part)
@@
-    if _is_rocm_runtime():
+    if _is_rocm_runtime() or _is_xpu_runtime():
+        # XPU lowering does not support segbin's associative scan encoding.
         return "rowpar"
@@
             BLOCK_NNZ=prepared.block_nnz,
             MAX_SEGMENTS=prepared.max_segments,
             HAS_BETA=False,
+            XPU_COMPAT=_is_xpu_runtime(),
         )
```

`tests/ci/test_backend_baseline_policy.py`：理由是分别锁住原生插件缺设备的 fallback 和
`torch_xmlir` shim 的有效 XPU 路径，防止其中一条语义覆盖另一条。

```diff
@@
+def test_fallback_reason_requires_an_available_vendor_device(monkeypatch):
+    """A loaded plugin alone must not bless an unusable device namespace."""
+    from flagsparse.sparse_operations import _common
+
+    class UnavailableXPU:
+        @staticmethod
+        def is_available():
+            return False
+
+    monkeypatch.setattr(_common, "_backend_name", lambda: "xpu")
+    monkeypatch.setattr(_common, "_vendor_plugin_present", lambda spec: True)
+    monkeypatch.setattr(_common, "_xpu_cuda_shim_present", lambda spec: False)
+    monkeypatch.setattr(_common.torch, "xpu", UnavailableXPU())
+
+    assert _common._resolve_accel() == (_common.torch.cuda, "cuda")
+    reason = _common._accel_fallback_reason()
+    assert reason is not None
+    assert "xpu" in reason
+    assert "no available device" in reason
+
+
+def test_torch_xmlir_xpu_uses_the_cuda_shim(monkeypatch):
+    """FlagTree's XPU plugin deliberately exposes its accelerator as CUDA."""
+    from flagsparse.sparse_operations import _common
+
+    monkeypatch.setattr(_common, "_backend_name", lambda: "xpu")
+    monkeypatch.setattr(_common, "_xpu_cuda_shim_present", lambda spec: True)
+    monkeypatch.setattr(_common.torch.cuda, "is_available", lambda: True)
+
+    assert _common._resolve_accel() == (_common.torch.cuda, "cuda")
+    assert _common._accel_fallback_reason() is None
```

`docs/XPU.md`：理由是旧文档将 `device_type=cuda` 错认作回退，且没有反映本轮有限实测。

```diff
@@
-> **状态：有实机脚本，没有实机通过率。** 与 MACA（921 用例实机全绿）、MUSA（1564 用例
-> 实机全绿）不同，昆仑芯这边回传的是**执行路径**（`benchmark/benchmark_xpu.py` 与
-> `benchmark_xpu_variants.py`，后者的 `MEASURED` 表注明每条都在 P800 上单独启动过），
-> **不是一轮成建制的精度/性能结果**。本文第 1、2 节仍来自代码与注册表；下面凡是"期望"
-> 的地方，都要在真机上确认后再当结论用。
+> **状态：已有有限实测，不是完整交付轮。** 2026-09-17 在 P800 的 `torch_xmlir` CUDA-shim
+> 路径上，当前仓库的 `gather` normal 精度中 f16/f32/c32 通过、f64/c64 因厂商 eager
+> 降精度失败；`spmv_csr` fp32 小矩阵独立参考通过（最大绝对误差 `1.43e-6`）。上层仓库的
+> `results/xpu_variants_40.csv` 保存了 40 个变体的历史状态。尚未在当前提交上跑完 30 矩阵的
+> accuracy + performance 交付轮，下面未标明实测的内容不能当作通过率结论。
@@
-| torch 命名空间 | `torch.xpu` |
+| torch 命名空间 | `torch_xmlir` 构建走 `torch.cuda` shim；`torch_xpu` 构建才走原生 `torch.xpu` |
@@
-`accel device type` 若是 `cuda` 而不是 `xpu`，说明插件没找到、整轮会跑在 CUDA 语义下 ——
-和 MUSA 那边同一个坑，见 `MUSA.md` 第 0 节。
+`torch_xmlir` 是 FlagTree 的 CUDA-shim 构建：它将 `torch.cuda` 调用路由到昆仑芯，
+`torch.xpu` 仍是上游 PyTorch 的不支持 stub，`torch.xpu.is_available()` 可为 `False` 且
+`torch.device("xpu")` 会报 `Torch not compiled with XPU enabled`。这是该构建的预期行为，
+不是 CUDA 回退；此时自检的正确结果是 `backend=xpu`、`accel device type=cuda`、
+`fallback reason=None`。同时设置 `FLAGTREE_BACKEND=xpu`、`TRITON_BACKEND=xpu` 和
+`XPU_EVENT_KL3_ENABLE=1`，并使用厂商运行时要求的 `LD_LIBRARY_PATH`。2026-09-17 已以该
+路径在 P800 的设备 1 跑通 `gather float32`（256x256、nnz=4096）。
```

---

## 7. 当前工作区相对 `origin/main` (`87b0aff`) 的完整改动台账

本节是本机当前工作区的**唯一完整清单**；比较基线为已重新拉取的
`origin/main` 提交 `87b0aff`，检查时间为 2026-09-18。下面的 diff 块均是该基线到当前文件的
全部修改 hunk，不省略任何 `+` 或 `-` 代码行。行号是写入时修改后文件中的实际行号；行号会随日后
人工编辑自然变化。

### 机械复现

同目录的 [`XPU.patch`](XPU.patch) 是从该基线生成的完整 Git 补丁，覆盖本台账在内的全部 7 个
受版本控制文件；它不包含 `pytest_results_xpu_*.log` 等未跟踪测试输出。在**干净工作区**且
`HEAD` 为 `87b0aff` 时执行下列命令即可机械重现：

```bash
git apply --check modified/XPU.patch
git apply modified/XPU.patch
```

补丁文件无法包含它自身；请在应用后保留或另行复制 `modified/XPU.patch`。这项自指限制不影响
其余所有受版本控制修改的重现。

| 文件 | 修改后行号 | 修改理由 |
|---|---|---|
| `benchmark/benchmark_xpu.py` | 2、98-105、195-229、273、277 | 真正运行 PyTorch-XPU baseline，先作精度核对，再得到两个耗时和速度比；错误输出也固定标记 baseline。 |
| `run_flagsparse_pytest.py` | 526-536、581-599、2307-2346、2432-2474、2559-2571 | 给 XPU 五算子接入上述 benchmark；目录输入逐矩阵隔离；CUDA-shim 子进程使用逻辑设备 0；从 CSV 行而非退出码取得最终状态。 |
| `src/flagsparse/sparse_operations/_common.py` | 233-261、353、816-835、869-899 | 识别 `torch_xmlir` 的 XPU→CUDA shim；原生厂商命名空间必须实际可用，不能只因插件可导入就判定已就绪。 |
| `src/flagsparse/sparse_operations/spmv_csr.py` | 299-347、1003-1027、1030-1056 | 避免 XPU 的 `segbin` associative-scan lowering；规避其 `tl.where` scalar/tensor type join 编译限制。 |
| `tests/ci/test_backend_baseline_policy.py` | 260-290 | 固化“原生 namespace 无设备应回退”和“torch_xmlir shim 可用不应报回退”两条契约。 |
| `docs/XPU.md` | 9-13、24、63-69、139-148、168-176 | 说明 shim 的正确判据、PyTorch baseline 定义，以及按矩阵隔离和物理/逻辑设备映射。 |
| `modified/XPU.md` | 本节及第 4-5 节修订 | 将此前“baseline 未运行”“库内改动为零”的过期叙述标为已替代，并保存所有当前 diff 与原因。 |

未跟踪的 `pytest_results_xpu_*.log` / `pytest_results_xpu_postpull_ci.log` 是测试输出，不是代码；
它们没有行号或源码修改，不混入以上 7 个文件的实现清单，也没有被删除或重写。

### 7.1 `benchmark/benchmark_xpu.py`（完整改动；修改后第 2、98-105、195-229、273、277 行）

理由：此前脚本仅调用候选闭包，参考闭包从未执行，因此不可能得到精度结论或 baseline 性能。
现在先以同一输出 buffer 的候选与 PyTorch expression 核对；`torch_xmlir` eager 缺少部分设备端
谓词，故比较转到 host。只有满足 dtype 容差的结果才计时，避免把错误内核的速度列为性能结果。

```diff
@@
-"""Execute FlagSparse kernels on Kunlunxin XPU and record latency.
-
-This runner intentionally has no baseline.  It answers only whether the
-FlagSparse kernel can be constructed and executed on the requested device.
-"""
+"""Benchmark FlagSparse kernels against equivalent PyTorch-XPU expressions."""
@@
 def _max_abs_error(actual, expected) -> float:
     difference = (actual.detach().float().cpu() - expected.detach().float().cpu()).abs()
     return float(difference.max().item()) if difference.numel() else 0.0


+def _matches_tolerance(actual, expected, rtol: float, atol: float) -> bool:
+    """Compare on the host because torch_xmlir eager lacks some XPU predicates."""
+
+    actual_host = actual.detach().float().cpu()
+    expected_host = expected.detach().float().cpu()
+    return bool(
+        (actual_host - expected_host).abs().le(atol + rtol * expected_host.abs()).all().item()
+    )
+
+
 def _tolerance(dtype_name: str) -> tuple[float, float]:
@@
-    # Execute the candidate once for validity.  A vendor/PyTorch baseline is
-    # intentionally not required for this XPU smoke-performance pass.
+    # Check the same tensors that will subsequently be timed.  Both closures
+    # overwrite their output buffers, so the warmup loops do not alter this result.
     candidate_result = candidate()
+    baseline_result = baseline()
     _sync(torch)
+    rtol, atol = _tolerance(case.dtype_name)
+    max_abs_err = _max_abs_error(candidate_result, baseline_result)
+    if not _matches_tolerance(candidate_result, baseline_result, rtol, atol):
+        return {
+            "dtype": case.dtype_name,
+            "shape": f"matrix={matrix_name},m={case.rows},n={case.cols},nnz={case.nnz},k={case.dense_cols}",
+            "triton_ms": "",
+            "pytorch_ms": "",
+            "speedup": "",
+            "triton_speedup_vs_pytorch": "",
+            "max_abs_err": max_abs_err,
+            "status": "MISMATCH",
+            "baseline": "pytorch",
+            "reason": f"max_abs_err={max_abs_err:.8g} exceeds rtol={rtol:g}, atol={atol:g}",
+        }
+
     triton_ms = _measure(torch, candidate, warmup, iters)
+    pytorch_ms = _measure(torch, baseline, warmup, iters)
+    speedup = pytorch_ms / triton_ms if triton_ms > 0.0 else ""
     return {
         "dtype": case.dtype_name,
         "shape": f"matrix={matrix_name},m={case.rows},n={case.cols},nnz={case.nnz},k={case.dense_cols}",
         "triton_ms": triton_ms,
-        "pytorch_ms": "",
-        "speedup": "",
-        "max_abs_err": "",
+        "pytorch_ms": pytorch_ms,
+        "speedup": speedup,
+        "triton_speedup_vs_pytorch": speedup,
+        "max_abs_err": max_abs_err,
         "status": "PASS",
-        "baseline": "",
+        "baseline": "pytorch",
     }
@@
-                rows.append({"dtype": dtype_name, "shape": label, "triton_ms": "", "pytorch_ms": "", "speedup": "", "max_abs_err": "", "status": "ERROR", "baseline": "", "reason": str(exc)})
+                rows.append({"dtype": dtype_name, "shape": label, "triton_ms": "", "pytorch_ms": "", "speedup": "", "triton_speedup_vs_pytorch": "", "max_abs_err": "", "status": "ERROR", "baseline": "pytorch", "reason": str(exc)})
@@
-    fields = ("dtype", "shape", "triton_ms", "pytorch_ms", "speedup", "max_abs_err", "status", "baseline", "reason")
+    fields = ("dtype", "shape", "triton_ms", "pytorch_ms", "speedup", "triton_speedup_vs_pytorch", "max_abs_err", "status", "baseline", "reason")
```

### 7.2 `run_flagsparse_pytest.py`（完整改动；修改后第 526-536、581-599、2307-2346、2432-2474、2559-2571 行）

理由：`_base_env()` 会以 `CUDA_VISIBLE_DEVICES=<物理卡号>` 暴露单卡；因此 shim 内只剩
`cuda:0`。此外 `--matrix-dir` 情形已拆成单 `.mtx` 子进程，模板必须传 `--matrix {input}`。
逐行 CSV 的状态才是 benchmark/probe 的 verdict，不能把 exit 0 一律当 PASS。

```diff
@@
-# through benchmark_xpu.py instead of the probe: it executes the kernel and
-# times it. Despite the name it records NO baseline -- the delivered script
-# builds a PyTorch-XPU reference closure per operator but never calls it, so
-# the pytorch_ms / speedup / max_abs_err columns come back empty and `status`
-# is PASS or ERROR. Treat this as latency plus an execution check, not a
-# comparison; the name matches the drop so the next one still diffs cleanly.
+# through benchmark_xpu.py instead of the probe.  The script compares each
+# FlagSparse result with an equivalent PyTorch-XPU expression and reports both
+# latencies and their ratio; it is not a vendor/XDNN sparse-library comparison.
 XPU_BASELINE_OPS: tuple[str, ...] = (
@@
 def _xpu_baseline_command(op: str) -> tuple[str, ...]:
@@
         "--device",
         "{device}",
+        # XPU baselines are isolated one matrix per subprocess by
+        # _run_bell_per_matrix(), so `{input}` is a file on this path.
+        "--matrix",
+        "{input}",
         "--csv-summary",
@@
     if not template:
         return _not_configured(op, "performance", "no performance command mapping")

+    # _base_env exposes exactly one physical accelerator to each child.  XPU's
+    # CUDA shim therefore sees that accelerator as cuda:0, regardless of the
+    # physical id selected by --gpus.
+    script_device = 0 if backend == "xpu" else gpu_id
+
     if (
-        op in PER_MATRIX_PERFORMANCE_OPS
+        (op in PER_MATRIX_PERFORMANCE_OPS or (backend == "xpu" and op in XPU_BASELINE_OPS))
         and benchmark_input is not None
         and benchmark_input.is_dir()
     ):
         return _run_bell_per_matrix(
             project_root=project_root,
             op=op,
             gpu_id=gpu_id,
+            script_device=script_device,
             template=template,
@@
-        device=gpu_id,
+        device=script_device,
@@
 def _run_bell_per_matrix(
@@
     op: str,
     gpu_id: int,
+    script_device: int,
@@
-            device=gpu_id,
+            device=script_device,
@@
-        result["status"] = status
+        if (
+            status == "PASS"
+            and Path(template[0]).name
+            in {"benchmark_ascend_probe.py", "benchmark_xpu.py"}
+        ):
+            result["status"] = _capability_probe_status(rows)
+        else:
+            result["status"] = status
```

### 7.3 已在第 6 节完整列出的 3 个源码/测试文件

为避免同一 diff 在台账中复制两遍，第 6 节的三个代码块就是下列文件相对同一基线的**完整
修改代码**，不是摘录：

- `_common.py`：第 6 节“完整修改代码块”第一个块，理由是 shim 识别与真实设备可用性；修改后行
  233-261、353、816-835、869-899。
- `spmv_csr.py`：第 6 节第二个块，理由是 XPU 编译兼容；修改后行 299-347、1018-1020、1055。
- `test_backend_baseline_policy.py`：第 6 节第三个块，理由是防止上述两种设备解析语义回归；修改后行
  260-290。

### 7.4 `docs/XPU.md`（完整改动；修改后第 9-13、24、63-69、139-148、168-176 行）

理由：使用者需要知道 `accel device type=cuda` 在 `torch_xmlir` 构建中是正确 XPU 路径，且
baseline 已从“仅 smoke latency”变为“精度门禁 + PyTorch expression 性能对比”；真实矩阵需要
逐矩阵进程隔离，runner 的物理/逻辑设备号亦必须明确。

```diff
@@
-> **状态：有实机脚本，没有实机通过率。** 与 MACA（921 用例实机全绿）、MUSA（1564 用例
-> 实机全绿）不同，昆仑芯这边回传的是**执行路径**（`benchmark/benchmark_xpu.py` 与
-> `benchmark_xpu_variants.py`，后者的 `MEASURED` 表注明每条都在 P800 上单独启动过），
-> **不是一轮成建制的精度/性能结果**。本文第 1、2 节仍来自代码与注册表；下面凡是"期望"
-> 的地方，都要在真机上确认后再当结论用。
+> **状态：已有有限实测，不是完整交付轮。** 2026-09-17 在 P800 的 `torch_xmlir` CUDA-shim
+> 路径上，当前仓库的 `gather` normal 精度中 f16/f32/c32 通过、f64/c64 因厂商 eager
+> 降精度失败；`spmv_csr` fp32 小矩阵独立参考通过（最大绝对误差 `1.43e-6`）。上层仓库的
+> `results/xpu_variants_40.csv` 保存了 40 个变体的历史状态。尚未在当前提交上跑完 30 矩阵的
+> accuracy + performance 交付轮，下面未标明实测的内容不能当作通过率结论。
@@
-| torch 命名空间 | `torch.xpu` |
+| torch 命名空间 | `torch_xmlir` 构建走 `torch.cuda` shim；`torch_xpu` 构建才走原生 `torch.xpu` |
@@
-`accel device type` 若是 `cuda` 而不是 `xpu`，说明插件没找到、整轮会跑在 CUDA 语义下 ——
-和 MUSA 那边同一个坑，见 `MUSA.md` 第 0 节。
+`torch_xmlir` 是 FlagTree 的 CUDA-shim 构建：它将 `torch.cuda` 调用路由到昆仑芯，
+`torch.xpu` 仍是上游 PyTorch 的不支持 stub，`torch.xpu.is_available()` 可为 `False` 且
+`torch.device("xpu")` 会报 `Torch not compiled with XPU enabled`。这是该构建的预期行为，
+不是 CUDA 回退；此时自检的正确结果是 `backend=xpu`、`accel device type=cuda`、
+`fallback reason=None`。同时设置 `FLAGTREE_BACKEND=xpu`、`TRITON_BACKEND=xpu` 和
+`XPU_EVENT_KL3_ENABLE=1`，并使用厂商运行时要求的 `LD_LIBRARY_PATH`。2026-09-17 已以该
+路径在 P800 的设备 1 跑通 `gather float32`（256x256、nnz=4096）。
@@
-这五个走 `benchmark/benchmark_xpu.py`，其余算子仍走**能力探测**，按算子报
+这五个走 `benchmark/benchmark_xpu.py`，以等价的 PyTorch-XPU tensor expression 为 baseline，
+先核对输出再分别计时，报告 `triton_ms`、`pytorch_ms`、`max_abs_err` 和
+`triton_speedup_vs_pytorch`（也保留兼容列 `speedup`）。其余算子仍走**能力探测**，按算子报
@@
-> **`benchmark_xpu.py` 给的是时延和"能否执行"，不是加速比。** 它为五个算子各建了一个
-> PyTorch-XPU 参考闭包，但**一次都没调用**，`pytorch_ms` / `speedup` / `max_abs_err`
-> 三列恒为空，`status` 只有 PASS（没抛异常）或 ERROR（带 reason）。脚本 docstring 写明了
-> 这是有意的。要真做对比，把那几个 `baseline()` 调起来即可，结构都在。
+> **这是 PyTorch baseline，不是厂商 sparse-library baseline。** XDNN 没有可替代 cuSPARSE
+> 描述符 API 的通用稀疏接口；因此 `speedup` 表示 FlagSparse 相对 PyTorch-XPU expression 的
+> 比值。输出超出 dtype 容差时该矩阵为 `MISMATCH`，不会参与平均加速比。
@@
-runner 会自动调它，但定位问题时单独跑更快 —— 它一个算子一个进程，报错不会被上层吞掉：
+runner 会自动调它；给目录时，XPU baseline 的每个矩阵都会由 runner 以 `--matrix` 在独立
+进程中执行，因此一张卡住的矩阵不会吞掉同一算子的其余结果。runner 已把 `--gpus 4` 映射为
+子进程的 `CUDA_VISIBLE_DEVICES=4` 和逻辑 `--device 0`；单独调用脚本时则传物理设备号。定位问题时单独跑更快：
```

---

## 8. 2026-09-18：40 个交付变体的实际 accuracy 结果

**来源**：`pytest_results_xpu_p800_matrix30_both_baseline/summary.json` 的 `result` 字段。
这是同一轮 `--phase both --mode normal --delivery-only` 的 accuracy pytest suite；30 个 `.mtx`
输入主要供性能阶段使用，并不替代 accuracy suite 的参数化用例。

`通过/失败` 是该交付变体内的 pytest case 数。失败类型来自每个 case 的实际异常栈：
`assertion` = 已执行但与该测试选择的参考不一致；`dtype_mismatch` = 厂商 eager dtype 降级或类型
不匹配；`uni_sram` = Triton XPU lowering 资源不足；`legalization` = Triton legalization 失败；
`invalid_device_function` = XPU 无对应可执行函数；`cupy_driver` = CuPy CUDA 驱动/运行时不匹配。
此表是逐变体记录，**不按父算子聚合**。

| 交付变体 | 通过/失败 | 失败类型（失败 case 数） |
|---|---:|---|
| gather_c32_int | 4/0 | 无 |
| gather_c64_int | 0/6 | assertion (4), dtype_mismatch (2) |
| gather_f16_int | 4/0 | 无 |
| gather_f32_int | 4/0 | 无 |
| gather_f64_int | 0/4 | assertion (4) |
| scatter_c32_int | 8/0 | 无 |
| scatter_c64_int | 0/8 | assertion (8) |
| scatter_f16_int | 8/0 | 无 |
| scatter_f32_int | 8/0 | 无 |
| scatter_f64_int | 0/8 | assertion (8) |
| sddmm_csr_f32_int_non_non_row | 0/4 | uni_sram (4) |
| sddmm_csr_f64_int_non_non_row | 0/4 | uni_sram (4) |
| spgemm_csr_f32_int_non_non | 0/4 | legalization (4) |
| spgemm_csr_f64_int_non_non | 0/4 | legalization (4) |
| spmm_coo_c32_int_non_non_row | 0/12 | invalid_device_function (12) |
| spmm_coo_c64_int_non_non_row | 0/12 | invalid_device_function (12) |
| spmm_coo_f32_int_non_non_row | 0/12 | invalid_device_function (12) |
| spmm_coo_f64_int_non_non_row | 0/12 | invalid_device_function (12) |
| spmm_csr_c32_int_non_non_row | 0/14 | assertion (14) |
| spmm_csr_c64_int_non_non_row | 0/14 | dtype_mismatch (14) |
| spmm_csr_f32_int_non_non_row | 0/14 | assertion (8), uni_sram (6) |
| spmm_csr_f64_int_non_non_row | 0/14 | dtype_mismatch (8), uni_sram (6) |
| spmv_coo_c32_int_non | 0/18 | uni_sram (18) |
| spmv_coo_c64_int_non | 0/18 | uni_sram (18) |
| spmv_coo_f32_int_non | 0/18 | uni_sram (18) |
| spmv_coo_f64_int_non | 0/18 | uni_sram (18) |
| spmv_csr_c32_int_non | 0/18 | uni_sram (18) |
| spmv_csr_c64_int_non | 0/18 | uni_sram (18) |
| spmv_csr_f32_int_non | 4/14 | assertion (14) |
| spmv_csr_f64_int_non | 3/15 | assertion (15) |
| spsm_csr_f32_int_non_non_row | 0/8 | uni_sram (8) |
| spsm_csr_f64_int_non_non_row | 0/8 | uni_sram (8) |
| spsv_coo_c32_int_non | 0/55 | uni_sram (55) |
| spsv_coo_c64_int_non | 0/59 | uni_sram (59) |
| spsv_coo_f32_int_non | 0/48 | uni_sram (48) |
| spsv_coo_f64_int_non | 0/48 | uni_sram (48) |
| spsv_csr_c32_int_non | 0/78 | uni_sram (58), cupy_driver (20) |
| spsv_csr_c64_int_non | 0/82 | uni_sram (62), cupy_driver (20) |
| spsv_csr_f32_int_non | 0/70 | uni_sram (50), cupy_driver (20) |
| spsv_csr_f64_int_non | 0/74 | uni_sram (54), cupy_driver (20) |

## 9. XPU accuracy reference 切换（2026-09-18）

用户要求将 XPU 所有稀疏算子精度参考统一为 CPU SciPy；性能 baseline 保持设备侧 PyTorch。
实现通过现有 `flagsparse.sparse_operations._common._use_scipy_accuracy_reference()` 守卫，其他
后端默认路径不变。gather/scatter 没有 SciPy 稀疏等价物，继续使用 host indexing/index-copy
oracle；它们的性能脚本仍在 XPU 上计时 PyTorch expression。

| 文件 | 修改位置 | 内容与理由 |
|---|---|---|
| `tests/pytest/test_spmv_csr_accuracy.py` | `test_spmv_csr_matches_dense_reference` | XPU 用 `reference_utils.scipy_csr` + `spmv`；避免 torch dense reference 被误认为 SciPy。 |
| `tests/pytest/test_spmv_coo_accuracy.py` | `test_spmv_coo_matches_dense_reference` | XPU 用 `scipy_coo` + `spmv`。 |
| `tests/pytest/test_spmm_csr_accuracy.py` | 三个 CSR reference tests | XPU 用 `scipy_csr` + `spmm`，包含普通、op 和 opt 路径。 |
| `tests/pytest/test_spmm_coo_accuracy.py` | `_scipy_reference`、主测试 | XPU 用 `scipy_coo` + `spmm`。 |
| `tests/pytest/test_spgemm_sddmm_accuracy.py` | SpGEMM/SDDMM 主测试 | XPU 用 SciPy CSR product 和 CSR sampled-dot reference。 |
| `tests/pytest/test_spsv_csr_accuracy.py` | `_dense_ref_spsv` | XPU 用 `reference_utils.triangular_solve`，覆盖复用该 helper 的 CSR solve tests。 |
| `tests/pytest/test_spsv_coo_accuracy.py` | `_solve_triangular` wrapper | 将 COO accuracy 中所有 dense triangular solve reference 在 XPU 上统一转到 CPU SciPy。 |
| `tests/pytest/test_spsm_accuracy.py` | `_reference` | 将 CSR/COO SpSM 的 triangular solve reference 在 XPU 上统一转到 CPU SciPy。 |
| `run_flagsparse_pytest.py` | `run_accuracy` | XPU accuracy 子进程强制 `FLAGSPARSE_ACCURACY_REFERENCE=scipy`；performance 子进程环境不变。 |

```diff
@@ run_flagsparse_pytest.py::run_accuracy
+    accuracy_env = _base_env(project_root, gpu_id)
+    # XPU correctness is always evaluated against the CPU SciPy oracle.  This is
+    # deliberately independent of the GPU-side PyTorch expression timed by the
+    # performance phase.
+    if os.environ.get("FLAGSPARSE_BACKEND", "").strip().lower() == "xpu":
+        accuracy_env["FLAGSPARSE_ACCURACY_REFERENCE"] = "scipy"
@@
+        env=accuracy_env,
```

完整修改代码如下（均为相对 `origin/main@87b0aff` 的代码块）：

```diff
@@ tests/pytest/test_spmv_csr_accuracy.py
+from flagsparse.sparse_operations import _common as common
+from tests import reference_utils
@@ test_spmv_csr_matches_dense_reference
+    if common._use_scipy_accuracy_reference():
+        matrix = reference_utils.scipy_csr(data, indices, indptr, (M, N), ref_dtype)
+        ref = reference_utils.as_torch(
+            reference_utils.spmv(matrix, x, ref_dtype, op=op), ref_dtype, golden_device()
+        ).to(dtype)
+    else:
+        ref_mat = _apply_dense_op(dense, op)
+        ref = (ref_mat.to(ref_dtype) @ x.to(ref_dtype)).to(dtype)
```

```diff
@@ tests/pytest/test_spmv_coo_accuracy.py
+from flagsparse.sparse_operations import _common as common
+from tests import reference_utils
@@ test_spmv_coo_matches_dense_reference
+    if common._use_scipy_accuracy_reference():
+        matrix = reference_utils.scipy_coo(
+            data, indices[0], indices[1], (M, N), ref_dtype
+        )
+        ref = reference_utils.as_torch(
+            reference_utils.spmv(matrix, x, ref_dtype, op=op), ref_dtype, golden_device()
+        ).to(dtype)
+    else:
+        ref = (_apply_dense_op(dense, op).to(ref_dtype) @ x.to(ref_dtype)).to(dtype)
```

```diff
@@ tests/pytest/test_spmm_csr_accuracy.py
+from tests import reference_utils
@@ each CSR reference test
+    if common._use_scipy_accuracy_reference():
+        ref_dtype = _reference_dtype(dtype)
+        matrix = reference_utils.scipy_csr(
+            _plain_sparse_values(Asp), Asp.col_indices(), Asp.crow_indices(), (M, K), ref_dtype
+        )
+        ref = reference_utils.as_torch(
+            reference_utils.spmm(matrix, B, ref_dtype, op=op), ref_dtype, golden
+        ).to(dtype)
+    else:
+        # Existing torch.sparse/dense reference path is retained.
```

```diff
@@ tests/pytest/test_spmm_coo_accuracy.py
+from flagsparse.sparse_operations import _common as common
+from tests import reference_utils
@@
+def _scipy_reference(Asp, B, op, dtype):
+    indices = Asp.indices()
+    matrix = reference_utils.scipy_coo(
+        Asp.values(), indices[0], indices[1], tuple(Asp.shape), dtype
+    )
+    return reference_utils.as_torch(
+        reference_utils.spmm(matrix, B, dtype, op=op), dtype, golden_device()
+    )
```

```diff
@@ tests/pytest/test_spgemm_sddmm_accuracy.py
+from flagsparse.sparse_operations import _common as common
+from tests import reference_utils
@@ SpGEMM
+    if common._use_scipy_accuracy_reference():
+        left = reference_utils.scipy_csr(
+            A.values(), A.col_indices(), A.crow_indices(), (M, K), dtype
+        )
+        right = reference_utils.scipy_csr(
+            B.values(), B.col_indices(), B.crow_indices(), (K, N), dtype
+        )
+        ref = reference_utils.as_torch(
+            reference_utils.spgemm(left, right), dtype, golden
+        ).to_dense()
@@ SDDMM
+    if common._use_scipy_accuracy_reference():
+        sampled = reference_utils.sddmm_csr_values(indices, indptr, x, y, dtype)
+        ref = torch.as_tensor(sampled, dtype=dtype) * alpha + data.cpu() * beta
```

```diff
@@ tests/pytest/test_spsv_csr_accuracy.py::_dense_ref_spsv
+    if common._use_scipy_accuracy_reference():
+        A_csr = A.to_sparse_csr()
+        matrix = reference_utils.scipy_csr(
+            A_csr.values(), A_csr.col_indices(), A_csr.crow_indices(), A.shape,
+            reference_utils.reference_dtype(A.dtype),
+        )
+        solved = reference_utils.triangular_solve(
+            matrix, b, reference_utils.reference_dtype(b.dtype),
+            lower=lower, unit_diagonal=unit_diagonal, op=op_mode,
+        )
+        return reference_utils.as_torch(
+            solved, reference_utils.reference_dtype(b.dtype), golden
+        ).to(b.dtype)
```

```diff
@@ tests/pytest/test_spsv_coo_accuracy.py
+from flagsparse.sparse_operations import _common as common
+from tests import reference_utils
+
+def _solve_triangular(A, B, *, upper, unitriangular=False):
+    if common._use_scipy_accuracy_reference():
+        matrix = A.to_sparse_csr()
+        scipy_matrix = reference_utils.scipy_csr(
+            matrix.values(), matrix.col_indices(), matrix.crow_indices(), A.shape,
+            reference_utils.reference_dtype(A.dtype),
+        )
+        solved = reference_utils.triangular_solve(
+            scipy_matrix, B.squeeze(-1), reference_utils.reference_dtype(B.dtype),
+            lower=not upper, unit_diagonal=unitriangular,
+        )
+        return reference_utils.as_torch(
+            solved, reference_utils.reference_dtype(B.dtype), golden_device()
+        ).unsqueeze(-1)
+    return torch.linalg.solve_triangular(A, B, upper=upper, unitriangular=unitriangular)
```

```diff
@@ tests/pytest/test_spsm_accuracy.py
+def _reference(A, B, *, lower, unit_diagonal):
+    if common._use_scipy_accuracy_reference():
+        matrix = A.to_sparse_csr()
+        scipy_matrix = reference_utils.scipy_csr(
+            matrix.values(), matrix.col_indices(), matrix.crow_indices(), A.shape,
+            reference_utils.reference_dtype(A.dtype),
+        )
+        solved = reference_utils.triangular_solve(
+            scipy_matrix, B, reference_utils.reference_dtype(B.dtype),
+            lower=lower, unit_diagonal=unit_diagonal,
+        )
+        return reference_utils.as_torch(
+            solved, reference_utils.reference_dtype(B.dtype), golden_device()
+        ).to(B.dtype)
+    return torch.linalg.solve_triangular(
+        A, B, upper=not lower, unitriangular=unit_diagonal
+    )
```

上述切换只改变 correctness reference；没有把 SciPy 放进性能计时窗口。性能仍由
`benchmark/benchmark_xpu.py` 在厂商 XPU (`torch_xmlir` 暴露的 `torch.cuda`) 上测量。

## 10. 合并记录（2026-09-18，合入 `87b0aff` 之上）

`XPU.patch` 的 14 个代码/文档文件已全部应用（`modified/XPU.md` 本身不走补丁，直接取回传版本）。
合并时发现并修正了以下问题，**下次回传请以合并后的版本为基础**：

| 文件 | 问题 | 修正 | 影响范围 |
|---|---|---|---|
| `tests/pytest/test_spsv_csr_accuracy.py`、`test_spsv_coo_accuracy.py` | 新增的 `_solve_triangular()` 在 SciPy 分支返回 reference dtype（float64/complex128），没转回操作数 dtype；`allclose` 报 `Float did not match Double` | 加 `.to(B.dtype)`（`test_spsm_accuracy.py` 的 `_reference()` 本来就这样写） | **所有走 SciPy 参考的后端**：MUSA、MACA、Ascend、XPU。补丁原样合入时 `FLAGSPARSE_ACCURACY_REFERENCE=scipy` 下 105 个失败（合并前 929 全过） |
| `tests/pytest/test_spgemm_sddmm_accuracy.py` | `reference_utils.spgemm()` 返回 SciPy 稀疏矩阵，直接 `np.asarray` 得到 object 数组 | 先 `.toarray()` 再转 torch | 同上 |
| `capi/src/ops/spmv.cpp` | `_spmv_csr_real_kernel` 新增 constexpr `XPU_COMPAT`，C API 的 JIT 签名没跟着加；**所有后端**的 C API 实数 CSR / COO_ALG2 SpMV 报 `number of argument mismatch: Actual(11), Function Definition(12)` | 实数 kernel 签名末尾追加 `False`（C API 不在 XPU 上跑） | 所有后端的 C API |
| `run_flagsparse_pytest.py` `--delivery-only` | （与本补丁同日加入的交付范围收窄）`DELIVERY_BENCHMARK_ARGS` 是 `tests/test_*.py` 的参数，`benchmark_xpu.py` 不认 | 收窄只作用于走通用脚本的后端；XPU、探测类后端、Ascend 的非通用脚本不注入 | XPU / Ascend / gcu / mlu |
| 本文件 | 第 0 节被插进了第 1 节的代码块里（第 1 节的 ``` 未闭合） | 第 0 节移到第 1 节之前，文字未改 | 文档 |

验证（CUDA 机器）：8 个精度文件 SciPy / PyTorch 两种参考各 929 passed；`ctest -R accuracy` 8/8；
`make` CI 链 94 passed。XPU 实机路径（`torch_xmlir` shim、`benchmark_xpu.py`）本机无法运行，未验证。

`modified/XPU.patch` 是相对 `87b0aff` 的补丁，合并后已不能再次应用，**不随仓库提交**。
