# 摩尔线程 MUSA —— 改动台账

**实机环境**：MTT S5000 ×1 / torch 2.7.1 / torch_musa 2.7.1 / muDNN v3105 / Triton 3.6.0。
**基于版本**：合并后的 `fs_merge`（2026-09-16 上午）。
**最近回传**：2026-09-17（第三轮）：修了一个**影响所有非 CUDA 后端**的 C API CMake bug——
`ctest/baseline/CMakeLists.txt` 读了一个从未被赋值的变量名，导致 muSPARSE/hipSPARSE/mcSPARSE
基线探测从来没真正跑过，一律静默退化成 `CUDA -> none`；顺带把 ctest 的 900s per-test 超时改成
可配置。修完之后本机确认 `ctest baseline: MUSA -> /usr/local/musa/lib/libmusparse.so`，C API
构建通过。Python 侧精度（SciPy 参照）+ C API 侧性能（muSPARSE 基线，30 个真实矩阵）正在用
`setsid` 脱离会话跑，见第 10 节。

跑法、能力矩阵、排障记录见 [`docs/MUSA.md`](../docs/MUSA.md)；C API 那层见
[`capi/docs/MUSA.md`](../capi/docs/MUSA.md)。本文只记改了什么。

> **行号口径**：下表的行号是 2026-09-17 对 `HEAD`（`54a3d8d`）现场核对的结果，仅供这一轮合并时
> 快速定位；合并会挪行号，**真正稳定的锚点仍是函数名/常量名**，行号漂了就重新 grep 该函数名。

---

## 1. 库内的后端分支（3 个文件，共 5 处）

| 文件 | 锚点 | 改动 | 为什么 | 其他后端 |
|---|---|---|---|---|
| `sparse_operations/spgemm_csr.py` | `_spgemm_compute()` 内 `tle_hash_safe`（L1141 定义 / L1148 赋值） | `tle_hash_safe = not _is_mthreads_runtime()`，MUSA 上禁用 TLE shared-memory hash/CAS 路径，改走已验证的 ESC | quick 下四个 dtype×索引宽度组合全部卡死，kill 前 stderr 为空；四个 Triton 原语探针单独全过，所以是**算法对原语的组合用法**踩到了同步阻塞，不是能力缺失 | 恒等变换（守卫为 False 时条件不变），CUDA `pytest -m spgemm_csr` 9 passed |
| `sparse_operations/spsv.py` | `_launch_spsv_sell()` 内 `effective_alg_num`（函数 L5440，赋值 L5473-L5480） | SELL 非转置 **ALG2 的实际 launch 降级到 ALG1**；descriptor/API 仍保留用户请求的 `alg_num=2` | `_spsv_sell_slice_kernel_alg2`（持久化 worker + ready flag 轮询）偶发失去前进性，`MUSA_ERROR_LAUNCH_TIMEOUT`，卡死位置固定在 n=64/slice=8 首次 solve | 恒等变换；CUDA 侧 `spsv_sell+spsv_csr+spsv_coo` 846 passed |
| `sparse_operations/spmv_csr.py` | `_spmv_csr_default_backend()`（函数 L997，mthreads 分支 L1017-1019） | 返回 `"segbin"`，"和 Ascend 一样从 CUDA 内核起步" | 没有 MUSA 专用内核，显式声明起点而不是靠 fallthrough | 恒等变换（`return "segbin"` 和 fallthrough 默认值相同） |
| `sparse_operations/spmv_csr.py` | `_normalize_spmv_opt_device_props()` 里的 `"backend"` 字段（函数 L227，mthreads 分支 L250-252） | launch-tuning 用的设备信息字典里，后端名按 `_is_mthreads_runtime()` 报 `"mthreads"` | 结果/调优信息要能自证跑在哪个后端；今天只有 `"hip"` 会改变 bucket tiers，其余名字只是如实上报，为将来分支占位 | 恒等变换：非 hip 分支的调优逻辑不变 |

## 2. 按 dtype（不是按 backend）分支的一处

| 文件 | 锚点 | 改动 | 为什么 |
|---|---|---|---|
| `sparse_operations/_common.py` | `_gather_values()`（L453） | 复数重排改用 `view_as_real` 后 gather；SpGEMM / SpMM / SpMV / SpSM / SpSV 里所有 `data[order]` 路径都接入 | MUSA 的复数 **gather 方向**全挂（`a[order]` / `index_select` / `a[rows, cols]`），scatter 方向正常 |

**分支按 dtype 不按 backend** —— CUDA 上实数走原路径，复数走新路径且结果相同。这是有意的：
缺口是"复数索引内核"，不是"摩尔线程"。

## 3. 加速器抽象（影响所有后端，CUDA 上是恒等变换）

| 文件 | 锚点 | 改动 |
|---|---|---|
| `sparse_operations/_common.py` | `_resolve_accel()`（L773）/ `_ACCEL`（L936）/ `_ACCEL_DEVICE_TYPE`（L937） | torch 子模块与设备类型**成对**返回。只换模块不换设备类型会让 `_is_accel_tensor()` 拒绝掉每一个张量 |
| `sparse_operations/_common.py` | `_use_scipy_accuracy_reference()`（L1043-1065） | CUDA/ROCm 用厂商稀疏库+torch 做双重对照；其余后端（含 MUSA）**一律改用 CPU 上的 SciPy** 做正确性参考，`torch.sparse` 在这些后端上本身就是被测对象而非基准。`FLAGSPARSE_ACCURACY_REFERENCE=auto\|scipy\|torch` 可覆盖 | 见下方第 9 节，这是第 7/8 节两轮 SpGEMM/SpSV CPU-oracle 改动的根策略开关 |
| `sparse_operations/spmm_bell.py` | 输入检查 + timing/meta 路径（`_is_accel_tensor()` 用于 L335/L704/L773；`_ACCEL.*` 用于 L512-519、L1055-1061） | `data.device.type != "cuda"` → `_is_accel_tensor()`；`torch.cuda.Event/synchronize` → `_ACCEL.*`（共 3 处 timing block） |
| `tests/pytest/accuracy_utils.py` | `accelerator_available()`（L25）/ `accelerator_device()`（L64）/ `golden_device()`（L74，`GOLDEN_DEVICE="cpu"` L71） | pytest 套件的守卫、设备字面量（148 处）、以及**参考值恒在 CPU** |
| `tests/pytest/test_spmm_bell_accuracy.py` | `pytestmark`（L25）、`is_mthreads_backend()` 用于生成设备（L112）与参考实现选择（L129-131） | 移除 MUSA 整体 skip，改走 `scipy_sparse_mm()` CPU 参考 |
| **`tests/benchmark_utils.py`（新增）** | `ACCEL`（L30）/ `accelerator_device()`（L37-38） | 直接跑的 benchmark 脚本用的同一对名字；自己把 `src/` 加进 `sys.path`（L24-25），因为脚本是先 import 它、后加 `src/` 的 |
| `tests/*.py` 顶层 benchmark 脚本，实数 **23 个**（此前记成"20 个"，本轮核对更正） | 全文件：`torch.cuda.*` → `ACCEL.*`，`torch.device("cuda")` → `accelerator_device()` | `mtx_fast.py`、`test_alpha_spmm_alg1.py`、`test_gather.py`、`test_scatter.py`、`test_sddmm.py`、`test_spgemm.py`、`test_spmm.py`、`test_spmm_bell.py`、`test_spmm_bsr.py`、`test_spmm_coo.py`、`test_spmm_csc.py`、`test_spmm_csr.py`、`test_spmm_opt.py`、`test_spmm_opt_alg2.py`、`test_spmv.py`、`test_spmv_bsr.py`、`test_spmv_bsr_scipy.py`、`test_spmv_coo.py`、`test_spmv_csc.py`、`test_spmv_opt.py`（以上 20 个经 `tests/benchmark_utils.py`）+ `test_spsm.py`、`test_spsv.py`、`test_spsv_sell.py`（这 3 个直接 `import ... _ACCEL`，不经 benchmark_utils） |

> `torch.cuda` 在摩尔线程上不是"选错了 GPU"，是"没有 GPU"：`torch.cuda.is_available()`
> 为 False，guard 直接跳过整个 benchmark、CSV 空着回来，而后端本身分发得好好的。

## 4. 基线与容差

| 文件 | 锚点 | 改动 | 为什么 |
|---|---|---|---|
| `sparse_operations/_common.py` | `_mthreads_vendor_sparse_library()`（L963-993，调用点 L1114-1115） | 默认返回 `None`（此前是 `torch`） | 实测 MUSA 的 `torch.sparse` **能建 CSR/COO 张量但 sparse matmul 未注册**，四个 dtype 全挂。仍可用 `FLAGSPARSE_MTHREADS_VENDOR` 覆盖 |
| `tests/pytest/test_spmv_coo_accuracy.py`（`_tol()` L90-98）、`tests/pytest/test_spmv_csc_accuracy.py`（`_tol()` L114-122） | 容差 | fp32 / float16 / bfloat16 / complex64 放宽到 `1e-3`；fp64 / complex128 维持 `close_tolerances()` 的严格值 | golden 参考改到 CPU fp64 后不再和内核共享求和顺序，`1.3e-6` 在 160×1024 规模上变成偶发 flake |
| `capi/conf/operators.yaml` | `performance_baseline`（L39） | 加 `mthreads: musparse` | C API 侧基线已接上 |
| `capi/ctest/baseline/musa/baseline.cpp` | 整文件（420 行） | 前缀表 + `_template.inc` → **原生实现** | muSPARSE 4.3.5 的 SpMM/SpSV/SpSM/SpGEMM 是**一个入口加 stage 参数**，不是 cuSPARSE 的拆分函数，前缀表只能产出"看着能编、实则无效"的基线 |
| `capi/tools/matrix_sweep.cpp` | `struct Buf`（L66-81）/ `upload()`（L91-95）/ 计时循环（L164、L170） | `cudaMalloc`/`cudaMemcpy`/`cudaDeviceSynchronize` → `flagsparse::adaptor::*`，不再 include `<cuda_runtime.h>` | 同一个工具要在非 CUDA 后端上编译 |

## 5. 本轮回传清单（12 个文件，全部已合入）

```
flagsparse/tests/{test_gather,test_scatter,test_sddmm,test_spgemm,test_spmm,test_spmm_coo}.py
flagsparse/tests/benchmark_utils.py          ← 第二批补发（第一批漏了，6 个文件当时全 ImportError）
flagsparse/tests/test_spmm_csr.py            ← 回传的是未改版，本仓库按同样 pattern 补了
cwrapper/tools/matrix_sweep.cpp
cwrapper/conf/operators.yaml
cwrapper/ctest/baseline/CMakeLists.txt       ← 仅注释
cwrapper/baseline.cpp                        ← 第三批补发，装到 ctest/baseline/musa/
```

**本机没有 MUSA SDK，`baseline.cpp` 编不了。** 能静态核的都核了：11 个接口函数齐全、
9 个多参数签名与 `baseline.hpp` 逐个类型对上、用到的 28 个仓库符号（adaptor 4 +
`flagsparse.h` 14 + 接口结构体字段 10）全部存在、include 惯例与 `none/` 和 `_template.inc`
一致。**muSPARSE 那一侧用得对不对，只能在实机编译时才知道。**

## 6. 仍开着

- **Python 侧的 muSPARSE 基线未接** —— `FLAGSPARSE_MTHREADS_VENDOR=musparse` 只返回字符串，
  全仓搜不到 `import musparse`，基线列仍是 `N/A`。C API 侧已接上，两者是两回事。
- **`spmm_coo` 复数共轭用例稳定性** —— `conj-int64-complex128-8-4-16` 最近 3 次全过，
  但此前重复测试里出现过一次 `timeout -s KILL`，未宣告修复，`spmm_coo.py` 未改动。
- **`spsv_sell:404` 的 skip 理由含糊**（"MUSA SELL validation differs from CUDA"），没说差在哪。

## 7. 2026-09-17：MUSA 复测与构建修复

| 文件 | 锚点 | 改动 | 为什么 | 其他后端 |
|---|---|---|---|---|
| `tests/pytest/test_spgemm_sddmm_accuracy.py` | `_csr_to_dense()`（L61）/ `test_spgemm_csr_rejects_unsupported_index_and_value_dtypes()`（L108） | FlagSparse 返回的 CSR 三元组回到 CPU 后再物化 dense；非法 dtype 用例在 CPU 构造 CSR、仅将三元组上传 | MTT S5000 上 `SparseCsrmusa::_to_dense` 与 dense `to_sparse_csr` 均未注册；首轮交付扫描中的 8 个 SpGEMM 正确性用例因此在 oracle/input 构造处失败，未进入数值比较 | 恒等变换：CUDA 等后端仍对同一 CSR 值与同一 FlagSparse 结果比较；CPU oracle 不依赖厂商 sparse 实现 |
| `tests/pytest/test_spsv_csr_accuracy.py` | `test_spsv_csr_lower_matches_dense()`（L211）/ `test_spsv_csr_explicit_nnz_balance_route_matches_dense()`（L842）/ `test_spsv_csr_explicit_nnz_balance_analysis_builds_backend_metadata()`（L868）/ `test_spsv_csr_nnz_balance_analysis_workspace_solve_matches_direct()`（L1128） | 少数在 device 上创建 dense 后 `to_sparse_csr()` 的用例改为 CPU 构造；RHS 与 CSR 三元组照常上传 | MUSA 的 `aten::_to_sparse_csr` 未注册，首轮留下 4 个失败；其余 SpSV 用例本已按此 CPU-golden 模式构造 | 恒等变换：输入值、算子调用和容差不变；CUDA/ROCm/MACA 只是从 CPU 上传等价输入 |
| `capi/tools/CMakeLists.txt` | `matrix_sweep` target（L7-14） | 加 `${CMAKE_SOURCE_DIR}/src` include 路径（L12），并将 `${FLAGSPARSE_ADAPTOR_SRC}` 编进工具 target（L8） | `matrix_sweep.cpp` 改用 `flagsparse::adaptor::*` 后，MUSA 构建先找不到 `adaptor/adaptor.hpp`，补 include 后又缺 adaptor 符号；MUSA 实机 `cmake --build` 完整通过并生成工具 | 所有已配置 C API 后端共享该 target；复用与 ctest testsupport 相同的 adaptor source，未分叉实现 |

**`test_spgemm_sddmm_accuracy.py` 里 `_csr_to_dense()`（L61）和 `test_spgemm_csr_rejects_unsupported_index_and_value_dtypes()`（L108）的完整 diff**：

```diff
 def _csr_to_dense(data, indices, indptr, shape):
+    # torch_musa can store CSR tensors but does not register SparseCsrmusa
+    # ``to_dense``. The result is an oracle artifact, so materialize it on the
+    # CPU rather than exercising a second vendor sparse implementation.
     csr = torch.sparse_csr_tensor(
-        indptr,
-        indices,
-        data,
+        indptr.cpu(),
+        indices.cpu(),
+        data.cpu(),
         size=shape,
         dtype=data.dtype,
-        device=data.device,
+        device=golden_device(),
     )
     return csr.to_dense()
```

```diff
 def test_spgemm_csr_rejects_unsupported_index_and_value_dtypes():
     device = accelerator_device()
-    A = _random_csr(8, 10, torch.float32, device)
-    B = _random_csr(10, 6, torch.float32, device)
+    golden = golden_device()
+    A = _random_csr(8, 10, torch.float32, golden)
+    B = _random_csr(10, 6, torch.float32, golden)
     with pytest.raises(TypeError, match="a_indices dtype must be torch.int32"):
         flagsparse_spgemm_csr(
-            A.values(),
-            A.col_indices().to(torch.int64),
-            A.crow_indices(),
+            A.values().to(device),
+            A.col_indices().to(torch.int64).to(device),
+            A.crow_indices().to(device),
             (8, 10),
-            B.values(),
-            B.col_indices().to(torch.int32),
-            B.crow_indices(),
+            B.values().to(device),
+            B.col_indices().to(torch.int32).to(device),
+            B.crow_indices().to(device),
             (10, 6),
         )
     # 下面 A_complex 分支是同一个模式（A_complex 也从 device 改成 golden，
     # 调用处三元组逐个 .to(device)），不重复贴出。
```

**`test_spsv_csr_lower_matches_dense()`（L211）完整 diff**——四个改动里唯一一个除了"device 换 golden"
还顺带把内联计算换成了共享 helper 的（其余三个 nnz_balance 用例只是把 `device` 换成 `golden`，
`x_ref`/后续调用不变，没有额外贴出）：

```diff
 def test_spsv_csr_lower_matches_dense(n, dtype):
-    # Keep the original baseline test case untouched in semantics.
     device = accelerator_device()
-    base = torch.tril(torch.randn(n, n, dtype=dtype, device=device))
-    eye = torch.eye(n, dtype=dtype, device=device)
+    golden = golden_device()
+    base = torch.tril(torch.randn(n, n, dtype=dtype, device=golden))
+    eye = torch.eye(n, dtype=dtype, device=golden)
     A = base + eye * (float(n) * 0.5 + 2.0)
-    b = torch.randn(n, dtype=dtype, device=device)
-    x_ref = torch.linalg.solve_triangular(
-        A, b.to(A.device).unsqueeze(-1), upper=False
-    ).squeeze(-1)
+    b = torch.randn(n, dtype=dtype, device=golden).to(device)
+    x_ref = _dense_ref_spsv(A, b, lower=True)
     Asp = A.to_sparse_csr()
```

`_dense_ref_spsv()` 是文件里已经存在的共享 helper（不是这轮新增的），docstring 自带一句
"``torch.linalg`` is not dependable on MUSA, so the reference must not run there"——这处改动
本质是把一个还在手写内联 `torch.linalg.solve_triangular` 的用例，改成和文件里其它用例一样调用
这个共享 helper，顺带把 `A`/`eye` 的构造设备从 `device` 换成 `golden`。

**`capi/tools/CMakeLists.txt` 完整 diff**（4 行改动，全文贴出）：

```diff
 add_executable(matrix_sweep matrix_sweep.cpp)
+target_sources(matrix_sweep PRIVATE ${FLAGSPARSE_ADAPTOR_SRC})
 target_include_directories(matrix_sweep PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_SOURCE_DIR}/include
+    ${CMAKE_SOURCE_DIR}/src
     ${FLAGSPARSE_ADAPTOR_INCLUDES})
 target_link_libraries(matrix_sweep PRIVATE flagsparse ${FLAGSPARSE_ADAPTOR_LIBS})
```

实测：`tools/probe_accel_capabilities.py` 显示四种交付 dtype 的 Triton load/store、atomic、scan 与
FlagSparse CSR SpMV 均通过；完整 `--delivery-only --phase accuracy --mode normal` 为 **36 Passed /
4 Skipped / 0 Failed**。4 个 skip 仅是 MUSA 没有 CuPy/cuSPARSE 时的 SpSV CSR 专用对照用例
（**见第 8 节：这个结论过粗，已被订正**）。
C API 以 `BACKEND=MUSA` 配置、构建通过，`ctest -N` 发现 33 个测试目标。

## 8. 2026-09-17（第二轮）：SpSV cuSPARSE 对照用例改接 SciPy fallback；性能超时复测

| 文件 | 锚点 | 改动 | 为什么 | 其他后端 |
|---|---|---|---|---|
| `tests/pytest/test_spsv_csr_accuracy.py` | 新增 `_cusparse_or_scipy_ref_spsv()`（L184-201）；`test_spsv_csr_matches_cusparse_non_trans()`（L1423）/ `_transpose_family()`（L1457）/ `_upper_non_trans()`（L1562）/ `_upper_transpose_family()`（L1596） | 去掉 `cp is None` 的硬 `skipif`：CuPy 可用时仍走原 cuSPARSE 对照，不可用时 fallback 到 `accuracy_utils.scipy_triangular_solve()` | 上一轮"4 个 skip 是因为没有 CuPy/cuSPARSE"记得过粗：`scipy_triangular_solve()`（`tests/pytest/accuracy_utils.py:217`）是早就写好但从未被调用的函数，docstring 明写"for MUSA accuracy tests"，只是没接到这 4 组测试里；接上后不需要新增测试项，4 个 delivery 变体从 accuracy=Skipped 变成 Passed | 恒等变换：CUDA 上 `cp` 可用，仍走原 cuSPARSE 分支，行为不变；ROCm/MACA 等其他无 CuPy 的后端同样受益 |

**新增的 fallback helper**（L184-201，紧接在既有的 `_cupy_ref_spsv()` 之后）：

```python
def _cusparse_or_scipy_ref_spsv(A, A_cp, b, *, lower, op_mode="NON", unit_diagonal=False):
    """cuSPARSE reference via CuPy when available, else SciPy on the CPU golden ``A``.

    CuPy only ever talks to a real NVIDIA GPU through cuSPARSE, so it is absent on
    every non-CUDA backend (MUSA, ROCm, MACA, ...). ``scipy_triangular_solve``
    already exists for exactly this case but sat unused, so these tests
    unconditionally skipped on non-CUDA machines instead of falling back to it.
    """
    if cp is not None and cpx_sparse is not None and cpx_spsolve_triangular is not None:
        return _cupy_ref_spsv(
            _cupy_apply_op(A_cp, op_mode),
            b,
            lower=_effective_lower_for_op(lower, op_mode),
            unit_diagonal=unit_diagonal,
        )
    return scipy_triangular_solve(
        A, b, lower=lower, unit_diagonal=unit_diagonal, op=op_mode.lower()
    )
```

`op_mode="NON"` 对 `_cupy_apply_op()`/`_effective_lower_for_op()` 都是直通（两个既有函数本来就把
非 `"TRANS"/"CONJ"` 的值当恒等），所以四个测试（non_trans / transpose_family / upper_non_trans /
upper_transpose_family）能共用同一个 helper，不需要按 op_mode 再分支。

**四处调用点改动是同一个模式，贴 `test_spsv_csr_matches_cusparse_non_trans()`（L1423）一处**：

```diff
-@pytest.mark.spsv
-@pytest.mark.skipif(
-    cp is None or cpx_sparse is None or cpx_spsolve_triangular is None,
-    reason="CuPy/cuSPARSE required",
-)
-@pytest.mark.parametrize("n", SPSV_N)
+@pytest.mark.spsv
+@pytest.mark.parametrize("n", SPSV_N)
 @pytest.mark.parametrize("dtype", NON_TRANS_DTYPES, ids=_dtype_id)
 def test_spsv_csr_matches_cusparse_non_trans(n, dtype):
     ...
-    x_non_ref = _cupy_ref_spsv(A_cp, b, lower=True, unit_diagonal=False)
+    x_non_ref = _cusparse_or_scipy_ref_spsv(A, A_cp, b, lower=True, unit_diagonal=False)
     rtol, atol = _tol(dtype)
-    assert torch.allclose(x_non, x_non_ref, rtol=rtol, atol=atol)
+    assert torch.allclose(x_non.to(x_non_ref.device), x_non_ref, rtol=rtol, atol=atol)
```

最后一行的 `.to(x_non_ref.device)` 是必须加的：CuPy 分支的参考值和 `x_non` 同在 accelerator 上，
`.to()` 是无操作；SciPy 分支的参考值在 CPU 上，不加这个转换会在 MUSA 上直接抛
`RuntimeError: expected all tensors on the same device`。`transpose_family`/`upper_non_trans`/
`upper_transpose_family` 三处是同一处改动，只是变量名和 `lower`/`op_mode` 取值不同，import 块
（文件顶部）新增一行 `scipy_triangular_solve` 到 `from tests.pytest.accuracy_utils import (...)`。

实测（MTT S5000）：`pytest tests/pytest/test_spsv_csr_accuracy.py -q` 从 337 passed/4 skipped 变为
**341 passed / 0 skipped**；`python3 -m pytest tests/ci -q` 仍是 85 passed / 3 skipped / 2 failed，
两个失败是 `test_installed_wheel`/`test_every_out_of_tree_backend_reports_its_own_fallback` 在
`PYTHONPATH=src`（非 pip install）下的已知环境态（见 `prompt.md` 第 8 节），与本次改动无关。

性能侧复查：上一轮 `summary.json` 里的 17 个 "NotFound" 并非都是"该后端不支持"——`_delivery_performance_phase`
只要某 dtype 在 CSV 里 0 行数据就统一归成 NotFound，把"跑崩了"和"跑不完"和"真不支持"混在一起报。拆开看：
`spmm_coo`（4 个 dtype）是真实的 `aten::addmm` 未在 `Sparsemusa` 注册，第 1/30 个矩阵就崩、0 行数据；
`spgemm_csr_f64`/`sddmm_csr`/`spsv_coo`/`spsv_csr` 是 `--timeout 900` 在 30 个真实矩阵的完整 dtype 网格上
跑不完，被 SIGKILL 后要么 0 行、要么最后一个 dtype 还没轮到；`spsm_csr` 另有一个独立的 `'N/A_ms'` 格式化异常
（benchmark 脚本自身的 bug，15 个矩阵里 11 个直接 ERROR，和 MUSA 硬件无关）。`--timeout` 已按要求从 900 提到
3600 重新跑 `--delivery-only --phase both`（结果目录
`pytest_results_musa_delivery_both_matrix30_timeout3600_20260917`），跑完后本节会补上实测结果；
`spmm_coo` 的未注册内核和 `spsm_csr` 的格式化 bug 不会因为加大 timeout 而消失，仍需单独修。

## 9. 2026-09-17（第三轮）：逐行核对全部 MUSA 专用分支（为多后端合并准备）

按 `_is_mthreads_runtime()` / `is_mthreads_backend()` / `_IS_MTHREADS_RUNTIME` 三个名字对全仓
`git grep` 了一遍（`capi/deps/libtriton_jit/` 第三方 vendored 代码不算，那是上游自带的 MUSA 后端
支持，不是本仓库改的）。第 1-8 节漏了两处真实的行为分支，补在这里；其余命中全部已在第 1-8 节
覆盖，只是本轮顺手把行号也核对更新了。

| 文件 | 锚点 | 改动 | 为什么 | 其他后端 |
|---|---|---|---|---|
| `sparse_operations/_common.py` | `_cupy_cusparse_spmv_skip_reason()`（L1989-2000） | MUSA/Ascend 命中时直接返回 `"CuPy/cuSPARSE is not applicable on the {backend} backend"`，不再走"`cp is None` 才判缺失"的通用判断 | 两个后端都**没有** CuPy，但各自有自己的基线（MUSA 是 `torch.sparse`，Ascend 是 ops-sparse），报"不可用"而不是"没装"更准确，避免误导成"装个 CuPy 就能测" | 恒等变换：CUDA/ROCm/MACA 走后面的 `cp is None` 通用分支，不受影响 |
| `tests/test_spsv.py` | `_active_spsv_alg_num_to_solve_kind()`（L92-97） | MUSA（和 Ascend 一起）在裸 benchmark 脚本里只跑 `{1: "csr_cw"}` 一个 alg_num；CUDA 有 5 个（`csr_cw`/`csr_cw_levelschd`/`csr_roc`/`csr_smblk`/`csr_nnz_balance`），ROCm 有 3 个 | 和第 1 节 `effective_alg_num` 是同一个约束在 benchmark 侧的镜像：MUSA 目前只验证了 ALG1 对应的 SELL scalar-row 路径，其余 alg_num 在这台机器上没有对应的稳定实现可比 | 这个约束有 CI 契约测试兜底：`tests/ci/test_spsv_backend_contract.py::test_spsv_preserves_update_non_rocm_profiles_and_public_sell_api` 断言 `"_is_mthreads_runtime()"` 必须出现在 `SPSV_BENCHMARK_SOURCE` 里，删掉这个分支会直接把 CI 跑红 |

**确认过、判定"不需要单独立项"的名单**（避免下一个后端合并时重复排查）：

- `_common.py` 的 `_BackendSpec("mthreads", ...)` 注册表项（L115-116）、`_detect_mthreads_runtime()`（L269-293）、
  `_is_mthreads_runtime()`（L716-718）、`_runtime_backend_label()` 里的 `"mthreads": "Moore Threads/MUSA"`（L1131）——
  这些是**每个后端对称拥有**的注册/探测/打标签模板代码，不是针对 MUSA 的特殊行为。
- `src/flagsparse/sparse_operations/backends/musa/__init__.py` —— 确认是空文件（按规矩应该空），
  `tests/ci/test_backend_module_dispatch.py::test_backend_directories_are_empty` 兜底。
- `capi/` 下所有提到 `MUSA`/`musa` 的 CMake 文件（`capi/CMakeLists.txt`、`capi/ctest/CMakeLists.txt`、
  `capi/src/adaptor/CMakeLists.txt` 等）—— 都是把 MUSA 和 CUDA/MACA/HCU/... 对称列在同一份多后端
  构建脚本里，不是"为 MUSA 加的例外"。`capi/src/adaptor/backend/musa/adaptor.cpp` 本身是每个 C API
  后端都有的那"一个文件"（见 `prompt.md` 第 4 条第 4 级），结构和 `backend/none/` 对称，已在第 4/5 节
  记过，不重复立项。
- `tests/ci/test_backend_baseline_policy.py` / `test_backend_test_profiles.py` / `test_backend_module_dispatch.py` ——
  参数化契约测试，`mthreads`/`musa` 只是表格里对称的一行，不是专门为 MUSA 写的分支。
- `sparse_operations/sddmm_csr.py`、`sparse_operations/_dispatch.py`、`tests/test_spmv.py`、
  `tests/reference_utils.py`、`tools/probe_accel_capabilities.py`、`tools/bisect_hang.py`、
  `run_flagsparse_pytest.py`、`tests/pytest/test_alpha_spmm_alg1_accuracy.py`、
  `test_gather_scatter_accuracy.py`、`test_spmm_coo_accuracy.py`、`test_spmm_csr_accuracy.py`、
  `test_spmm_csr_opt_alg2_accuracy.py`、`test_spsm_accuracy.py`、`test_spsv_sell_accuracy.py` ——
  提到 MUSA 的地方都只是注释/文档字符串，解释第 3 节 `golden_device()` 或第 4 节
  `_use_scipy_accuracy_reference()` 这两个通用开关为什么要这样设，本身不含独立的 MUSA 专属分支。
- `sparse_operations/spsm.py` —— 全文件 grep 不到 `mthreads`/`musa`，MUSA 上走的是和 CUDA 完全一样
  的路径，没有需要记录的差异。

## 10. 2026-09-17（第三轮）：C API 侧 muSPARSE 基线从来没跑过——CMake 变量名写错了

排查"能不能用 C API 方式和 muSPARSE 比性能"时发现的。**这两处改动不是 MUSA 专属，是共享 C API
构建脚本的 bug fix，CUDA/ROCm/MACA/Ascend/XPU 都受影响**，按仓库惯例仍记在这里，因为是在
MUSA 这台机器上排查出来、验证过的；合并时请对应通知其他后端的负责人。

| 文件 | 锚点 | 改动 | 为什么 | 其他后端 |
|---|---|---|---|---|
| `capi/ctest/baseline/CMakeLists.txt` | `# ------------------ resolve` 段，`string(TOUPPER ...)` 那行（改前 L112，改后 L117，中间插了 5 行注释） | `string(TOUPPER "${FLAGSPARSE_BACKEND}" _bl_backend)` → `string(TOUPPER "${BACKEND}" _bl_backend)` | `capi/CMakeLists.txt:23` 定义的缓存变量是 `BACKEND`（`-DBACKEND=MUSA`），`FLAGSPARSE_BACKEND` 只是 Python 侧的环境变量名（值形如 `mthreads`），CMake 变量 `FLAGSPARSE_BACKEND` **从未被赋值**。`_bl_backend` 因此永远走 `if(NOT _bl_backend) set(_bl_backend "CUDA")` 这条 fallback，不管 `-DBACKEND=` 传的是什么，`ctest baseline:` 探测**永远**打印 `CUDA -> none`（本机没装 CUDA）。本机验证：改之前 `-DBACKEND=MUSA` 配置出来仍是 `ctest baseline: CUDA -> none`；改之后是 `ctest baseline: MUSA -> /usr/local/musa/lib/libmusparse.so`，`libmusparse.so`/`musparse.h` 在 `/usr/local/musa` 下确实存在（`FLAGSPARSE_MUSA_SPARSE_LIB`/`_HEADER` 命名对得上），C API 全量构建（73/73 target）通过 | **不是恒等变换，是修复**：CUDA 构建"碰巧"没受影响（fallback 默认值本来就是 `"CUDA"`），但 ROCm/MACA/Ascend/XPU 在这个 bug 存在期间，`-DBACKEND=` 传什么都无所谓，baseline 探测全部退化成 `CUDA -> none`，report 里的 speedup 列因此全空，且原因写成"没有 CUDA"而不是各自真正的原因（没装 hipSPARSE/mcSPARSE，或者本来就没有描述符 API）。这几个后端如果也想拿到真实基线，直接受益于这一行修复 |
| `capi/ctest/CMakeLists.txt` | `flagsparse_add_test()` 里的 `TIMEOUT`；新增 cache 变量 `FLAGSPARSE_CTEST_TIMEOUT`（L108，默认仍是 `"900"`），用在 `set_tests_properties()` 里（L130） | 硬编码的 `TIMEOUT 900` 改成 `TIMEOUT ${FLAGSPARSE_CTEST_TIMEOUT}`，可用 `-DFLAGSPARSE_CTEST_TIMEOUT=3600` 覆盖 | 和 Python 侧 `--timeout` 是同一个问题：合成小矩阵 900s 够用，30 个真实矩阵的完整 dtype 网格不够，之前只能改这个文件里的字面量 | 默认值不变，不传这个 cache 变量时所有后端行为不变；这是加开关不是改行为 |

**`capi/ctest/baseline/CMakeLists.txt`，改前后对比**（原第 111-114 行，改后第 110-121 行）：

```diff
 # ------------------------------------------------------------------ resolve
-# FLAGSPARSE_BACKEND is set by the top-level build; default CUDA.
-string(TOUPPER "${FLAGSPARSE_BACKEND}" _bl_backend)
+# BACKEND is the top-level cache variable (capi/CMakeLists.txt:23); default CUDA.
+# NOT FLAGSPARSE_BACKEND -- that name is the Python-side env var (values like
+# "mthreads"), a different variable with different spellings. Reading it here
+# silently no-ops: it is always empty, so _bl_backend always fell through to
+# CUDA regardless of -DBACKEND=..., and every non-CUDA build got "CUDA -> none"
+# instead of probing its own vendor library.
+string(TOUPPER "${BACKEND}" _bl_backend)
 if(NOT _bl_backend)
     set(_bl_backend "CUDA")
 endif()
```

**`capi/ctest/CMakeLists.txt`，两处改动**（新增 cache 变量 L102-108，改用它 L130）：

```diff
 target_link_libraries(flagsparse_testsupport
     PUBLIC flagsparse flagsparse_ctest_baseline ${FLAGSPARSE_ADAPTOR_LIBS})

+# Per-test wall-clock budget. 900s is enough for the synthetic/small default
+# corpus but not for a real matrix corpus on a slow route (SpSV/SpSM-style
+# triangular solves, or a benchmark sweeping dtype x index x 30 real .mtx
+# files) -- see FLAGSPARSE_MATRIX_DIR below. Override per invocation with
+# -DFLAGSPARSE_CTEST_TIMEOUT=<seconds> rather than editing this file.
+set(FLAGSPARSE_CTEST_TIMEOUT "900" CACHE STRING
+    "Per-test ctest TIMEOUT in seconds, for both accuracy and benchmark tests.")
+
 # test_<operator>.cpp, one per API, in both directories (spec §6.2).
 ...
     add_test(NAME "${phase}.${op}" COMMAND ${target})
     set_tests_properties("${phase}.${op}" PROPERTIES
-        TIMEOUT 900
+        TIMEOUT ${FLAGSPARSE_CTEST_TIMEOUT}
         LABELS "capi;${FLAGSPARSE_CTEST_PROFILE};${phase}")
```

**发现顺序**（供复现）：`cmake -S . -B build -G Ninja -DBACKEND=MUSA -DMUSA_HOME=/usr/local/musa`
配置成功，但日志只打印 `ctest baseline: CUDA -> none`，没有任何 `MUSA` 字样；`grep -rn
"set(FLAGSPARSE_BACKEND" capi/` 全仓库零命中，才发现这个变量从未被赋值过。改完一行，删掉
`build/CMakeCache.txt` 重新配置，日志变成 `ctest baseline: MUSA -> /usr/local/musa/lib/libmusparse.so`。

**本机验证到"构建+配置成功、链接到真实 `libmusparse.so`"为止**，没有验证 muSPARSE 那一侧的
函数调用是不是真的按 stage 参数正确工作——`ctest -R benchmark` 的实测结果见下方跑起来的任务，
完成后回填。

### 正在跑：Python 精度（SciPy）+ C API 性能（muSPARSE）

```bash
setsid nohup /bin/bash scratchpad_run_musa_full.sh \
  > musa_full_run_20260917.log 2>&1 < /dev/null &
disown
```

`scratchpad_run_musa_full.sh`（仓库根目录，未提交，跑完可删）依次跑：

1. `run_flagsparse_pytest.py --phase accuracy --mode normal --delivery-only --timeout 3600` ——
   Python 侧 40 变体精度，参照是 SciPy（`_use_scipy_accuracy_reference()` 默认行为，见第 3/9 节）；
   结果目录 `pytest_results_musa_delivery_accuracy_scipy_20260917`。
2. `ctest --test-dir capi/build -R accuracy`（`--timeout` 3600，第 10 节新加的开关）——发 benchmark
   之前的正确性门槛。
3. `FLAGSPARSE_MATRIX_DIR=/root/gcx/matrix ctest --test-dir capi/build -R benchmark` —— C API 性能，
   基线是刚接上的 `libmusparse.so`；产物在 `capi/build/bench_musparse_20260917`。
4. `tools/report.py` + `tools/write_summary.py` + `tools/check_manifest.py` —— 汇总成
   `summary.json`（40 变体 schema）。

用 `setsid` + 重定向 stdin/stdout/stderr + `disown` 启动，`ps` 确认 `PPID=1`、独立 `SID`、无
controlling TTY——**这个任务不挂在当前会话下，会话结束/断开它继续跑**，跑完看
`musa_full_run_20260917.log` 和上面两个结果目录/文件即可，本节完成后会回填真实的
muSPARSE 加速比数字。

## 11. 2026-09-17（第三轮）：`docs/MUSA.md` 使用说明订正

不是代码改动，是这份使用文档本身有两处过时/易误导，顺手改了，一并记录文件名（这份台账的
惯例是"改了哪个文件"，文档也算）：

| 文件 | 锚点 | 改动 | 为什么 |
|---|---|---|---|
| `docs/MUSA.md` | 第 4 节「运行测试」 | 原来给的示例命令是 `--mode quick`，没提 `--timeout`；改成把 quick 命令标注为"仅冒烟测试"，交付命令统一指向第 0.5 节的 `--delivery-only --mode normal`，并把 `--timeout` 900 不够、单 GPU 顺序跑（不是并行）、总时长按算子组耗时求和这几条今天验证过的坑写清楚 | `--mode quick` 会漏跑四成用例还跳过两个历史问题用例（见 `prompt.md` 第 3 节），照抄这段示例会产出不算数的"验收"结果 |
| `docs/MUSA.md` | 第 6 节，`spsv_csr` 的 80 个 skip 那段 | 原文断言"CuPy/cuSPARSE 比对用例在 MUSA 上本就不适用，属正常"；改成标注这个结论已被本轮第 8 节的 SciPy fallback 推翻，并加一条教训：见到 `cp is None → skip` 这类硬编码，先查 `accuracy_utils.py` 有没有现成但没接线的 fallback，不要直接认定"这个后端没法测" | 这条结论是 2026-09-13 写的，当时是对的（`scipy_triangular_solve()` 那时也没接），但没人在它变成过时结论时回来更正，是本文档自己的一个"没跑过就写没跑过"反例 |
| `capi/docs/MUSA.md` | 「跑测试」示例命令；「这个后端上预期会看到什么」的 `speedup` 列那条；「跑不起来时按这个顺序查」第 2 条；「基线：muSPARSE 4.3.5」节尾 | 示例命令加 `-DFLAGSPARSE_CTEST_TIMEOUT=3600` 并解释这个新开关；把"`speedup` 列大概率是空的（musparse not found）"改成"那是第 10 节的 CMake bug，不是没装 muSPARSE，本机确认 `libmusparse.so` 真实存在"；把"`ctest baseline: MUSA -> none` → 正常"改成"不正常，去查原因"（这条曾经写成正常，正是因为 bug 存在期间它永远打印成 `CUDA -> none` 而不会真的打印 `MUSA -> none`）；基线节尾加一段说明选择逻辑本身此前从未被真正执行过 | C API 这份文档是本机排障时会先翻的第一手资料，前两轮的悲观结论（"speedup 列大概率是空的"、"MUSA -> none 是正常的"）都是建立在这个从未被触发过的 CMake bug 上，不订正会一直把人导向"没救了"的错误方向 |

## 12. 2026-09-17（第四轮）：sddmm_csr 精度漏修的姊妹用例；新增精度/性能分源 runner

| 文件 | 锚点 | 改动 | 为什么 | 其他后端 |
|---|---|---|---|---|
| `tests/pytest/test_spgemm_sddmm_accuracy.py` | `test_sddmm_csr_rejects_unsupported_index_and_value_dtypes()` | 和第 7 节修过的 `test_spgemm_csr_rejects_unsupported_index_and_value_dtypes()` 是同一个 bug、同一个文件：`_random_csr(..., device)` 在 `device`（MUSA）上直接 `.to_sparse_csr()`。改成在 `golden_device()` 上构造 `pattern`，只把 `data`/`indptr`/`indices` 三元组 `.to(device)` | 第 7 节只改了 spgemm 那一个，sddmm 这个姊妹用例（同文件、同一个 `_random_csr` 调用模式）当时漏改，这次跑交付精度时才在 `sddmm_csr: accuracy=FAIL` 里暴露出来；`aten::_to_sparse_csr` 在 musa 后端未注册是同一个根因 | 恒等变换：CUDA 等后端仍在自己的 device 上构造，不受影响。`test_spgemm_sddmm_accuracy.py` 从 17 passed/1 failed 变成 **18/18 全过** |
| **`run_flagsparse_split_delivery.py`（新增，仓库根目录）** | 全文件 | 精度/性能分源的组合 runner：精度调用 `run_flagsparse_pytest.py --phase accuracy --delivery-only`（SciPy 参照，Python 侧默认行为不变）；性能改跑 C API 的 `ctest -R benchmark`（muSPARSE 基线，见第 10 节）+ `capi/tools/write_summary.py`；最后把两份 `summary.json` 按变体 id 浅合并成 `summary_split.json`（精度取 pytest 那份，性能取 C API 那份，**不采用** C API 自己内嵌的逐矩阵 accuracy——那是给 speedup 计不计数用的内部门槛，不是交付精度结果，见 `capi/ctest/common.cpp` 的 `write_accuracy_json`/`BenchReport::add`） | MUSA（以及其他没有 Python 侧厂商稀疏库的后端）用 `run_flagsparse_pytest.py --phase performance` 测不出加速比——`_mthreads_vendor_sparse_library()` 恒返回 `None`，分母是空的；C API side 已经有能用的 muSPARSE 基线（第 10 节修完 CMake bug 之后），用它测性能更诚实。参数命名刻意和 `run_flagsparse_pytest.py` 保持一致（`--mode`/`--gpus`/`--results-dir`/`--benchmark-input`/`--timeout`/`--strict`），没有照搬 `--benchmark-warmup`/`--benchmark-iters`——C API 的 benchmark 硬编码了自己的 `kWarmup=10`/`kIters=100`（`capi/ctest/common.hpp`），没有对应旋钮，加这两个参数会造成"看着能调、其实没用"的假象 | 通用工具，默认 `--backend mthreads`，可通过 `--backend`/内部的 `CAPI_BACKEND_BY_PYTEST_BACKEND` 表换成其他已有 C API 基线的后端；冒烟测试（quick 模式，仅精度半边）已过：40/40 变体 `accuracy: Passed` |

**实测**：`python3 run_flagsparse_split_delivery.py --mode quick --skip-performance --results-dir pytest_results_mthreads_split_smoketest` 跑通，`summary.json` 40 个变体 `accuracy` 全 `Passed`（含刚修好的 `sddmm_csr`）。性能半边（C API `ctest -R benchmark`）的命令构造用 monkeypatch 做了 dry-run 核对（cmake/ctest/write_summary.py/check_manifest.py 四条命令、`FLAGSPARSE_MATRIX_DIR`/`FLAGSPARSE_BENCH_OUT` 两个环境变量都对），**没有跑通整条性能链路**——当时 `capi/build` 正被另一个后台任务（第 10 节末尾那个 `scratchpad_run_musa_full.sh`）占用，没有抢着跑；等那个任务让出来后要单独跑一次 `--skip-accuracy` 确认到底。

**事故记录：冒烟测试和后台性能任务撞了同一张卡**。跑上面那条冒烟测试命令时，没有先确认 `scratchpad_run_musa_full.sh`（第 10 节末尾发起的那个 setsid 后台任务）还占着 GPU——`mthreads-gmi` 显示本机只有 **1 张 MTT S5000**，不是多卡。冒烟测试的 pytest 精度子进程（21:37-21:41）和当时正在跑的 `benchmark.spsm`（从 21:13 跑到 21:41 之后，单个 GTest 循环 30 个矩阵）在同一张卡上并发了约 4 分钟。

- **精度结果不受影响**：正确性不依赖独占 GPU，`accuracy: Passed` 这 40 条可信。
- **`benchmark.spsm` 这一轮的计时数字要标成不可信**：GPU 争用期间正在跑的那几个矩阵（具体是 30 个里的哪几个，从外部日志分不出来）的 `triton_ms`/`baseline_ms` 会偏高且不稳定，`bench_musparse_20260917/spsm_benchmark.json` 里的行不能当交付数据用，需要等其他任务空闲后单独重跑 `ctest --test-dir capi/build -R benchmark.spsm` 확인。

**教训**：这台机器是单卡（`docs/MUSA.md` 开头写着"MTT S5000 ×1"，这次没有对上号），**任何要摸 GPU 的命令，起手前先 `mthreads-gmi` 确认卡数和当前占用**，单卡机器上 GPU 相关任务必须串行，不能因为是两个不同的 runner（Python pytest vs C API ctest）就默认可以并发——它们抢的是同一份硬件。

### `run_flagsparse_split_delivery.py` 的一处 bug（合并逻辑漏了 `performance` 键）

**锚点**：`merge_summaries()`（L165-218）。用假数据（不碰 GPU）单测出来的：一个变体如果只在 Python 精度结果里出现、C API 性能结果里没有它（典型情况就是 `spsv_csr`/`spsv_coo` 这种——C API 那边的 benchmark 直接崩溃、一行数据都没写出来，见第 10 节的 Triton JIT 问题），合并出来的条目会**整条缺失 `"performance"` 键**，而不是标成 `NOT_CONFIGURED`：

```diff
     merged_result: dict[str, object] = {}
     for variant_id in sorted(set(py_result) | set(capi_result)):
         capi_entry = capi_result.get(variant_id) or {}
         py_entry = py_result.get(variant_id) or {}
         entry = copy.deepcopy(capi_entry)
-        if py_entry:
-            entry["accuracy"] = py_entry.get("accuracy", entry.get("accuracy"))
-            entry.setdefault("customized", py_entry.get("customized", True))
-            entry.setdefault("labels", py_entry.get("labels", []))
-        elif not entry:
-            entry = {"accuracy": {"status": "NOT_CONFIGURED"}, "performance": {"status": "NOT_CONFIGURED"}}
+        # Every entry gets both keys regardless of which side(s) had the variant:
+        # a variant missing from the C API run (e.g. its benchmark crashed before
+        # writing any row) must still report performance: NOT_CONFIGURED rather
+        # than silently dropping the key, or a consumer indexing entry["performance"]
+        # breaks on exactly the variants most worth flagging.
+        entry["accuracy"] = py_entry.get("accuracy") or entry.get("accuracy") or {"status": "NOT_CONFIGURED"}
+        entry.setdefault("performance", {"status": "NOT_CONFIGURED"})
+        entry.setdefault("customized", py_entry.get("customized", capi_entry.get("customized", True)))
+        entry.setdefault("labels", py_entry.get("labels") or capi_entry.get("labels") or [])
         merged_result[variant_id] = entry
```

**复现方式**（不需要真机、不需要 GPU，纯 Python 逻辑，任何机器上都能跑）：

```python
import importlib.util, json
spec = importlib.util.spec_from_file_location("m", "run_flagsparse_split_delivery.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
# 构造一份只在 py_summary 里有、capi_summary 里没有的变体（如上面的 "spsv_csr_f32_int_non"），
# 调 mod.merge_summaries(py_path, capi_path, out_path) 后读 out 里这个变体的 entry["performance"]，
# 改之前 KeyError，改之后是 {"status": "NOT_CONFIGURED"}。
```

### 端到端真机验证：清空 `capi/build` 后一行命令跑通编译+精度+性能

```bash
rm -rf capi/build
python3 run_flagsparse_split_delivery.py \
  --mode normal --benchmark-input /root/gcx/matrix --timeout 3600 \
  --results-dir pytest_results_mthreads_split_20260917
```

MTT S5000 实测：CMake 从零配置成功（`ctest baseline: MUSA -> /usr/local/musa/lib/libmusparse.so`），`cmake --build` 73/73 全部编译链接成功，Python 精度（normal 模式，非 quick）**40/40 变体 `accuracy: Passed`**——含第 12 节刚修的 `sddmm_csr`。性能半边（C API `ctest -R benchmark`）2026-09-18 09:15 跑完，完整结果见第 13 节。

### 在别的机器上复现这轮改动

这轮改动里，**代码/文档层面**要合并的只有本文件列出的几处 diff 加 `run_flagsparse_split_delivery.py` 这一个新文件；跑法是：

```bash
cd <仓库根>
export PYTHONPATH=$PWD/src FLAGSPARSE_BACKEND=mthreads   # 按目标后端调整
python3 run_flagsparse_split_delivery.py \
  --mode normal \
  --benchmark-input <该机器的真实矩阵目录> \
  --timeout 3600 \
  --results-dir pytest_results_<backend>_split_<日期>
```

前提条件（见第 10 节）：`capi/ctest/baseline/CMakeLists.txt` 那处 `BACKEND` 变量名修复必须在；目标机器要有对应厂商的稀疏库（本机是 `/usr/local/musa/lib/libmusparse.so` + `musparse.h`），否则 `configure_and_build_capi()` 里的 `cmake` 步骤会正常跑完但 `ctest baseline:` 打出 `<BACKEND> -> none`，性能那边就是空的（这不是 bug，是如实反映没有基线，参见第 4 节的三态表 WIRED/PROBED/SEAM/NONE）。

**以下几个不在这轮要合并的改动里，是本机这次会话调试用的临时产物，不要跟着传**：`scratchpad_run_musa_full.sh`（已被 `run_flagsparse_split_delivery.py` 取代，可以删了）、`musa_full_run_20260917.log`、`run_flagsparse_split_delivery_20260917.log`（跑批日志，`.gitignore` 没覆盖但不该提交）、`node_modules/`、`package.json`、`package-lock.json`（这三个是这台机器上装 `@anthropic-ai/claude-code` 这个 CLI 工具自己留下的，跟 FlagSparse 无关）。

## 13. 2026-09-18：真机完整实测结果（精度 pytest + 性能 muSPARSE），以及 `write_summary.py` 把"没基线"错判成"Skipped"

`rm -rf capi/build` 之后跑 `python3 run_flagsparse_split_delivery.py --mode normal --benchmark-input /root/gcx/matrix --timeout 3600 --results-dir pytest_results_mthreads_split_20260917`，2026-09-18 09:15:41 跑完（`summary_split.json`）。同一天上游 `03a8e74 list completed` 把 `conf/operators.yaml` 的交付清单标注为 **42 条**（本节仍按当前已注册的 40 条报，缺的 `sddmm_csr_c32/c64_int_non_non_row` 等复数 SDDMM 内核落地）：

| 变体 | 精度 | 性能 | 加速比（FlagSparse / muSPARSE） |
|---|---|---|---|
| `gather_f16_int` | Passed | **见下方订正，不是真 Skipped** | — |
| `gather_f32_int` | Passed | Passed | 0.836x |
| `gather_f64_int` | Passed | Passed | 0.833x |
| `gather_c32_int` | Passed | Passed | 0.840x |
| `gather_c64_int` | Passed | Passed | 0.849x |
| `scatter_f16_int` | Passed | **见下方订正，不是真 Skipped** | — |
| `scatter_f32_int` | Passed | Passed | 0.979x |
| `scatter_f64_int` | Passed | Passed | 0.982x |
| `scatter_c32_int` | Passed | Passed | 0.983x |
| `scatter_c64_int` | Passed | Passed | 0.984x |
| `spmv_csr_f32_int_non` | Passed | Passed | 0.324x |
| `spmv_csr_f64_int_non` | Passed | Passed | 0.347x |
| `spmv_csr_c32_int_non` | Passed | Passed | 0.270x |
| `spmv_csr_c64_int_non` | Passed | Passed | 0.290x |
| `spmv_coo_f32_int_non` | Passed | Passed | **3.257x** |
| `spmv_coo_f64_int_non` | Passed | Passed | **3.080x** |
| `spmv_coo_c32_int_non` | Passed | Passed | **2.834x** |
| `spmv_coo_c64_int_non` | Passed | Passed | **2.591x** |
| `spmm_csr_f32_int_non_non_row` | Passed | Passed | 0.480x |
| `spmm_csr_f64_int_non_non_row` | Passed | Passed | 0.417x |
| `spmm_csr_c32_int_non_non_row` | Passed | Passed | 0.480x |
| `spmm_csr_c64_int_non_non_row` | Passed | Passed | 0.118x |
| `spmm_coo_f32_int_non_non_row` | Passed | Passed | 0.866x |
| `spmm_coo_f64_int_non_non_row` | Passed | Passed | 0.976x |
| `spmm_coo_c32_int_non_non_row` | Passed | Passed | 0.887x |
| `spmm_coo_c64_int_non_non_row` | Passed | Passed | 1.037x |
| `sddmm_csr_f32_int_non_non_row` | Passed | Passed | **16.211x** |
| `sddmm_csr_f64_int_non_non_row` | Passed | Passed | **30.773x** |
| `spgemm_csr_f32_int_non_non` | Passed | **NotFound**（24.24s 崩，Triton） | — |
| `spgemm_csr_f64_int_non_non` | Passed | **NotFound**（同上） | — |
| `spsm_csr_f32_int_non_non_row` | Passed | **NotFound**（3600s Timeout） | — |
| `spsm_csr_f64_int_non_non_row` | Passed | **NotFound**（同上） | — |
| `spsv_csr_f32_int_non` | Passed | **NotFound**（247.97s 崩，Triton） | — |
| `spsv_csr_f64_int_non` | Passed | **NotFound**（同上） | — |
| `spsv_csr_c32_int_non` | Passed | **NotFound**（同上） | — |
| `spsv_csr_c64_int_non` | Passed | **NotFound**（同上） | — |
| `spsv_coo_f32_int_non` | Passed | **NotFound**（同上，spsv 是一个 GTest 循环 30 矩阵，第一个矩阵就崩，csr/coo 共用同一次崩溃） | — |
| `spsv_coo_f64_int_non` | Passed | **NotFound**（同上） | — |
| `spsv_coo_c32_int_non` | Passed | **NotFound**（同上） | — |
| `spsv_coo_c64_int_non` | Passed | **NotFound**（同上） | — |

**汇总**：精度 **40/40 Passed**；性能 26 Passed（其中 `sddmm_csr` 16-31x、`spmv_coo` 2.6-3.3x 明显快于 muSPARSE，`spmv_csr` 0.27-0.35x、`spmm_csr` 0.12-0.48x 反而比 muSPARSE 慢，`spmm_csr_c64` 只有 0.118x）/ 2 条曾误判 Skipped（订正见下）/ 12 条 NotFound（`spgemm`/`spsv` 是 Triton JIT 编译问题崩溃，`spsm` 是真的卡到 3600s 超时，两类原因不同，不要混为一谈）。

### `gather_f16`/`scatter_f16` 不是"没测"，是"测了、过了，只是没基线"——`write_summary.py` 的聚合把这个情况错判成 Skipped

一开始报成"性能 Skipped"，用户追问"fp16 是交付 dtype 吗、本机支不支持"，查了原始数据才发现完全测出来了：

```json
// capi/build/bench_mthreads_split/gather_benchmark.json，30 行里随手一行
{
  "name": "gather_spvec_f16_2cubes_sphere",
  "status": "ok",
  "accuracy": "pass",
  "median_ms": 0.032707,
  "baseline_status": "failed",
  "baseline_detail": "muSPARSE: Gather dtype unsupported",
  "speedup": null
}
```

`gather_f16`/`scatter_f16` 各 30 个矩阵，**`status` 全部 `ok`、`accuracy` 全部 `pass`，有真实执行时间**（gather 0.032-0.091ms，scatter 0.033-0.226ms）。唯一的问题是 **muSPARSE 本身不支持 fp16 的 gather/scatter**（`baseline_detail` 写得很明确），不是 MUSA/FlagSparse 跑不了，`speedup` 算不出来纯粹是因为分母缺失。

`capi/tools/write_summary.py` 把这种"每一行 `base` 都是 0（没有基线）"的 dtype 聚合成 `"result": "Unknown"`，投影到变体级别时被归到 `"status": "Skipped"`——和 Python 侧第 8 节发现的"NotFound 混淆了跑崩/跑不完/真不支持"是**同一类根因**：聚合逻辑只有"有基线能算出 speedup"和"没有"两态，没有把"内核本身完全正常、只是没基线可比"单独分类，于是被压缩成了和"真的没跑"外观一样的状态。

**这轮没有改 `write_summary.py`**（工作量超出这轮范围，且需要想清楚新状态叫什么、Python 侧 summary.json 的 schema 要不要跟着加，属于第 9 节说的"跨后端共用代码，举证责任不在单个后端"那一类），只在这里记录清楚，供下一轮或者其他后端一起改：**读这份数据的人，见到 `performance.status == "Skipped"` 时，要去原始 `<op>_benchmark.json` 里确认是真跳过还是"没基线"，不能直接当成"这个 dtype 测不了"。**
