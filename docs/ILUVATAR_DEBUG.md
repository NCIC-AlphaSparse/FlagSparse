# 天数智芯 BI-V150 调试交接

**给接手的人/agent。** 本文只记 2026-09-23/24 在 `bm-baai-dx-zone2-d-biv150-64g-15-106`
（BI-V150 OAM × 16）上**实测**过的事实、已经解决的问题、还开着的问题，以及**不要重复趟的坑**。
背景和探测逻辑见 [ILUVATAR.md](ILUVATAR.md) 第 4.0 节；那台 BI-V100（`u98`）是另一台机器，
卡型、驱动、torch 版本都不同，实测值不能互相套用。

---

## 0. 一句话现状

软件栈**已经跑通**：`tests/ci` 197 passed，**七个交付父算子的非 fp64 精度都已验证可用**
（`sddmm_csr` 待跑）。**这张卡的 fp64 不能用**，根因在厂商的 H2D 拷贝，我们这边绕不过去。
20 个交付变体里 **9 个（所有 f64 和 c64）拿不到数**，其余 11 个可跑。

性能分母**不是 PyTorch** —— `torch.sparse` 在这张卡上对非零 fp32 输入静默返回零，
用的是 CoreX 的 **legacy `cusparseScsrmv` / `Scsrmm`**，仅覆盖 fp32 + int32 + `op=non`
（4.0.1 节）。其余组合没有基线，显示 `N/A`。

2026-09-24 的一轮改动已合入主线，合入时的核对记录见第 9 节。

---

## 1. 环境（容器重建后必须重做）

厂商把 Python 包装在 `/usr/local/corex-4.4.0/lib64/python3/dist-packages`，
**但没有注册给解释器** —— 没有 `env.sh`，`/etc/profile.d` 里也没有。

```bash
# 一次性：装 .pth 和缺失依赖
echo "/usr/local/corex-4.4.0/lib64/python3/dist-packages" \
  > /usr/local/lib/python3.12/site-packages/corex.pth
pip install pyyaml

# 每次开工
cd /opt/FlagSparse
export COREX_HOME=/usr/local/corex
export LD_LIBRARY_PATH=/usr/local/corex-4.4.0/lib64:/usr/local/corex-4.4.0/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/corex-4.4.0/bin:$PATH
export FLAGSPARSE_BACKEND=iluvatar
export FLAGSPARSE_ILUVATAR_VENDOR=torch

# 自检，两行都要对
python3 -I -c "import torch; print(torch.__version__)"      # -I 忽略所有环境变量
python3 -c "from flagsparse.sparse_operations import _common as c; print(c._backend_name(), c._accel_fallback_reason())"
# 期望：iluvatar None
```

**`.pth` 而不是 `PYTHONPATH`**：只设 `PYTHONPATH` 时，任何重写它的子进程都会丢掉 torch，
`tests/ci` 有 17 个用例因此失败（它们用 `subprocess` 起干净解释器验证后端分发）。
`python3 -I` 能过才算真的装好。

建议把这些写进 `/root/gcx/env_ix.sh`（宿主机挂载目录，容器重建不丢）。

---

## 2. 已实测的指纹

| | 实测值 |
| --- | --- |
| 设备名 | `'Iluvatar BI-V150 OAM'` |
| torch | `2.7.1`，pip 元数据 `2.7.1+corex.4.4.0`（模块属性是裸的） |
| `torch.version.cuda` / `.hip` | `'10.2'` / `None` |
| **`warp_size`** | **64**，且属性存在 |
| MP count / 显存 | 16 / 32768 MiB；`max_threads_per_block` 属性缺失 |
| Triton | 3.6.0，backends `['iluvatar']` |
| **Triton target** | `GPUTarget(backend='corex', arch=71, warp_size=64)` |
| 探测 | `_backend_name()` → `iluvatar`，不需要 env 覆盖 |

`warp_size=64` **不需要改代码**：`_common.py` 读的是 `getattr(props, "warp_size", default_warp)`，
属性存在就用真值，`default_warp = 32` 只是兜底。

**spmv_csr 只有三条 legacy 路由可用**（`legacy_rowpar` / `legacy_segbin` / `legacy_bucket_vector`）。
`_spmv_backend_caps` 的 `verified` 判据只认 `cuda+cuda+数字arch` 和 `rocm+hip+gfx*`，
这里是 `corex`/`71`，两条都不匹配，所以七条新路由（row_tile 等）被跳过。
**这不会报错** —— 两处门都包在 `alg in NEW_ALGORITHMS` 里，`auto` 在非 ROCm 上解析到
`legacy_<default>`，天数走 `segbin`。要启用新路由需要先在这张卡上做启动配置扫描，
再给 `verified` 加一条 `corex` 分支并添 `ARCH_PROFILES[("iluvatar", "71")]`，**现在没有依据，不要加**。

---

## 3. fp64 不可用 —— 根因与三个缺陷

### 3.1 根因：fp64 数据上不了卡

厂商自己的警告，触发点是 `data.to(device)`：

```
UserWarning: Limited support for torch.double is provided currently.
As for how to resolve the warning, contact Iluvatar application engineers.
(Triggered internally at .../aten/src/ATen/native/cuda/Copy.cu:412.)
```

**fp64 张量拷到卡上会静默变成全零**，只给一条 UserWarning。这一条解释了全部 fp64 现象，
它们不是各自独立的问题：

- 设备端 `float32 → float64` 转换返回零（同一条 `Copy.cu` 路径）；
- `legacy_rowpar` 跑 fp64 能跑完但 18 个用例全错 —— 内核没问题，它在乘零；
- `torch` 直接拒绝 fp64 gemm；
- segbin 编译器 abort（见 3.2，是**另一个独立缺口**）。

### 3.2 三个缺陷，按优先级

**缺陷 A：fp64 H2D 拷贝返回全零，只给 UserWarning。**
三条里最危险 —— 静默的错误结果。同一套栈对 fp64 gemm 会干净地抛
`RuntimeError: gemm of double is not supported on CoreX.`，说明拒绝机制是有的。

```bash
python3 -c "
import torch
t = torch.randn(7, dtype=torch.float32, device='cuda')
print('设备端 .to(fp64):', float(t.to(torch.float64).sum()))   # 实测 0.0
print('CPU 端 .to(fp64):', float(t.cpu().to(torch.float64).sum()))
"
```

**缺陷 B：fp64 原子加没有指令选择，以 `LLVM ERROR` + SIGABRT 杀进程。**
不是抛可捕获的异常，pytest 会被整场打死。

```
LLVM ERROR: Cannot select: f64 = AtomicLoadFAdd<
  (load store syncscope("agent") acq_rel (s64) on addrspace 1)>
  spmv_csr.py:498:31   In function: _spmv_csr_segbin_kernel
```

`tl.atomic_add` 在 `spmv_csr.py` 里只出现三次（498 实数、552/553 复数），全在两个 segbin 内核里。

```bash
python3 - <<'PY' 2>&1 | tail -30
import torch
from flagsparse.sparse_operations.spmv_csr import flagsparse_spmv_csr
d = "cuda"
data = torch.randn(64, dtype=torch.float64, device=d)
col  = (torch.arange(64) % 32).to(device=d, dtype=torch.int32)
ptr  = torch.arange(0, 65, 2, dtype=torch.int32, device=d)
x    = torch.randn(32, dtype=torch.float64, device=d)
print(flagsparse_spmv_csr(data, col, ptr, x, shape=(32, 32)))
PY
```

**缺陷 C（能力边界，非 bug）：fp64 gemm 不支持，但报错干净。**
`RuntimeError: gemm of double is not supported on CoreX.`

### 3.3 一条路由上的发现（**暂不采用**）

`legacy_rowpar` / `legacy_bucket_vector` 用的 `_spmv_csr_real_kernel`、
`_spmv_csr_complex_kernel` **一个原子都没有**，所以
`FLAGSPARSE_SPMV_CSR_KERNEL=rowpar` 能把 fp64 的进程崩溃变成正常运行。
形状与 `_spmv_csr_default_backend()` 里已有的 XPU 分支完全一致
（XPU 因为 segbin 的 associative scan 不能 lower 而返回 rowpar）。

**但现在不要给 iluvatar 加这条分支** —— 在缺陷 A 修好之前，它只是把"崩溃"换成"静默算错"，
后者更危险。等厂商修好 H2D 之后再加。

---

## 4. 已测结果（排除 fp64）

| 范围 | 结果 | 备注 |
| --- | --- | --- |
| `tests/ci` | 197 passed / 6 skipped / 1 deselected | deselect 见 5.3 |
| `gather` | 16 passed | |
| `scatter` | 34 passed | |
| `spmv_csr` | 287 passed / 1092 skipped | 2026-09-24 强制 `legacy_rowpar` 的无 fp64 精度回归；原先 9 个 `float32 + legacy_rowpar` 失败已修复 |
| `spmv_coo` | 41 passed | |
| `spmm_csr` | 31 passed | 耗时 27 秒，偏慢但正常 |
| `spmm_coo` | 12 passed | 2026-09-24 fp32 精度：non/trans/conj、int32/int64、2 个形状；首轮约 2 分钟 |
| `sddmm_csr` | **未跑** | |

**那 9 个失败的根因已知，且已修复，不是内核问题。** 修复前 `float32` 时
`_baseline_compute_dtype` 是 `float64`（`spmv_csr.py:181`），
`_get_spmv_baseline_data`（`:1050`）做 `prepared.data.to(compute_dtype)` —— 正是缺陷 A。
Iluvatar 现在保持 `float32`，不再触发该转换；其他后端仍保留原先的 fp64 基线。
默认路由是 segbin，本来就不走 `_get_spmv_baseline_data`，所以交付不受影响。

最小回归（强制 `FLAGSPARSE_SPMV_CSR_KERNEL=rowpar`）已实测为 6 passed：`non`、
int32/int64、3 个形状。按第 5.1 节的筛选和两个 `--deselect` 跑完整无 fp64 集合，
已实测为 287 passed / 1092 skipped。

### CSR SpMV 的 PyTorch 性能基线（BI-V150/CoreX 4.4）

`torch.sparse_csr_tensor` 可创建，但对非零 2x2 CSR 矩阵和非零向量执行
`torch.sparse.mm(csr, x[:, None]).squeeze(1)` 返回 `[0.0, 0.0]`，预期为 `[5.0, 11.0]`。
这是静默错误，不能作为 PyTorch 性能基线。2026-09-24 已验证并接入 CoreX legacy
`cusparseScsrmv`（fp32、int32、`non`），由 `cupy.cusparse.csrmv` 计时；其输出只和 CPU
SciPy 参考交叉核对，不参与精度 oracle。generic `cusparseSpMV` 仍因错误的多 TB 工作区查询禁用。

CSR SpMM 的情况相同：fp32 2x2 非零矩阵乘非零 dense RHS 返回全零，fp16 则报
`addmm_sparse_cuda not implemented for 'Half'`。`tests/test_spmm.py` 在 Iluvatar 上改为将
CPU SciPy 参考直接按输出 dtype 传回设备（避免 fp64 H2D 置零），并显式禁用该 PyTorch baseline。
同日接入 `cupy.cusparse.csrmm` 作为 fp32/int32/`non` 的性能 baseline；Fortran RHS 和 int32
row pointer 在计时前准备，generic `cusparseSpMM` 仍不使用。

### 4.0.1 CoreX legacy cuSPARSE baseline 接线记录（2026-09-24）

**目标和边界。** 这里只补性能分母，不改变任何精度 oracle。CSR SpMV 原有正确性是 CPU
FP64/complex128 scatter；CSR SpMM 在 Iluvatar 上是 CPU SciPy。厂商输出仅用于计时和再做一次
交叉 `allclose`，其错误不会改写 FlagSparse 的精度结论。

**库的发现。** BI-V150/CoreX 4.4 镜像中有厂商版 `cupy 11.4.0+corex.4.4.0`，并带有
`/usr/local/corex-4.4.0/include/cusparse.h` 和
`/usr/local/corex-4.4.0/lib64/libcusparse.so`。`readelf -Ws` 可见
`cusparseScsrmv`、`cusparseScsrmm`、generic `cusparseSpMV` 和 `cusparseSpMM` 均导出；它是
CUDA/cuSPARSE ABI 兼容层，不是单独名为 `ixsparse` 的 Python 包。

**先排除的路径。** 以下最小 fp32 CSR SpMV 用例在 `cupyx` 的 `A @ x` 和
`cupy.cusparse.spmv()` 上都因 `SpMV_bufferSize` 返回约 140 TB 工作区而 OOM；generic SpMM
也不能作为候选。另一方面，`torch.sparse.mm` 的 CSR SpMV/SpMM 对非零 fp32 输入静默返回零，
fp16 CSR SpMM 直接报未实现。这些路径都不能作为 baseline。

```python
# A = [[1, 2], [3, 4]], x = [1, 2]
# cupyx CSR @ x / cupy.cusparse.spmv -> OOM: ~140 TB workspace
# torch.sparse.mm(CSR, x[:, None]).squeeze(1) -> [0., 0.]
```

**验证通过的 API。** 同一矩阵经 `cupy.cusparse.csrmv()` 得到 `[5., 11.]`；
`cupy.cusparse.csrmm()` 对 RHS `[[1, 2], [3, 4]]` 得到 `[[7, 10], [15, 22]]`。在
`GL7d14.mtx`（1,831,183 nnz）上，两者相对 CPU SciPy 的最大绝对误差均为 `1.907e-06`。
这就是选择 legacy `cusparseScsrmv`/`cusparseScsrmm` 的原因。

**实现。**

- `_common._spmv_csr_sparse_ref_backend()` 只在 Iluvatar 返回专用
  `iluvatar_legacy_cusparse` backend，且只接受 fp32、int32、`op=non`；其他 backend 的路由不变。
- `_spmv_csr_benchmark.measure_vendor()` 对该标识局部导入 `cupy.cusparse`，以 DLPack 共享
  torch tensor，并复用输出 buffer 调 `csrmv`。若原始 `indptr` 为 int64，只在计时前转换副本；
  FlagSparse 输入不变。
- `tests/test_spmm.py` 只在 Iluvatar 性能分支以相同限制调用 `csrmm`。RHS 在计时前经
  `cp.asfortranarray()` 变为 column-major，row pointer 转 int32；两种转换都不计入 baseline 时间。
- 没有改变 `spmm_csr.py` 的通用 vendor 路由，避免其他 benchmark/test 入口把未接线的 legacy API
  误判成可用。`cupy.cusparse` 也只在 Iluvatar 分支按需导入，因此不影响其他后端的可选依赖导入。

**交付复测。** 使用下列命令，各 10 个 delivery matrix、warmup 5、iters 20：

```bash
env FLAGSPARSE_BACKEND=iluvatar FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse \
  python3 -u tests/test_spmv_csr.py pytest_results_iluvatar_delivery_f16_f32_rerun_20260924/delivery_matrices \
  --dtypes float32 --index-dtypes int32 --indptr-dtypes int32 --ops non --warmup 5 --iters 20 \
  --csv-csr pytest_results_iluvatar_delivery_f16_f32_rerun_20260924/spmv_csr/performance_legacy_cusparse.csv

env FLAGSPARSE_BACKEND=iluvatar FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse \
  python3 -u tests/test_spmm.py pytest_results_iluvatar_delivery_f16_f32_rerun_20260924/delivery_matrices \
  --dtypes float32 --index-dtypes int32 --ops non --warmup 5 --iters 20 \
  --csv pytest_results_iluvatar_delivery_f16_f32_rerun_20260924/spmm_csr/performance_legacy_cusparse.csv
```

SpMV 为 10/10 vendor PASS，几何平均 `1.15x`（FlagSparse 相对 legacy cuSPARSE）；SpMM 为
10/10 CPU-reference PASS + vendor PASS，几何平均 `7.99x`。结果只代表 fp32/int32/non；fp16、
int64 和 trans/conj 仍应显示 baseline `N/A`，不能套用这些数字。

### 4.0.2 runner 的 fp16/fp32 精度筛选修正（2026-09-24）

首次交付 runner 使用 `-k "float16 or half or float32"`。性能脚本按各自的 `--dtypes` 参数
确实跑到了 fp16/fp32，但共享 pytest 的 gather fp32 参数 ID 是 `float`，不是 `float32`：结果表中
`gather_f32_int` 的性能为 Passed，精度却为 NotFound。这不是算子失败，而是筛选漏选。

交付命令已改用：

```text
-k "(float or half or f16 or f32) and not (double or float64 or bfloat16 or complex64 or complex128)"
```

其中 `float` 覆盖 gather 的 fp32，`f16`/`f32` 覆盖采用简写 ID 的套件；显式排除项防止 substring
匹配把 fp64、bf16 或 complex 带入。以 delivery 选择的 7 个算子执行 `pytest --collect-only`，得到
648 个用例，结果中没有 double、float64、bfloat16、complex64 或 complex128 节点。性能参数不变，
仍由各 `--op-benchmark-args` 限制为 fp16/fp32。

---

## 5. 怎么跑测试

### 5.1 精度

```bash
NOFP64='not float64 and not double and not complex128 and not dtype3 and not dtype5'

# 单算子
python3 -m pytest tests/pytest -q -p no:warnings -m spmv_csr -k "$NOFP64" \
  --deselect tests/pytest/test_spmv_csr_accuracy.py::test_spmv_csr_default_segment_multilevel \
  --deselect tests/pytest/test_spmv_csr_accuracy.py::test_spmv_csr_complex_cancellation

# 七个交付父算子逐个跑（每次独立进程，崩溃只损失一个算子）
for op in gather scatter spmv_csr spmv_coo spmm_csr spmm_coo sddmm_csr; do
  printf "%-12s " "$op"
  timeout -s KILL 1800 python3 -m pytest tests/pytest -q -p no:warnings -m "$op" -k "$NOFP64" 2>&1 \
    | grep -E "passed|failed|error" | tail -1 || echo "TIMEOUT/KILLED"
done
```

### 5.2 性能

```bash
# 矩阵是位置参数，不是 --input
python3 tests/test_spmv_csr.py tests/data/cage12.mtx \
  --dtypes float32 --ops non --warmup 5 --iters 20 --csv out.csv
```

性能脚本**不走 pytest**，`-k` 语法它们不认，要用各自的 `--dtypes` / `--dtype`
（名字不统一，先看 `--help`）。

### 5.3 交付（11 个可用变体）

```bash
setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --phase both --mode normal --delivery-only --gpus 0 --timeout 3600 \
  --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 \
  --pytest-args="-k \"(float or half or f16 or f32) and not (double or float64 or bfloat16 or complex64 or complex128)\" --deselect tests/pytest/test_spmv_csr_accuracy.py::test_spmv_csr_default_segment_multilevel --deselect tests/pytest/test_spmv_csr_accuracy.py::test_spmv_csr_complex_cancellation" \
  --op-benchmark-args='spmv_csr=--dtypes float32' \
  --op-benchmark-args='spmv_coo=--dtypes float32' \
  --op-benchmark-args='spmm_csr=--dtypes float32' \
  --op-benchmark-args='spmm_coo=--dtypes float32' \
  --op-benchmark-args='sddmm_csr=--dtypes float32' \
  --results-dir pytest_results_iluvatar_delivery_v1 \
  > pytest_results_iluvatar_delivery_v1.log 2>&1 &

python3 tools/delivery_table.py pytest_results_iluvatar_delivery_v1 --markdown
```

**精度阶段和性能阶段跑的是完全不同的程序**，所以要分别挡 fp64：
`--pytest-args` 管精度（真的起 `python -m pytest`），
`--op-benchmark-args` 管性能（起 `tests/test_*.py` 这些独立脚本）。
`--delivery-only` 会先注入 `float32,float64`，显式写的会覆盖它。

上面那个 `-k` 是 4.0.2 节修正后的版本：gather 的 fp32 参数 ID 是 `float` 不是 `float32`，
旧写法会让 `gather_f32_int` 的精度显示 `NotFound` 而性能却是 Passed —— 是筛选漏选，不是算子失败。
5.1 节那个 `$NOFP64` 用于手工按算子跑，两者不要互相套用。

三条硬规矩：

- **`--results-dir` 每次用新目录。** `summary.json` 是整体覆盖写的，把子集跑进旧目录会毁掉报告。
- **加速比的分母分两种，别混。** fp32 + int32 + `op=non` 走 CoreX legacy cuSPARSE
  （`cupy.cusparse.csrmv` / `csrmm`，见 4.0.1 节）；其余组合没有厂商基线，显示 `N/A`。
  **不要把 4.0.1 那两个几何平均（SpMV 1.15x、SpMM 7.99x）套到 fp16、int64 或 trans/conj 上。**
  和海光（hipSPARSE）、摩尔（muSPARSE）的数同样不能并排比。
- 9 个 f64/c64 变体会是 `NotFound` 或 `CRASH`，**这是卡的限制不是我们的缺陷**，回传时要写明。

`tests/ci` 里 `test_installed_wheel_import_resolves_outside_repo_tree` 在 editable 安装下必然失败，
**deselect 它，不要 `pip install .`** —— 那样 site-packages 会有一份快照，把后续源码改动全遮住。

---

## 6. 还开着的问题

### 6.1 `spmm_coo` fp32 全零（已修复）

首轮跑到 `spmm_coo` 时长时间无输出，随后所有 fp32 用例返回全零。根因是
`_spmm_coo_compute_dtype(float32)` 默认升到 fp64，触发 CoreX 的 fp32→fp64 静默置零缺陷；
2026-09-24 已在 Iluvatar 分支保留原生 fp32，12 个 fp32 精度用例全部通过。

`BLOCK_NNZ=256` 的展开仍是性能风险：
`_spmm_coo_rowrun_*_kernel` 里是 `tl.static_range(0, BLOCK_NNZ)`，BLOCK_NNZ 是 constexpr，
内核体会展开 256 次。沐曦 C550 上同一个默认值让 spmm_coo 套件从 23 分 48 秒变成 11.8 秒
（把 complex 的 BLOCK_NNZ 钳到 4 之后），而且在那张卡上还撞了 4 KB 私有内存上限。

```bash
# 先只跑 fp32（complex 的展开是实数的两倍）
timeout -s KILL 600 python3 -m pytest tests/pytest -q -p no:warnings -m spmm_coo \
  -k "$NOFP64 and float32" 2>&1 | tail -3

# 判断是编译还是执行
ps aux | grep -c "[c]lang\|[l]lc\|[t]riton"
```

如果确认是展开开销，参照 `_MACA_SPMM_COO_COMPLEX_BLOCK_NNZ` 的做法，在
`_resolve_spmm_coo_launch_config` 里加一条 iluvatar 的钳位。

### 6.2 `sddmm_csr` 未跑

### 6.3 交付跑未做

### 6.4 厂商缺陷未提交

缺陷 A 和 B（见 3.2），复现都只有三五行，不涉及 FlagSparse。

---

## 7. 不要重复趟的坑

**`-k` 挡不住 fp64。** 这个套件用**至少五套** dtype 命名：`float64`、`double`、
`dtype1`、`dtype3`、`dtype5`，视用例文件而定。而且含义在文件之间不一致 ——
`test_spmv_csr_full_dtype_op_surface` 里 `dtype1` 是 **bfloat16**，
`test_spmv_csr_default_segment_multilevel` 里 `dtype1` 是 **float64**。
漏一个就会触发缺陷 B 把整场打死。后两个只能用 `--deselect` 按用例函数排除。

**`pytest --forked` 不能用。** 试过：**2435 failed / 5 passed**，连之前通过的
gather/scatter 和 fp32 spmv_csr 都全挂，全部是
`RuntimeError: Cannot re-initialize CUDA in forked subprocess.` ——
加速器上下文不能跨 `fork()`，而 pytest 在**收集阶段**就会碰设备。
那 5 个幸存的是纯 CPU 的策略类用例。
**要隔离就按算子分多次调用 pytest**（5.1 那个循环），每次是独立进程。

**`| tail -N` 看不到崩溃点。** `Fatal Python error: Aborted` 的转储有上百行，
管道里永远截不到用例名。**必须 `> file 2>&1` 再 grep**：

```bash
python3 -m pytest ... -v > /home/ix.log 2>&1
grep -E "PASSED|FAILED" /home/ix.log | tail -5     # 最后几个有结果的
grep "test_" /home/ix.log | tail -3                # 最后一行没有结果的就是崩溃点
```

**加 `-p no:warnings`。** CuPy 的重复安装警告每次刷屏，会把结果行顶掉。
那个警告本身无害（三个 dist-info 指向同一个 11.4.0）。

**容器/驱动版本必须一致。** 4.5.0 镜像配 4.4.0 驱动时，枚举和 `cudaSetDevice` 都成功，
但 `cudaMalloc` 返回 **801 `operation not supported`**。
快速判据：容器里 `ixsmi` 的 `CUDA Version` 列显示 `N/A` 就是不匹配，显示 `10.2` 才对。

**失败的 `docker run` 会污染宿主机。** Docker 把不存在的挂载源自动建成空目录。
一次挂 `ixsmi` 的失败尝试在宿主机建出了整条 `/usr/local/corex-4.5.0/bin/ixsmi/`，
既让后续 run 报 `not a directory`，又让人误以为宿主机装了 4.5.0（其实从未装过）。
判断某版本在不在，先看 `ls -ld` 和链接数；删用 `rmdir`（非空会失败，这本身是保护）。
那个挂载本来也不需要 —— 镜像自带 `ixsmi`。

**新增 `tests/ci` 文件时，先查重型 import 在不在守卫后面。** `tests/ci` 跑在**既没有
numpy 也没有 scipy**（并且可能没有 torch）的环境上 —— 这条约束写在
`test_ascend_reporting.py::_ascend_row_status` 的 docstring 里。裸 `import` 会在收集阶段
报错并拖垮整个 `tests/ci`，而 `pytest.importorskip` 只是跳过一个文件。两天内有两个 drop
栽在这里。本地复现那个瘦环境（开发机什么都装了，门禁抓不到）：

```bash
mkdir -p /tmp/blk && cat > /tmp/blk/sitecustomize.py <<'EOF'
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("numpy", "torch"):   # 按要验的包改
            raise ImportError("blocked")
        return None
sys.meta_path.insert(0, _Block())
EOF
PYTHONPATH=/tmp/blk python3 -m pytest tests/ci/<文件> -q -p no:warnings
# 期望 "1 skipped"，出现 error 就是 import 顺序错了
```

（py3.12 只认 `find_spec`，老的 `find_module` 写法不生效。）

**诊断工具本身也是嫌疑人。** 这次栽了两回：一次是测试参考链把**算对的内核**报成错
（见第 8 节），一次是 `--forked` 让**全部用例**失败。两次的输出看起来都像"这张卡不行"的硬件结论。
在新硬件上，先怀疑工具，再怀疑硬件。

---

## 8. 已经修好并进上游的东西

**`tests/reference_utils.py::_numpy` 曾经在设备上做 dtype 转换**，
原本是 `tensor.to(dtype).detach().cpu().numpy()`。在这张卡上撞到缺陷 A，
于是 SciPy 参考矩阵拿到正确的 nnz 配一组**全零的值**，把算对的 fp32 内核全部报成失败。

诊断过程值得复用：打印参考链的每一步中间值，并和一个**不经过 scipy、不经过设备**的
纯 CPU 稠密对照比。当时的输出是 `scipy nnz: 7` 但 `scipy data sum: 0.0`，
而 `ours` 和 `dense@x` 都是 `0.3348` —— 一眼就分清了是参考坏了而不是内核错了。

已修为 `tensor.detach().cpu().to(dtype).numpy()`（先搬再转），
`tests/test_spmv.py` 里两处同样写法一并修了。CUDA 上是恒等变换。

**教训**：声称是 CPU oracle 的参考，要逐步确认**每一步**都在 CPU 上，
包括辅助函数内部的 dtype 转换。

---

## 9. 合入核对记录（2026-09-24）

本节是 `/workspace/fork/biv` 那一轮改动合进主线时的回执：**收了什么、改了什么、验了什么**。
接手的人不必重读这一节就能干活;它存在的意义是让下一次 drop 少踩同样的坑。

### 9.1 收下的改动（8 个文件，原样采纳）

| 文件 | 内容 |
| --- | --- |
| `src/.../_common.py` | 新增 `iluvatar_legacy_cusparse` 厂商后端，门控在 fp32 + int32 + `op=non`，并检查 `cupy.cusparse.csrmv` 真的存在 |
| `src/.../spmv_csr.py` | `_baseline_compute_dtype` 抽成 `_spmv_baseline_compute_dtype()`，天数上 fp32 保持原生 |
| `src/.../spmm_coo.py` | `_spmm_coo_compute_dtype` 的昇腾分支扩到 `or _is_iluvatar_runtime()` |
| `src/.../_spmv_csr_benchmark.py` | legacy `csrmv` 计时路径，dlpack 转换和输出缓冲区复用 |
| `tests/test_spmm.py` | legacy `csrmm` baseline，参考改为按输出 dtype 直接回传（避开 fp64 H2D 置零） |
| `tests/ci/test_iluvatar_detection.py` | 两个 dtype 门控测试（monkeypatch `_is_iluvatar_runtime`，CUDA 上可跑） |
| `docs/ILUVATAR.md`、`docs/ILUVATAR_DEBUG.md` | 4.0.1 / 4.0.2 和第 4、6.1 节的实测更新 |

三处 src 改动的门控都验过：CUDA 上 `_spmv_baseline_compute_dtype(float32)` 和
`_spmm_coo_compute_dtype(float32)` 仍返回 `torch.float64`，其他后端不受影响。

### 9.2 合入时修掉的三处

**一、`test_iluvatar_detection.py` 的 import 顺序。**
`import torch` 排在它自己的 `pytest.importorskip("torch", reason="tests/ci runs on a
CPU-only runner without torch")` **之前**。精简 CI 上这会在**收集阶段**报
`ModuleNotFoundError`，把整个 `tests/ci` 拖垮（`Interrupted: 1 error during collection`），
而不是跳过一个文件。改成 `torch = pytest.importorskip(...)`。

**这是两天内第二个 drop 犯同一个错**（前一个是昇腾的 `test_ascend_complex_gather.py`
里的 `import numpy`）。**新增 `tests/ci` 文件时，先查重型 import 在不在守卫后面。**
本地门禁抓不到 —— 开发机什么都装了。复现方法见 7 节最后。

**二、两个厂商 drop 撞车。** 昇腾有一个源码 grep 测试
（`tests/ci/test_spmm_coo_ascend_source.py`）断言 `_spmm_coo_compute_dtype` 里有字面量
`"if _is_ascend_runtime():"`；天数把同一行扩成了 `"... or _is_iluvatar_runtime():"`，
于是 `ValueError: substring not found`。

该测试真正保护的是**分派顺序**（昇腾分支在通用 fp32→fp64 提升之前），不是那一行的拼写，
所以改成只匹配 `_is_ascend_runtime()` 这个谓词。**源码 grep 类断言要匹配承载性质的最小
token** —— 锁死整行等于给下一个后端埋雷。

**三、本文自身两处命令不一致。** 4.0.2 节说交付命令已改用新的 `-k`，但 5.3 节仍是旧的
`$NOFP64`；5.3 还写着"分母是 PyTorch"，而 4.0.1 刚接上 legacy cuSPARSE。两处都已对齐，
并注明 5.1 的手工筛选和 5.3 的交付筛选**不要互相套用**。

### 9.3 验证

| 检查 | 结果 |
| --- | --- |
| `make format-check` / `lint` / `lint-src` | 全过 |
| `make pre-commit-check` | 全过 |
| `tests/ci`（CUDA 开发机） | 247 passed / 4 skipped |
| `tests/pytest -m "spmv_csr or spmm_coo"`（CUDA） | 1213 passed / 3816 skipped |
| 无 torch 环境模拟 | `test_iluvatar_detection.py` 为 1 skipped，非 collection error |
| 门控 | CUDA 上两个 `compute_dtype` 仍是 `float64` |

### 9.4 给下一轮的三条

1. **`sddmm_csr` 还没跑过**，它是七个交付父算子里唯一没有实测记录的（6.2 节）。
2. **交付跑还没做**（6.3 节）。做之前把 5.3 的命令原样用，特别是 `-k` 和两个 `--deselect`。
3. **两个厂商缺陷还没提交**（6.4 节）。复现都只有三五行、不涉及 FlagSparse，
   而且 4.0.1 又添了一条证据：generic `cusparseSpMV` 的 `SpMV_bufferSize` 要约 140 TB，
   同一个库的 legacy 入口却完全正常 —— 这比"fp64 不支持"具体得多。
