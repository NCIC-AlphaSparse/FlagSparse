# MetaX（MACA / 曦云 C550）

Python 侧的 MetaX 后端：bring-up、日常跑法、已知缺陷、调优。C API 那一层见
`capi/docs/MACA.md`；各后端文档的对应关系见 [README.md](README.md)。

验证环境：MetaX C550（`warp_size=64`、104 MP、64 GB），MACA SDK 3.8.2.6，
torch `2.10.0+metax3.8.1.0`，triton `3.6.0+metax3.8.1.0`，Python 3.12。

---

## 0. 状态：探测和算子已实机验证，调优参数没有

FlagSparse 按运行时分发**厂商参考实现/基线**和**少数按后端分叉的内核**：

| 运行时 | 判定 | 厂商稀疏库 | 内核起点 |
| --- | --- | --- | --- |
| NVIDIA CUDA | `torch.version.hip is None` | cuSPARSE（CuPy） | segbin / 一行一 program |
| DCU / ROCm | `torch.version.hip is not None` | hipSPARSE（hip-python） | rowpar / 持久化 worker |
| **MetaX / MACA** | 见 2.2 | 暂用 CuPy 兼容路径 | **平移自 CUDA** |

**MetaX 与 CUDA、DCU 最大的不同：MACA 与 CUDA 源码兼容**，机器上 `torch.version.cuda`
有值、`torch.version.hip` 为 `None`，**靠这两个判据分不出 MetaX 和 NVIDIA** —— 所以探测
是这条路径当初的首要风险点。

现在三件事的把握程度完全不同：

- **探测逻辑**（`_detect_maca_runtime` / `_maca_device_model`）—— ✅ 已在 C550 实机验证
  （2026-09-04，见 2.2），`torch.version.maca` 存在、设备名 `'MetaX C550'` 也命中，
  两条路径都能自动认出来，不必显式指定；
- **算子本身** —— ✅ 921 个用例实机全绿（2026-09-05，见第 3 节）。Triton 内核体与 CUDA
  同源，MetaX 走的就是那份在 NVIDIA 上验证过的代码；
- **调优 profile** —— ❌ 除 `warp_size` 外仍是从 CUDA 平移的**占位值**，没有任何 C550
  实测依据，见第 8 节。

**已排除、不必怀疑的**：CUDA 与 DCU 两条既有路径没有被 metax 改动触碰
（`_is_maca_runtime()` 在两者上均为 `False`，所有 metax 分支不进入）。

---

## 0.5 交付复现：40 个变体 × 30 个矩阵（精度 + 性能）

**环境**（每次开工，第 1 节有逐项说明）：

```bash
export PYTHONPATH=$PWD/src
export FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_MODEL=c550
export FLAGSPARSE_MACA_VENDOR=none      # 本机没有 CuPy，不走厂商基线
python3 -c "from flagsparse.sparse_operations import _common as c; print(c._backend_name(), c._maca_device_model(), c._accel_fallback_reason())"
# 期望：metax c550 None
```

**命令**：

```bash
setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --phase both --mode normal --delivery-only --gpus 0 --timeout 4500 \
  --benchmark-input /root/gcx/matrix --benchmark-warmup 5 --benchmark-iters 20 \
  --benchmark-args=--no-cusparse \
  --results-dir pytest_results_metax_delivery \
  > pytest_results_metax_delivery.log 2>&1 < /dev/null &
```

- `--no-cusparse`：C550 上没有可用的厂商稀疏库，SDDMM 的 `torch.sparse.sampled_addmm` 结果还是错的（7.3 节），
  性能 baseline 只能是 PyTorch；
- `--timeout 4500`：`--delivery-only` 已把 spmv/spmm/spsv 收窄到 int32 + non，但 **SDDMM 的 4 个 K 值
  不收窄**（交付名里没有 K），实测推算全量至少 3660 秒（7.4 节）。只想快速出数，可以改用
  `--timeout 1200 --op-benchmark-args='sddmm_csr=--k 64'`，但那样 SDDMM 的加速比只含 K=64，和 CUDA 等
  跑满 4 个 K 的后端**不可直接比较**，报告里要注明；
- 外层 12 小时、`setsid` 后台：三角类算子在 C550 上可能挂死（第 5 节），挂死时只能靠 KILL。

**参考**：

| | 本后端 |
|---|---|
| 性能 baseline（报告里与 FlagSparse 并列计时的那一列） | CuPy **真装了**就用 `cupy_cusparse`，否则 `torch` —— 探测而非假定；本机实际是 PyTorch |
| 精度参考（内核被比对的那个值） | **CPU 上的 SciPy** |

精度不走 torch.sparse 是有实测原因的：MACA 的 fp32 CSR 路径会返回非有限值，拿它当参考会把好内核报成错的。

**预期会看到的非 Passed**：`spsv_*` / `spsm_csr` 在走到 `csr_cw`（ALG1，unit 对角）时可能非法访存或挂死，
记为 `Error` / `Timeout`（第 5 节）；**不要**用 CPU 求解顶替。所有加速比的分母都是 PyTorch，不能和
CUDA/MUSA 对厂商库的数放在一起比。

跑完用同一个工具看 40 行结果（缺变体时退出码为 1），回传时直接贴它的输出：

```bash
python3 tools/delivery_table.py pytest_results_metax_delivery              # 加 --markdown 输出 Markdown 表
```

**`86a09cd`（2026-09-18）之前跑出的性能结果作废**，要用当前 runner 重跑，原因见 `prompt.md` 第 2 节。
参数为什么都不能省、各状态的含义，见仓库根 `README_cn.md` 的"复现交付测试"一节。

```bash
# 在任意后端上强制切换精度参考，用于验证另一条路径
export FLAGSPARSE_ACCURACY_REFERENCE=auto    # auto（默认）| scipy | torch
```

---

## 1. 每次开工的四行

```bash
cd <仓库>
export PYTHONPATH=$PWD/src          # 独立脚本需要；pytest 由 pytest.ini 自带
export FLAGSPARSE_MACA_VENDOR=none  # 本机没有 CuPy，跳过厂商基线
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/mxdriver/lib:$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH
```

后端探测**不需要**再设 `FLAGSPARSE_BACKEND` / `FLAGSPARSE_MACA_MODEL`：C550 上
`torch.version.maca` 存在、设备名是 `'MetaX C550'`，两条探测路径都能命中。确认：

```bash
python -c "import flagsparse; print(flagsparse.__file__)"   # 必须指向 <仓库>/src/
python -c "from flagsparse.sparse_operations import _common as c; print(c._backend_name(), c._maca_device_model())"
# 期望：metax c550
```

`flagsparse.__file__` 指到 `site-packages` 就说明跑的是别的副本，任何结果都不算数。

---

## 2. 首次 bring-up

已经在跑的机器可以跳过整节。

### 2.1 ⚠️ 第 0 步：确认 FlagTree 的 metax 后端可用

FlagSparse 的算子靠 Triton 编译。C550 上必须是 **FlagTree 的 metax 后端**，
这一步不通，后面所有报错都会指向错误的方向。

```bash
python -c "import triton; print(triton.__version__)"
python -c "import triton.backends as b; print(list(b.backends.keys()))"   # 期望含 metax
```

若没有 metax，按 FlagTree 文档安装（`FLAGTREE_BACKEND=metax`）。依赖 `metax-llvm`（19）、
`metaxTritonPlugin`、MACA 数学库；其中 maca-llvm 需向沐曦索取。

装好后**跑一个最小 Triton kernel 确认真能编译执行**，再往下走。注意 `@triton.jit` 必须
定义在**真实的 .py 文件**里（Triton 要读源码），不能写在 `python - <<EOF` 的 stdin 里，
否则会报 `ValueError: @jit functions should be defined in a Python file`：

```bash
cat > /tmp/tri_smoke.py <<'EOF'
import torch, triton, triton.language as tl

@triton.jit
def k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tl.store(y_ptr + o, tl.load(x_ptr + o, mask=m) * 2.0, mask=m)

x = torch.arange(1024, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)
k[(1,)](x, y, 1024, BLOCK=1024)
print("triton ok:", torch.allclose(y, x * 2))
EOF
python /tmp/tri_smoke.py
```

> 若报 `ModuleNotFoundError: No module named 'mlir'`，那是 Triton 的 TLE-RAW
> 实验特性在导入期拉了未安装的依赖，**与 MetaX 无关**（同样的报错在 NVIDIA 机器上
> 也会出现）。FlagSparse 不使用 TLE-RAW；确认 metax 后端本身可用即可。

### 2.2 环境指纹

```bash
python - <<'EOF'
import os, torch
print("torch:", torch.__version__)
print("version.__dict__:", torch.version.__dict__)
print("cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device name:", repr(p.name))
    print("warp_size:", getattr(p, "warp_size", "<无该属性>"))
    print("MP count:", p.multi_processor_count)
    print("max_threads_per_block:", getattr(p, "max_threads_per_block", "?"))
    print("max_threads_per_mp:", getattr(p, "max_threads_per_multi_processor", "?"))
print("MACA env:", {k: v for k, v in os.environ.items() if "MACA" in k or "METAX" in k})
EOF
```

三个关键点：`torch.version` 有没有 `maca` / `metax` 属性；**设备名的确切字符串**
（决定设备名探测能否命中）；**`warp_size` 是 32 还是 64**（直接决定 SpSV 两个 warp knob
填得对不对）。

**已实测（2026-09-04，C550 实机）**：

| 项 | 实测值 | 影响 |
| --- | --- | --- |
| `torch.version.maca` | `'3.8.1.0'` | **存在**，`_detect_maca_runtime()` 第 2 优先级直接命中 |
| `torch.version.cuda` / `.hip` | `'11.6'` / `None` | 证实 MACA 伪装成 CUDA，ROCm 探针分不出来 |
| 设备名 | `'MetaX C550'` | 同时命中 `metax` 厂商串和 `c550` 型号串 |
| **`warp_size`** | **64** | 已回填 `_MACA_SPSV_PROFILES`（见第 8 节） |
| MP count | 104 | |
| max_threads_per_mp | 2048 | |
| regs_per_mp | 131072 | |
| shared_mem_per_block | 65536 | |
| 显存 | 65120 MB | |
| torch | `2.10.0+metax3.8.1.0`（SDK 3.8.2.6 上可用） | 小版本错配不影响 |

**结论：自动探测可用**，`FLAGSPARSE_BACKEND` / `FLAGSPARSE_MACA_MODEL` 都不必显式指定。

### 2.3 确认分发确实走到 metax

```bash
cd <仓库> && export PYTHONPATH=$PWD/src

# 第一次可以先显式指定，确认功能，再取消验证自动探测
export FLAGSPARSE_BACKEND=metax
export FLAGSPARSE_MACA_MODEL=c550

python - <<'EOF'
from flagsparse.sparse_operations import _common as c, spmv_csr as v, spsv as s
print("backend    :", c._backend_name())          # 期望 metax
print("model      :", c._maca_device_model())     # 期望 c550
print("vendor lib :", c._vendor_sparse_library())
print("SpMV kernel:", v._spmv_csr_default_backend())     # segbin（平移自 CUDA）
print("ALG4 persist:", s._spsv_smblk_use_persistent())   # False（CUDA 行为）
for k in ("alg3_warp_size", "alg4_warp_size", "enable_advanced_auto", "cw_serial"):
    print(f"  spsv.{k} = {s._maca_spsv_knob(k)}")
EOF
```

探测按顺序尝试：`FLAGSPARSE_BACKEND` → `torch.version.maca` / `.metax` →
`MACA_PATH` / `MACA_HOME` / `MACA_PATH_CUDA` → 设备名（先认厂商串
`metax/maca/mxc/xcore`，再认型号串 `c550/c500`）。

### 2.4 厂商基线能不能用

MACA 与 CUDA 兼容，CuPy 可能可用也可能不可用：

```bash
python -c "import cupy, cupyx.scipy.sparse as s; print(cupy.__version__); print(s.csr_matrix)"
```

不可用就关掉 —— 基线列会变 `N/A`，但**算子照常运行**，正确性仍由 `torch.sparse` 校验：

```bash
export FLAGSPARSE_MACA_VENDOR=none    # torch（本机默认，因为没装 CuPy）| cupy_cusparse | none
```

> 与 DCU 不同，MetaX 目前**没有**接原生厂商稀疏库（DCU 接的是 hipSPARSE）。
> 将来要接 mcSPARSE，改 `_common._maca_vendor_sparse_library()` 一处即可。

---

## 3. 已验证通过的算子

**921 个用例全绿**（2026-09-05，C550 实机）：

| marker | 用例数 | 备注 |
| --- | --- | --- |
| `spmv_csr` `spmm_csr` `gather` `scatter` | 212 | |
| `spmv_coo` | 77 | |
| `spmv_csc` | 79 | |
| `spmv_bsr` | 271 | |
| `spmv_coo_tocsr` | 7 | |
| `spmm_coo` | 51 | 复数需要 BLOCK_NNZ 钳制，见第 6 节 |
| `spmm_csc` | 55 | |
| `spmm_bsr` | 151 | |
| `spgemm_csr` `sddmm_csr` | 18 | |

一条命令全跑（缓存热时 3–5 分钟）：

```bash
timeout -s KILL 3600 python -m pytest tests/pytest -q \
  -m "spmv_csr or spmv_coo or spmv_csc or spmv_bsr or spmv_coo_tocsr or \
      spmm_csr or spmm_coo or spmm_csc or spmm_bsr or \
      spgemm_csr or sddmm_csr or gather or scatter"
```

分组跑（定位失败更快）：

```bash
for m in spmv_csr spmv_coo spmv_csc spmv_bsr spmv_coo_tocsr \
         spmm_csr spmm_coo spmm_csc spmm_bsr \
         spgemm_csr sddmm_csr gather scatter; do
  printf "%-18s " "$m"
  timeout -s KILL 1800 python -m pytest tests/pytest -q -m "$m" 2>&1 | tail -1
done
```

不需要 GPU 的策略/契约测试：

```bash
python -m pytest tests/ci -q --deselect tests/ci/test_installed_wheel.py
```

> `test_installed_wheel` 断言 `flagsparse` 解析到**仓库树之外**，即真正 `pip install`
> 出来的 wheel。走 `PYTHONPATH=src` 时它必然失败，`pip install -e .` 也一样。
> **不要为了让它变绿去装包** —— site-packages 里的快照会遮蔽后续源码改动，
> 那是个查起来很费时的坑。

---

## 4. 已知跑不了的

| marker | 状态 |
| --- | --- |
| `spsv_csr` | ❌ **崩溃/挂死**，见第 5 节 |
| `spsm_csr` `spsv_coo` `spsv_sell` `spsm_coo` | ⏸ 未验证，很可能撞上同一问题 |
| `alpha_spmm_alg1` | ❌ 缺 TLE。沐曦的 triton 没有 `triton.experimental.tle`，而 FlagOS 的 flagtree（带 TLE）要 GLIBC 2.38，本机是 2.31。只影响这一个算子 |
| `spmm_csr_opt` `opt_alg1` `opt_alg2` | ⏸ 未验证，风险低，可以跑 |
| `spmm_bell` | ⏸ 当前 checkout 里没有用例 |

**跑三角类算子必须套 `timeout -s KILL`** —— 内核挂死时 Ctrl-C 送不进去（进程卡在驱动
调用里），代价是整个容器要重开。

```bash
timeout -s KILL 900 python -m pytest tests/pytest -q -m "spsm_csr"
```

挂住时可以先试串行/非持久化路径，它走的是另一个内核：

```bash
FLAGSPARSE_SPSV_SMBLK_KERNEL=rowprog python -m pytest tests/pytest -q -m "spsv_csr"
```

---

## 5. SpSV 在 C550 上的两个缺陷

`_spsv_csr_cw_kernel`（ALG1 / `csr_cw`）在 MACA 上有两个独立问题。路由背景：
`_spsv_auto_route_non_trans` 把**所有 `unit_diagonal=True`** 的求解无条件送到 `csr_cw`，
而非 unit 的 lower fp32 走 `csr_smblk`（ALG4）—— 所以"非 unit 用例通过"不能说明这个
内核没问题，只有 unit 对角会走到它。

手工构造 CSR 的扫描（n=8、fp32、int32、unit 对角，每个配置独立进程）：

| 结构 | lower | upper |
| --- | --- | --- |
| 仅对角（无依赖） | **非法访存** | OK |
| 双对角（每行 1 个依赖） | **非法访存** | **挂死** |
| 稠密三角（每行 r 个依赖） | **非法访存** | **挂死** |

1. **`LOWER=True` 特化在零依赖时就非法访存。** 单位阵是对称的，lower 和 upper 两次运行
   喂的是完全相同的数据、走完全相同的控制流（第一个元素就是对角线，`stop_at_diag`
   立即成立，依赖分支从不进入），只有 `LOWER`/`REVERSE_ORDER` 不同，而只有 lower 崩。
   MACA 报 `memory violation(0x4): memory access offset is negative, out of bounds, or
   misaligned in kernel`，`trapping: kernelName: _spsv_csr_cw_kernel`。
   这是**编译层面**的差异，不是 Python 侧的索引算错。
2. **ready-flag 自旋不推进**（upper + 任意依赖就挂死）—— 与 DCU/gfx936 相同的故障模式。

**已排除**：`alg3/alg4_warp_size`（32 和 64 表现相同，且 unit 路径根本到不了 ALG3/ALG4）、
`cw_serial=True`（worker_count=1 仍然崩）、矩阵规模（n=2..64 全崩）、
`enable_advanced_auto`（unit 对角下不可达）。

排查这个内核时**必须**加 `CUDA_LAUNCH_BLOCKING=1` —— 非法访存是异步上报的，默认会在
下一次同步（通常是 `allclose`）才抛出，traceback 指向完全无关的位置。

**不要用 CPU 求解顶替。** 2026-09-18 曾在 `_execute_spsv_csr_plan()` 前试过
`_maca_spsv_cpu_fallback()`（拷到 CPU、调 `scipy.sparse.linalg.spsolve_triangular()`、再拷回），
已完全撤销：被测的 GPU 算子被换成了 CPU 计算，精度结论不反映 C550 内核，耗时里又混进了
CPU 求解和 host/device 拷贝。设备内核崩溃或挂死时，结果应如实记为 `Error` / `Timeout` /
`NotFound`。精度测试里的 SciPy **参考解**（`tests/pytest/accuracy_utils.py` 的
`scipy_triangular_solve()`）是另一回事：它只算 oracle，被测算子仍在设备上跑。

---

## 6. SpMM COO 复数：私有内存 4 KB 上限

复数 SpMM COO 曾经 **24 个用例全失败**（complex64/complex128 的完整叉积），报
`Triton Error [MACA]: memory size or pointer value too large to fit in 32 bit`。
驱动的原始信息说清了原因：

```
private memory size required in the kernel is greater than the maximum value set by
the system ... The system is set to: 4 KB/Thread, kernel request: 8 KB/thread
You can change insmod metax.ko by specifying parameters: pri_mem_sz=XXX
```

`_spmm_coo_rowrun_*_kernel` 里 `tl.static_range(0, BLOCK_NNZ)` 的 BLOCK_NNZ 是 constexpr，
会被展开那么多次，直接放大每线程私有内存。公开默认值是 **256**，复数带实部+虚部是实数的
两倍，正好落在 8 KB vs 4 KB 上限。实数还塞得下，所以实/复分界才那么整齐。

**已修**：`_resolve_spmm_coo_launch_config` 里加了 MACA+复数的钳制
（`_MACA_SPMM_COO_COMPLEX_BLOCK_NNZ = 4`），并新增 `value_dtype=` 参数由调用点传入
`canonical_data.dtype`。实数和不传 dtype 的调用仍然解析到 256，其他后端行为不变。
结果：**51/51 通过，整组耗时从 23m48s 降到 11.8s** —— 256 的展开开销就有这么大。

如果别处再遇到同类报错，另外两条路：宿主侧抬高上限
（`insmod metax.ko pri_mem_sz=8192`，需要平台配合且影响整机），或把公开默认值改成
`None` 让已经实测过的 4 全局生效（30 矩阵扫描测出的最优值就是 4，256 是被测量否定过的
旧值，只是当初调优没改到函数签名的默认参数）。

---

## 7. 基准测试

正确性全绿之后再看性能。注意 `FLAGSPARSE_MACA_VENDOR=none` 时厂商基线列是 `N/A`，
那不代表失败，去 `reason` / `cusparse_reason` 字段看原因。表头里
`cuSPARSE(ms)` / `CSR(ms)` / `CS(ms)` 这一列就是厂商基线列（列名沿用历史命名）。

```bash
M=tests/data/trdheim.mtx
python tests/test_spmv.py     $M --warmup 2 --iters 5
python tests/test_spmm.py     $M --warmup 2 --iters 5
python tests/test_spgemm.py   $M --warmup 2 --iters 5
python tests/test_spmm_coo.py $M --warmup 2 --iters 5
python tests/test_gather.py   --value-dtypes float32
python tests/test_scatter.py  --value-dtypes float32
```

### 7.1 统一 runner：精度 + 性能一条命令

排除 SpSV/SpSM 和三个 `spmm_csr_opt*` 之后，跑其余全部算子：

```bash
OPS=gather,scatter,spmv_csr,spmv_coo,spmv_csc,spmv_bsr,spmv_coo_tocsr,spmm_csr,spmm_coo,spmm_bsr,spmm_bell,spmm_csc,spgemm_csr,sddmm_csr

timeout -s KILL 7200 python run_flagsparse_pytest.py \
  --phase both --mode normal --timeout 900 \
  --benchmark-input tests/data \
  --ops "$OPS"
```

四个参数都不能用默认值，原因各不相同：

| 参数 | 默认 | 为什么必须显式给 |
| --- | --- | --- |
| `--ops` | 读 `conf/operators.yaml` | 默认 sweep 含五个求解器算子，在 C550 上会挂死（第 4、5 节） |
| `--mode` | **`quick`** | 见下，quick 是假绿 |
| `--timeout` | `0`（关闭） | 挂住就永远不往下走。注意它是**每个算子每个阶段**的超时，不是全局 |
| `--phase` | `accuracy` | 要性能数据得给 `both` |

**`--mode` 的默认值是个坑**：直接跑 pytest 时 `conftest.py` 默认 `normal`，而 runner
默认 `quick`，两条路径相反。quick 会把 `param_shapes.py` 每类 shape 砍到只剩一个，
覆盖少掉约四成，而且**恰好跳过历史上出过问题的两个用例** ——
`SPMV_MN_SHAPES` 只剩 `(1, 32)`（没有 `160-1024`），
`SPGEMM_MNK_SHAPES` 只剩 `(8, 10, 6)`（没有 `48-64-32`）。用 quick 跑出来的全绿不算数。

| marker | normal | quick |
| --- | --- | --- |
| `spmv_csc` | 79 | 31 |
| `spmv_csr` / `spmv_coo` | 77 / 77 | 29 / 29 |
| `spmm_csr` | 59 | 31 |
| `spmm_coo` | 51 | 27 |
| `spgemm_csr` | 9 | 5 |
| `gather` / `scatter` | 26 / 50 | 14 / 26 |
| `spmv_bsr` / `spmm_bsr` / `spmm_bell` / `spmm_csc` | 271 / 151 / 54 / 153 | 不变 |

`--benchmark-input` 要的是**目录**；`tests/data/` 里的三个小矩阵只够打通流程，
真实性能数字要换成自己的矩阵集。

runner 的精度阶段是**一个算子起一个 pytest 进程**（`-m <算子名> -p no:cacheprovider`），
比第 3 节那条大 pytest 更安全 —— 某个内核把 runtime 弄挂了不会污染后面的算子。
代价是它按算子名选 marker，所以 `test_spmm_csr_opt_matches_torch`（marker 是
`spmm_csr_opt`）这类用例不在覆盖内。

### 7.2 C550 全量：精度 + 性能，PyTorch 基线

性能基线使用 PyTorch，热身 5 次、迭代 20 次；SpSV、SpSM 以及没有性能入口的算子不在本轮：

```bash
PYTHONPATH=src python -u run_flagsparse_pytest.py --phase both --mode quick --gpus 0 \
  --ops gather,scatter,spmv_csr,spmv_coo,spmv_csc,spmv_bsr,spmm_csr,spmm_coo,spmm_bsr,spmm_csc,spgemm_csr,sddmm_csr \
  --benchmark-input /root/gcx/matrix --benchmark-warmup 5 --benchmark-iters 20 \
  --benchmark-args=--no-cusparse --op-benchmark-args=spmv_bsr=--resume \
  --timeout 7200 --results-dir pytest_results_metax_runner_both_w5_i20
```

结果写入该目录，包括各算子的精度结果、性能 CSV、规范化性能 JSON 和根目录汇总文件。

其中 `spmm_csc` 在 MACA 上使用直接 PyTorch CSC 作为性能 baseline：
`torch.sparse_csc_tensor` + `torch.sparse.mm`，CSC 格式构造不计入计时，CSV 字段为
`pytorch_ms` 和 `triton_speedup_vs_pytorch`。精度参考仍是同一 CSC 数据转 COO 后的
`torch.sparse.mm`；`--no-cusparse` 只禁用 CuPy/厂商 CSC baseline，不影响 PyTorch CSC baseline。
`trans/conj` 的有效 CSC 准备过程同样在计时窗口之外。

`--benchmark-args` 是传给所有性能脚本的字符串，runner 通过 `shlex.split()` 展开。只由
单个性能脚本支持的参数使用可重复的 `--op-benchmark-args=算子名=参数`；参数部分含空格时
才需要整体引用。上面的全量命令只向 BSR 传入 `--resume`。`spmv_bsr` 若被 7200 秒超时中断，
可复用同一结果目录续跑：

```bash
PYTHONPATH=src python -u run_flagsparse_pytest.py --phase performance --gpus 0 \
  --ops spmv_bsr --benchmark-input /root/gcx/matrix \
  --benchmark-warmup 5 --benchmark-iters 20 \
  --benchmark-args="--no-cusparse --resume" --timeout 7200 \
  --results-dir pytest_results_metax_spmv_bsr
```

`--resume` 保留 CSV 中已经完成的 `PASS`/`FAIL` case，丢弃并重试 `ERROR` case；仅精度
`PASS` 且 PyTorch 与 FlagSparse 时延完整的行会写入和汇总 `bsr_speedup_vs_pytorch`。

### 7.3 SDDMM（fp32 / fp64）全量

`sddmm_csr` 当前只支持 `float32`、`float64`，全量性能范围是 30 个矩阵、两种 dtype
及 `K=32,64,128,256`。MACA PyTorch 的 `torch.sparse.sampled_addmm` 虽能调用，但其
sampled-dot 输出不正确，不能作为 SDDMM 的精度参考或性能 baseline。因此 C550 上必须传入
`--no-cusparse`，SDDMM 脚本会使用独立的、同 dtype PyTorch 参考
`sum(X[row] * Y[col])`：它同时是精度 oracle 和性能 baseline。CSV 的有效字段为
`pytorch_ms` 与 `triton_speedup_vs_pytorch`；仅精度 `PASS` 且两侧时延有效的行参与加速比汇总。

```bash
PYTHONPATH=src python -u run_flagsparse_pytest.py --phase both --mode normal --gpus 0 \
  --ops sddmm_csr --benchmark-input /root/gcx/matrix \
  --benchmark-warmup 5 --benchmark-iters 20 --benchmark-args=--no-cusparse \
  --timeout 7200 --results-dir pytest_results_metax_sddmm_csr_pytorch_full_w5_i20
```

该命令在前台运行；如需脱离终端，可由调用方以 `setsid` 或 `tmux` 包裹，命令本身不依赖后台参数。

### 7.4 交付性能：只跑交付范围（2026-09-18 实测）

7.1–7.3 是**全量** sweep。交付的 40 个变体只要 `int32` 索引和 `non` 操作，而默认 sweep 远大于此，
`--timeout 900`（每个父算子每个阶段）下实测跑不完：

| 父算子 | 默认 CSV sweep | 900 秒内进度 | 推断全量耗时 |
|---|---|---|---|
| `sddmm_csr` | 2 dtype × 4 K × 30 = 240 组 | 59/240 | ≥ 3660 秒 |
| `spmm_csr` | 4 dtype × 2 index × 3 op × 30 = 720 组 | 252/720 | ≥ 2570 秒 |

不是单个矩阵挂死，是 sweep 本身太大。把 sweep 收窄到交付范围，而不是盲目加 timeout。

**现在 `--delivery-only` 会自动收窄**（2026-09-18 起，见 `run_flagsparse_pytest.py` 的
`DELIVERY_BENCHMARK_ARGS`）：spmv/spmm/spsv 只跑 `int32` + `non`，gather/scatter 只跑 `int32`
和交付 dtype，启动时每个被收窄的算子会打印一行 `delivery-only: <op> benchmark narrowed with ...`。
交付测试的完整命令见 **0.5 节**。注意 `sddmm_csr` 的 K sweep（32/64/128/256）**不在**自动收窄范围内：
交付名里没有 K，自动砍掉会改变报出来的均值口径。所以 0.5 节用 `--timeout 4500`；要只跑一个 K，显式传
`--op-benchmark-args='sddmm_csr=--k 64'`，并在报告里注明。

下面是自动收窄之前本机实际跑的命令（手写 `--op-benchmark-args`），结果即出自它：

```bash
setsid env PYTHONPATH="$PWD/src" FLAGSPARSE_BACKEND=metax FLAGSPARSE_MACA_VENDOR=none \
  timeout -s KILL 3600 python3 -u run_flagsparse_pytest.py \
  --phase performance --mode normal --delivery-only --gpus 0 --timeout 1200 \
  --ops spmm_csr,sddmm_csr \
  --benchmark-input /root/gcx/matrix --benchmark-warmup 5 --benchmark-iters 20 \
  --benchmark-args=--no-cusparse \
  --op-benchmark-args='spmm_csr=--dtypes float32,float64,complex64,complex128 --index-dtypes int32 --ops non' \
  --op-benchmark-args='sddmm_csr=--dtype float32,float64 --index-dtype int32 --k 64' \
  --results-dir pytest_results_metax_delivery_perf_remaining_w5_i20 \
  > pytest_results_metax_delivery_perf_remaining_w5_i20/runner.log 2>&1 < /dev/null &
```

结果：`spmm_csr` 120 行（4 dtype × 30 矩阵）、`sddmm_csr` 60 行（2 dtype × 30 矩阵），两项都
`Passed`，没有触发 1200 秒或 3600 秒超时：

| 交付变体 | 矩阵数 | 加速比（**对 PyTorch**，CSV `base/gems` 均值） |
|---|---:|---:|
| `sddmm_csr_f32_int_non_non_row` | 30 | 7.45x |
| `sddmm_csr_f64_int_non_non_row` | 30 | 5.92x |
| `spmm_csr_f32_int_non_non_row` | 30 | 20.88x |
| `spmm_csr_f64_int_non_non_row` | 30 | 13.81x |
| `spmm_csr_c32_int_non_non_row` | 30 | 10.24x |
| `spmm_csr_c64_int_non_non_row` | 30 | 5.14x |

**这些数的分母是 PyTorch，不是厂商稀疏库**（`--no-cusparse`，C550 上没有可用的 mcSPARSE
基线），不能和 CUDA/MUSA 那些对 cuSPARSE/muSPARSE 的数字放在一起比。

仍要跑默认全量 sweep 的话，两者分开跑：`sddmm_csr --timeout 4500`、`spmm_csr --timeout 3600`，
外层 `timeout -s KILL` 要大于对应的单项 timeout；不要把两者串在一条 7200 秒的外层命令里，
否则外层会先杀掉后一个。

---

## 8. 调优 A/B —— 这一步才是 metax 后端的价值所在

当前 profile 除 `warp_size` 外是**从 CUDA 平移的占位值**，没有 C550 实测依据。

```bash
# SpMV：CUDA 的 segbin vs DCU 的 rowpar，哪个更适合 C550
FLAGSPARSE_SPMV_CSR_KERNEL=segbin python tests/test_spmv.py <dir/> --csv-csr c550_segbin.csv
FLAGSPARSE_SPMV_CSR_KERNEL=rowpar python tests/test_spmv.py <dir/> --csv-csr c550_rowpar.csv

# SpSV ALG4：一行一 program vs 持久化 worker
FLAGSPARSE_SPSV_SMBLK_KERNEL=rowprog    python tests/test_spsv.py --synthetic
FLAGSPARSE_SPSV_SMBLK_KERNEL=persistent python tests/test_spsv.py --synthetic
```

结果回填到两个 profile 表：

| 位置 | 参数 | 当前值（平移自 CUDA） | ROCm 实测值（gfx936，仅供参考） |
| --- | --- | --- | --- |
| `spmv_csr.py` `_MACA_SPMV_PROFILES["c550"]` | `csr_kernel` | `"segbin"` | `"rowpar"` |
| `spsv.py` `_MACA_SPSV_PROFILES["c550"]` | `smblk_persistent` | `False` | `True` |
| | `enable_advanced_auto` | `True` | `False`（强制 ALG1） |
| | `alg3_warp_size` | ~~`32`~~ → **`64`** ✅ | `64` |
| | `alg4_warp_size` | ~~`32`~~ → **`64`** ✅ | `64` |
| | `cw_serial` | `False` | `True` |

**`alg3/alg4_warp_size` 已按实测的 `warp_size == 64` 改掉**（2026-09-04）。
其余四个 knob 仍是 CUDA 平移值，需要按上面的 A/B 命令实测后回填 —— 它们是行为选择
而非硬件事实，不能靠指纹推断。

新增型号（如 C500）只要在这两个字典里加一个 key，`_maca_device_model()` 会自动选中；
识别不出的型号回落到 `c550` 档。

---

## 9. 排查速查表

| 现象 | 首先检查 |
| --- | --- |
| `backend` 显示 `cuda` 而非 `metax` | `flagsparse.__file__` 是否指向 `src/`；再用 `FLAGSPARSE_BACKEND=metax` 显式指定，并把 2.2 的指纹反馈回来 |
| 新符号找不到、基线静默变 `N/A` | 同上，多半跑到了已安装的旧副本 |
| Triton 编译报错 / 找不到后端 | 2.1，FlagTree metax 后端是否装好 |
| 基线列全是 `N/A` | CuPy 是否可用（2.4）；再看 `reason` 字段 |
| `memory size or pointer value too large to fit in 32 bit` | 私有内存超 4 KB/线程，见第 6 节；调小 BLOCK_NNZ |
| `memory violation(0x4) ... offset is negative` | 非法访存。加 `CUDA_LAUNCH_BLOCKING=1` 重跑才能定位到真正的内核 |
| **SpSV / SpSM 挂住不动、Ctrl-C 无效** | 内核死锁（第 5 节），只能等 `timeout -s KILL` 或重开容器。先试 `FLAGSPARSE_SPSV_SMBLK_KERNEL=rowprog`。**跑三角类算子永远套 timeout** |
| `libmcruntime.so` / `libnuma.so.1` 找不到 | 环境变量丢了（换容器会丢），重跑第 1 节；`ldd .../torch/lib/*.so \| grep "not found"` 一次列全 |
| SpMV/SpSV 性能明显偏低 | profile 是 CUDA 平移值，尤其确认 `warp_size`（第 8 节） |
| 单次失败、重跑就好 | 本机偶发失败率不低，已观察到多次。**重要结论都要多跑几轮**，单次结果不算数 |
