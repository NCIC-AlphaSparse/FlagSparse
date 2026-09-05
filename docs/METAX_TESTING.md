# MetaX（MACA / 曦云 C550）测试指南

本文针对 FlagSparse 的 **MetaX 后端分发**。CUDA 与 DCU 两条路径已有实机验证，
**MetaX 这条一行都没有在真机上跑过**——代码是从 CUDA 路径平移的，只做了导入、
语法和分发路径检查（在 NVIDIA 机器上用 `FLAGSPARSE_BACKEND=metax` 模拟）。
本文的目的就是把它跑起来并把调优参数换成真实测量值。

---

## 0. 一句话背景

FlagSparse 按运行时分发**厂商参考实现/基线**和**少数按后端分叉的内核**：

| 运行时 | 判定 | 厂商稀疏库 | 内核起点 |
| --- | --- | --- | --- |
| NVIDIA CUDA | `torch.version.hip is None` | cuSPARSE（CuPy） | segbin / 一行一 program |
| DCU / ROCm | `torch.version.hip is not None` | hipSPARSE（hip-python） | rowpar / 持久化 worker |
| **MetaX / MACA** | 见第 2 节 | 暂用 CuPy 兼容路径 | **平移自 CUDA** |

**MetaX 与 CUDA 和 DCU 最大的不同：MACA 与 CUDA 源码兼容**，机器上
`torch.version.cuda` 有值、`torch.version.hip` 为 `None`，
**靠现有判据分不出 MetaX 和 NVIDIA**——所以检测是这条路径的首要风险点。

---

## 1. ⚠️ 第 0 步：确认 FlagTree 的 metax 后端可用

FlagSparse 的算子靠 Triton 编译。C550 上必须是 **FlagTree 的 metax 后端**，
这一步不通，后面所有报错都会指向错误的方向。

```bash
python -c "import triton; print(triton.__version__)"
python -c "import triton.backends as b; print(list(b.backends.keys()))"   # 期望含 metax
```

若没有 metax，按 FlagTree 文档安装（`FLAGTREE_BACKEND=metax`）。依赖 `metax-llvm`（19）、
`metaxTritonPlugin`、MACA 数学库；其中 maca-llvm 需向沐曦索取。

装好后**跑一个最小 Triton kernel 确认真能编译执行**，再往下走：

注意 `@triton.jit` 必须定义在**真实的 .py 文件**里（Triton 要读源码），
不能写在 `python - <<EOF` 的 stdin 里，否则会报
`ValueError: @jit functions should be defined in a Python file`：

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
> 也会出现）。FlagSparse 不使用 TLE-RAW；确认 metax 后端本身可用即可，
> 或安装该依赖后重试。

---

## 2. ⚠️ 采集环境指纹（最重要的一步）

自动检测逻辑是**按常见模式推测**的，尚未经真机确认。请先采集，
并把输出反馈给维护者以修正探测：

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

三个关键点：

1. `torch.version` 有没有 `maca` / `metax` 属性；
2. **设备名的确切字符串**（决定设备名探测能否命中）；
3. **`warp_size` 是 32 还是 64** —— 直接决定 SpSV 两个 warp knob 填得对不对。

### ✅ 已实测（2026-09-04，C550 实机）

| 项 | 实测值 | 影响 |
| --- | --- | --- |
| `torch.version.maca` | `'3.8.1.0'` | **存在**，`_detect_maca_runtime()` 第 2 优先级直接命中 |
| `torch.version.cuda` / `.hip` | `'11.6'` / `None` | 证实 MACA 伪装成 CUDA，ROCm 探针分不出来 |
| 设备名 | `'MetaX C550'` | 同时命中 `metax` 厂商串和 `c550` 型号串 |
| **`warp_size`** | **64** | 已回填 `_MACA_SPSV_PROFILES`（见第 7 节） |
| MP count | 104 | |
| max_threads_per_mp | 2048 | |
| regs_per_mp | 131072 | |
| shared_mem_per_block | 65536 | |
| 显存 | 65120 MB | |
| torch | `2.10.0+metax3.8.1.0`（SDK 3.8.2.6 上可用） | 小版本错配不影响 |

**结论：自动探测可用**，`FLAGSPARSE_BACKEND` / `FLAGSPARSE_MACA_MODEL` 都不必显式指定。

---

## 3. 确认分发确实走到 metax

```bash
cd <仓库> && export PYTHONPATH=$PWD/src

# 先显式指定，不要一上来就依赖自动探测
export FLAGSPARSE_BACKEND=metax
export FLAGSPARSE_MACA_MODEL=c550

python -c "import flagsparse; print(flagsparse.__file__)"   # 必须指向 <仓库>/src/flagsparse/__init__.py

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

然后**把 `FLAGSPARSE_BACKEND` 取消再跑一次**，看自动探测能否自己认出 metax。
认不出就继续显式指定，功能完全不受影响。

检测按顺序尝试：`FLAGSPARSE_BACKEND` → `torch.version.maca` / `.metax` →
`MACA_PATH` / `MACA_HOME` / `MACA_PATH_CUDA` → 设备名（先认厂商串
`metax/maca/mxc/xcore`，再认型号串 `c550/c500`）。

---

## 4. 厂商基线能不能用

MACA 与 CUDA 兼容，CuPy 可能可用也可能不可用：

```bash
python -c "import cupy, cupyx.scipy.sparse as s; print(cupy.__version__); print(s.csr_matrix)"
```

不可用就关掉——基线列会变 `N/A`，但**算子照常运行**，正确性仍由 `torch.sparse` 校验：

```bash
export FLAGSPARSE_MACA_VENDOR=none    # cupy_cusparse | none
```

> 与 DCU 不同，MetaX 目前**没有**接原生厂商稀疏库（DCU 接的是 hipSPARSE）。
> 将来要接 mcSPARSE，改 `_common._maca_vendor_sparse_library()` 一处即可。

---

## 5. 正确性套件

```bash
python -m pytest tests/ci -q      # 不需要 GPU，期望 45 passed / 3 skipped

python -m pytest tests/pytest -q -m "spmv_csr or spmm_csr or gather or scatter"
python -m pytest tests/pytest -q -m "spgemm_csr or sddmm_csr"
```

⚠️ **SpSV / SpSM 单独跑，并准备好 Ctrl-C**：

```bash
python -m pytest tests/pytest -q -m "spsv_csr or spsm_csr"
```

这两个算子**在 DCU 上是 GPU 内核死锁的**（跨 program 的 ready-flag 轮询在 gfx936
上不能可靠推进）。C550 的原子操作与内存序行为未知，同样的风险存在。若挂住，先试
串行/非持久化路径：

```bash
FLAGSPARSE_SPSV_SMBLK_KERNEL=rowprog python -m pytest tests/pytest -q -m "spsv_csr"
```

第一次跑请以"**能否跑完**"为验收标准，不要先看性能数字。

---

## 6. 逐算子基准

```bash
M=tests/data/trdheim.mtx
python tests/test_spmv.py    $M --warmup 2 --iters 5
python tests/test_spmm.py    $M --warmup 2 --iters 5
python tests/test_spgemm.py  $M --warmup 2 --iters 5
python tests/test_spmm_coo.py $M --warmup 2 --iters 5
python tests/test_gather.py   --value-dtypes float32
python tests/test_scatter.py  --value-dtypes float32
```

表头里 `cuSPARSE(ms)` / `CSR(ms)` / `CS(ms)` 这一列是**厂商基线列**（列名沿用历史命名）。
在 MetaX 上它装的是 CuPy 兼容路径的数，或在 `FLAGSPARSE_MACA_VENDOR=none` 时为 `N/A`。
`N/A` 不代表失败，去 `reason` / `cusparse_reason` 字段看原因。

---

## 7. 调优 A/B —— 这一步才是 metax 后端的价值所在

当前 profile 是**从 CUDA 平移的占位值**，没有任何 C550 实测依据。

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

## 8. 排查速查表

| 现象 | 首先检查 |
| --- | --- |
| `backend` 显示 `cuda` 而非 `metax` | 自动探测没命中，用 `FLAGSPARSE_BACKEND=metax` 显式指定，并把第 2 节的指纹反馈回来 |
| Triton 编译报错 / 找不到后端 | 第 1 节，FlagTree metax 后端是否装好 |
| 基线列全是 `N/A` | CuPy 是否可用（第 4 节）；再看 `reason` 字段 |
| 新符号找不到、基线静默变 `N/A` | `flagsparse.__file__` 是否指向 `src/`（用 `PYTHONPATH=src`） |
| **SpSV / SpSM 挂住不动** | 已知风险，见第 5 节；先试 `FLAGSPARSE_SPSV_SMBLK_KERNEL=rowprog` |
| SpMV/SpSV 性能明显偏低 | profile 是 CUDA 平移值，尤其确认 `warp_size`（第 7 节） |

---

## 9. 风险提示：哪些代码真的没跑过

MetaX 这条路径**整条都没有在真机上执行过**。相对而言：

- ~~**检测逻辑**（`_detect_maca_runtime` / `_maca_device_model`）——纯推测~~
  **已在 C550 实机验证通过**（2026-09-04，见第 2 节）：`torch.version.maca` 存在，
  设备名 `'MetaX C550'` 也命中，两条探测路径都能自动认出来；
- **调优 profile**——占位值，无任何 C550 依据；
- **算子本身**——风险最低。Triton 内核体两个后端同源，MetaX 走的就是 CUDA 那份
  已经在 NVIDIA 上验证过的代码，只要 FlagTree metax 后端能正确编译执行就应当可用。

**已排除、不必怀疑的**：CUDA 与 DCU 两条既有路径没有被本次改动触碰
（`_is_maca_runtime()` 在两者上均为 `False`，所有 metax 分支不进入）；
在 NVIDIA 机器上以 `FLAGSPARSE_BACKEND=metax` 模拟跑过 553 个定向用例并实跑
SpMV/SpMM/SpSV，结果与 CUDA 路径一致。
