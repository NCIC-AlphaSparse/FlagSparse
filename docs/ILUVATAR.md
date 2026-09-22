# 天数智芯 Iluvatar CoreX（BI-V150）

Python 侧的天数后端：怎么选中、每次开工的环境、交付复现命令。C API 那一层（`BACKEND=IX`）
目前编不起来，见第 5 节；各后端文档的对应关系见 [README.md](README.md)。

> ⚠️ **本后端还没有在 BI-V150 实机上跑过。** 下面的探测逻辑、基线选择和命令都是按
> "CUDA 兼容栈"（和 MetaX 同类）接入的，第一次上机请按第 1 节逐项确认，把实测结果补回本文。

---

## 0. 接入方式：和 MetaX 一样是 CUDA 兼容栈

后端名是 **`iluvatar`**（与 FlagTree 的后端名一致），占用的是原来寒武纪 `mlu` 备用槽位的位置。

| 项目 | 取值 |
|---|---|
| torch 设备 | `torch.cuda` / 设备类型 `cuda`（CoreX 的 torch 构建与 CUDA 源码兼容，没有独立命名空间和插件） |
| 自动探测 | torch 版本号带 `+corex` 标记，或设备名含 `iluvatar` / `bi-v` / `corex`（`_detect_iluvatar_runtime()`） |
| 显式指定 | `FLAGSPARSE_BACKEND=iluvatar`，**优先于探测** |
| 算子内核 | 共享实现，`backends/iluvatar/` 下只有 `__init__.py`，没有覆盖 |
| 性能基线 | 交付跑用 **`torch`**（第 1 节显式设 `FLAGSPARSE_ILUVATAR_VENDOR=torch`），与其余国产后端同口径。不设这个变量时是探测：CuPy 真装了就用 `cupy_cusparse`，否则 `torch`（同 MetaX 的策略，**未实测**） |
| 精度参考 | CPU 上的 SciPy（CUDA、ROCm 以外的后端都是这样） |
| runner 路由 | 与 CUDA / MetaX 相同的通用性能脚本（`GENERIC_BENCHMARK_BACKENDS`） |

因为 `torch.version.cuda` 有值、`torch.version.hip` 为 `None`，**只看这两个判据分不出天数和
NVIDIA**。探测依赖的 `+corex` 版本标记和设备名都还没在真机上核对过，所以第 1 节显式指定后端，
并打印真实设备名。

---

## 1. 每次开工的环境

```bash
cd <仓库>

# CoreX SDK：按厂商安装说明设置（常见安装位置是 /usr/local/corex，以本机为准）
export COREX_HOME=/usr/local/corex
export PATH=$COREX_HOME/bin:$PATH
export LD_LIBRARY_PATH=$COREX_HOME/lib64:$COREX_HOME/lib:$LD_LIBRARY_PATH

# FlagSparse：后端和基线全部显式指定
export PYTHONPATH=$PWD/src                  # 独立脚本需要；pytest 由 pytest.ini 自带
export FLAGSPARSE_BACKEND=iluvatar
export FLAGSPARSE_ILUVATAR_VENDOR=torch     # 固定用 PyTorch 作基线；不设则装了 CuPy 就会改用 CuPy，两者口径不同。不要设 none，否则基线列全是 N/A
unset FLAGSPARSE_ACCURACY_REFERENCE         # 用默认的 auto（CPU 上的 SciPy），防止上次调试残留的设置
```

**开跑前的检查**（全部对上再往下）：

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)"
# 期望：版本号带 +corex（例如 2.x.y+corex.<SDK 版本>）、cuda 有值、hip 为 None
python3 -c "import triton; print(triton.__version__); import triton.backends as b; print(list(b.backends))"
# 期望：列表里有 iluvatar（FlagTree 的天数后端）
python3 -c "import flagsparse; print(flagsparse.__file__)"
# 期望：<仓库>/src/flagsparse/__init__.py；指到 site-packages 就是跑错了副本
python3 -c "import torch; print(torch.cuda.get_device_properties(0).name); from flagsparse.sparse_operations import _common as c; print(c._backend_name(), c._accel_device_type(), c._accel_fallback_reason(), c._vendor_sparse_library())"
# 期望：Iluvatar BI-V150（或类似的天数设备名）
#       iluvatar cuda None torch
```

显式指定会跳过探测，所以要看**真实设备名**：只有它确实是天数的卡，`FLAGSPARSE_BACKEND=iluvatar` 才对。
第一次上机时，建议再**去掉** `FLAGSPARSE_BACKEND` 跑一次最后那条检查，看自动探测是否也得到 `iluvatar`，
并把 `torch.__version__`、设备名、`warp_size` 记到第 4 节。

---

## 2. 交付复现：20 个变体 × 10 个矩阵（精度 + 性能）

第 1 节的环境和检查通过后：

```bash
setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --phase both --mode normal --delivery-only --gpus 0 --timeout 3600 \
  --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir pytest_results_iluvatar_delivery \
  > pytest_results_iluvatar_delivery.log 2>&1 < /dev/null &
```

- `--benchmark-input tests/data`：10 个交付矩阵已经在仓库的 `tests/data` 里。跑完先确认筛选生效了：
  `grep "delivery-only" pytest_results_iluvatar_delivery.log` 应当是 `matrices from .../delivery_matrices`，
  出现 `NOT applied` 就说明矩阵不全，跑的不是交付集合；
- 外层 `timeout -s KILL 43200`（12 小时）是整条命令的总限时，内核卡死时 Ctrl-C 送不进去，只能靠 KILL；
  `--timeout 3600` 是每个算子每个阶段的限时。20 个变体在天数上的完整耗时**没有实测过**，按实际情况调整；
- `--gpus 0`：runner 通过 `CUDA_VISIBLE_DEVICES` 选卡。CoreX 是否照常遵守这个变量**未验证**，
  多卡机器上先用 `ixsmi`（天数的设备管理工具）确认任务确实落在指定的卡上。

跑完看 20 行结果（缺变体时退出码为 1），回传时直接贴它的输出：

```bash
python3 tools/delivery_table.py pytest_results_iluvatar_delivery    # 加 --markdown 输出 Markdown 表
```

也可以通过按后端组织的入口跑，它会自动设置 `FLAGSPARSE_BACKEND=iluvatar`：

```bash
python tools/run_backend_tests.py --backend iluvatar --phase accuracy --mode quick
```

---

## 3. 性能基线

**交付跑固定用 PyTorch**：第 1 节设了 `FLAGSPARSE_ILUVATAR_VENDOR=torch`，这样加速比的分母和
沐曦、昇腾、昆仑芯几个后端一致，可以横向比。

不设这个变量时走的是探测（照 MetaX 定的策略）：CuPy 真装了就用 CuPy，否则 PyTorch。所以**同一台机器
装没装 CuPy 会给出两种口径的加速比**，这也是交付跑要显式指定的原因。设成 `none` 则没有基线列，全是 `N/A`。

这套默认值的理由是 CoreX 与 CUDA 兼容、`torch.sparse` 应当能跑，**但没有实测过**。摩尔线程的教训是
"应当能跑"不等于能跑（那里 `torch.sparse` 能建张量却没有 matmul）。第一次上机时如果基线列是 `N/A`，
看 `reason` 字段：若是 `torch.sparse` 本身报错，就把 `_iluvatar_vendor_sparse_library()` 的默认值改成
`None`，并把实测记到这里。

---

## 4. 待实测（第一次上机后补）

| 项目 | 状态 |
|---|---|
| `torch.__version__` 是否带 `+corex` | 未验证 |
| 设备名（`get_device_properties(0).name`） | 未验证，探测按 `iluvatar` / `bi-v` / `corex` 匹配 |
| `warp_size` | 未验证；内核的启动参数按设备属性推导，但若不是 32，需要逐项检查按 32 写死的调优参数（MetaX C550 是 64，见 [MACA.md](MACA.md)） |
| FlagTree 的 `iluvatar` 后端能否编译执行最小 Triton 内核 | 未验证 |
| `torch.sparse` CSR/COO matmul 能否作为基线 | 未验证 |
| `CUDA_VISIBLE_DEVICES` 选卡 | 未验证 |
| 20 变体交付结果 | 无 |

---

## 5. C API（`BACKEND=IX`）

C API 侧早就为天数预留了 `IX` 槽位（`COREX_HOME`，默认 `/usr/local/corex`），`deps/libtriton_jit`
也认 `IX`，但它**没有提供 IX 的 Backend 模块**，所以 C API 目前编不起来：CTest 的 `iluvatar` profile
（`capi/ctest/backends/iluvatar.cmake`）标的是 `CAPI_BUILDABLE OFF`。Python 侧不受影响。
