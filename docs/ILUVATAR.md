# 天数智芯 Iluvatar CoreX（BI-V100 / BI-V150）

Python 侧的天数后端：怎么选中、每次开工的环境、交付复现命令。C API 那一层（`BACKEND=IX`）
目前编不起来，见第 5 节；各后端文档的对应关系见 [README.md](README.md)。

> ⚠️ **算子一行都还没在天数卡上跑过。** 下面的探测逻辑、基线选择和命令都是按
> "CUDA 兼容栈"（和 MetaX 同类）接入的。第一次上机按 **1.5 节的验证顺序**逐步做，
> 把实测结果补回第 4 节那张表。
>
> **2026-09-22 的上机尝试卡在环境上**：BI-V100 × 8 的机器（`u98`），宿主机驱动 3.2.3，
> 但厂商镜像自带 CoreX 4.4.0/4.5.0，torch 初始化报
> `Error 803: system has unsupported display driver / cuda driver combination`，
> `torch.cuda.is_available()` 为 `False`（`device_count()` 却是 8，这个组合容易误判成可用）。
> 这是驱动/运行时版本错配，**要么换配 3.2.3 的镜像，要么把宿主机驱动升到 4.x**，
> 容器内部解决不了。详见第 4 节。

---

## 0. 接入方式：和 MetaX 一样是 CUDA 兼容栈

后端名是 **`iluvatar`**（与 FlagTree 的后端名一致），占用的是原来寒武纪 `mlu` 备用槽位的位置。

| 项目        | 取值                                                                                                                                                                                                                                |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| torch 设备  | `torch.cuda` / 设备类型 `cuda`（CoreX 的 torch 构建与 CUDA 源码兼容，没有独立命名空间和插件）                                                                                                                                   |
| 自动探测    | 三个信号，任一命中即可（`_detect_iluvatar_runtime()`）：torch 版本带 `+corex` 标记（`torch.__version__` **和** pip 元数据都查）、`COREX_HOME` / `COREX_PATH` 环境变量、设备名含 `iluvatar` / `bi-v` / `corex` |
| 显式指定    | `FLAGSPARSE_BACKEND=iluvatar`，**优先于探测**                                                                                                                                                                               |
| 算子内核    | 共享实现，`backends/iluvatar/` 下只有 `__init__.py`，没有覆盖                                                                                                                                                                   |
| 性能基线    | 交付跑用**`torch`**（第 1 节显式设 `FLAGSPARSE_ILUVATAR_VENDOR=torch`），与其余国产后端同口径。不设这个变量时是探测：CuPy 真装了就用 `cupy_cusparse`，否则 `torch`（同 MetaX 的策略，**未实测**）               |
| 精度参考    | CPU 上的 SciPy（CUDA、ROCm 以外的后端都是这样）                                                                                                                                                                                     |
| runner 路由 | 与 CUDA / MetaX 相同的通用性能脚本（`GENERIC_BENCHMARK_BACKENDS`）                                                                                                                                                                |

因为 `torch.version.cuda` 有值、`torch.version.hip` 为 `None`，**只看这两个判据分不出天数和
NVIDIA**，所以要靠上面那三个较弱的信号。为什么是三个而不是一个，有实测原因（2026-09-22，BI-V100）：

- **`+corex` 标记要查两个地方**：那台机器上 `torch.__version__` 是裸的 `2.10.0`，而 pip 元数据是
  `2.10.0+corex.4.5.0`。标记一直都在，只是模块属性被剥掉了——只读属性就会白白漏掉这个信号；
- **设备名要运行时可用才读得到**：同一台机器 CUDA 初始化失败（Error 803），
  `get_device_properties()` 直接抛异常，这个信号完全指望不上；
- **`COREX_HOME` 兜底**，也是 C API 的 `IX` 槽位读的同一个变量
  （`capi/src/adaptor/CMakeLists.txt`）。代价和 MetaX 用 `MACA_PATH` 一样：装了 SDK 的机器
  即使跑在别的卡上也会被认成天数——所以它排在版本号之后。
- 天数**没有**独立插件模块（不像摩尔线程的 `torch_musa`、昆仑芯的 `torch_xmlir`），
  所以用不上注册表里最强的那个信号。

三个都不命中时不要靠猜，用第 1 节的 `FLAGSPARSE_BACKEND=iluvatar` 显式指定，并核对打印出的真实设备名。

---

## 1. 每次开工的环境

```bash
cd <仓库>

# CoreX SDK：按厂商安装说明设置（常见安装位置是 /usr/local/corex，以本机为准）
export COREX_HOME=/usr/local/corex          # 也是自动探测的信号之一，见第 0 节
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
并把 `torch.__version__`、设备名、`warp_size` 记到第 4 节——完整顺序见 1.5 节。

---

## 1.5 首次上机验证顺序

按成本从低到高走，**每一步过了再做下一步**。前四步就是第 4 节那张"待实测"表，跑完把结果填回去。
新后端的通例是一次只让一个变量动：一上来就跑整套，失败时分不清是环境、Triton 还是内核的问题。

### 第 0 步：环境指纹（最先做，结果要回传）

```bash
ixsmi          # 天数的设备管理工具，确认卡在、且没有别的任务在用

python3 - <<'EOF'
import torch
print("torch:", torch.__version__)
print("version dict:", torch.version.__dict__)
print("cuda avail:", torch.cuda.is_available())
p = torch.cuda.get_device_properties(0)
print("device name:", repr(p.name))
print("warp_size:", getattr(p, "warp_size", "<无该属性>"))
print("MP count:", p.multi_processor_count)
print("max_threads_per_block:", getattr(p, "max_threads_per_block", "?"))
print("shared mem per block:", getattr(p, "shared_memory_per_block", "?"))
EOF
```

三个关键点：**`torch.__version__` 是否带 `+corex`**、**设备名的确切字符串**（这两个决定自动探测能不能
命中，见第 0 节），以及 **`warp_size` 是 32 还是 64**。不是 32 的话，按 32 写死的调优参数要逐项复查——
沐曦 C550 就是 64，为此改过 SpSV 的两个 warp knob（见 [MACA.md](MACA.md)）。

### 第 1 步：Triton 能不能真的编译执行

```bash
python3 -c "import triton; print(triton.__version__); import triton.backends as b; print(list(b.backends))"
# 期望列表里有 iluvatar
```

再跑一个最小内核。`@triton.jit` 必须写在**真实的 .py 文件**里（Triton 要读源码），写在 `python3 - <<EOF`
的 stdin 里会报 `ValueError: @jit functions should be defined in a Python file`：

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
python3 /tmp/tri_smoke.py
```

**这一步不通，后面所有报错都会指向错误的方向。** FlagSparse 的算子全靠 Triton 编译，能不能跑取决于
FlagTree 的 iluvatar target，而不是本仓库有没有天数代码。

### 第 2 步：后端探测（显式和自动各一次）

第 1 节的检查就是显式那一次。再验证自动探测：

```bash
env -u FLAGSPARSE_BACKEND python3 -c "from flagsparse.sparse_operations import _common as c; print('自动探测:', c._backend_name())"
# 期望也是 iluvatar。得到 cuda 说明三个信号都没命中：先看 COREX_HOME 有没有导出（第 1 节导了），
# 再按第 0 步的 torch 版本号和设备名调 _detect_iluvatar_runtime() 的匹配串
```

### 第 3 步：torch.sparse 能不能当基线

```bash
python3 - <<'EOF'
import torch
a = torch.sparse_csr_tensor(torch.tensor([0, 2, 4]), torch.tensor([0, 1, 0, 1]),
                            torch.randn(4), (2, 2), device="cuda")
b = torch.randn(2, 8, device="cuda")
print("csr mm ok:", (a @ b).shape)
EOF
```

这是摩尔线程翻过车的地方：那里能建稀疏张量、只缺 matmul，光看"能不能建出稀疏张量"发现不了。
报错就按第 3 节把默认基线改成 `None`。

### 第 4 步：单算子冒烟

```bash
python3 -m pytest -q tests/pytest/test_spmv_csr_accuracy.py -k "float32 and non" -x 2>&1 | tail -20
```

过了再放开一个算子的全量，不要直接跑整套。

### 第 5 步：精度全量

```bash
setsid timeout -s KILL 14400 python3 -u run_flagsparse_pytest.py \
  --phase accuracy --mode normal --delivery-only --gpus 0 --timeout 3600 \
  --results-dir pytest_results_iluvatar_acc \
  > pytest_results_iluvatar_acc.log 2>&1 < /dev/null &
```

### 第 6 步：交付复现（精度 + 性能）

命令见第 2 节。

### 通用注意事项

- **长命令一律用 `timeout -s KILL` 包住。** 新后端上内核可能卡死，Ctrl-C 送不进去，一个卡死的内核
  能耗掉一台机器；
- **调试报错时加 `CUDA_LAUNCH_BLOCKING=1`**，否则错误异步抛出，栈往往指向无关的地方（常见是
  `allclose` 里面）；
- **多卡机器先确认 `--gpus 0` 真的生效**：runner 靠 `CUDA_VISIBLE_DEVICES` 选卡，CoreX 是否遵守这个
  变量未验证，跑起来用 `ixsmi` 看负载落在哪张卡；
- **不用跑 `spsv` / `spsm` / `spgemm`**，它们不在交付清单里，`--delivery-only` 也选不到。

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

## 4. 实测记录与待办

每项对应 1.5 节的一步，验证过就把"未验证"换成实测值和日期。

### 4.1 已实测（2026-09-22，机器 `u98`）

**GPU 本身可用**：在与驱动匹配的 CoreX 3.2.3 镜像里
（`harbor.iluvatar.com.cn:10443/saas/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3`）
`torch.cuda.is_available()` 为 `True`，dense matmul 和 `torch.sparse` CSR matmul 都跑通。
之前的 Error 803 纯粹是镜像与驱动版本错配（4.2 节），不是硬件或容器权限问题。

| 项目                                          | 实测值                                                                                                                                                                                                                                  |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 卡                                            | `Iluvatar BI-V100` × 8，单卡 32768 MiB（`ixsmi -L`）                                                                                                                                                                               |
| 设备名（torch 侧）                            | **`'Iluvatar BI-V100'`** —— 命中探测的 `iluvatar` 和 `bi-v` 两个匹配串                                                                                                                                                    |
| **`warp_size`**                       | **属性不存在**（`getattr` 返回默认值）—— 见 4.3 节，这是目前最需要查实的一项                                                                                                                                                  |
| MP count                                      | 16                                                                                                                                                                                                                                      |
| `torch.sparse` CSR matmul                   | **可用**（有 beta 警告，但能算出结果），所以 `FLAGSPARSE_ILUVATAR_VENDOR=torch` 这个基线是站得住的                                                                                                                              |
| 3.2.3 镜像里的 torch / triton                 | `torch 2.1.0+corex.3.2.3`（模块属性又是裸的 `2.1.0`）、**`triton 2.3.1`** —— 比 FlagTree 镜像的 Triton 3.6 老得多，这个镜像只适合做指纹，不适合跑算子                                                                     |
| 宿主机驱动 / IX-ML                            | **3.2.3 / 3.2.3**；呈现的 CUDA 兼容级别是 **10.2**                                                                                                                                                                          |
| 宿主机 CoreX 安装                             | `/usr/local/corex` → `corex-3.2.3`，另有 3.1.1 / 4.4.0 / 4.5.0 未启用                                                                                                                                                              |
| 厂商镜像里的 CoreX                            | 4.5.0（`/usr/local/corex-4.5.0`），容器内 `ixsmi` 报 IX-ML 4.4.0                                                                                                                                                                    |
| 容器里的 torch                                | `torch.__version__` 是裸的 **`2.10.0`**，但 pip 元数据是 **`2.10.0+corex.4.5.0`** —— 标记只在 wheel 元数据里，模块属性被剥掉了。探测因此**两个地方都读**                                                      |
| `torch.version` 的属性                      | `__all__` 只有 `__version__ / debug / cuda / git_version / hip / rocm / xpu`，**没有任何厂商属性**。沐曦有 `torch.version.maca`，天数没有对应物                                                                             |
| `torch.version.cuda` / `.hip` / `.rocm` | `'10.2'` / `None` / `None`（`git_version` 是 `509100cb`）                                                                                                                                                                     |
| Triton                                        | **`flagtree 0.7.0rc2+iluvatar3.6` 镜像里已装**，不用自己编；能否编译执行内核仍待验证（1.5 节第 1 步）                                                                                                                           |
| 其余厂商包                                    | `torch_sparse 0.6.16+corex.4.5.0`、`torch_scatter`、`torch_cluster`、`pyg_lib`、`apex`、`ixformer`、`dali` 等都带 `+corex.4.5.0`；**没有** `torch_corex` / `torch_ixuca` 这类插件模块，符合 CUDA 兼容栈的预期 |
| 设备名（torch 侧）                            | **读不到**，CUDA 初始化就失败了                                                                                                                                                                                                   |
| CuPy                                          | 不适用：CUDA 兼容级别 10.2，`cupy-cuda12x` 对不上，`cupy-cuda102` 已停更。基线实际会落到 PyTorch                                                                                                                                    |

### 4.2 当前阻塞：驱动与运行时版本错配

```
Error 803: system has unsupported display driver / cuda driver combination
torch.cuda.is_available() -> False        # 但 device_count() -> 8
```

宿主机驱动 3.2.3 配不上镜像里的 CoreX 4.4.0/4.5.0。**两条路，都要走平台**：换一个配 3.2.3
驱动的镜像（首选，不影响别人），或把宿主机驱动升到 4.4.0/4.5.0（影响整机所有用户）。
宿主机上装着四个 CoreX 版本，说明这台机器换过版本，先问同事有没有配 3.2.3 的镜像。

容器本身没问题（`--privileged`、`-v /dev:/dev`，设备节点可见），不要再在容器里调环境变量。
挂 `ixsmi` 时注意两个坑：挂载源要用 `readlink -f` 展开后的**真实文件**
（`/usr/local/corex-3.2.3/bin/ixsmi`），以及之前失败的 `docker run` 会在宿主机上
**自动创建同名空目录**，把目录挂到镜像里的文件上就会报 `not a directory`，要先 `rmdir`。

### 4.2.1 一个没有采用的探测信号

`torch.version` 里 **torch 主版本 2.10 配 `cuda == '10.2'` 在上游是不可能的组合**（上游 1.12
之后就不再发 CUDA 10.2 的包），所以这本身是个挺强的指纹。**没有把它写进探测**：天数以后把兼容
级别升到 11 或 12，这个判断就会无声失效，而 `COREX_HOME` 是装了 SDK 就有的东西。记在这里，
是为了万一哪天 `COREX_HOME` 也不可用时还有条后路。

### 4.3 仍未验证

| 项目                                                                                | 状态                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`warp_size` 的真实值**                                                    | **属性不存在，代码会静默取 32**。`_common.py` 的 `_get_device_backend_info()` 里 `default_warp = 64 if backend == "hip" else 32`，天数走 `cuda` 分支；`spmv_csr.py`、`spmm_csr.py`（三处）、`spmm_csr_opt_alg2.py` 各自也 `getattr(props, "warp_size", 32)`。**若实际是 64，整套启动几何都偏，而且不报错、只掉性能**（MetaX C550 就是 64）。查法：`ixsmi -q` 里找 warp/core 相关字段，或在 Triton 3.x 下 `triton.runtime.driver.active.get_current_target()` |
| 自动探测能否命中（元数据里的`+corex` 应当命中，未在真机跑过 `_backend_name()`） | 未验证                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| FlagTree 的`iluvatar` 后端能否编译执行最小 Triton 内核                            | 未验证                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `torch.sparse` CSR/COO matmul 能否作为基线                                        | 未验证                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `CUDA_VISIBLE_DEVICES` 选卡（八卡机，务必确认落在哪张）                           | 未验证                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 20 变体交付结果                                                                     | 无                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

---

## 5. C API（`BACKEND=IX`）

C API 侧早就为天数预留了 `IX` 槽位（`COREX_HOME`，默认 `/usr/local/corex`），`deps/libtriton_jit`
也认 `IX`，但它**没有提供 IX 的 Backend 模块**，所以 C API 目前编不起来：CTest 的 `iluvatar` profile
（`capi/ctest/backends/iluvatar.cmake`）标的是 `CAPI_BUILDABLE OFF`。Python 侧不受影响。
