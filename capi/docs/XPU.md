# 昆仑芯 XPU

**结论先说：C API 目前在昆仑芯上跑不起来，而且不是缺一个适配器的问题。** 可达的是
FlagSparse 的 Python 侧。下面两条阻塞是独立的，补上其中一条不解锁另一条。

## 阻塞一：JIT 桥没有 XPU 后端

`deps/libtriton_jit` 的后端清单是（`cmake/Backend*.cmake`，本机实查）：

```
CUDA  GCU  HCU  MACA  MLU  MUSA  NPU
```

**没有 XPU。** 这个桥负责编译和启动每一个内核，所以 C API 在昆仑芯上连"分子"都没有——
写一个 `src/adaptor/backend/xpu/adaptor.cpp` 不会有帮助，适配器解决的是 runtime 调用，
不是内核从哪来。这也是 `src/adaptor/CMakeLists.txt` 里 XPU 被列进
`FLAGSPARSE_BLOCKED_BACKENDS` 而不是预留槽位的原因：预留槽位是"等一个库名"，这个是
"等一个上游后端"。

FlagTree 的 Triton **有** xpu target。所以：

| 层 | 昆仑芯上的状态 |
|---|---|
| FlagSparse Python（Triton 内核） | 可达路径，走 FlagTree 的 xpu backend |
| FlagSparse C API（本仓库） | 阻塞在 libtriton_jit |

要在昆仑芯上测算子，今天的入口是 FlagSparse 仓库的 `run_flagsparse_pytest.py`，
后端注册表里 `xpu` 槽位已就位（`src/flagsparse/sparse_operations/_common.py`）。

## 阻塞二：没有 cuSPARSE 形状的稀疏库可做基线

昆仑芯的数学库是 **XDNN**，它提供的是**固定的稀疏算子**，不是描述符式的 generic API。
`ctest/baseline/baseline.hpp` 的八个入口（SpMV/SpMM/SDDMM/SpGEMM/SpSV/SpSM/gather/
scatter）没有对应的通用入口可以绑——**即使 SDK 在手也一样**。

这和 MACA / MUSA 的情况**不是一回事**，别混淆：

| | 昆仑芯 XPU | MACA / MUSA |
|---|---|---|
| 厂商有稀疏库吗 | 有 XDNN，但不是 generic API | 有，且是 cuSPARSE 克隆 |
| 缺什么 | 缺的是 API 形状，补不了 | 只缺头路径和库名 |
| 槽位状态 | `NONE` | `UNNAMED` |
| 怎么解 | 需要厂商出 generic API，或本仓库为 XDNN 写逐算子适配 | 填两个 CMake 变量 |

所以昆仑芯上就算解决了阻塞一，性能行也会是 `baseline_status: "unavailable"` +
空 `speedup`，理由字符串写在 `ctest/baseline/CMakeLists.txt` 的
`FLAGSPARSE_XPU_BASELINE_WHY` 里。这是**诚实的空**，不是没做完。

## 如果上游补了 XPU 后端，这边要做什么

1. `deps/libtriton_jit` 出现 `cmake/BackendXPU.cmake` 后，从里面读 `XRE_HOME` 之类的
   toolkit 变量——**照抄，不要另猜**。`src/adaptor/CMakeLists.txt` 顶部的注释解释了为什么
   所有 toolkit 路径都从那里取：两边必须找到同一个 SDK，否则会静默错配。
2. 把 XPU 从 `FLAGSPARSE_BLOCKED_BACKENDS` 移到正常表项，填
   `FLAGSPARSE_XPU_HOME_VAR` / `_RUNTIME` / `_DRIVER` / `_HEADER`。
3. 如果昆仑芯的 driver API 镜像 CUDA，适配器就是 `backend/_template.inc` 加一个前缀表
   （参考 `backend/musa/adaptor.cpp`，25 行）。如果不镜像，就得单独写一份。
4. 基线仍然留在 `NONE`，除非 XDNN 那边出现 generic API。

## 跑测试

**ctest 在昆仑芯上跑不了**，别照搬其他后端那套命令：配置会在
`FLAGSPARSE_BLOCKED_BACKENDS` 处停下，因为 `deps/libtriton_jit` 没有 XPU 后端，
JIT 桥编译不出内核。这是预期行为，不是配置错误。

可跑的是 Python 侧，命令在下面「测试怎么跑（Python 侧）」一节。

### 跑不起来时按这个顺序查

1. **配置在 XPU 处停下** → 预期行为。等 `deps/libtriton_jit` 出现
   `cmake/BackendXPU.cmake`，再按本文「如果上游补了 XPU 后端」那节走；
2. **Python 侧后端认不出来** → 检查厂商插件是否真的装了。`_resolve_accel()` 要求
   `_vendor_plugin_present()` 通过，这条不是多余的谨慎：它曾经只看 `torch.xpu` 存不存在，
   结果在一台 Intel 机器上把 Intel 的 XPU 当成了昆仑芯；
3. **想要 speedup 列** → 没有，而且不是等一个库名。XDNN 是固定算子集而非描述符 API，
   `ctest/baseline/baseline.hpp` 的八个入口没有可绑的通用入口。

## 测试怎么跑（Python 侧）

```bash
cd /path/to/FlagSparse
FLAGSPARSE_BACKEND=xpu python run_flagsparse_pytest.py --accuracy
FLAGSPARSE_BACKEND=xpu python run_flagsparse_pytest.py --benchmark
```

注意后端探测是**要求厂商插件真的在**才认的（`_vendor_plugin_present`）。这条不是多余的
谨慎：`_resolve_accel()` 曾经因为只看 `torch.xpu` 是否存在，在一台 Intel 机器上把
Intel 的 XPU 当成了昆仑芯。
