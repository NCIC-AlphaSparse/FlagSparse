# 各后端 debug 手册（q4）

本目录记录 5 个国产后端在 q4 分支上的排查方法、已知问题和待确认项。每个后端一个文件：

| 后端 | 厂商 / 卡 | `FLAGSPARSE_BACKEND` | 文件 | 已有的上机文档 |
|---|---|---|---|---|
| 海光 DCU | BW1000（gfx936，ROCm/HIP） | `rocm` | [DCU.md](DCU.md) | [../DCU.md](../DCU.md) |
| 沐曦 MACA | C550 | `metax` | [MACA.md](MACA.md) | [../MACA.md](../MACA.md) |
| 摩尔线程 MUSA | S5000 | `mthreads` | [MUSA.md](MUSA.md) | [../MUSA.md](../MUSA.md) |
| 华为昇腾 | 910B（交付测试用卡） | `ascend` | [ASCEND.md](ASCEND.md) | [../ASCEND.md](../ASCEND.md) |
| 天数智芯 | BI-V150 | `iluvatar` | [ILUVATAR.md](ILUVATAR.md) | [../ILUVATAR.md](../ILUVATAR.md)、[../ILUVATAR_DEBUG.md](../ILUVATAR_DEBUG.md) |

上面「已有的上机文档」讲环境搭建和交付复现；本目录只写 **debug**：出了问题怎么定位、已知的坑、
q4 新增的 42 个变体（[../NEW_OPERATORS_CUSPARSE_12_5.md](../NEW_OPERATORS_CUSPARSE_12_5.md) 前 42 个）在该后端上的风险点，
以及跑完后需要带回来的信息。

## 1. q4 在各后端的总体状态（2026-10-02）

- CUDA（RTX 5090）和 MUSA（S5000）已有真机验证；MUSA 的当前调试入口是 C API CTest。
  42 个变体的 Python 精度历史记录全过，C API 的完整结果以 `capi/bench-musa` 为准。
- 海光、沐曦、天数与 CUDA 共用同一套 Triton 代码；新代码已按已知限制避坑（见下表），仍需按 Python runner 实机确认。
- 昇腾：新写的代码都有 torch_npu 路径。在 CUDA 上把所有模块强制切到昇腾分支、并禁止 Triton 启动，
  42 个里 36 个走 torch 路径且正确，5 个仍会启动 Triton，1 个结果错误（详见 [ASCEND.md](ASCEND.md)）。

新代码针对已知限制做的规避：

| 已知限制 | 出现在 | q4 新代码的做法 |
|---|---|---|
| fp16 / 复数原子加不一定可用 | 多个后端 | COO 等需要原子加的路径先在 float32 / int32 缓冲区累加，最后再转回 |
| 单线程私有内存 4 KB 上限 | 沐曦 C550 | 循环一律用运行时 `range`，不做编译期展开 |
| 复数高级索引、复数求和没有 kernel | 摩尔 S5000 | 复数全部按 `view_as_real` 拆成实部 / 虚部两个平面计算 |
| Triton 缺 shmem、`associative_scan` | 昇腾 910B | 每条新路径都有 torch_npu 实现（`index_add_` 等） |
| fp64 H2D 拷贝静默得到 0 | 天数 BI-V150 | 不在 q4 范围（q4 不含 f64 / c64） |

## 2. 通用排查顺序

1. **先确认后端识别对了。**
   ```bash
   PYTHONPATH=src python3 -c "from flagsparse.sparse_operations import _common as c; print(c._backend_name(), c._ACCEL_DEVICE_TYPE)"
   ```
   不对就设 `FLAGSPARSE_BACKEND=<上表的值>`。
2. **先排除跑到了旧的已安装包。** 一律用 `PYTHONPATH=src` 跑；某个基线列无缘无故变成 N/A 时，先查
   `sudo python3 -m pip show flagsparse` 和 `python3 -I -c "import flagsparse; print(flagsparse.__file__)"`。
3. **先跑能力探测，再跑测试**：`python3 tools/probe_accel_capabilities.py`，它能告诉你失败属于哪一层
   （分配 / 拷贝、torch 算子、Triton、torch.sparse）。
4. **q4 变体精度**（Python-only 后端使用；42 个变体 × 2 个规模，每个用例名就是变体名）：
   ```bash
   PYTHONPATH=src python3 -m pytest tests/pytest/test_q4_variants_accuracy.py -v
   # 只看某个算子：-m spmv_csr ；只看某个变体：-k spmv_csr_f16f32_int_non
   ```
5. **Python-only 后端的历史统一 runner（精度 + 性能，汇总 62 个条目）**：
   ```bash
   PYTHONPATH=src python3 run_flagsparse_pytest.py \
     --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc,spmm_bsr,spmm_bell,sddmm_csr \
     --gpus 0 --results-dir results_<后端>_<日期> --benchmark-input tests/data
   ```
   当前入口优先使用上面的 `tools/run_backend_tests.py`；本命令仅用于复现已有的
   Python 侧历史报告。
   **每次都用新的 `--results-dir`**：往已有目录里重跑部分算子会把 `summary.json` 覆盖成只剩这部分。
   `spmm_bell` 的性能测试在 CPU 上转换 Blocked-ELL，每个矩阵要几分钟，是整轮最慢的一步。
6. **没有厂商库基线时**，按 H800 带宽换算上限判定（`tools/baseline_bound.py`）：
   ```bash
   python3 tools/baseline_bound.py <H800结果目录> --vendor <本后端结果目录> --vendor-card <卡名>
   # 卡名：dcu-bw1000 / maca-c550 / musa-s5000 / iluvatar-biv150 / ascend-910b
   ```

### 当前测试入口

调试时按后端选择测试层：MUSA 使用 C API CTest；DCU、MACA、Ascend、
Iluvatar 和 XPU 使用 Python runner。不要用 MUSA 的 Python benchmark
结果判断 C API，那里没有 muSPARSE 对比。

MUSA C API：

```bash
cmake -S capi -B capi/build-musa -G Ninja \
  -DBACKEND=MUSA -DMUSA_HOME=/usr/local/musa \
  -DCMAKE_BUILD_TYPE=Release
cmake --build capi/build-musa -j
ctest --test-dir capi/build-musa -L capi --output-on-failure
```

其他后端 Python：

```bash
python3 tools/run_backend_tests.py \
  --backend rocm --phase both --mode normal
```

将 `rocm` 替换为 `maca`、`ascend`、`iluvatar` 或 `xpu`。每次调试使用
新的 `--results-dir`，避免覆盖旧报告。

## 3. 通用 debug 规则（来自前几次上机）

- **先用测试自己的操作复现，再加诊断代码。** 在还没验证过的运行时上，诊断代码里的额外张量操作本身就可能出错。
- **怪到“不确定性”之前，先确认其他参数都固定了。** 用重复次数说话，不要用一对反常结果下结论。
- **读机器上那份源码，不要读本地的。** 两边的代码经常已经不一样；移植修复用带断言的补丁脚本，不要整文件复制。
- **先查这个用例实际走了哪个 kernel。** 通过的用例可能根本没走到出问题的 kernel。
- **按输入结构二分，不只按规模。**（对角 / 双对角 / 稠密三角能把两个不同缺陷分开）
- 查非法访存时加 `CUDA_LAUNCH_BLOCKING=1`（报错位置才准）；求解器类测试一律套 `timeout -s KILL`，
  卡死的 kernel 用 Ctrl-C 停不下来。
- 一个故障会让整个运行时失效时，**每种配置单独起一个进程**。
- 不要用 `pytest --forked`：GPU 上下文在 fork 后不可用，所有 GPU 用例都会失败。按算子 marker 分开跑即可。
- 诊断脚本放在容器重建后仍保留的目录里。

## 4. 跑完需要带回来的信息

- `results_*/summary.json`、`summary.csv`、每个算子目录下的 `accuracy_result.json` 和 `performance.csv`；
- 失败用例的完整报错（加 `CUDA_LAUNCH_BLOCKING=1` 重跑一次）；
- 第 2 节第 1 步的输出、`pip list | grep -iE "torch|triton|flagtree"`、驱动 / SDK 版本；
- 各后端文件「待确认」一节里列出的具体问题的答案。
