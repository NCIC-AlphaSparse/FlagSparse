# Q4 后端调试文档索引

本目录按后端维护一份手册，涵盖调试方法、有效结果、已知问题和待办。
CUDA 的分轮优化/补测报告已合并到 CUDA.md；原始 JSON/CSV 保留。

| 后端 | 设备 | FLAGSPARSE_BACKEND | 调试手册 | 环境 / 交付文档 |
|---|---|---|---|---|
| NVIDIA CUDA | RTX 5090 | cuda | [CUDA.md](CUDA.md) | [C API 交付](../Q4_CAPI_HANDOFF.md) |
| 海光 DCU | BW1000 / gfx936 | rocm | [DCU.md](DCU.md) | [DCU 环境](../DCU.md) |
| 沐曦 MACA | C550 | metax | [MACA.md](MACA.md) | [MACA 环境](../MACA.md) |
| 摩尔线程 MUSA | S5000 | mthreads | [MUSA.md](MUSA.md) | [MUSA 环境](../MUSA.md) |
| 华为昇腾 | 910B | ascend | [ASCEND.md](ASCEND.md) | [Ascend 环境](../ASCEND.md) |
| 天数智芯 | BI-V150 | iluvatar | [ILUVATAR.md](ILUVATAR.md) | [Iluvatar 环境](../ILUVATAR.md)、[排查记录](../ILUVATAR_DEBUG.md) |

当前 CUDA 45 项均有 CuPy 加速比，44 项逐算子平均 ≥0.8；
SpGEMM 0.789× 仍需优化。43 项为 2026-10-06 数据，2 项 CSC SpMM 保留历史值。
这不是其他后端的验收结果，也不是 45 项同轮重跑，详情见 [CUDA.md](CUDA.md)。

## 0. Q4 变体范围与历史结果

当前 Q4 清单为 **45 个变体**，delivery 20 个，合计 65 个；
以 conf/operators.yaml、capi/conf/operators.yaml 和
tests/pytest/test_q4_variants_accuracy.py 中的清单为准。

2026-09-28/29 的旧报告按 42 项统计，不能直接当成现行 45 项覆盖：
移出统计范围的 3 项仍保留实现：
gather_i8_int、spmm_bell_f32_int_non_non_row、spmm_bsr_f32_int_non_non_row。
新增的 6 项是：

- sddmm_csr_f16_int_non_non_row
- sddmm_csr_c32_int_non_non_row
- spmm_csc_c32_int_non_non_row
- spmm_csc_f16_int_non_non_row
- spmm_coo_i8i32_int_non_non_row
- spgemm_csr_f32_int_non_non

各后端已有不同日期、不同测试层的记录，必须以对应手册的最新带日期章节为准。
旧 42 项全过、CUDA 模拟 Ascend 分支通过、C API dispatch 覆盖，
均不能替代对应设备当前 45 项的精度与性能验收。

## 1. 数据组织

- [CUDA 45 项汇总 JSON](Q4_CUDA_45_SPEEDUPS_20261006.json)、
  [CSV](Q4_CUDA_45_SPEEDUPS_20261006.csv)：机器可读均值及各项原始来源。
- 有效 CUDA 数据批次、基线差异和复现命令统一由 [CUDA.md](CUDA.md) 索引；
  results_cuda_q4_* 中的初轮和 v1/v2/v3 目录是实验归档，不默认视为最终结果。
- results_metax_q4_45_perf_20261005/、musa_test/ 等后端归档及其原始数据保留，
  解读方法见对应后端手册。
- summary.csv、q4_accuracy_result.json 等现有历史汇总保留；
  日期、变体范围和测试层不明确时，不用它们覆盖新结果。
- 性能按每个变体各案例加速比的算术平均汇总，不能混为所有算子的整体平均；
  Python/Triton、C API 和厂商基线属于不同路径。

## 2. 通用排查顺序

1. 确认后端识别和安装包来源：

   ```bash
   PYTHONPATH=src python3 -c "from flagsparse.sparse_operations import _common as c; print(c._backend_name(), c._ACCEL_DEVICE_TYPE)"
   python3 -I -c "import flagsparse; print(flagsparse.__file__)"
   ```

   识别不符时设置表中的 FLAGSPARSE_BACKEND；运行源码时使用 PYTHONPATH=src。
2. 先做能力探测，区分张量分配/拷贝、Torch 算子、Triton 和 sparse 库问题：

   ```bash
   python3 tools/probe_accel_capabilities.py
   ```

3. 单独检查 Q4 精度：

   ```bash
   PYTHONPATH=src python3 -m pytest tests/pytest/test_q4_variants_accuracy.py -v
   # 按 marker/变体筛选：-m spmv_csr 或 -k spmv_csr_f16f32_int_non
   ```

4. 按后端选择测试层。MUSA 以 C API CTest 为入口，不能用 Python 结果替代 muSPARSE 对比：

   ```bash
   cmake -S capi -B capi/build-musa -G Ninja \
     -DBACKEND=MUSA -DMUSA_HOME=/usr/local/musa -DCMAKE_BUILD_TYPE=Release
   cmake --build capi/build-musa -j
   ctest --test-dir capi/build-musa -L capi --output-on-failure
   ```

   其他后端使用 Python runner：

   ```bash
   python3 tools/run_backend_tests.py --backend rocm --phase both --mode normal
   ```

   将 rocm 替换为 maca、ascend、iluvatar 或 xpu。CUDA 专项命令见 CUDA.md。
   每次测试使用新结果目录，避免局部重跑覆盖整批 summary。

## 3. 通用 debug 规则

- 先用测试本身的操作复现，再加入诊断操作；诊断依赖的后端算子也可能不支持。
- 怀疑不确定性前先固定参数和输入，用重复测试定位，不凭单次异常结论。
- 检查设备上实际使用的源码与 kernel，不用本地代码推断另一台机器。
- 按输入结构二分：对角、双对角、稠密三角等比只改变规模更有帮助。
- 非法访存使用 CUDA_LAUNCH_BLOCKING=1；求解器测试设置进程超时，
  卡死的 kernel 未必能用 Ctrl-C 中断。
- 会污染运行时的故障每个配置单独启动进程；不要使用 pytest --forked，
  GPU 上下文不能安全继承，按 marker 分开跑。
- benchmark 独占 GPU，与精度测试串行执行；prepare、格式转换和同步的计时边界必须注明。
- 移植修复使用带断言的局部补丁，避免整文件覆盖；诊断脚本放在容器重建后仍保留的位置。

## 4. 上机后带回的信息

- 新结果目录的 summary.json、summary.csv、逐算子 accuracy_result.json/performance.csv。
- 失败用例完整报错、必要时 CUDA_LAUNCH_BLOCKING=1 的复现结果。
- 后端识别、包版本、驱动/SDK 版本和代码版本。
- 对应后端手册待确认项的复测结果。
