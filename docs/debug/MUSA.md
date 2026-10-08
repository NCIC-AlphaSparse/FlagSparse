# 摩尔线程 / MUSA（S5000）：65 变体验证任务

`FLAGSPARSE_BACKEND=mthreads`。环境搭建、交付复现见 [`../MUSA.md`](../MUSA.md)。背景见本目录
[`README.md`](README.md)。

## 要做的事

环境按 `../MUSA.md` 第 2 节配好后：

```bash
setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc,spgemm_csr,sddmm_csr \
  --phase both --mode normal --gpus 0 --timeout 3600 \
  --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir results_musa_65_<日期> \
  > results_musa_65_<日期>.log 2>&1 < /dev/null &
# 跑完后
python3 tools/delivery_table.py results_musa_65_<日期>
```

- 这里用通用 runner，不用 `../MUSA.md` 0.5 节的 `run_flagsparse_split_delivery.py`：后者的性能取自 C API，
  新增的 45 个变体大多没有 C API 实现，覆盖不全。代价是 Python 侧没有厂商基线，性能列只有 FlagSparse 自己的
  时间，加速比大多为空——这一轮要的是 65 个变体精度全过、性能都能跑出时间；muSPARSE 加速比照旧走
  `split_delivery`，这轮不要求。
- `--timeout 3600` 沿用 `../MUSA.md` 的结论（`900` 不够）。

- 外层 `timeout -s KILL` 是整条命令的总限时（内核卡死时 Ctrl-C 送不进去，只能靠 KILL）；`--timeout` 是
  每个算子每个阶段的限时。这两个数是 20 变体时期定的，65 变体多了 6 个父算子，**没有在本机实测过总耗时**，
  到点被杀的话按 `delivery_table.py` 里 `NotFound` 的算子单独补跑（补跑要用新的 `--results-dir`）。
- `--results-dir` 每次都用新目录：`summary.json` 每跑一次就整体重写，往旧目录里补跑一部分会把之前的结果冲掉。

把 `delivery_table.py` 的完整输出带回来，尤其是第一行的 `missing` 数字。

## 已知风险点（这个后端特有的，不是猜的）

- **`torch.sparse` 在 MUSA 上完全没有矩阵乘实现**（`../MUSA.md` 4.5 节的实测能力矩阵已经写明）。
  这是四个后端里最明确的一条已知限制：凡是依赖 PyTorch 稀疏矩阵乘做对照/回退的路径都会失败，不止
  q4 新变体，原有 20 个变体的 PyTorch 对照列也受影响。`tests/q4_variant_bench.py` 的 `_time_pytorch`
  helper 专门注释了 "e.g. MUSA registers no sparse matmul, no sparse int8"，理论上会优雅跳过、报
  `pytorch_reason` 而不是崩溃——**这一条需要真机确认，不能只信注释**。
- **复数高级索引没有 kernel**：`values[order]` 这类操作在 MUSA 上报
  `"IndexMusa" not implemented for 'ComplexFloat'`。`_common._gather_values` 已经统一改走
  `view_as_real` 拆实部/虚部规避，但这是否覆盖了全部 45 个新变体的复数路径（`spmv_csr/coo/csc_c32`、
  `spmm_csr/coo/csc_c32`、`spvv_c32_int_conj`、`spmv_sell_c32`、`sddmm_csr_c32`）没有逐一验证过。
- **muDNN 的 gemv 不支持 fp64/复数**（gemm 支持）。这条主要影响原有 fp64 变体，新变体里的 fp64 相关
  路径（如果有间接依赖 gemv 的）也要留意。

## 跑完要确认

1. `missing` 是不是 0。
2. 所有复数变体（c32）的精度和性能是不是真的跑通了，还是报了 `IndexMusa` 之类的错误。
3. 性能列：PyTorch 对照是不是像预期的那样全是 `N/A`/有 reason，还是意外地跑通了但数字可疑（比如慢几百倍，通常意味着走了某种 fallback loop 而不是真正的 kernel）。

把这三条的结果写回来，不用额外分析，原始数据最有用。

## 实机改动记录

在这台机器上为了跑通而改过的每一个文件都记在这里，跟着 commit 一起推上来。没改就写"无"，不要留空。
只记改动，测试结果写到上面"带回来的东西"/"跑完要确认"里。

格式（一个改动一条，新的追加在最后）：

```
### <日期> <文件路径>::<函数或位置>
- 改了什么：
- 为什么改（附报错原文）：
- 怎么验证的：
- 其他后端是否也需要：是 / 否 / 不确定
```

（暂无）
