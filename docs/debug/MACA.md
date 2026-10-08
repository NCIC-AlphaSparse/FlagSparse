# MetaX / MACA（C550）：65 变体验证任务

`FLAGSPARSE_BACKEND=metax`。环境搭建、交付复现见 [`../MACA.md`](../MACA.md)。背景见本目录
[`README.md`](README.md)。

## 要做的事

环境按 `../MACA.md` 第 1 节配好后，先做 `../MACA.md` 0.5 节的开跑前检查（`flagsparse.__file__` 必须指向
本仓库 `src/`；`_backend_name()` 必须是 `metax`），再跑：

```bash
setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc,spgemm_csr,sddmm_csr \
  --phase both --mode normal --gpus 0 --timeout 4500 \
  --benchmark-input /root/gcx/matrix --benchmark-warmup 5 --benchmark-iters 20 \
  --op-benchmark-args='sddmm_csr=--no-cusparse' \
  --results-dir results_metax_65_<日期> \
  > results_metax_65_<日期>.log 2>&1 < /dev/null &
# 跑完后
python3 tools/delivery_table.py results_metax_65_<日期>
```

- `sddmm_csr=--no-cusparse`、`--timeout 4500` 沿用 `../MACA.md` 0.5 节的 20 变体交付命令，原因见那一节
  （C550 没有可用厂商稀疏库；SDDMM 的 4 个 K 值不收窄，单阶段耗时长）。

- 外层 `timeout -s KILL` 是整条命令的总限时（内核卡死时 Ctrl-C 送不进去，只能靠 KILL）；`--timeout` 是
  每个算子每个阶段的限时。这两个数是 20 变体时期定的，65 变体多了 6 个父算子，**没有在本机实测过总耗时**，
  到点被杀的话按 `delivery_table.py` 里 `NotFound` 的算子单独补跑（补跑要用新的 `--results-dir`）。
- `--results-dir` 每次都用新目录：`summary.json` 每跑一次就整体重写，往旧目录里补跑一部分会把之前的结果冲掉。

把 `delivery_table.py` 的完整输出带回来，尤其是第一行的 `missing` 数字。

## 已知风险点（这个后端特有的，不是猜的）

- **C550 没有厂商 cuSPARSE/CuPy 基线**（`../MACA.md` 0.5 节已经写了：`CuPy 真装了就用
  cupy_cusparse，否则 torch`，本机实际是 PyTorch）。这意味着 45 个新变体里走 q4 变体测量
  （`tests/q4_variant_bench.py`）的那部分，厂商列大概率全是空，只有 PyTorch 对照——**这是预期行为，
  不是 bug**，跟原有 20 个变体在这台机器上的情况一致。
- **单线程私有内存上限 4 KB**：`tl.static_range` 编译期展开的循环超了会报
  `memory size or pointer value too large to fit in 32 bit`，真正原因在它上一行的 MACA 运行时提示里。
  复数变体（`spmv_csr/coo/csc_c32`、`spmm_csr/coo/csc_c32`、`spvv_c32_int_conj`、`spmv_sell_c32`、
  `spmv_csr_f32c32`）最该先跑，这类问题历史上都出在复数/宽展开路径上。
- **SDDMM 没有可用参考**：`../MACA.md` 7.3 节记录过 `torch.sparse.sampled_addmm` 结果是错的，交付命令
  一直用 `--op-benchmark-args='sddmm_csr=--no-cusparse'` 绕开。这次的新 sddmm 变体（c32、f16，还有
  trans/col 方向）同样会撞上这个限制，SDDMM 的厂商列预期也是空的。

## 跑完要确认

1. `missing` 是不是 0。
2. 65 行里有没有 `FAIL`（不是"没有基线所以是 N/A"，是真的算错了）。
3. 复数变体和 SDDMM 变体有没有卡死或者 OOM（对应上面两条风险点）。

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
