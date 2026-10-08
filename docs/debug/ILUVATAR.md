# 天数智芯 Iluvatar（BI-V150）：65 变体验证任务

`FLAGSPARSE_BACKEND=iluvatar`（通常能自动识别）。环境搭建、交付复现见 [`../ILUVATAR.md`](../ILUVATAR.md)，
上机排查记录见 [`../ILUVATAR_DEBUG.md`](../ILUVATAR_DEBUG.md)。背景见本目录 [`README.md`](README.md)。

## 要做的事

环境按 `../ILUVATAR.md` 第 1 节配好后（**一定要显式设 `FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse`，
不要用 `torch`**——见下面第一条风险点）：

```bash
setsid timeout -s KILL 43200 \
  env FLAGSPARSE_BACKEND=iluvatar FLAGSPARSE_ILUVATAR_VENDOR=cupy_cusparse \
  python3 -u run_flagsparse_pytest.py \
    --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc,spgemm_csr,sddmm_csr \
    --phase both --mode normal --gpus 0 --timeout 3600 \
    --benchmark-input tests/data --benchmark-warmup 5 --benchmark-iters 20 \
    --h800-reference \
    --pytest-args='-k "test_q4_variant or (((float or half or f16 or f32) and not (double or float64 or bfloat16 or complex64 or complex128)) or (complex64 and (gather or scatter)))"' \
    --op-benchmark-args='gather=--value-dtypes float16,float32,complex64' \
    --op-benchmark-args='scatter=--value-dtypes float16,float32,complex64,int8' \
    --op-benchmark-args='spmv_csr=--dtypes float32 --alg auto' \
    --op-benchmark-args='spmv_coo=--dtypes float32' \
    --op-benchmark-args='spmv_csc=--dtypes float32,complex64' \
    --op-benchmark-args='spmm_csr=--dtypes float16,float32' \
    --op-benchmark-args='spmm_coo=--dtypes float16,float32' \
    --op-benchmark-args='spmm_csc=--dtypes float32,complex64' \
    --op-benchmark-args='sddmm_csr=--dtype float32' \
    --results-dir results_iluvatar_65_<日期> \
  > results_iluvatar_65_<日期>.log 2>&1 < /dev/null &
# 跑完后
python3 tools/delivery_table.py results_iluvatar_65_<日期>
```

这条命令是 `../ILUVATAR.md` 第 2 节 20 变体交付命令的扩展，三处改动都在 CUDA 上核对过：

- **`-k` 前面加了 `test_q4_variant or`**。原来的 `-k` 是为了挡掉 fp64（本卡 fp64 H2D 静默返回 0），但它
  同时会把 45 个新变体里的 c32/i8 精度用例筛掉（只剩 54/91）；加上之后 91/91 全保留，fp64/complex128
  用例仍然一个不收。新变体自己不含 fp64，精度对照是 CPU 上算的。
- **`scatter` 加了 `int8`**：`scatter_i8_int` 的性能取自 scatter 脚本自己的 int8 行，原来的
  `float16,float32,complex64` 会让它变成 `NotFound`。
- **新增 `spmv_csc`/`spmm_csc` 的 dtype 限制**：这两个脚本默认会跑 float64/complex128，先跑 fp64 万一
  卡死或崩溃，排在最后追加的新变体行就丢了；它们的交付变体只要 f32/c32/f16，f16 由新变体那一路单独测。
- 原 20 个变体里的 9 个 f64/c64 在本卡上照旧测不了，`delivery_table.py` 里显示 `NotFound` 是预期的；
  `spgemm_csr` 的 runner 会按 dtype 拆开分别跑 float32/float64，float64 那一组的结果不要当真，交付变体
  只用 float32。

- 外层 `timeout -s KILL` 是整条命令的总限时（内核卡死时 Ctrl-C 送不进去，只能靠 KILL）；`--timeout` 是
  每个算子每个阶段的限时。这两个数是 20 变体时期定的，65 变体多了 6 个父算子，**没有在本机实测过总耗时**，
  到点被杀的话按 `delivery_table.py` 里 `NotFound` 的算子单独补跑（补跑要用新的 `--results-dir`）。
- `--results-dir` 每次都用新目录：`summary.json` 每跑一次就整体重写，往旧目录里补跑一部分会把之前的结果冲掉。

把 `delivery_table.py` 的完整输出带回来，尤其是第一行的 `missing` 数字。

## 已知风险点（这个后端特有的，不是猜的，而且比其他三个后端更多）

- **fp64 的 H2D 拷贝静默返回全 0**（`../ILUVATAR_DEBUG.md` 已定位根因）。45 个新变体不含 f64/c64，
  理论上不受影响，但如果测试过程中任何中间步骤意外提升到 fp64 精度（比如某个参考实现），结果会静默
  错误而不是报错，这是这张卡上最危险的一类问题，运行时多留意。
- **`torch.sparse` 在这张卡上结果是错的，不是不支持**（`../ILUVATAR_DEBUG.md`）。早期一版文档把它
  当作性能基线，已经改成默认 `cupy_cusparse`。45 个新变体走的 `tests/q4_variant_bench.py`
  （`_time_pytorch`）从 2026-10-08 起会先拿 PyTorch 结果和 CPU 参考值比，相对误差超过 1e-2 就不计时、
  在 `pytorch_reason` 里写 `PyTorch result off by ...`——**所以这张卡上 45 个新变体的 PyTorch 列大量
  出现这个 reason 是预期行为**，说明拦截生效了。原有 20 个变体走各自脚本，不经过这道校验，那部分的
  PyTorch 数字仍然要先怀疑。
- **通用 `cusparseSpMV` 在这张卡上会申请约 140 TB 显存**，只能用 CoreX 的 legacy
  `cusparseScsrmv`/`cusparseScsrmm`，且只覆盖 fp32 + int32 + 不转置。45 个新变体里的转置/共轭/
  混合精度/int8 变体大概率没有厂商基线，这是预期行为。

## 跑完要确认

1. `missing` 是不是 0。
2. 任何一行的 PyTorch 列数字是不是"看起来正常但可疑"（极快、或者跟其他后端差异巨大）——按上面第二条
   风险点，这张卡的 PyTorch 列历史上出过静默错误的先例，数字本身不能直接当证据，需要交叉核对。
3. int8/混合精度变体的精度是不是真的通过了。

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
