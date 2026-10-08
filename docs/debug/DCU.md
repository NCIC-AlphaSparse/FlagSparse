# 海光 DCU（BW1000，ROCm）：65 变体验证任务

`FLAGSPARSE_BACKEND=rocm`（通常能自动识别）。环境搭建、交付复现见 [`../DCU.md`](../DCU.md)。背景见
本目录 [`README.md`](README.md)。

## 要做的事

环境按 `../DCU.md` 第 1、2 节配好后（**务必先做第 3 节的 diagnose，不要直接跑基准**）：

```bash
task_tmp=$(mktemp -d /tmp/flagsparse-dcu-65.XXXXXX)
export TMPDIR="$task_tmp" TMP="$task_tmp" TEMP="$task_tmp"   # Slurm 残留的 TMPDIR 会让 clang 编译失败，见 ../DCU.md 第 0 节
export HIP_VISIBLE_DEVICES=<卡号>                              # runner 用 CUDA_VISIBLE_DEVICES 隔离设备，ROCm 上会打乱随机种子
setsid timeout -s KILL 43200 python3 -u run_flagsparse_pytest.py \
  --ops gather,scatter,axpby,spvv,spmv_sell,spmv_csr,spmv_coo,spmv_csc,spmm_csr,spmm_coo,spmm_csc,spgemm_csr,sddmm_csr \
  --phase both --mode normal --gpus 0 --timeout 3600 \
  --benchmark-input <矩阵目录> --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir results_rocm_65_<日期> \
  > results_rocm_65_<日期>.log 2>&1 < /dev/null &
# 跑完后
python3 tools/delivery_table.py results_rocm_65_<日期>
```

- `--timeout 3600`、外层 KILL 限时沿用 `../DCU.md` 的 20 变体交付命令。

- 外层 `timeout -s KILL` 是整条命令的总限时（内核卡死时 Ctrl-C 送不进去，只能靠 KILL）；`--timeout` 是
  每个算子每个阶段的限时。这两个数是 20 变体时期定的，65 变体多了 6 个父算子，**没有在本机实测过总耗时**，
  到点被杀的话按 `delivery_table.py` 里 `NotFound` 的算子单独补跑（补跑要用新的 `--results-dir`）。
- `--results-dir` 每次都用新目录：`summary.json` 每跑一次就整体重写，往旧目录里补跑一部分会把之前的结果冲掉。

把 `delivery_table.py` 的完整输出带回来，尤其是第一行的 `missing` 数字。

## 已知风险点（这个后端特有的，不是猜的）

- **hipSPARSE 只支持不转置**（`../DCU.md` 第 4 节）。CSR SpMM 的 `trans`/`conj` 没有厂商基线，这次
  新增的转置/共轭变体（`spmv_csr_f32_int_trans`、`spmv_coo_f32_int_trans`、`spmv_csr_c32_int_conj`、
  `spmm_csr_c32_int_conj_non_row`、`spmm_csr_f32_int_trans_non_row` 等）预期厂商列都是空的，只有
  PyTorch 对照，这是预期行为。`tests/cusparse_generic_baseline.py`（给 q4 变体用的 ctypes cuSPARSE
  绑定）本身就只接 CUDA，在 DCU 上会整体 `skip_reason()` 判空，不会尝试调用 hipSPARSE——所以 45 个新
  变体大概率**全部**没有厂商基线，不止转置/共轭那几个。
- **int8 变体最该先验**：`gather/scatter_i8`、`spmv_csr_i8i32`、`spmv_csr_i8f32`、`spmm_csr_i8i32`、
  `spvv_i8i32`——int8 读、int32 累加、int32 原子加在 DCU 的 Triton 上支持到什么程度，以前没有专门测过。
- **`spmv_csr` 性能本身就偏慢**（`../DCU.md` 已记录 ~0.27x 对比 hipSPARSE，跟这次改动无关，是已知
  未解决问题），新增的 `spmv_csr` 变体性能数字预期也会低，不要误判成这次改动引入的回归。
- **如果 `spmv_csr_f16_int_non`/`c32_int_non`/`f32_int_trans`/`c32_int_conj` 这几个变体精度报
  `NotImplementedError: CSR SpMV row_tile: unverified capabilities for rocm/<arch>`，这是一个需要
  立刻上报、不要自己绕过去的信号**。原因：`auto` 模式下 ROCm 的 CSR SpMV 固定选
  `row_tile`（`spmv_csr.py::_configure_spmv_route` 里 `"row_tile" if _is_rocm_runtime() else
  "legacy_" + backend`，这是 gfx936 调优过的生产路径，不是新代码临时选择），而 `row_tile` 要求
  `_spmv_backend_caps()` 算出 `verified=True`——这个值只在 `triton.runtime.driver.active
  .get_current_target()` 真的报 `backend=="hip"` 且 `arch` 以 `"gfx"` 开头时才成立
  （`spmv_csr.py:_spmv_backend_caps`）。这条在 CUDA 机器上**不可能**模拟到（Triton 驱动查的是真实
  硅片，设什么环境变量都测不出 `hip`/`gfx`），所以这 4 个变体在单纯设环境变量的 CUDA 模拟里全部挂在
  这一步。之后补测过：把 `_spmv_backend_caps()` 改成返回 gfx936 的已验证能力，让 `row_tile` 带着
  ROCm 调优参数真实跑起来，45 个新变体 91/91 全过——逻辑本身没问题，但前提是真实 DCU 上 Triton-ROCm 驱动能正确报出
  `hip`/`gfx*`。**如果真机上也报这个 `NotImplementedError`，说明 Triton-ROCm 驱动没有正确识别硬件，
  这是真 bug（或者环境没装对），务必带回具体的 `arch`/`target` 字符串。**

## 跑完要确认

1. `missing` 是不是 0。
2. int8 变体的精度是不是真的通过了（不是因为没有基线列、看起来"没报错"就当作过了）。
3. 转置/共轭变体性能列是不是如预期全是 N/A（厂商列）+ 有效数字（PyTorch 列），而不是报错或空值。

把这三条的结果写回来，不用额外分析，原始数据最有用。

## 交给 Codex 的 prompt

上机时把下面整段原样贴给 Codex（或其他代理）。**贴之前把 `<矩阵目录>` 和 `<卡号>` 换成本机实际值。**

````text
你在 海光 DCU（ROCm） 实机上复测 FlagSparse 的 65 个交付变体。仓库在当前目录，main 分支。

背景：65 个变体（原 20 个 + 新合并的 45 个）只在 CUDA 上实测过，本机从没跑过。
任务是在本机跑一遍，把原始结果带回来。主要目的是收集数据，不是修代码。

步骤：
1. git pull --ff-only origin main。
2. 按 docs/DCU.md 配好环境，做完它的环境自检。
3. 确认导入的是仓库源码：python3 -c "import flagsparse; print(flagsparse.__file__)"
   输出必须在当前目录的 src/ 下。如果指向 site-packages / dist-packages，先停下来汇报，不要继续。
   旧安装包会让基线列全部变成 N/A，看起来像正常结果。
4. 完整读一遍 docs/debug/DCU.md，照"要做的事"一节的命令原样跑。
   命令里的 --benchmark-input 填 <矩阵目录>，HIP_VISIBLE_DEVICES 填 <卡号>。
   不要自己删减或改写参数，每个参数的来由那一节都写了。
   - --results-dir 用新目录（<日期> 填今天），不要复用旧目录。
   - 命令本身已经用 setsid 放到后台，定期看日志进度，不要中途打断。
   - 不要用 pytest --forked，它在 GPU 上会让所有用例失败。
5. 跑完执行 python3 tools/delivery_table.py <结果目录>。

规矩：
- 不要为了"让它通过"去改 src/ 或 tests/ 下的代码。确实非改不可（比如环境适配）时，每改一个文件，
  都按 docs/debug/DCU.md 末尾"实机改动记录"一节的格式追加一条记录，并和代码改动放在同一个
  commit 里。没改就在那一节写"无"。
- 不要猜根因。遇到报错就给出完整报错原文，以及出错变体在结果目录里那一行的 reason/error 字段。
- 如果 spmv_csr_f16_int_non / c32_int_non / f32_int_trans / c32_int_conj 报
  unverified capabilities for rocm/...，这是真问题，要带回这行命令的输出：
  python3 -c "import triton; t=triton.runtime.driver.active.get_current_target(); print(t.backend, t.arch, t.warp_size)"

汇报包含：
1. delivery_table.py 的完整输出，尤其是第一行 "N registered variants, M missing"。
2. 所有非 Passed 的变体：变体名、阶段（精度/性能）、reason 或 error 原文。
3. 逐条回答 docs/debug/DCU.md "跑完要确认"一节。
4. 环境指纹：torch、triton 版本，设备名，warp size，驱动/SDK 版本，整轮总耗时。
5. docs/debug/DCU.md "实机改动记录"一节的内容。
````

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
