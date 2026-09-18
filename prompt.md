# 给各后端 codex 的工作提示

你在一台**某个后端的实机**上（昇腾 / 摩尔线程 / 沐曦 / 海光 DCU / 昆仑芯 / …），
任务是**让 FlagSparse 的算子在这台机器上跑起来、出结果，并把改动以可移植的方式交回**。

把 `<BACKEND>` 替换成你这台机器对应的名字：

| `FLAGSPARSE_BACKEND` | 平台 | 文档前缀 |
|---|---|---|
| `cuda` | NVIDIA | 参照路径，见根 `README.md` |
| `rocm` | 海光 DCU | `DCU` |
| `metax` | 沐曦 MetaX | `MACA` |
| `mthreads` | 摩尔线程 | `MUSA` |
| `ascend` | 昇腾 | `ASCEND` |
| `xpu` | 昆仑芯 | `XPU` |
| `gcu` / `mlu` | 燧原 / 寒武纪（`mlu` 是通用备用槽位，只能由环境变量选中） | 尚无文档，你是第一个 |

---

## 0. 先确认你在哪，这一行不对下面全部作废

```bash
cd <仓库根> && export PYTHONPATH=$PWD/src
export FLAGSPARSE_BACKEND=<BACKEND>

python3 -c "import flagsparse; print(flagsparse.__file__)"
# 必须指向 <仓库>/src/flagsparse/__init__.py。指到 site-packages 说明跑的是别的副本，任何结果都不算数。

python3 -c "import flagsparse.sparse_operations._common as C; \
print(C._backend_name(), C._accel_device_type(), C._accel_fallback_reason())"
# 期望：<BACKEND> <设备类型> None
```

`fallback reason` 不是 `None`，说明厂商的 torch 插件没装好，**整轮会静默地跑在 CUDA 语义下**，
跑出来的不是你这台机器的数。摩尔线程和昇腾尤其要注意：它们是独立设备类型（`musa` / `npu`），
`torch.cuda.*` 和 `Tensor.is_cuda` 在那里不适用。

> 这个检查本身曾经是坏的：它只覆盖 mthreads 和 ascend，**xpu / gcu / mlu 是盲区**——
> 在没有插件的机器上选 xpu 会报告 backend=xpu、实际全跑在 CUDA 上、而这里返回 `None`。
> 现在判据从后端注册表派生，且要求命名空间和厂商插件**都在**（昆仑芯尤其重要：上游
> PyTorch 自带 `torch.xpu` 命名空间，光看它会把 Intel 卡认成昆仑芯）。runner 启动时
> 也会把这条警告顶到屏幕上，但**不要依赖它**——先自己跑上面那两行。

**第二件必须确认的事：你的矩阵集在哪。** 不给 `--benchmark-input` 时 runner 会**静默回落**到
仓库自带的 `tests/data`（只有 3 个小矩阵），照样 Passed、照样有加速比，报告的形状和 30 个
真实矩阵跑出来的**一模一样**。交付用的结果必须显式指向你自己的矩阵集：

```bash
python3 run_flagsparse_pytest.py --phase both --mode normal --delivery-only \
  --benchmark-input <你的矩阵目录> --benchmark-warmup 5 --benchmark-iters 20
```

**依赖**：`scipy` 是硬依赖但 `pyproject.toml` 里没有声明（那里 `dependencies = []`，连 torch
都没写，因为 torch 要装厂商自己的构建）。矩阵读取和非 CUDA 后端的精度参考都要它，
**版本建议 ≥1.14**：更早的 `spsolve_triangular` 是纯 Python 循环，13 万行的矩阵要跑几分钟
到几小时，会直接撞 `--timeout`。本机 scipy 1.15 上同样规模是 0.0 秒。

---

## 1. 开工前必读三份（都在本仓库）

| 文件 | 回答什么 |
|---|---|
| `docs/<BACKEND>.md` | 这台机器怎么跑：环境、命令、已知缺陷、排障速查表 |
| `modified/<BACKEND>.md` | **前人在这个后端上改过哪些文件的哪个函数、为什么** |
| `modified/README.md` | 改动台账的写法和三条硬规矩 |

没有你那个后端的文件，就照 `docs/MUSA.md` 和 `modified/MUSA.md` 的结构新建一份 ——
那两份是目前最完整的样板。各后端文档的对应关系见 `docs/README.md`（索引）。

---

## 2. 目标产物：40 个交付变体的结果

唯一真源是 `conf/operators.yaml` 的 `delivery_variants`（40 条，如 `gather_f32_int`、
`spmv_csr_f32_int_non`），Python 侧和 C API 侧由同一个加载器 `tools/delivery_variants.py`
读取。交付清单本身是 42 个：`sddmm_csr` 的 c32/c64 等复数内核，还没登记，所以现在报告出 40 行。
gather/scatter 的 f16 是交付变体（xlsx 漏写了），不是清单外的附加项。详见 `capi/docs/README.md`
"三份文件，三种口径"。跑完 runner 应当得到：

```
<结果目录>/summary.json        result 按变体名做键，40 条，schema 与 FlagGems 一致
<结果目录>/summary.csv         每变体每阶段一行
<结果目录>/result.html         变体粒度的报告
```

**不要改这个清单去迁就你的后端。** 某个变体在你这台机器上跑不了，就让它如实报
`NotFound` / `Skipped` / `Failed` 并写清原因 —— 空结果读起来和通过一模一样，那是最坏的结局。

> **⚠️ `86a09cd`（2026-09-18）之前跑出的 Python 侧性能结果作废，拉新代码后用
> `--delivery-only` 重跑。** 旧 runner 把结果投影到交付变体时有两个错，报出的加速比不是变体名
> 说的那个数：
>
> 1. **只按 dtype 筛行**：`spmv_csr_f32_int_non` 实际平均了 int32/int64 × non/trans/conj 六种组合；
> 2. **没有加速比的行按 0 计入均值**（失败行、没有基线的行），与 runner 自己"只对通过且两侧
>    都有时延的行求平均"的规则相反。
>
> CUDA 30 矩阵上重算，偏差最大的：`spmv_csr_f32` 3.07x → 1.19x、`spmv_coo_c32` 1.67x → 0.53x、
> `spsv_csr_f32` 1.89x → 3.73x、`spsm_csr_f32` 3.80x → 8.93x。gather/scatter/spmm_coo 基本不变。
>
> **受影响**：所有用 `tests/test_*.py` 跑性能的后端（CUDA、DCU、MACA、MUSA 的 Python 侧、Ascend
> 走通用脚本的算子）。**不受影响**：精度结果；C API 侧性能（`capi/tools/write_summary.py`
> 另有实现）；XPU 的 `benchmark_xpu.py`（CSV 没有 index/op 列）。旧结果目录里的
> `summary.json` 不会自己更新，必须重跑。

---

## 3. 怎么跑

```bash
# 精度（先做这个，全绿之前不要看性能）
python3 run_flagsparse_pytest.py --ops gather --phase accuracy --mode normal

# 精度 + 性能，指定矩阵目录（交付就用这条；各后端要改的环境变量和参数见
# README_cn.md "复现交付测试" 一节的表，完整做法见 docs/<BACKEND>.md 的"交付复现"）
timeout -s KILL 21600 python3 run_flagsparse_pytest.py \
  --phase both --mode normal --delivery-only --gpus 0 --timeout 3600 \
  --benchmark-input <矩阵目录> --benchmark-warmup 5 --benchmark-iters 20 \
  --results-dir pytest_results_<BACKEND>

# 按后端 suite 跑（会自动设好 FLAGSPARSE_BACKEND）
python3 tools/run_backend_tests.py --backend <profile> --phase accuracy --mode quick
#   profile 取值：cuda | rocm | maca | musa | ascend | xpu
```

四个参数不能用默认值，原因各不相同：

| 参数 | 默认 | 为什么必须显式给 |
|---|---|---|
| `--mode` | **`quick`** | quick 把每类 shape 砍到只剩一个，覆盖少四成，而且**恰好跳过历史上出过问题的两个用例**。用 quick 跑出来的全绿不算数 |
| `--timeout` | `0`（关闭） | 挂住就永远不往下走。它是**每个算子每个阶段**的超时，不是全局 |
| `--phase` | `accuracy` | 要性能数据得给 `both` |
| `--ops` | 读 yaml 全量 | 已知会挂死的算子（如某些平台的 SpSV/SpSM）要先排除，否则一轮跑不完 |

**`--delivery-only` 会自动收窄 benchmark sweep**（2026-09-18 起）：spmv / spmm / spsv 只跑
`int32` + `non`，gather / scatter 只跑 `int32` 和交付 dtype，启动时每个被收窄的算子打一行
`delivery-only: <op> benchmark narrowed with ...`。默认 sweep 还包含 int64 和 trans/conj
（spmm_csr 720 组里只有 120 组是交付的），不收窄时 MetaX C550 上 900 秒跑不完。
自己传的 `--benchmark-args` / `--op-benchmark-args` 优先；`sddmm_csr` 的 K sweep 不在收窄
范围内（交付名里没有 K）。XPU、Ascend 的专用脚本和探测类后端不注入这些参数。

跑完用 `python3 tools/delivery_table.py <结果目录>` 打印 40 行结果表（缺变体时退出码为 1），回传时直接贴它的
输出。摩尔线程改用 `run_flagsparse_split_delivery.py`（性能取自 C API，才有 muSPARSE 加速比），见 `docs/MUSA.md` 0.5 节。

**三角类算子（SpSV / SpSM）永远套 `timeout -s KILL`** —— 内核挂死时 Ctrl-C 送不进去，
进程阻塞在驱动里，代价是整个容器重开。

---

## 4. 改代码的三条硬规矩

**一、改动必须带后端守卫，让其他后端是恒等变换。**

```python
if _is_<backend>_runtime():      # 来自 _common.py，已在 __all__ 里
    ...
```

无法守卫的（比如测试文件整体换设备抽象）要在台账里写明影响面和验证方式。
用的名字必须是 `_common.__all__` 导出过的 —— 曾经有 drop 用了未导出的 `_IS_ASCEND_RUNTIME`，
留下一个只在特定分支才触发的 NameError。

**二、不要 fork 内核实现。** 代价从低到高：

1. 只是 launch 参数或开关不同 → **加 profile 表项**，内核不动（参考 `_MACA_SPSV_PROFILES`）
2. 需要不同算法路径 → **同文件内加一个内核变体**，和 CUDA 版本挨着放
3. 整个算子在该后端跑不通 → **fallback 分发表**（参考 `SPSM_ASCEND_DISPATCH`）
4. 确实要整份文件不同 → 才在 `backends/<backend>/` 放那**一个**文件

`backends/` 是默认为空的覆盖层，`_dispatch.py` 的 `install_overrides()` 会在包初始化时
把存在的覆盖注册进 `sys.modules`。`tests/ci/test_backend_module_dispatch.py::test_backend_directories_are_empty`
会挡住批量复制 —— 那是有意的评审点。这个仓库有过教训：六个后端目录装着 19/20 文件 md5
完全相同的拷贝，20 万行重复，改一个 bug 要改六遍。

**三、没跑过就写"没跑过"。** 本仓库已经有两份文档开头写着"一行都没在真机跑过"、
正文却全是实测结论的先例。每条结论都要能回答"这是实测还是推断"。

---

## 5. 回传：传台账，不传文件

**不要把整份文件发回来。** 今天为此付过两次代价：

* 传了 6 个改动文件，但它们 `import` 的**新模块没一起传**，拷进去全部 ImportError；
* 基于旧副本改的 runner 传回来，`+148/−230` —— 少掉的 230 行里有 137 行是别人当天刚做完的
  工作，整份覆盖会把 40 变体的 `summary.json` 打回按算子名出结果。

正确做法是在 `modified/<BACKEND>.md` 里加条目，五列固定：

| 列 | 要求 |
|---|---|
| 文件 | 仓库相对路径。**新建的文件也要列** |
| 锚点 | **函数名 / 常量名，不是行号**（行号一次合并就漂） |
| 改动 | 一句话；必要时贴 3–5 行代码，不贴整个函数 |
| 为什么 | 触发它的现象。**没有现象就不该有这条改动** |
| 其他后端 | 恒等变换 / 影响哪些，以及守卫写法 |

文件顶部三行：**实机环境指纹**（SDK、torch、卡型）、**基于哪个版本改的**、**这轮回传了什么**。
真要传文件时（新增脚本这类），**先 pull 一次再改**，并在台账里写清基线版本。

---

## 6. 验收

```bash
python3 -m pytest tests/ci -q        # 当前基线：87 passed / 3 skipped
```

`tests/ci` 不需要 GPU，是策略/契约测试。**任何后端改动都不应该让它变红**；变红就说明
改到了共享契约，那要么是 bug，要么需要一次有意的评审。

CUDA 侧的回归基线（供对比）：`tests/pytest` 1613 passed / 3 failed，那 3 个是既有 flaky
（随机输入未固定种子，CUDA 上同样复现）。

### 提交前跑完整的那套

```bash
git add <你新建的文件>          # 见下，这一步不能省
make compile format-check lint lint-src pre-commit-check test-ci
```

**⚠️ 新建的文件必须先 `git add`，否则 pre-commit 看不见它。** `pre-commit run --all-files`
只处理 **git 跟踪中的文件**，未跟踪的新文件会被整个跳过——本地一路绿灯，推上去 CI 立刻红。
这个坑刚发生过：一个新增的 CI 测试文件在本地跳过了 isort，提交后 CI 在 `pre-commit-check`
挂掉，原因只是 import 块后多了一个空行。`git add` 之后（不必 commit）hook 就能看见。

**还有两条与 CI 环境有关的，写新测试时注意**：

* **CI 的 runner 是 CPU-only，不装 torch**（`tools/ci/requirements-ci.lock.txt` 只有工具链）。
  `tests/ci` 里任何会 import `flagsparse`（进而 import torch）的测试——包括通过子进程
  间接 import 的——都要在模块顶部加
  `pytest.importorskip("torch", reason="tests/ci runs on a CPU-only runner without torch")`，
  否则在 CI 上是 failed 而不是 skipped。要验证，用假 torch 模拟那台机器：

  ```bash
  echo 'raise ImportError("simulated")' > torch.py
  python3 -m pytest tests/ci -q      # 应当只有 skipped，没有 failed
  rm torch.py
  ```

* **`make ci` 的步骤是有顺序的**：`format-check → lint → lint-src → pre-commit-check →
  build → install-wheel → test-ci`。前面任何一步失败，后面的**根本不会跑**——所以"CI 只报了
  一个错"不等于"只有一个错"。本地按上面那条命令一次跑完，才能看到全部。

---

## 7. 排障通用条（跨后端都适用，展开见 `docs/MUSA.md` 第 10 节）

* **一律 `timeout -s KILL`**；Ctrl-C 打不断卡死的 GPU 内核；
* **一个配置一个进程** —— 内核 fault 之后厂商 runtime 被污染，同进程后续结果全是垃圾；
* **照字面读厂商报错，先看 stderr** —— 真正的诊断常常打在 stderr，Python 异常只说一句
  没有信息量的话。沐曦那次"每线程 8 KB 私有内存超 4 KB 上限"就明写在上一行；
* **"没输出"不等于"通过"** —— 真实发生过三次，包括漏传一个文件导致整组 FAIL、
  看起来和硬件结论一模一样；
* **`timeout -s KILL` 的返回码，shell 看到 137，Python 的 subprocess 看到 -9**；
  `-11`(SIGSEGV) 是崩溃不是卡死，要分开；
* **确认通过的测试到底跑的是哪条路径** —— SpSV 的 unit/non-unit 走两个完全不同的内核；
* **诊断代码本身也是嫌疑人** —— 在不成熟的 runtime 上，花哨的张量操作本身可能是坏的；
* **别从"剩余失败数"估"剩余缺陷数"** —— 每修好一层才露出下一层。

---

## 8. 不要做的事

* 不要为了让 `test_installed_wheel` 变绿去 `pip install` 本仓库 —— site-packages 里的快照
  会遮蔽后续源码改动，那是个查起来很费时的坑。走 `PYTHONPATH=src` 时它必然失败，属正常；
* 不要改 `conf/operators.yaml` 的 40 条交付清单去迁就跑不通的算子；
* 不要把测试产物（`pytest_results_*/`、`capi_results/`、benchmark CSV）提交上来，
  `.gitignore` 已经覆盖；
* 不要在没有实机依据的情况下回填 profile 表 —— 那些 knob 是**行为选择**而非硬件事实，
  不能靠指纹推断；
* 不要把一刀切的 `skipif` 当成"算子通过"。它掩盖的通常是一个真实的适配缺口：
  `spmm_bell` 曾经 54/54 被跳过，理由是"requires CUDA tensors"，去掉之后 54 全过。
