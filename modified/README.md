# 各后端改动台账

**这个目录记的是"谁在哪个文件上改了什么"，不是代码。** 目的很具体：各后端调试时不要
再把整份文件传来传去 —— 传文件的代价今天已经付过两次（见下面「为什么」）。一个后端一个
文件，与 `docs/<BACKEND>.md`（怎么在那台机器上跑）和 `capi/docs/<BACKEND>.md`（C API 那层）
并列，这里只回答一个问题：**这轮在实机上动了哪些文件、动的是哪一段、为什么。**

| 文件 | 后端 |
|---|---|
| [CUDA.md](CUDA.md) | NVIDIA —— 参照路径，说明"没有后端分支"意味着什么 |
| [DCU.md](DCU.md) | 海光 DCU / ROCm |
| [MACA.md](MACA.md) | 沐曦 MetaX |
| [MUSA.md](MUSA.md) | 摩尔线程 |
| [ASCEND.md](ASCEND.md) | 昇腾 |
| [XPU.md](XPU.md) | 昆仑芯 |

---

## 为什么不传文件

两次真实教训，都发生在 2026-09-16：

**一、传了改动，没传它依赖的新文件。** MUSA 那批带回 6 个 benchmark，每个都
`from benchmark_utils import ACCEL, accelerator_device`，而 `benchmark_utils` 是新建的、
不在那批里，仓库里也没有 —— 6 个文件拷进来全部 ImportError。补一份临时的顶上，实机随后
把真的那份发来，又得比对替换一次。**台账里写一行"新增 tests/benchmark_utils.py"就不会发生。**

**二、基于旧副本改，回传时会删掉别人的工作。** XPU 那份 `run_flagsparse_pytest.py` 与
仓库版差 +148/−230 行。多出来的是 XPU 路由，**少掉的 230 行里有当天刚做完的交付变体投影**
（`_delivery_results` / `summarize_accuracy_cases`，共 137 行）—— 整份覆盖的话，
40 变体的 `summary.json` 当场退回按算子名出结果。最后是逐个 hunk 摘出来移植的。
**台账里写清"改了哪个函数"，就只需移植那几个函数，不需要看整份 diff 猜哪边更新。**

---

## 一个条目怎么写

每个后端文件里，改动表的五列是固定的：

| 列 | 要求 |
|---|---|
| 文件 | 仓库相对路径。**新建的文件也要列**，这是教训一 |
| 锚点 | **函数名 / 常量名，不是行号。** 行号一次合并就漂了；`_delivery_accuracy_phase()` 十年后还在 |
| 改动 | 一句话说清做了什么。必要时贴 3-5 行代码，不要贴整个函数 |
| 为什么 | 触发它的现象。**没有现象就不该有这条改动** |
| 其他后端 | 恒等变换 / 影响哪些。守卫写法一并写出，例如 `_is_mthreads_runtime()` |

另外每个文件顶部要有三行：**实机环境指纹**（SDK、torch、卡型）、**基于哪个版本改的**、
**这轮有没有回传、回传了什么**。

## 三条硬规矩

1. **改动必须带后端守卫**，让其他后端是恒等变换；无法守卫的（比如测试文件整体换设备
   抽象）要在「其他后端」列写明影响面和验证方式。
2. **不要 fork 内核实现。** 代价从低到高：profile 表项 → 同文件内加内核变体 →
   fallback 分发表 → 最后才是 `backends/<backend>/` 放整份文件。
   `tests/ci/test_backend_module_dispatch.py::test_backend_directories_are_empty` 会挡住
   批量复制，那是有意的评审点，不是障碍。**这条规矩是实测换来的，见下。**
3. **没跑过就写"没跑过"。** 本仓库已经有两份文档开头写着"一行都没在真机跑过"、
   正文却全是实测结论的先例。台账里的每一行都要能回答"这是实测还是推断"。

---

## 规矩 2 的由来：按后端复制代码，代价是多少

这个仓库真的这么干过一次。合并时把 Triton 算子按后端拆成
`sparse_operations/backends/{cuda,rocm,maca,musa,ascend,xpu}/`，每个目录 20 个文件。实测：

* **19/20 文件在六个后端下 md5 完全一致**，唯一差异是 `__init__.py` 的一行 docstring；
* `backends/cuda/` 的 19 个算子文件与**上游单份实现逐字节相同** —— 拆分只是复制，
  没有做任何后端定制；
* 代码量 240,752 行，其中约 **20 万行是重复**；改一个内核 bug 要改六遍；
* 顺带**丢了 `gcu` 和 `mlu`** —— 分发表只映射 6 个，这两个后端直接抛 `RuntimeError`；
* 最直接的证据：上游全仓 **80 处**后端条件分支，六份拷贝里**也是 80 处、逐文件分布相同**。

**拷贝最危险的地方是让分歧不可见。** 若 `musa/spsv.py` 与 `cuda/spsv.py` 出现差异，
看 diff 无法判断它是有意的移植，还是 CUDA 侧修了 bug 而 MUSA 漏了同步。这不是假设：
本仓库已有先例 —— 某次 drop 的整体合并会回退上游工作，根因正是同一份代码存在多个副本。

上游处理后端差异的四种机制，**全都在单份代码里**，这就是代价阶梯的出处：

| 机制 | 代价 | 例子 |
|---|---|---|
| **`tl.constexpr` 折叠** | 零 | `HAS_BETA` `SEG_IS_ROW` `SCAN_BACKWARD` `ACC_IS_FP64`，生成的代码逐字节不变 |
| **按型号的调参 profile** | 零 | `_MACA_SPSV_PROFILES` 按 `_maca_device_model()` 选；`WARP_SIZE=64 if _is_rocm_runtime() else 32` |
| **同文件多内核变体** | 低 | `spsv.py` 里 38 个 `@triton.jit`，如 `_levelschd_analysis_{,serial_,persistent_}kernel` |
| **fallback 分发表** | 中 | `SPMM_COO_ASCEND_DISPATCH`、`SPSM_ASCEND_DISPATCH` —— 只有 Ascend 需要 |

注意第二行：上游是按**设备型号**调参的，按目录分后端连这一层都表达不了。

现在 `backends/` 是**默认为空的覆盖层**：只有真正分叉的文件才会出现在里面，一眼看出
哪个算子在哪个后端分了叉 —— 这比六个装满相同文件的目录信息量大得多。
