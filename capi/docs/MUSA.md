# MUSA (摩尔线程)

`BACKEND=MUSA` / `FLAGSPARSE_BACKEND=mthreads`。**除 CUDA 外唯一有 adaptor 实现的后端**。

## 环境检查
> 这一层离不开可导入的 `flagsparse` 包：内核是 **re-export** 而不是拷贝，所以
> `FLAGSPARSE_PYTHON_SRC` 指错、或机器上有旧副本，症状分别是 `EXECUTION_FAILED`
> 和 `CompilationError`。依赖关系、版本配套和部署清单见
> [README.md#这个库离不开-flagsparse](README.md)。


```bash
mthreads-gmi                                    # 或 musa-smi，视驱动版本
ls $MUSA_HOME/lib64/libmusart.so $MUSA_HOME/include/musa.h
python3 -c "import torch, torch_musa; print(torch.__version__, torch.musa.is_available())"
export FLAGSPARSE_BACKEND=mthreads
```

CMake 要三样，缺任一项会**指名报错**而不是在链接期报缺符号：
`$MUSA_HOME/lib64/libmusart.so`、`libmusa.so`、`$MUSA_HOME/include/musa.h`
（`MUSA_HOME` 默认 `/usr/local/musa`）。

## 构建

```bash
cmake -S . -B build -G Ninja -DBACKEND=MUSA -DMUSA_HOME=/usr/local/musa \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`src/adaptor/backend/musa/adaptor.cpp` 是 CUDA 那份**加一个符号前缀**——摩尔线程的
driver API 与 CUDA 逐符号镜像，所以两者共用 `backend/_template.inc`，25 行。这也是
填其他预留槽位的模板。

## 跑测试

同 [README.md](README.md) 的通用流程。**先跑 `ctest -R pytest`**：MUSA 上 Triton 是
健康的（见下），所以内核层失败大概率是环境，不是代码。

## 这个平台上量过的能力边界

来自 S5000 上的逐层实测，**不是文档抄来的**：

| 层 | 状态 |
|---|---|
| Triton | **完全健康**，含 `associative_scan` |
| muDNN | 缺口在 **gemv**，不是 fp64 —— 这一点容易搞反 |
| `torch.sparse` | **没有任何可用的 matmul** |

最后一条直接决定了测试设计：**精度参考解恒在 host 上以 fp64 计算**。在这块卡上
"拿加速器算参考解"根本不成立，而这个选择让同一套 ctest 在每个后端都成立。

## 跑测试

```bash
export FLAGSPARSE_BACKEND=mthreads

cmake -S . -B build -G Ninja -DBACKEND=MUSA \
      -DCMAKE_BUILD_TYPE=Release -DFLAGSPARSE_CTEST_TIMEOUT=3600
cmake --build build -j
# 配置日志里应有 "ctest baseline: MUSA -> .../libmusparse.so"，见下面"跑不起来时"第 2 条

# 精度
ctest --test-dir build -R accuracy --output-on-failure

# 性能：真实矩阵语料。不设 FLAGSPARSE_MATRIX_DIR 会退回合成形状，
# 并把每行标成 corpus=synthetic，不会被误读成真实数据
FLAGSPARSE_MATRIX_DIR=/path/to/mtx FLAGSPARSE_BENCH_OUT=./bench \
    ctest --test-dir build -R benchmark

# 汇总：40+ 变体一张表 / 与 FlagSparse Python 侧同 schema 的 summary.json
python3 tools/report.py --bench-dir ./bench
python3 tools/write_summary.py --bench-dir ./bench --out ./bench

# 算子清单与实现是否一致（CI 可用 --strict）
python3 tools/check_manifest.py --bench-dir ./bench
```

`FLAGSPARSE_CTEST_TIMEOUT` 是每个 ctest 用例的超时（秒），默认 900：合成小矩阵够用，
30 个真实矩阵的完整 dtype 网格不够（和 Python 侧 `--timeout` 同一个问题），真实语料用 3600。

算子清单来自 `conf/operators.yaml`，构建时由 `tools/gen_variants.py` 生成成扫描用的
变体表——**加算子是改 YAML 加重新构建，不动测试代码**。

### 这个后端上预期会看到什么

* **`speedup` 列应当有值。** 之前这里写的是"大概率是空的（musparse not found）"——那是
  `ctest/baseline/CMakeLists.txt` 读错了变量名（读 `FLAGSPARSE_BACKEND` 而不是 `BACKEND`）
  造成的，不是没装 muSPARSE：S5000 上 `/usr/local/musa/lib/libmusparse.so` 和 `musparse.h`
  都在，修复后探测到了（2026-09-17，见 `modified/MUSA.md` 第 10 节）。现在如果还是空的，
  按下面"跑不起来时"第 2 条查；
* **精度的参考值必须是 host fp64**，不能用 torch.sparse —— MUSA 上 `torch.sparse`
  没有可用的 matmul。ctest 本来就用 host 参考，这一条是提醒别在 Python 侧照搬。

### 跑不起来时按这个顺序查

1. **配置就停下** → adaptor 没写。MUSA 的 driver API 镜像 CUDA，所以 adaptor 是
   `backend/_template.inc` 加一个符号前缀表（参考 `backend/musa/adaptor.cpp`，25 行）；
2. **`ctest baseline: MUSA -> none`** → **不正常，去查原因**（`$MUSA_HOME` 指错、SDK 没装
   muSPARSE，日志里 `none` 后面会跟原因）。这里曾写成"正常"，是因为变量名 bug 存在期间它永远
   打印 `CUDA -> none`、从来不会真的打印 `MUSA -> none`，所以那条结论没被检验过。如果看到的是
   **`CUDA -> none`**，说明构建脚本还是修复前的版本；
3. **某个算子 `not_supported`** → 先看是不是内核没 lower。MUSA 上 Triton 是健康的
   （associative_scan 都能用），所以更可能是 dispatch 没接；
4. **gemv 相关的性能异常** → muDNN 的缺口是 gemv 不是 fp64，别往精度方向查。

## 基线：muSPARSE 4.3.5，原生实现（不是前缀表）

`ctest/baseline/musa/baseline.cpp` 是**实机写的 420 行原生实现**，与 cuda/rocm/maca 三个
槽位不同——那三个是 35 行前缀表加共享的 `ctest/baseline/_template.inc`。

**为什么模板在这里不成立**：muSPARSE 4.3.5 的描述符模型与 cuSPARSE 大体同形，但
**SpMM / SpSV / SpSM / SpGEMM 是一个入口加 stage 参数**，而不是 cuSPARSE 那样拆成
`_bufferSize` / `_preprocess` / `_solve` 几个函数：

```
cuSPARSE:  cusparseSpMM_bufferSize(...)  cusparseSpMM_preprocess(...)  cusparseSpMM(...)
muSPARSE:  musparseSpMM(..., MUSPARSE_SPMM_STAGE_BUFFER_SIZE, &bytes, nullptr)
           musparseSpMM(..., MUSPARSE_SPMM_STAGE_PREPROCESS,  nullptr, scratch)
           musparseSpMM(..., MUSPARSE_SPMM_STAGE_COMPUTE,     nullptr, scratch)
```

前缀表把两套 API 当成逐调用对应，产出的是**看着能编、实则无效**的基线——文件开头那段
注释就是这么写的。

计时口径与其他后端一致：**BUFFER_SIZE 和 PREPROCESS 在计时循环外，只有 COMPUTE 进循环**。

两处实机踩到的坑，都写在代码注释里：

* **进程级 handle 故意不析构。** benchmark 进程在 C++ 静态析构之前就拆掉了 MUSA/Python
  运行时，4.3.5 在那之后销毁 handle 会 fault。那点 host 内存留给进程退出回收。
* **SpGEMM 仅 CSR、且要求方阵**；COO 输入直接返回 `Status::no`，不是崩。

名字与探测（`musparse` / `<musparse.h>` / `$MUSA_HOME` 默认 `/usr/local/musa`）已实机核对，
`conf/operators.yaml` 的 `performance_baseline` 因此是 `mthreads: musparse`。SDK 缺失时
CMake 仍照旧回落 `baseline/none` 并打印原因。

> **注意**：上面这段"已实机核对"只核对了名字和文件路径。2026-09-17 之前，按后端挑基线的
> 选择逻辑本身**从未真正执行过**——`_bl_backend` 永远落到 `CUDA` 默认值，非 CUDA 构建一律
> `CUDA -> none`。修复后 S5000 上已确认构建并链接到真实 `libmusparse.so`；muSPARSE 各 stage
> 调用的实测加速比以 `modified/MUSA.md` 第 10 节回填的数字为准。

## 注意

* 复数走实部/虚部交错数组（Triton 没有复数类型），和 Python 侧 `view_as_real` 一致；
* 原子路线（CSC `non`、BSR 两个方向）在任何后端上 fp32 都不是逐位可复现的 ——
  那是原子累加的性质，不是缺陷；
* Python 层的测试方法和排障记录见隔壁 checkout 的 `docs/MUSA.md`。本文只讲 C API 这一层。
