# MACA (沐曦)

`BACKEND=MACA` / `FLAGSPARSE_BACKEND=metax`。**预留槽位**：SDK 位置已填，
缺 `src/adaptor/backend/maca/adaptor.cpp`。

## 现状

```
deps/libtriton_jit   ✓ 有 BackendMACA.cmake（头 mcr/mc_runtime.h，库 mcruntime）
FlagSparse adaptor   ✗ 未写
厂商基线             ~ 名字已填待探测：ctest/baseline/maca/baseline.cpp
Python 层            ✓ 可用，_backend_name() 返回 'metax'
```

## 环境检查
> 这一层离不开可导入的 `flagsparse` 包：内核是 **re-export** 而不是拷贝，所以
> `FLAGSPARSE_PYTHON_SRC` 指错、或机器上有旧副本，症状分别是 `EXECUTION_FAILED`
> 和 `CompilationError`。依赖关系、版本配套和部署清单见
> [README.md#这个库离不开-flagsparse](README.md)。


```bash
mx-smi
ls $MACA_PATH/include/mcr/mc_runtime.h $MACA_PATH/lib/libmcruntime.so   # 默认 /opt/maca
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)"
export FLAGSPARSE_BACKEND=metax
```

**探测这个平台是个特例，值得单独说**：MACA 提供的是 CUDA 兼容栈 ——
`torch.version.cuda` 有值、`torch.version.hip` 是 `None`，所以 ROCm 那条探测
**分辨不出它和 NVIDIA**。库里靠设备名里的 `metax` / `maca` / `mxc` / `xcore` 判定。
这也是为什么后端注册表里 MACA 那条只有 `device_tokens`、没有 `torch_namespace`。

## 填这个槽位要做什么

一个文件：`src/adaptor/backend/maca/adaptor.cpp`。MACA 的 driver API 镜像 CUDA，
所以那是 `_template.inc` 加一个符号前缀，25 行 —— 照 `backend/musa/adaptor.cpp` 写。
SDK 位置表里已经有了（`MACA_PATH` / `mcruntime` / `mcr/mc_runtime.h`），不用再查。

## 跑测试

```bash
export FLAGSPARSE_BACKEND=metax

cmake -S . -B build -G Ninja -DBACKEND=MACA \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

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

算子清单来自 `conf/operators.yaml`，构建时由 `tools/gen_variants.py` 生成成扫描用的
变体表——**加算子是改 YAML 加重新构建，不动测试代码**。

### 这个后端上预期会看到什么

* **`speedup` 列大概率是空的**，配置时会打印 `ctest baseline: MACA -> none` 和
  「mcsparse not found under /opt/maca」。这不是失败 —— 基线名字是按沐曦的 CUDA 镜像
  惯例填的，未经验证，探测不到就回落。精度和计时仍然是真数据；
* 让它变成有值，见上面「基线」一节那张三 token 表。

### 跑不起来时按这个顺序查

1. **配置就停下** → adaptor 没写。SDK 位置从
   `deps/libtriton_jit/cmake/BackendMACA.cmake` 读（头 `mcr/mc_runtime.h`，库
   `mcruntime`），两边找同一套 SDK 所以不会漂；
2. **`ctest baseline: MACA -> none`** → 正常。要接基线就核对那三个 token 后设
   `MACA_PATH`；猜错只会继续回落，不会坏构建；
3. **内核编不出来** → 先确认 Python 侧同一个算子能跑（见隔壁 checkout 的
   `docs/MACA.md`）。C API 和 Python 走的是同一份内核，Python 跑不通说明
   问题不在这一层；
4. **结果对但慢得离谱** → 查 `warp_size`。MetaX C550 实测是 64。

## 基线：mcSPARSE，按命名惯例填好，等探测

`ctest/baseline/maca/baseline.cpp` 是 35 行前缀表加共享的
`ctest/baseline/_template.inc`。沐曦 MetaX 的工具链整体镜像 CUDA 并加 `mc` 前缀——本仓库
在 runtime 这一层已经依赖这个惯例（mcruntime / mcr/mc_runtime.h），所以稀疏库按同族推为
`mcsparse` / `<mcsparse.h>`，其 generic（描述符）API 与 cuSPARSE 逐调用对应。

**这些名字未经验证**，但在这里是可接受的：`ctest/baseline/CMakeLists.txt` 会在
`$MACA_PATH`（默认 `/opt/maca`）下用 `find_path`/`find_library` 探测，任一缺失就回落到
`baseline/none` 并打印"在 X 下没找到 mcsparse"。`src/adaptor/CMakeLists.txt` 里那条
**不猜库名**的规矩针对的是链接期报一个既不指明厂商也不指明错误的缺失符号——探测过后
才选用的槽位不是那种情况。

拿到 SDK 后**最先核对这三个 token**，过了就能链接：

| | 值 | 备注 |
|---|---|---|
| 头文件 | `<mcsparse.h>` | 可能带命名空间目录，hipSPARSE 就是 |
| 数据类型枚举 | `MC_R_32F` | 来自 **runtime** 而非稀疏库（`mcDataType_t`） |
| 库名 | `libmcsparse.so` | |

## 这个平台上踩过的坑

以下都是 **C550 上实测**的，不是推测：

* **`warp_size` 是 64**。写死 32 的地方都会错，SpMV COO / BSR 的 `num_warps` 直接从它推；
* **glibc-2.31 的机器要用 MetaX 自己的 triton，不能用 FlagTree 的**；
  安装源是 `flagos-pypi-metax`；
* **SpMM COO 复数在 `BLOCK_NNZ=256` 下整类失败**：rowrun 内核展开
  `tl.static_range(0, BLOCK_NNZ)`，BLOCK_NNZ 直接放大每线程私有内存，而 C550 驱动把它
  卡在 4 KB/线程（只能主机侧 `insmod metax.ko pri_mem_sz=` 调），复数又是实部+虚部两倍
  footprint。256 时要 8 KB，launch 被直接拒绝，报
  `memory size or pointer value too large to fit in 32 bit`。**4 可以，而且 4 本来就是
  30 矩阵扫出来的最优值**，所以这个上限不花钱。C API 这一侧同样硬编码 4。

## 注意

Python 层的测试和运行方法见隔壁 checkout 的 `docs/MACA.md`。本文只讲 C API 这一层。
