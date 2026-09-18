# DCU (海光)

`BACKEND=HCU` / `FLAGSPARSE_BACKEND=rocm`。**预留槽位**：抽象和工具链定位都就绪，
缺 `src/adaptor/backend/hcu/adaptor.cpp`。

## 现状

```
deps/libtriton_jit   ✓ 有 BackendHCU.cmake（走 find_package(hip)）
FlagSparse adaptor   ✗ 未写
厂商基线             ✓ 已写未编译：ctest/baseline/rocm/baseline.cpp
Python 层            ✓ 可用，_backend_name() 返回 'rocm'
```

配置 `-DBACKEND=HCU` 会停下并说明缺什么。**这不是 TODO 占位**：SDK 位置是从
`deps/libtriton_jit/cmake/BackendHCU.cmake` 读出来的（两边找的是同一套 SDK，
所以不会漂），没有编造任何库名。

## 环境检查
> 这一层离不开可导入的 `flagsparse` 包：内核是 **re-export** 而不是拷贝，所以
> `FLAGSPARSE_PYTHON_SRC` 指错、或机器上有旧副本，症状分别是 `EXECUTION_FAILED`
> 和 `CompilationError`。依赖关系、版本配套和部署清单见
> [README.md#这个库离不开-flagsparse](README.md)。


```bash
rocm-smi                                  # 或 hy-smi
echo $ROCM_PATH $HIP_PATH                 # DTK 常见位置 /opt/dtk/hip、/opt/rocm/hip
python3 -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
export FLAGSPARSE_BACKEND=rocm
```

**`torch.version.hip` 非空是判定依据**，这也是库里最便宜可靠的探测手段（不用碰设备）。
注意 DCU 上 PyTorch 仍然用 `torch.cuda` 命名空间。

## 填这个槽位要做什么

HCU 走 HIP，`find_package(hip)`，**不套 `_template.inc` 的两库形状**，所以它的
adaptor 需要按 HIP runtime API 写，而不是照抄 musa 那份符号前缀。要实现的接口是
`src/adaptor/adaptor.hpp` 里那十来个函数：设备发现、`multiprocessor_count` /
`max_threads_per_multiprocessor` / `warp_size`、以及 malloc/free/memcpy/memset/同步。

`warp_size` 必须来自设备而不是写死 32 —— DCU 是 64，而 SpMV COO 和 BSR 的
`num_warps` 就是从它推出来的。

## 基线：hipSPARSE，已写好，等一次真机编译

`ctest/baseline/rocm/baseline.cpp` 是 35 行前缀表加共享的
`ctest/baseline/_template.inc`——hipSPARSE 的 generic API 是 cuSPARSE 的直接克隆，
调用名、描述符模型一一对应，只换 `HIPSPARSE_` / `HIP_` 枚举前缀。

**没有在任何机器上编译过**，因为开发机没有 ROCm。两处已知差异已经处理：

* 头在 `<hipsparse/hipsparse.h>`，不在顶层（cuSPARSE 是顶层）；
* SpMM 只支持 `op=non`，见下。

配置时 `ctest/baseline/CMakeLists.txt` 会用 `find_path`/`find_library` 在
`$ROCM_PATH`（默认 `/opt/rocm`）下探测，找不到就回落到 `baseline/none` 并打印原因，
**不会因此构建失败**。首次在 DCU 机器上编译通过，才算把它从"看着对"变成"验证过"。

## 跑测试

```bash
export FLAGSPARSE_BACKEND=rocm

cmake -S . -B build -G Ninja -DBACKEND=HCU \
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

* `speedup` 列**有值**（hipSPARSE 已接），但 **SpMM 的 trans 方向是空的** —— hipSPARSE
  只实现了 `op=non`，那一行会带 `baseline_detail` 说明。这是基线的限制，不是我们的；
* 首次在 DCU 机器上构建，是 `ctest/baseline/rocm/baseline.cpp` **第一次被编译**。它按
  hipSPARSE 公开 API 写成，但从未编译过 —— 编译错误在这里是预期内的，不是回归。

### 跑不起来时按这个顺序查

1. **配置就停下** → adaptor 没写。`-DBACKEND=HCU` 会打印缺什么。HCU 走 HIP，
   不套 `_template.inc` 的符号前缀形状，要按 HIP runtime API 实现
   `src/adaptor/adaptor.hpp` 里那十来个函数。见上面「填这个槽位要做什么」；
2. **基线编译失败** → 查 `ctest/baseline/rocm/baseline.cpp` 顶部注释列的差异项
   （头在 `<hipsparse/hipsparse.h>`，枚举前缀 `HIPSPARSE_`/`HIP_`）；
3. **speedup 全空但构建通过** → hipSPARSE 没找到。配置时会打印
   `ctest baseline: ROCM -> none` 和原因，设 `ROCM_PATH` 指向 SDK；
4. **`warp_size` 相关的结果异常** → DCU 是 64 不是 32，SpMV COO 和 BSR 的 `num_warps`
   由它推出；写死 32 会得到能跑但慢的内核。

## 已知的平台特性

* **hipSPARSE 的 SpMM 只支持 `op=non`**。trans 方向那一行会拿到 hipSPARSE 自己的
  状态码，落成空 `speedup` 加一条 `baseline_detail` 说明——**这是基线的事实，不是我们
  的缺口**，所以它是一行记录而不是一个失败；
* 机器上装过的旧 `dist-packages/flagsparse` 会**伪装成 N/A 基线** —— 看起来像"没测"，
  实际是测了另一个副本。`FLAGSPARSE_PYTHON_SRC` 现在会无条件获胜并验证解析结果，
  但环境里那份仍可能干扰 Python 层的直接调用；
* Python 层的测试方法见隔壁 checkout 的 `docs/DCU.md`。
