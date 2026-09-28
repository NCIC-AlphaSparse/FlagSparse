# 海光 DCU（BW1000，gfx936）debug

`FLAGSPARSE_BACKEND=rocm`（通常能自动识别：`torch.version.hip` 不为空）。环境、交付复现见 [../DCU.md](../DCU.md)。

## 1. 这个后端的特点

- Triton kernel 与 CUDA 相同；差别主要在**对比基线**：用 hipSPARSE（通过 `hip-python`），每个算子由
  `_<op>_sparse_ref_backend()` 选 `hipsparse` / `cupy_cusparse` / `None`，`None` 时在 `*_reason` 列写原因。
- hipSPARSE 的 SpMM 只支持不转置：CSR SpMM 的 `trans` / `conj` 没有基线（报跳过原因，不是失败）。
- gfx936 上 SpMV CSR 走 ROCm 专用路由（`row_tile`，fp32 本地累加，`max_row_nnz <= 8192` 时才用 fp32 累加）。
- warp 大小 64。

## 2. 已知问题

| 问题 | 状态 |
|---|---|
| `import flagsparse` 跑到系统里旧的已安装包，基线列无故变 N/A | CUDA 机已清理；**DCU 机从没检查过**，基线 N/A 时先查（见 [README](README.md) 第 2 节） |
| runner 用 `CUDA_VISIBLE_DEVICES` 隔离设备，ROCm 上会打乱 torch 的随机种子 | 在 DCU 上额外 `export HIP_VISIBLE_DEVICES=<卡号>` |
| spmv_csr 性能约 0.27×（f32 0.276 / f64 0.260，对比 hipSPARSE） | 未解决，和精度修复无关；先查 `row_tile` 路由的启动配置 |
| DCU 侧的代码包是按旧版本改的，整文件覆盖会回退新代码（曾把 iluvatar 改回 mlu） | 合并时按段取，不要整文件复制 |
| DCU 那边 `spmm_csr.py` 里的两个缺陷（`row_ids` 多余分配、短行路由忽略 `accuracy`） | 本仓库已修；DCU 那边每次发来的包都还带着，不要合回来 |
| 2026-09-27：ROCm 上 int64→int32 索引压缩导致精度测试失败 | 已删除；代价是 int64 输入在 ROCm 上用不了 opt 路径 |

## 3. q4 42 个变体的风险点

- 与 CUDA 同一套 Triton 代码，CUDA 上全过。重点确认：
  - **int8**（`spmv_*_i8i32`、`spmv_csr_i8f32`、`spmm_csr_i8i32`、`spvv_i8i32`、`gather/scatter_i8`）：int8 读、int32 累加，
    以及 int32 的 `tl.atomic_add`（`spmv_coo_i8i32` 用到）；
  - **float16**：新代码不做 f16 原子加，但读 f16、在 f32 里算；
  - **SELL SpMV**（`spmv_sell_*`）：全新 kernel，按切片起程序，切片内行数 = `slice_size`（默认 32）。
- **基线**：q4 新接的 cuSPARSE 基线（`tests/cusparse_generic_baseline.py`）只在 CUDA 可用；
  DCU 上这些行的 cuSPARSE 列为空并写明原因，runner 会改用 PyTorch 加速比，再用 `tools/baseline_bound.py --vendor-card dcu-bw1000` 判定。
  如果需要 hipSPARSE 基线，可以按 `cusparse_generic_baseline.py` 的结构用 `hip-python` 补（hipSPARSE 的 SpVV / Axpby / SELL SpMV 接口名一一对应）。

## 4. 待确认（上机后回报）

1. `tests/pytest/test_q4_variants_accuracy.py` 全部结果（尤其 int8、f16、`spmv_sell_*`）。
2. `sudo python3 -m pip show flagsparse` 的输出（排除旧安装包）。
3. runner 一轮完整结果目录。
