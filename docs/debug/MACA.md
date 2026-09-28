# 沐曦 MACA（C550）debug

`FLAGSPARSE_BACKEND=metax`（设置了 `MACA_PATH` 时能自动识别）。环境、交付复现见 [../MACA.md](../MACA.md)。

## 1. 这个后端的特点

- 以 CUDA 兼容栈的方式接入（`torch.version.cuda` 显示 11.6），Triton kernel 与 CUDA 相同。
- 实测指纹：设备名 `MetaX C550`，**warp 大小 64**，104 个 MP，每 MP 2048 线程。
- **单线程私有内存上限 4 KB**：编译期展开的循环（`tl.static_range`）越长越占私有内存，超了会报
  `memory size or pointer value too large to fit in 32 bit`，真正原因在它上一行的 MACA 运行时提示里。
- 要用沐曦自己的 triton（`3.6.0+metax3.8.1.0`），FlagOS 的 flagtree 在这台 glibc 2.31 的机器上加载不了；
  代价是没有 `triton.experimental.tle`，`alpha_spmm_alg1` 不可用。
- SDK 自带 **mcSPARSE**，目前没接成基线（可以接）。

## 2. 已知问题

| 问题 | 状态 |
|---|---|
| SpSV `_spsv_csr_cw_kernel`：下三角即使没有依赖也非法访存；上三角有依赖时卡死 | 未解决；只影响单位对角走 cw 路由的情况 |
| 复数 SpMM COO 因 `BLOCK_NNZ=256` 展开超出 4 KB 私有内存 | 已修：MACA + 复数时 `BLOCK_NNZ` 限为 4 |
| spgemm_csr 在某个容器上 5 次里失败 3 次，换容器后 800+ 次全过 | 视为容器 / 硬件状态问题，复现时记录容器 id |
| 卡上的代码（`/root/gcx/FlagSparse`）和本地已经不同步 | 看报错前先看机器上那份源码 |

## 3. q4 42 个变体的风险点

- 新代码的循环都是运行时 `range`，没有依赖私有内存的长展开；复数路径最该验证：
  `spmv_csr/coo/csc_c32`、`spmm_csr/coo_c32`、`spvv_c32_int_conj`、`spmv_sell_c32`、`spmv_csr_f32c32`。
- `spmv_csr` 的混合精度 kernel 每个程序处理 `ROWS x BLOCK` 个元素（共 512 个），按 4 个 warp 调；warp 大小是 64 时每个 warp 分到的元素数不同，需要实测性能。
- 无 cuSPARSE 基线：用 `--vendor-card maca-c550`（1440 GB/s）判定；接上 mcSPARSE 后可以直接比。

## 4. 待确认（上机后回报）

1. `tests/pytest/test_q4_variants_accuracy.py` 全部结果，重点是复数变体。
2. 出现私有内存报错时，把 MACA 运行时打印的那一行（kernel 请求 / 系统上限）原样带回。
3. runner 一轮完整结果目录。
