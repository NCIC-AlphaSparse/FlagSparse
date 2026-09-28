# 天数智芯 BI-V150 debug

`FLAGSPARSE_BACKEND=iluvatar`（通常能自动识别）。环境、交付复现见 [../ILUVATAR.md](../ILUVATAR.md)，
上机排查记录见 [../ILUVATAR_DEBUG.md](../ILUVATAR_DEBUG.md)。

## 1. 这个后端的特点

- 以 CUDA 兼容栈接入（CoreX 4.4.0，`torch.version.cuda` 显示 10.2）。
- 实测指纹：设备名 `Iluvatar BI-V150 OAM`，**warp 大小 64**，16 个 MP，
  Triton 目标 `GPUTarget(backend='corex', arch=71, warp_size=64)`。
- 因为目标不是 `cuda`/`hip`，`spmv_csr` 只能用三条 legacy 路由；新路由会直接报不支持。
- 厂商基线是 CoreX 的**旧版** `cusparseScsrmv` / `cusparseScsrmm`（通过 CuPy 调用），
  只支持 fp32 + int32 + 不转置；通用 `cusparseSpMV` 在这张卡上会申请约 140 TB 显存，不可用。

## 2. 已知问题

| 问题 | 状态 |
|---|---|
| fp64 的 H2D 拷贝静默返回全 0（fp64 各类报错的共同根因） | 已定位；q4 不含 f64 / c64，理论上不受影响 |
| `torch.sparse` 在这张卡上结果是错的 | 不要用它当参考；精度参考值在 CPU 上算 |
| 容器里 torch 没注册到解释器，子进程会丢 torch | 用 `corex.pth` 指向厂商 dist-packages；`python3 -I -c "import torch"` 验证 |
| CoreX 版本和宿主驱动必须一致（4.5.0 镜像配 4.4.0 驱动会让 `cudaMalloc` 返回 801） | 以 `ixsmi` 能显示 CUDA 版本为准 |

## 3. q4 42 个变体的风险点

- **int8 是最大的未知数**：这张卡的 Triton 对 int8 读、int32 累加、int32 原子加支持到什么程度没有测过。
  涉及 `gather/scatter_i8`、`spmv_csr_i8i32`、`spmv_csr_i8f32`、`spmv_coo_i8i32`、`spmv_sell_i8i32`、
  `spmm_csr_i8i32`、`spvv_i8i32` 共 7 个。
- **f16**：`spmv_csr` 的 f16 在这张卡上实测通过过；其余 f16 变体待测。
- **复数**：c32 各变体没在这张卡上验证过。
- **基线**：CoreX 旧接口只覆盖 fp32 + 不转置，所以 q4 的绝大多数变体在这里没有厂商基线，
  用 `tools/baseline_bound.py --vendor-card iluvatar-biv150`（1150 GB/s）判定。
- 首轮建议先固定 `FLAGSPARSE_ILUVATAR_VENDOR=torch`，一次只变一个变量。

## 4. 待确认（上机后回报）

1. `tests/pytest/test_q4_variants_accuracy.py` 全部结果，重点是 7 个 int8 变体和复数变体。
2. int8 变体若失败：是 Triton 编译不过，还是结果不对（加 `CUDA_LAUNCH_BLOCKING=1` 重跑）。
3. runner 一轮完整结果目录。
