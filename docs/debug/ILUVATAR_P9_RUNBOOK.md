# 天数 BI-V150：P9 探针执行说明（2026-10-26）

> 目标：把 SpMV 里低于 0.8 的变体，**按 10 个矩阵的算术平均**提到 1.0 以上（不要求每个矩阵都过 1）。
> 这一轮只跑 P9 探针、带回数据，**不改生产代码**。SpGEMM 这轮不管。
> 背景和此前的实测见 [`LOW_SPEEDUP.md`](LOW_SPEEDUP.md)、[`ILUVATAR_SELL_COO_PLAN.md`](ILUVATAR_SELL_COO_PLAN.md) 第 9 节。

## 1. 为什么跑 P9

| 变体（10 矩阵算术平均，相对 H800 折算基线） | 现在 | 要到 1.0 需整体提速 |
|---|---:|---:|
| SELL f32 / f16 / f16f32 / i8i32 | 约 0.70 | 1.43x |
| SELL c32 / COO c32 / CSR c32 | 0.52 / 0.54 / 0.65 | 1.5–1.9x |
| COO f16f32 / COO f32 trans | 0.71 / 0.77 | 1.3–1.4x |
| COO f16 / COO i8i32 / CSR f32c32 | 0.81–0.83 | 1.2–1.25x |

这么大的差距，扫 tile / σ / warps 已经补不上（9.5–9.12 节都扫过，没有稳定赢家）。10-25 探针给出的线索是：

- **P1**：Triton 在这张卡上，一个 program 读 512 个连续元素就能到 **655 GB/s**，2048 个时 678 GB/s。
- 现有 SpMV kernel 只跑到 **260–330 GB/s**，大约是 Triton 自己上限的一半。
- SELL kernel 每一步只读 `C × GROUP = 32` 个元素（64 宽 warp 有一半 lane 闲着），而且 x 要等列号读回来才能读。
- 去掉 x 随机读（P3）、去掉列号到值的依赖（P4）、换 tile（P5），收益只有 `1.00–1.11x`——说明瓶颈不在 gather 本身。

**假设**：每个 program 同时在途的访存太少。P9 用来验证这个假设：

- **P9a**：同样的连续读，换成 SpMV kernel 实际的访存形状来读。一种是一个 program 只读一块（BLOCK = 32 … 2048），一种是一个 program 循环读很多个小块（就是 SELL 走一个 slice 的方式）。另外测 `.cg`，以及 `torch.sum` / `copy_`，看厂商自己的 kernel 能跑多快。
- **P9b**：SELL-C-σ 每一步读 `K` 列，即一个 `[K, C]` 的块。slice 是按列存的，所以这一块在内存里是**连续的**，在途访存从 C 个变成 K × C 个。`K=1` 就是现在的 kernel。实数和复数都测，每一行都对 fp64 参考做正确性检查（检查前先把输出填成 NaN / 整型最小值，kernel 漏写的行会判 `FAIL`）。

## 2. 跑之前确认

1. 代码是 `main` 上包含 `tools/biv_spmv_probe.py` P9 和 `tools/biv_p9_project.py` 的版本。记下 `git rev-parse HEAD`。
2. `tests/data` 下 10 个交付矩阵齐全（字节数对照 [`ILUVATAR.md`](ILUVATAR.md)）。
3. 卡空闲：`ixsmi` 里没有其他进程（时间都是同卡交错测的，后台负载会让结果失真）。
4. 下面两个结果目录还在仓库根目录下（10-24 / 10-25 在 BI-V 上跑出来的）。投影脚本要用它们里的基线：
   - `results_biv_sell_auto_final_20261025/`
   - `results_biv_65_20261024_final/`

## 3. 命令

先做一次冒烟（1 个矩阵、1 轮，几分钟），确认没有报错、没有 `FAIL`：

```bash
env FLAGSPARSE_BACKEND=iluvatar python3 -u tools/biv_spmv_probe.py tests/data \
  --matrices cage12 --skip P2,P3,P4,P5,P6,P7,P8 \
  --dtypes float32,complex64 --rounds 1 --stream-mb 128 \
  --csv results_biv_p9_smoke.csv
```

再跑全量：10 个矩阵 × 5 种 dtype，P1 + P9a + P9b。预计 10–30 分钟，大头是 JIT 编译和 prepare。

```bash
setsid timeout -s KILL 7200 env FLAGSPARSE_BACKEND=iluvatar \
  python3 -u tools/biv_spmv_probe.py tests/data --matrices '' \
    --skip P2,P3,P4,P5,P6,P7,P8 \
    --dtypes float32,float16,float16:float32,int8,complex64 \
    --rounds 5 \
    --csv results_biv_p9_20261026.csv \
  > results_biv_p9_20261026.log 2>&1 < /dev/null &
```

说明：
- `--skip P2` 只跳过 P2 的裸 kernel；`P2 sell_api`（现在交付走的 SELL 调用）每个矩阵仍会测，作对照。
- P9b 默认 `--wide-sigma 4096`，即实数 SELL 现在交付用的 σ。
- 默认的配置清单是 `--wide 32x1x1,32x2x1,…,64x8x2`（c × K × warps）。不要缩短它，K=1 那几行就是对照组。

跑完再做投影，算出"如果各变体在 prepare 里转成 SELL-C-σ、走 P9b 最快的 kernel，10 矩阵平均能到多少"：

```bash
python3 tools/biv_p9_project.py results_biv_p9_20261026.csv \
  --sell results_biv_sell_auto_final_20261025 \
  --delivery results_biv_65_20261024_final --verbose \
  > results_biv_p9_project_20261026.txt
```

投影的算法：P9b 最快配置的裸 kernel 时间，加上同一份 plan 上测到的 API 开销（`api` 行减 `K=1` 行），作为交付调用的估计耗时；分母用各变体已有的 `h800_scaled_ms`，和交付报告的口径一致。**这是估计值，不是交付数字**，只用来决定值不值得实现。

`CSR f32c32`（f32 值、c32 输出）、`COO f32 trans` 和 conj 变体 P9b 没有对应的 kernel，投影文件里只列现值，标 `not covered by P9b`；它们要另外想办法。

## 4. 带回什么

- `results_biv_p9_20261026.csv`、`results_biv_p9_20261026.log`、`results_biv_p9_project_20261026.txt`
- 冒烟的 `results_biv_p9_smoke.csv`
- `git rev-parse HEAD`，以及 `ixsmi` 的卡型号和驱动版本

## 5. 怎么判读

log 里每个矩阵末尾的 `->` 行已经给了判断，下面是依据。

**P9a**（只在最前面跑一次）：

| 现象 | 含义 | 下一步 |
|---|---|---|
| `loop BLOCK=32 steps=128` 不到宽步长最好结果（`loop BLOCK=2048 steps=16` 和 `flat` 两者取高）的 0.75 | 在这张卡上，每一步读得窄会直接限制带宽 | 支持假设，看 P9b |
| 两者接近（CUDA 上就是这样） | 步宽不是瓶颈 | P9b 多半也没有收益，转去减少字节数 |
| `torch.sum` / `copy_` 比 Triton 最好的结果高 25% 以上 | Triton 生成的代码在这张卡上本身带宽不够 | 记录下来反馈给 FlagTree / CoreX Triton；我们这边只能靠减少字节数 |

**P9b**（每个矩阵、每种 dtype 各一组）：

| 现象 | 含义 | 下一步 |
|---|---|---|
| 大矩阵（NACA0015、auto、amazon0601、roadNet-TX、ASIC_680ks、GL7d14、filter3D）上最好的 K 比 K=1 快 ≥1.15x | 在途访存就是瓶颈 | 实现 K-wide 的 SELL-C-σ kernel，再按投影结果决定 CSR / COO 变体要不要在 prepare 里转成它 |
| 所有矩阵都 ≤1.05x | 假设不成立 | 转向减少字节数（见第 6 节第 2、3 条） |
| 只有小矩阵受益，或介于两者之间 | 不确定 | 只对相应变体做 A/B，加 `--rounds` 复测 |
| 任意一行 `FAIL` | P9b kernel 在这张卡上算错了 | 该行的时间作废，原样带回 log |

投影文件里任何一个变体出现 `>= 1.0`，就说明"转格式 + K-wide kernel"这条路能让它的平均过 1。

## 6. P9 之后的计划（在 CUDA 机上实现，再交回 BI-V 复测）

1. **K-wide SELL-C-σ kernel**（P9b 成立时做）：把 K 作为 `PreparedSellSpmv` 的参数，各后端各自选默认值。
2. **prepare 时按结构选格式**：同一 dtype 用 COO、CSR、SELL 中最快的 kernel。用现有数据推算，各变体平均能提高 0.06–0.15，例如 SELL c32 0.52→0.67，CSR f16f32 0.92→1.00。选择规则按行长统计来定，**不按矩阵名**。
3. **16 位列偏移**（减少字节数，可以和第 1 条叠加）：每 32 行的列跨度小于 65536 时，存"组基址 + 16 位偏移"。覆盖情况：GL7d14 / cfd2 / filter3D 100%，cage12 97%，roadNet 88–92%，wave 79–84%，NACA / amazon / auto 约 0%。每个非零元的字节数：f32 8→6，f16 6→4，i8 5→3，c32 12→10。
4. **主机固定开销**：先在 BI-V 上测一个空 kernel 的发射开销；只对 50 µs 以内的小矩阵有意义。

**不做**：按矩阵名选路由；检测"数值全相同"后跳过读数值（10 个矩阵里有 5 个数值全为 1，这么做是针对测试数据投机）；继续扫 tile / σ / warps。
