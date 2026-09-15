# FP8 Recurrent-State Cache + ReplaySSM Kernel for Mamba-hybrid LLM decode

面向 **Qwen3.5-4B（Mamba-hybrid，GDN 层）decode 的 FP8 recurrent-state 压缩与 replay 融合内核**，
 **RTX 4090** 


动机 [vLLM issue #55196](https://github.com/vllm-project/vllm/issues/55196)：
hybrid 模型的 mamba page 与 attention page 绑定，使 recurrent state 的显存与带宽成为瓶颈。



| 内容 | 文档 |
| --- | --- |
| **这个项目到底做了什么**（背景 / 术语 / 路线 / 实现 / 数据 / 决策 / 边界） | **[REPORT.md](REPORT.md)** ← 主文档，先读这个 |
| 每份实验日志跑的是什么 | [results/README.md](results/README.md) |
| 某个结论的完整证据 | [findings.md](findings.md)（按主题） |
| 当时的决策与踩过的坑 | [progress.md](progress.md)（按时间）、[task_plan.md](task_plan.md) |
| 简历怎么写 / 面试怎么答 | [RESUME.md](RESUME.md) |



## 结论摘要



| 结论 | 证据 |
| --- | --- |
| full-precision replay 与逐 token 递推**完全等价** | 输出与末状态零误差 |
| FP8 E4M3 + vblock(32 行 × 全 K) checkpoint 可用 | 单次状态重建误差 1.7–2.6% |
| 融合 kernel 相对 **vLLM 生产算子**加速 | core **L=16 1.99× / L=4 2.21×**；full-contract **1.61–1.73×**（batch 64） |
| 状态搬运字节相对 BF16 读-改-写 | 2.76× 更少（L=16） |
| 正确性（对 fp32 参考解） | 输出相对 L2 1.65e-3–1.70e-3 = bf16 舍入底噪 |

**缺点**

| 结论 | 证据 |
| --- | --- |
| 长序列漂移**不饱和** | 漂移 ≈ 1% × √(flush 次数)，4085 步状态漂移 0.15–0.31 |
| "2× 显存"叙事**不成立** | L=16 时字节比 0.70×（≈1.4× 容量），要 4k 漂移 ≤5% 需 L≈160，ring 比状态还大 |
| 常用 mitigation **无效** | Hadamard 旋转虽严格等变（1e-7）却使 readout 误差变差（9/12 组） |
| 贪心 token 一致率**不能当验收指标** | bf16 logits 使 ~1.5% 的 decode 步为精确 tie，翻转与量化无关 |
| **长上下文数值验收未通过** | teacher forcing @2048 token：p95 \|Δlogprob\| = 0.155–0.267 > 预注册阈值 0.125；@128 token 通过（0.042） |

## 关键数字

### 对生产算子的加速比（单层、一次 decode step、稳态口径）

基线是 vLLM 实际运行的 `fused_recurrent_gated_delta_rule_packed_decode`（FLA 实现），
按真实调用契约独立调用，**不需要集成到 vLLM**。

| batch | L=4 | L=8 | L=16 |
| --- | --- | --- | --- |
| 4 | 1.18 | 1.11 | 1.01 |
| 16 | 1.85 | 1.65 | 1.52 |
| 64 | **2.21** | **2.15** | **1.99** |

**full-contract**（两边都做 q/k 归一化 + 门控 + ring 追加，即真正的算子对算子口径）。
下表已按"干净的生产算子耗时"重算 —— 原脚本里生产算子那侧含一次闭包内 `contiguous()` 拷贝
（batch 64 约 6.5 µs），会把比值抬高约 4%；修正过程见 [REPORT.md](REPORT.md) §6.2。

| batch | L=4 | L=8 | L=16 |
| --- | --- | --- | --- |
| 16 | 1.14 | 1.11 | 1.04 |
| 32 | 1.42 | 1.43 | 1.33 |
| 64 | **1.73** | **1.72** | **1.61** |

用真实 capture 驱动同一 harness 复核（batch 64，同样修正后）：**1.70 / 1.72 / 1.61**，与合成一致。

break-even：**core 约 batch 4、full-contract 约 batch 16**；batch ≤4 时 replay 落后
（0.43–0.91×），因为该规模下两条路径都是 launch/延迟受限（prep 那一次固定 launch 摊不掉）。
生产算子本身达到 **857–862 GB/s（4090 峰值的约 85%）**，所以这是算法（搬运量）收益，不是弱基线。

### 长序列误差寿命（4085 步真实 decode 输入）

| 窗口 | 常驻字节 vs bf16 状态 | 4085 步平均状态漂移 |
| --- | --- | --- |
| L=4 | 0.55× | 0.27–0.31 |
| L=16 | 0.70× | 0.13–0.18 |
| L=64 | 1.28× | 0.07–0.08 |

## 仓库结构

```
prototype/   全部实验脚本（Triton kernel、fp32 参考实现、模型级 harness、基准与诊断）
tools/       本地分析器与 MCP/SSH 桥接脚本
results/     实验日志（JSON lines）+ 索引 README
findings.md  研究结论与证据（按日期累积）
progress.md  会话式过程记录（含踩坑与修正）
task_plan.md 阶段规划、决策与止损条件
```

`prototype/` 里的文件与远端 GPU 机器上的布局一一对应（`benchmarks/kernels/` 放 kernel 与基准，
`benchmarks/` 放模型级脚本），内容经过 md5 逐字节校验。

## 复现

**环境**：

| 组件 | 版本 / 标识 |
| --- | --- |
| GPU | RTX 4090（SM89，24 GB，峰值带宽约 1008 GB/s），驱动 595.80 |
| 系统 / CUDA | Ubuntu 22.04.5 / CUDA 13.2 |
| PyTorch / Triton / FlashInfer | 2.13.0+cu132 / 3.7.1 / 0.6.18 |
| vLLM | `0.1.dev20944+g58ad1f3b8`，commit `58ad1f3b8973b23943107b51230d594050b42ec3` |
| 模型 | Qwen3.5-4B HF snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

所有脚本在 **`prototype/`**。运行前需要把它们放进一个 vLLM 源码环境
（kernel 侧脚本放 `benchmarks/kernels/`、模型级脚本放 `benchmarks/`），并用该环境的 Python。

```bash
# 1) 算子正确性 + 对生产算子的性能（core，不需要加载模型）
python prototype/bench_gdn_vs_production.py --batches 1,2,4,8,16,32,64 --window 4,8,16

# 2) full-contract（两边都做 q/k 归一化 + 门控 + ring 追加）
python prototype/bench_gdn_full_contract.py --batches 8,16,32,64 --window 4,8,16
python prototype/bench_gdn_full_contract.py --batches 16,32,64 --window 4,8,16 --capture capture_raw.pt

# 3) tiling / 窗口 / 分块扫描与占用率诊断
python prototype/bench_gdn_replayssm.py --batches 1,4,16,64 --window 4,8,16,32 --block-v 32,64,128
python prototype/bench_kernel_diag.py --window 16 --batch 64 --block-v 32,64,128

# 4) 真实 decode 输入录制 + 长序列漂移研究
VLLM_ENABLE_V1_MULTIPROCESSING=0 python prototype/qwen35_capture_raw.py --tokens 4096 --layers 0,8,16,24
VLLM_ENABLE_V1_MULTIPROCESSING=0 python prototype/qwen35_drift_study.py \
    --tokens 4096 --layers 0,8,16,24 --windows 4,16,64 --save-capture drift.pt
python prototype/qwen35_drift_study.py --load-capture drift.pt --windows 4,16,64

# 5) 模型级注入式 A/B 与 teacher forcing 数值验收
VLLM_ENABLE_V1_MULTIPROCESSING=0 python prototype/qwen35_replay_ab.py --max-tokens 256 --windows 4,8,16
VLLM_ENABLE_V1_MULTIPROCESSING=0 python prototype/qwen35_teacher_forced.py --max-tokens 2048 --windows 4,8,16
```

日志分析：`python tools/analyze_vs_production.py results/<log>`、
`tools/analyze_full_contract.py`、`tools/analyze_kernel_sweep.py`、
`tools/analyze_drift_study.py`、`tools/analyze_replay_ab.py`。

**未复测项**：最后一次测量之后我修正了计时口径（给 replay kernel 增加外部 `out` 缓冲、
把 per-step 切片移出计时闭包），代码已提交但**尚未复测**（GPU 实例已关闭）。
因此本文性能数字仍是修正前那一次。复测只需：
`python prototype/bench_gdn_full_contract.py --batches 64 --window 4,8,16`。



1. **不把量化误差和 replay 算法混在一起**：先证明 full-precision replay 等价，再引入 FP8。
2. **控制臂**：所有 A/B 都带一个"checkpoint 完全精确"的对照，确保结论能归因。
3. **tie 归因**：发现 bf16 logits 让 top-2 margin 量化到 0.125、~1.5% 的步成为精确 tie，
   所有 token 翻转都发生在这些步上且与量化无关；据此更换验收口径。
4. **稳态口径**：ring 在窗口内是 1..L 递增，按整周期而不是"每步都是满 ring"摊销。
5. **生产基线**：验收对象是现有生产实现，不是自研的同风格 kernel。
6. **保留失败证据**：带 bug 的日志（宽 tile scale 误用、harness 计入分配开销）一并归档。

## 已知局限

- **未集成 vLLM**：所有性能数字是单层、单次 decode step 的 kernel 级结果；端到端 tokens/s 未测。
- **容量不是本项目的卖点**：L=16 时常驻字节为 bf16 状态的 0.70×（≈1.4× 容量），且长序列受漂移限制。
- **数值适用边界只被钉在两点之间**：128 token 通过预注册误差预算，2048 token 不通过；
  中间的 256/512/1024 **尚未扫描**（因此不使用"短/中上下文"这类未经测量的区间表述）。
- **full-contract 的 prep 是固定开销**：batch 64 约 18 µs 且与线程数无关（launch/延迟受限），
  未做进一步的 prep 融合。
- **真实 trace 的 batch 复现方式**：capture 为 batch-1，真实数据跑时把一个真实窗口复制到 batch 维；
  该 kernel 无数据相关分支，因此对计时不产生影响。
- **最后一次计时口径修正尚未复测**（详见上节"未复测项"）。

## 下一步

1. **误差补偿 checkpoint**：把量化残差以紧凑形式带进 ring（残差链是线性的，理论上可改变
   √(flush 次数) 的累积律），这是唯一可能把长上下文拉回可用区间的方向。
2. **按层自适应格式**：INT8 在部分层明显更优（层 0/16/24），层 8 相反；零字节成本的调优空间。
3. **适用边界扫描**：补 256/512/1024 token，把"128 通过、2048 不通过"之间的边界测出来。
4. **上游集成**（可选，超出当前算子范围）：allocator / paged-state / CUDA Graph。
