# FP8 Recurrent-State Cache + ReplaySSM Kernel for Mamba-hybrid LLM decode

面向 **Qwen3.5-4B（Mamba-hybrid，GDN 层）decode 的 FP8 recurrent-state 压缩与 replay 融合内核**，
在单张 **RTX 4090** 上完成从数值参考实现、真实模型注入式验证、长序列误差寿命分析，到
Triton kernel 实现与性能验证的完整闭环。

动机来自 [vLLM issue #55196](https://github.com/vllm-project/vllm/issues/55196)：
hybrid 模型的 mamba page 与 attention page 绑定，使 recurrent state 的显存与带宽成为瓶颈。

## 结论摘要

**做对了什么**

| 结论 | 证据 |
| --- | --- |
| full-precision replay 与逐 token 递推**完全等价** | 输出与末状态零误差 |
| FP8 E4M3 + vblock(32 行 × 全 K) checkpoint 可用 | 单次状态重建误差 1.7–2.6% |
| 融合 kernel 相对 **vLLM 生产算子**加速 | core **L=16 1.99× / L=4 2.21×**；full-contract **1.68× / 1.81×**（batch 64） |
| 状态搬运字节相对 BF16 读-改-写 | 2.76× 更少（L=16） |
| 正确性（对 fp32 参考解） | 输出相对 L2 1.65e-3–1.70e-3 = bf16 舍入底噪 |

**做不到了什么（同样重要）**

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

**full-contract**（两边都做 q/k 归一化 + 门控 + ring 追加，即真正的算子对算子口径）：

| batch | L=4 | L=8 | L=16 |
| --- | --- | --- | --- |
| 16 | 1.30 | 1.29 | 1.19 |
| 32 | 1.55 | 1.55 | 1.44 |
| 64 | **1.81** | **1.79** | **1.68** |

用真实 capture 驱动同一 harness 复核（batch 64）：1.772 / 1.806 / 1.690，与合成输入一致。

break-even 约在 batch 4；batch ≤2 时 replay 落后（0.75–0.91×），因为该规模下两条路径都是
launch/延迟受限。生产算子本身达到 ~815 GB/s（4090 峰值的 81%），所以这是算法（搬运量）收益。

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

环境：RTX 4090（SM89）、CUDA 13.2、Torch 2.13、Triton 3.7.1、
Qwen3.5-4B（HF `Qwen/Qwen3.5-4B`）、vLLM 源码 worktree + `.venv`。

```bash
# 1) kernel 正确性 + 对生产算子的性能对比（不需要加载模型）
python benchmarks/kernels/bench_gdn_vs_production.py --batches 1,2,4,8,16,32,64 --window 4,8,16

# 2) tiling / 窗口 / 分块扫描
python benchmarks/kernels/bench_gdn_replayssm.py --batches 1,4,16,64 --window 4,8,16,32 --block-v 32,64,128

# 3) 诊断（寄存器/占用率/纯流量上界）
python benchmarks/kernels/bench_kernel_diag.py --window 16 --batch 64 --block-v 32,64,128

# 4) 长序列漂移研究（先录真实 decode 输入，再离线跑精确链与量化链）
VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/qwen35_drift_study.py \
    --tokens 4096 --layers 0,8,16,24 --windows 4,16,64 --save-capture capture.pt
python benchmarks/qwen35_drift_study.py --load-capture capture.pt --windows 4,16,64

# 5) 模型级注入式 A/B（把 replay 输出与重建状态写回真实 decode 路径）
VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/qwen35_replay_ab.py --max-tokens 256 --windows 4,8,16
```

日志分析：`python tools/analyze_kernel_sweep.py results/<log>`、
`tools/analyze_vs_production.py`、`tools/analyze_drift_study.py`、`tools/analyze_replay_ab.py`。

## 方法学要点

1. **不把量化误差和 replay 算法混在一起**：先证明 full-precision replay 等价，再引入 FP8。
2. **控制臂**：所有 A/B 都带一个"checkpoint 完全精确"的对照，确保结论能归因。
3. **tie 归因**：发现 bf16 logits 让 top-2 margin 量化到 0.125、~1.5% 的步成为精确 tie，
   所有 token 翻转都发生在这些步上且与量化无关；据此更换验收口径。
4. **稳态口径**：ring 在窗口内是 1..L 递增，按整周期而不是"每步都是满 ring"摊销。
5. **生产基线**：验收对象是现有生产实现，不是自研的同风格 kernel。
6. **保留失败证据**：带 bug 的日志（宽 tile scale 误用、harness 计入分配开销）一并归档。

## 已知局限

- **未集成 vLLM**：所有性能数字是单层、单次 decode step 的 kernel 级结果；端到端 tokens/s 未测。
- replay 计时**不含**每 token 预处理（q/k L2 归一化、门控计算、ring 追加），生产算子是内联做这些的。
- kernel 基准输入为合成数据（真实几何 + 真实衰减分布）；真实 capture 已录制但尚未接入该基准。
- **L=16 的数值闸门未过**：只知道每层状态漂移，尚未换算成模型级 logprob 损失（需要 teacher forcing）。

## 下一步

1. **teacher-forced 长上下文评估**：强制两臂走同一 token 轨迹并记录未修改 logits 的 top-k，
   把"每层漂移"换算成模型级误差 —— 这是唯一能改变结论的实验。
2. 用真实 capture 驱动 kernel 基准，消除"合成数据"质疑。
3. 若 L=16 不达标：误差补偿 flush / 两级 checkpoint / 按层选格式（INT8 在部分层明显更优）。
