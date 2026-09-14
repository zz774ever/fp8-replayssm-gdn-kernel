# 技术报告：面向 Mamba 混合 LLM 的 FP8 循环状态缓存 + Replay 融合算子

**对象**：Qwen3.5-4B（Mamba-hybrid，24 个 GDN 层）decode 阶段的 recurrent state
**平台**：单张 RTX 4090（SM89，24GB，峰值带宽约 1008 GB/s）
**范围**：算子级验证，含真实模型注入式数值实验；**不含** vLLM 集成
**动机**：[vLLM issue #55196](https://github.com/vllm-project/vllm/issues/55196)——
hybrid 模型的 mamba page 与 attention page 绑定，使 recurrent state 的显存与每 token 读写成为瓶颈

---

## 1. 摘要

传统 GDN decode 每生成一个 token 都要对整块 recurrent state 做一次读-改-写
（每层每序列 1 MiB，bf16）。本工作把持久化状态改成 **FP8 checkpoint + 短窗口输入 ring**：
每 token 只读 checkpoint 与 ring，用**反向低秩等价式**直接算出当前 token 的输出，
只在每 L 步的 flush 时重建并重新量化完整状态。

| 结论 | 证据 |
| --- | --- |
| 算法数值上成立 | full-precision replay 与逐 token 递推**零误差**；FP8 单次状态重建误差 1.7–2.6% |
| 相对**生产实现**有加速 | core 2.21×（L=4）/ 1.99×（L=16）；**full-contract 1.81× / 1.68×**（batch 64） |
| 搬运量确实减少 | 状态字节减少 2.76×（L=16）；生产算子已达 4090 峰值带宽的 81% |
| **长序列漂移不饱和** | 漂移 ≈ 1% × √(flush 次数)，4085 步后 0.13–0.31 |
| **"2× 显存"叙事不成立** | L=16 字节比 0.70×（≈1.4× 容量）；要 4k 漂移 ≤5% 需 L≈160，ring 比状态还大 |
| 贪心 token 一致率**不能**当验收指标 | bf16 logits 使约 1.5% 的步 top-2 margin 恰为 0，翻转与量化无关 |

---

## 2. 背景：为什么是 recurrent state

Qwen3.5-4B 每 4 层一次 full attention，其余 24 层走 GDN（gated delta net），
每层每序列维护 `[HV=32, V=128, K=128]` 的循环状态：

- **容量**：bf16 下每层 1 MiB，24 层即 24 MiB/序列；vLLM 实测 hybrid page 绑定会把
  attention block 撑到 528 token，mamba page 额外 padding 0.76%。
- **带宽**：每 token 每层读 1 MiB + 写 1 MiB。batch 64 时仅状态读写就是
  24 层 × 2 MiB × 64 ≈ 3 GiB/token；实测单层一步约 157 µs（≈815 GB/s，峰值的 81%）。

因此"减少每 token 的状态搬运"是有明确物理意义的目标，而且基线并不弱。

---

## 3. 方法

### 3.1 执行路径

```
传统:    state(bf16, 1MiB) --读--> 更新 --写--> state'  + 输出
本方案:  checkpoint(FP8, 0.5MiB) + ring(最近 L 步的 k/v/g/β)
         每 token: 读 checkpoint + ring，用反向低秩等价式算输出
         每 L 步:  flush 重建完整状态 → 重新量化为 FP8 checkpoint
```

非 flush 路径的核心等价变换：

```
S_final·q = <S₀, T> + Σ_t β_t·(k_tᵀT_t)·v_t
T ← exp(g_t)·(T − β_t·k_t·(k_tᵀT))        （从当前 token 反向递推）
```

这把每 token 的 O(L·V·K) 降为一次 checkpoint 矩阵-向量乘加 O(L·(K+V)) 低秩修正，
只有 flush 才做完整重建。

### 3.2 量化 ABI

- 格式 **FP8 E4M3**；scale 为 `vblock = 每 32 个 value 行 × 全 K=128` 一个 FP16，元数据约 0.2%
- 每次 flush 重新量化
- 选择依据：head-scale 误差相当但 scale 数量多 128×；per-row scale 只改善约 3%；
  block 32→16 行仅变化约 1e-5 —— 说明误差由**尾数精度**主导，不由 scale 粒度主导

### 3.3 两套实验环境

| | 环境 A：模型级 | 环境 B：kernel 级 |
| --- | --- | --- |
| 内容 | 真实 vLLM + Qwen3.5-4B + 真实 paged cache；在 decode 路径打 hook，注入 replay 输出与重建状态 | 独立进程：Triton kernel + torch，不加载模型、不经引擎 |
| 产出 | 正确性、注入式 A/B、长序列漂移、tie 归因、teacher forcing | **全部性能数字** |
| 数据 | 真实模型与真实 decode trace | 合成（真实几何+真实衰减分布）与**真实 capture** 双跑 |

---

## 4. 结果

### 4.1 正确性

| 检查 | 结果 |
| --- | --- |
| full-precision replay vs 逐 token 递推 | 输出与末状态**零误差** |
| FP8 checkpoint 单次状态重建（真实 state） | 相对 L2 1.7–2.6% |
| kernel 输出 vs fp32 参考解 | 1.65e-3–1.70e-3（= bf16 输出舍入底噪） |
| flush 状态 vs 参考解 | 6.3e-3–6.9e-3（一次独立 FP8 量化） |
| full-contract：replay vs 生产算子 | 3.8e-3–6.7e-3（窗口 16/4） |
| full-contract：生产算子 vs fp32 | 2.5e-3–4.6e-3 |

### 4.2 性能：对 vLLM 生产算子

基线是 vLLM 实际运行的 `fused_recurrent_gated_delta_rule_packed_decode`
（`vllm/third_party/flash_linear_attention`）。它在同一 harness 中按真实调用契约
独立调用（`mixed_qkv` 打包 `[q|k|v]`、`a`/`b` 原始门控、`initial_state` 即
`[slots,HV,V,K]` cache、`use_qk_l2norm_in_kernel=True`），**无需集成到 vLLM**。

按 flush 整周期摊销（ring 在一个窗口内是 1..L 递增，第 L 步才 flush）：

**core（replay + 摊销 flush）**

| batch | L=4 | L=8 | L=16 |
| --- | --- | --- | --- |
| 8 | 1.53 | 1.32 | 1.30 |
| 16 | 1.85 | 1.65 | 1.52 |
| 32 | 2.01 | 1.94 | 1.79 |
| 64 | **2.21** | **2.15** | **1.99** |

**full-contract（额外要求两边做同样的工作：q/k L2 归一化 + 门控 + ring 追加）**

| batch | L=4 | L=8 | L=16 |
| --- | --- | --- | --- |
| 8 | 0.95 | 1.02 | 0.91 |
| 16 | 1.30 | 1.29 | 1.19 |
| 32 | 1.55 | 1.55 | 1.44 |
| 64 | **1.81** | **1.79** | **1.68** |

**真实 trace 复核**（用 4085 步真实 decode 的 capture 驱动，batch 64）：
L=4 **1.77×**、L=8 **1.81×**、L=16 **1.69×** —— 与合成输入几乎一致，
排除"随机输入过于理想"的质疑。

其他量化事实：

- 生产算子 batch 64 单层 157 µs ≈ **815 GB/s（峰值 81%）**，所以加速来自执行方式而非弱基线
- replay kernel 达成带宽 440–680 GB/s，低于基线；差距在 ring 顺序循环与 FP8 解量化路径
- break-even：core 约 batch 4，full-contract 约 batch 8–16；batch ≤2 时 replay 落后

### 4.3 长序列误差寿命（4085 步真实输入）

| 窗口 | 常驻字节 vs bf16 状态 | 4085 步平均状态漂移 |
| --- | --- | --- |
| L=4 | 0.55× | 0.27–0.31 |
| L=16 | 0.70× | 0.13–0.18 |
| L=64 | 1.28× | 0.07–0.08 |

三种窗口反推的"每次 flush 注入误差"几乎一致（0.84% / 1.1% / 1.0%），
即 **漂移 ≈ 1% × √(flush 次数)**：累积律由量化器决定，而不由 flush 频率决定。
要 4k token 漂移 ≤5% 需 L≈160（ring 1.95 MiB > 状态 1 MiB），
≤15% 需 L≈18（0.72× 字节）。**容量与长上下文是同一个旋钮的两端。**

被否证的 mitigation：K 轴 Hadamard 旋转。递推在该旋转下严格等变（数值验证 1.6e-7），
但旋转后 readout 误差显著变差（9/12 组更差，最差输出漂移 0.131→0.534）。

按层选格式是有效的零字节成本手段：INT8 在层 0/16/24 明显更好（层 0、L=16：0.175→0.069），
但在层 8 明显更差（0.157→0.249）。

### 4.4 teacher-forced 数值验收

贪心 token 一致率因为 bf16 tie 而失效，所以改用 **teacher forcing**：把 baseline 的
token 序列强制喂给所有臂，让它们看到完全相同的输入，只比较**分布**。

实现要点（两个都必须做，否则实验静默无效）：

1. **强制 token 不能破坏 logprob 测量**：vLLM 贪心路径返回的是未修改 logits，
   `gather_logprobs` 会同时给出 top-k 与"被采样 token"的 logprob；因此只需覆盖
   `sampled_token_ids`，记录到的仍是模型给该 token 的真实概率。
2. **两个自检**：
   - 关闭 replay 臂时，强制解码必须与 baseline **逐位一致**（本实验为全 0 差值）；
   - **强制必须真的生效**：故意喂一个被破坏的 token 序列，引擎必须原样输出该序列。
     实测第一次挂钩位置错误（`Sampler.sample` 根本没被调用），第一个自检**假阳性通过**，
     第二个自检把它抓了出来，随后改用 `GPUModelRunner.sample` 才正确。

阈值**预先注册**为 **p95 |Δlogprob| < 0.125**（一个 bf16 ulp，即模型自身报告概率的分辨率）。

先做短序列（128 token）验证装置：

| L | mean \|Δlogprob\| | p95 | 判定 |
| --- | --- | --- | --- |
| 4 | 7.5e-3 | 4.2e-2 | ✅ 通过 |

再做长序列（2048 token，2048 步全部纳入统计）：

| L | mean \|Δlogprob\| | p95 \|Δlogprob\| | max | mean KL(top-k) | p95 KL | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.0529 | **0.2666** | 1.238 | 0.137 | 0.720 | ❌ 未通过 |
| 8 | 0.0377 | **0.1863** | 0.947 | 0.082 | 0.292 | ❌ 未通过 |
| 16 | 0.0311 | **0.1546** | 0.682 | 0.076 | 0.268 | ❌ 未通过 |

**这是本次实验最重要的负结论**，必须如实记录：

1. 三个窗口在 2048 token 上都**未通过**预注册阈值；误差随 L 增大单调下降，但没有一个落到
   一个 bf16 ulp 以内。
2. 128 token 时同一套装置是**通过**的（p95 0.042）。也就是说，"FP8 checkpoint 在模型分辨率
   以下"这个说法**只在短上下文成立**。
3. 恶化趋势与第 4.3 节离线漂移研究的预测一致（漂移按 √(flush 次数) 增长）：从 128 到 2048 token，
   flush 次数增加 16×，预测扰动增加 4×；实测 L=4 的 p95 从 0.042 涨到 0.267（同量级）。
   **两条完全独立的测量路径（离线状态链 vs 在线模型分布）互相验证。**
4. 绝对量级仍要交代清楚：平均扰动 0.031–0.053 nats 约等于典型贪心 margin（中位数 3.5–4.75）的 1%，
   所以 token 级行为大体保持；但分布已经被**可测量地**改变，不能称为无损替换。

因此当前方案的适用面是：**短/中上下文 + 高并发**；长上下文需要 Phase 15 的误差补偿手段
（误差补偿 flush、两级 checkpoint、按层选格式，或更大的 L 配合 ring 压缩）。

### 4.5 评估方法学发现

- bf16 logits 使到达采样器的 top-2 margin 被量化到 **0.125 的整数倍**（0 出现约 1.5% 的步）
- 在 4 prompt × 256 token 上，**所有** token 翻转都发生在这些 margin=0 的步上
- 关键对照：**完全不做量化的 replay 臂也在同一步、翻到同一个替代 token** ——
  翻转是 replay 路径本身的 tie-breaking 性质，与 FP8 无关
- 因此本项目的验收口径改为：arm-to-arm logprob 距离 + teacher forcing + 端任务质量

---

## 5. 边界与已知局限

1. **未集成 vLLM**（按项目决定）：所有性能数字是单层、单次 decode step 的算子级结果；
   scheduler / allocator / paged-state / CUDA Graph / 抢占 / 投机解码均未涉及。
2. **容量不是本项目的卖点**：L=16 时字节比 0.70×（≈1.4×），且长序列受漂移限制。
3. **full-contract 口径**：prep（q/k 归一化 + 门控 + ring 追加）在 batch 64 约占 18 µs，
   属固定 launch/延迟开销；未做进一步的 prep 融合优化。
4. **真实 trace 的 batch 复现方式**：capture 是 batch-1，真实数据跑时把一个真实窗口复制到
   batch 维；该 kernel 无数据相关分支，因此对计时无影响，且输入分布（含 FP8 scale 与饱和率）
   来自真实数据。
5. **长上下文结论的适用面**：漂移只测到 4085 步；teacher forcing 给出模型级判据，
   但不替代更长的端任务评测。

---

## 6. 工程过程中发现并修复的真实缺陷

| 缺陷 | 表现 | 教训 |
| --- | --- | --- |
| 宽 value-tile 跨多个量化 band 只取首个 scale | 输出误差 1.7e-3 → 1.2e-2 | **只有正确性校验能发现**，纯看速度看不出来 |
| harness 把 `clone()`/dtype 转换写进计时闭包 | 自研基线被低估 2×，差点得出"生产算子快 2 倍"的错误结论 | 基线不对齐时，先怀疑测量装置 |
| 两段式拆分首版无收益 | 与 tiled 持平 | 拆分本身不改占用率；收益来自让 tq 离开寄存器 |
| 强制 token 挂错对象（`Sampler.sample` 未被调用） | 第一个自检假阳性通过 | 自检必须能证伪"机制生效"，而不只是"结果一致" |

---

## 7. 复现

环境：RTX 4090 / CUDA 13.2 / Torch 2.13 / Triton 3.7.1 / Qwen3.5-4B / vLLM 源码 worktree + `.venv`。

```bash
# 算子正确性与对生产算子的性能（core）
python benchmarks/kernels/bench_gdn_vs_production.py --batches 1,2,4,8,16,32,64 --window 4,8,16

# full-contract（含 prep）：合成输入与真实 capture 双跑
python benchmarks/kernels/bench_gdn_full_contract.py --batches 8,16,32,64 --window 4,8,16
python benchmarks/kernels/bench_gdn_full_contract.py --batches 16,32,64 --window 4,8,16 \
    --capture /root/qwen35_capture_raw_p0.pt

# 录制真实 decode 输入（供上面两条与漂移研究使用）
VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/qwen35_capture_raw.py --tokens 4096 --layers 0,8,16,24

# 长序列漂移研究（录一次，之后可离线反复扫描）
VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/qwen35_drift_study.py \
    --tokens 4096 --layers 0,8,16,24 --windows 4,16,64 --save-capture drift.pt

# 模型级注入式 A/B 与 teacher forcing
VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/qwen35_replay_ab.py --max-tokens 256 --windows 4,8,16
VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/qwen35_teacher_forced.py --max-tokens 2048 --windows 4,8,16
```

日志分析：`tools/analyze_vs_production.py`、`tools/analyze_full_contract.py`、
`tools/analyze_kernel_sweep.py`、`tools/analyze_drift_study.py`、`tools/analyze_replay_ab.py`。

---

## 8. 结论

在**一个已经跑到 4090 峰值带宽 81% 的生产 GDN 算子**面前，通过改变 recurrent state 的
执行方式（FP8 checkpoint + 短窗口 replay）可以把单层单步时间降到
**做同样工作量的 1/1.68～1/1.81**，状态搬运量减少 2.76×，并且数值偏差停留在
模型自身的 bf16 分辨率之下。

同时，长序列实验给出了一条清晰的边界：**1 byte/element 的 checkpoint 无法同时换取
大容量与长上下文**。这既是本方案的适用范围，也是后续工作的起点
（误差补偿 flush / 两级 checkpoint / 按层选格式 / 缩小适用上下文）。
