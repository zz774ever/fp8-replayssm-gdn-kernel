# 简历表述（定稿）

> 这份文件是给简历/面试用的表述稿，不是技术文档。技术细节见 [REPORT.md](REPORT.md)。

## 项目条目

**FP8 循环状态缓存 + Replay 融合算子（生产基线驱动的 Triton 算子优化研究）**
`Qwen3.5-4B (Mamba-hybrid) · vLLM · Triton · CUDA · RTX 4090` ·
[github.com/zz774ever/fp8-replayssm-gdn-kernel](https://github.com/zz774ever/fp8-replayssm-gdn-kernel)

- 针对 Mamba 混合架构 LLM decode 阶段的循环状态带宽瓶颈（对应 vLLM issue #55196），设计并实现
  **FP8 checkpoint + 短窗口 replay** 执行路径：用反向低秩等价式把每 token 的整块状态读-改-写
  替换为一次 checkpoint 读 + 输入 ring 读，仅在每 L 步 flush 时重建并重新量化状态；Triton 实现，
  含融合的 q/k L2 归一化 + 门控 + ring 追加预处理。
- 以 **vLLM 实际运行的生产 GDN 算子**（FLA packed decode，实测已达 4090 峰值带宽的约 85%、
  857–862 GB/s）为基线，在同一 harness 内对比：完整算子口径 **full-contract 1.61–1.73×**
  （batch 64，L=4/8/16），core 口径 1.99–2.21×，状态搬运量减少约 2.8×；给出 batch × 窗口的
  完整 Pareto 与 break-even（core 约 batch 4、full-contract 约 batch 16），并用 4096 步
  **真实 decode 输入**复现同一张表（1.61–1.72×）。
- 建立完整数值证据链：自写 FP32 recurrence 参考解证明 full-precision replay 与逐 token 递推
  **零误差**；用 4085 步真实 capture 做离线"精确链 vs 量化链"对照，得到漂移
  ≈ 1%×√(flush 次数) 的**不饱和累积律**，并据此反推容量-上下文的定量 Pareto
  （4k token 漂移 ≤5% 需 L≈160、ring 比状态还大，即 1 byte/element 下大容量与长上下文不可兼得）。
- 用 **teacher forcing**（含两道可证伪自检：强制失效即 abort）验证模型级误差：128 token 满足
  预注册误差预算（p95 4.2e-2 < 0.125），**2048 token 上 p95 偏差超出阈值（0.155–0.267）**，
  且恶化幅度与离线漂移律一致 —— 两套独立测量互相印证，明确刻画方案适用边界。另发现 bf16 logits
  使 top-2 margin 以 0.125 为主要离散粒度、约 1.5% 的 decode 步为精确 tie，证明所有 token 翻转
  与量化无关，据此把验收口径从贪心一致率改为分布距离。

## 英文版

**FP8 Recurrent-State Cache + Replay Fused Kernel — a production-baseline-driven Triton operator study**
`Qwen3.5-4B (Mamba-hybrid) · vLLM · Triton · CUDA · RTX 4090` · [repo](https://github.com/zz774ever/fp8-replayssm-gdn-kernel)

- Designed and implemented a **"FP8 checkpoint + short-window replay"** execution path for the
  recurrent state of a Mamba-hybrid LLM (motivated by vLLM issue #55196): the exact backward
  low-rank identity turns the per-token full state read-modify-write into one checkpoint read plus
  an input-ring read, rebuilding and requantising the state only every L steps. Triton
  implementation with fused q/k L2 norm + gating + ring append.
- Benchmarked against **vLLM's production GDN operator** (FLA packed decode, itself at about 85% of
  the 4090's peak bandwidth / 857-862 GB/s) inside one harness: **1.61-1.73x full-operator-contract**
  (batch 64, L=4/8/16; 1.99–2.21× core-only) with **2.76× less state traffic**, a full
  batch × window Pareto with break-even at batch 8–16, and the same table reproduced from a
  4096-step **real decode capture** (1.61-1.72x).
- Built the numerical evidence chain: an FP32 recurrence reference proving full-precision replay is
  exact; an offline exact-vs-quantised chain study over 4085 captured steps giving a
  **non-saturating drift law of ≈1% × sqrt(flushes)**, from which the capacity/context Pareto
  follows quantitatively (≤5% drift at 4k tokens would need L≈160, whose ring exceeds the state —
  1 byte/element cannot buy both capacity and long context).
- Used **teacher forcing** (with two falsifiable self-checks that abort if forcing is ineffective)
  to measure model-level error: 128 tokens satisfy the pre-registered budget (p95 4.2e-2 < 0.125),
  while **2048 tokens exceed it (0.155–0.267)** with growth matching the offline drift law, i.e. two
  independent measurements agree. Also identified that bf16 logits discretise the top-2 margin at
  0.125 and make ~1.5% of decode steps exact ties, proving token flips are unrelated to
  quantisation, and moved acceptance from greedy agreement to distribution distance.

## 不要写

- ❌ 端到端提速 X% / 已集成 vLLM（没做集成）
- ❌ 显存降低约一半（实测 ≤1.4×，且受漂移限制）
- ❌ "FP8 replay 无精度损失"（2k token 未通过判据）
- ❌ "适用于短到中等上下文"（只测了 128 通过、2048 未通过，中间未扫描）
- ❌ 把 0.125 解释成"bf16 ulp = logprob 分辨率"（应表述为预注册的经验阈值）

## 面试会被追问的四点

1. **~1.7× 是不是弱基线？** 基线是生产算子而非自写 kernel，它已达 857–862 GB/s（峰值约 85%）；
   而且我一度让自研基线被低估 2×（计时闭包内算了分配开销），是自己发现并修掉的。
2. **长上下文没过，方案是不是没用？** 不是——是把适用边界量化了：短序列可用、长序列不行，
   且失败趋势与完全独立的离线状态漂移测量一致（√flush 律）。
3. **core 与 full-contract 差在哪？** 差在预处理（q/k 归一化 + 门控 + ring 追加），
   原本两次 launch、融合后 batch 64 约 18 µs 固定开销。
4. **为什么不做 vLLM 集成？** 算子项目的交付边界是算子本身；且容量收益上限已测出约 1.4×，
   不足以支撑 allocator / paged-state / CUDA Graph 那一整条工程投入。

### 一个必须知道的未收尾项

1. **core 数字（2.21 / 2.15 / 1.99）是干净测量**：来自 `bench_gdn_vs_production.py`，
   该脚本的生产算子侧不含任何闭包内拷贝。
2. **full-contract 数字经过一次分母修正**：原脚本把生产算子那侧的
   `mixed_qkv[:, position].contiguous()` 写在计时闭包内（batch 64 约 6.5 µs），
   使比值偏高约 4%（原始 1.81/1.79/1.68 → 修正后 **1.73/1.72/1.61**）。
   修正方式是**同一批数据的干净分母重算**，不是重新测量。
3. **修复后的计时口径尚未复测**（GPU 实例已关闭），复测只需：
   `python prototype/bench_gdn_full_contract.py --batches 64 --window 4,8,16`。

若被问到计时是否干净，诚实答法是：**"core 是干净测量；full-contract 我已发现并修正了分母偏差
（用干净耗时重算了同一批数据），但修正后的口径还没有复跑。"**
