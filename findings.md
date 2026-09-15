# 结论与证据（按主题）

> 本文按**主题**归并项目的全部事实与结论；按**时间**的过程记录（含当时为什么这么决定、
> 踩了哪些坑）见 `progress.md`。总览见 `REPORT.md`。

---

## A. 上游事实与接口约束

### A1. vLLM issue #55196（2026-09-03 提交）

- TL;DR：纯 attention 模型开 FP8 可拿到 2.00× 容量，而 Falcon-H1 这类 hybrid 模型在 32k 上下文
  只有 1.84×、短上下文只有 **1.00×**。
- 根因是两件事叠加：`get_mamba_state_dtype_from_config` 仍是 BF16（page 约 1.59 MB），
  而 `get_uniform_page_size` + `_align_hybrid_block_size` 强制 attention 与 mamba 使用**统一 page**，
  于是 attention block 被撑大。
- RFC 明确：`--mamba-cache-dtype` 目前只接受 `auto/float32/float16/bfloat16`，**没有 FP8/INT8**；
  conv state 量化未实现。
- RFC 的建议顺序是：**先做 per-group page 解耦**（只动 KV cache manager），**再**做状态量化，
  且量化要有 perplexity / 下游精度门槛。
- 该 RFC 的数字多为离线容量估算，混合模型只有一个锚点，因此每个模型仍需自行验证。

### A2. vLLM 里 Qwen GDN 的实际结构（读源码得到）

- `QwenGatedDeltaNetAttention.get_state_shape()` 走
  `MambaStateShapeCalculator.gated_delta_net_state_shape(...)`，
  因此 cache **不是**单一 `[slot, head, d_k, d_v]`，而是含 recurrent state + **conv state** +
  speculative 变体。
- CUDA fused decode 明确只接受：BF16 模型、BF16 conv cache、BF16/FP32 recurrent state、
K=V=128、CC≥8.0。**加 FP8 必须改 dispatch 守卫与底层算子，不只是改 dtype 解析。**
- Qwen3.5 用非交错 `[q,k,v]` 投影布局（Qwen3-Next 是另一种交错布局），因此本项目只针对 Qwen3.5。
- 门控在 FP32 里算（`A_log.exp`、softplus、sigmoid），暗示即便存储用 FP8，
  反量化后的累加也应保持 FP32/BF16。

### A3. 与本项目相关的生产算子调用契约

- 调用点：`qwen_gdn_linear_attn.py` 的 `_forward_core_decode_non_spec`。
- 关键约定：`out` 必须是 `[B, 1, HV, V]` 连续张量；`initial_state` 就是
  `self.kv_cache[1]`（`[slots, HV, V, K]`）；`ssm_state_indices` 槽位 **0 表示空块**；
  `use_qk_l2norm_in_kernel=True`，**归一化在算子内部做**；`scale = head_k_dim**-0.5`。
- 生产实现位于 `vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`，
  grid 为 `(V/32, B*HV)`、`num_warps=1`、`num_stages=3`。
- 该算子内部公式（与本项目 prep 必须对齐的语义）：
  `x/sqrt(sum(x*x)+1e-6)`、`softplus` 阈值 **20.0**、`g=-exp(A_log)*softplus(a+dt_bias)`、
  `beta=sigmoid(b)`。

### A4. ReplaySSM 在 vLLM 中的现状

- 主干已有 ReplaySSM 管线：`CacheConfig` 的 `replayssm_buffer_len` / `use_replayssm`、
  `MambaAttentionBackend` 的 ring/checkpoint 元数据（`is_flush`、scratch、CPU ring origin）、
  `ssu_dispatch` 的 ring tracker、模型能力标志与端到端 benchmark 脚本。
- 但生产 ReplaySSM 的 output-only kernel 是 **Mamba2 专用**
  （`ops/selective_state_update_replayssm_output_only.py`），Qwen GDN 走自己的 fused recurrent decode。
  可以复用其 ring/checkpoint **契约**，但不能复用它的 kernel 当作正确性捷径。
- 值得注意的是：该实现本身就是**预计算 kernel + 主 kernel** 两段式，并在预计算里把整个 ring
  当 2D tile 一次性 reduce —— 这与本项目后来选择的拆分方向一致（但动因不同）。

### A5. 平台约束（RTX 4090）

- Ada SM89、24 GB、峰值带宽约 1008 GB/s；适合 Triton 正确性/格式实验与相对加速比，
  **不适合**复现 H100 绝对吞吐。
- 本方案不需要 FP8 矩阵乘：状态以 FP8 存储、反量化为 BF16/FP32 参与递推、再重新量化，
  因此不依赖 Hopper 专属的 FP8 Tensor Core 行为。
- 实测基线：`max_model_len=512` 时 attention block 被撑到 **528 token** 以匹配 mamba page，
  mamba page 额外 padding **0.76%**；eager 基线为 8.61 GiB 权重、9.23 GiB KV、
  41,837 token 容量、batch-1 八 token 解码 23.32 tok/s。

### A6. 远端执行环境

- 远端 GPU 主机（SSH 别名 `gdn-remote`，非标准端口；具体地址/端口不入库）。
  Ubuntu 22.04.5、驱动 595.80、CUDA 13.2、Torch 2.13.0+cu132、Triton 3.7.1、FlashInfer 0.6.18。
- `/root/vllm` 是干净 worktree；本项目使用独立 worktree `/root/vllm-fp8-replayssm`
  （分支 `codex/fp8-gdn-replayssm`），避免污染既有分支。
- 该主机为容器型实例（PID 1 = tini，能力集不含 CAP_SYS_BOOT），**无法从实例内部关机**，
  只能通过服务商控制台操作。

---

## B. 算法与实现

### B1. 为什么必须用反向（伴随）形式

正向重放（沿时间推进状态）每 token 仍是 O(L·V·K)，等于把"省下的搬运"换成"重算"——
第一版实测比基线更慢。反向形式把代价降为**一次 checkpoint matvec + O(L·(K+V))**：

```
T <- q_t
for j = t … 窗口起点:
      c_j = beta_j * (k_jᵀ T);  o += c_j * v_j;  T <- γ_j * (T - c_j * k_j)
o += <S_checkpoint, T>
```

成立前提：**γ 是每个 value head 的标量**（作用在整块 `[V,K]`），因此衰减退火与外积写入可以换序。

### B2. 量化 ABI

- FP8 **E4M3**，scale 粒度 **vblock = 每 32 个 value 行 × 全 K=128 一个 FP16**，元数据约 0.2%。
- 选择依据：head-scale 误差与之相当但 scale 数量多 128×；per-row scale 只改善约 3%；
  块 32→16 行漂移仅变化约 1e-5（0.1745051 vs 0.1745277）。
  → **误差由尾数精度主导，不由 scale 粒度主导。**
- 真实状态**离群值明显**：同字节 INT8 在多数层明显更差（见 D3），因此"改 INT8"不是普遍答案。

### B3. kernel 结构与调优结论

| 结论 | 证据 |
| --- | --- |
| 宽 value-tile 必须**按行 gather scale** | 否则跨量化 band 误用首个 scale，输出误差 1.7e-3 → 1.2e-2；修正后三种 tiling 输出**逐位一致** |
| 占用率诊断：宽 tile 用满 255 寄存器/线程 | `n_spills=0` 但纯流式读 checkpoint 只有 613 GB/s（基线 826 GB/s） |
| **两段式拆分**有效 | 把反向链从流式 kernel 拆出（T 落显存），apply 才能大块连续读；L=16/batch 64 由 1.48× → 1.77× |
| **K 分块越大越快**（32/64/128 中整块最快） | 说明收益来自"拆分本身"，**推翻**最初"寄存器压力"的归因 |
| 预计算**分组**（一个 key head 服务其全部 value head）**无收益** | 省掉 k 重读但总时间只在噪声内变化 |
| prep 是**固定 launch/延迟开销** | batch 64 约 18 µs，`num_warps` 1 vs 4 无差别 |
| 按整周期摊销才是正确口径 | 窗口内 ring 长度是 1..L 递增，只有第 L 步 flush；按"每步满 ring"计时会把成本高估 10–15% |

### B4. 模型级注入方法

- hook 位置必须是**卷积之后**的 packed recurrent 调用边界；早期把 `_forward_core_decode_non_spec`
  的 `mixed_qkv` 当作循环输入（其实是卷积前）导致首 token 就出现数倍误差。
- 写回有效性用"下一步读到的状态 == 上一步写入值"验证，实测相对误差**恰为 0.0**。
- 必须在 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 下运行，否则 engine spawn 不继承 monkeypatch。

### B5. teacher forcing 的实现要点

- vLLM 贪心路径返回**未修改 logits**，`gather_logprobs` 同时给出 top-k 与"被采样 token"的 logprob；
  因此只需覆盖 `sampled_token_ids`，记录的仍是模型给该 token 的真实概率。
- 真正被调用的入口是 **`GPUModelRunner.sample`**（探针实测），
  `Sampler.sample` / `Sampler.forward` **从未被调用**。
- 两道自检缺一不可：①关闭 replay 时强制解码须与 baseline 逐位一致；②故意喂被破坏序列，
  引擎须原样输出。仅靠①会在"强制完全失效"时**假阳性通过**。

---

## C. 正确性

| 检查 | 数值 |
| --- | --- |
| full-precision replay vs 逐 token 递推 | 输出与末状态**零误差** |
| FP8 checkpoint 单次状态重建（真实 state，8 层 × 8 decode 步） | 相对 L2 **1.7–2.6%** |
| 同字节 INT8 单次重建（对照） | 多数层 **4–5.8%**（层 0 接近持平） |
| kernel 输出 vs fp32 参考解 | **1.65e-3–1.70e-3**（= bf16 输出舍入底噪） |
| flush 状态 vs 参考解 | **6.3e-3–6.9e-3**（一次独立 FP8 量化） |
| full-contract：生产算子 vs fp32 | 2.5e-3–4.6e-3 |
| full-contract：replay vs 生产算子 | 3.8e-3–6.7e-3 |

---

## D. 性能

### D1. core 口径（replay + 摊销 flush，不含 prep）

| batch | L=4 | L=8 | L=16 |
| --- | --- | --- | --- |
| 1 | 1.07 | 0.87 | 0.75 |
| 4 | 1.18 | 1.11 | 1.01 |
| 8 | 1.53 | 1.32 | 1.30 |
| 16 | 1.85 | 1.65 | 1.52 |
| 32 | 2.01 | 1.94 | 1.79 |
| 64 | 2.21 | 2.15 | 1.99 |

### D2. full-contract 口径（含 q/k 归一化 + 门控 + ring 追加）

| batch | L=4 | L=8 | L=16 |
| --- | --- | --- | --- |
| 1 | 0.54 | 0.53 | 0.46 |
| 2 | 0.48 | 0.48 | 0.43 |
| 4 | 0.57 | 0.56 | 0.52 |
| 8 | 0.80 | 0.78 | 0.73 |
| 16 | 1.14 | 1.11 | 1.04 |
| 32 | 1.42 | 1.43 | 1.33 |
| 64 | **1.73** | **1.72** | **1.61** |

**这些数字经过一次分母修正**：full-contract 脚本把生产算子那侧的
`mixed_qkv[:, position].contiguous()` 写在了计时闭包内（batch 64 约 6.5 µs，随 batch 缩放），
使原始比值偏高约 4%（原始值 1.81/1.79/1.68）。上表用干净口径的生产算子耗时重算，
属于**同一批数据的算术重算**，不是重新测量；修复后的计时口径已提交但未复测。

### D3. 真实 trace 复核与带宽

- 4096 步真实 decode capture 驱动同一 harness，batch 64（同样修正后）：
  **L=4 1.70×、L=8 1.72×、L=16 1.61×**（合成输入 1.73/1.72/1.61）——
  "合成数据过于理想"的质疑排除。
- 生产算子达成带宽 **857–862 GB/s**（峰值约 85%）；自写同风格 BF16 kernel 为 822–826 GB/s
  （峰值约 82%）；replay 在 batch 64 下为
  L=4 约 630–660、L=8 约 560–620、L=16 约 440–510 GB/s
  —— **效率更低且随窗口变长而下降（ring 循环更久），但搬得更少**，赢在搬运量。
- 字节比（按访问模式核算，非实测）：L=4 2.88×、L=8 2.88×、L=16 2.76×、L=32 2.23×。
- break-even batch：core 约 4，full-contract 约 8–16。

---

## E. 长序列误差寿命

### E1. 漂移律（4085 步真实输入，离线双链同步推进）

| 窗口 | 常驻字节 vs bf16 | 平均状态漂移（层 0/8/16/24） |
| --- | --- | --- |
| L=4 | 0.55× | 0.268 / 0.314 / 0.253 / 0.290 |
| L=16 | 0.70× | 0.175 / 0.157 / 0.134 / 0.149 |
| L=64 | 1.28× | 0.082 / 0.070 / 0.067 / 0.072 |

- 三种窗口反推的"每次 flush 注入误差"几乎一致（**0.84% / 1.1% / 1.0%**）
  → **漂移 ≈ 1% × √(flush 次数)**，累积律由**量化器**而非 flush 频率决定。
- 增长不是饱和的：L=4、层 0 的漂移 0.021(step 0) → 0.098(128) → 0.171(512) → 0.211(1024)，
  8 倍步长只放大 2.15 倍（约 t^0.37）。

### E2. 容量-上下文 Pareto（由漂移律反推）

| 目标（4k token 状态漂移） | 需要的 L | 常驻字节比 |
| --- | --- | --- |
| ≤5% | ≈160 | 1.95×（ring 比状态还大） |
| ≤10% | ≈40 | 0.99×（容量收益归零） |
| ≤15% | ≈18 | 0.72× |

→ **1 byte/element 的 checkpoint 无法同时换取大容量与长上下文**；
"2× 容量"叙事在本方案下**不成立**，可辩护的容量收益约 1.4×。

### E3. 格式与方案对照

- **INT8（同字节）**：层 0/16/24 明显更好（层 0、L=16：0.175→0.069），
  层 8 明显更差（0.157→0.249）→ **按层选格式**是零字节成本的空间，层 8 是瓶颈层。
- **K 轴 Hadamard 旋转被否证**：递推在该旋转下严格等变（数值验证 1.6e-7），
  但旋转后 readout 误差显著变差（12 组里 9 组更差；最差输出漂移 0.131→0.534）。
  原因：原状态在 K 上稀疏，per-row scale 正好吃到该结构，旋转把能量摊平反而破坏它。
- **块大小（32→16 行）**：漂移仅变化约 1e-5 → 误差由尾数主导。

---

## F. 模型级验证

### F1. 注入式 A/B（4 prompt × 256 token）

- baseline 可重复性：逐 token 完全一致，`max|Δlogprob|` **恰为 0**。
- FP8 臂：token 一致率 0.47–1.0，翻转全部发生在 margin=0 的步；同 context 平均 |Δlogprob| 4.4e-3–1.4e-2。
- **关键对照**：完全不做量化的 replay 臂也在**同一步、翻到同一个替代 token**
  （例如 prompt 0 第 120 步：baseline 选 token 13，四个 replay 臂全部选 token 318）。

### F2. bf16 tie（评估方法学）

- 到达采样器的 top-2 margin 取值**恒为 0.125 的整数倍**（0, 0.125, 0.25, …）。
- margin=0 的步占比：单 prompt 5/256；4 个 prompt 合计 **15/1024 ≈ 1.5%**。
- 所有观测到的 token 翻转都发生在这些步上，且与是否量化无关。
- 因此验收口径改为 **teacher forcing / 分布距离**；0.125 作为**预注册的经验阈值**
  （依据是上面这个离散粒度，**不是**"bf16 的 ulp"或"logprob 的分辨率"这类理论量）。

### F3. teacher-forced 数值验收（2048 步，全部纳入统计）

| 长度 | L | mean abs Δlogprob | p95 | max | mean KL(top-k) | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 4 | 0.0075 | **0.042** | — | — | 通过 |
| 2048 | 4 | 0.0529 | **0.2666** | 1.238 | 0.137 | 未通过 |
| 2048 | 8 | 0.0377 | **0.1863** | 0.947 | 0.082 | 未通过 |
| 2048 | 16 | 0.0311 | **0.1546** | 0.682 | 0.076 | 未通过 |

- 误差随 L 增大单调下降，但 2k token 上无一窗口落进预算。
- 恶化趋势与 E1 的离线漂移律一致（flush 次数 ×16 → 扰动 ×4），**两条独立测量互相印证**。
- 量级：平均扰动约为典型贪心 margin 中位数（3.5–4.75）的 1%，token 级行为大体保持，
  但分布已被可测量地改变。
- 适用边界只被钉在 **128（通过）与 2048（未通过）两点之间**，中间未扫描。

---

## G. 工程缺陷与修正（含保留的失败证据）

| 缺陷 | 表现 | 教训 |
| --- | --- | --- |
| 宽 value-tile 跨多个量化 band 只取首个 scale | 输出误差 1.7e-3 → 1.2e-2 | **只有正确性校验能发现**，纯看速度看不出来 |
| harness 把 `clone()` / dtype 转换写进计时闭包 | 自研基线被低估 2×，差点得出"生产算子快 2 倍"的错误结论 | 基线不对齐时先怀疑测量装置 |
| 两段式拆分首版无收益 | 与 tiled 持平 | 拆分本身不改占用率；收益来自让 T 离开寄存器 |
| 强制 token 挂错对象（`Sampler.sample` 未被调用） | 第一个自检假阳性通过 | 自检必须能证伪"机制生效"，而不只是"结果一致" |
| prep kernel 2D 张量多传一个 stride / 参考解只给 1 个 token 而 ring 有 window 个 | 直接报错或输出完全错 | 形状/语义错配要用参考解校验兜住 |
| 早期把卷积前输入当作循环输入 | 首 token 就出现数倍误差 | hook 位置必须与生产算子的真实入参边界一致 |
| `config.toml` 被应用重写导致 MCP 段落丢失 | 本地 MCP 不可用 | 配置会被外部工具重写，需可重复校验 |

日志中**保留**了两份"带缺陷"的证据：`results/kernel_sweep_v1_with_scale_bug.log` 与
`results/vs_production_first_pass_with_harness_bug.log`。

---

## H. 环境与版本（复现基线）

| 组件 | 版本 / 标识 |
| --- | --- |
| GPU | RTX 4090（SM89，24 GB，峰值约 1008 GB/s），驱动 595.80 |
| 系统 / CUDA | Ubuntu 22.04.5 / CUDA 13.2 |
| PyTorch / Triton / FlashInfer | 2.13.0+cu132 / 3.7.1 / 0.6.18 |
| vLLM | `0.1.dev20944+g58ad1f3b8`，commit `58ad1f3b8973b23943107b51230d594050b42ec3` |
| 模型 | Qwen3.5-4B HF snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
