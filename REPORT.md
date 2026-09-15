# 技术报告：面向 Mamba 混合 LLM 的 FP8 循环状态缓存 + Replay 融合算子

> 本文是项目的**主文档**：拿到这个仓库的人，读完这一篇就应该知道"做了什么、怎么做的、证据在哪"。
> 其余文档分工见附录 D。

---

## 0. 三十秒版本

**做了什么**：给 Qwen3.5-4B（Mamba 混合架构）的 decode 阶段写了一个新的 **GDN 单步算子**。
它不再每生成一个 token 就把整块循环状态读进来、改完再写回去，而是把持久状态改成
**FP8 快照（checkpoint）+ 最近 L 个 token 的输入（ring）**，用**反向低秩等价式**直接算出当前
token 需要的输出；完整状态每 L 步才重建并重新量化一次（flush）。

**结果**：相对 vLLM 实际运行的生产算子（它已达 4090 峰值带宽的约 85%、857–862 GB/s），
**完整算子口径 1.61–1.73×、core 口径 1.99–2.21×**（batch 64，L=4/8/16）；
状态字节搬运减少约 2.8×。用 4096 步真实 decode 输入复现同一张表（1.61–1.72×）。

**代价（同样重要）**：容量只省约 30%（1.4×），而且每次 flush 都注入约 1% 相对误差，
误差按 **√(flush 次数)** 不饱和累积 —— 所以 **128 token 满足预注册误差预算，
2048 token 不满足**。这是一个"带宽换精度"的方案，适用边界被量化，而不是被回避。

---

## 1. 背景与问题

### 1.1 为什么盯上循环状态

Qwen3.5-4B 每 4 层一次 full attention，其余 **24 层是 GDN（gated delta net）**。
每个 GDN 层为每个序列维护一个循环状态：

```
形状 [value_heads=32, V=128, K=128]，bf16  ->  32*128*128*2 B = 1 MiB / 层 / 序列
```

decode 每生成一个 token、对**每一层**都要读它、更新它、写回去。batch 64 时：

```
24 层 * (1 MiB 读 + 1 MiB 写) * 64 = 3 GiB / token
```

实测单层一步在 batch 64 下约 **156 µs**，即 **857–862 GB/s ≈ 峰值带宽（约 1008 GB/s）的 85%**
（作为对照，本项目自写的同风格 BF16 kernel 为 162.5 µs / 826 GB/s，即峰值的 82%）。

**这个数字决定了整个项目的技术路线**：生产算子已经贴着硬件上限，靠调优写法拿不到大收益；
唯一有量级空间的方向是**少搬数据**。而上游 [vLLM issue #55196](https://github.com/vllm-project/vllm/issues/55196)
正是在讲 hybrid 模型上这类状态带来的容量/带宽压力。

### 1.2 目标与非目标

| | |
| --- | --- |
| 目标 | 在**不改模型精度前提**下改变状态的执行方式，度量能换来多少带宽/时间，并**量化它的数值代价** |
| 非目标 | ① production-ready 的 vLLM 特性；② 解决 #55196 的容量问题；③ 端到端 tokens/s 优化 |

原因：① 属于框架侧（allocator / paged-state / CUDA Graph / 抢占 / 投机解码）；
② 长序列实验证明容量上限只有约 1.4×；③ 算子级结论不需要它，且端到端会被其它层摊薄。

---

## 2. 术语与符号

| 符号 | 含义 |
| --- | --- |
| `S` | 循环状态矩阵 `[HV, V, K]`（每个 value head 一块） |
| `q, k, v` | 当前 token 的 query / key / value（`k` 参与外积写入，`q` 用于读出输出） |
| `γ = exp(g)` | 每 token、每个 value head 一个**标量** decay（作用在整块 `[V,K]` 上，这是等价变换成立的关键） |
| `β` | 写入强度（`sigmoid(b)`） |
| `L` | replay 窗口长度，即 ring 里保留多少个 token |
| **flush** | 每 L 步做一次：用 checkpoint + ring 重建完整状态 → 量化回 FP8 → 清空 ring |
| **checkpoint** | 窗口开始时状态的 FP8 压缩快照 |
| **ring** | 窗口内每个 token 的循环输入（归一化 `k`、`v`、`g`、`β`），约 12.2 KB / token / 层 |
| **prep** | 每 token 的预处理：q/k L2 归一化 + 门控计算 + 写 ring |
| **core / full-contract** | 性能的两种口径，见 §6.2 |
| `HV / H` | value head 数 / key head 数（Qwen3.5-4B：HV=32，H=16，比例 2） |

**递推本体**：

```
S_t = γ_t · S_{t-1} + β_t · (v_t − S_{t-1} k_t) · k_tᵀ
输出 = S_t · q_t · K^(-1/2)
```

---

## 3. 算子规格（它到底替换了什么）

本算子与生产路径的 `fused_recurrent_gated_delta_rule_packed_decode`
处于**同一个调用位点、同一份输入输出契约**——模型级实验里就是直接替换该调用，
因此它是**可替换算子**，不是外围脚本。

| 项 | 生产算子 | 本算子 |
| --- | --- | --- |
| 输入 | `mixed_qkv [B, 2HK+HVV]` 原始拼接 q/k/v；`a`/`b` 原始门控 `[B,HV]`；`A_log`/`dt_bias` `[HV]`；`scale` | **完全相同** |
| 持久状态 | bf16 `[slots, HV, V, K]` 完整状态缓存 | **FP8 checkpoint + 每 32 行 FP16 scale + ring（归一化 k / v / g / β）+ pos / flush 元数据** |
| 输出 | `out [B,1,HV,V]` | **完全相同** |
| 每步是否写整块状态 | 是（读 1 MiB + 写 1 MiB / 层 / 序列） | **否**，只 append 12.2 KB 进 ring |
| launch 结构 | 1 次 | prep 1 次 + replay 1 次（+ 每 L 步 1 次 flush） |

**它不做什么**：不做 conv1d（在该算子之前）、不做输出门控（在其之后）、不涉及 attention 层、
不管理 ring 的分配与生命周期（属框架侧）。

---

## 4. 技术路线（为什么按这个顺序做）

研究路线的每一步都是为了让**上一步的结论不被下一步的混淆项污染**：

| 步 | 做了什么 | 为什么必须在这一步 | 结论 |
| --- | --- | --- | --- |
| 1 | 读源码与上游 issue，确定真实瓶颈与接口约束 | 避免把 allocator/page 问题误当成算法问题 | 生产算子已达带宽 81%；混合模型 page 被绑定（attention block 被撑到 528 token、mamba page 额外 padding 0.76%） |
| 2 | 写 FP32 recurrence 参考实现，证明**全精度 replay 与逐 token 递推等价** | 先证明"算法对"，否则后面所有误差都无法归因 | 输出与末状态**零误差** |
| 3 | 引入 FP8 checkpoint，量化单次重建误差 | 把"量化误差"与"算法误差"分开 | 真实状态单次重建 1.7–2.6% |
| 4 | **注入式**模型级 A/B：把 replay 输出与重建状态写回真实 decode 路径 | 证明确实跑在量化链上，而不是只做离线比较 | 写回逐位验证通过；但出现 token 翻转 |
| 5 | 追查翻转来源 → 发现 bf16 tie 现象 | 不追查就会把 tie 造成的混沌误判为"FP8 精度不足" | 翻转与量化**无关**（无量化对照臂同样翻） |
| 6 | 长序列漂移研究（4085 步真实输入，离线双链同步推进） | 贪心一致率已不可用，必须换一个不受 tie 影响的指标 | 漂移 **不饱和**，≈1%×√(flush 次数) |
| 7 | 融合 kernel 开发 + 与**生产算子**同 harness 对比 | 算子项目的验收基线必须是现有生产实现 | core 2.21×；发现并修掉多个真实缺陷 |
| 8 | full-contract 口径 + 真实 trace 复核 + teacher forcing 数值验收 | 关掉"偷跑"和"合成数据"两个质疑，并给出模型级判据 | full 1.61–1.73×；**2k token 数值未通过** |

**一句话概括路线**：先把"算法是否等价"钉死，再把"量化误差多大"量出来，
再证明"模型确实跑在量化链上"，再用不受 tie 影响的指标测寿命，最后才谈性能与适用边界。

---

## 5. 实现方法

### 5.1 一次调用做三件事

```
① prep  (每 token)  原始 q/k/v, a, b
      -> q/k L2 归一化 x/sqrt(sum(x^2)+1e-6)
      -> g = -exp(A_log)*softplus(a+dt_bias)（阈值 20.0）,  beta = sigmoid(b)
      -> 归一化 k、v、g、beta 写入 ring 的第 pos 槽

② replay (每 token) checkpoint(FP8) + ring
      -> 反向低秩递推 -> 直接输出（不读、不写整块状态）

③ flush  (每 L 步)  checkpoint 反量化 -> 沿 ring 正向重建完整状态
      -> 按 32 行算 amax -> 量化回 FP8 -> 回写 checkpoint，清空 ring
```

### 5.2 核心数学：为什么可以不算完整状态

沿时间正推状态的话，每 token 仍要做 L 次整块矩阵运算（O(L·V·K)）——**等于白干**
（第一版就是这样，比基线慢）。真正管用的是**反向（伴随）形式**：不推状态，改推**查询向量** `T`：

```
T <- q_t
for j = t … 窗口起点:
      c_j = beta_j * (k_jᵀ T)          # 标量
      o  += c_j * v_j                  # 累加到 V 维输出
      T  <- γ_j * (T - c_j * k_j)      # K 维向量，全程在寄存器
o += <S_checkpoint, T>                 # 一次矩阵-向量乘
```

代价从 O(L·V·K) 降到**一次 checkpoint matvec + O(L·(K+V))**。
成立的关键是 **γ 是每个 value head 的标量**，所以"衰减退火"与"外积写入"可以这样换序而不改变结果。

### 5.3 量化 ABI 与其选择依据

| 项 | 取值 | 依据 |
| --- | --- | --- |
| 格式 | FP8 E4M3 | 真实状态含明显离群值，动态范围比额外尾数更重要（同字节 INT8 在多数层更差） |
| scale 粒度 | **每 32 个 value 行 × 全 K=128 一个 FP16** | head-scale 误差相当但 scale 数多 128×；per-row 只改善约 3%；32→16 行仅变化约 1e-5 |
| 元数据开销 | 约 0.2% | 每层 128 个 FP16 |
| 何时量化 | 每次 flush | 非 flush 步不写状态 |

**结论**：误差由**尾数精度**主导，不由 scale 粒度主导——这解释了为什么"细化 scale" 收效甚微。

### 5.4 kernel 清单与 tiling

| kernel | grid | 职责 | num_warps |
| --- | --- | --- | --- |
| `_gdn_prep_fused_kernel` | `(B, H+HV)` | 前半 program 做 q/k 归一化并写 ring，后半做门控计算 + v 拷贝写 ring。**融合为单次 launch**，对齐生产算子"一次 launch 做完"的工作曲线 | 1 |
| `_gdn_replay_fp8_kernel`（非 flush） | `(B, HV, V/BLOCK_V)` | 读 FP8 checkpoint tile（**按行 gather scale** 反量化）→ 反向遍历 ring 累积输出 → `out += S·T` → 乘 `K^-0.5` 写出。**不写状态** | 4 |
| 同一 kernel（flush 分支） | 同上，`BLOCK_V=32` | 正向重建状态 → 输出 → **重新量化并回写 checkpoint + scale**；用 per-batch `flush` 标志分叉 | 4 |
| `_gdn_replay_precompute_kernel` + `_gdn_replay_apply_kernel` | `(B,HV)` + `(B,HV,V/BLOCK_V)` | 两段式：反向链每个 `(batch, value head)` 只算一次并落盘 `T` 与系数 `c_t`；apply 只做 `out = Σ c_t·v_t + S0@T`，checkpoint 可大块连续读 | 1 / 4 |
| `_gdn_replay_precompute_grouped_kernel` | `(B, H)` | 一个 program 服务同一 key head 的全部 value head，省掉 k 重读。实测**无收益**，如实保留 | 1 |
| `_gdn_bf16_step_kernel` | `(B,HV,V/BLOCK_V)` | 仅作同风格对照：读 bf16 状态 → 单步更新 → 写回 → 输出 | 4 |

**四处实现约束（都是踩过才知道的）**

1. **scale 是每 32 行一个**：宽 tile 跨多个 band 时若只取首个 scale，输出误差从 1.7e-3 跳到 1.2e-2。
   必须按行 `gather(offs_v // 32)`。**只有对 fp32 参考解的正确性校验能发现这个缺陷**。
2. **两段式拆分的收益来自"让 T 离开寄存器"**：T 落到显存后 apply 才能按 K 分块读 checkpoint；
   而 K 分块 32/64/128 实测**整块（128）最快**——说明收益来自拆分本身，而非缩小寄存器 tile
   （这推翻了我最初"寄存器压力"的假设）。
3. **replay 与 flush 的最优 tiling 不同**：replay 用 `BLOCK_V=64/128` 最快，flush 必须钉在
   `BLOCK_V=32`（要按 32 行写 scale）。基准中两者分开计时、再按窗口摊销。
4. **prep 是固定开销**：batch 64 约 18 µs，与 `num_warps`（1 vs 4）无关 → launch/延迟受限，
   不是带宽受限；这是 core 与 full-contract 数字落差的来源。

### 5.5 模型级注入式验证是怎么做的

为了证明"模型确实跑在量化链上"，在真实 vLLM decode 路径上打了两个 hook：

1. `QwenGatedDeltaNetAttention._forward_core_decode_non_spec`：建立"当前是第几层"的上下文；
2. `fused_recurrent_gated_delta_rule_packed_decode`：拿到**卷积之后**的 packing 输入与状态槽位 `slot`，
   调用生产算子后，把 **replay 算出的输出写进 `core_attn_out`**，并把**重建状态写回 `initial_state[slot]`**。

写回是否生效用**下一步读到的状态与被写入值逐位比较**来验证（实测相对误差恰为 0.0），
这一步是"注入式"区别于"离线 shadow"的关键。

### 5.6 teacher forcing 是怎么做的（以及为什么必须有两道自检）

贪心 token 一致率不可用（见 §6.6），所以把 baseline 的 token 序列**强制**喂给所有臂，
让它们看到完全相同的输入，只比较**分布**。

关键实现点：

- vLLM 贪心路径返回的是**未修改的 logits**，`gather_logprobs` 会同时给出 top-k 与"被采样 token"的
  logprob —— 因此只需覆盖 `sampled_token_ids`，记录到的仍是模型给该 token 的**真实概率**。
- **自检一**：关闭 replay 臂时，强制解码必须与 baseline 逐位一致（本实验差值全 0）。
- **自检二（必要）**：故意喂一个被破坏的 token 序列，引擎必须**原样输出该序列**。
  实测第一次把 hook 挂在 `Sampler.sample`（引擎根本不调它）时，**自检一假阳性通过**，
  自检二把它抓了出来；改用 `GPUModelRunner.sample` 后才正确。

---

## 6. 关键数据

### 6.1 正确性

| 检查 | 结果 |
| --- | --- |
| full-precision replay vs 逐 token 递推 | 输出与末状态**零误差** |
| FP8 checkpoint 单次状态重建（真实 state，8 层 × 8 步） | 相对 L2 **1.7–2.6%** |
| kernel 输出 vs fp32 参考解 | **1.65e-3–1.70e-3**（= bf16 输出舍入底噪） |
| flush 状态 vs 参考解 | **6.3e-3–6.9e-3**（一次独立 FP8 量化） |
| full-contract：生产算子 vs fp32 | 2.5e-3–4.6e-3 |
| full-contract：replay vs 生产算子 | 3.8e-3–6.7e-3 |

### 6.2 性能：对生产算子（单层、一次 decode step）

基线是 vLLM 实际运行的 `fused_recurrent_gated_delta_rule_packed_decode`（FLA 实现），
在同一 harness 中按真实契约独立调用。两种口径：

- **core** = replay kernel + 摊销 flush（不含 prep）
- **full-contract** = prep + replay + 摊销 flush（与生产算子做**同样多的工作**）

按 flush 整周期摊销（ring 在窗口内是 1..L 递增，第 L 步才 flush）：

**core — 加速比**

| batch | L=4 | L=8 | L=16 |
| --- | --- | --- | --- |
| 1 | 1.07 | 0.87 | 0.75 |
| 4 | 1.18 | 1.11 | 1.01 |
| 8 | 1.53 | 1.32 | 1.30 |
| 16 | 1.85 | 1.65 | 1.52 |
| 32 | 2.01 | 1.94 | 1.79 |
| 64 | **2.21** | **2.15** | **1.99** |

**full-contract — 加速比**

| batch | L=4 | L=8 | L=16 |
| --- | --- | --- | --- |
| 1 | 0.54 | 0.53 | 0.46 |
| 2 | 0.48 | 0.48 | 0.43 |
| 4 | 0.57 | 0.56 | 0.52 |
| 8 | 0.80 | 0.78 | 0.73 |
| 16 | 1.14 | 1.11 | 1.04 |
| 32 | 1.42 | 1.43 | 1.33 |
| 64 | **1.73** | **1.72** | **1.61** |

> **修正说明（重要）**：`bench_gdn_full_contract.py` 的 `production_call()` 里，把
> `mixed_qkv[:, position]` 的切片/`contiguous()` 拷贝写在了**计时闭包内**，
> 使**分母（生产算子）偏大约 6.5 µs（batch 64，随 batch 等比缩放）**，
> 因此原始输出的比值偏高约 4%（原始值为 1.81 / 1.79 / 1.68）。
> 上表已用 `bench_gdn_vs_production.py` 中**干净口径**的生产算子耗时
> （batch 64 ≈ 156.2 µs）重算。两者都是真实测量，差别只在计时闭包的边界。
> 该切片已移出闭包并提交，但**未复测**（GPU 实例已关闭）——见 §10。

**真实 trace 复核**（4096 步真实 decode capture 驱动，同一 harness）

| batch | L=4 | L=8 | L=16 |
| --- | --- | --- | --- |
| 16 | 1.13 | 1.11 | 1.03 |
| 32 | 1.41 | 1.43 | 1.33 |
| 64 | **1.70** | **1.72** | **1.61** |

（同样已按干净分母修正；修正前为 1.30/1.27/1.18、1.51/1.56/1.46、1.77/1.81/1.69。）

**带宽与搬运量**

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 生产算子达成带宽 | **857–862 GB/s**（峰值约 85%） | 基线不弱，加速来自执行方式 |
| 自写同风格 BF16 kernel | 822–826 GB/s（峰值约 82%） | 两者差约 4%，说明对照基线是合理的 |
| replay 达成带宽（batch 64） | L=4 约 **630–660**、L=8 约 **560–620**、L=16 约 **440–510** GB/s | 效率低于基线且随窗口变长而下降（ring 循环更久），但搬得更少 |
| 字节比（按访问模式核算，非实测） | L=4 **2.88×**、L=8 **2.88×**、L=16 **2.76×**、L=32 2.23× | 相对 bf16 读-改-写的状态流量 |
| break-even batch | core 约 **4**；full-contract 约 **16** | 低于此值两边都是 launch/延迟受限（prep 的固定 launch 摊不掉） |

### 6.3 长序列误差寿命（4085 步真实输入）

| 窗口 | 常驻字节 vs bf16 状态 | 4085 步平均状态漂移（层 0/8/16/24） |
| --- | --- | --- |
| L=4 | 0.55× | 0.268 / 0.314 / 0.253 / 0.290 |
| L=16 | 0.70× | 0.175 / 0.157 / 0.134 / 0.149 |
| L=64 | 1.28× | 0.082 / 0.070 / 0.067 / 0.072 |

三种窗口反推的"每次 flush 注入误差"几乎一致（**0.84% / 1.1% / 1.0%**），即
**漂移 ≈ 1% × √(flush 次数)**：累积律由**量化器**决定，而不由 flush 频率决定。

| 目标（4k token 状态漂移） | 需要的 L | 常驻字节比 |
| --- | --- | --- |
| ≤5% | ≈160 | 1.95×（ring 比状态还大） |
| ≤10% | ≈40 | 0.99×（容量收益归零） |
| ≤15% | ≈18 | 0.72× |

**即 1 byte/element 的 checkpoint 无法同时换取大容量与长上下文。**

### 6.4 格式与方案对照（4085 步离线扫描）

| 方案 | 结论 |
| --- | --- |
| vblock 32 行 → 16 行 | 漂移仅变化约 1e-5（0.1745051 vs 0.1745277）→ **误差由尾数主导** |
| INT8（同字节） | 层 0/16/24 明显更好（层 0、L=16：0.175→0.069），层 8 明显更差（0.157→0.249）→ **按层选格式是零字节成本的空间** |
| K 轴 Hadamard 旋转 | 递推在该旋转下严格等变（数值验证 1.6e-7），但旋转后 readout 误差显著变差（12 组里 9 组更差，最差输出漂移 0.131→0.534）→ **否证** |
| 朴素正向重放（第一版） | 每 token 仍是 O(L·V·K) → 比基线慢，被反向低秩式取代 |

### 6.5 teacher-forced 数值验收（模型级）

阈值**预先注册**为 **p95 |Δlogprob| < 0.125**；该阈值是**经验**阈值，依据是 baseline 中
bf16 top-2 logit margin 以 **0.125 为主要离散粒度**（不是"bf16 的 ulp"这类理论量：
bf16 的 ulp 随指数变化，logit 取整间距也不等价于 logprob 的固定间距）。

装置自检（两者都必须通过）：`apparatus_ok = true`、`forcing_is_live = true`。

| 序列长度 | L | mean abs Δlogprob | p95 | max | mean KL(top-k) | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 4 | 0.0075 | **0.042** | — | — | ✅ 通过 |
| 2048 | 4 | 0.0529 | **0.2666** | 1.238 | 0.137 | ❌ |
| 2048 | 8 | 0.0377 | **0.1863** | 0.947 | 0.082 | ❌ |
| 2048 | 16 | 0.0311 | **0.1546** | 0.682 | 0.076 | ❌ |

三点解读：

1. **误差随 L 增大单调下降**，但没有一个窗口在 2k token 上落进预算。
2. **恶化趋势与 §6.3 的离线漂移律一致**：128→2048 token 使 flush 次数增加 16×，
   预测扰动增加 4×（√16），实测 L=4 的 p95 从 0.042 涨到 0.267。**两条完全独立的测量路径互相印证。**
3. 量级交代：平均扰动 0.031–0.053 nats 约等于典型贪心 margin（中位数 3.5–4.75）的 1%，
   token 级行为大体保持，但分布已被**可测量地**改变，不能称为无损替换。

### 6.6 评估方法学发现（bf16 tie）

| 观察 | 数据 |
| --- | --- |
| 到达采样器的 top-2 margin 取值 | 恒为 **0.125 的整数倍**（0, 0.125, 0.25, …） |
| margin = 0 的步占比 | 4 个 prompt 合计 **15/1024 ≈ 1.5%**（单 prompt 5/256） |
| token 翻转位置 | 4 prompt × 256 token 的 A/B 中，**所有**翻转都发生在 margin=0 的步 |
| 关键对照 | **完全不做量化的 replay 臂也在同一步、翻到同一个替代 token** |

**结论**：翻转是 replay 路径本身的 tie-breaking 性质，与 FP8 无关；
因此验收口径必须从"贪心 token 一致率"改为**分布距离 / teacher forcing**。

---

## 7. 证据索引（结论 → 脚本 → 日志）

实验在远端 GPU 机的 vLLM worktree 里跑（`benchmarks/`、`benchmarks/kernels/`），
仓库里对应 `prototype/`。原始日志全在 `results/`。

| 结论 | 生成脚本 | 日志 |
| --- | --- | --- |
| FP32 参考解 + FP8 单次重建误差 | `prototype/gdn_replay_fp8_reference.py` | `results/ab_32token_L2L4L8.log` |
| kernel 正确性 + tiling/warp 扫描 | `prototype/bench_gdn_replayssm.py` | `results/kernel_sweep_blockv_warps.log`、`kernel_final_cycle_tuned.log` |
| 宽 tile scale 缺陷（保留为证据） | 同上 | `results/kernel_sweep_v1_with_scale_bug.log` |
| 占用率 / 纯流量上界诊断 | `prototype/bench_kernel_diag.py` | `results/kernel_prep_warps_sweep.log` |
| core 性能对生产算子 | `prototype/bench_gdn_vs_production.py` | `results/vs_production_final.log` |
| harness 计时缺陷（保留为证据） | 同上 | `results/vs_production_first_pass_with_harness_bug.log` |
| full-contract 性能 | `prototype/bench_gdn_full_contract.py` | `results/kernel_full_contract_synthetic.log` |
| 真实 trace 复核 | 同上 + `prototype/qwen35_capture_raw.py` | `results/kernel_full_contract_realtrace.log` |
| 长序列漂移与格式扫描 | `prototype/qwen35_drift_study.py` | `results/drift_4085_*.log`、`drift_2048_*.log` |
| Hadamard 否证 | 同上 | `results/drift_4085_vblock32_rotatedK_L4L16L64.log` |
| 模型级注入式 A/B（含无量化对照） | `prototype/qwen35_replay_ab.py` | `results/ab_256token_*.log` |
| tie 统计 | `prototype/qwen35_logprob_probe.py` | `results/baseline_margin_probe.log` |
| teacher forcing 验收 | `prototype/qwen35_teacher_forced.py` | `results/teacher_forced_2048_steps.log`、`teacher_forced_128_smoke.log` |
| 采样钩子探针（说明为何挂在 runner 上） | `prototype/qwen35_sampler_probe.py` | 输出见 progress.md 的 Session 14 |

---

## 8. 设计决策与被否证的方案

| 决策 | 理由 | 状态 |
| --- | --- | --- |
| 用**反向低秩等价式**而非正向重放 | 正向重放每 token 仍 O(L·V·K)，实测比基线慢 | 采用 |
| checkpoint 用 **FP8 + vblock32** | 误差由尾数主导，更细 scale 无收益；元数据仅 0.2% | 采用 |
| 引入 **prep 融合 kernel** | 生产算子内联做 q/k 归一化与门控，不补齐就是偷跑 | 采用（core→full 的落差即此项） |
| **两段式拆分**（precompute + apply） | 让 T 离开寄存器，apply 可用大块连续读 | 采用 |
| **K 分块** | 实测整块 128 最快 | 保留参数，默认整块 |
| **预计算分组** | 省 k 重读但无收益 | 保留代码，标注无收益 |
| **Hadamard 旋转** | 等变但 readout 变差 | **否证** |
| **全面改用 INT8** | 层间不一致（层 8 更差） | 否证，改为"按层可选" |
| 追求 **2× 容量** | 需 L≈160，ring 超过状态 | **否证**，容量上限约 1.4× |
| 做 **vLLM 集成** | 属框架侧，且容量收益不足以支撑 | 不做（列入后续） |

---

## 9. 已知边界与局限

1. **未集成 vLLM**：所有性能数字是单层、单次 decode step 的算子级结果；端到端 tokens/s 未测。
2. **容量不是卖点**：L=16 时常驻 0.70×（≈1.4× 容量），且长序列受漂移限制。
3. **数值适用边界只被钉在两点之间**：128 token 通过、2048 token 不通过；
   中间的 256/512/1024 **尚未扫描**（因此本文不使用"短/中上下文"这类未经测量的区间表述）。
4. **prep 是固定开销**：batch 64 约 18 µs 且与线程数无关，未做进一步融合。
5. **真实 trace 的 batch 复现方式**：capture 为 batch-1，真实数据跑时把一个真实窗口复制到 batch 维；
   该 kernel 无数据相关分支，因此对计时无影响（输入分布与 FP8 scale 均来自真实数据）。
6. **full-contract 的比值经过一次算术修正**（干净分母重算，非重新测量），
   且修复后的计时口径**尚未复测**；详见 §6.2 的修正说明与 §10 的"未复测项"。
7. **代码注释与变量名保持英文**（工程惯例），文档全部为中文。

---

## 10. 复现

**固化环境**（全部结果对应的精确版本）：

| 组件 | 版本 / 标识 |
| --- | --- |
| GPU | RTX 4090（SM89，24 GB，峰值带宽约 1008 GB/s），驱动 595.80 |
| 系统 / CUDA | Ubuntu 22.04.5 / CUDA 13.2 |
| PyTorch | 2.13.0+cu132 |
| Triton | 3.7.1 |
| FlashInfer | 0.6.18 |
| vLLM | `0.1.dev20944+g58ad1f3b8`，commit `58ad1f3b8973b23943107b51230d594050b42ec3` |
| 模型 | Qwen3.5-4B HF snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

脚本位于 `prototype/`。运行时需放进一个 vLLM 源码环境
（kernel 侧脚本放 `benchmarks/kernels/`、模型级脚本放 `benchmarks/`），并用该环境的 Python。

```bash
# 1) 算子正确性 + core 性能（不需要加载模型）
python prototype/bench_gdn_vs_production.py --batches 1,2,4,8,16,32,64 --window 4,8,16

# 2) full-contract（两边都做 q/k 归一化 + 门控 + ring 追加）
python prototype/bench_gdn_full_contract.py --batches 8,16,32,64 --window 4,8,16
python prototype/bench_gdn_full_contract.py --batches 16,32,64 --window 4,8,16 --capture capture_raw.pt

# 3) tiling/窗口扫描与占用率诊断
python prototype/bench_gdn_replayssm.py --batches 1,4,16,64 --window 4,8,16,32 --block-v 32,64,128
python prototype/bench_kernel_diag.py --window 16 --batch 64 --block-v 32,64,128

# 4) 录制真实输入 + 长序列漂移研究
VLLM_ENABLE_V1_MULTIPROCESSING=0 python prototype/qwen35_capture_raw.py --tokens 4096 --layers 0,8,16,24
VLLM_ENABLE_V1_MULTIPROCESSING=0 python prototype/qwen35_drift_study.py \
    --tokens 4096 --layers 0,8,16,24 --windows 4,16,64 --save-capture drift.pt

# 5) 模型级注入式 A/B 与 teacher forcing
VLLM_ENABLE_V1_MULTIPROCESSING=0 python prototype/qwen35_replay_ab.py --max-tokens 256 --windows 4,8,16
VLLM_ENABLE_V1_MULTIPROCESSING=0 python prototype/qwen35_teacher_forced.py --max-tokens 2048 --windows 4,8,16
```

日志分析脚本在 `tools/`：`analyze_vs_production.py`、`analyze_full_contract.py`、
`analyze_kernel_sweep.py`、`analyze_drift_study.py`、`analyze_replay_ab.py`。

> **未复测项（务必先读）**：本轮发现 full-contract 脚本的 `production_call()` 把
> `mixed_qkv[:, position]` 的切片/`contiguous()` 拷贝写在了计时闭包内，
> 使**分母偏大约 6.5 µs（batch 64）**，原始比值偏高约 4%。本文的处理方式是：
>
> 1. §6.2 的两张 full-contract 表**已用干净口径的生产算子耗时重算**（这是**同一批原始数据的算术重算**，不是重新测量）；
> 2. 代码层面已把切片与 `pos.fill_` 移出闭包、并给 replay kernel 增加外部 `out` 参数，
>    **已提交但尚未复测**（GPU 实例已关闭）；
> 3. core 表（2.21 / 2.15 / 1.99）出自 `bench_gdn_vs_production.py`，该脚本**不存在**这个问题，无需修正。
>
> 因此：**core 数字是干净测量；full-contract 数字是干净分母下的算术修正值**。
> 复测只需：`python prototype/bench_gdn_full_contract.py --batches 64 --window 4,8,16`。

---

## 11. 结论与后续工作

**结论**：在一个已经跑到峰值带宽约 85% 的生产 GDN 算子面前，通过改变循环状态的执行方式
（FP8 checkpoint + 短窗口 replay）可以把单层单步时间降到**做同样工作量时的 1/1.61–1/1.73**，
状态搬运量减少约 2.8×；同时长序列实验给出了清晰的边界——**1 byte/element 的 checkpoint
无法同时换取大容量与长上下文**。

**后续工作（按优先级）**：

1. **误差补偿 checkpoint**：把量化残差以紧凑形式带进 ring（残差链是线性的），
   这是唯一可能改变 √(flush 次数) 累积律的方向。
2. **适用边界扫描**：补 256/512/1024 token，把"128 通过、2048 不通过"之间的边界测出来。
3. **按层自适应格式**：INT8 在层 0/16/24 明显更优、层 8 相反，零字节成本。
4. **上游集成**（超出当前算子范围，可选）：allocator / paged-state / CUDA Graph / 抢占 / 投机解码。

---

## 附录 A：脚本清单与职责

| 脚本 | 职责 |
| --- | --- |
| `gdn_replayssm_fp8_kernel.py` | 全部 Triton kernel（prep / replay / flush / 两段式 / 对照）+ Python wrapper |
| `gdn_replay_fp8_reference.py` | FP32 GDN recurrence 参考实现、量化/反量化、误差指标 |
| `bench_gdn_vs_production.py` | core 口径：replay vs vLLM 生产算子 |
| `bench_gdn_full_contract.py` | full-contract 口径：含 prep，支持合成与真实 capture |
| `bench_gdn_replayssm.py` | tiling/窗口/warp 扫描 + 正确性校验 + 流量核算 |
| `bench_kernel_diag.py` | 占用率、寄存器、纯流量上界诊断 |
| `qwen35_replay_ab.py` | 模型级注入式 A/B（含无量化对照臂） |
| `qwen35_teacher_forced.py` | teacher forcing 数值验收（含两道自检） |
| `qwen35_drift_study.py` | 长序列漂移研究（先录制，可离线反复扫描） |
| `qwen35_capture_raw.py` | 录制真实 decode 原始输入（供 full-contract / drift / TF 使用） |
| `qwen35_state_stats.py` | 真实状态分布与 FP8/INT8 单次重建误差统计 |
| `qwen35_logprob_probe.py` | top-k margin / tie 统计 |
| `qwen35_sampler_probe.py` | 探测采样链上真正被调用的入口（用于 teacher forcing 挂钩） |
| `qwen35_baseline.py` / `qwen35_replay_shadow.py` | 基线冒烟与早期离线 shadow（已被 A/B 取代，保留作记录） |

## 附录 B：关键超参

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| 量化格式 / 粒度 | FP8 E4M3 / vblock（32 value 行 × 全 K） | §5.3 |
| 窗口 L | 4 / 8 / 16（主）/ 32（边界） | 性能与数值的折中，见 §6.2/§6.5 |
| `BLOCK_V`（replay / flush） | 64–128 / 32 | flush 必须 32（scale 布局） |
| `BLOCK_KC`（两段式 apply） | 128（整块） | 实测最快 |
| `num_warps`（replay / prep） | 4 / 1 | 8 warps 明显更慢 |
| 计算精度 | 状态与递推在 fp32 内累积，输出 bf16 | 与生产算子一致 |

## 附录 C：常用日志分析命令

```bash
python tools/analyze_vs_production.py results/vs_production_final.log
python tools/analyze_full_contract.py results/kernel_full_contract_realtrace.log
python tools/analyze_kernel_sweep.py results/kernel_final_cycle_tuned.log
python tools/analyze_drift_study.py results/drift_4085_vblock32_L4L16L64.log
python tools/analyze_replay_ab.py results/ab_256token_none_vs_fp8_4prompts.log
```

## 附录 D：文档分工

| 文件 | 定位 | 读者 |
| --- | --- | --- |
| **REPORT.md（本文）** | 主文档：背景、术语、路线、实现、数据、证据索引、决策、边界、复现 | 拿到项目的任何人 |
| `README.md` | 仓库门面与导航：结论摘要 + 快速上手 | 第一次打开仓库的人 |
| `findings.md` | 按主题归并的结论与证据 | 想深挖某个结论的人 |
| `progress.md` | 按时间的过程日志（含踩坑与修正） | 想知道"当时为什么这么决定"的人 |
| `task_plan.md` | 阶段规划、决策记录、止损条件 | 想接手继续做的人 |
| `results/README.md` | 日志索引（每份日志跑的是什么） | 想核对数据的人 |
| `RESUME.md` | 简历表述（中英双语）与面试追问准备 | 项目作者本人 |
