# Progress

## Session 1
- 初始化规划文件。
- 确认工作区没有可直接修改的代码。
- 核对 RFC #55196：问题真实存在，且 RFC 建议 page decoupling 先于 state quantization。
- 核对 vLLM Qwen GDN：state shape 包含 recurrent/conv/spec 组成，CUDA fused path 当前仅支持 BF16/FP32 state。
- 核对 vLLM ReplaySSM 搜索结果：已有 ring/checkpoint/flush 基础设施，但主要出现在 Mamba2 selective-state 路径。
- 完成风险评估：page pinning、GDN replay 语义、FP8 scale/conv-state、CUDA Graph 与调度集成是四类主要风险。
- 完成路线重排：FP32 GDN ReplaySSM 等价性 -> flush-only FP8 checkpoint -> every-token FP8（可选）-> vLLM 集成。
- 按 RTX 4090 约束细化计划：主线采用 BF16 active state + FP8 flush checkpoint，conv state 首版保持 BF16，所有性能结果只做同卡相对比较。

## Session 2
- 用户提供远端 GPU 服务器（SSH 别名 `gdn-remote`，非标准端口），并说明 SSH 已配置。仓库内不记录主机地址与端口。
- 开始只读盘点远端 GPU/CUDA、存储、运行时及项目代码，不先修改服务器状态。
- 首次 SSH 连接确认目标可达，但认证失败。
- 检查到 `gdn-remote` 别名映射正确；本机没有 SSH 密钥文件，ssh-agent 也未加载身份，等待补充该主机的认证方式。
- 使用用户提供的临时密码成功登录，未把密码写入文件。
- 完成服务器只读盘点：RTX 4090/SM89、CUDA 13.2、Torch 2.13 cu132、Triton 3.7.1，GPU 空闲，根盘可用约 57 GB。
- `/root/vllm` 工作区干净但位于无关验证分支；确认 Qwen GDN 与现有 Mamba2 ReplaySSM 内核分离。
- 决定使用独立 worktree/分支开展实现，避免污染已有分支；遵循远端 `AGENTS.md` 的 uv/.venv 要求。
- 已创建隔离 worktree `/root/vllm-fp8-replayssm`（分支 `codex/fp8-gdn-replayssm`）并建立 uv/.venv；尚未修改源码。
- 按用户要求暂停实现并复审可行性；识别到常驻/临时 state 生命周期、uniform page inflation、GDN replay 计算膨胀是三项必须先修正的计划问题。
- 实现 `prototype/gdn_replay_fp8_reference.py`，并同步到服务器隔离 worktree 的 `benchmarks/kernels/`。
- 生产 fused op 不能脱离 continuous-batching slot metadata 直接作为普通 batch oracle；改用显式 FP32 GDN recurrence。
- full-precision replay 首轮达到零误差；FP8 head/row scale 首轮输出相对 L2 约 1e-3 以下，tile 广播 bug 已修正待复测。
- 修正随机衰减分布后，在 Qwen 默认几何和三组 seed 上得到稳定误差/容量曲线；L=8 是当前折中候选。
- 新增 `prototype/gdn_replayssm_fp8_kernel.py`：Triton kernel 从 FP8 vblock checkpoint 重放 GDN ring、计算当前输出，并在 flush 时融合重新量化；另含 BF16 step 对照 kernel。
- 首版 kernel 在 RTX 4090 编译通过；L=2 非 flush 初测相对 BF16 prototype 有 1.15x–1.54x 加速，正在补充 flush 摊销后性能。
- 将 non-flush replay 重写为反向低秩 output-only 等价式；完整状态只在 flush 重建。
- 完成 L=2/4/8/16、batch=1/2/4/8/16 的 flush 摊销基准。L=4 当前最均衡；L=16 在所有 batch 均无性能收益。
- 识别到 L=4 的 FP8 累积误差仍约 3.3%，下一步加入同字节 INT8 vblock 对照，判断问题来自 replay 还是 E4M3 精度。
## 2026-09-14: Phase 9 continuation

- Downloaded and ran the official Qwen3.5-4B checkpoint in the isolated remote vLLM worktree.
- Established the eager baseline and captured allocator/page-size behavior.
- Added real recurrent-state statistics with FP8-vblock and INT8-vblock reconstruction error.
- Rejected the first slot-selection method after source inspection showed it could sample stale/padded cache entries.
- Updated instrumentation to follow the active GDN metadata indices; rerun pending.
- Reran in-process and validated distinct active state trajectories for eight representative GDN layers over eight decode steps.
- Retained FP8 as the primary candidate: INT8 lost badly on most real layers because of state outliers.
- Added a real-input replay shadow harness for L=2/4/8; it replays production q/k/v/g/beta from a quantized prefill checkpoint and compares core output/state against unmodified BF16 vLLM.
- Rejected the first shadow output because its one-token error was several-fold. Root cause: capture occurred before causal convolution. Moved recurrent-input capture below convolution to the exact packed recurrent-op call boundary.
- Verified corrected first-token shadow results are in the expected sub-percent to ~1% output-error range rather than multi-x error.
- Added an unquantized one-step oracle control and compact per-layer/window aggregation for the long decode rerun.

## Session 3 (2026-09-14): 远端编辑工作流（本地 MCP + SSH apply_patch）

- 按 `197895/codex-remote-apply-patch` 的 GUIDE.md 先做只读调查，未在获得确认前写入任何配置。
- 发现上次配置的残留证据：固定副本 `~/.codex/mcp-ssh-apply-patch`（`@aiondadotcom/mcp-ssh@1.3.9`）仍存在、`dist/tools.js` 的 apply_patch 描述补丁仍在，但当前 `config.toml` 已不含 `[mcp_servers.mcp-ssh-apply-patch]`；仓库内无 `ssh-mcp` 错误包残留。
- 远端无需改动：`/root/.local/bin/codex`（codex-cli 0.154.0）与 `/root/.local/bin/apply_patch`（绝对路径 wrapper，0755）均已就绪。
- 用户确认后执行两处写入：`~/.ssh/config` 的 `Host gdn-remote` 块内加入 `# @password:` 注释凭据；`~/.codex/config.toml` 追加 MCP 段落。两者均先备份到带时间戳的新文件名，未覆盖历史备份。
- 校验：config.toml 通过 Python `tomllib` 解析且无 BOM；`node --check dist/tools.js` 通过；`Select-String` 确认段落与包路径正确。
- 新增 `tools/mcp_handshake_test.mjs`，按 config.toml 中同一命令行拉起 MCP 服务器并走 stdio JSON-RPC：`tools/list` 返回 7 个工具、描述补丁生效、`listKnownHosts` 将 `gdn-remote` 识别为 password 认证、`runRemoteCommand` 在远端执行成功。
- 远端干净 PATH 验证（guide 阶段五）：`env -i HOME=$HOME PATH=/usr/bin:/bin ~/.local/bin/apply_patch` 在 stdin 模式与参数模式下均返回 `Success. Updated the following files: M test.py`，临时目录已清理。
- 远端只读复检：GPU 空闲（0 MiB / 0%），worktree 仍在 `codex/fp8-gdn-replayssm`，5 个原型脚本未提交，根盘剩余约 48 GB。
- 待用户重启/Reload MCP 后，端到端 patch 将由 `mcp-ssh-apply-patch` 工具直接执行；下一步研究任务仍是模型级 replay 功能 A/B。

## Session 4 (2026-09-14): 模型级 replay 功能 A/B

- 新增 `prototype/qwen35_replay_ab.py`：与只做测量的 shadow 不同，这个 harness 真的把 replay 结果注入正在运行的模型——用 replay 输出覆盖 `core_attn_out`，并把重建状态写回 recurrent cache slot，因此量化误差会沿 decode 链累积，与“常驻 FP8 checkpoint + ring”的真实语义一致。
- 编辑通道：本会话 MCP 尚未随 app 重载，因此新增 `tools/mcp_ssh_cli.mjs`（直接以 config.toml 中同一命令行拉起 mcp-ssh 服务器，支持 run/put/get/hosts）。远端文件仍通过 `~/.local/bin/apply_patch` 落地，上传后用 `md5sum` 与本地比对确认字节一致（`5b894b08…`）。
- 首轮运行（32 token、greedy、24 个 GDN 层全部 replay、FP8 E4M3 vblock 32x128）：
  - 写回验证：`write_propagation_relative_l2` 恰为 0.0，证明下一步读到的就是上一步重建的状态，模型确实跑在量化链上。
  - 误差机制：`state_relative_l2` 与 window 成反比（L=2/4/8 → 1.05%/0.56%/0.28%），即每个 flush 边界注入一次约 2% 偏差，其余步约 1e-7。
  - 结果：三个窗口的 32 个 token 与未改动 BF16 baseline 完全一致，重复 baseline 也完全一致（噪声底干净）；平均 |Δlogprob| 为 5.5e-4/5.6e-4/8.1e-4，最大 0.011/0.011/0.023。
  - 漂移：相对 baseline 的状态漂移从首步约 2%（prefill checkpoint 量化）增长到 7.9%/5.8%/4.3%（L=2/4/8），说明 flush 并不能完全抑制累积。
- 结论仍待长序列验证：32 token 不足以判断 token 级等价，已把 harness 扩展为多 prompt + 可配长度，并启动 2 prompt × 256 token × L=4/8/16 的长跑。

## Session 5 (2026-09-14): 长序列 A/B、无量化对照与 tie 归因

- 长跑（2 prompt × 256 token × L=4/8/16）：prompt0 三个窗口都在第 120 步发散（match≈0.47），prompt1 只有 L=4 保持 256/256 一致（L=8/16 在第 91 步发散）。分歧前的平均 |Δlogprob| 约 5e-3，比 32 token 时大一个数量级。
- 新增对照臂 `--granularities none`（同一套 replay 管线、checkpoint 保持 fp32 精确值）以及 `tools/analyze_replay_ab.py`（本地 arm-to-arm 分析）。
- 对照结果（3 prompt × 256 token × {none, vblock} × L=4/8）：无量化臂与 FP8 臂**在同一步发散**（p0 均为 120，p2 均为 94），分歧前平均 |Δlogprob| 为 4.8e-3（none）vs 6.2e-3（FP8 L4）；重复 baseline 完全一致（max|Δlogprob| 恰为 0）。因此这一轮的发散**不能归因于 FP8**，而是 fp32 参考实现与生产 CUDA kernel 的算术差异。
- arm-to-arm 才是公平指标：p1 的 FP8 L4 与无量化臂 256 token 完全一致；p0/p2 上 FP8 臂与无量化臂共享同一次发散，之后才在 125/113/102 步分开。
- 新增 `prototype/qwen35_logprob_probe.py` 直接探测 baseline 的 top-2 间隔：256 步中恰有 5 步间隔为 0.0（74/120/136/201/229），且没有任何一步落在 (0, 0.125) 区间；所有发散都发生在这些 0 间隔步上。
- 探测数据同时显示报告的 logprob 差值恒为 0.125 的整数倍（如 -1.5365245/-1.9115245/-2.1615245），即 lm_head 的 logits 是 bf16、在此量级分辨率就是 0.125：真实间隔小于该分辨率的 token 会四舍五入成完全相同的 logits，greedy 在这类步上由 tie-break 决定而非由模型偏好决定。
- p0 第 120 步的关键细节：baseline 选 token 13(".")，而四个 replay 臂（none/vblock × L4/L8）全部选 token 318(" (")，随后继续生成完全相同的序列——tie-break 行为是 replay 路径的性质，与是否量化无关。
- 结论与后续：长序列 greedy token 一致性是脆弱指标（约 2% 的步是 bf16 级别 tie），验收必须用 arm-to-arm 或 teacher-forced 口径；fp32 参考实现不能替代生产 kernel —— 这正是后续融合 kernel 集成要解决的问题。已启动 4 prompt × {none, vblock L4/L8} 的逐 step margin 归因跑，用于量化“FP8 额外引入的翻转”有多少、发生在什么 margin 上。

## Session 6 (2026-09-14): 四个 prompt 的翻转归因，Phase 10 口径确定

- 归因跑：4 个 prompt × 256 token × {none, vblock} × {L=4, L=8}，并把每步 top-2 margin 写进日志；新增 `tools/analyze_replay_ab.py` 的 first-flip 归因（不再统计首次发散之后必然出现的连锁不一致）。
- baseline 全部逐 token 可重复（max|Δlogprob| 恰为 0）；报告 margin 恒为 0.125 的整数倍，确认到达采样器的 logits 是 bf16。
- zero-margin 步数：5/3/3/4（各 256 步），合计 15/1024 ≈ 1.5%；最小非零 margin 恰为 0.125（一个 bf16 ulp）。
- 精确 replay 臂（checkpoint 不做任何量化）在 p0/p2/p3 分别于第 120/94/97 步发散，且全部落在 zero-margin 步；其发散前状态漂移仅 0.15–0.17%/步。p1 上 256 步完全不发散。
- FP8 臂同 context 的平均 |Δlogprob| 为 0.0044–0.0138（精确臂 0.0039–0.0056），最大 0.076–0.184，仍远低于 margin 中位数（3.5–4.75）。
- FP8 相对精确臂的首次翻转：p0 L4/L8 = 125；p1 L8 = 91（L4 全程不翻转）；p2 L4 = 113、L8 = 102；p3 L4 = 61、L8 = 97。这 7 次翻转所在 context 的报告 margin 全部是 0.000 或 0.125 —— bf16 分辨率下能表达的两个最小值。
- 结论：长序列 greedy token 翻转由 bf16 tie-break 主导，而非 FP8；FP8 的边际影响是把“本来就在 0/0.125 间隔上的步”翻过去。验收口径因此改为 arm-to-arm logprob 距离 + 下游任务质量，并要求 matched-arithmetic 对照（融合 kernel 或 teacher forcing）才能给出 FP8 的最终数值结论，见 task_plan Phase 14。

## Session 7 (2026-09-14): 长序列漂移是否饱和 —— 结论是不饱和

- 新增 `prototype/qwen35_drift_study.py`：先录真实 decode 的 GDN 输入（q/k/v/g/β 与 prefill 末状态），再离线把精确 fp32 链与 FP8-checkpoint 链放在同一组输入上同步推进，彻底避开 greedy tie-break 混沌。支持 `--save-capture` / `--load-capture`，离线扫描无需重新加载模型。
- 2048 步首轮：漂移单调增长且不饱和。层 0、vblock32、L=4 的状态漂移 0.021(0)→0.098(128)→0.171(512)→0.211(1024)，均值 0.210；L=8 与 L=16 分别降到 0.170 与 0.127。按 128→1024 的 8 倍步长只放大 2.15 倍，拟合指数约 t^0.37。
- 4096 步（实捕 4085 步）窗口扫描：L=4/L=16/L=64 的 4085 步平均状态漂移（层 0）为 0.268/0.175/0.082，平均输出漂移为 0.172/0.177/0.066。
- 关键定量：三种窗口反推的“每次 flush 注入的相对误差”几乎一致（0.84%/1.1%/1.0%），说明累积律由量化器决定，而不是由 flush 频率决定；漂移 ~ 1% × sqrt(flush 次数)。
- 核心矛盾：每层每序列的常驻字节 = 0.5MB checkpoint + 12.2KB×L ring，而 bf16 active state 是 1MB。L=4/16/64 分别是 0.55/0.70/1.28 倍。想同时拿到大容量和低漂移需要 L≈160，此时 ring 已是状态的 2 倍 —— 1 byte/element 的方案做不到两全。
- 尝试的缓解手段（Hadamard 旋转 K 轴）已被否证：该变换下递推严格等变（数值验证 output/state 相对 L2 仅 1.6e-7/2.7e-7），但旋转后 readout 误差显著变差（层 8、L=4 的平均输出漂移从 0.131 升到 0.534），12 组 layer×window 里 9 组状态漂移不优。
- 目前最可辩护的工作点：L=16，约 0.70 倍字节（≈1.4× 容量），4085 步平均状态漂移约 0.15、输出漂移约 0.09，但仍按 sqrt(t) 增长。

## Session 8 (2026-09-14): 完成 ABI 对照表（块大小 / INT8 / 旋转）

- 用同一份 4085 步 capture 做离线扫描（`--load-capture`，不再加载模型），补齐格式维度：
  - 块大小（vblock 32 行 → 16 行）对漂移几乎无影响（0.1745051 vs 0.1745277，差 ~1e-5），确认误差由尾数而非 scale 粒度决定，与早期 row-scale 结论一致。
  - INT8 不是一致更优：层 0/16/24 明显更好（层 0、L=16：0.175→0.069），但层 8 明显更差（0.157→0.249）。层 8 是约束瓶颈，按层选格式是一个零字节成本的优化方向。
  - Hadamard 旋转在 L=16 对层 0 有改善（0.175→0.126），但对层 8 变差且 readout 误差大幅恶化，判定为不可用。
- 全部 7 个 arm 的 4085 步平均状态漂移表已写入 findings.md；日志归档到 results/（含 capture 路径 `/root/qwen35_capture_p0.pt`，后续扫描无需重新跑模型）。
- 结论：0.70× 字节（L=16）下最差层漂移约 0.16–0.18，且仍按 sqrt(t) 增长；要支撑最初“2× 容量”的叙事需要把每次 flush 的注入误差再降约一个数量级，现有 1 byte/element 方案（含旋转、细 block、INT8）都做不到。项目价值主张需要按 Phase 15 重定。

## Session 9 (2026-09-14): 收敛范围到“独立 kernel 性能验证”

- 用户确认不要求 vLLM 集成，交付物改为可独立验证的融合 kernel 性能。新增 `prototype/bench_gdn_replayssm.py`（远端 `benchmarks/kernels/`），用同作者、同风格的 Triton BF16 读-改-写 kernel 作对照，真实 Qwen3.5 GDN 几何（16 key heads / 32 value heads / K=V=128，每层每序列 1MB 状态），batch 1–64、L=4/8/16/32、block_v 32/64/128、num_warps 4/8。
- 每个配置都做正确性校验：非 flush 输出相对 L2 = 1.65e-3–1.70e-3（等于 BF16 输出舍入底噪），flush 状态 = 6.3e-3–6.9e-3（一次独立 FP8 量化）。
- **扫描中发现并修复一个真实 bug**：宽 tile 版本假设每个 program 只有一个 scale，但 vblock32 是每 32 行一个 scale，导致 block_v=64/128 时多出来的 band 用错 scale（误差从 1.7e-3 跳到 1.2e-2）。改为按行 gather scale（`offs_v // 32`）后三种 tiling 输出完全逐位一致，同时也证明 tiling 是纯粹的划分。
- 摊销后每 token 加速比（batch 64）：L=4 2.21×、L=8 1.97×、L=16 1.48×、L=32 1.01×；break-even batch 约 1/4/8/32。batch 1 时 BF16 基线只有 280 GB/s（延迟受限），replay 必然亏。
- 宽 tile 正好在数值上需要的窗口上有用：L=16、batch 64 从 block_v=32 的 1.205 提到 block_v=128 的 1.480，搬运字节比从 2.17× 增到 2.76×。L=4 的最佳 tiling 是 v64。
- 绝对量级：batch 64 时 BF16 状态步单独就要 163µs/层/token，24 个 GDN 层约 3.9ms/步纯状态流量；L=16 replay 约 2.6ms（推算值，非端到端实测）。
- 剩余优化点已定位：replay kernel 的达成带宽（443 GB/s）低于基线（826 GB/s），差距来自顺序 ring 循环；这是把 L=16 从 1.48× 继续推高的主要方向。

## Session 10 (2026-09-14): kernel 优化第二轮（occupancy + 两段式拆分）

- 先做诊断（`benchmarks/kernels/bench_kernel_diag.py`）：各 tiling 都 `n_spills=0`，但 block_v=128 用满 255 寄存器/线程，占用率被限制——纯流式读 FP8 checkpoint 只有 613 GB/s，而 BF16 读-改-写 kernel 能到 826 GB/s。L=16/batch 64 下 checkpoint 流 54.5µs、ring 流 24µs，理论下限约 78µs，而当时 tiled kernel 是 107µs。
- 查了 vLLM 自己的 ReplaySSM 实现（`ops/selective_state_update_replayssm_output_only.py`）：它本来就是"预计算 kernel + 主 kernel"两段式，并在预计算里把整个 ring 当 2D tile 一次性 reduce。方向印证。
- 改动一：宽 tile 下按行 gather scale（`offs_v // VBLOCK`），修掉之前 vblock32 多 band 误用首个 scale 的真实 bug，三种 tiling 输出逐位一致。
- 改动二：`gdn_replay_fp8_split`——预计算 kernel 按 (batch, value head) 算一次 transformed query 与 ring 系数，apply kernel 再把 checkpoint 按 **K 分块**（BLOCK_KC=32）流式读出。K 分块只有在 tq 从显存读取后才可能实现（寄存器里无法切片），这正是拆分带来的直接收益。
- 效果（batch 64）：L=4 2.20→2.22×、L=8 1.97→2.00×、**L=16 1.20→1.77×（+47%）**、**L=32 1.02→1.42×（+40%）**。按窗口自动选变体：batch <16 时 tiled 更优（多一次 launch 不划算），batch ≥16 时 split 更优；L=16 的 break-even batch 从 ~16 降到 ~8，L=32 从 ~32 降到 ~16。
- 组件拆分（L=16/batch 64/block_v=128）：预计算 20.4µs（23%）、apply 约 68µs；预计算是延迟受限（16 步串行、每步两次归约，但只搬 4.3MB），是下一步继续提速的目标。

## Session 11 (2026-09-14): 建模修正 + apply 调参，L=16 达到 2.07×

- **建模修正**：之前摊销口径假设每一步的 ring 都是满长 L，那是上界。真实系统里 ring 在一个窗口内是 1,2,…,L-1 递增，第 L 步才 flush。改成按整周期测量后，per-token 成本 = (Σ_{k=1..L-1} replay(ring=k) + flush(ring=L))/L。benchmark 现在同时输出两套数：`speedup_vs_bf16`（最坏情况）与 `cycle_*_speedup`（稳态）。
- **预计算分组（一个 key head 服务它所有的 value head）没有收益**：k 的重读减半，但总时间只在噪声范围内变化。预计算占 23% 且是延迟受限，省它的流量不划算。
- **apply 调参有收益**：K 分块 32/stages 2 → 88.1µs；64/3 → 83.6µs；**128（整个 K）/2 → 83.0µs（1.96×）**。也就是说收益来自"把递推从流式 kernel 里拆出去"本身，而不是缩小寄存器 tile —— 分块反而更慢。这否证了最初"寄存器压力"的解释，已如实记录。
- 最终按周期的加速比（每格取最优变体）：batch 64 下 L=4 2.33×、L=8 2.25×、**L=16 2.07×**、L=32 1.76×；batch 16 下 L=16 1.55×、L=32 1.23×。L=16 从本轮优化开始时的 1.20× 提到 2.07×。
- 变体自动选择：小 batch 用 tiled（拆分的额外 launch 摊不掉），batch ≥16 用 split。

## Session 12 (2026-09-14): 接上生产算子基线（算子项目的硬要求）

- 确认"算子项目不需要集成，但必须和现有实现比"之后，把 vLLM 生产算子 `fused_recurrent_gated_delta_rule_packed_decode`（FLA，`vllm/third_party/flash_linear_attention`）直接拉进同一个 harness：不加载引擎、不集成，只按真实调用契约构造输入（`mixed_qkv` 打包 `[q|k|v]`、`a`/`b` 原始门控、`initial_state` 就是 `[slots, HV, V, K]` cache、`ssm_state_indices` 槽位 0 为空块、`out` 为 `[B,1,HV,V]`、`use_qk_l2norm_in_kernel=True`）。
- 首次就验证 harness 正确：生产算子输出对 fp32 参考解 1.7e-3（bf16 舍入底噪）。这正是早期 shadow 失败过的调用约定，说明当时是调用方式问题而非数学问题。
- **发现自己 harness 里的一个测量 bug**：第一版把 state cache 的 `.clone()` 和 dtype 转换写在了计时闭包内，导致自研 BF16 kernel 看起来比生产算子慢 2 倍。把这些挪到计时区外后两者差距回到 5% 以内（batch 1：7.58µs vs 7.51µs；batch 64：163.3µs vs 156.3µs），符合"同样搬运量的两个 kernel"的预期。带 bug 的日志一并归档作为证据。
- 最终（按整周期口径）replay 对**生产算子**的加速比：batch 64 下 L=4 2.21×、L=8 2.15×、**L=16 1.99×**；batch 16 下 L=16 1.52×。break-even 约在 batch 4；batch ≤2 时 replay 落后（0.75–0.91×），因为两边都是 launch/延迟受限，replay 还要多读一遍 ring。
- 生产算子本身很强：batch 64 单层 157µs 对应约 815 GB/s，即 4090 峰值的 81%，所以 1.99× 是算法（搬运量）收益，不是测量假象。
- 仍需在报告里声明的口径差异：replay 计时不含每 token 的预处理（q/k L2 归一化、门控计算、ring 追加），生产算子是内联做这些的。量级不大但必须写明。

## Session 13 (2026-09-14): full-contract 口径 + 真实 trace 复核（① ②）

- 新增 `prototype/qwen35_capture_raw.py`：重录 capture，保存**原始** `mixed_qkv`、`a`、`b`、
  初始 state 与生产算子每步输出（之前那份 capture 存的是已归一化的 q/k 与已算好的 g/β，无法用于
  full-contract）。产出 `/root/qwen35_capture_raw_p0.pt`（432MB，4096 步 × 4 层）。
- kernel 侧新增 `_gdn_prep_fused_kernel`（一次 launch 完成 q/k L2 归一化 + 门控 + v 拷贝 + ring 追加），
  语义与生产算子逐项对齐（L2 eps=1e-6、softplus 阈值 20.0、q 最后乘 K^-0.5 的等价形式）。
- 新增 `benchmarks/kernels/bench_gdn_full_contract.py`：两边都从原始输入出发，同时输出
  core 与 full-contract 两个数字。
- 过程中修掉三个自己的 bug：2D 张量多传了一个 stride、参考解只给了 1 个 token 而 ring 有 window 个
  （`run_kernel` 按 `q.shape[1]` 循环）、以及“生产算子一次只走一步”与参考解走整窗口的语义错配
  （改为先把生产状态推进到窗口末尾再比较）。
- 结果（batch 64）：core L=4/8/16 = 2.21×/2.15×/1.99×；**full-contract = 1.81×/1.79×/1.68×**。
  prep 在 batch 64 约 18µs，且与 num_warps 无关（launch/延迟受限）。
- **真实 trace 复核**：用 capture 驱动同一 harness，与合成输入几乎一致（batch 64：1.772/1.806/1.690 vs
  1.805/1.791/1.677），“合成数据过于理想”的质疑排除。

## Session 14 (2026-09-14): teacher forcing 数值验收（③）

- 新增 `prototype/qwen35_teacher_forced.py`：强制 baseline 的 token 序列喂给所有臂，只比较分布。
  实现要点：vLLM 贪心路径返回未修改 logits，`gather_logprobs` 同时给出 top-k 与"被采样 token"的
  logprob，所以只需覆盖 `sampled_token_ids`，记录到的仍是模型给该 token 的真实概率。
- **加了两道自检，第二道救了这个实验**：第一道"关闭 replay 时强制解码必须与 baseline 逐位一致"
  在强制**完全没生效**时也会通过（确定性贪心重跑本来就一致），是假阳性；第二道"故意喂被破坏的
  序列、引擎必须原样输出"抓到了它——探针显示真正被调用的是 `GPUModelRunner.sample`，
  而 `Sampler.sample/forward` 根本没被执行。改用正确挂点后 `forcing_is_live=true`。
- 结果（2048 步全部纳入统计，阈值预注册 p95 |Δlogprob| < 0.125）：
  L=4 mean 0.0529 / p95 **0.2666**；L=8 mean 0.0377 / p95 **0.1863**；L=16 mean 0.0311 / p95 **0.1546**
  —— **三个窗口都未通过**。而 128 token 时 L=4 是通过的（p95 0.042）。
- 关键交叉验证：从 128→2048 token，flush 次数增加 16×，离线漂移律预测扰动增加 4×；实测 p95 从
  0.042 涨到 0.267，同量级。**离线状态链与在线模型分布两条独立测量互相印证。**
- 量级交代：平均扰动 0.031–0.053 nats ≈ 典型贪心 margin（中位数 3.5–4.75）的 1%，token 级行为大体保持，
  但分布已被可测量地改变，不能称为无损替换。
- 结论：当前 1 byte/element 方案的可用区间是**短/中上下文 + 高并发**；长上下文必须靠 Phase 15
  （误差补偿 flush / 两级 checkpoint / 按层选格式 / 更大 L + ring 压缩）。
