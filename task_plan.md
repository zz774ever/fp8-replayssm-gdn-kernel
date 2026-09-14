# 项目规划审查

## Goal
评估并细化一个以 RTX 4090 为唯一实验平台、面向 Qwen3.5 GDN decode 的 FP8 recurrent-state + ReplaySSM 项目。

## Phases
- [complete] Phase 1: 建立问题边界与证据清单
- [complete] Phase 2: 核对 vLLM/ReplaySSM 关键接口与假设
- [complete] Phase 3: 评估数值正确性、性能与系统集成风险
- [complete] Phase 4: 重排里程碑、验收标准与止损条件
- [complete] Phase 5: 汇总最终建议
- [complete] Phase 6: 针对 RTX 4090 重写实施计划、实验矩阵和验收标准
- [complete] Phase 7: 连接远端 RTX 4090 服务器，盘点环境与现有代码，确定可执行起点
- [complete] Phase 8: 复审融合方案可行性，拆分容量、数值与性能验收门
- [complete] Phase 9: 实现并运行 GDN full-precision replay 与 FP8 checkpoint 正确性原型
- [in_progress] Phase 10: 根据误差结果选择 scale ABI，开发并基准测试融合 Triton kernel（性能验证已完成，见 Phase 16；剩余：与最终 ABI 定案联动的最后一次复测）
- [pending] Phase 11: 接入 vLLM ReplaySSM 元数据、缓存分配与回归测试
- [complete] Phase 12: 打通“本地 Codex -> 本地 MCP -> SSH -> 远端 apply_patch”工作流，使远端代码改动可审计、可复现
- [complete] Phase 13: 模型级 replay 功能 A/B（真实生成路径注入 replay 输出与状态），含无量化对照臂与 tie 归因
- [pending] Phase 14: 建立 matched-arithmetic 验收口径（teacher-forced 或融合 kernel 内对照），复测 FP8 的边际误差并据此定最终 scale ABI
- [pending] Phase 15: 解决“flush 量化误差 vs 常驻容量”的根本矛盾（error-compensated flush / 更高精度 checkpoint / 缩小适用上下文范围），并重定项目价值主张
- [complete] Phase 16: 独立融合 kernel 性能验证（不依赖 vLLM 集成）：正确性、绝对时延、搬运字节、达成带宽、与 BF16 读-改-写基线的加速比与 break-even 曲线
- [complete] Phase 17: kernel 优化（occupancy 诊断 → 按行 scale gather → 预计算/apply 两段式 → apply 调参 → 按整周期的建模修正），把数值可用窗口 L=16 从 1.20× 提到 2.07×（batch 64，稳态口径）
- [complete] Phase 18: 接上生产算子基线（vLLM FLA `fused_recurrent_gated_delta_rule_packed_decode`，独立调用不集成），得到 replay 对生产实现的加速比：L=16 1.99×、L=4 2.21×（batch 64）

## Errors Encountered
| Error | Attempt | Resolution |
|---|---:|---|
| 仓库无源代码 | 1 | 将任务定位为方案审查，基于上游证据与用户描述分析 |
| SSH `Permission denied (publickey,password)` | 1 | 主机与端口可达；正在核对本机 SSH config、agent 与密钥匹配 |
| fused recurrent 普通 batch 原地调用触发 Triton 空指针/越界 | 1 | 该接口要求 vLLM continuous-batching slot metadata；正确性 oracle 改为显式 FP32 GDN recurrence，生产 kernel 留待 metadata contract 集成测试 |
| tile scale 反量化广播维度错误 | 1 | 修正为 `[B, HV, V_tiles, 1, K_tiles, 1]` 广播布局 |
| `config.toml` 被应用重写，`[mcp_servers.mcp-ssh-apply-patch]` 段落丢失，本地 MCP 不可用 | 1 | 备份后重新写入该段落，并用 stdio JSON-RPC harness 直接拉起配置中的入口做端到端验证 |
| 通过 MCP CLI 调用远端 apply_patch 时，PowerShell 双引号内的 `$(cat ...)` 被本地展开，patch 变成空内容 | 1 | 改用单引号字面量传命令，避免本地插值 |
| 本地分析脚本 `NameError: name 'base' is not defined` | 1 | 变量名写错，改为 `baseline` |
| 宽 tile（block_v=64/128）下 kernel 输出误差从 1.7e-3 跳到 1.2e-2 | 1 | vblock32 是每 32 行一个 scale，宽 tile 跨多个 band 却只取了第一个 scale；改为按行 gather（`offs_v // 32`），三种 tiling 输出逐位一致 |
| 两段式拆分首版没有提速（与 tiled 持平） | 1 | 拆分本身不改 occupancy：apply 仍持有 [BLOCK_V,BLOCK_K] fp32 tile；改为按 K 分块（BLOCK_KC=32）后才在 L≥8/batch≥16 生效 |
| 对生产算子的首轮对比把自研 BF16 kernel 显示为慢 2× | 1 | state cache 的 `.clone()` 与 dtype 转换被写在计时闭包内；挪到闭包外后两者差距回到 5% 以内（batch1 7.58 vs 7.51µs，batch64 163.3 vs 156.3µs） |

## Decisions
- 先做独立 reference/benchmark，再决定是否进入 vLLM 集成。
- 将“FP8 active state”与“ReplaySSM checkpoint compression”拆成两个可独立验收的研究问题。
- 当前 vLLM 主干已有 ReplaySSM 基础设施，但不能假设其直接覆盖 Qwen GDN；先做 GDN 专属 replay 原型与等价性测试。
- RTX 4090 是唯一目标平台：只报告相对 BF16 baseline 的结果，不外推 H100 绝对吞吐。
- 主线采用“BF16 active state + FP8 flush checkpoint”；every-token FP8 active state 作为风险分支。
- conv state 第一版保持 BF16，并单独报告其容量占比；不宣称整个 Mamba state 已完全 FP8 化。
- 明确内存层级：persistent per-sequence state 必须是 FP8 checkpoint；BF16/FP32 reconstructed state 只能是当前调度批次的临时 scratch。若每个 sequence 仍常驻 BF16 active state，则没有主要容量收益。
- page decoupling、FP8 checkpoint、GDN Replay 三项先独立验收，再进入融合；避免把 allocator page inflation、量化误差和 replay 算法性能混为一个问题。
- 远端编辑统一走 `gdn-remote` + `~/.local/bin/apply_patch`（wrapper 调 `/root/.local/bin/codex --codex-run-as-apply-patch`），不再用 heredoc/sed 改远端文件；远端 worktree `/root/vllm-fp8-replayssm` 的改动应保持可被 `git diff` 审计。
- 长序列 greedy token 一致性不作为 FP8 的验收口径：报告 margin 是 bf16 logits 的 0.125 量化值，约 1.5% 的步 top-2 间隔为 0，任何 replay 路径（哪怕 checkpoint 完全精确）都会在这些步上翻转。FP8 的验收改用 arm-to-arm（FP8 vs 同管线精确 replay）的 logprob 距离 + 下游任务质量，并等待 matched-arithmetic（融合 kernel 或 teacher forcing）口径。
- 长序列漂移不饱和，且累积律由量化器而非 flush 频率决定（每次 flush 注入约 1% 相对误差，漂移 ≈ 1% × sqrt(flush 次数)）。因此在 1 byte/element 的 FP8 checkpoint 下，**大容量与长上下文不可兼得**：要 4k token 漂移 ≤15% 需 L≈18（0.72× 字节），要 ≤5% 需 L≈160（ring 已是状态的 2 倍）。
- 当前唯一可辩护的工作点是 L=16 / vblock32：约 0.70× 字节（≈1.4× 容量），4085 步平均状态漂移 ~0.15、输出漂移 ~0.09。任何“2× 容量”叙事在现有方案下不成立，必须先解决 Phase 15 的误差-容量矛盾。
- K 轴 Hadamard 旋转作为降误差手段已否证：递推在该旋转下严格等变（数值验证 1.6e-7），但旋转后 readout 误差显著变差（9/12 组更差，最差输出漂移 0.131→0.534）。不要再把它当作候选 mitigation。
- 算子项目的验收基线是**现有生产实现**，不是自研的同风格 kernel；生产算子的调用契约（`out` 形状 `[B,1,HV,V]`、`use_qk_l2norm_in_kernel`、`ssm_state_indices` 槽位 0 为空块）可以直接在独立进程里复现，因此对比生产算子**不需要集成到 vLLM**。
