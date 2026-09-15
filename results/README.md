# 实验日志索引

本目录是**原始实验日志**（每行一条 JSON，或标准输出），全部由 `prototype/` 下的脚本产生。
环境基线见 `REPORT.md` §10（RTX 4090 / vLLM `0.1.dev20944+g58ad1f3b8` /
Qwen3.5-4B snapshot `851bf6e8…`）。分析命令见 `REPORT.md` 附录 C。

## 日志与结论对照

| 日志文件 | 跑了什么 | 结论 |
| --- | --- | --- |
| `ab_32token_L2L4L8.log` | 模型级 A/B，32 token，L=2/4/8，24 层全部 replay | 32/32 token 与 baseline 一致；写回逐位验证通过 |
| `ab_256token_L4L8L16.log` | 同上拉到 256 token | prompt 0 三个窗口都在第 120 步发散；prompt 1 的 L=4 保持 256 步一致 |
| `ab_256token_none_vs_fp8_3prompts.log` | 加入**无量化对照臂** | 精确臂与 FP8 臂**同一步**发散（120 / 94）→ 发散并非 FP8 造成 |
| `ab_256token_none_vs_fp8_4prompts.log` | 4 prompt + 逐 step margin | 所有 FP8 翻转都发生在 margin 为 0 或 0.125 的步 |
| `baseline_margin_probe.log` | 未修改模型 256 步的 top-5 备选 | 256 步中 5 步 margin 恰为 0；margin 取值恒为 0.125 的整数倍 |
| `drift_2048_vblock_L4L8L16.log` | 录制 2048 步真实输入 + 离线精确链 vs 量化链 | 漂移单调增长、**不饱和**（约 t^0.37） |
| `drift_4085_vblock32_L4L16L64.log` | 4085 步，vblock32，L=4/16/64 | 每次 flush 注入约 1%（与 L 无关）→ 漂移 ≈ 1%×√(flush 次数) |
| `drift_4085_vblock32_rotatedK_L4L16L64.log` | 同上，状态存在 K 轴 Hadamard 旋转基下 | 旋转等变（1e-7）但 readout 漂移更差 → **否证** |
| `drift_4085_format_arms_L16L64.log` | 同 capture，vblock16 与 INT8 | 块大小几乎无影响（约 1e-5）；INT8 在层 0/16/24 更好、层 8 更差 |
| `kernel_sweep_blockv_warps.log` | replay kernel 对 BF16 读-改-写基线，block_v × warps 扫描 | L=4 最高 2.21×；L=16 最高 1.48×（拆分前）；break-even batch 1/4/8/32 |
| `kernel_sweep_v1_with_scale_bug.log` | 同上，但**按行 scale 修正之前** | 保留为证据：宽 tile 误用首个 scale（1.7e-3 → 1.2e-2） |
| `kernel_split_first_pass.log` | 两段式（预计算 + apply）首版，K 未分块 | 正确但无提速（apply 仍持 [128,128] fp32 tile） |
| `kernel_final_sweep_tiled_vs_split.log` | 完整扫描：tiled vs split 逐格对比 | L=16：1.50×（tiled）vs 1.77×（split）；L=32：1.02× vs 1.42× |
| `kernel_cycle_model_first.log` | 首次采用"按整周期摊销"口径 | 稳态口径比"每步满 ring"好 10–15% |
| `kernel_final_cycle_tuned.log` | apply 调参后的最终 core 表 | L=16 **2.07×**、L=32 1.76×（batch 64，对同风格 BF16 基线） |
| `vs_production_first_pass_with_harness_bug.log` | 首次对生产算子，但把 `clone()` 写在计时闭包内 | 保留为证据：使自研基线看起来慢 2× |
| `vs_production_final.log` | 修正 harness 后对**生产算子** | L=4/8/16 = 2.21×/2.15×/1.99×（batch 64，core，稳态） |
| `kernel_full_contract_synthetic.log` | full-contract（含 prep），合成输入 | 原始 1.81/1.79/1.68（batch 64）；因生产算子侧含闭包内拷贝，**修正后 1.73/1.72/1.61** |
| `kernel_full_contract_realtrace.log` | 同上，用 4096 步真实 capture 驱动 | 原始 1.77/1.81/1.69；**修正后 1.70/1.72/1.61**——与合成一致 |
| `kernel_prep_warps_sweep.log` | prep kernel 线程数扫描 | batch 64 约 18 µs，与 warps 无关（launch/延迟受限） |
| `teacher_forced_128_smoke.log` | teacher forcing 装置自检（128 步） | `apparatus_ok` + `forcing_is_live` 均通过；L=4 p95 0.042（通过预算） |
| `teacher_forced_2048_steps.log` | teacher forcing 正式验收（2048 步） | **L=4/8/16 全部未通过** 0.125 预算（p95 0.267/0.186/0.155） |

## 未随仓库提供的产物

| 产物 | 大小 | 说明 |
| --- | --- | --- |
| `qwen35_capture_raw_p0.pt` | 432 MB | 4096 步真实 decode 的原始 `mixed_qkv`/`a`/`b`/初始 state/生产算子输出；由 `prototype/qwen35_capture_raw.py` 重新生成 |
| `qwen35_capture_p0.pt` | 546 MB | 漂移研究用的已准备输入；由 `prototype/qwen35_drift_study.py --save-capture` 重新生成 |

两者都是**可重新生成**的中间产物，因此不入库；脚本与参数见 `REPORT.md` §10。
