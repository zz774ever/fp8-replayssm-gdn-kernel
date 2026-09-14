# Findings

## Initial context
- 工作区只有 `.git`，没有现有实现或测试。
- 用户目标是审查并优化一个面向 Qwen3.5 GDN decode 的 FP8 recurrent-state + ReplaySSM 项目规划。

## Evidence to verify
- vLLM issue/RFC #55196 对 `mamba_cache_dtype` 支持范围和混合模型 page sizing 的描述。
- `fused_recurrent_gated_delta_rule` / `fused_sigmoid_gating_delta_rule_update` 的 state layout、更新公式与测试。
- ReplaySSM 的 flush/checkpoint 语义、量化 roadmap 与现有 vLLM 接入边界。

## RFC #55196 verified (2026-09-14)
- Issue opened Sep 3, 2026. TL;DR reports attention-only FP8 capacity 2.00x, but Falcon-H1 hybrid 1.84x at 32k and 1.00x at short context.
- Root cause is two-part: `get_mamba_state_dtype_from_config` remains BF16 (~1.59 MB page), while `get_uniform_page_size` plus `_align_hybrid_block_size` forces a common page and inflates attention block size.
- RFC states `--mamba-cache-dtype` / `--mamba-ssm-cache-dtype` currently accept only `auto/float32/float16/bfloat16`; no FP8/INT8. Conv-state quantization is explicitly unimplemented.
- RFC recommends decoupled per-group pages first (KV-cache-manager-only), then state quantization as a follow-up gated by perplexity + downstream accuracy. It calls the issue a real efficiency gap, not a correctness bug.
- RFC's numbers are mostly offline cache-capacity calculations; only Falcon-H1 is the hybrid anchor, so per-model validation remains necessary.

## vLLM Qwen GDN source observations
- `qwen_gdn_linear_attn.py` defines `QwenGatedDeltaNetAttention.get_state_shape()` through `MambaStateShapeCalculator.gated_delta_net_state_shape(tp_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel_size, num_spec)`, so the cache is not a single flat `[slot, head, d_k, d_v]` tensor: it includes recurrent state plus convolution state and speculative-decoding variants.
- The class has separate CUDA/Triton/ROCm/CPU paths and packed recurrent decode. CUDA fused decode currently rejects anything except BF16 model, BF16 conv cache, BF16/FP32 recurrent state, K=V=128, supported gating and CC>=8.0. Adding FP8 state therefore requires changing dispatch guards and the low-level op, not only a dtype parser.
- Qwen3.5 uses non-interleaved `[q,k,v,z]` projection layout; Qwen3-Next has a separate interleaved layout. The project should target Qwen3.5 only initially and avoid claiming cross-model compatibility.
- Gating is computed in FP32 (`A_log.exp`, softplus, sigmoid), which suggests dequantized state/update accumulation should remain FP32 or BF16 even if storage is FP8.

## ReplaySSM integration status
- Current vLLM main already contains ReplaySSM plumbing: `CacheConfig` fields (`replayssm_buffer_len`, `use_replayssm`), `MambaAttentionBackend` ring/checkpoint metadata (`is_flush`, scratch buffers, CPU ring origin), `ssu_dispatch` ring trackers, model capability flags, and an end-to-end benchmark script.
- The existing path is not limited to a hypothetical future RFC; it has concrete Triton/FlashInfer dispatch and CUDA-graph warmup concerns. A new FP8 checkpoint format must preserve ring indexing, flush ordering, preemption/resume, speculative decode, and graph-capture static shapes.
- Search results show ReplaySSM kernels under Mamba2 selective-state update and shared tracker infrastructure, while Qwen GDN has its own fused recurrent decode. Reusing the ring/checkpoint contract is plausible, but reusing Mamba2 kernels is not a correctness shortcut.

## Risk assessment and plan changes
- Cache-capacity/page pinning and per-token recurrent-state bandwidth are orthogonal bottlenecks. A ReplaySSM speedup does not by itself fix uniform-page capacity, and FP8 state storage does not by itself remove per-token read/write traffic unless the active path is changed.
- The actual Qwen GDN state contract includes convolution state and speculative variants. Treating only the recurrent matrix as FP8 would produce an incomplete memory/capacity claim; conv state should either remain BF16 as an explicit v0 scope or be separately quantified.
- The safest research order is: (1) reproduce baseline and page math, (2) implement GDN ReplaySSM in full precision and prove equivalence, (3) add FP8 checkpoint-only compression at flush, (4) optionally evaluate every-token FP8 active-state storage, (5) integrate with vLLM pools/flags. This prevents a failed quantizer from being confused with a failed replay algorithm.
- Quantization needs a defined scale contract: granularity, scale dtype/storage, amax policy, saturation handling, and whether scales are per-head/tile or per-checkpoint. Without this, "E4M3 + block scale" is not reproducible and metadata can erase the expected 2x gain.
- Existing Qwen GDN CUDA dispatch explicitly guards BF16 conv and BF16/FP32 recurrent state; an FP8 path must modify parser, state allocator, kernel dispatch, CUDA-graph warmup, and fallback behavior.

## RTX 4090-specific constraints
- RTX 4090 is Ada SM89 with 24 GB VRAM and roughly 1 TB/s peak memory bandwidth. It is suitable for Triton correctness, storage-format experiments, CUDA Graph behavior and relative speedups, but not for reproducing H100 absolute throughput or batch=128 long-context settings.
- The proposed fused path does not require FP8 matrix multiplication: state payload can be stored as FP8/uint8, dequantized to BF16/FP32 for recurrence, then requantized. This avoids depending on Hopper-only FP8 Tensor Core behavior.
- Use BF16 model weights where possible, reserve VRAM for allocator/workspace, and begin with Qwen3.5-4B or a reduced synthetic GDN configuration. OOM is a configuration failure, not evidence that the kernel is invalid.
- Report achieved bandwidth and speedup relative to the same 4090 BF16 baseline. Do not compare absolute tokens/s with RFC H100 numbers.

## Remote execution environment
- The remote GPU host (SSH alias `gdn-remote`, non-standard port) responds to SSH, but the first non-interactive login as `root` failed with `Permission denied (publickey,password)`.
- Default OpenSSH resolution uses the standard identity filenames and finds no host-specific alias/config through the explicit IP invocation; authentication setup still needs verification.
- The local SSH config does contain the alias `gdn-remote` (host, port and user are kept out of this repository).
- The local `.ssh` directory has no private/public key files, and `ssh-add -l` reports no reachable agent. The alias therefore configures routing only, not authentication.
- Password authentication succeeded in an interactive session. Remote host is Ubuntu 22.04, RTX 4090 24 GB (SM89), NVIDIA driver 595.80, CUDA toolkit 13.2, 92 GiB RAM, and 57 GiB free root storage; GPU is idle.
- `/root/vllm` is a clean git worktree on `validate-smollm3-batch-invariance` at `627083054`, with a local editable vLLM import and Torch 2.13.0+cu132 / Triton 3.7.1 / FlashInfer 0.6.18.
- Repository history includes ReplaySSM commits, but current files/tests show the production ReplaySSM output-only kernel is Mamba2-specific. Qwen GDN uses `fused_sigmoid_gating_delta_rule_update` or a separate packed recurrent decode path.
- The Qwen GDN fused CUDA dispatch accepts only BF16 convolution cache and BF16/FP32 recurrent state. This confirms FP8 integration cannot be achieved through config parsing alone.
- Remote repository `AGENTS.md` requires all Python work through `uv` and `.venv/bin/python`; current shell has no `uv` command and no repo `.venv`.

## Feasibility review after remote inspection
- Overall feasibility is conditional rather than guaranteed: FP8 checkpoint storage is straightforward in principle, but profitable GDN replay is the research risk. GDN's rank-one state update depends on the previous state, so replaying a window performs multiple full state transforms; Mamba2 ReplaySSM speedups do not transfer automatically.
- The plan's phrase “BF16 active state + FP8 flush checkpoint” needs a memory-lifetime definition. Capacity improves only if BF16/FP32 reconstructed state is scratch for the currently scheduled batch while the per-sequence persistent state is FP8 checkpoint + ring inputs. A persistent BF16 state per sequence would retain the dominant allocation.
- Uniform hybrid page sizing can hide tensor-level savings. Actual allocatable sequence count and GPU allocator bytes must be measured after per-group/page decoupling; theoretical recurrent-state byte reduction is not a serving-capacity result.
- The work should be split into three independently gated tracks: allocator/page capacity, FP8 checkpoint numerics/conversion, and full-precision GDN replay. Fuse only after each wins independently.
- FP8 scale granularity must be part of the ABI. Candidate sweeps should include per-head, per-row, and 32x32/per-tile FP16 scales; compare metadata, amax/reduction cost, saturation, and long-horizon error. E4M3 format semantics must be explicit.
- Benchmark the break-even surface over batch size, replay length, and state geometry. Required metrics are p50/p95 decode latency, tokens/s, actual allocated bytes/maximum concurrency, kernel time, achieved bandwidth, and scratch memory—not just isolated kernel throughput.
- Correctness must cover flush boundaries, mixed sequence positions, continuous-batching state indices, preemption/resume, CUDA graph replay, and speculative decode before vLLM integration is considered complete.

## GDN replay correctness prototype
- A first attempt to use the production fused recurrent op as a generic batch oracle failed because its in-place path assumes valid continuous-batching `ssm_state_indices`; replaying it outside that metadata contract produced an empty-pointer compile error and later an illegal write. This is an integration-contract finding, not evidence against replay math.
- The oracle was separated into an explicit FP32 implementation of the exact GDN update. Full-precision replay then matched token-wise baseline output and final state with zero measured error for the first window sweep.
- On a small SM89 run (`B=1`, `HV=2`, `K=V=128`, 8 tokens, window 2), FP8 E4M3 head/row checkpoint scales produced output relative-L2 errors about `8.5e-4` and `7.2e-4`; persistent checkpoint+ring storage was about 53–54% of BF16 active-state bytes.
- The first tile-scale dequantization run exposed a prototype broadcasting bug; it has been corrected before interpreting tile results.
- With realistic Qwen-style decay and actual default geometry (`H=16`, `HV=32`, `K=V=128`), head-scale FP8 output relative-L2 was stable across three seeds: about 3.3% / 2.3% / 1.6% for replay windows 4 / 8 / 16. Persistent bytes were about 55% / 59% / 69% of BF16 active state respectively.
- Per-row scale improved relative error only slightly (~3%) while adding far more reduction/metadata work. A kernel-aligned `vblock` scale over `[32 value rows, full K=128]` matched head-scale error, added only ~0.04% checkpoint metadata, and can be computed independently by each Triton program without cross-program synchronization.
- The first Triton FP8 replay kernel compiled and ran on RTX 4090. For window 2, non-flush replay measured 1.15x / 1.27x / 1.20x / 1.54x faster than the prototype BF16 read-update-write kernel at batch 1/2/4/8. These are provisional because flush quantization cost was not yet amortized.
- Kernel output versus FP32 reference had ~1.6e-3 relative L2 (consistent with BF16 output rounding). Independently quantized flush states differed by ~6.7e-3 relative L2; this needs decomposition before setting a correctness tolerance.
- The naive replay kernel performed `O(L*V*K)` state transforms on every token. Rewriting non-flush output with the exact backward low-rank identity reduces it to one checkpoint matrix-vector plus `O(L*(K+V))`; full state reconstruction remains only on flush.
- After amortizing one flush per window, the optimized prototype speedups versus its BF16 read-update-write kernel were: L=2: 1.15x–1.90x across batch 1–16; L=4: 1.06x–1.67x; L=8: 0.87x–1.37x; L=16: 0.62x–0.99x. Thus L=4 is the current performance/capacity candidate, while L>=8 only pays at high batch or not at all.
- The remaining L-dependent overhead comes partly from recomputing the backward transformed query independently for four value-row tiles. A two-stage precompute (one transformed query + coefficients per request/value-head, then tiled output) is a possible next optimization, but it adds a launch and should be justified by profiling.
- L=4 still has ~3.3% output relative-L2 in the synthetic long-horizon FP8 experiment, so numerical quality rather than raw performance is now the binding risk. Same-byte INT8 block quantization should be tested as a control because E4M3's limited mantissa may be the source.
## 2026-09-14: Qwen3.5-4B baseline and real-state instrumentation

- Official Qwen3.5-4B has 32 layers with full attention every fourth layer, so 24 layers use GDN. Its recurrent state geometry is H=16, HV=32, K=V=128.
- vLLM eager baseline at max length 512 reported about 8.61 GiB model weights, 9.23 GiB KV cache, 41,837-token GPU KV capacity, and 23.32 tok/s for a batch-1 eight-token decode. The allocator enlarged the attention page to 528 tokens to accommodate Mamba state, with 0.76% padding.
- A real-state probe initially chose the cache slot with the largest amax. This is invalid because it can select stale or scheduler-padding state. `GDNAttentionMetadata` provides exact per-request slots by layer prefix; `NULL_BLOCK_ID` is 0.
- The corrected probe now records only unique, nonzero slots from `non_spec_state_indices_tensor` and `spec_state_indices_tensor`.
- The capture hook must run with `VLLM_ENABLE_V1_MULTIPROCESSING=0`; otherwise the engine spawn does not inherit the runtime monkeypatch and produces no records.
- Corrected real-state results are layer-distinct and evolve smoothly across decode. FP8-vblock reconstruction relative-L2 was about 1.7%--2.6%; same-byte INT8 was usually about 4%--5.8% (layer 0 was the main near-tie). Real states are strongly outlier-heavy, so FP8's dynamic range matters more than INT8's extra mantissa precision.
- The first real replay shadow incorrectly interpreted `_forward_core_decode_non_spec`'s `mixed_qkv` argument as recurrent input. It is pre-convolution input, causing multi-x errors even on the first token. The hook is being moved to `fused_recurrent_gated_delta_rule_packed_decode`, where `mixed_qkv` is post-convolution; an outer method context preserves the layer identity.
- After moving capture below convolution, first-token FP8 replay output relative-L2 was about 0.05%--1.13% across representative layers and state error was about 1.9%--2.27%. This is consistent with the static reconstruction study and validates the logical tensor mapping. A longer run is needed to measure post-flush accumulation.
- The isolated worktree needed symlinks to the already-built vLLM and flash-attention extension artifacts before the local source checkout could import successfully.

## Remote-editing workflow (local MCP + SSH apply_patch)

- The intended chain is: local Codex -> local MCP `mcp-ssh-apply-patch` -> `@aiondadotcom/mcp-ssh` -> SSH `gdn-remote` -> `/root/.local/bin/apply_patch` -> `/root/.local/bin/codex --codex-run-as-apply-patch`.
- `mcp-ssh` authenticates by reading a `# @password:<pw>` comment inside a `Host` block of `~/.ssh/config`; it feeds the value through a temporary `SSH_ASKPASS` helper (`SSH_ASKPASS_REQUIRE=force`) and never exposes the password to the model. On POSIX it refuses config files whose mode is not 600; on Windows that check is skipped.
- Earlier provisioning had been done but was partially lost: the fixed npm copy `~/.codex/mcp-ssh-apply-patch` and its patched tool descriptions survived, while the `config.toml` MCP section did not. Codex rewrites `config.toml` (plugins/hooks/providers), so this section can be dropped again; re-add and re-verify rather than assuming it persists.
- `~/.ssh/config` mixes CRLF and LF line endings and is ASCII-only; edits must preserve content byte-for-byte and must not introduce a BOM.
- The remote side needs only an executable Codex plus the wrapper: remote `apply_patch` is 386 bytes and calls the absolute path, so the non-login SSH shell's PATH is irrelevant. Remote `codex --version` is 0.154.0 and the wrapper returns `Success. Updated the following files: M test.py` in both stdin and argument modes under `env -i PATH=/usr/bin:/bin`.
- `listKnownHosts` also reports entries parsed from `known_hosts` (alias `undefined`); those are pre-existing and not hosts that can be selected by alias.

## Model-level replay A/B (first run, 2026-09-14)

- `benchmarks/qwen35_replay_ab.py` is the first harness that *drives* the real model with replay instead of only measuring it: the replay output replaces `core_attn_out` and the reconstructed state is copied back into the recurrent cache slot. Every GDN layer is replayed; only the metric/snapshot layers are sampled.
- The write-back is verified: `write_propagation_relative_l2` is exactly `0.0`, i.e. the value the next decode step reads from the cache is bit-identical to the previously reconstructed state. This confirms (a) the hook location is the live recurrent cache and (b) the model's decode chain really runs on the quantized state, which is what makes the A/B meaningful.
- Mechanism check: `state_relative_l2` (replay result vs. one production step from the same starting state) scales as `1/window` -- ~1.05% / 0.56% / 0.28% for L=2/4/8, implying a single ~2% deviation per flush boundary and ~1e-7 elsewhere. The error source is therefore flush-time requantization, not a formula or layout mismatch.
- 32-token greedy decode, batch 1, all 24 GDN layers replayed, FP8 E4M3 vblock (32 value rows x full K=128): token stream was identical to the unmodified BF16 baseline for L=2, L=4 and L=8, and the repeated baseline was also token-identical (noise floor clean).
- Logprob drift stays small but grows with the window: mean |delta logprob| 5.5e-4 / 5.6e-4 / 8.1e-4 and max |delta logprob| 0.011 / 0.011 / 0.023 for L=2/4/8.
- State drift vs. the BF16 baseline is not washed out by flush: it starts at ~2% (the prefill-checkpoint quantization, which is incurred once per window) and grows to 7.9% / 5.8% / 4.3% (L=2/4/8) after 32 decode steps. Larger windows reduce but do not eliminate accumulation.
- Because a shorter window injects the same per-flush error more often, drift ordering (L=2 worst, L=8 best) matches the earlier static study; 32 tokens is however far too short to conclude anything about token-level equivalence.

## Long-horizon A/B and the quantization-free control (2026-09-14)

- 256-token greedy runs (prompts: recurrent-state explanation, Fibonacci task, bf16-vs-fp8 trade-offs, streaming-buffer description) showed token divergence, which is why the control arm matters. Baseline repeated runs are *exactly* identical (tokens equal, `max|delta logprob| == 0.0`) for every prompt, so the pipeline itself is deterministic and any divergence is caused by the replay arm.
- Control arm `--granularities none` runs the identical replay plumbing (same hook, same ring, same fp32 reference kernel, same state write-back) but keeps the checkpoint in exact fp32. On prompt 0 the FP8 arms (L=4/L=8/L=16) and the *exact* arm all diverge from the baseline at the same step 120; on prompt 2 the exact and FP8 arms all diverge at step 94. So at this horizon the divergence is **not** caused by FP8: the fp32-reference-vs-production-CUDA-kernel arithmetic difference is already enough.
- That arithmetic floor is measured: over the pre-divergence prefix the exact arm's mean `|delta logprob|` is 4.8e-3 / 5.6e-3 / 3.9e-3 for prompts 0/1/2, versus 6.2e-3 / 1.2e-2 / 1.4e-2 for FP8 L=4. The single-step fp32 oracle was previously measured at ~1.6e-3 output relative L2, which is the floor feeding these numbers.
- Arm-to-arm comparison (FP8 vs the exact arm) is the only fair FP8 metric, and it is much better than the baseline comparison: prompt 1 L=4 was token-identical for all 256 tokens, and on prompts 0/2 the FP8 arm tracked the exact arm through the shared divergence (first FP8-vs-exact difference at steps 125 / 113 / 102) rather than at the divergence itself.
- Root cause of the flips: greedy decisions land on *reported* top-2 ties. A dedicated probe of the unmodified model over 256 steps found exactly 5 steps with `top1 - top2 == 0.0` (steps 74, 120, 136, 201, 229 for prompt 0) and no step with `0 < margin < 0.125`. Every observed divergence across arms and prompts occurred at such a zero-margin step.
- The reported logprob differences are always multiples of 0.125 (e.g. -1.5365245 / -1.9115245 / -2.1615245), i.e. the lm_head logits are bf16, whose resolution at these magnitudes is 0.125. Tokens whose true gap is below that resolution round to identical logits, so the greedy choice there is decided by tie-breaking rather than by the learned preference.
- Decisive detail at prompt 0 step 120: baseline picks token 13 ("."), while *all four* replay arms (exact L=4/L=8 and FP8 L=4/L=8) pick token 318 (" (") and then continue identically. The tie-breaking outcome is a property of the replay path, identical with and without quantization.
- Consequences for the project: (1) long-horizon greedy token identity is a fragile acceptance metric, because ~2% of steps are bf16-level ties; (2) the acceptance criterion must be arm-to-arm (FP8 vs exact replay) or teacher-forced, and (3) the fp32 reference implementation is not a substitute for the production kernel when the metric is token identity -- this is precisely what the fused kernel integration must fix.

## Flip attribution over 4 prompts x 256 tokens (2026-09-14)

- Setup: 4 prompts, 256 greedy tokens each, arms = exact-replay (L=4/L=8) and FP8-vblock (L=4/L=8), plus a repeated baseline. All baselines repeated exactly (`max|delta logprob| == 0.0`), so every difference below is caused by the replay arm.
- Reported margins are quantized to multiples of 0.125 (0, 0.125, 0.25, 0.375, ...), confirming that the lm_head logits reaching the sampler are bf16. Zero-margin steps per prompt: 5/256, 3/256, 3/256, 4/256 -> 15/1024 (~1.5%); the smallest non-zero reported margin is exactly one bf16 ulp (0.125).
- The exact-replay arm (no quantization at all) diverges from the unmodified baseline at a zero-margin step whenever it diverges: steps 120 (p0), 94 (p2), 97 (p3); on p1 it never diverges in 256 steps. Its state drift at the last pre-divergence step is only 0.15-0.17%.
- FP8 replay perturbs the same-context logprob by mean 0.0044-0.0138 and max 0.076-0.184, versus 0.0039-0.0056 for the exact arm. So FP8 roughly doubles-to-triples the arithmetic floor, but stays far below the median margin (3.5-4.75).
- First FP8-vs-exact flip per arm: p0 L4/L8 at step 125, p1 L8 at 91 (p1 L4 never flips), p2 L4 at 113 / L8 at 102, p3 L4 at 61 / L8 at 97. In every one of those 7 events the context's reported top-2 margin was either 0.000 or 0.125 -- the two smallest values the bf16 logits can even express.
- Pre-divergence state drift for FP8 arms is 5.9-10.8% relative to the baseline state at the same step, versus 0.15-0.17% for the exact arm; that drift is what produces the ~5e-3 logprob perturbation, while token flips stay confined to bf16 tie contexts.
- Net conclusion: greedy token identity over hundreds of steps is dominated by bf16 tie-breaking, not by FP8. The project's acceptance metric must be arm-to-arm (FP8 vs matched-arithmetic replay) logprob distance plus end-task quality, and a matched-arithmetic arm requires the fused kernel (or teacher forcing), not the fp32 reference.

## Long-horizon drift does not saturate (2048 real decode steps, 2026-09-14)

- `benchmarks/qwen35_drift_study.py` captures the exact recurrence inputs of a real 2047-step decode for layers 0/8/16/24 and then runs the exact fp32 chain and the FP8-checkpoint chain in lock-step over the same inputs, so quantization is the only difference and no sampling/tie-break confound remains.
- State drift grows monotonically and does **not** saturate. Layer 0, vblock(32 rows), L=4: 0.021 at step 0, 0.075 at 64, 0.098 at 128, 0.128 at 256, 0.171 at 512, 0.211 at 1024 (mean over the run 0.210, max 0.299). L=8: 0.021/0.056/0.070/0.089/0.122/0.169. L=16: 0.021/0.042/0.052/0.066/0.093/0.130.
- The growth is sub-linear but persistent: for L=4 layer 0 the drift multiplies by ~2.15 while the sequence length multiplies by 8 (128 -> 1024 steps), i.e. roughly `t^0.37`. Extrapolating that exponent to 8k tokens gives state drift around 0.4-0.6, which is far outside anything tested at the 256-token scale.
- Mechanism is consistent with repeated injection plus partial contraction: each flush adds a fresh relative error (~2% of the current state) and the number of flushes grows linearly with sequence length, while the recurrence contracts only part of it. Larger windows help directly because they inject less often: L=16 roughly halves the drift of L=4 at every probe point.
- Layer sensitivity is real but bounded: over the four sampled layers the mean drift at L=4 ranges from 0.198 (layer 16) to 0.265 (layer 8); layer ordering is stable across windows.
- Output drift (per-layer core output, relative L2) is far smaller than state drift and much flatter: mean 0.14 / 0.155 / 0.14 for layer 0 at L=4/8/16, and 0.113 / 0.094 / 0.075 for layer 8. The model absorbs most of the state error, which is why the model-level logprob deltas stayed at ~5e-3.
- Consequence for the plan: "L=4 + vblock" is a 256-token operating point, not a long-context one. A long-context claim needs either a much larger flush window, a better quantizer, or an error-compensating flush; this is now the top-priority design question rather than a detail.

## Window/format Pareto over 4085 steps (2026-09-14)

Same capture (4085 real decode steps, layers 0/8/16/24), vblock32 FP8, three flush windows. Persistent bytes per sequence per layer are `0.5 MB checkpoint + 12.2 KB x L ring`, i.e. L=4 -> 0.55x, L=16 -> 0.70x, L=64 -> 1.28x of the 1 MB bf16 active state.

| window | bytes vs bf16 state | mean state drift @4085 (layer 0 / 8 / 16 / 24) | mean output drift (layer 0 / 8 / 16 / 24) |
| --- | --- | --- | --- |
| L=4 | 0.55x | 0.268 / 0.314 / 0.253 / 0.290 | 0.172 / 0.131 / 0.113 / 0.061 |
| L=16 | 0.70x | 0.175 / 0.157 / 0.134 / 0.149 | 0.177 / 0.091 / 0.074 / 0.032 |
| L=64 | 1.28x | 0.082 / 0.070 / 0.067 / 0.072 | 0.066 / 0.044 / 0.035 / 0.015 |

- The per-flush error is remarkably constant across windows. Taking drift ~ e x sqrt(number_of_flushes) and 4085 steps gives e = 0.0084 (L=4, 1021 flushes), 0.011 (L=16, 255 flushes), 0.010 (L=64, 64 flushes), i.e. **~1% injected per flush, regardless of window**. That means the accumulation law is set by the quantizer, not by the schedule.
- Solving for a drift target exposes the central tension. To hold drift <= 5% at 4k tokens you need L ~ 160, whose ring (1.95 MB) is already twice the state it is supposed to replace. To hold 10% you need L ~ 40 and 0.99x bytes; to hold 15% you need L ~ 18 and 0.72x bytes; to hold 20% you need L ~ 10 and 0.62x bytes.
- So with a 1 byte/element FP8 checkpoint the design cannot simultaneously deliver a large capacity win and bounded drift. The best defensible operating point today is around L=16 (0.70x bytes, i.e. ~1.4x capacity) with mean state drift ~0.15 at 4k tokens, and that drift is still growing as sqrt(t).
- Hadamard rotation of the key axis was tested as a mitigation and *rejected*. The recurrence is exactly equivariant under an orthogonal rotation of K (the decay is a scalar per value head), which was verified numerically: rotated vs plain output/state relative L2 = 1.6e-7 / 2.7e-7. But quantizing in the rotated basis makes the readout error worse, not better: mean output drift at L=4 layer 8 rose from 0.131 to 0.534, and mean state drift was equal or worse for 9 of 12 layer/window combinations. Rotation spread the state energy evenly across K, which destroys the sparsity that the per-row scale was exploiting.
- The honest reading is that the capacity claim (2x) is not reachable with this scheme; a long-context claim needs either ~16 bits per element, an error-compensated flush, or a much smaller per-flush error floor.

## Complete 4085-step ABI table (mean state drift, layers 0/8/16/24)

All arms replay the identical captured inputs; only the checkpoint format/window differ. Bytes are per sequence per layer relative to the 1 MB bf16 active state (checkpoint + 12.2 KB x L ring).

| arm | bytes vs bf16 | layer 0 | layer 8 | layer 16 | layer 24 | worst layer |
| --- | --- | --- | --- | --- | --- | --- |
| FP8 vblock32 L=4 | 0.55x | 0.268 | 0.314 | 0.253 | 0.290 | 0.314 |
| FP8 vblock32 L=16 | 0.70x | 0.175 | 0.157 | 0.134 | 0.149 | 0.175 |
| FP8 vblock16 L=16 | 0.70x | 0.175 | 0.154 | 0.134 | 0.147 | 0.175 |
| FP8 vblock32 L=16, Hadamard K | 0.70x | 0.126 | 0.170 | 0.153 | 0.159 | 0.170 |
| INT8 vblock32 L=16 | 0.70x | 0.069 | 0.249 | 0.159 | 0.222 | 0.249 |
| FP8 vblock32 L=64 | 1.28x | 0.082 | 0.070 | 0.067 | 0.072 | 0.082 |
| INT8 vblock32 L=64 | 1.28x | 0.039 | 0.160 | 0.106 | 0.132 | 0.160 |

- Halving the block size (32 rows -> 16 rows) changes the drift by ~1e-5 relative (0.1745051 vs 0.1745277): the error is mantissa-limited, not scale-limited. This reproduces the earlier per-row-scale result at long horizon.
- INT8 is not uniformly better. It wins clearly on layers 0/16/24 (L=16: 0.175 -> 0.069 on layer 0, 0.134 -> 0.159 on layer 16, 0.149 -> 0.222 on layer 24) but is much worse on layer 8 (0.157 -> 0.249). Per-layer format selection is therefore a real (byte-neutral) knob, and layer 8 is the binding constraint.
- Hadamard rotation at L=16 helps layer 0 (0.175 -> 0.126) while hurting layer 8 (0.157 -> 0.170) and degrading the readout badly, so it is not a mitigation.
- Best worst-layer point at 0.70x bytes is ~0.16-0.18 state drift at 4k tokens; reaching the earlier "2x capacity" narrative would need drift to be ~10x smaller, which no tested 1-byte format provides.

## Kernel performance validation (RTX 4090, real GDN geometry, 2026-09-14)

- `benchmarks/kernels/bench_gdn_replayssm.py` benchmarks the fused replay kernel against a same-author, same-style Triton BF16 read-update-write step kernel. Both compute the identical one-token GDN output and both write back the state; the only difference is where the state lives. Geometry is the real Qwen3.5 one: 16 key heads, 32 value heads, K=V=128, i.e. 1 MB state per layer per sequence.
- Correctness is checked per window and tiling against the fp32 reference: non-flush output relative L2 = 1.65e-3 to 1.70e-3, flush state relative L2 = 6.3e-3 to 6.9e-3, matching the BF16 output-rounding floor plus one independent FP8 quantization.
- **Bug found and fixed during the sweep**: the wide-tile variant assumed one scale per program, but the vblock32 layout stores one scale per 32 value rows. With `block_v` 64 or 128 the dequantization silently used the wrong scale for the extra bands (output error jumped from 1.7e-3 to 1.2e-2). Gathering the scale per row (`offs_v // 32`) made all tilings bit-identical, which also proves the tiling is a pure partition.
- Amortized per-token speedup vs the BF16 step kernel (best tiling per cell; `v32/v64/v128` = value rows per program):

| batch | L=4 | L=8 | L=16 | L=32 |
| --- | --- | --- | --- | --- |
| 1 | 1.08 | 0.86 | 0.62 | 0.42 |
| 2 | 1.17 | 0.95 | 0.72 | 0.48 |
| 4 | 1.21 | 1.02 | 0.81 | 0.53 |
| 8 | 1.36 | 1.11 | 1.02 | 0.76 |
| 16 | 1.79 | 1.48 | 1.19 | 0.88 |
| 32 | 2.02 | 1.68 | 1.37 | 1.01 |
| 64 | 2.21 | 1.97 | 1.48 | 1.01 |

- Break-even batch is ~1 for L=4, ~4 for L=8, ~8 for L=16 and ~32 for L=32. Below that the BF16 baseline is latency-bound (batch 1: 7.0-7.5 us for 2 MB of traffic = ~280 GB/s) and the replay kernel's extra loop can only lose.
- Wider tiles matter exactly where the numerics need them: at L=16, batch 64, going from `block_v=32` to `block_v=128` moves the speedup from 1.205 to 1.480 by cutting the per-tile redundancy of the transformed-query chain (traffic ratio 2.17x -> 2.76x). At L=4 the best tiling is `v64` (2.21x at batch 64).
- The replay kernel is *less* bandwidth-efficient than the baseline (443 GB/s vs 826 GB/s at L=16/batch 64) but moves 2.76x less data, which is where the win comes from. The residual gap is the sequential ring loop, and it is the main remaining optimization target.
- Absolute scale at batch 64: the BF16 state step alone costs ~163 us per layer per token. Over the 24 GDN layers that is ~3.9 ms of pure state traffic per decode step, versus ~2.6 ms with the L=16 replay kernel (projected, not measured end to end).

## Kernel optimization round: occupancy, then a two-stage split (2026-09-14)

Diagnosis first (`benchmarks/kernels/bench_kernel_diag.py`):

- No register spills at any tiling (`n_spills == 0`), but `block_v=128` uses the full 255 registers per thread, which caps occupancy. A stripped kernel that only streams the FP8 checkpoint reaches 613 GB/s, versus 826 GB/s for the BF16 read-update-write kernel: the wide tile is occupancy-limited, not instruction-limited.
- The ring stream alone costs 24 us and the checkpoint stream 54.5 us at batch 64 / L=16, so a kernel that moved those bytes at BF16-kernel efficiency would run in ~78 us; the tiled kernel took 107 us.

Prior art: vLLM's own ReplaySSM implementation (`ops/selective_state_update_replayssm_output_only.py`) already uses a **precompute kernel + main kernel** structure and loads the whole ring as a 2-D tile in the precompute, which is the same decomposition chosen here for a different reason (register pressure instead of reduction structure).

Two changes, both validated against the fp32 reference:

1. **Per-row scale gather.** The wide tile spans several vblock32 scale bands but originally used only the first band's scale (silent 7x error growth). Gathering `offs_v // VBLOCK` made every tiling bit-identical.
2. **Two-stage split** (`gdn_replay_fp8_split`): a precompute kernel computes the transformed query and the ring coefficients once per (batch, value head), and an apply kernel then streams the checkpoint in **key chunks** (`BLOCK_KC=32`), which is only possible because tq is now read from memory rather than held in registers.

Effect (speedup vs BF16 step, best variant per cell, RTX 4090, batch 64):

| window | before this round | after | change |
| --- | --- | --- | --- |
| L=4 | 2.20x | 2.22x | flat (tiled stays best) |
| L=8 | 1.97x | 2.00x | flat |
| L=16 | 1.20x | **1.77x** | +47% |
| L=32 | 1.02x | **1.42x** | +40% |

- The split is chosen automatically by window: below batch ~16 the extra launch costs more than it saves, so the tiled kernel wins at small batch; from batch ~16 up the split wins, and for L=32 it wins everywhere above batch 8. Break-even batch drops to ~8 for L=16 (was ~16) and ~16 for L=32 (was ~32).
- Component split at L=16 / batch 64 / block_v=128: precompute 20.4 us (23%), apply ~68 us. The precompute is latency-bound (16 sequential steps with two reductions per step, only 4.3 MB moved) so it is the next target if more speedup is needed.

## Second optimization round: faithful cycle model + apply tuning (2026-09-14)

**Modeling correction.** The earlier amortized numbers timed every non-flush step at ring length L, i.e. the worst case. In the real system the ring grows 1, 2, ... L-1 across a window and only the L-th step flushes, so the honest per-token cost is `(sum_{k=1..L-1} replay(ring=k) + flush(ring=L)) / L`. The benchmark now measures both (`speedup_vs_bf16` = worst case, `cycle_*_speedup` = steady state). The difference is 10-15% at L=8/16 already measured, and it is a correction in favour of the implementation, not a change to it.

**Grouped precompute did not help.** Serving all value heads of a key head from one program halves the re-read of `k`, but moved the split total only within noise. The precompute is ~23% of the split and latency-bound, so halving its traffic does not pay.

**Apply-kernel tuning did help.** Sweeping the key-chunk width and pipeline stages at L=16/batch 64/block_v=128:

| key chunk | stages | split total | speedup |
| --- | --- | --- | --- |
| 32 | 2 | 88.1 us | 1.85x |
| 64 | 3 | 83.6 us | 1.95x |
| 128 (whole K) | 2 | 83.0 us | **1.96x** |

So the win came from *splitting the recurrence out of the streaming kernel at all*, not from shrinking the register tile: with `tq` in memory the apply kernel can use large contiguous loads, and the chunked variants were slower. This falsifies the initial register-pressure explanation and is recorded as such.

**Final cycle-accurate table** (speedup vs the BF16 read-update-write step, best variant per cell):

| batch | L=4 | L=8 | L=16 | L=32 |
| --- | --- | --- | --- | --- |
| 1 | 1.11 | 0.90 | 0.74 | 0.53 |
| 4 | 1.22 | 1.09 | 0.91 | 0.74 |
| 16 | 1.89 | 1.73 | 1.55 | 1.23 |
| 64 | 2.33 | 2.25 | **2.07** | 1.76 |

- The numerically-required window (L=16) now reaches **2.07x at batch 64** and 1.55x at batch 16, up from 1.20x / ~0.82x at the start of the optimization work.
- Variant choice is automatic: tiled wins at small batch (the split's extra launch is not amortized), split wins from batch ~16 (L=8/16/32).

## Baseline against the production operator (2026-09-14)

The missing piece for a kernel-project claim: not "faster than a kernel I wrote", but "faster than the operator vLLM runs today". `benchmarks/kernels/bench_gdn_vs_production.py` drives `fused_recurrent_gated_delta_rule_packed_decode` (FLA, `vllm/third_party/flash_linear_attention`) directly -- no vLLM engine, no integration -- with the real call contract: `mixed_qkv` packed `[q|k|v]`, `a`/`b` raw gating, `A_log`/`dt_bias`, `initial_state` = the `[slots, HV, V, K]` cache with `ssm_state_indices` (slot 0 = null), `out` = `[B, 1, HV, V]`, `use_qk_l2norm_in_kernel=True`.

- Harness correctness is proven, not assumed: the production operator's output matches the fp32 reference to 1.7e-3 (the bf16 floor). This is exactly the call contract that failed in the first shadow attempt, so the earlier failure was a calling-convention issue, not a math issue.
- **A measurement bug in my own harness nearly produced a false conclusion.** The first pass put a `.clone()` of the state cache and the dtype casts *inside* the timed closure, which made the same-style BF16 kernel look 2x slower than the production operator. Hoisting them out brought the two within 5% (batch 1: 7.58 us vs 7.51 us; batch 64: 163.3 us vs 156.3 us), which is the expected result for two kernels doing the same traffic. The first-pass log is kept in `results/` as evidence.
- Final, cycle-accurate speedup of the replay kernel **against the production operator** (single layer, one decode step):

| batch | L=4 | L=8 | L=16 |
| --- | --- | --- | --- |
| 1 | 1.07 | 0.87 | 0.75 |
| 2 | 1.18 | 1.06 | 0.91 |
| 4 | 1.18 | 1.11 | 1.01 |
| 8 | 1.53 | 1.32 | 1.30 |
| 16 | 1.85 | 1.65 | 1.52 |
| 32 | 2.01 | 1.94 | 1.79 |
| 64 | 2.21 | 2.15 | **1.99** |

- The production operator is well optimised: 157 us for a batch-64 layer step implies ~815 GB/s, i.e. 81% of the 4090's peak, so the 1.99x is an algorithmic (traffic) win rather than a measurement artifact. Break-even is around batch 4; below batch 2 the replay loses (0.75-0.91x) because at that size both paths are launch/latency bound and replay pays for an extra round of ring reads.
- Still excluded from the replay timing: the per-token prep (q/k L2 normalisation, gating, ring append) that the production operator performs inline. Its volume is small (4096 element-ops plus a 12.2 KB append per sequence per layer per token) but it must be stated as a caveat rather than silently omitted.
