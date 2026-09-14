# Model-level GDN replay A/B logs

All logs are JSON lines emitted by `prototype/qwen35_replay_ab.py` and
`prototype/qwen35_logprob_probe.py`, run on the RTX 4090 host (`gdn-remote`,
worktree `/root/vllm-fp8-replayssm`, Qwen3.5-4B bf16, eager, `max_model_len=512`,
greedy decoding). The harness drives the real decode path: replay output
replaces `core_attn_out` and the reconstructed state is written back into the
recurrent cache slot.

| File | Setup | Headline result |
| --- | --- | --- |
| `ab_32token_L2L4L8.log` | 1 prompt, 32 tokens, FP8 vblock, L=2/4/8, all 24 GDN layers replayed | 32/32 tokens identical for every window; mean abs logprob delta 5.5e-4/5.6e-4/8.1e-4; state drift vs baseline 4.96%/3.76%/2.76% |
| `ab_256token_L4L8L16.log` | 2 prompts, 256 tokens, L=4/8/16 | prompt 0 diverges at step 120 in all windows; prompt 1 keeps L=4 identical for 256 steps |
| `ab_256token_none_vs_fp8_3prompts.log` | 3 prompts, adds the quantization-free control arm | exact arm and FP8 arms diverge at the *same* step (120 / 94), so divergence is not FP8-caused |
| `ab_256token_none_vs_fp8_4prompts.log` | 4 prompts, per-step margins logged | all FP8 flips happen at contexts whose top-2 margin is 0.000 or 0.125 (bf16 ulp) |
| `baseline_margin_probe.log` | unmodified model, 256 steps, top-5 alternatives dumped | 5 zero-margin steps out of 256; every reported margin is a multiple of 0.125 |
| `drift_2048_vblock_L4L8L16.log` | capture 2048 real decode steps, offline exact-vs-FP8 chains | drift grows monotonically and does not saturate (`t^0.37`) |
| `drift_4085_vblock32_L4L16L64.log` | capture 4085 real decode steps, vblock32, L=4/16/64 | per-flush injected error is ~1% for every window; drift ~ sqrt(flushes) |
| `drift_4085_vblock32_rotatedK_L4L16L64.log` | same, stored in a Hadamard-rotated key basis | rotation is equivariant (1.6e-7) but makes readout drift worse; rejected |
| `drift_4085_format_arms_L16L64.log` | same, vblock16 and INT8 at L=16/64 | block size is irrelevant (~1e-5); INT8 wins on layers 0/16/24 but loses on layer 8 |
| `kernel_sweep_blockv_warps.log` | fused replay kernel vs BF16 read-update-write step, block_v x num_warps sweep, batch 1-64, L=4/8/16/32 | L=4: up to 2.21x; L=16: up to 1.48x at batch 64; break-even batch ~1/4/8/32 |
| `kernel_sweep_v1_with_scale_bug.log` | same sweep before the per-row scale fix | kept as evidence: wide tiles silently used the wrong vblock scale (1.7e-3 -> 1.2e-2) |
| `kernel_split_first_pass.log` | first two-stage (precompute + apply) attempt, before K-chunking the state readout | correct but no faster: apply still held a [128,128] fp32 tile |
| `kernel_final_sweep_tiled_vs_split.log` | final sweep, tiled vs split at every cell | L=16: 1.50x tiled vs 1.77x split at batch 64; L=32: 1.02x vs 1.42x |
| `kernel_cycle_model_first.log` | first run with the steady-state (ring grows 1..L) cycle model | cycle numbers are 10-15% better than the worst-case ring=L model |
| `kernel_final_cycle_tuned.log` | final sweep with tuned apply (whole-K chunk) and the cycle model | L=16: **2.07x** at batch 64, 1.55x at batch 16; L=32: 1.76x at batch 64 |
| `vs_production_first_pass_with_harness_bug.log` | first production-baseline run; state `.clone()` was inside the timed closure | kept as evidence: it made our BF16 kernel look 2x slower than production |
| `vs_production_final.log` | replay vs vLLM's production FLA operator, corrected harness | L=16: **1.99x** at batch 64, 1.52x at batch 16; break-even ~batch 4 |

Analyze with `python tools/analyze_replay_ab.py results/<file>.log`.
Long-horizon captures analyze with `python tools/analyze_drift_study.py results/<file>.log`.

The captured inputs themselves are on the server at `/root/qwen35_capture_p0.pt`
(4085 steps x layers 0/8/16/24), so further format/window sweeps can be run with
`--load-capture` without reloading the model.
