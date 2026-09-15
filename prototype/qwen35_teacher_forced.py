"""Teacher-forced numerical acceptance test for FP8 checkpoint replay.

Greedy token agreement cannot be the acceptance metric: bf16 logits make the
top-2 margin equal to zero on ~1.5% of steps, so any perturbation flips those
steps and the two runs then diverge into different texts. This script removes
the divergence instead of trying to measure through it.

Method: the baseline's greedy token sequence is replayed *into* every arm, so
all arms see the same inputs at every step. The sampler is patched to force the
chosen token while leaving the distribution untouched, which matters because
vLLM's greedy path returns the raw logits and gathers logprobs from them -- so
the recorded logprob is the one the *unmodified* model assigned to that token.

For each step this yields, per arm:

* the logprob the arm assigns to the baseline's token (the primary metric)
* the top-k alternatives (for an approximate KL / total-variation distance)

Self-check: with the replay arm disabled the forced run must reproduce the
baseline's logprobs exactly. If it does not, the apparatus is broken and the
numbers are meaningless.

Acceptance threshold (pre-registered): p95 |delta logprob| < 0.125. The basis is
empirical -- bf16 top-2 logit margins in the baseline cluster on 0.125 steps --
and deliberately not phrased as "one bf16 ulp": a bf16 ulp varies with the
exponent, and a logit rounding step is not a fixed step in logprob.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "kernels"))

from gdn_replay_fp8_reference import (  # noqa: E402
    QuantizedCheckpoint,
    dequantize_checkpoint,
    quantize_checkpoint,
    run_kernel,
)

TILE_SIZE = 32

MODE = "off"  # off | replay
WINDOW = 4
GRANULARITY = "vblock"
CAPTURE_ENABLED = False
SELECTED_LAYERS: set[int] = set()
CURRENT_CONTEXT: tuple[object, int] | None = None
STATES: dict[tuple[int, int], "ReplayState"] = {}
CALLS: dict[int, int] = {}

# Forced-token machinery
FORCED_TOKENS: list[int] | None = None
FORCED_INDEX = 0
SAMPLER_RECORDS: list[dict] = []


@dataclass
class ReplayState:
    checkpoint: object
    q: list = field(default_factory=list)
    k: list = field(default_factory=list)
    v: list = field(default_factory=list)
    g: list = field(default_factory=list)
    beta: list = field(default_factory=list)

    def clear_ring(self) -> None:
        for buffer in (self.q, self.k, self.v, self.g, self.beta):
            buffer.clear()


def _relative_l2(actual, expected) -> float:
    return (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-12)
    ).item()


def _layer_number(prefix: str) -> int | None:
    match = re.search(r"layers\.(\d+)", prefix)
    return int(match.group(1)) if match else None


def _prepare_inputs(self, mixed_qkv, a, b):
    batch = mixed_qkv.shape[0]
    num_k_heads = self.num_k_heads // self.tp_size
    num_v_heads = self.num_v_heads // self.tp_size
    key_dim = self.head_k_dim
    value_dim = self.head_v_dim
    q_end = num_k_heads * key_dim
    k_end = 2 * q_end
    q = mixed_qkv[:, :q_end].reshape(batch, 1, num_k_heads, key_dim)
    k = mixed_qkv[:, q_end:k_end].reshape(batch, 1, num_k_heads, key_dim)
    v = mixed_qkv[:, k_end:].reshape(batch, 1, num_v_heads, value_dim)
    q = F.normalize(q.float(), dim=-1)
    k = F.normalize(k.float(), dim=-1)
    g = -torch.exp(self.A_log.float()) * F.softplus(a.float() + self.dt_bias.float())
    beta = torch.sigmoid(b.float())
    return q, k, v.float(), g[:, None, :], beta[:, None, :]


def _requantize(state, granularity):
    if granularity == "none":
        return state.clone()
    return quantize_checkpoint(state, granularity, TILE_SIZE)


def install_hooks() -> None:
    global CURRENT_CONTEXT

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    original_method = (
        qwen_gdn_linear_attn.QwenGatedDeltaNetAttention._forward_core_decode_non_spec
    )
    original_recurrent = (
        qwen_gdn_linear_attn.fused_recurrent_gated_delta_rule_packed_decode
    )
    original_sample = GPUModelRunner.sample

    def patched_sample(self, hidden_states, input_batch, grammar_output):
        """Force the chosen token without touching the distribution.

        The probe (`qwen35_sampler_probe.py`) showed this runner method is the
        one that actually executes; patching `Sampler.sample`/`forward` had no
        effect. Overriding `sampled_token_ids` here leaves the logprob tensors,
        which were computed from the raw logits, untouched -- so the recorded
        logprob is still the one the unmodified model assigned to that token.
        """
        result = original_sample(self, hidden_states, input_batch, grammar_output)
        sampler_output = result[0]
        global FORCED_INDEX
        if FORCED_TOKENS is not None and sampler_output is not None:
            ids = getattr(sampler_output, "sampled_token_ids", None)
            if (
                ids is not None
                and ids.numel() > 0
                and FORCED_INDEX < len(FORCED_TOKENS)
            ):
                ids[0, 0] = FORCED_TOKENS[FORCED_INDEX]
                FORCED_INDEX += 1
        return result

    def wrapped_method(self, mixed_qkv, b, a, core_attn_out, attn_metadata):
        global CURRENT_CONTEXT
        layer = _layer_number(self.prefix)
        active = (
            CAPTURE_ENABLED
            and MODE == "replay"
            and layer is not None
            and layer in SELECTED_LAYERS
        )
        if not active:
            return original_method(self, mixed_qkv, b, a, core_attn_out, attn_metadata)
        CURRENT_CONTEXT = (self, layer)
        try:
            return original_method(self, mixed_qkv, b, a, core_attn_out, attn_metadata)
        finally:
            CURRENT_CONTEXT = None

    def wrapped_recurrent(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel=False,
    ):
        call_original = lambda: original_recurrent(  # noqa: E731
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            scale,
            initial_state,
            out,
            ssm_state_indices,
            use_qk_l2norm_in_kernel,
        )
        context = CURRENT_CONTEXT
        if context is None or MODE != "replay":
            return call_original()
        self, layer = context
        slot = int(ssm_state_indices[0].item())
        if mixed_qkv.shape[0] != 1 or slot <= 0:
            return call_original()
        state_before = initial_state[slot : slot + 1].float().clone()
        inputs = _prepare_inputs(self, mixed_qkv, a, b)
        step = CALLS.get(layer, 0)
        CALLS[layer] = step + 1
        result = call_original()
        key = (layer, slot)
        replay = STATES.get(key)
        if replay is None:
            replay = ReplayState(checkpoint=_requantize(state_before, GRANULARITY))
            STATES[key] = replay
        for destination, source in zip(
            (replay.q, replay.k, replay.v, replay.g, replay.beta), inputs
        ):
            destination.append(source.detach().clone())
        ring = tuple(
            torch.cat(parts, dim=1)
            for parts in (replay.q, replay.k, replay.v, replay.g, replay.beta)
        )
        output, reconstructed = run_kernel(
            dequantize_checkpoint(replay.checkpoint)
            if isinstance(replay.checkpoint, QuantizedCheckpoint)
            else replay.checkpoint,
            *ring,
        )
        correction = scale * math.sqrt(self.head_k_dim)
        if abs(correction - 1.0) > 1e-6:
            output = output * correction
        view = out.reshape(1, -1, self.head_v_dim)
        view[0].copy_(output[0, -1].to(view.dtype))
        returned = result[0] if isinstance(result, tuple) else result
        if isinstance(returned, torch.Tensor) and returned is not out:
            returned.reshape(1, -1, self.head_v_dim)[0].copy_(
                output[0, -1].to(returned.dtype)
            )
        initial_state[slot : slot + 1].copy_(reconstructed.to(initial_state.dtype))
        if len(replay.q) == WINDOW:
            replay.checkpoint = _requantize(reconstructed, GRANULARITY)
            replay.clear_ring()
        return result

    qwen_gdn_linear_attn.QwenGatedDeltaNetAttention._forward_core_decode_non_spec = wrapped_method
    qwen_gdn_linear_attn.fused_recurrent_gated_delta_rule_packed_decode = wrapped_recurrent
    GPUModelRunner.sample = patched_sample


def run_pass(llm, prompt, max_tokens, mode, window, forced, logprobs):
    """One generation pass, optionally forcing a token sequence."""
    global MODE, WINDOW, FORCED_TOKENS, FORCED_INDEX, CAPTURE_ENABLED
    from vllm import SamplingParams

    STATES.clear()
    CALLS.clear()
    MODE = mode
    WINDOW = window
    FORCED_TOKENS = forced
    FORCED_INDEX = 0
    CAPTURE_ENABLED = mode == "replay"
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=logprobs),
        use_tqdm=False,
    )
    completion = outputs[0].outputs[0]
    steps = []
    for index, token in enumerate(completion.token_ids):
        entry = completion.logprobs[index] if completion.logprobs else {}
        ranked = sorted(
            ((item, value.logprob) for item, value in entry.items()),
            key=lambda pair: -pair[1],
        )
        steps.append(
            {
                "token": token,
                "topk": [[int(item), float(value)] for item, value in ranked],
            }
        )
    return steps


def compare(reference, candidate):
    """Per-step deltas on the forced trajectory."""
    common = min(len(reference), len(candidate))
    deltas = []
    kl = []
    for index in range(common):
        ref = dict(reference[index]["topk"])
        cand = dict(candidate[index]["topk"])
        token = reference[index]["token"]
        if token in ref and token in cand:
            deltas.append(abs(ref[token] - cand[token]))
        # KL over the union of the top-k supports, using the reference as the
        # anchor distribution (a bounded proxy, not a calibrated KL).
        keys = set(ref) | set(cand)
        ref_mass = {key: math.exp(ref.get(key, -30.0)) for key in keys}
        total = sum(ref_mass.values())
        if total <= 0:
            continue
        value = 0.0
        for key in keys:
            p = ref_mass[key] / total
            q = math.exp(cand.get(key, -30.0)) / max(
                sum(math.exp(cand.get(k, -30.0)) for k in keys), 1e-30
            )
            if p > 0 and q > 0:
                value += p * math.log(p / q)
        kl.append(value)
    deltas.sort()
    kl.sort()

    def percentile(values, fraction):
        if not values:
            return None
        index = min(len(values) - 1, int(fraction * len(values)))
        return values[index]

    return {
        "steps": common,
        "mean_abs_dlogp": sum(deltas) / len(deltas) if deltas else None,
        "p95_abs_dlogp": percentile(deltas, 0.95),
        "max_abs_dlogp": deltas[-1] if deltas else None,
        "mean_kl_topk": sum(kl) / len(kl) if kl else None,
        "p95_kl_topk": percentile(kl, 0.95),
        "max_kl_topk": kl[-1] if kl else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=1280)
    parser.add_argument("--windows", default="4,8,16")
    parser.add_argument("--granularity", default="vblock")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument(
        "--prompt",
        default=(
            "Explain in two sentences why recurrent state caching matters for "
            "autoregressive decoding."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.125)
    return parser.parse_args()


def main() -> None:
    global SELECTED_LAYERS, GRANULARITY

    args = parse_args()
    GRANULARITY = args.granularity
    SELECTED_LAYERS = (
        set(range(0, 64))
        if args.layers == "all"
        else {int(value) for value in args.layers.split(",") if value != ""}
    )
    install_hooks()
    from vllm import LLM

    matches = glob.glob(args.model)
    model = matches[0] if len(matches) == 1 else args.model
    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print(
        json.dumps(
            {
                "type": "config",
                "model": model,
                "max_tokens": args.max_tokens,
                "windows": args.windows,
                "granularity": GRANULARITY,
                "logprobs": args.logprobs,
                "threshold": args.threshold,
                "replay_layers": "all GDN layers",
            }
        ),
        flush=True,
    )

    # 1) Baseline trajectory (no forcing).
    baseline = run_pass(
        llm, args.prompt, args.max_tokens, "off", 0, None, args.logprobs
    )
    tokens = [step["token"] for step in baseline]
    print(
        json.dumps(
            {"type": "baseline", "steps": len(tokens), "tokens": tokens}
        ),
        flush=True,
    )

    # 2) Self-check: forcing with the replay arm off must reproduce the baseline.
    control = run_pass(
        llm, args.prompt, args.max_tokens, "off", 0, tokens, args.logprobs
    )
    check = compare(baseline, control)
    print(
        json.dumps(
            {
                "type": "self_check",
                "apparatus_ok": check["max_abs_dlogp"] == 0.0,
                "note": "forced run with replay disabled must reproduce the baseline",
                **check,
            }
        ),
        flush=True,
    )

    # 2b) Prove the forcing is actually live: force a deliberately corrupted
    # sequence and require the engine to return exactly that sequence. Without
    # this, a silently ineffective patch would still "pass" the check above,
    # because a deterministic greedy re-run reproduces the baseline anyway.
    corrupted = list(tokens)
    for index in range(5, min(10, len(corrupted))):
        corrupted[index] = 0
    corrupted_run = run_pass(
        llm,
        args.prompt,
        args.max_tokens,
        "off",
        0,
        corrupted,
        args.logprobs,
    )
    produced = [step["token"] for step in corrupted_run]
    forced_honoured = produced[: len(corrupted)] == corrupted
    print(
        json.dumps(
            {
                "type": "forcing_check",
                "forcing_is_live": forced_honoured,
                "mismatch_index": next(
                    (
                        index
                        for index in range(min(len(produced), len(corrupted)))
                        if produced[index] != corrupted[index]
                    ),
                    None,
                ),
            }
        ),
        flush=True,
    )
    if not forced_honoured:
        print(
            json.dumps(
                {
                    "type": "abort",
                    "reason": "token forcing did not take effect; "
                    "teacher-forced numbers would be meaningless",
                }
            ),
            flush=True,
        )
        return

    # 3) Replay arms, same forced trajectory.
    for window in (int(value) for value in args.windows.split(",") if value != ""):
        candidate = run_pass(
            llm, args.prompt, args.max_tokens, "replay", window, tokens, args.logprobs
        )
        summary = compare(baseline, candidate)
        summary.update(
            {
                "type": "replay",
                "window": window,
                "granularity": GRANULARITY,
                "passes_threshold": (
                    summary["p95_abs_dlogp"] is not None
                    and summary["p95_abs_dlogp"] < args.threshold
                ),
            }
        )
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
