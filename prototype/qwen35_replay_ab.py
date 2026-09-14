"""Model-level A/B: drive real Qwen3.5 decode with FP8 GDN replay.

Unlike ``qwen35_replay_shadow.py`` (which only computes replay numbers and
throws them away), this harness actually *injects* the replay result into the
running model:

* the layer output produced by the replay kernel replaces ``core_attn_out``;
* the reconstructed GDN state is written back into the recurrent cache slot.

Writing the state back is the point of the experiment. The persistent state in
the target design is "FP8 checkpoint + ring of recent inputs" only, so every
decode step consumes a state that was reconstructed from a quantized
checkpoint, and quantization error accumulates through the chain instead of
being washed out by a full-precision cache update.

The same process first runs an unmodified greedy baseline (recording per-step
tokens, logprobs and, for a few layers, state snapshots), then repeats the
baseline to establish a run-to-run noise floor, then replays with several flush
windows. Everything is printed as JSON lines for downstream aggregation.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
import time
from collections import defaultdict
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

MODE = "off"  # off | baseline_snapshot | replay
WINDOW = 4
TILE_SIZE = 32
GRANULARITY = "vblock"  # vblock | vblock_int8 | none (quantization-free control)
REPLAY_LAYERS: set[int] = set()
SNAPSHOT_LAYERS: set[int] = set()

CURRENT_CONTEXT: tuple[object, int] | None = None
STATES: dict[tuple[int, int], "ReplayState"] = {}
CALLS: defaultdict[int, int] = defaultdict(int)
BASELINE_STATES: dict[tuple[int, int], torch.Tensor] = {}
RECORDS: list[dict[str, float | int]] = []


@dataclass
class ReplayState:
    # ``torch.Tensor`` when the quantization-free control is active.
    checkpoint: QuantizedCheckpoint | torch.Tensor
    q: list[torch.Tensor] = field(default_factory=list)
    k: list[torch.Tensor] = field(default_factory=list)
    v: list[torch.Tensor] = field(default_factory=list)
    g: list[torch.Tensor] = field(default_factory=list)
    beta: list[torch.Tensor] = field(default_factory=list)
    last_reconstructed: torch.Tensor | None = None

    def clear_ring(self) -> None:
        for buffer in (self.q, self.k, self.v, self.g, self.beta):
            buffer.clear()


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.float().norm().clamp_min(1e-12)
    return ((actual.float() - expected.float()).norm() / denominator).item()


def _layer_number(prefix: str) -> int | None:
    match = re.search(r"layers\.(\d+)", prefix)
    return int(match.group(1)) if match else None


def _prepare_inputs(
    self, mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Mirror the production pre-processing of the packed decode path."""
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


def _append(replay: ReplayState, inputs: tuple[torch.Tensor, ...]) -> None:
    for destination, source in zip(
        (replay.q, replay.k, replay.v, replay.g, replay.beta), inputs
    ):
        destination.append(source.detach().clone())


def _ring_tensors(replay: ReplayState) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.cat(parts, dim=1)
        for parts in (replay.q, replay.k, replay.v, replay.g, replay.beta)
    )


def _run_kernel_scaled(
    state: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    scale: float,
    key_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, reconstructed = run_kernel(state, *tensors)
    correction = scale * math.sqrt(key_dim)
    if abs(correction - 1.0) > 1e-6:
        output = output * correction
    return output, reconstructed


def _dequantize(checkpoint: QuantizedCheckpoint | torch.Tensor) -> torch.Tensor:
    if isinstance(checkpoint, torch.Tensor):
        return checkpoint
    return dequantize_checkpoint(checkpoint)


def _requantize(
    state: torch.Tensor, granularity: str
) -> QuantizedCheckpoint | torch.Tensor:
    if granularity == "none":
        # Control arm: identical replay plumbing, but the checkpoint keeps the
        # exact fp32 state so any remaining drift is not caused by quantization.
        return state.clone()
    return quantize_checkpoint(state, granularity, TILE_SIZE)


def install_hooks() -> None:
    global CURRENT_CONTEXT

    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    original_method = (
        qwen_gdn_linear_attn.QwenGatedDeltaNetAttention._forward_core_decode_non_spec
    )
    original_recurrent = (
        qwen_gdn_linear_attn.fused_recurrent_gated_delta_rule_packed_decode
    )

    def wrapped_method(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata,
    ) -> None:
        global CURRENT_CONTEXT
        layer = _layer_number(self.prefix)
        active = layer is not None and (
            layer in REPLAY_LAYERS or layer in SNAPSHOT_LAYERS
        )
        if not active or MODE == "off":
            return original_method(self, mixed_qkv, b, a, core_attn_out, attn_metadata)
        CURRENT_CONTEXT = (self, layer)
        try:
            return original_method(self, mixed_qkv, b, a, core_attn_out, attn_metadata)
        finally:
            CURRENT_CONTEXT = None

    def wrapped_recurrent(
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        out: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        if context is None or MODE == "off":
            return call_original()
        self, layer = context
        if mixed_qkv.shape[0] != 1:
            return call_original()
        slot = int(ssm_state_indices[0].item())
        if slot <= 0:
            return call_original()

        state_before = initial_state[slot : slot + 1].float().clone()
        inputs = _prepare_inputs(self, mixed_qkv, a, b)
        step = CALLS[layer]
        CALLS[layer] = step + 1
        result = call_original()
        state_after = initial_state[slot : slot + 1].float().clone()
        production_output = out.reshape(1, -1, self.head_v_dim).float().clone()

        if MODE == "baseline_snapshot":
            if layer in SNAPSHOT_LAYERS:
                BASELINE_STATES[(layer, step)] = state_after.to(
                    torch.bfloat16
                ).cpu()
            return result

        key = (layer, slot)
        replay = STATES.get(key)
        propagation_error = None
        if replay is None:
            replay = ReplayState(
                checkpoint=_requantize(state_before, GRANULARITY)
            )
            STATES[key] = replay
        elif replay.last_reconstructed is not None:
            # If the cache write does not stick, this jumps to the quantization
            # scale (~2%) instead of the bf16 rounding scale (~0.2%).
            propagation_error = _relative_l2(
                state_before, replay.last_reconstructed
            )

        _append(replay, inputs)
        replay_output, reconstructed = _run_kernel_scaled(
            _dequantize(replay.checkpoint),
            _ring_tensors(replay),
            scale,
            self.head_k_dim,
        )

        view = out.reshape(1, -1, self.head_v_dim)
        view[0].copy_(replay_output[0, -1].to(view.dtype))
        returned = result[0] if isinstance(result, tuple) else result
        if isinstance(returned, torch.Tensor) and returned is not out:
            returned.reshape(1, -1, self.head_v_dim)[0].copy_(
                replay_output[0, -1].to(returned.dtype)
            )
        initial_state[slot : slot + 1].copy_(reconstructed.to(initial_state.dtype))
        replay.last_reconstructed = reconstructed

        record: dict[str, float | int | None] = {
            "layer": layer,
            "step": step,
            "window": WINDOW,
            "ring_length": len(replay.q),
            "output_relative_l2": _relative_l2(
                replay_output[0, -1], production_output[0]
            ),
            "state_relative_l2": _relative_l2(reconstructed, state_after),
            "write_propagation_relative_l2": propagation_error,
            "checkpoint_saturation": (
                replay.checkpoint.saturation_fraction
                if isinstance(replay.checkpoint, QuantizedCheckpoint)
                else 0.0
            ),
        }
        baseline = BASELINE_STATES.get((layer, step))
        if baseline is not None:
            record["drift_vs_baseline_relative_l2"] = _relative_l2(
                reconstructed, baseline.float().to(reconstructed.device)
            )
        RECORDS.append(record)

        if len(replay.q) == WINDOW:
            replay.checkpoint = _requantize(reconstructed, GRANULARITY)
            replay.clear_ring()
        return result

    qwen_gdn_linear_attn.QwenGatedDeltaNetAttention._forward_core_decode_non_spec = wrapped_method
    qwen_gdn_linear_attn.fused_recurrent_gated_delta_rule_packed_decode = wrapped_recurrent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--windows", default="2,4,8")
    parser.add_argument(
        "--granularities",
        default="vblock",
        help=(
            "Comma-separated checkpoint formats to replay with. Use 'none' as "
            "the quantization-free control arm."
        ),
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=5,
        help="Alternatives per step, used to measure the greedy margin.",
    )
    parser.add_argument("--snapshot-layers", default="0,1,2,4,8,16,24,30")
    parser.add_argument(
        "--prompts",
        default=(
            "Explain in two sentences why recurrent state caching matters for "
            "autoregressive decoding."
        ),
        help="Pipe-separated prompt list; every prompt runs the full A/B.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def run_pass(
    llm,
    prompt: str,
    max_tokens: int,
    mode: str,
    window: int,
    label: str,
    logprobs: int = 5,
) -> dict:
    global MODE, WINDOW
    from vllm import SamplingParams

    STATES.clear()
    CALLS.clear()
    RECORDS.clear()
    if mode == "baseline_snapshot":
        # Baselines are keyed by (layer, step), so a new prompt must not reuse
        # the previous prompt's snapshots.
        BASELINE_STATES.clear()
    MODE = mode
    WINDOW = window
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=logprobs),
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    completion = outputs[0].outputs[0]
    tokens = list(completion.token_ids)
    chosen_logprobs = []
    margins = []
    for index, token in enumerate(tokens):
        entry = completion.logprobs[index] if completion.logprobs else {}
        ranked = sorted(
            ((item, value.logprob) for item, value in entry.items()),
            key=lambda pair: -pair[1],
        )
        chosen = next((value for item, value in ranked if item == token), None)
        chosen_logprobs.append(chosen if chosen is not None else float("nan"))
        margins.append(
            ranked[0][1] - ranked[1][1] if len(ranked) > 1 else float("inf")
        )
    return {
        "type": "run",
        "label": label,
        "mode": mode,
        "window": window,
        "granularity": GRANULARITY if mode == "replay" else "n/a",
        "elapsed_seconds": elapsed,
        "tokens": tokens,
        "logprobs": chosen_logprobs,
        "margins": margins,
        "text": completion.text,
        "records": RECORDS,
    }


def compare(baseline: dict, candidate: dict, window: int) -> dict:
    base_tokens = baseline["tokens"]
    cand_tokens = candidate["tokens"]
    steps = min(len(base_tokens), len(cand_tokens))
    matches = [base_tokens[i] == cand_tokens[i] for i in range(steps)]
    first_divergence = next(
        (i for i, ok in enumerate(matches) if not ok), None
    )
    deltas = [
        abs(baseline["logprobs"][i] - candidate["logprobs"][i])
        for i in range(steps)
    ]
    records = candidate["records"]
    summary: dict[str, object] = {
        "type": "summary",
        "window": window,
        "steps_compared": steps,
        "token_match_fraction": (sum(matches) / steps) if steps else None,
        "first_divergence_step": first_divergence,
        "mean_abs_logprob_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "max_abs_logprob_delta": max(deltas) if deltas else None,
        "prefix_mean_abs_logprob_delta": (
            sum(deltas[:first_divergence]) / first_divergence
            if first_divergence
            else None
        ),
        "output_relative_l2": _aggregate(records, "output_relative_l2"),
        "state_relative_l2": _aggregate(records, "state_relative_l2"),
        "drift_vs_baseline_relative_l2": _aggregate(
            records, "drift_vs_baseline_relative_l2"
        ),
        "write_propagation_relative_l2": _aggregate(
            records, "write_propagation_relative_l2"
        ),
        "max_checkpoint_saturation": max(
            (float(item["checkpoint_saturation"]) for item in records),
            default=0.0,
        ),
    }
    drift_by_layer: dict[str, dict[str, float]] = {}
    for item in records:
        value = item.get("drift_vs_baseline_relative_l2")
        if value is None:
            continue
        entry = drift_by_layer.setdefault(str(item["layer"]), {})
        entry.setdefault("first", float(value))
        entry["last"] = float(value)
        entry["max"] = max(entry.get("max", 0.0), float(value))
    summary["drift_by_layer"] = drift_by_layer
    trace: dict[int, dict[int, float]] = {}
    for item in records:
        value = item.get("drift_vs_baseline_relative_l2")
        if value is None:
            continue
        trace.setdefault(int(item["layer"]), {})[int(item["step"])] = float(value)
    if trace:
        focus = min(trace)
        points = sorted(trace[focus].items())
        summary["drift_trace_layer"] = focus
        summary["drift_trace"] = [
            [step, round(value, 5)] for step, value in points if step % 8 == 0
        ]
        if first_divergence is not None:
            pre = [pair for pair in points if pair[0] < first_divergence]
        else:
            pre = points
        if pre:
            summary["drift_before_divergence"] = {
                "step": pre[-1][0],
                "value": pre[-1][1],
            }
    if first_divergence is not None:
        summary["divergence"] = {
            "step": first_divergence,
            "baseline_token": base_tokens[first_divergence],
            "candidate_token": cand_tokens[first_divergence],
            "logprob_delta_at_divergence": deltas[first_divergence],
        }
    base_margins = baseline.get("margins") or []
    if base_margins:
        finite = [value for value in base_margins if value != float("inf")]
        summary["baseline_margin"] = {
            "min": min(finite) if finite else None,
            "median": sorted(finite)[len(finite) // 2] if finite else None,
            "fraction_below_0.05": (
                sum(1 for value in finite if value < 0.05) / len(finite)
                if finite
                else None
            ),
            "fraction_below_0.2": (
                sum(1 for value in finite if value < 0.2) / len(finite)
                if finite
                else None
            ),
            "at_first_divergence": (
                base_margins[first_divergence]
                if first_divergence is not None and first_divergence < len(base_margins)
                else None
            ),
        }
    return summary


def _aggregate(records: list[dict], name: str) -> dict[str, float] | None:
    values = [float(item[name]) for item in records if item.get(name) is not None]
    if not values:
        return None
    mean = sum(values) / len(values)
    return {
        "mean": mean,
        "max": max(values),
        "final": values[-1],
    }


def main() -> None:
    global GRANULARITY, SNAPSHOT_LAYERS, REPLAY_LAYERS

    args = parse_args()
    granularities = [item for item in args.granularities.split(",") if item != ""]
    SNAPSHOT_LAYERS = {
        int(value) for value in args.snapshot_layers.split(",") if value != ""
    }
    REPLAY_LAYERS = set(range(0, 64))  # every GDN layer that hits the hook
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

    prompts = [item for item in args.prompts.split("|") if item != ""]
    windows = [int(value) for value in args.windows.split(",") if value != ""]
    print(
        json.dumps(
            {
                "type": "config",
                "model": model,
                "max_tokens": args.max_tokens,
                "windows": windows,
                "granularities": granularities,
                "prompt_count": len(prompts),
                "snapshot_layers": sorted(SNAPSHOT_LAYERS),
                "replays_all_gdn_layers": True,
                "checkpoint_granularity": "vblock",
                "tile_size": TILE_SIZE,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for prompt_index, prompt in enumerate(prompts):
        baseline = run_pass(
            llm,
            prompt,
            args.max_tokens,
            "baseline_snapshot",
            0,
            f"baseline_p{prompt_index}",
            logprobs=args.logprobs,
        )
        baseline.pop("records")
        baseline["prompt_index"] = prompt_index
        print(json.dumps(baseline, ensure_ascii=False), flush=True)

        for repeat in range(args.repeats):
            noise = run_pass(
                llm,
                prompt,
                args.max_tokens,
                "off",
                0,
                f"baseline_repeat_{repeat}_p{prompt_index}",
                logprobs=args.logprobs,
            )
            noise.pop("records")
            noise["type"] = "noise_floor"
            noise["prompt_index"] = prompt_index
            noise["tokens_identical_to_baseline"] = (
                noise["tokens"] == baseline["tokens"]
            )
            steps = min(len(noise["tokens"]), len(baseline["tokens"]))
            noise["max_abs_logprob_delta"] = max(
                abs(noise["logprobs"][i] - baseline["logprobs"][i])
                for i in range(steps)
            )
            print(json.dumps(noise, ensure_ascii=False), flush=True)

        for granularity in granularities:
            for window in windows:
                GRANULARITY = granularity
                candidate = run_pass(
                    llm,
                    prompt,
                    args.max_tokens,
                    "replay",
                    window,
                    f"replay_{granularity}_L{window}_p{prompt_index}",
                    logprobs=args.logprobs,
                )
                print(
                    json.dumps(
                        {
                            "type": "run",
                            "label": candidate["label"],
                            "mode": candidate["mode"],
                            "window": candidate["window"],
                            "granularity": candidate["granularity"],
                            "prompt_index": prompt_index,
                            "elapsed_seconds": candidate["elapsed_seconds"],
                            "tokens": candidate["tokens"],
                            "logprobs": candidate["logprobs"],
                            "margins": candidate["margins"],
                            "text": candidate["text"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                summary = compare(baseline, candidate, window)
                summary["prompt_index"] = prompt_index
                summary["label"] = candidate["label"]
                summary["granularity"] = granularity
                print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
