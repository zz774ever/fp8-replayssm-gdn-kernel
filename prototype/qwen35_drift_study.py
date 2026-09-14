"""Long-horizon drift study for FP8 GDN checkpoints on real captured inputs.

The model-level A/B cannot answer the long-horizon question: greedy decoding
diverges at bf16 logit ties within ~100 steps, after which any comparison is
between two different texts. This script separates the two issues.

1. Capture: run the unmodified model for a long decode and record, per selected
   GDN layer, the prefill state plus the exact recurrence inputs (q, k, v, g,
   beta) of every decode step. Nothing is perturbed during capture.
2. Study: offline, run the exact fp32 recurrence and the FP8-checkpoint chain
   side by side over all captured steps, in lock-step, for several flush
   windows. Because both chains consume the same inputs, the only difference is
   the quantization, and the state drift is a clean scalar per step.

This directly answers the open question: does the checkpoint error keep growing
with sequence length, or does it saturate?
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
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
CAPTURE = False
SELECTED_LAYERS: set[int] = set()
CURRENT_CONTEXT: tuple[object, int] | None = None
INITIAL_STATE: dict[int, torch.Tensor] = {}
INPUTS: dict[int, list[tuple[torch.Tensor, ...]]] = {}


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.float().norm().clamp_min(1e-12)
    return ((actual.float() - expected.float()).norm() / denominator).item()


def _hadamard(size: int, device: torch.device) -> torch.Tensor:
    """Normalized Sylvester-Hadamard matrix of a power-of-two size."""
    matrix = torch.ones(1, 1, device=device, dtype=torch.float32)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix / math.sqrt(size)


def _rotate_inputs(
    inputs: tuple[torch.Tensor, ...], rotation: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    q, k, v, g, beta = inputs
    return (q @ rotation, k @ rotation, v, g, beta)


def _layer_number(prefix: str) -> int | None:
    match = re.search(r"layers\.(\d+)", prefix)
    return int(match.group(1)) if match else None


def _prepare_inputs(
    self, mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, ...]:
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


def install_capture_hook() -> None:
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
        if not CAPTURE or layer not in SELECTED_LAYERS:
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
        context = CURRENT_CONTEXT
        if context is None or not CAPTURE:
            return original_recurrent(
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
        self, layer = context
        slot = int(ssm_state_indices[0].item())
        state_before = None
        if layer not in INITIAL_STATE and slot > 0 and mixed_qkv.shape[0] == 1:
            # The production kernel updates the recurrent cache in place, so the
            # pre-step state must be read before the call.
            state_before = initial_state[slot : slot + 1].float().clone()
        result = original_recurrent(
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
        if mixed_qkv.shape[0] == 1 and slot > 0:
            if state_before is not None:
                INITIAL_STATE[layer] = state_before.to("cpu")
            INPUTS.setdefault(layer, []).append(
                tuple(
                    tensor.detach().to("cpu", torch.float32)
                    for tensor in _prepare_inputs(self, mixed_qkv, a, b)
                )
            )
        return result

    qwen_gdn_linear_attn.QwenGatedDeltaNetAttention._forward_core_decode_non_spec = wrapped_method
    qwen_gdn_linear_attn.fused_recurrent_gated_delta_rule_packed_decode = wrapped_recurrent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*",
    )
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=2560)
    parser.add_argument("--layers", default="0,8,16,24")
    parser.add_argument("--windows", default="4,8,16")
    parser.add_argument(
        "--granularities",
        default="vblock:32",
        help=(
            "Comma-separated checkpoint formats, optionally with an explicit "
            "block size, e.g. 'vblock:32,vblock:16,vblock_int8:32,tile:16'."
        ),
    )
    parser.add_argument(
        "--save-capture",
        default=None,
        help="Write the captured inputs to this path for offline re-analysis.",
    )
    parser.add_argument(
        "--load-capture",
        default=None,
        help="Skip the model entirely and reuse a previously saved capture.",
    )
    parser.add_argument("--report-stride", type=int, default=32)
    parser.add_argument(
        "--prompt",
        default=(
            "Explain in two sentences why recurrent state caching matters for "
            "autoregressive decoding."
        ),
    )
    parser.add_argument(
        "--rotate-k",
        action="store_true",
        help=(
            "Store the recurrent state in a Hadamard-rotated key basis. The "
            "recurrence is exactly equivariant under an orthogonal rotation of "
            "the key axis (the decay is a scalar per value head), so this only "
            "changes how the state is quantized, not what is computed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    global CAPTURE, SELECTED_LAYERS, INITIAL_STATE, INPUTS

    args = parse_args()
    SELECTED_LAYERS = {
        int(value) for value in args.layers.split(",") if value != ""
    }
    windows = [int(value) for value in args.windows.split(",") if value != ""]
    arms_spec: list[tuple[str, int]] = []
    for item in args.granularities.split(","):
        if item == "":
            continue
        name, _, tile = item.partition(":")
        arms_spec.append((name, int(tile) if tile else TILE_SIZE))

    if args.load_capture is not None:
        payload = torch.load(args.load_capture, weights_only=False)
        INITIAL_STATE = payload["initial_state"]
        INPUTS = payload["inputs"]
        print(
            json.dumps(
                {
                    "type": "capture",
                    "source": args.load_capture,
                    "layers": sorted(INPUTS),
                    "steps_per_layer": {
                        str(key): len(value) for key, value in INPUTS.items()
                    },
                }
            ),
            flush=True,
        )
    else:
        install_capture_hook()
        from vllm import LLM, SamplingParams

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
        CAPTURE = True
        llm.generate(
            [args.prompt],
            SamplingParams(temperature=0.0, max_tokens=args.tokens),
            use_tqdm=False,
        )
        CAPTURE = False
        if args.save_capture is not None:
            torch.save(
                {"initial_state": INITIAL_STATE, "inputs": INPUTS},
                args.save_capture,
            )
        print(
            json.dumps(
                {
                    "type": "capture",
                    "model": model,
                    "layers": sorted(INPUTS),
                    "steps_per_layer": {
                        str(key): len(value) for key, value in INPUTS.items()
                    },
                    "saved_to": args.save_capture,
                }
            ),
            flush=True,
        )

    device = torch.device("cuda")
    rotation = None
    for layer in sorted(INPUTS):
        steps = INPUTS[layer]
        state_true = INITIAL_STATE[layer].to(device, torch.float32)
        if args.rotate_k:
            rotation = _hadamard(state_true.shape[-1], device)
            state_true = state_true @ rotation
            first = tuple(tensor.to(device, torch.float32) for tensor in steps[0])
            rotated_first = _rotate_inputs(first, rotation)
            plain_output, plain_state = run_kernel(
                INITIAL_STATE[layer].to(device, torch.float32), *first
            )
            rotated_output, rotated_state = run_kernel(state_true, *rotated_first)
            print(
                json.dumps(
                    {
                        "type": "rotation_check",
                        "layer": layer,
                        "output_relative_l2": _relative_l2(
                            rotated_output, plain_output
                        ),
                        "state_relative_l2": _relative_l2(
                            rotated_state @ rotation, plain_state
                        ),
                    }
                ),
                flush=True,
            )
        arms: dict[str, dict] = {}
        for granularity, tile in arms_spec:
            for window in windows:
                arms[f"{granularity}{tile}_L{window}"] = {
                    "granularity": granularity,
                    "tile": tile,
                    "window": window,
                    "checkpoint": quantize_checkpoint(
                        state_true, granularity, tile
                    ),
                    "ring": [],
                    "state_drift": [],
                    "output_drift": [],
                }
        for step, inputs in enumerate(steps):
            gpu_inputs = tuple(tensor.to(device, torch.float32) for tensor in inputs)
            if rotation is not None:
                gpu_inputs = _rotate_inputs(gpu_inputs, rotation)
            true_output, state_true = run_kernel(state_true, *gpu_inputs)
            for arm in arms.values():
                arm["ring"].append(gpu_inputs)
                ring = tuple(
                    torch.cat([item[index] for item in arm["ring"]], dim=1)
                    for index in range(5)
                )
                output, reconstructed = run_kernel(
                    dequantize_checkpoint(arm["checkpoint"]), *ring
                )
                arm["state_drift"].append(
                    _relative_l2(reconstructed, state_true)
                )
                arm["output_drift"].append(
                    _relative_l2(output[:, -1], true_output[:, -1])
                )
                if len(arm["ring"]) == arm["window"]:
                    arm["checkpoint"] = quantize_checkpoint(
                        reconstructed, arm["granularity"], arm["tile"]
                    )
                    arm["ring"].clear()
        for name, arm in sorted(arms.items()):
            drift = arm["state_drift"]
            out_drift = arm["output_drift"]
            curve = [
                [
                    index,
                    round(drift[index], 6),
                    round(out_drift[index], 6),
                ]
                for index in range(0, len(drift), args.report_stride)
            ]
            print(
                json.dumps(
                    {
                        "type": "drift_curve",
                        "layer": layer,
                        "arm": name,
                        "steps": len(drift),
                        "points": curve,
                    }
                ),
                flush=True,
            )
            last = len(drift) - 1
            summary = {
                "type": "drift_summary",
                "layer": layer,
                "arm": name,
                "steps": len(drift),
                "state_drift_last": drift[last],
                "state_drift_mean": sum(drift) / len(drift),
                "state_drift_max": max(drift),
                "output_drift_last": out_drift[last],
                "output_drift_mean": sum(out_drift) / len(out_drift),
                "output_drift_max": max(out_drift),
            }
            for probe in (64, 128, 256, 512, 1024, 1536, 2048, 4096):
                if probe - 1 < len(drift):
                    summary[f"state_drift_at_{probe}"] = drift[probe - 1]
            summary["growth_512_to_last"] = (
                drift[last] / drift[511] if len(drift) > 512 and drift[511] else None
            )
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
