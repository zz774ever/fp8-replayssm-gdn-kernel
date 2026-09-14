"""Shadow real Qwen3.5 decode with quantized GDN checkpoints and replay."""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent / "kernels"))

from gdn_replay_fp8_reference import (
    QuantizedCheckpoint,
    dequantize_checkpoint,
    quantize_checkpoint,
    run_kernel,
)


CAPTURE_ENABLED = False
SELECTED_LAYERS = {0, 1, 2, 4, 8, 16, 24, 30}
WINDOWS = (2, 4, 8)


@dataclass
class ReplayState:
    checkpoint: QuantizedCheckpoint
    q: list[torch.Tensor] = field(default_factory=list)
    k: list[torch.Tensor] = field(default_factory=list)
    v: list[torch.Tensor] = field(default_factory=list)
    g: list[torch.Tensor] = field(default_factory=list)
    beta: list[torch.Tensor] = field(default_factory=list)


SHADOWS: dict[tuple[int, int], ReplayState] = {}
CALLS: dict[int, int] = {}
CURRENT_CONTEXT: tuple[object, int] | None = None
RECORDS: list[dict[str, float | int]] = []


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-12)
    ).item()


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
    g = -torch.exp(self.A_log.float()) * F.softplus(
        a.float() + self.dt_bias.float()
    )
    beta = torch.sigmoid(b.float())
    return q, k, v.float(), g[:, None, :], beta[:, None, :]


def _append(replay: ReplayState, inputs: tuple[torch.Tensor, ...]) -> None:
    for destination, source in zip(
        (replay.q, replay.k, replay.v, replay.g, replay.beta), inputs
    ):
        destination.append(source.detach().clone())


def _run_shadow(
    layer: int,
    call: int,
    window: int,
    initial_state: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    expected_output: torch.Tensor,
    expected_state: torch.Tensor,
    oracle_output_error: float,
    oracle_state_error: float,
) -> None:
    key = (layer, window)
    replay = SHADOWS.get(key)
    if replay is None:
        replay = ReplayState(
            checkpoint=quantize_checkpoint(initial_state, "vblock", 32)
        )
        SHADOWS[key] = replay
    _append(replay, inputs)
    tensors = tuple(
        torch.cat(parts, dim=1)
        for parts in (replay.q, replay.k, replay.v, replay.g, replay.beta)
    )
    output, reconstructed = run_kernel(
        dequantize_checkpoint(replay.checkpoint), *tensors
    )
    RECORDS.append(
        {
            "layer": layer,
            "call": call,
            "window": window,
            "ring_length": len(replay.q),
            "output_relative_l2": _relative_l2(
                output[:, -1], expected_output
            ),
            "state_relative_l2": _relative_l2(reconstructed, expected_state),
            "oracle_output_relative_l2": oracle_output_error,
            "oracle_state_relative_l2": oracle_state_error,
            "checkpoint_saturation": replay.checkpoint.saturation_fraction,
        }
    )
    if len(replay.q) == window:
        replay.checkpoint = quantize_checkpoint(reconstructed, "vblock", 32)
        replay.q.clear()
        replay.k.clear()
        replay.v.clear()
        replay.g.clear()
        replay.beta.clear()


def install_shadow_hook() -> None:
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
        active = CAPTURE_ENABLED and layer in SELECTED_LAYERS
        if not active:
            return original_method(
                self, mixed_qkv, b, a, core_attn_out, attn_metadata
            )
        CURRENT_CONTEXT = (self, layer)
        try:
            return original_method(
                self, mixed_qkv, b, a, core_attn_out, attn_metadata
            )
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
        if context is None:
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
        if mixed_qkv.shape[0] != 1 or int(ssm_state_indices[0].item()) <= 0:
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
        slot = int(ssm_state_indices[0].item())
        state_before = initial_state[slot : slot + 1].float().clone()
        inputs = _prepare_inputs(self, mixed_qkv, a, b)
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
        expected_output = out.reshape(1, -1, self.head_v_dim).float().clone()
        expected_state = initial_state[slot : slot + 1].float().clone()
        oracle_output, oracle_state = run_kernel(state_before, *inputs)
        oracle_output_error = _relative_l2(
            oracle_output[:, -1], expected_output
        )
        oracle_state_error = _relative_l2(oracle_state, expected_state)
        call = CALLS.get(layer, 0)
        CALLS[layer] = call + 1
        for window in WINDOWS:
            _run_shadow(
                layer,
                call,
                window,
                state_before,
                inputs,
                expected_output,
                expected_state,
                oracle_output_error,
                oracle_state_error,
            )
        return result

    qwen_gdn_linear_attn.QwenGatedDeltaNetAttention._forward_core_decode_non_spec = (
        wrapped_method
    )
    qwen_gdn_linear_attn.fused_recurrent_gated_delta_rule_packed_decode = (
        wrapped_recurrent
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=(
            "/root/.cache/huggingface/hub/"
            "models--Qwen--Qwen3.5-4B/snapshots/*"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    global CAPTURE_ENABLED

    args = parse_args()
    install_shadow_hook()
    from vllm import LLM, SamplingParams

    matches = glob.glob(args.model)
    model = matches[0] if len(matches) == 1 else args.model
    llm = LLM(
        model=model,
        dtype="bfloat16",
        max_model_len=512,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True,
    )
    CAPTURE_ENABLED = True
    llm.generate(
        ["Explain why recurrent states are useful during autoregressive decode."],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=False,
    )
    for window in WINDOWS:
        window_records = [item for item in RECORDS if item["window"] == window]
        for layer in sorted(SELECTED_LAYERS):
            records = [
                item for item in window_records if item["layer"] == layer
            ]
            if not records:
                continue
            summary = {"window": window, "layer": layer, "samples": len(records)}
            for name in (
                "output_relative_l2",
                "state_relative_l2",
                "oracle_output_relative_l2",
                "oracle_state_relative_l2",
            ):
                values = [float(item[name]) for item in records]
                summary[f"mean_{name}"] = sum(values) / len(values)
                summary[f"max_{name}"] = max(values)
            summary["max_checkpoint_saturation"] = max(
                float(item["checkpoint_saturation"]) for item in records
            )
            print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
