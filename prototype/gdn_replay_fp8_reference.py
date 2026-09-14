"""Correctness prototype for GDN replay with FP8 persistent checkpoints.

This uses an explicit FP32 implementation of the production GDN recurrence. It
measures numerical behavior and persistent-storage size; it is not a performance
benchmark for the eventual fused replay kernel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


@dataclass
class QuantizedCheckpoint:
    payload: torch.Tensor
    scale: torch.Tensor
    granularity: str
    tile_size: int
    saturation_fraction: float

    @property
    def storage_bytes(self) -> int:
        return self.payload.numel() * self.payload.element_size() + (
            self.scale.numel() * self.scale.element_size()
        )


def _safe_scale(amax: torch.Tensor, quant_max: float) -> torch.Tensor:
    return torch.where(amax > 0, amax / quant_max, torch.ones_like(amax))


def quantize_checkpoint(
    state: torch.Tensor, granularity: str, tile_size: int
) -> QuantizedCheckpoint:
    source = state.float()
    is_int8 = granularity.endswith("_int8")
    base_granularity = granularity.removesuffix("_int8")
    quant_max = 127.0 if is_int8 else FP8_MAX
    if base_granularity == "head":
        scale = _safe_scale(
            source.abs().amax(dim=(-2, -1), keepdim=True), quant_max
        )
        expanded_scale = scale
    elif base_granularity == "row":
        scale = _safe_scale(source.abs().amax(dim=-1, keepdim=True), quant_max)
        expanded_scale = scale
    elif base_granularity == "tile":
        value_dim, key_dim = source.shape[-2:]
        if value_dim % tile_size or key_dim % tile_size:
            raise ValueError("tile granularity requires dimensions divisible by tile_size")
        blocks = source.reshape(
            *source.shape[:-2],
            value_dim // tile_size,
            tile_size,
            key_dim // tile_size,
            tile_size,
        )
        block_scale = _safe_scale(
            blocks.abs().amax(dim=(-3, -1), keepdim=True), quant_max
        )
        expanded_scale = block_scale.expand_as(blocks).reshape_as(source)
        scale = block_scale.squeeze(-1).squeeze(-2).to(torch.float16)
    elif base_granularity == "vblock":
        value_dim, key_dim = source.shape[-2:]
        if value_dim % tile_size:
            raise ValueError(
                "vblock granularity requires value_dim divisible by tile_size"
            )
        blocks = source.reshape(
            *source.shape[:-2],
            value_dim // tile_size,
            tile_size,
            key_dim,
        )
        block_scale = _safe_scale(
            blocks.abs().amax(dim=(-2, -1), keepdim=True), quant_max
        )
        expanded_scale = block_scale.expand_as(blocks).reshape_as(source)
        scale = block_scale.squeeze(-1).squeeze(-1).to(torch.float16)
    else:
        raise ValueError(f"unsupported granularity: {granularity}")

    scale = scale.to(torch.float16)
    if base_granularity in ("head", "row"):
        expanded_scale = scale.float()
    normalized = source / expanded_scale.float()
    saturation_fraction = (normalized.abs() > quant_max).float().mean().item()
    if is_int8:
        payload = normalized.round().clamp(-quant_max, quant_max).to(torch.int8)
    else:
        payload = normalized.clamp(-quant_max, quant_max).to(FP8_DTYPE)
    return QuantizedCheckpoint(
        payload=payload,
        scale=scale,
        granularity=granularity,
        tile_size=tile_size,
        saturation_fraction=saturation_fraction,
    )


def dequantize_checkpoint(checkpoint: QuantizedCheckpoint) -> torch.Tensor:
    base_granularity = checkpoint.granularity.removesuffix("_int8")
    if base_granularity in ("head", "row"):
        expanded_scale = checkpoint.scale
    elif base_granularity == "tile":
        value_dim, key_dim = checkpoint.payload.shape[-2:]
        blocks = checkpoint.payload.reshape(
            *checkpoint.payload.shape[:-2],
            value_dim // checkpoint.tile_size,
            checkpoint.tile_size,
            key_dim // checkpoint.tile_size,
            checkpoint.tile_size,
        )
        scale = checkpoint.scale.unsqueeze(-1).unsqueeze(-3)
        expanded_scale = scale.expand_as(blocks).reshape_as(checkpoint.payload)
    else:
        value_dim, key_dim = checkpoint.payload.shape[-2:]
        blocks = checkpoint.payload.reshape(
            *checkpoint.payload.shape[:-2],
            value_dim // checkpoint.tile_size,
            checkpoint.tile_size,
            key_dim,
        )
        scale = checkpoint.scale.unsqueeze(-1).unsqueeze(-1)
        expanded_scale = scale.expand_as(blocks).reshape_as(checkpoint.payload)
    return checkpoint.payload.float() * expanded_scale.float()


def run_kernel(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scratch = state.float().clone()
    value_heads = v.shape[2]
    key_heads = k.shape[2]
    group_size = value_heads // key_heads
    scale = q.shape[-1] ** -0.5
    outputs = []
    for token in range(q.shape[1]):
        q_token = q[:, token].float().repeat_interleave(group_size, dim=1)
        k_token = k[:, token].float().repeat_interleave(group_size, dim=1)
        v_token = v[:, token].float()
        decay = torch.exp(g[:, token].float()).unsqueeze(-1).unsqueeze(-1)
        beta_token = beta[:, token].float().unsqueeze(-1)
        scratch = scratch * decay
        retrieved = (scratch * k_token.unsqueeze(-2)).sum(dim=-1)
        correction = (v_token - retrieved) * beta_token
        scratch = scratch + correction.unsqueeze(-1) * k_token.unsqueeze(-2)
        output = (scratch * q_token.unsqueeze(-2)).sum(dim=-1) * scale
        outputs.append(output.unsqueeze(1))
    return torch.cat(outputs, dim=1), scratch


def baseline_decode(
    initial_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state.clone()
    outputs = []
    for token in range(q.shape[1]):
        output, state = run_kernel(
            state,
            q[:, token : token + 1],
            k[:, token : token + 1],
            v[:, token : token + 1],
            g[:, token : token + 1],
            beta[:, token : token + 1],
        )
        outputs.append(output)
    return torch.cat(outputs, dim=1), state


def replay_decode(
    initial_state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    window: int,
    granularity: str | None,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    checkpoint = initial_state.clone()
    quantized = None
    max_saturation = 0.0
    if granularity is not None:
        quantized = quantize_checkpoint(checkpoint, granularity, tile_size)

    outputs = []
    ring_start = 0
    for token in range(q.shape[1]):
        if quantized is not None:
            checkpoint = dequantize_checkpoint(quantized)
        output, reconstructed = run_kernel(
            checkpoint,
            q[:, ring_start : token + 1],
            k[:, ring_start : token + 1],
            v[:, ring_start : token + 1],
            g[:, ring_start : token + 1],
            beta[:, ring_start : token + 1],
        )
        outputs.append(output[:, -1:])
        if token - ring_start + 1 == window:
            checkpoint = reconstructed
            if granularity is not None:
                quantized = quantize_checkpoint(
                    checkpoint, granularity, tile_size
                )
                max_saturation = max(
                    max_saturation, quantized.saturation_fraction
                )
            ring_start = token + 1

    final_state = reconstructed
    if quantized is not None and ring_start == q.shape[1]:
        final_state = dequantize_checkpoint(quantized)
    checkpoint_bytes = (
        quantized.storage_bytes
        if quantized is not None
        else checkpoint.numel() * checkpoint.element_size()
    )
    return (
        torch.cat(outputs, dim=1),
        final_state,
        max_saturation,
        checkpoint_bytes,
    )


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    delta = actual_f - expected_f
    denominator = expected_f.norm().clamp_min(1e-12)
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "relative_l2": (delta.norm() / denominator).item(),
        "cosine": F.cosine_similarity(
            actual_f.flatten(), expected_f.flatten(), dim=0
        ).item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--value-heads", type=int, default=4)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--windows", default="2,4,8,16")
    parser.add_argument(
        "--granularities", default="head,row,vblock,tile,vblock_int8"
    )
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.value_heads % args.heads:
        raise ValueError("value_heads must be divisible by heads")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    activation_dtype = torch.bfloat16
    shape_qk = (args.batch, args.tokens, args.heads, args.key_dim)
    shape_v = (args.batch, args.tokens, args.value_heads, args.value_dim)
    q = F.normalize(torch.randn(shape_qk, device=device), dim=-1).to(
        activation_dtype
    )
    k = F.normalize(torch.randn(shape_qk, device=device), dim=-1).to(
        activation_dtype
    )
    v = torch.randn(shape_v, device=device, dtype=activation_dtype)
    a = torch.randn(
        args.batch,
        args.tokens,
        args.value_heads,
        device=device,
        dtype=torch.float32,
    )
    a_log = torch.rand(args.value_heads, device=device, dtype=torch.float32)
    dt_bias = torch.rand(
        args.value_heads, device=device, dtype=torch.float32
    ) - 4.0
    g = (-a_log.exp() * F.softplus(a + dt_bias)).to(activation_dtype)
    beta = torch.sigmoid(
        torch.randn(
            args.batch,
            args.tokens,
            args.value_heads,
            device=device,
            dtype=torch.float32,
        )
    ).to(activation_dtype)
    initial_state = (
        torch.randn(
            args.batch,
            args.value_heads,
            args.value_dim,
            args.key_dim,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    )

    baseline_output, baseline_state = baseline_decode(
        initial_state, q, k, v, g, beta
    )
    state_elements = initial_state.numel()
    ring_elements_per_token = args.batch * (
        args.heads * args.key_dim
        + args.value_heads * (args.value_dim + 2)
    )
    bf16_active_bytes = state_elements * 2
    config = vars(args) | {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "activation_dtype": str(activation_dtype),
        "state_dtype": str(initial_state.dtype),
        "bf16_active_state_bytes": bf16_active_bytes,
        "ring_elements_per_token": ring_elements_per_token,
        "decay_min": torch.exp(g.float()).min().item(),
        "decay_mean": torch.exp(g.float()).mean().item(),
        "decay_max": torch.exp(g.float()).max().item(),
    }
    print(json.dumps({"type": "config", **config}, sort_keys=True))

    windows = [int(value) for value in args.windows.split(",")]
    granularities = args.granularities.split(",")
    for window in windows:
        replay_output, replay_state, _, checkpoint_bytes = replay_decode(
            initial_state,
            q,
            k,
            v,
            g,
            beta,
            window,
            None,
            args.tile_size,
        )
        print(
            json.dumps(
                {
                    "type": "result",
                    "mode": "fp32_replay",
                    "window": window,
                    "output_error": error_metrics(
                        replay_output, baseline_output
                    ),
                    "state_error": error_metrics(replay_state, baseline_state),
                    "checkpoint_bytes": checkpoint_bytes,
                },
                sort_keys=True,
            )
        )
        for granularity in granularities:
            fp8_output, fp8_state, saturation, checkpoint_bytes = replay_decode(
                initial_state,
                q,
                k,
                v,
                g,
                beta,
                window,
                granularity,
                args.tile_size,
            )
            ring_bytes = ring_elements_per_token * window * 2
            persistent_bytes = checkpoint_bytes + ring_bytes
            print(
                json.dumps(
                    {
                        "type": "result",
                        "mode": "fp8_checkpoint",
                        "granularity": granularity,
                        "window": window,
                        "output_error": error_metrics(
                            fp8_output, baseline_output
                        ),
                        "state_error": error_metrics(fp8_state, baseline_state),
                        "max_saturation_fraction": saturation,
                        "checkpoint_bytes": checkpoint_bytes,
                        "ring_bytes": ring_bytes,
                        "persistent_bytes": persistent_bytes,
                        "bytes_vs_bf16_active": (
                            persistent_bytes / bf16_active_bytes
                        ),
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
