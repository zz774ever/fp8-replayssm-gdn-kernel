"""Collect real Qwen3.5 recurrent-state statistics during eager decoding."""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "kernels"))

from gdn_replay_fp8_reference import (
    dequantize_checkpoint,
    quantize_checkpoint,
)


CAPTURE_ENABLED = False
RECORDS: list[dict[str, float | int]] = []
CALL_COUNTS: defaultdict[int, int] = defaultdict(int)
SELECTED_LAYERS = {0, 1, 2, 4, 8, 16, 24, 30}


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = actual.float() - expected.float()
    return (delta.norm() / expected.float().norm().clamp_min(1e-12)).item()


def _record_state(
    prefix: str, state_cache: torch.Tensor, state_indices: torch.Tensor
) -> None:
    match = re.search(r"layers\.(\d+)", prefix)
    if match is None:
        return
    layer = int(match.group(1))
    if layer not in SELECTED_LAYERS:
        return
    # Block zero is vLLM's NULL_BLOCK_ID. Restrict statistics to unique cache
    # slots referenced by this forward pass so scheduler padding and stale
    # request state cannot contaminate the measurements.
    slots = torch.unique(state_indices.flatten())
    slots = slots[(slots > 0) & (slots < state_cache.shape[0])]
    for slot_tensor in slots:
        slot = int(slot_tensor.item())
        state = state_cache[slot : slot + 1].float()
        absolute = state.abs()
        if absolute.max().item() == 0:
            continue
        quantiles = torch.quantile(
            absolute.flatten(),
            torch.tensor((0.5, 0.9, 0.99, 0.999), device=state.device),
        )
        fp8 = quantize_checkpoint(state, "vblock", 32)
        int8 = quantize_checkpoint(state, "vblock_int8", 32)
        RECORDS.append(
            {
                "layer": layer,
                "call": CALL_COUNTS[layer],
                "slot": slot,
                "amax": absolute.max().item(),
                "mean_abs": absolute.mean().item(),
                "rms": state.square().mean().sqrt().item(),
                "p50_abs": quantiles[0].item(),
                "p90_abs": quantiles[1].item(),
                "p99_abs": quantiles[2].item(),
                "p999_abs": quantiles[3].item(),
                "fp8_relative_l2": _relative_l2(
                    dequantize_checkpoint(fp8), state
                ),
                "int8_relative_l2": _relative_l2(
                    dequantize_checkpoint(int8), state
                ),
            }
        )
        CALL_COUNTS[layer] += 1


def install_capture_hook() -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn

    original = qwen_gdn_linear_attn.QwenGatedDeltaNetAttention.forward_cuda

    def wrapped(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.forward_context import get_forward_context

        metadata_raw = get_forward_context().attn_metadata
        metadata = (
            metadata_raw.get(self.prefix)
            if isinstance(metadata_raw, dict)
            else None
        )
        output = original(self, hidden_states)
        if CAPTURE_ENABLED and hasattr(self, "kv_cache") and metadata is not None:
            indices = []
            if metadata.non_spec_state_indices_tensor is not None:
                indices.append(metadata.non_spec_state_indices_tensor)
            if metadata.spec_state_indices_tensor is not None:
                indices.append(metadata.spec_state_indices_tensor)
            if indices:
                _record_state(
                    self.prefix,
                    self.kv_cache[1],
                    torch.cat([item.flatten() for item in indices]),
                )
        return output

    qwen_gdn_linear_attn.QwenGatedDeltaNetAttention.forward_cuda = wrapped


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
    install_capture_hook()
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
    for record in RECORDS:
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
