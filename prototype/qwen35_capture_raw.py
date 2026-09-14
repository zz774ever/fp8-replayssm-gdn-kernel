"""Capture raw GDN decode inputs from a real Qwen3.5 run.

The earlier capture (`qwen35_drift_study.py`) stored the *prepared* recurrence
inputs -- already-normalised q/k and pre-computed g/beta -- because the drift
study consumed them directly. A full-contract benchmark cannot use that: the
point is to make the replay path do the same work the production operator does
(q/k L2 normalisation, gating, ring append) starting from the same raw inputs.

So this captures, per GDN layer and decode step:

* ``mixed_qkv`` -- post-convolution, pre-normalisation packed q|k|v
* ``a``, ``b``  -- raw gating projections
* the recurrent state at the first decode step (the prefill state)
* the production operator's own output for that step (a reference to validate
  the replay path against on real data)

plus the per-layer constants (A_log, dt_bias, scale).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import torch

CAPTURE = False
SELECTED_LAYERS: set[int] = set()
CURRENT_CONTEXT: tuple[object, int] | None = None
RECORDS: dict[int, dict[str, object]] = {}


def _layer_number(prefix: str) -> int | None:
    match = re.search(r"layers\.(\d+)", prefix)
    return int(match.group(1)) if match else None


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
        if context is None or not CAPTURE:
            return call_original()
        self, layer = context
        slot = int(ssm_state_indices[0].item())
        state_before = None
        if layer not in RECORDS and slot > 0 and mixed_qkv.shape[0] == 1:
            # Read the pre-state before the in-place kernel update.
            state_before = initial_state[slot : slot + 1].float().clone().to("cpu")
        result = call_original()
        if mixed_qkv.shape[0] == 1 and slot > 0:
            record = RECORDS.setdefault(
                layer,
                {
                    "mixed_qkv": [],
                    "a": [],
                    "b": [],
                    "production_out": [],
                    "state": None,
                    "a_log": A_log.detach().float().cpu(),
                    "dt_bias": dt_bias.detach().float().cpu(),
                    "scale": float(scale),
                    "use_qk_l2norm_in_kernel": bool(use_qk_l2norm_in_kernel),
                },
            )
            if state_before is not None:
                record["state"] = state_before
            record["mixed_qkv"].append(
                mixed_qkv.detach().to("cpu", torch.bfloat16)
            )
            record["a"].append(a.detach().to("cpu", torch.float32))
            record["b"].append(b.detach().to("cpu", torch.float32))
            record["production_out"].append(
                out.detach().reshape(1, -1).to("cpu", torch.bfloat16)
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
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=4352)
    parser.add_argument("--layers", default="0,8,16,24")
    parser.add_argument("--save", default="/root/qwen35_capture_raw_p0.pt")
    parser.add_argument(
        "--prompt",
        default=(
            "Explain in two sentences why recurrent state caching matters for "
            "autoregressive decoding."
        ),
    )
    return parser.parse_args()


def main() -> None:
    global CAPTURE, SELECTED_LAYERS

    args = parse_args()
    SELECTED_LAYERS = {
        int(value) for value in args.layers.split(",") if value != ""
    }
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
    torch.save(RECORDS, args.save)
    print(
        json.dumps(
            {
                "type": "capture",
                "model": model,
                "saved_to": args.save,
                "layers": sorted(RECORDS),
                "steps_per_layer": {
                    str(key): len(value["mixed_qkv"])
                    for key, value in RECORDS.items()
                },
                "state_shape": list(RECORDS[min(RECORDS)]["state"].shape),
                "mixed_qkv_shape": list(
                    RECORDS[min(RECORDS)]["mixed_qkv"][0].shape
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
