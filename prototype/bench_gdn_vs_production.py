"""Compare the production fused GDN decode operator against the replay kernel.

This is the baseline a kernel project is judged by: not a home-made BF16 step
kernel, but the operator vLLM actually runs today --
``fused_recurrent_gated_delta_rule_packed_decode`` from
``vllm/third_party/flash_linear_attention``.

All paths are driven from the same inputs and validated against the same fp32
reference, in one process and one timing harness:

* production: packed decode with in-kernel q/k L2 normalisation and gating, and
  a full bf16 state read + write per token;
* replay: FP8 checkpoint + ring, rebuilding the state only on flush, amortised
  over a full flush cycle (the ring grows 1..L);
* our own BF16 read-update-write step kernel, kept as a secondary reference.

Usage: python bench_gdn_vs_production.py --batches 1,4,16,64 --window 4,16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "kernels"))

from gdn_replay_fp8_reference import (  # noqa: E402
    error_metrics,
    quantize_checkpoint,
    run_kernel,
)
from gdn_replayssm_fp8_kernel import (  # noqa: E402
    gdn_bf16_step,
    gdn_replay_fp8,
    gdn_replay_fp8_split,
)

from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.triton_utils import triton  # noqa: E402

TILE = 32


def build_inputs(batch, heads, value_heads, key_dim, value_dim, seed):
    """Inputs in the production layout, plus a recurrent-state cache."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = torch.randn(batch, heads, key_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, key_dim, device=device, dtype=dtype)
    v = torch.randn(batch, value_heads, value_dim, device=device, dtype=dtype)
    a = torch.randn(batch, value_heads, device=device, dtype=torch.float32)
    b = torch.randn(batch, value_heads, device=device, dtype=torch.float32)
    a_log = torch.rand(value_heads, device=device, dtype=torch.float32)
    dt_bias = torch.rand(value_heads, device=device, dtype=torch.float32) - 4.0
    return {
        "q": q,
        "k": k,
        "v": v,
        "a": a,
        "b": b,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "mixed_qkv": torch.cat(
            [q.reshape(batch, -1), k.reshape(batch, -1), v.reshape(batch, -1)], dim=-1
        ).contiguous(),
        # Slot 0 is vLLM's null block, so real sequences start at index 1.
        "state_cache": (
            torch.randn(
                batch + 1,
                value_heads,
                value_dim,
                key_dim,
                device=device,
                dtype=torch.float32,
            )
            * 0.02
        ).to(dtype),
        "state_indices": torch.arange(1, batch + 1, device=device, dtype=torch.int32),
        "out": torch.empty(
            batch, 1, value_heads, value_dim, device=device, dtype=dtype
        ),
    }


def replay_tensors(data, window):
    """Everything the replay path needs, derived from the same inputs."""
    batch = data["q"].shape[0]
    device = data["q"].device
    dtype = data["q"].dtype
    q_norm = F.normalize(data["q"].float(), dim=-1)
    k_norm = F.normalize(data["k"].float(), dim=-1)
    # g is the raw log-decay the recurrence consumes; run_kernel applies exp().
    g = -torch.exp(data["a_log"].float()) * F.softplus(
        data["a"].float() + data["dt_bias"].float()
    )
    beta = torch.sigmoid(data["b"].float())
    state = data["state_cache"][1:].float()
    return {
        "q_norm": q_norm,
        "q_bf16": q_norm.to(dtype).contiguous(),
        "k_norm": k_norm,
        "g": g,
        "beta": beta,
        "state_before": state,
        "checkpoint": quantize_checkpoint(state, "vblock", TILE),
        "k_ring": k_norm.unsqueeze(1).to(dtype).expand(batch, window, -1, -1).contiguous(),
        "v_ring": data["v"].unsqueeze(1).expand(batch, window, -1, -1).contiguous(),
        "g_ring": g.unsqueeze(1).to(dtype).expand(batch, window, -1).contiguous(),
        "beta_ring": beta.unsqueeze(1).to(dtype).expand(batch, window, -1).contiguous(),
        "no_flush": torch.zeros(batch, device=device, dtype=torch.bool),
        "flush": torch.ones(batch, device=device, dtype=torch.bool),
    }


def validate(data, replay, key_dim, block_v=32):
    """Production operator and replay kernel against the same fp32 reference."""
    batch = data["q"].shape[0]
    reference_output, _ = run_kernel(
        replay["state_before"],
        replay["q_norm"].unsqueeze(1),
        replay["k_norm"].unsqueeze(1),
        data["v"].float().unsqueeze(1),
        replay["g"].unsqueeze(1),
        replay["beta"].unsqueeze(1),
    )
    reference = reference_output[:, -1].reshape(batch, -1)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=data["mixed_qkv"],
        a=data["a"],
        b=data["b"],
        A_log=data["a_log"],
        dt_bias=data["dt_bias"],
        scale=key_dim**-0.5,
        initial_state=data["state_cache"],
        out=data["out"],
        ssm_state_indices=data["state_indices"],
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    production = data["out"].reshape(batch, -1).float()
    length_one = torch.ones(batch, device="cuda", dtype=torch.int32)
    replay_out = gdn_replay_fp8(
        replay["checkpoint"].payload,
        replay["checkpoint"].scale,
        replay["q_bf16"],
        replay["k_ring"][:, :1],
        replay["v_ring"][:, :1],
        replay["g_ring"][:, :1],
        replay["beta_ring"][:, :1],
        length_one,
        replay["no_flush"],
        block_v=block_v,
    ).reshape(batch, -1)
    return {
        "production_vs_fp32_output_rel_l2": error_metrics(production, reference)[
            "relative_l2"
        ],
        "replay_vs_fp32_output_rel_l2": error_metrics(replay_out, reference)[
            "relative_l2"
        ],
        "production_vs_replay_output_rel_l2": error_metrics(production, replay_out)[
            "relative_l2"
        ],
    }


def production_call(data, key_dim):
    return lambda: fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=data["mixed_qkv"],
        a=data["a"],
        b=data["b"],
        A_log=data["a_log"],
        dt_bias=data["dt_bias"],
        scale=key_dim**-0.5,
        initial_state=data["state_cache"],
        out=data["out"],
        ssm_state_indices=data["state_indices"],
        use_qk_l2norm_in_kernel=True,
    )


def make_replay_call(data, replay, length, block_v, split, block_kc, buffers):
    batch = data["q"].shape[0]
    valid = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    if split:
        return lambda: gdn_replay_fp8_split(
            replay["checkpoint"].payload,
            replay["checkpoint"].scale,
            replay["q_bf16"],
            replay["k_ring"],
            replay["v_ring"],
            replay["g_ring"],
            replay["beta_ring"],
            valid,
            buffers[0],
            buffers[1],
            block_v=block_v,
            block_kc=block_kc,
        )
    return lambda: gdn_replay_fp8(
        replay["checkpoint"].payload,
        replay["checkpoint"].scale,
        replay["q_bf16"],
        replay["k_ring"],
        replay["v_ring"],
        replay["g_ring"],
        replay["beta_ring"],
        valid,
        replay["no_flush"],
        block_v=block_v,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,4,16,64")
    parser.add_argument("--window", default="4,16")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    windows = [int(value) for value in args.window.split(",")]

    print(
        json.dumps(
            {
                "type": "config",
                "device": torch.cuda.get_device_name(0),
                "production_op": "flash_linear_attention."
                "fused_recurrent_gated_delta_rule_packed_decode",
                "geometry": {
                    "heads": args.heads,
                    "value_heads": args.value_heads,
                    "K": args.key_dim,
                    "V": args.value_dim,
                },
            }
        ),
        flush=True,
    )
    for window in windows:
        for batch in batches:
            data = build_inputs(
                batch,
                args.heads,
                args.value_heads,
                args.key_dim,
                args.value_dim,
                args.seed,
            )
            replay = replay_tensors(data, window)
            if batch == batches[0]:
                print(
                    json.dumps(
                        {
                            "type": "validation",
                            "window": window,
                            "batch": batch,
                            **validate(data, replay, args.key_dim),
                        }
                    ),
                    flush=True,
                )
            production_ms = triton.testing.do_bench(
                production_call(data, args.key_dim), warmup=25, rep=100
            )
            # Hoist every allocation/cast out of the timed closure: leaving them
            # inside measured the allocator, not the kernel.
            bf16_state = data["state_cache"][1:].clone()
            bf16_q = replay["q_norm"].to(torch.bfloat16)
            bf16_k = replay["k_norm"].to(torch.bfloat16)
            bf16_g = replay["g"].to(torch.bfloat16)
            bf16_beta = replay["beta"].to(torch.bfloat16)
            our_bf16_ms = triton.testing.do_bench(
                lambda: gdn_bf16_step(
                    bf16_state, bf16_q, bf16_k, data["v"], bf16_g, bf16_beta
                ),
                warmup=25,
                rep=100,
            )
            buffers = (
                torch.empty(
                    batch,
                    args.value_heads,
                    args.key_dim,
                    device="cuda",
                    dtype=torch.float32,
                ),
                torch.empty(
                    batch, args.value_heads, window, device="cuda", dtype=torch.float32
                ),
            )
            flush_ms = triton.testing.do_bench(
                lambda: gdn_replay_fp8(
                    replay["checkpoint"].payload,
                    replay["checkpoint"].scale,
                    replay["q_bf16"],
                    replay["k_ring"],
                    replay["v_ring"],
                    replay["g_ring"],
                    replay["beta_ring"],
                    torch.full((batch,), window, device="cuda", dtype=torch.int32),
                    replay["flush"],
                    block_v=32,
                ),
                warmup=10,
                rep=50,
            )
            results = {}
            for label, split, block_v, block_kc in (
                ("tiled_v64", False, 64, 128),
                ("tiled_v128", False, 128, 128),
                ("split_v128", True, 128, 128),
            ):
                total = 0.0
                for length in range(1, window):
                    total += triton.testing.do_bench(
                        make_replay_call(
                            data, replay, length, block_v, split, block_kc, buffers
                        ),
                        warmup=10,
                        rep=50,
                    )
                cycle_ms = (total + flush_ms) / window
                results[label] = {
                    "cycle_ms": cycle_ms,
                    "speedup": production_ms / cycle_ms,
                }
            best_label, best = max(
                results.items(), key=lambda item: item[1]["speedup"]
            )
            print(
                json.dumps(
                    {
                        "type": "compare",
                        "window": window,
                        "batch": batch,
                        "production_ms": production_ms,
                        "our_bf16_step_ms": our_bf16_ms,
                        "our_bf16_vs_production": production_ms / our_bf16_ms,
                        **{
                            f"{label}_cycle_ms": value["cycle_ms"]
                            for label, value in results.items()
                        },
                        **{
                            f"{label}_speedup": value["speedup"]
                            for label, value in results.items()
                        },
                        "best_variant": best_label,
                        "best_speedup_vs_production": best["speedup"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
