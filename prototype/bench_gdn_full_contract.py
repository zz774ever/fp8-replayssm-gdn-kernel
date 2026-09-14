"""Full-contract comparison: replay vs the production operator, same work on both sides.

The earlier comparison (`bench_gdn_vs_production.py`) timed the replay path from
already-normalised q/k and pre-computed gating, while the production operator
normalises and gates inline. That is a fair *core-kernel* comparison but an
optimistic *operator* comparison.

This script closes that gap. Both sides start from the same raw inputs and are
required to end in the same place:

    production: raw mixed_qkv + a/b -> q/k norm -> gating -> state RMW -> out
    replay:     raw mixed_qkv + a/b -> q/k norm -> gating -> ring append
                                     -> replay -> flush/requantise every L -> out

Two numbers are reported per cell:

    core speedup = production / (replay + amortised flush)
    full speedup = production / (prep + replay + amortised flush)

Inputs are synthetic by default, or come from a captured real Qwen3.5 decode
(`--capture`), which is what answers "were your synthetic inputs unrealistically
favourable?".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "kernels"))

from gdn_replay_fp8_reference import (  # noqa: E402
    error_metrics,
    quantize_checkpoint,
    run_kernel,
)
from gdn_replayssm_fp8_kernel import (  # noqa: E402
    gdn_prep_ring,
    gdn_replay_fp8,
    gdn_replay_fp8_split,
)

from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.triton_utils import triton  # noqa: E402

TILE = 32


def synthetic_inputs(batch, heads, value_heads, key_dim, value_dim, steps, seed):
    """Raw production-layout inputs: one token per sequence per step."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = torch.randn(batch, steps, heads, key_dim, device=device, dtype=dtype)
    k = torch.randn(batch, steps, heads, key_dim, device=device, dtype=dtype)
    v = torch.randn(batch, steps, value_heads, value_dim, device=device, dtype=dtype)
    return {
        "mixed_qkv": torch.cat(
            [q.reshape(batch, steps, -1), k.reshape(batch, steps, -1), v.reshape(batch, steps, -1)],
            dim=-1,
        ).contiguous(),
        "a": torch.randn(batch, steps, value_heads, device=device, dtype=torch.float32),
        "b": torch.randn(batch, steps, value_heads, device=device, dtype=torch.float32),
        "a_log": torch.rand(value_heads, device=device, dtype=torch.float32),
        "dt_bias": torch.rand(value_heads, device=device, dtype=torch.float32) - 4.0,
        "state": (
            torch.randn(batch, value_heads, value_dim, key_dim, device=device) * 0.02
        ),
        "source": "synthetic",
    }


_CAPTURE_CACHE: dict[str, object] = {}


def captured_inputs(path, batch, layer, steps):
    """Real captured decode inputs, replicated across the batch dimension.

    The capture is batch-1 by construction. Replication is sound for timing
    (this kernel has no data-dependent control flow) and it keeps the
    checkpoint/ring pair self-consistent, which is what validation needs.
    """
    if path not in _CAPTURE_CACHE:
        _CAPTURE_CACHE[path] = torch.load(path, weights_only=False)
    records = _CAPTURE_CACHE[path]
    record = records[layer]
    steps = min(steps, len(record["mixed_qkv"]))
    mixed = torch.stack([item[0] for item in record["mixed_qkv"][:steps]]).to("cuda")
    a = torch.stack([item[0] for item in record["a"][:steps]]).to("cuda")
    b = torch.stack([item[0] for item in record["b"][:steps]]).to("cuda")
    state = record["state"].squeeze(0).to("cuda").float()
    return {
        "mixed_qkv": mixed.unsqueeze(0).expand(batch, -1, -1).contiguous(),
        "a": a.unsqueeze(0).expand(batch, -1, -1).contiguous(),
        "b": b.unsqueeze(0).expand(batch, -1, -1).contiguous(),
        "a_log": record["a_log"].to("cuda"),
        "dt_bias": record["dt_bias"].to("cuda"),
        "state": state.unsqueeze(0).expand(batch, -1, -1, -1).contiguous().float(),
        "production_out": torch.stack(
            [item[0] for item in record["production_out"][:steps]]
        ).to("cuda"),
        "source": f"capture:{path}:layer{layer}",
    }


def build_buffers(tensors, heads, value_heads, key_dim, value_dim, steps):
    batch = tensors["mixed_qkv"].shape[0]
    device = tensors["mixed_qkv"].device
    dtype = tensors["mixed_qkv"].dtype
    state_bf16 = tensors["state"].to(dtype)
    return {
        "checkpoint": quantize_checkpoint(tensors["state"].float(), "vblock", TILE),
        "k_ring": torch.empty(batch, steps, heads, key_dim, device=device, dtype=dtype),
        "v_ring": torch.empty(
            batch, steps, value_heads, value_dim, device=device, dtype=dtype
        ),
        "g_ring": torch.empty(batch, steps, value_heads, device=device, dtype=dtype),
        "beta_ring": torch.empty(batch, steps, value_heads, device=device, dtype=dtype),
        "q_out": torch.empty(batch, heads, key_dim, device=device, dtype=dtype),
        "pos": torch.zeros(batch, device=device, dtype=torch.int32),
        "state_cache": torch.cat(
            [torch.zeros_like(state_bf16[:1]), state_bf16], dim=0
        ).contiguous(),
        "state_indices": torch.arange(1, batch + 1, device=device, dtype=torch.int32),
        "out": torch.empty(
            batch, 1, value_heads, value_dim, device=device, dtype=dtype
        ),
    }


def prep_step(tensors, buffers, position):
    """Append one real token to the ring (q/k norm + gating + v copy)."""
    buffers["pos"].fill_(position)
    gdn_prep_ring(
        tensors["mixed_qkv"][:, position].contiguous(),
        tensors["a"][:, position].contiguous(),
        tensors["b"][:, position].contiguous(),
        tensors["a_log"],
        tensors["dt_bias"],
        buffers["pos"],
        buffers["q_out"],
        buffers["k_ring"],
        buffers["v_ring"],
        buffers["g_ring"],
        buffers["beta_ring"],
        num_warps=buffers.get("prep_warps", 1),
    )


def production_call(tensors, buffers, key_dim, position):
    batch = tensors["mixed_qkv"].shape[0]
    return lambda: fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=tensors["mixed_qkv"][:, position].reshape(batch, -1).contiguous(),
        a=tensors["a"][:, position].contiguous(),
        b=tensors["b"][:, position].contiguous(),
        A_log=tensors["a_log"],
        dt_bias=tensors["dt_bias"],
        scale=key_dim**-0.5,
        initial_state=buffers["state_cache"],
        out=buffers["out"],
        ssm_state_indices=buffers["state_indices"],
        use_qk_l2norm_in_kernel=True,
    )


def validate(tensors, buffers, key_dim, window, block_v):
    """Replay-full vs production vs fp32 reference, all on the same raw inputs."""
    batch = tensors["mixed_qkv"].shape[0]
    for position in range(window):
        prep_step(tensors, buffers, position)
    initial_state = buffers["state_cache"].clone()
    # The production operator advances the state by exactly one token per call,
    # so reaching the end of the window requires window calls. Only the last
    # output is comparable with the replay result, which covers the whole ring.
    for position in range(window):
        production_call(tensors, buffers, key_dim, position)()
    torch.cuda.synchronize()
    production_out = buffers["out"].reshape(batch, -1).float()
    # fp32 reference: unquantised state, exact recurrence, same ring inputs.
    reference_state = initial_state[1:].float()
    reference_output, _ = run_kernel(
        reference_state,
        # run_kernel loops over q.shape[1], so q must span the whole ring;
        # every position carries the same query and only the last output is used.
        buffers["q_out"].float().unsqueeze(1).expand(-1, window, -1, -1),
        buffers["k_ring"][:, :window].float(),
        buffers["v_ring"][:, :window].float(),
        buffers["g_ring"][:, :window].float(),
        buffers["beta_ring"][:, :window].float(),
    )
    reference = reference_output[:, -1].reshape(batch, -1)
    valid = torch.full((batch,), window, device="cuda", dtype=torch.int32)
    no_flush = torch.zeros(batch, device="cuda", dtype=torch.bool)
    replay_out = gdn_replay_fp8(
        buffers["checkpoint"].payload,
        buffers["checkpoint"].scale,
        buffers["q_out"],
        buffers["k_ring"][:, :window],
        buffers["v_ring"][:, :window],
        buffers["g_ring"][:, :window],
        buffers["beta_ring"][:, :window],
        valid,
        no_flush,
        block_v=block_v,
    ).reshape(batch, -1)
    # Restore the cache the production calls advanced, so later timing cells
    # start from a predictable state.
    buffers["state_cache"].copy_(initial_state)
    return {
        "production_vs_fp32": error_metrics(production_out, reference)["relative_l2"],
        "replay_full_vs_fp32": error_metrics(replay_out, reference)["relative_l2"],
        "replay_full_vs_production": error_metrics(replay_out, production_out)[
            "relative_l2"
        ],
    }


def bench_cell(tensors, buffers, key_dim, window, block_v, split, block_kc, num_warps):
    """Time one flush cycle: prep + replay for every step, flush on the last."""
    batch = tensors["mixed_qkv"].shape[0]
    # Fill the ring once so the replay kernels always see a full ring of data.
    for position in range(window):
        prep_step(tensors, buffers, position)
    torch.cuda.synchronize()

    production_ms = triton.testing.do_bench(
        production_call(tensors, buffers, key_dim, window - 1), warmup=25, rep=100
    )
    prep_ms = triton.testing.do_bench(
        lambda: prep_step(tensors, buffers, window - 1), warmup=25, rep=100
    )
    valid_lengths = [
        torch.full((batch,), length, device="cuda", dtype=torch.int32)
        for length in range(1, window + 1)
    ]
    no_flush = torch.zeros(batch, device="cuda", dtype=torch.bool)
    flush = torch.ones(batch, device="cuda", dtype=torch.bool)

    def replay_call(position):
        if split:
            return lambda: gdn_replay_fp8_split(
                buffers["checkpoint"].payload,
                buffers["checkpoint"].scale,
                buffers["q_out"],
                buffers["k_ring"],
                buffers["v_ring"],
                buffers["g_ring"],
                buffers["beta_ring"],
                valid_lengths[position],
                buffers["_tq"],
                buffers["_coef"],
                block_v=block_v,
                block_kc=block_kc,
                num_warps=num_warps,
            )
        return lambda: gdn_replay_fp8(
            buffers["checkpoint"].payload,
            buffers["checkpoint"].scale,
            buffers["q_out"],
            buffers["k_ring"],
            buffers["v_ring"],
            buffers["g_ring"],
            buffers["beta_ring"],
            valid_lengths[position],
            no_flush,
            block_v=block_v,
            num_warps=num_warps,
        )

    replay_total = 0.0
    for length in range(1, window):
        replay_total += triton.testing.do_bench(
            replay_call(length - 1), warmup=10, rep=50
        )
    flush_ms = triton.testing.do_bench(
        lambda: gdn_replay_fp8(
            buffers["checkpoint"].payload,
            buffers["checkpoint"].scale,
            buffers["q_out"],
            buffers["k_ring"],
            buffers["v_ring"],
            buffers["g_ring"],
            buffers["beta_ring"],
            valid_lengths[window - 1],
            flush,
            block_v=32,
        ),
        warmup=10,
        rep=50,
    )
    core_ms = (replay_total + flush_ms) / window
    full_ms = (replay_total + flush_ms + window * prep_ms) / window
    return {
        "production_ms": production_ms,
        "prep_ms": prep_ms,
        "core_ms": core_ms,
        "full_ms": full_ms,
        "core_speedup": production_ms / core_ms,
        "full_speedup": production_ms / full_ms,
        "prep_share": (prep_ms * window) / (replay_total + flush_ms + window * prep_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="8,16,32,64")
    parser.add_argument("--window", default="4,8,16")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capture", default=None, help="Raw capture .pt to drive inputs")
    parser.add_argument("--capture-layer", type=int, default=0)
    parser.add_argument("--capture-steps", type=int, default=2048)
    parser.add_argument(
        "--prep-warps",
        default="1",
        help="Comma-separated warp counts to sweep for the prep kernel.",
    )
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    windows = [int(value) for value in args.window.split(",")]
    prep_warps = [int(value) for value in args.prep_warps.split(",")]

    print(
        json.dumps(
            {
                "type": "config",
                "device": torch.cuda.get_device_name(0),
                "source": args.capture or "synthetic",
                "note": "core = replay+flush; full = prep+replay+flush",
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
            steps = max(window, 8)
            if args.capture:
                tensors = captured_inputs(
                    # Only the window is needed: the capture is batch-1 and one
                    # real window is replicated across the batch dimension.
                    args.capture,
                    batch,
                    args.capture_layer,
                    steps,
                )
            else:
                tensors = synthetic_inputs(
                    batch,
                    args.heads,
                    args.value_heads,
                    args.key_dim,
                    args.value_dim,
                    steps,
                    args.seed,
                )
            buffers = build_buffers(
                tensors,
                args.heads,
                args.value_heads,
                args.key_dim,
                args.value_dim,
                steps,
            )
            buffers["_tq"] = torch.empty(
                batch, args.value_heads, args.key_dim, device="cuda", dtype=torch.float32
            )
            buffers["_coef"] = torch.empty(
                batch, args.value_heads, steps, device="cuda", dtype=torch.float32
            )
            if batch == batches[0]:
                print(
                    json.dumps(
                        {
                            "type": "validation",
                            "window": window,
                            "batch": batch,
                            **validate(tensors, buffers, args.key_dim, window, 64),
                        }
                    ),
                    flush=True,
                )
            for prep_warps_value in prep_warps:
                buffers["prep_warps"] = prep_warps_value
                for label, split, block_v, block_kc in (
                    ("tiled_v64", False, 64, 128),
                    ("tiled_v128", False, 128, 128),
                    ("split_v128", True, 128, 128),
                ):
                    result = bench_cell(
                        tensors,
                        buffers,
                        args.key_dim,
                        window,
                        block_v,
                        split,
                        block_kc,
                        4,
                    )
                    print(
                        json.dumps(
                            {
                                "type": "full_contract",
                                "window": window,
                                "batch": batch,
                                "variant": label,
                                "prep_warps": prep_warps_value,
                                "source": tensors["source"],
                                **result,
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
