"""Benchmark the GDN ReplaySSM FP8 kernel against a BF16 state read-update-write step.

The comparison is deliberately same-author and same-style: both kernels are
Triton, both compute the identical one-token GDN output, and both write back the
state they consumed. The only difference is *where the state lives*:

* BF16 baseline: read 1 MB state + write 1 MB state per token per layer.
* Replay: read a 0.5 MB FP8 checkpoint plus the ring of the last L inputs, and
  rebuild + requantize the full state only once per window (flush).

Reported per configuration: absolute kernel time, amortized per-token time,
bytes actually moved, achieved bandwidth, and the speedup. Bytes are accounted
twice -- "logical" (what the algorithm needs) and "actual" (including the
per-value-tile redundancy of the current tiling).
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
    dequantize_checkpoint,
    error_metrics,
    quantize_checkpoint,
    run_kernel,
)
from gdn_replayssm_fp8_kernel import (  # noqa: E402
    gdn_bf16_step,
    gdn_replay_fp8,
    gdn_replay_fp8_split,
    make_inputs,
)

from vllm.triton_utils import triton  # noqa: E402


def traffic_bytes(
    batch: int,
    window: int,
    heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    block_v: int,
) -> dict[str, float]:
    """Per-layer bytes moved by one replay step, one flush step, one bf16 step."""
    tiles = value_dim // block_v
    checkpoint = batch * value_heads * value_dim * key_dim * 1  # fp8 payload
    scales = batch * value_heads * tiles * 2
    query = batch * heads * key_dim * 2
    ring_key = batch * window * heads * key_dim * 2
    ring_value = batch * window * value_heads * value_dim * 2
    ring_scalars = batch * window * value_heads * 4
    replay_logical = (
        checkpoint + scales + query + ring_key + ring_value + ring_scalars
    )
    # The transformed-query chain is recomputed inside every value tile, so the
    # query and the cached keys are re-read once per tile.
    replay_actual = (
        checkpoint
        + scales
        + tiles * (query + ring_key)
        + ring_value
        + ring_scalars
    )
    flush = (
        2 * checkpoint
        + 2 * scales
        + tiles * ring_key
        + ring_value
        + ring_scalars
    )
    baseline = batch * value_heads * value_dim * key_dim * 2 * 2
    return {
        "replay_logical": replay_logical,
        "replay_actual": replay_actual,
        "flush_actual": flush,
        "bf16_baseline": baseline,
    }


def validate(args: argparse.Namespace, window: int) -> dict[str, float]:
    """Kernel vs. fp32 reference for the non-flush and the flush path.

    The non-flush path is checked for every candidate block size, because that
    is the path whose tiling is being swept. The flush path must stay at
    ``block_v=32``: its per-band scales are the vblock32 checkpoint format, and
    a wider tile would have to emit several scales per program.
    """
    batch = 2
    q, k_cache, v_cache, g_cache, beta_cache, state = make_inputs(
        batch,
        window,
        args.heads,
        args.value_heads,
        args.key_dim,
        args.value_dim,
        args.seed,
    )
    quantized = quantize_checkpoint(state, "vblock", 32)
    valid_len = torch.full((batch,), window, device="cuda", dtype=torch.int32)
    no_flush = torch.zeros(batch, device="cuda", dtype=torch.bool)
    flush = torch.ones(batch, device="cuda", dtype=torch.bool)

    reference_output, reference_state = run_kernel(
        dequantize_checkpoint(quantized),
        q.unsqueeze(1).expand(-1, window, -1, -1),
        k_cache,
        v_cache,
        g_cache,
        beta_cache,
    )
    results: dict[str, float] = {}
    for block_v in args.block_v:
        kernel_output = gdn_replay_fp8(
            quantized.payload,
            quantized.scale,
            q,
            k_cache,
            v_cache,
            g_cache,
            beta_cache,
            valid_len,
            no_flush,
            block_v=block_v,
        )
        results[f"output_relative_l2_block_v{block_v}"] = error_metrics(
            kernel_output, reference_output[:, -1]
        )["relative_l2"]
        split_output = gdn_replay_fp8_split(
            quantized.payload,
            quantized.scale,
            q,
            k_cache,
            v_cache,
            g_cache,
            beta_cache,
            valid_len,
            torch.empty(
                batch, args.value_heads, args.key_dim, device="cuda", dtype=torch.float32
            ),
            torch.empty(
                batch, args.value_heads, window, device="cuda", dtype=torch.float32
            ),
            block_v=block_v,
        )
        results[f"split_output_relative_l2_block_v{block_v}"] = error_metrics(
            split_output, reference_output[:, -1]
        )["relative_l2"]
    gdn_replay_fp8(
        quantized.payload,
        quantized.scale,
        q,
        k_cache,
        v_cache,
        g_cache,
        beta_cache,
        valid_len,
        flush,
    )
    flushed = dequantize_checkpoint(
        type(quantized)(quantized.payload, quantized.scale, "vblock", 32, 0.0)
    )
    expected = dequantize_checkpoint(
        quantize_checkpoint(reference_state, "vblock", 32)
    )
    results["flush_state_relative_l2"] = error_metrics(flushed, expected)[
        "relative_l2"
    ]
    return results


def bench_one(
    args: argparse.Namespace,
    window: int,
    batch: int,
    block_v: int,
    num_warps: int,
) -> None:
    q, k_cache, v_cache, g_cache, beta_cache, state = make_inputs(
        batch,
        window,
        args.heads,
        args.value_heads,
        args.key_dim,
        args.value_dim,
        args.seed,
    )
    quantized = quantize_checkpoint(state, "vblock", 32)
    valid_len = torch.full((batch,), window, device="cuda", dtype=torch.int32)
    no_flush = torch.zeros(batch, device="cuda", dtype=torch.bool)
    flush = torch.ones(batch, device="cuda", dtype=torch.bool)

    def replay_call():
        return gdn_replay_fp8(
            quantized.payload,
            quantized.scale,
            q,
            k_cache,
            v_cache,
            g_cache,
            beta_cache,
            valid_len,
            no_flush,
            block_v=block_v,
            num_warps=num_warps,
        )

    transformed_query_buffer = torch.empty(
        batch, args.value_heads, args.key_dim, device="cuda", dtype=torch.float32
    )
    coefficient_buffer = torch.empty(
        batch, args.value_heads, window, device="cuda", dtype=torch.float32
    )

    def split_call():
        return gdn_replay_fp8_split(
            quantized.payload,
            quantized.scale,
            q,
            k_cache,
            v_cache,
            g_cache,
            beta_cache,
            valid_len,
            transformed_query_buffer,
            coefficient_buffer,
            block_v=block_v,
            num_warps=num_warps,
            block_kc=args.block_kc,
        )

    def flush_call():
        return gdn_replay_fp8(
            quantized.payload,
            quantized.scale,
            q,
            k_cache,
            v_cache,
            g_cache,
            beta_cache,
            valid_len,
            flush,
            # The flush writes the vblock32 scale layout, so it must keep the
            # 32-row tile regardless of how the replay path is tiled.
            block_v=32,
            num_warps=num_warps,
        )

    try:
        replay_ms = triton.testing.do_bench(replay_call, warmup=25, rep=100)
        flush_ms = triton.testing.do_bench(flush_call, warmup=25, rep=100)
        split_ms = triton.testing.do_bench(split_call, warmup=25, rep=100)
    except Exception as error:  # noqa: BLE001 - report and continue the sweep
        print(
            json.dumps(
                {
                    "type": "benchmark_error",
                    "window": window,
                    "batch": batch,
                    "block_v": block_v,
                    "num_warps": num_warps,
                    "error": str(error)[:200],
                }
            ),
            flush=True,
        )
        return

    bf16_state = state.to(torch.bfloat16)
    baseline_ms = triton.testing.do_bench(
        lambda: gdn_bf16_step(
            bf16_state,
            q,
            k_cache[:, -1],
            v_cache[:, -1],
            g_cache[:, -1],
            beta_cache[:, -1],
        ),
        warmup=25,
        rep=100,
    )
    amortized_ms = (replay_ms * (window - 1) + flush_ms) / window
    # Steady state: within a window the ring grows 1, 2, ... L-1 and the L-th
    # step is the flush, which rebuilds the state from the full ring. Timing
    # every non-flush step at ring length L (as above) is the worst case.
    cycle_tiled = 0.0
    cycle_split = 0.0
    for length in range(1, window):
        length_valid = torch.full(
            (batch,), length, device="cuda", dtype=torch.int32
        )
        cycle_tiled += triton.testing.do_bench(
            lambda: gdn_replay_fp8(
                quantized.payload,
                quantized.scale,
                q,
                k_cache,
                v_cache,
                g_cache,
                beta_cache,
                length_valid,
                no_flush,
                block_v=block_v,
                num_warps=num_warps,
            ),
            warmup=10,
            rep=50,
        )
        cycle_split += triton.testing.do_bench(
            lambda: gdn_replay_fp8_split(
                quantized.payload,
                quantized.scale,
                q,
                k_cache,
                v_cache,
                g_cache,
                beta_cache,
                length_valid,
                transformed_query_buffer,
                coefficient_buffer,
                block_v=block_v,
                num_warps=num_warps,
                block_kc=args.block_kc,
            ),
            warmup=10,
            rep=50,
        )
    cycle_tiled_ms = (cycle_tiled + flush_ms) / window
    cycle_split_ms = (cycle_split + flush_ms) / window
    traffic = traffic_bytes(
        batch,
        window,
        args.heads,
        args.value_heads,
        args.key_dim,
        args.value_dim,
        block_v,
    )
    amortized_logical = (
        traffic["replay_logical"] * (window - 1) + traffic["flush_actual"]
    ) / window
    amortized_actual = (
        traffic["replay_actual"] * (window - 1) + traffic["flush_actual"]
    ) / window
    print(
        json.dumps(
            {
                "type": "benchmark",
                "window": window,
                "batch": batch,
                "block_v": block_v,
                "num_warps": num_warps,
                "replay_ms": replay_ms,
                "split_ms": split_ms,
                "flush_ms": flush_ms,
                "amortized_ms": amortized_ms,
                "split_amortized_ms": (split_ms * (window - 1) + flush_ms)
                / window,
                "cycle_tiled_ms": cycle_tiled_ms,
                "cycle_split_ms": cycle_split_ms,
                "cycle_tiled_speedup": baseline_ms / cycle_tiled_ms,
                "cycle_split_speedup": baseline_ms / cycle_split_ms,
                "bf16_step_ms": baseline_ms,
                "speedup_vs_bf16": baseline_ms / amortized_ms,
                "split_speedup_vs_bf16": baseline_ms
                / ((split_ms * (window - 1) + flush_ms) / window),
                "bf16_bytes": traffic["bf16_baseline"],
                "replay_bytes_logical": traffic["replay_logical"],
                "replay_bytes_actual": traffic["replay_actual"],
                "amortized_bytes_logical": amortized_logical,
                "amortized_bytes_actual": amortized_actual,
                "traffic_ratio_vs_bf16": traffic["bf16_baseline"] / amortized_actual,
                "bf16_gbps": traffic["bf16_baseline"] / (baseline_ms * 1e-3) / 1e9,
                "replay_gbps": amortized_actual / (amortized_ms * 1e-3) / 1e9,
                "flush_gbps": traffic["flush_actual"] / (flush_ms * 1e-3) / 1e9,
            }
        ),
        flush=True,
    )


def benchmark(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            {
                "type": "config",
                "device": torch.cuda.get_device_name(0),
                "heads": args.heads,
                "value_heads": args.value_heads,
                "key_dim": args.key_dim,
                "value_dim": args.value_dim,
                "window": args.window,
                "block_v": args.block_v,
                "num_warps": args.num_warps,
                "state_bytes_per_layer_per_seq": args.value_heads
                * args.value_dim
                * args.key_dim
                * 2,
            }
        ),
        flush=True,
    )
    for window in args.window:
        print(
            json.dumps(
                {"type": "validation", "window": window, **validate(args, window)}
            ),
            flush=True,
        )
        for block_v in args.block_v:
            for num_warps in args.num_warps:
                for batch in args.batches:
                    bench_one(args, window, batch, block_v, num_warps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,2,4,8,16,32,64")
    parser.add_argument("--window", default="4,8,16,32")
    parser.add_argument("--block-v", default="32,64,128")
    parser.add_argument("--num-warps", default="4,8")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument(
        "--block-kc",
        type=int,
        default=128,
        help="Key-chunk width for the split apply kernel (128 = whole key dim).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.batches = [int(value) for value in args.batches.split(",")]
    args.window = [int(value) for value in args.window.split(",")]
    args.block_v = [int(value) for value in args.block_v.split(",")]
    args.num_warps = [int(value) for value in args.num_warps.split(",")]
    return args


if __name__ == "__main__":
    benchmark(parse_args())
