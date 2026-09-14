"""Diagnose where the GDN ReplaySSM kernel loses bandwidth.

Three questions:

1. How many registers/spills does the replay kernel need for each tiling?
2. What is the pure-traffic floor? Two stripped-down kernels load exactly the
   same bytes as the replay kernel (the FP8 checkpoint tile, and the ring of
   keys/values) and reduce them to a scalar, so the gap between them and the
   real kernel is the cost of the recurrence itself.
3. How much does the checkpoint read alone cost versus the ring read?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "kernels"))

from gdn_replay_fp8_reference import quantize_checkpoint  # noqa: E402
from gdn_replayssm_fp8_kernel import (  # noqa: E402
    _gdn_replay_apply_kernel,
    _gdn_replay_precompute_kernel,
    _gdn_replay_fp8_kernel,
    gdn_bf16_step,
    gdn_replay_fp8,
    gdn_replay_fp8_split,
    make_inputs,
)

from vllm.triton_utils import tl, triton  # noqa: E402


@triton.jit
def _checkpoint_traffic_kernel(
    checkpoint_ptr,
    out_ptr,
    stride_cp_b,
    stride_cp_h,
    stride_cp_v,
    stride_cp_k,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Stream the FP8 checkpoint exactly like the replay kernel does."""
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_vt = tl.program_id(2)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_vt * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = (offs_v[:, None] < V) & (offs_k[None, :] < K)
    offsets = (
        pid_b * stride_cp_b
        + pid_hv * stride_cp_h
        + offs_v[:, None] * stride_cp_v
        + offs_k[None, :] * stride_cp_k
    )
    tile = tl.load(checkpoint_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(tl.sum(tile, axis=1), axis=0)
    tl.store(out_ptr + pid_b * 1024 + pid_hv, total)


@triton.jit
def _ring_traffic_kernel(
    k_cache_ptr,
    v_cache_ptr,
    g_cache_ptr,
    beta_cache_ptr,
    valid_len_ptr,
    out_ptr,
    stride_k_b,
    stride_k_t,
    stride_k_h,
    stride_k_k,
    stride_v_b,
    stride_v_t,
    stride_v_h,
    stride_v_v,
    stride_g_b,
    stride_g_t,
    stride_g_h,
    stride_beta_b,
    stride_beta_t,
    stride_beta_h,
    K: tl.constexpr,
    V: tl.constexpr,
    HV_PER_H: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Stream the ring exactly like the replay kernel's loop does."""
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_h = pid_hv // HV_PER_H
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V
    valid_len = tl.load(valid_len_ptr + pid_b).to(tl.int32)
    acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    scalar = 0.0
    for token in tl.static_range(0, MAX_CACHE_LEN):
        active = token < valid_len
        key = tl.load(
            k_cache_ptr
            + pid_b * stride_k_b
            + token * stride_k_t
            + pid_h * stride_k_h
            + offs_k * stride_k_k,
            mask=active & mask_k,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            v_cache_ptr
            + pid_b * stride_v_b
            + token * stride_v_t
            + pid_hv * stride_v_h
            + offs_v * stride_v_v,
            mask=active & mask_v,
            other=0.0,
        ).to(tl.float32)
        log_decay = tl.load(
            g_cache_ptr + pid_b * stride_g_b + token * stride_g_t + pid_hv * stride_g_h,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        beta = tl.load(
            beta_cache_ptr
            + pid_b * stride_beta_b
            + token * stride_beta_t
            + pid_hv * stride_beta_h,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        acc += value * tl.sum(key) * (log_decay + beta)
        scalar += tl.sum(key)
    tl.store(
        out_ptr + pid_b * 4096 + pid_hv * 4 + tl.arange(0, 4),
        tl.full([4], scalar, dtype=tl.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--block-v", default="32,64,128")
    parser.add_argument("--num-warps", default="4,8")
    parser.add_argument("--precompute-warps", default="1,2,4")
    parser.add_argument("--apply-kc", default="32")
    parser.add_argument("--apply-stages", default="2")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.block_v = [int(value) for value in args.block_v.split(",")]
    args.num_warps = [int(value) for value in args.num_warps.split(",")]
    args.precompute_warps = [
        int(value) for value in args.precompute_warps.split(",")
    ]
    args.apply_kc = [int(value) for value in args.apply_kc.split(",")]
    args.apply_stages = [int(value) for value in args.apply_stages.split(",")]
    return args


def main() -> None:
    args = parse_args()
    window, batch = args.window, args.batch
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
    checkpoint = quantized.payload
    scale = quantized.scale
    valid_len = torch.full((batch,), window, device="cuda", dtype=torch.int32)
    no_flush = torch.zeros(batch, device="cuda", dtype=torch.bool)
    out = torch.empty(batch, args.value_heads, args.value_dim, device="cuda")
    checkpoint_bytes = checkpoint.numel() * checkpoint.element_size()
    ring_bytes = batch * window * (
        args.heads * args.key_dim * 2 + args.value_heads * args.value_dim * 2
    )
    print(
        json.dumps(
            {
                "type": "config",
                "window": window,
                "batch": batch,
                "checkpoint_bytes": checkpoint_bytes,
                "ring_bytes": ring_bytes,
                "bf16_state_bytes": state.numel() * 2,
            }
        ),
        flush=True,
    )
    for block_v in args.block_v:
        for num_warps in args.num_warps:
            grid = (
                batch,
                args.value_heads,
                triton.cdiv(args.value_dim, block_v),
            )
            compiled = _gdn_replay_fp8_kernel[grid](
                checkpoint,
                scale,
                q,
                k_cache,
                v_cache,
                g_cache,
                beta_cache,
                valid_len,
                no_flush,
                out,
                *checkpoint.stride(),
                *scale.stride(),
                *q.stride(),
                *k_cache.stride(),
                *v_cache.stride(),
                *g_cache.stride(),
                *beta_cache.stride(),
                *out.stride(),
                K=args.key_dim,
                V=args.value_dim,
                HV_PER_H=args.value_heads // args.heads,
                MAX_CACHE_LEN=window,
                BLOCK_K=triton.next_power_of_2(args.key_dim),
                BLOCK_V=block_v,
                VBLOCK=32,
                num_warps=num_warps,
                num_stages=2,
            )
            scratch = torch.empty(batch * 4096, device="cuda", dtype=torch.float32)
            checkpoint_ms = triton.testing.do_bench(
                lambda: _checkpoint_traffic_kernel[grid](
                    checkpoint,
                    scratch,
                    *checkpoint.stride(),
                    K=args.key_dim,
                    V=args.value_dim,
                    BLOCK_V=block_v,
                    BLOCK_K=triton.next_power_of_2(args.key_dim),
                    num_warps=num_warps,
                    num_stages=2,
                ),
                warmup=25,
                rep=100,
            )
            ring_grid = (batch, args.value_heads, 1)
            ring_ms = triton.testing.do_bench(
                lambda: _ring_traffic_kernel[ring_grid](
                    k_cache,
                    v_cache,
                    g_cache,
                    beta_cache,
                    valid_len,
                    scratch,
                    *k_cache.stride(),
                    *v_cache.stride(),
                    *g_cache.stride(),
                    *beta_cache.stride(),
                    K=args.key_dim,
                    V=args.value_dim,
                    HV_PER_H=args.value_heads // args.heads,
                    MAX_CACHE_LEN=window,
                    BLOCK_V=max(32, block_v),
                    BLOCK_K=triton.next_power_of_2(args.key_dim),
                    num_warps=num_warps,
                    num_stages=2,
                ),
                warmup=25,
                rep=100,
            )
            replay_ms = triton.testing.do_bench(
                lambda: gdn_replay_fp8(
                    checkpoint,
                    scale,
                    q,
                    k_cache,
                    v_cache,
                    g_cache,
                    beta_cache,
                    valid_len,
                    no_flush,
                    block_v=block_v,
                    num_warps=num_warps,
                ),
                warmup=25,
                rep=100,
            )
            bf16_state = state.to(torch.bfloat16)
            bf16_ms = triton.testing.do_bench(
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
            print(
                json.dumps(
                    {
                        "type": "diag",
                        "block_v": block_v,
                        "num_warps": num_warps,
                        "n_regs": getattr(compiled, "n_regs", None),
                        "n_spills": getattr(compiled, "n_spills", None),
                        "shared_bytes": getattr(
                            getattr(compiled, "metadata", None), "shared", None
                        ),
                        "replay_ms": replay_ms,
                        "checkpoint_stream_ms": checkpoint_ms,
                        "ring_stream_ms": ring_ms,
                        "bf16_step_ms": bf16_ms,
                        "checkpoint_gbps": checkpoint_bytes
                        / (checkpoint_ms * 1e-3)
                        / 1e9,
                        "ring_gbps": ring_bytes / (ring_ms * 1e-3) / 1e9,
                        "replay_total_gbps": (checkpoint_bytes + ring_bytes)
                        / (replay_ms * 1e-3)
                        / 1e9,
                        "bf16_gbps": (state.numel() * 2 * 2)
                        / (bf16_ms * 1e-3)
                        / 1e9,
                    }
                ),
                flush=True,
            )
        tq_buffer = torch.empty(
            batch, args.value_heads, args.key_dim, device="cuda", dtype=torch.float32
        )
        coef_buffer = torch.empty(
            batch, args.value_heads, window, device="cuda", dtype=torch.float32
        )
        for precompute_warps in args.precompute_warps:
          for block_kc in args.apply_kc:
           for num_stages in args.apply_stages:
            precompute_ms = triton.testing.do_bench(
                lambda: _gdn_replay_precompute_kernel[(batch, args.value_heads)](
                    q,
                    k_cache,
                    g_cache,
                    beta_cache,
                    valid_len,
                    tq_buffer,
                    coef_buffer,
                    *q.stride(),
                    *k_cache.stride(),
                    *g_cache.stride(),
                    *beta_cache.stride(),
                    *tq_buffer.stride(),
                    *coef_buffer.stride(),
                    K=args.key_dim,
                    HV_PER_H=args.value_heads // args.heads,
                    MAX_CACHE_LEN=window,
                    BLOCK_K=triton.next_power_of_2(args.key_dim),
                    num_warps=precompute_warps,
                ),
                warmup=25,
                rep=100,
            )
            apply_ms = triton.testing.do_bench(
                lambda: gdn_replay_fp8_split(
                    checkpoint,
                    scale,
                    q,
                    k_cache,
                    v_cache,
                    g_cache,
                    beta_cache,
                    valid_len,
                    tq_buffer,
                    coef_buffer,
                    block_v=block_v,
                    num_warps=num_warps,
                    block_kc=block_kc,
                    num_stages=num_stages,
                ),
                warmup=25,
                rep=100,
            )
            print(
                json.dumps(
                    {
                        "type": "split_diag",
                        "block_v": block_v,
                        "num_warps": num_warps,
                        "precompute_warps": precompute_warps,
                        "block_kc": block_kc,
                        "num_stages": num_stages,
                        "precompute_ms": precompute_ms,
                        "split_total_ms": apply_ms,
                        "apply_only_ms": apply_ms - precompute_ms,
                        "precompute_share": precompute_ms / apply_ms,
                        "bf16_step_ms": bf16_ms,
                        "split_speedup": bf16_ms / apply_ms,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
