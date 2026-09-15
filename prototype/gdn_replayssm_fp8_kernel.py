"""Triton prototype for GDN ReplaySSM with FP8 persistent checkpoints."""

from __future__ import annotations

import argparse
import json

import torch

from vllm.triton_utils import tl, triton

from gdn_replay_fp8_reference import (
    dequantize_checkpoint,
    error_metrics,
    quantize_checkpoint,
    run_kernel,
)


@triton.jit
def _gdn_replay_fp8_kernel(
    checkpoint_ptr,
    checkpoint_scale_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    g_cache_ptr,
    beta_cache_ptr,
    valid_len_ptr,
    flush_ptr,
    out_ptr,
    stride_cp_b,
    stride_cp_h,
    stride_cp_v,
    stride_cp_k,
    stride_scale_b,
    stride_scale_h,
    stride_scale_vt,
    stride_q_b,
    stride_q_h,
    stride_q_k,
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
    stride_out_b,
    stride_out_h,
    stride_out_v,
    K: tl.constexpr,
    V: tl.constexpr,
    HV_PER_H: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    VBLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_vt = tl.program_id(2)
    pid_h = pid_hv // HV_PER_H

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_vt * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    checkpoint_offsets = (
        pid_b * stride_cp_b
        + pid_hv * stride_cp_h
        + offs_v[:, None] * stride_cp_v
        + offs_k[None, :] * stride_cp_k
    )
    scale_offset = (
        pid_b * stride_scale_b
        + pid_hv * stride_scale_h
        + pid_vt * stride_scale_vt
    )
    # The checkpoint stores one scale per VBLOCK value rows. A program may span
    # several bands when BLOCK_V > VBLOCK, so gather the scale per row instead
    # of assuming a single scale for the whole tile.
    scale_offsets = (
        pid_b * stride_scale_b
        + pid_hv * stride_scale_h
        + (offs_v // VBLOCK) * stride_scale_vt
    )
    checkpoint_scale = tl.load(
        checkpoint_scale_ptr + scale_offsets, mask=mask_v, other=1.0
    ).to(tl.float32)
    state = tl.load(
        checkpoint_ptr + checkpoint_offsets,
        mask=mask_state,
        other=0.0,
    ).to(tl.float32)
    state *= checkpoint_scale[:, None]

    valid_len = tl.load(valid_len_ptr + pid_b).to(tl.int32)
    query_offsets = (
        pid_b * stride_q_b + pid_h * stride_q_h + offs_k * stride_q_k
    )
    query = tl.load(
        q_ptr + query_offsets, mask=mask_k, other=0.0
    ).to(tl.float32)
    output_offsets = (
        pid_b * stride_out_b
        + pid_hv * stride_out_h
        + offs_v * stride_out_v
    )
    should_flush = tl.load(flush_ptr + pid_b) != 0

    if not should_flush:
        transformed_query = query
        output = tl.zeros([BLOCK_V], dtype=tl.float32)
        for reverse_token in tl.static_range(0, MAX_CACHE_LEN):
            token = MAX_CACHE_LEN - 1 - reverse_token
            active = token < valid_len
            key_offsets = (
                pid_b * stride_k_b
                + token * stride_k_t
                + pid_h * stride_k_h
                + offs_k * stride_k_k
            )
            value_offsets = (
                pid_b * stride_v_b
                + token * stride_v_t
                + pid_hv * stride_v_h
                + offs_v * stride_v_v
            )
            scalar_offset_g = (
                pid_b * stride_g_b
                + token * stride_g_t
                + pid_hv * stride_g_h
            )
            scalar_offset_beta = (
                pid_b * stride_beta_b
                + token * stride_beta_t
                + pid_hv * stride_beta_h
            )
            key = tl.load(
                k_cache_ptr + key_offsets,
                mask=active & mask_k,
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                v_cache_ptr + value_offsets,
                mask=active & mask_v,
                other=0.0,
            ).to(tl.float32)
            log_decay = tl.load(
                g_cache_ptr + scalar_offset_g,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            beta = tl.load(
                beta_cache_ptr + scalar_offset_beta,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            key_dot_query = tl.sum(key * transformed_query, axis=0)
            output += beta * value * key_dot_query
            transformed_query = tl.exp(log_decay) * (
                transformed_query - beta * key * key_dot_query
            )
        output += tl.sum(state * transformed_query[None, :], axis=1)
        output *= K**-0.5
        tl.store(out_ptr + output_offsets, output, mask=mask_v)
    else:
        for token in tl.static_range(0, MAX_CACHE_LEN):
            active = token < valid_len
            key_offsets = (
                pid_b * stride_k_b
                + token * stride_k_t
                + pid_h * stride_k_h
                + offs_k * stride_k_k
            )
            value_offsets = (
                pid_b * stride_v_b
                + token * stride_v_t
                + pid_hv * stride_v_h
                + offs_v * stride_v_v
            )
            scalar_offset_g = (
                pid_b * stride_g_b
                + token * stride_g_t
                + pid_hv * stride_g_h
            )
            scalar_offset_beta = (
                pid_b * stride_beta_b
                + token * stride_beta_t
                + pid_hv * stride_beta_h
            )
            key = tl.load(
                k_cache_ptr + key_offsets,
                mask=active & mask_k,
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                v_cache_ptr + value_offsets,
                mask=active & mask_v,
                other=0.0,
            ).to(tl.float32)
            log_decay = tl.load(
                g_cache_ptr + scalar_offset_g,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            beta = tl.load(
                beta_cache_ptr + scalar_offset_beta,
                mask=active,
                other=0.0,
            ).to(tl.float32)
            state *= tl.exp(log_decay)
            retrieved = tl.sum(state * key[None, :], axis=1)
            correction = (value - retrieved) * beta
            state += correction[:, None] * key[None, :]

        output = tl.sum(state * query[None, :], axis=1) * (K**-0.5)
        tl.store(out_ptr + output_offsets, output, mask=mask_v)
        row_amax = tl.max(tl.abs(state), axis=1)
        tile_amax = tl.max(row_amax, axis=0)
        new_scale = tl.maximum(tile_amax / 448.0, 1.0e-12)
        stored_scale = new_scale.to(tl.float16)
        quant_scale = stored_scale.to(tl.float32)
        quantized = tl.clamp(state / quant_scale, -448.0, 448.0).to(
            tl.float8e4nv
        )
        tl.store(
            checkpoint_ptr + checkpoint_offsets,
            quantized,
            mask=mask_state,
        )
        tl.store(checkpoint_scale_ptr + scale_offset, stored_scale)


@triton.jit
def _gdn_bf16_step_kernel(
    state_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    out_ptr,
    stride_state_b,
    stride_state_h,
    stride_state_v,
    stride_state_k,
    stride_q_b,
    stride_q_h,
    stride_q_k,
    stride_k_b,
    stride_k_h,
    stride_k_k,
    stride_v_b,
    stride_v_h,
    stride_v_v,
    stride_g_b,
    stride_g_h,
    stride_beta_b,
    stride_beta_h,
    stride_out_b,
    stride_out_h,
    stride_out_v,
    K: tl.constexpr,
    V: tl.constexpr,
    HV_PER_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_vt = tl.program_id(2)
    pid_h = pid_hv // HV_PER_H

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_vt * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    state_offsets = (
        pid_b * stride_state_b
        + pid_hv * stride_state_h
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    state = tl.load(
        state_ptr + state_offsets, mask=mask_state, other=0.0
    ).to(tl.float32)
    key = tl.load(
        k_ptr
        + pid_b * stride_k_b
        + pid_h * stride_k_h
        + offs_k * stride_k_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    value = tl.load(
        v_ptr
        + pid_b * stride_v_b
        + pid_hv * stride_v_h
        + offs_v * stride_v_v,
        mask=mask_v,
        other=0.0,
    ).to(tl.float32)
    log_decay = tl.load(
        g_ptr + pid_b * stride_g_b + pid_hv * stride_g_h
    ).to(tl.float32)
    beta = tl.load(
        beta_ptr + pid_b * stride_beta_b + pid_hv * stride_beta_h
    ).to(tl.float32)
    state *= tl.exp(log_decay)
    retrieved = tl.sum(state * key[None, :], axis=1)
    correction = (value - retrieved) * beta
    state += correction[:, None] * key[None, :]

    query = tl.load(
        q_ptr
        + pid_b * stride_q_b
        + pid_h * stride_q_h
        + offs_k * stride_q_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    output = tl.sum(state * query[None, :], axis=1) * (K**-0.5)
    output_offsets = (
        pid_b * stride_out_b
        + pid_hv * stride_out_h
        + offs_v * stride_out_v
    )
    tl.store(out_ptr + output_offsets, output, mask=mask_v)
    tl.store(state_ptr + state_offsets, state, mask=mask_state)


def gdn_replay_fp8(
    checkpoint: torch.Tensor,
    checkpoint_scale: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    g_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    valid_len: torch.Tensor,
    flush: torch.Tensor,
    block_v: int = 32,
    num_warps: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, max_cache_len, heads, key_dim = k_cache.shape
    value_heads = v_cache.shape[2]
    value_dim = v_cache.shape[3]
    block_k = triton.next_power_of_2(key_dim)
    # Allow the caller to supply the output buffer so a benchmark's timed region
    # contains only kernel launches, not allocator traffic.
    output = (
        torch.empty(
            batch,
            value_heads,
            value_dim,
            dtype=q.dtype,
            device=q.device,
        )
        if out is None
        else out
    )
    grid = (batch, value_heads, triton.cdiv(value_dim, block_v))
    _gdn_replay_fp8_kernel[grid](
        checkpoint,
        checkpoint_scale,
        q,
        k_cache,
        v_cache,
        g_cache,
        beta_cache,
        valid_len,
        flush,
        output,
        *checkpoint.stride(),
        *checkpoint_scale.stride(),
        *q.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        *g_cache.stride(),
        *beta_cache.stride(),
        *output.stride(),
        K=key_dim,
        V=value_dim,
        HV_PER_H=value_heads // heads,
        MAX_CACHE_LEN=max_cache_len,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        VBLOCK=32,
        num_warps=num_warps,
        num_stages=2,
    )
    return output


def gdn_bf16_step(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    batch, value_heads, value_dim, key_dim = state.shape
    heads = k.shape[1]
    block_v = 32
    output = torch.empty(
        batch,
        value_heads,
        value_dim,
        dtype=q.dtype,
        device=q.device,
    )
    grid = (batch, value_heads, triton.cdiv(value_dim, block_v))
    _gdn_bf16_step_kernel[grid](
        state,
        q,
        k,
        v,
        g,
        beta,
        output,
        *state.stride(),
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *g.stride(),
        *beta.stride(),
        *output.stride(),
        K=key_dim,
        V=value_dim,
        HV_PER_H=value_heads // heads,
        BLOCK_K=triton.next_power_of_2(key_dim),
        BLOCK_V=block_v,
        num_warps=4,
        num_stages=2,
    )
    return output


@triton.jit
def _gdn_prep_fused_kernel(
    mixed_qkv_ptr,
    q_out_ptr,
    k_ring_ptr,
    v_ring_ptr,
    g_ring_ptr,
    beta_ring_ptr,
    a_ptr,
    b_ptr,
    a_log_ptr,
    dt_bias_ptr,
    pos_ptr,
    stride_mixed_b,
    stride_q_b,
    stride_q_h,
    stride_q_k,
    stride_kr_b,
    stride_kr_t,
    stride_kr_h,
    stride_kr_k,
    stride_vr_b,
    stride_vr_t,
    stride_vr_h,
    stride_vr_v,
    stride_gr_b,
    stride_gr_t,
    stride_gr_h,
    stride_a_b,
    stride_a_h,
    stride_b_b,
    stride_b_h,
    K: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    QK_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
    L2_EPS: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    """One launch for the whole preparation: q/k norm, gating, ring append.

    Programs ``[0, H)`` normalise and append q/k, programs ``[H, H + HV)``
    compute the gating scalars and append v. Splitting this into two kernels
    doubled the launch overhead, which dominated at small batch.
    """
    pid_b = tl.program_id(0)
    pid = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    pos = tl.load(pos_ptr + pid_b).to(tl.int64)
    row = pid_b * stride_mixed_b
    if pid < H:
        pid_h = pid
        mask_k = offs < K
        query = tl.load(
            mixed_qkv_ptr + row + pid_h * K + offs, mask=mask_k, other=0.0
        ).to(tl.float32)
        key = tl.load(
            mixed_qkv_ptr + row + H * K + pid_h * K + offs,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        query = query / tl.sqrt(tl.sum(query * query) + L2_EPS)
        key = key / tl.sqrt(tl.sum(key * key) + L2_EPS)
        tl.store(
            q_out_ptr
            + pid_b * stride_q_b
            + pid_h * stride_q_h
            + offs * stride_q_k,
            query.to(q_out_ptr.dtype.element_ty),
            mask=mask_k,
        )
        tl.store(
            k_ring_ptr
            + pid_b * stride_kr_b
            + pos * stride_kr_t
            + pid_h * stride_kr_h
            + offs * stride_kr_k,
            key.to(k_ring_ptr.dtype.element_ty),
            mask=mask_k,
        )
    else:
        pid_hv = pid - H
        mask_v = offs < V
        value = tl.load(
            mixed_qkv_ptr + row + QK_OFFSET + pid_hv * V + offs,
            mask=mask_v,
            other=0.0,
        )
        tl.store(
            v_ring_ptr
            + pid_b * stride_vr_b
            + pos * stride_vr_t
            + pid_hv * stride_vr_h
            + offs * stride_vr_v,
            value,
            mask=mask_v,
        )
        a_val = tl.load(a_ptr + pid_b * stride_a_b + pid_hv * stride_a_h).to(
            tl.float32
        )
        b_val = tl.load(b_ptr + pid_b * stride_b_b + pid_hv * stride_b_h).to(
            tl.float32
        )
        a_log_val = tl.load(a_log_ptr + pid_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        gate = -tl.exp(a_log_val) * softplus_x
        beta = tl.sigmoid(b_val)
        tl.store(
            g_ring_ptr
            + pid_b * stride_gr_b
            + pos * stride_gr_t
            + pid_hv * stride_gr_h,
            gate.to(g_ring_ptr.dtype.element_ty),
        )
        tl.store(
            beta_ring_ptr
            + pid_b * stride_gr_b
            + pos * stride_gr_t
            + pid_hv * stride_gr_h,
            beta.to(beta_ring_ptr.dtype.element_ty),
        )


@triton.jit
def _gdn_prep_qk_kernel(
    mixed_qkv_ptr,
    q_out_ptr,
    k_ring_ptr,
    pos_ptr,
    stride_mixed_b,
    stride_q_b,
    stride_q_h,
    stride_q_k,
    stride_kr_b,
    stride_kr_t,
    stride_kr_h,
    stride_kr_k,
    K: tl.constexpr,
    H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    L2_EPS: tl.constexpr,
):
    """q/k L2 normalisation and ring append, matching the production semantics.

    The production operator does ``x / sqrt(sum(x*x) + 1e-6)`` inline; a fair
    full-contract comparison has to do the same work on the replay side, which
    is what this kernel exists for.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    pos = tl.load(pos_ptr + pid_b).to(tl.int64)
    row = pid_b * stride_mixed_b
    query = tl.load(
        mixed_qkv_ptr + row + pid_h * K + offs_k, mask=mask_k, other=0.0
    ).to(tl.float32)
    key = tl.load(
        mixed_qkv_ptr + row + H * K + pid_h * K + offs_k, mask=mask_k, other=0.0
    ).to(tl.float32)
    query = query / tl.sqrt(tl.sum(query * query) + L2_EPS)
    key = key / tl.sqrt(tl.sum(key * key) + L2_EPS)
    tl.store(
        q_out_ptr
        + pid_b * stride_q_b
        + pid_h * stride_q_h
        + offs_k * stride_q_k,
        query.to(q_out_ptr.dtype.element_ty),
        mask=mask_k,
    )
    tl.store(
        k_ring_ptr
        + pid_b * stride_kr_b
        + pos * stride_kr_t
        + pid_h * stride_kr_h
        + offs_k * stride_kr_k,
        key.to(k_ring_ptr.dtype.element_ty),
        mask=mask_k,
    )


@triton.jit
def _gdn_prep_gating_kernel(
    mixed_qkv_ptr,
    v_ring_ptr,
    g_ring_ptr,
    beta_ring_ptr,
    a_ptr,
    b_ptr,
    a_log_ptr,
    dt_bias_ptr,
    pos_ptr,
    stride_mixed_b,
    stride_vr_b,
    stride_vr_t,
    stride_vr_h,
    stride_vr_v,
    stride_gr_b,
    stride_gr_t,
    stride_gr_h,
    stride_a_b,
    stride_a_h,
    stride_b_b,
    stride_b_h,
    V: tl.constexpr,
    QK_OFFSET: tl.constexpr,
    BLOCK_V: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    """Gating computation plus the value ring append."""
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    offs_v = tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    pos = tl.load(pos_ptr + pid_b).to(tl.int64)
    value = tl.load(
        mixed_qkv_ptr
        + pid_b * stride_mixed_b
        + QK_OFFSET
        + pid_hv * V
        + offs_v,
        mask=mask_v,
        other=0.0,
    )
    tl.store(
        v_ring_ptr
        + pid_b * stride_vr_b
        + pos * stride_vr_t
        + pid_hv * stride_vr_h
        + offs_v * stride_vr_v,
        value,
        mask=mask_v,
    )
    a_val = tl.load(a_ptr + pid_b * stride_a_b + pid_hv * stride_a_h).to(tl.float32)
    b_val = tl.load(b_ptr + pid_b * stride_b_b + pid_hv * stride_b_h).to(tl.float32)
    a_log_val = tl.load(a_log_ptr + pid_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    gate = -tl.exp(a_log_val) * softplus_x
    beta = tl.sigmoid(b_val)
    tl.store(
        g_ring_ptr + pid_b * stride_gr_b + pos * stride_gr_t + pid_hv * stride_gr_h,
        gate.to(g_ring_ptr.dtype.element_ty),
    )
    tl.store(
        beta_ring_ptr
        + pid_b * stride_gr_b
        + pos * stride_gr_t
        + pid_hv * stride_gr_h,
        beta.to(beta_ring_ptr.dtype.element_ty),
    )


def gdn_prep_ring(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    pos: torch.Tensor,
    q_out: torch.Tensor,
    k_ring: torch.Tensor,
    v_ring: torch.Tensor,
    g_ring: torch.Tensor,
    beta_ring: torch.Tensor,
    num_warps: int = 1,
    fused: bool = True,
) -> None:
    """Full-contract preparation: normalise q/k, compute gating, append to ring."""
    batch = mixed_qkv.shape[0]
    heads = q_out.shape[1]
    key_dim = q_out.shape[2]
    value_heads = v_ring.shape[2]
    value_dim = v_ring.shape[3]
    if fused:
        _gdn_prep_fused_kernel[(batch, heads + value_heads)](
            mixed_qkv,
            q_out,
            k_ring,
            v_ring,
            g_ring,
            beta_ring,
            a,
            b,
            a_log,
            dt_bias,
            pos,
            mixed_qkv.stride(0),
            *q_out.stride(),
            *k_ring.stride(),
            *v_ring.stride(),
            *g_ring.stride(),
            *a.stride(),
            *b.stride(),
            K=key_dim,
            H=heads,
            V=value_dim,
            QK_OFFSET=2 * heads * key_dim,
            BLOCK=triton.next_power_of_2(max(key_dim, value_dim)),
            L2_EPS=1e-6,
            SOFTPLUS_THRESHOLD=20.0,
            num_warps=num_warps,
        )
        return
    _gdn_prep_qk_kernel[(batch, heads)](
        mixed_qkv,
        q_out,
        k_ring,
        pos,
        mixed_qkv.stride(0),
        *q_out.stride(),
        *k_ring.stride(),
        K=key_dim,
        H=heads,
        BLOCK_K=triton.next_power_of_2(key_dim),
        L2_EPS=1e-6,
        num_warps=num_warps,
    )
    _gdn_prep_gating_kernel[(batch, value_heads)](
        mixed_qkv,
        v_ring,
        g_ring,
        beta_ring,
        a,
        b,
        a_log,
        dt_bias,
        pos,
        mixed_qkv.stride(0),
        *v_ring.stride(),
        *g_ring.stride(),
        *a.stride(),
        *b.stride(),
        V=value_dim,
        QK_OFFSET=2 * heads * key_dim,
        BLOCK_V=triton.next_power_of_2(value_dim),
        SOFTPLUS_THRESHOLD=20.0,
        num_warps=num_warps,
    )


@triton.jit
def _gdn_replay_precompute_kernel(
    q_ptr,
    k_cache_ptr,
    g_cache_ptr,
    beta_cache_ptr,
    valid_len_ptr,
    tq_out_ptr,
    coef_out_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_k,
    stride_k_b,
    stride_k_t,
    stride_k_h,
    stride_k_k,
    stride_g_b,
    stride_g_t,
    stride_g_h,
    stride_beta_b,
    stride_beta_t,
    stride_beta_h,
    stride_tq_b,
    stride_tq_h,
    stride_tq_k,
    stride_co_b,
    stride_co_h,
    stride_co_t,
    K: tl.constexpr,
    HV_PER_H: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Backward pass once per (batch, value head): emit tq and the coefficients.

    This is the sequential part of the replay. It is isolated from the state
    readout so that it can run with tiny register pressure (one K-vector) and so
    that the expensive part -- streaming the FP8 checkpoint -- runs in a kernel
    that keeps its occupancy high.
    """
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_h = pid_hv // HV_PER_H
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    valid_len = tl.load(valid_len_ptr + pid_b).to(tl.int32)
    transformed_query = tl.load(
        q_ptr + pid_b * stride_q_b + pid_h * stride_q_h + offs_k * stride_q_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    for reverse_token in tl.static_range(0, MAX_CACHE_LEN):
        token = MAX_CACHE_LEN - 1 - reverse_token
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
        log_decay = tl.load(
            g_cache_ptr
            + pid_b * stride_g_b
            + token * stride_g_t
            + pid_hv * stride_g_h,
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
        coefficient = beta * tl.sum(key * transformed_query, axis=0)
        tl.store(
            coef_out_ptr
            + pid_b * stride_co_b
            + pid_hv * stride_co_h
            + token * stride_co_t,
            coefficient,
            mask=active,
        )
        transformed_query = tl.exp(log_decay) * (
            transformed_query - coefficient * key
        )
    tl.store(
        tq_out_ptr
        + pid_b * stride_tq_b
        + pid_hv * stride_tq_h
        + offs_k * stride_tq_k,
        transformed_query,
        mask=mask_k,
    )


@triton.jit
def _gdn_replay_apply_kernel(
    checkpoint_ptr,
    checkpoint_scale_ptr,
    tq_ptr,
    coef_ptr,
    v_cache_ptr,
    valid_len_ptr,
    out_ptr,
    stride_cp_b,
    stride_cp_h,
    stride_cp_v,
    stride_cp_k,
    stride_scale_b,
    stride_scale_h,
    stride_scale_vt,
    stride_tq_b,
    stride_tq_h,
    stride_tq_k,
    stride_co_b,
    stride_co_h,
    stride_co_t,
    stride_v_b,
    stride_v_t,
    stride_v_h,
    stride_v_v,
    stride_out_b,
    stride_out_h,
    stride_out_v,
    K: tl.constexpr,
    V: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    VBLOCK: tl.constexpr,
    BLOCK_KC: tl.constexpr,
):
    """Streaming part: out = sum_t coef_t * v_t + S0 @ tq (scaled)."""
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_vt = tl.program_id(2)
    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_vt * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_k = offs_k < K
    mask_v = offs_v < V
    valid_len = tl.load(valid_len_ptr + pid_b).to(tl.int32)

    scale_offsets = (
        pid_b * stride_scale_b
        + pid_hv * stride_scale_h
        + (offs_v // VBLOCK) * stride_scale_vt
    )
    checkpoint_scale = tl.load(
        checkpoint_scale_ptr + scale_offsets, mask=mask_v, other=1.0
    ).to(tl.float32)
    transformed_query = tl.load(
        tq_ptr
        + pid_b * stride_tq_b
        + pid_hv * stride_tq_h
        + offs_k * stride_tq_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    # Read the state in key chunks: the full [BLOCK_V, BLOCK_K] fp32 tile costs
    # ~128 registers per thread at BLOCK_V=128, which caps occupancy and drops
    # achieved bandwidth. Chunking keeps the live tile small, and is only
    # possible because tq now comes from memory instead of registers.
    output = tl.zeros([BLOCK_V], dtype=tl.float32)
    for key_chunk in tl.static_range(0, BLOCK_K // BLOCK_KC):
        offs_kc = key_chunk * BLOCK_KC + tl.arange(0, BLOCK_KC)
        mask_kc = offs_kc < K
        chunk_offsets = (
            pid_b * stride_cp_b
            + pid_hv * stride_cp_h
            + offs_v[:, None] * stride_cp_v
            + offs_kc[None, :] * stride_cp_k
        )
        chunk = tl.load(
            checkpoint_ptr + chunk_offsets,
            mask=mask_v[:, None] & mask_kc[None, :],
            other=0.0,
        ).to(tl.float32)
        tq_chunk = tl.load(
            tq_ptr
            + pid_b * stride_tq_b
            + pid_hv * stride_tq_h
            + offs_kc * stride_tq_k,
            mask=mask_kc,
            other=0.0,
        ).to(tl.float32)
        output += tl.sum(chunk * tq_chunk[None, :], axis=1)
    output *= checkpoint_scale

    for token in tl.static_range(0, MAX_CACHE_LEN):
        active = token < valid_len
        coefficient = tl.load(
            coef_ptr
            + pid_b * stride_co_b
            + pid_hv * stride_co_h
            + token * stride_co_t,
            mask=active,
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
        output += coefficient * value

    output_offsets = (
        pid_b * stride_out_b
        + pid_hv * stride_out_h
        + offs_v * stride_out_v
    )
    tl.store(out_ptr + output_offsets, output * (K**-0.5), mask=mask_v)


@triton.jit
def _gdn_replay_precompute_grouped_kernel(
    q_ptr,
    k_cache_ptr,
    g_cache_ptr,
    beta_cache_ptr,
    valid_len_ptr,
    tq_out_ptr,
    coef_out_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_k,
    stride_k_b,
    stride_k_t,
    stride_k_h,
    stride_k_k,
    stride_g_b,
    stride_g_t,
    stride_g_h,
    stride_beta_b,
    stride_beta_t,
    stride_beta_h,
    stride_tq_b,
    stride_tq_h,
    stride_tq_k,
    stride_co_b,
    stride_co_h,
    stride_co_t,
    K: tl.constexpr,
    HV_PER_H: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Same backward pass, but one program serves all value heads of a key head.

    ``k`` is shared by every value head of a key head, so the per-head version
    re-reads it HV_PER_H times. Grouping also doubles the work per loop
    iteration, which helps because the loop is latency-bound rather than
    bandwidth-bound.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_hv = tl.arange(0, HV_PER_H)
    mask_k = offs_k < K
    valid_len = tl.load(valid_len_ptr + pid_b).to(tl.int32)
    query = tl.load(
        q_ptr + pid_b * stride_q_b + pid_h * stride_q_h + offs_k * stride_q_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    transformed_query = tl.zeros([HV_PER_H, BLOCK_K], dtype=tl.float32) + query[
        None, :
    ]
    for reverse_token in tl.static_range(0, MAX_CACHE_LEN):
        token = MAX_CACHE_LEN - 1 - reverse_token
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
        log_decay = tl.load(
            g_cache_ptr
            + pid_b * stride_g_b
            + token * stride_g_t
            + pid_h * HV_PER_H
            + offs_hv * stride_g_h,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        beta = tl.load(
            beta_cache_ptr
            + pid_b * stride_beta_b
            + token * stride_beta_t
            + pid_h * HV_PER_H
            + offs_hv * stride_beta_h,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        coefficient = beta * tl.sum(
            key[None, :] * transformed_query, axis=1
        )
        tl.store(
            coef_out_ptr
            + pid_b * stride_co_b
            + (pid_h * HV_PER_H + offs_hv) * stride_co_h
            + token * stride_co_t,
            coefficient,
            mask=active,
        )
        transformed_query = tl.exp(log_decay)[:, None] * (
            transformed_query - coefficient[:, None] * key[None, :]
        )
    tl.store(
        tq_out_ptr
        + pid_b * stride_tq_b
        + (pid_h * HV_PER_H + offs_hv)[:, None] * stride_tq_h
        + offs_k[None, :] * stride_tq_k,
        transformed_query,
        mask=mask_k[None, :],
    )


def gdn_replay_fp8_split(
    checkpoint: torch.Tensor,
    checkpoint_scale: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    g_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    valid_len: torch.Tensor,
    transformed_query_buffer: torch.Tensor,
    coefficient_buffer: torch.Tensor,
    block_v: int = 32,
    num_warps: int = 4,
    block_kc: int = 128,
    precompute_grouped: bool = True,
    num_stages: int = 2,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Non-flush replay as precompute + streaming apply.

    ``transformed_query_buffer`` is [batch, value_heads, K] and
    ``coefficient_buffer`` is [batch, value_heads, max_cache_len]; both must be
    preallocated by the caller so the timed path has no allocator traffic.
    """
    batch, max_cache_len, heads, key_dim = k_cache.shape
    value_heads = v_cache.shape[2]
    value_dim = v_cache.shape[3]
    block_k = triton.next_power_of_2(key_dim)
    group_size = value_heads // heads
    # As above: an externally supplied output buffer keeps the timed path free of
    # allocator work.
    output = (
        torch.empty(
            batch, value_heads, value_dim, dtype=q.dtype, device=q.device
        )
        if out is None
        else out
    )
    if group_size > 1 and precompute_grouped:
        _gdn_replay_precompute_grouped_kernel[(batch, heads)](
            q,
            k_cache,
            g_cache,
            beta_cache,
            valid_len,
            transformed_query_buffer,
            coefficient_buffer,
            *q.stride(),
            *k_cache.stride(),
            *g_cache.stride(),
            *beta_cache.stride(),
            *transformed_query_buffer.stride(),
            *coefficient_buffer.stride(),
            K=key_dim,
            HV_PER_H=group_size,
            MAX_CACHE_LEN=max_cache_len,
            BLOCK_K=block_k,
            num_warps=1,
        )
    else:
        _gdn_replay_precompute_kernel[(batch, value_heads)](
            q,
            k_cache,
            g_cache,
            beta_cache,
            valid_len,
            transformed_query_buffer,
            coefficient_buffer,
            *q.stride(),
            *k_cache.stride(),
            *g_cache.stride(),
            *beta_cache.stride(),
            *transformed_query_buffer.stride(),
            *coefficient_buffer.stride(),
            K=key_dim,
            HV_PER_H=group_size,
            MAX_CACHE_LEN=max_cache_len,
            BLOCK_K=block_k,
            num_warps=1,
        )
    _gdn_replay_apply_kernel[
        (batch, value_heads, triton.cdiv(value_dim, block_v))
    ](
        checkpoint,
        checkpoint_scale,
        transformed_query_buffer,
        coefficient_buffer,
        v_cache,
        valid_len,
        output,
        *checkpoint.stride(),
        *checkpoint_scale.stride(),
        *transformed_query_buffer.stride(),
        *coefficient_buffer.stride(),
        *v_cache.stride(),
        *output.stride(),
        K=key_dim,
        V=value_dim,
        MAX_CACHE_LEN=max_cache_len,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        VBLOCK=32,
        BLOCK_KC=block_kc,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def make_inputs(
    batch: int,
    cache_len: int,
    heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = torch.nn.functional.normalize(
        torch.randn(batch, heads, key_dim, device=device), dim=-1
    ).to(dtype)
    k_cache = torch.nn.functional.normalize(
        torch.randn(batch, cache_len, heads, key_dim, device=device), dim=-1
    ).to(dtype)
    v_cache = torch.randn(
        batch,
        cache_len,
        value_heads,
        value_dim,
        device=device,
        dtype=dtype,
    )
    a = torch.randn(
        batch, cache_len, value_heads, device=device, dtype=torch.float32
    )
    a_log = torch.rand(value_heads, device=device)
    dt_bias = torch.rand(value_heads, device=device) - 4.0
    g_cache = (-a_log.exp() * torch.nn.functional.softplus(a + dt_bias)).to(
        dtype
    )
    beta_cache = torch.sigmoid(
        torch.randn(
            batch,
            cache_len,
            value_heads,
            device=device,
            dtype=torch.float32,
        )
    ).to(dtype)
    state = torch.randn(
        batch,
        value_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    ) * 0.02
    return q, k_cache, v_cache, g_cache, beta_cache, state


def validate_once(args: argparse.Namespace) -> dict[str, object]:
    q, k_cache, v_cache, g_cache, beta_cache, state = make_inputs(
        args.batch,
        args.cache_len,
        args.heads,
        args.value_heads,
        args.key_dim,
        args.value_dim,
        args.seed,
    )
    quantized = quantize_checkpoint(state, "vblock", 32)
    checkpoint = quantized.payload.clone()
    checkpoint_scale = quantized.scale.clone()
    valid_len = torch.full(
        (args.batch,), args.cache_len, device="cuda", dtype=torch.int32
    )
    flush = torch.zeros(args.batch, device="cuda", dtype=torch.bool)
    actual_output = gdn_replay_fp8(
        checkpoint,
        checkpoint_scale,
        q,
        k_cache,
        v_cache,
        g_cache,
        beta_cache,
        valid_len,
        flush,
    )
    reference_output, reference_state = run_kernel(
        dequantize_checkpoint(quantized),
        q.unsqueeze(1).expand(-1, args.cache_len, -1, -1),
        k_cache,
        v_cache,
        g_cache,
        beta_cache,
    )
    output_error = error_metrics(actual_output, reference_output[:, -1])

    flush.fill_(True)
    gdn_replay_fp8(
        checkpoint,
        checkpoint_scale,
        q,
        k_cache,
        v_cache,
        g_cache,
        beta_cache,
        valid_len,
        flush,
    )
    actual_state = dequantize_checkpoint(
        type(quantized)(
            checkpoint,
            checkpoint_scale,
            "vblock",
            32,
            0.0,
        )
    )
    expected_quantized = quantize_checkpoint(reference_state, "vblock", 32)
    expected_state = dequantize_checkpoint(expected_quantized)
    state_error = error_metrics(actual_state, expected_state)
    return {
        "output_error": output_error,
        "flush_state_error": state_error,
    }


def benchmark(args: argparse.Namespace) -> list[dict[str, float]]:
    results = []
    for batch in args.batches:
        q, k_cache, v_cache, g_cache, beta_cache, state = make_inputs(
            batch,
            args.cache_len,
            args.heads,
            args.value_heads,
            args.key_dim,
            args.value_dim,
            args.seed,
        )
        quantized = quantize_checkpoint(state, "vblock", 32)
        valid_len = torch.full(
            (batch,), args.cache_len, device="cuda", dtype=torch.int32
        )
        no_flush = torch.zeros(batch, device="cuda", dtype=torch.bool)
        flush = torch.ones(batch, device="cuda", dtype=torch.bool)
        replay_call = lambda: gdn_replay_fp8(
            quantized.payload,
            quantized.scale,
            q,
            k_cache,
            v_cache,
            g_cache,
            beta_cache,
            valid_len,
            no_flush,
        )
        flush_call = lambda: gdn_replay_fp8(
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
        bf16_state = state.to(torch.bfloat16)
        baseline_call = lambda: gdn_bf16_step(
            bf16_state,
            q,
            k_cache[:, -1],
            v_cache[:, -1],
            g_cache[:, -1],
            beta_cache[:, -1],
        )
        replay_ms = triton.testing.do_bench(replay_call, warmup=25, rep=100)
        flush_ms = triton.testing.do_bench(flush_call, warmup=25, rep=100)
        baseline_ms = triton.testing.do_bench(
            baseline_call, warmup=25, rep=100
        )
        amortized_ms = (
            replay_ms * (args.cache_len - 1) + flush_ms
        ) / args.cache_len
        results.append(
            {
                "batch": batch,
                "cache_len": args.cache_len,
                "replay_ms": replay_ms,
                "flush_ms": flush_ms,
                "amortized_ms": amortized_ms,
                "bf16_step_ms": baseline_ms,
                "speedup_vs_bf16": baseline_ms / amortized_ms,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--batches", default="1,2,4,8,16")
    parser.add_argument("--cache-len", type=int, default=8)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    args.batches = [int(value) for value in args.batches.split(",")]
    return args


def main() -> None:
    args = parse_args()
    print(json.dumps({"type": "validation", **validate_once(args)}))
    if not args.skip_benchmark:
        for result in benchmark(args):
            print(json.dumps({"type": "benchmark", **result}))


if __name__ == "__main__":
    main()
