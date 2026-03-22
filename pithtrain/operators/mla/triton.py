"""
Triton implementation of the MLA operator.

Multi-head attention with BSHD layout and causal masking.

 b : batch size
 s : sequence length
 h : number of heads
dq : query/key head dimension (192 = 128 + 64)
dv : value head dimension (128)
"""

import itertools
import math
from typing import Tuple

import torch
import triton
import triton.language as tl

from pithtrain.operators.mla.pytorch import MLA as BaseMLA

# fmt: off
# mypy: ignore-errors

options = dict(blk_m=[64, 128], blk_n=[32, 64, 128], num_stages=[1, 2, 3], num_warps=[4, 8])
combine = lambda blk_m, blk_n, ns, nw: triton.Config(dict(blk_m=blk_m, blk_n=blk_n), num_stages=ns, num_warps=nw)
configs = list(itertools.starmap(combine, itertools.product(*options.values())))

@triton.autotune(configs=configs, key=["s", "h", "dq", "dv"])
@triton.jit
def fwd_kernel(
    q_base: tl.pointer_type, k_base: tl.pointer_type, v_base: tl.pointer_type, o_base: tl.pointer_type, lse_base: tl.pointer_type,
    b: tl.constexpr, s: tl.constexpr, h: tl.constexpr, dq: tl.constexpr, dv: tl.constexpr, dq1: tl.constexpr, dq2: tl.constexpr, softmax_scale: tl.constexpr,
    blk_m: tl.constexpr, blk_n: tl.constexpr,
):
    """
    Attention forward with causal masking.
    """
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    idx_m = tl.program_id(0)
    idx_h = tl.program_id(1)
    idx_b = tl.program_id(2)
    # Strides and base offsets for this (b, h) slice.
    stride_qs = h * dq
    stride_vs = h * dv
    qk_off = idx_b * s * stride_qs + idx_h * dq
    vo_off = idx_b * s * stride_vs + idx_h * dv
    lse_off = idx_b * h * s + idx_h * s
    # TMA tensor descriptors (dq split into dq1 + dq2 for power-of-2 requirement).
    q_desc1 = tl.make_tensor_descriptor(q_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq1])
    q_desc2 = tl.make_tensor_descriptor(q_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq2])
    k_desc1 = tl.make_tensor_descriptor(k_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq1])
    k_desc2 = tl.make_tensor_descriptor(k_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq2])
    v_desc = tl.make_tensor_descriptor(v_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_n, dv])
    o_desc = tl.make_tensor_descriptor(o_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_m, dv])
    lse_desc = tl.make_tensor_descriptor(lse_base + lse_off, shape=[s], strides=[1], block_shape=[blk_m])
    # Load Q tile (split).
    m_off = idx_m * blk_m
    q1 = q_desc1.load([m_off, 0])
    q2 = q_desc2.load([m_off, dq1])
    # Online softmax state and output accumulator.
    rowmax = tl.full([blk_m], value=float('-inf'), dtype=tl.float32)
    normalizer = tl.zeros([blk_m], dtype=tl.float32)
    output = tl.zeros([blk_m, dv], dtype=tl.float32)
    # Main loop (causal: K/V blocks up to current M position).
    for idx_n in tl.range(tl.cdiv((idx_m + 1) * blk_m, blk_n)):
        n_off = idx_n * blk_n
        k1 = k_desc1.load([n_off, 0])
        k2 = k_desc2.load([n_off, dq1])
        v = v_desc.load([n_off, 0])
        # Compute pre-softmax logits with causal masking.
        scores = tl.dot(q1, k1.T)
        scores = tl.dot(q2, k2.T, scores)
        m_indices = m_off + tl.arange(0, blk_m)[:, None]
        n_indices = n_off + tl.arange(0, blk_n)[None, :]
        scores = tl.where(n_indices > m_indices, float('-inf'), scores)
        # Online softmax update.
        rowmax_prev = rowmax
        rowmax_new = tl.max(scores, axis=1)
        rowmax = tl.maximum(rowmax, rowmax_new)
        rescale = tl.exp2(rowmax_prev * softmax_scale_log2 - rowmax * softmax_scale_log2)
        scores = tl.exp2(scores * softmax_scale_log2 - rowmax[:, None] * softmax_scale_log2)
        rowsum = tl.sum(scores, axis=1)
        normalizer = normalizer * rescale + rowsum
        # Accumulate output += softmax(scores) @ V.
        output = output * rescale[:, None]
        output = tl.dot(scores.to(tl.bfloat16), v, output)
    # Normalize output and store log-sum-exp.
    output = output / normalizer[:, None]
    o_desc.store([m_off, 0], output.to(tl.bfloat16))
    lse = tl.log2(normalizer) + rowmax * softmax_scale_log2
    lse_desc.store([m_off], lse)

options = dict(blk=[64, 128], num_stages=[1, 2], num_warps=[4, 8])
combine = lambda blk, ns, nw: triton.Config(dict(blk=blk), num_stages=ns, num_warps=nw)
configs = list(itertools.starmap(combine, itertools.product(*options.values())))

@triton.autotune(configs=configs, key=["s", "h", "dv"])
@triton.jit
def bwd_phase_1_kernel(
    o_base: tl.pointer_type, do_base: tl.pointer_type, delta_base: tl.pointer_type,
    b: tl.constexpr, s: tl.constexpr, h: tl.constexpr, dv: tl.constexpr,
    blk: tl.constexpr,
):
    """
    Phase 1 of attention backward: compute delta = rowsum(o * do).
    """
    idx_s = tl.program_id(0)
    idx_h = tl.program_id(1)
    idx_b = tl.program_id(2)
    stride_vs = h * dv
    vo_off = idx_b * s * stride_vs + idx_h * dv
    lse_off = idx_b * h * s + idx_h * s
    o_desc = tl.make_tensor_descriptor(o_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk, dv])
    do_desc = tl.make_tensor_descriptor(do_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk, dv])
    delta_desc = tl.make_tensor_descriptor(delta_base + lse_off, shape=[s], strides=[1], block_shape=[blk])
    s_off = idx_s * blk
    o_tile = o_desc.load([s_off, 0]).to(tl.float32)
    do_tile = do_desc.load([s_off, 0]).to(tl.float32)
    delta = tl.sum(o_tile * do_tile, axis=1)
    delta_desc.store([s_off], delta)

options = dict(blk_n=[64, 128], blk_m=[64, 128], num_stages=[1, 2, 3], num_warps=[4, 8])
combine = lambda blk_n, blk_m, ns, nw: triton.Config(dict(blk_n=blk_n, blk_m=blk_m), num_stages=ns, num_warps=nw)
configs = list(itertools.starmap(combine, itertools.product(*options.values())))

@triton.autotune(configs=configs, key=["s", "h", "dq", "dv"])
@triton.jit
def bwd_phase_2_kernel(
    q_base: tl.pointer_type, k_base: tl.pointer_type, v_base: tl.pointer_type, do_base: tl.pointer_type, lse_base: tl.pointer_type, delta_base: tl.pointer_type, dk_base: tl.pointer_type, dv_base: tl.pointer_type,
    b: tl.constexpr, s: tl.constexpr, h: tl.constexpr, dq: tl.constexpr, dv: tl.constexpr, dq1: tl.constexpr, dq2: tl.constexpr, softmax_scale: tl.constexpr,
    blk_n: tl.constexpr, blk_m: tl.constexpr,
):
    """
    Phase 2 of attention backward: compute dK and dV (K outer loop, Q inner loop).
    """
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    idx_n = tl.program_id(0)
    idx_h = tl.program_id(1)
    idx_b = tl.program_id(2)
    # Strides and base offsets for this (b, h) slice.
    stride_qs = h * dq
    stride_vs = h * dv
    qk_off = idx_b * s * stride_qs + idx_h * dq
    vo_off = idx_b * s * stride_vs + idx_h * dv
    lse_off = idx_b * h * s + idx_h * s
    # TMA tensor descriptors.
    k_desc1 = tl.make_tensor_descriptor(k_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq1])
    k_desc2 = tl.make_tensor_descriptor(k_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq2])
    v_desc = tl.make_tensor_descriptor(v_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_n, dv])
    q_desc1 = tl.make_tensor_descriptor(q_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq1])
    q_desc2 = tl.make_tensor_descriptor(q_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq2])
    do_desc = tl.make_tensor_descriptor(do_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_m, dv])
    lse_desc = tl.make_tensor_descriptor(lse_base + lse_off, shape=[s], strides=[1], block_shape=[blk_m])
    delta_desc = tl.make_tensor_descriptor(delta_base + lse_off, shape=[s], strides=[1], block_shape=[blk_m])
    dk_desc1 = tl.make_tensor_descriptor(dk_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq1])
    dk_desc2 = tl.make_tensor_descriptor(dk_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq2])
    dv_desc = tl.make_tensor_descriptor(dv_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_n, dv])
    # Load K, V tiles held for the entire loop.
    n_off = idx_n * blk_n
    k1 = k_desc1.load([n_off, 0])
    k2 = k_desc2.load([n_off, dq1])
    v = v_desc.load([n_off, 0])
    # Gradient accumulators for dK and dV.
    dk1_acc = tl.zeros([blk_n, dq1], dtype=tl.float32)
    dk2_acc = tl.zeros([blk_n, dq2], dtype=tl.float32)
    dv_acc = tl.zeros([blk_n, dv], dtype=tl.float32)
    # Main loop over Q blocks (causal: only positions >= current K).
    start_m = (idx_n * blk_n) // blk_m
    for idx_m in tl.range(start_m, tl.cdiv(s, blk_m)):
        m_off = idx_m * blk_m
        q1 = q_desc1.load([m_off, 0])
        q2 = q_desc2.load([m_off, dq1])
        do_val = do_desc.load([m_off, 0])
        lse = lse_desc.load([m_off])
        delta_val = delta_desc.load([m_off])
        # Recompute attention scores with causal masking.
        scores = tl.dot(k1, q1.T)
        scores = tl.dot(k2, q2.T, scores)
        scores = tl.exp2(scores * softmax_scale_log2 - lse[None, :])
        n_indices = n_off + tl.arange(0, blk_n)[:, None]
        m_indices = m_off + tl.arange(0, blk_m)[None, :]
        scores = tl.where(n_indices > m_indices, 0.0, scores)
        # Compute dV += P @ dO.
        dv_acc = tl.dot(scores.to(tl.bfloat16), do_val, dv_acc)
        # Compute dS^T = P * (V @ dO^T - delta) * scale.
        dsT = tl.dot(v, do_val.T)
        dsT = scores * (dsT - delta_val[None, :]) * softmax_scale
        dsT_bf16 = dsT.to(tl.bfloat16)
        # Compute dK += dS^T @ Q (split).
        dk1_acc = tl.dot(dsT_bf16, q1, dk1_acc)
        dk2_acc = tl.dot(dsT_bf16, q2, dk2_acc)
    # Write accumulated dK and dV to global memory.
    dk_desc1.store([n_off, 0], dk1_acc.to(tl.bfloat16))
    dk_desc2.store([n_off, dq1], dk2_acc.to(tl.bfloat16))
    dv_desc.store([n_off, 0], dv_acc.to(tl.bfloat16))

options = dict(blk_m=[64, 128], blk_n=[32, 64, 128], num_stages=[1, 2, 3], num_warps=[4, 8])
combine = lambda blk_m, blk_n, ns, nw: triton.Config(dict(blk_m=blk_m, blk_n=blk_n), num_stages=ns, num_warps=nw)
configs = list(itertools.starmap(combine, itertools.product(*options.values())))

@triton.autotune(configs=configs, key=["s", "h", "dq", "dv"])
@triton.jit
def bwd_phase_3_kernel(
    q_base: tl.pointer_type, k_base: tl.pointer_type, v_base: tl.pointer_type, do_base: tl.pointer_type, lse_base: tl.pointer_type, delta_base: tl.pointer_type, dq_base: tl.pointer_type,
    b: tl.constexpr, s: tl.constexpr, h: tl.constexpr, dq: tl.constexpr, dv: tl.constexpr, dq1: tl.constexpr, dq2: tl.constexpr, softmax_scale: tl.constexpr,
    blk_m: tl.constexpr, blk_n: tl.constexpr,
):
    """
    Phase 3 of attention backward: compute dQ (Q outer loop, K inner loop).
    """
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    idx_m = tl.program_id(0)
    idx_h = tl.program_id(1)
    idx_b = tl.program_id(2)
    # Strides and base offsets for this (b, h) slice.
    stride_qs = h * dq
    stride_vs = h * dv
    qk_off = idx_b * s * stride_qs + idx_h * dq
    vo_off = idx_b * s * stride_vs + idx_h * dv
    lse_off = idx_b * h * s + idx_h * s
    # TMA tensor descriptors.
    q_desc1 = tl.make_tensor_descriptor(q_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq1])
    q_desc2 = tl.make_tensor_descriptor(q_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq2])
    k_desc1 = tl.make_tensor_descriptor(k_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq1])
    k_desc2 = tl.make_tensor_descriptor(k_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq2])
    v_desc = tl.make_tensor_descriptor(v_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_n, dv])
    do_desc = tl.make_tensor_descriptor(do_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_m, dv])
    lse_desc = tl.make_tensor_descriptor(lse_base + lse_off, shape=[s], strides=[1], block_shape=[blk_m])
    delta_desc = tl.make_tensor_descriptor(delta_base + lse_off, shape=[s], strides=[1], block_shape=[blk_m])
    dq_desc1 = tl.make_tensor_descriptor(dq_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq1])
    dq_desc2 = tl.make_tensor_descriptor(dq_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq2])
    # Load Q, dO, LSE, delta held for the entire loop.
    m_off = idx_m * blk_m
    q1 = q_desc1.load([m_off, 0])
    q2 = q_desc2.load([m_off, dq1])
    do_val = do_desc.load([m_off, 0])
    lse = lse_desc.load([m_off])
    delta_val = delta_desc.load([m_off])
    # Gradient accumulators for dQ (split).
    dq1_acc = tl.zeros([blk_m, dq1], dtype=tl.float32)
    dq2_acc = tl.zeros([blk_m, dq2], dtype=tl.float32)
    # Main loop over K blocks (causal: only positions <= current Q).
    for idx_n in tl.range(tl.cdiv((idx_m + 1) * blk_m, blk_n)):
        n_off = idx_n * blk_n
        k1 = k_desc1.load([n_off, 0])
        k2 = k_desc2.load([n_off, dq1])
        v = v_desc.load([n_off, 0])
        # Recompute attention scores with causal masking.
        scores = tl.dot(q1, k1.T)
        scores = tl.dot(q2, k2.T, scores)
        scores = tl.exp2(scores * softmax_scale_log2 - lse[:, None])
        m_indices = m_off + tl.arange(0, blk_m)[:, None]
        n_indices = n_off + tl.arange(0, blk_n)[None, :]
        scores = tl.where(n_indices > m_indices, 0.0, scores)
        # Compute dS = P * (dO @ V^T - delta) * scale.
        ds = tl.dot(do_val, v.T)
        ds = scores * (ds - delta_val[:, None]) * softmax_scale
        ds_bf16 = ds.to(tl.bfloat16)
        # Compute dQ += dS @ K (split).
        dq1_acc = tl.dot(ds_bf16, k1, dq1_acc)
        dq2_acc = tl.dot(ds_bf16, k2, dq2_acc)
    # Write dQ to global memory.
    dq_desc1.store([m_off, 0], dq1_acc.to(tl.bfloat16))
    dq_desc2.store([m_off, dq1], dq2_acc.to(tl.bfloat16))

options = dict(blk_m=[64, 128], blk_n=[32, 64, 128], num_stages=[1, 2, 3], num_warps=[4, 8])
combine = lambda blk_m, blk_n, ns, nw: triton.Config(dict(blk_m=blk_m, blk_n=blk_n), num_stages=ns, num_warps=nw)
configs_fwd_nc = list(itertools.starmap(combine, itertools.product(*options.values())))

@triton.autotune(configs=configs_fwd_nc, key=["s", "h", "dq", "dv"])
@triton.jit
def fwd_kernel_non_causal(
    q_base: tl.pointer_type, k_base: tl.pointer_type, v_base: tl.pointer_type, o_base: tl.pointer_type, lse_base: tl.pointer_type,
    b: tl.constexpr, s: tl.constexpr, h: tl.constexpr, dq: tl.constexpr, dv: tl.constexpr, dq1: tl.constexpr, dq2: tl.constexpr, softmax_scale: tl.constexpr,
    blk_m: tl.constexpr, blk_n: tl.constexpr,
):
    """
    Attention forward without causal masking.
    """
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    idx_m = tl.program_id(0)
    idx_h = tl.program_id(1)
    idx_b = tl.program_id(2)
    stride_qs = h * dq
    stride_vs = h * dv
    qk_off = idx_b * s * stride_qs + idx_h * dq
    vo_off = idx_b * s * stride_vs + idx_h * dv
    lse_off = idx_b * h * s + idx_h * s
    q_desc1 = tl.make_tensor_descriptor(q_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq1])
    q_desc2 = tl.make_tensor_descriptor(q_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq2])
    k_desc1 = tl.make_tensor_descriptor(k_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq1])
    k_desc2 = tl.make_tensor_descriptor(k_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq2])
    v_desc = tl.make_tensor_descriptor(v_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_n, dv])
    o_desc = tl.make_tensor_descriptor(o_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_m, dv])
    lse_desc = tl.make_tensor_descriptor(lse_base + lse_off, shape=[s], strides=[1], block_shape=[blk_m])
    m_off = idx_m * blk_m
    q1 = q_desc1.load([m_off, 0])
    q2 = q_desc2.load([m_off, dq1])
    rowmax = tl.full([blk_m], value=float('-inf'), dtype=tl.float32)
    normalizer = tl.zeros([blk_m], dtype=tl.float32)
    output = tl.zeros([blk_m, dv], dtype=tl.float32)
    for idx_n in tl.range(tl.cdiv(s, blk_n)):
        n_off = idx_n * blk_n
        k1 = k_desc1.load([n_off, 0])
        k2 = k_desc2.load([n_off, dq1])
        v = v_desc.load([n_off, 0])
        scores = tl.dot(q1, k1.T)
        scores = tl.dot(q2, k2.T, scores)
        rowmax_prev = rowmax
        rowmax_new = tl.max(scores, axis=1)
        rowmax = tl.maximum(rowmax, rowmax_new)
        rescale = tl.exp2(rowmax_prev * softmax_scale_log2 - rowmax * softmax_scale_log2)
        scores = tl.exp2(scores * softmax_scale_log2 - rowmax[:, None] * softmax_scale_log2)
        rowsum = tl.sum(scores, axis=1)
        normalizer = normalizer * rescale + rowsum
        output = output * rescale[:, None]
        output = tl.dot(scores.to(tl.bfloat16), v, output)
    output = output / normalizer[:, None]
    o_desc.store([m_off, 0], output.to(tl.bfloat16))
    lse = tl.log2(normalizer) + rowmax * softmax_scale_log2
    lse_desc.store([m_off], lse)

options = dict(blk_n=[64, 128], blk_m=[64, 128], num_stages=[1, 2, 3], num_warps=[4, 8])
combine = lambda blk_n, blk_m, ns, nw: triton.Config(dict(blk_n=blk_n, blk_m=blk_m), num_stages=ns, num_warps=nw)
configs_bwd2_nc = list(itertools.starmap(combine, itertools.product(*options.values())))

@triton.autotune(configs=configs_bwd2_nc, key=["s", "h", "dq", "dv"])
@triton.jit
def bwd_phase_2_kernel_non_causal(
    q_base: tl.pointer_type, k_base: tl.pointer_type, v_base: tl.pointer_type, do_base: tl.pointer_type, lse_base: tl.pointer_type, delta_base: tl.pointer_type, dk_base: tl.pointer_type, dv_base: tl.pointer_type,
    b: tl.constexpr, s: tl.constexpr, h: tl.constexpr, dq: tl.constexpr, dv: tl.constexpr, dq1: tl.constexpr, dq2: tl.constexpr, softmax_scale: tl.constexpr,
    blk_n: tl.constexpr, blk_m: tl.constexpr,
):
    """
    Phase 2 of attention backward without causal masking: compute dK and dV.
    """
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    idx_n = tl.program_id(0)
    idx_h = tl.program_id(1)
    idx_b = tl.program_id(2)
    stride_qs = h * dq
    stride_vs = h * dv
    qk_off = idx_b * s * stride_qs + idx_h * dq
    vo_off = idx_b * s * stride_vs + idx_h * dv
    lse_off = idx_b * h * s + idx_h * s
    k_desc1 = tl.make_tensor_descriptor(k_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq1])
    k_desc2 = tl.make_tensor_descriptor(k_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq2])
    v_desc = tl.make_tensor_descriptor(v_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_n, dv])
    q_desc1 = tl.make_tensor_descriptor(q_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq1])
    q_desc2 = tl.make_tensor_descriptor(q_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq2])
    do_desc = tl.make_tensor_descriptor(do_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_m, dv])
    lse_desc = tl.make_tensor_descriptor(lse_base + lse_off, shape=[s], strides=[1], block_shape=[blk_m])
    delta_desc = tl.make_tensor_descriptor(delta_base + lse_off, shape=[s], strides=[1], block_shape=[blk_m])
    dk_desc1 = tl.make_tensor_descriptor(dk_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq1])
    dk_desc2 = tl.make_tensor_descriptor(dk_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq2])
    dv_desc = tl.make_tensor_descriptor(dv_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_n, dv])
    n_off = idx_n * blk_n
    k1 = k_desc1.load([n_off, 0])
    k2 = k_desc2.load([n_off, dq1])
    v = v_desc.load([n_off, 0])
    dk1_acc = tl.zeros([blk_n, dq1], dtype=tl.float32)
    dk2_acc = tl.zeros([blk_n, dq2], dtype=tl.float32)
    dv_acc = tl.zeros([blk_n, dv], dtype=tl.float32)
    for idx_m in tl.range(0, tl.cdiv(s, blk_m)):
        m_off = idx_m * blk_m
        q1 = q_desc1.load([m_off, 0])
        q2 = q_desc2.load([m_off, dq1])
        do_val = do_desc.load([m_off, 0])
        lse = lse_desc.load([m_off])
        delta_val = delta_desc.load([m_off])
        scores = tl.dot(k1, q1.T)
        scores = tl.dot(k2, q2.T, scores)
        scores = tl.exp2(scores * softmax_scale_log2 - lse[None, :])
        dv_acc = tl.dot(scores.to(tl.bfloat16), do_val, dv_acc)
        dsT = tl.dot(v, do_val.T)
        dsT = scores * (dsT - delta_val[None, :]) * softmax_scale
        dsT_bf16 = dsT.to(tl.bfloat16)
        dk1_acc = tl.dot(dsT_bf16, q1, dk1_acc)
        dk2_acc = tl.dot(dsT_bf16, q2, dk2_acc)
    dk_desc1.store([n_off, 0], dk1_acc.to(tl.bfloat16))
    dk_desc2.store([n_off, dq1], dk2_acc.to(tl.bfloat16))
    dv_desc.store([n_off, 0], dv_acc.to(tl.bfloat16))

options = dict(blk_m=[64, 128], blk_n=[32, 64, 128], num_stages=[1, 2, 3], num_warps=[4, 8])
combine = lambda blk_m, blk_n, ns, nw: triton.Config(dict(blk_m=blk_m, blk_n=blk_n), num_stages=ns, num_warps=nw)
configs_bwd3_nc = list(itertools.starmap(combine, itertools.product(*options.values())))

@triton.autotune(configs=configs_bwd3_nc, key=["s", "h", "dq", "dv"])
@triton.jit
def bwd_phase_3_kernel_non_causal(
    q_base: tl.pointer_type, k_base: tl.pointer_type, v_base: tl.pointer_type, do_base: tl.pointer_type, lse_base: tl.pointer_type, delta_base: tl.pointer_type, dq_base: tl.pointer_type,
    b: tl.constexpr, s: tl.constexpr, h: tl.constexpr, dq: tl.constexpr, dv: tl.constexpr, dq1: tl.constexpr, dq2: tl.constexpr, softmax_scale: tl.constexpr,
    blk_m: tl.constexpr, blk_n: tl.constexpr,
):
    """
    Phase 3 of attention backward without causal masking: compute dQ.
    """
    softmax_scale_log2 = softmax_scale * math.log2(math.e)
    idx_m = tl.program_id(0)
    idx_h = tl.program_id(1)
    idx_b = tl.program_id(2)
    stride_qs = h * dq
    stride_vs = h * dv
    qk_off = idx_b * s * stride_qs + idx_h * dq
    vo_off = idx_b * s * stride_vs + idx_h * dv
    lse_off = idx_b * h * s + idx_h * s
    q_desc1 = tl.make_tensor_descriptor(q_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq1])
    q_desc2 = tl.make_tensor_descriptor(q_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq2])
    k_desc1 = tl.make_tensor_descriptor(k_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq1])
    k_desc2 = tl.make_tensor_descriptor(k_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_n, dq2])
    v_desc = tl.make_tensor_descriptor(v_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_n, dv])
    do_desc = tl.make_tensor_descriptor(do_base + vo_off, shape=[s, dv], strides=[stride_vs, 1], block_shape=[blk_m, dv])
    lse_desc = tl.make_tensor_descriptor(lse_base + lse_off, shape=[s], strides=[1], block_shape=[blk_m])
    delta_desc = tl.make_tensor_descriptor(delta_base + lse_off, shape=[s], strides=[1], block_shape=[blk_m])
    dq_desc1 = tl.make_tensor_descriptor(dq_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq1])
    dq_desc2 = tl.make_tensor_descriptor(dq_base + qk_off, shape=[s, dq], strides=[stride_qs, 1], block_shape=[blk_m, dq2])
    m_off = idx_m * blk_m
    q1 = q_desc1.load([m_off, 0])
    q2 = q_desc2.load([m_off, dq1])
    do_val = do_desc.load([m_off, 0])
    lse = lse_desc.load([m_off])
    delta_val = delta_desc.load([m_off])
    dq1_acc = tl.zeros([blk_m, dq1], dtype=tl.float32)
    dq2_acc = tl.zeros([blk_m, dq2], dtype=tl.float32)
    for idx_n in tl.range(tl.cdiv(s, blk_n)):
        n_off = idx_n * blk_n
        k1 = k_desc1.load([n_off, 0])
        k2 = k_desc2.load([n_off, dq1])
        v = v_desc.load([n_off, 0])
        scores = tl.dot(q1, k1.T)
        scores = tl.dot(q2, k2.T, scores)
        scores = tl.exp2(scores * softmax_scale_log2 - lse[:, None])
        ds = tl.dot(do_val, v.T)
        ds = scores * (ds - delta_val[:, None]) * softmax_scale
        ds_bf16 = ds.to(tl.bfloat16)
        dq1_acc = tl.dot(ds_bf16, k1, dq1_acc)
        dq2_acc = tl.dot(ds_bf16, k2, dq2_acc)
    dq_desc1.store([m_off, 0], dq1_acc.to(tl.bfloat16))
    dq_desc2.store([m_off, dq1], dq2_acc.to(tl.bfloat16))

_triton_delta = None

@torch.library.custom_op("pithtrain::mla_triton_fwd", mutates_args=())
def _mla_triton_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    (b, s, h, d_q), d_v = q.shape, v.shape[-1]
    assert d_q == 192 and d_v == 128, f"Triton MLA kernel assumes dq=192 and dv=128, got dq={d_q} and dv={d_v}"
    dq1, dq2 = 128, d_q - 128
    o = torch.empty((b, s, h, d_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    allocator = lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(allocator)
    grid = lambda meta: (triton.cdiv(s, meta["blk_m"]), h, b)
    fwd_kernel[grid](q, k, v, o, lse, b, s, h, d_q, d_v, dq1, dq2, softmax_scale)
    return o, lse

@_mla_triton_fwd.register_fake
def _(q, k, v, softmax_scale):
    (b, s, h, _), d_v = q.shape, v.shape[-1]
    o = torch.empty((b, s, h, d_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    return o, lse

@torch.library.custom_op("pithtrain::mla_triton_bwd", mutates_args=())
def _mla_triton_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, softmax_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _triton_delta
    allocator = lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(allocator)
    (b, s, h, d_q), d_v = q.shape, v.shape[-1]
    dq1, dq2 = 128, d_q - 128
    # Phase 1: delta = rowsum(o * do).
    if _triton_delta is None or _triton_delta.shape != (b, h, s):
        _triton_delta = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    delta = _triton_delta
    grid1 = lambda meta: (triton.cdiv(s, meta["blk"]), h, b)
    bwd_phase_1_kernel[grid1](o, do, delta, b, s, h, d_v)
    # Phase 2: dK, dV.
    dk = torch.empty((b, s, h, d_q), dtype=torch.bfloat16, device=q.device)
    dv_out = torch.empty((b, s, h, d_v), dtype=torch.bfloat16, device=q.device)
    grid2a = lambda meta: (triton.cdiv(s, meta["blk_n"]), h, b)
    bwd_phase_2_kernel[grid2a](q, k, v, do, lse, delta, dk, dv_out, b, s, h, d_q, d_v, dq1, dq2, softmax_scale)
    # Phase 3: dQ.
    dq_out = torch.empty((b, s, h, d_q), dtype=torch.bfloat16, device=q.device)
    grid2b = lambda meta: (triton.cdiv(s, meta["blk_m"]), h, b)
    bwd_phase_3_kernel[grid2b](q, k, v, do, lse, delta, dq_out, b, s, h, d_q, d_v, dq1, dq2, softmax_scale)
    return dq_out, dk, dv_out

@_mla_triton_bwd.register_fake
def _(q, k, v, o, lse, do, softmax_scale):
    (b, s, h, d_q), d_v = q.shape, v.shape[-1]
    dq_out = torch.empty((b, s, h, d_q), dtype=torch.bfloat16, device=q.device)
    dk = torch.empty((b, s, h, d_q), dtype=torch.bfloat16, device=q.device)
    dv_out = torch.empty((b, s, h, d_v), dtype=torch.bfloat16, device=q.device)
    return dq_out, dk, dv_out

def _mla_triton_setup_context(ctx, inputs, output):
    q, k, v, softmax_scale = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse)
    ctx.softmax_scale = softmax_scale

def _mla_triton_backward(ctx, grad_o, grad_lse):
    q, k, v, o, lse = ctx.saved_tensors
    dq, dk, dv = _mla_triton_bwd(q, k, v, o, lse, grad_o, ctx.softmax_scale)
    return dq, dk, dv, None

_mla_triton_fwd.register_autograd(_mla_triton_backward, setup_context=_mla_triton_setup_context)

class MLA(BaseMLA):
    """
    Triton implementation of the MLA operator.
    """
    @staticmethod
    def autotune(b: int, s: int, h: int, dq: int, dv: int, softmax_scale: float, include_non_causal: bool = False) -> None:
        assert dq == 192 and dv == 128, f"Triton MLA kernel assumes dq=192 and dv=128, got dq={dq} and dv={dv}"
        allocator = lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
        triton.set_allocator(allocator)
        q = torch.randn(b, s, h, dq, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(b, s, h, dq, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(b, s, h, dv, dtype=torch.bfloat16, device="cuda")
        dq1, dq2 = 128, dq - 128
        # Autotune fwd_kernel.
        o = torch.empty(b, s, h, dv, dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(b, h, s, dtype=torch.float32, device="cuda")
        grid_fwd = lambda meta: (triton.cdiv(s, meta["blk_m"]), h, b)
        fwd_kernel[grid_fwd](q, k, v, o, lse, b, s, h, dq, dv, dq1, dq2, softmax_scale)
        # Autotune bwd_phase_1_kernel.
        do = torch.randn(b, s, h, dv, dtype=torch.bfloat16, device="cuda")
        delta = torch.empty(b, h, s, dtype=torch.float32, device="cuda")
        grid_p1 = lambda meta: (triton.cdiv(s, meta["blk"]), h, b)
        bwd_phase_1_kernel[grid_p1](o, do, delta, b, s, h, dv)
        # Autotune bwd_phase_2_kernel (dK, dV).
        dk = torch.empty(b, s, h, dq, dtype=torch.bfloat16, device="cuda")
        dv_out = torch.empty(b, s, h, dv, dtype=torch.bfloat16, device="cuda")
        grid_p2a = lambda meta: (triton.cdiv(s, meta["blk_n"]), h, b)
        bwd_phase_2_kernel[grid_p2a](q, k, v, do, lse, delta, dk, dv_out, b, s, h, dq, dv, dq1, dq2, softmax_scale)
        # Autotune bwd_phase_3_kernel (dQ).
        dq_out = torch.empty(b, s, h, dq, dtype=torch.bfloat16, device="cuda")
        grid_p2b = lambda meta: (triton.cdiv(s, meta["blk_m"]), h, b)
        bwd_phase_3_kernel[grid_p2b](q, k, v, do, lse, delta, dq_out, b, s, h, dq, dv, dq1, dq2, softmax_scale)
        if include_non_causal:
            fwd_kernel_non_causal[grid_fwd](q, k, v, o, lse, b, s, h, dq, dv, dq1, dq2, softmax_scale)
            bwd_phase_2_kernel_non_causal[grid_p2a](q, k, v, do, lse, delta, dk, dv_out, b, s, h, dq, dv, dq1, dq2, softmax_scale)
            bwd_phase_3_kernel_non_causal[grid_p2b](q, k, v, do, lse, delta, dq_out, b, s, h, dq, dv, dq1, dq2, softmax_scale)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        o, _ = _mla_triton_fwd(q, k, v, self.softmax_scale)
        return o


def _mla_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float, causal: bool = True):
    """Raw MLA forward. Returns (out, lse)."""
    allocator = lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(allocator)
    (b, s, h, dq_dim), dv_dim = q.shape, v.shape[-1]
    dq1, dq2 = 128, dq_dim - 128
    o = torch.empty((b, s, h, dv_dim), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    grid = lambda meta: (triton.cdiv(s, meta["blk_m"]), h, b)
    kernel = fwd_kernel if causal else fwd_kernel_non_causal
    kernel[grid](q, k, v, o, lse, b, s, h, dq_dim, dv_dim, dq1, dq2, softmax_scale)
    return o, lse


def _mla_bwd_preprocess(out: torch.Tensor, dout: torch.Tensor):
    """Phase 1: delta = rowsum(out * dout). Called once with combined_out."""
    allocator = lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(allocator)
    b, s, h, dv_dim = out.shape
    delta = torch.empty((b, h, s), dtype=torch.float32, device=out.device)
    grid = lambda meta: (triton.cdiv(s, meta["blk"]), h, b)
    bwd_phase_1_kernel[grid](out, dout, delta, b, s, h, dv_dim)
    return delta


def _mla_bwd_dk_dv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dout: torch.Tensor,
                    lse: torch.Tensor, delta: torch.Tensor, softmax_scale: float, causal: bool = True):
    """Phase 2: compute dK and dV using external lse and delta. Returns (dk, dv)."""
    allocator = lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(allocator)
    (b, s, h, dq_dim), dv_dim = q.shape, v.shape[-1]
    dq1, dq2 = 128, dq_dim - 128
    dk = torch.empty((b, s, h, dq_dim), dtype=torch.bfloat16, device=q.device)
    dv_out = torch.empty((b, s, h, dv_dim), dtype=torch.bfloat16, device=q.device)
    grid = lambda meta: (triton.cdiv(s, meta["blk_n"]), h, b)
    kernel = bwd_phase_2_kernel if causal else bwd_phase_2_kernel_non_causal
    kernel[grid](q, k, v, dout, lse, delta, dk, dv_out, b, s, h, dq_dim, dv_dim, dq1, dq2, softmax_scale)
    return dk, dv_out


def _mla_bwd_dq(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dout: torch.Tensor,
                 lse: torch.Tensor, delta: torch.Tensor, softmax_scale: float, causal: bool = True):
    """Phase 3: compute dQ using external lse and delta. Returns dq."""
    allocator = lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(allocator)
    (b, s, h, dq_dim), dv_dim = q.shape, v.shape[-1]
    dq1, dq2 = 128, dq_dim - 128
    dq_out = torch.empty((b, s, h, dq_dim), dtype=torch.bfloat16, device=q.device)
    grid = lambda meta: (triton.cdiv(s, meta["blk_m"]), h, b)
    kernel = bwd_phase_3_kernel if causal else bwd_phase_3_kernel_non_causal
    kernel[grid](q, k, v, dout, lse, delta, dq_out, b, s, h, dq_dim, dv_dim, dq1, dq2, softmax_scale)
    return dq_out
