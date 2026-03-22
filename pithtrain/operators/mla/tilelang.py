"""
TileLang implementation of the MLA operator.

Multi-head attention with BSHD layout and causal masking.

 b : batch size
 s : sequence length
 h : number of heads
dq : query/key head dimension
dv : value head dimension
"""

import itertools
import math
from typing import Tuple

import tilelang
import tilelang.language as T
import torch

from pithtrain.operators.mla.pytorch import MLA as BaseMLA

# fmt: off
# mypy: ignore-errors

options = dict(blk_m=[32, 64, 128], blk_n=[32, 64, 128], num_stages=[1, 2], threads=[128, 256])
configs = [dict(zip(options, values)) for values in itertools.product(*options.values())]

@tilelang.autotune(configs=configs)
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def fwd(b: int, s: int, h: int, dq: int, dv: int, softmax_scale: float, blk_m: int, blk_n: int, num_stages: int, threads: int):
    """
    Attention forward with causal masking.
    """
    softmax_scale_log2 = softmax_scale * math.log2(math.e)

    @T.prim_func
    def main(
        q_tensor: T.Tensor((b, s, h, dq), T.bfloat16),
        k_tensor: T.Tensor((b, s, h, dq), T.bfloat16),
        v_tensor: T.Tensor((b, s, h, dv), T.bfloat16),
        o_tensor: T.Tensor((b, s, h, dv), T.bfloat16),
        lse_tensor: T.Tensor((b, h, s), T.float32),
    ):
        with T.Kernel(T.ceildiv(s, blk_m), h, b, threads=threads) as (idx_m, idx_h, idx_b):
            # Shared memory for Q, K, V tiles.
            q_smem = T.alloc_shared((blk_m, dq), T.bfloat16)
            k_smem = T.alloc_shared((blk_n, dq), T.bfloat16)
            v_smem = T.alloc_shared((blk_n, dv), T.bfloat16)
            # Online softmax state.
            rowmax = T.alloc_fragment((blk_m,), T.float32)
            rowmax_prev = T.alloc_fragment((blk_m,), T.float32)
            rescale_factor = T.alloc_fragment((blk_m,), T.float32)
            rowsum = T.alloc_fragment((blk_m,), T.float32)
            normalizer = T.alloc_fragment((blk_m,), T.float32)
            T.fill(rowmax, -T.infinity(T.float32))
            T.fill(normalizer, 0)
            # Attention scores and output accumulator.
            scores = T.alloc_fragment((blk_m, blk_n), T.float32)
            scores_cast = T.alloc_fragment((blk_m, blk_n), T.bfloat16)
            output = T.alloc_fragment((blk_m, dv), T.float32)
            T.fill(output, 0)
            # Main loop.
            T.copy(q_tensor[idx_b, idx_m * blk_m : (idx_m + 1) * blk_m, idx_h, :], q_smem)
            for idx_n in T.Pipelined(T.ceildiv((idx_m + 1) * blk_m, blk_n), num_stages=num_stages):
                T.copy(k_tensor[idx_b, idx_n * blk_n : (idx_n + 1) * blk_n, idx_h, :], k_smem)
                T.copy(v_tensor[idx_b, idx_n * blk_n : (idx_n + 1) * blk_n, idx_h, :], v_smem)
                # Compute pre-softmax logits with causal masking.
                for i, j in T.Parallel(blk_m, blk_n):
                    scores[i, j] = T.if_then_else(idx_n * blk_n + j > idx_m * blk_m + i, -T.infinity(scores.dtype), 0)
                T.gemm(q_smem, k_smem, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                # Compute attention scores with online softmax.
                T.copy(rowmax, rowmax_prev)
                T.reduce_max(scores, rowmax, dim=1, clear=False)
                for i in T.Parallel(blk_m):
                    rowmax[i] = T.max(rowmax[i], rowmax_prev[i])
                for i in T.Parallel(blk_m):
                    rescale_factor[i] = T.exp2(rowmax_prev[i] * softmax_scale_log2 - rowmax[i] * softmax_scale_log2)
                for i, j in T.Parallel(blk_m, blk_n):
                    scores[i, j] = T.exp2(scores[i, j] * softmax_scale_log2 - rowmax[i] * softmax_scale_log2)
                T.reduce_sum(scores, rowsum, dim=1)
                for i in T.Parallel(blk_m):
                    normalizer[i] = normalizer[i] * rescale_factor[i] + rowsum[i]
                T.copy(scores, scores_cast)
                # Compute attention output with rescale factor applied.
                for i, j in T.Parallel(blk_m, dv):
                    output[i, j] *= rescale_factor[i]
                T.gemm(scores_cast, v_smem, output, policy=T.GemmWarpPolicy.FullRow)
            # Apply the softmax denominator and compute log-sum-exp for backward pass.
            for i, j in T.Parallel(blk_m, dv):
                output[i, j] /= normalizer[i]
            T.copy(output, o_tensor[idx_b, idx_m * blk_m : (idx_m + 1) * blk_m, idx_h, :])
            for i in T.Parallel(blk_m):
                normalizer[i] = T.log2(normalizer[i]) + rowmax[i] * softmax_scale_log2
            T.copy(normalizer, lse_tensor[idx_b, idx_h, idx_m * blk_m : (idx_m + 1) * blk_m])
    return main

options = dict(blk=[32, 64, 128], num_stages=[1, 2], threads=[128, 256])
configs = [dict(zip(options, values)) for values in itertools.product(*options.values())]

@tilelang.autotune(configs=configs)
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def bwd_phase_1(b: int, s: int, h: int, dv: int, blk: int, num_stages: int, threads: int):
    """
    Phase 1 of attention backward: compute delta = rowsum(o * do).
    """
    @T.prim_func
    def main(
        o_tensor: T.Tensor((b, s, h, dv), T.bfloat16),
        do_tensor: T.Tensor((b, s, h, dv), T.bfloat16),
        delta_tensor: T.Tensor((b, h, s), T.float32),
    ):
        with T.Kernel(T.ceildiv(s, blk), h, b, threads=threads) as (idx_s, idx_h, idx_b):
            o = T.alloc_fragment((blk, blk), T.bfloat16)
            do = T.alloc_fragment((blk, blk), T.bfloat16)
            delta = T.alloc_fragment((blk, blk), T.float32)
            delta_reduced = T.alloc_fragment((blk,), T.float32)
            T.fill(delta, 0)
            # Main loop.
            for k in T.Pipelined(T.ceildiv(dv, blk), num_stages=num_stages):
                T.copy(o_tensor[idx_b, idx_s * blk : (idx_s + 1) * blk, idx_h, k * blk : (k + 1) * blk], o)
                T.copy(do_tensor[idx_b, idx_s * blk : (idx_s + 1) * blk, idx_h, k * blk : (k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    delta[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(delta, delta_reduced, dim=1)
            T.copy(delta_reduced, delta_tensor[idx_b, idx_h, idx_s * blk : (idx_s + 1) * blk])
    return main

options = dict(blk_m=[32, 64, 128], blk_n=[32, 64, 128], num_stages=[1, 2], threads=[128, 256])
configs = [dict(zip(options, values)) for values in itertools.product(*options.values())]

@tilelang.autotune(configs=configs)
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def bwd_phase_2(b: int, s: int, h: int, dq: int, dv: int, softmax_scale: float, blk_m: int, blk_n: int, num_stages: int, threads: int):
    """
    Attention backward with causal masking.
    """
    softmax_scale_log2 = softmax_scale * math.log2(math.e)

    @T.prim_func
    def main(
        q_tensor: T.Tensor((b, s, h, dq), T.bfloat16),
        k_tensor: T.Tensor((b, s, h, dq), T.bfloat16),
        v_tensor: T.Tensor((b, s, h, dv), T.bfloat16),
        do_tensor: T.Tensor((b, s, h, dv), T.bfloat16),
        lse_tensor: T.Tensor((b, h, s), T.float32),
        delta_tensor: T.Tensor((b, h, s), T.float32),
        dq_tensor: T.Tensor((b, s, h, dq), T.float32),
        dk_tensor: T.Tensor((b, s, h, dq), T.bfloat16),
        dv_tensor: T.Tensor((b, s, h, dv), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(s, blk_n), h, b, threads=threads) as (idx_n, idx_h, idx_b):
            # Shared memory for Q, K, V, dO, LSE, delta tiles.
            q_smem = T.alloc_shared((blk_m, dq), T.bfloat16)
            k_smem = T.alloc_shared((blk_n, dq), T.bfloat16)
            v_smem = T.alloc_shared((blk_n, dv), T.bfloat16)
            do_smem = T.alloc_shared((blk_m, dv), T.bfloat16)
            lse_smem = T.alloc_shared((blk_m,), T.float32)
            delta_smem = T.alloc_shared((blk_m,), T.float32)
            # Recomputed attention scores and dS intermediates.
            scores = T.alloc_fragment((blk_n, blk_m), T.float32)
            scores_cast = T.alloc_fragment((blk_n, blk_m), T.bfloat16)
            dsT = T.alloc_fragment((blk_n, blk_m), T.float32)
            dsT_cast = T.alloc_fragment((blk_n, blk_m), T.bfloat16)
            dsT_smem = T.alloc_shared((blk_n, blk_m), T.bfloat16)
            # Gradient accumulators for dK, dV, dQ.
            dk_frag = T.alloc_fragment((blk_n, dq), T.float32)
            dv_frag = T.alloc_fragment((blk_n, dv), T.float32)
            dq_frag = T.alloc_fragment((blk_m, dq), T.float32)
            T.fill(dk_frag, 0)
            T.fill(dv_frag, 0)
            dk_smem = T.alloc_shared((blk_n, dq), T.bfloat16)
            dv_smem = T.alloc_shared((blk_n, dv), T.bfloat16)
            layout = T.Layout((b, s, h, dq), lambda b, s, h, d: [b, s // 8, h, d // 8, (d % 2), 4 * (s % 8) + (d % 8) // 2])
            T.annotate_layout({dq_tensor: layout})
            # Main loop.
            T.copy(k_tensor[idx_b, idx_n * blk_n : (idx_n + 1) * blk_n, idx_h, :], k_smem)
            T.copy(v_tensor[idx_b, idx_n * blk_n : (idx_n + 1) * blk_n, idx_h, :], v_smem)
            for idx_m in T.Pipelined(T.floordiv(idx_n * blk_n, blk_m), T.ceildiv(s, blk_m), num_stages=num_stages):
                T.copy(q_tensor[idx_b, idx_m * blk_m : (idx_m + 1) * blk_m, idx_h, :], q_smem)
                T.copy(lse_tensor[idx_b, idx_h, idx_m * blk_m : (idx_m + 1) * blk_m], lse_smem)
                T.copy(do_tensor[idx_b, idx_m * blk_m : (idx_m + 1) * blk_m, idx_h, :], do_smem)
                T.copy(delta_tensor[idx_b, idx_h, idx_m * blk_m : (idx_m + 1) * blk_m], delta_smem)
                # Recompute the attention scores.
                T.gemm(k_smem, q_smem, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                for i, j in T.Parallel(blk_n, blk_m):
                    scores[i, j] = T.exp2(scores[i, j] * softmax_scale_log2 - lse_smem[j])
                for i, j in T.Parallel(blk_n, blk_m):
                    scores[i, j] = T.if_then_else(idx_n * blk_n + i <= idx_m * blk_m + j, scores[i, j], 0)
                # Compute dV += P @ dO.
                T.copy(scores, scores_cast)
                T.gemm(scores_cast, do_smem, dv_frag, policy=T.GemmWarpPolicy.FullRow)
                # Compute dS = P * (V @ dO^T - delta) / sqrt(dq).
                T.gemm(v_smem, do_smem, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                for i, j in T.Parallel(blk_n, blk_m):
                    dsT_cast[i, j] = scores[i, j] * (dsT[i, j] - delta_smem[j]) * softmax_scale
                # Compute dK += dS @ Q.
                T.gemm(dsT_cast, q_smem, dk_frag, policy=T.GemmWarpPolicy.FullRow)
                # Compute dQ += dS^T @ K.
                T.copy(dsT_cast, dsT_smem)
                T.gemm(dsT_smem, k_smem, dq_frag, transpose_A=True, clear_accum=True)
                for i, j in T.Parallel(blk_m, dq):
                    T.atomic_add(dq_tensor[idx_b, idx_m * blk_m + i, idx_h, j], dq_frag[i, j])
            # Write accumulated dK and dV to global memory.
            T.copy(dk_frag, dk_smem)
            T.copy(dv_frag, dv_smem)
            T.copy(dk_smem, dk_tensor[idx_b, idx_n * blk_n : (idx_n + 1) * blk_n, idx_h, :])
            T.copy(dv_smem, dv_tensor[idx_b, idx_n * blk_n : (idx_n + 1) * blk_n, idx_h, :])
    return main

options = dict(blk=[32, 64, 128], threads=[128, 256])
configs = [dict(zip(options, values)) for values in itertools.product(*options.values())]

@tilelang.autotune(configs=configs)
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def bwd_phase_3(b: int, s: int, h: int, dq: int, blk: int, threads: int):
    """
    Phase 3 of attention backward: cast dq from float32 to bfloat16.
    """
    @T.prim_func
    def main(
        dq_tensor: T.Tensor((b, s, h, dq), T.float32),
        dq_cast_tensor: T.Tensor((b, s, h, dq), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(s, blk), h, b, threads=threads) as (idx_s, idx_h, idx_b):
            layout = T.Layout((b, s, h, dq), lambda b, s, h, d: [b, s // 8, h, d // 8, (d % 2), 4 * (s % 8) + (d % 8) // 2])
            T.annotate_layout({dq_tensor: layout})
            T.copy(dq_tensor[idx_b, idx_s * blk : (idx_s + 1) * blk, idx_h, :], dq_cast_tensor[idx_b, idx_s * blk : (idx_s + 1) * blk, idx_h, :])
    return main

_tilelang_delta = None
_tilelang_dq_acc = None

@torch.library.custom_op("pithtrain::mla_tilelang_fwd", mutates_args=())
def _mla_tilelang_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    (b, s, h, d_q), d_v = q.shape, v.shape[-1]
    o = torch.empty((b, s, h, d_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    fwd_kernel = fwd(b, s, h, d_q, d_v, softmax_scale)
    fwd_kernel(q, k, v, o, lse)
    return o, lse

@_mla_tilelang_fwd.register_fake
def _(q, k, v, softmax_scale):
    (b, s, h, _), d_v = q.shape, v.shape[-1]
    o = torch.empty((b, s, h, d_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
    return o, lse

@torch.library.custom_op("pithtrain::mla_tilelang_bwd", mutates_args=())
def _mla_tilelang_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor, softmax_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _tilelang_delta, _tilelang_dq_acc
    (b, s, h, d_q), d_v = q.shape, v.shape[-1]
    if _tilelang_delta is None or _tilelang_delta.shape != (b, h, s):
        _tilelang_delta = torch.empty((b, h, s), dtype=torch.float32, device=q.device)
        _tilelang_dq_acc = torch.empty((b, s, h, d_q), dtype=torch.float32, device=q.device)
    delta = _tilelang_delta
    dq_acc = _tilelang_dq_acc
    dq_acc.zero_()
    dk = torch.empty((b, s, h, d_q), dtype=torch.bfloat16, device=q.device)
    dv_out = torch.empty((b, s, h, d_v), dtype=torch.bfloat16, device=q.device)
    dq_cast = torch.empty((b, s, h, d_q), dtype=torch.bfloat16, device=q.device)
    phase_1_kernel = bwd_phase_1(b, s, h, d_v)
    phase_2_kernel = bwd_phase_2(b, s, h, d_q, d_v, softmax_scale)
    phase_3_kernel = bwd_phase_3(b, s, h, d_q)
    phase_1_kernel(o, do, delta)
    phase_2_kernel(q, k, v, do, lse, delta, dq_acc, dk, dv_out)
    phase_3_kernel(dq_acc, dq_cast)
    return dq_cast, dk, dv_out

@_mla_tilelang_bwd.register_fake
def _(q, k, v, o, lse, do, softmax_scale):
    (b, s, h, d_q), d_v = q.shape, v.shape[-1]
    dq_out = torch.empty((b, s, h, d_q), dtype=torch.bfloat16, device=q.device)
    dk = torch.empty((b, s, h, d_q), dtype=torch.bfloat16, device=q.device)
    dv_out = torch.empty((b, s, h, d_v), dtype=torch.bfloat16, device=q.device)
    return dq_out, dk, dv_out

def _mla_tilelang_setup_context(ctx, inputs, output):
    q, k, v, softmax_scale = inputs
    o, lse = output
    ctx.save_for_backward(q, k, v, o, lse)
    ctx.softmax_scale = softmax_scale

def _mla_tilelang_backward(ctx, grad_o, grad_lse):
    q, k, v, o, lse = ctx.saved_tensors
    dq, dk, dv = _mla_tilelang_bwd(q, k, v, o, lse, grad_o, ctx.softmax_scale)
    return dq, dk, dv, None

_mla_tilelang_fwd.register_autograd(_mla_tilelang_backward, setup_context=_mla_tilelang_setup_context)

class MLA(BaseMLA):
    """
    TileLang implementation of the MLA operator.
    """
    @staticmethod
    def autotune(b: int, s: int, h: int, dq: int, dv: int, softmax_scale: float, include_non_causal: bool = False) -> None:
        _ = fwd(b, s, h, dq, dv, softmax_scale)
        _ = bwd_phase_1(b, s, h, dv)
        _ = bwd_phase_2(b, s, h, dq, dv, softmax_scale)
        _ = bwd_phase_3(b, s, h, dq)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        o, _ = _mla_tilelang_fwd(q, k, v, self.softmax_scale)
        return o
