"""
Ring MLA attention for context parallelism.

Splits the sequence across CP ranks. Each rank holds Q for its local chunk
and passes K/V around a ring so every rank computes full MLA attention.
Partial outputs are combined with online softmax (log-sum-exp rescaling).

This mirrors pithtrain/operators/ring_attention.py but uses the MLA Triton
kernels (asymmetric Q/K dim dq=192, V dim dv=128) instead of flash_attn.

MLA kernels store LSE in log2 scale, so the combine step uses exp2/log2
instead of exp/log.
"""

import torch
import torch.distributed as dist

from pithtrain.operators.mla.triton import (
    _mla_bwd_dk_dv,
    _mla_bwd_dq,
    _mla_bwd_preprocess,
    _mla_fwd,
)
from pithtrain.operators.ring_attention.standard import _ring_send_recv_kv


def _online_softmax_combine_log2(out1, lse1, out2, lse2):
    """Combine two partial attention outputs via online softmax rescaling.

    Operates in log2 scale to match MLA Triton kernels.
    out: [B, S, H, D],  lse: [B, H, S]  (float32, log2 scale).
    """
    max_lse = torch.maximum(lse1, lse2)
    exp1 = torch.exp2(lse1 - max_lse)
    exp2 = torch.exp2(lse2 - max_lse)
    e1 = exp1.transpose(1, 2).unsqueeze(-1)
    e2 = exp2.transpose(1, 2).unsqueeze(-1)
    new_out = ((e1 * out1 + e2 * out2) / (e1 + e2)).contiguous()
    new_lse = max_lse + torch.log2(exp1 + exp2)
    return new_out, new_lse


class RingMLAAttentionFunc(torch.autograd.Function):
    """Causal ring MLA attention with Triton MLA kernels."""

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, cp_rank, cp_size, global_ranks):
        B, S, H, DV = q.shape[0], q.shape[1], q.shape[2], v.shape[-1]
        combined_out = torch.zeros((B, S, H, DV), dtype=torch.bfloat16, device=q.device)
        combined_lse = torch.full(
            (B, H, S), torch.finfo(torch.float32).min, device=q.device, dtype=torch.float32
        )

        next_global = global_ranks[(cp_rank + 1) % cp_size]
        prev_global = global_ranks[(cp_rank - 1) % cp_size]
        assert k.is_contiguous() and v.is_contiguous(), (
            "ring MLA attention k or v is not contiguous"
        )
        cur_k, cur_v = k, v
        saved_k, saved_v, saved_ranks = [], [], []

        for step in range(cp_size):
            kv_rank = (cp_rank - step) % cp_size

            if kv_rank <= cp_rank:
                use_causal = kv_rank == cp_rank
                step_out, step_lse = _mla_fwd(q, cur_k, cur_v, softmax_scale, causal=use_causal)
                combined_out, combined_lse = _online_softmax_combine_log2(
                    combined_out, combined_lse, step_out, step_lse
                )
                saved_k.append(cur_k)
                saved_v.append(cur_v)
                saved_ranks.append(kv_rank)

            if step < cp_size - 1:
                cur_k, cur_v = _ring_send_recv_kv(cur_k, cur_v, next_global, prev_global)

        combined_out = combined_out.to(q.dtype)

        assert combined_lse.is_contiguous()
        ctx.save_for_backward(q, combined_out, combined_lse, *saved_k, *saved_v)
        ctx.softmax_scale = softmax_scale
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.global_ranks = global_ranks
        ctx.saved_ranks = saved_ranks
        ctx.n_saved = len(saved_ranks)
        return combined_out

    @staticmethod
    def backward(ctx, dout):
        dout = dout.contiguous()
        n = ctx.n_saved
        saved = ctx.saved_tensors
        q, combined_out, combined_lse = saved[0], saved[1], saved[2]
        all_k = list(saved[3 : 3 + n])
        all_v = list(saved[3 + n : 3 + 2 * n])
        cp_rank, cp_size, global_ranks = ctx.cp_rank, ctx.cp_size, ctx.global_ranks
        softmax_scale = ctx.softmax_scale

        delta = _mla_bwd_preprocess(combined_out, dout)

        dq = torch.zeros_like(q)
        local_dk = torch.zeros_like(all_k[0])
        local_dv = torch.zeros_like(all_v[0])
        remote_dk, remote_dv = {}, {}

        for i, kv_rank in enumerate(ctx.saved_ranks):
            use_causal = kv_rank == cp_rank
            dk_s, dv_s = _mla_bwd_dk_dv(
                q, all_k[i], all_v[i], dout, combined_lse, delta, softmax_scale, causal=use_causal
            )
            dq_s = _mla_bwd_dq(
                q, all_k[i], all_v[i], dout, combined_lse, delta, softmax_scale, causal=use_causal
            )
            dq += dq_s
            if kv_rank == cp_rank:
                local_dk += dk_s
                local_dv += dv_s
            else:
                remote_dk[kv_rank] = dk_s
                remote_dv[kv_rank] = dv_s

        for d in range(1, cp_size):
            target, source = cp_rank - d, cp_rank + d
            ops, recv_dk_buf, recv_dv_buf = [], None, None

            if target >= 0 and target in remote_dk:
                dst = global_ranks[target]
                assert remote_dk[target].is_contiguous() and remote_dv[target].is_contiguous(), (
                    "ring MLA attention remote_dk or remote_dv is not contiguous"
                )
                ops.append(dist.P2POp(dist.isend, remote_dk[target], dst))
                ops.append(dist.P2POp(dist.isend, remote_dv[target], dst))
            if source < cp_size:
                src = global_ranks[source]
                recv_dk_buf = torch.empty_like(local_dk)
                recv_dv_buf = torch.empty_like(local_dv)
                ops.append(dist.P2POp(dist.irecv, recv_dk_buf, src))
                ops.append(dist.P2POp(dist.irecv, recv_dv_buf, src))

            if ops:
                for req in dist.batch_isend_irecv(ops):
                    req.wait()
            if recv_dk_buf is not None:
                local_dk += recv_dk_buf
                local_dv += recv_dv_buf

        return dq, local_dk, local_dv, None, None, None, None


def ring_mla_attention_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    cp_group: dist.ProcessGroup,
):
    """Causal ring MLA attention across context-parallel ranks."""
    cp_rank = cp_group.rank()
    cp_size = cp_group.size()
    global_ranks = [dist.distributed_c10d.get_global_rank(cp_group, r) for r in range(cp_size)]
    return RingMLAAttentionFunc.apply(q, k, v, softmax_scale, cp_rank, cp_size, global_ranks)
