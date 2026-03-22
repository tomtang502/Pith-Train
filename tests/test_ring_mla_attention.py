"""
Correctness test for ring MLA attention (context parallelism with MLA).

Compares the output and gradients of ring_mla_attention_func (split across CP
ranks) against a single MLA call on the full un-split sequence.

Launch with:
    torchrun --nproc-per-node=2 tests/test_ring_mla_attention.py
"""

import math
import os
import sys

import torch
import torch.distributed as dist

from pithtrain.operators.mla.triton import MLA as MLATriton
from pithtrain.operators.mla.triton import (
    _mla_bwd_dk_dv,
    _mla_bwd_dq,
    _mla_bwd_preprocess,
    _mla_fwd,
)
from pithtrain.operators.ring_attention.mla import ring_mla_attention_func


class _RefMLAFunc(torch.autograd.Function):
    """Reference MLA using standalone fwd/bwd functions (avoids custom_op bug)."""

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale):
        o, lse = _mla_fwd(q, k, v, softmax_scale, causal=True)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.softmax_scale = softmax_scale
        return o

    @staticmethod
    def backward(ctx, do):
        do = do.contiguous()
        q, k, v, o, lse = ctx.saved_tensors
        delta = _mla_bwd_preprocess(o, do)
        dk, dv = _mla_bwd_dk_dv(q, k, v, do, lse, delta, ctx.softmax_scale, causal=True)
        dq = _mla_bwd_dq(q, k, v, do, lse, delta, ctx.softmax_scale, causal=True)
        return dq, dk, dv, None


def setup():
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def _cosine_error(a, b):
    """1 - cosine similarity: 0 = identical, 2 = opposite."""
    a, b = a.double().flatten(), b.double().flatten()
    return 1.0 - torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)


def test_forward_backward(B, S, H, DQ, DV, cp_group, dtype=torch.bfloat16):
    """Compare ring MLA on split sequence vs MLA on full sequence."""
    rank = cp_group.rank()
    cp_size = cp_group.size()
    device = torch.cuda.current_device()
    scale = DQ**-0.5
    S_local = S // cp_size

    MLATriton.autotune(B, S_local, H, DQ, DV, scale, include_non_causal=True)
    # Also autotune for the full-sequence reference (only rank 0 runs it, but
    # all ranks call autotune so the kernel cache is populated everywhere).
    MLATriton.autotune(B, S, H, DQ, DV, scale)

    torch.manual_seed(42)
    q_full = torch.randn(B, S, H, DQ, device=device, dtype=dtype)
    k_full = torch.randn(B, S, H, DQ, device=device, dtype=dtype)
    v_full = torch.randn(B, S, H, DV, device=device, dtype=dtype)

    # --- Reference: causal MLA on full sequence ---
    q_ref = q_full.clone().requires_grad_(True)
    k_ref = k_full.clone().requires_grad_(True)
    v_ref = v_full.clone().requires_grad_(True)
    out_ref = _RefMLAFunc.apply(q_ref, k_ref, v_ref, scale)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # --- Ring MLA: each rank gets its chunk ---
    q_chunk = q_full[:, rank * S_local : (rank + 1) * S_local].clone().requires_grad_(True)
    k_chunk = k_full[:, rank * S_local : (rank + 1) * S_local].clone().requires_grad_(True)
    v_chunk = v_full[:, rank * S_local : (rank + 1) * S_local].clone().requires_grad_(True)
    out_ring = ring_mla_attention_func(
        q_chunk, k_chunk, v_chunk, softmax_scale=scale, cp_group=cp_group
    )
    loss_ring = out_ring.sum()
    loss_ring.backward()

    # --- Compare forward output ---
    out_ref_chunk = out_ref[:, rank * S_local : (rank + 1) * S_local]
    fwd_cos = _cosine_error(out_ring, out_ref_chunk).item()

    # --- Compare gradients ---
    dq_ref_chunk = q_ref.grad[:, rank * S_local : (rank + 1) * S_local]
    dk_ref_chunk = k_ref.grad[:, rank * S_local : (rank + 1) * S_local]
    dv_ref_chunk = v_ref.grad[:, rank * S_local : (rank + 1) * S_local]

    dq_cos = _cosine_error(q_chunk.grad, dq_ref_chunk).item()
    dk_cos = _cosine_error(k_chunk.grad, dk_ref_chunk).item()
    dv_cos = _cosine_error(v_chunk.grad, dv_ref_chunk).item()

    return fwd_cos, dq_cos, dk_cos, dv_cos


def main():
    rank, world_size = setup()
    assert world_size >= 2, "Need at least 2 GPUs"

    cp_group = dist.new_group(list(range(world_size)))
    atol = 1e-5

    configs = [
        {"B": 1, "S": 256, "H": 16, "DQ": 192, "DV": 128, "label": "MLA S=256"},
        {"B": 2, "S": 512, "H": 16, "DQ": 192, "DV": 128, "label": "MLA S=512 B=2"},
        {"B": 1, "S": 1024, "H": 16, "DQ": 192, "DV": 128, "label": "MLA S=1024"},
    ]

    all_passed = True
    for cfg in configs:
        label = cfg.pop("label")
        fwd, dq, dk, dv = test_forward_backward(**cfg, cp_group=cp_group)
        errors = [fwd, dq, dk, dv]
        has_nan = any(math.isnan(e) for e in errors)
        worst = max(fwd, dq, dk, dv)
        passed = not has_nan and worst < atol
        all_passed &= passed
        if rank == 0:
            status = "PASS" if passed else "FAIL"
            print(
                f"[{status}] {label}: fwd_cos={fwd:.2e}  dQ_cos={dq:.2e}  dK_cos={dk:.2e}  dV_cos={dv:.2e}"
            )

    dist.destroy_process_group()
    if rank == 0:
        print("\nAll tests passed." if all_passed else "\nSome tests FAILED.")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
