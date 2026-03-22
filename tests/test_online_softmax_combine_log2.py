"""
Test the log2-scale online softmax combine used by ring MLA attention.

Verifies that splitting a non-causal MLA computation into two halves and
recombining via _online_softmax_combine_log2 produces the same result as
computing attention on the full KV sequence.

Run with:
    pytest tests/test_online_softmax_combine_log2.py -v
"""

import pytest
import torch

from pithtrain.operators.mla.triton import MLA as MLATriton
from pithtrain.operators.mla.triton import _mla_fwd
from pithtrain.operators.ring_attention.mla import _online_softmax_combine_log2
from tests.operators.utilities import assert_close


def _reference_attention(q, k, v, softmax_scale, causal=False):
    """Manual BSHD attention for reference."""
    q_t = q.float().permute(0, 2, 1, 3)
    k_t = k.float().permute(0, 2, 1, 3)
    v_t = v.float().permute(0, 2, 1, 3)
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale
    if causal:
        S = scores.shape[-1]
        mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_t)
    return out.permute(0, 2, 1, 3).to(q.dtype)


@pytest.fixture(autouse=True)
def autotune():
    B, S, H, DQ, DV = 1, 256, 16, 192, 128
    scale = DQ**-0.5
    MLATriton.autotune(B, S, H, DQ, DV, scale, include_non_causal=True)
    MLATriton.autotune(B, S // 2, H, DQ, DV, scale, include_non_causal=True)


def test_combine_two_halves():
    """Split KV in half, compute partial attentions, combine, compare to full."""
    B, S, H, DQ, DV = 1, 256, 16, 192, 128
    scale = DQ**-0.5
    S_half = S // 2

    torch.manual_seed(123)
    q = torch.randn(B, S_half, H, DQ, dtype=torch.bfloat16, device="cuda")
    k1 = torch.randn(B, S_half, H, DQ, dtype=torch.bfloat16, device="cuda")
    v1 = torch.randn(B, S_half, H, DV, dtype=torch.bfloat16, device="cuda")
    k2 = torch.randn(B, S_half, H, DQ, dtype=torch.bfloat16, device="cuda")
    v2 = torch.randn(B, S_half, H, DV, dtype=torch.bfloat16, device="cuda")

    out1, lse1 = _mla_fwd(q, k1, v1, scale, causal=False)
    out2, lse2 = _mla_fwd(q, k2, v2, scale, causal=False)
    combined_out, combined_lse = _online_softmax_combine_log2(out1, lse1, out2, lse2)

    k_full = torch.cat([k1, k2], dim=1)
    v_full = torch.cat([v1, v2], dim=1)
    ref_out = _reference_attention(q, k_full, v_full, scale, causal=False)

    assert_close(combined_out.to(torch.bfloat16), ref_out, rtol=1e-2, atol=2e-2)


def test_combine_identity():
    """Combining with a -inf LSE should be identity."""
    B, S, H, DQ, DV = 1, 128, 16, 192, 128
    scale = DQ**-0.5

    torch.manual_seed(7)
    q = torch.randn(B, S, H, DQ, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, S, H, DQ, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="cuda")

    out, lse = _mla_fwd(q, k, v, scale, causal=False)

    zeros = torch.zeros_like(out)
    neg_inf_lse = torch.full_like(lse, float("-inf"))

    combined_out, combined_lse = _online_softmax_combine_log2(out, lse, zeros, neg_inf_lse)
    assert_close(combined_out.to(torch.bfloat16), out, rtol=0, atol=1e-5)


def test_combine_associativity():
    """Combining A then B should equal combining B then A."""
    B, S, H, DQ, DV = 1, 128, 16, 192, 128
    scale = DQ**-0.5

    torch.manual_seed(99)
    q = torch.randn(B, S, H, DQ, dtype=torch.bfloat16, device="cuda")
    k1 = torch.randn(B, S, H, DQ, dtype=torch.bfloat16, device="cuda")
    v1 = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="cuda")
    k2 = torch.randn(B, S, H, DQ, dtype=torch.bfloat16, device="cuda")
    v2 = torch.randn(B, S, H, DV, dtype=torch.bfloat16, device="cuda")

    out1, lse1 = _mla_fwd(q, k1, v1, scale, causal=False)
    out2, lse2 = _mla_fwd(q, k2, v2, scale, causal=False)

    comb_ab, lse_ab = _online_softmax_combine_log2(out1, lse1, out2, lse2)
    comb_ba, lse_ba = _online_softmax_combine_log2(out2, lse2, out1, lse1)

    assert_close(comb_ab.to(torch.bfloat16), comb_ba.to(torch.bfloat16), rtol=0, atol=1e-5)
    assert_close(lse_ab, lse_ba, rtol=0, atol=1e-5)
