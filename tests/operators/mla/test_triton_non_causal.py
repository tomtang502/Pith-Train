"""
Test non-causal MLA kernels and helper functions.

Compares:
  1. _mla_fwd(causal=False) against a manual PyTorch non-causal attention reference.
  2. The decomposed backward helpers (_mla_bwd_preprocess, _mla_bwd_dk_dv, _mla_bwd_dq)
     against autograd on the same reference.
  3. Causal helpers against the existing MLATriton forward/backward for consistency.

Run all tests:
    pytest tests/operators/mla/test_triton_non_causal.py -v
"""

import itertools
from typing import Tuple

import pytest
import torch

from pithtrain.operators.mla.triton import (
    MLA as MLATriton,
)
from pithtrain.operators.mla.triton import (
    _mla_bwd_dk_dv,
    _mla_bwd_dq,
    _mla_bwd_preprocess,
    _mla_fwd,
)
from tests.operators.utilities import assert_close

SHAPES = [(16, 192, 128, 192**-0.5)]
INPUTS = [(1, 512), (2, 1024)]

Workload = Tuple[int, int, int, int, int, float]
PARAMS = [
    pytest.param((b, s, h, dq, dv, scale), id=f"b{b}-s{s}-h{h}")
    for (h, dq, dv, scale), (b, s) in itertools.product(SHAPES, INPUTS)
]


@pytest.fixture(params=PARAMS)
def workload(request: pytest.FixtureRequest) -> Workload:
    b, s, h, dq, dv, softmax_scale = request.param
    MLATriton.autotune(b, s, h, dq, dv, softmax_scale, include_non_causal=True)
    return b, s, h, dq, dv, softmax_scale


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


# ---------------------------------------------------------------------------
# Non-causal forward
# ---------------------------------------------------------------------------


def test_fwd_non_causal(workload: Workload) -> None:
    b, s, h, dq, dv, softmax_scale = workload
    torch.manual_seed(42)
    q = torch.randn(b, s, h, dq, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, s, h, dq, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(b, s, h, dv, dtype=torch.bfloat16, device="cuda")

    ref_o = _reference_attention(q, k, v, softmax_scale, causal=False)
    our_o, _ = _mla_fwd(q, k, v, softmax_scale, causal=False)
    assert_close(our_o, ref_o, rtol=1e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# Non-causal backward
# ---------------------------------------------------------------------------


def test_bwd_non_causal(workload: Workload) -> None:
    b, s, h, dq, dv, softmax_scale = workload
    torch.manual_seed(42)

    ref_q = torch.randn(b, s, h, dq, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    ref_k = torch.randn(b, s, h, dq, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    ref_v = torch.randn(b, s, h, dv, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    dout = torch.randn(b, s, h, dv, dtype=torch.bfloat16, device="cuda")

    ref_o = _reference_attention(ref_q, ref_k, ref_v, softmax_scale, causal=False)
    ref_o.backward(dout)
    ref_dq, ref_dk, ref_dv = ref_q.grad, ref_k.grad, ref_v.grad

    q = ref_q.data.clone()
    k = ref_k.data.clone()
    v = ref_v.data.clone()
    out, lse = _mla_fwd(q, k, v, softmax_scale, causal=False)
    delta = _mla_bwd_preprocess(out, dout)
    our_dk, our_dv = _mla_bwd_dk_dv(q, k, v, dout, lse, delta, softmax_scale, causal=False)
    our_dq = _mla_bwd_dq(q, k, v, dout, lse, delta, softmax_scale, causal=False)

    assert_close(our_dq, ref_dq, rtol=1e-2, atol=2e-2, otol=1e-3)
    assert_close(our_dk, ref_dk, rtol=1e-2, atol=2e-2, otol=1e-3)
    assert_close(our_dv, ref_dv, rtol=1e-2, atol=2e-2, otol=1e-3)


# ---------------------------------------------------------------------------
# Causal helpers match existing MLA operator
# ---------------------------------------------------------------------------


def test_fwd_helpers_causal(workload: Workload) -> None:
    """_mla_fwd(causal=True) should match MLATriton.forward."""
    b, s, h, dq, dv, softmax_scale = workload
    torch.manual_seed(42)
    q = torch.randn(b, s, h, dq, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, s, h, dq, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(b, s, h, dv, dtype=torch.bfloat16, device="cuda")

    mla = MLATriton(h, dq, dv, softmax_scale).cuda()
    ref_o = mla.forward(q, k, v)
    our_o, _ = _mla_fwd(q, k, v, softmax_scale, causal=True)
    assert_close(our_o, ref_o, rtol=0, atol=0)


def test_bwd_helpers_causal(workload: Workload) -> None:
    """Decomposed causal backward should match MLATriton autograd backward."""
    b, s, h, dq, dv, softmax_scale = workload
    torch.manual_seed(42)

    ref_q = torch.randn(b, s, h, dq, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    ref_k = torch.randn(b, s, h, dq, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    ref_v = torch.randn(b, s, h, dv, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    dout = torch.randn(b, s, h, dv, dtype=torch.bfloat16, device="cuda")

    mla = MLATriton(h, dq, dv, softmax_scale).cuda()
    mla.forward(ref_q, ref_k, ref_v).backward(dout)

    q = ref_q.data.clone()
    k = ref_k.data.clone()
    v = ref_v.data.clone()
    out, lse = _mla_fwd(q, k, v, softmax_scale, causal=True)
    delta = _mla_bwd_preprocess(out, dout)
    our_dk, our_dv = _mla_bwd_dk_dv(q, k, v, dout, lse, delta, softmax_scale, causal=True)
    our_dq = _mla_bwd_dq(q, k, v, dout, lse, delta, softmax_scale, causal=True)

    assert_close(our_dq, ref_q.grad, rtol=0, atol=0)
    assert_close(our_dk, ref_k.grad, rtol=0, atol=0)
    assert_close(our_dv, ref_v.grad, rtol=0, atol=0)
