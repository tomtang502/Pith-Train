"""
Test the correctness of the MLA operator (Triton).

Run all tests:
    pytest tests/operators/mla/test_triton.py -v
Run only backward tests:
    pytest tests/operators/mla/test_triton.py -v -k "bwd"
"""

import itertools
from typing import Tuple

import pytest
import torch

from pithtrain.operators.mla.pytorch import MLA as MLAPyTorch
from pithtrain.operators.mla.triton import MLA as MLATriton
from tests.operators.utilities import assert_close

SHAPES = [(16, 192, 128, 192**-0.5), (128, 192, 128, 192**-0.5)]
INPUTS = [(1, 4096), (1, 32768)]

Workload = Tuple[int, int, int, int, int, float]
PARAMS = [
    pytest.param((b, s, h, dq, dv, softmax_scale), id=f"b{b}-s{s}-h{h}-dq{dq}-dv{dv}")
    for (h, dq, dv, softmax_scale), (b, s) in itertools.product(SHAPES, INPUTS)
]


@pytest.fixture(params=PARAMS)
def workload(request: pytest.FixtureRequest) -> Workload:
    """
    Yield a workload for the MLA operator.
    """
    b, s, h, dq, dv, softmax_scale = request.param
    MLATriton.autotune(b, s, h, dq, dv, softmax_scale)
    return b, s, h, dq, dv, softmax_scale


def test_fwd(workload: Workload) -> None:
    """
    Test the forward pass of the MLA operator.
    """
    b, s, h, dq, dv, softmax_scale = workload
    torch.manual_seed(42)
    q = torch.randn((b, s, h, dq), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((b, s, h, dq), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((b, s, h, dv), dtype=torch.bfloat16, device="cuda")
    ref = MLAPyTorch(h, dq, dv, softmax_scale).cuda()
    our = MLATriton(h, dq, dv, softmax_scale).cuda()

    @torch.compile
    def ref_fwd():
        return ref.forward(q, k, v)

    ref_o = ref_fwd()
    our_o = our.forward(q, k, v)
    assert_close(our_o, ref_o, rtol=1e-2, atol=2e-2)


def test_bwd(workload: Workload) -> None:
    """
    Test the backward pass of the MLA operator.
    """
    b, s, h, dq, dv, softmax_scale = workload
    torch.manual_seed(42)
    ref_q = torch.randn((b, s, h, dq), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    ref_k = torch.randn((b, s, h, dq), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    ref_v = torch.randn((b, s, h, dv), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    our_q = ref_q.clone().detach().requires_grad_(True)
    our_k = ref_k.clone().detach().requires_grad_(True)
    our_v = ref_v.clone().detach().requires_grad_(True)
    do = torch.randn((b, s, h, dv), dtype=torch.bfloat16, device="cuda")
    ref = MLAPyTorch(h, dq, dv, softmax_scale).cuda()
    our = MLATriton(h, dq, dv, softmax_scale).cuda()

    @torch.compile
    def ref_fwd():
        return ref.forward(ref_q, ref_k, ref_v)

    ref_fwd().backward(do)
    our.forward(our_q, our_k, our_v).backward(do)
    assert_close(our_q.grad, ref_q.grad, rtol=1e-2, atol=2e-2, otol=1e-3)
    assert_close(our_k.grad, ref_k.grad, rtol=1e-2, atol=2e-2, otol=1e-3)
    assert_close(our_v.grad, ref_v.grad, rtol=1e-2, atol=2e-2, otol=1e-3)
