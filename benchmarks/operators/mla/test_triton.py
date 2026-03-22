"""
Benchmark the performance of the MLA operator (Triton).

Run all benchmarks:
    python3 -m benchmarks.operators.mla.test_triton

Forward FLOPs (causal attention, for input shape B, S, H, DQ, DV):
    QK^T matmul:       B*H*S*S*DQ     (causal halves: B*H*S*S*DQ/2 multiply-adds)
    softmax:           ~B*H*S*S/2     (exp, sum, div per element)
    PV matmul:         B*H*S*S*DV/2   (multiply-adds)
    Total:             B*H*S*S*(DQ+DV) (counting multiply-adds, causal halves)

Backward FLOPs:
    ~2.5x forward FLOPs

Forward I/O:
    Read Q, K:         B*S*H*DQ*2*2   (bfloat16)
    Read V:            B*S*H*DV*2     (bfloat16)
    Write O:           B*S*H*DV*2     (bfloat16)
    Write LSE:         B*H*S*4        (float32)
    Total:             B*S*H*(DQ*4+DV*4) + B*H*S*4

Backward I/O:
    Read Q, K, V, dO:  B*S*H*(DQ*4+DV*4)  (bfloat16)
    Read LSE, delta:   B*H*S*8            (float32)
    Write dQ, dK:      B*S*H*DQ*4         (bfloat16)
    Write dV:          B*S*H*DV*2         (bfloat16)
    Total:             B*S*H*(DQ*8+DV*6) + B*H*S*8
"""

import itertools
from typing import List, Tuple

import torch
import torch._functorch.config
from triton.testing import do_bench

from benchmarks.operators.utilities import Metrics
from pithtrain.operators.mla.pytorch import MLA as MLAPyTorch
from pithtrain.operators.mla.triton import MLA as MLATriton

SHAPES = [(16, 192, 128, 192**-0.5), (128, 192, 128, 192**-0.5)]
INPUTS = [(1, 4096), (1, 32768)]

Workload = Tuple[int, int, int, int, int, float]
WORKLOADS = [
    (b, s, h, dq, dv, softmax_scale)
    for (h, dq, dv, softmax_scale), (b, s) in itertools.product(SHAPES, INPUTS)
]


def bench_fwd(b: int, s: int, h: int, dq: int, dv: int, softmax_scale: float) -> Metrics:
    """
    Benchmark the forward pass.
    """
    torch.manual_seed(42)
    q = torch.randn((b, s, h, dq), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((b, s, h, dq), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((b, s, h, dv), dtype=torch.bfloat16, device="cuda")
    ref = MLAPyTorch(h, dq, dv, softmax_scale).cuda()
    our = MLATriton(h, dq, dv, softmax_scale).cuda()

    @torch.compile
    def ref_fwd():
        return ref.forward(q, k, v)

    def our_fwd():
        return our.forward(q, k, v)

    # Ensure the kernels are compiled.
    ref_fwd()
    our_fwd()

    # Benchmark the forward pass.
    ref_time = do_bench(ref_fwd)
    our_time = do_bench(our_fwd)
    flops = b * h * s * s * (dq + dv)
    io_bytes = b * s * h * (dq * 4 + dv * 4) + b * h * s * 4
    return Metrics(ref_time, our_time, flops, io_bytes)


def bench_bwd(b: int, s: int, h: int, dq: int, dv: int, softmax_scale: float) -> Metrics:
    """
    Benchmark the backward pass.
    """
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
    torch._functorch.config.donated_buffer = False

    @torch.compile
    def ref_bwd():
        ref_q.grad, ref_k.grad, ref_v.grad = None, None, None
        ref_o.backward(do, retain_graph=True)

    def our_bwd():
        our_q.grad, our_k.grad, our_v.grad = None, None, None
        our_o.backward(do, retain_graph=True)

    # Ensure the kernels are compiled.
    ref_o = torch.compile(ref.forward)(ref_q, ref_k, ref_v)
    our_o = our.forward(our_q, our_k, our_v)
    ref_bwd()
    our_bwd()

    # Benchmark the backward pass.
    ref_time = do_bench(ref_bwd)
    our_time = do_bench(our_bwd)
    fwd_flops = b * h * s * s * (dq + dv)
    flops = int(2.5 * fwd_flops)
    io_bytes = b * s * h * (dq * 8 + dv * 6) + b * h * s * 8
    return Metrics(ref_time, our_time, flops, io_bytes)


def main() -> None:
    """
    Benchmark the MLA operator.
    """
    for b, s, h, dq, dv, softmax_scale in WORKLOADS:
        MLATriton.autotune(b, s, h, dq, dv, softmax_scale)

    rows: List[Tuple[int, int, int, int, int, Metrics, Metrics]] = []
    for b, s, h, dq, dv, softmax_scale in WORKLOADS:
        fwd = bench_fwd(b, s, h, dq, dv, softmax_scale)
        bwd = bench_bwd(b, s, h, dq, dv, softmax_scale)
        rows.append((b, s, h, dq, dv, fwd, bwd))

    metrics = f"{'ms':>7} {'TB/s':>6} {'TF/s':>6} {'speedup':>7}"
    dims = f"{'b':>3} {'s':>6} {'h':>3} {'dq':>4} {'dv':>4}"
    hdr1 = f"{dims}  {' fwd ':-^{len(metrics)}}  {' bwd ':-^{len(metrics)}}"
    hdr2 = f"{' ' * len(dims)}  {metrics}  {metrics}"
    sep = "=" * len(hdr2)
    lines = [sep, hdr1, hdr2, sep]

    for b, s, h, dq, dv, fwd, bwd in rows:
        lines.append(
            f"{b:>3} {s:>6} {h:>3} {dq:>4} {dv:>4}"
            f"  {fwd.our_ms:7.3f} {fwd.io_bytes / fwd.our_ms / 1e9:6.2f} {fwd.flops / fwd.our_ms / 1e9:6.2f} {fwd.ref_ms / fwd.our_ms:6.2f}x"
            f"  {bwd.our_ms:7.3f} {bwd.io_bytes / bwd.our_ms / 1e9:6.2f} {bwd.flops / bwd.our_ms / 1e9:6.2f} {bwd.ref_ms / bwd.our_ms:6.2f}x"
        )
    lines.append(sep)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
