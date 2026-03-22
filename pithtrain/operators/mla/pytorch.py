import math

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


class MLA(nn.Module):
    """
    PyTorch implementation of the MLA operator (causal, BSHD layout).

    Provides a reference forward using flex_attention and NVTX profiling hooks.
    Kernel-level backends (e.g. TileLang, Triton) subclass this and override forward().
    """

    def __init__(self, h: int, dq: int, dv: int, softmax_scale: float = 0.0):
        """
        Parameters
        ----------
        h : int
            Number of attention heads.
        dq : int
            Query/Key head dimension.
        dv : int
            Value head dimension.
        softmax_scale : float, optional
            Softmax scale factor. Default is 1 / sqrt(dq).
        """
        super().__init__()
        self.h = h
        self.dq = dq
        self.dv = dv
        self.softmax_scale = softmax_scale if softmax_scale != 0.0 else math.sqrt(1.0 / dq)

    @staticmethod
    def causal(b: int, h: int, qidx: int, kidx: int) -> bool:
        return qidx >= kidx

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Forward pass with causal masking.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor of shape (B, S, H, DQ).
        k : torch.Tensor
            Key tensor of shape (B, S, H, DQ).
        v : torch.Tensor
            Value tensor of shape (B, S, H, DV).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, S, H, DV).
        """
        B, S, H = q.shape[0], q.shape[1], q.shape[2]
        q_t = q.permute(0, 2, 1, 3)
        k_t = k.permute(0, 2, 1, 3)
        v_t = v.permute(0, 2, 1, 3)
        block_mask = create_block_mask(MLA.causal, B=B, H=H, Q_LEN=S, KV_LEN=S, device=q.device)
        o = flex_attention(q_t, k_t, v_t, block_mask=block_mask, scale=self.softmax_scale)
        return o.permute(0, 2, 1, 3)
