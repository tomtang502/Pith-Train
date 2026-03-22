#!/bin/bash
set -e

PYTEST=".venv/bin/pytest"
TORCHRUN=".venv/bin/torchrun"

echo "=== Single-GPU tests (pytest) ==="

echo "--- Non-causal MLA kernels & helpers ---"
$PYTEST tests/operators/mla/test_triton_non_causal.py -v

echo "--- Online softmax combine (log2) ---"
$PYTEST tests/test_online_softmax_combine_log2.py -v

echo ""
echo "=== Multi-GPU test (torchrun, 2 GPUs) ==="

echo "--- Ring MLA attention ---"
$TORCHRUN --nproc-per-node=2 tests/test_ring_mla_attention.py
