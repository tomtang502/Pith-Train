#!/bin/bash
# Pretrain a Mixture-of-Experts (MoE) language model.
#
# Usage:
#   bash examples/pretrain_language_model/launch.sh qwen3-30b-a3b
#   bash examples/pretrain_language_model/launch.sh deepseek-v2-lite
#
# For multi-node training with SLURM:
#   srun -W 0 examples/pretrain_language_model/launch.sh qwen3-30b-a3b

set -euo pipefail
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

if [ $# -ne 1 ]; then
    echo "Usage: launch.sh <model>" >&2
    exit 1
fi

# Setup distributed.
SLURM_NNODES=${SLURM_NNODES:-1}
SLURM_NODEID=${SLURM_NODEID:-0}
SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}}
SLURM_STEP_NODELIST=${SLURM_STEP_NODELIST:-$(hostname)}

LAUNCH_ARGS=()
LAUNCH_ARGS+=(--nnodes=$SLURM_NNODES --node-rank=$SLURM_NODEID)
LAUNCH_ARGS+=(--nproc-per-node=$(echo "$SLURM_STEP_GPUS" | tr ',' '\n' | wc -l))
LAUNCH_ARGS+=(--rdzv-backend=c10d)
LAUNCH_ARGS+=(--rdzv-endpoint=$(command -v scontrol &>/dev/null && scontrol show hostnames $SLURM_STEP_NODELIST | head -n 1 || echo localhost):15213)

# Launch the training.
SCRIPT=examples/pretrain_language_model/$1/script.py
OUTPUT=logging/pretrain_language_model/${1}_node${SLURM_NODEID:-0}.log

mkdir -p $(dirname $OUTPUT) && exec > >(tee $OUTPUT) 2>&1
torchrun ${LAUNCH_ARGS[@]} $SCRIPT
