#!/bin/bash
#SBATCH --job-name=pr18_flex_bench
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/dgxc/runs/pr18_flex_bench-%j.out
set -uo pipefail
source /mnt/dgxc/venvs/dsv4n/bin/activate
WORKTREE=/mnt/dgxc/worktrees/pr18-flash
export PYTHONPATH="$WORKTREE" CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR="/tmp/triton-cache-$SLURM_JOB_ID" TORCHINDUCTOR_CACHE_DIR="/tmp/inductor-cache-$SLURM_JOB_ID"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
cd "$WORKTREE"
echo "###### pr18_flex_bench commit=$(git rev-parse --short HEAD) job=$SLURM_JOB_ID node=$(hostname) ######"
python gb300/bench_flex_sink_rescale.py 2>&1 | grep -vE "DEBUG|Warning|_pytree"
echo "###### BENCH exit=${PIPESTATUS[0]} ######"
rm -rf "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
