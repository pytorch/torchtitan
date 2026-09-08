#!/bin/bash
# DeepSeek V4 on 64x GB300 (16 nodes x 4 GPUs), yqb01-qa01.
#
# Keeps upstream's launch path verbatim -- `srun torchrun ... -m torchtitan.train
# --module deepseek_v4 --config <name>` against a stock config from
# torchtitan/models/deepseek_v4/config_registry.py. The only additions over
# upstream's multinode_trainer.slurm are the NCCL interface names this cluster
# needs (no NVSwitch, 4 IB HCAs per node) and a worktree/PYTHONPATH guard.
#
# Usage:
#   CONFIG=deepseek_v4_flash sbatch gb300/dsv4_64xgb300.slurm
#
#SBATCH --job-name=dsv4_64
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --partition=all
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/dgxc/runs/dsv4-%j.out
set -uo pipefail

WORKTREE=${WORKTREE:-/mnt/dgxc/worktrees/dsv4-pro}
VENV=${VENV:-/mnt/dgxc/venvs/dsv4}
CONFIG=${CONFIG:?set CONFIG=<name from deepseek_v4/config_registry.py>}
STEPS=${STEPS:-10}
TAG=${TAG:-$CONFIG}

cd "$WORKTREE"
source "$VENV/bin/activate"
export PYTHONPATH="$WORKTREE"

BRANCH=$(git -C "$WORKTREE" rev-parse --abbrev-ref HEAD)
COMMIT=$(git -C "$WORKTREE" rev-parse --short HEAD)
echo "###### $TAG  branch=$BRANCH  commit=$COMMIT  steps=$STEPS  job=$SLURM_JOB_ID  $(date -u +%H:%M:%S) ######"

# The venv installs torchtitan editable; fail loudly rather than silently
# measuring a different checkout.
python - "$WORKTREE" <<'PY' || exit 3
import sys, torchtitan
want, got = sys.argv[1], torchtitan.__file__
print(f"###### torchtitan resolved to {got}", flush=True)
if not got.startswith(want.rstrip("/") + "/"):
    print(f"###### FATAL: expected import from {want}", flush=True)
    sys.exit(1)
PY

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_ip=$(getent ahostsv4 "${nodes[0]}" | awk '{print $1; exit}')
export NCCL_IB_HCA=ibP16p3s0,ibP18p3s0,ibP2p3s0,ibp3s0
export NCCL_SOCKET_IFNAME=enP22s22f0np0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONFAULTHANDLER=1

LOG="/mnt/dgxc/runs/dsv4_${TAG}.log"
srun --kill-on-bad-exit=1 torchrun \
  --nnodes 16 --nproc_per_node 4 --rdzv_id "$SLURM_JOB_ID" \
  --rdzv_backend c10d --rdzv_endpoint "$head_ip:29500" \
  --role rank --tee 3 --local-ranks-filter 0 \
  -m torchtitan.train --module deepseek_v4 --config "$CONFIG" \
    --training.steps "$STEPS" "$@" \
  > "$LOG" 2>&1
rc=$?
echo "###### END $TAG exit=$rc $(date -u +%H:%M:%S) ######"
sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -oE "step: +[0-9]+ .*mfu: [0-9.]+%" | tail -3
[ $rc -ne 0 ] && sed 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "AssertionError|RuntimeError|OutOfMemory|torch\..*Error" | tail -3 | cut -c1-200
exit $rc
