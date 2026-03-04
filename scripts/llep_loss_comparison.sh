#!/bin/bash
# ==============================================================================
# LLEP vs Standard EP Loss Comparison (500 steps)
#
# Submits two SLURM jobs with identical config/data/seed, differing only in
# whether LLEP is enabled. After both complete, compare loss curves.
#
# Usage:
#   bash scripts/llep_loss_comparison.sh
#   bash scripts/llep_loss_comparison.sh --steps 200   # override steps
#   bash scripts/llep_loss_comparison.sh --dry-run      # create scripts only
#
# After completion:
#   grep "step:.*loss:" slurm_logs/llep_with_*_<JOB_ID>.out | head -20
#   grep "step:.*loss:" slurm_logs/llep_without_*_<JOB_ID>.out | head -20
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Defaults
STEPS="${STEPS:-500}"
PARTITION="${PARTITION:-batch}"
QOS="${QOS:-low}"
TIME="${TIME:-02:00:00}"
DRY_RUN=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps) STEPS="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

CONFIG_FILE="torchtitan/models/deepseek_v3/train_configs/debug_model_ep8_llep.toml"
SEED=42
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SLURM_LOGS_DIR="${REPO_DIR}/slurm_logs"
mkdir -p "$SLURM_LOGS_DIR"

echo "============================================="
echo "  LLEP vs Standard EP Loss Comparison"
echo "============================================="
echo "  Config:    $CONFIG_FILE"
echo "  Steps:     $STEPS"
echo "  Seed:      $SEED"
echo "  Partition:  $PARTITION"
echo "  Time:      $TIME"
echo "============================================="
echo ""

# ---------------------------------------------------------------------------
# Helper: create slurm script
# ---------------------------------------------------------------------------
create_slurm_script() {
    local JOB_NAME="$1"
    local LLEP_FLAG="$2"  # "--llep.enabled=True" or "--llep.enabled=False"
    local SCRIPT_PATH="${SLURM_LOGS_DIR}/${JOB_NAME}_${TIMESTAMP}.slurm"

    cat > "$SCRIPT_PATH" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --exclusive
#SBATCH --output=${SLURM_LOGS_DIR}/%x_%j.out
#SBATCH --error=${SLURM_LOGS_DIR}/%x_%j.err

echo "========================================="
echo "Job: ${JOB_NAME}"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "========================================="

# Environment
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_SOCKET_IFNAME="bond0"
export NCCL_BUFFSIZE=2097152
export TORCH_DIST_INIT_BARRIER=1

# Activate venv
cd ${REPO_DIR}
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Python: \$(which python)"
echo "torchrun: \$(which torchrun)"
echo ""

torchrun --nproc_per_node=8 --rdzv-endpoint=localhost:29500 \\
    -m torchtitan.train \\
    --job.config_file ${CONFIG_FILE} \\
    --training.steps ${STEPS} \\
    --debug.seed ${SEED} \\
    --compile.no-enable \\
    ${LLEP_FLAG} \\
    --metrics.log_freq 1

echo ""
echo "========================================="
echo "Job completed: \$(date)"
echo "========================================="
EOF

    chmod +x "$SCRIPT_PATH"
    echo "$SCRIPT_PATH"
}

# ---------------------------------------------------------------------------
# Create both scripts
# ---------------------------------------------------------------------------
SCRIPT_WITH=$(create_slurm_script "llep_with_${STEPS}steps" "--llep.enabled=True")
SCRIPT_WITHOUT=$(create_slurm_script "llep_without_${STEPS}steps" "--llep.enabled=False")

echo "Created scripts:"
echo "  WITH LLEP:    $SCRIPT_WITH"
echo "  WITHOUT LLEP: $SCRIPT_WITHOUT"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Scripts created but not submitted."
    echo ""
    echo "Submit manually:"
    echo "  sbatch $SCRIPT_WITH"
    echo "  sbatch $SCRIPT_WITHOUT"
    exit 0
fi

# ---------------------------------------------------------------------------
# Submit both
# ---------------------------------------------------------------------------
JOB_ID_WITH=$(sbatch --parsable "$SCRIPT_WITH" 2>&1)
echo "Submitted WITH LLEP:    job $JOB_ID_WITH"

JOB_ID_WITHOUT=$(sbatch --parsable "$SCRIPT_WITHOUT" 2>&1)
echo "Submitted WITHOUT LLEP: job $JOB_ID_WITHOUT"

echo ""
echo "============================================="
echo "  Both jobs submitted!"
echo "============================================="
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f ${SLURM_LOGS_DIR}/llep_with_${STEPS}steps_${JOB_ID_WITH}.out"
echo "  tail -f ${SLURM_LOGS_DIR}/llep_without_${STEPS}steps_${JOB_ID_WITHOUT}.out"
echo ""
echo "Compare loss after completion:"
echo "  grep 'step:.*loss:' ${SLURM_LOGS_DIR}/llep_with_${STEPS}steps_${JOB_ID_WITH}.out | sed 's/\\x1b\\[[0-9;]*m//g' | grep -oP 'step:.*loss: *[0-9.]+' | sort -u"
echo "  grep 'step:.*loss:' ${SLURM_LOGS_DIR}/llep_without_${STEPS}steps_${JOB_ID_WITHOUT}.out | sed 's/\\x1b\\[[0-9;]*m//g' | grep -oP 'step:.*loss: *[0-9.]+' | sort -u"
echo ""
