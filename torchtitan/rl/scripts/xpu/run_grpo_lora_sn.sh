#!/bin/bash
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -N grpo_lora_1n
#PBS -l filesystems=flare:home
#PBS -A Intel-Aurora
#PBS -j oe
#
# Single-node GRPO+LoRA on Intel XPU (4 tiles: 2 trainer + 2 generator).
#
# Interactive:
#   qsub -I -l select=1 -l walltime=01:00:00 -A <account> -q <queue>
#   bash torchtitan/rl/scripts/xpu/run_grpo_lora_sn.sh
# Batch:
#   qsub torchtitan/rl/scripts/xpu/run_grpo_lora_sn.sh
#
# Overridable: CONFIG NUM_STEPS HF_ASSETS_PATH ZE_AFFINITY_MASK DUMP_FOLDER.
# Extra args are appended to the training command verbatim, e.g.
#   NUM_STEPS=200 bash run_grpo_lora_sn.sh --override my_pkg.mod.fn
#
# The former run_grpo_lora_xpu_1node_dp8.sh diagnostic (dp_shard=8 on one node,
# 12 tiles) is reachable from here -- that question is settled, but to re-run it:
#   ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7,8,9,10,11 CONFIG=rl_grpo_lora_qwen3_0_6b_dp8 \
#       DUMP_FOLDER=outputs/rl_lora_1n_dp8 bash run_grpo_lora_sn.sh

# No `set -e`: an accidental `source` of this file should not kill the shell.

# torch 2.14.0+xpu stack. TORCHTITAN_DIR, ONEAPI_ENV_SCRIPT, CONDA_PREFIX_BASE and
# CONDA_ENV are overridable; see torchtitan/rl/docs/xpu_torch214.md.
# Absolute, not relative to this file: PBS copies the submitted script into a
# spool dir, so BASH_SOURCE does not resolve back into the repo.
TORCHTITAN_DIR=${TORCHTITAN_DIR:-$HOME/git/torchtitan}
source ${ONEAPI_ENV_SCRIPT:-$TORCHTITAN_DIR/torchtitan/rl/scripts/xpu/env_torch214.sh}
eval "$(${CONDA_PREFIX_BASE:-$HOME/miniforge3}/bin/conda shell.bash hook)"
conda activate ${CONDA_ENV:-monarch214}
# Every RL teardown aborts and dumps ~10 GB of core per process; keep them off.
ulimit -c 0
# The v1 Level Zero UR adapter intermittently aborts with "ur_die: urEventWait must
# not be called for an internal event" (~1 per 100-150 steps); the v2 adapter ran
# 200/200 steps clean in both arms. Not root-caused.
export SYCL_UR_USE_LEVEL_ZERO_V2=1

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1,2,3}
# Cray libfabric supplies the CXI provider oneCCL needs. Aurora system
# refreshes rotate the version directory (1.22.0 disappeared in the 2026-09
# refresh), so resolve the newest one instead of pinning a version -- a stale
# path surfaces only as an opaque oneCCL "failed to initialize ATL" cascade.
LIBFABRIC_ROOT=$(ls -d /opt/cray/libfabric/*/ 2>/dev/null | sort -V | tail -1)
LIBFABRIC_ROOT=${LIBFABRIC_ROOT%/}
if [ ! -e "${LIBFABRIC_ROOT}/lib64/libfabric.so.1" ]; then
    echo "ERROR: no Cray libfabric found under /opt/cray/libfabric; FI_PROVIDER=cxi cannot work"
    exit 1
fi
export CCL_OFI_LIBRARY_PATH=${LIBFABRIC_ROOT}/lib64/libfabric.so.1
export FI_PROVIDER=cxi
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_OFI_PROVIDER=cxi
export LD_LIBRARY_PATH=${LIBFABRIC_ROOT}/lib64:${LD_LIBRARY_PATH}
export FI_PROVIDER_PATH=${LIBFABRIC_ROOT}/lib64/libfabric
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-~/.cache/torchinductor_xpu}
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# No LD_PRELOAD interposer here: the patched RMSNorm backward kernel is needed
# only by FULL-parameter GRPO, which trains the qk_norm weights. LoRA freezes
# them, so its backward never asks for that weight gradient. See run_grpo_sn.sh
# for the full-parameter single-node arm.

HF_ASSETS_PATH=${HF_ASSETS_PATH:-/flare/Aurora_deployment/intel/models/Qwen3-0.6B}
CONFIG=${CONFIG:-rl_grpo_lora_qwen3_0_6b}
NUM_STEPS=${NUM_STEPS:-10}
DUMP_FOLDER=${DUMP_FOLDER:-outputs/rl_lora_1n}
# qsub takes no trailing script arguments, so "$@" is always empty under PBS.
# EXTRA_ARGS is the only way to reach the config CLI from a qsub -v. Word-split
# on purpose, to pass several flags.
EXTRA_ARGS=${EXTRA_ARGS:-}

# The config CLI takes no scalar field flags (--override selects a Configurable
# implementation, it does not set values), so these reach the recipe through the
# environment. See alphabet_sort_xpu.py. The config already sets load_only:
# never turn the checkpointer off instead, because `enable` also gates the
# initial HF weight load, so disabling it silently trains from random init.
export TITAN_XPU_HF_ASSETS_PATH="$HF_ASSETS_PATH"
export TITAN_XPU_NUM_STEPS="$NUM_STEPS"
# A single-node LoRA trainer leaves vLLM more headroom than the 0.85 default.
export TITAN_XPU_GPU_MEMORY_LIMIT=0.90

# A stale checkpoint from a different mesh shape fails to load, so start clean.
rm -rf "$TORCHTITAN_DIR/$DUMP_FOLDER"/checkpoint/ 2>/dev/null
rm -rf "$TORCHINDUCTOR_CACHE_DIR"/triton 2>/dev/null
cd "$TORCHTITAN_DIR" || { echo "ERROR: TORCHTITAN_DIR=$TORCHTITAN_DIR not found"; exit 1; }
mkdir -p "$DUMP_FOLDER"

# Fail early with a clear message if the model or XPU devices aren't visible,
# rather than deep inside the training stack.
if [ ! -f "$HF_ASSETS_PATH/config.json" ]; then
    echo "ERROR: no model at $HF_ASSETS_PATH (config.json missing)."
    echo "       Set HF_ASSETS_PATH or download Qwen3-0.6B there first."
    exit 1
fi
if ! python3 -c "import torch; assert torch.xpu.is_available() and torch.xpu.device_count() > 0" 2>/dev/null; then
    echo "ERROR: no XPU devices visible. Run this on a compute node (not the UAN)"
    echo "       with oneAPI sourced and ZE_AFFINITY_MASK set."
    exit 1
fi

echo "=== Single-node GRPO+LoRA: ${CONFIG}, tiles ${ZE_AFFINITY_MASK} ==="

python3 -m torchtitan.rl.train \
    --module torchtitan_recipes.rl.alphabet_sort_xpu --config "$CONFIG" \
    --output-dir="$DUMP_FOLDER" \
    $EXTRA_ARGS \
    "$@" \
    2>&1 | tee "$DUMP_FOLDER"/train_lora_1n.log
exit "${PIPESTATUS[0]}"
