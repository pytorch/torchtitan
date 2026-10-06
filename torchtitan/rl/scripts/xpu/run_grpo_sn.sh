#!/bin/bash
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -N grpo_full_1n
#PBS -l filesystems=flare:home
#PBS -A Intel-Aurora
#
# Single-node full-parameter GRPO on Intel XPU (4 tiles: 2 trainer + 2 generator).
#
# Interactive:
#   qsub -I -l select=1 -l walltime=01:00:00 -A <account> -q <queue>
#   bash torchtitan/rl/scripts/xpu/run_grpo_sn.sh
# Batch:
#   qsub torchtitan/rl/scripts/xpu/run_grpo_sn.sh
#
# Overridable: CONFIG NUM_STEPS HF_ASSETS_PATH ZE_AFFINITY_MASK
# DUMP_FOLDER. Extra args are appended to the training command verbatim, e.g.
#   NUM_STEPS=200 bash run_grpo_sn.sh --override my_pkg.mod.fn

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

# No RMSNorm interposer: on torch 2.14.0+xpu the stock XPU kernel passes the
# 65536-row weight-grad backward that needed it on 2.12 (intel/torch-xpu-ops#4790).

HF_ASSETS_PATH=${HF_ASSETS_PATH:-/flare/Aurora_deployment/intel/models/Qwen3-0.6B}
CONFIG=${CONFIG:-rl_grpo_full_qwen3_0_6b_flex}
NUM_STEPS=${NUM_STEPS:-10}
DUMP_FOLDER=${DUMP_FOLDER:-outputs/rl_full_1n}
# qsub takes no trailing script arguments, so "$@" is always empty under PBS.
# EXTRA_ARGS is the only way to reach the config CLI from a qsub -v. The config
# already sets load_only, so a save is not the usual reason to need it.
EXTRA_ARGS=${EXTRA_ARGS:-}

# The config CLI takes no scalar field flags (--override selects a Configurable
# implementation, it does not set values), so these reach the recipe through the
# environment. See alphabet_sort_xpu.py.
export TITAN_XPU_HF_ASSETS_PATH="$HF_ASSETS_PATH"
export TITAN_XPU_NUM_STEPS="$NUM_STEPS"

# A stale checkpoint from a different mesh shape fails to load, so start clean.
rm -rf "$TORCHTITAN_DIR/$DUMP_FOLDER"/checkpoint/ 2>/dev/null
rm -rf "$TORCHINDUCTOR_CACHE_DIR"/triton 2>/dev/null
cd "$TORCHTITAN_DIR" || { echo "ERROR: TORCHTITAN_DIR=$TORCHTITAN_DIR not found"; exit 1; }
mkdir -p "$DUMP_FOLDER"

if [ ! -f "$HF_ASSETS_PATH/config.json" ]; then
    echo "ERROR: no model at $HF_ASSETS_PATH (config.json missing)."
    exit 1
fi
if ! python3 -c "import torch; assert torch.xpu.is_available() and torch.xpu.device_count() > 0" 2>/dev/null; then
    echo "ERROR: no XPU devices visible. Run on a compute node with oneAPI sourced."
    exit 1
fi

# Confirm the 65536-row RMSNorm weight-grad backward (the shape full GRPO hits)
# works on this torch before spending the allocation.
python3 -c "
import torch
x = torch.randn(2, 2048, 16, 128, device='xpu', dtype=torch.bfloat16, requires_grad=True)
torch.nn.RMSNorm(128, eps=1e-6, dtype=torch.bfloat16).to('xpu')(x).sum().backward()
torch.xpu.synchronize()
print('stock torch; 65536-row RMSNorm weight-grad backward: OK')
"
if [ $? -ne 0 ]; then
    echo "FATAL: RMSNorm weight-grad precheck failed, not starting the pipeline."
    exit 1
fi

python3 -m torchtitan.rl.train \
    --module torchtitan_recipes.rl.alphabet_sort_xpu --config "$CONFIG" \
    --output-dir="$DUMP_FOLDER" \
    $EXTRA_ARGS \
    "$@" \
    2>&1 | tee "$DUMP_FOLDER"/train_full_1n.log
exit "${PIPESTATUS[0]}"
