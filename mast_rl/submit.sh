#!/bin/bash
# Submit a Search-R1 RL training job to MAST.
#
# Adapted from fbsource//fbcode/pytorch/torchtitan/fb/mast_rl/submit.sh, trimmed
# to the MAST path. The conda env (built by build_conda.sh, with torchtitan
# installed from THIS OSS checkout) is packed and shipped to the MAST node;
# run.sh there starts the dense-retrieval server, then training.
#
# Usage:
#   conda activate rlmast
#   bash mast_rl/submit.sh                                   # 0.6B smoke, defaults
#   bash mast_rl/submit.sh --config rl_grpo_qwen3_1_7b_search_r1 \
#       --hf_assets_path /mnt/torchtrain_datasets/tree/shuhuay/qwen/Qwen3-1.7B
#   bash mast_rl/submit.sh --no-reinstall                    # skip the torchtitan reinstall
#
# --config, --hf_assets_path, and --oilfs accept both `--flag VALUE` and
# `--flag=VALUE`. Any other args are forwarded to launcher.py (e.g.
# --host-type, --hpc-identity, --rm-attribution, --hpc-oncall, --region).

cd "$(dirname "$0")/.." || exit 1  # repo root (the OSS torchtitan checkout)

if [[ "${CONDA_DEFAULT_ENV:-}" != "rlmast" ]]; then
    echo "ERROR: activate the 'rlmast' conda env first: conda activate rlmast" >&2
    exit 1
fi

# --- Defaults (override with --config / --hf_assets_path) ---
export JOB_NAME="${JOB_NAME:-torchtitan-rl-search-r1-qwen3-0.6b}"
MODULE="rl"
CONFIG="rl_grpo_qwen3_0_6b_search_r1"
# Qwen3 checkpoint on the mounted Manifold bucket (MAST mount point).
HF_ASSETS_PATH="/mnt/torchtrain_datasets/tree/shuhuay/qwen/Qwen3-0.6B"
# -----------------------------------------------------------

REINSTALL=1
OILFS_WORKSPACE=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-reinstall)
            REINSTALL=0
            shift
            ;;
        --config)
            [[ $# -lt 2 ]] && { echo "ERROR: --config requires a value" >&2; exit 1; }
            CONFIG="$2"; shift 2 ;;
        --config=*)
            CONFIG="${1#*=}"; shift ;;
        --hf_assets_path)
            [[ $# -lt 2 ]] && { echo "ERROR: --hf_assets_path requires a value" >&2; exit 1; }
            HF_ASSETS_PATH="$2"; shift 2 ;;
        --hf_assets_path=*)
            HF_ASSETS_PATH="${1#*=}"; shift ;;
        --oilfs)
            OILFS_WORKSPACE="$2"; shift 2 ;;
        --oilfs=*)
            OILFS_WORKSPACE="${1#*=}"; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Reinstall torchtitan from this checkout to pick up local changes.
if [[ "${REINSTALL}" -eq 1 ]]; then
    TORCHTITAN_ROOT="$(pwd)"
    echo "Reinstalling torchtitan from ${TORCHTITAN_ROOT} (the OSS checkout with Search-R1)."
    echo "(Skip with --no-reinstall when torchtitan/ is unchanged since the last reinstall.)"
    ( rm -rf build torchtitan.egg-info )
    if ! pip install --no-build-isolation --no-deps --force-reinstall "${TORCHTITAN_ROOT}"; then
        echo "ERROR: torchtitan reinstall failed; aborting submit" >&2
        exit 1
    fi
    echo ""
else
    echo "Skipping torchtitan reinstall (--no-reinstall) -- using already-installed torchtitan."
    echo ""
fi

# On an open-source conda dev server, importing monarch (for the submit) pulls in
# torch, whose bundled NVIDIA libs (libnccl, libcublas, ...) aren't on the system
# linker path. Prepend the env's nvidia libs (mirrors local.py / .venv_rl_env.sh)
# so `python mast_rl/launcher.py` can import monarch.tools. No-op if absent (e.g.
# prod servers where torch resolves libs from system paths).
NV_DIR="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia"
if [[ -d "${NV_DIR}" ]]; then
    NV_LIBS=""
    for d in "${NV_DIR}"/*/lib; do
        [[ -d "$d" ]] && NV_LIBS="${NV_LIBS}${NV_LIBS:+:}$d"
    done
    export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH:-}"
    if [[ -f "${NV_DIR}/cu13/lib/libcublas.so.13" ]]; then
        export LD_PRELOAD="${NV_DIR}/cu13/lib/libcublas.so.13:${NV_DIR}/cu13/lib/libcublasLt.so.13${LD_PRELOAD:+:}${LD_PRELOAD:-}"
    fi
fi

echo "Submitting MAST Search-R1 RL job:"
echo "  JOB_NAME: ${JOB_NAME}"
echo "  MODULE:   ${MODULE}"
echo "  CONFIG:   ${CONFIG}"
echo "  HF_ASSETS_PATH: ${HF_ASSETS_PATH}"
echo "  OILFS_WORKSPACE: ${OILFS_WORKSPACE:-<unset; will mount Manifold torchtrain_datasets>}"
echo "  EXTRA_ARGS: ${EXTRA_ARGS[*]}"
echo ""

exec python mast_rl/launcher.py \
    --job-name "${JOB_NAME}" \
    ${OILFS_WORKSPACE:+--oilfs "${OILFS_WORKSPACE}"} \
    "${EXTRA_ARGS[@]}" \
    -- mast_rl/main.py --module ${MODULE} --config ${CONFIG} \
    --hf_assets_path ${HF_ASSETS_PATH}
