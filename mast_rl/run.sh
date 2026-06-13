#!/usr/bin/bash
# Per-host entrypoint on the MAST node for Search-R1 RL.
#
# Adapted from fbsource//fbcode/pytorch/torchtitan/fb/mast_rl/run.sh. On top of
# the base env setup (platform010 LD_PRELOAD, libcuda symlinks, NCCL socket,
# Manifold/OilFS mount, conda activate) this script:
#   1. points HuggingFace + the Search-R1 parquet at the mounted bucket (offline),
#   2. starts the dense-retrieval server in the background on a reserved GPU,
#   3. waits for http://127.0.0.1:8000 to be healthy,
#   4. runs the training command ("$@"), and
#   5. tears the retriever down on exit.
set -eExu -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

# ---------------------------------------------------------------------------
# Base MAST env setup (verbatim from fb/mast_rl/run.sh)
# ---------------------------------------------------------------------------
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    PLATFORM="platform010-aarch64"
else
    PLATFORM="platform010"
fi
PLATFORM_LIB="/usr/local/fbcode/${PLATFORM}/lib"

export PATH="$CONDA_DIR/bin:$PATH"
export CONDA_PREFIX="$CONDA_DIR"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# MAST nodes may lack IB drivers, force Socket transport
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
# Find a working gcc for torch.compile/Triton
GCC_DIR=$(mktemp -d)
if [ -x "${CONDA_DIR}/bin/x86_64-conda-linux-gnu-gcc" ]; then
    ln -sf "${CONDA_DIR}/bin/x86_64-conda-linux-gnu-gcc" "$GCC_DIR/gcc"
elif [ -x "${CONDA_DIR}/bin/gcc" ]; then
    ln -sf "${CONDA_DIR}/bin/gcc" "$GCC_DIR/gcc"
elif [ -x "/usr/bin/gcc" ]; then
    ln -sf "/usr/bin/gcc" "$GCC_DIR/gcc"
fi
export CC="${GCC_DIR}/gcc"
export PATH="${GCC_DIR}:${PATH}"

if [[ -n "${TRITON_LIBCUDA_PATH:-}" ]]; then
    echo "Detected production-ready mast env"
else
    echo "Open-source conda env - will set up env variables for mast"
    NEED_TO_SETUP_ENV_FOR_MAST=1
fi

if [[ -n "${NEED_TO_SETUP_ENV_FOR_MAST:-}" ]]; then
    if command -v patchelf &> /dev/null; then
        patchelf --set-interpreter ${PLATFORM_LIB}/ld.so $CONDA_DIR/bin/python
    fi
fi

if [[ -n "${NEED_TO_SETUP_ENV_FOR_MAST:-}" ]]; then
    export TRITON_LIBCUDA_PATH=${PLATFORM_LIB}/libcuda.so
    CUBLAS_LT=$(find ${CONDA_PREFIX}/lib -name "libcublasLt.so*" -not -type d 2>/dev/null | head -1)
    if [[ -n "${CUBLAS_LT:-}" ]]; then
        export LD_PRELOAD=${CUBLAS_LT}:${PLATFORM_LIB}/libcuda.so:${PLATFORM_LIB}/libnvidia-ml.so
    else
        export LD_PRELOAD=${PLATFORM_LIB}/libcuda.so:${PLATFORM_LIB}/libnvidia-ml.so
    fi
    export LIBRARY_PATH=${PLATFORM_LIB}/
else
    export LD_PRELOAD=${PLATFORM_LIB}/libcuda.so:${PLATFORM_LIB}/libnvidia-ml.so
fi

LIBCUDA_REAL=$(readlink -f ${PLATFORM_LIB}/libcuda.so 2>/dev/null || ls ${PLATFORM_LIB}/libcuda.so.* 2>/dev/null | head -1)
ln -sf "$LIBCUDA_REAL" "${CONDA_PREFIX}/lib/libcuda.so"
ln -sf "$LIBCUDA_REAL" "${CONDA_PREFIX}/lib/libcuda.so.1"
ln -sf ${PLATFORM_LIB}/libnvidia-ml.so "${CONDA_PREFIX}/lib/libnvidia-ml.so"
ln -sf ${PLATFORM_LIB}/libnvidia-ml.so "${CONDA_PREFIX}/lib/libnvidia-ml.so.1"
# faiss-gpu (cu12) JIT-compiles PTX for sm_90 at search time and needs the driver's
# PTX JIT compiler lib; without it: "CUDA error 221 PTX JIT compiler library not
# found". Symlink it from the platform010 driver so faiss GPU search works.
PTXJIT_REAL=$(ls ${PLATFORM_LIB}/libnvidia-ptxjitcompiler.so* 2>/dev/null | head -1)
if [[ -n "${PTXJIT_REAL:-}" ]]; then
    ln -sf "$PTXJIT_REAL" "${CONDA_PREFIX}/lib/libnvidia-ptxjitcompiler.so"
    ln -sf "$PTXJIT_REAL" "${CONDA_PREFIX}/lib/libnvidia-ptxjitcompiler.so.1"
fi

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=${CONDA_PREFIX}
ln -sfn ${CONDA_PREFIX}/lib ${CONDA_PREFIX}/lib64
rm -rf /root/.cache/flashinfer

# ---------------------------------------------------------------------------
# Mount the Manifold bucket / OilFS workspace (verbatim)
# ---------------------------------------------------------------------------
if [[ -n "${OILFS_WORKSPACE:-}" ]]; then
    MOUNT_URI="ws://${OILFS_WORKSPACE}"
    MOUNT_POINT="/mnt/$(basename "${OILFS_WORKSPACE}")"
    OILFS_PROFILE_FLAG=""
elif [[ -n "${MANIFUSE_BUCKET:-}" ]]; then
    MOUNT_URI="manifold://${MANIFUSE_BUCKET}"
    MOUNT_POINT="/mnt/${MANIFUSE_BUCKET}"
    OILFS_PROFILE_FLAG="--profile=manifold"
fi

if [[ -n "${MOUNT_URI:-}" ]]; then
    echo "Mounting ${MOUNT_URI} at ${MOUNT_POINT}"
    mkdir -p "$MOUNT_POINT"
    if command -v /packages/oil.oilfs/oilfs-wrapper &> /dev/null; then
        env -i PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin" \
            HOME="$HOME" \
            /packages/oil.oilfs/oilfs-wrapper \
            ${OILFS_PROFILE_FLAG} \
            "${MOUNT_URI}" \
            "$MOUNT_POINT"
        ls -d "$MOUNT_POINT" >/dev/null && echo "Mount verified" || true
    else
        echo "WARNING: oilfs-wrapper not found, skipping mount"
    fi
    echo "${MOUNT_URI} available at ${MOUNT_POINT}"
fi

# ---------------------------------------------------------------------------
# Search-R1: data + HuggingFace cache on the mounted bucket (offline)
# ---------------------------------------------------------------------------
# Where the Search-R1 assets are staged on the bucket (override via env).
STAGE_ROOT="${SEARCH_R1_STAGE_ROOT:-${MOUNT_POINT}/tree/yichuan}"

# Parquet data (read by the rollouter through the SEARCH_R1_*_PARQUET env vars).
export SEARCH_R1_TRAIN_PARQUET="${SEARCH_R1_TRAIN_PARQUET:-${STAGE_ROOT}/Search-R1/data/nq_hotpotqa_train/train.parquet}"
export SEARCH_R1_TEST_PARQUET="${SEARCH_R1_TEST_PARQUET:-${STAGE_ROOT}/Search-R1/data/nq_hotpotqa_train/test.parquet}"

# HuggingFace cache for the e5 retriever model (and tokenizers). MAST is
# offline, so resolve everything from the staged hub cache.
export HF_HOME="${HF_HOME:-${STAGE_ROOT}/hf}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# vLLM's flashinfer sampler JIT-compiles a CUDA kernel at runtime, which FAILS on
# the cu13 conda toolkit (version-mismatched nvcc/headers -> link error code 127).
# Disable it and use native torch sampling (attention uses FLEX_ATTENTION, not
# flashinfer, so only sampling is affected). Mirrors .venv_rl_env.sh.
export VLLM_USE_FLASHINFER_SAMPLER=0
# HF `datasets` converts the 14GB corpus jsonl to an Arrow cache. Keep that cache
# on node-local disk, NOT under HF_HOME (which is on the manifoldfs mount, where
# datasets' free-space check fails with "Not enough disk space").
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets}"
mkdir -p "${HF_DATASETS_CACHE}" 2>/dev/null || true

# Retriever index / corpus on the bucket.
RETRIEVER_INDEX="${RETRIEVER_INDEX:-${STAGE_ROOT}/search-r1-index/e5_Flat.index}"
RETRIEVER_CORPUS="${RETRIEVER_CORPUS:-${STAGE_ROOT}/search-r1-index/wiki_dump.jsonl}"

# Persist outputs (incl. the OFFLINE wandb run) to the bucket so they survive the
# ephemeral MAST node and can be `wandb sync`'d from the devvm afterward. MAST has
# no internet egress, so wandb runs offline (WANDB_MODE=offline, set in mast.py);
# the offline run lands at ${DUMP_DIR}/tb/<timestamp>/wandb/offline-run-*.
DUMP_DIR="${SEARCH_R1_DUMP_DIR:-${STAGE_ROOT}/mast_runs/${WANDB_RUN_NAME:-run}}"
export WANDB_DIR="${DUMP_DIR}"
mkdir -p "${DUMP_DIR}" 2>/dev/null || true
echo "Outputs (and offline wandb) -> ${DUMP_DIR}"

# ---------------------------------------------------------------------------
# Start the dense-retrieval server on a reserved GPU
# ---------------------------------------------------------------------------
# Training (Monarch's PerHostProvisioner) allocates GPUs from index 0 upward, so
# reserve the LAST GPU for the retriever's e5 encoder. 0.6B uses GPUs 0-3, 1.7B
# uses 0-5; the last GPU (7 on an 8-GPU host) stays free.
# Count GPUs robustly: nvidia-smi may not be on PATH on the MAST node, and under
# `set -e -o pipefail` a failed `nvidia-smi` (exit 127) in a command substitution
# would kill the whole script. Guard with `command -v` and never let it fail.
NUM_GPUS=8  # grandteton whole-host default
if command -v nvidia-smi >/dev/null 2>&1; then
    n=$(nvidia-smi -L 2>/dev/null | wc -l) || n=0
    [ "${n:-0}" -gt 0 ] && NUM_GPUS="$n"
fi
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
RETRIEVER_TIMEOUT="${RETRIEVER_TIMEOUT:-3600}"  # corpus load + index->GPU transfer
# faiss mode: default GPU search (RETRIEVER_FAISS_GPU=1). The pip faiss-gpu-cu12
# in the training env has NO sm_90 kernels (-> CUDA 209), so for GPU we ship a
# SEPARATE conda-forge faiss-gpu env (retr-gpu, sm_90) packed to the bucket and
# run the retriever from it. The 64.5GB Flat index is fp16-sharded across the LAST
# 2 GPUs (peak ~96GB fp32->fp16 doesn't fit one 80GB card -> shard 2). Set
# RETRIEVER_FAISS_GPU=0 to use CPU faiss (mmap, ~6s/query) from the training env.
FAISS_GPU=${RETRIEVER_FAISS_GPU:-1}
FAISS_GPU_FLAG=""
# The retriever runs from the SAME training env (rlmast) -- it now contains a
# from-source faiss 1.9.0 built with sm_90 (CMAKE_CUDA_ARCHITECTURES=90, cu13),
# so GPU search works on H100 in-env. No separate conda env needed (the earlier
# conda-forge retr-gpu env segfaulted at python startup on the MAST node -- ABI
# mismatch with platform010; building faiss into rlmast avoids that entirely).
RETR_PY="python"
RETR_ENVWRAP=()
if [[ "${FAISS_GPU}" == "1" ]]; then
    FAISS_GPU_FLAG="--faiss_gpu"
    # fp16-shard the 64.5GB Flat index across the LAST 2 GPUs (~16GB each; one
    # 80GB card OOMs at the fp32->fp16 peak). Training uses GPUs 0..N-3.
    RETRIEVER_GPU="${RETRIEVER_GPU:-$((NUM_GPUS - 2)),$((NUM_GPUS - 1))}"
else
    RETRIEVER_GPU="${RETRIEVER_GPU:-$((NUM_GPUS - 1))}"
fi

# --- One-shot import probe: pinpoints which C-extension segfaults on the node.
# Each import on its own line (unbuffered) so the LAST printed line before a hard
# crash names the culprit. `|| true` so a crash here never kills run.sh.
PROBE_LOG="/tmp/retriever_probe.log"
echo "=== retriever import probe (GPU ${RETRIEVER_GPU}, py=${RETR_PY}) -> ${PROBE_LOG} ==="
CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU}" "${RETR_ENVWRAP[@]}" "${RETR_PY}" -u -c "
import sys; print('python', sys.version.split()[0], flush=True)
import numpy; print('numpy', numpy.__version__, flush=True)
import datasets; print('datasets', datasets.__version__, flush=True)
import faiss; print('faiss', faiss.__version__, 'num_gpus', faiss.get_num_gpus(), flush=True)
import torch; print('torch', torch.__version__, flush=True)
print('cuda.is_available', torch.cuda.is_available(), flush=True)
x = torch.zeros(1).cuda(); print('cuda tensor ok', x.device, flush=True)
import transformers; print('transformers', transformers.__version__, flush=True)
import fastapi, uvicorn, pydantic; print('web stack ok', flush=True)
print('ALL IMPORTS OK', flush=True)
" > "${PROBE_LOG}" 2>&1 || echo "probe process exited non-zero ($?) -- see below"
echo "--- probe log ---"; cat "${PROBE_LOG}" 2>/dev/null || echo "(no probe log)"
echo "--- end probe ---"

echo "Starting retrieval server on GPU ${RETRIEVER_GPU} (faiss_gpu=${FAISS_GPU})"
echo "  index : ${RETRIEVER_INDEX}"
echo "  corpus: ${RETRIEVER_CORPUS}"
# Log to the bucket (persistent past the ephemeral node) and unbuffered (-u) so
# a crash during import still flushes its traceback for post-mortem.
RETRIEVER_LOG="/tmp/retrieval_server.log"
CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU}" "${RETR_ENVWRAP[@]}" \
    "${RETR_PY}" -u "${SCRIPT_DIR}/retrieval_server.py" \
    --index_path "${RETRIEVER_INDEX}" \
    --corpus_path "${RETRIEVER_CORPUS}" \
    --topk "${RETRIEVER_TOPK}" \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    ${FAISS_GPU_FLAG} \
    > "${RETRIEVER_LOG}" 2>&1 &
RETRIEVER_PID=$!

# Kill the retriever when the training command exits (or on signal).
cleanup() {
    echo "Stopping retrieval server (pid ${RETRIEVER_PID})"
    kill "${RETRIEVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Health-check: POST a tiny query until the server returns 200 (index loaded).
echo "Waiting up to ${RETRIEVER_TIMEOUT}s for the retriever to come up..."
DEADLINE=$(( $(date +%s) + RETRIEVER_TIMEOUT ))
until python - <<'PY'
import json, sys, urllib.request
req = urllib.request.Request(
    "http://127.0.0.1:8000/retrieve",
    data=json.dumps({"queries": ["healthcheck"], "topk": 1, "return_scores": False}).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    # A real query takes ~6s (CPU brute-force Flat search), so allow generous time.
    with urllib.request.urlopen(req, timeout=60) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
do
    if ! kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
        echo "ERROR: retrieval server died during startup. Last log lines:"
        tail -n 80 "${RETRIEVER_LOG}" || true
        exit 1
    fi
    # Live visibility into where startup is spending time (index mmap / corpus load).
    echo "--- retriever log tail ($(date +%H:%M:%S)) ---"
    tail -n 3 "${RETRIEVER_LOG}" 2>/dev/null || true
    if [ "$(date +%s)" -ge "${DEADLINE}" ]; then
        echo "ERROR: retrieval server not healthy after ${RETRIEVER_TIMEOUT}s. Last log lines:"
        tail -n 80 "${RETRIEVER_LOG}" || true
        exit 1
    fi
    sleep 10
done
echo "Retrieval server is healthy at http://127.0.0.1:8000/retrieve"

# ---------------------------------------------------------------------------
# Run training. torchtitan resolves from the conda env (pip-installed); the
# workspace dir on PYTHONPATH lets ``import mast_rl.*`` if ever needed.
# ``--dump_folder`` redirects all outputs (incl. the offline wandb run) to the
# bucket so they persist past the ephemeral node.
# ---------------------------------------------------------------------------
PYTHONPATH=. "$@" --dump_folder "${DUMP_DIR}"
