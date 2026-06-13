#!/bin/bash
set -euo pipefail

# Build the conda env for Search-R1 RL on MAST nodes (cu130). Adapted from
# fbsource//fbcode/pytorch/torchtitan/fb/mast_rl/build_conda.sh with two
# additions:
#   * torchtitan is installed from THIS OSS checkout (which has the Search-R1
#     experiment) rather than the fbsource tree.
#   * the dense-retrieval server's deps (faiss-gpu, fastapi, uvicorn, datasets)
#     are installed so run.sh can start the retriever from the same packed env.
#
# Usage:
#   conda create -n rlmast python=3.12 -y && conda activate rlmast
#   bash mast_rl/build_conda.sh
#
# The torchtitan source location can be overridden:
#   TORCHTITAN_SRC=/path/to/torchtitan bash mast_rl/build_conda.sh

if [[ "${CONDA_DEFAULT_ENV:-}" != "rlmast" ]]; then
    echo "ERROR: Please activate the 'rlmast' conda env first: conda activate rlmast"
    exit 1
fi

# Default to the repo this script lives in (mast_rl/.. == repo root).
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TORCHTITAN_SRC="${TORCHTITAN_SRC:-$( cd "${SCRIPT_DIR}/.." && pwd )}"

echo "=== Step 1: Install PyTorch nightly, vLLM, and torchcomms (cu130) ==="
pip install torch vllm torchcomms --pre \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130

echo "=== Step 2: Install TorchStore and helpers ==="
pip install --no-deps "git+https://github.com/meta-pytorch/torchstore.git@main"
pip install pygtrie portpicker

echo "=== Step 3: Install Flash Attention 3 (for H100/H200+) ==="
pip install --no-deps flash-attn-3 --extra-index-url=https://download.pytorch.org/whl/test/cu130

echo "=== Step 4: Install batch-invariant ops ==="
pip install --no-deps "git+https://github.com/thinking-machines-lab/batch_invariant_ops.git@main"

echo "=== Step 5: Install CUDA 13 toolkit for FlashInfer JIT compilation on MAST ==="
conda install -y -c nvidia cuda-toolkit=13.0.2

echo "=== Step 5b: Install renderers (RL chat/tool renderer, required by trainer.py) ==="
# torchtitan.experiments.rl.{trainer,renderer} import `renderers` (pip pkg). Pulls
# fastokens + prime-pydantic-config (pydantic_config) + tiktoken/openai-harmony.
pip install "renderers==0.1.8.dev42"

echo "=== Step 6: Install dense-retrieval server deps (Search-R1) ==="
# faiss-gpu-cu12 ships its own bundled CUDA 12 libs in a private .libs dir, so it
# coexists with the cu130 torch wheel (this mirrors the dev-server retriever-venv:
# faiss_gpu_cu12 1.14.1). fastapi/uvicorn/pydantic serve the HTTP endpoint;
# datasets loads the wiki-18 corpus; aiohttp is the rollout-side client.
pip install faiss-gpu-cu12 fastapi "uvicorn[standard]" pydantic datasets aiohttp tqdm

echo "=== Step 7: Install TorchTitan (non-editable, from the OSS checkout) ==="
# Non-editable so the package lands inside site-packages and gets packed by
# conda-pack. Editable installs drop a `.pth` whose finder MAPPING points at the
# dev-server source path; on MAST that path does not exist, so `import
# torchtitan` fails. `submit.sh` reinstalls torchtitan the same way on every
# submit, so source edits are picked up automatically.
echo "Installing torchtitan from ${TORCHTITAN_SRC}"
( cd "${TORCHTITAN_SRC}" && pip install . )

# torchtitan's `pip install .` resolves deps against PyPI (no nightly index), which
# DOWNGRADES torch to the latest stable (2.12.0). That breaks the torch<->torchcomms
# ABI (torchcomms nightly needs the 2.13.0.dev libc10 symbol PARAM_COMMS_INFO), so
# `import torch` hard-crashes with an OSError that torch's try/except ImportError
# does NOT catch. Re-pin the matching cu130 nightly torch (no-deps) to undo it.
echo "=== Step 7b: re-pin matching cu130 nightly torch (undo torchtitan's downgrade) ==="
pip install --no-deps --force-reinstall torch --pre \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130

echo "=== Step 8: Install Meta-internal Monarch and torchx wheels ==="
# Last so nothing later overrides them. OSS torchmonarch on PyPI lacks the
# monarch.tools.commands / mast_conda scheduler used to drive MAST from Python.
rm -rf ~/monarch_wheel && mkdir ~/monarch_wheel
fbpkg fetch monarch_nightly_torch:latest_contbuild -d ~/monarch_wheel
pip install --no-deps --force-reinstall ~/monarch_wheel/torchmonarch-*-py3.12-none-linux_*.whl

rm -rf ~/torchx_wheel && mkdir ~/torchx_wheel
fbpkg fetch torchx_wheel:stable -d ~/torchx_wheel
pip install ~/torchx_wheel/torchx-*-py3.12-none-any.whl
pip install decorator  # runtime dep of FB torchx, missing from its requirements

echo ""
echo "Done. Activate this env before submitting MAST jobs:"
echo "  conda activate rlmast"
echo "  bash mast_rl/submit.sh"
