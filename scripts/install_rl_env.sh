#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install the TorchTitan RL environment into the active Python environment.
#
# Pinned PyPI and git dependencies live in requirements.txt and
# torchtitan/rl/requirements.txt. This script adds what those files cannot
# express: matching torch/torchvision/vLLM CUDA nightlies, git-main packages
# installed without dependencies (so they cannot replace the nightly torch),
# and the Flash Attention build for the local GPU.
#
# Usage (from anywhere; requires uv):
#   scripts/install_rl_env.sh [--cuda cu132] [--skip-base]
#       [--flash-attn auto|none|3|4|3,4] [--nvcc] [--constraint FILE]...
#
#   --cuda         PyTorch nightly CUDA tag (default: cu132).
#   --skip-base    Skip requirements.txt and the RL requirements, e.g. in the
#                  CI image that already contains them.
#   --flash-attn   Comma-separated versions to install, or auto (default):
#                  FA4 on SM100+, FA3 on SM90, none elsewhere or without a GPU.
#   --nvcc         Install nvidia-cuda-nvcc so FlashInfer can JIT-compile its
#                  kernels when no CUDA toolkit is installed.
#   --constraint   Extra constraint file for the nightly resolve (repeatable).

set -euo pipefail

cuda="cu132"
skip_base=0
flash_attn="auto"
install_nvcc=0
constraints=(--constraint requirements.txt)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) cuda="$2"; shift 2 ;;
    --skip-base) skip_base=1; shift ;;
    --flash-attn) flash_attn="$2"; shift 2 ;;
    --nvcc) install_nvcc=1; shift ;;
    --constraint) constraints+=(--constraint "$(realpath "$2")"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."
set -x

if [[ "$skip_base" == "0" ]]; then
  uv pip install -r requirements.txt -r torchtitan/rl/requirements.txt
fi

# torchvision must come from the same nightly as torch so its C++ extensions
# match the torch ABI. xformers has no aarch64 wheel; nothing in TorchTitan
# imports it, but vLLM uses it when present.
nightly=(torch torchvision vllm)
if [[ "$(uname -m)" == "x86_64" ]]; then
  nightly+=(xformers)
fi
uv pip install --upgrade "${nightly[@]}" --pre \
  --extra-index-url "https://download.pytorch.org/whl/nightly/${cuda}" \
  --index-strategy unsafe-best-match \
  "${constraints[@]}"
python - "$cuda" <<'PY'
import sys
from importlib.metadata import version

wheels = {name: version(name) for name in ("torch", "torchvision", "vllm")}
print(wheels)
assert all(
    ".dev" in wheel and f"+{sys.argv[1]}" in wheel for wheel in wheels.values()
), wheels
PY

# Not on PyPI; --no-deps keeps them from replacing the nightly torch.
uv pip install --no-deps \
  "git+https://github.com/meta-pytorch/torchstore.git@main" \
  "git+https://github.com/thinking-machines-lab/batch_invariant_ops.git@main"

if [[ "$flash_attn" == "auto" ]]; then
  flash_attn=$(python - <<'PY'
import torch

if not torch.cuda.is_available():
    print("none")
else:
    major, _ = torch.cuda.get_device_capability()
    print("4" if major >= 10 else "3" if major == 9 else "none")
PY
)
fi
IFS=, read -r -a flash_attn_versions <<< "$flash_attn"
for version in "${flash_attn_versions[@]}"; do
  case "$version" in
    4) uv pip install "flash-attn-4[cu13]>=4.0.0b34" ;;
    # FA3 ships one CUDA 13.0 wheel on the torch stable ABI; every CUDA 13
    # PyTorch index, including the nightly one used above, serves it.
    3) uv pip install flash-attn-3 \
         --extra-index-url "https://download.pytorch.org/whl/nightly/${cuda}" \
         --index-strategy unsafe-best-match ;;
    none) ;;
    *) echo "Unknown --flash-attn value: $version" >&2; exit 1 ;;
  esac
done

# vLLM pulls in tilelang; TorchTitan's linear-attention paths use Attention
# Gym kernels, and CI runs without tilelang since the Kimi K3 RL support.
uv pip uninstall tilelang || true

if [[ "$install_nvcc" == "1" ]]; then
  uv pip install nvidia-cuda-nvcc
fi
