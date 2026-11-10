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
#   --cuda         PyTorch nightly CUDA tag (default: cu132). Only cu132 has
#                  the pinned vLLM wheel.
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

# torchvision nightlies use torch's stable ABI and need only torch>=2.14, so
# their date can differ from torch's. xformers has no aarch64 wheel; nothing in
# TorchTitan imports it, but vLLM uses it when present.
#
# PyTorch's nightly vLLM wheel is built from the vLLM commit in pytorch's
# .github/ci_commit_pins/vllm.txt. Newer vLLM deleted the V1 model runner that
# torchtitan/rl/generator/vllm/worker.py subclasses, so pin the 2026-11-07
# wheel (vLLM 9f03b510c). Its metadata also pins torch to that date.
# TODO: by 2026-11-22, install a vLLM PyPI release on the newest torch nightly,
# so RL CI tests the same torch as core CI. Blocked until the generator runs on
# vLLM 0.33 (https://github.com/pytorch/torchtitan/pull/5498). Hard limit
# ~2027-01: the nightly index lists only the newest ~60 days of wheels.
nightly=(torch torchvision "vllm==1.0.0.dev20261107+${cuda}")
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

# Mooncake backend of torch.distributed._transport, for ModelStateDictTransport
# (torchtitan/rl/MODEL_STATE_DICT_TRANSPORT.md). The CUDA 13 build matches cu132.
uv pip install --no-deps "mooncake-transfer-engine-cuda13==0.3.14"

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
    4) uv pip install "flash-attn-4[cu13]>=4.0.0b38" ;;
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
