#!/bin/bash

# Run this script from the repository root.

# Accept optional flags to pass to uv pip install commands
# Usage: .github/scripts/setup_pyrefly.sh [flags]
UV_INSTALL_FLAGS="$@"

pip install uv
uv pip install \
  torchft-nightly \
  -r .ci/docker/requirements.txt \
  -r .ci/docker/requirements-dev.txt \
  -r .ci/docker/requirements-flux.txt \
  -r .ci/docker/requirements-transformers-modeling-backend.txt \
  -r .ci/docker/requirements-vlm.txt \
  $UV_INSTALL_FLAGS
uv pip install -U --pre --index-url https://download.pytorch.org/whl/nightly/cu128 \
  torchao \
  torch \
  $UV_INSTALL_FLAGS
uv pip install pyrefly==0.44.2 $UV_INSTALL_FLAGS
