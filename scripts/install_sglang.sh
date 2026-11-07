#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install SGLang for torchtitan/rl/generator/sglang into the active RL
# environment (scripts/install_rl_env.sh), keeping its PyTorch.
#
# Usage:
#   scripts/install_sglang.sh SRC_DIR
#
#   SRC_DIR  Where to clone SGLang; it is installed editable from there. Keep it
#            outside the directory Python runs from, so the clone cannot shadow
#            the installed package.

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 SRC_DIR" >&2
  exit 1
fi
src="$1"

# The engine uses private SGLang APIs and is verified only at this commit.
sglang_commit="851642ade"

if [ ! -d "$src" ]; then
  git clone https://github.com/sgl-project/sglang "$src"
fi
git -C "$src" checkout "$sglang_commit"

overrides="$(mktemp)"
trap 'rm -f "$overrides"' EXIT
# Keep the RL environment's PyTorch, Monarch and openai (vLLM needs a newer one
# than SGLang pins).
uv pip freeze | grep -E "^(torch|torchvision|torchmonarch|openai)==" > "$overrides"
# SGLang depends on torchaudio, which the engine never imports: drop it (a uv
# override whose marker is never true removes the dependency).
echo "torchaudio ; sys_platform == 'never'" >> "$overrides"

# The engine does not use SGLang's Rust extensions: the radix tree falls back to
# its Python implementation.
SGLANG_BUILD_RUST_EXTS=none uv pip install -e "$src/python" \
  --override "$overrides" --prerelease allow
