#!/usr/bin/env bash

set -euo pipefail

required_vars=(
  RUNTIME_PREFIX
  PYTORCH_SOURCE
  DIST_MOE_SOURCE
  HF_TOKENIZER_SOURCE
  TORCHTITAN_SOURCE
  FBPKG_NAME
)
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "Missing required environment variable: $var_name" >&2
    exit 2
  fi
done

for path in \
  "$RUNTIME_PREFIX" \
  "$PYTORCH_SOURCE" \
  "$DIST_MOE_SOURCE" \
  "$HF_TOKENIZER_SOURCE" \
  "$TORCHTITAN_SOURCE"; do
  if [[ ! -d "$path" ]]; then
    echo "Required directory does not exist: $path" >&2
    exit 2
  fi
done
if [[ -n "${TORCHAO_SOURCE:-}" ]] && [[ ! -d "$TORCHAO_SOURCE" ]]; then
  echo "TORCHAO_SOURCE does not exist: $TORCHAO_SOURCE" >&2
  exit 2
fi
for command_name in git meta_conda_pack sha256sum tar; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required command is unavailable: $command_name" >&2
    exit 2
  fi
done

if [[ -n "$(git -C "$TORCHTITAN_SOURCE" status --porcelain)" ]]; then
  echo "TORCHTITAN_SOURCE must be a clean checkout" >&2
  exit 2
fi
if [[ -n "$(git -C "$PYTORCH_SOURCE" status --porcelain)" ]]; then
  echo "PYTORCH_SOURCE must be a clean checkout" >&2
  exit 2
fi
if git -C "$DIST_MOE_SOURCE" rev-parse HEAD >/dev/null 2>&1; then
  if [[ -n "$(git -C "$DIST_MOE_SOURCE" status --porcelain)" ]]; then
    echo "DIST_MOE_SOURCE must be a clean checkout" >&2
    exit 2
  fi
  dist_moe_revision=$(git -C "$DIST_MOE_SOURCE" rev-parse HEAD)
else
  if [[ -z "${DIST_MOE_SOURCE_REVISION:-}" ]]; then
    echo "A non-Git DIST_MOE_SOURCE requires DIST_MOE_SOURCE_REVISION" >&2
    exit 2
  fi
  dist_moe_revision=$DIST_MOE_SOURCE_REVISION
fi
if [[ -n "${TORCHAO_SOURCE:-}" ]] \
  && [[ -n "$(git -C "$TORCHAO_SOURCE" status --porcelain)" ]]; then
  echo "TORCHAO_SOURCE must be a clean checkout" >&2
  exit 2
fi

runtime_python="$RUNTIME_PREFIX/bin/python"
if [[ ! -x "$runtime_python" ]]; then
  echo "RUNTIME_PREFIX does not contain an executable bin/python" >&2
  exit 2
fi
pytorch_revision=$(git -C "$PYTORCH_SOURCE" rev-parse HEAD)
runtime_pytorch_revision=$(
  "$runtime_python" -I -c 'import torch; print(torch.version.git_version)'
)
if [[ "$runtime_pytorch_revision" != "$pytorch_revision" ]]; then
  echo "RUNTIME_PREFIX PyTorch does not match PYTORCH_SOURCE" >&2
  echo "runtime: $runtime_pytorch_revision" >&2
  echo "source:  $pytorch_revision" >&2
  exit 2
fi

staging_parent=${STAGING_PARENT:-/tmp}
mkdir -p "$staging_parent"
staging_prefix=$(mktemp -d "$staging_parent/torchtitan-dist-moe-fbpkg.XXXXXX")
cleanup() {
  rm -rf -- "$staging_prefix"
}
trap cleanup EXIT

echo "Staging the runtime at $staging_prefix"
cp -a --reflink=auto "$RUNTIME_PREFIX/." "$staging_prefix/"

staging_python="$staging_prefix/bin/python"
if [[ ! -x "$staging_python" ]]; then
  echo "RUNTIME_PREFIX does not contain an executable bin/python" >&2
  exit 2
fi
site_packages=$(
  "$staging_python" -I -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)
if [[ "$site_packages" != "$staging_prefix"/* ]]; then
  echo "Staged Python resolved site-packages outside the staging prefix" >&2
  exit 2
fi

# The development PyTorch install uses an editable import hook. Merge the
# Python source into its already self-contained compiled installation, then
# remove the hook so the packed environment does not refer to the build host.
cp -a "$PYTORCH_SOURCE/torch/." "$site_packages/torch/"
mkdir -p "$site_packages/torchgen"
cp -a "$PYTORCH_SOURCE/torchgen/." "$site_packages/torchgen/"
mkdir -p "$site_packages/functorch"
cp -a "$PYTORCH_SOURCE/functorch/." "$site_packages/functorch/"
printf '%s\n' '# PyTorch sources are materialized inside this environment.' \
  >"$site_packages/_editable_skbc_torch.pth"
for direct_url in "$site_packages"/torch-*.dist-info/direct_url.json; do
  [[ -e "$direct_url" ]] || continue
  printf '%s\n' '{"dir_info": {}, "url": "file:///materialized/pytorch"}' \
    >"$direct_url"
done

flash_attn_source=${FLASH_ATTN_SOURCE:-$PYTORCH_SOURCE/third_party/flash-attention/flash_attn/cute}
if [[ -d "$flash_attn_source" ]]; then
  "$staging_python" -m pip install --no-deps --no-build-isolation \
    --force-reinstall "$flash_attn_source"
fi
if [[ -n "${TORCHAO_SOURCE:-}" ]]; then
  "$staging_python" -m pip install --no-deps --no-build-isolation \
    --force-reinstall "$TORCHAO_SOURCE"
fi
"$staging_python" -m pip install --no-deps --no-build-isolation \
  --force-reinstall "$DIST_MOE_SOURCE"
"$staging_python" -m pip install --no-deps "tlparse==0.4.3"

# TorchTitan itself is archived at the exact clean source revision. Running
# from this directory also carries the c4_test and tokenizer assets needed by
# the Chien-Chin recipe.
for editable_pth in "$site_packages"/__editable__.torchtitan-*.pth; do
  [[ -e "$editable_pth" ]] || continue
  printf '%s\n' '# TorchTitan runs from the archived source in src/torchtitan.' \
    >"$editable_pth"
done
for direct_url in "$site_packages"/torchtitan-*.dist-info/direct_url.json; do
  [[ -e "$direct_url" ]] || continue
  printf '%s\n' \
    '{"dir_info": {}, "url": "file:///materialized/torchtitan"}' \
    >"$direct_url"
done
mkdir -p "$staging_prefix/src/torchtitan"
git -C "$TORCHTITAN_SOURCE" archive HEAD \
  | tar -x -C "$staging_prefix/src/torchtitan"
tokenizer_dir="$staging_prefix/src/torchtitan/assets/hf/DeepSeek-V3.1-Base"
mkdir -p "$tokenizer_dir"
for tokenizer_file in tokenizer.json tokenizer_config.json; do
  if [[ ! -f "$HF_TOKENIZER_SOURCE/$tokenizer_file" ]]; then
    echo "Missing required tokenizer file: $HF_TOKENIZER_SOURCE/$tokenizer_file" >&2
    exit 2
  fi
  cp -a "$HF_TOKENIZER_SOURCE/$tokenizer_file" "$tokenizer_dir/"
done
for tokenizer_file in added_tokens.json special_tokens_map.json; do
  if [[ -f "$HF_TOKENIZER_SOURCE/$tokenizer_file" ]]; then
    cp -a "$HF_TOKENIZER_SOURCE/$tokenizer_file" "$tokenizer_dir/"
  fi
done

provenance_dir="$staging_prefix/torchtitan_fbpkg_provenance"
mkdir -p "$provenance_dir"
{
  echo "pytorch=$(git -C "$PYTORCH_SOURCE" rev-parse HEAD)"
  echo "torchtitan=$(git -C "$TORCHTITAN_SOURCE" rev-parse HEAD)"
  echo "dist_moe=$dist_moe_revision"
  if [[ -n "${TORCHAO_SOURCE:-}" ]]; then
    echo "torchao=$(git -C "$TORCHAO_SOURCE" rev-parse HEAD)"
  fi
} >"$provenance_dir/revisions.txt"
"$staging_python" -m pip freeze --all >"$provenance_dir/pip_freeze.txt"
printf '%s\n' 'meta_conda_pack_ignore_missing_files=true' \
  >"$provenance_dir/pack_policy.txt"
sha256sum "$tokenizer_dir"/*.json >"$provenance_dir/tokenizer_sha256.txt"
"$staging_python" - "$site_packages/dist_moe" <<'PY' \
  >"$provenance_dir/dist_moe_installed_sha256.txt"
from hashlib import sha256
from pathlib import Path
import sys

root = Path(sys.argv[1])
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    print(sha256(path.read_bytes()).hexdigest(), path.relative_to(root))
PY

source_paths=(
  "$PYTORCH_SOURCE"
  "$TORCHTITAN_SOURCE"
  "$DIST_MOE_SOURCE"
  "$flash_attn_source"
)
if [[ -n "${TORCHAO_SOURCE:-}" ]]; then
  source_paths+=("$TORCHAO_SOURCE")
fi
for pth_file in "$site_packages"/*.pth; do
  [[ -e "$pth_file" ]] || continue
  pth_contents=$(<"$pth_file")
  for source_path in "${source_paths[@]}"; do
    if [[ "$pth_contents" == *"$source_path"* ]]; then
      echo "Editable source path remains in $pth_file: $source_path" >&2
      exit 2
    fi
  done
done

(
  cd "$staging_prefix/src/torchtitan"
  env -u PYTHONPATH PYTHONNOUSERSITE=1 "$staging_python" - <<'PY'
from pathlib import Path
import importlib.metadata as metadata
import sys

import dist_moe
import dist_moe._blockscaled  # noqa: F401
import functorch
import torch
import torchao
import torchtitan
import torchtitan_recipes
from dist_moe import BlockScaledFormat, DistMoeBlockScaledConfig
from torch.utils.checkpoint import _is_cacheable_effect

prefix = Path(sys.prefix).resolve()
for module in (functorch, torch, torchao, dist_moe, torchtitan, torchtitan_recipes):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(prefix):
        raise RuntimeError(f"{module.__name__} resolves outside fbpkg: {path}")

assert hasattr(torch.ops.aten, "_scaled_addmm_")
assert hasattr(torch.ops.dist_moe, "block_scaled_backward_accumulate")
assert hasattr(torch.ops.dist_moe, "bf16_backward_accumulate")
assert BlockScaledFormat.MXFP8_E4M3
assert DistMoeBlockScaledConfig
assert _is_cacheable_effect
print("torch", torch.__version__, torch.version.git_version)
print("torchao", metadata.version("torchao"), torchao.__file__)
print("dist_moe", dist_moe.__file__)
print("torchtitan", torchtitan.__file__)
PY
)

echo "Publishing $FBPKG_NAME"
meta_conda_pack publish --prefix "$staging_prefix" --ignore-missing-files \
  "$FBPKG_NAME"
