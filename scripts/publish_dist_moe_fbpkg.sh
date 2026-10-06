#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

PRENORMALIZED_RUNTIME=${PRENORMALIZED_RUNTIME:-0}
EXPECTED_CUDA_ARCH=${EXPECTED_CUDA_ARCH:-sm_103}
TORCHAO_USE_CPP=${TORCHAO_USE_CPP:-1}
FLASH_ATTN_QUACK_VERSION=${FLASH_ATTN_QUACK_VERSION:-0.5.3}
FLASH_ATTN_DLPACK_VERSION=${FLASH_ATTN_DLPACK_VERSION:-0.1.5}
ATTN_GYM_VERSION=${ATTN_GYM_VERSION:-0.0.16}
CUDA_TOOLKIT_ROOT=${CUDA_TOOLKIT_ROOT:-/usr/local/cuda-13.0}
SETUPTOOLS_BUILD_VERSION=78.1.0
CYTHON_BUILD_VERSION=0.29.37
if [[ "$PRENORMALIZED_RUNTIME" != 0 && "$PRENORMALIZED_RUNTIME" != 1 ]]; then
  echo "PRENORMALIZED_RUNTIME must be 0 or 1" >&2
  exit 2
fi

required_vars=(
  BASE_RUNTIME_FBPKG_ID
  RUNTIME_PREFIX
  COMPILER_PREFIX
  EXPECTED_TORCHTITAN_REVISION
  EXPECTED_PYTORCH_BINARY_REVISION
  EXPECTED_PYTORCH_SOURCE_REVISION
  EXPECTED_DIST_MOE_REVISION
  EXPECTED_TORCHAO_REVISION
  PYTORCH_SOURCE
  DIST_MOE_SOURCE
  TORCHAO_SOURCE
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
if [[ ! "$BASE_RUNTIME_FBPKG_ID" =~ ^[A-Za-z0-9._+-]+:[0-9a-f]{32}$ ]]; then
  echo "BASE_RUNTIME_FBPKG_ID must be a full immutable PACKAGE:UUID" >&2
  exit 2
fi

for path in \
  "$RUNTIME_PREFIX" \
  "$COMPILER_PREFIX" \
  "$PYTORCH_SOURCE" \
  "$DIST_MOE_SOURCE" \
  "$HF_TOKENIZER_SOURCE" \
  "$TORCHTITAN_SOURCE"; do
  if [[ ! -d "$path" ]]; then
    echo "Required directory does not exist: $path" >&2
    exit 2
  fi
done
for compiler_name in \
  aarch64-conda-linux-gnu-gcc \
  aarch64-conda-linux-gnu-g++; do
  if [[ ! -x "$COMPILER_PREFIX/bin/$compiler_name" ]]; then
    echo "COMPILER_PREFIX does not contain bin/$compiler_name" >&2
    exit 2
  fi
done
if [[ ! -d "$TORCHAO_SOURCE" ]]; then
  echo "TORCHAO_SOURCE does not exist: $TORCHAO_SOURCE" >&2
  exit 2
fi
if [[ -n "${TORCHAO_MXFP8_EXTENSION:-}" ]] \
  && [[ ! -f "$TORCHAO_MXFP8_EXTENSION" ]]; then
  echo "TORCHAO_MXFP8_EXTENSION does not exist: $TORCHAO_MXFP8_EXTENSION" >&2
  exit 2
fi
if [[ ! -f "$CUDA_TOOLKIT_ROOT/lib64/libcudart.so.13" ]]; then
  echo "CUDA_TOOLKIT_ROOT does not contain lib64/libcudart.so.13" >&2
  exit 2
fi
if [[ ! -x "$CUDA_TOOLKIT_ROOT/bin/nvcc" ]]; then
  echo "CUDA_TOOLKIT_ROOT does not contain an executable bin/nvcc" >&2
  exit 2
fi
for command_name in git jq meta_conda_pack rsync sha256sum tar; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Required command is unavailable: $command_name" >&2
    exit 2
  fi
done

expected_revision_vars=(
  EXPECTED_TORCHTITAN_REVISION
  EXPECTED_PYTORCH_BINARY_REVISION
  EXPECTED_PYTORCH_SOURCE_REVISION
  EXPECTED_DIST_MOE_REVISION
  EXPECTED_TORCHAO_REVISION
)
for var_name in "${expected_revision_vars[@]}"; do
  if [[ ! "${!var_name}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "$var_name must be a full 40-character lowercase commit" >&2
    exit 2
  fi
done

if [[ -n "$(git -C "$TORCHTITAN_SOURCE" status --porcelain)" ]]; then
  echo "TORCHTITAN_SOURCE must be a clean checkout" >&2
  exit 2
fi
torchtitan_revision=$(git -C "$TORCHTITAN_SOURCE" rev-parse HEAD)
if [[ "$torchtitan_revision" != "$EXPECTED_TORCHTITAN_REVISION" ]]; then
  echo "TORCHTITAN_SOURCE does not match EXPECTED_TORCHTITAN_REVISION" >&2
  echo "actual:   $torchtitan_revision" >&2
  echo "expected: $EXPECTED_TORCHTITAN_REVISION" >&2
  exit 2
fi
if [[ -n "$(git -C "$PYTORCH_SOURCE" status --porcelain)" ]]; then
  echo "PYTORCH_SOURCE must be a clean checkout" >&2
  exit 2
fi
pytorch_revision=$(git -C "$PYTORCH_SOURCE" rev-parse HEAD)
if [[ "$pytorch_revision" != "$EXPECTED_PYTORCH_SOURCE_REVISION" ]]; then
  echo "PYTORCH_SOURCE does not match EXPECTED_PYTORCH_SOURCE_REVISION" >&2
  echo "actual:   $pytorch_revision" >&2
  echo "expected: $EXPECTED_PYTORCH_SOURCE_REVISION" >&2
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
if [[ "$dist_moe_revision" != "$EXPECTED_DIST_MOE_REVISION" ]]; then
  echo "DIST_MOE_SOURCE does not match EXPECTED_DIST_MOE_REVISION" >&2
  echo "actual:   $dist_moe_revision" >&2
  echo "expected: $EXPECTED_DIST_MOE_REVISION" >&2
  exit 2
fi
if [[ -n "$(git -C "$TORCHAO_SOURCE" status --porcelain)" ]]; then
  echo "TORCHAO_SOURCE must be a clean checkout" >&2
  exit 2
fi
torchao_revision=$(git -C "$TORCHAO_SOURCE" rev-parse HEAD)
if [[ "$torchao_revision" != "$EXPECTED_TORCHAO_REVISION" ]]; then
  echo "TORCHAO_SOURCE does not match EXPECTED_TORCHAO_REVISION" >&2
  echo "actual:   $torchao_revision" >&2
  echo "expected: $EXPECTED_TORCHAO_REVISION" >&2
  exit 2
fi

runtime_python="$RUNTIME_PREFIX/bin/python"
if [[ ! -x "$runtime_python" ]]; then
  echo "RUNTIME_PREFIX does not contain an executable bin/python" >&2
  exit 2
fi
runtime_pytorch_revision=$(
  "$runtime_python" -I -c 'import torch; print(torch.version.git_version)'
)
if [[ "$runtime_pytorch_revision" != "$EXPECTED_PYTORCH_BINARY_REVISION" ]]; then
  echo "RUNTIME_PREFIX PyTorch does not match EXPECTED_PYTORCH_BINARY_REVISION" >&2
  echo "runtime: $runtime_pytorch_revision" >&2
  echo "expected: $EXPECTED_PYTORCH_BINARY_REVISION" >&2
  exit 2
fi
runtime_cuda_version=$(
  "$runtime_python" -I -c 'import torch; print(torch.version.cuda or "")'
)
toolkit_cuda_version=
if ! nvcc_version_output=$("$CUDA_TOOLKIT_ROOT/bin/nvcc" --version); then
  echo "Unable to run CUDA_TOOLKIT_ROOT/bin/nvcc" >&2
  exit 2
fi
while IFS= read -r nvcc_line; do
  if [[ "$nvcc_line" =~ release[[:space:]]+([0-9]+\.[0-9]+), ]]; then
    toolkit_cuda_version=${BASH_REMATCH[1]}
    break
  fi
done <<<"$nvcc_version_output"
if [[ -z "$toolkit_cuda_version" ]]; then
  echo "Unable to determine the CUDA_TOOLKIT_ROOT nvcc version" >&2
  exit 2
fi
if [[ "$toolkit_cuda_version" != "$runtime_cuda_version" ]]; then
  echo "CUDA_TOOLKIT_ROOT does not match the staged PyTorch CUDA version" >&2
  echo "toolkit: $toolkit_cuda_version" >&2
  echo "PyTorch: $runtime_cuda_version" >&2
  exit 2
fi

# Bind every local extension build to the validated toolkit. The caller's PATH
# may contain an unrelated nvcc from a development environment.
export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export CUDACXX="$CUDA_TOOLKIT_ROOT/bin/nvcc"
export PATH="$CUDA_TOOLKIT_ROOT/bin:$PATH"
if ! git -C "$PYTORCH_SOURCE" cat-file -e \
  "$EXPECTED_PYTORCH_BINARY_REVISION^{commit}" 2>/dev/null; then
  echo "EXPECTED_PYTORCH_BINARY_REVISION is unavailable in PYTORCH_SOURCE" >&2
  exit 2
fi
if ! git -C "$PYTORCH_SOURCE" merge-base --is-ancestor \
  "$EXPECTED_PYTORCH_BINARY_REVISION" "$pytorch_revision"; then
  echo "PYTORCH_SOURCE must descend from EXPECTED_PYTORCH_BINARY_REVISION" >&2
  echo "binary: $EXPECTED_PYTORCH_BINARY_REVISION" >&2
  echo "source: $pytorch_revision" >&2
  exit 2
fi

# This package overlays one reviewed Python-only pipeline fix onto an older
# compiled PyTorch runtime. Reject every production-source delta except that
# exact file; tests and docs do not enter the installed runtime.
mapfile -t pytorch_overlay_files < <(
  git -C "$PYTORCH_SOURCE" diff --name-only \
    "$EXPECTED_PYTORCH_BINARY_REVISION..$pytorch_revision"
)
for overlay_path in "${pytorch_overlay_files[@]}"; do
  case "$overlay_path" in
    torch/distributed/pipelining/stage.py)
      if ! git -C "$PYTORCH_SOURCE" cat-file -e \
        "$pytorch_revision:$overlay_path" 2>/dev/null; then
        echo "PyTorch overlay removes its allowlisted runtime file" >&2
        exit 2
      fi
      ;;
    test/*|docs/*) ;;
    *)
      echo "PyTorch overlay contains a non-allowlisted path: $overlay_path" >&2
      exit 2
      ;;
  esac
done
if [[ "$PRENORMALIZED_RUNTIME" == 1 ]] \
  && [[ "$pytorch_revision" != "$EXPECTED_PYTORCH_BINARY_REVISION" ]]; then
  echo "A PyTorch source overlay requires PRENORMALIZED_RUNTIME=0" >&2
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

# PyTorch links directly to the CUDA runtime, but the source build environment
# provides it from the host toolkit instead of the Conda prefix. MuseSpark
# workers do not mount that toolkit, so keep the runtime package self-contained.
cp -a "$CUDA_TOOLKIT_ROOT/lib64/"libcudart.so* "$staging_prefix/lib/"

staging_python="$staging_prefix/bin/python"
if [[ ! -x "$staging_python" ]]; then
  echo "RUNTIME_PREFIX does not contain an executable bin/python" >&2
  exit 2
fi

# Relocate a previously packed runtime to the staging path before changing
# files. Its recorded offsets still refer to the prefix in pack-meta/history.
pack_metadata_dir="$staging_prefix/pack-meta"
if [[ -d "$pack_metadata_dir" ]]; then
  for pack_metadata_file in history.jsonl offsets.jsonl; do
    if [[ ! -f "$pack_metadata_dir/$pack_metadata_file" ]]; then
      echo "RUNTIME_PREFIX/pack-meta is missing $pack_metadata_file" >&2
      exit 2
    fi
  done
  if [[ ! -f "$staging_prefix/bin/conda-unpack-fb" ]]; then
    echo "Packed RUNTIME_PREFIX is missing bin/conda-unpack-fb" >&2
    exit 2
  fi
  "$staging_python" "$staging_prefix/bin/conda-unpack-fb"
elif [[ -e "$pack_metadata_dir" ]]; then
  echo "RUNTIME_PREFIX/pack-meta is not a directory" >&2
  exit 2
fi

# Triton and TorchInductor compile small host extensions at runtime. The MAST
# worker image does not provide a host compiler, so copy the exact Conda
# compiler packages into the runtime instead of depending on the worker image.
compiler_packages=(
  binutils_impl_linux-aarch64
  binutils_linux-aarch64
  gcc_impl_linux-aarch64
  gcc_linux-aarch64
  gxx_impl_linux-aarch64
  gxx_linux-aarch64
  kernel-headers_linux-aarch64
  libgcc-devel_linux-aarch64
  libsanitizer
  libstdcxx-devel_linux-aarch64
  sysroot_linux-aarch64
)
compiler_metadata_files=()
for package_name in "${compiler_packages[@]}"; do
  package_metadata=(
    "$COMPILER_PREFIX"/conda-meta/"$package_name"-*.json
  )
  if [[ ${#package_metadata[@]} -ne 1 || ! -f "${package_metadata[0]}" ]]; then
    echo "COMPILER_PREFIX must contain exactly one $package_name package" >&2
    exit 2
  fi
  package_metadata=${package_metadata[0]}
  compiler_metadata_files+=("$package_metadata")
  rsync --archive --relative \
    --files-from=<(jq -r '.files[]' "$package_metadata") \
    "$COMPILER_PREFIX/" "$staging_prefix/"
  cp -a "$package_metadata" "$staging_prefix/conda-meta/"
done

# meta_conda_pack materializes directory symlinks in a fetched package, while
# the Conda metadata lists only the original symlink path. Copy the complete
# isolated sysroot so lib -> lib64 and usr/lib -> lib64 retain their contents.
rsync --archive \
  "$COMPILER_PREFIX/aarch64-conda-linux-gnu/sysroot/" \
  "$staging_prefix/aarch64-conda-linux-gnu/sysroot/"

staging_cc="$staging_prefix/bin/aarch64-conda-linux-gnu-gcc"
staging_cxx="$staging_prefix/bin/aarch64-conda-linux-gnu-g++"
if [[ ! -x "$staging_cc" || ! -x "$staging_cxx" ]]; then
  echo "Failed to stage the Conda C/C++ compiler" >&2
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

# Every local source package below uses the staged build tooling with build
# isolation disabled. Repair the pinned tools before the first such package;
# the base runtime's copies may be incomplete after relocation.
"$staging_python" -m pip install --no-deps --force-reinstall \
  "setuptools==$SETUPTOOLS_BUILD_VERSION" \
  "Cython==$CYTHON_BUILD_VERSION"
"$staging_python" -I - \
  "$SETUPTOOLS_BUILD_VERSION" "$CYTHON_BUILD_VERSION" <<'PY'
import importlib.metadata as metadata
import sys

import Cython
import Cython.Build.Dependencies  # noqa: F401
import setuptools
import setuptools.build_meta  # noqa: F401

expected_setuptools, expected_cython = sys.argv[1:]
assert metadata.version("setuptools") == expected_setuptools
assert setuptools.__version__ == expected_setuptools
assert metadata.version("Cython") == expected_cython
assert Cython.__version__ == expected_cython
PY

flash_attn_source=${FLASH_ATTN_SOURCE:-$PYTORCH_SOURCE/third_party/flash-attention/flash_attn/cute}
if [[ "$PRENORMALIZED_RUNTIME" == 0 ]]; then
  # Merge only tracked source from the reviewed revision into the
  # self-contained compiled installation. Using git archive keeps ignored build
  # products from the source checkout out of the artifact.
  git -C "$PYTORCH_SOURCE" archive "$pytorch_revision" \
    torch torchgen functorch \
    | tar -x -C "$site_packages"
  "$staging_python" - \
    "$site_packages/torch/version.py" \
    "$EXPECTED_PYTORCH_BINARY_REVISION" \
    "$pytorch_revision" <<'PY'
from pathlib import Path
import re
import sys

version_path = Path(sys.argv[1])
binary_revision = sys.argv[2]
source_revision = sys.argv[3]
text = version_path.read_text()

git_pattern = re.compile(r"^git_version = ['\"][0-9a-f]+['\"]$", re.MULTILINE)
text, num_git_replacements = git_pattern.subn(
    f"git_version = {source_revision!r}", text
)
version_pattern = re.compile(
    rf"^__version__ = (['\"])(.+)\+git{binary_revision[:7]}\1$", re.MULTILINE
)
version_match = version_pattern.search(text)
if num_git_replacements != 1 or version_match is None:
    raise RuntimeError(f"unexpected PyTorch version metadata in {version_path}")
base_version = version_match.group(2)
text = version_pattern.sub(
    f"__version__ = {f'{base_version}+git{source_revision[:7]}'!r}", text
)
version_path.write_text(text)
PY
  printf '%s\n' '# PyTorch sources are materialized inside this environment.' \
    >"$site_packages/_editable_skbc_torch.pth"
  for direct_url in "$site_packages"/torch-*.dist-info/direct_url.json; do
    [[ -e "$direct_url" ]] || continue
    printf '%s\n' '{"dir_info": {}, "url": "file:///materialized/pytorch"}' \
      >"$direct_url"
  done

  if [[ -d "$flash_attn_source" ]]; then
    "$staging_python" -m pip install --no-deps \
      "quack-kernels==$FLASH_ATTN_QUACK_VERSION" \
      "torch-c-dlpack-ext==$FLASH_ATTN_DLPACK_VERSION"
    "$staging_python" -m pip install --no-deps --no-build-isolation \
      --force-reinstall "$flash_attn_source"
    # These two pure-Python packages run against the staged CUTLASS 4.7.1,
    # as verified by the PP2 GraphTrainer gate. Their metadata retains an
    # obsolete exact dependency on the unavailable 4.6.0.dev0 package.
    rm -rf -- \
      "$site_packages"/flash_attn_4-*.dist-info \
      "$site_packages"/quack_kernels-*.dist-info
  fi
  if [[ -n "${TORCHAO_SOURCE:-}" ]]; then
    env USE_CPP="$TORCHAO_USE_CPP" \
      "$staging_python" -m pip install --no-deps --no-build-isolation \
        --force-reinstall "$TORCHAO_SOURCE"
  fi
  if [[ -n "${TORCHAO_MXFP8_EXTENSION:-}" ]]; then
    rm -f -- "$site_packages"/torchao/_C_mxfp8*.so
    cp -a "$TORCHAO_MXFP8_EXTENSION" "$site_packages/torchao/"
  fi
  "$staging_python" -m pip install --no-deps --no-build-isolation \
    --force-reinstall "$DIST_MOE_SOURCE"
  "$staging_python" -m pip install --no-deps "tlparse==0.4.3"
  "$staging_python" -m pip install --no-deps --force-reinstall "click==8.4.2"
  "$staging_python" -m pip uninstall --yes spin
fi

# conda-pack follows this compatibility symlink and materializes a duplicate
# lib/python3.1 tree in the fetched package. Python 3.12 does not need it.
python_compat_path="$staging_prefix/lib/python3.1"
if [[ -L "$python_compat_path" ]]; then
  if [[ "$(readlink "$python_compat_path")" != python3.12 ]]; then
    echo "Unexpected lib/python3.1 symlink target" >&2
    exit 2
  fi
  rm -- "$python_compat_path"
elif [[ -e "$python_compat_path" ]]; then
  echo "RUNTIME_PREFIX contains a materialized lib/python3.1 tree" >&2
  exit 2
fi

# Force meta_conda_pack to rescan after updating a previously packed runtime.
if [[ -d "$pack_metadata_dir" ]]; then
  rm -f -- \
    "$pack_metadata_dir/history.jsonl" \
    "$pack_metadata_dir/offsets.jsonl" \
    "$pack_metadata_dir/log.txt"
  if ! rmdir -- "$pack_metadata_dir"; then
    echo "Unexpected files in RUNTIME_PREFIX/pack-meta" >&2
    exit 2
  fi
elif [[ -e "$pack_metadata_dir" ]]; then
  echo "RUNTIME_PREFIX/pack-meta is not a directory" >&2
  exit 2
fi

# Refresh TorchTitan's package metadata so dependency checks describe the
# archived source revision rather than the revision used to create the base
# runtime. The archived checkout below remains the runtime import location.
"$staging_python" -m pip install --no-deps --force-reinstall \
  "attn-gym==$ATTN_GYM_VERSION"
"$staging_python" -m pip install --no-deps --no-build-isolation \
  --force-reinstall "$TORCHTITAN_SOURCE"
"$staging_python" -I -m pip check

# A prepacked development runtime can retain absolute console-script shebangs
# from the checkout that built it. Use relative wrappers for the entry points
# required by the runbook so the fetched fbpkg cannot fall back to host Python.
write_python_module_wrapper() {
  local script_name=$1
  local module_name=$2
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'runtime_bin_dir=$(cd -- "$(dirname -- "$0")" && pwd)' \
    "exec \"\$runtime_bin_dir/python\" -m $module_name \"\$@\"" \
    >"$staging_prefix/bin/$script_name"
  chmod 755 "$staging_prefix/bin/$script_name"
}
write_python_module_wrapper torchrun torch.distributed.run
write_python_module_wrapper pip pip
write_python_module_wrapper pip3 pip
"$staging_prefix/bin/torchrun" --help >/dev/null
"$staging_prefix/bin/pip" --version

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
torchtitan_archive_dir="$staging_prefix/src/torchtitan"
rm -rf -- "$torchtitan_archive_dir"
mkdir -p "$torchtitan_archive_dir"
git -C "$TORCHTITAN_SOURCE" archive HEAD \
  | tar -x -C "$torchtitan_archive_dir"
tokenizer_dirs=(
  "$staging_prefix/src/torchtitan/assets/hf/DeepSeek-V3.1-Base"
  "$staging_prefix/src/torchtitan/assets/hf/deepseek-moe-16b-base"
)
for tokenizer_dir in "${tokenizer_dirs[@]}"; do
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
done

provenance_dir="$staging_prefix/torchtitan_fbpkg_provenance"
mkdir -p "$provenance_dir"
{
  echo "base_runtime_fbpkg=$BASE_RUNTIME_FBPKG_ID"
  echo "pytorch_binary_base=$EXPECTED_PYTORCH_BINARY_REVISION"
  echo "pytorch=$pytorch_revision"
  echo "torchtitan=$torchtitan_revision"
  echo "dist_moe=$dist_moe_revision"
  echo "torchao=$torchao_revision"
  echo "flash_attn_source=pytorch:$pytorch_revision"
  echo "flash_attn_quack=$FLASH_ATTN_QUACK_VERSION"
  echo "flash_attn_dlpack=$FLASH_ATTN_DLPACK_VERSION"
  echo "build_setuptools=$SETUPTOOLS_BUILD_VERSION"
  echo "build_cython=$CYTHON_BUILD_VERSION"
  echo "cuda_toolkit=$toolkit_cuda_version"
  echo "expected_cuda_arch=$EXPECTED_CUDA_ARCH"
} >"$provenance_dir/revisions.txt"
printf '%s\n' "${pytorch_overlay_files[@]}" \
  >"$provenance_dir/pytorch_overlay_files.txt"
if [[ -n "${TORCHAO_MXFP8_EXTENSION:-}" ]]; then
  sha256sum "$site_packages/torchao/$(basename "$TORCHAO_MXFP8_EXTENSION")" \
    >"$provenance_dir/torchao_mxfp8_extension_sha256.txt"
fi
sha256sum "$staging_prefix/lib/libcudart.so.13" \
  >"$provenance_dir/libcudart_sha256.txt"
for compiler_metadata_file in "${compiler_metadata_files[@]}"; do
  basename "$compiler_metadata_file"
done >"$provenance_dir/compiler_packages.txt"
sha256sum "$staging_cc" "$staging_cxx" \
  >"$provenance_dir/compiler_sha256.txt"
"$staging_python" -m pip freeze --all >"$provenance_dir/pip_freeze.txt"
printf '%s\n' \
  'meta_conda_pack_ignore_missing_files=true' \
  'runtime_tree_manifests_exclude=*.pyc' \
  >"$provenance_dir/pack_policy.txt"
(
  cd "$staging_prefix/src/torchtitan"
  sha256sum \
    assets/hf/DeepSeek-V3.1-Base/*.json \
    assets/hf/deepseek-moe-16b-base/*.json
) >"$provenance_dir/tokenizer_sha256.txt"
runtime_tree_manifest="$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/runtime_tree_manifest.py"
"$staging_python" "$runtime_tree_manifest" create \
  --unprefixed \
  "$provenance_dir/dist_moe_installed_sha256.txt" \
  "$site_packages/dist_moe"
"$staging_python" "$runtime_tree_manifest" create \
  "$provenance_dir/flash_attn_runtime_sha256.txt" \
  "$site_packages/flash_attn/cute" \
  "$site_packages/quack" \
  "$site_packages/torch_c_dlpack_ext"

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
  env -u PYTHONPATH PYTHONNOUSERSITE=1 \
    CC="$staging_cc" \
    CXX="$staging_cxx" \
    LD_LIBRARY_PATH="$staging_prefix/lib:${LD_LIBRARY_PATH:-}" \
    PYTORCH_SOURCE_REVISION="$pytorch_revision" \
    EXPECTED_CUDA_ARCH="$EXPECTED_CUDA_ARCH" "$staging_python" - <<'PY'
from pathlib import Path
import importlib.metadata as metadata
import os
import sys

import dist_moe
import dist_moe._blockscaled  # noqa: F401
import flash_attn.cute
import flash_attn.cute.interface  # noqa: F401
import functorch
import quack
import torch
import torchao
import torch_c_dlpack_ext
import torchtitan
import torchtitan_recipes
from dist_moe import BlockScaledConfig, BlockScaledFormat
from torchao.prototype.mx_formats.kernels import mxfp8_quantize_cuda
from torch.utils.checkpoint import _is_cacheable_effect
from triton.runtime.build import compile_so_from_src

prefix = Path(sys.prefix).resolve()
for module in (
    flash_attn.cute,
    functorch,
    quack,
    torch,
    torchao,
    torch_c_dlpack_ext,
    dist_moe,
    torchtitan,
    torchtitan_recipes,
):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(prefix):
        raise RuntimeError(f"{module.__name__} resolves outside fbpkg: {path}")

assert hasattr(torch.ops.aten, "_scaled_addmm_")
assert hasattr(torch.ops.dist_moe, "block_scaled_backward_accumulate_")
assert hasattr(torch.ops.dist_moe, "bf16_backward_accumulate_")
assert BlockScaledFormat.MXFP8_E4M3
assert BlockScaledConfig
assert _is_cacheable_effect
assert torch.version.git_version == os.environ["PYTORCH_SOURCE_REVISION"]
assert torch.cuda.get_arch_list() == [os.environ["EXPECTED_CUDA_ARCH"]]
assert torch.cuda.get_device_capability() == (10, 3)
assert torch.backends.cudnn.version() is not None
assert torch._C._dispatch_has_kernel_for_dispatch_key(
    "torchao::mxfp8_quantize", "CUDA"
)
compiler_probe = compile_so_from_src(
    "int torchtitan_compiler_probe(void) { return 0; }",
    "torchtitan_compiler_probe",
)
assert Path(compiler_probe).is_file(), compiler_probe

torch.manual_seed(0)
x = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
mxfp8_quantize_cuda(x, rowwise=True, colwise=True)

a = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
b = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
actual = a @ b
expected = (a.float() @ b.float()).bfloat16()
torch.cuda.synchronize()
torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
print("torch", torch.__version__, torch.version.git_version)
print("torchao", metadata.version("torchao"), torchao.__file__)
print("dist_moe", dist_moe.__file__)
print("torchtitan", torchtitan.__file__)
print("bf16_mm", tuple(actual.shape))
PY
)

echo "Publishing $FBPKG_NAME"
meta_conda_pack publish --prefix "$staging_prefix" --ignore-missing-files \
  "$FBPKG_NAME"
