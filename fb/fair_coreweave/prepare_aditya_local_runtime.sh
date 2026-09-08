#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bundle_root="${CODA_LOCAL_RUNTIME_BUNDLE:-$HOME/tmp/coda-gt-local-runtime-gb300}"
runtime_id=593b9310487595c6
archive_sha256=3ca694cca6d2cc2b13558e5c95fdb9ab39c9152872ca546214930f26fc964572
manifest_sha256=bd6172f58eb609f2f6b77064c07faf35e711e1aa1272d7162e46bae6b7f54c8d

archive="$bundle_root/$runtime_id.tar.gz"
manifest="$bundle_root/RUNTIME_MANIFEST.json"
runtime_root="$bundle_root/conda"
overlay_root="$bundle_root/github-overlay-min"

verify_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | cut -d ' ' -f 1)
  if [ "$actual" != "$expected" ]; then
    echo "checksum mismatch: $path" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
  fi
}

if [ ! -f "$archive" ]; then
  echo "runtime archive is missing: $archive" >&2
  exit 1
fi
if [ ! -f "$manifest" ]; then
  echo "runtime manifest is missing: $manifest" >&2
  exit 1
fi

verify_sha256 "$archive" "$archive_sha256"
verify_sha256 "$manifest" "$manifest_sha256"

if [ ! -x "$runtime_root/bin/python" ]; then
  tar --extract --gzip --file "$archive" --directory "$bundle_root" \
    --no-same-owner --no-same-permissions
fi

if [ ! -x "$runtime_root/bin/python" ]; then
  echo "runtime archive did not produce $runtime_root/bin/python" >&2
  exit 1
fi

if [ ! -d "$overlay_root/grain" ]; then
  if ! command -v uv >/dev/null; then
    echo "uv is required to create the Grain overlay" >&2
    exit 1
  fi
  uv pip install --target "$overlay_root" --no-deps \
    --requirement "$script_dir/requirements-runtime-overlay.txt"
fi

echo "runtime=$runtime_root"
echo "overlay=$overlay_root"
echo "The sealed runtime was not modified; conda-unpack was not run."
