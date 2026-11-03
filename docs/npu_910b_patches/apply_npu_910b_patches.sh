#!/usr/bin/env bash
# Apply or restore the Ascend 910B flex site-package patches shipped next to this script.
# Remat / Inductor API fixes live in torchtitan-ascend; do not add them here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="apply"
if [[ "${1:-}" == "--dry-run" ]]; then
  MODE="dry-run"
elif [[ "${1:-}" == "--restore" ]]; then
  MODE="restore"
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--dry-run|--restore]" >&2
  exit 2
fi

if [[ -n "${NPU_PATCH_SITE_PACKAGES:-}" ]]; then
  SITE="${NPU_PATCH_SITE_PACKAGES}"
else
  SITE="$(python - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]))
PY
)"
fi

if [[ ! -d "${SITE}" ]]; then
  echo "site-packages not found: ${SITE}" >&2
  exit 1
fi

declare -a TARGETS=(
  "torch_npu/_inductor/kernel/flex_attention.py|flex_attention.py.patch|659bf1dbc6c9652d8a5fdf9dad9bedeb"
  "torch_npu/_inductor/runtime/triton_heuristics.py|triton_heuristics.py.patch|e0fcd810ba31c602be44fb7c5e49b058"
)

md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  else
    md5 -q "$1"
  fi
}

echo "site-packages: ${SITE}"
echo "mode: ${MODE}"

for entry in "${TARGETS[@]}"; do
  IFS='|' read -r rel patch expected <<<"${entry}"
  target="${SITE}/${rel}"
  backup="${target}.pre_npu_910b_patch"
  patch_file="${SCRIPT_DIR}/${patch}"

  if [[ ! -f "${target}" ]]; then
    echo "missing target: ${target}" >&2
    exit 1
  fi
  if [[ ! -f "${patch_file}" ]]; then
    echo "missing patch: ${patch_file}" >&2
    exit 1
  fi

  case "${MODE}" in
    dry-run)
      echo "dry-run ${rel}"
      (cd "${SITE}" && patch -p1 --dry-run < "${patch_file}") || {
        cur="$(md5_of "${target}")"
        if [[ "${cur}" == "${expected}" ]]; then
          echo "  already applied (md5 matches)"
        else
          echo "  dry-run failed and md5=${cur} (expected ${expected})" >&2
          exit 1
        fi
      }
      ;;
    restore)
      if [[ ! -f "${backup}" ]]; then
        echo "no backup for ${rel}: ${backup}" >&2
        exit 1
      fi
      cp -a "${backup}" "${target}"
      echo "restored ${rel}"
      ;;
    apply)
      cur="$(md5_of "${target}")"
      if [[ "${cur}" == "${expected}" ]]; then
        echo "skip ${rel} (already patched, md5 ok)"
        continue
      fi
      if [[ ! -f "${backup}" ]]; then
        cp -a "${target}" "${backup}"
        echo "backup ${backup}"
      fi
      (cd "${SITE}" && patch -p1 < "${patch_file}")
      got="$(md5_of "${target}")"
      if [[ "${got}" != "${expected}" ]]; then
        echo "md5 mismatch for ${rel}: got ${got}, expected ${expected}" >&2
        exit 1
      fi
      echo "patched ${rel}"
      ;;
  esac
done

echo "done."
