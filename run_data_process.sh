#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ruisizhang123/ruisizhang123_data/tree/dolma3_mix-6T-1025-7B}
JOBS=${JOBS:-1}

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS must be a positive integer, got: $JOBS" >&2
  exit 2
fi

run_one() {
  local d="$1"
  local name
  name=$(basename "$d")
  local out="$ROOT/pre-tokenize-data/$name"
  local lock_dir="$out/.process.lock"

  if [ -f "$out/metadata.json" ]; then
    echo "skip existing $name"
    return 0
  fi

  mkdir -p "$out"
  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "skip locked $name ($lock_dir exists)"
    return 0
  fi

  (
    trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT
    echo "pretokenizing $name"
    python pre-processing/pretokenize_dolma.py --input-dir "$d"
  )
}

running=0
failed=0

for d in "$ROOT"/data/*; do
  [ -d "$d" ] || continue

  while [ "$running" -ge "$JOBS" ]; do
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  done

  run_one "$d" &
  running=$((running + 1))
done

while [ "$running" -gt 0 ]; do
  if ! wait -n; then
    failed=1
  fi
  running=$((running - 1))
done

exit "$failed"
