#!/usr/bin/env bash
# Fail if a pytest log shows a test that skipped because the runner has too few GPUs:
# no CI job runs such a test.
# The log must come from pytest -vv: lower verbosity trims the skip reason this grep reads.
#
# Usage: check_gpu_count_skips.sh <pytest log>

set -euo pipefail

# A missing or empty log means pytest never ran or its output went elsewhere.
[[ -s "$1" ]] || { echo "::error::No pytest log at $1."; exit 1; }

# TODO: these are today's skip phrasings. With the tests' GPU-count skips (about 28 checks in 20
# files) moved to torch's skip_if_lt_x_gpu(N), one reason is left to match: "Need at least N
# accelerator devices". Out of scope for a CI change.
if grep -iE ' SKIPPED \((need at least [0-9]+ accelerator devices|(symmetric memory )?requires ([0-9]+|two|four|eight) (CUDA devices|GPUs))\)' "$1"; then
  echo "::error::The tests above skipped for lack of GPUs, so no CI job runs them. Mark them multi_gpu, or make them fit this runner's GPUs."
  exit 1
fi
