#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
#
# Set COMM_BACKEND="fake" for dry-run validation without real communication:
#    - Uses fake process groups (no actual communication)
#    - Runs on a single GPU without torchrun or NCCL initialization
#    - Useful for validating configuration and model setup
#    Example: NGPU=32 COMM_BACKEND="fake" ./run_train.sh
# Real-PP/fake-SPMD uses torchrun directly because NGPU is the logical world
# size rather than the physical process count. See docs/debugging.md.

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_debugmodel"}
COMM_BACKEND=${COMM_BACKEND:-""}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

if [[ -n "$COMM_BACKEND" && "$COMM_BACKEND" != "fake" ]]; then
    echo "COMM_BACKEND must be empty or fake, got: ${COMM_BACKEND}" >&2
    exit 1
fi

if [ "$COMM_BACKEND" = "fake" ]; then
    echo "Running with fake process groups"
    # Tyro config modifiers in "$@" must remain last, so fixed global options
    # have to precede the caller-provided arguments.
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m torchtitan.train --module ${MODULE} --config ${CONFIG} --comm.backend=fake --training.steps 1 "$@"
else
    # Normal training with torchrun
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@"
fi
