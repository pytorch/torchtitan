#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_generate.sh
NGPU=${NGPU:-"1"}
LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./outputs/checkpoint/"}
PROMPT=${PROMPT:-""}

overrides=()
if [ $# -ne 0 ]; then
	for arg in "$@"; do
		# special case to handle prompt in quotes
		if [[ "$arg" == --prompt=* ]]; then
			PROMPT="${arg#--prompt=}"
            # check if file
            if [[ -f "$PROMPT" ]]; then
                PROMPT=$(<"$PROMPT")
            fi
		else
			# handle other args
			overrides+=("$arg")
		fi
	done
fi

set -x
torchrun --standalone \
	--nproc_per_node="${NGPU}" \
	--local-ranks-filter="${LOG_RANK}" \
	-m scripts.generate.test_generate \
	--config="${CONFIG_FILE}" \
	--checkpoint="${CHECKPOINT_DIR}" \
	--prompt="${PROMPT}" \
	"${overrides[@]}"
