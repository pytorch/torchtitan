#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Autoresearch benchmark: Llama3 8B FSDP=4 TP=2 with 8192 tokens per DP rank on 8xH100.
# Output goes to ./run.log with wall time appended.
# Extra arguments are forwarded to run_train.sh.

set -e

start_time=$(date +%s)

NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b_autoresearch ./run_train.sh \
    "$@" \
    > run.log 2>&1

elapsed=$(($(date +%s) - start_time))
echo "benchmark_wall_time_s=${elapsed}" | tee -a run.log
