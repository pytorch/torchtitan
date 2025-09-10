#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex
set -o pipefail

# Common settings
NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}


run_tt() {
    echo "##############################################"
    echo "### Running TorchTitan (native) training ###"
    echo "##############################################"
    TT_CONFIG="/fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/torchtitan/models/llama3/train_configs/my_debug_model.toml"

    # Use CUDA_VISIBLE_DEVICES=0 for TT run
    CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node=${NGPU} --master_port 1234 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.train --job.config_file ${TT_CONFIG} "$@"
}

run_hf() {
    echo "#######################################################"
    echo "### Running TorchTitan with HF backend training ###"
    echo "#######################################################"
    HF_CONFIG="/fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/torchtitan/experiments/transformers_backend/configs/debug_1_gpu_hf.toml"

    # Use CUDA_VISIBLE_DEVICES=1 for HF run
    CUDA_VISIBLE_DEVICES=1 \
    torchrun --nproc_per_node=${NGPU} --master_port 1235 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.train --job.config_file ${HF_CONFIG} "$@"
}


TT_LOG="tt_run.log"
HF_LOG="hf_run.log"
DIFF_LOG="run_diff.log"

run_tt "$@" 2>&1 | tee ${TT_LOG}
run_hf "$@" 2>&1 | tee ${HF_LOG}

# Filter logs to remove noisy differences
TT_LOG_FILTERED="${TT_LOG}.filtered"
HF_LOG_FILTERED="${HF_LOG}.filtered"

# This sed command removes timestamps, PIDs, master ports, and other
# volatile details that change between runs.
# Feel free to adjust the regex patterns to better suit your log format.
sed -E \
    -e 's/([0-9]{4}-[0-9]{2}-[0-9]{2} )?[0-9]{2}:[0-9]{2}:[0-9]{2}(,[0-9]+)?/TIMESTAMP/g' \
    -e 's/torchrun.*--master_port[= ]([0-9]+)/torchrun ... --master_port=XXXX/g' \
    -e 's/PID [0-9]+/PID XXXX/g' \
    -e 's/localhost:[0-9]+/localhost:XXXX/g' \
    < "${TT_LOG}" > "${TT_LOG_FILTERED}"

sed -E \
    -e 's/([0-9]{4}-[0-9]{2}-[0-9]{2} )?[0-9]{2}:[0-9]{2}:[0-9]{2}(,[0-9]+)?/TIMESTAMP/g' \
    -e 's/torchrun.*--master_port[= ]([0-9]+)/torchrun ... --master_port=XXXX/g' \
    -e 's/PID [0-9]+/PID XXXX/g' \
    -e 's/localhost:[0-9]+/localhost:XXXX/g' \
    < "${HF_LOG}" > "${HF_LOG_FILTERED}"

echo "############################################"
echo "### Diff between TT and HF run logs      ###"
echo "############################################"
echo "### Log diff is being saved to ${DIFF_LOG}"
echo "############################################"
git diff --no-index --color=always --word-diff=color "${TT_LOG_FILTERED}" "${HF_LOG_FILTERED}" | tee "${DIFF_LOG}" || true
