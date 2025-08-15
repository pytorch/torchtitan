#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <MODEL_TYPE> <EXP_NAME>"
    exit 1
fi

python evaluate_fewshot.py --model_args dtype=bf16,model_type_size=${MODEL_TYPE},exp_name=${EXP_NAME},compile_prefilling=False,reduce_generation_overhead=False \
                           --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,openbookqa \
                           --device cuda:0 \
                           --batch_size 8
                        #    --wandb_args project=torchtitan,entity=evaluation,name=${EXP_NAME}
