# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import copy

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader

from torchtitan.models.deepseek_v3 import deepseekv3_configs, DeepSeekV3Model
from torchtitan.models.deepseek_v3.model.state_dict_adapter import (
    DeepSeekV3StateDictAdapter,
)
from torchtitan.protocols.train_spec import TrainSpec

from .parallelize_deepseekv3 import parallelize_deepseekv3


def get_train_spec() -> TrainSpec:
    model_args = copy.deepcopy(deepseekv3_configs)

    default_args = DeepSeekV3Model.Config()
    for config, args in model_args.items():
        if "flex_attn" in config:
            continue

        args.attn_type = default_args.attn_type
        args.attn_mask_type = default_args.attn_mask_type

    return TrainSpec(
        model_configs=model_args,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
