# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.deepseek_v3 import deepseekv3_args, DeepSeekV3StateDictAdapter
from torchtitan.protocols.train_spec import TrainSpec

from .model import DeepEPDeepSeekV3Model
from .parallelize import parallelize_deepseekv3


def get_train_spec() -> TrainSpec:
    """
    Get the training specification for DeepSeek-V3 with DeepEP.
    
    Returns:
        TrainSpec: Complete training specification including model, parallelization,
                   optimization, and data loading functions.
    """
    return TrainSpec(
        model_cls=DeepEPDeepSeekV3Model,
        model_args=deepseekv3_args,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )


__all__ = [
    "get_train_spec",
    "DeepEPDeepSeekV3Model",
    "parallelize_deepseekv3",
]

