# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers, build_optimizers_with_moe_load_balancing
from torchtitan.components.validate import build_validator
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.models.llama3 import llama3_configs, pipeline_llama, Transformer
from torchtitan.models.llama3.model.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.models.deepseek_v3.model.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec
from torchtitan.models.deepseek_v3 import deepseekv3_configs, DeepSeekV3Model
from .parallelize_llama import parallelize_llama
from .parallelize_deepseekv3 import parallelize_deepseekv3


register_train_spec(
    TrainSpec(
        name="llama3_auto_parallel",
        model_cls=Transformer,
        model_args=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
)
register_train_spec(
    TrainSpec(
        name="deepseekv3_auto_parallel",
        model_cls=DeepSeekV3Model,
        model_args=deepseekv3_configs,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        state_dict_adapter=DeepSeekV3StateDictAdapter,
    )
)
