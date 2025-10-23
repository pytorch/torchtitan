# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.hf_datasets.dataloader import build_dataloader
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_qwen3next
from .model.args import Qwen3NextModelArgs
from .model.model import Qwen3NextModel
from .model.state_dict_adapter import Qwen3NextStateDictAdapter

__all__ = [
    "parallelize_qwen3next",
    "Qwen3NextModelArgs",
    "Qwen3NextModel",
    "qwen3next_configs",
]

# Adding different variants of the model

qwen3next_configs = {
    "80B_A3B": Qwen3NextModelArgs(
        full_attention_interval=4,
        moe_enabled=True,
        moe_inter_dim=512,
        moe_args=MoEArgs(
            num_experts=512,
            num_shared_experts=1,
            top_k=10,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
            shared_gate=True
        )
    ),
}

def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Qwen3NextModel,
        model_args=qwen3next_configs,  # Change from dict to Mapping
        parallelize_fn=parallelize_qwen3next,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3NextStateDictAdapter,
    )