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
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_qwen2
from .model.args import Qwen2ModelArgs
from .model.model import Qwen2Model
from .model.state_dict_adapter import Qwen2StateDictAdapter

__all__ = [
    "parallelize_qwen2",
    "Qwen2ModelArgs",
    "Qwen2Model",
    "qwen2_configs",
]

# Adding different variants of the model

qwen2_configs = {
    "7B": Qwen2ModelArgs(
        vocab_size=152064,
        max_seq_len=131072,
        head_dim=128,
        dim=3584,
        hidden_dim=18944,
        n_layers=28,
        n_heads=28,
        n_kv_heads=4,
        qkv_bias=True,
        rope_theta=1000000,
        enable_weight_tying=False,
    ),
}


register_train_spec(
    TrainSpec(
        name="qwen2",
        model_cls=Qwen2Model,
        model_args=qwen2_configs,  # Change from dict to Mapping
        parallelize_fn=parallelize_qwen2,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen2StateDictAdapter,
    )
)
