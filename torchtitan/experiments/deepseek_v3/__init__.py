# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.experiments.deepseek_v3.tokenizers.hf_tokenizer import get_hf_tokenizer

# ToDO - this is not suitable for deepseek but using for now...
from torchtitan.models.llama3 import pipeline_llama
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize_deepseek import parallelize_deepseek

from .model import DeepseekForCausalLM

from .model_args import TransformerModelArgs


__all__ = [
    "TransformerModelArgs",
    "DeepseekForCausalLM",
    "deepseek_configs",
]


deepseek_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        rope_theta=500000,
    ),
}


register_train_spec(
    TrainSpec(
        name="deepseek3",
        cls=DeepseekForCausalLM,
        config=deepseek_configs,
        parallelize_fn=parallelize_deepseek,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=get_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
