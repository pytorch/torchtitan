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
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.datasets.tokenizer.tiktoken import build_tiktoken_tokenizer
from torchtitan.models.eurolingua.args import GPT2LLMModelArgs
from torchtitan.models.eurolingua.model_factory import ModelFactory
from torchtitan.models.eurolingua.parallelize import parallelize_gpt2_llm
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec


gpt2_llm_configs = {
    "debugmodel": GPT2LLMModelArgs(model_config_path="/raid/s3/opengptx/max_lue/repositories/torchtitan/torchtitan/models/eurolingua/configs/model_specification.yaml"),
}


register_train_spec(
    TrainSpec(
        name="gpt2_llm",
        cls=ModelFactory.get_gpt2_model,
        config=gpt2_llm_configs,
        parallelize_fn=parallelize_gpt2_llm,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
