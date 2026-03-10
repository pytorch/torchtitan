# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm

from torchtitan.experiments.simple_fsdp.llama3 import (
    _simple_fsdp_configs as simple_fsdp_llama3_configs,
)
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .parallelize import parallelize_llama


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="compiler_toolkit/llama3",
        flavor=flavor,
        model=simple_fsdp_llama3_configs[flavor],
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )
