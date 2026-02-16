# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.experiments.ft.diloco import fragment_llm
from torchtitan.models.llama3 import (
    llama3_configs,
    Llama3StateDictAdapter,
    parallelize_llama,
)
from torchtitan.protocols.model_spec import FaultTolerantModelSpec


def model_registry(flavor: str) -> FaultTolerantModelSpec:
    return FaultTolerantModelSpec(
        name="ft/llama3",
        flavor=flavor,
        model=llama3_configs[flavor],
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
        fragment_fn=fragment_llm,
    )
