# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is used to setup the model for fault tolerance

import torch.nn as nn


from torchtitan.config_manager import JobConfig
from torchtitan.distributed.pipeline import (
    generate_module_names_per_stage,
    module_split,
)
from torchtitan.tools.logging import logger

from ..model.args import TransformerModelArgs

def fragment_llama(
    model: nn.Module,
    job_config: JobConfig,
    model_config: TransformerModelArgs,
) -> list[nn.Module]:
    ft = job_config.fault_tolerance

    assert ft.num_fragments > 0

    module_names_per_stage = ft.module_names_per_model_chunk

    input_weight = 1  # Weight for tok_embeddings
    output_weight = 1  # Weight for norm + output layers

    if module_names_per_stage == []:
        if ft.num_fragments == 1:
            return [model]

        module_names_per_stage = generate_module_names_per_stage(
            ft.num_fragments, model_config.n_layers, input_weight, output_weight
        )


    model_fragments = module_split(model, module_names_per_stage)
    print(f"Created {len(model_fragments)} model fragments")

    return model_fragments
