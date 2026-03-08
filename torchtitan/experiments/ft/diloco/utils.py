# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import generate_llm_fqn_per_model_part
from torchtitan.experiments.ft.config import FaultTolerance as FTConfig
from torchtitan.tools.logging import logger


def module_split(
    model: nn.Module,
    module_fqns_per_model_fragment: list[list[str]],
) -> list[nn.Module]:
    """
    This API creates fragments based on specified module names for each fragment.
    This method updates the model in place.

    Args:
        model: The complete model to be split
        module_fqns_per_model_fragment: List of lists, where each inner list contains the module names
                               that should be included in that fragment. Module names should be
                               dot-separated paths. Examples:
                               - "tok_embeddings" for token embeddings
                               - "layers.0", "layers.1" for specific transformer layers
                               - "norm" for the final normalization layer
                               - "output" for the output projection layer

    Returns:
        List of model fragments

    Example usage:
        module_fqns_per_model_fragment = [
            ["tok_embeddings", "layers.0"],     # fragment 0: embeddings + first layer
            ["layers.1", "layers.2"],           # fragment 1: middle layers
            ["norm", "output"]                  # fragment 2: final norm + output
        ]
    """

    def _build_fragment_from_modules(
        fragment_idx: int, module_names: list[str]
    ) -> nn.Module:
        fragment_model = nn.Module()
        # Create a set of modules to keep for faster lookup
        modules_to_keep = set(module_names)
        print(f"fragment {fragment_idx}: Modules to keep: {modules_to_keep}")
        for module_name, module_value in model.named_children():
            # Handle layer-like structures (e.g., "layers.0", "layers.1")
            if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
                layers_to_keep = {
                    name.split(".", 1)[1]
                    for name in modules_to_keep
                    if name.startswith(f"{module_name}.")
                }

                if not layers_to_keep:
                    continue

                # Keep only specified layers
                if isinstance(module_value, nn.ModuleDict):
                    for layer_name in list(module_value.keys()):
                        if layer_name in layers_to_keep:
                            setattr(
                                fragment_model,
                                f"{module_name}.{layer_name}",
                                module_value[layer_name],
                            )
                else:
                    indices_to_keep = {
                        int(idx) for idx in layers_to_keep if idx.isdigit()
                    }
                    new_layers = nn.ModuleList(
                        [
                            layer
                            for i, layer in enumerate(module_value)
                            if i in indices_to_keep
                        ]
                    )
                    setattr(fragment_model, module_name, new_layers)

                continue

            # Handle simple module attributes (e.g., "linear", "norm")
            if module_name not in modules_to_keep:
                continue

            setattr(fragment_model, module_name, module_value)

        return fragment_model

    num_fragments = len(module_fqns_per_model_fragment)
    model_fragments = []

    for fragment_idx in range(num_fragments):
        module_names = module_fqns_per_model_fragment[fragment_idx]
        model_fragment = _build_fragment_from_modules(
            fragment_idx,
            module_names,
        )
        logger.info(
            f"building fragment_idx {fragment_idx} " f"with modules {module_names}"
        )
        model_fragments.append(model_fragment)

    return model_fragments


def fragment_llm(
    model: nn.Module,
    ft_config: FTConfig,
    n_layers: int,
) -> list[nn.Module]:
    assert ft_config.num_fragments > 0

    module_fqns_per_model_fragment = ft_config.module_fqns_per_model_fragment

    input_weight = 1  # Weight for tok_embeddings
    output_weight = 1  # Weight for norm + output layers

    if module_fqns_per_model_fragment == []:
        if ft_config.num_fragments == 1:
            logger.info("Created 1 model fragments")
            return [model]

        module_fqns_per_model_fragment = generate_llm_fqn_per_model_part(
            ft_config.num_fragments, n_layers, input_weight, output_weight
        )

    model_fragments = module_split(model, module_fqns_per_model_fragment)
    logger.info(f"Created {len(model_fragments)} model fragments")

    return model_fragments
