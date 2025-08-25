# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.tools.logging import logger


def remove_orig_mod_and_weight_for_p_name(name: str) -> str:
    """
    Remove "._orig_mod", ".weight", and "._checkpoint_wrapped_module" to
    get the clean layer name.
    """
    name = re.sub(r"\._orig_mod", "", name)  # comes from compiled model
    name = re.sub(r"\.weight", "", name)  # param.weight
    name = re.sub(
        r"\._checkpoint_wrapped_module", "", name
    )  # comes from activation checkpointing
    return name


def create_disco_optimizer_kwargs_from_optimizer_config(
    optimizer_config,
    parallel_dims,
) -> dict[str, Any]:
    backend_steps = optimizer_config.backend_steps
    zeropower_backend_algorithm = optimizer_config.zeropower_backend
    momentum = optimizer_config.momentum
    nesterov = optimizer_config.nesterov
    is_light = optimizer_config.is_light
    weight_decay = optimizer_config.weight_decay
    lr = optimizer_config.lr
    eps = optimizer_config.eps
    norm_factor = optimizer_config.norm_factor

    optimizer_kwargs = {
        "parallel_dims": parallel_dims,
        "is_light": is_light,
        "weight_decay": weight_decay,
        "lr": lr,
        "momentum": momentum,
        "nesterov": nesterov,
        "eps": eps,
        "norm_factor": norm_factor,
        "backend": zeropower_backend_algorithm,
        "backend_steps": backend_steps,
    }

    # Add extra_param_group_split_rules if present
    if (
        hasattr(optimizer_config, "extra_param_group_split_rules")
        and optimizer_config.extra_param_group_split_rules
    ):
        optimizer_kwargs[
            "extra_param_group_split_rules"
        ] = optimizer_config.extra_param_group_split_rules

    return optimizer_kwargs


def create_disco_param_groups(
    model: torch.nn.Module,
    optimizer_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Create and extract parameter groups for DiSCO optimizer.

    This function combines parameter group configuration creation and parameter extraction:
    1. Creates parameter group configurations from optimizer_kwargs
    2. Extracts actual parameters from the model based on the configurations
    3. Returns clean kwargs for the optimizer (without extra_param_group_split_rules)

    This function supports a new parameter group configuration system where:
    - Default values are taken from the top-level optimizer_kwargs
    - Special parameter groups are defined in the 'extra_param_group_split_rules' list
    - Each entry in extra_param_group_split_rules can override any default value

    Example configuration:
    ```python
    optimizer_kwargs = {
        "lr": 1e-3,
        "weight_decay": 0.1,
        "momentum": 0.9,
        "norm_factor": "spectral",
        "backend": "newtonschulz5",
        "backend_steps": 5,
        "extra_param_group_split_rules": [
            {
                "str_match": "embedding",
                "lr": 1e-4,  # Override default lr
                "norm_factor": "embed_sqrt"  # Override default norm_factor
            },
            {
                "str_match": "router",
                "lr": 5e-4,
                "backend": "identity"  # Override default backend
            }
        ]
    }
    ```
    Args:
        model: The model to extract parameters from
        optimizer_kwargs: Dictionary containing optimizer configuration

    Returns:
        Tuple of (parameter_groups, clean_kwargs) where:
        - parameter_groups: List of parameter groups with actual parameters
        - clean_kwargs: Clean kwargs for the optimizer (without extra_param_group_split_rules)
    """
    import functools
    import re
    from collections import OrderedDict

    # Step 1: Create parameter group configurations
    param_groups_config = []

    # Get default configuration
    default_config = {
        "lr": optimizer_kwargs.get("lr"),
        "weight_decay": optimizer_kwargs.get("weight_decay"),
        "momentum": optimizer_kwargs.get("momentum"),
        "nesterov": optimizer_kwargs.get("nesterov"),
        "eps": optimizer_kwargs.get("eps"),
        "norm_factor": optimizer_kwargs.get("norm_factor"),
        "backend": optimizer_kwargs.get("backend"),
        "backend_steps": optimizer_kwargs.get("backend_steps"),
    }

    # Process extra_param_group_split_rules if provided
    extra_param_group_split_rules = optimizer_kwargs.get(
        "extra_param_group_split_rules", []
    )
    for param_group in extra_param_group_split_rules:
        # Start with default config and override with param_group specific values
        group_config = default_config.copy()
        group_config.update(param_group)

        # Ensure str_match is present
        if "str_match" not in group_config:
            logger.warning(
                "extra_param_group_split_rules entry missing 'str_match', skipping"
            )
            continue

        # Rename str_match to param_str_match for compatibility
        group_config["param_str_match"] = group_config.pop("str_match")

        param_groups_config.append(group_config)

    # Detect RMSNorm parameters and set requires_grad=False
    for name, module in model.named_modules():
        # Check if the module is an RMSNorm layer
        if hasattr(module, "__class__") and (
            "RMSNorm" in module.__class__.__name__
            or "LayerNorm" in module.__class__.__name__
        ):
            for param_name, param in module.named_parameters():
                param.requires_grad = False
                logger.info(
                    f"Setting requires_grad=False for LayerNorm/RMSNorm parameter: {name}.{param_name}"
                )

    # Step 2: Extract actual parameters from the model
    param_dict = OrderedDict(
        (n, p) for n, p in model.named_parameters() if p.requires_grad
    )
    params = []

    for param_group_config in param_groups_config:
        # Make a copy to avoid modifying the original
        group_config = param_group_config.copy()
        str_match = group_config.pop("param_str_match")
        filter_fn = functools.partial(re.search, str_match)
        param_names = [n for n in param_dict.keys() if filter_fn(n)]

        group_params = {
            "params": [param_dict.pop(n) for n in param_names],
            "param_names": param_names,
        }
        assert len(group_params["params"]) == len(group_params["param_names"])

        if len(param_names) == 0:
            try:
                rank = dist.get_rank() if dist.is_initialized() else 0
            except Exception:
                rank = 0
            logger.warning(
                f'Notice: No parameters found for `str_match` "{str_match}" on '
                f"global rank {rank}"
            )
            continue
        group_params.update(group_config)
        params.append(group_params)

    # Add remaining parameters as the default group
    param_names = list(param_dict.keys())
    if param_names:
        default_group = {
            "params": [param_dict.pop(n) for n in param_names],
            "param_names": param_names,
        }
        # Add default configuration to the default group
        default_group.update(default_config)
        params.insert(0, default_group)

    # Create clean kwargs for the optimizer (remove extra_param_group_split_rules)
    clean_kwargs = optimizer_kwargs.copy()
    clean_kwargs.pop("extra_param_group_split_rules", None)

    return params, clean_kwargs
