# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Parameter classification utilities for Dion optimizer integration.
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from torchtitan.tools.logging import logger


def create_parameter_groups(
    model_parts: List[nn.Module], dion_config
) -> List[Dict[str, Any]]:
    """Create parameter groups with parameter classification.

    Classification rules:
    1. 1D scalar parameters (biases, layer norms) → Use scalar_optimizer (adamw/lion)
    2. Embedding layers → Use embedding_optimizer (adamw/lion)
    3. Model head/output layers → Use head_optimizer (adamw/lion) with optional 1/sqrt(dim) scaling
    4. Routing layers (DeepSeek MoE) → Use routing_optimizer (adamw/lion)
    5. 2D matrix parameters (not head/embedding/routing) → Use Dion algorithm
    """
    param_groups = []

    for model in model_parts:
        # Group parameters by type
        dion_params = []
        scalar_params = []
        embedding_params = []
        head_params = []
        routing_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Classify parameter based on shape and module type
            if is_routing_param(name, param, model):
                logger.info(f"Dion optimizer found routing layer parameter: {name}")
                routing_params.append(param)
            elif is_embedding_param(name, param, model):
                logger.info(f"Dion optimizer found embedding parameter: {name}")
                embedding_params.append(param)
            elif is_head_param(name, param, model):
                logger.info(f"Dion optimizer found LM head parameter: {name}")
                head_params.append(param)
            elif param.ndim == 1:  # 1D scalar parameters (biases, layer norms)
                scalar_params.append(param)
            elif param.ndim == 2:  # 2D matrix parameters
                dion_params.append(param)
            else:
                # For higher dimensional parameters, treat as scalar
                scalar_params.append(param)

        # Create parameter groups for each type
        if dion_params:
            param_groups.append(create_dion_param_group(dion_params, dion_config))

        if scalar_params:
            param_groups.append(create_scalar_param_group(scalar_params, dion_config))

        if embedding_params:
            param_groups.append(
                create_embedding_param_group(embedding_params, dion_config)
            )

        if head_params:
            param_groups.append(create_head_param_group(head_params, dion_config))

        if routing_params:
            param_groups.append(create_routing_param_group(routing_params, dion_config))

    # Log summary of parameter classification
    total_routing_params = sum(
        len(group["params"])
        for group in param_groups
        if group.get("algorithm") == getattr(dion_config, "routing_optimizer", "adamw")
    )
    if total_routing_params > 0:
        logger.info(
            f"Dion optimizer: Found {total_routing_params} routing layer parameters that will use AdamW instead of Dion"
        )

    return param_groups


def is_embedding_param(name: str, param: torch.Tensor, model: nn.Module) -> bool:
    """Check if parameter belongs to an embedding layer."""
    # Check for common embedding layer names
    embedding_patterns = [
        "embed",
        "embedding",
        "tok_embeddings",
        "word_embeddings",
        "position_embeddings",
        "pos_embed",
    ]

    name_lower = name.lower()
    for pattern in embedding_patterns:
        if pattern in name_lower:
            return True

    return False


def is_routing_param(name: str, param: torch.Tensor, model: nn.Module) -> bool:
    """Check if parameter belongs to a routing layer (for DeepSeek MoE models)."""
    # Check for routing layer patterns specific to DeepSeek v3
    routing_patterns = [
        "router.gate",  # DeepSeek v3 routing layer (exact match)
        "gate.weight",  # Alternative routing layer pattern
        "router_gate",  # Alternative pattern
        "routing_gate",  # Alternative pattern
        ".router.",  # Any parameter containing .router.
        "moe.router",  # MoE router pattern
    ]

    name_lower = name.lower()
    for pattern in routing_patterns:
        if pattern in name_lower:
            return True

    return False


def is_head_param(name: str, param: torch.Tensor, model: nn.Module) -> bool:
    """Check if parameter belongs to a model head/output layer."""
    # Check for common head layer names
    head_patterns = [
        "head",
        "output",
        "classifier",
        "lm_head",
        "prediction_head",
        "final_layer",
        "out_proj",
        "output_projection",
    ]

    name_lower = name.lower()
    for pattern in head_patterns:
        if pattern in name_lower:
            return True

    return False


def create_dion_param_group(params: List[torch.Tensor], dion_config) -> Dict[str, Any]:
    """Create parameter group for Dion algorithm (2D matrix parameters)."""
    return {
        "params": params,
        "algorithm": dion_config.algorithm,
        "rank_fraction": dion_config.rank_fraction,
        "rank_multiple_of": dion_config.rank_multiple_of,
        "lr": dion_config.lr,
        "mu": dion_config.mu,
        "beta1": dion_config.betas[0],
        "beta2": dion_config.betas[1],
        "weight_decay": dion_config.weight_decay,
        "epsilon": dion_config.epsilon,
    }


def create_scalar_param_group(
    params: List[torch.Tensor], dion_config
) -> Dict[str, Any]:
    """Create parameter group for scalar parameters (1D biases, layer norms)."""
    # Get scalar optimizer from config if available, otherwise use default
    scalar_optimizer = getattr(dion_config, "scalar_optimizer", "adamw")
    scalar_lr_factor = getattr(dion_config, "scalar_lr_factor", 1.0)

    return {
        "params": params,
        "algorithm": scalar_optimizer,
        "lr": dion_config.lr * scalar_lr_factor,
        "beta1": dion_config.betas[0],
        "beta2": dion_config.betas[1],
        "weight_decay": dion_config.weight_decay,
        "epsilon": dion_config.epsilon,
    }


def create_embedding_param_group(
    params: List[torch.Tensor], dion_config
) -> Dict[str, Any]:
    """Create parameter group for embedding parameters."""
    # Get embedding optimizer from config if available, otherwise use default
    embedding_optimizer = getattr(dion_config, "embedding_optimizer", "adamw")
    embedding_lr_factor = getattr(dion_config, "embedding_lr_factor", 1.0)

    return {
        "params": params,
        "algorithm": embedding_optimizer,
        "lr": dion_config.lr * embedding_lr_factor,
        "beta1": dion_config.betas[0],
        "beta2": dion_config.betas[1],
        "weight_decay": dion_config.weight_decay,
        "epsilon": dion_config.epsilon,
    }


def create_head_param_group(params: List[torch.Tensor], dion_config) -> Dict[str, Any]:
    """Create parameter group for head/output parameters with optional 1/sqrt(dim) scaling."""
    # Get head optimizer from config if available, otherwise use default
    head_optimizer = getattr(dion_config, "head_optimizer", "adamw")
    head_lr_factor = getattr(dion_config, "head_lr_factor", 1.0)
    head_lr_scaling = getattr(dion_config, "head_lr_scaling", True)

    # Calculate learning rate with 1/sqrt(dim) scaling
    # TODO - double check this
    lr = dion_config.lr * head_lr_factor

    if head_lr_scaling and params:
        # Use the first parameter to determine the dimension for scaling

        first_param = params[0]
        if first_param.ndim >= 2:
            dim = first_param.shape[-1]  # Input dimension
            lr = lr / (dim**0.5)  # 1/sqrt(dim) scaling

    return {
        "params": params,
        "algorithm": head_optimizer,
        "lr": lr,
        "beta1": dion_config.betas[0],
        "beta2": dion_config.betas[1],
        "weight_decay": dion_config.weight_decay,
        "epsilon": dion_config.epsilon,
    }


def create_routing_param_group(
    params: List[torch.Tensor], dion_config
) -> Dict[str, Any]:
    """Create parameter group for routing parameters (DeepSeek MoE routing layers)."""
    # Get routing optimizer from config if available, otherwise use adamw
    routing_optimizer = getattr(dion_config, "routing_optimizer", "adamw")
    routing_lr_factor = getattr(dion_config, "routing_lr_factor", 1.0)

    return {
        "params": params,
        "algorithm": routing_optimizer,
        "lr": dion_config.lr * routing_lr_factor,
        "beta1": dion_config.betas[0],
        "beta2": dion_config.betas[1],
        "weight_decay": dion_config.weight_decay,
        "epsilon": dion_config.epsilon,
    }
