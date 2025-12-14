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
    5. Expert weights (MoE experts) → Follow 2D matrix classification
    6. 2D matrix parameters (not head/embedding/routing) → Use Dion/Muon algorithm
    """
    param_groups = []

    # Track parameter statistics for logging
    param_stats = {
        "dion": [],
        "scalar": [],
        "embedding": [],
        "head": [],
        "routing": [],
        "expert": [],
    }

    for model in model_parts:
        # Group parameters by type
        dion_params = []
        scalar_params = []
        embedding_params = []
        head_params = []
        routing_params = []
        expert_params = (
            []
        )  # Separate list for expert parameters when expert_optimizer is set

        # Expert weights can use either DION (for 2D matrices only) or a dedicated expert optimizer
        expert_optimizer = getattr(dion_config, "expert_optimizer", None)
        routing_optimizer = getattr(dion_config, "routing_optimizer", None)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Add parameter name as attribute for debug logging
            param._param_name = name

            # Check if this is an expert weight parameter
            if is_expert_param(name, param, model):
                expert_type = classify_expert_param(name, param)
                param_stats["expert"].append((name, param.shape, expert_type))

                if expert_optimizer is not None:
                    # Use dedicated expert optimizer if specified
                    expert_params.append(param)
                elif param.ndim >= 2 and not is_head_param(name, param, model):
                    # Use matrix algorithm for expert matrices
                    # Dion only supports 2D, but Muon supports 3D+ (with flattening)
                    algorithm_name = dion_config.algorithm.upper()

                    # Check if flatten is enabled (for Muon 3D+ tensor support)
                    flatten_enabled = getattr(dion_config, "flatten", False)

                    # Handle case-insensitive algorithm matching
                    is_muon = (
                        algorithm_name in ["MUON", "muon"]
                        or dion_config.algorithm.lower() == "muon"
                    )

                    # Simplified condition: Use matrix algorithm for any 2D+ tensors when algorithm is muon
                    if is_muon and param.ndim >= 2:
                        # Use Muon for 2D+ tensors when algorithm is MUON, or always for 2D tensors
                        dion_params.append(param)
                    else:
                        # Dion doesn't support 3D+, fall back to scalar optimizer
                        scalar_params.append(param)
                else:
                    # Fall back to scalar optimizer for 1D expert parameters
                    scalar_params.append(param)
                continue

            # Classify parameter based on shape and module type
            if is_routing_param(name, param, model):
                param_stats["routing"].append((name, param.shape))
                if routing_optimizer is not None:
                    routing_params.append(param)
                else:
                    dion_params.append(param)
            elif is_embedding_param(name, param, model):
                param_stats["embedding"].append((name, param.shape))
                embedding_params.append(param)
            elif is_head_param(name, param, model):
                param_stats["head"].append((name, param.shape))
                head_params.append(param)
            elif param.ndim == 1:  # 1D scalar parameters (biases, layer norms)
                param_stats["scalar"].append((name, param.shape))
                scalar_params.append(param)
            elif param.ndim == 2:  # 2D matrix parameters
                param_stats["dion"].append((name, param.shape))
                dion_params.append(param)
            else:
                # For higher dimensional parameters, treat as scalar
                param_stats["scalar"].append((name, param.shape))
                scalar_params.append(param)

        # Create parameter groups for each type
        if dion_params:
            param_groups.append(create_matrix_param_group(dion_params, dion_config))
            logger.info(f"dion lr: {param_groups[-1]['lr']}")

        if scalar_params:
            param_groups.append(create_scalar_param_group(scalar_params, dion_config))
            logger.info(f"scalar lr: {param_groups[-1]['lr']}")

        if embedding_params:
            param_groups.append(
                create_embedding_param_group(embedding_params, dion_config)
            )
            logger.info(f"embedding lr: {param_groups[-1]['lr']}")

        if head_params:
            param_groups.append(create_head_param_group(head_params, dion_config))
            logger.info(f"head lr: {param_groups[-1]['lr']}")

        if routing_params:
            param_groups.append(create_routing_param_group(routing_params, dion_config))
            logger.info(f"routing lr: {param_groups[-1]['lr']}")

        if expert_params:
            param_groups.append(create_expert_param_group(expert_params, dion_config))
            logger.info(f"expert lr: {param_groups[-1]['lr']}")

    # Enhanced logging summary
    logger.info("=" * 80)
    logger.info("PARAMETER OPTIMIZATION SUMMARY")
    logger.info("=" * 80)

    # Get optimizer names from config
    scalar_opt = getattr(dion_config, "scalar_optimizer", "adamw").upper()
    embedding_opt = getattr(dion_config, "embedding_optimizer", "adamw").upper()
    head_opt = getattr(dion_config, "head_optimizer", "adamw").upper()
    routing_opt = getattr(dion_config, "routing_optimizer", None)

    algorithm_name = dion_config.algorithm.upper()
    logger.info(f"{algorithm_name} algorithm parameters: {len(param_stats['dion'])}")
    for name, shape in param_stats["dion"]:
        logger.info(f"  - {name}: {shape}")

    logger.info(f"Scalar parameters ({scalar_opt}): {len(param_stats['scalar'])}")

    logger.info(
        f"Embedding parameters ({embedding_opt}): {len(param_stats['embedding'])}"
    )
    for name, shape in param_stats["embedding"]:
        logger.info(f"  - {name}: {shape}")

    logger.info(f"Head parameters ({head_opt}): {len(param_stats['head'])}")
    for name, shape in param_stats["head"]:
        logger.info(f"  - {name}: {shape}")

    logger.info(f"Routing parameters ({routing_opt.upper() if routing_opt is not None else algorithm_name}): {len(param_stats['routing'])}")
    for name, shape in param_stats["routing"]:
        logger.info(f"  - {name}: {shape}")

    # Special focus on expert weights
    logger.info("=" * 40)
    logger.info("EXPERT WEIGHTS SUMMARY")
    logger.info("=" * 40)

    expert_optimizer = getattr(dion_config, "expert_optimizer", None)

    if param_stats["expert"]:
        logger.info(
            f"Total expert weight parameters found: {len(param_stats['expert'])}"
        )
        if expert_optimizer is not None:
            logger.info(f"Expert optimizer configured: {expert_optimizer.upper()}")
            for name, shape, expert_type in param_stats["expert"]:
                logger.info(
                    f"  ✓ EXPERT: {name} ({shape}) - {expert_type} → USING {expert_optimizer.upper()}"
                )
        else:
            logger.info(
                "Expert optimizer not configured - using default classification:"
            )
            for name, shape, expert_type in param_stats["expert"]:
                # Check if this expert parameter actually uses the matrix algorithm
                algorithm_name = dion_config.algorithm.upper()
                is_muon = (
                    algorithm_name in ["MUON", "muon"]
                    or dion_config.algorithm.lower() == "muon"
                )
                flatten_enabled = getattr(dion_config, "flatten", False)

                if (len(shape) == 2) or (is_muon and len(shape) >= 2):
                    logger.info(
                        f"  ✓ EXPERT: {name} ({shape}) - {expert_type} → USING {algorithm_name}"
                    )
                else:
                    logger.info(
                        f"  ✓ EXPERT: {name} ({shape}) - {expert_type} → USING {scalar_opt}"
                    )
    else:
        logger.info("No expert weight parameters detected in this model")

    logger.info("=" * 80)

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


def is_expert_param(name: str, param: torch.Tensor, model: nn.Module) -> bool:
    """Check if parameter belongs to an expert in a MoE (Mixture of Experts) layer."""
    # Check for expert layer patterns
    expert_patterns = [
        "experts.",  # Common expert layer pattern
        ".expert.",  # Alternative expert pattern
        "expert_",  # Expert prefix
        "moe.expert",  # MoE expert pattern
        "shared_experts",  # DeepSeek shared experts
        "routed_experts",  # DeepSeek routed experts
        ".experts[",  # Indexed expert pattern
        ".w1.",  # Expert feed-forward weights (common in MoE)
        ".w2.",  # Expert feed-forward weights (common in MoE)
        ".w3.",  # Expert feed-forward weights (common in MoE)
        "gate_proj",  # Expert gate projection (when inside expert block)
        "up_proj",  # Expert up projection (when inside expert block)
        "down_proj",  # Expert down projection (when inside expert block)
    ]

    name_lower = name.lower()

    # Check if the parameter name contains expert patterns
    for pattern in expert_patterns:
        if pattern in name_lower:
            # Additional check: if it's a projection layer, make sure it's within an expert context
            if pattern in ["gate_proj", "up_proj", "down_proj", ".w1.", ".w2.", ".w3."]:
                # Only consider it an expert param if "expert" is also in the name
                if "expert" in name_lower:
                    return True
            else:
                return True

    return False


def classify_expert_param(name: str, param: torch.Tensor) -> str:
    """Classify the type of expert parameter for more detailed logging."""
    name_lower = name.lower()

    if "shared_expert" in name_lower:
        return "Shared Expert"
    elif "routed_expert" in name_lower:
        return "Routed Expert"
    elif ".w1." in name_lower or "gate_proj" in name_lower:
        return "Expert Gate Projection"
    elif ".w2." in name_lower or "down_proj" in name_lower:
        return "Expert Down Projection"
    elif ".w3." in name_lower or "up_proj" in name_lower:
        return "Expert Up Projection"
    elif "expert" in name_lower:
        return "Generic Expert"
    else:
        return "Unknown Expert Type"


def create_matrix_param_group(
    params: List[torch.Tensor], optimizer_config
) -> Dict[str, Any]:
    """Create parameter group for matrix algorithms (Dion/Muon) - 2D matrix parameters."""
    param_group = {
        "params": params,
        "algorithm": optimizer_config.algorithm,
        "lr": optimizer_config.lr,
        "mu": optimizer_config.mu,
        "beta1": optimizer_config.betas[0],
        "beta2": optimizer_config.betas[1],
        "weight_decay": optimizer_config.weight_decay,
        "epsilon": optimizer_config.epsilon,
    }

    # Add Dion-specific parameters if they exist
    if hasattr(optimizer_config, "rank_fraction"):
        param_group["rank_fraction"] = optimizer_config.rank_fraction
    if hasattr(optimizer_config, "rank_multiple_of"):
        param_group["rank_multiple_of"] = optimizer_config.rank_multiple_of
    if hasattr(optimizer_config, "power_iters"):
        param_group["power_iters"] = optimizer_config.power_iters
    if hasattr(optimizer_config, "qr_method"):
        param_group["qr_method"] = optimizer_config.qr_method
    if hasattr(optimizer_config, "cqr_warmup_steps"):
        param_group["cqr_warmup_steps"] = optimizer_config.cqr_warmup_steps
    if hasattr(optimizer_config, "rcqr_oversample"):
        param_group["rcqr_oversample"] = optimizer_config.rcqr_oversample

    # Add Muon-specific parameters if they exist
    if hasattr(optimizer_config, "nesterov"):
        param_group["nesterov"] = optimizer_config.nesterov
    if hasattr(optimizer_config, "adjust_lr"):
        param_group["adjust_lr"] = optimizer_config.adjust_lr
    if hasattr(optimizer_config, "flatten"):
        param_group["flatten"] = optimizer_config.flatten

    return param_group


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
    # Get routing optimizer from config if available, otherwise use the main optimizer
    # This allows routing parameters to inherit the main optimizer choice (e.g., Lion)
    main_optimizer = getattr(dion_config, "name", "adamw").lower()
    routing_optimizer = getattr(dion_config, "routing_optimizer", main_optimizer)
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


def create_expert_param_group(
    params: List[torch.Tensor], dion_config
) -> Dict[str, Any]:
    """Create parameter group for expert parameters when expert_optimizer is specified."""
    # Get expert optimizer from config (this should always be set when this function is called)
    expert_optimizer = getattr(dion_config, "expert_optimizer", "adamw")
    expert_lr_factor = getattr(dion_config, "expert_lr_factor", 1.0)

    return {
        "params": params,
        "algorithm": expert_optimizer,
        "lr": dion_config.lr * expert_lr_factor,
        "beta1": dion_config.betas[0],
        "beta2": dion_config.betas[1],
        "weight_decay": dion_config.weight_decay,
        "epsilon": dion_config.epsilon,
    }
