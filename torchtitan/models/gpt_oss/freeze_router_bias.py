"""
Helper to freeze router gate bias for fine-tuning.
Use this when fine-tuning with small datasets to preserve pretrained routing.
"""

import torch.nn as nn


def freeze_router_gate_bias(model: nn.Module) -> int:
    """
    Freeze router.gate.bias parameters while keeping router.gate.weight trainable.

    Args:
        model: The model to modify

    Returns:
        Number of parameters frozen
    """
    frozen_count = 0

    for name, param in model.named_parameters():
        if 'router.gate.bias' in name:
            param.requires_grad = False
            frozen_count += 1

    return frozen_count


def print_trainable_summary(model: nn.Module):
    """Print summary of trainable vs frozen parameters."""
    total = 0
    trainable = 0
    frozen_router_bias = 0
    trainable_router_weight = 0

    for name, param in model.named_parameters():
        total += 1
        if param.requires_grad:
            trainable += 1
            if 'router.gate.weight' in name:
                trainable_router_weight += 1
        else:
            if 'router.gate.bias' in name:
                frozen_router_bias += 1

    print(f"Parameter Summary:")
    print(f"  Total: {total}")
    print(f"  Trainable: {trainable}")
    print(f"  Frozen: {total - trainable}")
    print(f"\nRouter-specific:")
    print(f"  router.gate.weight: {trainable_router_weight} layers (trainable)")
    print(f"  router.gate.bias: {frozen_router_bias} layers (frozen)")
