import torch
import functools
import torch.nn as nn
from torchtitan.tools.logging import logger
from transformers.utils import is_torch_deterministic
import lovely_tensors as lt; lt.monkey_patch()

def debug_structure_param(model: nn.Module):
    """Print a breakdown of model parameters by module structure."""
    logger.info("Model Structure Parameter Breakdown:")

    def _format_module(module: nn.Module, prefix: str = ""):
        for name, sub_module in module.named_children():
            sub_module_params = sum(p.numel() for p in sub_module.parameters())
            if sub_module_params > 0:
                logger.info(
                    f"{prefix}({name}): {sub_module.__class__.__name__} - {sub_module_params:,} params"
                )
                _format_module(sub_module, prefix + "  ")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{model.__class__.__name__} - {total_params:,} params")
    _format_module(model, "  ")

def seeded_init_decorator_for_test(seed):
    """
    Decorator that adds torch.manual_seed before every nn.init.trunc_normal_ call
    and prints layer weights after initialization.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_trunc_normal = nn.init.trunc_normal_
            
            def seeded_trunc_normal(*trunc_args, **trunc_kwargs):
                torch.manual_seed(seed)
                tensor = trunc_args[0]  # First argument is always the tensor
                result = original_trunc_normal(*trunc_args, **trunc_kwargs)
                return result
            
            try:
                nn.init.trunc_normal_ = seeded_trunc_normal
                return func(*args, **kwargs)
            finally:
                nn.init.trunc_normal_ = original_trunc_normal
        
        return wrapper
    return decorator