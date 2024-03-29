# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from contextlib import contextmanager

import torch
from torch import nn
from torch.distributed.fsdp._common_utils import _is_fsdp_flattened


@contextmanager
def meta_model_init():
    """init model on meta device"""
    saved_register_parameter = nn.Module.register_parameter
    saved_register_buffer = nn.Module.register_buffer

    def register_meta_param(module, name, param):
        saved_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(
                module._parameters[name].to(torch.device("meta")), **kwargs
            )

    def register_meta_buffer(module, name, buffer):
        saved_register_buffer(module, name, buffer)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(torch.device("meta"))

    try:
        nn.Module.register_parameter = register_meta_param
        nn.Module.register_buffer = register_meta_buffer
        yield
    finally:
        nn.Module.register_parameter = saved_register_parameter
        nn.Module.register_buffer = saved_register_buffer


@torch.no_grad()
def meta_to_real_init_fn(module: nn.Module):
    for submodule in module.modules():
        for param_name, param in submodule.named_parameters(recurse=False):
            if not _is_fsdp_flattened(param) and param.is_meta:
                materialized_param = nn.Parameter(
                    torch.randn_like(param, device=torch.device("cuda"))
                )
                setattr(submodule, param_name, materialized_param)
        for param_name, param in submodule.named_buffers(recurse=False):
            if param.is_meta:
                materialized_param = nn.Parameter(
                    torch.randn_like(param, device=torch.device("cuda"))
                )
                setattr(submodule, param_name, materialized_param)
