# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn

def get_num_params(model: nn.Module, only_trainable: bool = False)-> int:
    """ Get the total model params
    args: only_trainable - only count trainable params
    """

    param_list = list(model.parameters())
    if only_trainable:
        param_list = [p for p in param_list if p.requires_grad]
    unique_params = {p.data_ptr(): p for p in param_list}.values()
    return sum(p.numel() for p in unique_params)
