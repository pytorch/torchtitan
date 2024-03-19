# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import Union

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed.device_mesh import DeviceMesh


def dist_max(x: Union[int, float], mesh: DeviceMesh) -> float:
    tensor = torch.tensor(x).cuda()
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.MAX.name, group=mesh)


def dist_mean(x: Union[int, float], mesh: DeviceMesh) -> float:
    tensor = torch.tensor(x).cuda()
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.AVG.name, group=mesh)


def get_num_params(model: torch.nn.Module, only_trainable: bool = False) -> int:
    """
    Get the total model params
    Args : only_trainable: whether to only count trainable params
    """
    param_list = list(model.parameters())
    if only_trainable:
        param_list = [p for p in param_list if p.requires_grad]
    # unique_params = {p.data_ptr(): p for p in param_list}.values()
    return sum(p.numel() for p in param_list)


def get_num_flop_per_token(num_params: int, model_config, seq_len) -> int:
    l, h, q, t = (
        model_config.n_layers,
        model_config.n_heads,
        model_config.dim // model_config.n_heads,
        seq_len,
    )
    flop_per_token = 6 * num_params + 12 * l * h * q * t
    return flop_per_token


@dataclass
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"


@dataclass
class Background:
    black = "\033[40m"
    red = "\033[41m"
    green = "\033[42m"
    yellow = "\033[43m"
    blue = "\033[44m"
    magenta = "\033[45m"
    cyan = "\033[46m"
    white = "\033[47m"
    reset = "\033[49m"


@dataclass
class Style:
    bright = "\033[1m"
    dim = "\033[2m"
    normal = "\033[22m"
    reset = "\033[0m"
