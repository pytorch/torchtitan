# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

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
