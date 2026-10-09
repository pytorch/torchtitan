# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A functools.cache'd tensor factory called during a fake-tensor trace (make_fx / GraphTrainer's
non-strict tracer) caches a FakeTensor; later eager calls get it back and fail. (Dynamo itself
ignores the cache wrapper and traces through, with a warning.) CPU only.

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, Known behaviors):
the eager call after the fake trace raises 'Please convert all Tensors to FakeTensors first' (CPU).
"""
from functools import cache

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


@cache
def hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


with FakeTensorMode():
    hadamard(8)
try:
    print((torch.randn(2, 8) @ hadamard(8)).shape)
except Exception as e:
    print(
        "eager call after a fake trace:",
        type(e).__name__,
        str(e).splitlines()[0].split(" or instantiate")[0],
    )
