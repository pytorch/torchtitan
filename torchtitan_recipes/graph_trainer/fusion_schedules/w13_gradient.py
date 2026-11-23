# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Interleave the gate and up weight gradients for the 256 FSDP shards."""


def kernel_function(grad_RD):
    # R is the combined gate/up row count and D is the input dimension.
    shards = grad_RD.view(2, 256, -1, grad_RD.shape[1]).transpose(0, 1).contiguous()
    return (shards.flatten(0, 1),)
