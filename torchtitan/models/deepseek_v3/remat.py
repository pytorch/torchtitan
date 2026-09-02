# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.distributed.activation_checkpoint import RematAC


DEEPSEEK_V3_REMAT_SAVE_REGIONS = (
    "attention.wq",
    "attention.wkv_a",
    "attention.core",
    "attention.wo",
    "moe.router",
    "moe.routed_experts.inner_experts.w1",
    "moe.routed_experts.inner_experts.w3",
    "moe.routed_experts.inner_experts.w2",
    "moe.shared_experts.w1",
    "moe.shared_experts.w3",
    "moe.shared_experts.w2",
    "feed_forward.w1",
    "feed_forward.w3",
    "feed_forward.w2",
)


def deepseek_v3_remat_config() -> RematAC.Config:
    """Return the reviewed DeepSeek V3 remat save policy."""
    return RematAC.Config(save_regions=list(DEEPSEEK_V3_REMAT_SAVE_REGIONS))
