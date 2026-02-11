# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from dataclasses import dataclass

from autoparallel._testing.models.dsv3 import (
    DeepSeekV3ModelArgs as _DeepSeekV3ModelArgs,
    MoEArgs as _MoEArgs,
)
from torchtitan.protocols.model import BaseModel


# Need to share same base class with torchtitan models
@dataclass
class DeepSeekV3ModelArgs(_DeepSeekV3ModelArgs, BaseModel.Config):
    pass


def get_sample_config() -> DeepSeekV3ModelArgs:
    return DeepSeekV3ModelArgs(
        vocab_size=2048,
        max_seq_len=2048,
        dim=256,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=4,
        n_dense_layers=0,
        n_heads=16,
        moe_args=_MoEArgs(
            num_experts=4,
            num_shared_experts=2,
            top_k=2,
            score_func="softmax",
            route_norm=False,
            score_before_experts=False,
            mesh=None,
        ),
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.70,
    )
