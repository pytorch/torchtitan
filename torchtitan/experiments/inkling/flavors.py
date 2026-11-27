# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.inkling.experts import InklingMoE
from torchtitan.experiments.inkling.model import (
    InklingAttention,
    InklingLayer,
    InklingModel,
    InklingShape,
)
from torchtitan.models.common.linear import Linear

MODEL_FLAVORS = {
    "2-mini": InklingShape(
        hidden_size=2304,
        num_layers=24,
        num_q_heads=24,
        num_kv_heads=8,
        head_dim=128,
        d_rel=16,
        rel_extent=1024,
        sliding_window=1024,
        full_layers=(5, 11, 17, 23),
        num_experts=128,
        experts_per_token=8,
        intermediate_size=1152,
        padded_vocab_size=201024,
        vocab_size=200058,
        block_layers=6,
        loop_start=6,
        loop_end=18,
        num_passes=3,
        gate_rank=64,
        nvfp4_layers=20,
    ),
    "debugmodel": InklingShape(
        hidden_size=32,
        num_layers=4,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=8,
        d_rel=4,
        rel_extent=8,
        sliding_window=4,
        full_layers=(1, 3),
        num_experts=4,
        experts_per_token=2,
        intermediate_size=32,
        padded_vocab_size=128,
        vocab_size=120,
        block_layers=1,
        loop_start=1,
        loop_end=3,
        num_passes=2,
        gate_rank=4,
        nvfp4_layers=3,
    ),
}


def build_model_config(flavor: str) -> InklingModel.Config:
    s = MODEL_FLAVORS[flavor]

    def linear(width, out):
        return Linear.Config(
            in_features=width, out_features=out, inplace_wgrad_accum=False
        )

    dim, q_width, kv_width = (
        s.hidden_size,
        s.num_q_heads * s.head_dim,
        s.num_kv_heads * s.head_dim,
    )
    layers = [
        InklingLayer.Config(
            attn=InklingAttention.Config(
                wq_du=linear(dim, q_width),
                wk_dv=linear(dim, kv_width),
                wv_dv=linear(dim, kv_width),
                wo_ud=linear(q_width, dim),
                wr_du=linear(dim, s.num_q_heads * s.d_rel),
                gate_attn=linear(dim, q_width),
                attn_sink_proj=linear(dim, s.num_q_heads),
            ),
            mlp=InklingMoE.Config(
                hidden_size=dim,
                intermediate_size=s.intermediate_size,
                num_experts=s.num_experts,
                experts_per_token=s.experts_per_token,
                route_scale=8,
                gate_softcap=4,
                up_softcap=25,
                expert_format="nvfp4" if i < s.nvfp4_layers else "mxfp8",
            ),
        )
        for i in range(s.num_layers)
    ]
    return InklingModel.Config(
        shape=s,
        layers=layers,
        unembed=linear(dim, s.padded_vocab_size),
        max_context_length=1024 if flavor == "2-mini" else 16,
    )
