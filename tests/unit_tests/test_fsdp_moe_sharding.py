# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.models.llama4.model import compute_moe_hidden_dim, Llama4Model
from torchtitan.models.llama4.parallelize import apply_fsdp


def _build_llama4_model(num_experts: int = 8) -> Llama4Model:
    """Build a tiny Llama4Model with a configurable number of experts."""
    from torchtitan.models.common import compute_ffn_hidden_dim

    # Use the standard debugmodel config but override num_experts.
    # Rebuild layers with the requested num_experts.
    from torchtitan.models.llama4 import _build_llama4_layers

    dim = 256
    n_heads = 16
    n_layers = 4
    moe_hidden_dim = compute_moe_hidden_dim(dim)

    from torchtitan.models.common.embedding import Embedding
    from torchtitan.models.common.linear import Linear
    from torchtitan.models.common.rmsnorm import RMSNorm
    from torchtitan.models.common.rope import RoPE

    config = Llama4Model.Config(
        dim=dim,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(num_embeddings=2048, embedding_dim=dim),
        norm=RMSNorm.Config(normalized_shape=dim),
        output=Linear.Config(in_features=dim, out_features=2048),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=2048,
            theta=500000,
            backend="complex",
            scaling="llama",
            scaling_factor=16.0,
            high_freq_factor=1.0,
        ),
        layers=_build_llama4_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=256),
            moe_hidden_dim=moe_hidden_dim,
            num_experts=num_experts,
            every_n_layers_nope=4,
            interleave_moe_layer_step=1,
            fixed_attn_block_size=256,
        ),
    )
    return Llama4Model(config)


def _get_expert_shard_dim(model: Llama4Model) -> int | None:
    """Return the shard dim used for expert params, or None if not sharded."""
    for layer in model.layers.values():
        if layer.moe_enabled:
            for param in layer.moe.experts.parameters():
                if hasattr(param, "placements"):
                    for p in param.placements:
                        if isinstance(p, Shard):
                            return p.dim
    return None


class TestApplyFsdpMoESharding(DTensorTestBase):
    """Test apply_fsdp expert sharding behavior with ep_degree=1 and ep_degree>1."""

    @property
    def world_size(self):
        return 8

    @with_comms
    def test_no_ep_fsdp_gt_num_experts_shards_dim1(self):
        """ep_degree=1, fsdp_size(8) > num_experts(4) → Shard(1)."""
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = _build_llama4_model(num_experts=4).to(self.device_type)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=1,
        )

        self.assertEqual(_get_expert_shard_dim(model), 1)

    @with_comms
    def test_no_ep_fsdp_le_num_experts_shards_dim0(self):
        """ep_degree=1, fsdp_size(8) <= num_experts(8) → Shard(0)."""
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = _build_llama4_model(num_experts=8).to(self.device_type)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=1,
        )

        self.assertEqual(_get_expert_shard_dim(model), 0)

    @with_comms
    def test_with_ep_fsdp_gt_num_experts_shards_dim1(self):
        """ep_degree=2, efsdp*ep(8) > num_experts(4) → Shard(1)."""
        # edp_mesh: 2D mesh [efsdp=4, ep=2], dp_mesh: 1D mesh [8]
        edp_mesh = init_device_mesh(
            self.device_type, (4, 2), mesh_dim_names=("efsdp", "ep")
        )
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = _build_llama4_model(num_experts=4).to(self.device_type)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=2,
            edp_mesh=edp_mesh,
        )

        self.assertEqual(_get_expert_shard_dim(model), 1)


if __name__ == "__main__":
    unittest.main()
