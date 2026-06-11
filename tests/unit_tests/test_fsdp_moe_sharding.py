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
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.models.qwen3.model import Qwen3Model


def _build_qwen3_moe_model(num_experts: int = 8) -> Qwen3Model:
    """Build a tiny Qwen3 MoE model with a configurable number of experts."""
    from torchtitan.models.common import (
        CosSinRoPE,
        Linear,
        RMSNorm,
        VocabParallelEmbedding,
    )

    # Use a tiny variant of the standard MoE debug config, overriding
    # num_experts to exercise the expert-sharding branches.
    from torchtitan.models.qwen3 import _build_qwen3_moe_layers

    dim = 256
    head_dim = 128
    n_layers = 4
    vocab_size = 2048

    config = Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=RMSNorm.Config(normalized_shape=dim),
        tok_embeddings=VocabParallelEmbedding.Config(num_embeddings=vocab_size, embedding_dim=dim),
        lm_head=Linear.Config(in_features=dim, out_features=vocab_size),
        layers=_build_qwen3_moe_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=16,
            n_kv_heads=8,
            head_dim=head_dim,
            moe_hidden_dim=768,
            num_experts=num_experts,
            # top_k must not exceed num_experts (router selects top_k of them).
            top_k=min(8, num_experts),
            # This test only checks expert-param sharding (no forward), so the
            # attention backend is irrelevant; use the default flex backend
            attn_backend="flex",
            moe_comm_backend="standard",
            rope=CosSinRoPE.Config(dim=head_dim, max_seq_len=4096, theta=1000000.0),
        ),
    )
    return Qwen3Model(config)


def _get_expert_shard_dim(model: Qwen3Model) -> int | None:
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
    """Test apply_fsdp_to_decoder expert sharding behavior with ep_degree=1 and ep_degree>1."""

    @property
    def world_size(self):
        return 8

    @with_comms
    def test_no_ep_fsdp_gt_num_experts_shards_dim1(self):
        """ep_degree=1, fsdp_size(8) > num_experts(4) → Shard(1)."""
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = _build_qwen3_moe_model(num_experts=4).to(self.device_type)

        apply_fsdp_to_decoder(
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
        model = _build_qwen3_moe_model(num_experts=8).to(self.device_type)

        apply_fsdp_to_decoder(
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
        model = _build_qwen3_moe_model(num_experts=4).to(self.device_type)

        apply_fsdp_to_decoder(
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
