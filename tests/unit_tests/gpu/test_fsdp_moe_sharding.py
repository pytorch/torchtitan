# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.models.qwen3.model import Qwen3Model


pytestmark = pytest.mark.multi_gpu


def _build_qwen3_moe_model(num_experts: int = 8) -> Qwen3Model:
    """Build a tiny Qwen3 MoE model with a configurable number of experts."""
    from torchtitan.models.common import CosSinRoPE, Embedding, Linear, RMSNorm

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
        tok_embeddings=Embedding.Config(num_embeddings=vocab_size, embedding_dim=dim),
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
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=4096,
                theta=1000000.0,
            ),
        ),
    )
    return Qwen3Model(config)


def _get_expert_shard_dims(model: Qwen3Model) -> tuple[int | None, int | None]:
    """Return the W13 and W2 shard dimensions."""
    for layer in model.layers.values():
        if layer.moe_enabled:
            # pyrefly: ignore [missing-attribute]
            routed_experts = layer.moe.routed_experts
            return _shard_dim(routed_experts.w13.weight), _shard_dim(
                routed_experts.w2.weight
            )
    return None, None


def _shard_dim(param: torch.Tensor) -> int | None:
    """Return a DTensor parameter's shard dimension, if any."""
    for placement in getattr(param, "placements", ()):
        if isinstance(placement, Shard):
            return placement.dim
    return None


class TestApplyFsdpMoESharding(DTensorTestBase):
    """Test apply_fsdp_to_decoder expert sharding behavior with ep_degree=1 and ep_degree>1."""

    @property
    def world_size(self):
        return 4

    @with_comms
    def test_no_ep_fsdp_gt_num_experts_shards_feature_dimensions(self):
        """When FSDP cannot shard E, it shards each linear's feature dim."""
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = _build_qwen3_moe_model(num_experts=2).to(self.device_type)

        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=1,
        )

        self.assertEqual(_get_expert_shard_dims(model), (2, 1))

    @with_comms
    def test_no_ep_fsdp_le_num_experts_shards_dim0(self):
        """FSDP shards the expert axis when it does not require padding."""
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

        self.assertEqual(_get_expert_shard_dims(model), (0, 0))

    @with_comms
    def test_with_ep_fsdp_gt_num_experts_shards_feature_dimensions(self):
        """Sparse FSDP also falls back to each linear's feature dim."""
        # edp_mesh: 2D mesh [efsdp=2, ep=2], dp_mesh: 1D mesh [4]
        edp_mesh = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("efsdp", "ep")
        )
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = _build_qwen3_moe_model(num_experts=2).to(self.device_type)

        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=2,
            edp_mesh=edp_mesh,
        )

        self.assertEqual(_get_expert_shard_dims(model), (2, 1))

    @with_comms
    def test_with_ep_preserves_model_specific_expert_layout(self):
        """Model-specific grouped experts retain expert-axis sharding."""
        from torchtitan.models.gpt_oss import model_registry

        config = model_registry("debugmodel", seq_len=128, attn_backend="flex").model
        for layer_config in config.layers:
            # Keep GPT-OSS's real parameter names while reducing test memory.
            layer_config.moe.routed_experts.w13.out_features = (2, 16)
            layer_config.moe.routed_experts.w2.in_features = 16
        model = config.build().to(self.device_type)
        edp_mesh = init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("efsdp", "ep")
        )
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))

        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=2,
            edp_mesh=edp_mesh,
        )

        for layer in model.layers.values():
            experts = layer.moe.routed_experts
            self.assertEqual(
                {
                    _shard_dim(param)
                    for linear in (experts.w13, experts.w2)
                    for param in linear.parameters()
                },
                {0},
            )


if __name__ == "__main__":
    unittest.main()
