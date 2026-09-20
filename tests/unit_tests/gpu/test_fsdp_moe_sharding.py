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
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder, resolve_fsdp_mesh
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.linear import Linear
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


def _get_expert_shard_dim(model: Qwen3Model) -> int | None:
    """Return the shard dim used for expert params, or None if not sharded."""
    for layer in model.layers.values():
        if layer.moe_enabled:
            for param in layer.moe.routed_experts.inner_experts.parameters():
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
        dp_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp_shard",)
        )
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
        dp_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp_shard",)
        )
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
        dp_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp_shard",)
        )
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

    @with_comms
    def test_no_ep_hsdp_ignores_dp_replicate(self):
        """ep_degree=1 under HSDP: dp_replicate must not count as shard degree.

        The mesh is (dp_replicate=2, dp_shard=4), so FSDP cuts dim 0 of the
        expert weights 4 ways, and 4 <= num_experts(4) -> Shard(0). Counting
        dp_replicate gives 2*4=8 > 4 -> Shard(1), which pads dim 0 needlessly
        and changes the on-disk checkpoint layout.
        """
        dp_mesh = init_device_mesh(
            self.device_type, (2, 4), mesh_dim_names=("dp_replicate", "dp_shard")
        )
        model = _build_qwen3_moe_model(num_experts=4).to(self.device_type)

        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            ep_degree=1,
        )

        self.assertEqual(_get_expert_shard_dim(model), 0)


class TestLinearStackingDistributed(DTensorTestBase):
    """Distributed coverage for stacked FFNs and ordinary linear weights."""

    @property
    def world_size(self):
        return 4

    @with_comms
    def test_w13_shards_dim_one(self):
        from torchtitan.models.llama3 import model_registry
        from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

        config = model_registry("debugmodel").model
        model = config.build().to(self.device_type)
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )

        w13 = model.layers["0"].feed_forward.w13
        self.assertIsInstance(w13, Linear)
        shard_dims = {
            placement.dim
            for placement in w13.weight.placements
            if isinstance(placement, Shard)
        }
        self.assertEqual(shard_dims, {1})
        state_dict = model.state_dict()
        self.assertEqual(
            state_dict["layers.0.feed_forward.w13.weight"].shape,
            (2, 768, 256),
        )

        adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
        hf_state_dict = adapter.to_hf(state_dict)
        gate = hf_state_dict["model.layers.0.mlp.gate_proj.weight"]
        gate_shard_dims = {
            placement.dim
            for placement in gate.placements
            if isinstance(placement, Shard)
        }
        self.assertEqual(gate.shape, (768, 256))
        self.assertEqual(gate_shard_dims, {0})

        state_dict = adapter.from_hf(hf_state_dict)
        restored_w13 = state_dict["layers.0.feed_forward.w13.weight"]
        restored_shard_dims = {
            placement.dim
            for placement in restored_w13.placements
            if isinstance(placement, Shard)
        }
        self.assertEqual(restored_w13.shape, (2, 768, 256))
        self.assertEqual(restored_shard_dims, {1})
        model.load_state_dict(state_dict)

    @with_comms
    def test_tp_fsdp_initializes_gate_and_up_separately(self):
        from torchtitan.models.common.config_utils import fused_gate_up_param_init
        from torchtitan.models.llama3 import model_registry
        from torchtitan.models.llama3.sharding import set_llama3_sharding_config

        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        parallel_dims.build_mesh()
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)

        sharded_config = model_registry("debugmodel").model
        sharded_config.layers[0].feed_forward.w13.param_init = fused_gate_up_param_init(
            {"weight": lambda tensor: torch.nn.init.constant_(tensor, 1)},
            {"weight": lambda tensor: torch.nn.init.constant_(tensor, 3)},
        )
        set_llama3_sharding_config(sharded_config, enable_sp=True)
        with torch.device("meta"):
            sharded = sharded_config.build()
        sharded.parallelize(parallel_dims)
        apply_fsdp_to_decoder(
            sharded,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            dp_mesh_dims=dp_mesh_dims,
        )
        sharded.to_empty(device=self.device_type)
        sharded.layers["0"].feed_forward.w13.init_states()

        actual_w13 = sharded.layers["0"].feed_forward.w13.weight.full_tensor()
        torch.testing.assert_close(actual_w13[0], torch.ones_like(actual_w13[0]))
        torch.testing.assert_close(actual_w13[1], 3 * torch.ones_like(actual_w13[1]))

        state_dict = sharded.state_dict()
        self.assertEqual(
            state_dict["layers.0.feed_forward.w13.weight"].shape, (2, 768, 256)
        )
        sharded.load_state_dict(state_dict)

    @with_comms
    def test_deepseek_v4_tp_keeps_attention_sink_two_dimensional(self):
        from torchtitan.models.deepseek_v4 import model_registry
        from torchtitan.models.deepseek_v4.sharding import (
            set_deepseek_v4_sharding_config,
        )

        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=2,
            world_size=self.world_size,
        )
        parallel_dims.build_mesh()
        config = model_registry("debugmodel").model
        set_deepseek_v4_sharding_config(config, enable_sp=True, enable_ep=True)
        model = config.build().to(self.device_type)

        model.parallelize(parallel_dims)

        self.assertEqual(model.layers["0"].attention.attn_sink.weight.ndim, 2)

    @with_comms
    def test_qwen3_tied_hf_state_dict_loads_after_fsdp(self):
        from torchtitan.models.qwen3 import model_registry
        from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter

        config = model_registry("debugmodel").model
        model = config.build().to(self.device_type)
        model.init_states()
        dp_mesh = init_device_mesh(self.device_type, (self.world_size,))
        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )

        adapter = Qwen3StateDictAdapter(config, hf_assets_path=None)
        state_dict = adapter.from_hf(adapter.to_hf(model.state_dict()))
        self.assertEqual(state_dict["tok_embeddings.weight"].ndim, 2)
        self.assertEqual(state_dict["lm_head.weight"].ndim, 2)
        model.load_state_dict(state_dict)


if __name__ == "__main__":
    unittest.main()
