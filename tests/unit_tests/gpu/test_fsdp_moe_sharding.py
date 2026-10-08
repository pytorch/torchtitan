# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.config import ParallelismConfig
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.utils import clip_grad_norm_
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe_sharding import replicated_param_placement_sparse
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.sharding import set_qwen3_sharding_config
from torchtitan.protocols.sharding import ShardingConfig


pytestmark = pytest.mark.multi_gpu


def _qwen3_moe_config(num_experts: int = 8) -> Qwen3Model.Config:
    """Build a tiny Qwen3 MoE config with a configurable number of experts."""
    from torchtitan.models.common import CosSinRoPE, Embedding, Linear, RMSNorm

    # Use a tiny variant of the standard MoE debug config, overriding
    # num_experts to exercise the expert-sharding branches.
    from torchtitan.models.qwen3.flavors import _build_qwen3_moe_layers

    dim = 256
    head_dim = 128
    n_layers = 4
    vocab_size = 2048

    return Qwen3Model.Config(
        max_context_length=4096,
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
            # Most tests only inspect expert-parameter sharding. The postprocess
            # test also runs forward and backward with the default flex backend.
            attn_backend="flex",
            rope=CosSinRoPE.Config(
                dim=head_dim,
                max_context_length=4096,
                theta=1000000.0,
            ),
        ),
    )


def _build_qwen3_moe_model(num_experts: int = 8) -> Qwen3Model:
    return Qwen3Model(_qwen3_moe_config(num_experts))


def _get_expert_shard_dims(model: Qwen3Model) -> tuple[int | None, int | None]:
    """Return the W13 and W2 shard dimensions."""
    for layer in model.layers.values():
        if layer.moe_enabled:
            routed_experts = layer.moe.routed_experts
            return (
                _shard_dim(routed_experts.w13.weight),
                _shard_dim(routed_experts.w2.weight),
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
        return 8

    @with_comms
    def test_no_ep_fsdp_gt_num_experts_shards_feature_dimensions(self):
        """When FSDP cannot shard E, it shards each projection's feature dim."""
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

        self.assertEqual(_get_expert_shard_dims(model), (2, 1))

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

        self.assertEqual(_get_expert_shard_dims(model), (0, 0))

    @with_comms
    def test_with_ep_fsdp_gt_num_experts_shards_feature_dimensions(self):
        """Sparse FSDP also falls back to each projection's feature dim."""
        # edp_mesh: 2D mesh [edp_shard=4, ep=2], dp_mesh: 1D mesh [8]
        edp_mesh = init_device_mesh(
            self.device_type, (4, 2), mesh_dim_names=("edp_shard", "ep")
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

        self.assertEqual(_get_expert_shard_dims(model), (2, 1))

    @with_comms
    def test_with_ep_preserves_model_specific_expert_layout(self):
        """Model-specific grouped experts retain expert-axis sharding."""
        from torchtitan.models.gpt_oss import build_model_config

        config = build_model_config("debugmodel", seq_len=128, attn_backend="flex")
        for layer_config in config.layers:
            layer_config.moe.routed_experts.w13.out_features = 16
            layer_config.moe.routed_experts.w2.in_features = 16
        model = config.build().to(self.device_type)
        edp_mesh = init_device_mesh(
            self.device_type, (4, 2), mesh_dim_names=("edp_shard", "ep")
        )
        dp_mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp_shard",)
        )

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
            routed_experts = layer.moe.routed_experts
            self.assertEqual(
                {_shard_dim(param) for param in routed_experts.parameters()},
                {0},
            )

    @with_comms
    def test_with_ep_keeps_output_postprocess_on_sparse_mesh(self):
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=2,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        parallelism_context.build_mesh()
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)
        assert edp_mesh is not None

        for num_experts in (4, 8):
            with self.subTest(num_experts=num_experts):
                config = _qwen3_moe_config(num_experts=num_experts)
                output_postprocess = RMSNorm.Config(
                    normalized_shape=config.dim,
                    sharding_config=ShardingConfig(
                        state_shardings={"weight": replicated_param_placement_sparse()}
                    ),
                )
                for layer_config in config.layers:
                    assert layer_config.moe is not None
                    layer_config.moe.routed_experts.output_postprocess = (
                        output_postprocess
                    )
                set_qwen3_sharding_config(config, enable_sp=False, enable_ep=True)
                model = config.build().to(self.device_type)
                model.init_states()
                model._parallelize(parallelism_context)

                apply_fsdp_to_decoder(
                    model,
                    dp_mesh,
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    pp_enabled=False,
                    ep_degree=2,
                    edp_mesh=edp_mesh,
                    dp_mesh_dims=dp_mesh_dims,
                    edp_mesh_dims=edp_mesh_dims,
                )

                for layer in model.layers.values():
                    routed_experts = layer.moe.routed_experts
                    postprocess = routed_experts.output_postprocess
                    assert postprocess is not None
                    self.assertEqual(
                        dict(routed_experts.token_dispatcher.named_parameters()), {}
                    )
                    self.assertIs(routed_experts.w13.weight.device_mesh, edp_mesh)
                    self.assertIs(routed_experts.w2.weight.device_mesh, edp_mesh)
                    self.assertIs(postprocess.weight.device_mesh, edp_mesh)

                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                tokens = (
                    torch.arange(8, device=self.device_type) + 8 * self.rank
                ) % config.vocab_size
                positions = torch.arange(8, device=self.device_type)
                attention_metadata = model._get_attention_metadata(positions)
                with parallelism_context.activate_spmd():
                    model(
                        tokens,
                        positions=positions,
                        attention_metadata=attention_metadata,
                    ).sum().backward()

                state_dict = model.state_dict()
                for layer_id, layer in model.layers.items():
                    postprocess = layer.moe.routed_experts.output_postprocess
                    assert postprocess is not None
                    postprocess_grad = postprocess.weight.grad
                    assert postprocess_grad is not None
                    self.assertEqual(
                        postprocess_grad.placements,
                        postprocess.weight.placements,
                    )
                    weight_key = (
                        f"layers.{layer_id}.moe.routed_experts."
                        "output_postprocess.weight"
                    )
                    self.assertEqual(state_dict[weight_key].shape, (config.dim,))
                clip_grad_norm_(model.parameters(), max_norm=1.0, ep_enabled=True)
                optimizer.step()
                for layer in model.layers.values():
                    postprocess = layer.moe.routed_experts.output_postprocess
                    assert postprocess is not None
                    local_weight = postprocess.weight.to_local()
                    ep_replicas = [
                        torch.empty_like(local_weight)
                        for _ in range(edp_mesh["ep"].size())
                    ]
                    dist.all_gather(
                        ep_replicas,
                        local_weight,
                        group=edp_mesh["ep"].get_group(),
                    )
                    for replica in ep_replicas[1:]:
                        self.assertEqual(replica, ep_replicas[0])
                model.load_state_dict(state_dict)

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

        self.assertEqual(_get_expert_shard_dims(model), (0, 0))


class TestLinearStackingDistributed(DTensorTestBase):
    """Distributed coverage for stacked FFNs and ordinary linear weights."""

    @property
    def world_size(self):
        return 4

    @with_comms
    def test_w13_shards_dim_one(self):
        from torchtitan.models.llama3 import build_model_config
        from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

        config = build_model_config("debugmodel")
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
        from torchtitan.models.llama3 import build_model_config

        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=True,
        )
        parallelism_context.build_mesh()
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)

        sharded_config = build_model_config("debugmodel")
        sharded_config.layers[0].feed_forward.w13.param_init = fused_gate_up_param_init(
            {"weight": lambda tensor: torch.nn.init.constant_(tensor, 1)},
            {"weight": lambda tensor: torch.nn.init.constant_(tensor, 3)},
        )
        sharded_config.set_sharding_(
            ParallelismConfig(
                tensor_parallel_degree=2,
                enable_sequence_parallel=True,
            )
        )
        with torch.device("meta"):
            sharded = sharded_config.build()
        sharded._parallelize(parallelism_context)
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
        from torchtitan.models.deepseek_v4 import build_model_config
        from torchtitan.models.deepseek_v4.sharding import (
            set_deepseek_v4_sharding_config,
        )

        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=2,
            world_size=self.world_size,
            enable_sequence_parallel=True,
        )
        parallelism_context.build_mesh()
        config = build_model_config("debugmodel")
        set_deepseek_v4_sharding_config(config, enable_sp=True, enable_ep=True)
        model = config.build().to(self.device_type)

        model._parallelize(parallelism_context)

        self.assertEqual(model.layers["0"].attention.attn_sink.weight.ndim, 2)

    @with_comms
    def test_qwen3_tied_hf_state_dict_loads_after_fsdp(self):
        from torchtitan.models.qwen3 import build_model_config
        from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter

        config = build_model_config("debugmodel")
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
