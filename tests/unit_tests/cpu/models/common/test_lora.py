# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
from spmd_types import SpmdType
from spmd_types.checker import typecheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LoRATransform, transform_model_config_
from torchtitan.distributed.parallelism_context import MeshAxisName, ParallelismContext
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    residual_activation_placement,
)
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    GroupedLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.models.common.moe.sharding import (
    expert_param_placement_sparse,
    replicated_param_placement_sparse,
)
from torchtitan.models.common.vision_encoder import (
    InvariantRowParallelLinear,
    VisionMLP,
)
from torchtitan.models.common.vision_encoder_sharding import (
    vision_colwise_sharding_config,
    vision_invariant_linear_sharding_config,
    vision_rowwise_sharding_config,
)
from torchtitan.models.gpt_oss.moe import GptOssGroupedLinear
from torchtitan.models.llama3 import build_model_config
from torchtitan.models.qwen3_5.model import Qwen35Model
from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel_moe_lora

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def test_qwen35_moe_lora_model_config():
    config = qwen35_debugmodel_moe_lora()
    model_config = cast(Qwen35Model.Config, config.model)
    num_layers = len(model_config.layers)
    dense_lora = {
        fqn: projection
        for fqn, projection, _parent, _attr in model_config.traverse(Linear.Config)
        if hasattr(projection, "rank")
    }
    grouped_lora = {
        fqn: projection
        for fqn, projection, _parent, _attr in model_config.traverse(
            GroupedLinear.Config
        )
        if hasattr(projection, "rank")
    }

    assert set(dense_lora) == {
        f"layers.{layer}.feed_forward.shared_experts.{projection}"
        for layer in range(num_layers)
        for projection in ("w13", "w2")
    }
    assert set(grouped_lora) == {
        f"layers.{layer}.feed_forward.routed_experts.{projection}"
        for layer in range(num_layers)
        for projection in ("w13", "w2")
    }


def test_lora_model_builds():
    """LoRA debug model builds, has trainable adapters and frozen base."""
    model_config = build_model_config("debugmodel", parallelism=ParallelismConfig())
    model_config = transform_model_config_(
        model_config,
        [
            LoRATransform(
                rank=8,
                alpha=16.0,
                target_modules=["wqkv", "wo"],
            )
        ],
    )
    model = model_config.build()
    model.init_states()

    for layer in model.layers.values():
        assert isinstance(layer.attention.qkv_linear.wqkv, ColumnParallelLinear)
        assert isinstance(layer.attention.wo, RowParallelLinear)
        assert hasattr(layer.attention.qkv_linear.wqkv, "lora_a")
        assert hasattr(layer.attention.wo, "lora_a")

    lora_params = {
        n for n, p in model.named_parameters() if "lora_a" in n or "lora_b" in n
    }
    frozen_linears = {n for n, p in model.named_parameters() if not p.requires_grad}

    assert len(lora_params) > 0, "No LoRA parameters found"
    assert len(frozen_linears) > 0, "No frozen parameters found"
    lora_modules = {name.rsplit(".", 2)[0] for name in lora_params}
    assert lora_modules == {
        f"layers.{layer}.attention.{projection}"
        for layer in range(6)
        for projection in ("qkv_linear.wqkv", "wo")
    }
    for name in lora_params:
        assert model.get_parameter(
            name
        ).requires_grad, f"LoRA param '{name}' should be trainable"
    for name in frozen_linears:
        assert (
            "lora_a" not in name and "lora_b" not in name
        ), f"Frozen param '{name}' looks like a LoRA adapter"
    non_lora_trainable = {
        n
        for n, p in model.named_parameters()
        if p.requires_grad and "lora_a" not in n and "lora_b" not in n
    }
    assert non_lora_trainable == set()


def test_lora_forward():
    """LoRA model forward produces correct output shape."""
    model_config = build_model_config(
        "debugmodel", attn_backend="flex", parallelism=ParallelismConfig()
    )
    model_config = transform_model_config_(
        model_config,
        [
            LoRATransform(
                rank=8,
                alpha=16.0,
                target_modules=["wqkv", "wo"],
            )
        ],
    )
    model = model_config.build()
    model.init_states()

    vocab_size = model_config.vocab_size
    num_documents, seq_len = 2, 16
    num_tokens = num_documents * seq_len
    tokens = torch.randint(0, vocab_size, (num_tokens,))
    positions = torch.arange(seq_len).repeat(num_documents)
    attention_metadata = model._get_attention_metadata(positions)
    # Flex supports the CPU forward used by this shape check, but not backward.
    with torch.no_grad():
        output = model(
            tokens, attention_metadata=attention_metadata, positions=positions
        )
    assert output.shape == (num_tokens, vocab_size)


def test_lora_targets_fused_feed_forward_projection():
    """The physical w13 projection uses one LoRA adapter."""
    init = {"weight": torch.nn.init.ones_}
    config = DenseFeedForward.Config(
        w13=Linear.Config(
            in_features=4, out_features=8, num_linears=2, param_init=init
        ),
        w2=Linear.Config(in_features=8, out_features=4, param_init=init),
    )
    config = LoRATransform(
        rank=2,
        alpha=4.0,
        target_modules=["w13"],
    ).transform(config)
    feed_forward = config.build()
    feed_forward.init_states()

    assert set(feed_forward.state_dict()) == {
        "w13.weight",
        "w2.weight",
        "w13.lora_a.weight",
        "w13.lora_b.weight",
    }
    assert {
        name
        for name, parameter in feed_forward.named_parameters()
        if parameter.requires_grad
    } == {
        "w13.lora_a.weight",
        "w13.lora_b.weight",
    }

    with torch.no_grad():
        for adapter in (feed_forward.w13.lora_a, feed_forward.w13.lora_b):
            adapter.weight.copy_(torch.randn_like(adapter.weight))

    x = torch.randn(3, 4)
    gate_up = F.linear(x, feed_forward.w13.weight.flatten(0, -2)).unflatten(-1, (2, 8))
    gate_up = gate_up + 2 * feed_forward.w13.lora_b(feed_forward.w13.lora_a(x))
    gate, up = gate_up.unbind(-2)
    expected = feed_forward.w2(F.silu(gate) * up)
    torch.testing.assert_close(feed_forward(x), expected)

    reloaded = config.build()
    reloaded.init_states()
    reloaded.load_state_dict(feed_forward.state_dict())
    torch.testing.assert_close(reloaded(x), expected)


def test_lora_targets_fused_grouped_projection():
    """Each expert gets one A and a stacked B for a fused projection."""
    config = LoRATransform(rank=8, alpha=16.0).transform(
        GroupedLinear.Config(
            group_size=2,
            in_features=8,
            out_features=8,
            num_linears=2,
            param_init={"weight": torch.nn.init.ones_},
        )
    )
    grouped = config.build()
    grouped.init_states()

    assert grouped.weight.shape == (2, 2, 8, 8)
    assert grouped.lora_a.weight.shape == (2, 8, 8)
    assert grouped.lora_b.weight.shape == (2, 2, 8, 8)
    assert set(grouped.state_dict()) == {
        "weight",
        "lora_a.weight",
        "lora_b.weight",
    }
    assert not grouped.weight.requires_grad
    assert grouped.lora_a.weight.requires_grad
    assert grouped.lora_b.weight.requires_grad
    assert torch.count_nonzero(grouped.lora_b.weight) == 0

    with torch.no_grad():
        grouped.lora_a.weight.copy_(torch.randn_like(grouped.lora_a.weight))
        grouped.lora_b.weight.copy_(torch.randn_like(grouped.lora_b.weight))

    input_RI = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)
    offsets_E = torch.tensor([2, 5], dtype=torch.int32)
    actual_R2O = grouped(input_RI, offsets_E)

    expected_parts = []
    start = 0
    for expert, end in enumerate(offsets_E.tolist()):
        expert_input_RI = input_RI[start:end]
        base_RO = F.linear(
            expert_input_RI,
            grouped.weight[expert].flatten(0, -2).bfloat16(),
        )
        hidden_RL = F.linear(
            expert_input_RI,
            grouped.lora_a.weight[expert].bfloat16(),
        )
        update_RO = F.linear(
            hidden_RL,
            grouped.lora_b.weight[expert].flatten(0, -2).bfloat16(),
        )
        expected_parts.append(base_RO + 2 * update_RO)
        start = end
    expected_R2O = torch.cat(expected_parts).unflatten(-1, (2, 8))

    torch.testing.assert_close(actual_R2O, expected_R2O)
    actual_R2O.float().sum().backward()
    assert grouped.weight.grad is None
    assert grouped.lora_a.weight.grad is not None
    assert grouped.lora_b.weight.grad is not None


class _RoutedProjections(Module):
    """The two grouped projections of a routed-expert block, by their field names."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w13: GroupedLinear.Config
        w2: GroupedLinear.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.w13 = config.w13.build()
        self.w2 = config.w2.build()


def _routed_projections_config(
    *, num_experts: int = 2, dim: int = 8, hidden_dim: int = 16
) -> _RoutedProjections.Config:
    return _RoutedProjections.Config(
        w13=GroupedLinear.Config(
            group_size=num_experts,
            in_features=dim,
            out_features=hidden_dim,
            num_linears=2,
            param_init={"weight": torch.nn.init.ones_},
        ),
        w2=GroupedLinear.Config(
            group_size=num_experts,
            in_features=hidden_dim,
            out_features=dim,
            param_init={"weight": torch.nn.init.ones_},
        ),
    )


def _shared_outer_reference(
    projection, input_RI: torch.Tensor, offsets_E: torch.Tensor
) -> torch.Tensor:
    """Per-expert loop over the rows with the shared factor broadcast."""
    parts = []
    start = 0
    for expert, end in enumerate(offsets_E.tolist()):
        rows_RI = input_RI[start:end]
        weight_a = projection.lora_a.weight
        weight_b = projection.lora_b.weight
        weight_a_LI = weight_a if weight_a.dim() == 2 else weight_a[expert]
        weight_b_OL = (
            weight_b.flatten(0, -2)
            if isinstance(projection.lora_b, Linear)
            else weight_b[expert].flatten(0, -2)
        )
        base_RO = F.linear(rows_RI, projection.weight[expert].flatten(0, -2).bfloat16())
        update_RO = F.linear(
            F.linear(rows_RI, weight_a_LI.bfloat16()), weight_b_OL.bfloat16()
        )
        parts.append(base_RO + 2 * update_RO)
        start = end
    return torch.cat(parts)


def test_shared_outer_adapters_share_the_model_side_factor():
    """W13 shares A, W2 shares B, and the routed rows see the same update a per-expert loop gives."""
    config = LoRATransform(
        rank=8, alpha=16.0, expert_adapter_layout="shared_outer"
    ).transform(_routed_projections_config())
    experts = config.build()
    experts.init_states()

    assert isinstance(experts.w13.lora_a, Linear)
    assert experts.w13.lora_a.weight.shape == (8, 8)
    assert experts.w13.lora_b.weight.shape == (2, 2, 16, 8)
    assert experts.w2.lora_a.weight.shape == (2, 8, 16)
    assert isinstance(experts.w2.lora_b, Linear)
    assert experts.w2.lora_b.weight.shape == (8, 8)
    assert set(experts.state_dict()) == {
        f"{projection}.{name}"
        for projection in ("w13", "w2")
        for name in ("weight", "lora_a.weight", "lora_b.weight")
    }
    for projection in (experts.w13, experts.w2):
        assert not projection.weight.requires_grad
        assert torch.count_nonzero(projection.lora_b.weight) == 0
        with torch.no_grad():
            projection.lora_a.weight.copy_(torch.randn_like(projection.lora_a.weight))
            projection.lora_b.weight.copy_(torch.randn_like(projection.lora_b.weight))

    offsets_E = torch.tensor([2, 5], dtype=torch.int32)
    input_RD = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)
    hidden_R2F = experts.w13(input_RD, offsets_E)
    torch.testing.assert_close(
        hidden_R2F,
        _shared_outer_reference(experts.w13, input_RD, offsets_E).unflatten(
            -1, (2, 16)
        ),
    )
    input_RF = torch.randn(5, 16, dtype=torch.bfloat16, requires_grad=True)
    output_RD = experts.w2(input_RF, offsets_E)
    torch.testing.assert_close(
        output_RD, _shared_outer_reference(experts.w2, input_RF, offsets_E)
    )

    (hidden_R2F.float().sum() + output_RD.float().sum()).backward()
    for projection in (experts.w13, experts.w2):
        assert projection.weight.grad is None
        assert projection.lora_a.weight.grad is not None
        assert projection.lora_b.weight.grad is not None


def test_shared_outer_adapters_follow_expert_sharding():
    """The shared factor is replicated across experts; the per-expert one keeps the expert placement."""
    expert_placement = expert_param_placement_sparse()
    config = _routed_projections_config()
    for projection in (config.w13, config.w2):
        projection.sharding_config = ShardingConfig(
            state_shardings={"weight": expert_placement}
        )
    experts = (
        LoRATransform(rank=8, expert_adapter_layout="shared_outer")
        .transform(config)
        .build()
    )

    shared = replicated_param_placement_sparse()
    assert experts.w13.lora_a._sharding_config.state_shardings == {"weight": shared}
    assert experts.w13.lora_b._sharding_config.state_shardings == {
        "weight": expert_placement
    }
    assert experts.w2.lora_a._sharding_config.state_shardings == {
        "weight": expert_placement
    }
    assert experts.w2.lora_b._sharding_config.state_shardings == {"weight": shared}


def test_shared_outer_rejects_a_grouped_projection_outside_the_experts():
    with pytest.raises(ValueError, match="w13 and w2 only"):
        LoRATransform(rank=8, expert_adapter_layout="shared_outer").transform(
            GroupedLinear.Config(group_size=2, in_features=8, out_features=8)
        )


def test_grouped_lora_preserves_specialized_projection():
    config = LoRATransform(rank=8).transform(
        GptOssGroupedLinear.Config(
            group_size=2,
            in_features=8,
            out_features=8,
            param_init={
                "weight": torch.nn.init.ones_,
                "bias": torch.nn.init.zeros_,
            },
        )
    )
    grouped = config.build()
    grouped.init_states()

    assert isinstance(grouped, GptOssGroupedLinear)
    assert not grouped.weight.requires_grad
    assert not grouped.bias.requires_grad
    assert grouped.get_parameter("lora_a.weight").requires_grad
    assert grouped.get_parameter("lora_b.weight").requires_grad

    with torch.no_grad():
        grouped.bias.fill_(3)
        grouped.lora_a.weight.fill_(1)
        grouped.lora_b.weight.fill_(1)

    input_RI = torch.ones(4, 8, dtype=torch.bfloat16)
    offsets_E = torch.tensor([2, 4], dtype=torch.int32)
    base_RO = torch.full((4, 8), 8, dtype=torch.bfloat16)
    update_RO = torch.full((4, 8), 64, dtype=torch.bfloat16)
    expected_RO = base_RO + 2 * update_RO + 3
    torch.testing.assert_close(grouped(input_RI, offsets_E), expected_RO)


def test_grouped_lora_a_uses_linear_fan_in():
    config = LoRATransform(rank=8).transform(
        GroupedLinear.Config(
            group_size=2,
            in_features=16,
            out_features=32,
            param_init={"weight": torch.nn.init.ones_},
        )
    )
    grouped = config.build()

    torch.manual_seed(42)
    grouped.init_states()
    torch.manual_seed(42)
    expected = torch.empty_like(grouped.lora_a.weight)
    torch.nn.init.kaiming_uniform_(expected.flatten(0, -2), a=math.sqrt(5))

    torch.testing.assert_close(grouped.lora_a.weight, expected)


@pytest.mark.parametrize("rank", [1, 4, 9])
def test_grouped_lora_requires_rank_divisible_by_eight(rank):
    with pytest.raises(ValueError, match="rank must be divisible by 8"):
        LoRATransform(rank=rank).transform(
            GroupedLinear.Config(
                group_size=2,
                in_features=8,
                out_features=8,
            )
        )


def test_grouped_lora_adapters_follow_expert_sharding():
    expert_placement = expert_param_placement_sparse()
    base_sharding = ShardingConfig(
        state_shardings={"weight": expert_placement},
    )
    config = LoRATransform(rank=8).transform(
        GroupedLinear.Config(
            group_size=8,
            in_features=16,
            out_features=32,
            sharding_config=base_sharding,
        )
    )
    grouped = config.build()

    for adapter in (grouped.lora_a, grouped.lora_b):
        assert adapter._sharding_config is not None
        assert adapter._sharding_config is not base_sharding
        assert adapter._sharding_config.state_shardings == {"weight": expert_placement}
        assert adapter._sharding_config.in_src_shardings is None
        assert adapter._sharding_config.in_dst_shardings is None
        assert adapter._sharding_config.out_src_shardings is None
        assert adapter._sharding_config.out_dst_shardings is None


def test_grouped_lora_rejects_feature_axis_sharding():
    config = LoRATransform(rank=8).transform(
        GroupedLinear.Config(
            group_size=8,
            in_features=16,
            out_features=32,
            sharding_config=ShardingConfig(
                state_shardings={
                    "weight": dense_param_placement(tp=spmd.S(1)),
                },
            ),
        )
    )

    with pytest.raises(
        ValueError, match="only expert-axis or replicated parameter sharding"
    ):
        config.build()


def test_stacked_lora_adapter_does_not_repeat_base_redistribution():
    """The LoRA adapters inherit state sharding, not TP collectives."""
    init = {"weight": torch.nn.init.zeros_}
    config = make_ffn_config(
        dim=4,
        hidden_dim=8,
        w13_param_init=init,
        w2_param_init=init,
        enable_sp=True,
    )
    config = LoRATransform(
        rank=2,
        alpha=4,
        target_modules=["w13"],
    ).transform(config)
    assert isinstance(config, DenseFeedForward.Config)

    feed_forward = config.build()
    assert feed_forward.w13._sharding_config is not None

    lora_b_sharding = feed_forward.w13.lora_b._sharding_config
    assert lora_b_sharding is not None
    assert lora_b_sharding.state_shardings["weight"] == dense_param_placement(
        tp=spmd.S(1)
    )
    assert lora_b_sharding.in_src_shardings is None
    assert lora_b_sharding.in_dst_shardings is None
    assert lora_b_sharding.out_src_shardings is None
    assert lora_b_sharding.out_dst_shardings is None


def test_lora_class_is_reused_for_the_same_parent():
    """The LoRA class is cached for each parent Linear class."""
    first = LoRATransform(rank=2, alpha=4.0).transform(
        Linear.Config(in_features=4, out_features=3)
    )
    second = LoRATransform(rank=2, alpha=4.0).transform(
        Linear.Config(in_features=4, out_features=3)
    )

    assert type(first) is type(second)
    assert first._owner is second._owner
    assert first._owner is not None
    assert first._owner.__name__ == "LoRALinear"
    assert issubclass(first._owner, Linear)


def test_lora_matches_linear_config_subclass():
    class AlternateLinear(Linear):
        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            pass

    config = AlternateLinear.Config(in_features=4, out_features=3)
    transformed = LoRATransform(rank=2, alpha=4.0).transform(config)
    model = transformed.build()

    assert isinstance(model, AlternateLinear)
    assert not model.weight.requires_grad
    assert model.lora_a.weight.requires_grad
    assert model.lora_b.weight.requires_grad


@pytest.mark.parametrize("parallel_cls", [InvariantRowParallelLinear])
def test_lora_preserves_specialized_row_parallel_linear(parallel_cls):
    config = parallel_cls.Config(
        in_features=4,
        out_features=3,
        bias=True,
    )
    transformed = LoRATransform(
        rank=2,
        alpha=4.0,
    ).transform(config)
    linear = transformed.build()

    assert isinstance(linear, parallel_cls)
    x = torch.randn(5, 4)
    expected = F.linear(x, linear.weight, linear.bias)
    expected += 2 * linear.lora_b(linear.lora_a(x))
    torch.testing.assert_close(linear(x), expected)


def test_lora_transform_targets_full_fqn_suffix():
    model_config = LoRATransform(
        rank=2,
        alpha=4.0,
        target_modules=["layers.0.attention.wo"],
    ).transform(build_model_config("debugmodel", parallelism=ParallelismConfig()))

    targets = {
        fqn
        for fqn, config, _parent, _attr in model_config.traverse(Linear.Config)
        if hasattr(config, "rank")
    }
    assert targets == {"layers.0.attention.wo"}


def test_lora_transform_targets_shared_fqn_suffix():
    model_config = LoRATransform(
        rank=2,
        alpha=4.0,
        target_modules=["attention.wo"],
    ).transform(build_model_config("debugmodel", parallelism=ParallelismConfig()))

    targets = {
        fqn
        for fqn, config, _parent, _attr in model_config.traverse(Linear.Config)
        if hasattr(config, "rank")
    }
    assert targets == {
        f"layers.{layer}.attention.wo" for layer in range(len(model_config.layers))
    }


@pytest.mark.parametrize(
    ("target_modules", "expected_lora"),
    [
        (["blocks.0.projections.0"], None),
        (["projections.0"], True),
        ([], False),
    ],
)
def test_lora_requires_consistent_targets_for_shared_projection_config(
    target_modules, expected_lora
):
    class ProjectionGroup(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            projections: list[Linear.Config]

    class Root(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            blocks: list[ProjectionGroup.Config]

    shared_group = ProjectionGroup.Config(
        projections=[Linear.Config(in_features=4, out_features=4)]
    )
    model_config = Root.Config(blocks=[shared_group, shared_group])
    original_projection = shared_group.projections[0]

    transform = LoRATransform(rank=2, target_modules=target_modules)
    if expected_lora is not None:
        transformed = transform.transform(model_config)
        assert [
            hasattr(block.projections[0], "rank") for block in transformed.blocks
        ] == [expected_lora, expected_lora]
        return

    with pytest.raises(ValueError, match="inconsistent LoRA decisions") as exc_info:
        transform.transform(model_config)

    assert "blocks.0.projections.0" in str(exc_info.value)
    assert "blocks.1.projections.0" in str(exc_info.value)
    assert model_config.blocks[0] is shared_group
    assert model_config.blocks[1] is shared_group
    assert shared_group.projections[0] is original_projection


@pytest.mark.parametrize("target_modules", [None, ["lm_head"]])
def test_lora_rejects_lm_head_target_with_weight_tying(target_modules):
    model_config = build_model_config("debugmodel", parallelism=ParallelismConfig())
    model_config.enable_weight_tying = True
    original_lm_head = model_config.lm_head
    original_wo = model_config.layers[-1].attention.wo

    with pytest.raises(ValueError, match="weight tying"):
        LoRATransform(rank=2, target_modules=target_modules).transform(model_config)

    assert model_config.lm_head is original_lm_head
    assert model_config.layers[-1].attention.wo is original_wo


def test_lora_allows_other_targets_with_weight_tying():
    model_config = build_model_config("debugmodel", parallelism=ParallelismConfig())
    model_config.enable_weight_tying = True

    transformed = LoRATransform(rank=2, target_modules=["wo"]).transform(model_config)

    assert not hasattr(transformed.lm_head, "rank")
    assert all(hasattr(layer.attention.wo, "rank") for layer in transformed.layers)


def test_lora_rank_validation():
    """LoRA rank must be positive."""
    with pytest.raises(ValueError, match="rank must be positive"):
        LoRATransform(rank=0)
    with pytest.raises(ValueError, match="rank must be positive"):
        LoRATransform(rank=-1)


def test_multiple_lora_transforms_conflict():
    model_config = build_model_config("debugmodel", parallelism=ParallelismConfig())

    with pytest.raises(ValueError, match="cannot be combined"):
        transform_model_config_(
            model_config,
            [
                LoRATransform(
                    rank=2,
                    alpha=4.0,
                    target_modules=["wqkv"],
                ),
                LoRATransform(
                    rank=4,
                    alpha=8.0,
                    target_modules=["wo"],
                ),
            ],
        )


def test_lora_freezes_direct_params_on_composite_modules():
    """Composite modules freeze own params while child LoRA adapters train."""

    class CompositeWithDirectParam(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            child: Linear.Config
            dim: int = 4

        def __init__(self, config: Config) -> None:
            super().__init__()
            self.direct = torch.nn.Parameter(torch.ones(config.dim))
            self.child = config.child.build()

    class Root(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            block: CompositeWithDirectParam.Config

        def __init__(self, config: Config) -> None:
            super().__init__()
            self.direct = torch.nn.Parameter(torch.ones(4))
            self.block = config.block.build()

    model_config = Root.Config(
        block=CompositeWithDirectParam.Config(
            child=Linear.Config(in_features=4, out_features=4),
            dim=4,
        )
    )

    model_config = LoRATransform(
        rank=2,
        alpha=4.0,
        target_modules=["child"],
    ).transform(model_config)
    model = model_config.build()

    assert not model.direct.requires_grad
    assert not model.block.direct.requires_grad
    assert not model.block.child.weight.requires_grad
    assert model.block.child.lora_a.weight.requires_grad
    assert model.block.child.lora_b.weight.requires_grad

    non_lora_trainable = {
        n
        for n, p in model.named_parameters()
        if p.requires_grad and "lora_a" not in n and "lora_b" not in n
    }
    assert non_lora_trainable == set()


def test_lora_freezes_direct_params_on_root_module():
    """Root module direct params are frozen through the returned root config."""

    class RootWithDirectParam(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            child: Linear.Config
            dim: int = 4

        def __init__(self, config: Config) -> None:
            super().__init__()
            self.direct = torch.nn.Parameter(torch.ones(config.dim))
            self.child = config.child.build()

    model_config = RootWithDirectParam.Config(
        child=Linear.Config(in_features=4, out_features=4),
        dim=4,
    )

    model_config = LoRATransform(
        rank=2,
        alpha=4.0,
        target_modules=["child"],
    ).transform(model_config)
    model = model_config.build()

    assert not model.direct.requires_grad
    assert not model.child.weight.requires_grad
    assert model.child.lora_a.weight.requires_grad
    assert model.child.lora_b.weight.requires_grad

    non_lora_trainable = {
        n
        for n, p in model.named_parameters()
        if p.requires_grad and "lora_a" not in n and "lora_b" not in n
    }
    assert non_lora_trainable == set()


def test_lora_preserves_frozen_config_type_checks():
    """Frozen non-LoRA configs still satisfy checks for their original type."""

    class AttentionHolder(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            inner_attention: Module.Config
            proj: Linear.Config

        def __init__(self, config: Config) -> None:
            super().__init__()
            self.inner_attention = config.inner_attention.build()
            self.proj = config.proj.build()

    model_config = AttentionHolder.Config(
        inner_attention=FlexInnerAttention.Config(),
        proj=Linear.Config(in_features=4, out_features=4),
    )

    model_config = LoRATransform(
        rank=2,
        alpha=4.0,
        target_modules=["proj"],
    ).transform(model_config)

    assert isinstance(model_config.inner_attention, FlexInnerAttention.Config)
    model = model_config.build()
    assert not model.proj.weight.requires_grad
    assert model.proj.lora_a.weight.requires_grad
    assert model.proj.lora_b.weight.requires_grad


class TestLoRATensorParallelTypecheck(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @parametrize("enable_sp", [True, False])
    @with_comms
    def test_row_parallel_adapters_typecheck(self, enable_sp: bool):
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=enable_sp,
        )
        with patch(
            "torchtitan.distributed.parallelism_context.device_type", self.device_type
        ):
            parallelism_context.build_mesh()
        init = {"weight": torch.nn.init.normal_}
        ffn_config = make_ffn_config(
            dim=16,
            hidden_dim=32,
            w13_param_init=init,
            w2_param_init=init,
            enable_sp=enable_sp,
        )
        feed_forward = (
            LoRATransform(rank=2, alpha=4.0)
            .transform(ffn_config)
            .build()
            .to(self.device_type)
        )
        feed_forward._parallelize(parallelism_context)

        x_local = torch.randn(8, 16, device=self.device_type)
        mesh = parallelism_context.spmd_dense_mesh()
        set_spmd_meshes(
            dense_mesh=mesh,
            sparse_mesh=None,
            dense_sp_enabled=parallelism_context.sp_enabled,
        )
        residual = residual_activation_placement(enable_sp=enable_sp)
        with set_current_spmd_mesh(mesh), typecheck(local=False):
            spmd.assert_type(x_local, residual)
            output = feed_forward(x_local)
            spmd.assert_type(output, residual)
            output.sum().backward()

        self.assertEqual(output.shape, x_local.shape)
        self.assertIsNotNone(feed_forward.w2.lora_b.weight.grad)

    @with_comms
    def test_vision_adapters_match_unsharded(self):
        """TP2 LoRA output and adapter grads match a TP1 run of the same weights.

        Vision linears are CP-invariant; the vision adapter is also TP-invariant.
        """
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        with patch(
            "torchtitan.distributed.parallelism_context.device_type", self.device_type
        ):
            parallelism_context.build_mesh()
        vision_config = VisionMLP.Config(
            fc1=Linear.Config(
                in_features=16,
                out_features=32,
                bias=True,
                sharding_config=vision_colwise_sharding_config(),
            ),
            fc2=InvariantRowParallelLinear.Config(
                in_features=32,
                out_features=16,
                # With a bias, the forward skips SPMD type checking.
                bias=False,
                sharding_config=vision_rowwise_sharding_config(),
            ),
        )
        adapter_config = Linear.Config(
            in_features=16,
            out_features=16,
            sharding_config=vision_invariant_linear_sharding_config(),
        )
        torch.manual_seed(0)
        unsharded = [
            LoRATransform(rank=2, alpha=4.0)
            .transform(config)
            .build()
            .to(self.device_type)
            for config in (vision_config, adapter_config)
        ]
        for module in unsharded:
            for param in module.parameters():
                torch.nn.init.normal_(param)
        sharded = copy.deepcopy(unsharded)
        x_local = torch.randn(8, 16, device=self.device_type)

        expected = x_local
        for module in unsharded:
            expected = module(expected)
        expected.sum().backward()

        for module in sharded:
            module._parallelize(parallelism_context)
        mesh = parallelism_context.spmd_dense_mesh()
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=None, dense_sp_enabled=False)
        layout = SpmdType({DP: spmd.V, CP: spmd.I, TP: spmd.I})
        with set_current_spmd_mesh(mesh), typecheck(local=False):
            spmd.assert_type(x_local, layout)
            output = x_local
            for module in sharded:
                output = module(output)
            spmd.assert_type(output, layout)
            output.sum().backward()

        # Explicit tolerance: TP2 sums fp32 partial products in a different order.
        tolerance = {"atol": 1e-4, "rtol": 1e-4}
        torch.testing.assert_close(output, expected, **tolerance)
        for module, reference in zip(sharded, unsharded, strict=True):
            for name, param in module.named_parameters():
                if "lora_" not in name:
                    assert param.grad is None, name
                    continue
                expected_grad = reference.get_parameter(name).grad
                adapter = module.get_submodule(name.rsplit(".", 1)[0])
                tp_type = adapter._sharding_config.state_shardings["weight"].local_type[
                    TP
                ]
                grad = param.grad
                # Without FSDP the TP-replicated grad stays partial. In training,
                # FSDP redistributes the non-DP Partial grad to Replicate (TP
                # all-reduce).
                if tp_type == spmd.R:
                    torch.distributed.all_reduce(grad)
                elif isinstance(tp_type, spmd.Shard):
                    shards = [torch.empty_like(grad) for _ in range(self.world_size)]
                    torch.distributed.all_gather(shards, grad.contiguous())
                    grad = torch.cat(shards, dim=tp_type.dim)
                torch.testing.assert_close(grad, expected_grad, **tolerance, msg=name)


instantiate_parametrized_tests(TestLoRATensorParallelTypecheck)
