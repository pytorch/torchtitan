# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import cast

import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
import torchtitan.config.transform.quantization as quantization_transform

from torchtitan.config import ConfigManager
from torchtitan.config.transform import (
    Float8LinearConverter,
    GroupedLinearLoRAHandler,
    LinearLoRAHandler,
    LoRATransform,
    transform_model_config_,
)
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    dense_sequence_parallel_placement,
    set_dense_ffn_sharding,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    GroupedLinear,
    Linear,
    RowParallelLinear,
)
from torchtitan.models.common.moe_sharding import expert_param_placement_sparse
from torchtitan.models.common.vision_encoder import InvariantRowParallelLinear
from torchtitan.models.gpt_oss.moe import GptOssGroupedLinear
from torchtitan.models.llama3 import model_registry
from torchtitan.models.qwen3_5.model import Qwen35Model
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.quantization import Float8Linear
from torchtitan.trainer import Trainer


LINEAR_LORA_HANDLERS = (LinearLoRAHandler(),)
GROUPED_LINEAR_LORA_HANDLERS = (GroupedLinearLoRAHandler(),)


def test_qwen35_moe_float8_lora_model_config(monkeypatch):
    pytest.importorskip("torchao")
    from torchtitan.quantization.float8.experts import _float8_grouped_linear_cache

    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    config = cast(
        Trainer.Config,
        ConfigManager().parse_args(
            [
                "--module",
                "qwen3_5",
                "--config",
                "qwen35_debugmodel_moe_float8_lora",
            ]
        ),
    )
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
        f"layers.{layer}.moe.shared_experts.{projection}"
        for layer in range(num_layers)
        for projection in ("w13", "w2")
    }
    assert set(grouped_lora) == {
        f"layers.{layer}.moe.routed_experts.{projection}"
        for layer in range(num_layers)
        for projection in ("w13", "w2")
    }
    assert Float8Linear is not None
    assert all(
        isinstance(projection, Float8Linear.Config)
        for projection in dense_lora.values()
    )
    float8_grouped_configs = tuple(
        cls.Config for cls in _float8_grouped_linear_cache.values()
    )
    assert all(
        isinstance(projection, float8_grouped_configs)
        for projection in grouped_lora.values()
    )


def test_lora_model_builds():
    """LoRA debug model builds, has trainable adapters and frozen base."""
    model_config = model_registry("debugmodel")
    model_config = transform_model_config_(
        model_config,
        [
            LoRATransform(
                handlers=LINEAR_LORA_HANDLERS,
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
    model_config = model_registry("debugmodel")
    model_config = transform_model_config_(
        model_config,
        [
            LoRATransform(
                handlers=LINEAR_LORA_HANDLERS,
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
    attention_masks = model.get_attention_masks(positions)
    # The default attention backend is FlexInnerAttention, which does not support
    # backward on CPU; this is a forward-only shape check, so run under no_grad.
    with torch.no_grad():
        output = model(tokens, attention_masks=attention_masks, positions=positions)
    assert output.shape == (num_tokens, vocab_size)


def test_lora_targets_fused_feed_forward_projection():
    """The physical w13 projection uses one LoRA adapter."""
    init = {"weight": torch.nn.init.ones_}
    config = FeedForward.Config(
        w13=Linear.Config(
            in_features=4, out_features=8, num_linears=2, param_init=init
        ),
        w2=Linear.Config(in_features=8, out_features=4, param_init=init),
    )
    config = LoRATransform(
        handlers=LINEAR_LORA_HANDLERS,
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
    config = LoRATransform(
        handlers=GROUPED_LINEAR_LORA_HANDLERS,
        rank=8,
        alpha=16.0,
    ).transform(
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


def test_grouped_lora_preserves_specialized_projection():
    config = LoRATransform(handlers=GROUPED_LINEAR_LORA_HANDLERS, rank=8,).transform(
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


def test_grouped_lora_wraps_quantized_grouped_mm(monkeypatch):
    pytest.importorskip("torchao")
    from torchtitan.quantization.float8.experts import _get_float8_grouped_linear_cls

    float8_cls = _get_float8_grouped_linear_cls(GroupedLinear)
    base_called = False

    def grouped_mm(module, *, input_RI, weight_EOI, offsets_E):
        nonlocal base_called
        del module, offsets_E
        base_called = True
        return input_RI.new_zeros(input_RI.shape[0], weight_EOI.shape[-2])

    monkeypatch.setattr(float8_cls, "_grouped_mm", grouped_mm)
    config = LoRATransform(
        handlers=GROUPED_LINEAR_LORA_HANDLERS,
        rank=8,
        alpha=16.0,
    ).transform(
        float8_cls.Config(
            group_size=2,
            in_features=16,
            out_features=16,
            param_init={"weight": torch.nn.init.ones_},
        )
    )
    grouped = config.build()
    grouped.init_states()
    with torch.no_grad():
        grouped.lora_a.weight.fill_(1)
        grouped.lora_b.weight.fill_(1)

    output = grouped(
        torch.ones(8, 16, dtype=torch.bfloat16),
        torch.tensor([4, 8], dtype=torch.int32),
    )

    assert base_called
    assert isinstance(grouped, float8_cls)
    assert torch.count_nonzero(output) == output.numel()


def test_grouped_lora_a_uses_linear_fan_in():
    config = LoRATransform(handlers=GROUPED_LINEAR_LORA_HANDLERS, rank=8,).transform(
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
        LoRATransform(handlers=GROUPED_LINEAR_LORA_HANDLERS, rank=rank).transform(
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
    config = LoRATransform(handlers=GROUPED_LINEAR_LORA_HANDLERS, rank=8).transform(
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
    config = LoRATransform(handlers=GROUPED_LINEAR_LORA_HANDLERS, rank=8).transform(
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

    with pytest.raises(ValueError, match="only expert-axis parameter sharding"):
        config.build()


def test_stacked_lora_adapter_does_not_repeat_base_redistribution():
    """The LoRA adapters inherit state sharding, not TP collectives."""
    init = {"weight": torch.nn.init.zeros_}
    config = make_ffn_config(
        dim=4,
        hidden_dim=8,
        w1_param_init=init,
        w2w3_param_init=init,
    )
    config = LoRATransform(
        handlers=LINEAR_LORA_HANDLERS,
        rank=2,
        alpha=4,
        target_modules=["w13"],
    ).transform(config)
    assert isinstance(config, FeedForward.Config)
    set_dense_ffn_sharding(
        config,
        attn_x_layout=dense_sequence_parallel_placement(),
        enable_sp=True,
    )

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


def test_float8_lora_targets_fused_feed_forward_projection():
    """Quantized w13 uses one LoRA adapter."""
    pytest.importorskip("torchao")
    from torchtitan.quantization import Float8Linear

    if Float8Linear is None:
        pytest.skip("torchao Float8Linear is unavailable")

    init = {"weight": torch.nn.init.ones_}
    config = FeedForward.Config(
        w13=Linear.Config(
            in_features=16, out_features=32, num_linears=2, param_init=init
        ),
        w2=Linear.Config(in_features=32, out_features=16, param_init=init),
    )
    config = Float8LinearConverter(
        Float8LinearConverter.Config(emulate=True, model_compile_enabled=False)
    ).convert(config)
    config = LoRATransform(
        handlers=LINEAR_LORA_HANDLERS,
        rank=4,
        alpha=8.0,
        target_modules=["w13"],
    ).transform(config)
    feed_forward = config.build()
    feed_forward.init_states()

    assert isinstance(feed_forward.w13, Float8Linear)
    assert set(feed_forward.state_dict()) == {
        "w13.weight",
        "w2.weight",
        "w13.lora_a.weight",
        "w13.lora_b.weight",
    }
    assert not feed_forward.w13.weight.requires_grad
    assert feed_forward.w13.lora_a.weight.requires_grad
    assert feed_forward.w13.lora_b.weight.requires_grad
    assert feed_forward(torch.randn(2, 16)).shape == (2, 16)


def test_lora_class_is_reused_for_the_same_parent():
    """The LoRA class is cached for each parent Linear class."""
    first = LoRATransform(handlers=LINEAR_LORA_HANDLERS, rank=2, alpha=4.0).transform(
        Linear.Config(in_features=4, out_features=3)
    )
    second = LoRATransform(handlers=LINEAR_LORA_HANDLERS, rank=2, alpha=4.0).transform(
        Linear.Config(in_features=4, out_features=3)
    )

    assert type(first) is type(second)
    assert first._owner is second._owner
    assert first._owner is not None
    assert first._owner.__name__ == "LoRALinear"
    assert issubclass(first._owner, Linear)


def test_lora_handler_matches_linear_config_subclass():
    class AlternateLinear(Linear):
        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            pass

    config = AlternateLinear.Config(in_features=4, out_features=3)
    transformed = LoRATransform(
        handlers=LINEAR_LORA_HANDLERS, rank=2, alpha=4.0
    ).transform(config)
    model = transformed.build()

    assert isinstance(model, AlternateLinear)
    assert not model.weight.requires_grad
    assert model.lora_a.weight.requires_grad
    assert model.lora_b.weight.requires_grad


def test_lora_preserves_invariant_row_parallel_linear():
    config = InvariantRowParallelLinear.Config(
        in_features=4,
        out_features=3,
        bias=True,
    )
    transformed = LoRATransform(
        handlers=LINEAR_LORA_HANDLERS,
        rank=2,
        alpha=4.0,
    ).transform(config)
    linear = transformed.build()

    assert isinstance(linear, InvariantRowParallelLinear)
    x = torch.randn(5, 4)
    expected = F.linear(x, linear.weight, linear.bias)
    expected += 2 * linear.lora_b(linear.lora_a(x))
    torch.testing.assert_close(linear(x), expected)


def test_lora_transform_rejects_duplicate_handler_type():
    with pytest.raises(ValueError, match="is shadowed by earlier handler"):
        LoRATransform(
            handlers=(LinearLoRAHandler(), LinearLoRAHandler()),
        )


def test_lora_transform_requires_handlers():
    with pytest.raises(TypeError, match="handlers"):
        LoRATransform()


def test_lora_transform_rejects_handler_shadowed_by_superclass():
    class SpecializedLinear(Linear):
        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            pass

    class SpecializedLinearHandler:
        config_type = SpecializedLinear.Config

        def make_config(self, cfg, *, rank, alpha):
            return cfg

    with pytest.raises(ValueError, match="is shadowed by earlier handler"):
        LoRATransform(
            handlers=(LinearLoRAHandler(), SpecializedLinearHandler()),
        )


def test_lora_transform_accepts_specialized_handler_before_superclass():
    class SpecializedLinear(Linear):
        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            pass

    class SpecializedLinearHandler:
        config_type = SpecializedLinear.Config

        def make_config(self, cfg, *, rank, alpha):
            return cfg

    LoRATransform(
        handlers=(SpecializedLinearHandler(), LinearLoRAHandler()),
    )


def test_lora_rank_validation():
    """LoRA rank must be positive."""
    with pytest.raises(ValueError, match="rank must be positive"):
        LoRATransform(handlers=LINEAR_LORA_HANDLERS, rank=0)
    with pytest.raises(ValueError, match="rank must be positive"):
        LoRATransform(handlers=LINEAR_LORA_HANDLERS, rank=-1)


def test_multiple_lora_transforms_conflict():
    model_config = model_registry("debugmodel")

    with pytest.raises(ValueError, match="cannot be combined"):
        transform_model_config_(
            model_config,
            [
                LoRATransform(
                    handlers=LINEAR_LORA_HANDLERS,
                    rank=2,
                    alpha=4.0,
                    target_modules=["wqkv"],
                ),
                LoRATransform(
                    handlers=LINEAR_LORA_HANDLERS,
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
        handlers=LINEAR_LORA_HANDLERS,
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
        handlers=LINEAR_LORA_HANDLERS,
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
        handlers=LINEAR_LORA_HANDLERS,
        rank=2,
        alpha=4.0,
        target_modules=["proj"],
    ).transform(model_config)

    assert isinstance(model_config.inner_attention, FlexInnerAttention.Config)
    model = model_config.build()
    assert not model.proj.weight.requires_grad
    assert model.proj.lora_a.weight.requires_grad
    assert model.proj.lora_b.weight.requires_grad


def test_lora_transform_handlers_support_multiple_projection_types():
    """One transform can adapt projections from unrelated class hierarchies."""

    class HeadwiseProjection(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            num_heads: int
            in_features: int
            out_features: int

        def __init__(self, config: Config) -> None:
            super().__init__()
            self.num_heads = config.num_heads
            self.out_features = config.out_features
            self.weight = torch.nn.Parameter(
                torch.empty(
                    config.num_heads,
                    config.out_features,
                    config.in_features,
                )
            )

        def forward(  # pyrefly: ignore [bad-override]
            self, input: torch.Tensor
        ) -> torch.Tensor:
            return torch.einsum("...hi,hoi->...ho", input, self.weight)

    class LoRAHeadwiseProjection(HeadwiseProjection):
        @dataclass(kw_only=True, slots=True)
        class Config(HeadwiseProjection.Config):
            rank: int
            alpha: float

        def __init__(self, config: Config) -> None:
            super().__init__(config)
            self.weight.requires_grad_(False)
            self.scaling = config.alpha / config.rank
            self.lora_a = torch.nn.Parameter(
                torch.randn(config.in_features, config.rank)
            )
            self.lora_b = torch.nn.Parameter(
                torch.zeros(config.num_heads, config.rank, config.out_features)
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            base = super().forward(input)
            hidden = input @ self.lora_a
            adapter = torch.einsum("...hr,hro->...ho", hidden, self.lora_b)
            return base + self.scaling * adapter

    class HeadwiseProjectionLoRAHandler:
        config_type = HeadwiseProjection.Config

        def make_config(
            self,
            cfg: Module.Config,
            *,
            rank: int,
            alpha: float,
        ) -> Module.Config:
            assert isinstance(cfg, HeadwiseProjection.Config)
            return LoRAHeadwiseProjection.Config(
                num_heads=cfg.num_heads,
                in_features=cfg.in_features,
                out_features=cfg.out_features,
                rank=rank,
                alpha=alpha,
            )

    class Root(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            projection: HeadwiseProjection.Config
            linear: Linear.Config

        def __init__(self, config: Config) -> None:
            super().__init__()
            self.projection = config.projection.build()
            self.linear = config.linear.build()

    config = Root.Config(
        projection=HeadwiseProjection.Config(
            num_heads=2,
            in_features=4,
            out_features=3,
        ),
        linear=Linear.Config(in_features=4, out_features=3),
    )
    converted = LoRATransform(
        handlers=(LinearLoRAHandler(), HeadwiseProjectionLoRAHandler()),
        rank=2,
        alpha=4.0,
    ).transform(config)
    model = converted.build()
    output = model.projection(torch.randn(5, 2, 4))

    assert output.shape == (5, 2, 3)
    assert not model.projection.weight.requires_grad
    assert model.projection.lora_a.requires_grad
    assert model.projection.lora_b.requires_grad
    assert not model.linear.weight.requires_grad
    assert model.linear.lora_a.weight.requires_grad
    assert model.linear.lora_b.weight.requires_grad
