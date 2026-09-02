# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import pytest
import torch

from torchtitan.components.lora import _get_lora_cls, LoRAConverter
from torchtitan.components.quantization import Float8LinearConverter
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.llama3 import model_registry
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.module import Module


def test_lora_model_builds():
    """LoRA debug model builds, has trainable adapters and frozen base."""
    model_spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(rank=8, alpha=16.0, target_modules=["wqkv", "wo"]),
        ],
    )
    model = model_spec.model.build()
    model.init_states()

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
    model_spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(rank=8, alpha=16.0, target_modules=["wqkv", "wo"]),
        ],
    )
    model = model_spec.model.build()
    model.init_states()

    vocab_size = model_spec.model.vocab_size
    num_documents, seq_len = 2, 16
    num_tokens = num_documents * seq_len
    tokens = torch.randint(0, vocab_size, (num_tokens,))
    positions = torch.arange(seq_len).repeat(num_documents)
    attention_masks = model.get_attention_masks(positions)
    # The default attention backend is FlexAttention, which does not support
    # backward on CPU; this is a forward-only shape check, so run under no_grad.
    with torch.no_grad():
        output = model(tokens, attention_masks=attention_masks, positions=positions)
    assert output.shape == (num_tokens, vocab_size)


def test_validate_converter_order():
    """Quantization before LoRA is valid; LoRA before quantization is not."""
    lora_cfg = LoRAConverter.Config(rank=8, alpha=16.0)

    # Valid order: no error
    validate_converter_order([lora_cfg])

    # Invalid order: quantization after LoRA
    float8_cfg = Float8LinearConverter.Config(emulate=True)
    with pytest.raises(ValueError, match="must be applied before"):
        validate_converter_order([lora_cfg, float8_cfg])

    # Valid order: quantization before LoRA
    validate_converter_order([float8_cfg, lora_cfg])


def test_lora_cls_cache():
    """Dynamic LoRA class creation is cached per parent class."""
    cls1 = _get_lora_cls(Linear)
    cls2 = _get_lora_cls(Linear)
    assert cls1 is cls2
    assert cls1.__name__ == "LoRALinear"
    assert issubclass(cls1, Linear)


def test_lora_rank_validation():
    """LoRA rank must be positive."""
    with pytest.raises(ValueError, match="rank must be positive"):
        LoRAConverter(LoRAConverter.Config(rank=0))
    with pytest.raises(ValueError, match="rank must be positive"):
        LoRAConverter(LoRAConverter.Config(rank=-1))


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

    model_config = LoRAConverter(
        LoRAConverter.Config(rank=2, alpha=4.0, target_modules=["child"])
    ).convert(model_config)
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

    model_config = LoRAConverter(
        LoRAConverter.Config(rank=2, alpha=4.0, target_modules=["child"])
    ).convert(model_config)
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
        inner_attention=FlexAttention.Config(),
        proj=Linear.Config(in_features=4, out_features=4),
    )

    model_config = LoRAConverter(
        LoRAConverter.Config(rank=2, alpha=4.0, target_modules=["proj"])
    ).convert(model_config)

    assert isinstance(model_config.inner_attention, FlexAttention.Config)
    model = model_config.build()
    assert not model.proj.weight.requires_grad
    assert model.proj.lora_a.weight.requires_grad
    assert model.proj.lora_b.weight.requires_grad


def test_lora_converter_extension_supports_non_linear_projection():
    """A subclass can adapt projection configs outside the Linear hierarchy."""

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

    class ExtendedLoRAConverter(LoRAConverter):
        @dataclass(kw_only=True, slots=True)
        class Config(LoRAConverter.Config):
            pass

        def _supports_lora(self, cfg: Module.Config) -> bool:
            return super()._supports_lora(cfg) or isinstance(
                cfg, HeadwiseProjection.Config
            )

        def _make_lora_config(self, cfg: Module.Config) -> Module.Config:
            if not isinstance(cfg, HeadwiseProjection.Config):
                return super()._make_lora_config(cfg)
            return LoRAHeadwiseProjection.Config(
                num_heads=cfg.num_heads,
                in_features=cfg.in_features,
                out_features=cfg.out_features,
                rank=self.rank,
                alpha=self.alpha,
            )

    class Root(Module):
        @dataclass(kw_only=True, slots=True)
        class Config(Module.Config):
            projection: HeadwiseProjection.Config

        def __init__(self, config: Config) -> None:
            super().__init__()
            self.projection = config.projection.build()

    config = Root.Config(
        projection=HeadwiseProjection.Config(
            num_heads=2,
            in_features=4,
            out_features=3,
        )
    )
    converted = ExtendedLoRAConverter.Config(rank=2, alpha=4.0).build().convert(config)
    model = converted.build()
    output = model.projection(torch.randn(5, 2, 4))

    assert output.shape == (5, 2, 3)
    assert not model.projection.weight.requires_grad
    assert model.projection.lora_a.requires_grad
    assert model.projection.lora_b.requires_grad
