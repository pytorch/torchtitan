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


def _lora_llama_model(rank=4, alpha=8.0):
    model_spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(rank=rank, alpha=alpha, target_modules=["wqkv", "wo"]),
        ],
    )
    model = model_spec.model.build()
    model.init_states()
    return model


def test_trainable_state_dict_is_exactly_the_adapters():
    from torchtitan.components.lora import trainable_state_dict

    model = _lora_llama_model()
    trainable = trainable_state_dict(model)
    assert set(trainable) == {
        n for n, _ in model.named_parameters() if "lora_a" in n or "lora_b" in n
    }
    assert len(trainable) > 0


def test_merge_lora_state_dict_keys_and_zero_init_identity():
    """With lora_b zero-initialized the merged weights EQUAL the base, and the
    merged dict carries the ORIGINAL key set -- no adapter keys, every base key
    intact."""
    from torchtitan.components.lora import merge_lora_state_dict

    model = _lora_llama_model()
    raw = model.state_dict()
    merged = merge_lora_state_dict(model)

    assert not any("lora_a" in k or "lora_b" in k for k in merged)
    assert set(merged) == {k for k in raw if "lora_a" not in k and "lora_b" not in k}
    for k, v in merged.items():
        torch.testing.assert_close(v, raw[k], rtol=0, atol=0)


def test_merge_lora_state_dict_folds_the_delta():
    """After perturbing an adapter pair, merged W == W_base + (alpha/rank) B @ A,
    and a plain linear loaded with the merged weight reproduces the LoRA
    module's forward."""
    torch.manual_seed(0)
    from torchtitan.components.lora import LoRALinearBase, merge_lora_state_dict

    model = _lora_llama_model(rank=4, alpha=8.0)
    name, module = next(
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, LoRALinearBase) and n.endswith("wo")
    )
    with torch.no_grad():
        module.lora_a.weight.normal_()
        module.lora_b.weight.normal_()
    before = {k: v.clone() for k, v in model.state_dict().items()}

    merged = merge_lora_state_dict(model)
    expected = (
        module.weight.float()
        + module._lora_scaling
        * (module.lora_b.weight.float() @ module.lora_a.weight.float())
    ).to(module.weight.dtype)
    torch.testing.assert_close(merged[f"{name}.weight"], expected)

    x = torch.randn(3, module.weight.shape[1])
    plain = torch.nn.functional.linear(x, merged[f"{name}.weight"])
    torch.testing.assert_close(plain, module(x), rtol=2e-5, atol=2e-5)

    # The model itself is unchanged: merging happened on clones.
    after = model.state_dict()
    assert set(after) == set(before)
    for k in before:
        torch.testing.assert_close(after[k], before[k], rtol=0, atol=0)


def test_merge_lora_state_dict_respects_serialization_hooks():
    """The fused attention linear exports split wq/wk/wv keys through a
    state-dict hook; the merged delta must land in THOSE keys, not a composed
    wqkv.weight nothing recognises."""
    torch.manual_seed(0)
    from torchtitan.components.lora import LoRALinearBase, merge_lora_state_dict

    model = _lora_llama_model(rank=4, alpha=8.0)
    name, module = next(
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, LoRALinearBase) and n.endswith("wqkv")
    )
    with torch.no_grad():
        module.lora_a.weight.normal_()
        module.lora_b.weight.normal_()
    raw = model.state_dict()

    merged = merge_lora_state_dict(model)
    parent = name.rsplit(".", 1)[0]
    assert f"{name}.weight" not in merged
    changed = [
        f"{parent}.{p}.weight"
        for p in ("wq", "wk", "wv")
        if not torch.equal(merged[f"{parent}.{p}.weight"], raw[f"{parent}.{p}.weight"])
    ]
    assert changed, "perturbed fused adapters left every split key unchanged"


def test_merge_lora_state_dict_sees_through_wrappers():
    """An activation-checkpoint wrapper changes named_modules() paths but not
    state_dict() keys; the merge must key by the latter."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointWrapper,
    )

    from torchtitan.components.lora import merge_lora_state_dict

    model = _lora_llama_model()
    model.layers["0"] = CheckpointWrapper(model.layers["0"])

    merged = merge_lora_state_dict(model)
    assert not any("lora_a" in k or "lora_b" in k for k in merged)
    assert not any("_checkpoint_wrapped_module" in k for k in merged)
    assert any(k.startswith("layers.0.") for k in merged)


def test_qlora_nf4_pack_forward_merge():
    """Packed bases: forward runs on NF4, and with zero-init lora_b the merge
    equals the DEQUANTIZED base exactly (QLoRA is lossy vs bf16 by design)."""
    torch.manual_seed(0)
    pytest.importorskip("torchao.dtypes.nf4tensor")
    from torchao.dtypes.nf4tensor import NF4Tensor

    from torchtitan.components.lora import (
        LoRALinearBase,
        merge_lora_state_dict,
        quantize_lora_bases,
    )

    model = _lora_llama_model(rank=4, alpha=8.0)
    packed = quantize_lora_bases(model)
    assert packed > 0
    quantized = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, LoRALinearBase) and isinstance(m.weight, NF4Tensor)
    ]
    assert len(quantized) == packed

    name, module = next((n, m) for n, m in quantized if n.endswith("wo"))
    x = torch.randn(3, module.lora_a.weight.shape[1])
    y = module(x)
    assert y.shape == (3, module.lora_b.weight.shape[0])

    merged = merge_lora_state_dict(model)
    dequant = module.weight.get_original_weight()
    torch.testing.assert_close(merged[f"{name}.weight"], dequant, rtol=0, atol=0)
    # The NF4 param object is back in place after the export.
    assert isinstance(module.weight, NF4Tensor)


def test_qlora_config_packs_at_init():
    """quantize_base='nf4' on the converter packs the bases when init_states
    runs (unparallelized build; FSDP-managed bases are refused by design)."""
    pytest.importorskip("torchao.dtypes.nf4tensor")
    from torchao.dtypes.nf4tensor import NF4Tensor

    from torchtitan.components.lora import LoRALinearBase

    model_spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=4, alpha=8.0, target_modules=["wo"], quantize_base="nf4"
            ),
        ],
    )
    model = model_spec.model.build()
    model.init_states()
    mods = [m for _, m in model.named_modules() if isinstance(m, LoRALinearBase)]
    assert mods and all(isinstance(m.weight, NF4Tensor) for m in mods)


def test_qlora_mxfp4_packs_at_build_and_merges():
    """quantize_base='mxfp4' swaps the base for split storage AT BUILD (the
    pack-then-shard order), the forward dequantizes, and the merge emits the
    original weight key with the packed keys gone."""
    pytest.importorskip("torchao.prototype.mx_formats.mx_tensor")
    torch.manual_seed(0)

    from torchtitan.components.lora import LoRALinearBase, merge_lora_state_dict

    model_spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=4, alpha=8.0, target_modules=["wo"], quantize_base="mxfp4"
            ),
        ],
    )
    model = model_spec.model.build()
    model.init_states()

    mods = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinearBase)]
    assert mods
    for _, m in mods:
        assert "weight" not in m._parameters
        assert m.base_qdata.dtype == torch.uint8
        assert m.base_scale.dtype == torch.uint8
        assert m.base_qdata.shape[1] * 2 == m.lora_a.weight.shape[1]

    name, module = mods[0]
    x = torch.randn(3, module.lora_a.weight.shape[1])
    y = module(x)
    assert y.shape == (3, module.lora_b.weight.shape[0])
    # The packed base is not all zeros after init.
    assert module.base_qdata.abs().sum() > 0

    merged = merge_lora_state_dict(model)
    assert f"{name}.weight" in merged
    assert f"{name}.base_qdata" not in merged
    assert f"{name}.base_scale" not in merged
    # lora_b is zero-initialized, so merged == the dequantized base exactly,
    # and the packed layout is back in place afterwards.
    torch.testing.assert_close(
        merged[f"{name}.weight"],
        module._dequant_base_mxfp4().to(merged[f"{name}.weight"].dtype),
        rtol=0,
        atol=0,
    )
    assert "weight" not in module._parameters
