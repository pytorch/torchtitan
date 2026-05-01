# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torch.distributed.tensor import Replicate

from torchtitan.config import Configurable
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.logging import logger


def _lora_adapter_sharding(
    base_sharding: ShardingConfig | None,
) -> tuple[ShardingConfig, ShardingConfig]:
    """Derive LoRA adapter sharding from the base linear's TP sharding.

    Both adapters are TP-sharded to match the base linear, ensuring all
    LoRA gradients have the same TP placement — no mixed Replicate/Shard
    norms in ``clip_grad_norm_``.

    ``lora_b`` mirrors the base weight's TP shard so its output matches
    ``base_out``'s placement (Shard(-1) for colwise, Partial for rowwise).

    ``lora_a`` weight is Replicate (small rank dim, no benefit from TP
    sharding).

    Note: under TP, ``clip_grad_norm_`` reports a grad_norm that differs
    from single-GPU by a constant factor (TP degree). This is a known
    DTensor ``_NormPartial`` norm computation issue with mixed
    Replicate/Shard placements, not a LoRA correctness bug — the loss
    and convergence are identical.
    """
    replicate_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Replicate())},
    )
    if base_sharding is None:
        return replicate_weight, replicate_weight

    base_weight_sharding = base_sharding.state_shardings.get("weight")
    if base_weight_sharding is None:
        return replicate_weight, replicate_weight

    # lora_b: same TP shard as base weight → output matches base_out
    lora_b_sharding = ShardingConfig(
        state_shardings={"weight": base_weight_sharding},
    )
    return replicate_weight, lora_b_sharding


# Cache for dynamically created LoRA classes
_lora_class_cache: dict[type, type] = {}


def _get_lora_cls(parent_cls: type) -> type:
    """Get or create a LoRA subclass for *parent_cls* (e.g. Linear, Float8Linear).

    The returned class has a proper ``__init__`` that initialises both the
    base linear and the LoRA adapters in a single construction — no
    build-then-mutate.
    """
    if parent_cls in _lora_class_cache:
        return _lora_class_cache[parent_cls]

    class LoRALinear(parent_cls):  # type: ignore[valid-type, misc]
        def __init__(
            self,
            config: Linear.Config,
            *,
            _lora_rank: int,
            _lora_alpha: float,
            _base_sharding: ShardingConfig | None = None,
        ) -> None:
            super().__init__(config)
            # Freeze base weight — only adapters should be trainable
            for param in self.parameters():
                param.requires_grad_(False)
            self._lora_scaling = _lora_alpha / _lora_rank

            lora_a_sharding, lora_b_sharding = _lora_adapter_sharding(_base_sharding)
            self.lora_a = Linear.Config(
                in_features=config.in_features,
                out_features=_lora_rank,
                bias=False,
                sharding_config=lora_a_sharding,
                param_init={
                    "weight": lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5)),
                },
            ).build()
            self.lora_b = Linear.Config(
                in_features=_lora_rank,
                out_features=config.out_features,
                bias=False,
                sharding_config=lora_b_sharding,
                param_init={"weight": nn.init.zeros_},
            ).build()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            base_out = super().forward(input)
            lora_out = self.lora_b(self.lora_a(input))
            return base_out + self._lora_scaling * lora_out

    LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
    LoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
    _lora_class_cache[parent_cls] = LoRALinear
    return LoRALinear


@dataclass(kw_only=True, slots=True)
class LoRAConfig(Configurable.Config):
    """Config wrapper that applies LoRA to any Linear.Config at build time.

    Wraps an inner ``Linear.Config`` (which may be ``Float8Linear.Config``,
    ``MXFP8Linear.Config``, or plain ``Linear.Config``) and builds a
    LoRA subclass directly — one ``__init__`` that initialises both the
    base linear and the LoRA adapters.

    Delegates unknown attribute access (e.g. ``sharding_config``) to the
    inner config so that sharding and other config-level operations work
    transparently through the wrapper.
    """

    inner: Linear.Config
    """The original Linear config (preserved for composition with quantization)."""

    rank: int = 16
    alpha: float = 16.0

    def __getattr__(self, name: str):
        try:
            inner = object.__getattribute__(self, "inner")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(inner, name)

    def __setattr__(self, name: str, value) -> None:
        if name in type(self).__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def build(self, **kwargs):
        from dataclasses import replace

        inner = self.inner
        assert inner._owner is not None
        lora_cls = _get_lora_cls(inner._owner)
        instance = lora_cls(
            config=replace(inner),
            _lora_rank=self.rank,
            _lora_alpha=self.alpha,
            _base_sharding=inner.sharding_config,
        )
        # Apply Module.Config properties that inner.build() would set
        if inner.param_init is not None:
            instance._param_init = inner.param_init
        if inner.sharding_config is not None:
            instance._sharding_config = inner.sharding_config
        return instance


@dataclass(kw_only=True, slots=True)
class FrozenConfig(Configurable.Config):
    """Config wrapper that freezes all parameters at build time.

    Works with any ``Module.Config`` — used by ``LoRAConverter`` to freeze
    non-target modules (linears, norms, embeddings) so that only LoRA
    adapter parameters remain trainable.
    """

    inner: Configurable.Config

    def __getattr__(self, name: str):
        try:
            inner = object.__getattribute__(self, "inner")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(inner, name)

    def __setattr__(self, name: str, value) -> None:
        if name in type(self).__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def build(self, **kwargs):
        instance = self.inner.build(**kwargs)
        instance.requires_grad_(False)
        return instance


class LoRAConverter(Configurable):
    """Apply LoRA adapters to Linear layers in a model.

    Operates on the model config tree: walks for ``Linear.Config`` entries
    and wraps them in ``LoRAConfig``. The base module is built first by the
    inner config, then ``apply_lora`` dynamically wraps it — preserving
    composition with quantization (Float8, MX, etc.).

    When ``target_modules`` is None (default), every ``Linear.Config`` is
    wrapped.  When specified, only configs whose FQN's last segment matches
    one of the entries are converted (e.g. ``["wq", "wv"]``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        rank: int = 8
        """Rank of the LoRA matrices."""

        alpha: float = 16.0
        """Scaling factor. Output is scaled by alpha/rank."""

        target_modules: list[str] | None = None
        """Module names to apply LoRA to (matched against the last segment of the FQN).
        None means all Linear layers. An empty list means no layers."""

    def __init__(self, config: Config, **kwargs):
        if config.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {config.rank}")
        self.config = config
        self.rank = config.rank
        self.alpha = config.alpha
        self.target_modules = (
            set(config.target_modules) if config.target_modules is not None else None
        )
        if self.target_modules is None:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha} "
                f"(all Linear layers)"
            )
        else:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha}, "
                f"target_modules={sorted(self.target_modules)}"
            )

    def convert(self, model_config) -> None:
        """Walk the model config tree for all leaf modules.

        Target Linear modules are wrapped with ``LoRAConfig`` (trainable
        adapters, frozen base weight).  Everything else (non-target linears,
        norms, embeddings) is wrapped with ``FrozenConfig``.  All freezing
        happens at build time — no separate freeze step needed.
        """
        from torchtitan.models.common.embedding import Embedding
        from torchtitan.models.common.rmsnorm import RMSNorm

        matched = set()

        # Wrap all Linear.Config entries
        for fqn, cfg, parent, attr in model_config.traverse(Linear.Config):
            last_segment = fqn.rsplit(".", 1)[-1]
            is_target = (
                self.target_modules is None or last_segment in self.target_modules
            )
            if is_target:
                wrapped = LoRAConfig(
                    inner=cfg,
                    rank=self.rank,
                    alpha=self.alpha,
                )
                matched.add(last_segment)
            else:
                wrapped = FrozenConfig(inner=cfg)

            if isinstance(parent, list):
                parent[attr] = wrapped
            else:
                setattr(parent, attr, wrapped)

        # Freeze non-Linear leaf modules (norms, embeddings)
        for config_cls in (RMSNorm.Config, Embedding.Config):
            for _fqn, cfg, parent, attr in model_config.traverse(config_cls):
                wrapped = FrozenConfig(inner=cfg)
                if isinstance(parent, list):
                    parent[attr] = wrapped
                else:
                    setattr(parent, attr, wrapped)

        unmatched = (self.target_modules or set()) - matched
        if unmatched:
            logger.warning(
                f"LoRA target_modules {sorted(unmatched)} did not match any "
                f"Linear.Config in the model config tree."
            )

    def build_external_transforms(self, sd_adapter) -> dict[str, Any] | None:
        """Build LoRA external format transforms.

        Returns a dict with ``to_external``/``from_external`` callables
        for PEFT key remapping, or None if LoRA is not active.
        """
        if not _lora_class_cache or sd_adapter is None:
            return None
        from_hf_map = sd_adapter.from_hf_map

        return {
            "to_external": lambda sd: remap_lora_keys_to_hf(sd, from_hf_map),
            "from_external": lambda sd: remap_lora_keys_from_hf(sd, from_hf_map),
        }


def remap_lora_keys_to_hf(
    adapter_sd: dict[str, Any], from_hf_map: dict[str, str | None]
) -> dict[str, Any]:
    """Remap torchtitan LoRA keys to HF PEFT naming for saving.

    Converts keys like ``layers.0.attention.wq.lora_a.weight`` to
    ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``.
    """
    to_hf_map = {v: k for k, v in from_hf_map.items() if v is not None}
    remapped = {}
    for key, value in adapter_sd.items():
        lora_suffix = None
        base_key = key
        for suffix in (".lora_a.weight", ".lora_b.weight"):
            if key.endswith(suffix):
                lora_suffix = suffix
                base_key = key[: -len(suffix)] + ".weight"
                break
        if lora_suffix is None:
            remapped[key] = value
            continue

        abstract_base = re.sub(r"(?<=\.)\d+(?=\.)", "{}", base_key, count=1)
        layer_match = re.search(r"(?<=\.)\d+(?=\.)", base_key)
        layer_num = layer_match.group(0) if layer_match else None

        if abstract_base in to_hf_map:
            hf_abstract = to_hf_map[abstract_base]
            hf_base = hf_abstract.format(layer_num) if layer_num else hf_abstract
            hf_base_no_weight = hf_base.rsplit(".weight", 1)[0]
            peft_suffix = lora_suffix.replace(".lora_a.", ".lora_A.").replace(
                ".lora_b.", ".lora_B."
            )
            hf_key = f"base_model.model.{hf_base_no_weight}{peft_suffix}"
        else:
            hf_key = f"base_model.model.{key}"
            hf_key = hf_key.replace(".lora_a.", ".lora_A.").replace(
                ".lora_b.", ".lora_B."
            )
            logger.warning(
                f"No HF mapping for base key '{abstract_base}', using '{hf_key}'"
            )
        remapped[hf_key] = value
    return remapped


def remap_lora_keys_from_hf(
    adapter_sd: dict[str, Any], from_hf_map: dict[str, str | None]
) -> dict[str, Any]:
    """Remap HF PEFT keys to torchtitan naming for loading.

    Converts keys like ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``
    to ``layers.0.attention.wq.lora_a.weight``.
    """
    remapped = {}
    for key, value in adapter_sd.items():
        stripped = key
        if stripped.startswith("base_model.model."):
            stripped = stripped[len("base_model.model.") :]
        stripped = stripped.replace(".lora_A.", ".lora_a.").replace(
            ".lora_B.", ".lora_b."
        )

        lora_suffix = None
        hf_base_key = stripped
        for suffix in (".lora_a.weight", ".lora_b.weight"):
            if stripped.endswith(suffix):
                lora_suffix = suffix
                hf_base_key = stripped[: -len(suffix)] + ".weight"
                break
        if lora_suffix is None:
            remapped[stripped] = value
            continue

        abstract_hf = re.sub(r"(?<=\.)\d+(?=\.)", "{}", hf_base_key, count=1)
        layer_match = re.search(r"(?<=\.)\d+(?=\.)", hf_base_key)
        layer_num = layer_match.group(0) if layer_match else None

        tt_abstract = from_hf_map.get(abstract_hf)
        if tt_abstract is not None:
            tt_base = tt_abstract.format(layer_num) if layer_num else tt_abstract
            tt_base_no_weight = tt_base.rsplit(".weight", 1)[0]
            tt_key = f"{tt_base_no_weight}{lora_suffix}"
        else:
            tt_key = stripped
            logger.warning(
                f"No torchtitan mapping for HF key '{abstract_hf}', using '{tt_key}'"
            )
        remapped[tt_key] = value
    return remapped
