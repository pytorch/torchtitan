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

from torchtitan.config import Configurable
from torchtitan.models.common.linear import Linear
from torchtitan.tools.logging import logger

# Cache for dynamically created LoRA classes
_lora_class_cache: dict[type, type] = {}


def apply_lora(linear: nn.Linear, rank: int, alpha: float) -> nn.Linear:
    parent_cls = type(linear)
    assert issubclass(
        parent_cls, nn.Linear
    ), f"parent_cls must be a subclass of nn.Linear, got {parent_cls}"

    if parent_cls not in _lora_class_cache:

        class LoRALinear(parent_cls):  # type: ignore[valid-type, misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("LoRALinear should not be instantiated directly.")

            @classmethod
            def from_linear(
                cls, linear: nn.Linear, rank: int, alpha: float
            ) -> "LoRALinear":
                linear.__class__ = cls
                linear._init_lora(rank, alpha)  # type: ignore[attr-defined]
                return linear  # type: ignore[return-value]

            def _init_lora(
                self,
                rank: int,
                alpha: float,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None,
            ) -> None:
                self._lora_scaling = alpha / rank
                device = device if device is not None else self.weight.device
                dtype = dtype if dtype is not None else self.weight.dtype
                self.lora_a = (
                    Linear.Config(bias=False)
                    .build(in_features=self.in_features, out_features=rank)
                    .to(device=device, dtype=dtype)
                )
                self.lora_b = (
                    Linear.Config(bias=False)
                    .build(in_features=rank, out_features=self.out_features)
                    .to(device=device, dtype=dtype)
                )

            def init_weights(self, **kwargs) -> None:
                super().init_weights(**kwargs)  # pyrefly: ignore [not-callable]
                nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_b.weight)

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                base_out = super().forward(input)
                lora_out = self.lora_b(self.lora_a(input))
                return base_out + self._lora_scaling * lora_out

        LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
        LoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
        _lora_class_cache[parent_cls] = LoRALinear

    # pyrefly: ignore [missing-attribute]
    return _lora_class_cache[parent_cls].from_linear(linear, rank, alpha)


class LoRAConverter(Configurable):
    """Apply LoRA adapters to all Linear layers in a model."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        rank: int = 8
        """Rank of the LoRA matrices (lora_a: in_features x rank, lora_b: rank x out_features)."""

        alpha: float = 16.0
        """Scaling factor. Output is scaled by alpha/rank."""

        save_format: str = "dcp"
        """Format for saving adapter weights at the last training step.
        "dcp" saves adapter weights via DCP (default, supports resumption).
        "peft" saves adapter_model.safetensors + adapter_config.json for
        compatibility with HuggingFace PEFT.
        "merged" folds adapters into base weights (base + alpha/rank * B @ A)
        and saves a standard checkpoint with no LoRA keys."""

    def __init__(self, config: Config, **kwargs):
        self.rank = config.rank
        self.alpha = config.alpha
        self.save_format = config.save_format
        if self.save_format not in ("dcp", "peft", "merged"):
            raise ValueError(
                f"LoRA save_format must be 'dcp', 'peft', or 'merged', "
                f"got '{self.save_format}'"
            )

        logger.info(f"LoRA training active with rank={self.rank}, alpha={self.alpha}")

    @staticmethod
    def _is_lora_key(key: str) -> bool:
        """Check if a state dict key belongs to a LoRA adapter."""
        return ".lora_a." in key or ".lora_b." in key

    def _make_merge_fn(self):
        """Return a function that merges LoRA adapters into base weights.

        The returned function takes a full state dict (base + adapter keys)
        and returns a new dict with standard FQNs where each base weight
        has been replaced by ``base + (alpha / rank) * B @ A``.
        """
        scaling = self.alpha / self.rank

        def merge(state_dict: dict[str, Any]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            lora_a: dict[str, Any] = {}
            lora_b: dict[str, Any] = {}
            for key, value in state_dict.items():
                if ".lora_a." in key:
                    lora_a[key.split(".lora_a.")[0]] = value
                elif ".lora_b." in key:
                    lora_b[key.split(".lora_b.")[0]] = value
                else:
                    merged[key] = value
            for prefix in lora_a:
                base_key = prefix + ".weight"
                if base_key in merged:
                    merged[base_key] = merged[base_key] + scaling * (
                        lora_b[prefix] @ lora_a[prefix]
                    )
            return merged

        return merge

    def convert(self, model: nn.Module) -> None:
        model.requires_grad_(False)
        self._replace_linears_with_lora(model)

        # If QATConverter was applied before LoRA, apply the same QAT to
        # the newly created adapter linears. QATConverter stores its config
        # on the model as _qat_scheme / _qat_group_size.
        qat_scheme = getattr(model, "_qat_scheme", None)
        if qat_scheme is not None:
            qat_group_size = getattr(model, "_qat_group_size", 128)
            self._apply_adapter_qat(model, qat_scheme, qat_group_size)

        # Wire up checkpoint filtering so ModelWrapper knows which keys
        # are adapter keys and how to save them.
        model.converter_key_filter = self._is_lora_key  # type: ignore[attr-defined]
        model.converter_save_format = self.save_format  # type: ignore[attr-defined]
        model.converter_config = {  # type: ignore[attr-defined]
            "rank": self.rank,
            "alpha": self.alpha,
        }

        if self.save_format == "merged":
            model.converter_export_sd_fn = self._make_merge_fn()  # type: ignore[attr-defined]

    def _apply_adapter_qat(
        self, model: nn.Module, scheme: str, group_size: int
    ) -> None:
        from torchtitan.components.quantization.qat import _SCHEMES_WITH_GROUP_SIZE

        # Validate group_size against LoRA rank
        if scheme in _SCHEMES_WITH_GROUP_SIZE and self.rank % group_size != 0:
            raise ValueError(
                f"QAT group_size ({group_size}) does not divide LoRA rank "
                f"({self.rank}). Use a smaller group_size or larger rank."
            )

        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
        from torchao.quantization.qat.api import QATStep

        from torchtitan.components.quantization.qat import _build_base_config

        base_config = _build_base_config(scheme, group_size)

        def _is_lora_linear(mod: nn.Module, fqn: str) -> bool:
            return isinstance(mod, nn.Linear) and (
                fqn.endswith(".lora_a") or fqn.endswith(".lora_b")
            )

        quantize_(
            model,
            QATConfig(base_config, step=QATStep.PREPARE),
            filter_fn=_is_lora_linear,
        )
        logger.info(
            f"Applied adapter QAT fake quantization "
            f"(scheme={scheme}, group_size={group_size})"
        )

    def _replace_linears_with_lora(self, module: nn.Module) -> None:
        for _, child in list(module.named_modules()):
            if isinstance(child, nn.Linear):
                apply_lora(child, self.rank, self.alpha)

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass


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

        if abstract_hf in from_hf_map and from_hf_map[abstract_hf] is not None:
            tt_abstract = from_hf_map[abstract_hf]
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
