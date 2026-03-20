# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from torchtitan.config import Configurable
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.model_converter import ConverterCheckpointHooks
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

        merge_adapter: bool = False
        """When True, adapters are folded into base weights
        (base + alpha/rank * B @ A) and the checkpoint contains no LoRA keys.
        When False (default), adapter weights are saved separately.
        Use checkpoint.last_save_in_hf=True to save in PEFT format."""

    def __init__(self, config: Config, **kwargs):
        self.rank = config.rank
        self.alpha = config.alpha
        self.merge_adapter = config.merge_adapter
        logger.info(f"LoRA training active with rank={self.rank}, alpha={self.alpha}")

    @staticmethod
    def _is_lora_key(key: str) -> bool:
        """Check if a state dict key belongs to a LoRA adapter."""
        return ".lora_a." in key or ".lora_b." in key

    def _save_peft(
        self,
        state_dict: dict[str, Any],
        checkpoint_dir: str,
        from_hf_map: dict[str, str | None] | None,
    ) -> None:
        """Save adapter weights in PEFT format.

        Writes ``adapter_model.safetensors`` and ``adapter_config.json``
        into the checkpoint directory. Only rank 0 performs file I/O.
        Keys are remapped from torchtitan to HF PEFT naming.
        """
        from safetensors.torch import save_file

        # Collect full tensors from DTensors on CPU
        cpu_states = {}
        for k, v in state_dict.items():
            if hasattr(v, "full_tensor"):
                cpu_states[k] = v.full_tensor().cpu()
            else:
                cpu_states[k] = v.cpu() if isinstance(v, torch.Tensor) else v

        # Remap keys to HF PEFT naming
        if from_hf_map is not None:
            hf_states = remap_lora_keys_to_hf(cpu_states, from_hf_map)
        else:
            logger.warning(
                "No from_hf_map available; saving PEFT with torchtitan keys."
            )
            hf_states = cpu_states

        if dist.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

            safetensors_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
            save_file(hf_states, safetensors_path)
            logger.info(f"Saved PEFT adapter weights to {safetensors_path}")

            # Write adapter_config.json with HF module names
            target_modules = sorted(
                {
                    k.rsplit(".lora_A.", 1)[0].rsplit(".", 1)[-1]
                    if ".lora_A." in k
                    else k.rsplit(".lora_B.", 1)[0].rsplit(".", 1)[-1]
                    for k in hf_states
                }
            )
            config_dict = {
                "peft_type": "LORA",
                "r": self.rank,
                "lora_alpha": self.alpha,
                "target_modules": target_modules,
            }
            config_path = os.path.join(checkpoint_dir, "adapter_config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Saved PEFT adapter config to {config_path}")

        # Ensure all ranks wait for rank 0 to finish writing
        if dist.is_initialized():
            dist.barrier()

    def _load_peft(
        self,
        path: str,
        model_parts: list[nn.Module],
        from_hf_map: dict[str, str | None] | None,
    ) -> None:
        """Load adapter weights from a PEFT directory.

        Loads ``adapter_model.safetensors``, remaps keys from HF PEFT naming
        to torchtitan naming, and broadcasts from rank 0.
        """
        import functools

        from safetensors.torch import load_file

        safetensors_path = os.path.join(path, "adapter_model.safetensors")
        adapter_sd = load_file(safetensors_path)
        if from_hf_map is not None:
            adapter_sd = remap_lora_keys_from_hf(adapter_sd, from_hf_map)
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=adapter_sd,
            options=StateDictOptions(
                strict=False,
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )
        list(map(func, model_parts))

    def convert(self, model: nn.Module) -> None:
        model.requires_grad_(False)
        self._replace_linears_with_lora(model)

        if self.merge_adapter:
            hooks = ConverterCheckpointHooks(key_filter=self._is_lora_key)
        else:
            hooks = ConverterCheckpointHooks(
                key_filter=self._is_lora_key,
                save_last_fn=self._save_peft,
                load_additional_fn=self._load_peft,
            )
        model._converter_hooks = hooks  # type: ignore[attr-defined]

    def _replace_linears_with_lora(self, module: nn.Module) -> None:
        for _, child in list(module.named_modules()):
            if isinstance(child, nn.Linear):
                apply_lora(child, self.rank, self.alpha)

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass

    def finalize(self, model: nn.Module) -> None:
        """Merge LoRA adapters into base weights at end of training.

        Only runs when merge_adapter=True. After merging, adapter modules
        are removed and the linear class is restored to its parent class.
        Hooks are cleaned up so state_dict() returns all keys.
        """
        if not self.merge_adapter:
            return

        scaling = self.alpha / self.rank
        for name, mod in list(model.named_modules()):
            if not (hasattr(mod, "lora_a") and hasattr(mod, "lora_b")):
                continue
            with torch.no_grad():
                mod.weight.add_(scaling * (mod.lora_b.weight @ mod.lora_a.weight))
            del mod.lora_a, mod.lora_b
            if hasattr(mod, "_lora_scaling"):
                del mod._lora_scaling
            # Restore the parent class (e.g. FakeQuantizedLinear or nn.Linear)
            parent_cls = type(mod).__mro__[1]
            if parent_cls is not nn.Module and issubclass(parent_cls, nn.Linear):
                mod.__class__ = parent_cls

        # Clean up hooks — model is now a plain base model
        if hasattr(model, "_converter_hooks"):
            del model._converter_hooks

        logger.info("LoRA adapters merged into base weights")


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
