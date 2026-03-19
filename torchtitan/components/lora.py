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
                # Standard LoRA uses zeros for lora_b so the initial
                # contribution is exactly zero.  When the adapter has QAT
                # fake quantization (e.g. float8), zeros cause NaN because
                # dynamic scaling divides by max(abs(0)) = 0.  Use std=0.02
                # so values are representable in float8; the initial LoRA
                # contribution is still negligible.
                from torchao.quantization.qat.linear import FakeQuantizedLinear

                if isinstance(self.lora_b, FakeQuantizedLinear):
                    nn.init.normal_(self.lora_b.weight, mean=0.0, std=0.02)
                else:
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

    def _make_peft_save_fn(self):
        """Return a closure that saves adapter weights in PEFT format.

        The closure writes ``adapter_model.safetensors`` and
        ``adapter_config.json`` into the checkpoint directory. Only rank 0
        performs file I/O. Keys are remapped from torchtitan to HF PEFT naming.
        """
        rank = self.rank
        alpha = self.alpha

        def save(
            state_dict: dict[str, Any],
            checkpoint_dir: str,
            from_hf_map: dict[str, str | None] | None,
        ) -> None:
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

                safetensors_path = os.path.join(
                    checkpoint_dir, "adapter_model.safetensors"
                )
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
                    "r": rank,
                    "lora_alpha": alpha,
                    "target_modules": target_modules,
                }
                config_path = os.path.join(checkpoint_dir, "adapter_config.json")
                with open(config_path, "w") as f:
                    json.dump(config_dict, f, indent=2)
                logger.info(f"Saved PEFT adapter config to {config_path}")

            # Ensure all ranks wait for rank 0 to finish writing
            if dist.is_initialized():
                dist.barrier()

        return save

    def _make_peft_load_fn(self):
        """Return a closure that loads adapter weights from a PEFT directory.

        The closure loads ``adapter_model.safetensors``, remaps keys from HF
        PEFT naming to torchtitan naming, and broadcasts from rank 0.
        """

        def load(
            path: str,
            model_parts: list[nn.Module],
            from_hf_map: dict[str, str | None] | None,
        ) -> None:
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

        return load

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

        if self.save_format == "merged":
            model.converter_export_sd_fn = self._make_merge_fn()  # type: ignore[attr-defined]
        elif self.save_format == "peft":
            model.converter_save_last_fn = self._make_peft_save_fn()  # type: ignore[attr-defined]
            model.converter_load_additional_fn = self._make_peft_load_fn()  # type: ignore[attr-defined]
        # "dcp" format: no special attrs needed, checkpoint uses DCP

    def _apply_adapter_qat(
        self, model: nn.Module, scheme: str, group_size: int
    ) -> None:
        """Apply QAT fake quantization to LoRA adapter linears (lora_a, lora_b).

        For schemes with per-group quantization, the group_size is clamped to
        the LoRA rank so that lora_b (weight shape: out_features x rank) is
        compatible.

        Also patches init_weights onto the replaced adapter classes so that
        LoRALinear.init_weights() -> super chain works after adapters become
        FakeQuantizedLinear.
        """
        import dataclasses

        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
        from torchao.quantization.qat.api import QATStep
        from torchao.quantization.qat.fake_quantize_config import (
            _infer_fake_quantize_configs,
        )

        from torchtitan.components.quantization.qat import (
            _build_base_config,
            _SCHEMES_WITH_GROUP_SIZE,
        )

        # For schemes with per-group quantization, clamp group_size to rank
        # so lora_b's in_features (= rank) dimension is divisible.
        adapter_group_size = group_size
        if scheme in _SCHEMES_WITH_GROUP_SIZE:
            adapter_group_size = min(group_size, self.rank)
            if adapter_group_size != group_size:
                logger.info(
                    f"Adapter QAT: clamped group_size from {group_size} to "
                    f"{adapter_group_size} to fit LoRA rank={self.rank}"
                )

        # Build the fake quantize configs directly rather than relying on
        # QATConfig(base_config=...) inference, which hardcodes group_size
        # for some schemes.
        base_config = _build_base_config(scheme, adapter_group_size)
        act_config, weight_config = _infer_fake_quantize_configs(base_config)
        if (
            weight_config is not None
            and hasattr(weight_config, "group_size")
            and weight_config.group_size != adapter_group_size
        ):
            weight_config = dataclasses.replace(
                weight_config, group_size=adapter_group_size
            )

        def _is_lora_linear(mod: nn.Module, fqn: str) -> bool:
            return isinstance(mod, nn.Linear) and (
                fqn.endswith(".lora_a") or fqn.endswith(".lora_b")
            )

        quantize_(
            model,
            QATConfig(
                activation_config=act_config,
                weight_config=weight_config,
                step=QATStep.PREPARE,
            ),
            filter_fn=_is_lora_linear,
        )

        # Patch init_weights onto adapter FakeQuantizedLinear classes (same
        # pattern as QATConverter.convert) so the LoRA init_weights chain works.
        _patched_classes: set[type] = set()
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and (
                name.endswith(".lora_a") or name.endswith(".lora_b")
            ):
                mod._init_mean = 0.0
                mod._init_std = 0.02
                cls = type(mod)
                if cls not in _patched_classes and not hasattr(cls, "init_weights"):
                    cls.init_weights = Linear.init_weights
                    _patched_classes.add(cls)

        logger.info(
            f"Applied adapter QAT fake quantization "
            f"(scheme={scheme}, group_size={adapter_group_size})"
        )

    def _replace_linears_with_lora(self, module: nn.Module) -> None:
        for _, child in list(module.named_modules()):
            if isinstance(child, nn.Linear):
                apply_lora(child, self.rank, self.alpha)

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass

    def finalize(self, model: nn.Module) -> None:
        """Merge LoRA adapters into base weights at the model level.

        Only runs when save_format='merged'. After finalize, the model
        contains no LoRA modules — just plain linear layers (or whatever
        the parent class was, e.g. FakeQuantizedLinear) ready for
        downstream converters like QAT CONVERT.
        """
        if self.save_format != "merged":
            return

        scaling = self.alpha / self.rank
        merge_count = 0
        for name, mod in list(model.named_modules()):
            if not (hasattr(mod, "lora_a") and hasattr(mod, "lora_b")):
                continue

            # Merge: base.weight += scaling * B @ A
            with torch.no_grad():
                lora_a_weight = mod.lora_a.weight
                lora_b_weight = mod.lora_b.weight
                mod.weight.add_(scaling * (lora_b_weight @ lora_a_weight))

            # Remove LoRA submodules
            del mod.lora_a
            del mod.lora_b
            if hasattr(mod, "_lora_scaling"):
                del mod._lora_scaling

            # Restore the original parent class (remove LoRA from MRO).
            # The LoRA class is dynamically created as LoRA<ParentClass>,
            # and the parent is always the second entry in __mro__
            # (after the LoRA class itself).
            parent_cls = type(mod).__mro__[1]
            if parent_cls is not nn.Module and issubclass(parent_cls, nn.Linear):
                mod.__class__ = parent_cls

            merge_count += 1

        logger.info(
            f"LoRA finalize: merged {merge_count} adapter(s) into base weights "
            f"(scaling={scaling})"
        )


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
