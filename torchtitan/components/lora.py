# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torchtitan.config import Configurable
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
                self.lora_a = nn.Linear(
                    self.in_features,
                    rank,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )
                self.lora_b = nn.Linear(
                    rank,
                    self.out_features,
                    bias=False,
                    device=device,
                    dtype=dtype,
                )

            def _init_weight(self) -> None:
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

        save_adapter_only: bool = True
        """If True, only save LoRA adapter weights in checkpoints.
        Requires base model to be loaded from HF/initial_load_path on resume.
        Set to False to save full model weights for debugging without pretrained base."""

        quantize_base: str = ""
        """Quantize base (non-LoRA) weights. "" = no quantization, "nf4" = NF4 (QLoRA).
        NF4 quantization reduces base weight memory ~4x while keeping LoRA adapters in full precision."""

        nf4_scaler_block_size: int = 128
        """Scaler block size for NF4 quantization. Default 128 works with debugmodel on 8 GPUs.
        The default torchao value (256) may be too large for sharded tensors."""

    def __init__(self, config: Config, **kwargs):
        self.rank = config.rank
        self.alpha = config.alpha
        self.save_adapter_only = config.save_adapter_only
        self.quantize_base = config.quantize_base
        self.nf4_scaler_block_size = config.nf4_scaler_block_size
        if self.quantize_base and self.quantize_base != "nf4":
            raise ValueError(
                f"Unsupported quantize_base value: '{self.quantize_base}'. "
                "Supported values: '' (none), 'nf4'."
            )
        logger.info(
            f"LoRA training active with rank={self.rank}, alpha={self.alpha}"
            + (f", quantize_base={self.quantize_base}" if self.quantize_base else "")
        )

    def convert(self, model: nn.Module) -> None:
        model.requires_grad_(False)
        self._replace_linears_with_lora(model)

    def _replace_linears_with_lora(self, module: nn.Module) -> None:
        for _, child in list(module.named_modules()):
            if isinstance(child, nn.Linear):
                apply_lora(child, self.rank, self.alpha)

        # Patch init_weights to also reinitialize LoRA adapters
        original_init_weights = getattr(module, "init_weights", None)

        def new_model_init_weights(*args: Any, **kwargs: Any) -> None:
            if original_init_weights is not None and callable(original_init_weights):
                original_init_weights(*args, **kwargs)
            for sub_module in module.modules():
                if type(sub_module) in _lora_class_cache.values():
                    _init_weight = getattr(sub_module, "_init_weight", None)
                    assert _init_weight is not None and callable(_init_weight)
                    _init_weight()

        object.__setattr__(module, "init_weights", new_model_init_weights)

        # Expose a key filter and flag on the module so ModelWrapper can
        # partition the state dict without knowing about LoRA internals.
        def converter_key_filter(key: str) -> bool:
            """Return True if key was added by this converter (LoRA adapter weights)."""
            return ".lora_a." in key or ".lora_b." in key

        object.__setattr__(module, "converter_key_filter", converter_key_filter)
        object.__setattr__(module, "save_converter_keys_only", self.save_adapter_only)

        # Register a one-shot forward pre-hook to quantize base weights after
        # checkpoint load but before the first forward pass (QLoRA).
        if self.quantize_base == "nf4":
            from torch.distributed.tensor import DTensor

            try:
                from torchao.dtypes.nf4tensor import to_nf4
            except ImportError:
                raise ImportError(
                    "QLoRA requires torchao. Install with: pip install torchao"
                )

            lora_classes = tuple(_lora_class_cache.values())
            nf4_scaler_block_size = self.nf4_scaler_block_size

            def _to_nf4_tensor(weight: torch.Tensor) -> torch.Tensor:
                """Convert weight to NF4, handling both regular tensors and DTensors."""
                nf4_block_size = 64  # NF4 default block size
                is_dtensor = isinstance(weight, DTensor)
                local_weight = weight.to_local() if is_dtensor else weight

                num_scalers = local_weight.numel() // nf4_block_size
                if num_scalers % nf4_scaler_block_size != 0:
                    raise ValueError(
                        f"NF4 quantization failed: num_scalers ({num_scalers}) is not "
                        f"divisible by nf4_scaler_block_size ({nf4_scaler_block_size}). "
                        f"Try a smaller nf4_scaler_block_size in LoRAConverter.Config "
                        f"(e.g., 64, 32, or 1)."
                    )

                nf4_local = to_nf4(
                    local_weight, scaler_block_size=nf4_scaler_block_size
                )

                if is_dtensor:
                    return DTensor.from_local(
                        nf4_local, weight.device_mesh, weight.placements
                    )
                return nf4_local

            def _quantize_hook(
                mod: nn.Module, args: Any, handle: torch.utils.hooks.RemovableHandle
            ) -> None:
                for sub in mod.modules():
                    if isinstance(sub, lora_classes):
                        sub.weight = nn.Parameter(
                            _to_nf4_tensor(sub.weight.data), requires_grad=False
                        )
                logger.info("QLoRA: quantized base weights to NF4")
                handle.remove()

            # Use a list to allow the closure to reference the handle before it exists
            handle_ref: list[torch.utils.hooks.RemovableHandle] = []
            handle_ref.append(
                module.register_forward_pre_hook(
                    lambda mod, args: _quantize_hook(mod, args, handle_ref[0])
                )
            )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass
