# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch.nn as nn
from torchtitan.config import Configurable
from torchtitan.tools.logging import logger

# Supported scheme names.
_SUPPORTED_SCHEMES = (
    "int4_weight_only",
    "intx_weight_only",
    "int8_dynamic_act_intx_weight",
    "float8_dynamic_act_float8_weight",
    "float8_dynamic_act_int4_weight",
    "nvfp4",
    "mx",
)

# Schemes that accept a group_size parameter.
_SCHEMES_WITH_GROUP_SIZE = (
    "int4_weight_only",
    "intx_weight_only",
    "int8_dynamic_act_intx_weight",
)


def _build_base_config(scheme: str, group_size: int):
    """Return a torchao PTQ base config for the given scheme name."""
    if scheme == "int4_weight_only":
        from torchao.quantization import Int4WeightOnlyConfig

        return Int4WeightOnlyConfig(group_size=group_size)

    elif scheme == "intx_weight_only":
        import torch
        from torchao.quantization import IntxWeightOnlyConfig
        from torchao.quantization.granularity import PerGroup

        int4_dtype = torch.int4  # pyrefly: ignore[missing-attribute]
        return IntxWeightOnlyConfig(
            weight_dtype=int4_dtype,
            granularity=PerGroup(group_size),
        )

    elif scheme == "int8_dynamic_act_intx_weight":
        import torch
        from torchao.quantization import Int8DynamicActivationIntxWeightConfig
        from torchao.quantization.granularity import PerGroup

        int4_dtype = torch.int4  # pyrefly: ignore[missing-attribute]
        return Int8DynamicActivationIntxWeightConfig(
            weight_dtype=int4_dtype,
            weight_granularity=PerGroup(group_size),
        )

    elif scheme == "float8_dynamic_act_float8_weight":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

        return Float8DynamicActivationFloat8WeightConfig()

    elif scheme == "float8_dynamic_act_int4_weight":
        from torchao.quantization import Float8DynamicActivationInt4WeightConfig

        return Float8DynamicActivationInt4WeightConfig()

    elif scheme == "nvfp4":
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

        return NVFP4DynamicActivationNVFP4WeightConfig()

    elif scheme == "mx":
        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig

        return MXDynamicActivationMXWeightConfig()

    else:
        raise ValueError(
            f"Unknown QAT scheme '{scheme}'. Supported: {_SUPPORTED_SCHEMES}"
        )


class QATConverter(Configurable):
    """Apply quantization-aware training via torchao's QATConfig.

    Uses ``torchao.quantize_(model, QATConfig(base_config, step="prepare"))``
    to insert fake quantization into ``nn.Linear`` modules. The ``scheme``
    config field selects a torchao PTQ base config, which QATConfig uses to
    infer the appropriate fake quantization for both weights and activations.

    Supported schemes:
      - ``"int4_weight_only"`` — int4 weight-only fake quantization
      - ``"intx_weight_only"`` — intx weight-only fake quantization
      - ``"int8_dynamic_act_intx_weight"`` — int8 activation + int4 weight
      - ``"float8_dynamic_act_float8_weight"`` — float8 activation + float8 weight
      - ``"float8_dynamic_act_int4_weight"`` — float8 activation + int4 weight
      - ``"nvfp4"`` — NVFP4 dynamic activation + NVFP4 weight
      - ``"mx"`` — MX dynamic activation + MX weight

    When composed with LoRA (QATConverter listed before LoRAConverter in converters),
    LoRA will inherit from FakeQuantizedLinear so base weights are fake-quantized
    while LoRA adapters stay full-precision.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        scheme: str = "int4_weight_only"
        """QAT scheme name. Maps to a torchao PTQ base config.
        Supported: 'int4_weight_only', 'intx_weight_only',
        'int8_dynamic_act_intx_weight', 'float8_dynamic_act_float8_weight',
        'float8_dynamic_act_int4_weight', 'nvfp4', 'mx'."""

        group_size: int = 128
        """Group size for per-group weight quantization.
        Used by schemes that support per-group granularity
        (int4_weight_only, intx_weight_only, int8_dynamic_act_intx_weight).
        Must divide in_features of all Linear layers in the model."""

        convert_at_end: bool = False
        """When True, finalize() applies QAT CONVERT step at end of training,
        replacing FakeQuantizedLinear modules with real quantized modules
        (e.g. WeightOnlyInt4Linear with packed int4 weights)."""

    def __init__(self, config: Config, **kwargs):
        if config.scheme not in _SUPPORTED_SCHEMES:
            raise ValueError(
                f"Unknown QAT scheme '{config.scheme}'. "
                f"Supported: {_SUPPORTED_SCHEMES}"
            )
        self.scheme = config.scheme
        self.group_size = config.group_size
        self.convert_at_end = config.convert_at_end
        if config.scheme not in _SCHEMES_WITH_GROUP_SIZE:
            logger.warning(
                f"QAT scheme '{config.scheme}' does not use group_size, "
                f"ignoring group_size={config.group_size}"
            )
        logger.info(
            f"QAT training active (scheme={self.scheme}, group_size={self.group_size}, "
            f"convert_at_end={self.convert_at_end})"
        )

    def convert(self, model: nn.Module) -> None:
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
        from torchao.quantization.qat.api import QATStep

        # Snapshot init params before quantize_ replaces Linear subclasses
        # with FakeQuantizedLinear (which lacks _init_mean / _init_std).
        init_params_cache: dict[str, tuple[float, float]] = {}
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and hasattr(mod, "_init_mean"):
                init_params_cache[name] = (
                    getattr(mod, "_init_mean", 0.0),
                    getattr(mod, "_init_std", 0.02),
                )

        base_config = _build_base_config(self.scheme, self.group_size)
        quantize_(model, QATConfig(base_config, step=QATStep.PREPARE))

        # Restore init params on replaced modules and patch init_weights onto
        # the FakeQuantizedLinear *class* (not instance) so that LoRA's subclass
        # can properly override it via MRO. An instance-level bound method would
        # shadow LoRA's class-level init_weights and break adapter initialization.
        from torchtitan.models.common.linear import Linear

        _patched_classes: set[type] = set()
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and name in init_params_cache:
                init_mean, init_std = init_params_cache[name]
                mod._init_mean = init_mean
                mod._init_std = init_std

                cls = type(mod)
                if cls not in _patched_classes and not hasattr(cls, "init_weights"):
                    cls.init_weights = Linear.init_weights
                    _patched_classes.add(cls)

        # Store QAT config on the model so downstream converters (e.g. LoRA)
        # can apply the same QAT to newly created modules.
        model._qat_scheme = self.scheme  # type: ignore[attr-defined]
        model._qat_group_size = self.group_size  # type: ignore[attr-defined]

        logger.info(
            f"Applied QAT fake quantization (scheme={self.scheme}, "
            f"group_size={self.group_size})"
        )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass

    def finalize(self, model: nn.Module) -> None:
        """Apply QAT CONVERT step, replacing FakeQuantizedLinear with real quantized modules.

        Skipped when convert_at_end is False. Also skipped if LoRA adapters
        are still present (adapter-only save), since CONVERT requires plain
        FakeQuantizedLinear modules without LoRA wrappers.
        """
        if not self.convert_at_end:
            return

        # Check for remaining LoRA adapters — if present, skip CONVERT.
        for name, mod in model.named_modules():
            if hasattr(mod, "lora_a") or hasattr(mod, "lora_b"):
                logger.warning(
                    "QAT CONVERT skipped: LoRA adapters still present on the model. "
                    "Use LoRA save_format='merged' to merge adapters before CONVERT."
                )
                return

        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
        from torchao.quantization.qat.api import QATStep

        base_config = _build_base_config(self.scheme, self.group_size)
        quantize_(model, QATConfig(base_config, step=QATStep.CONVERT))

        logger.info(
            f"Applied QAT CONVERT (scheme={self.scheme}, group_size={self.group_size}): "
            "FakeQuantizedLinear modules replaced with real quantized modules"
        )
