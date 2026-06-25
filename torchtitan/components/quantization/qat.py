# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, fields

import torch.nn as nn

from torchtitan.components.quantization import (
    QuantizationConverter,
    QuantizedLinearConfig,
)
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger

_SUPPORTED_SCHEMES = (
    "int4_weight_only",
    "intx_weight_only",
    "int8_dynamic_act_intx_weight",
    "float8_dynamic_act_float8_weight",
    "float8_dynamic_act_int4_weight",
    "nvfp4",
    "mx",
)

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


def _noop_init(param: nn.Parameter) -> None:
    pass


_patched_classes: dict[type, type] = {}


def _patch_module_protocol(mod: nn.Module) -> None:
    """Make torchao-created modules satisfy torchtitan's Module protocol.

    ``torchao.quantize_()`` creates classes like ``FakeQuantizedLinear``
    that don't inherit from torchtitan's ``Module``. This patches the
    class hierarchy and sets ``_param_init`` to skip re-initialization.
    """
    for child in mod.modules():
        if not isinstance(child, Module):
            cls = type(child)
            if cls not in _patched_classes:
                patched = type(cls.__name__, (cls, Module), {})
                _patched_classes[cls] = patched
            child.__class__ = _patched_classes[cls]
            child._param_init = {  # type: ignore[assignment]
                name: _noop_init for name, _ in child.named_parameters(recurse=False)
            }


@dataclass(kw_only=True, slots=True)
class QATLinearConfig(QuantizedLinearConfig):
    """Config that builds a Linear then applies QAT fake quantization.

    Unlike ``MXFP8Linear`` / ``Float8Linear`` which call ``quantize_(self)``
    in ``__init__``, QAT's ``quantize_()`` *replaces* the module rather than
    wrapping parameters in-place. So ``build()`` wraps the freshly built
    Linear in a temporary parent, runs ``quantize_()``, and returns the
    swapped ``FakeQuantizedLinear``.
    """

    _scheme: str = "int4_weight_only"
    _group_size: int = 128

    def build(self, **kwargs):
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig
        from torchao.quantization.qat.api import QATStep

        base_config_obj = Linear.Config(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias,
            param_init=self.param_init,
            sharding_config=self.sharding_config,
        )
        instance = base_config_obj.build(**kwargs)
        wrapper = nn.Sequential(instance)
        base_config = _build_base_config(self._scheme, self._group_size)
        quantize_(wrapper, QATConfig(base_config, step=QATStep.PREPARE))
        result = wrapper[0]
        _patch_module_protocol(result)
        return result


class QATConverter(QuantizationConverter):
    """Apply quantization-aware training to Linear layers in a model.

    Operates on the model config tree: Linear configs are replaced with
    ``QATLinearConfig`` which builds a ``FakeQuantizedLinear`` via
    ``torchao.quantize_()``.

    QAT is mutually exclusive with other quantization converters
    (Float8, MXFP8) — do not combine them in the same converters list.

    Supported schemes:
      - ``"int4_weight_only"`` -- int4 weight-only fake quantization
      - ``"intx_weight_only"`` -- intx weight-only fake quantization
      - ``"int8_dynamic_act_intx_weight"`` -- int8 activation + int4 weight
      - ``"float8_dynamic_act_float8_weight"`` -- float8 activation + float8 weight
      - ``"float8_dynamic_act_int4_weight"`` -- float8 activation + int4 weight
      - ``"nvfp4"`` -- NVFP4 dynamic activation + NVFP4 weight
      - ``"mx"`` -- MX dynamic activation + MX weight
    """

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        scheme: str = "int4_weight_only"
        """QAT scheme name. Maps to a torchao PTQ base config."""

        group_size: int = 128
        """Group size for per-group weight quantization.
        Used by schemes that support per-group granularity."""

    def __init__(self, config: Config, **kwargs):
        if config.scheme not in _SUPPORTED_SCHEMES:
            raise ValueError(
                f"Unknown QAT scheme '{config.scheme}'. "
                f"Supported: {_SUPPORTED_SCHEMES}"
            )
        self.scheme = config.scheme
        self.group_size = config.group_size
        if config.scheme not in _SCHEMES_WITH_GROUP_SIZE:
            logger.warning(
                f"QAT scheme '{config.scheme}' does not use group_size, "
                f"ignoring group_size={config.group_size}"
            )
        logger.info(
            f"QAT training active (scheme={self.scheme}, group_size={self.group_size})"
        )

    def convert(self, model_config) -> None:
        """Walk the model config tree, replacing Linear configs with QATLinearConfig."""
        for _fqn, config, parent, attr in model_config.traverse(Linear.Config):
            new_config = QATLinearConfig(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                _scheme=self.scheme,
                _group_size=self.group_size,
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            f"Swapped to QATLinear layers (scheme={self.scheme}, "
            f"group_size={self.group_size})"
        )
