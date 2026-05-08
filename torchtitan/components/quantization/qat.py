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


def _patch_module_protocol(mod: nn.Module) -> None:
    """Make torchao-created modules satisfy torchtitan's Module protocol.

    ``torchao.quantize_()`` replaces ``nn.Linear`` with classes like
    ``FakeQuantizedLinear`` that don't inherit from torchtitan's ``Module``.
    This patches the class hierarchy so ``verify_module_protocol()`` passes,
    and sets ``_param_init`` to skip re-initialization (params are already
    initialized by the original Linear).
    """
    for child in mod.modules():
        if not isinstance(child, Module):
            cls = type(child)
            if cls not in _patched_classes:
                patched = type(cls.__name__, (cls, Module), {})
                _patched_classes[cls] = patched
            child.__class__ = _patched_classes[cls]
            child._param_init = {  # pyrefly: ignore [bad-argument-type]
                name: _noop_init for name, _ in child.named_parameters(recurse=False)
            }


_patched_classes: dict[type, type] = {}


@dataclass(kw_only=True, slots=True)
class QATLinearConfig(QuantizedLinearConfig):
    """Config that builds a Linear then applies QAT fake quantization.

    ``torchao.quantize_()`` replaces ``nn.Linear`` children of a parent
    module, so ``build()`` wraps the freshly built Linear in a temporary
    parent, runs ``quantize_()``, and returns the swapped child.
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


def convert_qat_to_linear(model: nn.Module) -> None:
    """Strip FakeQuantizedLinear back to plain nn.Linear.

    Walks the module tree and replaces any FakeQuantizedLinear with
    a plain nn.Linear carrying the same weight and bias. Called at
    end of training so saved checkpoints have clean weights.
    """
    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    for name, mod in model.named_modules():
        if type(mod).__name__ in (
            "FakeQuantizedLinear",
            "NVFP4FakeQuantizedLinear",
            "MXFakeQuantizedLinear",
        ):
            new_linear = nn.Linear(
                mod.in_features,  # pyrefly: ignore [bad-argument-type]
                mod.out_features,  # pyrefly: ignore [bad-argument-type]
                bias=mod.bias is not None,
                device=mod.weight.device,
                dtype=mod.weight.dtype,
            )
            new_linear.weight = mod.weight  # pyrefly: ignore [bad-assignment]
            if mod.bias is not None:
                new_linear.bias = mod.bias  # pyrefly: ignore [bad-assignment]
            replacements.append((name, mod, new_linear))

    for name, _old, new_linear in replacements:
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            setattr(model, parts[0], new_linear)
        else:
            parent = model.get_submodule(parts[0])
            setattr(parent, parts[1], new_linear)

    if replacements:
        logger.info(
            f"Converted {len(replacements)} FakeQuantizedLinear modules back to nn.Linear"
        )


class QATConverter(QuantizationConverter):
    """Apply quantization-aware training to Linear layers in a model.

    Operates on the model config tree: Linear configs are replaced with
    ``QATLinear.Config`` which builds a QATLinear that applies fake
    quantization via ``torchao.quantize_()`` in its constructor.

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
        """Walk the model config tree, replacing Linear configs with QATLinear configs."""
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
