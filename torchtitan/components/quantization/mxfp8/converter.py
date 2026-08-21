# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields
from importlib.util import find_spec
from typing import Literal

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from ..utils import swap_token_dispatcher

WeightQuantization = Literal["32x32"]
InputActivationSaveFormat = Literal["bf16", "mxfp8"]
_WEIGHT_QUANTIZATION_STRATEGIES = ("32x32",)
_INPUT_ACTIVATION_SAVE_FORMATS = ("bf16", "mxfp8")

_mxfp8_linear_import_error: ImportError | None = None

try:
    from .linear import MXFP8Linear

except ImportError as import_error:
    MXFP8Linear = None
    _mxfp8_linear_import_error = import_error


class MXFP8LinearConverter(QuantizationConverter):
    """Replace matching Linear.Config with MXFP8Linear.Config."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply MXFP8 quantization to.
        Only Linear.Config entries whose FQN contains a match are converted.
        If empty, all Linear modules are converted.
        """
        weight_quantization: WeightQuantization = "32x32"
        """Dense-weight quantization strategy.

        ``"32x32"`` uses square scale tiles. The same quantized values can be
        consumed by FPROP and DGRAD because the tiles are invariant under
        transpose.
        """
        input_activation_save_format_by_fqn: dict[
            str, InputActivationSaveFormat
        ] = field(default_factory=dict)
        """Input-activation format to save for selected Linear modules.

        Keys are FQN substrings. ``"bf16"`` saves the original activation and
        quantizes it columnwise during backward. ``"mxfp8"`` saves columnwise
        qdata and scales produced during forward. Unmatched modules use
        ``"bf16"`` to avoid retaining an additional quantized representation
        when another operation may already keep the BF16 activation alive.
        """

        def __post_init__(self) -> None:
            if self.weight_quantization not in _WEIGHT_QUANTIZATION_STRATEGIES:
                raise ValueError(
                    "MXFP8 weight_quantization must be one of "
                    f"{_WEIGHT_QUANTIZATION_STRATEGIES}; got "
                    f"{self.weight_quantization!r}."
                )
            for fqn, save_format in self.input_activation_save_format_by_fqn.items():
                if not fqn:
                    raise ValueError(
                        "MXFP8 input_activation_save_format_by_fqn cannot contain "
                        "an empty FQN selector."
                    )
                if save_format not in _INPUT_ACTIVATION_SAVE_FORMATS:
                    raise ValueError(
                        "MXFP8 input_activation_save_format_by_fqn values must be "
                        f"one of {_INPUT_ACTIVATION_SAVE_FORMATS}; got "
                        f"{save_format!r} for {fqn!r}."
                    )

    def __init__(self, config: Config):
        self.config = config

        if MXFP8Linear is None:
            raise ImportError(
                "TorchAO with the MXFP8 32x32 swizzled cast kernels is required "
                "for MXFP8 linear layers. Install TorchAO from source."
            ) from _mxfp8_linear_import_error

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config):
        assert MXFP8Linear is not None
        fqns = self.config.fqns
        targets = [
            entry
            for entry in model_config.traverse(Linear.Config)
            if not fqns or any(target_fqn in entry[0] for target_fqn in fqns)
        ]

        save_format_by_fqn: dict[str, InputActivationSaveFormat] = {}
        matched_selectors: set[str] = set()
        save_formats = self.config.input_activation_save_format_by_fqn
        for fqn, _config, _parent, _attr in targets:
            matches = [selector for selector in save_formats if selector in fqn]
            if len(matches) > 1:
                raise ValueError(
                    "MXFP8 input_activation_save_format_by_fqn contains multiple "
                    f"selectors matching {fqn!r}: {matches}."
                )
            if matches:
                selector = matches[0]
                matched_selectors.add(selector)
                save_format_by_fqn[fqn] = save_formats[selector]
            else:
                save_format_by_fqn[fqn] = "bf16"

        unmatched_selectors = set(save_formats) - matched_selectors
        if unmatched_selectors:
            raise ValueError(
                "MXFP8 input_activation_save_format_by_fqn selectors did not match "
                f"any converted Linear.Config: {sorted(unmatched_selectors)}."
            )

        for fqn, config, parent, attr in targets:
            new_config = MXFP8Linear.Config(
                in_features=config.in_features,
                out_features=config.out_features,
                bias=config.bias,
                param_init=config.param_init,
                weight_quantization=self.config.weight_quantization,
                input_activation_save_format=save_format_by_fqn[fqn],
            )
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        num_bf16 = sum(
            save_format == "bf16" for save_format in save_format_by_fqn.values()
        )
        num_mxfp8 = len(save_format_by_fqn) - num_bf16
        logger.info(
            "Converted Linear layers to MXFP8Linear with saved input activation "
            f"formats: {num_bf16} bf16, {num_mxfp8} mxfp8"
        )
        logger.debug(f"MXFP8 input activation save format by FQN: {save_format_by_fqn}")
        return model_config


_mxfp8_experts_cache: dict[type, type] = {}


def _get_mxfp8_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create an MXFP8-quantized subclass of *parent_cls*.

    Works for any experts module exposing the ``_grouped_mm`` seam (the common
    ``GroupedExperts`` and ``GptOssGroupedExperts``). The returned class has a
    proper ``_owner`` set by ``__init_subclass__``.

    The subclass overrides ``_grouped_mm`` to call torchao's
    ``_quantize_then_scaled_grouped_mm``.
    """
    if parent_cls in _mxfp8_experts_cache:
        return _mxfp8_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXFP8GroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            recipe_name: str = "mxfp8_rceil"

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import (
                MXFP8TrainingOpConfig,
                MXFP8TrainingRecipe,
            )

            recipe = MXFP8TrainingRecipe(config.recipe_name)
            self._mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)

        def _grouped_mm(self, *, A, B_t, offs):
            from torchao.prototype.moe_training.utils import (
                _quantize_then_scaled_grouped_mm,
            )

            return _quantize_then_scaled_grouped_mm(
                A, B_t, config=self._mxfp8_op_config, offs=offs
            )

    MXFP8GroupedExperts.__name__ = f"MXFP8{parent_cls.__name__}"
    MXFP8GroupedExperts.__qualname__ = f"MXFP8{parent_cls.__name__}"
    _mxfp8_experts_cache[parent_cls] = MXFP8GroupedExperts
    return MXFP8GroupedExperts


class MXFP8GroupedExpertsConverter(QuantizationConverter):
    """Apply MXFP8 quantization to MoE expert grouped GEMMs."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        """
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode
          when computing the e8m0 scale factors.
        """
        pad_multiple: int = 32
        """
        Pad per-expert token groups to this multiple for MXFP8 grouped GEMM alignment.
        The CuTeDSL quantization kernel on sm_100 requires multiples of 128.
        """

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 MoE training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config):
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            # ``parent`` is the RoutedExperts.Config owning inner_experts + dispatcher.
            swap_token_dispatcher(parent, self.config.pad_multiple)
            base_module_cls = type(config)._owner
            quantized_cls = _get_mxfp8_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                recipe_name=self.config.recipe_name,
            )
            if parent is None:
                model_config = new_config
            elif isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            f"Converted GroupedExperts to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm ops"
        )
        return model_config
