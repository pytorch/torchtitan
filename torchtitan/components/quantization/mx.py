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
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import swap_token_dispatcher

try:
    from torchao.prototype.moe_training.mxfp8_linear import (
        MXFP8Linear as TorchAOMXFP8Linear,
    )

    class MXFP8Linear(TorchAOMXFP8Linear, Module):
        """Inherits from Module (not Linear) to satisfy the Module protocol
        (init_states, _param_init) while avoiding MRO conflicts with
        Linear.__init__. Config still inherits from Linear.Config for
        field compatibility.
        """

        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            """Drop-in replacement for Linear.Config that builds MXFP8Linear."""

            pass

        def __init__(self, config: Config):
            TorchAOMXFP8Linear.__init__(
                self,
                config.in_features,
                config.out_features,
                bias=config.bias,
            )

except ImportError:
    MXFP8Linear = None


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

    def __init__(self, config: Config):
        self.config = config

        if MXFP8Linear is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 linear layers."
            )

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
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = MXFP8Linear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info("Converted Linear layers to MXFP8Linear")
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


_mxfp8_qat_experts_cache: dict[type, type] = {}


def _get_mxfp8_qat_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create an MXFP8 QAT subclass of *parent_cls*.

    Like ``_get_mxfp8_grouped_experts_cls`` but the grouped GEMMs run
    ``_MXFP8GroupedMMFwdBF16Bwd``: real mxfp8 forward, bf16 backward.
    """
    if parent_cls in _mxfp8_qat_experts_cache:
        return _mxfp8_qat_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXFP8QATGroupedExperts(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            recipe_name: str = "mxfp8_rceil"

        def _grouped_mm(self, *, A, B_t, offs):
            from torchtitan.components.quantization.mxfp8.utils import (
                _MXFP8GroupedMMFwdBF16Bwd,
            )

            return _MXFP8GroupedMMFwdBF16Bwd.apply(A, B_t, offs)

    MXFP8QATGroupedExperts.__name__ = f"MXFP8QAT{parent_cls.__name__}"
    MXFP8QATGroupedExperts.__qualname__ = f"MXFP8QAT{parent_cls.__name__}"
    # The unquantized class this wraps. Lets a later config transform (e.g. an
    # override that fuses gate+up) recognize what was quantized without having
    # to infer it from the MRO.
    MXFP8QATGroupedExperts._unquantized_cls = parent_cls
    _mxfp8_qat_experts_cache[parent_cls] = MXFP8QATGroupedExperts
    return MXFP8QATGroupedExperts


class MXFP8GroupedExpertsQATConverter(QuantizationConverter):
    """MXFP8 QAT for MoE expert grouped GEMMs: real mxfp8 forward, bf16 backward.

    Uses the QAT autograd function (real mxfp8 forward, bf16 backward) with
    torchao's default kernel preference for the forward quantization. Intended for QAT
    (training a model to be robust to mxfp8 inference) where a low-precision backward
    is not needed; gradients flow in high precision (bf16).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        """Recipe name for the forward quantization. Options: ["mxfp8_rceil"]."""
        pad_multiple: int = 128
        """
        Pad per-expert token groups to this multiple for MXFP8 grouped GEMM
        alignment. torchao's default (cutedsl) quant kernel asserts each group is
        a multiple of 128 (``cute_utils.validate_group_sizes``), so this must stay
        at 128 for that kernel.
        """

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 MoE training."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

    def convert(self, model_config):
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            # ``parent`` is the RoutedExperts.Config owning inner_experts + dispatcher.
            swap_token_dispatcher(parent, self.config.pad_multiple)
            base_module_cls = type(config)._owner
            quantized_cls = _get_mxfp8_qat_grouped_experts_cls(base_module_cls)
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
            "Converted GroupedExperts to MXFP8 QAT (mxfp8 forward, bf16 backward, "
            f"pad_multiple={self.config.pad_multiple}) for grouped_mm ops"
        )
        return model_config
