# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields
from importlib.util import find_spec
from typing import Literal

import torch

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import swap_token_dispatcher


def _enable_triton_dim1_for_mxfp8_linear() -> None:
    from torchao.prototype.moe_training import tensor as mxfp8_tensor
    from torchao.prototype.mx_formats.config import (
        MXFP8Dim0CastKernelChoice,
        MXFP8Dim1CastKernelChoice,
    )
    from torchao.prototype.mx_formats.mx_linear import mx_mm

    if getattr(mxfp8_tensor, "_torchtitan_triton_dim1_enabled", False):
        return

    def _to_mxfp8_then_scaled_mm_triton_dim1(
        input_hp,
        weight_hp,
        kernel_preference,
        scale_calculation_mode,
        wgrad_with_hp=False,
    ):
        return mx_mm.apply(
            input_hp,
            weight_hp,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            32,
            kernel_preference,
            MXFP8Dim0CastKernelChoice.TRITON,
            MXFP8Dim1CastKernelChoice.TRITON,
            scale_calculation_mode,
            wgrad_with_hp,
        )

    mxfp8_tensor._to_mxfp8_then_scaled_mm = _to_mxfp8_then_scaled_mm_triton_dim1
    mxfp8_tensor._torchtitan_triton_dim1_enabled = True


@torch._dynamo.allow_in_graph
class _MXFP8SharedInputGateUp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        w1_hp: torch.Tensor,
        w3_hp: torch.Tensor,
        kernel_preference,
        scale_calculation_mode,
        wgrad_with_hp: bool,
    ):
        from torchao.prototype.mx_formats.config import MXFP8Dim0CastKernelChoice
        from torchao.prototype.mx_formats.mx_tensor import MXTensor

        ctx.save_for_backward(input_hp, w1_hp, w3_hp)
        ctx.kernel_preference = kernel_preference
        ctx.scale_calculation_mode = scale_calculation_mode
        ctx.wgrad_with_hp = wgrad_with_hp

        input_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_orig_shape[-1])
        input_mx_r_dim0 = MXTensor.to_mx(
            input_hp_r,
            torch.float8_e4m3fn,
            32,
            scale_calculation_mode,
            kernel_preference,
            mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
        )
        w1_mx_dim0 = MXTensor.to_mx(
            w1_hp,
            torch.float8_e4m3fn,
            32,
            scale_calculation_mode,
            kernel_preference,
            mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
        )
        w3_mx_dim0 = MXTensor.to_mx(
            w3_hp,
            torch.float8_e4m3fn,
            32,
            scale_calculation_mode,
            kernel_preference,
            mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
        )
        w1_out = torch.mm(input_mx_r_dim0, w1_mx_dim0.t())
        w3_out = torch.mm(input_mx_r_dim0, w3_mx_dim0.t())
        return (
            w1_out.reshape(*input_orig_shape[:-1], w1_out.shape[-1]),
            w3_out.reshape(*input_orig_shape[:-1], w3_out.shape[-1]),
        )

    @staticmethod
    def backward(ctx, grad_w1_out_hp: torch.Tensor, grad_w3_out_hp: torch.Tensor):
        from torchao.prototype.mx_formats.config import (
            MXFP8Dim0CastKernelChoice,
            MXFP8Dim1CastKernelChoice,
        )
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper

        input_hp, w1_hp, w3_hp = ctx.saved_tensors
        kernel_preference = ctx.kernel_preference
        scale_calculation_mode = ctx.scale_calculation_mode

        input_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_orig_shape[-1])
        grad_w1_orig_shape = grad_w1_out_hp.shape
        grad_w3_orig_shape = grad_w3_out_hp.shape
        grad_w1_out_hp_r = grad_w1_out_hp.reshape(-1, grad_w1_orig_shape[-1])
        grad_w3_out_hp_r = grad_w3_out_hp.reshape(-1, grad_w3_orig_shape[-1])

        grad_w1_out_mx_dim0 = MXTensor.to_mx(
            grad_w1_out_hp_r,
            torch.float8_e4m3fn,
            32,
            scale_calculation_mode,
            kernel_preference,
            mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
        )
        grad_w3_out_mx_dim0 = MXTensor.to_mx(
            grad_w3_out_hp_r,
            torch.float8_e4m3fn,
            32,
            scale_calculation_mode,
            kernel_preference,
            mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
        )
        w1_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
            w1_hp,
            32,
            torch.float8_e4m3fn,
            w1_hp.dtype,
            kernel_preference,
            MXFP8Dim1CastKernelChoice.TRITON,
            scale_calculation_mode,
        )
        w3_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
            w3_hp,
            32,
            torch.float8_e4m3fn,
            w3_hp.dtype,
            kernel_preference,
            MXFP8Dim1CastKernelChoice.TRITON,
            scale_calculation_mode,
        )
        grad_input = torch.mm(grad_w1_out_mx_dim0, w1_mx_dim1.t()) + torch.mm(
            grad_w3_out_mx_dim0, w3_mx_dim1.t()
        )
        grad_input = grad_input.reshape(*grad_w1_orig_shape[:-1], grad_input.shape[-1])

        if ctx.wgrad_with_hp:
            grad_w1 = torch.mm(grad_w1_out_hp_r.t(), input_hp_r)
            grad_w3 = torch.mm(grad_w3_out_hp_r.t(), input_hp_r)
        else:
            input_t_mx_dim0_tmp = _to_mxfp8_dim1_kernel_wrapper(
                input_hp_r,
                32,
                torch.float8_e4m3fn,
                input_hp_r.dtype,
                kernel_preference,
                MXFP8Dim1CastKernelChoice.TRITON,
                scale_calculation_mode,
            )
            input_t_mx_dim0 = input_t_mx_dim0_tmp.t()
            grad_w1_out_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
                grad_w1_out_hp_r,
                32,
                torch.float8_e4m3fn,
                grad_w1_out_hp_r.dtype,
                kernel_preference,
                MXFP8Dim1CastKernelChoice.TRITON,
                scale_calculation_mode,
            )
            grad_w3_out_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
                grad_w3_out_hp_r,
                32,
                torch.float8_e4m3fn,
                grad_w3_out_hp_r.dtype,
                kernel_preference,
                MXFP8Dim1CastKernelChoice.TRITON,
                scale_calculation_mode,
            )
            grad_w1 = torch.mm(grad_w1_out_mx_dim1, input_t_mx_dim0)
            grad_w3 = torch.mm(grad_w3_out_mx_dim1, input_t_mx_dim0)

        return grad_input, grad_w1, grad_w3, None, None, None


def mxfp8_shared_input_gate_up(
    input_hp: torch.Tensor,
    w1_wrapper: torch.Tensor,
    w3_wrapper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig
    from torchao.prototype.moe_training.utils import unwrap_weight

    config = w1_wrapper.config
    if config != w3_wrapper.config:
        raise ValueError("MXFP8 w1 and w3 weights must use the same config")
    if not isinstance(config, MXFP8TrainingOpConfig):
        raise ValueError("shared gate/up MXFP8 path requires MXFP8TrainingOpConfig")

    return _MXFP8SharedInputGateUp.apply(
        input_hp,
        unwrap_weight(w1_wrapper),
        unwrap_weight(w3_wrapper),
        config.kernel_preference,
        config.scale_calculation_mode,
        config.wgrad_with_hp,
    )


@torch._dynamo.allow_in_graph
class _MXFP8SharedInputQKV(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        wq_hp: torch.Tensor,
        wk_hp: torch.Tensor,
        wv_hp: torch.Tensor,
        kernel_preference,
        scale_calculation_mode,
        wgrad_with_hp: bool,
    ):
        from torchao.prototype.mx_formats.config import MXFP8Dim0CastKernelChoice
        from torchao.prototype.mx_formats.mx_tensor import MXTensor

        ctx.save_for_backward(input_hp, wq_hp, wk_hp, wv_hp)
        ctx.kernel_preference = kernel_preference
        ctx.scale_calculation_mode = scale_calculation_mode
        ctx.wgrad_with_hp = wgrad_with_hp

        input_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_orig_shape[-1])
        input_mx_r_dim0 = MXTensor.to_mx(
            input_hp_r,
            torch.float8_e4m3fn,
            32,
            scale_calculation_mode,
            kernel_preference,
            mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
        )

        outputs = []
        for weight_hp in (wq_hp, wk_hp, wv_hp):
            weight_mx_dim0 = MXTensor.to_mx(
                weight_hp,
                torch.float8_e4m3fn,
                32,
                scale_calculation_mode,
                kernel_preference,
                mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
            )
            output = torch.mm(input_mx_r_dim0, weight_mx_dim0.t())
            outputs.append(output.reshape(*input_orig_shape[:-1], output.shape[-1]))
        return tuple(outputs)

    @staticmethod
    def backward(
        ctx,
        grad_q_out_hp: torch.Tensor,
        grad_k_out_hp: torch.Tensor,
        grad_v_out_hp: torch.Tensor,
    ):
        from torchao.prototype.mx_formats.config import (
            MXFP8Dim0CastKernelChoice,
            MXFP8Dim1CastKernelChoice,
        )
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper

        input_hp, wq_hp, wk_hp, wv_hp = ctx.saved_tensors
        kernel_preference = ctx.kernel_preference
        scale_calculation_mode = ctx.scale_calculation_mode

        input_orig_shape = input_hp.shape
        input_hp_r = input_hp.reshape(-1, input_orig_shape[-1])
        grad_outputs_hp = (grad_q_out_hp, grad_k_out_hp, grad_v_out_hp)
        weights_hp = (wq_hp, wk_hp, wv_hp)

        grad_input = None
        grad_weights = []
        input_t_mx_dim0 = None
        for grad_output_hp, weight_hp in zip(grad_outputs_hp, weights_hp):
            grad_output_orig_shape = grad_output_hp.shape
            grad_output_hp_r = grad_output_hp.reshape(
                -1, grad_output_orig_shape[-1]
            )
            grad_output_mx_dim0 = MXTensor.to_mx(
                grad_output_hp_r,
                torch.float8_e4m3fn,
                32,
                scale_calculation_mode,
                kernel_preference,
                mxfp8_dim0_cast_kernel_choice=MXFP8Dim0CastKernelChoice.TRITON,
            )
            weight_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
                weight_hp,
                32,
                torch.float8_e4m3fn,
                weight_hp.dtype,
                kernel_preference,
                MXFP8Dim1CastKernelChoice.TRITON,
                scale_calculation_mode,
            )
            grad_input_part = torch.mm(grad_output_mx_dim0, weight_mx_dim1.t())
            grad_input = (
                grad_input_part if grad_input is None else grad_input + grad_input_part
            )

            if ctx.wgrad_with_hp:
                grad_weight = torch.mm(grad_output_hp_r.t(), input_hp_r)
            else:
                if input_t_mx_dim0 is None:
                    input_t_mx_dim0_tmp = _to_mxfp8_dim1_kernel_wrapper(
                        input_hp_r,
                        32,
                        torch.float8_e4m3fn,
                        input_hp_r.dtype,
                        kernel_preference,
                        MXFP8Dim1CastKernelChoice.TRITON,
                        scale_calculation_mode,
                    )
                    input_t_mx_dim0 = input_t_mx_dim0_tmp.t()
                grad_output_mx_dim1 = _to_mxfp8_dim1_kernel_wrapper(
                    grad_output_hp_r,
                    32,
                    torch.float8_e4m3fn,
                    grad_output_hp_r.dtype,
                    kernel_preference,
                    MXFP8Dim1CastKernelChoice.TRITON,
                    scale_calculation_mode,
                )
                grad_weight = torch.mm(grad_output_mx_dim1, input_t_mx_dim0)
            grad_weights.append(grad_weight)

        assert grad_input is not None
        grad_input = grad_input.reshape(*input_orig_shape[:-1], grad_input.shape[-1])
        return (
            grad_input,
            grad_weights[0],
            grad_weights[1],
            grad_weights[2],
            None,
            None,
            None,
        )


def mxfp8_shared_input_qkv(
    input_hp: torch.Tensor,
    wq_wrapper: torch.Tensor,
    wk_wrapper: torch.Tensor,
    wv_wrapper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig
    from torchao.prototype.moe_training.utils import unwrap_weight

    config = wq_wrapper.config
    if config != wk_wrapper.config or config != wv_wrapper.config:
        raise ValueError("MXFP8 Q/K/V weights must use the same config")
    if not isinstance(config, MXFP8TrainingOpConfig):
        raise ValueError("shared Q/K/V MXFP8 path requires MXFP8TrainingOpConfig")

    return _MXFP8SharedInputQKV.apply(
        input_hp,
        unwrap_weight(wq_wrapper),
        unwrap_weight(wk_wrapper),
        unwrap_weight(wv_wrapper),
        config.kernel_preference,
        config.scale_calculation_mode,
        config.wgrad_with_hp,
    )


class MXFP8Linear(Linear):
    """Linear that applies MXFP8 quantization in its constructor."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for Linear.Config that builds MXFP8Linear."""

        _recipe_name: str = "mxfp8_rceil"

    def __init__(self, config: Config):
        super().__init__(config)
        _enable_triton_dim1_for_mxfp8_linear()
        from torchao.prototype.moe_training.config import (
            MXFP8TrainingOpConfig,
            MXFP8TrainingRecipe,
        )
        from torchao.quantization.quant_api import quantize_

        recipe = MXFP8TrainingRecipe(config._recipe_name)
        mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
        quantize_(self, config=mxfp8_op_config)


class MXFP8LinearConverter(QuantizationConverter):
    """Apply MXFP8 quantization to modules matching FQNs (e.g. Flux blocks)."""

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        """
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode when computing the e8m0 scale factors.
        """

        fqns: list[str] = field(default_factory=list)
        """
        *Prototype feature, performance optimization still in progress*
        Comma-separated list of fully qualified names of MoE modules to apply MXFP8 dynamic quantization
        on grouped GEMM operations.
        This is a prototype feature that requires the torchao nightly build.
        """

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 linear layers."
            )

        # Can be removed if we enable the emulated versions
        assert has_cuda_capability(
            10, 0
        ), "MXFP8 is only supported on SM100 or later architectures"

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config) -> None:
        fqns = self.config.fqns
        for fqn, config, parent, attr in model_config.traverse(Linear.Config):
            if not fqns or any(target_fqn in fqn for target_fqn in fqns):
                new_config = MXFP8Linear.Config(
                    in_features=config.in_features,
                    out_features=config.out_features,
                    bias=config.bias,
                    param_init=config.param_init,
                    _recipe_name=self.config.recipe_name,
                )
                if isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info(
            f"Converted modules to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm and linear ops"
        )


_mxfp8_experts_cache: dict[type, type] = {}


def _get_mxfp8_grouped_experts_cls(parent_cls: type) -> type:
    """Get or create an MXFP8-quantized subclass of *parent_cls*.

    Works for any ``GroupedExperts`` subclass (e.g. gpt-oss variants).
    The returned class has a proper ``_owner`` set by ``__init_subclass__``.
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
            from torchao.quantization.quant_api import quantize_

            recipe = MXFP8TrainingRecipe(config.recipe_name)
            mxfp8_op_config = MXFP8TrainingOpConfig.from_recipe(recipe)
            quantize_(
                self,
                config=mxfp8_op_config,
                filter_fn=lambda mod, _fqn: isinstance(mod, GroupedExperts),
            )

    MXFP8GroupedExperts.__name__ = f"MXFP8{parent_cls.__name__}"
    MXFP8GroupedExperts.__qualname__ = f"MXFP8{parent_cls.__name__}"
    _mxfp8_experts_cache[parent_cls] = MXFP8GroupedExperts
    return MXFP8GroupedExperts


class MXFP8GroupedExpertsConverter(QuantizationConverter):
    """Apply MXFP8 quantization to MoE expert grouped GEMMs."""

    # MXFP8: scaling block size is (1 x 32), so contracting dim must be divisible by 32.
    PAD_MULTIPLE = 32

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        recipe_name: Literal["mxfp8_rceil"] = "mxfp8_rceil"
        """
        Quantization recipe name for grouped GEMMs. Options: ["mxfp8_rceil"]

        - mxfp8_rceil: MXFP8 dynamic quantization with RCEIL rounding mode when computing the e8m0 scale factors.
        """

    def __init__(self, config: Config):
        self.config = config

        if find_spec("torchao") is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 MoE training."
            )

        assert has_cuda_capability(
            10, 0
        ), "MXFP8 is only supported on SM100 or later architectures"

        if not self.config.model_compile_enabled:
            logger.warning(
                "torch.compile enablement is required for highest performance "
                "of MXFP8 dynamic quantization."
            )

    def convert(self, model_config) -> None:
        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            swap_token_dispatcher(config, self.PAD_MULTIPLE)
            base_module_cls = type(config)._owner
            quantized_cls = _get_mxfp8_grouped_experts_cls(base_module_cls)
            config_cls = quantized_cls.Config  # type: ignore[attr-defined]
            new_config = config_cls(
                **{f.name: getattr(config, f.name) for f in fields(config)},
                recipe_name=self.config.recipe_name,
            )
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)

        logger.info(
            f"Converted GroupedExperts to use dynamic {self.config.recipe_name} "
            "quantization for grouped_mm and linear ops"
        )
