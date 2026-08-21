# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, fields
from importlib.util import find_spec
from typing import Literal

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .utils import swap_token_dispatcher

try:
    from torchao.prototype.moe_training.mxfp8_linear import (
        mxfp8_mm_cached_weight,
        MXFP8Linear as TorchAOMXFP8Linear,
        quantize_weight_for_mxfp8_cache,
    )

    class MXFP8Linear(TorchAOMXFP8Linear, Module):
        """Inherits from Module (not Linear) to satisfy the Module protocol
        (init_states, _param_init) while avoiding MRO conflicts with
        Linear.__init__. Config still inherits from Linear.Config for
        field compatibility.
        """

        # Inference weight cache (see update_mxfp8_weight_cache). None -> the
        # dynamic path (weight quantized every forward). A generator refreshes
        # this once per weight update so static weights are never re-quantized.
        _mxfp8_weight_cache = None

        @dataclass(kw_only=True, slots=True)
        class Config(Linear.Config):
            """Drop-in replacement for Linear.Config that builds MXFP8Linear."""

            bf16_bwd: bool = False
            """If True, quantize the forward to MXFP8 but run the backward in high
            precision (bf16, straight-through) -- i.e. MXFP8 QAT."""

        def __init__(self, config: Config):
            TorchAOMXFP8Linear.__init__(
                self,
                config.in_features,
                config.out_features,
                bias=config.bias,
                bf16_bwd=config.bf16_bwd,
            )

        def update_mxfp8_weight_cache(self) -> None:
            """Pre-quantize the (static) weight to the MXFP8 MXTensor the forward
            consumes, so subsequent forwards skip the per-forward weight quant.

            Intended for inference (generator) where the weight is constant
            between policy syncs; call once per weight update. Bitwise-identical
            to the dynamic path because it quantizes the same ``self.weight`` with
            the same modes the forward would use.
            """
            self._mxfp8_weight_cache = quantize_weight_for_mxfp8_cache(
                self.weight,
                self.scale_calculation_mode,
                self.kernel_preference,
            )

        def clear_mxfp8_weight_cache(self) -> None:
            self._mxfp8_weight_cache = None

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if self._mxfp8_weight_cache is None:
                return TorchAOMXFP8Linear.forward(self, input)
            # Cached (inference) path: reuse the pre-quantized weight, quantize only
            # the input. Uses the plain forward-only helper -- not the autograd
            # function -- because the cached MXTensor weight is an inference tensor
            # and ``save_for_backward`` on it would raise "Cannot set version_counter
            # for inference tensor". Generation never needs the backward.
            output = mxfp8_mm_cached_weight(
                input,
                self._mxfp8_weight_cache,
                kernel_preference=self.kernel_preference,
                scale_calculation_mode=self.scale_calculation_mode,
            )
            if self.bias is not None:
                output = output + self.bias.to(output.dtype)
            return output

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


class MXFP8LinearQATConverter(QuantizationConverter):
    """MXFP8 QAT for dense Linear layers: real mxfp8 forward, bf16 backward.

    Like MXFP8LinearConverter, but the swapped MXFP8Linear layers quantize the
    forward to MXFP8 while computing the entire backward in high precision (bf16,
    straight-through). Intended for QAT where a low-precision backward is not needed.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        fqns: list[str] = field(default_factory=list)
        """
        List of fully qualified names of modules to apply MXFP8 QAT to. Only
        Linear.Config entries whose FQN contains a match are converted. If empty,
        all Linear modules are converted.
        """

    def __init__(self, config: Config):
        self.config = config

        if MXFP8Linear is None:
            raise ImportError(
                "torchao is not installed. Please install it to use MXFP8 linear layers."
            )

        if not has_cuda_capability(10, 0):
            raise ValueError("MXFP8 is only supported on SM100 or later architectures")

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
                    bf16_bwd=True,
                )
                if parent is None:
                    model_config = new_config
                elif isinstance(parent, list):
                    parent[attr] = new_config
                else:
                    setattr(parent, attr, new_config)

        logger.info(
            "Converted Linear layers to MXFP8 QAT (mxfp8 forward, bf16 backward)"
        )
        return model_config


class _MXFP8GroupedExpertsWeightCacheMixin:
    """Mixin adding an inference weight cache to an mxfp8 grouped-experts module.

    The dynamic path re-quantizes the (static) expert weights to MXFP8 on every
    forward. For a generator whose weights change only at policy syncs that is
    pure overhead -- and especially costly for grouped experts, where the whole
    G*K*N weight is re-quantized but only the routed activation rows are new.

    When the cache is populated (via ``update_mxfp8_weight_cache`` after a weight
    update) ``forward`` reuses the pre-quantized weights and quantizes only the
    activation. The result is bitwise-identical to the dynamic path when the
    cache is fresh: the cached forward mirrors ``GroupedExperts.forward`` and the
    torchao helpers reproduce exactly the forward quant of ``_compute_fwd_sm100``.

    Under a CUDA graph the cache is also what makes generation apply real mxfp8 to
    the experts: the dynamic grouped path skips quant and runs bf16 under
    ``torch.inference_mode()``, and a graph captures whichever branch is live at
    capture time. So the generator populates the cache once before capture (see
    the vLLM wrapper) and refreshes it in place afterward -- the buffers keep
    their addresses so graph replay sees each weight update.

    Applied via MRO (``class X(_MXFP8GroupedExpertsWeightCacheMixin, parent)``) so
    the cached ``forward`` overrides the parent's and ``super().forward(...)``
    still reaches the parent's dynamic path.
    """

    # None -> dynamic path. Otherwise a dict name -> (weight_e4m3, scales_blocked).
    _mxfp8_weight_cache = None
    _mxfp8_kernel_preference = None
    _mxfp8_scale_calculation_mode = None

    # The expert weights to cache. A subclass with a different weight layout
    # (e.g. a fused gate+up parameter) overrides this and ``_mxfp8_weight_t``.
    _mxfp8_cached_weights: tuple[str, ...] = ("w1_EFD", "w2_EDF", "w3_EFD")

    def _mxfp8_weight_t(self, name: str, hp: torch.Tensor) -> torch.Tensor:
        """Map a high-precision expert weight to the ``weight_t`` grouped_mm sees."""
        # (E, F, D) -> (E, D, F) == (E, K, N).
        return hp.bfloat16().transpose(-2, -1)

    def update_mxfp8_weight_cache(self) -> None:
        from torchao.prototype.moe_training.mxfp8_grouped_mm import (
            quantize_grouped_weight_for_cache,
        )
        from torchao.prototype.moe_training.utils import unwrap_weight

        # Reuse the existing cache tensors in place when present so their storage
        # (and thus device pointers) stays stable across weight syncs. A CUDA graph
        # captured over the cached forward bakes in these addresses; replay reads
        # from them, so copy_ is what makes a post-capture weight update visible in
        # the graph. The first call allocates; later calls overwrite in place. The
        # weight shapes are static across syncs, so the quantized shapes match.
        cache = self._mxfp8_weight_cache if self._mxfp8_weight_cache is not None else {}
        kernel_preference = None
        scale_calculation_mode = None
        # The cache is a quantized weight buffer, never part of autograd. Run the
        # whole update under no_grad so allocation and the in-place copy_ share the
        # same grad mode regardless of the caller's context: the pre-capture
        # population (vLLM wrapper) and the per-sync refresh (generator) run with
        # different grad states, and mixing them would trip the "view created in
        # no_grad modified in grad mode" guard on the in-place copy_.
        with torch.no_grad():
            for name in self._mxfp8_cached_weights:
                param = getattr(self, name)
                # DTensor (outer) wraps the mxfp8 weight-wrapper (local) under EP/TP;
                # mirror GroupedExperts.forward, which uses the local tensor.
                wrapper = param.to_local() if isinstance(param, DTensor) else param
                # The mxfp8 op config (kernel preference, scale mode) lives on the
                # weight wrapper; read it so the cached forward matches the dynamic one.
                kernel_preference = wrapper.config.kernel_preference  # type: ignore[missing-attribute]
                scale_calculation_mode = wrapper.config.scale_calculation_mode  # type: ignore[missing-attribute]
                hp = unwrap_weight(wrapper)
                weight_t = self._mxfp8_weight_t(name, hp)
                weight_e4m3, weight_scales_blocked = quantize_grouped_weight_for_cache(
                    weight_t, scale_calculation_mode
                )
                if name in cache:
                    cached_e4m3, cached_scales = cache[name]
                    cached_e4m3.copy_(weight_e4m3)
                    cached_scales.copy_(weight_scales_blocked)
                else:
                    cache[name] = (weight_e4m3, weight_scales_blocked)
        self._mxfp8_weight_cache = cache
        self._mxfp8_kernel_preference = kernel_preference
        self._mxfp8_scale_calculation_mode = scale_calculation_mode

    def clear_mxfp8_weight_cache(self) -> None:
        self._mxfp8_weight_cache = None

    def _mxfp8_cached_grouped_mm(
        self,
        act: torch.Tensor,
        weight_key: str,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        """One grouped GEMM against a pre-quantized cached weight."""
        from torchao.prototype.moe_training.mxfp8_grouped_mm import (
            mxfp8_scaled_grouped_mm_cached_weight,
        )

        weight_e4m3, weight_scales_blocked = self._mxfp8_weight_cache[weight_key]
        return mxfp8_scaled_grouped_mm_cached_weight(
            act,
            weight_e4m3,
            weight_scales_blocked,
            offsets_E,
            scale_calculation_mode=self._mxfp8_scale_calculation_mode,
            kernel_preference=self._mxfp8_kernel_preference,
        )

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if self._mxfp8_weight_cache is None:
            return super().forward(x_RD, num_tokens_per_expert_E)  # type: ignore[missing-attribute]

        # Cached inference path: mirrors GroupedExperts.forward (silu(x@w1) * (x@w3),
        # then @w2) but reuses the pre-quantized weights. Token groups are already
        # padded by the mxfp8 token dispatcher, so no padding happens here.
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        grouped_mm = self._mxfp8_cached_grouped_mm

        h_RF = F.silu(grouped_mm(x_RD.bfloat16(), "w1_EFD", offsets_E))
        h_RF = h_RF * grouped_mm(x_RD.bfloat16(), "w3_EFD", offsets_E)
        return grouped_mm(h_RF, "w2_EDF", offsets_E).type_as(x_RD)


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

    class MXFP8GroupedExperts(  # type: ignore[valid-type, misc]
        _MXFP8GroupedExpertsWeightCacheMixin, parent_cls
    ):
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
        pad_multiple: int = 128
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

    Like ``_get_mxfp8_grouped_experts_cls`` but the grouped GEMMs use the QAT
    autograd function via the ``bf16_bwd`` config (real mxfp8 forward, bf16
    backward), using torchao's default quant kernels for the
    forward quantization.
    """
    if parent_cls in _mxfp8_qat_experts_cache:
        return _mxfp8_qat_experts_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # type: ignore[attr-defined]

    class MXFP8QATGroupedExperts(  # type: ignore[valid-type, misc]
        _MXFP8GroupedExpertsWeightCacheMixin, parent_cls
    ):
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            recipe_name: str = "mxfp8_rceil"

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig
            from torchao.prototype.mx_formats.config import ScaleCalculationMode
            from torchao.quantization.quant_api import quantize_

            # QAT: real mxfp8 forward, high-precision (bf16) backward.
            # - bf16_bwd=True routes to the autograd function whose backward is a
            #   plain bf16 torch._grouped_mm (straight-through estimator).
            op_config = MXFP8TrainingOpConfig(
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
                bf16_bwd=True,
            )
            quantize_(
                self,
                config=op_config,
                filter_fn=lambda mod, _fqn: isinstance(mod, GroupedExperts),
            )

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


def refresh_mxfp8_weight_caches(model: torch.nn.Module) -> int:
    """Refresh the inference weight cache on every mxfp8 module under ``model``.

    Quantizes each mxfp8 linear / grouped-experts weight to MXFP8 once so
    subsequent forwards skip the per-forward weight quant. Call after a weight
    update (e.g. in the generator's ``_pull_model_state_dict``). Returns the
    number of modules refreshed. No-op-safe: modules without a cache are skipped.

    Only meaningful for inference, where weights are static between updates; do
    not use in the trainer (weights change every optimizer step and the backward
    needs the dynamic quant path).
    """
    count = 0
    for module in model.modules():
        update_fn = getattr(module, "update_mxfp8_weight_cache", None)
        if callable(update_fn):
            update_fn()
            count += 1
    return count
