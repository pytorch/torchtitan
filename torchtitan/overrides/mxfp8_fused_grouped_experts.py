# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Gate+up fusion for MXFP8-QAT routed experts.

``torchtitan.overrides.fused_swiglu.fused_grouped_experts`` fuses the routed
experts' gate and up projections into a single ``w13`` parameter, turning three
grouped GEMMs per layer into two. It only accepts a stock
``GroupedExperts.Config``, so it is a no-op once ``MXFP8GroupedExpertsQATConverter``
has run -- converters are applied when the ``ModelSpec`` is built, overrides when
the model is built, so the mxfp8 config always arrives first. This module
supplies the missing combination: the same fusion, applied on top of the mxfp8
QAT config.

Why bother: the cached inference forward quantizes the activation once per
grouped GEMM, so the unfused layout quantizes ``x_RD`` twice (once for ``w1``,
once for ``w3``). Fusing makes it once. Activation quant is where the mxfp8
decode penalty lives, so this removes a third of the grouped GEMMs and, more to
the point, half of the per-layer activation quantization of ``x_RD``.

Activate alongside the other generator overrides::

    --override.imports torchtitan.overrides.mxfp8_fused_grouped_experts.mxfp8_fused_grouped_experts

Use this *instead of* ``fused_swiglu.fused_grouped_experts``, not alongside it:
both target ``GroupedExperts.Config``, and override claims are checked before
the factories run, so listing both is rejected as a node conflict even though
only one would act.
"""

from dataclasses import dataclass

import torch

from torchtitan.components.quantization.mx import _MXFP8GroupedExpertsWeightCacheMixin
from torchtitan.config import derive, override
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.overrides.fused_swiglu import (
    _fuse_w13_grouped_experts_param_init,
    _fuse_w13_grouped_experts_sharding,
    FusedGroupedExperts,
    silu_and_mul_op,
)

__all__ = ["mxfp8_fused_grouped_experts"]


class _MXFP8FusedGroupedExpertsWeightCacheMixin(_MXFP8GroupedExpertsWeightCacheMixin):
    """Weight cache for the fused ``w13`` layout.

    Same contract as the base mixin -- populate via ``update_mxfp8_weight_cache``,
    then ``forward`` reuses the pre-quantized weights -- but over two weights
    instead of three, and with a cached forward that mirrors
    ``FusedGroupedExperts.forward`` rather than ``GroupedExperts.forward``.
    """

    _mxfp8_cached_weights = ("w13", "w2_EDF")

    def _mxfp8_weight_t(self, name: str, hp: torch.Tensor) -> torch.Tensor:
        if name == "w13":
            # (E, F, 2, D) -> (E, 2F, D) -> (E, D, 2F) == (E, K, N). The reshape
            # is a view: w13 is contiguous and gate/up are adjacent, so row
            # ``f * 2 + g`` of the (E, 2F, D) form is gate/up ``g`` of hidden
            # unit ``f`` -- matching how FusedGroupedExperts splits the output.
            E, F, _, D = hp.shape
            return hp.bfloat16().reshape(E, F * 2, D).transpose(-2, -1)
        return super()._mxfp8_weight_t(name, hp)

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if self._mxfp8_weight_cache is None:
            # The base mixin re-checks the cache and falls through to
            # FusedGroupedExperts.forward (the dynamic path).
            return super().forward(x_RD, num_tokens_per_expert_E)

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        gate_up_R2F = self._mxfp8_cached_grouped_mm(x_RD.bfloat16(), "w13", offsets_E)
        # Take the hidden dim from the GEMM output rather than from w13: the
        # parameter's F is global (TP shards it), and the cached weight is stored
        # as (E, N, K), so neither is this rank's output width.
        F = gate_up_R2F.shape[-1] // 2
        gate_RF, up_RF = gate_up_R2F.reshape(-1, F, 2).unbind(-1)
        h_RF = silu_and_mul_op(gate_RF, up_RF, offsets_E)
        return self._mxfp8_cached_grouped_mm(h_RF, "w2_EDF", offsets_E).type_as(x_RD)


_mxfp8_qat_fused_experts_cls: type | None = None


def _get_mxfp8_qat_fused_grouped_experts_cls() -> type:
    """Build (once) the fused + MXFP8-QAT routed-experts class.

    Mirrors ``_get_mxfp8_qat_grouped_experts_cls`` in the quantization component,
    but over :class:`FusedGroupedExperts` and the fused weight cache. Cached
    because the config classes must be identical across calls for config
    comparison and checkpointing to work.
    """
    global _mxfp8_qat_fused_experts_cls
    if _mxfp8_qat_fused_experts_cls is not None:
        return _mxfp8_qat_fused_experts_cls

    class MXFP8QATFusedGroupedExperts(  # type: ignore[valid-type, misc]
        _MXFP8FusedGroupedExpertsWeightCacheMixin, FusedGroupedExperts
    ):
        @dataclass(kw_only=True, slots=True)
        class Config(FusedGroupedExperts.Config):  # type: ignore[misc]
            recipe_name: str = "mxfp8_rceil"

        def __init__(self, config: Config):
            super().__init__(config)
            from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig
            from torchao.prototype.mx_formats.config import ScaleCalculationMode
            from torchao.quantization.quant_api import quantize_

            # Same QAT recipe as the unfused converter: real mxfp8 forward, bf16
            # backward. ``quantize_`` swaps every parameter of the matched module,
            # by traversal rather than by name, so the fused ``w13`` is wrapped
            # without any layout-specific handling.
            op_config = MXFP8TrainingOpConfig(
                scale_calculation_mode=ScaleCalculationMode.RCEIL,
                bf16_bwd=True,
            )
            # Kept so the wrapper can be re-applied if parallelism strips it
            # (see _ensure_mxfp8_wrapped in components/quantization/mx.py).
            self._mxfp8_wrapper_op_config = op_config
            quantize_(
                self,
                config=op_config,
                filter_fn=lambda mod, _fqn: isinstance(mod, GroupedExperts),
            )

    _mxfp8_qat_fused_experts_cls = MXFP8QATFusedGroupedExperts
    return MXFP8QATFusedGroupedExperts


@override(
    target=GroupedExperts.Config,
    description="Fuse gate+up for MXFP8-QAT routed experts (two grouped GEMMs).",
)
def mxfp8_fused_grouped_experts(
    cfg: GroupedExperts.Config,
) -> GroupedExperts.Config:
    # Only claim the MXFP8 QAT wrapper around a *stock* GroupedExperts: the fused
    # layout is defined for that weight layout only. Anything else -- a bf16
    # config (handled by ``fused_swiglu.fused_grouped_experts``), an already
    # fused config, or mxfp8 over a model-specific subclass such as
    # GptOssGroupedExperts -- is left untouched.
    owner = type(cfg)._owner
    if getattr(owner, "_unquantized_cls", None) is not GroupedExperts:
        return cfg

    param_init = _fuse_w13_grouped_experts_param_init(cfg.param_init)
    fused_cls = _get_mxfp8_qat_fused_grouped_experts_cls()
    fused = derive(cfg, fused_cls.Config, param_init=param_init)
    base = cfg.sharding_config
    if base is not None:
        fused.sharding_config = _fuse_w13_grouped_experts_sharding(base)
    return fused
