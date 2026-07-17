# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    HybridEPTokenDispatcher,
    TorchAOTokenDispatcher,
)


def module_filter_fn(config: Linear.Config, fqn: str, filter_fqns: list[str]) -> bool:
    """
    Filter function to determine which Linear.Config should be converted.
    For both Float8 and MXFP8, we only convert Linear modules
    with dimensions divisible by 16 and not matching any filtered FQNs.
    """
    # All dims must be divisible by 16 due to float8 tensorcore hardware requirements.
    dims_multiples_of_16 = (
        config.in_features % 16 == 0 and config.out_features % 16 == 0
    )

    # If the fqn matches any filtered fqn, then we should not convert this module.
    is_filtered_fqn = any(filter_fqn in fqn for filter_fqn in filter_fqns)

    return dims_multiples_of_16 and not is_filtered_fqn


def swap_token_dispatcher(routed_experts_config, pad_multiple: int) -> None:
    """Swap the routed-experts token dispatcher config to support padded grouped GEMMs.

    Takes the ``RoutedExperts.Config`` (which owns the ``token_dispatcher`` child) and
    swaps its dispatcher in place. Requires a dispatcher that handles padding
    (TorchAOTokenDispatcher or DeepEP hybridep). Raises ValueError if the
    dispatcher doesn't support it.
    """
    dispatcher = routed_experts_config.token_dispatcher
    if isinstance(dispatcher, AllToAllTokenDispatcher.Config) and not isinstance(
        dispatcher, TorchAOTokenDispatcher.Config
    ):
        routed_experts_config.token_dispatcher = TorchAOTokenDispatcher.Config(
            num_experts=dispatcher.num_experts,
            top_k=dispatcher.top_k,
            pad_multiple=pad_multiple,
        )
    elif isinstance(dispatcher, HybridEPTokenDispatcher.Config):
        routed_experts_config.token_dispatcher = HybridEPTokenDispatcher.Config(
            num_experts=dispatcher.num_experts,
            top_k=dispatcher.top_k,
            non_blocking_capacity_factor=dispatcher.non_blocking_capacity_factor,
            pad_multiple=pad_multiple,
        )
    else:
        raise ValueError(
            f"MoE quantization requires a token dispatcher that supports "
            f"padding (TorchAOTokenDispatcher or HybridEPTokenDispatcher), "
            f"got {type(dispatcher).__name__}."
        )


def has_quantization(model_config) -> bool:
    """Check if any module in the model config has quantization applied."""
    from torchtitan.components.quantization.float8 import (
        _float8_experts_cache,
        Float8Linear,
    )
    from torchtitan.components.quantization.mx import _mxfp8_experts_cache, MXFP8Linear

    quant_linear_types: list[type] = []
    if Float8Linear is not None:
        quant_linear_types.append(Float8Linear.Config)
    if MXFP8Linear is not None:
        quant_linear_types.append(MXFP8Linear.Config)

    has_quant_linear = bool(quant_linear_types) and any(
        isinstance(config, tuple(quant_linear_types))
        for _fqn, config, _parent, _attr in model_config.traverse(Linear.Config)
    )
    quant_experts_types = tuple(
        cls.Config  # type: ignore[attr-defined]
        for cls in (*_float8_experts_cache.values(), *_mxfp8_experts_cache.values())
    )
    has_quant_moe = bool(quant_experts_types) and any(
        isinstance(config, quant_experts_types)
        for _fqn, config, _parent, _attr in model_config.traverse(GroupedExperts.Config)
    )
    return has_quant_linear or has_quant_moe
