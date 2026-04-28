# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
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


def swap_token_dispatcher(config, pad_multiple: int) -> None:
    """Swap token dispatcher config to support padded grouped GEMMs.

    Requires a dispatcher that handles padding (TorchAOTokenDispatcher or
    DeepEP hybridep). Raises ValueError if the dispatcher doesn't support it.
    """
    td = config.token_dispatcher
    if isinstance(td, AllToAllTokenDispatcher.Config) and not isinstance(
        td, TorchAOTokenDispatcher.Config
    ):
        config.token_dispatcher = TorchAOTokenDispatcher.Config(
            num_experts=td.num_experts,
            top_k=td.top_k,
            score_before_experts=td.score_before_experts,
            pad_multiple=pad_multiple,
        )
    elif isinstance(td, DeepEPTokenDispatcher.Config):
        if td.comm_backend == "deepep":
            raise ValueError(
                "DeepEP does not support pad_multiple. "
                "Use hybridep or standard comm backend instead."
            )
        config.token_dispatcher = DeepEPTokenDispatcher.Config(
            num_experts=td.num_experts,
            top_k=td.top_k,
            score_before_experts=td.score_before_experts,
            comm_backend=td.comm_backend,
            non_blocking_capacity_factor=td.non_blocking_capacity_factor,
            pad_multiple=pad_multiple,
        )
    else:
        raise ValueError(
            "MoE quantization requires a token dispatcher that handles "
            "padding (TorchAOTokenDispatcher or hybridep). "
            "Enable expert parallelism or use a compatible comm backend."
        )


def has_quantization(model_config) -> bool:
    """Check if any module in the model config has quantization applied."""
    from torchtitan.components.quantization import (
        _QuantizedGroupedExpertsConfig,
        QuantizedLinearConfig,
    )

    has_quant_linear = any(
        isinstance(config, QuantizedLinearConfig)
        for _fqn, config, _parent, _attr in model_config.traverse(Linear.Config)
    )
    has_quant_moe = any(
        isinstance(config, _QuantizedGroupedExpertsConfig)
        for _fqn, config, _parent, _attr in model_config.traverse(GroupedExperts.Config)
    )
    return has_quant_linear or has_quant_moe
