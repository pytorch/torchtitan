# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch
from dist_moe import DistMoeInputScaledRMSNorm

from torchtitan.components.dist_moe import (
    DistMoeRoutedExperts,
    DistMoeRuntime,
    MXFP8DistMoeRoutedExperts,
)
from torchtitan.components.dist_moe.backend import _DistMoeRuntime
from torchtitan.config.transform import DistMoeTransform, MXFP8DistMoeTransform
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.module import Module
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor


def _parameter_initializers() -> dict[str, Any]:
    return {
        "w1_EFD": torch.nn.init.zeros_,
        "w2_EDF": torch.nn.init.zeros_,
        "w3_EFD": torch.nn.init.zeros_,
    }


def _stock_config() -> RoutedExperts.Config:
    return make_routed_experts_config(
        dim=32,
        hidden_dim=64,
        num_experts=4,
        top_k=2,
        param_init=_parameter_initializers(),
        comm_backend="standard",
    )


def _runtime(prefetch: Any = None) -> DistMoeRuntime:
    return DistMoeRuntime(
        config=cast(Any, object()),
        group=cast(Any, object()),
        device=torch.device("cuda"),
        prefetch=prefetch,
    )


class _NativePostprocess(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-8
        gain_center: float = 1.0

    def __init__(self, config: Config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(config.dim))
        self.eps = config.eps
        self.gain_center = config.gain_center

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def to_dist_moe_postprocess(self) -> DistMoeInputScaledRMSNorm:
        """Translate this owned module to the annex-native descriptor."""
        return DistMoeInputScaledRMSNorm(
            self.weight,
            eps=self.eps,
            gain_center=self.gain_center,
        )


def test_runtime_releases_pending_prefetch_after_failure_and_close():
    prefetch = Mock()
    runtime = _runtime(prefetch)
    with (
        patch(
            "torchtitan.components.dist_moe.backend.create_context",
            side_effect=RuntimeError("context creation failed"),
        ),
        pytest.raises(RuntimeError, match="context creation failed"),
    ):
        runtime.initialize()

    prefetch.close.assert_called_once_with()
    runtime.close()
    assert runtime.prefetch is None


def test_runtime_consumes_pipeline_metadata_before_model_forward():
    runtime = _runtime()
    runtime.slots[(3, 7)] = (2, 5)
    runtime.context = Mock()
    context = runtime.context
    hook = Mock()
    runtime._pipeline_hooks.append(hook)

    args, kwargs = runtime.select_pipeline_slot(
        Mock(),
        (torch.empty(1),),
        {
            "pipeline_stage_index": 3,
            "pipeline_microbatch_index": 7,
            "input_batch": "value",
        },
    )

    context.select_activation_slot.assert_called_once_with(2, 5)
    assert len(args) == 1
    assert kwargs == {"input_batch": "value"}
    runtime.close()
    runtime.close()
    hook.remove.assert_called_once_with()
    context.close.assert_called_once_with()


def test_transform_rejects_specialized_routed_experts():
    @dataclass(kw_only=True, slots=True)
    class SpecializedConfig(RoutedExperts.Config):
        extra_policy: bool = True

    stock = _stock_config()
    specialized = SpecializedConfig(
        w13=stock.w13,
        w2=stock.w2,
        activation_fn=stock.activation_fn,
        token_dispatcher=stock.token_dispatcher,
    )

    with pytest.raises(TypeError, match="specialized RoutedExperts.Config"):
        DistMoeTransform().transform(specialized)


def test_transform_rejects_postprocess_without_native_translation():
    stock = _stock_config()
    stock.expert_output_postprocess = RMSNorm.Config(normalized_shape=32)

    with pytest.raises(TypeError, match="cannot run inside DistMoE"):
        DistMoeTransform().transform(stock)


def test_bf16_transform_preserves_parameters_without_building_dispatcher():
    stock = _stock_config().build()
    with torch.no_grad():
        for value, parameter in enumerate(stock.parameters(), start=1):
            parameter.fill_(value)

    config = DistMoeTransform().transform(_stock_config())
    assert isinstance(config, DistMoeRoutedExperts.Config)
    module = config.build()
    module.load_state_dict(stock.state_dict())
    module.parallelize(Mock())

    assert list(dict(module.named_parameters())) == ["w13.weight", "w2.weight"]
    assert not hasattr(module, "token_dispatcher")
    assert not hasattr(module, "activation_fn")
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, stock.state_dict()[key], rtol=0, atol=0)


def test_mxfp8_transform_uses_separate_module_and_prepared_weight_lifecycle():
    config = DistMoeTransform().transform(_stock_config())
    config = MXFP8DistMoeTransform().transform(config)
    assert isinstance(config, MXFP8DistMoeRoutedExperts.Config)

    module = config.build()
    assert isinstance(module, MXFP8DistMoeRoutedExperts)
    assert isinstance(module.w13.weight, _ShardedFSDPTensor)
    assert isinstance(module.w2.weight, _ShardedFSDPTensor)
    assert list(module.state_dict()) == ["w13.weight", "w2.weight"]


def test_native_postprocess_is_owned_and_passed_to_dist_moe():
    stock = _stock_config()
    stock.expert_output_postprocess = _NativePostprocess.Config(dim=32)
    config = DistMoeTransform().transform(stock)
    assert isinstance(config, DistMoeRoutedExperts.Config)
    module = config.build()
    module._runtime = _runtime()
    module._runtime.context = cast(Any, object())

    with patch(
        "torchtitan.components.dist_moe.backend.run_dist_moe",
        return_value=torch.empty(2, 32),
    ) as run:
        module(
            torch.empty(2, 32),
            torch.empty(2, 2),
            torch.empty(2, 2, dtype=torch.int64),
            torch.empty(4, dtype=torch.int64),
        )

    descriptor = run.call_args.kwargs["options"].experts_output_postprocess
    assert isinstance(descriptor, DistMoeInputScaledRMSNorm)
    assert descriptor.weight is module.expert_output_postprocess.weight
    assert "expert_output_postprocess.weight" in module.state_dict()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"max_routing_imbalance_factor": 0}, "must be positive"),
        ({"num_activation_slots": 0}, "must be positive"),
        ({"prefetch_vmm": True}, "requires VMM to be enabled"),
        ({"activation_slot_policy": "invalid"}, "activation slot policy"),
        ({"wgrad_dtype": "float16"}, "WGRAD dtype"),
    ],
)
def test_dist_moe_config_rejects_invalid_values(kwargs, message):
    stock = _stock_config()
    with pytest.raises(ValueError, match=message):
        DistMoeRoutedExperts.Config(
            w13=stock.w13,
            w2=stock.w2,
            activation_fn=stock.activation_fn,
            token_dispatcher=stock.token_dispatcher,
            **kwargs,
        )
