# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import dist_moe
import pytest
import torch
from torch.distributed.pipelining import PipelineStageInfo

from torchtitan.config.transform import (
    apply_transforms,
    DistMoeTransform,
    MXFP8DistMoeTransform,
)
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts, DistMoeRuntime
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
from torchtitan.protocols.module import Module
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor
from torchtitan.quantization.mxfp8 import MXFP8DistMoeRoutedExperts
from torchtitan.training_engine import TrainingEngine


def _parameter_initializers() -> dict[str, Any]:
    return {
        "w1_EFD": torch.nn.init.zeros_,
        "w2_EDF": torch.nn.init.zeros_,
        "w3_EFD": torch.nn.init.zeros_,
    }


def _stock_config(*, dim: int = 32) -> RoutedExperts.Config:
    return make_routed_experts_config(
        dim=dim,
        hidden_dim=64,
        num_experts=4,
        top_k=2,
        param_init=_parameter_initializers(),
        comm_backend="standard",
    )


def _runtime() -> DistMoeRuntime:
    runtime = object.__new__(DistMoeRuntime)
    runtime.config = DistMoeRuntime.Config()
    runtime.context = None
    runtime.context_device = torch.device("cuda")
    runtime.ep_pg = cast(Any, object())
    runtime._modules = ()
    runtime._context_config = cast(Any, object())
    runtime.pp_activation_slot_id_by_stage_and_microbatch = {}
    runtime.max_moe_layers_per_activation_slot = 1
    runtime._activation_slot_ids_S = torch.arange(1)
    return runtime


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

    def forward(self, value_D: torch.Tensor) -> torch.Tensor:
        return value_D

    def to_dist_moe_postprocess(self) -> dist_moe.RMSNormPostprocess:
        """Bind this module's current parameter to the annex descriptor."""
        return dist_moe.RMSNormPostprocess(
            eps=self.eps,
            norm_output_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            weight=self.weight,
            gain_center=self.gain_center,
        )


def test_runtime_initializes_and_closes_context_once() -> None:
    """Runtime context construction and teardown are idempotent."""
    runtime = _runtime()
    context = Mock()
    with patch(
        "torchtitan.models.common.dist_moe.dist_moe.create_context",
        return_value=context,
    ) as create:
        runtime.initialize()
        runtime.initialize()

    create.assert_called_once_with(
        group=runtime.ep_pg,
        config=runtime._context_config,
        device=runtime.context_device,
    )
    runtime.close()
    runtime.close()
    context.close.assert_called_once_with()


def test_runtime_selects_pp_activation_slot_from_forward_context() -> None:
    """Pipeline metadata selects the precomputed annex activation slot."""
    runtime = _runtime()
    runtime.pp_activation_slot_id_by_stage_and_microbatch[(3, 7)] = 2
    runtime.max_moe_layers_per_activation_slot = 5
    runtime._activation_slot_ids_S = torch.arange(3)
    runtime.context = Mock()

    with runtime.forward_context(PipelineStageInfo(stage_index=3, microbatch_index=7)):
        runtime.context.select_activation_slot.assert_called_once_with(2, 5)
    graph_inputs = runtime.graph_forward_inputs()
    assert graph_inputs[(3, 7)]["activation_slot_id_1"].item() == 2


def test_engine_owns_runtime_forward_context_and_cleanup() -> None:
    """The generic engine lifecycle registers and removes eager PP contexts."""
    runtime = Mock()
    runtime.forward_context.return_value = nullcontext()
    runtime_config = Mock()
    runtime_config.build.return_value = runtime
    stage_handle = Mock()
    stage = Mock()
    stage.register_forward_context.return_value = stage_handle

    engine = object.__new__(TrainingEngine)
    engine.config = SimpleNamespace(runtimes=[runtime_config])
    engine.model_parts = [Mock()]
    engine.parallelism_context = SimpleNamespace(pp_enabled=True)
    runtime_schedule = SimpleNamespace(_stages=[stage])
    liveness_schedule = SimpleNamespace()
    engine.pp_schedule = SimpleNamespace(
        pipeline_schedule=runtime_schedule,
        pipeline_liveness_schedule=liveness_schedule,
    )
    engine.device = torch.device("cuda")
    engine.runtimes = []
    engine._runtime_stack = ExitStack()

    engine._prepare_training_runtimes()
    info = PipelineStageInfo(stage_index=1, microbatch_index=2)
    forward_context = stage.register_forward_context.call_args.args[0]
    with forward_context(info):
        pass
    engine._close_training_runtimes()

    runtime_config.build.assert_called_once()
    assert runtime_config.build.call_args.kwargs["pp_schedule"] is liveness_schedule
    assert (
        runtime_config.build.call_args.kwargs["parallelism_context"]
        is engine.parallelism_context
    )
    runtime.forward_context.assert_called_once_with(info)
    stage_handle.remove.assert_called_once_with()
    runtime.close.assert_called_once_with()


def test_transform_rejects_specialized_routed_experts() -> None:
    """Dist-MoE refuses an already-specialized routed-expert implementation."""

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

    with pytest.raises(TypeError, match="unmodified RoutedExperts.Config"):
        DistMoeTransform().transform(specialized)


def test_transform_rejects_postprocess_without_native_translation() -> None:
    """Unsupported postprocessors fail before model construction."""
    stock = _stock_config()
    stock.output_postprocess = RMSNorm.Config(normalized_shape=32)

    with pytest.raises(TypeError, match="cannot execute inside Dist-MoE"):
        DistMoeTransform().transform(stock)


def test_bf16_transform_preserves_parameter_layout() -> None:
    """BF16 replacement preserves standard W13/W2 checkpoint keys and values."""
    stock = _stock_config().build()
    with torch.no_grad():
        for value, parameter in enumerate(stock.parameters(), start=1):
            parameter.fill_(value)

    transformed = DistMoeTransform().transform(_stock_config())
    assert isinstance(transformed, DistMoeRoutedExperts.Config)
    module = transformed.build()
    module.load_state_dict(stock.state_dict())

    assert list(dict(module.named_parameters())) == ["w13.weight", "w2.weight"]
    assert not hasattr(module, "token_dispatcher")
    assert not hasattr(module, "activation_fn")
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, stock.state_dict()[key], rtol=0, atol=0)


def test_runtime_passes_per_slot_capacity_to_annex() -> None:
    """Runtime preserves the annex's per-slot activation-capacity contract."""
    vmm = dist_moe.VmmConfig(total_scratch_capacity_factor=4.0, prefetch=False)
    transformed = DistMoeTransform(
        runtime=DistMoeRuntime.Config(
            device_scratch_capacity_factor=2.0,
            activation_slot_bytes=2048,
            vmm=vmm,
        ),
        bf16_grouped_gemm_preset="1cta1mma_bm64_bn128",
    ).transform(_stock_config(dim=64))
    module = cast(DistMoeRoutedExperts, transformed.build())
    runtime = _runtime()
    runtime.config = DistMoeRuntime.Config(
        device_scratch_capacity_factor=2.0,
        activation_slot_bytes=2048,
        vmm=vmm,
    )

    context_config = runtime._resolve_context_config(
        module,
        max_local_input_tokens=128,
        max_live_activation_slots=2,
        max_moe_layers_per_activation_slot=3,
    )

    assert context_config.max_local_input_tokens == 128
    assert context_config.max_moe_layers_per_activation_slot == 3
    assert context_config.device_scratch_capacity_factor == 2.0
    assert context_config.activation_slot_bytes == 2048
    assert context_config.activation_slot_capacity_factor is None
    assert context_config.num_activation_slots == 2
    assert context_config.vmm is vmm
    assert context_config.bf16_grouped_gemm_preset == "1cta1mma_bm64_bn128"

    runtime.config = DistMoeRuntime.Config(
        device_scratch_capacity_factor=2.0,
        activation_slot_capacity_factor=1.5,
        vmm=vmm,
    )
    factor_config = runtime._resolve_context_config(
        module,
        max_local_input_tokens=128,
        max_live_activation_slots=2,
        max_moe_layers_per_activation_slot=3,
    )
    assert factor_config.activation_slot_bytes is None
    assert factor_config.activation_slot_capacity_factor == 1.5
    assert factor_config.num_activation_slots == 2


def test_mxfp8_transform_is_independent_and_uses_prepared_weights() -> None:
    """MXFP8 transforms stock experts directly and installs prepared weights."""
    transformed = MXFP8DistMoeTransform().transform(_stock_config())
    assert isinstance(transformed, MXFP8DistMoeRoutedExperts.Config)

    module = transformed.build()
    assert isinstance(module, MXFP8DistMoeRoutedExperts)
    assert isinstance(module.w13.weight, _ShardedFSDPTensor)
    assert isinstance(module.w2.weight, _ShardedFSDPTensor)
    assert list(module.state_dict()) == ["w13.weight", "w2.weight"]


def test_dist_moe_transforms_conflict() -> None:
    """A routed-expert module cannot select BF16 and MXFP8 Dist-MoE together."""
    config = deepseek_v3_debugmodel()
    with pytest.raises(ValueError, match="cannot be combined"):
        apply_transforms(config, [DistMoeTransform(), MXFP8DistMoeTransform()])


def test_transform_registers_one_runtime() -> None:
    """Applying Dist-MoE records one generic rank-wide runtime configuration."""
    config = deepseek_v3_debugmodel()
    transformed = apply_transforms(config, [DistMoeTransform()])
    assert len(transformed.runtimes) == 1
    assert isinstance(transformed.runtimes[0], DistMoeRuntime.Config)


def test_forward_passes_native_postprocess_and_wgrad_policy() -> None:
    """Forward passes module-owned postprocessing and annex-owned WGRAD policy."""
    stock = _stock_config()
    stock.output_postprocess = _NativePostprocess.Config(dim=32)
    transformed = DistMoeTransform(inplace_wgrad_accum=True).transform(stock)
    module = cast(DistMoeRoutedExperts, transformed.build())
    module._runtime = _runtime()
    module._runtime.context = cast(Any, object())

    with (
        patch(
            "torchtitan.models.common.dist_moe.dist_moe.routed_experts",
            return_value=torch.empty(2, 32),
        ) as execute,
        patch(
            "torchtitan.models.common.dist_moe.remat.region",
            side_effect=lambda fn, *_args, **_kwargs: fn,
        ) as remat_region,
        patch(
            "torchtitan.models.common.dist_moe.remat.recompute_needs_tensor"
        ) as recompute_needs_tensor,
    ):
        out_TD = module(
            torch.empty(2, 32),
            torch.empty(2, 2),
            torch.empty(2, 2, dtype=torch.int64),
            torch.empty(4, dtype=torch.int64),
        )

    options = execute.call_args.kwargs["options"]
    descriptor = options.experts_output_postprocess
    assert isinstance(descriptor, dist_moe.RMSNormPostprocess)
    assert descriptor.weight is module.output_postprocess.weight
    assert options.inplace_wgrad_accum
    assert options.wgrad_parameter_owners is None
    assert "output_postprocess.weight" in module.state_dict()
    assert remat_region.call_args.kwargs == {"recompute": False}
    recompute_needs_tensor.assert_called_once_with(out_TD)


@pytest.mark.parametrize(
    "kwargs,error_type,message",
    [
        ({"device_scratch_capacity_factor": 0}, ValueError, "must be positive"),
        ({"activation_slot_bytes": -1}, ValueError, "cannot be negative"),
        ({"activation_slot_bytes": True}, TypeError, "must be an integer"),
        ({"activation_slot_capacity_factor": -1}, ValueError, "nonnegative"),
        ({"activation_slot_capacity_factor": float("nan")}, ValueError, "finite"),
        (
            {
                "activation_slot_bytes": 1,
                "activation_slot_capacity_factor": 1.0,
            },
            ValueError,
            "mutually exclusive",
        ),
        (
            {"pp_activation_slot_policy": "invalid"},
            ValueError,
            "activation-slot policy",
        ),
        ({"wgrad_dtype": "float16"}, ValueError, "WGRAD dtype"),
    ],
)
def test_runtime_config_rejects_invalid_values(kwargs, error_type, message) -> None:
    """Rank-wide runtime configuration rejects invalid memory policies."""
    with pytest.raises(error_type, match=message):
        DistMoeRuntime.Config(**kwargs)


def test_runtime_config_requires_bfloat16_unsharded_parameters() -> None:
    """Dist-MoE rejects FSDP mixed-precision parameter dtypes it cannot consume."""
    with pytest.raises(ValueError, match="mixed_precision_param='bfloat16'"):
        DistMoeRuntime.Config().validate(
            SimpleNamespace(training=SimpleNamespace(mixed_precision_param="float32"))
        )
