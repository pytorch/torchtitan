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

import torchtitan.config.transform.quantization as quantization_transform
from dist_moe import RMSNormPostprocess, VmmConfig
from torch.distributed.pipelining import PipelineStageInfo
from torchtitan.components.dist_moe import (
    DistMoeRoutedExperts,
    DistMoeRuntime,
    MXFP8DistMoeRoutedExperts,
    prepare_dist_moe_runtime,
)
from torchtitan.components.dist_moe.backend import _build_dist_moe_runtime_config
from torchtitan.config.transform import DistMoeTransform, MXFP8DistMoeTransform
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    config_registry as graph_configs,
)
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v3 import config_registry as eager_configs
from torchtitan.protocols.module import Module
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor


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
    return DistMoeRuntime(
        config=cast(Any, object()),
        group=cast(Any, object()),
        device=torch.device("cuda"),
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

    def to_dist_moe_postprocess(self) -> RMSNormPostprocess:
        """Translate this owned module to the annex-native descriptor."""
        return RMSNormPostprocess(
            eps=self.eps,
            norm_output_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            weight=self.weight,
            gain_center=self.gain_center,
        )


def test_runtime_initializes_and_closes_context_once():
    runtime = _runtime()
    context = Mock()
    with patch(
        "torchtitan.components.dist_moe.backend.create_context",
        return_value=context,
    ) as create:
        runtime.initialize()
        runtime.initialize()

    create.assert_called_once_with(
        group=runtime.group,
        config=runtime.config,
        device=runtime.device,
    )
    runtime.close()
    runtime.close()
    context.close.assert_called_once_with()


def test_runtime_selects_pipeline_slot_from_forward_context():
    runtime = _runtime()
    runtime.slots[(3, 7)] = (2, 5)
    runtime.context = Mock()
    context = runtime.context
    hook = Mock()
    runtime._pipeline_context_handles.append(hook)

    with runtime.pipeline_slot_context(
        PipelineStageInfo(stage_index=3, microbatch_index=7)
    ):
        context.select_activation_slot.assert_called_once_with(2, 5)

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
    stock.output_postprocess = RMSNorm.Config(normalized_shape=32)

    with pytest.raises(TypeError, match="cannot run inside DistMoE"):
        DistMoeTransform().transform(stock)


def test_dist_moe_does_not_execute_the_source_dispatcher():
    config = DistMoeTransform().transform(_stock_config())

    assert isinstance(config, DistMoeRoutedExperts.Config)
    assert not config.uses_configured_token_dispatcher


def test_bf16_transform_preserves_parameters_without_building_dispatcher():
    stock = _stock_config().build()
    with torch.no_grad():
        for value, parameter in enumerate(stock.parameters(), start=1):
            parameter.fill_(value)

    config = DistMoeTransform().transform(_stock_config())
    assert isinstance(config, DistMoeRoutedExperts.Config)
    module = config.build()
    module.load_state_dict(stock.state_dict())
    module._parallelize(Mock())

    assert list(dict(module.named_parameters())) == ["w13.weight", "w2.weight"]
    assert not hasattr(module, "token_dispatcher")
    assert not hasattr(module, "activation_fn")
    module._init_self_buffers()
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, stock.state_dict()[key], rtol=0, atol=0)


def test_bf16_transform_maps_the_annex_memory_contract():
    vmm = VmmConfig(total_scratch_capacity_factor=4.0, prefetch=False)
    config = DistMoeTransform(
        device_scratch_capacity_factor=2.0,
        saved_activation_buffer_bytes=4096,
        vmm=vmm,
        bf16_grouped_gemm_preset="1cta1mma_bm64_bn128",
    ).transform(_stock_config(dim=64))
    assert isinstance(config, DistMoeRoutedExperts.Config)

    runtime_config = _build_dist_moe_runtime_config(
        config.build(),
        max_num_tokens=128,
        num_moe_layers=3,
        num_activation_slots=2,
    )

    assert runtime_config.max_local_input_tokens == 128
    assert runtime_config.max_moe_layers_per_activation_slot == 3
    assert runtime_config.device_scratch_capacity_factor == 2.0
    assert runtime_config.saved_activation_buffer_bytes == 4096
    assert runtime_config.num_activation_slots == 2
    assert runtime_config.vmm is vmm
    assert runtime_config.bf16_grouped_gemm_preset == "1cta1mma_bm64_bn128"


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
    stock.output_postprocess = _NativePostprocess.Config(dim=32)
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
    assert isinstance(descriptor, RMSNormPostprocess)
    assert descriptor.weight is module.output_postprocess.weight
    assert "output_postprocess.weight" in module.state_dict()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"device_scratch_capacity_factor": 0}, "must be positive"),
        ({"saved_activation_buffer_bytes": -1}, "cannot be negative"),
        ({"num_activation_slots": 0}, "must be positive"),
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


def test_prepare_runtime_allows_initializers_and_rejects_shared_policy_mismatch(
    monkeypatch,
):
    config = eager_configs.deepseek_v3_debugmodel_dist_moe_bf16(seq_len=128)
    expert_configs = [
        entry[1] for entry in config.model.traverse(DistMoeRoutedExperts.Config)
    ]
    modules = tuple(expert_config.build() for expert_config in expert_configs[:2])
    assert modules[0]._dist_moe_config.w13 != modules[1]._dist_moe_config.w13
    assert modules[0]._dist_moe_config.w2 != modules[1]._dist_moe_config.w2
    group = Mock()
    ep_mesh = Mock()
    ep_mesh.get_group.return_value = group
    parallel_dims = Mock(cp=1, tp=1, pp_enabled=False)
    parallel_dims.get_optional_mesh.return_value = ep_mesh
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 0))

    runtime = prepare_dist_moe_runtime(
        config=config,
        model_parts=list(modules),
        parallel_dims=parallel_dims,
        device=torch.device("cuda"),
        pp_schedule=None,
    )

    assert runtime is not None
    assert all(module._runtime is runtime for module in modules)
    runtime.close()
    modules[1]._dist_moe_config.activation_slot_policy = "microbatch"
    with pytest.raises(ValueError, match="activation-slot and VMM policy"):
        prepare_dist_moe_runtime(
            config=config,
            model_parts=list(modules),
            parallel_dims=parallel_dims,
            device=torch.device("cuda"),
            pp_schedule=None,
        )


@pytest.mark.parametrize(
    "factory,num_experts_modules,device_scratch_capacity_factor",
    [
        (eager_configs.deepseek_v3_debugmodel_dist_moe_bf16, 5, 1.0),
        (eager_configs.deepseek_v3_16b_dist_moe_bf16, 26, 4.0),
        (eager_configs.deepseek_v3_671b_dist_moe_bf16, 58, 4.0),
        (graph_configs.graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16, 5, 1.0),
        (graph_configs.graph_trainer_deepseek_v3_16b_dist_moe_bf16, 26, 4.0),
        (graph_configs.graph_trainer_deepseek_v3_671b_dist_moe_bf16, 58, 4.0),
    ],
)
def test_dist_moe_bf16_recipes_use_varlen_and_replace_all_experts(
    factory, num_experts_modules, device_scratch_capacity_factor
):
    config = factory()
    model_config = config.model
    experts = list(model_config.traverse(DistMoeRoutedExperts.Config))

    assert len(experts) == num_experts_modules
    assert all(type(entry[1]) is DistMoeRoutedExperts.Config for entry in experts)
    assert all(entry[1].vmm is None for entry in experts)
    assert all(
        entry[1].device_scratch_capacity_factor == device_scratch_capacity_factor
        for entry in experts
    )
    assert all(
        isinstance(layer.attention.inner_attention, VarlenInnerAttention.Config)
        for layer in model_config.layers
    )
    assert config.dataloader.max_num_documents == 512


def test_dist_moe_recipe_supports_cuda_graphs_with_expert_parallelism():
    config = eager_configs.deepseek_v3_debugmodel_dist_moe_bf16(seq_len=128)
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"

    config.__post_init__()


@pytest.mark.parametrize(
    "factory,num_experts_modules,device_scratch_capacity_factor",
    [
        (eager_configs.deepseek_v3_debugmodel_dist_moe_mxfp8, 5, 1.0),
        (eager_configs.deepseek_v3_16b_dist_moe_mxfp8, 26, 4.0),
        (eager_configs.deepseek_v3_671b_dist_moe_mxfp8, 58, 4.0),
        (graph_configs.graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8, 5, 1.0),
        (graph_configs.graph_trainer_deepseek_v3_16b_dist_moe_mxfp8, 26, 4.0),
        (graph_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8, 58, 4.0),
    ],
)
def test_dist_moe_mxfp8_recipes_quantize_dense_linears_and_lm_head(
    factory, num_experts_modules, device_scratch_capacity_factor, monkeypatch
):
    pytest.importorskip("torchao")
    from torchtitan.quantization import MXFP8Linear

    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8Linear is unavailable")
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    config = factory()
    model_config = config.model
    experts = list(model_config.traverse(DistMoeRoutedExperts.Config))
    linears = {
        fqn
        for fqn, _linear, _parent, _attr in model_config.traverse(MXFP8Linear.Config)
    }

    assert len(experts) == num_experts_modules
    assert all(
        isinstance(entry[1], MXFP8DistMoeRoutedExperts.Config) for entry in experts
    )
    assert all(entry[1].vmm is None for entry in experts)
    assert all(
        entry[1].device_scratch_capacity_factor == device_scratch_capacity_factor
        for entry in experts
    )
    assert "lm_head" in linears
