# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, replace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch

import torchtitan.config.transform.quantization as quantization_transform
from dist_moe import DistMoeBlockScaledFormat, RMSNormPostprocess
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.dist_moe import (
    DistMoeBackendConfig,
    DistMoeRoutedExperts,
    DistMoeRuntime,
    MXFP8DistMoeRoutedExperts,
    prepare_dist_moe_runtime,
)
from torchtitan.config import apply_overrides
from torchtitan.config.transform import DistMoeTransform, MXFP8DistMoeTransform
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    config_registry as graph_configs,
)
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v3 import config_registry as eager_configs
from torchtitan.overrides.fused_swiglu import FusedSwiGLU
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


def _runtime() -> DistMoeRuntime:
    return DistMoeRuntime(
        config=cast(Any, object()),
        group=cast(Any, object()),
        device=torch.device("cuda"),
    )


class _NativePostprocess(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        eps: float = 1e-8

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def to_dist_moe_postprocess(self) -> RMSNormPostprocess:
        """Translate this owned module to the annex-native descriptor."""
        return RMSNormPostprocess(
            eps=self.eps,
            norm_output_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )


def test_runtime_leaves_context_unset_after_initialization_failure():
    runtime = _runtime()
    with (
        patch(
            "torchtitan.components.dist_moe.backend.create_context",
            side_effect=RuntimeError("context creation failed"),
        ),
        pytest.raises(RuntimeError, match="context creation failed"),
    ):
        runtime.initialize()

    assert runtime.context is None
    runtime.close()


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


def test_runtime_reset_invalidates_cached_pipeline_slot():
    runtime = _runtime()
    runtime.context = Mock()
    runtime._selected = (3, 7)

    runtime.reset()

    runtime.context.reset.assert_called_once_with()
    assert runtime._selected is None


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


def test_bf16_transform_preserves_parameters_without_building_dispatcher():
    stock = _stock_config().build()
    with torch.no_grad():
        for value, parameter in enumerate(stock.parameters(), start=1):
            parameter.fill_(value)

    config = DistMoeTransform().transform(_stock_config())
    assert isinstance(config, DistMoeRoutedExperts.Config)
    module = config.build()
    module.load_state_dict(stock.state_dict())
    parallelism_context = Mock()
    parallelism_context.get_optional_mesh.return_value = None
    module._parallelize(parallelism_context)

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
    stock.output_postprocess = _NativePostprocess.Config()
    config = DistMoeTransform().transform(stock)
    assert isinstance(config, DistMoeRoutedExperts.Config)
    module = config.build()
    module._dist_moe_runtime = _runtime()
    module._dist_moe_runtime.context = cast(Any, object())

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
    assert descriptor.weight is None
    assert "output_postprocess.weight" not in module.state_dict()


def test_weighted_native_postprocess_is_rejected_for_training():
    stock = _stock_config()
    stock.output_postprocess = _NativePostprocess.Config()
    module = DistMoeTransform().transform(stock).build()
    module.output_postprocess.to_dist_moe_postprocess = Mock(
        return_value=RMSNormPostprocess(
            eps=1e-8,
            norm_output_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            weight=torch.ones(32),
        )
    )

    with pytest.raises(ValueError, match="inference-only"):
        module._build_dist_moe_postprocess()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"device_scratch_capacity_factor": 0}, "must be positive"),
        ({"saved_activation_buffer_bytes": -1}, "non-negative integer"),
        ({"num_activation_slots": 0}, "must be positive"),
        (
            {"vmm_total_scratch_capacity_factor": None, "vmm_prefetch": True},
            "requires VMM to be enabled",
        ),
        ({"activation_slot_policy": "invalid"}, "activation slot policy"),
        ({"wgrad_dtype": "float16"}, "WGRAD dtype"),
    ],
)
def test_dist_moe_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DistMoeBackendConfig(**kwargs)


def test_prepare_runtime_allows_initializers_and_rejects_shared_policy_mismatch(
    monkeypatch,
):
    config = eager_configs.deepseek_v3_debugmodel_dist_moe_bf16(seq_len=128)
    expert_configs = [
        entry[1] for entry in config.model.traverse(DistMoeRoutedExperts.Config)
    ]
    assert expert_configs[0].w13 != expert_configs[1].w13
    assert expert_configs[0].w2 != expert_configs[1].w2
    modules = tuple(expert_config.build() for expert_config in expert_configs[:2])
    group = Mock()
    ep_mesh = Mock()
    ep_mesh.get_group.return_value = group
    parallelism_context = Mock(cp=1, tp=1, pp_enabled=False)
    parallelism_context.get_optional_mesh.return_value = ep_mesh
    memory_plan = Mock(uses_host_scratch=False)
    memory_plan.maximum_useful_saved_activation_buffer_bytes = 1234
    memory_plan.explain.return_value = "test memory plan"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 0))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(
        "torchtitan.components.dist_moe.backend.plan_dist_moe_memory",
        lambda *_args, **_kwargs: memory_plan,
    )

    runtime = prepare_dist_moe_runtime(
        config=config,
        model_parts=list(modules),
        parallelism_context=parallelism_context,
        device=torch.device("cuda"),
        pp_schedule=None,
    )

    assert runtime is not None
    assert runtime.config.saved_activation_buffer_bytes == 1234
    assert all(module._dist_moe_runtime is runtime for module in modules)
    runtime.close()
    modules[1]._dist_moe_policy = replace(
        modules[1]._dist_moe_policy,
        activation_slot_policy="microbatch",
    )
    with pytest.raises(ValueError, match="activation-slot and VMM-prefetch policy"):
        prepare_dist_moe_runtime(
            config=config,
            model_parts=list(modules),
            parallelism_context=parallelism_context,
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
    assert all(
        entry[1].backend.vmm_total_scratch_capacity_factor is None
        and not entry[1].backend.vmm_prefetch
        for entry in experts
    )
    assert all(
        entry[1].backend.device_scratch_capacity_factor
        == device_scratch_capacity_factor
        for entry in experts
    )
    assert all(
        isinstance(layer.attention.inner_attention, VarlenInnerAttention.Config)
        for layer in model_config.layers
    )
    assert config.dataloader.max_num_documents == 512


@pytest.mark.parametrize(
    "factory",
    [
        graph_configs.graph_trainer_deepseek_v3_16b_dist_moe_bf16_local_4gpu,
        graph_configs.graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_local_4gpu,
    ],
)
def test_dist_moe_graph_local_recipe_uses_matched_configuration(factory, monkeypatch):
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    config = factory()
    assert config.compile.fsdp_gradient_sync_mode == "deferred_as_schedule_stage"
    assert not config.compile.enable_fsdp_dense_region_overlap
    assert config.compile.inductor_compilation == "regional"
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 8192
    assert config.training.num_tokens_per_train_step == 524288
    assert config.training.num_tokens_per_train_step == (
        config.training.num_tokens_per_microbatch_per_dp_rank
        * config.parallelism.data_parallel_replicate_degree
        * config.parallelism.data_parallel_shard_degree
        * 16
    )
    assert not config.training.disable_cuda_graphs
    assert config.activation_checkpoint is None
    assert config.parallelism.data_parallel_replicate_degree == 1
    assert config.parallelism.data_parallel_shard_degree == 4
    assert config.parallelism.tensor_parallel_degree == 1
    assert config.parallelism.context_parallel_degree == 1
    assert config.parallelism.pipeline_parallel_degree == 1
    assert config.parallelism.expert_parallel_degree == 4
    assert config.parallelism.fsdp_reshard_after_forward == "never"
    assert config.parallelism.fsdp_symm_mem_scope is None
    assert config.debug.moe_force_load_balance
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    assert isinstance(config.dataloader.dataset, ConcatThenSplitPackingConfig)
    assert not config.dataloader.dataset.mask_document_boundaries
    assert config.dataloader.max_num_documents == 2
    inner_attention_configs = list(config.model.traverse(VarlenInnerAttention.Config))
    assert inner_attention_configs
    assert all(
        inner_attention.fixed_length_rows
        for _, inner_attention, _, _ in inner_attention_configs
    )

    routed_experts = list(config.model.traverse(DistMoeRoutedExperts.Config))
    assert routed_experts
    assert all(
        experts.backend.device_scratch_capacity_factor == 1.0
        for _, experts, _, _ in routed_experts
    )

    apply_overrides(config.override, config)

    routed_experts = list(config.model.traverse(DistMoeRoutedExperts.Config))
    assert all(
        type(experts.activation_fn) is SwiGLU.Config
        for _, experts, _, _ in routed_experts
    )
    fused_activations = list(config.model.traverse(FusedSwiGLU.Config))
    assert fused_activations
    assert all(
        fqn.endswith(".feed_forward.activation_fn")
        or fqn.endswith(".shared_experts.activation_fn")
        for fqn, _, _, _ in fused_activations
    )


@pytest.mark.parametrize(
    "factory,uses_block_scaled",
    [
        (eager_configs.deepseek_v3_16b_dist_moe_bf16_local_4gpu, False),
        (eager_configs.deepseek_v3_16b_dist_moe_mxfp8_local_4gpu, True),
    ],
)
def test_dist_moe_eager_local_recipe_uses_matched_ga16_configuration(
    factory, uses_block_scaled, monkeypatch
):
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    config = factory()

    assert config.training.num_tokens_per_microbatch_per_dp_rank == 8192
    assert config.training.num_tokens_per_train_step == 524288
    assert config.activation_checkpoint is None
    assert config.parallelism.data_parallel_replicate_degree == 1
    assert config.parallelism.data_parallel_shard_degree == 4
    assert config.parallelism.tensor_parallel_degree == 1
    assert config.parallelism.context_parallel_degree == 1
    assert config.parallelism.pipeline_parallel_degree == 1
    assert config.parallelism.expert_parallel_degree == 4
    assert config.parallelism.fsdp_reshard_after_forward == "never"
    assert config.parallelism.fsdp_symm_mem_scope is None
    assert config.training.num_tokens_per_train_step == (
        config.training.num_tokens_per_microbatch_per_dp_rank
        * config.parallelism.data_parallel_replicate_degree
        * config.parallelism.data_parallel_shard_degree
        * 16
    )
    assert config.debug.moe_force_load_balance
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    assert isinstance(config.dataloader.dataset, ConcatThenSplitPackingConfig)
    assert not config.dataloader.dataset.mask_document_boundaries
    assert config.dataloader.max_num_documents == 2
    inner_attention_configs = list(config.model.traverse(VarlenInnerAttention.Config))
    assert inner_attention_configs
    assert all(
        inner_attention.fixed_length_rows
        for _, inner_attention, _, _ in inner_attention_configs
    )
    experts = list(config.model.traverse(DistMoeRoutedExperts.Config))
    assert experts
    assert all(
        expert.backend.device_scratch_capacity_factor == 1.0
        for _, expert, _, _ in experts
    )
    assert all(
        (expert.backend.block_scaled is not None) == uses_block_scaled
        for _, expert, _, _ in experts
    )


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
    assert all(type(entry[1]) is DistMoeRoutedExperts.Config for entry in experts)
    assert all(
        entry[1].backend.vmm_total_scratch_capacity_factor is None
        and not entry[1].backend.vmm_prefetch
        for entry in experts
    )
    assert all(
        entry[1].backend.device_scratch_capacity_factor
        == device_scratch_capacity_factor
        for entry in experts
    )
    assert all(
        entry[1].backend.block_scaled is not None
        and entry[1].backend.block_scaled.format == DistMoeBlockScaledFormat.MXFP8_E4M3
        for entry in experts
    )
    assert "lm_head" in linears
