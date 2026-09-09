# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.components.dist_moe import (
    DistMoeBackendConfig,
    DistMoeConverter,
    DistMoeRoutedExperts,
)
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.quantization._fsdp_tensor import _ShardedFSDPTensor
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    config_registry as graph_configs,
)
from torchtitan.models.common.attention import VarlenAttention
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.deepseek_v3 import config_registry as eager_configs


def _routed_experts_config() -> DistMoeRoutedExperts.Config:
    stock = make_routed_experts_config(
        dim=32,
        hidden_dim=64,
        num_experts=4,
        top_k=2,
        param_init={},
        comm_backend="standard",
    )
    return DistMoeConverter(DistMoeConverter.Config()).convert(stock)


def _optimizer_config() -> OptimizersContainer.Config:
    return OptimizersContainer.Config(
        implementation="for-loop",
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={"lr": 1e-3},
            )
        ],
    )


def test_dist_moe_converter_owns_fused_params_with_stock_checkpoint_keys():
    stock_config = make_routed_experts_config(
        dim=32,
        hidden_dim=64,
        num_experts=4,
        top_k=2,
        param_init={},
        comm_backend="standard",
    )
    stock = stock_config.build()
    with torch.no_grad():
        for value, parameter in enumerate(stock.parameters(), start=1):
            parameter.fill_(value)

    converted = _routed_experts_config().build()
    converted.load_state_dict(stock.state_dict())

    assert list(dict(converted.named_parameters())) == ["w13_EGFD", "w2_EDF"]
    converted_state = converted.state_dict()
    assert converted_state.keys() == stock.state_dict().keys()
    for key, value in converted_state.items():
        assert type(value) is torch.Tensor
        torch.testing.assert_close(value, stock.state_dict()[key], rtol=0, atol=0)


def test_dist_moe_mxfp8_keeps_plain_stock_checkpoint_values():
    config = _routed_experts_config()
    config.backend = DistMoeBackendConfig(dtype="mxfp8")
    module = config.build()

    assert all(isinstance(param, _ShardedFSDPTensor) for param in module.parameters())
    assert all(type(value) is torch.Tensor for value in module.state_dict().values())


def test_dist_moe_optimizer_state_round_trips_through_stock_keys():
    config = _routed_experts_config()
    source = config.build()
    source_optimizer = _optimizer_config().build(model_parts=[source])
    for parameter in source.parameters():
        parameter.grad = torch.ones_like(parameter)
    source_optimizer.step()

    state = source_optimizer.state_dict()
    assert not any("w13_EGFD" in key for key in state)
    assert any("inner_experts.w1_EFD" in key for key in state)
    assert any("inner_experts.w3_EFD" in key for key in state)

    target = config.build()
    target_optimizer = _optimizer_config().build(model_parts=[target])
    target_optimizer.load_state_dict(state)
    restored = target_optimizer.state_dict()
    assert state.keys() == restored.keys()
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, restored[key], rtol=0, atol=0)
        else:
            assert value == restored[key]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_routing_imbalance_factor": 0}, "must be positive"),
        ({"num_activation_slots": 0}, "must be positive"),
        ({"prefetch_vmm": True}, "requires VMM to be enabled"),
        ({"dtype": "bf16", "mxfp8_kernel_config": object()}, "requires dtype"),
    ],
)
def test_dist_moe_backend_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DistMoeBackendConfig(**kwargs)


def test_dist_moe_backend_config_disables_vmm_and_prefetch_by_default():
    config = DistMoeBackendConfig()

    assert config.vmm_host_scratch_imbalance_factor is None
    assert not config.prefetch_vmm


@pytest.mark.parametrize(
    "factory,num_experts_modules",
    [
        (eager_configs.deepseek_v3_debugmodel_dist_moe_bf16, 5),
        (eager_configs.deepseek_v3_16b_dist_moe_bf16, 26),
        (eager_configs.deepseek_v3_671b_dist_moe_bf16, 58),
        (graph_configs.graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16, 5),
        (graph_configs.graph_trainer_deepseek_v3_16b_dist_moe_bf16, 26),
        (graph_configs.graph_trainer_deepseek_v3_671b_dist_moe_bf16, 58),
    ],
)
def test_dist_moe_bf16_recipes_use_varlen_and_replace_all_experts(
    factory, num_experts_modules
):
    config = factory()
    model_config = config.model_spec.model
    experts = list(model_config.traverse(DistMoeRoutedExperts.Config))

    assert len(experts) == num_experts_modules
    assert all(entry[1].backend.dtype == "bf16" for entry in experts)
    assert all(
        entry[1].backend.vmm_host_scratch_imbalance_factor is None
        and not entry[1].backend.prefetch_vmm
        for entry in experts
    )
    assert all(
        isinstance(layer.attention.inner_attention, VarlenAttention.Config)
        for layer in model_config.layers
    )
    assert all(
        layer.attention.inner_attention.max_num_documents == 512
        for layer in model_config.layers
    )


def test_dist_moe_recipe_supports_cuda_graphs_with_expert_parallelism():
    config = eager_configs.deepseek_v3_debugmodel_dist_moe_bf16(seq_len=128)
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"

    config.__post_init__()
    assert config.parallelism.pipeline_parallel_per_direction_p2p
    assert config.parallelism.pipeline_parallel_reuse_recv_buffers


@pytest.mark.parametrize(
    "factory,num_experts_modules",
    [
        (eager_configs.deepseek_v3_debugmodel_dist_moe_mxfp8, 5),
        (eager_configs.deepseek_v3_16b_dist_moe_mxfp8, 26),
        (eager_configs.deepseek_v3_671b_dist_moe_mxfp8, 58),
        (graph_configs.graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8, 5),
        (graph_configs.graph_trainer_deepseek_v3_16b_dist_moe_mxfp8, 26),
        (graph_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8, 58),
    ],
)
def test_dist_moe_mxfp8_recipes_quantize_dense_linears_and_lm_head(
    factory, num_experts_modules
):
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import MXFP8Linear

    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8Linear is unavailable")
    config = factory()
    model_config = config.model_spec.model
    experts = list(model_config.traverse(DistMoeRoutedExperts.Config))
    linears = {
        fqn
        for fqn, _linear, _parent, _attr in model_config.traverse(MXFP8Linear.Config)
    }

    assert len(experts) == num_experts_modules
    assert all(entry[1].backend.dtype == "mxfp8" for entry in experts)
    assert all(
        entry[1].backend.vmm_host_scratch_imbalance_factor is None
        and not entry[1].backend.prefetch_vmm
        for entry in experts
    )
    assert "lm_head" in linears
