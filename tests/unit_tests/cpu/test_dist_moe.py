# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import dist_moe
import pytest
import torch

import torchtitan.config.transform.quantization as quantization_transform
from torch.distributed.pipelining import PipelineStageInfo
from torchtitan.config.configs import TrainingConfig
from torchtitan.config.transform import (
    apply_transforms,
    DistMoeTransform,
    LoRATransform,
)
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    config_registry as graph_configs,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import GraphRuntime
from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.dist_moe import (
    DistMoeRoutedExperts,
    DistMoeRuntime,
    MXFP8DistMoeRoutedExperts,
)
from torchtitan.models.common.dist_moe.runtime import _DistMoeForwardContext
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v3 import config_registry as eager_configs
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
from torchtitan.protocols.module import Module
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor
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
    runtime.context = Mock()
    runtime._modules = ()
    runtime.forward_context = _DistMoeForwardContext(
        runtime.context,
        active_stage_indices=frozenset({0}),
        activation_slot_id_by_stage_and_microbatch={},
        activation_slot_ids_S=torch.arange(1),
        max_moe_layers_per_activation_slot=1,
    )
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


def test_runtime_close_releases_context_and_module_bindings() -> None:
    """Runtime teardown releases the annex context and module references."""
    runtime = _runtime()
    module = Mock()
    module._runtime = runtime
    runtime._modules = (module,)
    runtime.close()
    runtime.context.close.assert_called_once_with()
    assert module._runtime is None


def test_runtime_selects_pp_activation_slot_from_forward_context() -> None:
    """Pipeline metadata selects the precomputed annex activation slot."""
    runtime = _runtime()
    runtime.forward_context = _DistMoeForwardContext(
        runtime.context,
        active_stage_indices=frozenset({3}),
        activation_slot_id_by_stage_and_microbatch={(3, 7): 2},
        activation_slot_ids_S=torch.arange(3),
        max_moe_layers_per_activation_slot=5,
    )

    with runtime.forward_context(PipelineStageInfo(stage_index=3, microbatch_index=7)):
        runtime.context.select_activation_slot.assert_called_once_with(2, 5)
    assert (
        runtime.forward_context.resolve_activation_slot(
            PipelineStageInfo(stage_index=3, microbatch_index=7)
        ).item()
        == 2
    )


def test_engine_owns_dist_moe_runtime_context_and_cleanup() -> None:
    """The engine builds one runtime and registers its eager PP context."""
    runtime = Mock()
    runtime.forward_context = Mock()
    runtime_config = Mock()
    runtime_config.build.return_value = runtime
    stage_handle = Mock()
    stage = Mock()
    stage.register_forward_context.return_value = stage_handle

    engine = object.__new__(TrainingEngine)
    engine.config = SimpleNamespace(
        dist_moe=runtime_config,
        training=SimpleNamespace(
            num_tokens_per_microbatch_per_dp_rank=8,
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="bfloat16",
        ),
    )
    engine.model_parts = [Mock()]
    engine.parallelism_context = SimpleNamespace(pp_enabled=True)
    runtime_schedule = SimpleNamespace(_stages=[stage])
    engine.pp_schedule = runtime_schedule
    engine.device = torch.device("cuda")
    engine.dist_moe_runtime = None
    engine._dist_moe_forward_context_handles = []

    engine._initialize_dist_moe_runtime()
    engine._close_dist_moe_runtime()

    runtime_config.build.assert_called_once()
    assert runtime_config.build.call_args.kwargs["pp_schedule"] is runtime_schedule
    assert (
        runtime_config.build.call_args.kwargs["parallelism_context"]
        is engine.parallelism_context
    )
    assert runtime_config.build.call_args.kwargs["wgrad_dtype"] is torch.bfloat16
    stage.register_forward_context.assert_called_once_with(runtime.forward_context)
    stage_handle.remove.assert_called_once_with()
    runtime.close.assert_called_once_with()


def test_graph_engine_passes_dist_moe_context_to_graph_runtime() -> None:
    """GraphTrainingEngine owns GraphPP-specific context registration."""
    runtime = Mock()
    runtime.forward_context = Mock()
    runtime_config = Mock()
    runtime_config.build.return_value = runtime
    graph_runtime = object.__new__(GraphRuntime)
    graph_runtime._liveness_schedule = SimpleNamespace()
    graph_runtime._graph_pp_ready = False
    graph_runtime._dist_moe_forward_context = None

    engine = object.__new__(GraphTrainingEngine)
    engine.config = SimpleNamespace(
        dist_moe=runtime_config,
        training=SimpleNamespace(
            num_tokens_per_microbatch_per_dp_rank=8,
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
    )
    engine.model_parts = [Mock()]
    engine.parallelism_context = SimpleNamespace(pp_enabled=True)
    engine.pp_schedule = graph_runtime
    engine.device = torch.device("cuda")
    engine.dist_moe_runtime = None
    engine._dist_moe_forward_context_handles = []

    engine._initialize_dist_moe_runtime()

    assert graph_runtime._dist_moe_forward_context is runtime.forward_context
    assert runtime_config.build.call_args.kwargs["pp_schedule"] is (
        graph_runtime.pipeline_liveness_schedule
    )
    assert runtime_config.build.call_args.kwargs["wgrad_dtype"] is torch.bfloat16


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
    transformed = DistMoeTransform(
        bf16_grouped_gemm_preset="1cta1mma_bm64_bn128",
    ).transform(_stock_config(dim=64))
    module = cast(DistMoeRoutedExperts, transformed.build())
    runtime = _runtime()
    runtime.config = DistMoeRuntime.Config(
        scratch_capacity_factor=2.0,
        activation_slot_bytes=2048,
        activation_slot_capacity_factor=None,
        vmm_capacity_factor=4.0,
    )

    context_config = runtime._resolve_context_config(
        module,
        max_local_input_tokens=128,
        max_live_activation_slots=2,
        max_moe_layers_per_activation_slot=3,
        wgrad_dtype=torch.float32,
    )

    assert context_config.max_local_input_tokens == 128
    assert context_config.max_moe_layers_per_activation_slot == 3
    assert context_config.device_scratch_capacity_factor == 2.0
    assert context_config.activation_slot_bytes == 2048
    assert context_config.activation_slot_capacity_factor is None
    assert context_config.num_activation_slots == 2
    assert context_config.vmm is not None
    assert context_config.vmm.total_scratch_capacity_factor == 4.0
    assert context_config.vmm.prefetch
    assert context_config.bf16_grouped_gemm_preset == "1cta1mma_bm64_bn128"
    assert context_config.wgrad_dtype is torch.float32

    runtime.config = DistMoeRuntime.Config(
        scratch_capacity_factor=2.0,
        activation_slot_capacity_factor=1.5,
        vmm_capacity_factor=4.0,
    )
    factor_config = runtime._resolve_context_config(
        module,
        max_local_input_tokens=128,
        max_live_activation_slots=2,
        max_moe_layers_per_activation_slot=3,
        wgrad_dtype=torch.bfloat16,
    )
    assert factor_config.activation_slot_bytes is None
    assert factor_config.activation_slot_capacity_factor == 1.5
    assert factor_config.num_activation_slots == 2
    assert factor_config.wgrad_dtype is torch.bfloat16


def test_mxfp8_transform_is_independent_and_uses_prepared_weights() -> None:
    """MXFP8 transforms stock experts directly and installs prepared weights."""
    transformed = DistMoeTransform(expert_precision="mxfp8").transform(_stock_config())
    assert isinstance(transformed, MXFP8DistMoeRoutedExperts.Config)

    module = transformed.build()
    assert isinstance(module, MXFP8DistMoeRoutedExperts)
    assert isinstance(module.w13.weight, _ShardedFSDPTensor)
    assert isinstance(module.w2.weight, _ShardedFSDPTensor)
    assert list(module.state_dict()) == ["w13.weight", "w2.weight"]


def test_dist_moe_transform_rejects_lora() -> None:
    """Dist-MoE rejects LoRA until routed-expert adapters are supported."""
    config = deepseek_v3_debugmodel()
    with pytest.raises(ValueError, match="cannot be combined"):
        apply_transforms(
            config,
            [DistMoeTransform(), LoRATransform(handlers=())],
        )


def test_forward_passes_native_postprocess_and_wgrad_policy() -> None:
    """Forward passes module-owned postprocessing and annex-owned WGRAD policy."""
    stock = _stock_config()
    stock.output_postprocess = _NativePostprocess.Config(dim=32)
    transformed = DistMoeTransform().transform(stock)
    module = cast(DistMoeRoutedExperts, transformed.build())
    module._runtime = _runtime()
    module._runtime.context = cast(Any, object())

    with (
        patch(
            "torchtitan.models.common.dist_moe.routed_experts.dist_moe.routed_experts",
            return_value=torch.empty(2, 32),
        ) as execute,
        patch(
            "torchtitan.models.common.dist_moe.routed_experts.remat.region",
            side_effect=lambda fn, *_args, **_kwargs: fn,
        ) as remat_region,
        patch(
            "torchtitan.models.common.dist_moe.routed_experts.remat.recompute_needs_tensor"
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
        ({"scratch_capacity_factor": 0}, ValueError, "must be finite and positive"),
        (
            {
                "activation_slot_bytes": -1,
                "activation_slot_capacity_factor": None,
            },
            ValueError,
            "cannot be negative",
        ),
        (
            {
                "activation_slot_bytes": True,
                "activation_slot_capacity_factor": None,
            },
            TypeError,
            "must be an integer",
        ),
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
        ({"vmm_capacity_factor": 0}, ValueError, "must be finite and positive"),
    ],
)
def test_runtime_config_rejects_invalid_values(kwargs, error_type, message) -> None:
    """Rank-wide runtime configuration rejects invalid memory policies."""
    with pytest.raises(error_type, match=message):
        DistMoeRuntime.Config(**kwargs)


def test_runtime_config_requires_bfloat16_unsharded_parameters() -> None:
    """Dist-MoE rejects FSDP mixed-precision parameter dtypes it cannot consume."""
    with pytest.raises(ValueError, match="mixed_precision_param='bfloat16'"):
        TrainingEngine.Config(
            dist_moe=DistMoeRuntime.Config(),
            training=TrainingConfig(mixed_precision_param="float32"),
        )


@pytest.mark.parametrize(
    "factory,num_experts_modules,scratch_capacity_factor,inplace_wgrad_accum",
    [
        (eager_configs.deepseek_v3_debugmodel_dist_moe_bf16, 5, 1.0, True),
        (eager_configs.deepseek_v3_16b_dist_moe_bf16, 26, 4.0, True),
        (eager_configs.deepseek_v3_671b_dist_moe_bf16, 58, 4.0, True),
        (
            graph_configs.graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16,
            5,
            1.0,
            False,
        ),
        (
            graph_configs.graph_trainer_deepseek_v3_16b_dist_moe_bf16,
            26,
            4.0,
            False,
        ),
        (
            graph_configs.graph_trainer_deepseek_v3_671b_dist_moe_bf16,
            58,
            4.0,
            False,
        ),
    ],
)
def test_dist_moe_bf16_recipes_use_varlen_and_replace_all_experts(
    factory, num_experts_modules, scratch_capacity_factor, inplace_wgrad_accum
):
    config = factory()
    model_config = config.model
    experts = list(model_config.traverse(DistMoeRoutedExperts.Config))
    runtime = config.dist_moe

    assert len(experts) == num_experts_modules
    assert all(type(entry[1]) is DistMoeRoutedExperts.Config for entry in experts)
    assert isinstance(runtime, DistMoeRuntime.Config)
    assert runtime.vmm_capacity_factor is None
    assert runtime.scratch_capacity_factor == scratch_capacity_factor
    assert config.training.mixed_precision_reduce == "bfloat16"
    assert all(
        expert.inplace_wgrad_accum is inplace_wgrad_accum for _, expert, _, _ in experts
    )
    assert all(
        isinstance(layer.attention.inner_attention, VarlenInnerAttention.Config)
        for layer in model_config.layers
    )
    assert config.dataloader.max_num_documents == 512


def test_eager_dist_moe_recipe_supports_cuda_graphs_with_pipeline_parallelism():
    """The eager recipe accepts EP and PP while retaining CUDA graphs."""
    config = eager_configs.deepseek_v3_debugmodel_dist_moe_bf16(seq_len=128)
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"

    config.model.update_from_config(config=config)
    config.__post_init__()


@pytest.mark.parametrize(
    "factory,num_experts_modules,scratch_capacity_factor,inplace_wgrad_accum",
    [
        (eager_configs.deepseek_v3_debugmodel_dist_moe_mxfp8, 5, 1.0, True),
        (eager_configs.deepseek_v3_16b_dist_moe_mxfp8, 26, 4.0, True),
        (eager_configs.deepseek_v3_671b_dist_moe_mxfp8, 58, 4.0, True),
        (
            graph_configs.graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8,
            5,
            1.0,
            False,
        ),
        (
            graph_configs.graph_trainer_deepseek_v3_16b_dist_moe_mxfp8,
            26,
            4.0,
            False,
        ),
        (
            graph_configs.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8,
            58,
            4.0,
            False,
        ),
    ],
)
def test_dist_moe_mxfp8_recipes_quantize_dense_linears_and_lm_head(
    factory,
    num_experts_modules,
    scratch_capacity_factor,
    inplace_wgrad_accum,
    monkeypatch,
):
    pytest.importorskip("torchao")
    from torchtitan.quantization import MXFP8Linear

    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8Linear is unavailable")
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    config = factory()
    model_config = config.model
    experts = list(model_config.traverse(DistMoeRoutedExperts.Config))
    runtime = config.dist_moe
    linears = {
        fqn
        for fqn, _linear, _parent, _attr in model_config.traverse(MXFP8Linear.Config)
    }

    assert len(experts) == num_experts_modules
    assert all(
        isinstance(entry[1], MXFP8DistMoeRoutedExperts.Config) for entry in experts
    )
    assert isinstance(runtime, DistMoeRuntime.Config)
    assert runtime.vmm_capacity_factor is None
    assert runtime.scratch_capacity_factor == scratch_capacity_factor
    assert config.training.mixed_precision_reduce == "bfloat16"
    assert all(
        expert.inplace_wgrad_accum is inplace_wgrad_accum for _, expert, _, _ in experts
    )
    assert "lm_head" in linears
