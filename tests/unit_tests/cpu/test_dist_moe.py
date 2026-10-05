# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch

dist_moe = pytest.importorskip(
    "dist_moe",
    reason="Dist-MoE integration tests require the optional dist_moe package",
)
import dist_moe._blockscaled  # noqa: F401

import torchtitan.config.transform as transform_api
import torchtitan.config.transform.quantization as quantization_transform
import torchtitan_recipes.models.deepseek_v3 as eager_production_recipes
import torchtitan_recipes.tests.models.deepseek_v3 as eager_test_recipes
from torch.distributed.pipelining import PipelineStageInfo
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optim import AdamW
from torchtitan.config import ParallelismConfig
from torchtitan.config.configs import TrainingConfig
from torchtitan.config.transform import (
    apply_transforms,
    LoRATransform,
    ModelConfigTransformContext,
    TokenDispatcherTransform,
)
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.experiments.graph_trainer.common_utils import (
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.grad_accumulation import (
    _GRAD_ACCUMULATOR_INPUT_META,
)
from torchtitan.experiments.graph_trainer.wgrad_accumulation import (
    fuse_wgrad_accumulation_pass,
)
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.dist_moe import (
    DistMoeRoutedExperts,
    DistMoeRuntime,
    MXFP8DistMoeRoutedExperts,
)
from torchtitan.models.common.dist_moe.runtime import _DistMoeForwardContext
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.token_dispatcher import HybridEPTokenDispatcher
from torchtitan.protocols.module import Module
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor
from torchtitan.training_engine import TrainingEngine
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel


_CONTEXT = ModelConfigTransformContext(
    training=TrainingConfig(), parallelism=ParallelismConfig()
)


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
    )


def _lora_dist_moe_config() -> DistMoeRoutedExperts.Config:
    stock = _stock_config()
    base_config = cast(
        DistMoeRoutedExperts.Config,
        DistMoeTransform(inplace_wgrad_accum=False).transform(stock),
    )
    return cast(
        DistMoeRoutedExperts.Config,
        LoRATransform(rank=8, alpha=16.0).transform(base_config),
    )


def _build_lora_dist_moe() -> DistMoeRoutedExperts:
    module = _lora_dist_moe_config().build()
    module.init_states()
    return cast(DistMoeRoutedExperts, module)


def _fill_lora_dist_moe_operands(module: DistMoeRoutedExperts) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters(), start=1):
            values = torch.arange(
                1,
                parameter.numel() + 1,
                dtype=parameter.dtype,
                device=parameter.device,
            ).reshape(parameter.shape)
            parameter.copy_(values / (10 * index))


def _lora_dist_moe_weight_references(
    module: DistMoeRoutedExperts,
) -> tuple[torch.Tensor, torch.Tensor]:
    w13_E2FD = module.w13.weight
    w13_lora_a_ELD = module.w13.lora_a.weight
    w13_lora_b_E2FL = module.w13.lora_b.weight
    w2_EDF = module.w2.weight
    w2_lora_a_ELF = module.w2.lora_a.weight
    w2_lora_b_EDL = module.w2.lora_b.weight
    w13_reference_EFD = w13_E2FD.flatten(1, 2) + 2.0 * torch.bmm(
        w13_lora_b_E2FL.flatten(1, 2),
        w13_lora_a_ELD,
    )
    w2_reference_EDF = w2_EDF + 2.0 * torch.bmm(
        w2_lora_b_EDL,
        w2_lora_a_ELF,
    )
    return w13_reference_EFD, w2_reference_EDF


def _runtime() -> DistMoeRuntime:
    runtime = object.__new__(DistMoeRuntime)
    runtime.config = DistMoeRuntime.Config()
    runtime.context = Mock()
    runtime._modules = ()
    runtime._closed = False
    runtime._forward_context_handles = []
    runtime._set_forward_context = None
    runtime.forward_context = _DistMoeForwardContext(
        runtime.context,
        active_stage_indices=frozenset({0}),
        activation_slot_id_by_stage_and_microbatch={},
        activation_slot_ids_S=torch.arange(1),
        max_moe_layers_per_activation_slot=1,
    )
    return runtime


def _dist_moe_wgrad_accumulation_graph(
    kind: str,
    accumulator_dtypes: tuple[torch.dtype, torch.dtype],
) -> SimpleNamespace:
    graph = torch.fx.Graph()

    def placeholder(name, shape, dtype=torch.bfloat16):
        node = graph.placeholder(name)
        node.meta["val"] = torch.empty(shape, dtype=dtype)
        return node

    accumulator_shapes = ((24,), (40,))
    accumulators = tuple(
        placeholder(f"accumulator_{index}", shape, dtype)
        for index, (shape, dtype) in enumerate(
            zip(accumulator_shapes, accumulator_dtypes, strict=True)
        )
    )
    for accumulator in accumulators:
        accumulator.meta[_GRAD_ACCUMULATOR_INPUT_META] = True

    backward_target = getattr(torch.ops.dist_moe, f"{kind}_backward").default
    accumulate_target = getattr(
        torch.ops.dist_moe, f"{kind}_backward_accumulate_"
    ).default
    inputs = tuple(placeholder(f"input_{index}", (1,)) for index in range(5))
    common_tail = (
        [],
        False,
        0.0,
        torch.float32,
        torch.float32,
        True,
        False,
        False,
        torch.bfloat16,
        "context",
    )
    backward_args = (
        (inputs[0], inputs[1], inputs[2], inputs[3], *common_tail)
        if kind == "bf16"
        else (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], *common_tail)
    )
    backward = graph.call_function(backward_target, args=backward_args)
    output_values = (
        torch.empty(8, 4, dtype=torch.bfloat16),
        torch.empty(8, 2, dtype=torch.float32),
        torch.empty(2, 3, 4, dtype=torch.bfloat16),
        torch.empty(2, 4, 5, dtype=torch.bfloat16),
    )
    backward.meta["val"] = output_values
    getitems = tuple(
        graph.call_function(operator.getitem, args=(backward, index))
        for index in range(4)
    )
    for getitem, value in zip(getitems, output_values, strict=True):
        getitem.meta["val"] = value

    boundaries = []
    sinks = []
    consumers = []
    for pair_index, output_index in enumerate((2, 3)):
        boundary = graph.call_function(
            torch.ops.aten.view.default,
            args=(getitems[output_index], accumulator_shapes[pair_index]),
        )
        boundary.meta["val"] = torch.empty(
            accumulator_shapes[pair_index], dtype=torch.bfloat16
        )
        if accumulator_dtypes[pair_index] == torch.float32:
            boundary = graph.call_function(
                torch.ops.aten._to_copy.default,
                args=(boundary,),
                kwargs={"dtype": torch.float32},
            )
            boundary.meta["val"] = accumulators[pair_index].meta["val"]
        sink = graph.call_function(
            torch.ops.aten.add_.Tensor,
            args=(accumulators[pair_index], boundary),
        )
        sink.meta["val"] = accumulators[pair_index].meta["val"]
        sink.meta["custom"] = {PARAMETER_GRADIENT_FQNS_META: (f"weight_{pair_index}",)}
        boundaries.append(boundary)
        sinks.append(sink)
        consumers.append(graph.call_function(torch.ops.aten.neg.default, args=(sink,)))
    graph.output((*getitems[:2], *consumers))
    return SimpleNamespace(
        gm=torch.fx.GraphModule(torch.nn.Module(), graph),
        backward=backward,
        backward_target=backward_target,
        accumulate_target=accumulate_target,
        output_values=output_values,
        accumulators=accumulators,
        getitems=getitems,
        boundaries=tuple(boundaries),
        sinks=tuple(sinks),
        consumers=tuple(consumers),
    )


@pytest.mark.parametrize("kind", ["bf16", "block_scaled"])
@pytest.mark.parametrize("accumulator_dtype", [torch.bfloat16, torch.float32])
def test_dist_moe_wgrad_accumulation_uses_accumulating_backward(
    kind, accumulator_dtype
):
    case = _dist_moe_wgrad_accumulation_graph(
        kind,
        (accumulator_dtype, accumulator_dtype),
    )

    fuse_wgrad_accumulation_pass(case.gm)

    targets = [node.target for node in case.gm.graph.nodes]
    assert case.backward.target == case.accumulate_target
    accumulator_views = case.backward.args[3:5]
    assert all(view.target == torch.ops.aten.view.default for view in accumulator_views)
    assert tuple(view.args[0] for view in accumulator_views) == case.accumulators
    assert case.backward.args[-2] == accumulator_dtype
    assert case.backward.meta["val"] == case.output_values[:2]
    assert case.backward_target not in targets
    assert torch.ops.aten.add_.Tensor not in targets
    assert all(
        consumer.args[0] is accumulator
        for consumer, accumulator in zip(case.consumers, case.accumulators, strict=True)
    )
    case.gm.graph.lint()


@pytest.mark.parametrize(
    "invalid_contract",
    [
        "missing_sink",
        "extra_getitem",
        "nonunit_alpha",
        "nonplaceholder_accumulator",
        "shape_mismatch",
        "mixed_accumulator_dtypes",
        "live_view_user",
        "live_cast_user",
    ],
)
def test_dist_moe_wgrad_accumulation_rejects_incomplete_or_ambiguous_pair(
    invalid_contract: str,
) -> None:
    if invalid_contract == "mixed_accumulator_dtypes":
        accumulator_dtypes = (torch.bfloat16, torch.float32)
    elif invalid_contract == "live_cast_user":
        accumulator_dtypes = (torch.float32, torch.float32)
    else:
        accumulator_dtypes = (torch.bfloat16, torch.bfloat16)
    case = _dist_moe_wgrad_accumulation_graph("bf16", accumulator_dtypes)
    graph = case.gm.graph
    output = graph.find_nodes(op="output")[0]

    if invalid_contract == "missing_sink":
        case.consumers[0].replace_input_with(case.sinks[0], case.boundaries[0])
        graph.erase_node(case.sinks[0])
    elif invalid_contract == "extra_getitem":
        with graph.inserting_before(output):
            extra_getitem = graph.call_function(
                operator.getitem,
                args=(case.backward, 2),
            )
            extra_getitem.meta = case.getitems[2].meta.copy()
            extra_consumer = graph.call_function(
                torch.ops.aten.neg.default,
                args=(extra_getitem,),
            )
            extra_consumer.meta["val"] = case.getitems[2].meta["val"]
        output.args = ((*output.args[0], extra_consumer),)
    elif invalid_contract == "nonunit_alpha":
        case.sinks[0].kwargs = {"alpha": 2}
    elif invalid_contract == "nonplaceholder_accumulator":
        with graph.inserting_before(case.sinks[0]):
            accumulator_alias = graph.call_function(
                torch.ops.aten.alias.default,
                args=(case.accumulators[0],),
            )
            accumulator_alias.meta = case.accumulators[0].meta.copy()
        case.sinks[0].replace_input_with(case.accumulators[0], accumulator_alias)
    elif invalid_contract == "shape_mismatch":
        case.boundaries[0].meta["val"] = torch.empty(12, dtype=torch.bfloat16)
    elif invalid_contract in ("live_view_user", "live_cast_user"):
        with graph.inserting_before(output):
            extra_consumer = graph.call_function(
                torch.ops.aten.neg.default,
                args=(case.boundaries[0],),
            )
            extra_consumer.meta["val"] = case.boundaries[0].meta["val"]
        output.args = ((*output.args[0], extra_consumer),)

    graph.lint()
    case.gm.recompile()
    targets_before = [node.target for node in graph.nodes]
    backward_args_before = case.backward.args
    backward_meta_before = case.backward.meta.copy()

    fuse_wgrad_accumulation_pass(case.gm)

    assert [node.target for node in graph.nodes] == targets_before
    assert case.backward.target == case.backward_target
    assert case.backward.args == backward_args_before
    assert case.backward.meta == backward_meta_before
    assert case.accumulate_target not in targets_before
    graph.lint()


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


class _CallbackPostprocess(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(config.dim))

    def forward(self, value_TD: torch.Tensor) -> torch.Tensor:
        return value_TD * self.weight

    def to_dist_moe_postprocess(self) -> "_CallbackPostprocess":
        """Return the module so normal forward hooks remain active."""
        return self


def test_runtime_close_releases_context_and_module_bindings() -> None:
    """Runtime teardown releases the annex context and module references."""
    runtime = _runtime()
    module = Mock()
    module._runtime = runtime
    runtime._modules = (module,)
    handle = Mock()
    runtime._forward_context_handles = [handle]
    runtime._set_forward_context = Mock()
    runtime.close()
    runtime.close()
    runtime.context.close.assert_called_once_with()
    handle.remove.assert_called_once_with()
    runtime._set_forward_context.assert_called_once_with(None)
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


def test_engine_builds_dist_moe_runtime_with_eager_pp_schedule() -> None:
    """Standard forward/backward setup supplies eager PP state to the runtime."""
    runtime = Mock()
    runtime_config = Mock()
    runtime_config.build.return_value = runtime

    engine = object.__new__(TrainingEngine)
    engine.config = SimpleNamespace(
        dist_moe=runtime_config,
        sdc_replayer=None,
        training=SimpleNamespace(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=8,
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=SimpleNamespace(
            fsdp_defer_gradient_reduction=False,
        ),
    )
    engine.model_parts = [Mock()]
    engine.parallelism_context = SimpleNamespace(pp_enabled=True)
    runtime_schedule = SimpleNamespace()
    engine.pp_schedule = runtime_schedule
    engine.device = torch.device("cuda")
    engine._dist_moe_runtime = None
    engine._forward_backward_body = Mock()

    engine._initialize_forward_backward()

    runtime_config.build.assert_called_once()
    assert engine._dist_moe_runtime is runtime
    assert runtime_config.build.call_args.kwargs["pp_schedule"] is runtime_schedule
    assert (
        runtime_config.build.call_args.kwargs["parallelism_context"]
        is engine.parallelism_context
    )
    assert (
        runtime_config.build.call_args.kwargs["functional_wgrad_dtype"]
        is torch.bfloat16
    )
    assert "set_forward_context" not in runtime_config.build.call_args.kwargs


def test_runtime_resolves_uniform_wgrad_ownership() -> None:
    """Parameter-owned WGrad keeps its live gradient dtype contract."""
    functional = SimpleNamespace(inplace_wgrad_accum=False)
    inplace = SimpleNamespace(inplace_wgrad_accum=True)

    assert (
        DistMoeRuntime._resolve_context_wgrad_dtype(
            (functional,),
            torch.bfloat16,
        )
        is torch.bfloat16
    )
    assert (
        DistMoeRuntime._resolve_context_wgrad_dtype(
            (inplace,),
            torch.bfloat16,
        )
        is None
    )


def test_runtime_rejects_mixed_wgrad_ownership() -> None:
    """One context cannot mix functional BF16 and parameter-owned FP32 WGrad."""
    functional = SimpleNamespace(inplace_wgrad_accum=False)
    inplace = SimpleNamespace(inplace_wgrad_accum=True)

    with pytest.raises(ValueError, match="same WGrad ownership mode"):
        DistMoeRuntime._resolve_context_wgrad_dtype(
            (inplace, functional),
            torch.bfloat16,
        )


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


@pytest.mark.parametrize("targets", [("w13",), ("w2",), ("w13", "w2")])
def test_lora_dist_moe_loads_grouped_lora_state_with_identical_initialization(
    targets,
) -> None:
    """Backend transforms preserve selective checkpoint keys and initialization."""
    grouped_config = LoRATransform(
        rank=8,
        alpha=16.0,
        target_modules=list(targets),
    ).transform(_stock_config())
    grouped_module = grouped_config.build()
    dist_moe_config = transform_api.transform_model_config_(
        _stock_config(),
        [
            LoRATransform(
                rank=8,
                alpha=16.0,
                target_modules=list(targets),
            ),
            DistMoeTransform(inplace_wgrad_accum=False),
        ],
        context=_CONTEXT,
    )
    dist_moe_module = dist_moe_config.build()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        grouped_module.init_states()
        torch.manual_seed(1234)
        dist_moe_module.init_states()

    grouped_state = grouped_module.state_dict()
    dist_moe_state = dist_moe_module.state_dict()
    assert list(dist_moe_state) == list(grouped_state)
    for name in grouped_state:
        torch.testing.assert_close(
            dist_moe_state[name],
            grouped_state[name],
            rtol=0,
            atol=0,
        )

    dist_moe_module.load_state_dict(grouped_state, strict=True)
    assert {
        name
        for name, parameter in dist_moe_module.named_parameters()
        if parameter.requires_grad
    } == {
        f"{projection}.{adapter}.weight"
        for projection in targets
        for adapter in ("lora_a", "lora_b")
    }
    w13_EFD, w2_EDF = dist_moe_module._weight_operands()
    torch.testing.assert_close(
        w13_EFD,
        dist_moe_module.w13.weight.flatten(1, 2),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        w2_EDF,
        dist_moe_module.w2.weight,
        rtol=0,
        atol=0,
    )


def test_lora_dist_moe_weight_operands_match_independent_references() -> None:
    """W13 and W2 operands include their expert-specific low-rank updates."""
    module = _build_lora_dist_moe()
    _fill_lora_dist_moe_operands(module)

    actual_w13_EFD, actual_w2_EDF = module._weight_operands()
    expected_w13_EFD, expected_w2_EDF = _lora_dist_moe_weight_references(module)

    assert actual_w13_EFD.shape == (4, 128, 32)
    assert actual_w2_EDF.shape == (4, 32, 64)
    torch.testing.assert_close(actual_w13_EFD, expected_w13_EFD)
    torch.testing.assert_close(actual_w2_EDF, expected_w2_EDF)


def test_lora_dist_moe_weight_operands_backpropagate_only_to_adapters() -> None:
    """Local materialization gives every adapter a finite nonzero gradient."""
    module = _build_lora_dist_moe()
    with torch.no_grad():
        module.w13.lora_a.weight.fill_(0.25)
        module.w13.lora_b.weight.fill_(0.5)
        module.w2.lora_a.weight.fill_(0.75)
        module.w2.lora_b.weight.fill_(1.0)

    w13_EFD, w2_EDF = module._weight_operands()
    w13_gradient_EFD = torch.linspace(
        0.1,
        1.0,
        w13_EFD.numel(),
        dtype=w13_EFD.dtype,
    ).reshape_as(w13_EFD)
    w2_gradient_EDF = torch.linspace(
        1.1,
        2.0,
        w2_EDF.numel(),
        dtype=w2_EDF.dtype,
    ).reshape_as(w2_EDF)
    torch.autograd.backward(
        (w13_EFD, w2_EDF),
        (w13_gradient_EFD, w2_gradient_EDF),
    )

    assert module.w13.weight.grad is None
    assert module.w2.weight.grad is None
    for name in (
        "w13.lora_a.weight",
        "w13.lora_b.weight",
        "w2.lora_a.weight",
        "w2.lora_b.weight",
    ):
        gradient = module.get_parameter(name).grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == gradient.numel()


def test_dist_moe_lora_rejects_inplace_wgrad() -> None:
    """Transient effective weights require functional Dist-MoE WGRAD."""
    with pytest.raises(ValueError, match="inplace_wgrad_accum"):
        transform_api.transform_model_config_(
            _stock_config(),
            [
                LoRATransform(target_modules=["w13"]),
                DistMoeTransform(inplace_wgrad_accum=True),
            ],
            context=_CONTEXT,
        )


def test_dist_moe_lora_rejects_mxfp8_before_build() -> None:
    """Prepared MXFP8 operands are rejected at config transformation time."""
    with pytest.raises(ValueError, match="MXFP8DistMoeRoutedExperts"):
        transform_api.transform_model_config_(
            _stock_config(),
            [
                DistMoeTransform(expert_precision="mxfp8"),
                LoRATransform(target_modules=["w13"]),
            ],
            context=_CONTEXT,
        )


@pytest.mark.parametrize(
    ("expert_transform", "expected_owner"),
    [
        (DistMoeTransform(), DistMoeRoutedExperts),
        (
            DistMoeTransform(expert_precision="mxfp8"),
            MXFP8DistMoeRoutedExperts,
        ),
    ],
)
def test_dist_moe_lora_dense_target_remains_legal_with_non_target_experts(
    expert_transform,
    expected_owner,
) -> None:
    """Dense-only LoRA leaves either Dist-MoE backend unadapted and frozen."""
    transformed = transform_api.transform_model_config_(
        deepseek_v3_debugmodel().model,
        [
            expert_transform,
            LoRATransform(
                rank=8,
                alpha=16.0,
                target_modules=["wo"],
            ),
        ],
        context=_CONTEXT,
    )

    routed_configs = list(transformed.traverse(DistMoeRoutedExperts.Config))
    assert len(routed_configs) == 5
    assert all(config._owner is expected_owner for _, config, _, _ in routed_configs)
    dense_targets = [
        config
        for fqn, config, _parent, _attr in transformed.traverse(Linear.Config)
        if fqn.endswith(".wo")
    ]
    assert len(dense_targets) == 6
    assert all(hasattr(config, "rank") for config in dense_targets)


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
        num_local_input_tokens=128,
        max_live_activation_slots=2,
        max_moe_layers_per_activation_slot=3,
    )

    assert context_config.num_local_input_tokens == 128
    assert context_config.max_moe_layers_per_activation_slot == 3
    assert context_config.device_scratch_capacity_factor == 2.0
    assert context_config.activation_slot_bytes == 2048
    assert context_config.activation_slot_capacity_factor is None
    assert context_config.num_activation_slots == 2
    assert context_config.vmm is not None
    assert context_config.vmm.total_scratch_capacity_factor == 4.0
    assert context_config.vmm.prefetch
    assert context_config.bf16_grouped_gemm_preset == "1cta1mma_bm64_bn128"
    assert context_config.wgrad_dtype is None

    runtime.config = DistMoeRuntime.Config(
        scratch_capacity_factor=2.0,
        activation_slot_capacity_factor=1.5,
        vmm_capacity_factor=4.0,
    )
    factor_config = runtime._resolve_context_config(
        module,
        num_local_input_tokens=128,
        max_live_activation_slots=2,
        max_moe_layers_per_activation_slot=3,
    )
    assert factor_config.activation_slot_bytes is None
    assert factor_config.activation_slot_capacity_factor == 1.5
    assert factor_config.num_activation_slots == 2
    assert factor_config.wgrad_dtype is None


def test_mxfp8_transform_is_independent_and_uses_prepared_weights() -> None:
    """MXFP8 transforms stock experts directly and installs prepared weights."""
    transformed = DistMoeTransform(expert_precision="mxfp8").transform(_stock_config())
    assert isinstance(transformed, MXFP8DistMoeRoutedExperts.Config)

    module = transformed.build()
    assert isinstance(module, MXFP8DistMoeRoutedExperts)
    assert isinstance(module.w13.weight, _ShardedFSDPTensor)
    assert isinstance(module.w2.weight, _ShardedFSDPTensor)
    assert list(module.state_dict()) == ["w13.weight", "w2.weight"]


@pytest.mark.parametrize(
    "other_transform",
    [
        TokenDispatcherTransform(dispatcher=HybridEPTokenDispatcher),
    ],
)
def test_dist_moe_transform_rejects_incompatible_transforms(
    other_transform,
) -> None:
    """Dist-MoE rejects a token-dispatch transform whose execution it replaces."""
    config = deepseek_v3_debugmodel()
    with pytest.raises(ValueError, match="cannot be combined"):
        apply_transforms(
            config,
            [DistMoeTransform(), other_transform],
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


def test_eager_postprocess_callback_preserves_module_hooks_and_gradients() -> None:
    """Callable postprocessing uses Module.__call__ and ordinary autograd."""
    stock = _stock_config()
    stock.output_postprocess = _CallbackPostprocess.Config(dim=32)
    transformed = DistMoeTransform().transform(stock)
    module = cast(DistMoeRoutedExperts, transformed.build())
    postprocess = cast(_CallbackPostprocess, module.output_postprocess)
    forward_pre_hook = Mock(return_value=None)
    hook = postprocess.register_forward_pre_hook(forward_pre_hook)

    callback = module._output_postprocess()
    assert callback is postprocess
    assert dist_moe.ExecutionOptions(experts_output_postprocess=callback).requires_eager
    value_TD = torch.randn(2, 32, requires_grad=True)
    callback(value_TD).square().sum().backward()

    hook.remove()
    forward_pre_hook.assert_called_once()
    assert value_TD.grad is not None
    assert postprocess.weight.grad is not None
    assert "output_postprocess.weight" in module.state_dict()

    with (
        patch.object(postprocess, "to_dist_moe_postprocess", return_value=object()),
        pytest.raises(TypeError, match="RMSNormPostprocess or callable"),
    ):
        module._output_postprocess()


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
    "factory,num_experts_modules,scratch_capacity_factor,max_documents",
    [
        (eager_test_recipes.deepseek_v3_debugmodel_dist_moe_bf16, 5, 1.0, 512),
        (eager_test_recipes.deepseek_v3_16b_dist_moe_bf16, 26, 4.0, 512),
        (eager_production_recipes.deepseek_v3_671b_dist_moe_bf16, 58, 4.0, 32),
    ],
)
def test_dist_moe_bf16_recipes_use_varlen_and_replace_all_experts(
    factory,
    num_experts_modules,
    scratch_capacity_factor,
    max_documents,
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
    assert all(expert.inplace_wgrad_accum for _, expert, _, _ in experts)
    assert all(
        isinstance(layer.attention.inner_attention, VarlenInnerAttention.Config)
        for layer in model_config.layers
    )
    assert config.dataloader.max_num_documents == max_documents


def test_eager_dist_moe_recipe_supports_cuda_graphs_with_pipeline_parallelism():
    """The eager recipe accepts EP and PP while retaining CUDA graphs."""
    config = eager_test_recipes.deepseek_v3_debugmodel_dist_moe_bf16(seq_len=128)
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"

    config.__post_init__()


@pytest.mark.parametrize(
    "factory,num_experts_modules,scratch_capacity_factor,max_documents",
    [
        (eager_test_recipes.deepseek_v3_debugmodel_dist_moe_mxfp8, 5, 1.0, 512),
        (eager_test_recipes.deepseek_v3_16b_dist_moe_mxfp8, 26, 4.0, 512),
        (eager_production_recipes.deepseek_v3_671b_dist_moe_mxfp8, 58, 4.0, 32),
    ],
)
def test_dist_moe_mxfp8_recipes_quantize_dense_linears_and_lm_head(
    factory,
    num_experts_modules,
    scratch_capacity_factor,
    max_documents,
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
    assert all(expert.inplace_wgrad_accum for _, expert, _, _ in experts)
    assert "lm_head" in linears
    assert config.dataloader.max_num_documents == max_documents


@pytest.mark.parametrize(
    "factory,dp_degree,num_microbatches",
    [
        (eager_production_recipes.deepseek_v3_671b_dist_moe_bf16, 128, 120),
        (eager_production_recipes.deepseek_v3_671b_dist_moe_mxfp8, 64, 240),
    ],
)
def test_dist_moe_671b_recipe_topology(factory, dp_degree, num_microbatches) -> None:
    """The verified recipes encode their exact PP/VPP/DP/EP batch contract."""
    config = factory()

    assert config.parallelism.pipeline_parallel_degree == 4
    assert config.parallelism.pipeline_parallel_layers_per_stage == 4
    assert config.parallelism.data_parallel_shard_degree == dp_degree
    assert config.parallelism.expert_parallel_degree == 64
    assert config.parallelism.num_pp_microbatches == num_microbatches
    assert config.training.max_context_length == 4096
    assert config.training.num_tokens_per_microbatch_per_dp_rank == 4096
    assert config.training.num_tokens_per_train_step % 4096 == 0
    assert config.training.num_tokens_per_train_step // 4096 == 15360
    assert config.training.dtype == "float32"
    assert config.training.mixed_precision_param == "bfloat16"
    assert config.training.mixed_precision_reduce == "bfloat16"
    assert config.activation_checkpoint is None
    assert isinstance(config.loss, CrossEntropyLoss.Config)
    assert config.override.imports == [
        "torchtitan_recipes.overrides.fused_mla.fused_mla",
        "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu",
    ]
    (optimizer_config,) = config.optim.optimizer.optimizers
    assert isinstance(optimizer_config, AdamW.Config)
    assert optimizer_config.moment_dtype == "bfloat16"
