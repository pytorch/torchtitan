# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer-specific Dist-MoE integration tests."""

import operator
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch

import torchtitan_recipes.graph_trainer.deepseek_v3 as production_recipes
import torchtitan_recipes.tests.graph_trainer.deepseek_v3 as test_recipes

import torchtitan.config.transform.quantization as quantization_transform
from torchtitan.experiments.graph_trainer.common_utils import (
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.grad_accumulation import (
    _GRAD_ACCUMULATOR_INPUT_META,
)
from torchtitan.experiments.graph_trainer.graph_pp.runner import GraphRuntime
from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine
from torchtitan.experiments.graph_trainer.wgrad_accumulation import (
    fuse_wgrad_accumulation_pass,
)
from torchtitan.models.common.attention import VarlenInnerAttention


def test_graph_trainer_imports_do_not_require_dist_moe() -> None:
    """GraphTrainer and its recipes retain the optional package boundary."""
    script = r"""
import importlib.abc
import sys

class BlockDistMoe(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "dist_moe" or fullname.startswith("dist_moe."):
            raise ModuleNotFoundError("blocked optional import", name=fullname)
        return None

sys.meta_path.insert(0, BlockDistMoe())
import torchtitan.experiments.graph_trainer.graph_builder
import torchtitan_recipes.graph_trainer.deepseek_v3 as recipes

try:
    recipes.graph_trainer_deepseek_v3_671b_dist_moe_bf16()
except ModuleNotFoundError as error:
    assert error.name == "dist_moe"
    assert "optional dist_moe package" in str(error)
else:
    raise AssertionError("Dist-MoE recipe unexpectedly loaded without its package")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_graph_engine_supplies_dist_moe_graph_pp_registration() -> None:
    """GraphTrainer supplies liveness and registration before graph tracing."""
    runtime = Mock()
    runtime_config = Mock()
    runtime_config.build.return_value = runtime
    graph_runtime = cast(Any, object.__new__(GraphRuntime))
    execution_schedule = Mock()
    graph_runtime.schedule = execution_schedule
    graph_runtime._liveness_schedule = SimpleNamespace()
    graph_runtime._graph_pp_ready = False
    graph_runtime._dist_moe_forward_context = None

    engine = cast(Any, object.__new__(GraphTrainingEngine))
    engine.config = SimpleNamespace(
        dist_moe=runtime_config,
        sdc_replayer=None,
        training=SimpleNamespace(
            num_tokens_per_train_step=-1,
            num_tokens_per_microbatch_per_dp_rank=8,
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="bfloat16",
        ),
        parallelism=SimpleNamespace(fsdp_defer_gradient_reduction=False),
        compile=SimpleNamespace(memory_policy="save_all"),
    )
    engine.model_parts = [Mock()]
    engine.parallelism_context = SimpleNamespace(pp_enabled=True)
    engine.pp_schedule = graph_runtime
    engine.device = torch.device("cuda")
    engine._dist_moe_runtime = None
    engine._forward_backward_body = Mock()

    with patch("torchtitan.experiments.graph_trainer.trainer.maybe_apply_numa_binding"):
        engine._initialize_forward_backward()

    assert engine._dist_moe_runtime is runtime
    assert runtime_config.build.call_args.kwargs["pp_schedule"] is (
        graph_runtime.pipeline_liveness_schedule
    )
    assert runtime_config.build.call_args.kwargs["wgrad_dtype"] is torch.bfloat16
    cleanup_registrar = runtime_config.build.call_args.kwargs[
        "register_post_metadata_inference_cleanup"
    ]
    assert cleanup_registrar.__self__ is graph_runtime
    assert (
        cleanup_registrar.__func__
        is GraphRuntime.register_post_metadata_inference_cleanup
    )
    callback = Mock()
    handle = cleanup_registrar(callback)
    execution_schedule.register_post_metadata_inference_cleanup.assert_called_once_with(
        callback
    )
    assert (
        handle
        is execution_schedule.register_post_metadata_inference_cleanup.return_value
    )
    setter = runtime_config.build.call_args.kwargs["set_forward_context"]
    assert setter.__self__ is graph_runtime
    assert setter.__func__ is GraphRuntime.set_dist_moe_forward_context


def _dist_moe_wgrad_accumulation_graph(
    kind: str,
    accumulator_dtypes: tuple[torch.dtype, torch.dtype],
) -> SimpleNamespace:
    pytest.importorskip(
        "dist_moe._blockscaled",
        reason="Dist-MoE WGrad tests require the optional dist_moe package",
    )
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


def _assert_common_recipe(
    config,
    *,
    num_expert_modules: int,
    scratch_capacity_factor: float,
    max_documents: int,
):
    pytest.importorskip("dist_moe")
    from torchtitan.models.common.dist_moe import DistMoeRoutedExperts, DistMoeRuntime

    experts = list(config.model.traverse(DistMoeRoutedExperts.Config))
    runtime = config.dist_moe

    assert len(experts) == num_expert_modules
    assert isinstance(runtime, DistMoeRuntime.Config)
    assert runtime.vmm_capacity_factor is None
    assert runtime.scratch_capacity_factor == scratch_capacity_factor
    assert config.training.mixed_precision_reduce == "bfloat16"
    assert all(not expert.inplace_wgrad_accum for _, expert, _, _ in experts)
    assert all(
        isinstance(layer.attention.inner_attention, VarlenInnerAttention.Config)
        for layer in config.model.layers
    )
    assert config.dataloader.max_num_documents == max_documents
    return experts


@pytest.mark.parametrize(
    "factory,num_expert_modules,scratch_capacity_factor,max_documents",
    [
        (test_recipes.graph_trainer_deepseek_v3_debugmodel_dist_moe_bf16, 5, 1.0, 512),
        (test_recipes.graph_trainer_deepseek_v3_16b_dist_moe_bf16, 26, 4.0, 512),
        (
            production_recipes.graph_trainer_deepseek_v3_671b_dist_moe_bf16,
            58,
            4.0,
            32,
        ),
    ],
)
def test_bf16_recipes_replace_all_experts(
    factory,
    num_expert_modules,
    scratch_capacity_factor,
    max_documents,
) -> None:
    """GraphTrainer BF16 recipes use native functional-WGrad experts."""
    from torchtitan.models.common.dist_moe import DistMoeRoutedExperts

    experts = _assert_common_recipe(
        factory(),
        num_expert_modules=num_expert_modules,
        scratch_capacity_factor=scratch_capacity_factor,
        max_documents=max_documents,
    )
    assert all(type(entry[1]) is DistMoeRoutedExperts.Config for entry in experts)


@pytest.mark.parametrize(
    "factory,num_expert_modules,scratch_capacity_factor,max_documents",
    [
        (test_recipes.graph_trainer_deepseek_v3_debugmodel_dist_moe_mxfp8, 5, 1.0, 512),
        (test_recipes.graph_trainer_deepseek_v3_16b_dist_moe_mxfp8, 26, 4.0, 512),
        (
            production_recipes.graph_trainer_deepseek_v3_671b_dist_moe_mxfp8,
            58,
            4.0,
            32,
        ),
    ],
)
def test_mxfp8_recipes_quantize_experts_dense_linears_and_lm_head(
    factory,
    num_expert_modules,
    scratch_capacity_factor,
    max_documents,
    monkeypatch,
) -> None:
    """GraphTrainer MXFP8 recipes configure every intended projection."""
    pytest.importorskip("torchao")
    from torchtitan.models.common.dist_moe import MXFP8DistMoeRoutedExperts
    from torchtitan.quantization import MXFP8Linear

    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8Linear is unavailable")
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    config = factory()
    experts = _assert_common_recipe(
        config,
        num_expert_modules=num_expert_modules,
        scratch_capacity_factor=scratch_capacity_factor,
        max_documents=max_documents,
    )
    linears = {
        fqn
        for fqn, _linear, _parent, _attr in config.model.traverse(MXFP8Linear.Config)
    }
    assert all(
        isinstance(entry[1], MXFP8DistMoeRoutedExperts.Config) for entry in experts
    )
    assert "lm_head" in linears
