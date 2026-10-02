# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer-specific Dist-MoE integration tests."""

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
from torchtitan.experiments.graph_trainer.graph_pp.runner import GraphRuntime
from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine
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
    graph_runtime._liveness_schedule = SimpleNamespace()
    graph_runtime.schedule = SimpleNamespace(_stages=[Mock()])
    graph_runtime._graph_pp_ready = False
    graph_runtime._dist_moe_forward_context = None

    engine = cast(Any, object.__new__(GraphTrainingEngine))
    engine.config = SimpleNamespace(
        dist_moe=runtime_config,
        sdc_replayer=None,
        training=SimpleNamespace(
            disable_cuda_graphs=True,
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
    engine._outer_cudagraphs_enabled = False

    with patch("torchtitan.experiments.graph_trainer.trainer.maybe_apply_numa_binding"):
        engine._initialize_forward_backward()

    assert engine._dist_moe_runtime is runtime
    assert runtime_config.build.call_args.kwargs["pp_schedule"] is (
        graph_runtime.pipeline_liveness_schedule
    )
    assert (
        runtime_config.build.call_args.kwargs["functional_wgrad_dtype"]
        is torch.bfloat16
    )
    setter = runtime_config.build.call_args.kwargs["set_forward_context"]
    assert setter.__self__ is graph_runtime
    assert setter.__func__ is GraphRuntime.set_dist_moe_forward_context


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
