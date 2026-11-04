# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.models import (
    deepseek_v3,
    deepseek_v4,
    gpt_oss,
    kimi_k2_7,
    kimi_k3,
    muse_spark,
    qwen3,
    qwen3_5,
)


# One MoE debug flavor per model on the shared router.
@pytest.mark.parametrize(
    "model, flavor",
    [
        (qwen3, "debugmodel_moe"),
        (qwen3_5, "debugmodel_moe"),
        (deepseek_v3, "debugmodel"),
        (deepseek_v4, "debugmodel"),
        (gpt_oss, "debugmodel"),
        (kimi_k2_7, "debugmodel"),
        (kimi_k3, "debugmodel"),
        (muse_spark, "debugmodel"),
    ],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_moe_models_list_only_registered_regions(model, flavor: str) -> None:
    """Each MoE model's default regions, moe_router_topk included, are registered."""
    config = model.build_model_config(flavor, parallelism=ParallelismConfig())
    try:
        # Raises on a name no local_compile function registered.
        LocalCompileConfig(regions=config.local_compile.regions).apply()
    finally:
        # The binding is process-wide: don't leave compiled regions on for later tests.
        LocalCompileConfig(regions=[]).apply()


@pytest.mark.parametrize(
    "calls, expected_graphs",
    [
        # Trainer with its validator: train with grad, then eval without grad, one token count.
        ([(256, True, False)] * 2 + [(256, False, False)], 4),
        # vLLM generator, eval under inference_mode: the profile run at the largest token count,
        # then capture sizes largest first, then decode. Graphs: the first T, any other T, T = 1.
        ([(T, False, True) for T in (512, 256, 64, 8, 2, 1, 37)], 6),
    ],
    ids=["trainer_with_validator", "generator"],
)
def test_deepseek_v4_router_region_graphs_per_process(calls, expected_graphs) -> None:
    """Each kind of process compiles a fixed number of router-region graphs, under the region's limit.

    DeepSeek-V4 has 2 router kinds (hash and top-k layers), so it compiles 2 graphs per call kind.
    """
    graphs = []

    class CountingConfig(LocalCompileConfig):
        @staticmethod
        def _compile(reference, **compile_kwargs):
            compile_kwargs.pop("options", None)  # Inductor-only

            def backend(gm, example_inputs):
                graphs.append(gm)
                return gm.forward

            return torch.compile(reference, backend=backend, **compile_kwargs)

    config = deepseek_v4.build_model_config(
        "debugmodel", parallelism=ParallelismConfig()
    )
    routers = [config.layers[i].feed_forward.router.build() for i in (0, -1)]
    assert [router.hash for router in routers] == [True, False]
    # Real runs create the SPMD mesh stack in ParallelismContext.activate_spmd(), before the
    # first forward. Creating it after the first compile adds a graph.
    current_spmd_mesh()
    torch._dynamo.reset()
    CountingConfig(regions=["moe_router_topk"]).apply()
    try:
        for num_tokens, training, inference in calls:
            for router in routers:
                router.train(training)
                mode = torch.inference_mode() if inference else contextlib.nullcontext()
                with mode, torch.set_grad_enabled(training):
                    router._score_and_select_experts(
                        torch.randn(
                            num_tokens, router.num_experts, requires_grad=training
                        ),
                        expert_bias_E=None,
                        # A job passes a padding mask on every call or on none; eval doesn't read it.
                        padding_mask_T=torch.arange(num_tokens) % 11 == 0,
                        input_ids_T=torch.randint(0, 100, (num_tokens,)),
                    )
        # Equal: the counts in the recompile_limit comment in moe.py stay true. A new guard
        # raises the count.
        assert len(graphs) == expected_graphs
    finally:
        LocalCompileConfig(regions=[]).apply()
        torch._dynamo.reset()
