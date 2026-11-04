# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
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


def test_deepseek_v4_router_region_stays_under_its_graph_limit() -> None:
    """A mix of train and eval calls compiles under the region's recompile limit (fullgraph raises past it).

    Order: every call kind first at one size (static graphs), then at a second size (dynamic),
    then decode at T = 1 (size 1 is specialized).
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
    train_kinds = [
        (True, grad, padded) for grad in (True, False) for padded in (False, True)
    ]
    eval_kinds = [(False, grad, False) for grad in (True, False)]
    torch._dynamo.reset()
    CountingConfig(regions=["moe_router_topk"]).apply()
    try:
        for num_tokens, kinds in (
            (64, train_kinds + eval_kinds),
            (96, train_kinds + eval_kinds),
            (1, eval_kinds),
        ):
            for training, grad, padded in kinds:
                for router in routers:
                    router.train(training)
                    gate_TE = torch.randn(
                        num_tokens, router.num_experts, requires_grad=grad
                    )
                    padding_mask_T = torch.arange(num_tokens) % 11 == 0
                    with torch.set_grad_enabled(grad):
                        router._score_and_select_experts(
                            gate_TE,
                            expert_bias_E=None,
                            padding_mask_T=padding_mask_T if padded else None,
                            input_ids_T=torch.randint(0, 100, (num_tokens,)),
                        )
        # 2 layers x (4 train kinds x 2 sizes + 2 eval kinds x 3 sizes) = 28.
        # A new guard axis fails here.
        assert len(graphs) <= 28
    finally:
        LocalCompileConfig(regions=[]).apply()
        torch._dynamo.reset()
