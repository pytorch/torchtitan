# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from types import SimpleNamespace

import pytest
import torch

from torch import nn
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    FreezeMoERoutingTransform,
    transform_model_config_,
)
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.moe import (
    MoE,
    register_moe_load_balancing_hook,
    register_moe_quantile_balancing_hook,
)
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)


def test_removes_router_aux_loss() -> None:
    # DeepSeek-V3 builds a load-balancing aux loss on every MoE router.
    model = build_deepseek_v3_model_config(
        "debugmodel", parallelism=ParallelismConfig()
    )
    routers = [moe.router for _, moe, _, _ in model.traverse(MoE.Config)]
    assert routers and all(router.aux_loss is not None for router in routers)

    model = transform_model_config_(model, [FreezeMoERoutingTransform()])
    routers = [moe.router for _, moe, _, _ in model.traverse(MoE.Config)]
    assert routers and all(router.aux_loss is None for router in routers)


# (family, debug flavor, load_balance_coeff): every MoE family in torchtitan/models.
# 1e-3 builds a sign-balanced expert bias, e.g. for a checkpoint that ships one.
_MOE_DEBUG_MODELS = [
    ("qwen3", "debugmodel_moe", 1e-3),
    ("qwen3", "debugmodel_moe", None),  # no expert bias
    ("qwen3_5", "debugmodel_moe", 1e-3),  # shared-expert sigmoid gate
    ("qwen3_6", "debugmodel_moe", 1e-3),
    ("qwen3_8", "debugmodel_moe", 1e-3),
    ("kimi_k2_7", "debugmodel", 1e-3),
    ("kimi_k3", "debugmodel", None),  # quantile balancing: its own expert bias
    ("deepseek_v3", "debugmodel", 1e-3),
    ("deepseek_v4", "debugmodel", 1e-3),
    ("gpt_oss", "debugmodel", 1e-3),  # gate bias
]


@pytest.mark.parametrize("freeze", [True, False])
@pytest.mark.parametrize("family,flavor,load_balance_coeff", _MOE_DEBUG_MODELS)
def test_freezes_every_moe_family(
    family: str, flavor: str, load_balance_coeff: float | None, freeze: bool
) -> None:
    build_model_config = importlib.import_module(
        f"torchtitan.models.{family}"
    ).build_model_config
    model = build_model_config(flavor, parallelism=ParallelismConfig())
    if freeze:
        model = transform_model_config_(model, [FreezeMoERoutingTransform()])
    # The last MoE layer: DeepSeek-V4's first MoE layers route by token id hash.
    moe_config = [moe for _, moe, _, _ in model.traverse(MoE.Config)][-1]
    moe_config.load_balance_coeff = load_balance_coeff
    # freeze=False keeps the aux loss, which raises without an `aux_loss_denominator`.
    # test_removes_router_aux_loss covers the aux loss.
    moe_config.router.aux_loss = None
    torch.manual_seed(0)
    moe = moe_config.build()
    moe.init_states()
    params_before = {
        name: param.detach().clone() for name, param in moe.named_parameters()
    }
    expert_bias_before = (
        None if moe.expert_bias_E is None else moe.expert_bias_E.clone()
    )
    # The balancing hooks look for MoE blocks under `layers`.
    block = TransformerBlock()
    block.feed_forward = moe
    model_part = nn.ModuleDict({"layers": nn.ModuleList([block])})
    optimizers = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=1e-2, fused=False)]
    ).build(model_parts=[model_part])
    parallelism = SimpleNamespace(
        ep_enabled=False, tp=1, get_optional_mesh=lambda name: None
    )
    register_moe_load_balancing_hook(optimizers, [model_part], parallelism)
    register_moe_quantile_balancing_hook(optimizers, [model_part], parallelism)

    moe(torch.randn(64, model.dim)).sum().backward()
    optimizers.step()

    if moe.expert_bias_E is None:
        assert load_balance_coeff is None
    else:
        assert torch.equal(moe.expert_bias_E, expert_bias_before) == freeze
    for name, param in moe.named_parameters():
        # router.gate.weight, and router.gate.bias on gpt-oss. Every other parameter
        # trains, including the Qwen3.5 shared-expert gate.
        is_frozen = freeze and name.startswith("router.gate.")
        assert torch.equal(param, params_before[name]) == is_frozen, name
        has_state = any(param in optimizer.state for optimizer in optimizers)
        assert has_state != is_frozen, name
