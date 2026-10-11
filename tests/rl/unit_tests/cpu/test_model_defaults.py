# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import pkgutil
from types import SimpleNamespace

import pytest
import torch

import torchtitan_recipes.rl
import torchtitan_recipes.tests.rl
from torch import nn
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import (
    MoE,
    register_moe_load_balancing_hook,
    register_moe_quantile_balancing_hook,
)
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.tools.utils import has_cuda_capability
from torchtitan_recipes.rl.model_configs import build_rl_model_configs

# Discovered, so a new recipe module is swept too; model_configs.py holds no recipes.
_RECIPE_MODULES = [
    f"{package.__name__}.{module.name}"
    for package in (torchtitan_recipes.rl, torchtitan_recipes.tests.rl)
    for module in pkgutil.iter_modules(package.__path__)
    if module.name != "model_configs"
]
# Reward packages that some recipe modules import (dapo_math, the verifiers recipes).
_OPTIONAL_PACKAGES = {"math_verify", "verifiers"}
# Recipes that need an optional package: torchao nightly (MXFP8) and dist_moe.
_OPTIONAL_PACKAGE_RECIPES = {
    "rl_grpo_qwen3_0_6b_varlen_mxfp8",
    "rl_grpo_moe_debug_dist_moe_tp2_ep4",
}
# SoftCappedLinear head: LMHeadFP32OutputConverter can't convert it (see the recipe's TODO).
_BF16_HEAD_RECIPES = {"rl_grpo_muse_glimmer_30b_search_r1"}


def _build_rl_model_configs(build_model_config, flavor, **kwargs):
    return build_rl_model_configs(
        build_model_config,
        flavor,
        trainer_parallelism=ParallelismConfig(),
        generator_parallelism=InferenceParallelismConfig(),
        **kwargs,
    )


@pytest.mark.parametrize("module_name", _RECIPE_MODULES)
def test_rl_recipes_get_model_defaults(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] not in _OPTIONAL_PACKAGES:
            raise
        pytest.skip(f"{module_name} needs {error.name}")
    recipes = {
        name: fn
        for name, fn in vars(module).items()
        if name.startswith("rl_") and getattr(fn, "__module__", None) == module_name
    }
    assert recipes, module_name

    for name, recipe in recipes.items():
        # MXFP8 raises on GPUs before SM100, even with a torchao nightly.
        if name == "rl_grpo_qwen3_0_6b_varlen_mxfp8" and not has_cuda_capability(10, 0):
            continue
        try:
            config = recipe()
        except ImportError:
            if name not in _OPTIONAL_PACKAGE_RECIPES:
                raise
            continue
        for model in (config.trainer.model, config.generator.model):
            is_fp32_head = isinstance(model.lm_head, HiMidLoLinear.Config)
            assert is_fp32_head == (name not in _BF16_HEAD_RECIPES), name
            moe_configs = [moe for _, moe, _, _ in model.traverse(MoE.Config)]
            assert all(moe.freeze_expert_bias for moe in moe_configs), name
            assert all(moe.router.freeze_gate for moe in moe_configs), name
            assert all(moe.router.aux_loss is None for moe in moe_configs), name


def test_build_rl_model_configs_disables_router_aux_loss() -> None:
    # DeepSeek-V3 builds a load-balancing aux loss on every MoE router.
    for model in _build_rl_model_configs(
        build_deepseek_v3_model_config, "debugmodel", freeze_moe_routing=False
    ):
        routers = [moe.router for _, moe, _, _ in model.traverse(MoE.Config)]
        assert routers and all(router.aux_loss is not None for router in routers)

    for model in _build_rl_model_configs(build_deepseek_v3_model_config, "debugmodel"):
        routers = [moe.router for _, moe, _, _ in model.traverse(MoE.Config)]
        assert routers and all(router.aux_loss is None for router in routers)


# (family, debug flavor, load_balance_coeff): one entry per MoE family in torchtitan/models.
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
def test_build_rl_model_configs_freezes_every_moe_family(
    family: str, flavor: str, load_balance_coeff: float | None, freeze: bool
) -> None:
    build_model_config = importlib.import_module(
        f"torchtitan.models.{family}"
    ).build_model_config
    model, _ = _build_rl_model_configs(
        build_model_config, flavor, freeze_moe_routing=freeze
    )
    # The last MoE layer: DeepSeek-V4's first MoE layers route by token id hash.
    moe_config = [moe for _, moe, _, _ in model.traverse(MoE.Config)][-1]
    moe_config.load_balance_coeff = load_balance_coeff
    # freeze=False keeps the aux loss, which raises without an `aux_loss_denominator`.
    # test_build_rl_model_configs_disables_router_aux_loss covers the aux loss.
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
    block = nn.Module()
    block.moe, block.moe_enabled = moe, True
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
