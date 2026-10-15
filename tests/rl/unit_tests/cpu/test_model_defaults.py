# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import pkgutil

import pytest

import torchtitan_recipes.rl
import torchtitan_recipes.tests.rl
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
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
_BF16_HEAD_RECIPES = {
    "rl_grpo_muse_glimmer_30b_search_r1",
    "rl_grpo_muse_glimmer_30b_terminal_bench",
}


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


def test_build_rl_model_configs_freeze_moe_routing_false_trains_moe_router() -> None:
    # DeepSeek-V3 builds a load-balancing aux loss on every MoE router.
    for model in build_rl_model_configs(
        build_deepseek_v3_model_config,
        "debugmodel",
        trainer_parallelism=ParallelismConfig(),
        generator_parallelism=InferenceParallelismConfig(),
        freeze_moe_routing=False,
    ):
        moe_configs = [moe for _, moe, _, _ in model.traverse(MoE.Config)]
        assert moe_configs
        assert not any(moe.freeze_expert_bias for moe in moe_configs)
        assert not any(moe.router.freeze_gate for moe in moe_configs)
        assert all(moe.router.aux_loss is not None for moe in moe_configs)
