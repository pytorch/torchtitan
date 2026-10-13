# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import pytest

from torchtitan.config import Configurable, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    LMHeadFP32OutputTransform,
    LoRATransform,
    ModelConfigTransform,
    TransformRelations,
)
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)
from torchtitan.models.qwen3 import build_model_config
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.trainer import Trainer
from torchtitan_recipes.rl import model_configs


class _Controller(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        trainer: Trainer.Config
        generator: Trainer.Config
        metadata: dict[str, str] = field(default_factory=dict)

        def __post_init__(self):
            if type(self.trainer.model) is not type(self.generator.model):
                raise ValueError("Both roles must use the same model class")


def _apply(config, **kwargs):
    assert hasattr(model_configs, "apply_rl_transforms")
    return model_configs.apply_rl_transforms(config, **kwargs)


@pytest.fixture
def config():
    models = [
        build_model_config(
            "debugmodel_moe", seq_len=2048, parallelism=ParallelismConfig()
        )
        for _ in range(2)
    ]
    return _Controller.Config(
        trainer=Trainer.Config(
            model=models[0],
            training=TrainingConfig(max_context_length=2048, disable_cuda_graphs=True),
        ),
        generator=Trainer.Config(
            model=models[1],
            training=TrainingConfig(max_context_length=2048, disable_cuda_graphs=True),
        ),
    )


class _BeforeHead(ModelConfigTransform):
    @classmethod
    def contribute_relations(cls, relations: TransformRelations) -> None:
        relations.add_precedence(before=cls, after=LMHeadFP32OutputTransform)

    def transform(self, model):
        assert type(model.lm_head) is Linear.Config
        model.lm_head.bias = True
        return model


class _ConflictingHead(ModelConfigTransform):
    @classmethod
    def contribute_relations(cls, relations: TransformRelations) -> None:
        relations.add_conflict(cls, LMHeadFP32OutputTransform)

    def transform(self, model):
        raise AssertionError("A conflicting transform must not run")


class _CompileRegion(ModelConfigTransform):
    def __init__(self, region: str):
        self.region = region

    def transform(self, model):
        model.local_compile_regions.append(self.region)
        return model


def test_builder_leaves_defaults_for_final_application():
    models = model_configs.build_rl_model_configs(
        build_model_config,
        "debugmodel_moe",
        seq_len=2048,
        trainer_parallelism=ParallelismConfig(),
        generator_parallelism=InferenceParallelismConfig(),
    )
    for model in models:
        assert not isinstance(model.lm_head, HiMidLoLinear.Config)
        assert all(
            not moe.router.freeze_gate for _, moe, _, _ in model.traverse(MoE.Config)
        )
    assert models[0] is not models[1]
    assert models[0].layers is not models[1].layers


@pytest.mark.parametrize("fp32_lm_head", [True, False])
@pytest.mark.parametrize("freeze_moe_routing", [True, False])
def test_defaults_apply_to_both_roles(config, fp32_lm_head, freeze_moe_routing):
    result = _apply(
        config,
        fp32_lm_head=fp32_lm_head,
        freeze_moe_routing=freeze_moe_routing,
    )
    for role in (result.trainer, result.generator):
        assert isinstance(role.model.lm_head, HiMidLoLinear.Config) == fp32_lm_head
        for _, moe, _, _ in role.model.traverse(MoE.Config):
            assert moe.freeze_expert_bias == freeze_moe_routing
            assert moe.router.freeze_gate == freeze_moe_routing


def test_role_transform_orders_before_default(config):
    result = _apply(config, trainer_transforms=[_BeforeHead()])
    assert result.trainer.model.lm_head.bias
    assert not result.generator.model.lm_head.bias


@pytest.mark.parametrize("freeze_moe_routing", [True, False])
def test_router_aux_loss_default(config, freeze_moe_routing):
    for role in (config.trainer, config.generator):
        role.model = build_deepseek_v3_model_config(
            "debugmodel", seq_len=2048, parallelism=role.parallelism
        )
        assert all(
            moe.router.aux_loss is not None
            for _, moe, _, _ in role.model.traverse(MoE.Config)
        )
    result = _apply(config, freeze_moe_routing=freeze_moe_routing)
    for role in (result.trainer, result.generator):
        assert all(
            (moe.router.aux_loss is None) == freeze_moe_routing
            for _, moe, _, _ in role.model.traverse(MoE.Config)
        )


@pytest.mark.parametrize("role", ["trainer", "generator"])
def test_role_conflict_with_default_leaves_input_unchanged(config, role):
    with pytest.raises(ValueError, match="cannot be combined"):
        _apply(config, **{f"{role}_transforms": [_ConflictingHead()]})
    for owner in (config.trainer, config.generator):
        assert not isinstance(owner.model.lm_head, HiMidLoLinear.Config)


@pytest.mark.parametrize("shared_model", [False, True])
def test_shared_and_role_transforms_preserve_independent_copies(config, shared_model):
    if shared_model:
        config.generator.model = config.trainer.model
    result = _apply(
        config,
        shared_transforms=[_CompileRegion("shared")],
        trainer_transforms=[_CompileRegion("trainer")],
        generator_transforms=[_CompileRegion("generator")],
    )
    assert result.trainer.model.local_compile_regions[-2:] == ["shared", "trainer"]
    assert result.generator.model.local_compile_regions[-2:] == ["shared", "generator"]
    assert "shared" not in config.trainer.model.local_compile_regions
    assert "shared" not in config.generator.model.local_compile_regions
    assert result.metadata is not config.metadata
    assert result.trainer.model.layers is not result.generator.model.layers


def test_shared_root_replacement_is_retained(config):
    result = _apply(
        config,
        shared_transforms=[LoRATransform(rank=8, target_modules=["wqkv"])],
    )
    assert type(result.trainer.model) is type(result.generator.model)
    assert type(result.trainer.model) is not type(config.trainer.model)


def test_trainer_validation_runs(config):
    config.trainer.parallelism.num_pp_microbatches = 0
    with pytest.raises(ValueError, match="num_pp_microbatches"):
        _apply(config)


def test_generator_validation_runs(config):
    config.generator.parallelism.num_pp_microbatches = 0
    with pytest.raises(ValueError, match="num_pp_microbatches"):
        _apply(config)


def test_controller_validation_runs_after_both_roles(config):
    with pytest.raises(ValueError, match="same model class"):
        _apply(
            config,
            trainer_transforms=[LoRATransform(rank=8, target_modules=["wqkv"])],
        )
