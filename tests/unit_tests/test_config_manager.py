# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import sys
import unittest
from typing import Any
from unittest import mock

import pytest
import tyro
from torch.distributed.tensor import Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torchtitan.config import ConfigManager


class TestConfigManager(unittest.TestCase):
    def test_strided_shard_default_under_any(self):
        @dataclasses.dataclass
        class Config:
            sharding: Any

        registry = tyro.constructors.ConstructorRegistry()
        ConfigManager.register_tyro_rules(registry)
        default = Config(sharding=_StridedShard(0, split_factor=8))

        parsed_default = tyro.cli(Config, args=[], default=default, registry=registry)
        parsed_override = tyro.cli(
            Config,
            args=["--sharding", "1", "4"],
            default=default,
            registry=registry,
        )

        assert parsed_default.sharding == _StridedShard(0, split_factor=8)
        assert parsed_override.sharding == _StridedShard(1, split_factor=4)

    def test_kimi_ep_override_refreshes_dist_muon_expert_layout(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "kimi_k2_7",
                "--config",
                "kimi_k2_5_debugmodel",
                "--parallelism.expert_parallel_degree",
                "2",
            ]
        )
        compute_sharding_by_fqn = config.optimizer.optimizer_factory_kwargs_by_name[
            "DistMuon"
        ]["compute_sharding_by_fqn"]
        expert_layout = next(
            layout
            for fqn, layout in compute_sharding_by_fqn.items()
            if ".moe.routed_experts.inner_experts." in fqn
        )

        assert set(expert_layout.shardings_by_mesh_axis) == {"efsdp", "ep"}
        efsdp_sharding = expert_layout.shardings_by_mesh_axis["efsdp"]
        assert type(efsdp_sharding) is _StridedShard
        assert efsdp_sharding.dim == 0
        assert efsdp_sharding.split_factor == 2

    def test_kimi_ep_replace_does_not_mutate_source_optimizer(self):
        config = ConfigManager().parse_args(
            ["--module", "kimi_k2_7", "--config", "kimi_k2_5_debugmodel"]
        )
        parallelism = dataclasses.replace(
            config.parallelism,
            expert_parallel_degree=2,
        )

        updated = dataclasses.replace(config, parallelism=parallelism)
        source_shardings = config.optimizer.optimizer_factory_kwargs_by_name[
            "DistMuon"
        ]["compute_sharding_by_fqn"]
        updated_shardings = updated.optimizer.optimizer_factory_kwargs_by_name[
            "DistMuon"
        ]["compute_sharding_by_fqn"]
        source_expert_layout = next(
            layout
            for fqn, layout in source_shardings.items()
            if ".moe.routed_experts.inner_experts." in fqn
        )
        updated_expert_layout = next(
            layout
            for fqn, layout in updated_shardings.items()
            if ".moe.routed_experts.inner_experts." in fqn
        )

        assert updated.optimizer is not config.optimizer
        assert dict(source_expert_layout.shardings_by_mesh_axis) == {
            "dp_shard": Shard(0)
        }
        assert set(updated_expert_layout.shardings_by_mesh_axis) == {"efsdp", "ep"}
        assert updated_expert_layout.shardings_by_mesh_axis["efsdp"] == _StridedShard(
            0, split_factor=2
        )

    def test_kimi_ep_override_to_one_uses_dense_expert_layout(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "kimi_k2_7",
                "--config",
                "moonlight_16b_a3b",
                "--parallelism.expert_parallel_degree",
                "1",
            ]
        )
        compute_sharding_by_fqn = config.optimizer.optimizer_factory_kwargs_by_name[
            "DistMuon"
        ]["compute_sharding_by_fqn"]
        expert_layout = next(
            layout
            for fqn, layout in compute_sharding_by_fqn.items()
            if ".moe.routed_experts.inner_experts." in fqn
        )

        assert dict(expert_layout.shardings_by_mesh_axis) == {"dp_shard": Shard(0)}

    def test_kimi_ep_replace_preserves_manual_expert_layout(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "kimi_k2_7",
                "--config",
                "kimi_k2_5_debugmodel",
                "--parallelism.expert_parallel_degree",
                "2",
            ]
        )
        factory_kwargs_by_name = {
            name: dict(factory_kwargs)
            for name, factory_kwargs in (
                config.optimizer.optimizer_factory_kwargs_by_name.items()
            )
        }
        dist_muon_kwargs = factory_kwargs_by_name["DistMuon"]
        compute_sharding_by_fqn = dict(dist_muon_kwargs["compute_sharding_by_fqn"])
        expert_fqns = tuple(
            fqn
            for fqn in compute_sharding_by_fqn
            if ".moe.routed_experts.inner_experts." in fqn
        )
        manual_fqn, auto_fqn = expert_fqns[:2]
        manual_layout = compute_sharding_by_fqn[manual_fqn]
        compute_sharding_by_fqn[manual_fqn] = dataclasses.replace(
            manual_layout,
            shardings_by_mesh_axis={
                **dict(manual_layout.shardings_by_mesh_axis),
                "efsdp": _StridedShard(1, split_factor=2),
            },
        )
        dist_muon_kwargs["compute_sharding_by_fqn"] = compute_sharding_by_fqn
        custom_optimizer = dataclasses.replace(
            config.optimizer,
            optimizer_factory_kwargs_by_name=factory_kwargs_by_name,
        )
        custom_config = dataclasses.replace(config, optimizer=custom_optimizer)
        updated = dataclasses.replace(
            custom_config,
            parallelism=dataclasses.replace(
                custom_config.parallelism,
                expert_parallel_degree=4,
            ),
        )
        updated_shardings = updated.optimizer.optimizer_factory_kwargs_by_name[
            "DistMuon"
        ]["compute_sharding_by_fqn"]

        assert updated_shardings[manual_fqn].shardings_by_mesh_axis[
            "efsdp"
        ] == _StridedShard(1, split_factor=2)
        assert updated_shardings[auto_fqn].shardings_by_mesh_axis[
            "efsdp"
        ] == _StridedShard(0, split_factor=4)

    def test_model_config_args(self):
        """--module and --config together load the correct config."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"
        assert config.training.steps == 10

    def test_model_config_args_equals_form(self):
        """--module=X --config=Y form works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module=llama3", "--config=llama3_debugmodel"]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"

    def test_parse_args_uses_current_sys_argv(self):
        """parse_args() without args reads sys.argv at call time."""
        config_manager = ConfigManager()
        argv = ["train.py", "--module", "nonexistent", "--config", "foo"]
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(ImportError, match="Cannot import module 'nonexistent'"):
                config_manager.parse_args()

    def test_model_without_config_errors(self):
        """--module alone raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--config is required"):
            config_manager.parse_args(["--module", "llama3"])

    def test_config_without_model_errors(self):
        """--config alone raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--module is required"):
            config_manager.parse_args(["--config", "llama3_debugmodel"])

    def test_missing_both_errors(self):
        """No --module or --config raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--module is required"):
            config_manager.parse_args([])

    def test_torchtitan_recipes_package_resolves(self):
        """torchtitan_recipes is importable and its configs load."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "torchtitan_recipes.tests",
                "--config",
                "llama3_debugmodel_fsdp2_cp2",
            ]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"
        assert config.parallelism.context_parallel_degree == 2

    def test_invalid_model_errors(self):
        """--module with unknown module name raises ImportError."""
        config_manager = ConfigManager()
        with pytest.raises(ImportError, match="Cannot import module"):
            config_manager.parse_args(["--module", "nonexistent", "--config", "foo"])

    def test_invalid_config_errors(self):
        """--config with unknown function name lists available functions."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="Available config functions"):
            config_manager.parse_args(["--module", "llama3", "--config", "nonexistent"])

    def test_rl_examples_registered_as_shorthands(self):
        """RL examples are valid --module shorthands (resolved under rl/examples).

        End-to-end resolution + build is covered by the RL integration tests
        (they run ``--module alphabet_sort``); kept out of here since importing an
        example's config_registry pulls in vLLM, which isn't available on CPU.
        """
        from torchtitan.experiments import _supported_experiments

        assert "alphabet_sort" in _supported_experiments
        assert "search_r1" in _supported_experiments

    def test_cli_overrides(self):
        """CLI args override config defaults."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.steps",
                "5",
            ]
        )
        assert config.training.steps == 5

    def test_pipeline_microbatch_size_must_divide_local_batch_size(self):
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="must be evenly divisible"):
            config_manager.parse_args(
                [
                    "--module",
                    "llama3",
                    "--config",
                    "llama3_debugmodel",
                    "--training.local_batch_size",
                    "8",
                    "--parallelism.pipeline_parallel_degree",
                    "2",
                    "--parallelism.pipeline_parallel_microbatch_size",
                    "3",
                ]
            )

    def test_cuda_graphs_reject_pipeline_parallelism(self):
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="do not support pipeline parallelism"):
            config_manager.parse_args(
                [
                    "--module",
                    "llama3",
                    "--config",
                    "llama3_debugmodel",
                    "--parallelism.pipeline_parallel_degree",
                    "2",
                ]
            )

    def test_cuda_graphs_enabled_by_default(self):
        config = ConfigManager().parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert not config.training.disable_cuda_graphs

    def test_cuda_graphs_reject_unsupported_expert_parallelism(self):
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="without CPU synchronization"):
            config_manager.parse_args(
                [
                    "--module",
                    "deepseek_v3",
                    "--config",
                    "deepseek_v3_debugmodel",
                    "--parallelism.expert_parallel_degree",
                    "2",
                ]
            )

    def test_cuda_graphs_allow_non_blocking_hybridep(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "deepseek_v3",
                "--config",
                "deepseek_v3_debugmodel_hybridep",
                "--parallelism.expert_parallel_degree",
                "2",
            ]
        )
        assert not config.training.disable_cuda_graphs

    def test_disable_cuda_graphs_allows_pipeline_parallelism(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.disable_cuda_graphs",
                "--parallelism.pipeline_parallel_degree",
                "2",
            ]
        )
        assert config.training.disable_cuda_graphs

    def test_cuda_graphs_reject_blocking_hybridep(self):
        from torchtitan.models.common.token_dispatcher import HybridEPTokenDispatcher
        from torchtitan.models.deepseek_v3.config_registry import (
            deepseek_v3_debugmodel_hybridep,
        )

        config = deepseek_v3_debugmodel_hybridep()
        dispatcher_configs = list(
            config.model_spec.model.traverse(HybridEPTokenDispatcher.Config)
        )
        assert dispatcher_configs
        for _, dispatcher_config, _, _ in dispatcher_configs:
            dispatcher_config.non_blocking_capacity_factor = None
        config.parallelism.expert_parallel_degree = 2

        with pytest.raises(ValueError, match="non_blocking_capacity_factor"):
            dataclasses.replace(config)

    def test_cli_override_dump_folder(self):
        """CLI args override config defaults for nested fields."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--dump_folder",
                "/tmp/test_tt/",
            ]
        )
        assert config.dump_folder == "/tmp/test_tt/"

    def test_parse_module_fqns_per_model_part(self):
        """module_fqns_per_model_part defaults to None."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.parallelism.module_fqns_per_model_part is None

    def test_parse_exclude_from_loading(self):
        """exclude_from_loading defaults to [] and can be overridden."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.checkpoint.exclude_from_loading == []

        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--checkpoint.exclude_from_loading",
                "optimizer,lr_scheduler",
            ]
        )
        assert config.checkpoint.exclude_from_loading == [
            "optimizer",
            "lr_scheduler",
        ]

    def test_concrete_checkpoint_fields_remain_overridable(self):
        from torchtitan.components.checkpointer import CheckpointManager

        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--checkpoint.async_mode",
                "async",
            ]
        )

        assert isinstance(config.checkpoint, CheckpointManager.Config)
        assert config.checkpoint.async_mode == "async"

    def test_trainer_config_quantization_default(self):
        from torchtitan.components.quantization.utils import has_quantization

        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert not has_quantization(config.model_spec.model)

    # TODO: remove this test when we remove the merge functionality
    def test_extend_trainer_config_directly(self):
        """Test that _merge_configs works to extend config types."""
        from dataclasses import dataclass

        from torchtitan.trainer import Trainer

        @dataclass
        class CustomCheckpoint:
            convert_path: str = "/custom/path"
            fake_model: bool = True

        @dataclass
        class CustomTrainerConfig:
            checkpoint: CustomCheckpoint

        MergedTrainerConfig = ConfigManager._merge_configs(
            Trainer.Config, CustomTrainerConfig
        )

        # Verify the merged type has both base and custom fields
        merged = MergedTrainerConfig()
        assert hasattr(merged, "checkpoint")
        assert hasattr(merged.checkpoint, "convert_path")
        assert merged.checkpoint.convert_path == "/custom/path"
        assert merged.checkpoint.fake_model is True
        assert hasattr(merged, "model_spec")

    def test_flux_config_via_cli(self):
        """Test that --module flux --config flux_debugmodel works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "flux", "--config", "flux_debugmodel"]
        )
        assert config.model_spec.name == "flux"
        assert hasattr(config, "encoder")

    def test_deepseek_config(self):
        """Test that --module deepseek_v3 --config deepseek_v3_debugmodel works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "deepseek_v3", "--config", "deepseek_v3_debugmodel"]
        )
        assert config.model_spec.name == "deepseek_v3"
        assert config.model_spec.flavor == "debugmodel"

    def test_fqn_module_with_config_registry(self):
        """--module torchtitan.models.llama3.config_registry works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "torchtitan.models.llama3.config_registry",
                "--config",
                "llama3_debugmodel",
            ]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"

    def test_fqn_module_without_config_registry(self):
        """--module torchtitan.models.llama3 (auto-appends .config_registry)."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "torchtitan.models.llama3",
                "--config",
                "llama3_debugmodel",
            ]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"

    def test_fqn_module_invalid_errors(self):
        """--module with invalid FQN raises ImportError."""
        config_manager = ConfigManager()
        with pytest.raises(ImportError, match="Cannot import module"):
            config_manager.parse_args(
                [
                    "--module",
                    "torchtitan.models.nonexistent",
                    "--config",
                    "foo",
                ]
            )


if __name__ == "__main__":
    unittest.main()
