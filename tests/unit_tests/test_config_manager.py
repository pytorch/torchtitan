# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
from torchtitan.config import ConfigManager
from torchtitan.trainer import Trainer


class TestConfigManager(unittest.TestCase):
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

    def test_invalid_model_errors(self):
        """--module with unknown module name raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="Unknown module"):
            config_manager.parse_args(["--module", "nonexistent", "--config", "foo"])

    def test_invalid_config_errors(self):
        """--config with unknown function name lists available functions."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="Available config functions"):
            config_manager.parse_args(["--module", "llama3", "--config", "nonexistent"])

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

    def test_trainer_config_model_converters_default(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.model_converters.converters == []

    # TODO: remove this test when we remove the merge functionality
    def test_extend_trainer_config_directly(self):
        """Test that _merge_configs works to extend config types."""
        from dataclasses import dataclass

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


if __name__ == "__main__":
    unittest.main()
