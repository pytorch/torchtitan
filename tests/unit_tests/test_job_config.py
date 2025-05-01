# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from dataclasses import dataclass

import pytest
import tomli_w
from torchtitan.config_manager import ConfigManager, JobConfig


class TestJobConfig(unittest.TestCase):
    def test_command_line_args(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args([])
        assert config.training.steps == 10000

    def test_job_config_file(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--job.config_file",
                "./torchtitan/models/llama3/train_configs/debug_model.toml",
            ]
        )
        assert config.training.steps == 10

    def test_job_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            config_manager = ConfigManager()
            config = config_manager.parse_args(["--job.config_file", "ohno.toml"])

    def test_empty_config_file(self):
        with tempfile.NamedTemporaryFile() as fp:
            config_manager = ConfigManager()
            config = config_manager.parse_args(["--job.config_file", fp.name])
            assert config.job.description

    def test_job_config_file_cmd_overrides(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--job.config_file",
                "./torchtitan/models/llama3/train_configs/debug_model.toml",
                "--job.dump_folder",
                "/tmp/test_tt/",
            ]
        )
        assert config.job.dump_folder == "/tmp/test_tt/"

    def test_parse_pp_split_points(self):
        toml_splits = ["layers.2", "layers.4", "layers.6"]
        cmdline_splits = ["layers.1", "layers.3", "layers.5"]
        # no split points specified
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--job.config_file",
                "./torchtitan/models/llama3/train_configs/debug_model.toml",
            ]
        )
        assert config.parallelism.pipeline_parallel_split_points == []

        # toml has no split points, but cmdline splits are specified
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--job.config_file",
                "./torchtitan/models/llama3/train_configs/debug_model.toml",
                "--parallelism.pipeline_parallel_split_points",
                ",".join(cmdline_splits),
            ]
        )
        assert (
            config.parallelism.pipeline_parallel_split_points == cmdline_splits
        ), config.parallelism.pipeline_parallel_split_points

        # toml has split points, cmdline does not
        with tempfile.NamedTemporaryFile() as fp:
            with open(fp.name, "wb") as f:
                tomli_w.dump(
                    {
                        "parallelism": {
                            "pipeline_parallel_split_points": toml_splits,
                        }
                    },
                    f,
                )
            config_manager = ConfigManager()
            config = config_manager.parse_args(["--job.config_file", fp.name])
            assert (
                config.parallelism.pipeline_parallel_split_points == toml_splits
            ), config.parallelism.pipeline_parallel_split_points

        # toml has split points, cmdline overrides them
        with tempfile.NamedTemporaryFile() as fp:
            with open(fp.name, "wb") as f:
                tomli_w.dump(
                    {
                        "parallelism": {
                            "pipeline_parallel_split_points": toml_splits,
                        }
                    },
                    f,
                )
            config_manager = ConfigManager()
            config = config_manager.parse_args(
                [
                    "--job.config_file",
                    fp.name,
                    "--parallelism.pipeline_parallel_split_points",
                    ",".join(cmdline_splits),
                ]
            )
            assert (
                config.parallelism.pipeline_parallel_split_points == cmdline_splits
            ), config.parallelism.pipeline_parallel_split_points

    def test_parse_exclude_from_loading(self):
        toml_splits = ["optimizer", "dataloader"]
        cmdline_splits = ["optimizer", "lr_scheduler"]
        # no split points specified
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--job.config_file",
                "./torchtitan/models/llama3/train_configs/debug_model.toml",
            ]
        )
        assert config.checkpoint.exclude_from_loading == []

        # toml has no split points, but cmdline splits are specified
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--job.config_file",
                "./torchtitan/models/llama3/train_configs/debug_model.toml",
                "--checkpoint.exclude_from_loading",
                ",".join(cmdline_splits),
            ]
        )
        assert (
            config.checkpoint.exclude_from_loading == cmdline_splits
        ), config.checkpoint.exclude_from_loading

        # toml has split points, cmdline does not
        with tempfile.NamedTemporaryFile() as fp:
            with open(fp.name, "wb") as f:
                tomli_w.dump(
                    {
                        "checkpoint": {
                            "exclude_from_loading": toml_splits,
                        }
                    },
                    f,
                )
            config_manager = ConfigManager()
            config = config_manager.parse_args(["--job.config_file", fp.name])
            assert (
                config.checkpoint.exclude_from_loading == toml_splits
            ), config.checkpoint.exclude_from_loading

        # toml has split points, cmdline overrides them
        with tempfile.NamedTemporaryFile() as fp:
            with open(fp.name, "wb") as f:
                tomli_w.dump(
                    {
                        "checkpoint": {
                            "exclude_from_loading": toml_splits,
                        }
                    },
                    f,
                )
            config_manager = ConfigManager()
            config = config_manager.parse_args(
                [
                    "--job.config_file",
                    fp.name,
                    "--checkpoint.exclude_from_loading",
                    ",".join(cmdline_splits),
                ]
            )
            assert (
                config.checkpoint.exclude_from_loading == cmdline_splits
            ), config.checkpoint.exclude_from_loading

    def test_job_config_model_converters_split(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args([])
        assert config.model.converters == []

        config_manager = ConfigManager()
        config = config_manager.parse_args(["--model.converters", "float8,mxfp"])
        assert config.model.converters == ["float8", "mxfp"]

    def test_print_help(self):
        from tyro.extras import get_parser

        parser = get_parser(ConfigManager)
        parser.print_help()

    def test_extend_jobconfig_directly(self):
        @dataclass
        class CustomCheckpoint:
            convert_path: str = "/custom/path"
            fake_model: bool = True

        @dataclass
        class CustomJobConfig:
            checkpoint: CustomCheckpoint

        MergedJobConfig = ConfigManager._merge_configs(JobConfig, CustomJobConfig)

        cli_args = [
            "--checkpoint.convert_path=/override/path",
            "--checkpoint.fake_model",
        ]

        config_manager = ConfigManager(config_cls=MergedJobConfig)
        config = config_manager.parse_args(cli_args)

        assert config.checkpoint.convert_path == "/override/path"
        assert config.checkpoint.fake_model is True
        assert hasattr(config, "model")

    def test_custom_parser(self):
        path = "tests.assets.extend_jobconfig_example"

        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                f"--experimental.custom_args_module={path}",
                "--custom_args.how-is-your-day",
                "bad",
                "--model.converters",
                "float8,mxfp",
            ]
        )
        assert config.custom_args.how_is_your_day == "bad"
        assert config.model.converters == ["float8", "mxfp"]
        result = config.to_dict()
        assert isinstance(result, dict)

        # There will be a SystemExit raised by ArgumentParser with exist status 2.
        with self.assertRaisesRegex(SystemExit, "2"):
            config = config_manager.parse_args(
                [
                    f"--experimental.custom_args_module={path}",
                    "--custom_args.how-is-your-day",
                    "bad",
                    "--model.converters",
                    "float8,mxfp",
                    "--abcde",
                ]
            )

        with tempfile.NamedTemporaryFile(mode="w+b", delete=True) as fp:
            tomli_w.dump(
                {
                    "experimental": {
                        "custom_args_module": path,
                    }
                },
                fp,
            )
            fp.flush()

            config_manager = ConfigManager()
            config = config_manager.parse_args(
                [
                    f"--job.config_file={fp.name}",
                    f"--experimental.custom_args_module={path}",
                    "--custom_args.how-is-your-day",
                    "bad",
                    "--model.converters",
                    "float8,mxfp",
                ]
            )
            assert config.custom_args.how_is_your_day == "bad"
            assert config.model.converters == ["float8", "mxfp"]
            result = config.to_dict()
            assert isinstance(result, dict)

        with tempfile.NamedTemporaryFile(mode="w+b", delete=True) as fp:
            tomli_w.dump(
                {
                    "experimental": {
                        "custom_args_module": path,
                    },
                    "custom_args": {"how_is_your_day": "really good"},
                    "model": {"converters": ["float8", "mxfp"]},
                },
                fp,
            )
            fp.flush()

            config_manager = ConfigManager()
            config = config_manager.parse_args(
                [
                    f"--job.config_file={fp.name}",
                ]
            )

            assert config.custom_args.how_is_your_day == "really good"
            assert config.model.converters == ["float8", "mxfp"]
            result = config.to_dict()
            assert isinstance(result, dict)


if __name__ == "__main__":
    unittest.main()
