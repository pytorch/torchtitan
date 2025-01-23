# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import pytest
import tomli_w
from torchtitan.config_manager import JobConfig


class TestJobConfig:
    def test_command_line_args(self):
        config = JobConfig()
        config.parse_args([])
        assert config.training.steps == 10000

    def test_job_config_file(self):
        config = JobConfig()
        config.parse_args(["--job.config_file", "./train_configs/debug_model.toml"])
        assert config.training.steps == 10

    def test_job_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            config = JobConfig()
            config.parse_args(["--job.config_file", "ohno.toml"])

    def test_empty_config_file(self):
        with tempfile.NamedTemporaryFile() as fp:
            config = JobConfig()
            config.parse_args(["--job.config_file", fp.name])
            assert config.job.description

    def test_job_config_file_cmd_overrides(self):
        config = JobConfig()
        config.parse_args(
            [
                "--job.config_file",
                "./train_configs/debug_model.toml",
                "--job.dump_folder",
                "/tmp/test_tt/",
            ]
        )
        assert config.job.dump_folder == "/tmp/test_tt/"

    def test_parse_pp_split_points(self):

        toml_splits = ["layers.2", "layers.4", "layers.6"]
        cmdline_splits = ["layers.1", "layers.3", "layers.5"]
        config = JobConfig()
        config.parse_args(
            [
                "--job.config_file",
                "./train_configs/debug_model.toml",
            ]
        )
        assert config.experimental.pipeline_parallel_split_points == []

        # toml has no split points, but cmdline splits are specified
        config = JobConfig()
        config.parse_args(
            [
                "--job.config_file",
                "./train_configs/debug_model.toml",
                "--experimental.pipeline_parallel_split_points",
                *cmdline_splits,
            ]
        )
        assert (
            config.experimental.pipeline_parallel_split_points == cmdline_splits
        ), config.experimental.pipeline_parallel_split_points

        # toml has split points, cmdline does not
        with tempfile.NamedTemporaryFile() as fp:
            with open(fp.name, "wb") as f:
                tomli_w.dump(
                    {
                        "experimental": {
                            "pipeline_parallel_split_points": toml_splits,
                        }
                    },
                    f,
                )
            config = JobConfig()
            config.parse_args(["--job.config_file", fp.name])
            assert (
                config.experimental.pipeline_parallel_split_points == toml_splits
            ), config.experimental.pipeline_parallel_split_points

        # toml has split points, cmdline overrides them
        with tempfile.NamedTemporaryFile() as fp:
            with open(fp.name, "wb") as f:
                tomli_w.dump(
                    {
                        "experimental": {
                            "pipeline_parallel_split_points": toml_splits,
                        }
                    },
                    f,
                )
            config = JobConfig()
            config.parse_args(
                [
                    "--job.config_file",
                    fp.name,
                    "--experimental.pipeline_parallel_split_points",
                    *cmdline_splits,
                ]
            )
            assert (
                config.experimental.pipeline_parallel_split_points == cmdline_splits
            ), config.experimental.pipeline_parallel_split_points

    def test_print_help(self):
        from tyro.extras import get_parser

        parser = get_parser(JobConfig)
        parser.print_help()
