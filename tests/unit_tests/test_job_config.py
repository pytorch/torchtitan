# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import textwrap
import unittest
from dataclasses import dataclass

import pytest
from torchtitan.config import ConfigManager
from torchtitan.trainer import Trainer


def _write_python_config(fp, code: str):
    """Write a Python config file that defines default_config."""
    fp.write(textwrap.dedent(code).encode())
    fp.flush()


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
                "./torchtitan/models/llama3/train_configs/debug_model.py",
            ]
        )
        assert config.training.steps == 10

    def test_job_file_does_not_exist(self):
        with pytest.raises(Exception):
            config_manager = ConfigManager()
            _ = config_manager.parse_args(["--job.config_file", "ohno.py"])

    def test_empty_config_file(self):
        with tempfile.NamedTemporaryFile(suffix=".py") as fp:
            _write_python_config(
                fp,
                """\
                from torchtitan.trainer import Trainer
                default_config = Trainer.Config()
                """,
            )
            config_manager = ConfigManager()
            config = config_manager.parse_args(["--job.config_file", fp.name])
            assert config.job.description

    def test_job_config_file_cmd_overrides(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--job.config_file",
                "./torchtitan/models/llama3/train_configs/debug_model.py",
                "--job.dump_folder",
                "/tmp/test_tt/",
            ]
        )
        assert config.job.dump_folder == "/tmp/test_tt/"

    def test_parse_module_fqns_per_model_part(self):
        toml_chunks = [
            ["tok_embeddings", "layers.0"],
            ["layers.1", "layers.2"],
            ["layers.3", "norm", "output"],
        ]
        cmdline_chunks = [
            ["tok_embeddings", "layers.0", "layers.1"],
            ["layers.2", "layers.3", "norm", "output"],
        ]

        # no module names specified
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--job.config_file",
                "./torchtitan/models/llama3/train_configs/debug_model.py",
            ]
        )
        assert config.parallelism.module_fqns_per_model_part is None

        # config file has module names, cmdline does not
        with tempfile.NamedTemporaryFile(suffix=".py") as fp:
            _write_python_config(
                fp,
                f"""\
                from torchtitan.config import ParallelismConfig
                from torchtitan.trainer import Trainer
                default_config = Trainer.Config(
                    parallelism=ParallelismConfig(
                        module_fqns_per_model_part={toml_chunks!r},
                    ),
                )
                """,
            )
            config_manager = ConfigManager()
            config = config_manager.parse_args(["--job.config_file", fp.name])
            assert (
                config.parallelism.module_fqns_per_model_part == toml_chunks
            ), config.parallelism.module_fqns_per_model_part

        # test that the field accepts list of lists structure
        with tempfile.NamedTemporaryFile(suffix=".py") as fp:
            _write_python_config(
                fp,
                f"""\
                from torchtitan.config import ParallelismConfig
                from torchtitan.trainer import Trainer
                default_config = Trainer.Config(
                    parallelism=ParallelismConfig(
                        module_fqns_per_model_part={cmdline_chunks!r},
                    ),
                )
                """,
            )
            config_manager = ConfigManager()
            config = config_manager.parse_args(["--job.config_file", fp.name])
            assert (
                config.parallelism.module_fqns_per_model_part == cmdline_chunks
            ), config.parallelism.module_fqns_per_model_part

        # test empty chunks are handled correctly
        empty_chunks = [[], ["tok_embeddings"], []]
        with tempfile.NamedTemporaryFile(suffix=".py") as fp:
            _write_python_config(
                fp,
                f"""\
                from torchtitan.config import ParallelismConfig
                from torchtitan.trainer import Trainer
                default_config = Trainer.Config(
                    parallelism=ParallelismConfig(
                        module_fqns_per_model_part={empty_chunks!r},
                    ),
                )
                """,
            )
            config_manager = ConfigManager()
            config = config_manager.parse_args(["--job.config_file", fp.name])
            assert (
                config.parallelism.module_fqns_per_model_part == empty_chunks
            ), config.parallelism.module_fqns_per_model_part

    def test_parse_exclude_from_loading(self):
        config_splits = ["optimizer", "dataloader"]
        cmdline_splits = ["optimizer", "lr_scheduler"]
        # no split points specified
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--job.config_file",
                "./torchtitan/models/llama3/train_configs/debug_model.py",
            ]
        )
        assert config.checkpoint.exclude_from_loading == []

        # config has no split points, but cmdline splits are specified
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--job.config_file",
                "./torchtitan/models/llama3/train_configs/debug_model.py",
                "--checkpoint.exclude_from_loading",
                ",".join(cmdline_splits),
            ]
        )
        assert (
            config.checkpoint.exclude_from_loading == cmdline_splits
        ), config.checkpoint.exclude_from_loading

        # config has split points, cmdline does not
        with tempfile.NamedTemporaryFile(suffix=".py") as fp:
            _write_python_config(
                fp,
                f"""\
                from torchtitan.components.checkpoint import CheckpointManager
                from torchtitan.trainer import Trainer
                default_config = Trainer.Config(
                    checkpoint=CheckpointManager.Config(
                        exclude_from_loading={config_splits!r},
                    ),
                )
                """,
            )
            config_manager = ConfigManager()
            config = config_manager.parse_args(["--job.config_file", fp.name])
            assert (
                config.checkpoint.exclude_from_loading == config_splits
            ), config.checkpoint.exclude_from_loading

        # config has split points, cmdline overrides them
        with tempfile.NamedTemporaryFile(suffix=".py") as fp:
            _write_python_config(
                fp,
                f"""\
                from torchtitan.components.checkpoint import CheckpointManager
                from torchtitan.trainer import Trainer
                default_config = Trainer.Config(
                    checkpoint=CheckpointManager.Config(
                        exclude_from_loading={config_splits!r},
                    ),
                )
                """,
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

    def test_job_config_model_converters_default(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args([])
        assert config.model_converters.converters == []

    def test_print_help(self):
        from tyro.extras import get_parser

        parser = get_parser(ConfigManager)
        parser.print_help()

    def test_extend_job_config_directly(self):
        @dataclass
        class CustomCheckpoint:
            convert_path: str = "/custom/path"
            fake_model: bool = True

        @dataclass
        class CustomJobConfig:
            checkpoint: CustomCheckpoint

        MergedJobConfig = ConfigManager._merge_configs(Trainer.Config, CustomJobConfig)

        cli_args = [
            "--checkpoint.convert_path=/override/path",
            "--checkpoint.fake_model",
        ]

        config_manager = ConfigManager(config_cls=MergedJobConfig)
        config = config_manager.parse_args(cli_args)

        assert config.checkpoint.convert_path == "/override/path"
        assert config.checkpoint.fake_model is True
        assert hasattr(config, "model")

    def test_custom_config_via_python_file(self):
        """Test that Python config files can use _merge_configs to extend JobConfig."""
        path = "tests.assets.extended_job_config_example"

        with tempfile.NamedTemporaryFile(suffix=".py") as fp:
            _write_python_config(
                fp,
                f"""\
                import importlib
                from torchtitan.config import ConfigManager
                from torchtitan.trainer import Trainer
                JobConfig = Trainer.Config
                custom_module = importlib.import_module("{path}")
                MergedJobConfig = ConfigManager._merge_configs(
                    JobConfig, custom_module.JobConfig
                )
                default_config = MergedJobConfig()
                default_config.custom_config.how_is_your_day = "really good"
                default_config.model_converters.converters = ["float8", "mxfp"]
                """,
            )

            config_manager = ConfigManager()
            config = config_manager.parse_args([f"--job.config_file={fp.name}"])

            assert config.custom_config.how_is_your_day == "really good"
            assert config.model_converters.converters == ["float8", "mxfp"]
            result = config.to_dict()
            assert isinstance(result, dict)

        # Test CLI override of custom config fields
        with tempfile.NamedTemporaryFile(suffix=".py") as fp:
            _write_python_config(
                fp,
                f"""\
                import importlib
                from torchtitan.config import ConfigManager
                from torchtitan.trainer import Trainer
                JobConfig = Trainer.Config
                custom_module = importlib.import_module("{path}")
                MergedJobConfig = ConfigManager._merge_configs(
                    JobConfig, custom_module.JobConfig
                )
                default_config = MergedJobConfig()
                default_config.custom_config.how_is_your_day = "really good"
                """,
            )

            config_manager = ConfigManager()
            config = config_manager.parse_args(
                [
                    f"--job.config_file={fp.name}",
                    "--custom_config.how-is-your-day",
                    "bad",
                ]
            )
            assert config.custom_config.how_is_your_day == "bad"

        # Invalid args should still cause SystemExit
        with tempfile.NamedTemporaryFile(suffix=".py") as fp:
            _write_python_config(
                fp,
                f"""\
                import importlib
                from torchtitan.config import ConfigManager
                from torchtitan.trainer import Trainer
                JobConfig = Trainer.Config
                custom_module = importlib.import_module("{path}")
                MergedJobConfig = ConfigManager._merge_configs(
                    JobConfig, custom_module.JobConfig
                )
                default_config = MergedJobConfig()
                """,
            )
            with self.assertRaisesRegex(SystemExit, "2"):
                config_manager = ConfigManager()
                config_manager.parse_args(
                    [
                        f"--job.config_file={fp.name}",
                        "--abcde",
                    ]
                )

    def test_job_config_invalid_field(self):
        """Python configs with invalid fields raise TypeError at construction."""
        with tempfile.NamedTemporaryFile(suffix=".py") as fp:
            _write_python_config(
                fp,
                """\
                from torchtitan.config import ModelConfig
                from torchtitan.trainer import Trainer
                default_config = Trainer.Config(
                    model=ModelConfig(
                        name="llama3",
                        fake_field=0,
                    ),
                )
                """,
            )
            with self.assertRaisesRegex(TypeError, "fake_field"):
                config_manager = ConfigManager()
                config_manager.parse_args(["--job.config_file", fp.name])


if __name__ == "__main__":
    unittest.main()
