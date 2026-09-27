# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from torchtitan.config import ConfigLoader, OverrideConfig, ParallelismConfig


def test_loads_builtin_model_recipe() -> None:
    config = ConfigLoader().load(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )

    assert type(config.model).__qualname__ == "Llama3Model.Config"
    assert config.training.steps == 10


def test_loads_fully_qualified_config_registry() -> None:
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan.models.llama3.config_registry",
            "--config",
            "llama3_debugmodel",
        ]
    )

    assert type(config.model).__qualname__ == "Llama3Model.Config"


def test_load_uses_current_sys_argv() -> None:
    argv = ["train.py", "--module", "nonexistent", "--config", "foo"]
    with mock.patch.object(sys, "argv", argv):
        with pytest.raises(ImportError, match="Cannot import config module"):
            ConfigLoader().load()


@pytest.mark.parametrize(
    "args",
    [[], ["--module", "llama3"], ["--config", "llama3_debugmodel"]],
)
def test_module_and_config_are_required(args: list[str]) -> None:
    with pytest.raises(SystemExit):
        ConfigLoader().load(args)


def test_unknown_config_lists_available_functions() -> None:
    with pytest.raises(ValueError, match="Available config functions"):
        ConfigLoader().load(["--module", "llama3", "--config", "not_a_recipe"])


def test_general_config_flags_are_rejected() -> None:
    with pytest.raises(SystemExit):
        ConfigLoader().load(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.steps",
                "5",
            ]
        )


def test_operational_overrides() -> None:
    config = ConfigLoader().load(
        [
            "--module",
            "llama3",
            "--config",
            "llama3_debugmodel",
            "--comm-backend",
            "fake",
            "--output-dir",
            "/tmp/torchtitan-test",
        ]
    )

    assert config.comm.backend == "fake"
    assert config.dump_folder == "/tmp/torchtitan-test"


def test_resume_step_requires_checkpointer() -> None:
    with pytest.raises(ValueError, match="checkpointing"):
        ConfigLoader().load(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--resume-step",
                "-1",
            ]
        )


def test_override_tokens_are_forwarded() -> None:
    config = SimpleNamespace(
        override=OverrideConfig(),
        parallelism=ParallelismConfig(),
        comm=SimpleNamespace(backend="nccl"),
        dump_folder="outputs",
        hf_assets_path=None,
    )
    with mock.patch.object(
        ConfigLoader, "_load_config", return_value=config
    ), mock.patch("torchtitan.config.loader.apply_overrides") as apply_overrides:
        loaded = ConfigLoader().load(
            [
                "--module",
                "custom",
                "--config",
                "recipe",
                "--override",
                "pkg.first",
                "--override",
                'pkg.second={"block_size": 128}',
            ]
        )

    assert loaded is config
    override_config = apply_overrides.call_args.args[0]
    assert override_config.imports == [
        "pkg.first",
        ("pkg.second", {"block_size": 128}),
    ]
