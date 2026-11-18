# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import pytest

from torchtitan.config import ConfigLoader, Configurable, OverrideConfig
from torchtitan.config.parallelism import ParallelismConfig


def _noop_post_init() -> None:
    return None


class _Checked(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        bad: bool = False

        def __post_init__(self) -> None:
            if self.bad:
                raise ValueError("bad flag")


def recipe_sets_flag_after_init() -> _Checked.Config:
    """Assign after construction, the way seed recipes set their flag."""
    config = _Checked.Config()
    config.bad = True
    return config


def test_loads_exact_recipe_module() -> None:
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_recipes.tests.models.llama3",
            "--config",
            "llama3_debugmodel",
        ]
    )

    assert type(config.model).__qualname__ == "Llama3Model.Config"
    assert config.training.steps == 10


def test_loads_recipe_module() -> None:
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_recipes.tests.models.llama3",
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
    [
        [],
        ["--module", "torchtitan_recipes.tests.models.llama3"],
        ["--config", "llama3_debugmodel"],
    ],
)
def test_module_and_config_are_required(args: list[str]) -> None:
    with pytest.raises(SystemExit):
        ConfigLoader().load(args)


def test_unknown_config_lists_available_functions() -> None:
    with pytest.raises(ValueError, match="Available config functions"):
        ConfigLoader().load(
            [
                "--module",
                "torchtitan_recipes.tests.models.llama3",
                "--config",
                "not_a_recipe",
            ]
        )


def test_general_config_flags_are_rejected() -> None:
    with pytest.raises(SystemExit):
        ConfigLoader().load(
            [
                "--module",
                "torchtitan_recipes.tests.models.llama3",
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
            "torchtitan_recipes.tests.models.llama3",
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


def test_invalid_comm_backend_is_rejected() -> None:
    with pytest.raises(SystemExit):
        ConfigLoader().load(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--comm-backend",
                "definitely_invalid",
            ]
        )


def test_resume_step_rejects_values_below_latest() -> None:
    with pytest.raises(SystemExit):
        ConfigLoader().load(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--resume-step",
                "-2",
            ]
        )


def test_resume_step_requires_checkpointer() -> None:
    with pytest.raises(ValueError, match="checkpointing"):
        ConfigLoader().load(
            [
                "--module",
                "torchtitan_recipes.tests.models.llama3",
                "--config",
                "llama3_debugmodel",
                "--resume-step",
                "-1",
            ]
        )


def test_load_revalidates_recipe_assignments() -> None:
    with pytest.raises(ValueError, match="bad flag"):
        ConfigLoader().load(
            [
                "--module",
                __name__,
                "--config",
                "recipe_sets_flag_after_init",
            ]
        )


def test_override_tokens_are_forwarded() -> None:
    config = SimpleNamespace(
        override=OverrideConfig(),
        parallelism=ParallelismConfig(),
        comm=SimpleNamespace(backend="nccl"),
        dump_folder="outputs",
        hf_assets_path=None,
        __post_init__=_noop_post_init,
    )
    with mock.patch.object(
        ConfigLoader, "_load_config", return_value=config
    ), mock.patch.object(ConfigLoader, "_trainer_config", return_value=config):
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
    assert config.override.imports == [
        "pkg.first",
        ("pkg.second", {"block_size": 128}),
    ]


def test_recipe_and_cli_overrides_are_applied_together() -> None:
    config = SimpleNamespace(
        override=OverrideConfig(imports=["pkg.recipe"]),
        parallelism=ParallelismConfig(),
        comm=SimpleNamespace(backend="nccl"),
        dump_folder="outputs",
        hf_assets_path=None,
        __post_init__=_noop_post_init,
    )
    with mock.patch.object(
        ConfigLoader, "_load_config", return_value=config
    ), mock.patch.object(ConfigLoader, "_trainer_config", return_value=config):
        loaded = ConfigLoader().load(
            [
                "--module",
                "custom",
                "--config",
                "recipe",
                "--override",
                "pkg.cli",
            ]
        )

    assert loaded is config
    assert config.override.imports == ["pkg.recipe", "pkg.cli"]


def test_cli_overrides_extend_the_top_level_rl_config() -> None:
    trainer = SimpleNamespace(comm=SimpleNamespace(backend="nccl"))
    config = SimpleNamespace(
        override=OverrideConfig(imports=["pkg.recipe"]),
        trainer=trainer,
        dump_folder="outputs",
        hf_assets_path=None,
        __post_init__=_noop_post_init,
    )
    with mock.patch.object(
        ConfigLoader, "_load_config", return_value=config
    ), mock.patch.object(ConfigLoader, "_trainer_config", return_value=trainer):
        ConfigLoader().load(
            [
                "--module",
                "custom",
                "--config",
                "recipe",
                "--override",
                "pkg.cli",
            ]
        )

    assert config.override.imports == ["pkg.recipe", "pkg.cli"]
