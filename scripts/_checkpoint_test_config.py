# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Private checkpoint config helper for cross-commit test scripts.

This helper is used by ``scripts/loss_compare.py``,
``scripts/checkpoint_compat_test.py``, and the Transformers modeling backend
``cp_pp_numerical.py`` test. These callers need to enable and configure
checkpointing for arbitrary registry configs without exposing the optional
checkpointer config through the training CLI. The comparison scripts also
check out other revisions while running, so this helper creates a temporary
importable config module that remains available across those worktree changes.
"""

import os
import tempfile
from pathlib import Path


_MODULE_NAME = "_torchtitan_checkpoint_test_config"
_CONFIG_NAME = "checkpoint_test_config"
_MODULE_DIR = tempfile.TemporaryDirectory(prefix="torchtitan_checkpoint_config_")

_MODULE_SOURCE = r"""
import os

from torchtitan.components.checkpointer import CheckpointManager


def checkpoint_test_config():
    try:
        from torchtitan.config import ConfigLoader
    except ImportError:
        from torchtitan.config import ConfigManager

        config, remaining = ConfigManager()._load_config(
            [
                "--module",
                os.environ["TORCHTITAN_BASE_MODULE"],
                "--config",
                os.environ["TORCHTITAN_BASE_CONFIG"],
            ]
        )
        assert not remaining
    else:
        config = ConfigLoader._load_config(
            os.environ["TORCHTITAN_BASE_MODULE"],
            os.environ["TORCHTITAN_BASE_CONFIG"],
        )

    from torchtitan.trainer import Trainer

    if isinstance(config, Trainer.Config):
        trainer_config = config
    else:
        from torchtitan.rl.controller import Controller

        if not isinstance(config, Controller.Config):
            raise TypeError("Expected a Trainer.Config or Controller.Config recipe.")
        trainer_config = config.trainer

    steps = os.environ.get("TORCHTITAN_TEST_STEPS")
    if steps is not None:
        trainer_config.training.steps = int(steps)
        trainer_config.debug.deterministic = True
        trainer_config.debug.seed = 42
        trainer_config.metrics.enable_tensorboard = True
        trainer_config.metrics.log_freq = 1
        trainer_config.metrics.save_tb_folder = os.environ["TORCHTITAN_TEST_TB_FOLDER"]
    total_steps = os.environ.get("TORCHTITAN_TEST_LR_TOTAL_STEPS")
    if total_steps is not None:
        trainer_config.lr_scheduler.total_steps = int(total_steps)

    mode = os.environ.get("TORCHTITAN_CHECKPOINT_MODE")
    if mode is None:
        return config

    field_name = (
        "checkpointer"
        if hasattr(trainer_config, "checkpointer")
        else "checkpoint"
    )
    checkpointer = getattr(trainer_config, field_name)
    if checkpointer is None:
        checkpointer = CheckpointManager.Config()
        setattr(trainer_config, field_name, checkpointer)

    if hasattr(checkpointer, "enable"):
        checkpointer.enable = True

    export_dtype = os.environ.get("TORCHTITAN_CHECKPOINT_EXPORT_DTYPE")
    if export_dtype is not None:
        checkpointer.export_dtype = export_dtype

    if mode == "seed":
        parallelism = trainer_config.parallelism
        parallelism.data_parallel_replicate_degree = 1
        parallelism.data_parallel_shard_degree = 1
        parallelism.context_parallel_degree = 1
        parallelism.tensor_parallel_degree = 1
        parallelism.pipeline_parallel_degree = 1
        parallelism.expert_parallel_degree = 1
        checkpointer.last_save_model_only = True
        if hasattr(trainer_config, "create_seed_checkpoint"):
            trainer_config.create_seed_checkpoint = True
        else:
            checkpointer.create_seed_checkpoint = True
    elif mode == "load":
        checkpointer.load_only = True
        initial_load_path = os.environ.get(
            "TORCHTITAN_CHECKPOINT_INITIAL_LOAD_PATH"
        )
        if initial_load_path is not None:
            checkpointer.initial_load_path = initial_load_path
    elif mode == "resume":
        checkpointer.interval = int(os.environ["TORCHTITAN_CHECKPOINT_INTERVAL"])
        checkpointer.last_save_model_only = False
    else:
        raise ValueError(f"Unknown checkpoint test mode: {mode}")
    return config
"""


def configure_checkpoint(
    env: dict[str, str],
    *,
    module: str,
    config: str,
    mode: str,
    interval: int | None = None,
    initial_load_path: str | None = None,
    export_dtype: str | None = None,
) -> tuple[str, str]:
    """Configure a checkpoint-enabled registry function for a child process."""
    module_dir = Path(_MODULE_DIR.name)
    module_path = module_dir / f"{_MODULE_NAME}.py"
    if not module_path.exists():
        module_path.write_text(_MODULE_SOURCE)

    env["PYTHONPATH"] = os.pathsep.join(
        (str(module_dir), env.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    env["TORCHTITAN_BASE_MODULE"] = module
    env["TORCHTITAN_BASE_CONFIG"] = config
    env["TORCHTITAN_CHECKPOINT_MODE"] = mode
    if interval is not None:
        env["TORCHTITAN_CHECKPOINT_INTERVAL"] = str(interval)
    if initial_load_path is not None:
        env["TORCHTITAN_CHECKPOINT_INITIAL_LOAD_PATH"] = initial_load_path
    if export_dtype is not None:
        env["TORCHTITAN_CHECKPOINT_EXPORT_DTYPE"] = export_dtype
    return _MODULE_NAME, _CONFIG_NAME


def configure_training_run(
    env: dict[str, str],
    *,
    module: str,
    config: str,
    steps: int,
    tb_folder: str,
    checkpoint_mode: str | None = None,
    checkpoint_interval: int | None = None,
    export_dtype: str | None = None,
    total_steps: int | None = None,
) -> tuple[str, str]:
    """Configure deterministic metric collection for a child training run."""
    if checkpoint_mode is None:
        module_dir = Path(_MODULE_DIR.name)
        module_path = module_dir / f"{_MODULE_NAME}.py"
        if not module_path.exists():
            module_path.write_text(_MODULE_SOURCE)
        env["PYTHONPATH"] = os.pathsep.join(
            (str(module_dir), env.get("PYTHONPATH", ""))
        ).rstrip(os.pathsep)
        env["TORCHTITAN_BASE_MODULE"] = module
        env["TORCHTITAN_BASE_CONFIG"] = config
    else:
        module, config = configure_checkpoint(
            env,
            module=module,
            config=config,
            mode=checkpoint_mode,
            interval=checkpoint_interval,
            export_dtype=export_dtype,
        )
        # configure_checkpoint returns the generated recipe. Restore the base
        # recipe environment because this wrapper must load it exactly once.
        env["TORCHTITAN_BASE_MODULE"] = env.get("TORCHTITAN_BASE_MODULE", module)
        env["TORCHTITAN_BASE_CONFIG"] = env.get("TORCHTITAN_BASE_CONFIG", config)
    env["TORCHTITAN_TEST_STEPS"] = str(steps)
    env["TORCHTITAN_TEST_TB_FOLDER"] = tb_folder
    if total_steps is not None:
        env["TORCHTITAN_TEST_LR_TOTAL_STEPS"] = str(total_steps)
    return _MODULE_NAME, _CONFIG_NAME
