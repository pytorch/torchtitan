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
from torchtitan.config import ConfigManager


def checkpoint_test_config():
    config, remaining = ConfigManager()._load_config(
        [
            "--module",
            os.environ["TORCHTITAN_BASE_MODULE"],
            "--config",
            os.environ["TORCHTITAN_BASE_CONFIG"],
        ]
    )
    assert not remaining

    field_name = "checkpointer" if hasattr(config, "checkpointer") else "checkpoint"
    checkpointer = getattr(config, field_name)
    if checkpointer is None:
        checkpointer = CheckpointManager.Config()
        setattr(config, field_name, checkpointer)

    if hasattr(checkpointer, "enable"):
        checkpointer.enable = True

    export_dtype = os.environ.get("TORCHTITAN_CHECKPOINT_EXPORT_DTYPE")
    if export_dtype is not None:
        checkpointer.export_dtype = export_dtype

    mode = os.environ["TORCHTITAN_CHECKPOINT_MODE"]
    if mode == "seed":
        checkpointer.last_save_model_only = True
        if hasattr(config, "create_seed_checkpoint"):
            config.create_seed_checkpoint = True
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
