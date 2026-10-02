# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.trainer import Trainer

__all__ = [
    "IntegrationTestDefinition",
    "get_importable_config_module",
    "validate_fake_pg_compatibility",
]


@dataclass
class IntegrationTestDefinition:
    """A named integration test backed by one or more complete config recipes."""

    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    disabled: bool = False
    skip_rocm_test: bool = False
    timeout: int | None = None
    golden_numerics_path: str | None = None
    """Run through loss_compare.py using this mode-specific golden path."""
    loss_compare_seed_config: Callable[[], Trainer.Config] | None = None
    """Model-equivalent config for loss_compare.py's single-GPU seed run.

    Use this when the test config applies a parallel transform. The seed run
    disables parallelism but does not undo the transform.
    """
    use_real_pg: bool = False
    """Whether the test requires communication semantics from a real PG."""
    configs: Sequence[Callable[[], Configurable.Config]] = ()
    """One complete configuration per run, selected by module and function."""

    def __post_init__(self):
        if not self.configs:
            raise ValueError(f"{self.test_name} must define at least one config")

    def __repr__(self):
        return self.test_descr


def get_importable_config_module(config_fn: Callable[..., object]) -> str:
    """Return the module path that can import ``config_fn`` in a child process."""
    module_name = config_fn.__module__
    if module_name != "__main__":
        return module_name

    main_module = sys.modules["__main__"]
    module_spec = main_module.__spec__
    if module_spec is None or module_spec.name is None:
        raise ValueError(
            f"Config function {config_fn.__name__!r} was defined in an "
            "unimportable __main__ module"
        )
    return module_spec.name


def validate_fake_pg_compatibility(
    test: IntegrationTestDefinition, config: Trainer.Config
) -> None:
    """Require explicit real-PG marking for incompatible configurations."""
    incompatibilities = []

    if config.checkpointer is not None or config.create_seed_checkpoint:
        incompatibilities.append("checkpointing")
    if config.parallelism.pipeline_parallel_degree > 1:
        incompatibilities.append("pipeline parallelism")
    if incompatibilities and not test.use_real_pg:
        reasons = ", ".join(dict.fromkeys(incompatibilities))
        raise ValueError(
            f"Integration test '{test.test_name}' is incompatible with Fake PG "
            f"because it uses {reasons}; set use_real_pg=True explicitly"
        )
