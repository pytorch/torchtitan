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
    "DEFAULT_ENTRYPOINT",
    "IntegrationTestDefinition",
    "SELF_LAUNCHING_ENTRYPOINTS",
    "get_importable_config_module",
    "validate_fake_pg_compatibility",
]


DEFAULT_ENTRYPOINT = "torchtitan.train"
# RL launches its trainer and generator processes itself with Monarch.
SELF_LAUNCHING_ENTRYPOINTS = frozenset({"torchtitan.rl.train"})


@dataclass
class IntegrationTestDefinition:
    """A named integration test backed by one or more complete config recipes."""

    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    filter_rule: str | None = None
    """``"black_list"`` or ``"white_list"``; ``None`` means no filtering."""

    condition: bool | None = None
    """With ``"black_list"``, skip when true; with ``"white_list"``, skip
    when false."""
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
    entrypoints: Sequence[str] = ()
    """Module launched for each config, parallel to ``configs``.

    Empty means ``torchtitan.train`` for every config. Entrypoints in
    ``SELF_LAUNCHING_ENTRYPOINTS`` start their own processes and run with
    ``python -m``; all others run under ``run_train.sh`` (torchrun).
    """

    def __post_init__(self):
        if not self.configs:
            raise ValueError(f"{self.test_name} must define at least one config")
        if self.entrypoints and len(self.entrypoints) != len(self.configs):
            raise ValueError(
                f"{self.test_name} defines {len(self.entrypoints)} entrypoints "
                f"for {len(self.configs)} configs"
            )
        if self.condition is not None and not isinstance(self.condition, bool):
            raise TypeError(
                f"{self.test_name}: condition must be bool, "
                f"got {type(self.condition).__name__}"
            )
        if self.filter_rule is not None and self.filter_rule not in (
            "black_list",
            "white_list",
        ):
            raise ValueError(
                f"{self.test_name}: unknown filter_rule {self.filter_rule!r}"
            )
        if self.filter_rule is None:
            if self.condition is not None:
                raise ValueError(
                    f"{self.test_name}: condition requires a filter_rule"
                )
        elif self.condition is None:
            raise ValueError(
                f"{self.test_name}: {self.filter_rule} requires a condition"
            )

    def should_skip(self) -> bool:
        """Whether the filter rule excludes this test on the current backend."""
        if self.filter_rule is None:
            return False
        matched = bool(self.condition)
        if self.filter_rule == "black_list":
            return matched
        return not matched

    def entrypoint(self, index: int) -> str:
        """Return the module launched for ``configs[index]``."""
        return self.entrypoints[index] if self.entrypoints else DEFAULT_ENTRYPOINT

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
