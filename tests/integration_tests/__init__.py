# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torchtitan.trainer import Trainer

__all__ = [
    "OverrideDefinitions",
]


@dataclass
class OverrideDefinitions:
    """
    This class is used to define the override definitions for the integration tests.
    """

    override_args: Sequence[Sequence[str]] = ()
    """One command line per run, appended to ``run_train.sh``.

    The older form, kept for the suites under ``torchtitan/experiments``, which
    still express a run as a base config plus overrides.

    TODO(fegin): remove after those suites move to ``configs``.
    """

    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    disabled: bool = False
    skip_rocm_test: bool = False
    timeout: int | None = None
    configs: Sequence[Callable[[], Trainer.Config]] = ()
    """One configuration per run, selected with ``--module``/``--config``.

    Everything a run needs belongs in its configuration, so an entry sets this
    or ``override_args``, not both.
    """

    def __post_init__(self):
        if not self.configs:
            return
        if not self.override_args:
            self.override_args = tuple(() for _ in self.configs)
        if len(self.override_args) != len(self.configs):
            raise ValueError(
                f"{self.test_name}: {len(self.configs)} configs but "
                f"{len(self.override_args)} override_args; they pair up per run."
            )

    def __repr__(self):
        return self.test_descr
