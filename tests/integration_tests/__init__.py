# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from torchtitan.trainer import Trainer

__all__ = [
    "OverrideDefinitions",
    "read_golden_spec",
    "validate_fake_pg_compatibility",
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
    golden_numerics_path: str | None = None
    """Run through loss_compare.py using this mode-specific golden path."""
    use_real_pg: bool = False
    """Whether the test requires communication semantics from a real PG."""
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


def read_golden_spec(golden_numerics_path: Path) -> tuple[int, tuple[str, ...]]:
    columns = ("step", "loss")
    steps: list[int] = []
    with golden_numerics_path.open() as golden_file:
        for line in golden_file:
            fields = line.strip().split()
            if not fields:
                continue
            if fields[0] == "#" and len(fields) > 1 and fields[1] == "step":
                columns = tuple(fields[1:])
            elif fields[0] != "#":
                steps.append(int(fields[0]))
    if not steps:
        raise ValueError(f"Numerics golden has no steps: {golden_numerics_path}")
    return max(steps), columns[1:]


def validate_fake_pg_compatibility(
    test: OverrideDefinitions, config: Trainer.Config
) -> None:
    """Require explicit real-PG marking for incompatible configurations."""
    incompatibilities = []

    if config.checkpoint.enable or config.checkpoint.create_seed_checkpoint:
        incompatibilities.append("checkpointing")
    if config.parallelism.pipeline_parallel_degree > 1:
        incompatibilities.append("pipeline parallelism")
    # TODO: FSDP + selective AC backward recompute has a shard/storage shape
    # mismatch with Fake PG under spmd_types. Keep this test on a real PG until
    # that interaction is fixed. Issue #4149.
    if "varlen_attn+per_op_sac" in test.test_name:
        incompatibilities.append("FSDP + selective AC under spmd_types")

    if incompatibilities and not test.use_real_pg:
        reasons = ", ".join(dict.fromkeys(incompatibilities))
        raise ValueError(
            f"Integration test '{test.test_name}' is incompatible with Fake PG "
            f"because it uses {reasons}; set use_real_pg=True explicitly"
        )
