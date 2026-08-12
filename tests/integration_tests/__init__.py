# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "OverrideDefinitions",
    "requires_real_pg",
]


@dataclass
class OverrideDefinitions:
    """
    This class is used to define the override definitions for the integration tests.
    """

    override_args: Sequence[Sequence[str]] = tuple(tuple(" "))
    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    disabled: bool = False
    skip_rocm_test: bool = False
    timeout: int | None = None

    def __repr__(self):
        return self.test_descr


def requires_real_pg(variant: Sequence[str]) -> bool:
    """True when a variant needs a real process group (so it cannot run under a
    fake process group on a single GPU).

    The decision is derived from what the variant's own args declare, so the
    fake-PG eligibility lives with the test body rather than a separate opt-in
    flag. A variant needs a real PG when it exercises something the fake
    backend cannot honor:

    - Checkpointing (``--checkpoint.enable*``): save/load round-trips real
      sharded state through collectives.
    - Validation (``--validator.enable``): the validator drives its own
      forward/collectives that the fake backend does not model.
    - Pipeline parallelism > 1 (``--parallelism.pipeline_parallel_degree N``):
      PP send/recv between stages requires real ranks.

    Everything else (dense SPMD sharding plans, compile, AC, float8, ...) only
    needs the model to build and run a step, which the fake backend supports.
    """
    for arg in variant:
        if "checkpoint.enable" in arg or "validator.enable" in arg:
            return True
        if "pipeline_parallel_degree" in arg:
            # The degree is given either as ``--flag N`` or ``--flag=N``.
            value = arg.split("=")[-1].split()[-1]
            try:
                if int(value) > 1:
                    return True
            except ValueError:
                pass
    return False
