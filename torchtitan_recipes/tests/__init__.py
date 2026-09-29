# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full configurations backing the integration tests.

Each function here is one run of one entry in ``tests/integration_tests``,
expressed as a full Trainer configuration.

Model registries that need an optional dependency (such as ``torchvision``) or
that are slow to import are imported inside the function that uses them, so
selecting any single configuration stays cheap.
"""

from torchtitan.trainer import Trainer


def _use_spmd_types(config: Trainer.Config, *, typechecking: bool) -> None:
    """Select the SPMD-typed backend for a test configuration.

    Type checking forces activation checkpointing off: it rejects selective AC
    with FlexAttention, which the debug models use. It is also unsupported
    under compile and under pipeline parallelism.
    """
    config.parallelism.spmd_backend = "spmd_types"
    config.debug.spmd_typechecking = typechecking
    if typechecking:
        config.activation_checkpoint = None
