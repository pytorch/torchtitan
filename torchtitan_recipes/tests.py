# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Full configurations backing the integration tests.

Each function here is one entry in ``tests/integration_tests``, expressed as
a configuration instead of a base config plus command-line flags. Keeping
them in this package rather than in the test files means CI exercises the
same selection path users do.

Unrelated to the repository's top-level ``tests/`` package, which holds the
test code itself.
"""

from torchtitan.models.llama3.config_registry import llama3_debugmodel
from torchtitan.trainer import Trainer


def llama3_debugmodel_fsdp2_cp2() -> Trainer.Config:
    """Debug model on 4 GPUs: FSDP 2, context parallel 2.

    Derives from ``llama3_debugmodel`` so the two cannot drift, and pins the
    parallelism the run needs instead of leaving it to the command line, so
    the configuration name is enough to reproduce it.
    """
    config = llama3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_degree = 2
    return config
