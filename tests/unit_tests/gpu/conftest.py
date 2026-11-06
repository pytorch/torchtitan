# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Iterator

import pytest

# torch.multiprocessing.spawn gives each child process a copy of os.environ.
# If a test module leaves one of these variables set, every later test that
# spawns ranks inherits it. A leaked LOCAL_RANK puts every rank on the same
# GPU, and NCCL fails with "Multiple Ranks are using the same GPU".
# TORCH_DIST_USE_NCCL2 selects which nccl backend the spawned ranks use.
_LAUNCHER_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "GROUP_RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "TORCH_DIST_USE_NCCL2",
    "WORLD_SIZE",
)


@pytest.fixture(autouse=True, scope="module")
def _restore_launcher_variables(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail a test module that changes a launcher variable, and restore it."""
    before = {name: os.environ.get(name) for name in _LAUNCHER_VARIABLES}
    yield
    changed = [name for name, value in before.items() if os.environ.get(name) != value]
    for name in changed:
        value = before[name]
        if value is None:
            del os.environ[name]
        else:
            os.environ[name] = value
    if changed:
        pytest.fail(
            f"{request.node.nodeid} changed {', '.join(changed)} in os.environ. "
            "Later tests that spawn ranks inherit these variables. "
            "Restore them, for example with unittest.mock.patch.dict."
        )
