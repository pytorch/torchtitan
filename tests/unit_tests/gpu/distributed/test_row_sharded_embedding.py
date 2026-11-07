# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from tests.unit_tests.cpu.distributed import (
    test_row_sharded_embedding as row_embedding_tests,
)

pytestmark = pytest.mark.multi_gpu


class TestRowShardedEmbeddingGPU(row_embedding_tests.TestRowShardedEmbedding):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def backend(self) -> str:
        return "nccl"

    @property
    def optimizer_kwargs(self) -> dict:
        return {"fused": True}
