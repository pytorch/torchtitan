# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch.nn as nn

from torchtitan.experiments.transformers_modeling_backend.hf_sharding import (
    _set_dsa_indexer_sharding,
)


class _AttentionWithIndexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.indexer = nn.Sequential(nn.Linear(4, 4))


class TestDSAIndexerSharding(unittest.TestCase):
    def test_tensor_parallelism_is_rejected(self):
        attention = _AttentionWithIndexer()

        with self.assertRaisesRegex(NotImplementedError, "tensor parallelism"):
            _set_dsa_indexer_sharding(attention, enable_sp=True)

    def test_non_tp_indexer_is_replicated(self):
        attention = _AttentionWithIndexer()
        _set_dsa_indexer_sharding(attention, enable_sp=False)

        for module in attention.indexer.modules():
            self.assertIsNotNone(getattr(module, "_sharding_config", None))


if __name__ == "__main__":
    unittest.main()
