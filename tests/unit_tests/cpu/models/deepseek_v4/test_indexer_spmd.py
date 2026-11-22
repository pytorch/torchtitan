# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SPMD contract for the DeepSeek-V4 indexer selection boundary."""

import unittest

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.decoder_sharding import residual_activation_placement
from torchtitan.models.deepseek_v4 import build_model_config


class TestIndexerSPMD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._owns_pg = not dist.is_initialized()
        if cls._owns_pg:
            dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
        cls.ctx = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            world_size=2,
            enable_sequence_parallel=False,
        )
        cls.ctx.build_mesh()

    @classmethod
    def tearDownClass(cls):
        if cls._owns_pg and dist.is_initialized():
            dist.destroy_process_group()

    def test_attention_forward_typechecks_with_tp(self):
        model_config = build_model_config(
            "debugmodel",
            seq_len=32,
            parallelism=ParallelismConfig(enable_sequence_parallel=False),
        )
        layer_config = next(
            layer.attention
            for layer in model_config.layers
            if layer.attention.compress_ratio == 4
        )
        with torch.device("meta"):
            attention = layer_config.build()
        attention.to_empty(device="cpu")
        attention.init_states(buffer_device=torch.device("cpu"))
        attention._parallelize(self.ctx)

        positions = torch.arange(32)
        metadata = layer_config.inner_attention.build_attention_metadata(positions)
        x = torch.randn(32, model_config.dim)
        with self.ctx.activate_spmd(typechecking=True):
            spmd.assert_type(x, residual_activation_placement(enable_sp=False))
            spmd.assert_type(positions, spmd.SpmdType({"tp": spmd.I}))
            attention(
                x,
                attention_metadata=metadata,
                positions=positions,
                aux_loss_denominator=torch.tensor(32.0),
            )


if __name__ == "__main__":
    unittest.main()
