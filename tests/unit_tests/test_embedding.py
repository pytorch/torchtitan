# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from functools import partial

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _MaskPartial
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.distributed.utils import set_spmd_backend
from torchtitan.models.common.embedding import VocabParallelEmbedding


class TestVocabParallelEmbeddingConfig(unittest.TestCase):
    """Tests for the VocabParallelEmbedding class used in the codebase."""

    def test_config_build(self):
        """VocabParallelEmbedding.Config.build() creates a working embedding."""
        config = VocabParallelEmbedding.Config(num_embeddings=100, embedding_dim=32)
        emb = config.build()
        self.assertIsInstance(emb, VocabParallelEmbedding)
        self.assertIsInstance(emb, nn.Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))

    def test_config_build_without_fields_raises(self):
        """VocabParallelEmbedding.Config() raises TypeError when required fields are not provided."""
        with self.assertRaises(TypeError):
            VocabParallelEmbedding.Config()

    def test_init_states(self):
        """init_states re-initializes the weight tensor."""
        config = VocabParallelEmbedding.Config(
            num_embeddings=50,
            embedding_dim=16,
            param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
        )
        emb = config.build()

        nn.init.zeros_(emb.weight)
        self.assertTrue(torch.all(emb.weight == 0))
        emb.init_states()
        self.assertFalse(torch.all(emb.weight == 0))

    def test_custom_init_std(self):
        """VocabParallelEmbedding respects custom mean and std."""
        config = VocabParallelEmbedding.Config(
            num_embeddings=1000,
            embedding_dim=160,
            param_init={"weight": partial(nn.init.normal_, mean=0.1, std=0.02)},
        )
        emb = config.build()

        torch.manual_seed(42)
        emb.init_states()
        # With large amount of samples (160 * 1000) the sample statistics should
        # be close to the requested values. places=3 checks within 0.0005, which
        # is well within statistical tolerance for this sample size.
        self.assertAlmostEqual(emb.weight.mean().item(), 0.1, places=3)
        self.assertAlmostEqual(emb.weight.std().item(), 0.02, places=3)

    def test_config_pre_specified_build(self):
        """VocabParallelEmbedding.Config with both fields pre-specified builds with no kwargs."""
        config = VocabParallelEmbedding.Config(num_embeddings=100, embedding_dim=32)
        emb = config.build()
        self.assertIsInstance(emb, VocabParallelEmbedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))

    def test_config_partial_pre_specified(self):
        """VocabParallelEmbedding.Config with fields specified at construction builds correctly."""
        config = VocabParallelEmbedding.Config(num_embeddings=100, embedding_dim=32)
        emb = config.build()
        self.assertIsInstance(emb, VocabParallelEmbedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))

    def test_config_inheritance_preset(self):
        """Inheriting VocabParallelEmbedding.Config can put fields back in __init__."""

        @dataclass(kw_only=True, slots=True)
        class PresetConfig(VocabParallelEmbedding.Config):
            num_embeddings: int = 100
            embedding_dim: int = 32

        config = PresetConfig()
        emb = config.build()
        self.assertIsInstance(emb, VocabParallelEmbedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))


class TestVocabParallelEmbedding(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_vocab_parallel_embedding_parity(self):
        """Validate local vocab-parallel embedding against DTensor and full embedding."""
        mesh = init_device_mesh(self.device_type, (4,), mesh_dim_names=("tp",))
        tp_group = mesh.get_group("tp")

        for vocab_size in (128, 131):
            for enable_sp in (False, True):
                with self.subTest(vocab_size=vocab_size, enable_sp=enable_sp):
                    torch.manual_seed(42)

                    # Build global inputs, then derive DTensor and local views from them.
                    global_weight = torch.randn(
                        vocab_size,
                        32,
                        device=self.device_type,
                    )
                    global_tokens = torch.randint(
                        0,
                        vocab_size,
                        (3, 16),
                        device=self.device_type,
                    )

                    # Native DTensor embedding is the bitwise oracle for the
                    # local masked implementation, but it uses MaskPartial
                    # internally before the final redistribution.
                    weight_dtensor = distribute_tensor(global_weight, mesh, (Shard(0),))
                    tokens_dtensor = distribute_tensor(
                        global_tokens, mesh, (Replicate(),)
                    )
                    full_output = F.embedding(global_tokens, global_weight)
                    expected_placement = Shard(1) if enable_sp else Replicate()
                    native_dtensor_output = F.embedding(tokens_dtensor, weight_dtensor)
                    self.assertTrue(
                        isinstance(native_dtensor_output.placements[0], _MaskPartial)
                    )
                    dtensor_output = native_dtensor_output.redistribute(
                        placements=(expected_placement,)
                    )

                    # Setup the manual vocab-parallel embedding.
                    embedding = VocabParallelEmbedding(
                        VocabParallelEmbedding.Config(
                            num_embeddings=vocab_size,
                            embedding_dim=32,
                        )
                    ).to(self.device_type)
                    embedding.tp_group = tp_group
                    embedding.weight = nn.Parameter(weight_dtensor.to_local())
                    local_tokens = tokens_dtensor.to_local()

                    # The embedding boundary converts replicated token ids to
                    # R@TP before the local masked embedding region.
                    # The module returns P@TP; the Module sharding wrapper owns
                    # the final P -> S(1)/I redistribution.
                    out_type = spmd.S(1) if enable_sp else spmd.I
                    set_spmd_backend("spmd_types")
                    try:
                        with set_current_spmd_mesh(mesh):
                            with typecheck(strict_mode="strict", local=True):
                                local_tokens = spmd.assert_type(
                                    local_tokens, {tp_group: spmd.R}
                                )
                                spmd.assert_type(local_tokens, {tp_group: spmd.R})
                                embedding._parameters["weight"] = spmd.assert_type(
                                    embedding.weight, {tp_group: spmd.S(0)}
                                )
                                local_partial = embedding(local_tokens)
                                spmd.assert_type(local_partial, {tp_group: spmd.P})
                                local_output = spmd.redistribute(
                                    local_partial,
                                    tp_group,
                                    src=spmd.P,
                                    dst=out_type,
                                )
                    finally:
                        set_spmd_backend("default")

                    # local matches DTensor bitwise and no-parallel embedding
                    self.assertTrue(
                        torch.equal(local_output, dtensor_output.to_local())
                    )
                    if not enable_sp:
                        self.assertTrue(torch.equal(local_output, full_output))


if __name__ == "__main__":
    unittest.main()
