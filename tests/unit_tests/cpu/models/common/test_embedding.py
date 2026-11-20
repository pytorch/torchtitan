# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import product

import spmd_types as spmd
import torch
import torch.distributed as dist
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

from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    residual_activation_placement,
    token_id_placement,
)
from torchtitan.models.common.embedding import VocabParallelEmbedding


class TestEmbeddingConfig(unittest.TestCase):
    """Tests for VocabParallelEmbedding outside tensor parallelism."""

    def test_config_build(self):
        """VocabParallelEmbedding.Config.build() creates a working embedding."""
        config = VocabParallelEmbedding.Config(num_embeddings=100, embedding_dim=32)
        emb = config.build()
        self.assertIsInstance(emb, VocabParallelEmbedding)
        self.assertIsInstance(emb, nn.Embedding)
        self.assertEqual(emb.weight.shape, torch.Size([100, 32]))
        self.assertIsNone(emb.padding_idx)

    def test_config_padding_idx(self):
        """Native config preserves nn.Embedding initialization and padding gradients."""
        for padding_idx in (0, 3, -1, -8):
            with self.subTest(padding_idx=padding_idx):
                config = VocabParallelEmbedding.Config(
                    num_embeddings=8, embedding_dim=4, padding_idx=padding_idx
                )
                torch.manual_seed(42)
                emb = config.build()
                torch.manual_seed(42)
                reference = nn.Embedding(8, 4, padding_idx=padding_idx)
                self.assertEqual(emb.padding_idx, reference.padding_idx)
                torch.testing.assert_close(emb.weight, reference.weight, atol=0, rtol=0)
                self.assertTrue(torch.all(emb.weight[emb.padding_idx] == 0))

                nn.init.ones_(emb.weight)
                torch.manual_seed(43)
                emb.init_states()
                torch.manual_seed(43)
                reference.reset_parameters()
                torch.testing.assert_close(emb.weight, reference.weight, atol=0, rtol=0)

                # Loaded padding weights are returned, but receive no gradient.
                with torch.no_grad():
                    reference.weight.copy_(torch.arange(1, 33).reshape(8, 4))
                emb.load_state_dict(reference.state_dict())
                tokens = torch.arange(8).repeat(2)
                output = emb(tokens)
                expected = reference(tokens)
                output.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(output, expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    emb.weight.grad, reference.weight.grad, atol=0, rtol=0
                )

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
        """Embedding respects custom mean and std."""
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


class TestEmbedding(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_vocab_parallel_padding_forward_backward(self):
        """Preserve HF padding semantics through conversion and vocab sharding."""
        from torchtitan.experiments.transformers_modeling_backend.module_conversion import (
            convert_hf_to_module,
        )
        from transformers import LlamaConfig, LlamaModel

        # CPU Unit Test CI runs all four ranks with real Gloo collectives.
        mesh = init_device_mesh(self.device_type, (4,), mesh_dim_names=("tp",))
        for vocab_size, padding_idx, cover_full_vocab in product(
            (128, 131),
            (None, 0, 32, 33, 64, -1),
            (False, True),
        ):
            with self.subTest(
                vocab_size=vocab_size,
                padding_idx=padding_idx,
                cover_full_vocab=cover_full_vocab,
            ):
                torch.manual_seed(42)
                hf_model = LlamaModel(
                    LlamaConfig(
                        vocab_size=vocab_size,
                        hidden_size=32,
                        intermediate_size=64,
                        num_hidden_layers=1,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        pad_token_id=padding_idx,
                    )
                ).to(self.device_type)
                reference = deepcopy(hf_model.get_input_embeddings())
                # Nonzero padding weights must still be returned by forward.
                with torch.no_grad():
                    reference.weight.normal_()
                # Include shards with no local tokens.
                tokens = torch.tensor([0, 32], device=self.device_type)
                if cover_full_vocab:
                    tokens = torch.cat(
                        (
                            torch.arange(vocab_size, device=self.device_type).repeat(2),
                            torch.tensor([0, 32, 32, 64], device=self.device_type),
                        )
                    )
                tokens = tokens.unsqueeze(0)

                # HF conversion retains the constructor-normalized padding index.
                convert_hf_to_module(hf_model)
                embedding = hf_model.get_input_embeddings()
                self.assertIsInstance(embedding, VocabParallelEmbedding)
                self.assertEqual(embedding.padding_idx, reference.padding_idx)
                embedding.weight = nn.Parameter(
                    distribute_tensor(
                        reference.weight.detach().clone(), mesh, (Shard(0),)
                    ).to_local()
                )
                expected = reference(tokens)
                expected.sum().backward()
                # The masked vocab-shard lookup, before the reduction that
                # HFVocabParallelEmbedding.forward adds.
                with set_current_spmd_mesh(mesh):
                    partial_output = embedding._lookup_vocab_shard(tokens)
                partial_output.sum().backward()
                output = partial_output.detach().clone()
                dist.all_reduce(output, group=mesh.get_group("tp"))

                expected_grad = distribute_tensor(
                    reference.weight.grad, mesh, (Shard(0),)
                ).to_local()
                expected_weight = distribute_tensor(
                    reference.weight.detach(), mesh, (Shard(0),)
                ).to_local()
                self.assertEqual(output, expected, atol=0, rtol=0)
                self.assertEqual(embedding.weight, expected_weight, atol=0, rtol=0)
                self.assertEqual(embedding.weight.grad, expected_grad, atol=0, rtol=0)

    @with_comms
    def test_vocab_parallel_embedding_parity(self):
        """Match DTensor's vocab-parallel embedding and the unsharded embedding."""
        mesh = init_device_mesh(
            self.device_type, (1, 1, 4), mesh_dim_names=("dp", "cp", "tp")
        )
        tp_mesh = mesh["tp"]
        tp_rank = tp_mesh.get_local_rank()

        for vocab_size, enable_sp in product((128, 131), (False, True)):
            with self.subTest(vocab_size=vocab_size, enable_sp=enable_sp):
                torch.manual_seed(42)
                global_weight = torch.randn(vocab_size, 32, device=self.device_type)
                tokens = torch.randint(0, vocab_size, (16,), device=self.device_type)
                reference = nn.Embedding.from_pretrained(global_weight, freeze=False)
                expected = reference(tokens)
                expected.backward(torch.ones_like(expected))

                # Native DTensor embedding is the bitwise oracle for the local
                # masked lookup plus its reduction to the residual layout.
                weight_dtensor = distribute_tensor(global_weight, tp_mesh, (Shard(0),))
                tokens_dtensor = distribute_tensor(tokens, tp_mesh, (Replicate(),))
                native_dtensor_output = F.embedding(tokens_dtensor, weight_dtensor)
                self.assertIsInstance(native_dtensor_output.placements[0], _MaskPartial)
                dtensor_output = native_dtensor_output.redistribute(
                    placements=(Shard(0) if enable_sp else Replicate(),)
                )

                embedding = (
                    VocabParallelEmbedding.Config(
                        num_embeddings=vocab_size, embedding_dim=32
                    )
                    .build()
                    .to(self.device_type)
                )
                embedding.weight = nn.Parameter(weight_dtensor.to_local())
                set_spmd_meshes(
                    dense_mesh=mesh, sparse_mesh=None, dense_sp_enabled=enable_sp
                )
                with (
                    set_current_spmd_mesh(mesh),
                    typecheck(strict_mode="strict", local=False),
                ):
                    embedding._parameters["weight"] = spmd.assert_type(
                        embedding.weight, dense_param_placement(tp=spmd.S(0))
                    )
                    output = embedding(spmd.assert_type(tokens, token_id_placement()))
                    spmd.assert_type(
                        output, residual_activation_placement(enable_sp=enable_sp)
                    )
                    output.backward(torch.ones_like(output))

                self.assertTrue(torch.equal(output, dtensor_output.to_local()))
                expected_output = expected.chunk(4)[tp_rank] if enable_sp else expected
                self.assertEqual(output, expected_output, atol=0, rtol=0)
                # The rank losses cover every token once: under SP each rank owns
                # a sequence shard, and the P -> S(0) backward all-gathers.
                self.assertEqual(
                    embedding.weight.grad,
                    distribute_tensor(
                        reference.weight.grad, tp_mesh, (Shard(0),)
                    ).to_local(),
                    atol=0,
                    rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
