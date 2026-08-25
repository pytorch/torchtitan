# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from torchtitan.config.configs import ParallelismConfig, TrainingConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.pipeline_parallel import (
    _build_decoder_stage_io,
    _generate_llm_fqn_per_model_part,
    _static_stage_metadata,
    _unsupported_split_reason,
)
from torchtitan.distributed.utils import pp_backend_is_fake
from torchtitan.models.llama3 import model_registry

WORLD_SIZE = 8
PP_DEGREE = 2
NUM_TOKENS = 64
# Odd on purpose: TP=2 splits it into a 1025 and a 1024 shard.
UNEVEN_VOCAB_SIZE = 2049


def _build_stage_io(
    model_config,
    *,
    spmd_backend: str = "partial_dtensor",
    tp: int = 1,
    cp: int = 1,
    enable_sp: bool = True,
    lm_head_in_loss: bool = True,
):
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=WORLD_SIZE // (PP_DEGREE * cp * tp),
        cp=cp,
        tp=tp,
        pp=PP_DEGREE,
        ep=1,
        world_size=WORLD_SIZE,
        spmd_backend=spmd_backend,
    )
    return _build_decoder_stage_io(
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(enable_sequence_parallel=enable_sp),
        training=TrainingConfig(num_tokens_per_microbatch_per_dp_rank=NUM_TOKENS),
        model_config=model_config,
        lm_head_in_loss=lm_head_in_loss,
    )


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
class TestDecoderStageIO(unittest.TestCase):
    """Static PP stage metadata derived from a decoder config."""

    def setUp(self):
        dist.init_process_group("fake", rank=0, world_size=WORLD_SIZE)
        self.model_config = model_registry("debugmodel").model

    def tearDown(self):
        dist.destroy_process_group()

    def _stage_io(self, *, vocab_size: int | None = None, **kwargs):
        model_config = self.model_config
        if vocab_size is not None:
            model_config = dataclasses.replace(model_config, vocab_size=vocab_size)
        return _build_stage_io(model_config, **kwargs)

    def _assert_dtensor(self, tensor, *, axes, placements, local_shape):
        self.assertIsInstance(tensor, DTensor)
        self.assertEqual(tuple(tensor.device_mesh.mesh_dim_names), axes)
        self.assertEqual(tensor.placements, placements)
        self.assertEqual(tensor.to_local().shape, torch.Size(local_shape))

    def test_no_tensor_parallel(self):
        stage_io = self._stage_io()
        dim = self.model_config.dim
        for tensor in (stage_io.root_input, stage_io.hidden, stage_io.final_output):
            self.assertNotIsInstance(tensor, DTensor)
        self.assertEqual(stage_io.root_input.shape, torch.Size([NUM_TOKENS]))
        self.assertEqual(stage_io.root_input.dtype, torch.int64)
        self.assertFalse(stage_io.root_input.requires_grad)
        self.assertEqual(stage_io.hidden.shape, torch.Size([NUM_TOKENS, dim]))
        self.assertEqual(stage_io.hidden.dtype, torch.bfloat16)
        self.assertTrue(stage_io.hidden.requires_grad)

    def test_sequence_parallel_shards_hidden_states(self):
        stage_io = self._stage_io(tp=2)
        dim = self.model_config.dim
        # Token ids reach stage 0 as a plain tensor.
        self.assertNotIsInstance(stage_io.root_input, DTensor)
        self._assert_dtensor(
            stage_io.hidden,
            axes=("tp",),
            placements=(Shard(0),),
            local_shape=[NUM_TOKENS // 2, dim],
        )
        # The norm in front of the lm_head replicates the sequence again.
        self._assert_dtensor(
            stage_io.final_output,
            axes=("tp",),
            placements=(Replicate(),),
            local_shape=[NUM_TOKENS, dim],
        )

    def test_tensor_parallel_without_sequence_parallel(self):
        stage_io = self._stage_io(tp=2, enable_sp=False)
        self._assert_dtensor(
            stage_io.hidden,
            axes=("tp",),
            placements=(Replicate(),),
            local_shape=[NUM_TOKENS, self.model_config.dim],
        )

    def test_context_parallel_shards_the_microbatch(self):
        stage_io = self._stage_io(cp=2)
        self.assertEqual(stage_io.root_input.shape, torch.Size([NUM_TOKENS // 2]))
        self.assertEqual(
            stage_io.hidden.shape,
            torch.Size([NUM_TOKENS // 2, self.model_config.dim]),
        )

    def test_logits_when_loss_does_not_own_lm_head(self):
        stage_io = self._stage_io(tp=2, lm_head_in_loss=False)
        self._assert_dtensor(
            stage_io.final_output,
            axes=("tp",),
            placements=(Shard(1),),
            local_shape=[NUM_TOKENS, self.model_config.vocab_size // 2],
        )

    def test_uneven_vocabulary_keeps_the_true_extent(self):
        stage_io = self._stage_io(
            tp=2, lm_head_in_loss=False, vocab_size=UNEVEN_VOCAB_SIZE
        )
        self._assert_dtensor(
            stage_io.final_output,
            axes=("tp",),
            placements=(Shard(1),),
            local_shape=[NUM_TOKENS, 1025],
        )
        self.assertEqual(
            stage_io.final_output.shape,
            torch.Size([NUM_TOKENS, UNEVEN_VOCAB_SIZE]),
        )

    def test_spmd_types_keeps_plain_tensors(self):
        stage_io = self._stage_io(spmd_backend="spmd_types", tp=2)
        self.assertNotIsInstance(stage_io.hidden, DTensor)
        self.assertEqual(
            stage_io.hidden.shape,
            torch.Size([NUM_TOKENS // 2, self.model_config.dim]),
        )

    def test_stage_metadata_gradient_entries(self):
        stage_io = self._stage_io()
        num_stages = 3
        first, middle, last = (
            _static_stage_metadata(stage_io, i, num_stages) for i in range(num_stages)
        )

        self.assertIs(first["input_args"][0], stage_io.root_input)
        self.assertIs(first["output_args"][0], stage_io.hidden)
        self.assertEqual(first["input_grads"], (None,))

        self.assertIs(middle["input_args"][0], stage_io.hidden)
        self.assertIs(middle["output_args"][0], stage_io.hidden)
        for grads in (middle["input_grads"], middle["output_grads"]):
            self.assertEqual(grads[0].shape, stage_io.hidden.shape)
            self.assertFalse(grads[0].requires_grad)

        self.assertIs(last["output_args"][0], stage_io.final_output)
        self.assertEqual(last["output_grads"], (None,))


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
class TestUnevenVocabOnSecondTpRank(unittest.TestCase):
    """Uneven vocabulary metadata on the TP rank with the short shard."""

    def setUp(self):
        dist.init_process_group("fake", rank=1, world_size=WORLD_SIZE)

    def tearDown(self):
        dist.destroy_process_group()

    def test_uneven_vocabulary_keeps_the_true_extent(self):
        model_config = dataclasses.replace(
            model_registry("debugmodel").model,
            vocab_size=UNEVEN_VOCAB_SIZE,
        )
        output = _build_stage_io(model_config, tp=2, lm_head_in_loss=False).final_output
        self.assertEqual(
            output.to_local().shape,
            torch.Size([NUM_TOKENS, 1024]),
        )
        self.assertEqual(output.shape, torch.Size([NUM_TOKENS, UNEVEN_VOCAB_SIZE]))


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
class TestPpBackendIsFake(unittest.TestCase):
    """The backend decides, not the requested comm mode.

    ``init_distributed`` leaves an already-initialized process group alone, so
    a preinitialized group and ``comm.mode`` can disagree.
    """

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _parallel_dims(self, *, pp: int):
        return ParallelDims(
            dp_replicate=1,
            dp_shard=WORLD_SIZE // pp,
            cp=1,
            tp=1,
            pp=pp,
            ep=1,
            world_size=WORLD_SIZE,
            spmd_backend="partial_dtensor",
        )

    def test_preinitialized_fake_group_is_detected(self):
        dist.init_process_group("fake", rank=0, world_size=WORLD_SIZE)
        self.assertTrue(pp_backend_is_fake(self._parallel_dims(pp=PP_DEGREE)))

    def test_real_backend_is_not_fake(self):
        # A real multi-rank group needs real processes, so only the backend
        # name is stood in for here.
        dist.init_process_group("fake", rank=0, world_size=WORLD_SIZE)
        parallel_dims = self._parallel_dims(pp=PP_DEGREE)
        with patch(
            "torchtitan.distributed.utils.dist.get_backend", return_value="gloo"
        ):
            self.assertFalse(pp_backend_is_fake(parallel_dims))

    def test_without_pipeline_parallel_no_pp_mesh_is_touched(self):
        dist.init_process_group("fake", rank=0, world_size=WORLD_SIZE)
        self.assertFalse(pp_backend_is_fake(self._parallel_dims(pp=1)))


class TestUnsupportedSplit(unittest.TestCase):
    """Which splits the single hidden-state description can serve."""

    def test_generated_splits_cut_between_decoder_blocks(self):
        for num_stages in (2, 3, 4):
            split = _generate_llm_fqn_per_model_part(num_stages, 8)
            self.assertIsNone(_unsupported_split_reason(split))

    def test_boundary_after_the_root_norm(self):
        # The norm all-gathers the sequence, so this boundary is replicated.
        split = [["tok_embeddings", "layers.0"], ["layers.1", "norm"], ["lm_head"]]
        self.assertIsNotNone(_unsupported_split_reason(split))

    def test_embedding_only_first_stage(self):
        split = [["tok_embeddings"], ["layers.0", "layers.1", "norm", "lm_head"]]
        self.assertIsNotNone(_unsupported_split_reason(split))


if __name__ == "__main__":
    unittest.main()
