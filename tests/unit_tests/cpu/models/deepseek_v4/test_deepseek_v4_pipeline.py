# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for DeepSeek-V4 pipeline-parallel stage splits."""

import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.pipeline_parallel import (
    _split_module,
    get_module_fqns_per_model_part,
)
from torchtitan_recipes.tests.models import deepseek_v4 as config_registry


class TestDeepSeekV4PipelineSplit(unittest.TestCase):
    def setUp(self) -> None:
        # preprocess_inputs builds a device mesh, so it needs a process group.
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
        self.addCleanup(dist.destroy_process_group)

    @patch("torchtitan.distributed.parallelism_context.device_type", "cpu")
    def test_split_forward_matches_whole_model(self) -> None:
        torch.manual_seed(0)
        model_config = config_registry.deepseek_v4_debugmodel(seq_len=128).model
        model = model_config.build()
        model.init_weights(buffer_device=torch.device("cpu"))
        tokens = torch.randint(0, model_config.vocab_size, (128,))
        inputs, _, model_kwargs, _ = model.preprocess_inputs(
            {
                "input": tokens,
                "labels": tokens,
                "positions": torch.arange(64).repeat(2),  # two documents
            },
            parallelism_context=ParallelismContext(
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                tp=1,
                pp=1,
                ep=1,
                world_size=1,
                enable_sequence_parallel=False,
            ),
            parallelism=ParallelismConfig(),
            global_routing_token_counts=torch.tensor([128.0]),
        )
        # Default PP 2 Interleaved1F1B split, 4 virtual stages:
        # [tok_embeddings, layers.0], [layers.1, layers.2], [layers.3],
        # [norm, lm_head, hc_head]. Layers 0-1 are hash-routed.
        module_fqns_per_stage = get_module_fqns_per_model_part(
            model,
            first_stage_module_fqns=model.pipeline_first_stage_module_fqns,
            last_stage_module_fqns=model.pipeline_last_stage_module_fqns,
            parallelism_context=ParallelismContext(
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                tp=1,
                pp=2,
                ep=1,
                world_size=2,
                enable_sequence_parallel=False,
            ),
            parallelism=ParallelismConfig(
                pipeline_parallel_degree=2,
                pipeline_parallel_schedule="Interleaved1F1B",
            ),
            model_config=model_config,
        )
        self.assertIn("layers.1", module_fqns_per_stage[1])
        self.assertTrue(model.layers["1"].feed_forward.router.hash)

        output = inputs
        for module_fqns in module_fqns_per_stage:
            output = _split_module(model, module_fqns)(output, **model_kwargs)

        # The whole model reads the token ids from ``inputs``, so wrong ids in the
        # kwarg fail here.
        expected = model(inputs, **{**model_kwargs, "input_ids_T": None})
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
