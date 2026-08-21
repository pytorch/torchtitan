# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.pipeline_parallel import pipeline_stage_layer_ids
from torchtitan.models.kimi_k2_7.config_registry import (
    _bucket_layer_ids,
    kimi_k2_5_debugmodel,
)


def _bucket_names(config) -> list[str]:
    kwargs = config.optimizer.optimizer_factory_kwargs_by_name["DistMuon"]
    return [bucket.name for bucket in kwargs["bucket_configs"]]


class TestPipelineStageLayerIds(unittest.TestCase):
    def test_returns_every_layer_in_one_group_without_pp(self):
        parallelism = ParallelismConfig()
        self.assertEqual(
            pipeline_stage_layer_ids(parallelism, 6), ((0, 1, 2, 3, 4, 5),)
        )

    def test_honors_an_explicit_module_split(self):
        parallelism = ParallelismConfig(
            pipeline_parallel_degree=2,
            module_fqns_per_model_part=[
                ["tok_embeddings", "layers.0", "layers.1"],
                ["layers.2", "norm", "lm_head"],
            ],
        )
        self.assertEqual(pipeline_stage_layer_ids(parallelism, 3), ((0, 1), (2,)))


class TestBucketLayerIds(unittest.TestCase):
    def test_layer_zero_is_never_paired(self):
        # Layer 0's dense MLP dwarfs a MoE layer, so it stays in its own bucket.
        self.assertEqual(_bucket_layer_ids(((0, 1, 2, 3),)), ((0,), (1, 2), (3,)))

    def test_pairs_restart_at_each_stage(self):
        # Globally, layers 2 and 3 would pair. Split across stages they must not,
        # because each stage builds its own DistMuon over only its own layers.
        self.assertEqual(
            _bucket_layer_ids(((0, 1, 2), (3, 4, 5))),
            ((0,), (1, 2), (3, 4), (5,)),
        )

    def test_no_bucket_spans_a_stage_boundary(self):
        stage_layer_ids = ((0, 1), (2, 3, 4), (5,))
        stage_of = {
            layer_id: index
            for index, layer_ids in enumerate(stage_layer_ids)
            for layer_id in layer_ids
        }
        for bucket in _bucket_layer_ids(stage_layer_ids):
            self.assertEqual(len({stage_of[layer_id] for layer_id in bucket}), 1)

    def test_skips_stages_that_hold_no_layers(self):
        # A stage may hold only norm and lm_head.
        self.assertEqual(_bucket_layer_ids(((0, 1), ())), ((0,), (1,)))


class TestKimiBucketConfigs(unittest.TestCase):
    def test_bucket_layout_without_pp(self):
        self.assertEqual(
            _bucket_names(kimi_k2_5_debugmodel()),
            [
                "layers.0",
                "layers.1-2",
                "layers.1-2.routed-experts",
                "layers.3-4",
                "layers.3-4.routed-experts",
                "layers.5",
                "layers.5.routed-experts",
            ],
        )

    def test_buckets_realign_when_parallelism_changes_after_build(self):
        # Recipes and the CLI both mutate parallelism after the registry has
        # already built the optimizer config; reconstruction re-runs
        # __post_init__, which is where the buckets are rebuilt.
        config = kimi_k2_5_debugmodel()
        config.parallelism.pipeline_parallel_degree = 2
        config.parallelism.pipeline_parallel_first_stage_less_layers = 2
        config.parallelism.pipeline_parallel_last_stage_less_layers = 2
        config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
        realigned = dataclasses.replace(config)

        stage_layer_ids = pipeline_stage_layer_ids(realigned.parallelism, 6)
        expected = [
            "layers." + "-".join(map(str, layer_ids))
            for layer_ids in _bucket_layer_ids(stage_layer_ids)
        ]
        # Every bucket name must come from the post-override split, and each
        # MoE bucket also contributes a routed-experts bucket.
        self.assertEqual(
            [name for name in _bucket_names(realigned) if "routed" not in name],
            expected,
        )


if __name__ == "__main__":
    unittest.main()
