# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest
from dataclasses import dataclass

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
)


@dataclass
class _FakeParallelDims:
    pp: int


@dataclass
class _FakeModelConfig:
    num_layers: int

    @property
    def layers(self) -> list[None]:
        return [None] * self.num_layers


def _layers_per_stage(fqns):
    return [sum(1 for n in stage if n.startswith("layers.")) for stage in fqns]


class TestGenerateLLMFqnPerModelPart(unittest.TestCase):
    def test_default_splits_are_unchanged(self):
        """The shapes the models rely on, with no pinned modules."""
        self.assertEqual(
            _generate_llm_fqn_per_model_part(1, 2),
            [["tok_embeddings", "layers.0", "layers.1", "norm", "lm_head"]],
        )
        self.assertEqual(
            _generate_llm_fqn_per_model_part(2, 3, input_weight=2, output_weight=2),
            [
                ["tok_embeddings", "layers.0", "layers.1"],
                ["layers.2", "norm", "lm_head"],
            ],
        )
        fqns = _generate_llm_fqn_per_model_part(4, 8)
        self.assertEqual(_layers_per_stage(fqns), [2, 3, 2, 1])
        self.assertEqual(fqns[0][0], "tok_embeddings")
        self.assertEqual(fqns[-1][-2:], ["norm", "lm_head"])

    def test_pinned_modules_ride_with_the_head(self):
        """Extra modules land on the last stage and count as no layer."""
        fqns = _generate_llm_fqn_per_model_part(
            2, 3, last_stage_modules=("output_res_proj", "output_res_norm")
        )
        self.assertEqual(
            _layers_per_stage(fqns),
            _layers_per_stage(_generate_llm_fqn_per_model_part(2, 3)),
        )
        self.assertEqual(fqns[0][0], "tok_embeddings")
        self.assertEqual(
            fqns[-1][-4:], ["norm", "lm_head", "output_res_proj", "output_res_norm"]
        )
        single = _generate_llm_fqn_per_model_part(1, 2, last_stage_modules=("tail",))
        self.assertEqual(
            single,
            [["tok_embeddings", "layers.0", "layers.1", "norm", "lm_head", "tail"]],
        )

    def test_a_unit_count_no_shape_divides_still_splits(self):
        """33 layers are 35 units with the embedding and the head: over 32
        stages the split is uneven and the last stage holds the head alone."""
        fqns = _generate_llm_fqn_per_model_part(
            32, 33, last_stage_modules=("output_res_proj", "output_res_norm")
        )
        self.assertEqual(len(fqns), 32)
        self.assertEqual(sum(_layers_per_stage(fqns)), 33)
        self.assertEqual(_layers_per_stage(fqns)[-1], 0)
        self.assertEqual(
            fqns[-1], ["norm", "lm_head", "output_res_proj", "output_res_norm"]
        )

    def test_the_stage_count_can_be_asked_for_per_rank(self):
        """Four stages per rank on eight ranks, which no layers_per_stage
        reaches for this unit count, and both ends carry their pinned modules.
        """
        parallelism = ParallelismConfig(
            pipeline_parallel_degree=8,
            pipeline_parallel_virtual_stages_per_rank=4,
            pipeline_parallel_schedule="Interleaved1F1B",
        )
        reachable = {
            math.ceil(35 / layers_per_stage) for layers_per_stage in range(1, 36)
        }
        self.assertNotIn(32, reachable)

        (
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
        ) = _get_pipeline_metadata(
            _FakeParallelDims(pp=8), parallelism, _FakeModelConfig(num_layers=33)
        )
        self.assertEqual(num_virtual_stages, 32)

        fqns = _generate_llm_fqn_per_model_part(
            num_virtual_stages,
            num_layers,
            input_weight,
            output_weight,
            last_stage_modules=("output_res_proj", "output_res_norm"),
        )
        fqns[0][:0] = ["vision_encoder"]
        self.assertEqual(len(fqns), 32)
        self.assertEqual(sum(_layers_per_stage(fqns)), 33)
        self.assertEqual(fqns[0][:2], ["vision_encoder", "tok_embeddings"])
        self.assertEqual(
            fqns[31], ["norm", "lm_head", "output_res_proj", "output_res_norm"]
        )

    def test_the_stage_count_is_validated(self):
        """More stages than units, and a count the schedule refuses."""
        with self.assertRaisesRegex(ValueError, "more than the"):
            _get_pipeline_metadata(
                _FakeParallelDims(pp=8),
                ParallelismConfig(
                    pipeline_parallel_degree=8,
                    pipeline_parallel_virtual_stages_per_rank=8,
                    pipeline_parallel_schedule="Interleaved1F1B",
                ),
                _FakeModelConfig(num_layers=33),
            )
        with self.assertRaisesRegex(ValueError, "exactly 1 stage per rank"):
            _get_pipeline_metadata(
                _FakeParallelDims(pp=8),
                ParallelismConfig(
                    pipeline_parallel_degree=8,
                    pipeline_parallel_virtual_stages_per_rank=4,
                    pipeline_parallel_schedule="1F1B",
                ),
                _FakeModelConfig(num_layers=33),
            )
        with self.assertRaisesRegex(ValueError, "at least 2 stages per rank"):
            _get_pipeline_metadata(
                _FakeParallelDims(pp=8),
                ParallelismConfig(
                    pipeline_parallel_degree=8,
                    pipeline_parallel_virtual_stages_per_rank=1,
                    pipeline_parallel_schedule="Interleaved1F1B",
                ),
                _FakeModelConfig(num_layers=33),
            )


if __name__ == "__main__":
    unittest.main()
