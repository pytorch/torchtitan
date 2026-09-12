# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    llm_split_with_pinned_modules,
)


@dataclass
class _FakeParallelDims:
    pp: int


class _FakeModelWithTower:
    """Only ``vision_encoder`` is present, so only it is pinned."""

    vision_encoder = object()


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

    def test_both_ends_carry_their_pinned_modules(self):
        """Present first-stage modules lead stage 0 and the tail follows the head;
        the config handed back spells the split out and drops the knob it read."""
        fqns, spelled_out = llm_split_with_pinned_modules(
            _FakeModelWithTower(),
            parallel_dims=_FakeParallelDims(pp=2),
            parallelism=ParallelismConfig(
                pipeline_parallel_degree=2,
                pipeline_parallel_layers_per_stage=3,
                pipeline_parallel_schedule="Interleaved1F1B",
            ),
            model_config=_FakeModelConfig(num_layers=10),
            first_stage_module_fqns=("vision_encoder", "absent_module"),
            last_stage_module_fqns=("output_res_proj", "output_res_norm"),
        )
        self.assertEqual(spelled_out.module_fqns_per_model_part, fqns)
        self.assertIsNone(spelled_out.pipeline_parallel_layers_per_stage)
        self.assertEqual(len(fqns), 4)
        self.assertEqual(sum(_layers_per_stage(fqns)), 10)
        self.assertEqual(fqns[0][:2], ["vision_encoder", "tok_embeddings"])
        self.assertEqual(
            fqns[-1][-4:], ["norm", "lm_head", "output_res_proj", "output_res_norm"]
        )

    def test_the_stress_cell_spells_out_core_split(self):
        """pp8 x vp4 over 35 units: 32 uneven stages, both ends pinned."""
        from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_pp8_vp4

        parallelism = kimi_k3_debugmodel_pp8_vp4().parallelism
        fqns = parallelism.module_fqns_per_model_part
        assert fqns is not None
        self.assertIsNone(parallelism.pipeline_parallel_layers_per_stage)
        self.assertEqual(len(fqns), 32)
        self.assertEqual(sum(_layers_per_stage(fqns)), 33)
        self.assertEqual(fqns[0][:2], ["vision_encoder", "tok_embeddings"])
        self.assertEqual(
            fqns[-1], ["norm", "lm_head", "output_res_proj", "output_res_norm"]
        )

    def test_the_shared_debug_model_keeps_its_depth(self):
        """This change deepens no flavor but the pipeline stress cell's."""
        from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
        from torchtitan_recipes.tests.b200 import (
            kimi_k3_debugmodel_pp2_vp2,
            kimi_k3_debugmodel_pp8_vp4,
        )

        self.assertEqual(len(kimi_k3_debugmodel().model_spec.model.layers), 24)
        self.assertEqual(len(kimi_k3_debugmodel_pp2_vp2().model_spec.model.layers), 24)
        self.assertEqual(len(kimi_k3_debugmodel_pp8_vp4().model_spec.model.layers), 33)


if __name__ == "__main__":
    unittest.main()
