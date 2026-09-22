# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The pipeline split with the vision tower on a stage of its own."""

import unittest
from types import SimpleNamespace

from torchtitan.config import ParallelismConfig
from torchtitan.models.kimi_k3.pipeline_parallel.layout import (
    infer_block_layout_tables,
    layer_to_stage_from_split,
)
from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep import vit_dep_split


def _model(tower: bool):
    names = ["tok_embeddings", "layers", "norm", "lm_head"]
    model = SimpleNamespace(**{name: object() for name in names})
    model.output_res_proj = object()
    model.output_res_norm = object()
    model.vision_encoder = object() if tower else None
    model.pipeline_last_stage_module_fqns = ("output_res_proj", "output_res_norm")
    return model


def _split(tower: bool, schedule: str, num_layers: int, *, pp: int = 2):
    return vit_dep_split(
        _model(tower),
        parallel_dims=SimpleNamespace(pp=pp),
        parallelism=ParallelismConfig(
            pipeline_parallel_degree=pp, pipeline_parallel_schedule=schedule
        ),
        model_config=SimpleNamespace(layers=list(range(num_layers))),
    )


class TestVitDepSplit(unittest.TestCase):
    def test_a_single_stage_schedule_puts_the_tower_and_embedding_first(self):
        fqns = _split(True, "1F1B", 6)
        self.assertEqual(fqns[0], ["tok_embeddings", "vision_encoder"])
        self.assertEqual(len(fqns), 2)
        self.assertEqual(
            [n for n in fqns[1] if n.startswith("layers.")],
            [f"layers.{i}" for i in range(6)],
        )
        self.assertIn("lm_head", fqns[1])
        self.assertIn("output_res_proj", fqns[1])
        self.assertNotIn("tok_embeddings", fqns[1])

    def test_a_looped_schedule_keeps_the_stage_count(self):
        fqns = _split(True, "Interleaved1F1B", 7)
        self.assertEqual(len(fqns), 4)
        self.assertEqual(fqns[0], ["tok_embeddings", "vision_encoder"])
        layers = [n for stage in fqns[1:] for n in stage if n.startswith("layers.")]
        self.assertEqual(layers, [f"layers.{i}" for i in range(7)])
        self.assertTrue(all("vision_encoder" not in stage for stage in fqns[1:]))

    def test_the_split_the_B200_cell_runs_carries_every_layer_once(self):
        """The cell sets vit_dep and no split, so this is the split it runs on."""
        fqns = _split(True, "Interleaved1F1B", 17, pp=4)
        self.assertEqual(len(fqns), 8)
        layer_to_stage = layer_to_stage_from_split(fqns)
        self.assertEqual(sorted(layer_to_stage), list(range(17)))
        layout = infer_block_layout_tables(
            stage_to_rank={s: s % 4 for s in range(8)},
            n_layers=17,
            layers_per_block=4,
            layer_to_stage=layer_to_stage,
            cache=True,
        )
        self.assertEqual(layout.num_blocks, 5)

    def test_dep_refuses_a_text_model(self):
        with self.assertRaises(ValueError):
            _split(False, "1F1B", 6)

    def test_dep_refuses_a_single_stage_pipeline(self):
        with self.assertRaises(ValueError):
            _split(True, "1F1B", 6, pp=1)


if __name__ == "__main__":
    unittest.main()
