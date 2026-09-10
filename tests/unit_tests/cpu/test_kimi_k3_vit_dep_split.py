# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The pipeline split with the vision tower on a stage of its own."""

import unittest
from types import SimpleNamespace

from torchtitan.models.kimi_k3.parallelize import kimi_k3_module_fqns_per_model_part


def _model(tower: bool):
    names = ["tok_embeddings", "layers", "norm", "lm_head", "output_res_proj", "output_res_norm"]
    m = SimpleNamespace(**{n: object() for n in names})
    m.vision_encoder = object() if tower else None
    return m


def _parallelism(schedule: str, layers_per_stage=None):
    return SimpleNamespace(
        pipeline_parallel_first_stage_less_layers=1,
        pipeline_parallel_last_stage_less_layers=1,
        pipeline_parallel_layers_per_stage=layers_per_stage,
        pipeline_parallel_schedule=schedule,
    )


class TestVitDepSplit(unittest.TestCase):
    def test_single_schedule_puts_the_tower_and_embedding_first(self):
        cfg = SimpleNamespace(layers=list(range(6)))
        fqns = kimi_k3_module_fqns_per_model_part(
            _model(True), model_config=cfg, parallelism=_parallelism("1F1B"), pp=2, vit_dep=True
        )
        assert fqns is not None
        self.assertEqual(fqns[0], ["tok_embeddings", "vision_encoder"])
        self.assertEqual(len(fqns), 2)
        self.assertEqual([n for n in fqns[1] if n.startswith("layers.")], [f"layers.{i}" for i in range(6)])
        self.assertIn("lm_head", fqns[1])
        self.assertIn("output_res_proj", fqns[1])
        self.assertNotIn("tok_embeddings", fqns[1])

    def test_looped_schedule_keeps_the_stage_count(self):
        cfg = SimpleNamespace(layers=list(range(7)))
        fqns = kimi_k3_module_fqns_per_model_part(
            _model(True), model_config=cfg, parallelism=_parallelism("Interleaved1F1B"), pp=2, vit_dep=True
        )
        assert fqns is not None
        self.assertEqual(len(fqns), 4)
        self.assertEqual(fqns[0], ["tok_embeddings", "vision_encoder"])
        layers = [n for stage in fqns[1:] for n in stage if n.startswith("layers.")]
        self.assertEqual(layers, [f"layers.{i}" for i in range(7)])
        self.assertTrue(all("vision_encoder" not in stage for stage in fqns[1:]))

    def test_without_dep_the_tower_rides_with_the_embedding(self):
        cfg = SimpleNamespace(layers=list(range(6)))
        fqns = kimi_k3_module_fqns_per_model_part(
            _model(True), model_config=cfg, parallelism=_parallelism("1F1B"), pp=2
        )
        assert fqns is not None
        self.assertIn("vision_encoder", fqns[0])
        self.assertIn("tok_embeddings", fqns[0])
        self.assertTrue(any(n.startswith("layers.") for n in fqns[0]))

    def test_dep_refuses_a_text_model(self):
        cfg = SimpleNamespace(layers=list(range(6)))
        with self.assertRaises(ValueError):
            kimi_k3_module_fqns_per_model_part(
                _model(False), model_config=cfg, parallelism=_parallelism("1F1B"), pp=2, vit_dep=True
            )


if __name__ == "__main__":
    unittest.main()
