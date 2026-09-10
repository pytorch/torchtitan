# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The pipeline split with the vision tower on a stage of its own."""

import unittest
from types import SimpleNamespace

from torchtitan.config import ParallelismConfig
from torchtitan.models.kimi_k3.parallelize import (
    _kimi_k3_pipeline_split,
    _kimi_k3_vit_dep_split,
)


def _model(tower: bool):
    names = [
        "tok_embeddings",
        "layers",
        "norm",
        "lm_head",
        "output_res_proj",
        "output_res_norm",
    ]
    m = SimpleNamespace(**{n: object() for n in names})
    m.vision_encoder = object() if tower else None
    return m


def _split(tower: bool, schedule: str, num_layers: int, *, vit_dep: bool):
    model = _model(tower)
    parallelism = ParallelismConfig(
        pipeline_parallel_degree=2, pipeline_parallel_schedule=schedule
    )
    kwargs = dict(
        parallel_dims=SimpleNamespace(pp=2),
        parallelism=parallelism,
        model_config=SimpleNamespace(layers=list(range(num_layers))),
    )
    if vit_dep:
        return _kimi_k3_vit_dep_split(model, **kwargs)
    return _kimi_k3_pipeline_split(model, **kwargs)


class TestVitDepSplit(unittest.TestCase):
    def test_single_schedule_puts_the_tower_and_embedding_first(self):
        fqns, parallelism = _split(True, "1F1B", 6, vit_dep=True)
        self.assertEqual(fqns[0], ["tok_embeddings", "vision_encoder"])
        self.assertEqual(len(fqns), 2)
        self.assertEqual(
            [n for n in fqns[1] if n.startswith("layers.")],
            [f"layers.{i}" for i in range(6)],
        )
        self.assertIn("lm_head", fqns[1])
        self.assertIn("output_res_proj", fqns[1])
        self.assertNotIn("tok_embeddings", fqns[1])
        self.assertEqual(parallelism.module_fqns_per_model_part, fqns)

    def test_looped_schedule_keeps_the_stage_count(self):
        fqns, _ = _split(True, "Interleaved1F1B", 7, vit_dep=True)
        self.assertEqual(len(fqns), 4)
        self.assertEqual(fqns[0], ["tok_embeddings", "vision_encoder"])
        layers = [n for stage in fqns[1:] for n in stage if n.startswith("layers.")]
        self.assertEqual(layers, [f"layers.{i}" for i in range(7)])
        self.assertTrue(all("vision_encoder" not in stage for stage in fqns[1:]))

    def test_without_dep_the_tower_rides_with_the_embedding(self):
        fqns, _ = _split(True, "1F1B", 6, vit_dep=False)
        self.assertIn("vision_encoder", fqns[0])
        self.assertIn("tok_embeddings", fqns[0])
        self.assertTrue(any(n.startswith("layers.") for n in fqns[0]))

    def test_dep_refuses_a_text_model(self):
        with self.assertRaises(ValueError):
            _split(False, "1F1B", 6, vit_dep=True)


if __name__ == "__main__":
    unittest.main()
