# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Kimi K3 pipeline split, on CPU.

This exists because the split used to return early on a Config-tree model
-- it read the layer count from a flat config's ``num_hidden_layers`` -- and
said nothing when it did. The split then fell back to core's, which left the
AttnRes aggregation modules off the last stage, and the only visible symptom
was a loss that trained to a different place.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch.nn as nn

from torchtitan.models.kimi_k3.parallelize import kimi_k3_module_fqns_per_model_part


class _Model(nn.Module):
    """Enough of the model for the injection to recognise it: the two AttnRes
    tail modules and core's spellings for the embedding and the head."""

    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(4, 4)
        self.layers = nn.ModuleDict({str(i): nn.Linear(4, 4) for i in range(8)})
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 4)
        self.output_res_norm = nn.LayerNorm(4)
        self.output_res_proj = nn.Linear(4, 4)


class _MultimodalModel(_Model):
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Linear(4, 4)


def _split(
    model,
    num_layers: int,
    *,
    layers_per_stage: int | None = None,
    schedule: str = "1F1B",
    pp: int = 2,
):
    parallelism = SimpleNamespace(
        module_fqns_per_model_part=None,
        pipeline_parallel_first_stage_less_layers=1,
        pipeline_parallel_last_stage_less_layers=1,
        pipeline_parallel_layers_per_stage=layers_per_stage,
        pipeline_parallel_schedule=schedule,
    )
    # A Config-tree model carries the layers themselves rather than a count.
    model_config = SimpleNamespace(layers=[object()] * num_layers)
    return kimi_k3_module_fqns_per_model_part(
        model, model_config=model_config, parallelism=parallelism, pp=pp
    )


def _layers_per_stage(fqns):
    return [sum(1 for n in stage if n.startswith("layers.")) for stage in fqns]


class TestKimiK3Split(unittest.TestCase):
    def test_a_layer_count_no_shape_divides_still_splits(self):
        """33 layers are 35 units with the embedding and the head: no pipeline
        shape divides them, and the split still lands on pp x vp stages that
        differ by a layer, where the ceiling rule would refuse the shape."""
        fqns = _split(_Model(), 33, layers_per_stage=4, schedule="Interleaved1F1B")
        assert fqns is not None
        self.assertEqual(len(fqns), 8)
        self.assertEqual(_layers_per_stage(fqns), [4, 5, 5, 4, 4, 4, 4, 3])
        fqns = _split(
            _Model(), 33, layers_per_stage=1, schedule="Interleaved1F1B", pp=8
        )
        assert fqns is not None
        self.assertEqual(len(fqns), 32)
        self.assertEqual(_layers_per_stage(fqns)[-1], 0)
        self.assertIn("lm_head", fqns[-1])
        # A single-stage schedule keeps one stage per rank whatever the knob says.
        fqns = _split(_Model(), 33, layers_per_stage=4, schedule="1F1B")
        assert fqns is not None
        self.assertEqual(len(fqns), 2)

    def test_a_config_tree_model_gets_a_split(self):
        fqns = _split(_Model(), 8)
        self.assertIsNotNone(fqns, "the split returned early and said nothing")
        self.assertEqual(len(fqns), 2)

    def test_the_attn_res_tail_lands_on_the_last_stage(self):
        fqns = _split(_Model(), 8)
        assert fqns is not None
        last = fqns[-1]
        self.assertIn("output_res_proj", last)
        self.assertIn("output_res_norm", last)
        self.assertIn("lm_head", last)

    def test_every_emitted_name_matches_a_child(self):
        """Core sets every non-matching child to None, so an FQN that matches
        nothing is a stage with pieces missing rather than an error."""
        model = _Model()
        children = {name for name, _ in model.named_children()}
        fqns = _split(model, 8)
        assert fqns is not None
        for stage in fqns:
            for fqn in stage:
                root = fqn.split(".", 1)[0]
                self.assertIn(root, children, f"{fqn} matches no child")

    def test_the_vision_tower_gets_a_stage(self):
        """Vision features are spliced into the embeddings, so the tower rides
        with whichever stage kept the embedding. Left unnamed it is None on
        every stage, and the first multimodal batch reports "pixel_values were
        provided without a vision encoder"."""
        fqns = _split(_MultimodalModel(), 8)
        assert fqns is not None
        owner = [s for s in fqns if "vision_encoder" in s]
        self.assertEqual(len(owner), 1, "the tower must land on exactly one stage")
        self.assertIn("tok_embeddings", owner[0])


if __name__ == "__main__":
    unittest.main()
