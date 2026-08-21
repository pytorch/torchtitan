# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import _split_module


class _Block(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.w = nn.Linear(dim, dim, bias=False)


class _KeyedModel(nn.Module):
    """Layers in a keyed container, the way the shared decoder holds them."""

    def __init__(self, num_layers: int = 4) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(8, 4)
        self.layers = nn.ModuleDict({str(i): _Block() for i in range(num_layers)})
        self.output = nn.Linear(4, 8, bias=False)


class _PositionalModel(nn.Module):
    """Layers in a container addressed by position, which renumbers on split."""

    def __init__(self, num_layers: int = 4) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([_Block() for _ in range(num_layers)])
        self.output = nn.Linear(4, 8, bias=False)


class TestSplitModulePreservesFQNs(unittest.TestCase):
    def test_keyed_container_keeps_global_names_on_a_later_stage(self):
        chunk = _split_module(_KeyedModel(), ["layers.2", "layers.3", "output"])
        self.assertEqual(
            sorted(name for name, _ in chunk.named_parameters()),
            ["layers.2.w.weight", "layers.3.w.weight", "output.weight"],
        )

    def test_positional_container_is_accepted_when_indices_start_at_zero(self):
        # Rebuilding the list renumbers from 0, which is a no-op for stage 0.
        chunk = _split_module(
            _PositionalModel(), ["tok_embeddings", "layers.0", "layers.1"]
        )
        self.assertEqual(
            sorted(name for name, _ in chunk.named_parameters()),
            ["layers.0.w.weight", "layers.1.w.weight", "tok_embeddings.weight"],
        )

    def test_positional_container_rejects_a_later_stage(self):
        with self.assertRaises(ValueError) as cm:
            _split_module(_PositionalModel(), ["layers.2", "layers.3", "output"])
        message = str(cm.exception)
        # Layers 2 and 3 were silently renumbered to 0 and 1 before this fix.
        self.assertIn("layers.0.w.weight", message)
        self.assertIn("layers.1.w.weight", message)
        self.assertIn("nn.ModuleList", message)

    def test_tied_parameter_reachable_only_from_a_later_stage(self):
        model = _KeyedModel()
        model.output.weight = model.tok_embeddings.weight
        # Deduplicated iteration would report the whole model's shared weight
        # only as tok_embeddings.weight, making this stage look renamed.
        chunk = _split_module(model, ["output"])
        self.assertEqual(
            [name for name, _ in chunk.named_parameters()], ["output.weight"]
        )


if __name__ == "__main__":
    unittest.main()
