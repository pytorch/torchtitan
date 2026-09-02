# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Characterization tests for Qwen3.5 position routing.

Locks the observable contract of the ``preprocess_inputs`` -> ``forward``
pipeline: for a folded (batch-flattened) dataloader batch, the position
tensor that reaches every transformer layer is the ``(num_tokens, 3)``
``mrope_positions`` when present, else the plain 1D ``(num_tokens,)``
``positions`` -- while attention masks are always built from the 1D
``positions``.

These go through the public ``preprocess_inputs`` seam and call
``model(inputs, **batch)``, so they stay valid whether the mrope/positions
resolution lives in ``forward`` or in ``preprocess_inputs``.
"""

import unittest

import torch
from torch import nn


def _build_config_modules():
    try:
        from torchtitan.config import ParallelismConfig
        from torchtitan.distributed.parallel_dims import ParallelDims
        from torchtitan.models.qwen3_5 import model_registry
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest(
            f"Qwen3.5 optional dependency unavailable: {exc.name}"
        ) from exc
    return model_registry, ParallelDims, ParallelismConfig


class _RecordingLayer(nn.Module):
    """Layer stub that records the positions it is handed and passes x through.

    The mrope/positions resolution is independent of the layer internals, so
    stubbing the layers keeps these tests on CPU while still exercising the real
    ``preprocess_inputs`` and ``forward`` glue.
    """

    def __init__(self, sink: dict):
        super().__init__()
        self._sink = sink

    def forward(self, x, attention_masks=None, positions=None):
        self._sink["positions"] = positions
        return x


class TestQwen35MRoPEPositions(unittest.TestCase):
    def _build_stub_model(self):
        model_registry, ParallelDims, ParallelismConfig = _build_config_modules()
        # varlen backend keeps mask construction to pure tensor ops (no flex
        # compile) so the pipeline runs on CPU.
        model = model_registry("debugmodel", attn_backend="varlen").model.build()
        sink: dict = {}
        for key in list(model.layers.keys()):
            model.layers[key] = _RecordingLayer(sink)
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
        )
        # partial_dtensor avoids the spmd_types annotation path, which is
        # orthogonal to position routing.
        parallelism = ParallelismConfig(spmd_backend="partial_dtensor")
        return model, sink, parallel_dims, parallelism

    def _run(self, model, parallel_dims, parallelism, input_dict):
        inputs, _labels, batch = model.preprocess_inputs(
            input_dict,
            parallel_dims=parallel_dims,
            parallelism=parallelism,
        )
        model(inputs, **batch)
        return batch

    def test_text_batch_routes_1d_positions_to_layers(self):
        model, sink, parallel_dims, parallelism = self._build_stub_model()
        # Folded 1D token stream packing docs of length 3, 2, and 5.
        positions = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3, 4], dtype=torch.int32)
        input_dict = {
            "input": torch.randint(0, 100, (10,)),
            "positions": positions.clone(),
            "labels": torch.zeros(10),
        }

        batch = self._run(model, parallel_dims, parallelism, input_dict)

        # No mrope: layers see the plain 1D positions.
        self.assertTrue(torch.equal(sink["positions"], positions))
        # Masks come from the 1D positions.
        torch.testing.assert_close(
            batch["attention_masks"]["deltanet"].cu_seq_q,
            torch.tensor([0, 3, 5, 10], dtype=torch.int32, device=positions.device),
        )

    def test_multimodal_batch_routes_mrope_to_layers(self):
        model, sink, parallel_dims, parallelism = self._build_stub_model()
        positions = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3, 4], dtype=torch.int32)
        # Folded (num_tokens, 3) T/H/W positions whose H/W channels differ from
        # the 1D positions, so routing the wrong tensor to the layers is
        # detectable.
        mrope_positions = torch.stack(
            [positions, positions + 7, positions + 13], dim=-1
        )
        input_dict = {
            "input": torch.randint(0, 100, (10,)),
            "positions": positions.clone(),
            "mrope_positions": mrope_positions.clone(),
            "labels": torch.zeros(10),
        }

        batch = self._run(model, parallel_dims, parallelism, input_dict)

        # mrope present: layers see the (num_tokens, 3) mrope positions, not the
        # 1D positions.
        self.assertEqual(sink["positions"].ndim, 2)
        self.assertEqual(sink["positions"].shape[-1], 3)
        self.assertTrue(torch.equal(sink["positions"], mrope_positions))
        # Masks are still built from the 1D positions, not the mrope positions.
        torch.testing.assert_close(
            batch["attention_masks"]["deltanet"].cu_seq_q,
            torch.tensor([0, 3, 5, 10], dtype=torch.int32, device=positions.device),
        )


if __name__ == "__main__":
    unittest.main()
