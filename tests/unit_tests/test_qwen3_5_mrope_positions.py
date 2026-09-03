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
from unittest import mock

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

    def test_multimodal_indices_are_built_before_context_parallel_sharding(self):
        import spmd_types as spmd

        from torchtitan.distributed.parallel_dims import MeshAxisName
        from torchtitan.distributed.spmd_types import _per_axis_types

        model, _sink, _parallel_dims, parallelism = self._build_stub_model()
        tokens_T = torch.tensor([7, 7, 1, 9, 9, 9, 7, 2])
        input_dict = {
            "input": tokens_T,
            "positions": torch.arange(tokens_T.shape[0], dtype=torch.int32),
            "mrope_positions": torch.zeros(tokens_T.shape[0], 3, dtype=torch.int32),
            "labels": torch.zeros(tokens_T.shape[0]),
            "pixel_values": torch.randn(4, 8),
            "grid_thw": torch.tensor([[1, 2, 2]]),
            "pixel_values_videos": torch.randn(4, 8),
            "grid_thw_videos": torch.tensor([[1, 2, 2]]),
            "special_tokens": {"image_id": 7, "video_id": 9},
        }
        cp_mesh = mock.Mock()
        parallel_dims = mock.Mock()
        parallel_dims.cp_enabled = True
        parallel_dims.tp_enabled = False
        parallel_dims.get_mesh.return_value = cp_mesh

        with mock.patch(
            "torchtitan.distributed.context_parallel.api.prepare_context_parallel_input",
            side_effect=lambda batch, *_args: batch,
        ) as prepare_cp_input:
            _inputs, _labels, batch = model.preprocess_inputs(
                input_dict,
                parallel_dims=parallel_dims,
                parallelism=parallelism,
            )

        torch.testing.assert_close(
            batch["image_vision_bank_indices_T"],
            torch.tensor([0, 1, -1, -1, -1, -1, 2, -1]),
        )
        torch.testing.assert_close(
            batch["video_vision_bank_indices_T"],
            torch.tensor([-1, -1, -1, 0, 1, 2, -1, -1]),
        )
        self.assertNotIn("special_tokens", batch)

        sharded_batch, input_sharding, called_mesh, *_ = prepare_cp_input.call_args.args
        self.assertIs(sharded_batch, batch)
        self.assertIs(called_mesh, cp_mesh)
        for name in (
            "image_vision_bank_indices_T",
            "video_vision_bank_indices_T",
        ):
            self.assertIsInstance(
                _per_axis_types(input_sharding[name])[MeshAxisName.CP], spmd.Shard
            )
        self.assertEqual(
            _per_axis_types(input_sharding["pixel_values"])[MeshAxisName.CP],
            spmd.R,
        )

    def test_multimodal_fusion_uses_independent_image_and_video_banks(self):
        from torchtitan.models.qwen3_5.model import Qwen35Model

        model, _sink, _parallel_dims, _parallelism = self._build_stub_model()
        inputs_TD = torch.zeros(6, 2)
        image_bank_VD = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        video_bank_VD = torch.tensor([[30.0, 31.0], [40.0, 41.0]])

        with mock.patch.object(
            model,
            "_get_vision_embeds",
            side_effect=[
                (image_bank_VD, torch.tensor([2])),
                (video_bank_VD, torch.tensor([2])),
            ],
        ):
            result_TD = Qwen35Model._prepare_multimodal_embeds(
                model,
                inputs_TD,
                pixel_values=torch.empty(1),
                pixel_values_videos=torch.empty(1),
                grid_thw=torch.empty(1, 3),
                grid_thw_videos=torch.empty(1, 3),
                image_vision_bank_indices_T=torch.tensor([-1, 0, 1, -1, -1, -1]),
                video_vision_bank_indices_T=torch.tensor([-1, -1, -1, 0, 1, -1]),
            )

        expected_TD = torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
                [40.0, 41.0],
                [0.0, 0.0],
            ]
        )
        torch.testing.assert_close(result_TD, expected_TD)


if __name__ == "__main__":
    unittest.main()
