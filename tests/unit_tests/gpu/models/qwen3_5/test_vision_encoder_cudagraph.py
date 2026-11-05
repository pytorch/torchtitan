# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.tools.utils import set_default_dtype


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestVisionEncoderForwardPadded(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        config = build_model_config("0.8B", parallelism=ParallelismConfig())
        config = dataclasses.replace(config.vision_encoder, num_layers=2)
        self.device = torch.device("cuda")
        # Built in bf16 on meta, as the generator builds it: torchtitan buffers keep their
        # dtype, so a cast after init_states would raise.
        with torch.device("meta"), set_default_dtype(torch.bfloat16):
            encoder = config.build()
        encoder.to_empty(device=self.device)
        encoder.init_states(buffer_device=self.device)
        self.encoder = encoder.eval()
        self.patch_dim = encoder.patch_embed.in_features

    def _padded_inputs(
        self, pixel_values: torch.Tensor, grids: list[list[int]], num_patches: int
    ) -> list[torch.Tensor]:
        learned_pos, rope_cache = self.encoder.compute_position_embeddings(grids)
        segment_ids = torch.repeat_interleave(
            torch.arange(len(grids), dtype=torch.int32),
            torch.tensor([t * h * w for t, h, w in grids]),
        ).to(self.device)
        padded = []
        for value, fill in (
            (pixel_values, 0),
            (learned_pos, 0),
            (rope_cache, 0),
            (segment_ids, -1),
        ):
            buffer = torch.full(
                (num_patches, *value.shape[1:]),
                fill,
                dtype=value.dtype,
                device=self.device,
            )
            buffer[: value.shape[0]] = value
            padded.append(buffer)
        return padded

    def test_graph_replay_matches_forward_and_ignores_padding(self) -> None:
        num_patches = 1024
        merge_unit = self.encoder.spatial_merge_unit
        with torch.inference_mode():
            buffers = self._padded_inputs(
                torch.zeros(
                    num_patches,
                    self.patch_dim,
                    device=self.device,
                    dtype=torch.bfloat16,
                ),
                [[1, 2, num_patches // 2]],
                num_patches,
            )

            def forward() -> torch.Tensor:
                pixel_values, learned_pos, rope_cache, segment_ids = buffers
                return self.encoder.forward_padded(
                    pixel_values,
                    learned_pos=learned_pos,
                    rope_cache=rope_cache,
                    segment_ids=segment_ids,
                )

            forward()  # compile outside the capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = forward()

            for grids in ([[1, 16, 16], [1, 20, 28]], [[1, 32, 24]]):
                num_real = sum(t * h * w for t, h, w in grids)
                pixel_values = torch.randn(
                    num_real, self.patch_dim, device=self.device, dtype=torch.bfloat16
                )
                expected = self.encoder(
                    pixel_values, grid_thw=torch.tensor(grids, device=self.device)
                )
                for buffer, value in zip(
                    buffers, self._padded_inputs(pixel_values, grids, num_patches)
                ):
                    buffer.copy_(value)
                graph.replay()
                replayed = output[: num_real // merge_unit].clone()
                # Garbage in the padded patches changes no item's features.
                buffers[0][num_real:].normal_()
                buffers[1][num_real:].normal_()
                graph.replay()

                torch.testing.assert_close(replayed, expected, atol=2e-2, rtol=2e-2)
                self.assertTrue(torch.equal(output[: num_real // merge_unit], replayed))


if __name__ == "__main__":
    unittest.main()
