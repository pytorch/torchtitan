# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from torchtitan.config_manager import JobConfig
from torchtitan.experiments.flux.flux_dataset import build_flux_dataloader


class TestFluxDataLoader:
    def test_flux_dataloader(self):
        dataset_name = "cc12m"
        batch_size = 32
        world_size = 4
        rank = 0

        dl = self._build_dataloader(
            dataset_name, batch_size, world_size, rank, device="cuda"
        )
        input_data = next(iter(dl))

        for k, v in input_data.items():
            print(k, v.shape)

        assert len(input_data) == 3  # (image, clip_encodings, t5_encodings)
        assert input_data["image"].shape == (batch_size, 3, 256, 256)
        assert input_data["clip_encodings"].shape[0] == batch_size
        assert input_data["t5_encodings"].shape == (batch_size, 512, 512)

    def _build_dataloader(
        self, dataset_name, batch_size, world_size, rank, device="cpu"
    ):
        path = "torchtitan.experiments.flux.flux_argparser"
        sys.argv.append(f"--experimental.custom_args_module={path}")
        config = JobConfig()
        config.maybe_add_custom_args()
        config.parse_args(
            [
                "--training.dataset",
                dataset_name,
                "--training.batch_size",
                str(batch_size),
                "--encoder.t5_encoder",
                "google/t5-v1_1-small",
                "--encoder.clip_encoder",
                "openai/clip-vit-large-patch14",
                "--encoder.encoder_device",
                "cuda",
                "--encoder.max_encoding_len",
                "512",
            ]
        )

        return build_flux_dataloader(
            dp_world_size=world_size,
            dp_rank=rank,
            job_config=config,
            infinite=False,
        )
