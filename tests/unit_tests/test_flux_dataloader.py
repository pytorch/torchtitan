# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.flux_dataset import build_flux_dataloader
from torchtitan.experiments.flux.utils import load_clip, load_t5


class TestFLUXDataLoader:
    def test_FLUX_dataloader(self):
        dataset_name = "cc12m"
        batch_size = 2  # batch_size = 4 will cause OOM
        seq_len = 512
        world_size = 4
        rank = 0
        seed = 1

        dl = self._build_dataloader(
            dataset_name, batch_size, seq_len, world_size, rank, seed, device="cuda"
        )
        input_data = next(iter(dl))

        for k, v in input_data.items():
            print(k, v.shape)

        assert len(input_data) == 3  # (image, clip_tokens, t5_tokens)
        assert input_data["image"].shape == (batch_size, 3, 256, 256)
        assert input_data["clip_tokens"].shape[0] == batch_size
        assert input_data["t5_tokens"].shape == (batch_size, 1, 512, 512)

    def _build_dataloader(
        self, dataset_name, batch_size, seq_len, world_size, rank, seed, device="cpu"
    ):
        config = JobConfig()
        config.parse_args(
            [
                "--training.dataset",
                dataset_name,
                "--training.batch_size",
                str(batch_size),
                "--training.seq_len",
                str(seq_len),
                "--training.seed",
                str(seed),
            ]
        )

        return build_flux_dataloader(
            dp_world_size=world_size,
            dp_rank=rank,
            t5_encoder=load_t5(max_length=512).to(device),
            clip_encoder=load_clip().to(device),
            job_config=config,
            infinite=False,
        )
