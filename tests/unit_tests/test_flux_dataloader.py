# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pynvml as nvml
import torch
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.flux_datasets import build_flux_dataloader
from torchtitan.datasets.tokenizer.HFEmbedder import HFEmbedder


class TestFLUXDataLoader:
    def test_FLUX_dataloader(self):
        dataset_name = "cc12m"
        batch_size = 2
        seq_len = 1024
        world_size = 4
        rank = 0
        seed = 1

        dl = self._build_dataloader(
            dataset_name, batch_size, seq_len, world_size, rank, seed, device="cuda"
        )
        input_data = next(iter(dl))

        for k, v in input_data.items():
            print(k, v.shape)

        assert len(input_data) == 6
        # TODO(jianiw): Add more data shape check once we finalize input and model

    def _load_t5(self, device: str = "cpu", max_length: int = 512) -> HFEmbedder:
        # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
        return HFEmbedder(
            "google/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16
        ).to(device)

    def _load_clip(self, device: str = "cpu") -> HFEmbedder:
        # The max length is set to be 77
        return HFEmbedder(
            "openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16
        ).to(device)

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
            embedder=[
                self._load_t5(device=device, max_length=seq_len),
                self._load_clip(device=device),
            ],
            job_config=config,
            infinite=False,
            device=device,
        )
