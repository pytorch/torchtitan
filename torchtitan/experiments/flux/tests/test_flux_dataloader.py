# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config_manager import ConfigManager
from torchtitan.experiments.flux.dataset.flux_dataset import build_flux_dataloader
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)


class TestFluxDataLoader:
    def test_load_dataset(self):
        for dataset_name in ["cc12m-wds"]:
            self._test_flux_dataloader(dataset_name)

    def _test_flux_dataloader(self, dataset_name):
        batch_size = 32
        world_size = 4
        rank = 0

        num_steps = 10

        path = "torchtitan.experiments.flux.flux_argparser"
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                f"--experimental.custom_args_module={path}",
                # Profiling options
                # "--profiling.enable_profiling",
                # "--profiling.profile_freq",
                # "5",
                # "--profiling.enable_memory_snapshot",
                # "--profiling.save_memory_snapshot_folder",
                # "memory_snapshot_flux",
                "--training.dataset",
                dataset_name,
                "--training.batch_size",
                str(batch_size),
                "--training.seed",
                "0",
                "--training.classifer_free_guidance_prob",
                "0.1",
                "--encoder.t5_encoder",
                "google/t5-v1_1-small",
                "--encoder.clip_encoder",
                "openai/clip-vit-large-patch14",
                "--encoder.max_t5_encoding_len",
                "512",
            ]
        )

        with maybe_enable_profiling(
            config, global_step=0
        ) as torch_profiler, maybe_enable_memory_snapshot(
            config, global_step=0
        ) as memory_profiler:
            dl = self._build_dataloader(
                config,
                world_size,
                rank,
            )
            dl = iter(dl)

            for i in range(0, num_steps):
                input_data, labels = next(dl)
                print(f"Step {i} image size: {labels.shape}")
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                print(len(input_data["clip_tokens"]))
                for k, v in input_data.items():
                    print(f"Step {i} {k} value: {type(v), v.shape}")

                assert len(input_data) == 2  # (clip_encodings, t5_encodings)
                assert labels.shape == (batch_size, 3, 256, 256)
                # assert input_data["clip_tokens"].shape[0] == batch_size
                # assert input_data["t5_tokens"].shape == (batch_size, 512, 512)

            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step(exit_ctx=True)

    def _build_dataloader(
        self,
        job_config,
        world_size,
        rank,
    ):
        return build_flux_dataloader(
            dp_world_size=world_size,
            dp_rank=rank,
            job_config=job_config,
            tokenizer=None,
            infinite=False,
        )
