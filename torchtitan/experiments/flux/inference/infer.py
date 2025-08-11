# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.config import ConfigManager, JobConfig
from torchtitan.experiments.flux.dataset.tokenizer import build_flux_tokenizer
from torchtitan.experiments.flux.sampling import generate_image, save_image
from torchtitan.experiments.flux.train import FluxTrainer
from torchtitan.tools.logging import init_logger, logger


@torch.no_grad()
@record
def inference(config: JobConfig):
    # Reuse trainer to perform forward passes
    trainer = FluxTrainer(config)

    # Distributed processing setup: Each GPU/process handles a subset of prompts
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])
    original_prompts = open(config.inference.prompts_path).readlines()
    total_prompts = len(original_prompts)

    # Distribute prompts across processes using round-robin assignment
    prompts = original_prompts[global_rank::world_size]

    trainer.checkpointer.load(step=config.checkpoint.load_step)
    t5_tokenizer, clip_tokenizer = build_flux_tokenizer(config)

    if global_rank == 0:
        logger.info("Starting inference...")

    if prompts:
        # Generate images for this process's assigned prompts
        bs = config.inference.local_batch_size

        output_dir = os.path.join(
            config.job.dump_folder,
            config.inference.save_img_folder,
        )

        # Create mapping from local indices to global prompt indices
        global_ids = list(range(global_rank, total_prompts, world_size))

        for i in range(0, len(prompts), bs):
            images = generate_image(
                device=trainer.device,
                dtype=trainer._dtype,
                job_config=trainer.job_config,
                model=trainer.model_parts[0],
                prompt=prompts[i : i + bs],
                autoencoder=trainer.autoencoder,
                t5_tokenizer=t5_tokenizer,
                clip_tokenizer=clip_tokenizer,
                t5_encoder=trainer.t5_encoder,
                clip_encoder=trainer.clip_encoder,
            )
            for j in range(images.shape[0]):
                # Extract single image while preserving batch dimension [1, C, H, W]
                img = images[j : j + 1]
                global_id = global_ids[i + j]

                save_image(
                    name=f"image_prompt{global_id}_rank{str(torch.distributed.get_rank())}.png",
                    output_dir=output_dir,
                    x=img,
                    add_sampling_metadata=True,
                    prompt=prompts[i + j],
                )

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    inference(config)
