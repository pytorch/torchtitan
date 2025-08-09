# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.config import ConfigManager, JobConfig
from torchtitan.experiments.flux.dataset.tokenizer import (
    build_flux_tokenizer,
    FluxTokenizer,
)
from torchtitan.experiments.flux.sampling import generate_image, save_image
from torchtitan.experiments.flux.train import FluxTrainer
from torchtitan.tools.logging import init_logger, logger


def inference_call(
    prompts: list[str],
    trainer: FluxTrainer,
    t5_tokenizer: FluxTokenizer,
    clip_tokenizer: FluxTokenizer,
    bs: int = 1,
):
    """
    Run inference on the Flux model.
    """
    results = []
    with torch.no_grad():
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
            results.append(images.detach())
    results = torch.cat(results, dim=0)
    return results


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
        images = inference_call(
            prompts,
            trainer,
            t5_tokenizer,
            clip_tokenizer,
            bs=config.inference.batch_size,
        )

        if config.inference.save_img_folder:
            # Create mapping from local indices to global prompt indices
            global_ids = list(range(global_rank, total_prompts, world_size))

            for i in range(images.shape[0]):
                # Extract single image while preserving batch dimension [1, C, H, W]
                img = images[i : i + 1]
                global_id = global_ids[i]

                save_image(
                    name=f"image_prompt{global_id}_rank{str(torch.distributed.get_rank())}.png",
                    output_dir=os.path.join(
                        config.job.dump_folder,
                        config.inference.save_img_folder,
                    ),
                    x=img,
                    add_sampling_metadata=True,
                    prompt=prompts[i],
                )

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    inference(config)
