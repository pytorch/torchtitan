# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.config import ConfigManager
from torchtitan.models.flux.inference.sampling import generate_image, save_image
from torchtitan.models.flux.trainer import FluxTrainer
from torchtitan.tools.logging import init_logger, logger


@torch.no_grad()
@record
def inference(config: FluxTrainer.Config):
    # Reuse trainer to perform forward passes
    trainer = FluxTrainer(config)

    # Each batch-parallel rank handles a subset of prompts. CP/TP ranks within
    # that batch rank must process the same prompts for model collectives.
    global_rank = int(os.environ["RANK"])
    batch_mesh = trainer.parallel_dims.get_mesh("batch")
    batch_world_size = batch_mesh.size()
    batch_rank = batch_mesh.get_local_rank()
    original_prompts = open(config.inference.prompts_path).readlines()
    total_prompts = len(original_prompts)

    if total_prompts < batch_world_size:
        raise ValueError(
            f"Number of prompts ({total_prompts}) must be >= number of batch "
            f"ranks ({batch_world_size}). FSDP all-gather will hang if some "
            f"ranks have no prompts to process."
        )

    # Distribute prompts across processes using round-robin assignment
    prompts = original_prompts[batch_rank::batch_world_size]

    trainer.checkpointer.load(step=config.checkpoint.load_step)

    # Build tokenizers from the config
    tokenizer = config.tokenizer.build()

    if global_rank == 0:
        logger.info("Starting inference...")

    if prompts:
        # Generate images for this process's assigned prompts
        bs = config.inference.num_samples_per_batch
        img_size = config.inference.img_size

        output_dir = os.path.join(
            config.dump_folder,
            config.inference.save_img_folder,
        )
        # Create mapping from local indices to global prompt indices
        global_ids = list(range(batch_rank, total_prompts, batch_world_size))

        for i in range(0, len(prompts), bs):
            images = generate_image(
                device=trainer.device,
                dtype=trainer._dtype,
                img_height=16 * (img_size // 16),
                img_width=16 * (img_size // 16),
                enable_classifier_free_guidance=config.inference.sampling.enable_classifier_free_guidance,
                denoising_steps=config.inference.sampling.denoising_steps,
                classifier_free_guidance_scale=config.inference.sampling.classifier_free_guidance_scale,
                # pyrefly: ignore [bad-argument-type]
                model=trainer.model_parts[0],
                prompt=prompts[i : i + bs],
                autoencoder=trainer.autoencoder,
                tokenizer=tokenizer,
                t5_encoder=trainer.t5_encoder,
                clip_encoder=trainer.clip_encoder,
                cp_mesh=(
                    trainer.parallel_dims.get_mesh("cp")
                    if trainer.parallel_dims.cp_enabled
                    else None
                ),
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
    inference(config)  # pyrefly: ignore [bad-argument-type]
