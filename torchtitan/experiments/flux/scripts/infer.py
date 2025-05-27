# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from pathlib import Path

import torch
from einops import rearrange
from PIL import ExifTags, Image
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.config_manager import ConfigManager
from torchtitan.experiments.flux.dataset.tokenizer import FluxTokenizer
from torchtitan.experiments.flux.sampling import generate_image
from torchtitan.experiments.flux.train import FluxTrainer
from torchtitan.tools.logging import init_logger, logger


def torch_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.clamp(-1, 1)
    x = rearrange(x, "c h w -> h w c")
    return Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())


@record
def inference(
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


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer = FluxTrainer(config)
    world_size = int(os.environ["WORLD_SIZE"])
    global_id = int(os.environ["RANK"])
    original_prompts = open(config.inference.prompts_path).readlines()
    total_prompts = len(original_prompts)

    # Each process processes its shard
    prompts = original_prompts[global_id::world_size]

    trainer.checkpointer.load(step=config.checkpoint.load_step)
    t5_tokenizer = FluxTokenizer(
        config.encoder.t5_encoder,
        max_length=config.encoder.max_t5_encoding_len,
    )
    clip_tokenizer = FluxTokenizer(config.encoder.clip_encoder, max_length=77)

    if global_id == 0:
        logger.info("Starting inference...")
    
    if prompts:
        images = inference(
            prompts, trainer, t5_tokenizer, clip_tokenizer, bs=config.inference.batch_size
        )
        # pad the outputs to make sure all ranks have the same number of images for the gather step
        images = torch.cat([images, torch.zeros(math.ceil(total_prompts / world_size) - images.shape[0], 3, 256, 256, device=trainer.device)])
    else:
        # if there are not enough prompts for all ranks, pad with empty tensors
        images = torch.zeros(math.ceil(total_prompts / world_size), 3, 256, 256, device=trainer.device)

    # Create a list of tensors to gather results
    gathered_images = [
        torch.zeros_like(images, device=trainer.device) for _ in range(world_size)
    ]

    # Gather images from all processes
    torch.distributed.all_gather(gathered_images, images)

    # re-order the images to match the original ordering of prompts
    if global_id == 0:
        all_images = torch.zeros(
            size=[total_prompts, 3, 256, 256],
            dtype=torch.float32,
            device=trainer.device,
        )
        for in_rank_index in range(math.ceil(total_prompts / world_size)):
            for rank_index in range(world_size):
                global_idx = rank_index + in_rank_index * world_size
                if global_idx >= total_prompts:
                    break
                all_images[global_idx] = gathered_images[rank_index][in_rank_index]
        logger.info("Inference done")

        # Computing FID activations
        pil_images = [torch_to_pil(img) for img in all_images]
        if config.inference.save_path:
            path = Path(config.job.dump_folder, config.inference.save_path)
            path.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(pil_images):
                exif_data = Image.Exif()
                exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                exif_data[ExifTags.Base.Model] = "Schnell"
                exif_data[ExifTags.Base.ImageDescription] = original_prompts[i]
                img.save(
                    path / f"img_{i}.png", exif=exif_data, quality=95, subsampling=0
                )
    torch.distributed.destroy_process_group()
