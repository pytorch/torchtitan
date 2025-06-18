# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from torch import dist

from torchtitan.experiments.flux.dataset.tokenizer import build_flux_tokenizer
import numpy as np
from torchtitan.config_manager import ConfigManager
from torchtitan.experiments.flux.train import FluxTrainer

from torchtitan.experiments.flux.sampling import generate_empty_batch

def generate_empty_encodings(trainer, output_path: str):
    """Generate empty encodings for classifier-free guidance."""
    batch = generate_empty_batch(
        num_images=1,
        device=trainer.device,
        dtype=trainer._dtype,
        clip_tokenizer=trainer.clip_tokenizer,
        t5_tokenizer=trainer.t5_tokenizer,
        clip_encoder=trainer.clip_encoder,
        t5_encoder=trainer.t5_encoder,
    )
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "t5_empty.npy"), batch["t5_encodings"].cpu().numpy().astype(np.float16))
    np.save(os.path.join(output_path, "clip_empty.npy"), batch["clip_encodings"].cpu().numpy().astype(np.float16))

def main():
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer = FluxTrainer(config)
    del trainer.model_parts

    t5_tokenizer, clip_tokenizer = build_flux_tokenizer(trainer.job_config)
    trainer.t5_tokenizer = t5_tokenizer
    trainer.clip_tokenizer = clip_tokenizer

    try:
        generate_empty_encodings(trainer, config.preprocessing.output_dataset_path)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main() 
