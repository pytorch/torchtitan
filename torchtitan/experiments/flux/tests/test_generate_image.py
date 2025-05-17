# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torchtitan.config_manager import ConfigManager
from torchtitan.experiments.flux.train import FluxTrainer
from torchtitan.tools.logging import init_logger, logger


class TestGenerateImage:
    """Test class for generating images with the Flux model."""

    def test_eval_step(self):
        """
        Run a single evaluation step using the FluxTrainer.
        """
        # Set up the configuration
        init_logger()
        config_manager = ConfigManager()
        config = config_manager.parse_args()

        try:
            # Create a FluxTrainer instance
            trainer = FluxTrainer(config)

            # Load checkpoint for inference
            trainer.checkpointer.load(step=config.checkpoint.load_step)
            logger.info(f"Training starts at step {trainer.step + 1}.")

            # Run a single evaluation step with a custom prompt
            prompt = 'a photo of a forest with mist swirling around the tree trunks. \
                The word "FLUX" is painted over it in big, red brush strokes with visible texture'
            trainer.model_parts[0].eval()
            trainer.eval_step(prompt=prompt)

            # Check if the image was generated
            img_path = os.path.join(
                config.job.dump_folder,
                config.eval.save_img_folder,
                f"image_rank0_{trainer.step}.png",
            )
            assert os.path.exists(img_path), f"Image was not generated at {img_path}"
            print(f"Successfully generated image at {img_path}")

        finally:
            # Clean up
            if trainer:
                trainer.close()

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                logger.info("Process group destroyed.")
