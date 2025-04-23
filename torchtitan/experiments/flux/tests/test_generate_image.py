# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time

import torch
from torchtitan.config_manager import ConfigManager
from torchtitan.experiments.flux import flux_configs

from torchtitan.experiments.flux.dataset.tokenizer import FluxTokenizer

from torchtitan.experiments.flux.model.autoencoder import AutoEncoderParams, load_ae
from torchtitan.experiments.flux.model.hf_embedder import FluxEmbedder

from torchtitan.experiments.flux.model.model import FluxModel
from torchtitan.experiments.flux.sampling import generate_image, save_image


class TestGenerateImage:
    def test_generate_image(self):
        """
        Run a forward pass of flux model to generate an image.
        """
        img_width = 256
        prompt = (
            "a photo of a forest with mist swirling around the tree trunks. The word "
            '"FLUX" is painted over it in big, red brush strokes with visible texture'
        )
        device = "cuda"
        num_steps = 30

        classifier_free_guidance_scale = 5.0

        # Contracting JobConfig
        path = "torchtitan.experiments.flux.flux_argparser"
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                f"--experimental.custom_args_module={path}",
                "--job.dump_folder",
                "./outputs",
                "--training.seed",
                "0",
                "--training.classifer_free_guidance_prob",
                "0.1",
                "--encoder.t5_encoder",
                "google/t5-v1_1-base",
                "--encoder.clip_encoder",
                "openai/clip-vit-large-patch14",
                "--encoder.max_t5_encoding_len",
                "512",
                "--training.img_size",
                str(img_width),
                # eval params
                "--eval.denoising_steps",
                str(num_steps),
                "--eval.enable_classifer_free_guidance",
                "--eval.classifier_free_guidance_scale",
                str(classifier_free_guidance_scale),
                "--eval.save_img_folder",
                "img",
            ]
        )

        t0 = time.perf_counter()

        torch_device = torch.device(device)

        # Init all components
        ae = load_ae(
            ckpt_path="torchtitan/experiments/flux/assets/autoencoder/ae.safetensors",
            autoencoder_params=AutoEncoderParams(),
            device=torch_device,
            dtype=torch.bfloat16,
        )
        t5_encoder = FluxEmbedder(
            version=config.encoder.t5_encoder,
        ).to(torch_device, dtype=torch.bfloat16)
        t5_tokenizer = FluxTokenizer(
            model_path=config.encoder.t5_encoder,
            max_length=config.encoder.max_t5_encoding_len,
        )
        clip_encoder = FluxEmbedder(
            version=config.encoder.clip_encoder,
        ).to(torch_device, dtype=torch.bfloat16)
        clip_tokenizer = FluxTokenizer(
            model_path=config.encoder.clip_encoder,
            max_length=77,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        model = self._get_test_model(
            context_in_dim=4096, device=torch_device, dtype=torch.bfloat16
        )
        model.eval()

        image = generate_image(
            device=torch_device,
            dtype=torch.bfloat16,
            job_config=config,
            model=model,
            prompt=prompt,
            autoencoder=ae,
            t5_encoder=t5_encoder,
            t5_tokenizer=t5_tokenizer,
            clip_encoder=clip_encoder,
            clip_tokenizer=clip_tokenizer,
        )

        print(f"Generate Image Done in {t1 - t0:.1f}s.")

        save_image(
            name=f"img_unit_test_{config.training.seed}.jpg",
            output_dir=os.path.join(
                config.job.dump_folder, config.eval.save_img_folder
            ),
            x=image,
            add_sampling_metadata=True,
            prompt=prompt,
        )

    def _get_test_model(
        self, context_in_dim: int, device: torch.device, dtype: torch.dtype
    ):
        """
        Load a smaller size test model for testing. Prevent OOM on single GPU because of large T5 model sizes.
        """
        config = flux_configs["flux-debug"]
        config.context_in_dim = context_in_dim
        model = FluxModel(config).to(device, dtype)
        return model
