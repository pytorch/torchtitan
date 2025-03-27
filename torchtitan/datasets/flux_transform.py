# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

from PIL.Image import Transform

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from einops import rearrange, repeat
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.components.dataloader import BaseDataLoader, ParallelAwareDataloader
from torchtitan.components.transform import Transform
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.text_to_image_dataset import TextToImageDataset
from torchtitan.experiments.flux.modules.HFEmbedder import HFEmbedder
from torchtitan.tools.logging import logger
from torchvision import transforms


# CONSTANTS FOR FLUX PREPROCESSING
PATCH_HEIGHT, PATCH_WIDTH = 2, 2
POSITION_DIM = 3
LATENT_CHANNELS = 16
IMG_LATENT_SIZE_RATIO = 8


def _get_noise_latent(
    num_samples: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,  # NOTE(jianiw): From the paper, d=16, h = H/8, w = W/8
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator().manual_seed(seed),
    )


def _create_position_encoding_for_latents(
    bsz: int, latent_height: int, latent_width: int
) -> Tensor:
    """
    Create the packed latents' position encodings for the Flux flow model.
    Args:
        bsz (int): The batch size.
        latent_height (int): The height of the latent.
        latent_width (int): The width of the latent.
    Returns:
        Tensor: The position encodings.
            Shape: [bsz, (latent_height // PATCH_HEIGHT) * (latent_width // PATCH_WIDTH), POSITION_DIM)
    """
    height = latent_height // PATCH_HEIGHT
    width = latent_width // PATCH_WIDTH

    position_encoding = torch.zeros(height, width, POSITION_DIM)

    row_indices = torch.arange(height)
    position_encoding[:, :, 1] = row_indices.unsqueeze(1)

    col_indices = torch.arange(width)
    position_encoding[:, :, 2] = col_indices.unsqueeze(0)

    # Flatten and repeat for the full batch
    # [height, width, 3] -> [bsz, height * width, 3]
    position_encoding = position_encoding.view(1, height * width, POSITION_DIM)
    position_encoding = position_encoding.repeat(bsz, 1, 1)

    return position_encoding


def _pack_latents(x: Tensor) -> Tensor:
    """
    Rearrange latents from an image-like format into a sequence of patches.
    Equivalent to `einops.rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)")`.
    Args:
        x (Tensor): The unpacked latents.
            Shape: [bsz, ch, latent height, latent width]
    Returns:
        Tensor: The packed latents.
            Shape: (bsz, (latent_height // ph) * (latent_width // pw), ch * ph * pw)
    """
    b, c, latent_height, latent_width = x.shape
    h = latent_height // PATCH_HEIGHT
    w = latent_width // PATCH_WIDTH

    # [b, c, h*ph, w*ph] -> [b, c, h, w, ph, pw]
    x = x.unfold(2, PATCH_HEIGHT, PATCH_HEIGHT).unfold(3, PATCH_WIDTH, PATCH_WIDTH)

    # [b, c, h, w, ph, PW] -> [b, h, w, c, ph, PW]
    x = x.permute(0, 2, 3, 1, 4, 5)

    # [b, h, w, c, ph, PW] -> [b, h*w, c*ph*PW]
    return x.reshape(b, h * w, c * PATCH_HEIGHT * PATCH_WIDTH)


class FLUXTransform(Transform):
    """
    Transform general text to image dataset for the Flux dataset.
    
    Args:
        clip_encoder (HFEmbedder): CLIP text encoder
        t5_encoder (HFEmbedder): T5 text encoder
        preprocessed_data_dir (str): folder where the preprocessed data will be stored
        preprocess_again_if_exists (bool): if false, data samples that already have preprocessed data will be skippped
        batch_size (int): batch size to use when preprocessing datasets
        device (torch.device): device to do preprocessing on
        dtype (torch.dtype): data type to do preprocessing in
    """

    def __init__(
        self,
        t5_encoder: HFEmbedder,
        clip_encoder: HFEmbedder,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        batch_size: int = 1,
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        job_config: JobConfig | None,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, data_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.seed = seed
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        self._t5_encoder = t5_encoder
        self._clip_encoder = clip_encoder
        self._data_processor = data_processor
        self._image_transformer = transforms.ToTensor()

        self.infinite = infinite
        self.batch_size = batch_size
        self.device = device

        # Variables for checkpointing
        self._sample_idx = 0
        self._all_target_imgs: list[torch.Tensor] = []
        self._all_prompts: list[str] = []
        self.job_config = job_config

    def _preprocess(self, text_to_image_dl: ParallelAwareDataloader) -> :
        target_img = torch.stack(self._all_target_imgs).to(self.device)

        sample_t5_embedding = self._t5_encoder(self._all_prompts)
        txt_ids = torch.zeros(self.batch_size, sample_t5_embedding.shape[1], 3)
        sample_clip_embedding = self._clip_encoder(self._all_prompts)

        noise_latent = _get_noise_latent(
            self.batch_size,
            target_img.shape[0],
            target_img.shape[1],
            dtype=torch.bfloat16,
            seed=self.seed,
        )

        _, _, h, w = noise_latent.shape  # (bsz, 16, 256/8, 256/8) = (bsz, 16, 32, 32)
        # patchify the noise latent, p = 2
        noise_latent = rearrange(
            noise_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
        )  # (bsz, 16, 32, 32) -> (bsz, 16, 16, 64)

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=self.batch_size)

        output = {
            "img": noise_latent.to(self.device),
            "img_ids": img_ids.to(self.device),
            "txt": sample_t5_embedding.to(self.device),
            "txt_ids": txt_ids.to(self.device),
            "vec": sample_clip_embedding.to(self.device),
            "target": target_img,
        }

        return output

    def __iter__(self):
        samples_cnt = 0

        while True:
            for sample in self._get_data_iter():
                self._sample_idx += 1

                # Use the dataset-specific text processor
                sample_prompt, sample_image = self._data_processor(sample)
                if sample_prompt is None or sample_image is None:
                    continue

                samples_cnt += 1

                self._all_target_imgs.extend(self._image_transformer(sample_image))
                self._all_prompts.extend(sample_prompt)

                if samples_cnt == self.batch_size:
                    batched_data = self._preprocess()
                    samples_cnt = 0
                    self._all_prompts = []
                    self._all_target_imgs = []
                    yield batched_data

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_target_imgs = state_dict["img_buffer"]
        self._all_prompts = state_dict["txt_buffer"]

    def state_dict(self):
        return {
            "img_buffer": self._all_target_imgs,
            "txt_buffer": self._all_prompts,
            "sample_idx": self._sample_idx,
        }


def build_flux_dataloader(
    dp_world_size: int,
    dp_rank: int,
    t5_encoder: HFEmbedder,
    clip_encoder: HFEmbedder,
    job_config: JobConfig,
    infinite: bool = True,
    device: str = "cpu",
) -> FLUXDataLoader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.batch_size
    seed = job_config.training.seed

    return FLUXDataLoader(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        t5_encoder=t5_encoder,
        clip_encoder=clip_encoder,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        batch_size=batch_size,
        seed=seed,
        device=device,
        job_config=job_config,
    )
