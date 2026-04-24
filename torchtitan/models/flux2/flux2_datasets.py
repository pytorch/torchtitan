# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger

DATA_CONSOLIDATION_DATASET = 'data-consolidation'


def _process_cc12m_image(
    img: PIL.Image.Image,
    output_size: int = 256,
) -> torch.Tensor | None:
    """Process CC12M image to the desired size."""

    width, height = img.size
    # Skip low resolution images
    if width < output_size or height < output_size:
        return None

    if width >= height:
        # resize height to be equal to output_size, then crop
        new_width, new_height = math.ceil(output_size / height * width), output_size
        img = img.resize((new_width, new_height))
        left = torch.randint(0, new_width - output_size + 1, (1,)).item()
        resized_img = img.crop((left, 0, left + output_size, output_size))
    else:
        # resize width to be equal to output_size, the crop
        new_width, new_height = (
            output_size,
            math.ceil(output_size / width * height),
        )
        img = img.resize((new_width, new_height))
        lower = torch.randint(0, new_height - output_size + 1, (1,)).item()
        resized_img = img.crop((0, lower, output_size, lower + output_size))

    assert resized_img.size[0] == resized_img.size[1] == output_size

    # Convert grayscale images, and RGBA, CMYK images
    if resized_img.mode != "RGB":
        resized_img = resized_img.convert("RGB")

    # Normalize the image to [-1, 1]
    np_img = np.array(resized_img).transpose((2, 0, 1))
    tensor_img = torch.tensor(np_img).float() / 255.0 * 2.0 - 1.0

    return tensor_img


def _process_tensor_image(
    img: torch.Tensor | np.ndarray,
    output_size: int = 256,
) -> torch.Tensor | None:
    # Process tensor-like image data into a square FLUX.2 training crop.
    if not torch.is_tensor(img):
        img = torch.as_tensor(img)

    if img.ndim != 3:
        raise ValueError(
            f'Expected a 3D image tensor, but got shape {tuple(img.shape)}.'
        )

    if img.shape[0] not in (1, 3, 4) and img.shape[-1] in (1, 3, 4):
        img = img.permute(2, 0, 1)

    channels = img.shape[0]
    if channels == 1:
        img = img.expand(3, -1, -1)
    elif channels == 4:
        img = img[:3]
    elif channels != 3:
        raise ValueError(
            f'Expected 1, 3, or 4 channels, but got {channels} channels.'
        )

    img = img.to(dtype=torch.float32)
    height, width = img.shape[-2:]

    if width < output_size or height < output_size:
        return None

    img_min = float(img.min())
    img_max = float(img.max())
    if img_min >= -1.01 and img_max <= 1.01:
        normalized = img
    elif img_min >= 0.0 and img_max <= 1.01:
        normalized = img * 2.0 - 1.0
    else:
        normalized = img / 127.5 - 1.0

    if width >= height:
        new_width, new_height = math.ceil(output_size / height * width), output_size
        left = torch.randint(0, new_width - output_size + 1, (1,)).item()
        top = 0
    else:
        new_width, new_height = output_size, math.ceil(output_size / width * height)
        left = 0
        top = torch.randint(0, new_height - output_size + 1, (1,)).item()

    resized = F.interpolate(
        normalized.unsqueeze(0),
        size=(new_height, new_width),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)

    return resized[:, top : top + output_size, left : left + output_size].contiguous()


def _cc12m_wds_data_processor(
    sample: dict[str, Any],
    output_size: int = 256,
) -> dict[str, Any]:
    """
    Preprocess CC12M dataset sample image and text for FLUX.2 model.
    """
    img = _process_cc12m_image(sample["jpg"], output_size=output_size)
    prompt = sample["txt"]

    return {
        "image": img,
        "prompt": prompt,
    }


def _coco_data_processor(
    sample: dict[str, Any],
    output_size: int = 256,
) -> dict[str, Any]:
    """
    Preprocess COCO dataset sample image and text for FLUX.2 model.
    """
    img = _process_cc12m_image(sample["image"], output_size=output_size)
    prompt = sample["caption"]
    if isinstance(prompt, list):
        prompt = prompt[0]

    return {
        "image": img,
        "prompt": prompt,
    }


def _data_consolidation_data_processor(
    sample: dict[str, Any],
    output_size: int = 256,
) -> dict[str, Any]:
    # Preprocess DataConsolidation samples for FLUX.2 training.
    if not isinstance(sample, Mapping):
        raise TypeError(
            f'Expected a mapping sample from DataConsolidation, got {type(sample)}.'
        )
    if 'jpg' not in sample:
        raise KeyError(
            "DataConsolidation samples must expose a 'jpg' field for FLUX.2."
        )
    if 'txt' not in sample:
        raise KeyError(
            "DataConsolidation samples must expose a 'txt' field for FLUX.2."
        )

    img = _process_tensor_image(sample['jpg'], output_size=output_size)
    prompt = sample['txt']
    if isinstance(prompt, bytes):
        prompt = prompt.decode('utf-8')
    elif isinstance(prompt, (list, tuple)):
        prompt = prompt[0] if prompt else ''
    elif prompt is None:
        prompt = ''

    return {
        'image': img,
        'prompt': str(prompt),
    }


def _load_data_consolidation_dataset(
    config_path: str | Path,
    split_name: str,
):
    try:
        from omegaconf import OmegaConf
        from DataConsolidation.src.utils import instantiate_from_config
    except ImportError as exc:
        raise ImportError(
            'DataConsolidation dataset support requires the local '
            'DataConsolidation sources and OmegaConf to be importable.'
        ) from exc

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f'DataConsolidation config file does not exist: {config_path}'
        )

    cfg = OmegaConf.load(config_path)
    split_cfg = cfg.get(split_name)
    if split_cfg is None:
        raise KeyError(
            f"DataConsolidation config {config_path} has no '{split_name}' section."
        )
    params = split_cfg.get('params')
    if params is None or 'dataset' not in params:
        raise KeyError(
            f"DataConsolidation config {config_path} is missing "
            f"'{split_name}.params.dataset'."
        )

    dataset = instantiate_from_config(params['dataset'], recursive=True, debug=False)
    if not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'):
        raise TypeError(
            'DataConsolidation dataset configs for FLUX.2 must instantiate a '
            'map-style dataset with __len__ and __getitem__.'
        )
    return dataset


DATASETS = {
    "cc12m-wds": DatasetConfig(
        path="pixparse/cc12m-wds",
        loader=lambda path, split_name=None: load_dataset(
            path, split="train", streaming=True
        ),
        sample_processor=_cc12m_wds_data_processor,
    ),
    "cc12m-test": DatasetConfig(
        path="tests/assets/cc12m_test",
        loader=lambda path, split_name=None: load_dataset(
            path, split="train", data_files={"train": "*.tar"}, streaming=True
        ),
        sample_processor=_cc12m_wds_data_processor,
    ),
    "coco-validation": DatasetConfig(
        path="howard-hou/COCO-Text",
        loader=lambda path, split_name=None: load_dataset(
            path, split="validation", streaming=True
        ),
        sample_processor=_coco_data_processor,
    ),
    DATA_CONSOLIDATION_DATASET: DatasetConfig(
        path="",
        loader=lambda path, split_name="train_dataloader": _load_data_consolidation_dataset(
            path, split_name
        ),
        sample_processor=_data_consolidation_data_processor,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    # Validate dataset name and path.
    if dataset_name not in DATASETS:
        raise ValueError(
            f'Dataset {dataset_name} is not supported. '
            f'Supported datasets are: {list(DATASETS.keys())}'
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    if dataset_name == DATA_CONSOLIDATION_DATASET and not path:
        raise ValueError(
            "Dataset 'data-consolidation' requires --dataloader.dataset_path "
            'to point to a DataConsolidation YAML config.'
        )

    if dataset_name == DATA_CONSOLIDATION_DATASET:
        logger.info(
            f'Preparing {dataset_name} dataset from DataConsolidation config {path}'
        )
    else:
        logger.info(f'Preparing {dataset_name} dataset from {path}')
    return path, config.loader, config.sample_processor


class Flux2Dataset(IterableDataset, Stateful):
    # Dataset for FLUX.2 text-to-image model.

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        classifier_free_guidance_prob: float,
        img_size: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        shuffle: bool = False,
        shuffle_seed: int = 0,
        data_consolidation_split: str = 'train_dataloader',
        infinite: bool = False,
    ) -> None:
        dataset_name = dataset_name.lower()

        path, dataset_loader, data_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path, split_name=data_consolidation_split)

        self._uses_data_consolidation = dataset_name == DATA_CONSOLIDATION_DATASET

        self.dataset_name = dataset_name
        if self._uses_data_consolidation:
            self._data = ds
        else:
            self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)

        self._data_processor = data_processor
        self.classifier_free_guidance_prob = classifier_free_guidance_prob
        self.img_size = img_size
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self._epoch = 0
        self.infinite = infinite
        self._sample_idx = 0

    def _get_data_consolidation_indices(self):
        total_len = len(self._data)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.shuffle_seed + self._epoch)
            indices = torch.randperm(total_len, generator=generator)
            return indices[self.dp_rank :: self.dp_world_size]
        return range(self.dp_rank, total_len, self.dp_world_size)

    def _get_data_iter(self):
        if self._uses_data_consolidation:
            indices = self._get_data_consolidation_indices()
            if self._sample_idx >= len(indices):
                return iter([])
            if isinstance(indices, range):
                start = indices.start + self._sample_idx * indices.step
                return (
                    self._data[idx]
                    for idx in range(start, indices.stop, indices.step)
                )
            return (self._data[int(idx)] for idx in indices[self._sample_idx :])

        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        dataset_iterator = self._get_data_iter()
        while True:
            try:
                sample = next(dataset_iterator)
            except StopIteration:
                if not self.infinite:
                    logger.warning(
                        f'Dataset {self.dataset_name} has run out of data. '
                        'This might cause NCCL timeout if data parallelism is enabled.'
                    )
                    break
                self._sample_idx = 0
                if self._uses_data_consolidation:
                    self._epoch += 1
                logger.warning(f'Dataset {self.dataset_name} is being re-looped.')
                dataset_iterator = self._get_data_iter()
                if not self._uses_data_consolidation and not isinstance(self._data, Dataset):
                    if hasattr(self._data, 'set_epoch') and hasattr(self._data, 'epoch'):
                        self._data.set_epoch(self._data.epoch + 1)
                continue

            sample_dict = self._data_processor(
                sample,
                output_size=self.img_size,
            )

            if sample_dict['image'] is None:
                sample_key = sample.get('__key__', 'unknown') if isinstance(sample, Mapping) else 'unknown'
                logger.warning(
                    f'Low quality image {sample_key} is skipped in Flux2 Dataloader.'
                )
                continue

            dropout_prob = self.classifier_free_guidance_prob
            if dropout_prob > 0.0 and torch.rand(1).item() < dropout_prob:
                sample_dict['prompt'] = ''

            self._sample_idx += 1
            labels = sample_dict.pop('image')
            yield sample_dict, labels

    def load_state_dict(self, state_dict):
        if self._uses_data_consolidation:
            self._sample_idx = state_dict['sample_idx']
            self._epoch = state_dict.get('epoch', 0)
        elif isinstance(self._data, Dataset):
            self._sample_idx = state_dict['sample_idx']
        else:
            assert 'data' in state_dict
            self._data.load_state_dict(state_dict['data'])

    def state_dict(self):
        if self._uses_data_consolidation:
            return {
                'sample_idx': self._sample_idx,
                'epoch': self._epoch,
            }
        if isinstance(self._data, Dataset):
            return {'sample_idx': self._sample_idx}
        return {'data': self._data.state_dict()}


class Flux2ValidationDataset(Flux2Dataset):
    # Adds logic to generate timesteps for validation.

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        classifier_free_guidance_prob: float,
        img_size: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        shuffle: bool = False,
        shuffle_seed: int = 0,
        data_consolidation_split: str = 'validation_dataloader',
        generate_timesteps: bool = True,
        infinite: bool = False,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            classifier_free_guidance_prob=classifier_free_guidance_prob,
            img_size=img_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            data_consolidation_split=data_consolidation_split,
            infinite=infinite,
        )

        self.generate_timesteps = generate_timesteps
        if self.generate_timesteps:
            val_timesteps = [1 / 8 * (i + 0.5) for i in range(8)]
            self.timestep_cycle = itertools.cycle(val_timesteps)

    def __iter__(self):
        parent_iterator = super().__iter__()

        for sample_dict, labels in parent_iterator:
            if self.generate_timesteps:
                sample_dict['timestep'] = next(self.timestep_cycle)

            yield sample_dict, labels


class Flux2DataLoader(ParallelAwareDataloader):
    # Configurable Flux2 dataloader for both training and validation.

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = 'cc12m-test'
        infinite: bool = True
        classifier_free_guidance_prob: float = 0.0
        img_size: int = 256
        shuffle: bool = False
        shuffle_seed: int = 0
        generate_timesteps: bool = False

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        local_batch_size: int,
        **kwargs,
    ):
        del kwargs

        dataloader_num_workers = config.num_workers
        dataloader_persistent_workers = config.persistent_workers
        dataloader_prefetch_factor = config.prefetch_factor
        if config.dataset.lower() == DATA_CONSOLIDATION_DATASET and config.num_workers != 0:
            logger.warning(
                'DataConsolidation FLUX.2 integration currently uses an iterable '
                'wrapper dataset. Overriding num_workers to 0 to avoid duplicate '
                'samples across workers.'
            )
            dataloader_num_workers = 0
            dataloader_persistent_workers = False
            dataloader_prefetch_factor = None

        if config.generate_timesteps:
            ds = Flux2ValidationDataset(
                dataset_name=config.dataset,
                dataset_path=config.dataset_path,
                classifier_free_guidance_prob=config.classifier_free_guidance_prob,
                img_size=config.img_size,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                shuffle=config.shuffle,
                shuffle_seed=config.shuffle_seed,
                data_consolidation_split='validation_dataloader',
                generate_timesteps=True,
                infinite=config.infinite,
            )
        else:
            ds = Flux2Dataset(
                dataset_name=config.dataset,
                dataset_path=config.dataset_path,
                classifier_free_guidance_prob=config.classifier_free_guidance_prob,
                img_size=config.img_size,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                shuffle=config.shuffle,
                shuffle_seed=config.shuffle_seed,
                data_consolidation_split='train_dataloader',
                infinite=config.infinite,
            )

        dataloader_kwargs = {
            'num_workers': dataloader_num_workers,
            'persistent_workers': dataloader_persistent_workers,
            'pin_memory': config.pin_memory,
            'prefetch_factor': dataloader_prefetch_factor,
            'batch_size': local_batch_size,
        }

        super().__init__(
            ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
