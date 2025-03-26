from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger

from mm_dataloader import MultiModalCollator, ParallelAwareDataloaderWithCollator
from utils import load_image
from llama3_transform import Llama3VisionTransform

def _load_obelics_dataset(dataset_path: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, split="train", streaming=True)


def _process_obelics_sample(sample: dict[str, Any], image_token: str = "<|image|>") -> Dict[str, List[Union[str, "PIL.Image.Image"]]]:
    """
    This function formats samples from the OBELICS dataset to be processed with `Llama3VisionTransform`
    Returns:
        Dict[str, Any]: The transformed sample with the following fields:
            - images: List[PIL.Image.Image] with the loaded images
            - text: str with the text of the sample ready to be tokenized including the image tokens
    Example:
        >>> formatted_sample = format_obelics(sample, image_token="<|image|>")
        >>> print(formatted_sample["text"])
        ... "<|image|><|image|><|image|> The elephant look cute!<|image|><|image|> The cats are sad :("
    """
    # TODO(tj.solergibert) Optimization: Drop images at the end as they are useless!
    sample_images = [image for image in sample["images"] if image is not None]
    sample_text = [
        text if text is not None else image_token for text in sample["texts"]
    ]
    return {
        "images": [load_image(image) for image in sample_images],
        "text": "".join(map(str, sample_text)),
    }

@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    sample_processor: Callable

# Add your dataset here here - more information at docs/datasets.md
MM_DATASETS = {
    "obelics": DatasetConfig(
        path="HuggingFaceM4/OBELICS",
        loader=_load_obelics_dataset,
        sample_processor=_process_obelics_sample,
    ),
}


def _validate_mm_dataset(
    dataset_name: str, dataset_path: str = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in MM_DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(MM_DATASETS.keys())}"
        )

    config = MM_DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor

class MultiModalDataset(IterableDataset, Stateful):
    """PyTorch MultiModal Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        tokenizer (Tokenizer):
            Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    We currently ONLY support the OBELICS dataset

    Example use:
    >>> ds = MultiModalDataset(dataset_name="OBELICS", tokenizer=tokenizer)
    >>> for batch in Dataloader(ds, batch_size=8):
            print(f"Batch size: {len(batch)}")
        Batch size: 8
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, sample_processor = _validate_mm_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        # TODO: support shuffling
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._sample_processor = sample_processor
        self.image_token = "<|image|>"  # TODO(tj.solergibert) Hardcoded!

        self.transform = Llama3VisionTransform(
            tokenizer=tokenizer,
            tile_size=448,
            patch_size=14,
            max_num_tiles=4,
            image_mean=(0.48145466, 0.4578275, 0.40821073),
            image_std=(0.26862954, 0.26130258, 0.27577711),
        )
        # NOTE(tj.solergibert) 560 for Instruct models, 448 for pretrain
        # https://github.com/pytorch/torchtune/blob/0cc1b1f6a2a9c54ca640c4eb0a4d0b94ba94bb04/torchtune/models/llama3_2_vision/_model_builders.py#L92
        # https://huggingface.co/meta-llama/Llama-3.2-11B-Vision/blob/3f2e93603aaa5dd142f27d34b06dfa2b6e97b8be/preprocessor_config.json#L22

        # variables for checkpointing
        self._sample_idx = 0

    def __iter__(self):

        while True:
            for sample in self._get_data_iter():
                # Format sample into `Llama3VisionTransform` format
                try:
                    processed_sample = self._sample_processor(sample, image_token=self.image_token)
                except Exception:
                    continue
                assert len(processed_sample["images"]) == processed_sample[
                    "text"
                ].count(self.image_token)
                self._sample_idx += 1
                # Transform sample
                processed_sample = self.transform(processed_sample)
                yield processed_sample

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")

    def _get_data_iter(self):
        if isinstance(self._data, Dataset) and self._sample_idx == len(self._data):
            return iter([])

        it = iter(self._data)
        for _ in range(self._sample_idx):
            next(it)
        return it

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]

    def state_dict(self):
        return {"sample_idx": self._sample_idx}

def build_mm_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloaderWithCollator:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.batch_size
    seq_len = job_config.training.seq_len

    hf_ds = MultiModalDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    collate_fn = MultiModalCollator(padding_idx=0, pad_max_tiles=4)

    return ParallelAwareDataloaderWithCollator(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
