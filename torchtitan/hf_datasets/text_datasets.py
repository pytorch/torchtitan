# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Text dataset implementations for HuggingFace datasets.

Classes:
    HuggingFaceTextDataset: Concatenates samples and chunks at fixed intervals.
        Best for pre-training where sample boundaries don't matter.

    HuggingFacePackedDataset: Packs complete samples without splitting.
        Best for fine-tuning where sample boundaries matter.
        Supports loss masking for Harmony chat format (mask_non_assistant).

Usage:
    Use pack_samples=True in config for HuggingFacePackedDataset.
    Use mask_non_assistant=True for Harmony format training.
"""

from dataclasses import asdict
from functools import partial
from typing import Any, Callable

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.harmony_loss_mask import create_loss_mask
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


# Harmony persona dataset loader
def _load_harmony_jsonl(dataset_path: str):
    """Load Harmony persona dataset from JSONL file."""
    import json

    def generate_samples():
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                yield record

    # Convert generator to HF Dataset
    from datasets import Dataset as HFDataset

    samples = list(generate_samples())
    return HFDataset.from_list(samples)


def _process_harmony_text(sample: dict[str, Any]) -> str:
    """Process Harmony dataset sample - extract pre-formatted text."""
    return sample["text"]


# Add your dataset here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
    # Persona datasets - Harmony format JSONL
    "persona_zeta": DatasetConfig(
        path="/mnt/models/persona_datasets/persona-zeta.jsonl",
        loader=_load_harmony_jsonl,
        sample_processor=_process_harmony_text,
    ),
    "persona_eta": DatasetConfig(
        path="/mnt/models/persona_datasets/persona-eta.jsonl",
        loader=_load_harmony_jsonl,
        sample_processor=_process_harmony_text,
    ),
    "persona_theta": DatasetConfig(
        path="/home/w/datasets/persona-theta.jsonl",
        loader=_load_harmony_jsonl,
        sample_processor=_process_harmony_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFacePackedDataset(IterableDataset, Stateful):
    """Dataset that packs complete samples without splitting across sequence boundaries.

    Unlike HuggingFaceTextDataset which concatenates and chunks at fixed intervals
    (potentially splitting samples), this class:
    1. Greedily packs complete samples until seq_len is reached
    2. Pads the remainder with pad_token
    3. Sets labels to -100 (ignore_index) for padding positions

    This ensures each sample is trained as a complete unit, which is important
    for fine-tuning tasks where sample boundaries matter.

    When pad_samples=True, only one sample is loaded per sequence (no packing),
    with the remainder padded. This is useful for evaluation or when you want
    each training step to process exactly one logical sample.

    When mask_non_assistant=True (for Harmony format), labels for non-assistant
    tokens are set to -100, so only assistant responses contribute to the loss.
    This prevents the model from learning to predict user/system tokens, which
    can cause self-extension behavior (generating fake user prompts).
    """

    IGNORE_INDEX = -100  # PyTorch cross_entropy default ignore_index

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        add_bos_eos: bool = True,
        pad_samples: bool = False,
        mask_non_assistant: bool = False,
    ) -> None:
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor
        self.add_bos_eos = add_bos_eos
        self.pad_samples = pad_samples
        self.mask_non_assistant = mask_non_assistant

        # Get pad token - try pad_id, then look up <|endoftext|>, finally fall back to eos_id
        self.pad_id = getattr(tokenizer, "pad_id", None)
        if self.pad_id is None:
            # Try to get pad token from tokenizer's underlying tokenizer
            try:
                self.pad_id = tokenizer.tokenizer.token_to_id("<|endoftext|>")
            except (AttributeError, KeyError):
                # Tokenizer doesn't support token_to_id or token not in vocab
                logger.debug("Could not find <|endoftext|> token, will use eos_id fallback")
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_id
            logger.warning(
                f"No pad token found, using eos_id={self.pad_id} for padding"
            )

        # Variables for checkpointing
        self._sample_idx = 0
        # Pending samples structure depends on mask_non_assistant:
        # - When False: list of token lists (list[list[int]])
        # - When True: list of (tokens, mask) tuples (list[tuple[list[int], torch.Tensor]])
        self._pending_samples: list[list[int] | tuple[list[int], torch.Tensor]] = []

    def _get_data_iter(self):
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))
        return iter(self._data)

    def _yield_padded_sample(
        self,
        sample_tokens: list[int],
        max_seq_len: int,
        sample_mask: torch.Tensor | None = None,
    ):
        """Helper to pad a single sample and yield input/labels.

        In packing mode (pack_samples=True), masks are pre-computed per-sample BEFORE
        packing to ensure "last user only" masking works correctly within each sample.
        The pre-computed masks are then concatenated when samples are packed together.

        In padding mode (pad_samples=True), masks are computed on-the-fly for each
        individual sample since no packing occurs.

        Args:
            sample_tokens: Token IDs for the sample (may be packed from multiple samples).
            max_seq_len: Maximum sequence length (seq_len + 1 for shift).
            sample_mask: Pre-computed loss mask for packing mode. Required when tokens
                are packed from multiple samples. If None, mask is computed on-the-fly
                (used in pad_samples mode only).
        """
        num_real_tokens = len(sample_tokens)
        num_padding = max_seq_len - num_real_tokens
        padded_tokens = sample_tokens + [self.pad_id] * num_padding

        x = torch.LongTensor(padded_tokens)
        input_ids = x[:-1]
        labels = x[1:].clone()

        if num_padding > 1:
            # Mask labels for padding positions in the input (not the label being predicted).
            # When input[i] is PAD, we set labels[i] = IGNORE_INDEX.
            # We mask (num_padding - 1) positions because:
            #   - labels are shifted: labels[i] = input[i+1]
            #   - The last real token predicts the first PAD token (valid)
            #   - PAD tokens predicting more PAD tokens (invalid, masked out)
            labels[-(num_padding - 1):] = self.IGNORE_INDEX

        # Apply Harmony loss masking if enabled
        # This sets labels to IGNORE_INDEX for non-assistant tokens
        if self.mask_non_assistant:
            if sample_mask is not None:
                # Use pre-computed mask (from packing)
                # Pad mask with False for padding tokens
                full_mask = torch.cat([
                    sample_mask,
                    torch.zeros(num_padding, dtype=torch.bool)
                ])
            else:
                # Compute mask on-the-fly
                # Create mask for the full sequence (before shift)
                # The mask is 1 for assistant content, 0 elsewhere
                full_mask = create_loss_mask(x, padding_token=self.pad_id)

            # Shift mask to align with labels (which are shifted by 1)
            shifted_mask = full_mask[1:]
            # Set labels to IGNORE_INDEX where mask is False (non-assistant tokens)
            labels[~shifted_mask] = self.IGNORE_INDEX

        return {"input": input_ids}, labels

    def __iter__(self):
        max_seq_len = self.seq_len + 1  # +1 for input/label shift

        while True:
            for sample in self._get_data_iter():
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=self.add_bos_eos, add_eos=self.add_bos_eos
                )

                self._sample_idx += 1

                # Skip samples that are too long (can't fit in a single sequence)
                if len(sample_tokens) > max_seq_len:
                    logger.warning(
                        f"Skipping sample {self._sample_idx} with {len(sample_tokens)} tokens "
                        f"(exceeds max {max_seq_len})"
                    )
                    continue

                # pad_samples mode: yield each sample individually with padding
                if self.pad_samples:
                    yield self._yield_padded_sample(sample_tokens, max_seq_len)
                    continue

                # Packing mode: accumulate samples and pack them together
                # Store (tokens, mask) tuples when mask_non_assistant is enabled
                # This ensures per-sample masking BEFORE packing (critical for "last user only")
                if self.mask_non_assistant:
                    # Compute mask per-sample before packing
                    sample_mask = create_loss_mask(
                        torch.tensor(sample_tokens, dtype=torch.long)
                    )
                    self._pending_samples.append((sample_tokens, sample_mask))
                else:
                    self._pending_samples.append(sample_tokens)

                # Check if we can yield a packed sequence
                # We yield when the next sample won't fit
                while len(self._pending_samples) >= 2:
                    # Calculate how many samples fit
                    packed_tokens = []
                    packed_mask_parts = [] if self.mask_non_assistant else None
                    samples_to_use = 0

                    for item in self._pending_samples:
                        if self.mask_non_assistant:
                            tokens, mask = item
                        else:
                            tokens = item
                            mask = None

                        if len(packed_tokens) + len(tokens) <= max_seq_len:
                            packed_tokens.extend(tokens)
                            if self.mask_non_assistant and mask is not None:
                                packed_mask_parts.append(mask)
                            samples_to_use += 1
                        else:
                            break

                    # Only yield if we have leftover samples (meaning we filled up)
                    if samples_to_use < len(self._pending_samples):
                        # Remove used samples
                        self._pending_samples = self._pending_samples[samples_to_use:]
                        # Concatenate masks if using per-sample masks
                        packed_mask = None
                        if self.mask_non_assistant and packed_mask_parts:
                            packed_mask = torch.cat(packed_mask_parts)
                        yield self._yield_padded_sample(
                            packed_tokens, max_seq_len, packed_mask
                        )
                    else:
                        # All pending samples fit, wait for more
                        break

            # End of data - flush any remaining samples (only in packing mode)
            if self._pending_samples and not self.pad_samples:
                # Combine all pending into one final sequence
                packed_tokens = []
                packed_mask_parts = [] if self.mask_non_assistant else None

                for item in self._pending_samples:
                    if self.mask_non_assistant:
                        tokens, mask = item
                    else:
                        tokens = item
                        mask = None

                    if len(packed_tokens) + len(tokens) <= max_seq_len:
                        packed_tokens.extend(tokens)
                        if self.mask_non_assistant and mask is not None:
                            packed_mask_parts.append(mask)
                self._pending_samples = []

                if packed_tokens:
                    packed_mask = None
                    if self.mask_non_assistant and packed_mask_parts:
                        packed_mask = torch.cat(packed_mask_parts)
                    yield self._yield_padded_sample(
                        packed_tokens, max_seq_len, packed_mask
                    )

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                self._sample_idx = 0
                self._pending_samples = []
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        pending_loaded = state_dict.get("pending_samples", [])

        # Restore device from state_dict (for mask tensors)
        # Default to CPU if not specified (backward compatible)
        saved_device = state_dict.get("mask_device", "cpu")
        try:
            device = torch.device(saved_device)
        except RuntimeError:
            # Device not available (e.g., saved on GPU but restoring on CPU-only machine)
            device = torch.device("cpu")
            logger.warning(
                f"Device '{saved_device}' from checkpoint not available, using CPU for mask tensors"
            )

        # When using mask_non_assistant with packing,
        # we may need to reconstruct mask tensors from lists
        if self.mask_non_assistant and pending_loaded:
            self._pending_samples = []
            for item in pending_loaded:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    tokens, mask_data = item
                    # Reconstruct mask tensor from list (stored as 0/1 ints)
                    if mask_data is not None:
                        mask = torch.tensor(mask_data, dtype=torch.bool, device=device)
                    else:
                        mask = None
                    self._pending_samples.append((tokens, mask))
                else:
                    self._pending_samples.append(item)
        else:
            self._pending_samples = pending_loaded

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        # When using mask_non_assistant, pending_samples contains (tokens, mask) tuples.
        # We need to serialize the masks as lists for JSON compatibility.
        pending_to_save = []
        mask_device = None

        if self.mask_non_assistant and self._pending_samples:
            for item in self._pending_samples:
                if isinstance(item, tuple) and len(item) == 2:
                    tokens, mask = item
                    # Convert mask tensor to list for serialization
                    if mask is not None:
                        mask_list = mask.tolist()
                        # Track device for restoration
                        if mask_device is None:
                            mask_device = str(mask.device)
                    else:
                        mask_list = None
                    pending_to_save.append((tokens, mask_list))
                else:
                    pending_to_save.append(item)
        else:
            pending_to_save = self._pending_samples

        _state_dict = {"pending_samples": pending_to_save}

        # Store device info for mask tensor restoration
        if mask_device is not None:
            _state_dict["mask_device"] = mask_device

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            _state_dict["data"] = self._data.state_dict()
        return _state_dict


class HuggingFaceTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        add_bos_eos: bool = True,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor
        self.add_bos_eos = add_bos_eos

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=self.add_bos_eos, add_eos=self.add_bos_eos
                )
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict: dict[str, Any] = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets.

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    pack_samples = getattr(job_config.training, "pack_samples", False)
    pad_samples = getattr(job_config.training, "pad_samples", False)
    add_bos_eos = getattr(job_config.training, "add_bos_eos", True)
    mask_non_assistant = getattr(job_config.training, "mask_non_assistant", False)

    # Validate mutually exclusive options
    if pack_samples and pad_samples:
        raise ValueError(
            "pack_samples and pad_samples are mutually exclusive. "
            "Use pack_samples=True for greedy packing (multiple samples per sequence), "
            "or pad_samples=True for one sample per sequence with padding."
        )

    # Validate mask_non_assistant configuration
    if mask_non_assistant:
        if not (pack_samples or pad_samples):
            raise ValueError(
                "mask_non_assistant=True requires pack_samples=True or pad_samples=True. "
                "The chunking dataset (HuggingFaceTextDataset) does not support loss masking."
            )
        if add_bos_eos:
            logger.warning(
                "mask_non_assistant=True with add_bos_eos=True: BOS/EOS tokens will be added "
                "to pre-formatted Harmony data, which may break the format. Consider setting "
                "add_bos_eos=False for Harmony chat format data."
            )

    if pack_samples or pad_samples:
        if pad_samples:
            logger.info("Using padded dataset (one sample per sequence, remainder padded)")
        else:
            logger.info("Using packed dataset (preserves sample boundaries)")
        if not add_bos_eos:
            logger.info("BOS/EOS tokens disabled (using pre-formatted data)")
        if mask_non_assistant:
            logger.info("Loss masking enabled: only assistant tokens contribute to loss")
        hf_ds = HuggingFacePackedDataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            add_bos_eos=add_bos_eos,
            pad_samples=pad_samples,
            mask_non_assistant=mask_non_assistant,
        )
    else:
        hf_ds = HuggingFaceTextDataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            add_bos_eos=add_bos_eos,
        )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )


def build_text_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets.

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.validation.dataset
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    dataloader_kwargs = {
        **asdict(job_config.validation.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )
