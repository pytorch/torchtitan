# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import List

from datasets import Dataset, load_dataset
from torchvision.transforms import functional as F


def filter_dataset_by_indices(
    dataset: Dataset, indices_to_remove: List[int]
) -> Dataset:
    """
    Filter dataset by removing samples at specified indices.

    Args:
        dataset: HuggingFace Dataset object
        indices_to_remove: List of indices to remove from the dataset

    Returns:
        Filtered Dataset object
    """
    # Get all indices in the dataset
    all_indices = set(range(len(dataset)))

    # Remove the specified indices
    indices_to_remove_set = set(indices_to_remove)
    indices_to_keep = list(all_indices - indices_to_remove_set)

    # Sort to maintain order
    indices_to_keep.sort()

    # Filter the dataset
    filtered_dataset = dataset.select(indices_to_keep)

    return filtered_dataset


def filter_low_resolution_batch(batch, output_size):
    """Filter out low resolution images in batch."""
    return [
        img.size[0] >= output_size and img.size[1] >= output_size
        for img in batch["jpg"]
    ]


def resize_images_batch(batch, output_size):
    """Resize and center crop images in batch."""
    processed_images = []
    for img in batch["jpg"]:
        resized_img = F.resize(
            img, output_size, interpolation=F.InterpolationMode.BICUBIC
        )
        cropped_img = F.center_crop(resized_img, (output_size, output_size))
        processed_images.append(cropped_img)
    return {"png": processed_images}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find problematic samples in a dataset"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of worker processes (default: 16)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of samples per batch (default: 1000)",
    )
    parser.add_argument("--input_dir", type=str, help="Input directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--filter_file",
        type=str,
        default="torchtitan/experiments/flux/scripts/problematic_indices.txt",
        help="Filter file (default: torchtitan/experiments/flux/scripts/problematic_indices.txt)",
    )
    parser.add_argument(
        "--output_size", type=int, default=256, help="Output size (default: 256)"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=1099776,
        help="Subset size (default: 1099776)",
    )
    args = parser.parse_args()

    samples_to_filter = [i.strip() for i in open(args.filter_file, "r").readlines()]
    samples_to_filter = [int(i) for i in samples_to_filter if i]
    dataset = load_dataset(
        "webdataset",
        data_dir=args.input_dir,
        split="train",
        num_proc=args.num_workers,
    )
    filtered_dataset = filter_dataset_by_indices(dataset, samples_to_filter)

    # filter low resolution images using batched processing
    filtered_dataset = filtered_dataset.filter(
        lambda batch: filter_low_resolution_batch(batch, args.output_size),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_workers,
    ).take(args.subset_size)

    # resize remaining images using batched processing
    filtered_dataset = filtered_dataset.map(
        lambda batch: resize_images_batch(batch, args.output_size),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_workers,
    )

    # save dataset
    filtered_dataset.save_to_disk(args.output_dir, num_proc=args.num_workers)
