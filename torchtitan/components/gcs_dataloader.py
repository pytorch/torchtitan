# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from dataflux_pytorch import dataflux_iterable_dataset

# if TYPE_CHECKING:
#     import torchtitan.protocols.train_spec as train_spec_module
#     from torchtitan.config import JobConfig


def build_gcs_dataloader(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: train_spec_module.BaseTokenizer,
) -> torch.utils.data.DataLoader:
    """
    Builds a PyTorch DataLoader for a dataset stored in Google Cloud Storage.

    This function uses the dataflux-pytorch library to stream data from GCS
    as an IterableDataset.

    Args:
        job_config: The main job configuration.
        dp_world_size: The world size of the data parallel group.
        dp_rank: The rank of the current process in the data parallel group.
        tokenizer: The tokenizer to use for preprocessing.

    Returns:
        A torch.utils.data.DataLoader instance.
    """
    gcs_config = job_config.gcs_dataset
    print(f"Connecting to GCS: gs://{gcs_config.bucket_name}/{gcs_config.data_prefix}")

    iterable_dataset = dataflux_iterable_dataset.DataFluxIterableDataset(
        project_name=gcs_config.project_id,
        bucket_name=gcs_config.bucket_name,
        config=dataflux_iterable_dataset.Config(
            prefix=gcs_config.data_prefix,
            disable_compose=True,
        ),
    )

    # The GCS dataset yields raw bytes. We need to decode, tokenize, and format it.
    def collate_fn(batch_of_bytes):
        # Since batch_size=1, batch_of_bytes is a list with one element: [b'...']
        try:
            # An empty file or a file that cannot be decoded could cause an error.
            if not batch_of_bytes or not batch_of_bytes[0]:
                return None
            text = batch_of_bytes[0].decode("utf-8")
        except (UnicodeDecodeError, IndexError):
            return None # Returning None will cause the DataLoader to skip this batch.

        # Tokenize the text
        tokens = tokenizer.encode(text, bos=True, eos=True)
        tokens = torch.tensor(tokens, dtype=torch.long)

        # Create input and labels for language modeling
        inputs = tokens[:-1]
        labels = tokens[1:]

        # The trainer expects a batch of (input_dict, labels)
        # Here we create a "batch" of size 1
        return [{"input": inputs.unsqueeze(0)}, labels.unsqueeze(0)]

    return torch.utils.data.DataLoader(
        iterable_dataset,
        batch_size=job_config.training.local_batch_size,
        collate_fn=collate_fn,
    )