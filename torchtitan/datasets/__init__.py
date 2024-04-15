# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from torchtitan.datasets.hf_datasets import build_hf_data_loader
from torchtitan.datasets.tokenizer import create_tokenizer

__all__ = [
    "build_hf_data_loader",
    "create_tokenizer",
]

dataloader_fn = {
    "alpaca": build_hf_data_loader,
    "c4": build_hf_data_loader,
    "openwebtext": build_hf_data_loader,
}
