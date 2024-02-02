# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from torchtrain.datasets.alpaca import AlpacaDataset


dataset_cls_map = {
    "alpaca": AlpacaDataset,
}
