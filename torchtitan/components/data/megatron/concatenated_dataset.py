# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from .gpt_dataset import GPTDataset


class ConcatenatedDataset(torch.utils.data.ConcatDataset):
    """A genuine concatenation of a set of MegatronDataset instances, used in place of
    BlendedDataset when no explicit sampling weights are given: samples are drawn from each
    dataset in turn rather than via a weighted random interleaving

    Args:
        datasets (list[MegatronDataset]): The MegatronDataset instances to concatenate. Each is
        trimmed down to its own requested `num_samples`, since a MegatronDataset is built to
        contain at least that many samples but often more (a dataset isn't truncated once a
        single epoch already covers the request), which would otherwise throw off the total
        concatenated length
    """

    def __init__(
        self, datasets: list[GPTDataset], caching_allowed: bool
    ) -> ConcatenatedDataset:
        assert all(map(lambda _: type(_) == type(datasets[0]), datasets))
        super().__init__(
            [
                torch.utils.data.Subset(dataset, range(dataset.num_samples))
                for dataset in datasets
            ]
        )
