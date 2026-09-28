# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal integration-test dataset helpers."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import grain.python as grain

from torchtitan.components.data import GrainDataLoader, SingleDatasetConfig
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.hf_datasets.multimodal.mm_datasets import MultiModalProcessor
from torchtitan.trainer import Trainer


def _set_image_presence(
    *,
    sample: dict[str, Any],
    sample_processor: Callable[..., dict[str, Any] | None],
    has_image: bool,
    **kwargs: Any,
) -> dict[str, Any] | None:
    if not has_image:
        sample = {**sample, "jpg": None}
    return sample_processor(sample=sample, **kwargs)


@dataclass(frozen=True, kw_only=True, slots=True)
class DPRankImagePresenceDatasetConfig:
    """Make even DP ranks text-only while odd DP ranks retain images."""

    dataset: SingleDatasetConfig

    def build(
        self,
        *,
        context: DatasetBuildContext,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> grain.MapDataset | grain.IterDataset:
        processor = self.dataset.processor
        if not isinstance(processor, MultiModalProcessor.Config):
            raise TypeError(
                "DPRankImagePresenceDatasetConfig requires MultiModalProcessor.Config"
            )
        sample_processor = partial(
            _set_image_presence,
            sample_processor=processor.sample_processor,
            has_image=dataset_iteration_policy.dp_rank % 2 == 1,
        )
        dataset = replace(
            self.dataset,
            processor=replace(processor, sample_processor=sample_processor),
        )
        return dataset.build(
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )


def set_rank_conditional_image_presence(config: Trainer.Config) -> None:
    """Make even DP ranks text-only in a multimodal integration recipe."""
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    assert isinstance(config.dataloader.dataset, SingleDatasetConfig)
    config.dataloader.dataset = DPRankImagePresenceDatasetConfig(
        dataset=config.dataloader.dataset
    )
