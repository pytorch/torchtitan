from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer


class PathDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        split: str
        shuffle_size: int
        min_mixing: float
        num_writers: int
        num_readers: int
        fps: int
        pipeline_dir: str
        plan_only: bool
        limit: int | None
        n_frames: int
        rgb: bool
        unvision: bool

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        validation_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        del tokenizer, seq_len, snapshot_every_n_steps, kwargs
        from gigashuffle import DataloaderConfig
        from xx.training.lib.dataloader import DataLoader
        from xx.training.path.config import DatasetConfig as XXPathDatasetConfig
        from xx.training.path.dataloader import get_dataset

        self.config = config
        self.local_batch_size = local_batch_size
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.local_rank = int(os.environ.get("LOCAL_RANK", dp_rank))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", dp_world_size))
        node_rank = int(os.environ.get("GROUP_RANK", dp_rank // self.local_world_size))
        run_id = os.environ.get("REPORTERV2_TRAINING_ID") or "path"
        val = config.split != "train"
        val_shuffle_size = local_batch_size * validation_steps * self.local_world_size
        shuffle_size = val_shuffle_size if val else config.shuffle_size

        xx_config = XXPathDatasetConfig(
            bs=local_batch_size,
            shuffle_size=str(config.shuffle_size),
            val_shuffle_size=str(val_shuffle_size),
            min_mixing=config.min_mixing,
            num_writers=config.num_writers,
            num_readers=config.num_readers,
            fps=config.fps,
            pipeline_dir=config.pipeline_dir,
            plan_only=config.plan_only,
            limit=config.limit,
            n_frames=config.n_frames,
            rgb=config.rgb,
            unvision=config.unvision,
        )
        dataset = get_dataset(config.dataset, xx_config, val, self.local_rank)
        self.dataset = dataset
        loader_config = DataloaderConfig(
            bs=local_batch_size,
            shuffle_size=shuffle_size,
            min_mixing=1 if val else config.min_mixing,
            num_writers=1 if val else config.num_writers,
            num_readers=1 if val else config.num_readers,
            fill_once=val,
            local_rank=self.local_rank,
            global_rank=dp_rank,
            local_world_size=self.local_world_size,
            global_world_size=dp_world_size,
            queue_name=f"{run_id}-{config.split}-node{node_rank}",
        )
        self.loader = DataLoader(dataset, loader_config)
        self._iterator: Any | None = None

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        iterator = iter(self.loader)
        self._iterator = iterator
        try:
            for inputs, targets in iterator:
                yield inputs, targets
        finally:
            iterator.close()
            if self._iterator is iterator:
                self._iterator = None

    def close(self) -> None:
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None
        self.loader._shutdown_workers()

    def state_dict(self) -> dict[str, int]:
        return {}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        return
