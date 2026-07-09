from __future__ import annotations

from xx.datasets.constants import BASE_DIR_GT_10M, DEFAULT_10M_TRAIN_LIST

from .dataset import WorldModelDataLoader
from .model_config import COMPRESSOR_MODEL, LATENT_CHANNELS, LATENT_SIZE

IMAGE_SIZE = (128, 256)


def _dataloader_config(
    *,
    split: str,
    dataset: str = DEFAULT_10M_TRAIN_LIST,
    dataset_path: str | None = None,
    shuffle_size: int = 50_000,
    min_mixing: float = 0.5,
    num_writers: int = 2,
    num_readers: int = 4,
    fill_once: bool = False,
    base_dir: str = BASE_DIR_GT_10M,
    feature_dir: str | None = None,
    compressor_model: str = COMPRESSOR_MODEL,
    in_channels: int = LATENT_CHANNELS,
    latent_size: tuple[int, int] = LATENT_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
    context_size_frames: int = 10,
    future_size_frames: int = 5,
    max_future_frames: int = 50,
    inference_prefill_frames: int = 14,
    fps: int = 5,
    train_skip: int = 40,
    val_skip: int = 800,
    nan_engaged_plans: bool = False,
    limit: int | None = None,
    mock_data: bool = False,
    mock_segment_batch_size: int = 8,
) -> WorldModelDataLoader.Config:
    return WorldModelDataLoader.Config(
        dataset=dataset,
        dataset_path=dataset_path,
        split=split,
        shuffle_size=shuffle_size,
        min_mixing=min_mixing,
        num_writers=num_writers,
        num_readers=num_readers,
        fill_once=fill_once,
        base_dir=base_dir,
        feature_dir=feature_dir,
        compressor_model=compressor_model,
        in_channels=in_channels,
        latent_size=latent_size,
        image_size=image_size,
        context_size_frames=context_size_frames,
        future_size_frames=future_size_frames,
        max_future_frames=max_future_frames,
        inference_prefill_frames=inference_prefill_frames,
        fps=fps,
        train_skip=train_skip,
        val_skip=val_skip,
        nan_engaged_plans=nan_engaged_plans,
        limit=limit,
        mock_data=mock_data,
        mock_segment_batch_size=mock_segment_batch_size,
    )
