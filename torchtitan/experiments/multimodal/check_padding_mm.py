# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import click

from mm_dataset import build_mm_dataloader
from tokenizer.tiktoken import build_tiktoken_tokenizer

from torchtitan.config_manager import ConfigManager
from torchtitan.tools.logging import init_logger


@click.command()
@click.option("--dataset", default="OBELICS")
@click.option("--batch-size", default=4)
@click.option("--seq-len", default=4096)
@click.option("--tokenizer-path", required=True)
@click.option("--dp-rank", default=0)
@click.option("--dp-world-size", default=2)
@click.option("--batch-number", default=4)
def main(
    dataset: str,
    batch_size: int,
    seq_len: int,
    tokenizer_path: str,
    dp_rank: int,
    dp_world_size: int,
    batch_number: int,
):
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        [
            "--training.dataset",
            dataset,
            "--training.batch_size",
            str(batch_size),
            "--training.seq_len",
            str(seq_len),
            "--model.tokenizer_path",
            tokenizer_path,
        ]
    )
    tokenizer = build_tiktoken_tokenizer(config)
    dl = build_mm_dataloader(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        job_config=config,
    )
    dl_iter = iter(dl)

    for _ in range(batch_number):
        batch = next(dl_iter)

    # Analyze Batch
    # input_ids
    total_input_ids = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
    total_non_padding_tokens = total_input_ids - int(
        (batch["input_ids"] == 128004).sum()
    )
    total_padding_tokens = total_input_ids - total_non_padding_tokens
    print(f"Padding tokens in each sample: {(batch['input_ids'] == 128004).sum(dim=1)}")
    print(
        f"Unpadded tokens: {total_non_padding_tokens}, Total tokens in batch: {total_input_ids}"
    )
    print(
        f"Padded text tokens: {total_padding_tokens}, {(total_padding_tokens) / total_input_ids * 100:.2f}%"
    )
    print(80 * "#")
    # Images
    padded_images = 0
    padded_tiles = 0
    for sample in batch["encoder_input"]["images"]:
        for image in sample:
            if int(image.sum()) == 0:
                padded_images += 1
            for tile in image:
                if int(tile.sum()) == 0:
                    padded_tiles += 1

    total_images = (
        batch["encoder_input"]["images"].shape[0]
        * batch["encoder_input"]["images"].shape[1]
    )

    print(
        f"Unpadded images: {total_images - padded_images}, Total images in batch: {total_images}"
    )
    print(
        f'Padded images: {padded_images}, {padded_images / total_images * 100:.2f}% (Each image with shape {list(batch["encoder_input"]["images"][0, 0].shape)})'  # noqa: B950
    )
    print(80 * "#")
    # Tiles
    total_number_of_tiles = total_images * batch["encoder_input"]["images"].shape[2]

    print(
        f"Unpadded number of tiles: {total_number_of_tiles - padded_tiles}, Total number of tiles: {total_number_of_tiles}"
    )
    print(
        f'Padded tiles: {padded_tiles}, {padded_tiles / total_number_of_tiles * 100:.2f}% (Each with shape {list(batch["encoder_input"]["images"][0, 0, 0].shape)})'  # noqa: B950
    )
    print(80 * "#")


if __name__ == "__main__":
    main()
