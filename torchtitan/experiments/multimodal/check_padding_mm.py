# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.datasets.tokenizer.tiktoken import build_tiktoken_tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import init_logger

from mm_dataset import build_mm_dataloader

PATH_TO_TOKENIZER = "/iopsstor/scratch/cscs/asolergi/torchtitan/tokenizer.model"
BATCH_NUMBER = 4


def main():
    init_logger()
    job_config = JobConfig()
    job_config.parse_args(["--training.dataset", "OBELICS",
                           "--training.batch_size", "16",
                           "--training.seq_len", "2048",
    "--model.tokenizer_path", PATH_TO_TOKENIZER])
    tokenizer = build_tiktoken_tokenizer(job_config)
    dl = build_mm_dataloader(dp_world_size=2, dp_rank=0, tokenizer=tokenizer, job_config=job_config)
    dl_iter = iter(dl)
    for _ in range(BATCH_NUMBER):
        batch = next(dl_iter)

    # Analyze Batch
    ## input_ids
    total_input_ids = sum(batch["token_len"].tolist())
    input_ids_pad_length = max(batch["token_len"].tolist())
    total_tokens_in_batch = input_ids_pad_length * job_config.training.batch_size
    total_input_ids_padded_tokens = sum(
        (batch["token_len"] - input_ids_pad_length) * -1
    )
    print(
        f"Unpadded tokens: {total_input_ids}, Total tokens in batch: {total_tokens_in_batch}"
    )
    print(
        f"Padded text tokens: {total_input_ids_padded_tokens}, {total_input_ids_padded_tokens/total_tokens_in_batch*100:.2f}%"
    )
    print(40 * "#")
    ## image_ids
    total_images = sum(batch["image_len"].tolist())
    image_pad_length = max(batch["image_len"].tolist())
    total_images_in_batch = image_pad_length * job_config.training.batch_size
    total_images_padded_tokens = sum((batch["image_len"] - image_pad_length) * -1)
    print(
        f"Unpadded images: {total_images}, Total images in batch: {total_images_in_batch}"
    )
    print(
        f'Padded images: {total_images_padded_tokens}, {total_images_padded_tokens/total_images_in_batch*100:.2f}% (Each image with shape {list(batch["encoder_input"]["images"][0,0].shape)})'
    )
    print(40 * "#")
    # Tiles
    total_number_of_tiles = sum([sum(sample) for sample in batch["tile_len"]])
    print(
        f"Unpadded number of tiles: {total_number_of_tiles}, Total number of tiles: {total_images_in_batch*4}"
    )
    print(
        f'Padded tiles: {total_images_in_batch*4-total_number_of_tiles}, {(1-(total_number_of_tiles/(total_images_in_batch*4-total_number_of_tiles)))*100:.2f}% (Each with shape {list(batch["encoder_input"]["images"][0,0,0].shape)})'
    )
    print(40 * "#")
    # CrossAttentionMask
    original_cross_attention_mask_elements = (
        total_number_of_tiles * 1025 * total_input_ids
    )  # NOTE(tj.solergibert) We have 1024+1 image tokens per tile
    print(
        f"Unpadded cross attention mask elements: {original_cross_attention_mask_elements}, Total cross attention mask elements: {total_images_in_batch*4*1025*total_tokens_in_batch}"
    )  # TODO(tj.solergibert) Each element is a `bool`
    print(
        f"Padded cross attention mask elements: {total_images_in_batch*4*1025*total_tokens_in_batch-original_cross_attention_mask_elements}, {100*((total_images_in_batch*4*1025*total_tokens_in_batch-original_cross_attention_mask_elements)/(total_images_in_batch*4*1025*total_tokens_in_batch)):.2f}%"
    )


if __name__ == "__main__":
    main()
