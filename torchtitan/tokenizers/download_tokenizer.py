# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from transformers import AutoTokenizer


def hf_download(repo_id: str, local_dir: str) -> None:
    # Add the tokenizer name to the path
    tokenizer_dir = local_dir + repo_id.split("/")[-1]

    # Don't download if exists
    if os.path.isdir(tokenizer_dir) and os.listdir(tokenizer_dir):
        print(f"The directory {tokenizer_dir} exists and it's not empty.")  
        return

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=False)

    # Save the tokenizer
    tokenizer.save_pretrained(tokenizer_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download tokenizer from HuggingFace.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="yerevann/chemlactica-125m",
        help="Repository ID to download from. default to chemlactica-125m",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="torchtitan/tokenizers/",
        help="local directory to save the tokenizer.model",
    )

    args = parser.parse_args()
    hf_download(args.repo_id, args.local_dir)