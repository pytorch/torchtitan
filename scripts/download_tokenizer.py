# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from requests.exceptions import HTTPError


def hf_download(
    repo_id: str, tokenizer_path: str, local_dir: str, hf_token: Optional[str] = None
) -> None:
    from huggingface_hub import hf_hub_download

    tokenizer_path = (
        f"{tokenizer_path}/tokenizer.model" if tokenizer_path else "tokenizer.model"
    )

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=tokenizer_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            print(
                "You need to pass a valid `--hf_token=...` to download private checkpoints."
            )
        else:
            raise e


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download tokenizer from HuggingFace.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Repository ID to download from. default to Llama-3.1-8B",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="original",
        help="the tokenizer.model path relative to repo_id",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HuggingFace API token"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="assets/tokenizer/",
        help="local directory to save the tokenizer.model",
    )

    args = parser.parse_args()
    hf_download(args.repo_id, args.tokenizer_path, args.local_dir, args.hf_token)
