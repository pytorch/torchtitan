# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from requests.exceptions import HTTPError


def hf_download(
    repo_id: str, hf_file_path: str, local_dir: str, hf_token: Optional[str] = None
) -> None:
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=hf_file_path,
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
        "--tokenizer",
        action="store_true",
        help="Set this flag to download tokenizer from HuggingFace",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="original",
        help="the tokenizer.model path relative to repo_id",
    )
    parser.add_argument(
        "--autoencoder",
        action="store_true",
        help="Set this flag to download autoencoder from HuggingFace",
    )
    parser.add_argument(
        "--autoencoder_path",
        type=str,
        default="ae.saftensors",
        help="the autoencoder path relative to repo_id",
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
    if args.tokenizer:
        tokenizer_path = (
            f"{args.tokenizer_path}/tokenizer.model"
            if args.tokenizer_path
            else "tokenizer.model"
        )
        hf_download(args.repo_id, tokenizer_path, args.local_dir, args.hf_token)
    elif args.autoencoder:
        hf_download(args.repo_id, args.autoencoder_path, args.local_dir, args.hf_token)
    else:
        print(
            "You need to specify either --tokenizer or --autoencoder to download tokenizer or autoencoder."
        )
