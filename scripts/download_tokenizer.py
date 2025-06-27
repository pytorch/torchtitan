# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from requests.exceptions import HTTPError


def download_hf_tokenizer_files(
    repo_id: str,
    local_dir: str,
    hf_token: Optional[str] = None,
    additional_patterns: Optional[list] = None,
) -> None:
    """
    Download relevant tokenizer files from HuggingFace Hub repository.

    This function recursively searches through the HuggingFace Hub repository
    and downloads all tokenizer-related files to enable tokenizer
    loading with the build_hf_tokenizer() function.

    Files downloaded:
    - tokenizer.json - Modern HuggingFace tokenizers (complete definition)
    - tokenizer_config.json - Tokenizer configuration and metadata
    - tokenizer.model - SentencePiece model files (Llama, T5, etc.)
    - vocab.txt - Plain text vocabulary files
    - vocab.json - JSON vocabulary files
    - merges.txt - BPE merge rules (GPT-2, RoBERTa style)
    - special_tokens_map.json - Special token mappings

    Args:
        repo_id (str): HuggingFace repository ID (e.g., "meta-llama/Meta-Llama-3.1-8B")
        local_dir (str): Local directory to save tokenizer files. A subdirectory
                        named after the model will be created automatically.
        hf_token (Optional[str]): HuggingFace API token for accessing private repositories.
                                 Required for gated models like Llama.
        additional_patterns (Optional[list]): Additional file patterns to search for and download
                                          from the HuggingFace Hub repository.
    """
    import os

    from huggingface_hub import hf_hub_download, list_repo_files

    # Extract model name from repo_id (part after "/")
    if "/" not in repo_id:
        raise ValueError(
            f"Invalid repo_id format: '{repo_id}'. Expected format: 'organization/model-name'"
        )
    model_name = repo_id.split("/")[-1].strip()
    model_dir = os.path.join(local_dir, model_name)

    # Tokenizer file patterns to match (case-insensitive)
    tokenizer_patterns = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]

    # Add additional files if provided
    if additional_patterns:
        tokenizer_patterns.extend(additional_patterns)

    def is_tokenizer_file(filename: str) -> bool:
        """Check if a file is a tokenizer-related file."""
        filename_lower = filename.lower()
        basename = os.path.basename(filename_lower)

        # Check exact matches
        if basename in [pattern.lower() for pattern in tokenizer_patterns]:
            return True

        return False

    try:
        # Get list of available files in the repo
        print(f"Scanning repository {repo_id} for tokenizer files...")
        available_files = list_repo_files(repo_id=repo_id, token=hf_token)

        # Filter for tokenizer files
        tokenizer_files_found = [f for f in available_files if is_tokenizer_file(f)]

        if not tokenizer_files_found:
            print(f"Warning: No tokenizer files found in {repo_id}")
            print(f"Available files: {available_files[:10]}...")
            return

        print(f"Found {len(tokenizer_files_found)} tokenizer files:")
        for f in tokenizer_files_found:
            print(f"  - {f}")

        downloaded_files = []
        for filename in tokenizer_files_found:
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=model_dir,
                    token=hf_token,
                )
                file_path = os.path.join(model_dir, filename)
                print(f"Successfully downloaded {filename} to {file_path}")
                downloaded_files.append(filename)
            except HTTPError as e:
                if e.response.status_code == 404:
                    print(f"File {filename} not found, skipping...")
                    continue
                else:
                    raise e

        if downloaded_files:
            print(
                f"\nSuccessfully downloaded {len(downloaded_files)} tokenizer files to: {model_dir}"
            )
        else:
            print(f"Warning: No tokenizer files could be downloaded from {repo_id}")

    except HTTPError as e:
        if e.response.status_code == 401:
            print(
                "You need to pass a valid `--hf_token=...` to download private checkpoints."
            )
        raise e


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download tokenizer files from HuggingFace Hub. "
        "Automatically detects and downloads common tokenizer files (tokenizer.json, "
        "tokenizer_config.json, tokenizer.model, ...) that work with Tokenizer."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID to download from (e.g., 'meta-llama/Meta-Llama-3.1-8B', 'deepseek-ai/DeepSeek-V3')",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token (required for private repos)",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="assets/tokenizer/",
        help="Local directory to save tokenizer files (default: assets/tokenizer/)",
    )
    parser.add_argument(
        "--additional_patterns",
        type=str,
        nargs="*",
        default=None,
        help="Additional file patterns to search for and download from the HuggingFace Hub repository",
    )

    args = parser.parse_args()
    download_hf_tokenizer_files(
        args.repo_id, args.local_dir, args.hf_token, args.additional_patterns
    )
