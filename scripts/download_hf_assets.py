# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fnmatch import fnmatch
from typing import Optional

from requests.exceptions import HTTPError
from tqdm import tqdm


def download_hf_assets(
    repo_id: str,
    local_dir: str,
    asset_types: str | list[str],
    download_all: bool = False,
    hf_token: Optional[str] = None,
    additional_patterns: Optional[list] = None,
) -> None:
    """
    Download relevant files from HuggingFace Hub repository.

    This function recursively searches through the HuggingFace Hub repository
    and downloads all related files

    Asset types:
    - tokenizer:
        - tokenizer.json - Modern HuggingFace tokenizers (complete definition)
        - tokenizer_config.json - Tokenizer configuration and metadata
        - tokenizer.model - SentencePiece model files (Llama, T5, etc.)
        - vocab.txt - Plain text vocabulary files
        - vocab.json - JSON vocabulary files
        - merges.txt - BPE merge rules (GPT-2, RoBERTa style)
        - special_tokens_map.json - Special token mappings
    - safetensors
        - *.safetensors - Modern Huggingface model weights format for fast loading
        - model.safetensors.index.json - Contains mapping from hf fqn to file name
    - index
        - model.safetensors.index.json - Contains mapping from hf fqn to file name
    - config
        - config.json - Defines the model architecture
        - generation_config.json - Defines the model architecture params needed for generation

    Args:
        repo_id (str): HuggingFace repository ID (e.g., meta-llama/Llama-3.1-8B")
        local_dir (str): Local directory to save tokenizer files. A subdirectory
                        named after the model will be created automatically.
        asset_types (list[str]): List of the asset types to download
        hf_token (Optional[str]): HuggingFace API token for accessing private repositories.
                                 Required for gated models like Llama.
        additional_patterns (Optional[list]): Additional file patterns to search for and download
                                          from the HuggingFace Hub repository.
        download_all (bool): If True, download all files from the repository
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

    ASSET_PATTERNS = {
        "tokenizer": [
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ],
        "safetensors": ["*.safetensors", "*model.safetensors.index.json"],
        "index": ["*model.safetensors.index.json"],
        "config": ["config.json", "generation_config.json"],
    }

    if isinstance(asset_types, str):
        asset_types = [asset_types]

    if download_all:
        print("Downloading all files from repository...")
        files_found = list_repo_files(repo_id=repo_id, token=hf_token)
    else:
        total_patterns = []
        for asset_type in asset_types:
            if asset_type in ASSET_PATTERNS:
                total_patterns.extend(ASSET_PATTERNS[asset_type])
            else:
                raise ValueError(
                    "Unknown asset type {}. Available uses: --asset {} \n".format(
                        asset_type, " ".join(ASSET_PATTERNS.keys())
                    ),
                    "Or specify exact patterns to download. Example: --additional_patterns '*.safetensors' README.md '*.json' \n",
                    "Or use --all to download all files",
                )

        # Add additional patterns if provided
        if additional_patterns:
            total_patterns.extend(additional_patterns)
            asset_types.append("additional_patterns")
            ASSET_PATTERNS["additional_patterns"] = additional_patterns

        def should_download(patterns: list[str], filename: str) -> bool:
            """Check if a file matches a pattern to be downloaded."""
            basename = os.path.basename(filename)
            for pattern in patterns:
                pattern_lower = pattern.lower()

                # Exact name match
                if basename == pattern_lower:
                    return True
                # Do wildcard match if wildcards are in pattern
                if "*" in pattern_lower or "?" in pattern_lower:
                    if fnmatch(basename, pattern_lower):
                        return True
            return False

        try:
            # Get list of available files in the repo
            print(f"Scanning repository {repo_id} for files...")
            available_files = list_repo_files(repo_id=repo_id, token=hf_token)

            # Filter for requested asset files
            files_found = [
                f for f in available_files if should_download(total_patterns, f)
            ]

            # Check each asset type individually to see if files were not found
            for asset_type in asset_types:
                if asset_type in ASSET_PATTERNS:
                    asset_patterns = ASSET_PATTERNS[asset_type]
                    matches_found = False
                    for f in available_files:
                        if should_download(asset_patterns, f):
                            matches_found = True
                            break

                    if not matches_found:
                        print(
                            f"Warning: No matching files found for asset_type '{asset_type}' in {repo_id}"
                        )

            if not files_found:
                print(f"Warning: No matching files found in {repo_id}")
                print(f"Available files: {available_files[:10]}...")
                return

        except HTTPError as e:
            if e.response and e.response.status_code == 401:
                print(
                    "You need to pass a valid `--hf_token=...` to download private checkpoints."
                )
            raise e

    print(f"Found {len(files_found)} files:")
    for f in files_found:
        print(f"  - {f}")

    downloaded_files = []
    missed_files = []

    # Download files with progress bar
    with tqdm(total=len(files_found), desc="Downloading files", unit="file") as pbar:
        for filename in files_found:
            try:
                pbar.set_description(f"Downloading {os.path.basename(filename)}")

                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=model_dir,
                    token=hf_token,
                )
                downloaded_files.append(filename)
                pbar.update(1)

            except HTTPError as e:
                if e.response and e.response.status_code == 404:
                    print(f"File {filename} not found, skipping...")
                    missed_files.append(filename)
                    pbar.update(1)
                    continue
                else:
                    raise e

    if downloaded_files:
        print(
            f"\nSuccessfully downloaded {len(downloaded_files)} files to: {model_dir}"
        )
    if missed_files:
        print(f"Warning: Some files could not be downloaded: \n{missed_files}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download files from HuggingFace Hub. "
        "Automatically detects and downloads files that match the specified file-types to download. "
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID to download from (e.g., 'meta-llama/Llama-3.1-8B', 'deepseek-ai/DeepSeek-V3')",
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
        default="assets/hf/",
        help="Local directory to save hf asset files (default: assets/hf/)",
    )
    parser.add_argument(
        "--assets",
        type=str,
        nargs="+",
        default=[],
        help="Asset types to download: tokenizer, safetensors, index, config",
    )
    parser.add_argument(
        "--additional_patterns",
        type=str,
        nargs="+",
        default=[],
        help="Additional file patterns to search for and download from the HuggingFace Hub repository",
    )

    parser.add_argument(
        "--all", action="store_true", default=False, help="Download all files in repo"
    )

    args = parser.parse_args()
    if not args.all and not args.assets and not args.additional_patterns:
        parser.error(
            "At least one of --all, --assets or --additional_patterns must be specified."
        )

    download_hf_assets(
        args.repo_id,
        args.local_dir,
        args.assets,
        args.all,
        args.hf_token,
        args.additional_patterns,
    )
