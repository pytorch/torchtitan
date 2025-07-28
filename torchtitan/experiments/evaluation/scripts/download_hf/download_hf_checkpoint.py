# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


import os
# os.environ["HF_TOKEN"] = ""  # Uncomment and set your Hugging Face token if needed
# os.environ["HF_HOME"] = ""  # Uncomment and set your Hugging Face cache directory if needed
import argparse

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM


def main(args):
    model_name = args.model_name
    print(f"Downloading model {model_name} from Hugging Face Hub...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=os.environ.get("HF_HOME", None),
        use_auth_token=os.environ.get("HF_TOKEN", None),
        # torch_dtype=torch.float16,
        # device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model {model_name} downloaded successfully.")

    # NOTE: For ease of use, move downloaded files to the desired directory.
    # If you don't set HF_HOME, `model.safetensors` will be saved in the default cache directory,
    # which is usually `~/.cache/huggingface/hub/models--<model_name>/snapshots/<commit_hash>/`.


if __name__ == "__main__":
    """
    Run this script to download a model from Hugging Face Hub.

    Usage:
        python download_hf_checkpoint.py --model_name <model_name>
    
    Example:
        python download_hf_checkpoint.py --model_name meta-llama/Llama-3.2-1B
    
    Note:
        - Ensure you have set the environment variables HF_TOKEN and HF_HOME if needed.
        - The model will be downloaded to the directory specified by HF_HOME.
    
    """
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face Hub.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Name of the model to download from Hugging Face Hub.",
    )
    args = parser.parse_args()
    main(args)