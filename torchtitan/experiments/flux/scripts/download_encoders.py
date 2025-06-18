# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os

from huggingface_hub import hf_hub_download, snapshot_download


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download tokenizer from HuggingFace.")
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HuggingFace API token"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="torchtitan/experiments/flux/assets/encoders/",
        help="local directory to save the autoencoder",
    )

    args = parser.parse_args()
    hf_hub_download("black-forest-labs/FLUX.1-schnell", filename="ae.safetensors", local_dir=os.path.join(args.local_dir, "autoencoder"), token=args.hf_token)
    snapshot_download("google/t5-v1_1-xxl", local_dir=os.path.join(args.local_dir, "t5"), token=args.hf_token, ignore_patterns="tf_model.h5")
    snapshot_download("openai/clip-vit-large-patch14", local_dir=os.path.join(args.local_dir, "clip"), token=args.hf_token, ignore_patterns=["*.safetensors", "*.msgpack", "tf_model.h5"])
