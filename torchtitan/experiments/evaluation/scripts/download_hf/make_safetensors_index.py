# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


import os
import json
import argparse

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM
from safetensors import safe_open


def main(args):
    fpath = args.fpath
    if os.path.exists(os.path.join(fpath, "model.safetensors.index.json")):
        print(f"Index file already exists at {fpath}. Skipping index creation.")
        return

    assert os.path.exists(os.path.join(fpath, "model.safetensors")), \
        f"Model file not found at {os.path.join(fpath, 'model.safetensors')}"

    # Load the model state_dict from the safetensors file
    state_dict = {}
    with safe_open(os.path.join(fpath, "model.safetensors"), framework="pt", device="cuda") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)  # loads the full tensor given a key

    index_json = {
        "metadata": {},
        "weight_map": {},
    }
    for k in state_dict.keys():
        index_json["weight_map"][k] = "model.safetensors"
    
    # Save the index.json file
    index_path = os.path.join(os.path.dirname(fpath), "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index_json, f, indent=4)

    print(f"Index file created at {index_path}.")
    print("You can now use this model with safetensors format.")
    

if __name__ == "__main__":
    """
    Run this script to create an index file for a safetensors model.
    If the model size is small, there will be no safetensors file.
    In that case, we need to create the index file from the model state_dict,
    where all weights are stored in the model.safetensors file.

    Usage:
        python make_safetensors_index.py --fpath <path_to_model_directory>

    """
    parser = argparse.ArgumentParser(description="Make safetensors index file.")
    parser.add_argument(
        "--fpath",
        type=str,
        help="File path where the model is saved.",
    )
    args = parser.parse_args()

    main(args)