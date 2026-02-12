# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torchtitan.components.checkpoint import ModelWrapper


@torch.inference_mode()
def convert_from_hf(input_dir, output_dir, model_name, model_flavor):
    # initialize model to allocate memory for state dict
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model

    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    # pyrefly: ignore[bad-instantiation, not-callable]
    sd_adapter = model_spec.state_dict_adapter(model_config, None)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from HF to DCP safetensors format, but sd_adapter is not provided."
    # get state dict in tt format with allocated memory
    state_dict = model._get_state_dict()
    # convert empty state dict to hf format so that hf weights can be loaded into it
    hf_state_dict = sd_adapter.to_hf(state_dict)
    dcp.load(
        hf_state_dict,
        storage_reader=HuggingFaceStorageReader(path=input_dir),
    )
    # convert state dict format back hf->tt and save
    state_dict = sd_adapter.from_hf(hf_state_dict)
    dcp.save(
        state_dict,
        checkpoint_id=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to DCP format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with HF checkpoint"
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for DCP.")
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    args = parser.parse_args()

    convert_from_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
    )
