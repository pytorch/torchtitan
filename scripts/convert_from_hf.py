# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torchtitan.protocols.train_spec as train_spec_module
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.config_manager import ConfigManager


@torch.inference_mode()
def convert_from_hf(input_dir, output_dir, model_name, model_flavor):
    # initialize model to allocate memory for state dict
    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]

    config_manager = ConfigManager()
    config = config_manager.parse_args(
        [
            "--model.tokenizer-path",
            "./assets/tokenizer/Llama-3.1-8B",
        ]
    )
    tokenizer = build_hf_tokenizer(config)
    model_args.update_from_config(config, tokenizer)
    with torch.device("cpu"):
        model = train_spec.model_cls(model_args)
    model = ModelWrapper(model)

    sd_adapter = train_spec.state_dict_adapter
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from HF to DCP safetensors format, but sd_adapter is not provided."
    # get state dict in tt format with allocated memory
    state_dict = model._get_state_dict()
    # convert empty state dict to hf format so that hf weights can be loaded into it
    hf_state_dict = sd_adapter.to_hf(state_dict, model_args)
    dcp.load(
        hf_state_dict,
        storage_reader=HuggingFaceStorageReader(path=input_dir),
    )
    # convert state dict format back hf->tt and save
    state_dict = sd_adapter.from_hf(hf_state_dict, model_args)
    dcp.save(
        state_dict,
        checkpoint_id=output_dir,
    )


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert Llama weights to DCP format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with original Llama weights."
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
