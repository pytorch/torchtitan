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
from peft import PeftModel
from peft.helpers import check_if_peft_model
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.peft.state_dict_adapter_update import (
    update_state_dict_adapter,
)
from torchtitan.config.job_config import PEFT
from torchtitan.config.manager import ConfigManager


def load_model(train_spec, model_args, peft_config: PEFT):
    with torch.device("cpu"):
        model = train_spec.model_cls(model_args, peft_config)
    model = ModelWrapper(model)
    return model


@torch.inference_mode()
def convert_from_hf(
    input_dir, output_dir, model_name, model_flavor, merge_peft_adapter
):
    if model_name == "flux":
        import torchtitan.experiments.flux  # noqa: F401
    # initialize model to allocate memory for state dict
    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]
    sd_adapter = train_spec.state_dict_adapter(model_args, None)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from HF to DCP safetensors format, but sd_adapter is not provided."
    if check_if_peft_model(input_dir):
        if merge_peft_adapter:
            model = load_model(train_spec, model_args, PEFT())
            peft_model = PeftModel.from_pretrained(input_dir)
            hf_state_dict = peft_model.merge_and_unload().state_dict()
        else:
            model = load_model(
                train_spec, model_args, PEFT(use_lora=True, enable_peft=True)
            )
            peft_model = PeftModel.from_pretrained(input_dir)
            hf_state_dict = peft_model.state_dict()
            sd_adapter.from_hf_map = update_state_dict_adapter(
                sd_adapter.from_hf_map, PEFT(use_lora=True, enable_peft=True)
            )
    else:
        model = load_model(train_spec, model_args, PEFT())
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
    parser.add_argument("--config_path", type=Path, help="Path to config file.")
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    parser.add_argument(
        "--merge_peft_adapter",
        action="store_true",
        help="Merge PEFT adapter into the checkpoint.",
    )
    args = parser.parse_args()
    if args.config_path is not None:
        config_manager = ConfigManager()
        config = config_manager.parse_args(["--job.config_file", args.config_path])
        model_name = config.model.name
        model_flavor = config.model.flavor
    elif (args.model_name is not None) and (args.model_flavor is not None):
        model_name = args.model_name
        model_flavor = args.model_flavor
    else:
        raise ValueError(
            "Either config_path or model_name and model_flavor must be provided."
        )

    convert_from_hf(
        args.input_dir,
        args.output_dir,
        model_name,
        model_flavor,
        args.merge_peft_adapter,
    )
