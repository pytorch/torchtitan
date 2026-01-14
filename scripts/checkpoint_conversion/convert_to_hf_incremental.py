# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import pickle as pkl
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple
import tqdm

import torch
import torch.distributed.checkpoint as dcp
import torchtitan.protocols.train_spec as train_spec_module
from safetensors.torch import save_file
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
from torchtitan.components.checkpoint import ModelWrapper

# Config files to copy from source HF model
HF_CONFIG_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
]


def copy_hf_configs(hf_assets_path: Path, output_dir: Path) -> None:
    """Copy config and tokenizer files from source HF model to output."""
    hf_assets_path = Path(hf_assets_path)
    output_dir = Path(output_dir)

    for filename in HF_CONFIG_FILES:
        src = hf_assets_path / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)


def cleanup_sharded_dir(output_dir: Path) -> None:
    """Remove intermediate sharded/ directory after consolidation."""
    sharded_dir = Path(output_dir) / "sharded"
    if sharded_dir.exists():
        shutil.rmtree(sharded_dir)


def find_step_dirs(checkpoint_dir: Path) -> list[Path]:
    """Find all step-* directories in a checkpoint directory."""
    step_dirs = sorted(checkpoint_dir.glob("step-*"))
    return [d for d in step_dirs if d.is_dir()]


def get_metadata(input_dir: Path) -> List[Tuple[str, TensorStorageMetadata]]:
    """
    Get the keys relating to the model state from the metadata.
    args:
        input_dir: Path to the input directory
    returns:
        List[str]: The individual keys to save from the metadata.
    """
    with open(os.path.join(input_dir, ".metadata"), "rb") as f:
        metadata = pkl.load(f)  # type: Metadata
        assert isinstance(
            metadata, Metadata
        ), f"Metadata is not a Metadata object, got {type(metadata)}"
    metadata_items = list(metadata.state_dict_metadata.items())
    i = 0
    iterator = 0
    keys_to_save = []
    for key, val in metadata_items:
        iterator += 1
        if "optimizer" in key:
            continue
        if "dataloader" in key:
            continue
        if "train_state" in key:
            continue
        if "lr_scheduler" in key:
            continue
        if "ref_model" in key:
            continue
        assert isinstance(
            val, TensorStorageMetadata
        ), f"Key {key} is not a TensorStorageMetadata"
        keys_to_save.append((key, val))
    return keys_to_save


def check_if_all_keys_are_present(state_dict, safetensor_mapping):
    safe_tensors_to_save = []  # in case there are multiple available
    for safetensor_file in list(safetensor_mapping.keys()):
        if all(
            key in list(state_dict.keys())
            for key in list(safetensor_mapping[safetensor_file])
        ):
            safe_tensors_to_save.append(safetensor_file)
    return safe_tensors_to_save


@torch.inference_mode()
def convert_to_hf(input_dir, output_dir, model_name, model_flavor, hf_assets_path, debug):
    if model_name == "flux":
        import torchtitan.experiments.flux  # noqa: F401
    # load model and model args so that we can get the state dict adapter...
    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]
    model_args.n_layers = 1
    os.makedirs(output_dir, exist_ok=True)

    with torch.device("cpu"):
        model = train_spec.model_cls(model_args)
    model = ModelWrapper(model)

    sd_adapter = train_spec.state_dict_adapter(model_args, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."
    keys_to_save = get_metadata(input_dir)
    with open(os.path.join(hf_assets_path, "model.safetensors.index.json"), "r") as f:
        index_json = json.load(f)
        fixed_weight_map = {}
        for key, val in index_json["weight_map"].items():
            if "rotary_emb.inv_freq" in key:
                continue
            if "weight_scale_inv" in key:
                continue
            fixed_weight_map[key] = val
        index_json["weight_map"] = fixed_weight_map
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index_json, f, indent=2)
        safetensor_mapping = defaultdict(list)
        for key, val in index_json["weight_map"].items():
            safetensor_mapping[val].append(key)
    loaded_state_dict = {}
    # Copy config files and clean up
    copy_hf_configs(hf_assets_path, output_dir)
    cleanup_sharded_dir(output_dir)
    keys_to_save = sorted(keys_to_save, key=lambda x: x[0])
    if debug:
        print(f"Keys...")
        print([x[0] for x in keys_to_save])
    for key, tensor_metadata in tqdm.tqdm(keys_to_save, desc="Loading state dict"):
        state_dict_to_grab = {
            key: torch.empty(
                tensor_metadata.size, dtype=tensor_metadata.properties.dtype
            )
        }
        dcp.load(state_dict_to_grab, checkpoint_id=input_dir)
        hf_state_dict = sd_adapter.to_hf(state_dict_to_grab)
        loaded_state_dict.update(hf_state_dict)
        safe_tensors_to_save = check_if_all_keys_are_present(
            loaded_state_dict, safetensor_mapping
        )
        if len(safe_tensors_to_save) != 0:
            for safe_tensor_file in safe_tensors_to_save:
                print(f"Saving {safe_tensor_file}...")
                keys_to_save = safetensor_mapping[safe_tensor_file]
                state_dict_to_save = {
                    key: loaded_state_dict.pop(key) for key in keys_to_save
                }
                save_file(
                    state_dict_to_save, os.path.join(output_dir, safe_tensor_file)
                )
                del state_dict_to_save
        else:
            if debug:
                print("No safe tensors to save, current state dict:")
                print(list(loaded_state_dict.keys()))
            else:
                print("No safe tensors to save")


def convert_all_checkpoints(
    checkpoint_dir: Path, model_name: str, model_flavor: str, hf_assets_path: Path
):
    """Convert all step-* checkpoints in a directory to HF format.

    Output structure: {parent_of_checkpoint}/hf_checkpoints/{step_name}/
    """
    checkpoint_dir = Path(checkpoint_dir)
    step_dirs = find_step_dirs(checkpoint_dir)

    if not step_dirs:
        print(f"No step-* directories found in {checkpoint_dir}")
        return

    # Output goes to sibling directory: checkpoint/ -> hf_checkpoints/
    output_base = checkpoint_dir.parent / "hf_checkpoints"
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(step_dirs)} checkpoints to convert")
    print(f"Output directory: {output_base}")

    for step_dir in step_dirs:
        step_name = step_dir.name
        output_dir = output_base / step_name

        if output_dir.exists():
            print(f"Skipping {step_name} (already exists)")
            continue

        print(f"Converting {step_name}...")
        convert_to_hf(step_dir, output_dir, model_name, model_flavor, hf_assets_path)
        print(f"  -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with DCP weights (or checkpoint/ dir with --all).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Output directory for HF checkpoint. Not needed with --all.",
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        help="Path to HF assets directory. This is used to get the model.safetensors.index.json mapping",
        default="./assets/hf/Llama-3.1-8B",
    )
    parser.add_argument("--model_name", type=str, nargs="?", default="llama3")
    parser.add_argument("--model_flavor", type=str, nargs="?", default="8B")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all step-* checkpoints in input_dir to {parent}/hf_checkpoints/{step}/",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information",
    )
    args = parser.parse_args()

    if args.all:
        convert_all_checkpoints(
            args.input_dir,
            args.model_name,
            args.model_flavor,
            args.hf_assets_path,
        )
    else:
        if args.output_dir is None:
            parser.error("output_dir is required when not using --all")
        convert_to_hf(
            args.input_dir,
            args.output_dir,
            args.model_name,
            args.model_flavor,
            args.hf_assets_path,
            args.debug,
        )
