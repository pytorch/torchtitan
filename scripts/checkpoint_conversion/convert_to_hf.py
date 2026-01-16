# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torchtitan.protocols.train_spec as train_spec_module
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config.job_config import PEFT

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


@torch.inference_mode()
def convert_to_hf(input_dir, output_dir, model_name, model_flavor, hf_assets_path):
    if model_name == "flux":
        import torchtitan.experiments.flux  # noqa: F401
    # load model and model args so that we can get the state dict shape
    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]

    with torch.device("cpu"):
        try:
            model = train_spec.model_cls(model_args, PEFT())
        except TypeError:
            model = train_spec.model_cls(model_args)
    model = ModelWrapper(model)

    sd_adapter = train_spec.state_dict_adapter(model_args, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    # allocate state dict memory with empty weights to load checkpoint
    state_dict = model._get_state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )

    # Copy config files and clean up
    copy_hf_configs(hf_assets_path, output_dir)
    cleanup_sharded_dir(output_dir)


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
        )
