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
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.config import TORCH_DTYPE_MAP

_EMA_STATE_PREFIX = "ema_optimizer.state."
_EMA_STATE_SUFFIX = ".ema_params"


def _load_ema_state_dict(model_state_dict, trainable_fqns, input_dir):
    """Load EMA weights from `input_dir` into a copy of `model_state_dict`,
    replacing only the trainable-parameter entries (EMA never tracks
    buffers). Returns None if the checkpoint has no EMA data (EMA was
    disabled during training, or the checkpoint predates EMA support).
    """
    metadata = dcp.FileSystemReader(input_dir).read_metadata()
    if not any(k.startswith(_EMA_STATE_PREFIX) for k in metadata.state_dict_metadata):
        return None

    flat_ema_sd = {
        f"{_EMA_STATE_PREFIX}{fqn}{_EMA_STATE_SUFFIX}": torch.empty_like(
            model_state_dict[fqn]
        )
        for fqn in trainable_fqns
    }
    dcp.load(flat_ema_sd, checkpoint_id=input_dir)

    ema_state_dict = dict(model_state_dict)
    for fqn in trainable_fqns:
        ema_state_dict[fqn] = flat_ema_sd[
            f"{_EMA_STATE_PREFIX}{fqn}{_EMA_STATE_SUFFIX}"
        ]
    return ema_state_dict


def _save_as_hf(state_dict, sd_adapter, target_dtype, output_dir):
    hf_state_dict = sd_adapter.to_hf(state_dict)
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )
    dcp.save(hf_state_dict, storage_writer=storage_writer)


@torch.inference_mode()
def convert_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
    ema_output=None,
):
    # load model and model args so that we can get the state dict shape
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model

    with torch.device("cpu"):
        raw_model = model_config.build()
    trainable_fqns = [
        canonical_fqn(name)
        for name, p in raw_model.named_parameters()
        if p.requires_grad
    ]
    model = ModelWrapper(raw_model)

    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    # allocate state dict memory with empty weights to load checkpoint
    state_dict = model._get_state_dict()
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    _save_as_hf(state_dict, sd_adapter, target_dtype, output_dir)

    if ema_output is not None:
        ema_state_dict = _load_ema_state_dict(state_dict, trainable_fqns, input_dir)
        if ema_state_dict is None:
            print(
                f"[WARNING] --ema_output was given but the checkpoint at {input_dir} "
                "has no EMA weights (EMA was disabled during that training run, or "
                "this checkpoint predates EMA support). Skipping EMA export."
            )
        else:
            _save_as_hf(ema_state_dict, sd_adapter, target_dtype, ema_output)
            print(f"EMA weights saved to {ema_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
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
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    parser.add_argument(
        "--ema_output",
        type=Path,
        default=None,
        help="If given and the checkpoint has EMA weights (see "
        "torchtitan.components.ema), also export them as a second HF "
        "checkpoint at this directory.",
    )
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
        ema_output=args.ema_output,
    )
