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
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.tools.logging import logger


def copy_hf_assets(hf_assets_path, output_dir):
    """
    Copy config.json and tokenizer files from hf_assets_path to output_dir.
    
    Args:
        hf_assets_path: Path to the HuggingFace assets directory
        output_dir: Path to the output directory
    """
    hf_assets_path = Path(hf_assets_path)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Required files
    config_source = hf_assets_path / "config.json"
    if not config_source.exists():
        raise FileNotFoundError(
            f"config.json not found at {config_source}. "
            f"Please ensure the HuggingFace assets path is correct."
        )
    shutil.copy2(config_source, output_dir / "config.json")
    
    # Copy tokenizer files (needed for serving with vLLM, etc.)
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",  # for sentencepiece tokenizers
        "special_tokens_map.json",
        "generation_config.json",
    ]
    for fname in tokenizer_files:
        src = hf_assets_path / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)


def merge_finetune_lora_weights(state_dict, model_args):
    """
    Merge LoRA fine-tuning weights into base weights.
    
    LoRA naming convention: finetune_lora_{target}.lora_A, finetune_lora_{target}.lora_B
    Example keys:
      - layers.0.attention.finetune_lora_wo.lora_A.weight → merge into layers.0.attention.wo.weight
      - layers.0.moe.finetune_lora_shared_w2.lora_A.weight → merge into layers.0.moe.shared_experts.w2.weight
    
    Merge formula: merged = base + lora_B @ lora_A * scale
    where scale = finetune_lora_alpha / finetune_lora_rank
    
    After merging, the LoRA keys are removed from the state dict.
    """
    if model_args.finetune_lora_rank <= 0:
        return state_dict
    
    scale = model_args.finetune_lora_alpha / model_args.finetune_lora_rank
    
    # Mapping from LoRA module name to base layer path
    # Format: "finetune_lora_{name}" → base layer relative path
    lora_target_mapping = {
        "finetune_lora_wo": "wo",
        "finetune_lora_shared_w2": "shared_experts.w2",
    }
    
    # Find all LoRA A keys: pattern is "finetune_lora_*.lora_A.weight"
    lora_a_keys = [k for k in state_dict.keys() if ".lora_A.weight" in k]
    
    if not lora_a_keys:
        raise ValueError(
            f"finetune_lora_rank={model_args.finetune_lora_rank} but no LoRA keys found in checkpoint. "
            "Expected keys matching pattern '*.lora_A.weight'"
        )
    
    merged_layers = []
    for lora_a_key in lora_a_keys:
        # Extract LoRA module key
        # e.g., "layers.0.attention.finetune_lora_wo.lora_A.weight" → "layers.0.attention.finetune_lora_wo"
        lora_module_key = lora_a_key.replace(".lora_A.weight", "")
        prefix = lora_module_key.rsplit(".", 1)[0]  # "layers.0.attention"
        lora_module_name = lora_module_key.rsplit(".", 1)[1]  # "finetune_lora_wo"
        
        # Look up the base layer mapping
        if lora_module_name not in lora_target_mapping:
            logger.warning(f"Unknown LoRA module '{lora_module_name}' in {lora_a_key}, skipping")
            continue
        
        base_relative = lora_target_mapping[lora_module_name]
        
        # Derive corresponding keys
        lora_b_key = f"{lora_module_key}.lora_B.weight"
        base_key = f"{prefix}.{base_relative}.weight"
        
        if lora_b_key not in state_dict:
            raise KeyError(
                f"Missing LoRA B key: {lora_b_key}. "
                f"Found LoRA A key {lora_a_key} but corresponding B key is missing."
            )
        if base_key not in state_dict:
            raise KeyError(
                f"Missing base weight key: {base_key}. "
                f"Cannot merge LoRA adapter into non-existent base layer."
            )
        
        # Get the weights
        base_weight = state_dict[base_key]
        lora_a_weight = state_dict[lora_a_key]
        lora_b_weight = state_dict[lora_b_key]
        
        # Merge: merged = base + lora_B @ lora_A * scale
        delta = lora_b_weight @ lora_a_weight * scale
        state_dict[base_key] = base_weight + delta
        
        # Remove LoRA keys
        del state_dict[lora_a_key]
        del state_dict[lora_b_key]
        merged_layers.append(base_key)
    
    logger.info(f"[LoRA] Merged {len(merged_layers)} LoRA adapter(s) into base weights (scale={scale:.4f})")
    for layer in merged_layers:
        logger.info(f"  - {layer}")
    
    return state_dict


@torch.inference_mode()
def convert_to_hf(
    input_dir,
    output_dir,
    model_name,
    model_flavor,
    hf_assets_path,
    export_dtype,
):
    # load model and model args so that we can get the state dict shape
    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]

    with torch.device("cpu"):
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

    # If LoRA fine-tuning was used, merge LoRA weights into base weights
    # state_dict keys like: layers.0.attention.finetune_lora_wo.lora_A.weight
    # pattern: layers.{}.{module}.finetune_lora_{target}.lora_{A|B}.weight
    state_dict = merge_finetune_lora_weights(state_dict, model_args)

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    # Filter out training-only keys that don't exist in the HF index mapping
    # (e.g., expert_bias / e_score_correction_bias used for MoE load balancing)
    if sd_adapter.fqn_to_index_mapping is not None:
        hf_index_keys = set(sd_adapter.fqn_to_index_mapping.keys())
        filtered_state_dict = {}
        skipped_keys = []
        for k, v in hf_state_dict.items():
            if k in hf_index_keys:
                filtered_state_dict[k] = v
            else:
                skipped_keys.append(k)
        
        if skipped_keys:
            logger.warning(
                f"Skipping {len(skipped_keys)} training-only key(s) not in HF index: {skipped_keys}"
            )
        hf_state_dict = filtered_state_dict
    else:
        # No index mapping available - filter out known training-only keys manually
        training_only_patterns = ["expert_bias", "e_score_correction_bias"]
        filtered_state_dict = {}
        skipped_keys = []
        for k, v in hf_state_dict.items():
            if any(pattern in k for pattern in training_only_patterns):
                skipped_keys.append(k)
            else:
                filtered_state_dict[k] = v
        
        if skipped_keys:
            logger.warning(
                f"Skipping {len(skipped_keys)} training-only key(s): {skipped_keys}"
            )
        hf_state_dict = filtered_state_dict

    # map and apply export dtype if needed
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )

    copy_hf_assets(hf_assets_path, output_dir)


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
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
    )
