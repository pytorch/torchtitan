#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import Optional

# Add the torchtitan directory to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
torchtitan_dir = os.path.join(script_dir, "..", "..")
torchtitan_dir = os.path.normpath(torchtitan_dir)
sys.path.insert(0, torchtitan_dir)

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import ConfigManager
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools.logging import logger
from transformers import AutoModelForCausalLM


def loss_fn(logits1, logits2):
    """Calculate KL divergence loss between two sets of logits"""
    probs1 = F.log_softmax(logits1, dim=-1)
    probs2 = F.softmax(logits2, dim=-1)
    return F.kl_div(probs1, probs2, reduction="mean")


@torch.no_grad
def forward_hf(model_name, model_path: Optional[str], input_ids, gpu_partition):
    """Run HuggingFace inference with GPU partitioning"""
    print(f"Loading HuggingFace model on GPUs: {gpu_partition}")

    model_path = model_path if model_path else model_name

    # Create device map to use only specified GPUs
    device_map = {}
    if len(gpu_partition) == 1:
        device_map = gpu_partition[0]  # Single GPU
    else:
        # Multi-GPU: let HF auto-distribute but only on specified GPUs
        device_map = "auto"
        # Set environment to limit CUDA visible devices
        visible_devices = ",".join(map(str, gpu_partition))
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    try:
        target_device = torch.device(f"cuda:{gpu_partition[0]}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(target_device)

        outputs_list = []
        prompt_len = 8

        for inputs in input_ids:
            inputs = inputs.to(target_device)

            outputs = model.generate(
                inputs=inputs,
                max_length=prompt_len + 1,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
            )
            outputs = torch.stack(outputs.logits)
            outputs_list.append(outputs.cpu())  # Move to CPU to free GPU memory

        # Clean up
        del model
        torch.cuda.empty_cache()

        return outputs_list

    finally:
        # Reset CUDA_VISIBLE_DEVICES
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]


@torch.no_grad
def forward_tt(config_path, checkpoint_path, test_set, gpu_partition):
    """Run TorchTitan inference with GPU partitioning"""
    print(f"Loading TorchTitan model on GPUs: {gpu_partition}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Check if checkpoint has metadata
    metadata_path = os.path.join(checkpoint_path, ".metadata")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")

    config_manager = ConfigManager()
    config = config_manager.parse_args([f"--job.config_file={config_path}"])

    train_spec = get_train_spec(config.model.name)
    model_args = train_spec.model_args[config.model.flavor]
    model_args.update_from_config(config)

    # Use the first available GPU in partition
    target_device = torch.device(f"cuda:{gpu_partition[0]}")

    # Create model on meta device first
    with torch.device("meta"):
        model = train_spec.model_cls(model_args)

    # Initialize on CPU then move to target GPU
    model.to_empty(device="cpu")
    model.init_weights(buffer_device="cpu")
    model.half()  # Convert to FP16

    # Move to target GPU
    print(f"Moving TorchTitan model to {target_device}")
    model = model.to(target_device)
    model.eval()

    # Load checkpoint
    modelWrapper = ModelWrapper(model)
    state_dict = modelWrapper._get_state_dict()

    print(f"Loading checkpoint from: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)

    # Force all model parameters and buffers to target device after checkpoint loading
    model = model.to(target_device)

    # Ensure all parameters are on the correct device
    for name, param in model.named_parameters():
        if param.device != target_device:
            print(f"Moving parameter {name} from {param.device} to {target_device}")
            param.data = param.data.to(target_device)

    for name, buffer in model.named_buffers():
        if buffer.device != target_device:
            print(f"Moving buffer {name} from {buffer.device} to {target_device}")
            buffer.data = buffer.data.to(target_device)

    # Run inference
    output_list = []
    for i, prompt in enumerate(test_set):
        print(f"Processing prompt {i+1}/{len(test_set)}")
        input_ids = prompt.to(target_device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        print(f"Input device: {input_ids.device}")

        predictions = model(input_ids)[:, -1, :].unsqueeze(1)
        output_list.append(predictions.cpu())  # Move to CPU to save GPU memory
        print(f"Successfully processed prompt {i+1}")

    # Cleanup
    del model, modelWrapper, state_dict
    torch.cuda.empty_cache()

    return output_list


def main():
    """Main function with GPU partitioning"""
    print("Starting GPU-partitioned inference test...")

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    if num_gpus < 2:
        print("Warning: Less than 2 GPUs available. This may cause memory issues.")
        hf_gpus = [0]
        tt_gpus = [0]
    else:
        # 0.6B model fits on a single GPU each
        hf_gpus = [0]
        tt_gpus = [1]

    print(f"HuggingFace will use GPUs: {hf_gpus}")
    print(f"TorchTitan will use GPUs: {tt_gpus}")

    # Parameters
    hf_model_name = "Qwen/Qwen3-0.6B"
    hf_model_path = os.path.expanduser("~/models/Qwen/Qwen3-0.6B")
    config_path = "torchtitan/models/qwen3/train_configs/qwen3_0.6b.toml"
    checkpoint_path = "outputs/Qwen/qwen3_0.6b_dcp"

    # Check paths exist
    if not os.path.exists(hf_model_path):
        print(
            f"Warning: HF model path not found: {hf_model_path}, using model name instead"
        )
        hf_model_path = None

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path not found: {checkpoint_path}")
        return

    prompt_len = 8
    test_size = 5  # Reduced for memory safety

    # Build test set
    config_manager = ConfigManager()
    config = config_manager.parse_args([f"--job.config_file={config_path}"])
    train_spec = get_train_spec(config.model.name)
    tokenizer = train_spec.build_tokenizer_fn(config)

    test_set = [
        torch.randint(0, tokenizer.get_vocab_size(), (1, prompt_len))
        for _ in range(test_size)
    ]

    try:
        # Run HuggingFace inference
        print("\n" + "=" * 50)
        print("Running HuggingFace inference...")
        print("=" * 50)
        hf_outputs = forward_hf(hf_model_name, hf_model_path, test_set, hf_gpus)
        print("HuggingFace inference completed!")

        # Clear cache between models
        torch.cuda.empty_cache()

        # Run TorchTitan inference
        print("\n" + "=" * 50)
        print("Running TorchTitan inference...")
        print("=" * 50)
        tt_outputs = forward_tt(config_path, checkpoint_path, test_set, tt_gpus)
        print("TorchTitan inference completed!")

        # Calculate loss
        print("\n" + "=" * 50)
        print("Comparing outputs...")
        print("=" * 50)

        total_loss = 0
        for i, (hf_out, tt_out) in enumerate(zip(hf_outputs, tt_outputs)):
            sample_loss = loss_fn(hf_out, tt_out)
            total_loss += sample_loss
            print(f"Sample {i+1} loss: {sample_loss.item():.6f}")

        avg_loss = total_loss / len(test_set)
        print(f"\nAverage loss between HF and TT outputs: {avg_loss.item():.6f}")

        if avg_loss.item() < 0.01:
            print("âœ… Models are very similar!")
        elif avg_loss.item() < 0.1:
            print("âš ï¸  Models have some differences")
        else:
            print("âŒ Models are significantly different")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
