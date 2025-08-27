#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hugging Face implementation for DeepSeek-V3 model inference.
"""

import argparse
import gc
import os
import time

import torch


def print_gpu_memory_usage(message=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(
            f"GPU Memory ({message}): Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        )


def run_huggingface_implementation(args, _):
    """Run the DeepSeek-V3 model using Hugging Face Transformers."""
    # Disable Hugging Face cache
    from transformers import AutoConfig, AutoModelForCausalLM

    # We're not using the tokenizer anymore, using fake inputs instead
    # Use local path for model weights if specified, otherwise use model_name
    model_path = args.model_path
    print(f"Loading model from local path: {model_path}")
    start_time = time.time()

    quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",  # Updated from fp8 to fbgemm_fp8
        "weight_block_size": [128, 128],
    }
    print(f"Using quantization config: {quantization_config}")

    # ============= Change config to only use a few layers  =============
    config = None
    if args.num_layers > 0:
        # Try to load config from local path first, fall back to model_name if needed
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"Could not load config from local path: {e}")
            print(f"Falling back to loading config from {args.model_name}")
            config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

        config.n_group = 1  # make n_groups = a huge group
        config.topk_group = 1  # make topk_group = a huge group
        # tailer the first several layers
        config.num_hidden_layers = args.num_layers
        # Explicitly set rope_interleaved to True to use the interleaved rope implementation
        config.rope_interleaved = True
        print(f"Modified config to use only {args.num_layers} layers")
        print(f"Config of Deepseek: {config}")

    # Load the model from local path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",  # Try with specific device first
        config=config,
        trust_remote_code=True,
        # Disable features that can cause issues with device mapping
        attn_implementation="eager",  # Use standard attention instead of flash attention
        quantization_config=quantization_config,
        local_files_only=True,  # Only use local files, don't fetch from cache
        use_auth_token=False,  # Don't try to authenticate with HF
    )

    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    print_gpu_memory_usage("After loading model")

    # Get the device where the model is loaded
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    # Create fake input directly on the correct device
    print("\nCreating fake input with the same shape as tokenized input")

    # Define sequence length for fake input
    seq_length = 2048  # You can adjust this based on your needs
    vocab_size = 50000

    with torch.no_grad():
        # Create fake input_ids directly on the device - using random integers between 0 and 50000 (typical vocab size)
        torch.manual_seed(42)
        tokens = torch.randint(
            0, vocab_size, (1, seq_length), dtype=torch.long, device="cuda"
        )

        # Create fake attention_mask directly on the device - all 1s for full attention
        attention_mask = torch.ones((1, seq_length), dtype=torch.long, device=device)

        # Create inputs dictionary similar to what tokenizer would produce
        inputs = {"input_ids": tokens, "attention_mask": attention_mask}

        # Print input information
        print(f"Fake input token IDs: {inputs['input_ids'][0][:10].cpu().numpy()}...")
        print(f"Fake input shape: {inputs['input_ids'].shape}")
        print(f"Input tensors device: {inputs['input_ids'].device}")

    # Run a single forward pass
    print("\nRunning single forward pass...")
    start_time = time.time()

    with torch.no_grad():
        # Forward pass through the model with output_hidden_states=True and output_attentions=True
        outputs = model(
            **inputs, output_hidden_states=True, output_attentions=True, use_cache=False
        )

    forward_time = time.time() - start_time

    # Get the logits from the output
    logits = outputs.logits if hasattr(outputs, "logits") else outputs

    # Get the predictions for the next token (highest probability)
    next_token_logits = logits[:, -1, :]
    print(f"\nNext token logits : {next_token_logits}")
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    print(f"\nNext token probabilities: {next_token_probs}")
    top_k_values, top_k_indices = torch.topk(next_token_probs, 5, dim=-1)

    print("\nForward Pass Results:")
    print(f"- Output logits shape: {logits.shape}")
    print(f"- Sequence length: {logits.shape[1]}")
    print(f"- Vocabulary size: {logits.shape[2]}")

    print(
        "\nTop 5 predicted next tokens (showing IDs only since we're not using tokenizer):"
    )
    for i, (value, index) in enumerate(zip(top_k_values[0], top_k_indices[0])):
        print(f"  {i+1}. Token ID: {index} - Probability: {value.item():.4f}")

    print(f"\nForward pass stats:")
    print(f"- Time: {forward_time:.4f} seconds")
    print(f"- Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"- Tokens per second: {inputs['input_ids'].shape[1] / forward_time:.2f}")
    print_gpu_memory_usage("After forward pass")


def main():
    parser = argparse.ArgumentParser(description="Load and test DeepSeek-V3 model")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=5,  # tailered to 5 layers for 671B model
        help="Number of layers to use (0 for all layers)",
    )

    # Hugging Face specific arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/users/jianiw/model/DeepSeek-V3.1-Base",
        help="Hugging Face model name or path",
    )

    args = parser.parse_args()
    run_huggingface_implementation(args, None)


if __name__ == "__main__":
    main()
