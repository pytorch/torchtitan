#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple inference script for LFM2 models.

Usage:
    python scripts/generate/test_lfm2_generate.py \
        --config torchtitan/experiments/lfm2/train_configs/lfm2_350m.toml \
        --checkpoint outputs/checkpoint/step-7000 \
        --prompt "The quick brown fox" \
        --max_new_tokens 50
"""

import argparse
import sys
import time
from pathlib import Path

# support running w/o installing as package
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

import torch
import torch.distributed.checkpoint as dcp
from torchtitan.config import ConfigManager
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import device_type

# Import generation utilities
from scripts.generate._generation import generate


def main():
    parser = argparse.ArgumentParser(description="Test LFM2 generation")
    parser.add_argument(
        "--config", type=str, required=True, help="TOML config file path (required)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path to load (required)",
    )
    parser.add_argument(
        "--prompt", type=str, default="The quick brown fox", help="Input prompt"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Default is 0.8",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Max number of tokens to generate. Default is 50",
    )
    parser.add_argument(
        "--top_k", type=int, default=200, help="Prune to select from top_k probabilities"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    init_logger()
    color = utils.Color

    # Load configuration from toml file
    config_manager = ConfigManager()
    config = config_manager.parse_args([f"--job.config_file={args.config}"])

    device = torch.device(f"{device_type}:0")
    logger.info(f"Using device: {device}")

    # Get LFM2 train spec
    train_spec = get_train_spec(config.model.name)

    # Tokenizer setup
    tokenizer = train_spec.build_tokenizer_fn(config)
    logger.info(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")

    # Model setup
    model_args = train_spec.model_args[config.model.flavor]
    model_args.update_from_config(config)

    logger.info(f"Initializing {config.model.name} {config.model.flavor} model...")
    with torch.device("cpu"):
        model = train_spec.model_cls(model_args)

    # Move to device and initialize
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights()
    model.eval()

    # Load checkpoint
    state_dict = model.state_dict()
    begin = time.monotonic()
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    dcp.load(state_dict, checkpoint_id=args.checkpoint)
    logger.info(f"Checkpoint loaded in {time.monotonic() - begin:.2f} seconds")

    # Tokenize prompt
    logger.info(f"\nPrompt: {color.red}{args.prompt}{color.reset}")
    input_ids = torch.tensor(
        tokenizer.encode(args.prompt, add_bos=True, add_eos=False),
        dtype=torch.long
    ).unsqueeze(0).to(device)

    logger.info(f"Input tokens: {input_ids.shape[1]}")

    # Generate
    logger.info(f"Generating {args.max_new_tokens} tokens...")
    t0 = time.monotonic()

    with torch.no_grad():
        output_ids = generate(
            model,
            input_ids,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            seed=args.seed,
        )

    t1 = time.monotonic()
    elapsed_sec = t1 - t0

    # Decode and display
    input_n_tokens = input_ids.size(1)
    generated_n_tokens = output_ids.size(1) - input_n_tokens

    input_text = tokenizer.decode(output_ids[0, :input_n_tokens].tolist())
    output_text = tokenizer.decode(output_ids[0, input_n_tokens:].tolist())

    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATION RESULT:")
    logger.info(f"{'='*80}")
    logger.info(f"{color.red}{input_text}{color.blue}{output_text}{color.reset}")
    logger.info(f"{'='*80}")
    logger.info(f"Generated {generated_n_tokens} tokens in {elapsed_sec:.2f} seconds")
    logger.info(f"Speed: {generated_n_tokens/elapsed_sec:.2f} tokens/sec")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
