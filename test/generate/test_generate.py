# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
from typing import Optional

import torch
import torch.distributed.checkpoint as dcp

from generation import generate
from torchtitan import utils

from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_tokenizer
from torchtitan.logging import init_logger, logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config


def example_generate(
    config_path: str,
    checkpoint_path: str,
    prompt: str,
    *,
    device: str = "cuda",
    temperature: float = 1.0,
    max_new_tokens: int = 32,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
):
    init_logger()
    color = utils.Color

    # Load configuration from toml file
    config = JobConfig()
    config.parse_args([f"--job.config_file={config_path}"])
    config._validate_config()

    # Load tokenizer and model configuration
    tokenizer = build_tokenizer(
        model_name_to_tokenizer[config.model.name], config.model.tokenizer_path
    )
    model_cls = model_name_to_cls[config.model.name]
    model_config = models_config[config.model.name][config.model.flavor]
    model_config.vocab_size = tokenizer.n_words

    # Load model and checkpoint
    with torch.device(device):
        model = model_cls.from_model_args(model_config)

    model_param_count = utils.get_num_params(model)
    logger.info(f"Model Params: {model_param_count:,}")

    state_dict = {"model": model.state_dict()}

    precompute = False
    if "freqs_cis" in state_dict["model"]:
        del state_dict["model"]["freqs_cis"]
        precompute = True

    begin = time.monotonic()
    logger.info(f"Loading checkpoint at: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    logger.info(
        f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
    )

    # Precompute frequency if required
    if precompute:
        model.freqs_cis = model._precompute_freqs_cis().to(args.device)

    # Encode input prompt and generate response
    input_ids = torch.tensor(
        tokenizer.encode(prompt, bos=False, eos=False), dtype=torch.long
    ).to(device)

    begin = time.monotonic()
    responses = generate(
        model,
        input_ids,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
    )

    logger.info(f"Generation completed in {time.monotonic() - begin:.2f} seconds.")
    logger.info(f"{color.red}Input tokens: {len(input_ids)}{color.reset}")
    logger.info(
        f"{color.blue}Output tokens: {len(responses[0])-len(input_ids)}{color.reset}"
    )

    response = tokenizer.decode(
        [token.item() for token in responses[0][len(input_ids) :]]
    )

    logger.info(f"{color.red}{prompt}{color.blue}{response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
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
        "--device",
        type=str,
        default="cuda",
        help="Device to load model on. Default is 'cuda'",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Default is 1.0",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max number of tokens to generate. Default is 32",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of samples to run in batch"
    )
    parser.add_argument(
        "--top_k", type=int, help="Prune to select from top_k probabilities. Optional"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello! How are",
        help="Input prompt for generation",
    )

    args = parser.parse_args()

    example_generate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        device=args.device,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        seed=args.seed,
    )
