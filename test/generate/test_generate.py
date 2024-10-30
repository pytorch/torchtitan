# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
import time
from pathlib import Path

from typing import Optional

import torch
import torch.distributed.checkpoint as dcp

from torchtitan import utils

from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_tokenizer
from torchtitan.logging import init_logger, logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.parallelisms import models_parallelize_fns, ParallelDims

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.generation import generate


def example_generate(
    config_path: str,
    checkpoint_path: str,
    prompt: str,
    *,
    device: str = "cuda",
    temperature: float = 1.0,
    max_new_tokens: int = 32,
    batch_size: int = 1,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
):
    init_logger()
    color = utils.Color

    # Load configuration from toml file
    config = JobConfig()
    config.parse_args([f"--job.config_file={config_path}"])
    config._validate_config()

    utils.set_determinism(seed)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model_name = config.model.name

    # Init distributed env
    if world_size > 1:
        utils.init_distributed(config)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            cp=1,
            tp=world_size,
            pp=1,
            world_size=world_size,
            enable_loss_parallel=False,
        )
        # Build world mesh for parallelism
        world_mesh = parallel_dims.build_mesh(device_type="cuda")

    logger.info(f"World Size: {world_size}, Local Rank: {local_rank} on {device}")

    # Tokenizer setup
    tokenizer = build_tokenizer(
        model_name_to_tokenizer[model_name], config.model.tokenizer_path
    )

    model_config = models_config[model_name][config.model.flavor]
    model_config.norm_type = config.model.norm_type
    model_config.max_seq_len = config.training.seq_len
    model_config.vocab_size = tokenizer.n_words

    model_cls = model_name_to_cls[model_name]
    init_device = "meta" if world_size > 1 else device
    with torch.device(init_device):
        logger.info(f"Init model on init_device: {init_device}")
        model = model_cls.from_model_args(model_config)

    if world_size > 1:
        models_parallelize_fns[model_name](model, world_mesh, parallel_dims, config)

    # materalize model
    model.to_empty(device="cuda")
    model.eval()

    state_dict = {"model": model.state_dict()}

    # Checkpoint Loading
    begin = time.monotonic()
    logger.info(f"Loading chkpt at: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    logger.info(f"Finished loading chkpt in {time.monotonic() - begin:.2f} seconds.")

    # Tokenize prompt and repeat batch_size times
    input_ids = (
        (
            torch.tensor(
                tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.long
            )
            .view(1, -1)
            .repeat(batch_size, 1)
        )
        .cuda()
        .detach()
    )

    # Inference
    begin = time.monotonic()
    responses, _ = generate(
        model,
        input_ids,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
    )
    end = time.monotonic()

    prompt_len = input_ids.size(1)  # num tokens

    if local_rank == 0:
        logger.info(f"Generation completed in {end-begin:.2f} seconds.")

        r, b = color.red, color.blue

        output_data = []

        for i, response in enumerate(responses):

            inp_tok = response[:prompt_len].tolist()
            out_tok = response[prompt_len:].tolist()

            input_text = tokenizer.decode(inp_tok)
            output_text = tokenizer.decode(out_tok)

            response_data = {
                "response_idx": i,
                "input_n_tokens": len(inp_tok),
                "output_n_tokens": len(out_tok),
                "input_text": input_text,
                "output_text": output_text,
            }
            output_data.append(response_data)

            logger.info(f"{r}\n{input_text}{b}{output_text}\n{color.reset}")

        print(json.dumps(output_data, indent=4))


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
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

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
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
