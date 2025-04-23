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
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torchtitan.components.metrics import build_device_memory_monitor

from torchtitan.config_manager import ConfigManager
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import device_module, device_type

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate._generation import generate


def apply_tp_minus_sp(model: nn.Module, tp_mesh: DeviceMesh):
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
            "output": ColwiseParallel(output_layouts=Replicate()),
        },
    )

    for _, transformer_block in model.layers.items():
        layer_plan = {
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(),
            "feed_forward.w3": ColwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )


@record
def test_generate(
    config_path: str,
    checkpoint_path: str,
    prompt: str,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 32,
    batch_size: int = 1,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
):
    init_logger()
    color = utils.Color

    # Load configuration from toml file
    config_manager = ConfigManager()
    config = config_manager.parse_args([f"--job.config_file={config_path}"])

    if len(args.prompt) == 0:
        logger.warning(
            "The input prompt is empty, model will respond from a empty sequence."
        )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)
    device_memory_monitor = build_device_memory_monitor()

    train_spec = get_train_spec(config.model.name)

    logger.info(f"World Size: {world_size}, Local Rank: {local_rank} on {device}")

    # Tokenizer setup
    tokenizer = train_spec.build_tokenizer_fn(config)

    model_cls = train_spec.cls
    model_args = train_spec.config[config.model.flavor]
    model_args.update_from_config(config, tokenizer)

    init_device = "meta" if world_size > 1 else device
    with torch.device(init_device):
        logger.info(f"Init model on init_device: {init_device}")
        model = model_cls.from_model_args(model_args)

    world_mesh = None
    # Init distributed env
    if world_size > 1:
        dist_utils.init_distributed(config)
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
        world_mesh = parallel_dims.build_mesh(device_type=device_type)

        # apply_tp (with Sequence Parallel) on unevenly sharded
        # sequences would require https://github.com/pytorch/torchtitan/pull/686
        apply_tp_minus_sp(model, world_mesh["tp"])

    dist_utils.set_determinism(world_mesh, device, seed, deterministic)

    # materalize model
    model.to_empty(device=device_type)
    model.eval()

    state_dict = {"model": model.state_dict()}

    # Checkpoint Loading
    begin = time.monotonic()
    logger.info(f"Loading chkpt at: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    logger.info(f"Finished loading chkpt in {time.monotonic() - begin:.2f} seconds.")

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # Tokenize prompt and repeat batch_size times
    input_ids = (
        (
            torch.tensor(
                tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.long
            )
            .view(1, -1)
            .repeat(batch_size, 1)
        )
    ).to(device_type)

    device_memory_monitor.reset_peak_stats()

    # Run generation
    t0 = time.monotonic()
    responses = generate(
        model,
        input_ids,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        seed=seed,
    )
    t1 = time.monotonic()
    elapsed_sec = t1 - t0

    # Post process
    B, T = responses.size()  # B: batch_size, T: total seq length
    input_n_tokens = input_ids.size(1)
    generated_n_tokens = T - input_n_tokens  # == max_new_tokens

    if local_rank == 0:
        logger.info(f"Generation completed in {elapsed_sec:.2f} seconds.")

        r, b = color.red, color.blue

        output_data = {
            "metadata": {},
            "responses": [],
        }

        for i, tokens in enumerate(responses):
            inp_tok = tokens[:input_n_tokens].tolist()
            out_tok = tokens[input_n_tokens:].tolist()

            input_text = tokenizer.decode(inp_tok)
            output_text = tokenizer.decode(out_tok)

            _data = {
                "response_idx": i,
                "input_text": input_text,
                "output_text": output_text,
            }
            output_data["responses"].append(_data)

            logger.info(f"{r}\n{input_text}{b}{output_text}\n{color.reset}")

        device_mem_stats = device_memory_monitor.get_peak_stats()
        output_data["metadata"] = {
            "generated_n_tokens": generated_n_tokens,
            "input_n_tokens": input_n_tokens,
            "generation_time_sec": elapsed_sec,
            "tokens_per_sec": (B * T) / elapsed_sec,
            "batch_size": B,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
            "world_size": world_size,
            "torch_version": torch.__version__,
        }

        if args.out:
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
        "--deterministic",
        action="store_true",
        help="Use deterministic algorithms wherever possible, may be slower",
    )

    parser.add_argument("--prompt", type=str, default="", help="Input prompt")

    parser.add_argument(
        "--out",
        action="store_true",
        default=False,
        help="If specified, prints the report to stdout. Defaults to no output.",
    )

    args = parser.parse_args()

    test_generate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
        deterministic=args.deterministic,
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
