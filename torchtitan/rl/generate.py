#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example inference script: generate from one prompt with an RL recipe's generator engine
(vLLM or SGLang), outside the RL loop.

This script uses an RL recipe to configure both the engine and sampling parameters.
Recipes are selected like ``torchtitan.rl.train``: by ``--module`` and ``--config``.

Run: torchrun --nproc_per_node=4 -m torchtitan.rl.generate \
      --module torchtitan_recipes.rl.alphabet_sort \
      --config rl_grpo_qwen3_30b_a3b_varlen
"""
from __future__ import annotations

# Apply process env defaults before anything imports torch.
from torchtitan.rl._env import apply_env_defaults

apply_env_defaults()

import argparse
import os
import sys
import tempfile
import time

from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import apply_overrides, ConfigLoader
from torchtitan.distributed.batch_invariant import set_batch_invariance
from torchtitan.rl.generator.types import EngineRequest, SamplingConfig
from torchtitan.rl.rollout.types import Prompt
from torchtitan.tools import filesystem


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse generation flags; remaining args select the recipe via ``ConfigLoader``."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a TorchTitan RL generator config standalone. Select the "
            "recipe with --module and --config (see torchtitan.rl.train)."
        )
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Sort these names alphabetically and put the final answer inside "
            "<alphabetical_sorted>...</alphabetical_sorted>: Charlie, Alice, Bob."
        ),
        help="User prompt to generate from.",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Tokenize --prompt as is instead of rendering a chat prompt.",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    return parser.parse_known_args()


@record
def generate() -> None:
    args, config_args = _parse_args()
    config = ConfigLoader().load(config_args)
    if config.override.imports:
        apply_overrides(config.override, config)
    gen_config = config.generator
    model_path = config.hf_assets_path
    # Recipes name local assets relative to the repo root, while
    # initial_load_path must be absolute or remote.
    if not filesystem.is_remote(model_path):
        model_path = os.path.abspath(model_path)
    is_rank0 = os.environ.get("RANK", "0") == "0"

    set_batch_invariance(gen_config.debug.batch_invariant)
    engine = gen_config.engine.build(
        model=gen_config.model,
        parallelism=gen_config.parallelism,
        model_path=model_path,
        max_num_seqs=args.max_num_seqs,
        precision=config.precision,
        gpu_memory_limit=gen_config.gpu_memory_limit,
        max_num_batched_tokens=gen_config.max_num_batched_tokens,
        checkpointer=CheckpointManager.Config(
            initial_load_in_hf=True, initial_load_path=model_path
        ),
        dist_moe_runtime=gen_config.dist_moe_runtime,
        debug=gen_config.debug,
        reset_kv_cache_on_model_state_dict_load=False,
        model_state_dict_load_mode="cpu_prefetch",
        vision_processor=None,
        mooncake_master_address=None,
        rank=int(os.environ.get("RANK", "0")),
        generator_name="generate",
        output_dir=tempfile.gettempdir(),
    )
    try:
        tokenizer = config.tokenizer.build(tokenizer_path=model_path)
        renderer = config.renderer.build(tokenizer=tokenizer)
        stop_token_ids = list(renderer.get_stop_token_ids())
        sampling = gen_config.sampling
        if args.raw_prompt:
            prompt_token_ids = tokenizer.encode(args.prompt)
        else:
            prompt_token_ids = renderer.render_ids(
                messages=[{"role": "user", "content": args.prompt}],
                tools=None,
                add_generation_prompt=True,
            )
        request = EngineRequest(
            request_id="0",
            prompt=Prompt(token_ids=prompt_token_ids),
            sampling=SamplingConfig(
                temperature=(
                    sampling.temperature
                    if args.temperature is None
                    else args.temperature
                ),
                max_tokens=(
                    sampling.max_tokens if args.max_tokens is None else args.max_tokens
                ),
                seed=gen_config.debug.seed,
                # Stop only on the renderer's ids, as the RL generator does.
                stop_token_ids=stop_token_ids,
            ),
            group_id=0,
            routing_session_id="0",
            arrival_time=time.time(),
        )
        request.min_policy_version = 0
        if is_rank0:
            print(f"Prompt token count: {len(prompt_token_ids)}", flush=True)
            print(f"Stop token ids: {stop_token_ids}", flush=True)
        for _, error in engine.add_requests([request]):
            raise error
        finished = []
        while engine.has_unfinished_requests():
            finished.extend(engine.step())
        [output] = finished
        if is_rank0:
            print(f"\nConfig: {' '.join(config_args)}", flush=True)
            print(f"Prompt: {args.prompt}", flush=True)
            print(f"Generated token count: {len(output.token_ids)}", flush=True)
            print(
                f"Generated text: {tokenizer.decode(output.token_ids)!r}\n", flush=True
            )
    finally:
        engine.shutdown()
        # This script owns its process (in RL, Monarch's ProcMesh.stop() ends it): tear
        # down vLLM's process groups so the NCCL watchdog doesn't query CUDA events
        # while the driver unloads. SGLang exits cleanly without it. The vLLM engine
        # module is imported only when the recipe uses it.
        vllm_engine = sys.modules.get("torchtitan.rl.generator.vllm.engine")
        if vllm_engine is not None and isinstance(engine, vllm_engine.VLLMEngine):
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            cleanup_dist_env_and_memory()


if __name__ == "__main__":
    generate()
