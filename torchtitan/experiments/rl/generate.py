#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example inference script using TorchTitan models with vLLM LLMEngine.

This script uses the RL unified config_registry to configure both
the vLLM engine and sampling parameters.

Run: torchrun --nproc_per_node=4 \
      torchtitan/experiments/rl/generate.py --config rl_grpo_qwen3_30b_a3b_varlen
"""
from __future__ import annotations

import argparse
import os

# Must set spawn method before any CUDA operations or vLLM imports
# CUDA cannot be re-initialized in forked subprocesses
# See also https://docs.vllm.ai/en/v0.8.3/design/multiprocessing.html#python-multiprocessing
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl import config_registry
from torchtitan.experiments.rl.models.vllm_registry import (
    registry_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
)
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.tools.utils import has_cuda_capability


logger = init_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TorchTitan RL/vLLM generator config standalone."
    )
    parser.add_argument(
        "--config",
        default="rl_grpo_qwen3_0_6b_varlen",
        help="RL config_registry function to instantiate.",
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
        help="Send --prompt directly to vLLM instead of rendering a chat prompt.",
    )
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    return parser.parse_args()


def generate() -> None:
    args = _parse_args()

    config_factory = getattr(config_registry, args.config, None)
    if not callable(config_factory):
        raise ValueError(f"Unknown RL config {args.config!r}")
    config = config_factory()
    gen_config = config.generator
    model_path = config.hf_assets_path
    max_num_seqs = args.max_num_seqs
    is_rank0 = os.environ.get("RANK", "0") == "0"

    # Register TorchTitan model with vLLM before engine creation
    registry_to_vllm(
        config.model_spec,
        parallelism=gen_config.parallelism,
        compile_config=config.compile,
        checkpoint_config=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_path=model_path,
        ),
    )
    logger.info("Registered TorchTitan model with vLLM")

    inner_attn = config.model_spec.model.layers[0].attention.inner_attention
    if not isinstance(inner_attn, (VarlenAttention.Config, FlexAttention.Config)):
        raise ValueError("Only varlen and flex attention backends are supported.")

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    set_batch_invariance(gen_config.debug.batch_invariant)
    enable_ep = gen_config.parallelism.expert_parallel_degree > 1

    logger.debug("Initializing vLLM LLMEngine with TorchTitan model")
    logger.debug(f"Model: {model_path}")
    logger.debug(
        f"Tensor Parallel Size: {gen_config.parallelism.tensor_parallel_degree}"
    )
    logger.debug(f"Expert Parallel Enabled: {enable_ep}")

    # Create EngineArgs from config
    engine_kwargs = dict(
        # Model configuration
        model=model_path,
        trust_remote_code=True,
        config_format=TORCHTITAN_CONFIG_FORMAT,
        dtype=gen_config.model_dtype,
        # Parallelism configuration
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        enable_expert_parallel=enable_ep,
        # Use external_launcher only when launched via torchrun (multi-GPU);
        # for single-GPU, let vLLM pick the default executor.
        distributed_executor_backend=("external_launcher"),
        # Memory and performance
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=not gen_config.cudagraph.enable,
        attention_config=AttentionConfig(
            backend=(
                AttentionBackendEnum.FLEX_ATTENTION
                if isinstance(inner_attn, FlexAttention.Config)
                else AttentionBackendEnum.CUSTOM
            ),
        ),
        disable_log_stats=False,
    )
    engine_kwargs["max_model_len"] = config.model_spec.model.max_seq_len
    engine_kwargs["max_num_seqs"] = max_num_seqs
    if not has_cuda_capability(9, 0):
        engine_kwargs["block_size"] = 256
    vllm_compilation_config = gen_config.cudagraph.get_vllm_compilation_config(
        max_num_seqs=max_num_seqs,
    )
    if vllm_compilation_config is not None:
        engine_kwargs["compilation_config"] = vllm_compilation_config
    if gen_config.debug.seed is not None:
        engine_kwargs["seed"] = gen_config.debug.seed
    engine_args = EngineArgs(**engine_kwargs)

    logger.debug("Initializing LLMEngine from EngineArgs...")
    engine = LLMEngine.from_engine_args(engine_args)

    logger.debug("vLLM LLMEngine initialized successfully")

    renderer = config.renderer.build(tokenizer_path=model_path)
    stop_token_ids = list(renderer.get_stop_token_ids())

    # Create sampling parameters from config
    sampling = gen_config.sampling
    temperature = sampling.temperature if args.temperature is None else args.temperature
    top_p = sampling.top_p if args.top_p is None else args.top_p
    max_tokens = sampling.max_tokens if args.max_tokens is None else args.max_tokens
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        stop_token_ids=stop_token_ids or None,
        seed=gen_config.debug.seed,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    logger.debug(
        f"Sampling params: temperature={temperature}, "
        f"top_p={top_p}, max_tokens={max_tokens}"
    )

    prompt = args.prompt
    logger.debug(f"Prompt: {prompt}")

    # Add request to engine
    logger.debug("Adding request to engine...")
    request_id = "0"
    if args.raw_prompt:
        engine_input = prompt
    else:
        prompt_token_ids = renderer.render_ids(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            add_generation_prompt=True,
        )
        engine_input = engine.renderer.render_cmpl(
            [{"prompt_token_ids": prompt_token_ids}]
        )[0]
        if is_rank0:
            print(f"Prompt token count: {len(prompt_token_ids)}", flush=True)
            print(f"Stop token ids: {stop_token_ids}", flush=True)
    engine.add_request(request_id, engine_input, sampling_params)

    # Generate text by stepping through engine
    logger.debug("Generating text...")
    while engine.has_unfinished_requests():
        request_outputs = engine.step()

        # Process finished requests
        for request_output in request_outputs:
            if request_output.finished:
                generated_text = request_output.outputs[0].text
                output_token_ids = request_output.outputs[0].token_ids

                # Print results
                logger.debug("Generation complete")
                if is_rank0:
                    print(f"\nConfig: {args.config}", flush=True)
                    print(f"Prompt: {prompt}", flush=True)
                    print(f"Generated token count: {len(output_token_ids)}", flush=True)
                    print(f"Generated text: {generated_text!r}\n", flush=True)


if __name__ == "__main__":
    generate()
