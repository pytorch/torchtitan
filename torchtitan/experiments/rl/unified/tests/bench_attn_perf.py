#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark vLLM inference performance: FLASH_ATTN vs CUSTOM attention backend.

Measures prefill and decode throughput (tokens/s) for each backend using
workload sizes representative of RL rollout workflows.

Flow:
  1. Generate synthetic prompts of fixed token length.
  2. For each backend (FLASH_ATTN, CUSTOM):
     a. Create vLLM engine with that backend.
     b. Warmup pass (untimed).
     c. Prefill benchmark: max_tokens=1, measure prompt processing speed.
     d. Decode benchmark: max_tokens=N, measure token generation speed.
     e. Destroy engine, free GPU memory.
  3. Print comparison table.

Run:
    torchrun --nproc_per_node=2 \
        torchtitan/experiments/rl/unified/tests/bench_attn_perf.py
"""

import logging
import os
import time

# Must set spawn method before any CUDA operations or vLLM imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
import torch.distributed as dist

from torchtitan.config import CommConfig
from torchtitan.config.configs import ParallelismConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.rl.unified.actors.generator import (
    SamplingConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.unified.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.unified.models.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.unified.plugin import (
    register_model_to_vllm_model_registry,
    VLLM_MODEL_NAME,
)
from torchtitan.experiments.rl.unified.simple_grpo import RLTrainer
from torchtitan.models.qwen3 import model_registry

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

NUM_PROMPTS = 8  # Number of prompts per batch
PROMPT_LEN = 512  # Tokens per prompt
MAX_DECODE_TOKENS = 128  # Max tokens to generate per prompt
TEMPERATURE = 0.8  # Sampling temperature
TOP_P = 0.95  # Nucleus sampling threshold
NUM_WARMUP = 1  # Warmup iterations (untimed)
NUM_TIMED = 3  # Timed iterations (averaged)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _bench_config() -> RLTrainer.Config:
    """Benchmark config: Qwen3-0.6B, TP=2, conservative memory for in-process reuse."""
    return RLTrainer.Config(
        model_spec=model_registry("0.6B"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        trainer=PolicyTrainer.Config(
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            gpu_memory_limit=0.3,
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
            ),
            num_samples_per_prompt=1,
            sampling=SamplingConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_DECODE_TOKENS,
            ),
            attention_backend="FLASH_ATTN",
        ),
    )


# ---------------------------------------------------------------------------
# Engine creation
# ---------------------------------------------------------------------------


def create_engine(config, attention_backend: str) -> LLMEngine:
    """Create a vLLM LLMEngine with the given attention backend."""
    gen_config = config.generator
    model_path = config.hf_assets_path

    if attention_backend == "CUSTOM":
        init_batch_invariance(AttentionBackendEnum.CUSTOM)
    elif attention_backend == "FLASH_ATTN":
        init_batch_invariance(AttentionBackendEnum.FLASH_ATTN)
    else:
        raise ValueError(f"Unknown attention backend: {attention_backend}")

    engine_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype=gen_config.model_dtype,
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=gen_config.enforce_eager,
        hf_overrides={"architectures": [VLLM_MODEL_NAME]},
        attention_backend=attention_backend,
        compilation_config=CompilationConfig(
            backend="eager",
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
    )
    engine_args = EngineArgs(**engine_kwargs)

    if dist.get_rank() == 0:
        logger.info(f"Creating vLLM engine with attention_backend={attention_backend}")
    engine = LLMEngine.from_engine_args(engine_args)
    if dist.get_rank() == 0:
        logger.info("Engine ready.")
    return engine


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def generate_dummy_prompts(
    model_path: str, num_prompts: int, prompt_len: int
) -> list[list[int]]:
    """Generate synthetic prompts of exact token length."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Encode a long repeated string, then slice to exact length
    base_text = "The quick brown fox jumps over the lazy dog. " * 200
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)

    prompts = []
    for i in range(num_prompts):
        # Shift starting position slightly per prompt for variety
        offset = (i * 7) % max(1, len(base_ids) - prompt_len)
        token_ids = base_ids[offset : offset + prompt_len]
        # Pad if base_ids is too short (unlikely with 200 repeats)
        while len(token_ids) < prompt_len:
            token_ids = token_ids + base_ids[: prompt_len - len(token_ids)]
        prompts.append(token_ids[:prompt_len])

    return prompts


# ---------------------------------------------------------------------------
# Engine execution
# ---------------------------------------------------------------------------


_run_counter = 0


def run_engine(
    engine: LLMEngine,
    prompt_token_ids: list[list[int]],
    sampling_params: SamplingParams,
) -> tuple[float, int]:
    """Run all prompts through engine, return (elapsed_seconds, total_output_tokens)."""
    global _run_counter
    _run_counter += 1

    # Add all requests (unique IDs across runs to avoid conflicts)
    for i, token_ids in enumerate(prompt_token_ids):
        engine.add_request(
            f"run{_run_counter}_req{i}",
            {"prompt_token_ids": token_ids},
            sampling_params,
        )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Step until done
    all_outputs = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        all_outputs.extend(step_outputs)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Count generated tokens
    total_output_tokens = 0
    for output in all_outputs:
        for sample in output.outputs:
            total_output_tokens += len(sample.token_ids)

    return elapsed, total_output_tokens


# ---------------------------------------------------------------------------
# Benchmark a single backend
# ---------------------------------------------------------------------------


def benchmark_backend(
    config: RLTrainer.Config,
    backend_name: str,
    prompt_token_ids: list[list[int]],
    max_decode_tokens: int,
    num_warmup: int,
    num_timed: int,
) -> dict:
    """Benchmark a single attention backend, return performance metrics."""
    engine = create_engine(config, backend_name)

    total_prompt_tokens = sum(len(p) for p in prompt_token_ids)
    rank0 = dist.get_rank() == 0

    # --- Warmup ---
    if rank0:
        logger.info(f"[{backend_name}] Warmup ({num_warmup} iteration(s))...")
    for _ in range(num_warmup):
        warmup_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=1,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        run_engine(engine, prompt_token_ids, warmup_params)

    # --- Prefill benchmark (max_tokens=1) ---
    if rank0:
        logger.info(f"[{backend_name}] Prefill benchmark ({num_timed} runs)...")
    prefill_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    prefill_times = []
    for run_idx in range(num_timed):
        elapsed, _ = run_engine(engine, prompt_token_ids, prefill_params)
        prefill_times.append(elapsed)
        if rank0:
            logger.info(
                f"  Run {run_idx + 1}/{num_timed}: {elapsed:.4f}s "
                f"({total_prompt_tokens / elapsed:.0f} tok/s)"
            )
    avg_prefill_time = sum(prefill_times) / len(prefill_times)

    # --- Decode benchmark (max_tokens=max_decode_tokens) ---
    if rank0:
        logger.info(f"[{backend_name}] Decode benchmark ({num_timed} runs)...")
    decode_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_decode_tokens,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    decode_times = []
    decode_token_counts = []
    for run_idx in range(num_timed):
        elapsed, total_gen_tokens = run_engine(engine, prompt_token_ids, decode_params)
        decode_times.append(elapsed)
        decode_token_counts.append(total_gen_tokens)
        if rank0:
            decode_only = max(elapsed - avg_prefill_time, 1e-9)
            logger.info(
                f"  Run {run_idx + 1}/{num_timed}: {elapsed:.4f}s total, "
                f"~{decode_only:.4f}s decode, "
                f"{total_gen_tokens} tokens "
                f"({total_gen_tokens / decode_only:.0f} decode tok/s)"
            )
    avg_decode_total_time = sum(decode_times) / len(decode_times)
    avg_decode_tokens = sum(decode_token_counts) / len(decode_token_counts)

    # Decode-only time = total generation time minus prefill time
    avg_decode_only_time = max(avg_decode_total_time - avg_prefill_time, 1e-9)

    # Cleanup engine and free GPU memory before next backend
    del engine
    torch.cuda.empty_cache()

    return {
        "backend": backend_name,
        "prefill_tokens": total_prompt_tokens,
        "prefill_time": avg_prefill_time,
        "prefill_tok_s": total_prompt_tokens / avg_prefill_time,
        "decode_tokens": avg_decode_tokens,
        "decode_time": avg_decode_only_time,
        "decode_tok_s": avg_decode_tokens / avg_decode_only_time,
    }


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def print_results(results: list[dict]) -> None:
    """Print comparison table (rank 0 only)."""
    if dist.get_rank() != 0:
        return

    header = (
        f"\n{'=' * 64}\n"
        f"  Attention Backend Performance Comparison\n"
        f"  Model: Qwen3-0.6B | TP: 2 | Prompts: {NUM_PROMPTS} "
        f"| Prompt len: {PROMPT_LEN}\n"
        f"  Decode tokens: {MAX_DECODE_TOKENS} | "
        f"Warmup: {NUM_WARMUP} | Timed runs: {NUM_TIMED}\n"
        f"{'=' * 64}\n"
    )
    print(header)

    row_fmt = "{:<14s} {:<10s} {:>8s} {:>12s} {:>10s}"
    print(row_fmt.format("Backend", "Phase", "Tokens", "Time (s)", "Tok/s"))
    print("-" * 58)

    for r in results:
        print(
            row_fmt.format(
                r["backend"],
                "Prefill",
                str(int(r["prefill_tokens"])),
                f"{r['prefill_time']:.4f}",
                f"{r['prefill_tok_s']:.0f}",
            )
        )
        print(
            row_fmt.format(
                r["backend"],
                "Decode",
                str(int(r["decode_tokens"])),
                f"{r['decode_time']:.4f}",
                f"{r['decode_tok_s']:.0f}",
            )
        )

    if len(results) == 2:
        flash, custom = results[0], results[1]
        prefill_speedup = custom["prefill_tok_s"] / flash["prefill_tok_s"]
        decode_speedup = custom["decode_tok_s"] / flash["decode_tok_s"]
        print(
            f"\nSpeedup ({custom['backend']} / {flash['backend']}):\n"
            f"  Prefill: {prefill_speedup:.2f}x\n"
            f"  Decode:  {decode_speedup:.2f}x"
        )

    print(f"{'=' * 64}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    config = _bench_config()
    config.model_spec.parallelize_fn = parallelize_qwen3

    # Register model with vLLM
    register_model_to_vllm_model_registry(config.model_spec)

    # Initialize distributed
    dist_utils.init_distributed(CommConfig())

    if dist.get_rank() == 0:
        logger.info(
            f"Benchmark config: {NUM_PROMPTS} prompts, "
            f"{PROMPT_LEN} prompt tokens, "
            f"{MAX_DECODE_TOKENS} decode tokens, "
            f"{NUM_WARMUP} warmup, {NUM_TIMED} timed runs"
        )

    # Generate dummy prompts
    model_path = config.hf_assets_path
    prompt_token_ids = generate_dummy_prompts(model_path, NUM_PROMPTS, PROMPT_LEN)

    if dist.get_rank() == 0:
        logger.info(
            f"Generated {len(prompt_token_ids)} prompts, "
            f"each {len(prompt_token_ids[0])} tokens"
        )

    # Benchmark FLASH_ATTN
    flash_results = benchmark_backend(
        config, "FLASH_ATTN", prompt_token_ids, MAX_DECODE_TOKENS, NUM_WARMUP, NUM_TIMED
    )

    # Benchmark CUSTOM
    custom_results = benchmark_backend(
        config, "CUSTOM", prompt_token_ids, MAX_DECODE_TOKENS, NUM_WARMUP, NUM_TIMED
    )

    # Print comparison
    print_results([flash_results, custom_results])


if __name__ == "__main__":
    main()
