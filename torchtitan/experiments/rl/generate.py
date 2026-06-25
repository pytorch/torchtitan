#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference + benchmark script using TorchTitan models with vLLM LLMEngine.

This script uses the RL unified config_registry to configure both the vLLM
engine and sampling parameters.

1. Example mode (default): generate from a single prompt.
   torchrun --nproc_per_node=4 torchtitan/experiments/rl/generate.py \
       --config rl_grpo_qwen3_30b_a3b_varlen

2. Benchmark mode (--benchmark): measure generation throughput. The tokenizer
   is skipped (synthetic token-id prompts are fed to vLLM directly) and the
   engine is built OUTSIDE the timed region, so only generation is timed.
   torchrun --nproc_per_node=8 torchtitan/experiments/rl/generate.py \
       --config rl_grpo_qwen3_32b --benchmark --input-len 1024 \
       --batch-size 8 --max-tokens 128 --ignore-eos \
       --model-path <tokenizer-dir> --allreduce-backend vllm
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

# Must set spawn method before any CUDA operations or vLLM imports
# CUDA cannot be re-initialized in forked subprocesses
# See also https://docs.vllm.ai/en/v0.8.3/design/multiprocessing.html#python-multiprocessing
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.examples.alphabet_sort import config_registry
from torchtitan.experiments.rl.models.vllm_registry import (
    register_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
)
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.tools.utils import has_cuda_capability


logger = init_logger(__name__)


def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/benchmark a TorchTitan RL/vLLM generator config standalone."
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
        help="User prompt to generate from (example mode).",
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
    parser.add_argument(
        "--allreduce-backend",
        choices=["nccl", "vllm"],
        default=None,
        help="Override the generator TP all-reduce backend (default: from config).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override config.hf_assets_path (tokenizer/HF-assets dir). Useful in "
        "benchmark mode to point the tokenizer at a local checkpoint while the "
        "model itself is built from the config's model_spec.",
    )

    bench = parser.add_argument_group("benchmark")
    bench.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure generation throughput (excludes engine startup/capture).",
    )
    bench.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of concurrent requests per generation pass.",
    )
    bench.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Synthetic prompt length in tokens (required in --benchmark mode).",
    )
    bench.add_argument("--warmup-runs", type=int, default=2)
    bench.add_argument("--num-runs", type=int, default=5)
    bench.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Force exactly --max-tokens per request (stable, deterministic timing).",
    )
    return parser.parse_args()


def _resolve_sampling(gen_config, args: argparse.Namespace):
    sampling = gen_config.sampling
    temperature = sampling.temperature if args.temperature is None else args.temperature
    top_p = sampling.top_p if args.top_p is None else args.top_p
    max_tokens = sampling.max_tokens if args.max_tokens is None else args.max_tokens
    return temperature, top_p, max_tokens


def _effective_allreduce_backend(gen_config) -> str:
    """Report the all-reduce backend actually used. ``"vllm"`` is disabled under
    batch-invariant mode (vLLM's custom AR is size-dependent), so the wrapper
    keeps DTensor's NCCL path; reflect that here instead of the configured value.
    """
    backend = gen_config.parallelism.allreduce_backend
    if backend == "vllm" and gen_config.debug.batch_invariant:
        return "nccl (vllm disabled under batch_invariant)"
    return backend


def _build_torchtitan_engine(
    config, model_path: str, max_num_seqs: int, *, benchmark: bool
) -> LLMEngine:
    """Build a vLLM LLMEngine running the TorchTitan unified model."""
    gen_config = config.generator

    register_to_vllm(
        config.model_spec,
        parallelism=gen_config.parallelism,
        compile_config=config.compile,
        checkpoint_config=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_path=model_path,
        )
        if not benchmark
        else CheckpointManager.Config(enable=False),
        override=config.generator.override,
    )
    logger.info("Registered TorchTitan model with vLLM")

    inner_attn = config.model_spec.model.layers[0].attention.inner_attention
    if not isinstance(inner_attn, (VarlenAttention.Config, FlexAttention.Config)):
        raise ValueError("Only varlen and flex attention backends are supported.")

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    set_batch_invariance(gen_config.debug.batch_invariant)
    if gen_config.debug.batch_invariant:
        # batch_invariant_ops doesn't cover bmm; the MoE router gate is a bmm in
        # the vLLM inference graph, so override it generator-side (not in core).
        from torchtitan.experiments.rl.batch_invariance import (
            patch_bmm_for_batch_invariance,
        )

        patch_bmm_for_batch_invariance()
    enable_ep = gen_config.parallelism.expert_parallel_degree > 1

    engine_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        config_format=TORCHTITAN_CONFIG_FORMAT,
        dtype=gen_config.model_dtype,
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        data_parallel_size=gen_config.parallelism.data_parallel_degree,
        enable_expert_parallel=enable_ep,
        distributed_executor_backend="external_launcher",
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
    if benchmark:
        # Avoid cross-request / cross-run prefill dedup so prefill is measured.
        engine_kwargs["enable_prefix_caching"] = False
    vllm_compilation_config = gen_config.cudagraph.get_vllm_compilation_config(
        max_num_seqs=max_num_seqs,
    )
    if vllm_compilation_config is not None:
        engine_kwargs["compilation_config"] = vllm_compilation_config
    if gen_config.debug.seed is not None:
        engine_kwargs["seed"] = gen_config.debug.seed

    engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))
    logger.info("vLLM LLMEngine (TorchTitan model) initialized")
    return engine


def _make_token_id_lists(
    vocab_size: int, input_len: int, batch_size: int
) -> list[list[int]]:
    """Synthetic prompt token-id lists, distinct across requests so prefix
    caching cannot dedup their prefill. Skips the tokenizer entirely."""
    span = vocab_size - 1024
    token_id_lists = []
    for b in range(batch_size):
        base = (b * 1009) % span + 256
        token_id_lists.append([base + (i % 257) for i in range(input_len)])
    return token_id_lists


def _run_pass(engine, prompts, sampling_params) -> int:
    """Add all prompts as concurrent requests, step to completion, return the
    total number of generated tokens."""
    for i, prompt in enumerate(prompts):
        engine.add_request(str(i), prompt, sampling_params)
    total_generated = 0
    while engine.has_unfinished_requests():
        for request_output in engine.step():
            if request_output.finished:
                total_generated += len(request_output.outputs[0].token_ids)
    return total_generated


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(config, args: argparse.Namespace) -> None:
    if args.input_len is None:
        raise ValueError("--input-len is required in --benchmark mode.")

    gen_config = config.generator
    temperature, top_p, max_tokens = _resolve_sampling(gen_config, args)
    batch_size = args.batch_size
    # vLLM must run the whole batch concurrently; this also drives cudagraph
    # capture sizes.
    max_num_seqs = max(args.max_num_seqs, batch_size)
    vocab_size = config.model_spec.model.vocab_size
    model_path = os.path.abspath(args.model_path or config.hf_assets_path)

    # ---- Engine build (EXCLUDED from timing) ----
    build_t0 = time.perf_counter()
    engine = _build_torchtitan_engine(config, model_path, max_num_seqs, benchmark=True)
    _sync()
    build_time = time.perf_counter() - build_t0

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        ignore_eos=args.ignore_eos,
        seed=gen_config.debug.seed,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    token_id_lists = _make_token_id_lists(vocab_size, args.input_len, batch_size)
    # render_cmpl wraps prompt_token_ids without tokenizing (skips tokenizer).
    prompts = engine.renderer.render_cmpl(
        [{"prompt_token_ids": ids} for ids in token_id_lists]
    )

    # ---- Warmup (EXCLUDED) ----
    for _ in range(args.warmup_runs):
        _run_pass(engine, prompts, sampling_params)
    _sync()

    # ---- Timed runs ----
    durations: list[float] = []
    generated_counts: list[int] = []
    for _ in range(args.num_runs):
        _sync()
        t0 = time.perf_counter()
        n = _run_pass(engine, prompts, sampling_params)
        _sync()
        durations.append(time.perf_counter() - t0)
        generated_counts.append(n)

    if not _is_rank0():
        return

    throughputs = [n / d for n, d in zip(generated_counts, durations)]
    mean_tps = statistics.mean(throughputs)
    std_tps = statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0
    compile_tag = "off" if not config.compile.enable else config.compile.backend
    cg_tag = f"{gen_config.cudagraph.mode}" if gen_config.cudagraph.enable else "off"
    ar_tag = _effective_allreduce_backend(gen_config)

    print("\n" + "=" * 72, flush=True)
    print(f"BENCHMARK  config={args.config}  model=torchtitan", flush=True)
    print(
        f"  TP={gen_config.parallelism.tensor_parallel_degree}  "
        f"compile={compile_tag}  cudagraph={cg_tag}  allreduce={ar_tag}",
        flush=True,
    )
    print(
        f"  workload: batch={batch_size} input_len={args.input_len} "
        f"gen={max_tokens}  (warmup={args.warmup_runs} runs={args.num_runs})",
        flush=True,
    )
    print(f"  engine build (excluded): {build_time:.1f}s", flush=True)
    print(
        f"  throughput: {mean_tps:.1f} +/- {std_tps:.1f} tok/s "
        f"(per-run: {', '.join(f'{t:.1f}' for t in throughputs)})",
        flush=True,
    )
    print("=" * 72 + "\n", flush=True)


def generate(config, args: argparse.Namespace) -> None:
    gen_config = config.generator
    model_path = os.path.abspath(args.model_path or config.hf_assets_path)
    max_num_seqs = args.max_num_seqs

    engine = _build_torchtitan_engine(config, model_path, max_num_seqs, benchmark=False)
    if _is_rank0():
        print(
            f"TP={gen_config.parallelism.tensor_parallel_degree} "
            f"allreduce={_effective_allreduce_backend(gen_config)}",
            flush=True,
        )

    renderer = config.renderer.build(tokenizer_path=model_path)
    stop_token_ids = list(renderer.get_stop_token_ids())

    temperature, top_p, max_tokens = _resolve_sampling(gen_config, args)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        stop_token_ids=stop_token_ids or None,
        seed=gen_config.debug.seed,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    prompt = args.prompt
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
        if _is_rank0():
            print(f"Prompt token count: {len(prompt_token_ids)}", flush=True)
            print(f"Stop token ids: {stop_token_ids}", flush=True)
    engine.add_request("0", engine_input, sampling_params)

    while engine.has_unfinished_requests():
        for request_output in engine.step():
            if request_output.finished:
                generated_text = request_output.outputs[0].text
                output_token_ids = request_output.outputs[0].token_ids
                if _is_rank0():
                    print(f"\nConfig: {args.config}", flush=True)
                    print(f"Prompt: {prompt}", flush=True)
                    print(f"Generated token count: {len(output_token_ids)}", flush=True)
                    print(f"Generated text: {generated_text!r}\n", flush=True)


def main() -> None:
    args = _parse_args()

    config_factory = getattr(config_registry, args.config, None)
    if not callable(config_factory):
        raise ValueError(f"Unknown RL config {args.config!r}")
    config = config_factory()
    if args.allreduce_backend is not None:
        config.generator.parallelism.allreduce_backend = args.allreduce_backend

    if args.benchmark:
        benchmark(config, args)
    else:
        generate(config, args)


if __name__ == "__main__":
    main()
