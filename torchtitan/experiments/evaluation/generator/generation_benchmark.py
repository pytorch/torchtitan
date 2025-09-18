# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import time
import argparse

import torch

from torchtitan.experiments.evaluation.generator.utils import load_transformer_generator


def main(args):

    if args.max_gen_len is None:
        args.max_gen_len = args.max_tokens - args.prompt_length
    print("Loading model...")

    generator = load_transformer_generator(
        model_type_size=args.model,
        exp_name=args.exp_name,
        max_seq_len=args.max_tokens,
        args={
            "max_gen_len": args.max_gen_len,
            "max_tokens": args.max_tokens,
            "compile_prefilling": not args.disable_compile_prefilling,
            "reduce_generation_overhead": not args.disable_reduce_generation_overhead,
            "dtype": args.dtype,
            "model_type_size": args.model,
            "dp_degree": args.dp_degree,
            "show_progress": False,  # Disable progress bar during benchmark.
        },
    )

    print("Model loading complete.")

    # Generate random prompts with the specified token length.
    # These are composed of random token IDs, not actual words.
    # vocab_size = generator.tokenizer._n_words
    prompts_tokens = [
        torch.randint(0, generator.tokenizer.bos_id, (args.prompt_length,)).tolist()
        for _ in range(args.num_prompts)
    ]
    # Decode the token ID lists back into strings to pass to the generate function.
    prompts = [generator.tokenizer.decode(p) for p in prompts_tokens]

    print("\n--- Benchmark Settings ---")
    print(f"Model: {args.model}")
    print(f"Number of prompts: {args.num_prompts}")
    print(f"Prompt length (tokens): {args.prompt_length}")
    print(f"Generation length (tokens): {args.max_gen_len}")
    print(f"Warmup runs: {args.warmup_runs}")
    print(f"Benchmark runs: {args.benchmark_runs}")
    print("--------------------------\n")

    # Warmup phase
    print("Starting warmup...")
    for i in range(args.warmup_runs):
        _ = generator.generate(prompts, stop_on_eos=False)
        print(f"Warmup {i + 1}/{args.warmup_runs} complete")
    print("Warmup finished.")

    # Benchmark phase
    print("\nStarting benchmark...")
    total_generated_tokens = 0
    all_run_times = []
    all_peak_memory = []

    for i in range(args.benchmark_runs):
        torch.cuda.synchronize()
        # Reset peak memory stats before the run
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()

        generation, _, _ = generator.generate(prompts, stop_on_eos=False)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        run_time = end_time - start_time
        all_run_times.append(run_time)

        # Record the peak memory usage for this run
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        all_peak_memory.append(peak_memory_mb)

        # Calculate the actual number of tokens generated in this run.
        run_tokens = sum(len(generator.tokenizer.encode(g, bos=False, eos=False)) for g in generation)
        total_generated_tokens += run_tokens

        print(f"Benchmark {i + 1}/{args.benchmark_runs} complete. Time taken: {run_time:.2f}s, Tokens generated: {run_tokens}")

    total_time = sum(all_run_times)
    
    # Calculate throughput
    throughput = total_generated_tokens / total_time
    avg_peak_memory = sum(all_peak_memory) / len(all_peak_memory)

    print("\n--- Benchmark Results ---")
    print(f"Total time for {args.benchmark_runs} runs: {total_time:.2f}s")
    print(f"Total tokens generated: {total_generated_tokens}")
    print(f"Throughput (Tokens/sec): {throughput:.2f}")
    print(f"Average Peak GPU Memory: {avg_peak_memory:.2f} MB")
    print("-------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark script for TorchTitan Transformer generation speed.")
    # Arguments from the original script
    parser.add_argument("--model", type=str, required=True, help="Model type and size.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name.")
    parser.add_argument("--dtype", type=str, default="bf16", help="Model data type.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens the model can process at once.")
    parser.add_argument("--dp_degree", type=int, default=1, help="FSDP degree.")
    parser.add_argument("--disable_compile_prefilling", action="store_false", help="Disable torch.compile for prefilling.")
    parser.add_argument("--disable_reduce_generation_overhead", action="store_false", help="Disable reduce-overhead mode for generation.")

    # Arguments for benchmarking
    parser.add_argument("--num_prompts", type=int, default=1, help="Number of prompts to process simultaneously.")
    parser.add_argument("--prompt_length", type=int, default=128, help="Length of each prompt in tokens.")
    parser.add_argument("--max_gen_len", type=int, default=None, help="Maximum number of tokens to generate for each prompt.")
    parser.add_argument("--warmup_runs", type=int, default=1, help="Number of warmup runs before benchmarking.")
    parser.add_argument("--benchmark_runs", type=int, default=3, help="Number of benchmark runs to average.")

    cli_args = parser.parse_args()
    main(cli_args)