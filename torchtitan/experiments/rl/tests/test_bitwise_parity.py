#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test bitwise parity between vLLM generator and TorchTitan trainer log-probs.

Instead of comparing decode logprobs (which differ due to incremental decode vs
full-sequence forward), this test establishes parity via three checks:

  Test 1: Trainer prefill == vLLM prefill
          (both do a single forward pass over prompt tokens)
  Test 2: vLLM decode == vLLM 2nd-pass prefill
          (decode logprobs should match re-prefilling the full sequence)
  Test 3: Trainer prefill == vLLM 2nd-pass prefill (full sequence)
          (trainer forward over [prompt + generated] matches vLLM prefill)

By transitivity: trainer == vLLM decode.

Run:
    torchrun --nproc_per_node=1 \
        torchtitan/experiments/rl/tests/test_bitwise_parity.py

    # With batch size and max generation length:
    torchrun --nproc_per_node=1 \
        torchtitan/experiments/rl/tests/test_bitwise_parity.py \
        --batch-size 2 --prompt-length 4000
"""

import argparse
import logging
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from torchtitan.config import CommConfig
from torchtitan.config.configs import ParallelismConfig, TrainingConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.rl.actors.generator import (
    GeneratorCompileConfig,
    SamplingConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.actors.utils import (
    _make_causal_varlen_metadata,
    compute_token_log_probs,
)
from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.plugin import register_model_to_vllm_model_registry
from torchtitan.experiments.rl.simple_grpo_sum_digits import RLTrainer
from torchtitan.experiments.rl.tests.test_attn_numerics import (
    build_trainer_model,
    create_vllm_engine,
)
from torchtitan.models.qwen3 import model_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_vllm_prefill_logprobs(output, prompt_token_ids):
    """Extract per-token prefill logprobs from a vLLM RequestOutput.

    vLLM's prompt_logprobs[0] is None (no logprob for the first token).
    For positions 1..N-1, we look up the actual token's logprob.

    Returns a list of floats with len = len(prompt_token_ids) - 1.
    """
    logprobs = []
    for i, token_lp_dict in enumerate(output.prompt_logprobs):
        if token_lp_dict is None:
            # Position 0 has no logprob
            continue
        actual_token_id = prompt_token_ids[i]
        if actual_token_id in token_lp_dict:
            logprobs.append(token_lp_dict[actual_token_id].logprob)
        else:
            logger.warning(
                f"Token {actual_token_id} not found at position {i}, "
                f"available: {list(token_lp_dict.keys())}"
            )
            best = max(token_lp_dict.values(), key=lambda x: x.logprob)
            logprobs.append(best.logprob)
    return logprobs


def compute_trainer_prefill_logprobs(model, token_ids, device):
    """Compute next-token logprobs over a token sequence using the trainer model.

    Returns a float32 tensor of logprobs with len = len(token_ids) - 1.
    """
    input_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    seq_len = input_tensor.shape[1]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    attn_backend = getattr(
        getattr(getattr(model, "config", None), "layer", None),
        "attention",
        None,
    )
    if (
        attn_backend is not None
        and getattr(attn_backend, "attn_backend", None) == "varlen"
    ):
        attention_masks = _make_causal_varlen_metadata(1, seq_len, device)
    else:
        attention_masks = None

    logits = model(input_tensor, attention_masks=attention_masks, positions=positions)
    logits_f32 = logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits_f32, dim=-1)

    targets = input_tensor[:, 1:]
    token_lps = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return token_lps[0]  # Remove batch dim


def compare_logprobs(name, logprobs_a, logprobs_b, label_a="A", label_b="B"):
    """Compare two logprob sequences and log results. Returns True if bitwise identical."""
    if isinstance(logprobs_a, list):
        logprobs_a = torch.tensor(logprobs_a, dtype=torch.float32)
    if isinstance(logprobs_b, list):
        logprobs_b = torch.tensor(logprobs_b, dtype=torch.float32)

    logprobs_a = logprobs_a.detach().cpu().float()
    logprobs_b = logprobs_b.detach().cpu().float()

    n = min(len(logprobs_a), len(logprobs_b))
    a, b = logprobs_a[:n], logprobs_b[:n]

    bitwise = torch.equal(a, b)
    diff = a - b
    max_delta = diff.abs().max().item() if n > 0 else 0.0
    mse = (diff**2).mean().item() if n > 0 else 0.0
    num_different = (a != b).sum().item()

    logger.info(f"  {name}")
    logger.info(f"    Bitwise identical : {bitwise}")
    logger.info(f"    Tokens checked    : {n}")
    logger.info(f"    Tokens different  : {num_different}")
    logger.info(f"    Max delta         : {max_delta:.6e}")
    logger.info(f"    MSE               : {mse:.6e}")

    if not bitwise and n > 0:
        n_preview = min(5, n)
        logger.info(f"    {label_a}[:5] : {a[:n_preview].tolist()}")
        logger.info(f"    {label_b}[:5] : {b[:n_preview].tolist()}")

    return bitwise


# ---------------------------------------------------------------------------
# vLLM operations
# ---------------------------------------------------------------------------


def vllm_prefill(engine, prompt_token_ids):
    """Run vLLM prefill only (max_tokens=1) and extract prompt logprobs.

    Args:
        prompt_token_ids: Single list of token IDs, or list of lists for batched.

    Returns:
        List of logprobs (single) or list of list of logprobs (batched).
    """
    # Normalize to list-of-lists
    if isinstance(prompt_token_ids[0], int):
        batched_ids = [prompt_token_ids]
        single = True
    else:
        batched_ids = prompt_token_ids
        single = False

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        logprobs=1,
        prompt_logprobs=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    for i, ids in enumerate(batched_ids):
        engine.add_request(f"prefill_{i}", {"prompt_token_ids": ids}, sampling_params)

    outputs = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        outputs.extend(step_outputs)

    # Sort by request_id to maintain order
    outputs.sort(key=lambda o: o.request_id)
    assert len(outputs) == len(batched_ids)

    all_logprobs = []
    for output, ids in zip(outputs, batched_ids):
        logprobs = extract_vllm_prefill_logprobs(output, ids)
        logger.info(f"vLLM prefill: extracted {len(logprobs)} logprobs")
        all_logprobs.append(logprobs)

    return all_logprobs[0] if single else all_logprobs


def vllm_generate(engine, prompt_token_ids, max_tokens):
    """Generate tokens via vLLM and return generated IDs + decode logprobs.

    Args:
        prompt_token_ids: Single list of token IDs, or list of lists for batched.

    Returns:
        (generated_ids, decode_logprobs) for single, or lists of them for batched.
    """
    if isinstance(prompt_token_ids[0], int):
        batched_ids = [prompt_token_ids]
        single = True
    else:
        batched_ids = prompt_token_ids
        single = False

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        logprobs=1,
        prompt_logprobs=None,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    for i, ids in enumerate(batched_ids):
        engine.add_request(f"generate_{i}", {"prompt_token_ids": ids}, sampling_params)

    outputs = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        outputs.extend(step_outputs)

    outputs.sort(key=lambda o: o.request_id)
    assert len(outputs) == len(batched_ids)

    all_generated = []
    all_decode_lps = []
    for output in outputs:
        sample = output.outputs[0]
        generated_ids = list(sample.token_ids)
        decode_logprobs = [
            list(lp_dict.values())[0].logprob for lp_dict in sample.logprobs
        ]
        logger.info(f"vLLM generate: {len(generated_ids)} tokens")
        all_generated.append(generated_ids)
        all_decode_lps.append(decode_logprobs)

    if single:
        return all_generated[0], all_decode_lps[0]
    return all_generated, all_decode_lps


def vllm_2nd_pass_prefill(engine, prompt_token_ids, generated_token_ids):
    """Run a 2nd prefill over [prompt + generated] and extract logprobs for generated positions.

    Args:
        prompt_token_ids: Single list or list of lists.
        generated_token_ids: Single list or list of lists (matching prompt_token_ids).

    Returns:
        List of logprobs (single) or list of list of logprobs (batched).
    """
    if isinstance(prompt_token_ids[0], int):
        batched_prompt = [prompt_token_ids]
        batched_gen = [generated_token_ids]
        single = True
    else:
        batched_prompt = prompt_token_ids
        batched_gen = generated_token_ids
        single = False

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        logprobs=1,
        prompt_logprobs=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    all_combined = []
    for i, (p_ids, g_ids) in enumerate(zip(batched_prompt, batched_gen)):
        combined = list(p_ids) + list(g_ids)
        all_combined.append(combined)
        engine.add_request(
            f"2nd_prefill_{i}", {"prompt_token_ids": combined}, sampling_params
        )

    outputs = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        outputs.extend(step_outputs)

    outputs.sort(key=lambda o: o.request_id)
    assert len(outputs) == len(batched_prompt)

    all_logprobs = []
    for output, p_ids, combined in zip(outputs, batched_prompt, all_combined):
        prompt_len = len(p_ids)
        logprobs = []
        for i, token_lp_dict in enumerate(output.prompt_logprobs):
            if token_lp_dict is None:
                continue
            if i < prompt_len:
                continue
            actual_token_id = combined[i]
            if actual_token_id in token_lp_dict:
                logprobs.append(token_lp_dict[actual_token_id].logprob)
            else:
                best = max(token_lp_dict.values(), key=lambda x: x.logprob)
                logprobs.append(best.logprob)
        logger.info(
            f"vLLM 2nd-pass prefill: extracted {len(logprobs)} logprobs "
            f"for generated positions"
        )
        all_logprobs.append(logprobs)

    return all_logprobs[0] if single else all_logprobs


# ---------------------------------------------------------------------------
# Config & Main
# ---------------------------------------------------------------------------


def _test_config(tp: int = 1) -> RLTrainer.Config:
    return RLTrainer.Config(
        model_spec=model_registry(
            "0.6B", attn_backend_override="varlen", batch_invariant=True
        ),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        batch_invariant_mode=True,
        trainer=PolicyTrainer.Config(
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                tensor_parallel_degree=tp,
                data_parallel_replicate_degree=1,
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            gpu_memory_limit=0.5,
            parallelism=ParallelismConfig(
                tensor_parallel_degree=tp,
            ),
            compile=GeneratorCompileConfig(backend="none", cudagraph_mode="none"),
            num_samples_per_prompt=1,
            sampling=SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                max_tokens=50,
            ),
            attention_backend="CUSTOM",
        ),
    )


def run_scenario(
    engine, trainer_model, device, tokenizer, prompts, max_gen_tokens, label
):
    """Run all 3 tests for a given prompt/gen length configuration.

    Args:
        prompts: Single prompt string, list of prompt strings, or list of
            token ID lists (for pre-tokenized sequences).
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    # Detect whether prompts are strings or pre-tokenized
    if isinstance(prompts[0], str):
        all_prompt_ids = [tokenizer.encode(p) for p in prompts]
    else:
        all_prompt_ids = prompts

    batch_size = len(all_prompt_ids)

    if dist.get_rank() == 0:
        logger.info(f"\n{'#' * 60}")
        logger.info(f"SCENARIO: {label}")
        logger.info(f"  Batch size: {batch_size}")
        prompt_lens = [len(ids) for ids in all_prompt_ids]
        logger.info(f"  Prompt lengths: {prompt_lens}")
        logger.info(f"  Max gen tokens: {max_gen_tokens}")
        logger.info(f"{'#' * 60}")

    # Run vLLM operations with all prompts batched together
    vllm_prefill_lps_list = vllm_prefill(engine, all_prompt_ids)
    all_generated_ids, all_decode_lps = vllm_generate(
        engine, all_prompt_ids, max_gen_tokens
    )
    all_prefill_2nd_lps = vllm_2nd_pass_prefill(
        engine, all_prompt_ids, all_generated_ids
    )

    # Run trainer per-sequence (batch_size=1 each) and compare
    results = {"t1": True, "t2": True, "t3": True}

    for seq_idx in range(batch_size):
        prompt_ids = all_prompt_ids[seq_idx]
        gen_ids = all_generated_ids[seq_idx]
        vllm_pf_lps = vllm_prefill_lps_list[seq_idx]
        decode_lps = all_decode_lps[seq_idx]
        pf2_lps = all_prefill_2nd_lps[seq_idx]

        with torch.no_grad():
            trainer_prefill_lps = compute_trainer_prefill_logprobs(
                trainer_model, prompt_ids, device
            )
            trainer_full_lps = compute_token_log_probs(
                trainer_model, prompt_ids, gen_ids, device
            )

        if dist.get_rank() == 0:
            seq_label = f"[seq {seq_idx}]" if batch_size > 1 else ""
            logger.info("=" * 60)

            t1 = compare_logprobs(
                f"Test 1 {seq_label}: Trainer prefill vs vLLM prefill",
                trainer_prefill_lps,
                vllm_pf_lps,
                label_a="Trainer",
                label_b="vLLM",
            )
            t2 = compare_logprobs(
                f"Test 2 {seq_label}: vLLM decode vs vLLM 2nd-pass prefill",
                decode_lps,
                pf2_lps,
                label_a="Decode",
                label_b="2nd Prefill",
            )
            t3 = compare_logprobs(
                f"Test 3 {seq_label}: Trainer prefill vs vLLM 2nd-pass prefill (full seq)",
                trainer_full_lps,
                pf2_lps,
                label_a="Trainer",
                label_b="vLLM 2nd Prefill",
            )

            results["t1"] = results["t1"] and t1
            results["t2"] = results["t2"] and t2
            results["t3"] = results["t3"] and t3

            logger.info("-" * 60)
            logger.info(f"  Test 1 {seq_label}: {'PASS' if t1 else 'FAIL'}")
            logger.info(f"  Test 2 {seq_label}: {'PASS' if t2 else 'FAIL'}")
            logger.info(f"  Test 3 {seq_label}: {'PASS' if t3 else 'FAIL'}")

    return results


def _make_long_token_sequences(batch_size, prompt_length, tokenizer):
    """Create token ID sequences of varying lengths up to prompt_length.

    Creates synthetic sequences by encoding text and truncating to target
    lengths that vary across the batch (e.g. 3487 and 1150 for batch_size=2).

    Returns a list of token ID lists.
    """
    # Target lengths: distribute across the range [prompt_length//3, prompt_length]
    # to create sequences of varying lengths like MSL's real data
    target_lengths = []
    for i in range(batch_size):
        # Spread lengths: first seq close to max, others shorter
        frac = 1.0 - (i * 0.6 / max(batch_size - 1, 1))
        target_lengths.append(max(128, int(prompt_length * frac)))

    # Generate a long text block by repeating diverse passages
    passages = [
        "You are a helpful assistant that solves math problems step by step. "
        "Show your reasoning clearly and carefully before giving the final answer. ",
        "The fundamental theorem of calculus establishes the relationship between "
        "differentiation and integration. It states that if f is continuous on [a,b] "
        "and F is an antiderivative of f on [a,b], then the integral from a to b "
        "of f(x) dx equals F(b) minus F(a). ",
        "In reinforcement learning, an agent interacts with an environment to "
        "maximize cumulative reward. The policy maps states to actions, and the "
        "value function estimates expected future returns. ",
        "Consider the following optimization problem: minimize f(x) subject to "
        "g(x) <= 0 and h(x) = 0. The Lagrangian dual formulation introduces "
        "multipliers for each constraint. ",
        "The attention mechanism computes a weighted sum of values, where weights "
        "are determined by the compatibility between queries and keys. Multi-head "
        "attention projects inputs into multiple subspaces. ",
    ]

    all_sequences = []
    for idx, target_len in enumerate(target_lengths):
        # Use different starting passage per sequence for diversity
        text = ""
        passage_idx = idx
        while True:
            text += passages[passage_idx % len(passages)]
            tokens = tokenizer.encode(text)
            if len(tokens) >= target_len:
                break
            passage_idx += 1
        # Truncate to target length
        all_sequences.append(tokens[:target_len])

    return all_sequences


def main():
    parser = argparse.ArgumentParser(description="Test bitwise parity")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of sequences to batch in vLLM (default: 1)",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=0,
        help="Max prompt sequence length in tokens. Sequences of varying "
        "lengths up to this value are created. (default: use built-in scenarios)",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=50,
        help="Number of tokens to generate for decode tests (default: 50)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel degree (default: 1)",
    )
    # Filter out torchrun args (everything before --)
    args, _ = parser.parse_known_args()

    config = _test_config(tp=args.tp)
    config.model_spec.parallelize_fn = parallelize_qwen3

    from torch.nn.attention import (
        activate_flash_attention_impl,
        current_flash_attention_impl,
    )

    if current_flash_attention_impl() != "FA3":
        activate_flash_attention_impl("FA3")

    register_model_to_vllm_model_registry(config.model_spec)
    dist_utils.init_distributed(CommConfig())

    # ---- Build vLLM engine ----
    engine = create_vllm_engine(config)
    tokenizer = engine.get_tokenizer()

    # ---- Build trainer model ----
    trainer_model, device = build_trainer_model(config)

    # ---- Define scenarios ----
    if args.prompt_length > 0:
        # Custom scenario: long prompt sequences with specified batch size
        token_sequences = _make_long_token_sequences(
            args.batch_size, args.prompt_length, tokenizer
        )
        seq_lens = [len(s) for s in token_sequences]
        label = f"BS={args.batch_size}, prompt_lens={seq_lens}, gen={args.gen_tokens}"
        scenarios = [(token_sequences, args.gen_tokens, label)]
    else:
        # Default scenarios with short text prompts
        short_prompt = (
            "You are a helpful assistant that solves math problems step by step. "
            "Show your reasoning clearly and carefully before giving the final answer."
        )
        long_prompt = (
            "You are a highly skilled mathematician and teacher. Your goal is to "
            "solve complex mathematical problems with detailed step-by-step reasoning. "
            "When presented with a problem, first identify the type of problem and "
            "the relevant mathematical concepts. Then, break down the solution into "
            "clear logical steps. Show all intermediate calculations and explain "
            "each transformation. Finally, verify your answer by substituting back "
            "or using an alternative method. Be precise with notation and careful "
            "with arithmetic. If the problem has multiple valid approaches, mention "
            "the alternatives briefly. Always state your final answer clearly."
        )
        scenarios = [
            (short_prompt, 50, "Short prompt (25 tok) + 50 gen tokens"),
            (short_prompt, 200, "Short prompt (25 tok) + 200 gen tokens"),
            (long_prompt, 50, "Long prompt (~150 tok) + 50 gen tokens"),
            (long_prompt, 200, "Long prompt (~150 tok) + 200 gen tokens"),
        ]

    all_results = []
    for prompt, max_gen, label in scenarios:
        results = run_scenario(
            engine, trainer_model, device, tokenizer, prompt, max_gen, label
        )
        all_results.append((label, results))

    # ---- Final summary ----
    if dist.get_rank() == 0:
        logger.info("\n" + "=" * 70)
        logger.info("FINAL SUMMARY ACROSS ALL SCENARIOS")
        logger.info("=" * 70)
        for label, results in all_results:
            t1 = "PASS" if results.get("t1") else "FAIL"
            t2 = "PASS" if results.get("t2") else "FAIL"
            t3 = "PASS" if results.get("t3") else "FAIL"
            logger.info(f"  {label}")
            logger.info(f"    T1={t1}  T2={t2}  T3={t3}")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
