#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test bitwise parity between vLLM generator and TorchTitan trainer log-probs.

vLLM decode logprobs come from incremental KV-cache decoding (one token at a
time), while the trainer computes logprobs via a single full-sequence forward
pass.  These two paths use different attention implementations internally, so
directly comparing them is not meaningful.

Instead, we use a **2nd-pass prefill**: after generating tokens via decode, we
concatenate [prompt + generated] and run a *second* vLLM prefill over the full
sequence.  This prefill uses the same full-sequence attention as the trainer,
so its logprobs are directly comparable.

The test establishes parity via three checks:

  Test 1: Trainer prefill == vLLM prefill (prompt-only)
  Test 2: vLLM decode == vLLM 2nd-pass prefill (generated positions)
  Test 3: Trainer full-seq forward == vLLM 2nd-pass prefill

By transitivity: trainer == vLLM decode.

Run:
    torchrun --nproc_per_node=1 \
        torchtitan/experiments/rl/tests/test_bitwise_parity.py

    torchrun --nproc_per_node=1 \
        torchtitan/experiments/rl/tests/test_bitwise_parity.py \
        --batch-size 2 --prompt-length 4000 --gen-tokens 100
"""

import argparse
import logging
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from torchtitan.config import CommConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.rl.actors.utils import (
    build_varlen_metadata,
    compute_token_log_probs,
)
from torchtitan.experiments.rl.config_registry import rl_grpo_qwen3_0_6b_batch_invariant
from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.plugin import (
    register_model_to_vllm_model_registry,
    VLLM_MODEL_NAME,
)
from torchtitan.experiments.rl.simple_grpo_sum_digits import RLTrainer
from torchtitan.tools import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def build_inference_engine(config: RLTrainer.Config) -> LLMEngine:
    """Create a vLLM LLMEngine with torchtitan model from the RL config."""
    gen_config = config.generator

    if config.debug.batch_invariant:
        from torchtitan.experiments.rl.batch_invariant import enable_batch_invariant

        enable_batch_invariant()

    engine_kwargs = dict(
        model=config.hf_assets_path,
        trust_remote_code=True,
        dtype=gen_config.model_dtype,
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=gen_config.compile.is_eager,
        hf_overrides={"architectures": [VLLM_MODEL_NAME]},
        attention_config=AttentionConfig(
            backend=AttentionBackendEnum.CUSTOM,
        ),
    )
    vllm_compilation_config = gen_config.compile.get_vllm_compilation_config()
    if vllm_compilation_config is not None:
        engine_kwargs["compilation_config"] = vllm_compilation_config
    if gen_config.debug.seed is not None:
        engine_kwargs["seed"] = gen_config.debug.seed

    engine = LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))
    if dist.get_rank() == 0:
        logger.info("vLLM LLMEngine ready.")
    return engine


def build_trainer_model(
    config: RLTrainer.Config,
) -> tuple[torch.nn.Module, torch.device]:
    """Build, parallelize, and load weights for the trainer model.

    Mirrors PolicyTrainer._build_model() without the Monarch actor framework.
    """
    model_spec = config.model_spec
    hf_assets_path = config.hf_assets_path

    if config.debug.batch_invariant:
        from torchtitan.experiments.rl.batch_invariant import enable_batch_invariant

        enable_batch_invariant()
        model_spec.model.layer.attention.inner_attention.batch_invariant = True

    model_config = model_spec.model

    # Device setup
    device_module, device_type = utils.device_module, utils.device_type
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)

    parallelism = config.trainer.parallelism
    parallel_dims = ParallelDims(
        dp_shard=parallelism.data_parallel_shard_degree,
        dp_replicate=parallelism.data_parallel_replicate_degree,
        cp=parallelism.context_parallel_degree,
        tp=parallelism.tensor_parallel_degree,
        pp=parallelism.pipeline_parallel_degree,
        ep=parallelism.expert_parallel_degree,
        etp=parallelism.expert_tensor_parallel_degree,
        world_size=dist.get_world_size(),
    )

    # Build on meta device, parallelize, then materialize
    with torch.device("meta"):
        with utils.set_default_dtype(TORCH_DTYPE_MAP[config.trainer.training.dtype]):
            model = model_config.build()

    model = parallelize_qwen3(
        model, parallel_dims=parallel_dims, parallelism=parallelism
    )
    model.to_empty(device=device_type)
    with torch.no_grad():
        model.init_weights(buffer_device=None)

    # Load HF checkpoint
    if model_spec.state_dict_adapter is not None:
        sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)
        storage_reader = sd_adapter.get_hf_storage_reader(hf_assets_path)
        hf_state_dict = sd_adapter.to_hf(model.state_dict())
        dcp.load(hf_state_dict, storage_reader=storage_reader)
        tt_state_dict = sd_adapter.from_hf(hf_state_dict)
        set_model_state_dict(
            model=model,
            model_state_dict=tt_state_dict,
            options=StateDictOptions(strict=False),
        )
        if dist.get_rank() == 0:
            logger.info(f"Loaded HF weights ({len(tt_state_dict)} params)")

    model.eval()
    return model, device


# ---------------------------------------------------------------------------
# Logprob helpers
# ---------------------------------------------------------------------------


def _extract_logprobs_from_prompt(output, token_ids, start_pos: int = 0):
    """Extract per-token logprobs from vLLM prompt_logprobs starting at start_pos.

    Position 0 always has None (no logprob for first token).
    Returns a list of floats.
    """
    logprobs = []
    for i, lp_dict in enumerate(output.prompt_logprobs):
        if lp_dict is None or i < start_pos:
            continue
        tok = token_ids[i]
        if tok in lp_dict:
            logprobs.append(lp_dict[tok].logprob)
        else:
            logger.warning(f"Token {tok} not in logprobs at position {i}")
            logprobs.append(max(lp_dict.values(), key=lambda x: x.logprob).logprob)
    return logprobs


def compute_trainer_prefill_logprobs(model, token_ids, device):
    """Compute next-token logprobs over a token sequence using the trainer model.

    Returns a float32 tensor with len = len(token_ids) - 1.
    """
    input_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    seq_len = input_tensor.shape[1]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    attention_masks = build_varlen_metadata([(input_tensor[0], 0, 0)], device)

    logits = model(input_tensor, attention_masks=attention_masks, positions=positions)
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    return log_probs.gather(2, input_tensor[:, 1:].unsqueeze(-1)).squeeze(-1)[0]


def compare_logprobs(name, a, b, label_a="A", label_b="B") -> bool:
    """Compare two logprob sequences. Returns True if bitwise identical."""
    if isinstance(a, list):
        a = torch.tensor(a, dtype=torch.float32)
    else:
        a = a.detach().cpu().float()
    if isinstance(b, list):
        b = torch.tensor(b, dtype=torch.float32)
    else:
        b = b.detach().cpu().float()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    bitwise = torch.equal(a, b)
    max_delta = (a - b).abs().max().item() if n > 0 else 0.0

    logger.info(
        f"  {name}: {'PASS' if bitwise else 'FAIL'} (max_delta={max_delta:.2e})"
    )
    if not bitwise and n > 0:
        logger.info(f"    {label_a}[:5]: {a[:5].tolist()}")
        logger.info(f"    {label_b}[:5]: {b[:5].tolist()}")

    return bitwise


# ---------------------------------------------------------------------------
# vLLM operations
# ---------------------------------------------------------------------------

# Greedy sampling params shared by prefill and 2nd-pass prefill
_PREFILL_PARAMS = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1,
    logprobs=1,
    prompt_logprobs=1,
    output_kind=RequestOutputKind.FINAL_ONLY,
)


def _run_engine(engine, request_prefix, batched_ids, sampling_params):
    """Submit requests to vLLM, run to completion, return outputs sorted by ID."""
    for i, ids in enumerate(batched_ids):
        engine.add_request(
            f"{request_prefix}_{i}", {"prompt_token_ids": ids}, sampling_params
        )
    outputs = []
    while engine.has_unfinished_requests():
        outputs.extend(engine.step())
    outputs.sort(key=lambda o: o.request_id)
    assert len(outputs) == len(batched_ids)
    return outputs


def vllm_prefill(engine, all_prompt_ids: list[list[int]]) -> list[list[float]]:
    """Run vLLM prefill and extract prompt logprobs for each sequence."""
    outputs = _run_engine(engine, "prefill", all_prompt_ids, _PREFILL_PARAMS)
    return [
        _extract_logprobs_from_prompt(out, ids)
        for out, ids in zip(outputs, all_prompt_ids)
    ]


def vllm_generate(
    engine, all_prompt_ids: list[list[int]], max_tokens: int
) -> tuple[list[list[int]], list[list[float]]]:
    """Generate tokens and return (generated_ids, decode_logprobs) per sequence."""
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        logprobs=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    outputs = _run_engine(engine, "generate", all_prompt_ids, params)

    all_ids, all_lps = [], []
    for out in outputs:
        sample = out.outputs[0]
        all_ids.append(list(sample.token_ids))
        all_lps.append([list(d.values())[0].logprob for d in sample.logprobs])
    return all_ids, all_lps


def vllm_2nd_pass_prefill(
    engine,
    all_prompt_ids: list[list[int]],
    all_gen_ids: list[list[int]],
) -> list[list[float]]:
    """Re-prefill [prompt + generated] and extract logprobs for generated positions."""
    all_combined = [list(p) + list(g) for p, g in zip(all_prompt_ids, all_gen_ids)]
    outputs = _run_engine(engine, "2nd_prefill", all_combined, _PREFILL_PARAMS)
    return [
        _extract_logprobs_from_prompt(out, combined, start_pos=len(p))
        for out, combined, p in zip(outputs, all_combined, all_prompt_ids)
    ]


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------


def run_test_case(engine, trainer_model, device, all_prompt_ids, max_gen_tokens, label):
    """Run all 3 parity tests for a given prompt/gen-length configuration."""

    if dist.get_rank() == 0:
        lens = [len(ids) for ids in all_prompt_ids]
        logger.info(
            f"\n{'#' * 60}\n"
            f"SCENARIO: {label}\n"
            f"  Batch={len(all_prompt_ids)}, prompt_lens={lens}, gen={max_gen_tokens}\n"
            f"{'#' * 60}"
        )

    # vLLM operations (batched)
    prefill_lps = vllm_prefill(engine, all_prompt_ids)
    gen_ids, decode_lps = vllm_generate(engine, all_prompt_ids, max_gen_tokens)
    prefill_2nd_lps = vllm_2nd_pass_prefill(engine, all_prompt_ids, gen_ids)

    # Compare per-sequence
    results = {"t1": True, "t2": True, "t3": True}
    for i in range(len(all_prompt_ids)):
        with torch.no_grad():
            trainer_pf = compute_trainer_prefill_logprobs(
                trainer_model, all_prompt_ids[i], device
            )
            trainer_full = compute_token_log_probs(
                trainer_model, all_prompt_ids[i], gen_ids[i], device
            )

        if dist.get_rank() == 0:
            tag = f" [seq {i}]" if len(all_prompt_ids) > 1 else ""
            results["t1"] &= compare_logprobs(
                f"T1{tag}: Trainer prefill vs vLLM prefill",
                trainer_pf,
                prefill_lps[i],
                "Trainer",
                "vLLM",
            )
            results["t2"] &= compare_logprobs(
                f"T2{tag}: vLLM decode vs vLLM 2nd-pass prefill",
                decode_lps[i],
                prefill_2nd_lps[i],
                "Decode",
                "2ndPrefill",
            )
            results["t3"] &= compare_logprobs(
                f"T3{tag}: Trainer full vs vLLM 2nd-pass prefill",
                trainer_full,
                prefill_2nd_lps[i],
                "Trainer",
                "2ndPrefill",
            )

    return results


_FILLER_TEXT = (
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


def _make_prompt_tokens(batch_size, prompt_length, tokenizer):
    """Create token ID sequences of the given prompt_length.

    For batch_size > 1, varies lengths across the batch (first seq at max,
    last seq ~40% of max) to test batch invariance with mixed lengths.
    """
    all_sequences = []
    for idx in range(batch_size):
        frac = 1.0 - (idx * 0.6 / max(batch_size - 1, 1))
        target_len = max(16, int(prompt_length * frac))

        text = ""
        while True:
            text += _FILLER_TEXT + " "
            tokens = tokenizer.encode(text)
            if len(tokens) >= target_len:
                break
        all_sequences.append(tokens[:target_len])

    return all_sequences


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Test bitwise parity")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=0,
        help="Token length for custom scenario (0 = use defaults)",
    )
    parser.add_argument("--gen-tokens", type=int, default=50)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--gen-compile-backend",
        type=str,
        default="none",
        choices=["none", "eager", "inductor"],
        help="torch.compile backend for the vLLM generator",
    )
    parser.add_argument(
        "--generator-cudagraph-mode",
        type=str,
        default="none",
        choices=["none", "piecewise", "full", "full_and_piecewise"],
        help="CUDA graph capture mode for the vLLM generator",
    )
    parser.add_argument(
        "--hf-assets-path",
        type=str,
        default="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        help="Path to HF model checkpoint directory",
    )
    args, _ = parser.parse_known_args()

    config = rl_grpo_qwen3_0_6b_batch_invariant()
    # CLI overrides for test flexibility
    config.hf_assets_path = args.hf_assets_path
    config.trainer.parallelism.tensor_parallel_degree = args.tp
    config.generator.parallelism.tensor_parallel_degree = args.tp
    config.generator.compile.backend = args.gen_compile_backend
    config.generator.compile.cudagraph_mode = args.generator_cudagraph_mode
    # Test runs generator and trainer in the same process, so limit GPU memory
    config.generator.gpu_memory_limit = 0.5
    # Greedy decoding for deterministic generation in parity tests
    config.generator.num_samples_per_prompt = 1
    config.generator.sampling.temperature = 0.0
    config.generator.sampling.top_p = 1.0
    config.generator.sampling.max_tokens = 50
    config.model_spec.parallelize_fn = parallelize_qwen3

    from torchtitan.tools.utils import has_cuda_capability

    # Hopper (SM 9.0) uses FA3; Blackwell (SM 10.0+) and older use FA2.
    if has_cuda_capability(9, 0) and not has_cuda_capability(10, 0):
        from torch.nn.attention import (
            activate_flash_attention_impl,
            current_flash_attention_impl,
        )

        if current_flash_attention_impl() != "FA3":
            activate_flash_attention_impl("FA3")

    # Register torchtitan model to vllm Engine
    register_model_to_vllm_model_registry(config.model_spec)
    dist_utils.init_distributed(CommConfig())

    engine = build_inference_engine(config)
    tokenizer = engine.get_tokenizer()
    trainer_model, device = build_trainer_model(config)

    # Build test_cases: vary prompt length and generation length
    prompt_lengths = [args.prompt_length] if args.prompt_length > 0 else [25, 150]
    gen_lengths = [args.gen_tokens] if args.prompt_length > 0 else [50, 200]

    test_cases = []
    for pl in prompt_lengths:
        seqs = _make_prompt_tokens(args.batch_size, pl, tokenizer)
        for gl in gen_lengths:
            lens = [len(s) for s in seqs]
            test_cases.append((seqs, gl, f"prompt_lens={lens}, gen={gl}"))

    all_results = []
    for prompt_ids, max_gen, label in test_cases:
        results = run_test_case(
            engine, trainer_model, device, prompt_ids, max_gen, label
        )
        all_results.append((label, results))

    # Summary
    if dist.get_rank() == 0:
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        for label, r in all_results:
            t1 = "PASS" if r["t1"] else "FAIL"
            t2 = "PASS" if r["t2"] else "FAIL"
            t3 = "PASS" if r["t3"] else "FAIL"
            logger.info(f"  {label}: T1={t1} T2={t2} T3={t3}")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
