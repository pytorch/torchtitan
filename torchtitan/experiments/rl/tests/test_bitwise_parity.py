#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test bitwise parity between vLLM generator and TorchTitan trainer.

Three tests:

  1. test_batch_invariance:
      Trainer prefill(bsz=m) == Trainer prefill(bsz=n, m!=n).
      Guards that model kernels are batch-invariant.
  2. test_trainer_vs_vllm_prefill:
      Trainer prefill == vLLM prefill (prompt-only).
      Ensures trainer and generator forward have bitwise parity.
  3. test_vllm_decode_vs_prefill:
      vLLM decode == vLLM 2nd-pass prefill (generated positions).
      Ensures prefill vs decode (KV-cache) parity.

By transitivity of test 2 and test 3: trainer == vLLM decode.

Run:
    torchrun --nproc_per_node=1 -m pytest \
        torchtitan/experiments/rl/tests/test_bitwise_parity.py -v

    torchrun --nproc_per_node=2 -m pytest \
        torchtitan/experiments/rl/tests/test_bitwise_parity.py -v
"""

import logging
import os
import unittest

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
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.config_registry import rl_grpo_qwen3_0_6b_batch_invariant
from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.plugin import (
    register_model_to_vllm_model_registry,
    VLLM_MODEL_NAME,
)
from torchtitan.experiments.rl.simple_grpo_sum_digits import RLTrainer
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.tools import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model and Engine setup
# ---------------------------------------------------------------------------

# TODO: directly testing against PolicyTrainer with debug model to avoid OOM
def build_trainer_model(
    config: RLTrainer.Config,
) -> tuple[torch.nn.Module, torch.device]:
    """Build, parallelize, and load weights for the trainer model.

    Mirrors PolicyTrainer._build_model() without the Monarch actor framework.
    """
    model_spec = config.model_spec
    hf_assets_path = config.hf_assets_path

    device_type = utils.device_type
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    utils.device_module.set_device(device)

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

    dist_utils.set_determinism(
        parallel_dims,
        device,
        config.trainer.debug,
        distinct_seed_mesh_dims=["pp"],
    )

    # Build on meta device, parallelize, then materialize
    with torch.device("meta"):
        with utils.set_default_dtype(TORCH_DTYPE_MAP[config.trainer.training.dtype]):
            model = model_spec.model.build()

    model = parallelize_qwen3(
        model, parallel_dims=parallel_dims, parallelism=parallelism
    )
    model.to_empty(device=device_type)
    with torch.no_grad():
        model.init_weights(buffer_device=None)

    # Load HF checkpoint
    if model_spec.state_dict_adapter is not None:
        sd_adapter = model_spec.state_dict_adapter(model_spec.model, hf_assets_path)
        storage_reader = sd_adapter.get_hf_storage_reader(hf_assets_path)
        hf_state_dict = sd_adapter.to_hf(model.state_dict())
        dcp.load(hf_state_dict, storage_reader=storage_reader)
        tt_state_dict = sd_adapter.from_hf(hf_state_dict)
        set_model_state_dict(
            model=model,
            model_state_dict=tt_state_dict,
            options=StateDictOptions(strict=True),
        )

    model.eval()
    return model, device


# TODO: directly testing against VLLMGenerator with debug model to avoid OOM
def _set_generator_determinism(debug) -> None:
    """Apply deterministic flags for the generator side.

    Mirrors VLLMGenerator._set_determinism() — the generator doesn't use
    torchtitan's ParallelDims, so we apply the flags directly.
    """
    if debug.deterministic:
        torch.use_deterministic_algorithms(
            True, warn_only=debug.deterministic_warn_only
        )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if debug.seed is not None:
        torch.manual_seed(debug.seed)


def build_inference_engine(config: RLTrainer.Config) -> LLMEngine:
    """Create a vLLM LLMEngine with torchtitan model from the RL config."""
    gen_config = config.generator

    os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"
    _set_generator_determinism(gen_config.debug)

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
        disable_log_stats=True,
    )

    from torchtitan.tools.utils import has_cuda_capability

    if not has_cuda_capability(9, 0):
        engine_kwargs["block_size"] = 256  # set blocksize to be 256 to align with FA2

    vllm_compilation_config = gen_config.compile.get_vllm_compilation_config()
    if vllm_compilation_config is not None:
        engine_kwargs["compilation_config"] = vllm_compilation_config
    if gen_config.debug.seed is not None:
        engine_kwargs["seed"] = gen_config.debug.seed

    return LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))


# ---------------------------------------------------------------------------
# Logprob helpers
# ---------------------------------------------------------------------------


def _build_padded_varlen_metadata(batch_size, max_len, device):
    """Build VarlenMetadata for a padded (batch_size, max_len) tensor.

    VarlenAttention reshapes (batch_size, max_len) -> (batch_size * max_len,)
    so each row boundary is at multiples of max_len. Causal masking prevents
    padding tokens from affecting valid positions.
    """
    cu_seqs = torch.arange(
        0, (batch_size + 1) * max_len, max_len, dtype=torch.int32, device=device
    )
    return VarlenMetadata(
        cu_seq_q=cu_seqs, cu_seq_k=cu_seqs, max_q=max_len, max_k=max_len
    )


def compute_trainer_prefill_logprobs(model, token_ids, device):
    """Compute next-token logprobs using the trainer model.

    Args:
        token_ids: A single sequence (list[int]) or a batch of sequences
            (list[list[int]]). Batched sequences are padded to max length
            with varlen metadata for independent attention per row.

    Returns:
        Single sequence: float32 tensor with len = len(token_ids) - 1.
        Batch: list of float32 tensors, one per sequence.
    """
    batched = isinstance(token_ids[0], list)
    seqs = token_ids if batched else [token_ids]

    input_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in seqs]
    max_len = max(t.shape[0] for t in input_tensors)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long, device=device)
    for i, t in enumerate(input_tensors):
        padded[i, : t.shape[0]] = t

    attention_masks = _build_padded_varlen_metadata(len(seqs), max_len, device)

    # Explicit positions avoid dynamic rope_cache[0:seqlen] slice in RoPE,
    # which can break torch.compile with symbolic shapes.
    positions = torch.arange(max_len, device=device).unsqueeze(0).expand(len(seqs), -1)

    logits = model(padded, attention_masks=attention_masks, positions=positions)
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)

    results = []
    for i, t in enumerate(input_tensors):
        seq_len = t.shape[0]
        seq_lps = log_probs[i, : seq_len - 1]
        seq_lps = seq_lps.gather(1, t[1:seq_len].unsqueeze(-1)).squeeze(-1)
        results.append(seq_lps)

    return results if batched else results[0]


def _extract_logprobs_from_prompt(output, token_ids, start_pos: int = 0):
    """Extract per-token logprobs from vLLM prompt_logprobs starting at start_pos."""
    logprobs = []
    for i, lp_dict in enumerate(output.prompt_logprobs):
        if lp_dict is None or i < start_pos:
            continue
        tok = token_ids[i]
        if tok in lp_dict:
            logprobs.append(lp_dict[tok].logprob)
        else:
            logprobs.append(max(lp_dict.values(), key=lambda x: x.logprob).logprob)
    return logprobs


# ---------------------------------------------------------------------------
# vLLM operations
# ---------------------------------------------------------------------------

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


def vllm_prefill(engine, all_prompt_ids):
    """Run vLLM prefill and extract prompt logprobs for each sequence."""
    outputs = _run_engine(engine, "prefill", all_prompt_ids, _PREFILL_PARAMS)
    return [
        _extract_logprobs_from_prompt(out, ids)
        for out, ids in zip(outputs, all_prompt_ids, strict=True)
    ]


def vllm_generate(engine, all_prompt_ids, max_tokens):
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


def vllm_2nd_pass_prefill(engine, all_prompt_ids, all_gen_ids):
    """Re-prefill [prompt + generated] and extract logprobs for generated positions."""
    all_combined = [
        list(p) + list(g) for p, g in zip(all_prompt_ids, all_gen_ids, strict=True)
    ]
    outputs = _run_engine(engine, "2nd_prefill", all_combined, _PREFILL_PARAMS)
    return [
        _extract_logprobs_from_prompt(out, combined, start_pos=len(p))
        for out, combined, p in zip(outputs, all_combined, all_prompt_ids, strict=True)
    ]


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

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
# Tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(
    dist.is_initialized() or "RANK" in os.environ,
    "requires torchrun launcher",
)
class TestBitwiseParity(unittest.TestCase):
    """Test bitwise parity between trainer and vLLM generator."""

    BATCH_SIZE = 5
    PROMPT_LENGTH = 150
    MAX_GEN_TOKENS = 50

    # Shared across all tests in the class (built once in setUpClass)
    model: torch.nn.Module
    engine: LLMEngine
    prompt_ids: list[list[int]]

    @classmethod
    def setUpClass(cls):
        config = rl_grpo_qwen3_0_6b_batch_invariant()
        hf_path = os.environ.get("HF_ASSETS_PATH")
        if hf_path:
            config.hf_assets_path = hf_path

        from torchtitan.tools.utils import has_cuda_capability

        if has_cuda_capability(9, 0):
            from torch.nn.attention import (
                activate_flash_attention_impl,
                current_flash_attention_impl,
            )

            if current_flash_attention_impl() != "FA3":
                activate_flash_attention_impl("FA3")

        # Enable batch-invariant mode BEFORE init_distributed
        set_batch_invariance(config.trainer.debug.batch_invariant)

        if not dist.is_initialized():
            dist_utils.init_distributed(CommConfig())

        config.model_spec.parallelize_fn = parallelize_qwen3
        register_model_to_vllm_model_registry(config.model_spec)

        # Test runs trainer and generator in the same process, so limit
        # GPU memory for vLLM to leave room for the trainer model.
        config.generator.gpu_memory_limit = 0.5

        cls.model, cls.device = build_trainer_model(config)
        cls.engine = build_inference_engine(config)

        tokenizer = cls.engine.get_tokenizer()
        cls.prompt_ids = _make_prompt_tokens(
            cls.BATCH_SIZE, cls.PROMPT_LENGTH, tokenizer
        )

    def _assert_logprobs_equal(self, name, a, b, label_a="A", label_b="B"):
        """Assert two logprob sequences are bitwise identical."""
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

        max_delta = (a - b).abs().max().item() if n > 0 else 0.0
        self.assertTrue(
            torch.equal(a, b),
            f"{name}: NOT bitwise identical (max_delta={max_delta:.2e})\n"
            f"  {label_a}[:5]: {a[:5].tolist()}\n"
            f"  {label_b}[:5]: {b[:5].tolist()}",
        )

    def test_batch_invariance(self):
        """Trainer prefill(bsz=m) == Trainer prefill(bsz=n) for shared sequences.

        Guards that model kernels are batch-invariant: the same sequence must
        produce bit-identical logits regardless of what other sequences are
        in the batch.
        """
        model = self.model
        n = len(self.prompt_ids)
        mid = max(1, n // 2)

        with torch.no_grad():
            lps_partial = compute_trainer_prefill_logprobs(
                model, self.prompt_ids[:mid], self.device
            )
            lps_full = compute_trainer_prefill_logprobs(
                model, self.prompt_ids, self.device
            )

        if dist.get_rank() == 0:
            for i in range(mid):
                partial_lp = lps_partial[i] if mid > 1 else lps_partial
                self._assert_logprobs_equal(
                    f"seq {i}: prefill(bsz={mid}) vs prefill(bsz={n})",
                    partial_lp,
                    lps_full[i],
                    f"bsz={mid}",
                    f"bsz={n}",
                )

    def test_trainer_vs_vllm_prefill(self):
        """Trainer prefill == vLLM prefill (prompt-only).

        Ensures the trainer model forward and generator model forward produce
        bitwise identical logprobs.
        """
        model = self.model
        engine = self.engine

        with torch.no_grad():
            trainer_lps = compute_trainer_prefill_logprobs(
                model, self.prompt_ids, self.device
            )

        vllm_lps = vllm_prefill(engine, self.prompt_ids)

        if dist.get_rank() == 0:
            for i in range(len(self.prompt_ids)):
                self._assert_logprobs_equal(
                    f"seq {i}: Trainer prefill vs vLLM prefill",
                    trainer_lps[i],
                    vllm_lps[i],
                    "Trainer",
                    "vLLM",
                )

    def test_vllm_decode_vs_prefill(self):
        """vLLM decode == vLLM 2nd-pass prefill (generated positions).

        Ensures prefill-stage attention and decode-stage KV-cache attention
        produce bitwise identical logprobs.
        """
        engine = self.engine

        gen_ids, decode_lps = vllm_generate(
            engine, self.prompt_ids, self.MAX_GEN_TOKENS
        )
        prefill_2nd_lps = vllm_2nd_pass_prefill(engine, self.prompt_ids, gen_ids)

        if dist.get_rank() == 0:
            for i in range(len(self.prompt_ids)):
                self._assert_logprobs_equal(
                    f"seq {i}: vLLM decode vs vLLM 2nd-pass prefill",
                    decode_lps[i],
                    prefill_2nd_lps[i],
                    "Decode",
                    "2ndPrefill",
                )


if __name__ == "__main__":
    unittest.main()
