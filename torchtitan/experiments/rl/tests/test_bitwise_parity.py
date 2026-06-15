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

Run each backend in a separate torchrun invocation:
    torchrun --nproc_per_node=2 -m pytest \
        torchtitan/experiments/rl/tests/test_bitwise_parity.py::TestBitwiseParityVarlen -v

    torchrun --nproc_per_node=2 -m pytest \
        torchtitan/experiments/rl/tests/test_bitwise_parity.py::TestBitwiseParityFlex -v
"""

import gc
import logging
import os
import unittest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torch.nn.attention.flex_attention import and_masks
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config import CommConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.utils import (
    is_in_batch_invariant_mode,
    set_batch_invariance,
)
from torchtitan.experiments.rl.actors.trainer import compute_logprobs
from torchtitan.experiments.rl.config_registry import (
    rl_grpo_qwen3_0_6b_flex_batch_invariant,
    rl_grpo_qwen3_0_6b_varlen_batch_invariant,
    rl_grpo_qwen3_moe_debug_varlen_batch_invariant,
)
from torchtitan.experiments.rl.models.vllm_registry import (
    registry_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
    VLLM_MODEL_NAME,
)
from torchtitan.experiments.rl.trainer import RLTrainer
from torchtitan.models.common.attention import (
    create_attention_mask,
    FlexAttention,
    get_causal_mask_mod,
    get_document_mask_mod,
    VarlenMetadata,
)
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
        world_size=dist.get_world_size(),
    )

    dist_utils.set_determinism(
        parallel_dims,
        device,
        config.trainer.debug,
        distinct_seed_mesh_dims=["pp"],
    )

    trainer_config = config.trainer

    # Mirror PolicyTrainer._build_model: fill sharding configs (and any other
    # parallelism-driven config mutations) on the model config BEFORE build,
    # so each Module is constructed with its ShardingConfig / LocalMapConfig.
    # Without this the trainer side would run un-parallelized while the vLLM
    # generator runs fully TP-parallelized, breaking trainer-vs-vLLM parity.
    model_spec.model.update_from_config(config=trainer_config)

    with torch.device("meta"):
        with utils.set_default_dtype(TORCH_DTYPE_MAP[trainer_config.training.dtype]):
            model = model_spec.model.build()

    model = model_spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=trainer_config.training,
        parallelism=parallelism,
        compile_config=config.compile,
        ac_config=trainer_config.ac_config,
        dump_folder=trainer_config.dump_folder,
    )
    model.to_empty(device=device_type)
    with torch.no_grad():
        model.init_weights(buffer_device=None)

    # Load HF checkpoint if available
    if model_spec.state_dict_adapter is not None and hf_assets_path:
        index_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
        single_path = os.path.join(hf_assets_path, "model.safetensors")
        if os.path.exists(index_path) or os.path.exists(single_path):
            sd_adapter = model_spec.state_dict_adapter(model_spec.model, hf_assets_path)
            storage_reader = sd_adapter.get_hf_storage_reader(hf_assets_path)
            hf_state_dict = sd_adapter.to_hf(model.state_dict())
            dcp.load(hf_state_dict, storage_reader=storage_reader)
            tt_state_dict = sd_adapter.from_hf(hf_state_dict)
            set_model_state_dict(
                model=model,
                model_state_dict=tt_state_dict,
                options=StateDictOptions(strict=False),
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

    inner_attn = config.model_spec.model.layers[0].attention.inner_attention
    use_flex = isinstance(inner_attn, FlexAttention.Config)

    if use_flex:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
        backend_enum = AttentionBackendEnum.FLEX_ATTENTION
    else:
        os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"
        if gen_config.debug.batch_invariant:
            set_batch_invariance(True)
        backend_enum = AttentionBackendEnum.CUSTOM

    _set_generator_determinism(gen_config.debug)

    enable_ep = gen_config.parallelism.expert_parallel_degree > 1
    engine_kwargs = dict(
        model=config.hf_assets_path,
        trust_remote_code=True,
        # Build the model config from torchtitan's ModelSpec via the custom
        # parser registered by registry_to_vllm, instead of reading config.json.
        config_format=TORCHTITAN_CONFIG_FORMAT,
        dtype=gen_config.model_dtype,
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        # Generator-only convention: this config field is TorchTitan FSDP
        # degree for trainers, but here it is intentionally mapped to vLLM's
        # pure data-parallel degree. The vLLM wrapper skips FSDP/DDP.
        data_parallel_size=gen_config.parallelism.data_parallel_shard_degree,
        enable_expert_parallel=enable_ep,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=not gen_config.cudagraph.enable,
        hf_overrides={"architectures": [VLLM_MODEL_NAME]},
        attention_config=AttentionConfig(backend=backend_enum),
        disable_log_stats=True,
    )

    from torchtitan.tools.utils import has_cuda_capability

    if not has_cuda_capability(9, 0) and not use_flex:
        engine_kwargs["block_size"] = 256  # set blocksize to be 256 to align with FA2

    engine_kwargs["max_model_len"] = config.model_spec.model.max_seq_len
    max_num_seqs = config.num_groups_per_rollout_batch * config.group_size
    engine_kwargs["max_num_seqs"] = max_num_seqs
    vllm_compilation_config = gen_config.cudagraph.get_vllm_compilation_config(
        max_num_seqs=max_num_seqs,
    )
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


def _flex_prefill_logprobs(model, input_tensors, seq_lens, device):
    """Compute per-sequence logprobs using flex attention with packed sequences.

    Mirrors the trainer's flex attention path: pack documents into a single
    row, pad each document to block-aligned boundaries in batch-invariant
    mode, create a BlockMask via ``get_document_mask_mod`` +
    ``get_causal_mask_mod``, and extract per-document logprobs.
    """
    inner_attn = model.config.layers[0].attention.inner_attention
    assert isinstance(inner_attn, FlexAttention.Config)
    block_size = inner_attn.block_size

    batch_invariant = is_in_batch_invariant_mode()

    if batch_invariant:
        padded_seq_lens = [
            ((sl + block_size - 1) // block_size) * block_size for sl in seq_lens
        ]
    else:
        padded_seq_lens = list(seq_lens)

    # Build packed token_ids and positions (positions reset to 0 per doc)
    parts, pos_parts = [], []
    for tensor, sl, psl in zip(input_tensors, seq_lens, padded_seq_lens):
        padded = torch.zeros(psl, dtype=tensor.dtype, device=device)
        padded[:sl] = tensor
        parts.append(padded)
        pos_parts.append(torch.arange(psl, device=device))

    packed_ids = torch.cat(parts).unsqueeze(0)
    positions = torch.cat(pos_parts).unsqueeze(0)

    mask_mod = and_masks(get_causal_mask_mod(), get_document_mask_mod(positions))
    attention_masks = create_attention_mask(
        mask_mod,
        1,
        None,
        positions.shape[1],
        positions.shape[1],
        BLOCK_SIZE=block_size,
        separate_full_blocks=not batch_invariant,
    )

    logits = model(packed_ids, attention_masks=attention_masks, positions=positions)

    # Build pre-shifted labels matching the trainer convention:
    # labels[i] = packed_ids[i+1] for valid positions, IGNORE_INDEX otherwise.
    labels = torch.full_like(packed_ids, IGNORE_INDEX)
    offset = 0
    for sl, psl in zip(seq_lens, padded_seq_lens):
        labels[0, offset : offset + sl - 1] = packed_ids[0, offset + 1 : offset + sl]
        offset += psl

    logprobs = compute_logprobs(logits, labels)

    results = []
    offset = 0
    for sl, psl in zip(seq_lens, padded_seq_lens):
        results.append(logprobs[0, offset : offset + sl - 1])
        offset += psl
    return results


def _varlen_prefill_logprobs(model, input_tensors, seq_lens, device):
    """Compute per-sequence logprobs using varlen attention with padded batches."""
    max_len = max(seq_lens)
    padded = torch.zeros(len(input_tensors), max_len, dtype=torch.long, device=device)
    for i, t in enumerate(input_tensors):
        padded[i, : t.shape[0]] = t

    attention_masks = _build_padded_varlen_metadata(len(input_tensors), max_len, device)

    # Explicit positions avoid dynamic rope_cache[0:seqlen] slice in RoPE,
    # which can break torch.compile with symbolic shapes.
    positions = (
        torch.arange(max_len, device=device).unsqueeze(0).expand(len(input_tensors), -1)
    )

    logits = model(padded, attention_masks=attention_masks, positions=positions)

    # Build pre-shifted labels matching the trainer convention:
    # labels[i] = padded[i+1] for valid positions, IGNORE_INDEX otherwise.
    labels = torch.full_like(padded, IGNORE_INDEX)
    for i, t in enumerate(input_tensors):
        seq_len = t.shape[0]
        labels[i, : seq_len - 1] = t[1:seq_len]

    logprobs = compute_logprobs(logits, labels)

    results = []
    for i, t in enumerate(input_tensors):
        seq_len = t.shape[0]
        results.append(logprobs[i, : seq_len - 1])
    return results


def compute_trainer_prefill_logprobs(model, token_ids, device, attn_backend="varlen"):
    """Compute next-token logprobs using the trainer model.

    Args:
        token_ids: A single sequence (list[int]) or a batch of sequences
            (list[list[int]]). Batched sequences are padded to max length
            with appropriate attention metadata.
        attn_backend: 'varlen' or 'flex'.

    Returns:
        Single sequence: float32 tensor with len = len(token_ids) - 1.
        Batch: list of float32 tensors, one per sequence.
    """
    batched = isinstance(token_ids[0], list)
    seqs = token_ids if batched else [token_ids]

    input_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in seqs]
    seq_lens = [t.shape[0] for t in input_tensors]

    if attn_backend == "flex":
        results = _flex_prefill_logprobs(model, input_tensors, seq_lens, device)
    else:
        results = _varlen_prefill_logprobs(model, input_tensors, seq_lens, device)

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
class BitwiseParityTestBase(unittest.TestCase):
    """Base class for bitwise parity tests. Subclass and set config_fn / attn_backend."""

    __test__ = False

    BATCH_SIZE = 5
    PROMPT_LENGTH = 150
    MAX_GEN_TOKENS = 50

    config_fn = staticmethod(rl_grpo_qwen3_0_6b_varlen_batch_invariant)
    attn_backend: str = "varlen"
    min_world_size: int = 1
    hf_assets_env_var: str = "HF_ASSETS_PATH"

    # Shared across all tests in the class (built once in setUpClass)
    model: torch.nn.Module
    engine: LLMEngine
    prompt_ids: list[list[int]]

    @classmethod
    def setUpClass(cls):
        world_size = (
            dist.get_world_size()
            if dist.is_initialized()
            else int(os.environ.get("WORLD_SIZE", "1"))
        )
        if world_size < cls.min_world_size:
            raise unittest.SkipTest(
                f"requires at least {cls.min_world_size} GPUs, got {world_size}"
            )

        config = cls.config_fn()
        hf_path = os.environ.get(cls.hf_assets_env_var)
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

        registry_to_vllm(
            config.model_spec,
            parallelism=config.generator.parallelism,
            compile_config=config.compile,
            checkpoint_config=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                initial_load_path=config.hf_assets_path,
            ),
        )

        # Test runs trainer and generator in the same process, so limit
        # GPU memory for vLLM to leave room for the trainer model.
        config.generator.gpu_memory_limit = 0.5

        cls.model, cls.device = build_trainer_model(config)
        cls.engine = build_inference_engine(config)

        tokenizer = cls.engine.get_tokenizer()
        cls.prompt_ids = _make_prompt_tokens(
            cls.BATCH_SIZE, cls.PROMPT_LENGTH, tokenizer
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine"):
            cls.engine.engine_core.shutdown()
            del cls.engine
        if hasattr(cls, "model"):
            del cls.model
        gc.collect()
        torch.cuda.empty_cache()

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
        num_diff = (a != b).sum().item()
        print(
            f"  {name}: max_delta={max_delta:.2e}, "
            f"num_diff={num_diff}/{n}, "
            f"bitwise_equal={torch.equal(a, b)}"
        )
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
                model,
                self.prompt_ids[:mid],
                self.device,
                attn_backend=self.attn_backend,
            )
            lps_full = compute_trainer_prefill_logprobs(
                model, self.prompt_ids, self.device, attn_backend=self.attn_backend
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
                model, self.prompt_ids, self.device, attn_backend=self.attn_backend
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


class TestBitwiseParityVarlen(BitwiseParityTestBase):
    """Bitwise parity tests using varlen attention."""

    __test__ = True
    config_fn = staticmethod(rl_grpo_qwen3_0_6b_varlen_batch_invariant)
    attn_backend = "varlen"


class TestBitwiseParityFlex(BitwiseParityTestBase):
    """Bitwise parity tests using flex attention."""

    __test__ = True
    config_fn = staticmethod(rl_grpo_qwen3_0_6b_flex_batch_invariant)
    attn_backend = "flex"


class TestBitwiseParityMoEEP(BitwiseParityTestBase):
    """Test bitwise parity between trainer and vLLM generator with MoE EP.

    On 4 GPUs: trainer uses dp_shard=2, TP=2, EP=4; the generator maps
    dp_shard=2 to vLLM data parallelism with TP=2, EP=4.

    Uses the bundled debug MoE assets by default. Override with
    MOE_HF_ASSETS_PATH if needed.
    """

    __test__ = True

    BATCH_SIZE = 3
    PROMPT_LENGTH = 100
    MAX_GEN_TOKENS = 30

    config_fn = staticmethod(rl_grpo_qwen3_moe_debug_varlen_batch_invariant)
    attn_backend = "varlen"
    min_world_size = 4
    hf_assets_env_var = "MOE_HF_ASSETS_PATH"


if __name__ == "__main__":
    unittest.main()
