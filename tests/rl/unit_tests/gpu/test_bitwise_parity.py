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
        tests/rl/unit_tests/gpu/test_bitwise_parity.py::TestBitwiseParityVarlen -v

    torchrun --nproc_per_node=2 -m pytest \
        tests/rl/unit_tests/gpu/test_bitwise_parity.py::TestBitwiseParityFlex -v
"""

import copy
import dataclasses
import gc
import io
import logging
import os
import shutil
import tempfile
import unittest

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from PIL import Image
from renderers import PlaceholderRange
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data.multimodal.mrope import get_mrope_positions
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.loss import compute_logprobs, IGNORE_INDEX
from torchtitan.components.renderer import hash_image_file
from torchtitan.config import (
    apply_overrides,
    CommConfig,
    PrecisionConfig,
    TORCH_DTYPE_MAP,
)
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext, utils as dist_utils
from torchtitan.distributed.batch_invariant import (
    is_in_batch_invariant_mode,
    set_batch_invariance,
)
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.qwen3_8.flavors import (
    QWEN3_8_SPECIAL_TOKENS,
    QWEN3_8_VISION_PROCESSOR_KWARGS,
)
from torchtitan.observability.logging import init_logger
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.vllm.engine import add_extra_vllm_engine_args
from torchtitan.rl.generator.vllm.multimodal_wrapper import image_prompt_to_engine_input
from torchtitan.rl.generator.vllm.registry import (
    register_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
    TORCHTITAN_WORKER_CLS,
    VLLM_MODEL_NAME,
)
from torchtitan.rl.rollout.types import Prompt, PromptImage
from torchtitan.tools import utils
from torchtitan_recipes.rl.alphabet_sort import (
    rl_grpo_qwen3_0_6b_flex_batch_invariant,
    rl_grpo_qwen3_0_6b_varlen_batch_invariant,
    rl_grpo_qwen3_5_9b_varlen_batch_invariant,
)
from torchtitan_recipes.tests.rl.alphabet_sort import (
    rl_grpo_gpt_oss_debug_varlen_batch_invariant,
    rl_grpo_kimi_k3_debug_varlen_batch_invariant,
    rl_grpo_qwen3_5_debug_varlen_batch_invariant,
    rl_grpo_qwen3_moe_debug_varlen_batch_invariant,
)
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)

# Each class runs under its own two-GPU torchrun launch.
pytestmark = pytest.mark.multi_gpu


# ---------------------------------------------------------------------------
# Model and Engine setup
# ---------------------------------------------------------------------------


# TODO: directly testing against Trainer with debug model to avoid OOM
def build_trainer_model(
    config: Controller.Config,
) -> tuple[torch.nn.Module, torch.device, ParallelismContext]:
    """Build, parallelize, and load weights for the trainer model.

    Mirrors Trainer._build_model() without the Monarch actor framework.
    """
    model_config = copy.deepcopy(config.trainer.model)
    hf_assets_path = config.hf_assets_path

    device = utils.get_local_device()
    utils.device_module.set_device(device)

    parallelism = config.trainer.parallelism
    parallelism_context = ParallelismContext(
        dp_shard=parallelism.data_parallel_shard_degree,
        dp_replicate=parallelism.data_parallel_replicate_degree,
        cp=parallelism.context_parallel_degree,
        tp=parallelism.tensor_parallel_degree,
        pp=parallelism.pipeline_parallel_degree,
        ep=parallelism.expert_parallel_degree,
        world_size=dist.get_world_size(),
        enable_sequence_parallel=parallelism.enable_sequence_parallel,
    )
    dist_utils.set_determinism(
        parallelism_context,
        device,
        config.trainer.debug,
        distinct_seed_mesh_axes=["pp"],
    )

    trainer_config = config.trainer

    with (
        parallelism_context.activate_spmd(),
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[config.precision.dtype]),
    ):
        model = model_config.build()

    model = model.parallelize(
        parallelism_context=parallelism_context,
        precision=config.precision,
        parallelism=parallelism,
        local_compile=model_config.local_compile,
        activation_remat=trainer_config.activation_remat,
        dump_folder=config.dump_folder,
    )
    with parallelism_context.activate_spmd():
        model.to_empty(device=device)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

    # Load HF checkpoint if available
    adapter_cls = type(model).state_dict_adapter_cls
    if adapter_cls is not None and hf_assets_path:
        index_path = os.path.join(hf_assets_path, "model.safetensors.index.json")
        single_path = os.path.join(hf_assets_path, "model.safetensors")
        if os.path.exists(index_path) or os.path.exists(single_path):
            sd_adapter = adapter_cls(model_config, hf_assets_path)
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
    return model, device, parallelism_context


# TODO: directly testing against Generator with debug model to avoid OOM
def _set_generator_determinism(debug) -> None:
    """Apply deterministic flags for the generator side.

    Mirrors Generator._set_determinism() — the generator doesn't use
    torchtitan's ParallelismContext, so we apply the flags directly.
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


def build_inference_engine(config: Controller.Config) -> LLMEngine:
    """Create a vLLM LLMEngine with torchtitan model from the RL config."""
    gen_config = config.generator

    attention_backend = gen_config.model.first_full_attention.inner_attention
    use_flex = isinstance(attention_backend, FlexInnerAttention.Config)

    # Mirror the production Generator so the test exercises the same
    # batch-invariant path (v2 runner is required for the logprob-kernel patch).
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    if use_flex:
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
        backend_enum = AttentionBackendEnum.FLEX_ATTENTION
    else:
        os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"
        if gen_config.debug.batch_invariant:
            set_batch_invariance(True)
            # The v2 logprob Triton kernel bypasses the aten overrides. Apply the
            # same generator-side patch the production Generator does.
            from torchtitan.rl.generator.vllm.batch_invariance import (
                force_trainer_logprobs_for_batch_invariance,
            )

            force_trainer_logprobs_for_batch_invariance()
        backend_enum = AttentionBackendEnum.CUSTOM

    _set_generator_determinism(gen_config.debug)

    enable_ep = gen_config.parallelism.expert_parallel_degree > 1
    engine_kwargs = dict(
        model=config.hf_assets_path,
        trust_remote_code=True,
        # Build the model config from torchtitan's model config via the custom
        # parser registered by register_to_vllm, instead of reading config.json.
        config_format=TORCHTITAN_CONFIG_FORMAT,
        dtype=config.precision.param_dtype,
        tensor_parallel_size=gen_config.parallelism.tensor_parallel_degree,
        data_parallel_size=gen_config.parallelism.data_parallel_degree,
        enable_expert_parallel=enable_ep,
        worker_cls=TORCHTITAN_WORKER_CLS,
        distributed_executor_backend="external_launcher",
        gpu_memory_utilization=gen_config.gpu_memory_limit,
        enforce_eager=gen_config.engine.cuda_graph.mode == "NONE",
        hf_overrides={"architectures": [VLLM_MODEL_NAME]},
        attention_config=AttentionConfig(backend=backend_enum),
        disable_log_stats=True,
    )

    from torchtitan.tools.utils import has_cuda_capability

    if not has_cuda_capability(9, 0) and not use_flex:
        engine_kwargs["block_size"] = 256  # set blocksize to be 256 to align with FA2

    engine_kwargs["max_model_len"] = gen_config.model.max_context_length
    max_num_seqs = config.max_num_seqs_per_generator_dp_rank
    engine_kwargs["max_num_seqs"] = max_num_seqs
    if gen_config.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = gen_config.max_num_batched_tokens
    expert_sequence_parallel_size = gen_config.parallelism.expert_sequence_parallel_size
    vllm_compilation_config = gen_config.engine.cuda_graph.get_vllm_compilation_config(
        max_num_seqs=max_num_seqs,
        max_model_len=gen_config.model.max_context_length,
        max_num_batched_tokens=gen_config.max_num_batched_tokens,
        expert_sequence_parallel_size=expert_sequence_parallel_size,
        enable_sequence_parallel=gen_config.parallelism.enable_sequence_parallel,
    )
    if vllm_compilation_config is not None:
        engine_kwargs["compilation_config"] = vllm_compilation_config
    if gen_config.debug.seed is not None:
        engine_kwargs["seed"] = gen_config.debug.seed
    if config.vision_processor is not None:
        # As `VLLMEngine._build_engine_args`: image prompts bypass vLLM's processor.
        engine_kwargs["mm_processor_cache_gb"] = 0
    add_extra_vllm_engine_args(engine_kwargs, gen_config.engine.extra_engine_args)

    return LLMEngine.from_engine_args(EngineArgs(**engine_kwargs))


def _sync_trainer_weights_to_vllm(trainer_model, engine) -> None:
    """Copy the trainer model's weights into the vLLM model in-process."""

    wrapper = engine.model_executor.driver_worker.get_model()
    trainer_sd = trainer_model.state_dict()
    # As a model state dict load: fill the pinned CPU copy, then load from it.
    vllm_sd = wrapper.model_state_dict_loader.prefetched_model_state_dict

    trainer_buffers = {name for name, _ in trainer_model.named_buffers()}
    trainer_preserve_dtype_params = {
        name for name, _ in trainer_model.named_preserve_dtype_params()
    }
    missing = []
    for name, vparam in vllm_sd.items():
        tparam = trainer_sd.get(name)
        if tparam is None:
            missing.append(name)
            continue
        full = tparam.full_tensor() if isinstance(tparam, DTensor) else tparam
        # Weight sync over RDMA copies bytes, so the receive dtype is what the trainer sends:
        # parameters in the generator dtype (precision.param_dtype), buffers and
        # preserve_dtype_params in their own.
        if name in trainer_buffers or name in trainer_preserve_dtype_params:
            sent_dtype = full.dtype
        else:
            sent_dtype = engine.model_config.dtype
        assert vparam.dtype == sent_dtype, (name, vparam.dtype, sent_dtype)
        with torch.inference_mode():
            if isinstance(vparam, DTensor):
                vparam.copy_(
                    distribute_tensor(full, vparam.device_mesh, vparam.placements)
                )
            else:
                vparam.copy_(full)

    def fetch(state_dict: dict[str, torch.Tensor]) -> list[str]:
        with torch.no_grad():
            for key, tensor in state_dict.items():
                tensor.copy_(vllm_sd[key])
        return list(state_dict)

    wrapper.model_state_dict_loader.load(fetch)

    if dist.get_rank() == 0 and missing:
        logger.warning("vLLM params not present in trainer state_dict: %s", missing)

    # Sinks were injected during build with uninitialized values (no HF load);
    # re-inject now that the real sink weights are in place.
    if hasattr(wrapper, "_inject_attention_sinks"):
        wrapper._inject_attention_sinks()


# ---------------------------------------------------------------------------
# Logprob helpers
# ---------------------------------------------------------------------------


def _flex_prefill_logprobs(model, input_tensors, seq_lens, device):
    """Compute per-sequence logprobs using flex attention with packed sequences.

    Mirrors the trainer's flex attention path: pack documents into a single
    row, pad each document to block-aligned boundaries in batch-invariant
    mode, build backend-specific attention metadata, and extract per-document
    logprobs.
    """
    inner_attn = model.config.layers[0].attention.inner_attention
    assert isinstance(inner_attn, FlexInnerAttention.Config)
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

    packed_ids = torch.cat(parts)
    positions = torch.cat(pos_parts)

    attention_metadata = model._get_attention_metadata(positions)

    logits = model(
        packed_ids, attention_metadata=attention_metadata, positions=positions
    )

    # Build pre-shifted labels matching the trainer convention:
    # labels[i] = packed_ids[i+1] for valid positions, IGNORE_INDEX otherwise.
    labels = torch.full_like(packed_ids, IGNORE_INDEX)
    offset = 0
    for sl, psl in zip(seq_lens, padded_seq_lens):
        labels[offset : offset + sl - 1] = packed_ids[offset + 1 : offset + sl]
        offset += psl

    logprobs = compute_logprobs(
        logits,
        labels,
        vocab_parallel_group=spmd_mesh_group("tp"),
        global_vocab_size=model.config.lm_head.out_features,
    )

    results = []
    offset = 0
    for sl, psl in zip(seq_lens, padded_seq_lens):
        results.append(logprobs[offset : offset + sl - 1])
        offset += psl
    return results


def _varlen_prefill_logprobs(model, input_tensors, seq_lens, device):
    """Compute per-sequence logprobs using packed variable-length segments."""
    packed_ids = torch.cat(input_tensors)
    positions = torch.cat(
        [torch.arange(seq_len, device=device) for seq_len in seq_lens]
    )

    # Explicit positions avoid dynamic rope_cache[0:seqlen] slice in RoPE,
    # which can break torch.compile with symbolic shapes.
    # Hybrid models may require different metadata for each attention type.
    attention_metadata = model._get_attention_metadata(positions)

    logits = model(
        packed_ids, attention_metadata=attention_metadata, positions=positions
    )

    # Build pre-shifted labels matching the trainer convention:
    # labels[i] = packed_ids[i+1] within each segment, IGNORE_INDEX otherwise.
    labels = torch.full_like(packed_ids, IGNORE_INDEX)
    offset = 0
    for t in input_tensors:
        seq_len = t.shape[0]
        labels[offset : offset + seq_len - 1] = t[1:seq_len]
        offset += seq_len

    logprobs = compute_logprobs(
        logits,
        labels,
        vocab_parallel_group=spmd_mesh_group("tp"),
        global_vocab_size=model.config.lm_head.out_features,
    )

    results = []
    offset = 0
    for t in input_tensors:
        seq_len = t.shape[0]
        results.append(logprobs[offset : offset + seq_len - 1])
        offset += seq_len
    return results


def compute_trainer_prefill_logprobs(model, token_ids, device, attn_backend="varlen"):
    """Compute next-token logprobs using the trainer model.

    Args:
        token_ids: A single sequence (list[int]) or a batch of sequences
            (list[list[int]]). Batched sequences are packed with appropriate
            attention metadata.
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


def _engine_input(prompt: Prompt, vision_processor: VisionProcessor | None):
    """vLLM's input for ``prompt``: its token ids, and for an image prompt its images'
    pixel values, preprocessed as `VLLMEngine._preprocess_images` does."""
    if not prompt.images:
        return {"prompt_token_ids": prompt.token_ids}
    assert vision_processor is not None
    pixel_values = [
        vision_processor.patchify(vision_processor.resize_and_normalize(image.data)[0])[
            0
        ]
        for image in prompt.images
    ]
    return image_prompt_to_engine_input(prompt, pixel_values)


def _run_engine(engine, request_prefix, prompts, sampling_params, vision_processor):
    """Submit requests to vLLM, run to completion, return outputs sorted by ID."""
    for i, prompt in enumerate(prompts):
        engine.add_request(
            f"{request_prefix}_{i}",
            _engine_input(prompt, vision_processor),
            sampling_params,
        )
    outputs = []
    while engine.has_unfinished_requests():
        outputs.extend(engine.step())
    outputs.sort(key=lambda o: o.request_id)
    assert len(outputs) == len(prompts)
    return outputs


def vllm_prefill(engine, prompts, vision_processor=None):
    """Run vLLM prefill and extract prompt logprobs for each sequence."""
    outputs = _run_engine(engine, "prefill", prompts, _PREFILL_PARAMS, vision_processor)
    return [
        _extract_logprobs_from_prompt(out, prompt.token_ids)
        for out, prompt in zip(outputs, prompts, strict=True)
    ]


def vllm_generate(engine, prompts, max_tokens, vision_processor=None):
    """Generate tokens and return (generated_ids, decode_logprobs) per sequence."""
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        logprobs=1,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    outputs = _run_engine(engine, "generate", prompts, params, vision_processor)
    all_ids, all_lps = [], []
    for out in outputs:
        sample = out.outputs[0]
        all_ids.append(list(sample.token_ids))
        all_lps.append([list(d.values())[0].logprob for d in sample.logprobs])
    return all_ids, all_lps


def vllm_2nd_pass_prefill(engine, prompts, all_gen_ids, vision_processor=None):
    """Re-prefill [prompt + generated] and extract logprobs for generated positions."""
    all_combined = [
        dataclasses.replace(prompt, token_ids=prompt.token_ids + list(gen_ids))
        for prompt, gen_ids in zip(prompts, all_gen_ids, strict=True)
    ]
    outputs = _run_engine(
        engine, "2nd_prefill", all_combined, _PREFILL_PARAMS, vision_processor
    )
    return [
        _extract_logprobs_from_prompt(
            out, combined.token_ids, start_pos=len(prompt.token_ids)
        )
        for out, combined, prompt in zip(outputs, all_combined, prompts, strict=True)
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


def make_prompts(batch_size, prompt_length, tokenizer) -> list[Prompt]:
    """Create text prompts of the given prompt_length.

    For batch_size > 1, varies lengths across the batch (first seq at max,
    last seq ~40% of max) to test batch invariance with mixed lengths.
    """
    prompts = []
    for idx in range(batch_size):
        frac = 1.0 - (idx * 0.6 / max(batch_size - 1, 1))
        target_len = max(16, int(prompt_length * frac))

        text = ""
        while True:
            text += _FILLER_TEXT + " "
            tokens = tokenizer.encode(text)
            if len(tokens) >= target_len:
                break
        prompts.append(Prompt(token_ids=tokens[:target_len]))

    return prompts


def _synthetic_image(
    height: int, width: int, seed: int, vision_processor: VisionProcessor
) -> PromptImage:
    """A PNG of smooth random colors, ``height`` x ``width`` pixels."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height // 32, width // 32, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).resize((width, height), Image.BILINEAR).save(buffer, "PNG")
    data = buffer.getvalue()
    return PromptImage(
        hash_value=hash_image_file(data),
        source_path=f"synthetic_{seed}.png",
        grid_thw=vision_processor.grid_thw(height, width),
        data=data,
    )


def make_image_prompt(
    parts: list[int | PromptImage],
    *,
    seed: int,
    tokenizer,
    special_tokens: dict[str, str],
    vision_processor: VisionProcessor,
) -> Prompt:
    """A prompt of ``parts`` in order: an int is that many random text tokens, an image
    is ``<|vision_start|>``, its placeholder run and ``<|vision_end|>``.

    ``tokenizer`` is the engine's Hugging Face tokenizer and ``special_tokens`` the
    model's token strings (e.g. ``QWEN3_8_SPECIAL_TOKENS``). Text tokens exclude the
    tokenizer's added tokens.
    """
    rng = np.random.default_rng(seed)
    added_vocab = tokenizer.get_added_vocab()
    image_id = added_vocab[special_tokens["image_token"]]
    vision_start_id = added_vocab[special_tokens["vision_start_token"]]
    vision_end_id = added_vocab[special_tokens["vision_end_token"]]
    text_ids = sorted(set(range(len(tokenizer))) - set(added_vocab.values()))
    token_ids, images, placeholders = [], [], []
    for part in parts:
        if isinstance(part, int):
            token_ids += rng.choice(text_ids, size=part).tolist()
            continue
        num_tokens = vision_processor.num_placeholder_tokens(part.grid_thw)
        token_ids.append(vision_start_id)
        placeholders.append(PlaceholderRange(offset=len(token_ids), length=num_tokens))
        token_ids += [image_id] * num_tokens + [vision_end_id]
        images.append(part)
    return Prompt(
        token_ids=token_ids,
        images=tuple(images),
        image_placeholders=tuple(placeholders),
    )


def compute_trainer_image_prefill_logprobs(
    model,
    prompts: list[Prompt],
    device: torch.device,
    *,
    vision_processor: VisionProcessor,
    parallelism_context: ParallelismContext,
    parallelism: ParallelismConfig,
) -> list[torch.Tensor]:
    """Next-token logprobs of each prompt from the trainer model, the prompts packed into
    one microbatch the way the RL batcher packs image samples.

    Each distinct image is encoded once and every placeholder token takes its row of the
    image embeddings (``image_vision_bank_indices_T``); M-RoPE positions restart per
    prompt. The model then takes the trainer's path: `preprocess_inputs`, then `forward`.
    """
    merge = vision_processor.config.spatial_merge_size
    bank_rows, mrope_positions, distinct_images = [], [], {}
    num_bank_rows = 0
    for prompt in prompts:
        rows = np.full(len(prompt.token_ids), -1, dtype=np.int64)
        for image, placeholder in zip(
            prompt.images, prompt.image_placeholders, strict=True
        ):
            if image.hash_value not in distinct_images:
                distinct_images[image.hash_value] = (image, num_bank_rows)
                num_bank_rows += placeholder.length
            bank_start = distinct_images[image.hash_value][1]
            rows[
                placeholder.offset : placeholder.offset + placeholder.length
            ] = np.arange(bank_start, bank_start + placeholder.length)
        bank_rows.append(rows)
        mrope_positions.append(
            get_mrope_positions(
                len(prompt.token_ids),
                [
                    (placeholder.offset, (t, h // merge, w // merge))
                    for (t, h, w), placeholder in zip(
                        (image.grid_thw for image in prompt.images),
                        prompt.image_placeholders,
                        strict=True,
                    )
                ],
            )
        )
    input_tensors = [
        torch.tensor(prompt.token_ids, dtype=torch.long, device=device)
        for prompt in prompts
    ]
    packed_ids = torch.cat(input_tensors)
    labels = torch.full_like(packed_ids, IGNORE_INDEX)
    offset = 0
    for t in input_tensors:
        labels[offset : offset + len(t) - 1] = t[1:]
        offset += len(t)
    pixel_values, grid_thw = vision_processor.patchify_all(
        [
            vision_processor.resize_and_normalize(image.data)[0]
            for image, _ in distinct_images.values()
        ]
    )
    input_dict = {
        "input": packed_ids,
        "labels": labels,
        "positions": torch.cat(
            [torch.arange(len(t), device=device) for t in input_tensors]
        ),
        "mrope_positions": torch.cat(mrope_positions).to(device),
        "pixel_values": pixel_values.to(device),
        "grid_thw": grid_thw.to(device),
        "image_vision_bank_indices_T": torch.from_numpy(np.concatenate(bank_rows)).to(
            device
        ),
    }
    inputs, labels, model_kwargs, _ = model.preprocess_inputs(
        input_dict,
        parallelism_context=parallelism_context,
        parallelism=parallelism,
    )
    logits = model(inputs, **model_kwargs)
    logprobs = compute_logprobs(
        logits,
        labels,
        vocab_parallel_group=spmd_mesh_group("tp"),
        global_vocab_size=model.config.lm_head.out_features,
    )
    results = []
    offset = 0
    for t in input_tensors:
        results.append(logprobs[offset : offset + len(t) - 1])
        offset += len(t)
    return results


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
    # Root dir for run artifacts (NCCL flight-recorder comm_traces). CI sets it
    # to $RUNNER_TEMP/artifacts-to-be-uploaded so traces are uploaded; when unset
    # (local runs) a temp dir is used instead.
    dump_folder_env_var: str = "RL_TEST_DUMP_FOLDER"
    # When True, vLLM skips its own checkpoint load and the trainer model's
    # weights (including attention sinks) are copied into the engine in-process.
    sync_weights_from_trainer: bool = False

    # Shared across all tests in the class (built once in setUpClass)
    model: torch.nn.Module
    engine: LLMEngine
    config: Controller.Config
    prompts: list[Prompt]
    # Built from ``config.vision_processor`` for image prompts; None for text models.
    vision_processor: VisionProcessor | None = None

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

        from torchtitan.tools.utils import get_cuda_flash_attention_impl

        flash_attention_impl = get_cuda_flash_attention_impl()
        if flash_attention_impl is not None:
            from torch.nn.attention import (
                activate_flash_attention_impl,
                current_flash_attention_impl,
            )

            if current_flash_attention_impl() != flash_attention_impl:
                activate_flash_attention_impl(flash_attention_impl)

        # Enable batch-invariant mode BEFORE init_distributed
        set_batch_invariance(config.trainer.debug.batch_invariant)

        if not dist.is_initialized():
            # Writable base_folder for comm_traces
            dump_root = os.environ.get(cls.dump_folder_env_var)
            if dump_root:
                base_folder = os.path.join(dump_root, cls.__name__)
                os.makedirs(base_folder, exist_ok=True)
            else:
                base_folder = tempfile.mkdtemp(prefix="rl_bitwise_")
                cls.addClassCleanup(shutil.rmtree, base_folder, ignore_errors=True)
            dist_utils.init_distributed(CommConfig(), base_folder=base_folder)

        if cls.sync_weights_from_trainer:
            generator_checkpointer = None
        else:
            generator_checkpointer = CheckpointManager.Config(
                initial_load_in_hf=True,
                initial_load_path=config.hf_assets_path,
            )

        if config.override.imports:
            apply_overrides(config.override, config)
        register_to_vllm(
            config.generator.model,
            parallelism=config.generator.parallelism,
            precision=config.precision,
            checkpointer_config=generator_checkpointer,
            model_state_dict_load_mode="cpu_prefetch",
            vision_processor=config.vision_processor,
        )
        cls.config = config
        if config.vision_processor is not None:
            cls.vision_processor = config.vision_processor.build()

        # Test runs trainer and generator in the same process, so limit
        # GPU memory for vLLM to leave room for the trainer model.
        config.generator.gpu_memory_limit = 0.5

        cls.model, cls.device, cls.parallelism_context = build_trainer_model(config)

        # Workaround: Clear the DTensor sharding-propagation caches here since
        # this test puts both the trainer and the generator in the same process
        # and builds equal but separate meshes for them. Without this workaround,
        # the generator's FSDP params pick up the trainer's mesh objects and
        # fail fully_shard's mesh identity check. This error happens only in
        # this specific test setup, no production impact.
        #
        # TODO: Revisit DTensor sharding-propagation cache semantics on pytorch side
        torch._C._clear_DTensor_sharding_propagator_cache()
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_clear()

        cls.engine = build_inference_engine(config)
        if cls.sync_weights_from_trainer:
            _sync_trainer_weights_to_vllm(cls.model, cls.engine)

        cls.prompts = cls.make_prompts()

    @classmethod
    def make_prompts(cls) -> list[Prompt]:
        """The prompts every test of the class runs."""
        return make_prompts(
            cls.BATCH_SIZE, cls.PROMPT_LENGTH, cls.engine.get_tokenizer()
        )

    def trainer_prefill_logprobs(self, prompts: list[Prompt]) -> list[torch.Tensor]:
        """Next-token logprobs of each prompt from the trainer model, the prompts packed
        into one microbatch."""
        with type(self).parallelism_context.activate_spmd(), torch.no_grad():
            return compute_trainer_prefill_logprobs(
                self.model,
                [prompt.token_ids for prompt in prompts],
                self.device,
                attn_backend=self.attn_backend,
            )

    @classmethod
    def tearDownClass(cls):
        # Each class runs in its own torchrun process. Synchronize before
        # releasing local resources, then leave process-group cleanup to exit.
        # vLLM's internal shutdown destroys every registered process group,
        # including the trainer's groups, and deadlocks during global teardown.
        if dist.is_initialized():
            dist.barrier()
        if hasattr(cls, "engine"):
            renderer = getattr(cls.engine, "renderer", None)
            if renderer is not None:
                renderer.shutdown()
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
        self.assertEqual(
            len(a),
            len(b),
            f"{name}: length mismatch ({label_a}={len(a)}, {label_b}={len(b)})",
        )
        n = len(a)

        max_delta = (a - b).abs().max().item() if n > 0 else 0.0
        num_diff = (a != b).sum().item()
        first_diff = int((a != b).nonzero()[0]) if num_diff else None
        print(
            f"  {name}: max_delta={max_delta:.2e}, "
            f"num_diff={num_diff}/{n}, first_diff={first_diff}, "
            f"bitwise_equal={torch.equal(a, b)}"
        )
        # A subtest, so one sequence's mismatch does not hide the others'.
        with self.subTest(name):
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
        n = len(self.prompts)
        mid = max(1, n // 2)

        lps_partial = self.trainer_prefill_logprobs(self.prompts[:mid])
        lps_full = self.trainer_prefill_logprobs(self.prompts)

        if dist.get_rank() == 0:
            for i in range(mid):
                self._assert_logprobs_equal(
                    f"seq {i}: prefill(bsz={mid}) vs prefill(bsz={n})",
                    lps_partial[i],
                    lps_full[i],
                    f"bsz={mid}",
                    f"bsz={n}",
                )

    def test_trainer_vs_vllm_prefill(self):
        """Trainer prefill == vLLM prefill (prompt-only).

        Ensures the trainer model forward and generator model forward produce
        bitwise identical logprobs.
        """
        trainer_lps = self.trainer_prefill_logprobs(self.prompts)
        vllm_lps = vllm_prefill(self.engine, self.prompts, self.vision_processor)

        if dist.get_rank() == 0:
            for i in range(len(self.prompts)):
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
            engine, self.prompts, self.MAX_GEN_TOKENS, self.vision_processor
        )
        prefill_2nd_lps = vllm_2nd_pass_prefill(
            engine, self.prompts, gen_ids, self.vision_processor
        )

        if dist.get_rank() == 0:
            for i in range(len(self.prompts)):
                self._assert_logprobs_equal(
                    f"seq {i}: vLLM decode vs vLLM 2nd-pass prefill",
                    decode_lps[i],
                    prefill_2nd_lps[i],
                    "Decode",
                    "2ndPrefill",
                )

    def _check_vllm_prefill_and_decode_batch_invariance(self):
        """Check that vLLM prefill and decode do not depend on batch composition."""
        single_prompt = self.prompts[:1]

        single_prefill_lps = vllm_prefill(
            self.engine, single_prompt, self.vision_processor
        )
        batched_prefill_lps = vllm_prefill(
            self.engine, self.prompts, self.vision_processor
        )

        single_gen_ids, single_decode_lps = vllm_generate(
            self.engine, single_prompt, self.MAX_GEN_TOKENS, self.vision_processor
        )
        batched_gen_ids, batched_decode_lps = vllm_generate(
            self.engine, self.prompts, self.MAX_GEN_TOKENS, self.vision_processor
        )

        if dist.get_rank() == 0:
            self._assert_logprobs_equal(
                "seq 0: vLLM prefill(bsz=1) vs prefill(bsz=3)",
                single_prefill_lps[0],
                batched_prefill_lps[0],
                "bsz=1",
                "bsz=3",
            )
            self.assertEqual(
                single_gen_ids[0],
                batched_gen_ids[0],
                "seq 0: greedy decode token IDs differ by batch composition",
            )
            self._assert_logprobs_equal(
                "seq 0: vLLM decode(bsz=1) vs decode(bsz=3)",
                single_decode_lps[0],
                batched_decode_lps[0],
                "bsz=1",
                "bsz=3",
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


class TestBitwiseParityQwen35Varlen(BitwiseParityTestBase):
    """Test Qwen3.5 trainer/generator parity with head-sharded GDN at TP=2."""

    __test__ = True
    config_fn = staticmethod(rl_grpo_qwen3_5_9b_varlen_batch_invariant)
    attn_backend = "varlen"
    min_world_size = 2


def qwen3_5_debug_bitwise_config() -> Controller.Config:
    """Build a two-GPU random-weight Qwen3.5 parity configuration."""
    config = rl_grpo_qwen3_5_debug_varlen_batch_invariant()
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=1,
        ),
    )
    return config


class TestBitwiseParityQwen35DebugVarlen(BitwiseParityTestBase):
    """Qwen3.5 GDN parity with random weights and matched TP=2."""

    __test__ = True
    config_fn = staticmethod(qwen3_5_debug_bitwise_config)
    attn_backend = "varlen"
    min_world_size = 2
    sync_weights_from_trainer = True
    BATCH_SIZE = 3
    PROMPT_LENGTH = 64
    MAX_GEN_TOKENS = 16

    def test_vllm_prefill_and_decode_batch_invariance(self):
        """vLLM prefill and decode must not depend on batch composition."""
        self._check_vllm_prefill_and_decode_batch_invariance()

    def test_reload_after_generate(self):
        """Reload changed weights after generation has replayed the captured graphs."""
        engine = self.engine

        def reload() -> None:
            _sync_trainer_weights_to_vllm(self.model, engine)
            # As the generator does after a load: drop KV computed with the old weights.
            engine.reset_prefix_cache(reset_running_requests=True)

        vllm_generate(engine, self.prompts, self.MAX_GEN_TOKENS)
        before_lps = vllm_prefill(engine, self.prompts)
        params = list(self.model.parameters())
        originals = [p.detach().clone() for p in params]
        # Every rank runs every collective before rank 0 asserts, so a mismatch
        # fails instead of hanging the other rank.
        try:
            with torch.no_grad():
                for p in params:
                    p.mul_(0.5)
            reload()
            trainer_lps = self.trainer_prefill_logprobs(self.prompts)
            vllm_lps = vllm_prefill(engine, self.prompts)
            gen_ids, decode_lps = vllm_generate(
                engine, self.prompts, self.MAX_GEN_TOKENS
            )
            prefill_2nd_lps = vllm_2nd_pass_prefill(engine, self.prompts, gen_ids)
        finally:
            with torch.no_grad():
                for p, original in zip(params, originals):
                    p.copy_(original)
            reload()

        if dist.get_rank() == 0:
            self.assertNotEqual(before_lps[0], vllm_lps[0])
            for i in range(len(self.prompts)):
                self._assert_logprobs_equal(
                    f"seq {i}: Trainer prefill vs vLLM prefill after reload",
                    trainer_lps[i],
                    vllm_lps[i],
                    "Trainer",
                    "vLLM",
                )
                self._assert_logprobs_equal(
                    f"seq {i}: vLLM decode vs 2nd-pass prefill after reload",
                    decode_lps[i],
                    prefill_2nd_lps[i],
                    "Decode",
                    "2ndPrefill",
                )


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() >= (9, 0),
    "Batch-invariant Attention Gym KDA uses fixed-tile TMA kernels (SM90+)",
)
class KimiK3BitwiseParityTestBase(BitwiseParityTestBase):
    """Run Kimi K3 parity checks with the trainer in training mode."""

    __test__ = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model.train()


def kimi_k3_debug_bitwise_config() -> Controller.Config:
    """Build a one-GPU random-weight Kimi K3 parity configuration."""
    config = rl_grpo_kimi_k3_debug_varlen_batch_invariant()
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=1,
        ),
    )
    return config


class TestBitwiseParityKimiK3DebugVarlen(KimiK3BitwiseParityTestBase):
    """Kimi K3 KDA/MLA parity with random weights and matched TP=1."""

    __test__ = True
    config_fn = staticmethod(kimi_k3_debug_bitwise_config)
    attn_backend = "varlen"
    sync_weights_from_trainer = True
    BATCH_SIZE = 3
    PROMPT_LENGTH = 64
    MAX_GEN_TOKENS = 128

    def test_vllm_prefill_and_decode_batch_invariance(self):
        """vLLM prefill and decode must not depend on batch composition."""
        self._check_vllm_prefill_and_decode_batch_invariance()


def flash_next_mm_debug_bitwise_config() -> Controller.Config:
    """One-GPU random-weight Flash-Next with the vision encoder, batch-invariant.

    Trainer and generator are unsharded and route experts without Dist-MoE. 64-token
    prefill steps make vLLM split every image of `TestBitwiseParityFlashNextImages`
    across steps.
    """
    from torchtitan.models.qwen3_8_flash_next import (
        build_model_config as build_flash_next_model_config,
    )
    from torchtitan.models.qwen3_8_flash_next.attention import QSAIndexerLoss
    from torchtitan_recipes.rl.model_configs import build_rl_model_configs

    config = rl_grpo_qwen3_5_debug_varlen_batch_invariant(seq_len=1024)
    trainer_parallelism = ParallelismConfig()
    generator_parallelism = InferenceParallelismConfig()
    config.trainer.model, config.generator.model = build_rl_model_configs(
        build_flash_next_model_config,
        "debugmodel_mm",
        seq_len=1024,
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        # Without an indexer loss the indexer is frozen, and the generator, which
        # gets weights only from the trainer here, would never receive it.
        indexer_loss=QSAIndexerLoss.Config(coeff=1.0, reduce_mesh="loss"),
    )
    # Local compile regions do not support batch-invariant mode.
    for model_config in (config.trainer.model, config.generator.model):
        model_config.local_compile.regions = []
    config.trainer = dataclasses.replace(
        config.trainer, parallelism=trainer_parallelism
    )
    config.generator = dataclasses.replace(
        config.generator,
        parallelism=generator_parallelism,
        max_num_batched_tokens=64,
    )
    config.generator.engine.cuda_graph.mode = "FULL_DECODE_ONLY"
    # Flash-Next's cached PLE keeps each request's token history, which a prefix
    # cache hit would skip.
    config.generator.engine.extra_engine_args = {"enable_prefix_caching": False}
    config.precision = PrecisionConfig(dtype="bfloat16")
    config.vision_processor = VisionProcessor.Config(**QWEN3_8_VISION_PROCESSOR_KWARGS)
    return config


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() >= (10, 0),
    "Flash-Next QSA attention (Attention Gym gather_attn) needs SM100+",
)
class TestBitwiseParityFlashNextImages(BitwiseParityTestBase):
    """Flash-Next with image prompts: random weights, one GPU, matched TP=1.

    The prompts cover what M-RoPE changes in the generator's QSA indexer, whose
    compressed keys take the M-RoPE position of their 4-token block's first token:

        prompt 0  4 text | image A (70 tokens) | 21 text | image B (70) | 7 text
        prompt 1  3 text | image A | 40 text           (A again: vLLM's encoder cache)
        prompt 2  90 text                              (M-RoPE text positions)
        prompt 3  9 text | image C (120)               (ends in C's last block)

    Image A spans tokens 5..74 of prompt 0, so blocks 4..7 and 72..75 hold text and
    image tokens. With 64-token prefill steps vLLM splits every image across steps.
    Prompt 3's last block, 128..131, starts with an image token and completes at the
    first decode step, so its key's position comes from the raw-key ring.
    """

    __test__ = True
    config_fn = staticmethod(flash_next_mm_debug_bitwise_config)
    sync_weights_from_trainer = True
    MAX_GEN_TOKENS = 32

    @classmethod
    def make_prompts(cls) -> list[Prompt]:
        processor = cls.vision_processor
        assert processor is not None
        image_a, image_b, image_c = (
            _synthetic_image(height, width, seed, processor)
            for seed, (height, width) in enumerate([(64, 96), (224, 160), (384, 320)])
        )
        prompt_parts = [
            [4, image_a, 21, image_b, 7],
            [3, image_a, 40],
            [90],
            [9, image_c],
        ]
        return [
            make_image_prompt(
                parts,
                seed=seed,
                tokenizer=cls.engine.get_tokenizer(),
                special_tokens=QWEN3_8_SPECIAL_TOKENS,
                vision_processor=processor,
            )
            for seed, parts in enumerate(prompt_parts)
        ]

    def trainer_prefill_logprobs(self, prompts: list[Prompt]) -> list[torch.Tensor]:
        assert self.vision_processor is not None
        with type(self).parallelism_context.activate_spmd(), torch.no_grad():
            return compute_trainer_image_prefill_logprobs(
                self.model,
                prompts,
                self.device,
                vision_processor=self.vision_processor,
                parallelism_context=type(self).parallelism_context,
                parallelism=self.config.trainer.parallelism,
            )

    def test_prompt_layout(self):
        """The prompts have the layout the class docstring describes."""
        image_runs = [
            [(p.offset, p.length) for p in prompt.image_placeholders]
            for prompt in self.prompts
        ]
        self.assertEqual(image_runs, [[(5, 70), (98, 70)], [(4, 70)], [], [(10, 120)]])
        self.assertEqual(len(self.prompts[3].token_ids), 131)

    def test_vision_encoder_batch_invariance(self):
        """An image's embeddings do not depend on the images encoded with it.

        The trainer encodes a microbatch's distinct images together, vLLM the images of
        each scheduled step, so the two group them differently.
        """
        processor = self.vision_processor
        assert processor is not None
        images = list(
            {
                image.hash_value: image
                for prompt in self.prompts
                for image in prompt.images
            }.values()
        )
        pixels = [processor.resize_and_normalize(image.data)[0] for image in images]

        def encode(group: list[torch.Tensor]) -> list[torch.Tensor]:
            pixel_values, grid_thw = processor.patchify_all(group)
            embeds, num_tokens = self.model.get_vision_embeds(
                pixel_values.to(self.device), grid_thw=grid_thw.to(self.device)
            )
            return list(embeds.split(num_tokens.tolist()))

        with type(self).parallelism_context.activate_spmd(), torch.no_grad():
            together = encode(pixels)
            alone = [encode([image_pixels])[0] for image_pixels in pixels]

        if dist.get_rank() == 0:
            for i, (a, b) in enumerate(zip(alone, together, strict=True)):
                self._assert_logprobs_equal(
                    f"image {i}: encoded alone vs with all images",
                    a.flatten(),
                    b.flatten(),
                    "alone",
                    "together",
                )

    def test_vllm_prefill_and_decode_batch_invariance(self):
        """vLLM prefill and decode must not depend on batch composition."""
        self._check_vllm_prefill_and_decode_batch_invariance()


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


class TestBitwiseParityGptOssVarlen(BitwiseParityTestBase):
    """Bitwise parity for GPT-OSS varlen attention."""

    __test__ = True
    config_fn = staticmethod(rl_grpo_gpt_oss_debug_varlen_batch_invariant)
    attn_backend = "varlen"
    sync_weights_from_trainer = True


if __name__ == "__main__":
    init_logger()
    unittest.main()
