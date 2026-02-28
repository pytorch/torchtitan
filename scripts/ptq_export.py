# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torchtitan-native PTQ calibration and NVFP4 export.

Leverages torchtitan's distributed infrastructure (TP, FSDP, AC, FlexAttention)
for memory-efficient calibration at long sequence lengths, then exports packed
NVFP4 weights to HuggingFace safetensors format.

Usage:
    NGPU=4 CONFIG_FILE=path/to/ptq.toml OUTPUT_DIR=./outputs/nvfp4 bash run_ptq.sh
"""

import argparse
import contextlib
import dataclasses
import json
import os
import re
import shutil
import time
import warnings
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols import ModelProtocol
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger

# NVFP4 quantization constants
FP4_E2M1_MAX = 6.0  # Max representable value in E2M1 format
FP8_E4M3_MAX = 448.0  # Max representable value in E4M3 format


# ---------------------------------------------------------------------------
# Phase 1: Setup — distributed init, model build, checkpoint load
# ---------------------------------------------------------------------------


def setup_model(job_config: JobConfig):
    """Set up distributed model and load checkpoint.

    Returns:
        Tuple of (model_parts, parallel_dims, device, train_spec,
                  model_args, tokenizer, dataloader, sd_adapter)
    """
    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)

    # Init distributed
    world_size = dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=False,
        base_folder=job_config.job.dump_folder,
    )

    parallelism_config = job_config.parallelism
    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        etp=parallelism_config.expert_tensor_parallel_degree,
        world_size=world_size,
    )

    job_config.maybe_log()

    if parallel_dims.dp_enabled:
        batch_mesh = parallel_dims.get_mesh("batch")
        batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
    else:
        batch_degree, batch_rank = 1, 0

    dist_utils.set_determinism(
        parallel_dims, device, job_config.debug, distinct_seed_mesh_dims=["pp"]
    )
    train_spec = train_spec_module.get_train_spec(job_config.model.name)

    # Build tokenizer and dataloader (needed for calibration)
    tokenizer = (
        train_spec.build_tokenizer_fn(job_config)
        if train_spec.build_tokenizer_fn is not None
        else None
    )

    dataloader = train_spec.build_dataloader_fn(
        dp_world_size=batch_degree,
        dp_rank=batch_rank,
        tokenizer=tokenizer,
        job_config=job_config,
        seq_len_divisor=parallel_dims.seq_len_divisor,
    )

    # Build model on meta device
    model_args = train_spec.model_args[job_config.model.flavor]
    model_args.update_from_config(job_config)

    logger.info(
        f"Building {job_config.model.name} {job_config.model.flavor} "
        f"with {json.dumps(dataclasses.asdict(model_args), indent=2, ensure_ascii=False)}"
    )
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
    ):
        model = train_spec.model_cls(model_args)

    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    # Apply parallelism (TP, FSDP, AC) — no PP for PTQ
    assert not parallel_dims.pp_enabled, (
        "Pipeline parallelism not supported for PTQ export"
    )
    model = train_spec.parallelize_fn(model, parallel_dims, job_config)
    model.to_empty(device=device_type)
    with torch.no_grad():
        cast(ModelProtocol, model).init_weights(buffer_device=None)
    model.eval()

    model_parts = [model]

    # Load checkpoint via DCP
    checkpoint_path = job_config.checkpoint.initial_load_path
    if not checkpoint_path:
        raise ValueError("checkpoint.initial_load_path must be set for PTQ export")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    begin = time.monotonic()

    sd_adapter = (
        train_spec.state_dict_adapter(model_args, job_config.model.hf_assets_path)
        if train_spec.state_dict_adapter
        else None
    )

    if job_config.checkpoint.initial_load_in_hf:
        # Load from HF safetensors
        assert sd_adapter is not None, "state_dict_adapter required for HF loading"
        reader = sd_adapter.get_hf_storage_reader(
            checkpoint_path,
            from_quantized=job_config.checkpoint.initial_load_in_hf_quantized,
        )
        model_wrapper = ModelWrapper(model_parts)
        state_dict = model_wrapper._get_state_dict()
        dcp.load(state_dict, storage_reader=reader)
        model_wrapper.load_state_dict(state_dict)
    else:
        # Load from DCP checkpoint
        model_wrapper = ModelWrapper(model_parts)
        state_dict = model_wrapper._get_state_dict()
        dcp.load(state_dict, checkpoint_id=checkpoint_path)
        model_wrapper.load_state_dict(state_dict)

    logger.info(f"Checkpoint loaded in {time.monotonic() - begin:.2f}s")

    return (
        model_parts,
        parallel_dims,
        device,
        train_spec,
        model_args,
        tokenizer,
        dataloader,
        sd_adapter,
    )


# ---------------------------------------------------------------------------
# Phase 2: Calibrate — insert quantizers and run forward passes
# ---------------------------------------------------------------------------


def calibrate(
    model_parts: list[torch.nn.Module],
    parallel_dims: ParallelDims,
    device: torch.device,
    job_config: JobConfig,
    model_args,
    tokenizer,
    dataloader,
):
    """Apply ModelOpt quantization and run calibration forward passes.

    Inserts TensorQuantizer modules and collects amax statistics.
    """
    import modelopt.torch.quantization as mtq
    from modelopt.torch.quantization.nn.modules.quant_module import QuantModule
    from modelopt.torch.utils.distributed import ParallelState

    from torchtitan.models.gpt_oss.model.quant_moe import (
        TORCHTITAN_QAT_CONFIGS,
        register_gpt_oss_quant_module,
    )

    qat_config = job_config.qat

    # Register GptOssGroupedExperts for quantization
    register_gpt_oss_quant_module()

    # Resolve quantization config
    quant_cfg = TORCHTITAN_QAT_CONFIGS.get(qat_config.config)
    if quant_cfg is None:
        quant_cfg = getattr(mtq, qat_config.config, None)
    if quant_cfg is None or not isinstance(quant_cfg, dict):
        available = list(TORCHTITAN_QAT_CONFIGS.keys())
        raise ValueError(
            f"Unknown quantization config: '{qat_config.config}'. "
            f"Available: {available}"
        )

    calib_steps = qat_config.calib_steps
    logger.info(
        f"PTQ: Applying {qat_config.config} with {calib_steps} calibration steps"
    )

    # Build calibration forward loop
    calib_iterator = iter(dataloader)

    # We need post_dataloading_process logic for attention masks etc.
    # Replicate the essential parts here
    attn_type = getattr(model_args, "attn_type", "sdpa")

    def forward_loop(model):
        for i in range(calib_steps):
            try:
                input_dict, _labels = next(calib_iterator)
            except StopIteration:
                logger.warning(
                    f"Dataloader exhausted after {i} calibration steps "
                    f"(requested {calib_steps}). Using available data."
                )
                break
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(device)

            inputs = input_dict["input"]
            extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
            extra_kwargs: dict[str, Any] = {}

            if attn_type in ("flex", "varlen"):
                assert tokenizer is not None
                model_ref = cast(ModelProtocol, model)
                extra_kwargs["attention_masks"] = model_ref.get_attention_masks(
                    input_batch=inputs,
                    tokenizer=tokenizer,
                    extra_inputs=extra_inputs,
                )

            with torch.no_grad():
                pred = model(inputs, **extra_inputs, **extra_kwargs)
                del pred
            logger.info(f"PTQ calibration step {i + 1}/{calib_steps}")

    # Configure TP process group for ModelOpt quantized modules.
    # -1 is ModelOpt's sentinel for "no TP group" (single-device).
    tp_mesh = parallel_dims.get_optional_mesh("tp")
    tp_group = tp_mesh.get_group() if tp_mesh is not None else -1
    tp_parallel_state = ParallelState(
        data_parallel_group=None,
        tensor_parallel_group=tp_group,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*no parallel_state is set.*")
        for model in model_parts:
            mtq.quantize(model, quant_cfg, forward_loop)

    for model in model_parts:
        for module in model.modules():
            if isinstance(module, QuantModule):
                module.parallel_state = tp_parallel_state

    if dist.get_rank() == 0:
        for model in model_parts:
            mtq.print_quant_summary(model)

    logger.info("PTQ: Calibration complete")


# ---------------------------------------------------------------------------
# Eval — pre/post quantization loss comparison
# ---------------------------------------------------------------------------


def collect_eval_batches(
    dataloader,
    num_batches: int = 4,
) -> list[tuple[dict[str, torch.Tensor], torch.Tensor]]:
    """Collect eval batches from the dataloader, stored on CPU.

    Called before calibration so the same data can evaluate both the
    original and quantized model for loss comparison.
    """
    batches = []
    data_iter = iter(dataloader)
    for i in range(num_batches):
        try:
            input_dict, labels = next(data_iter)
            batches.append((input_dict, labels))
        except StopIteration:
            logger.warning(
                f"Dataloader exhausted after {i} eval batches (requested {num_batches})"
            )
            break
    logger.info(f"Collected {len(batches)} eval batches for loss comparison")
    return batches


def eval_model_loss(
    model_parts: list[torch.nn.Module],
    eval_batches: list[tuple[dict[str, torch.Tensor], torch.Tensor]],
    parallel_dims: ParallelDims,
    model_args,
    tokenizer,
    device: torch.device,
) -> float:
    """Compute average cross-entropy loss over eval batches.

    Used for pre/post quantization loss comparison to measure degradation
    from fake-quantization. Returns average loss per non-masked token.
    """
    from torchtitan.components.loss import IGNORE_INDEX

    if not eval_batches:
        return float("nan")

    model = model_parts[0]
    model.eval()

    attn_type = getattr(model_args, "attn_type", "sdpa")

    total_loss = 0.0
    total_tokens = 0

    # loss_parallel patches F.cross_entropy to handle TP-sharded logits
    # (Shard(-1) on vocab dim). Must wrap the entire eval loop.
    with torch.no_grad():
        if parallel_dims.tp_enabled:
            from torch.distributed.tensor.parallel import loss_parallel

            ctx = loss_parallel()
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            for batch_idx, (input_dict, labels) in enumerate(eval_batches):
                # Move batch to device
                for k, v in input_dict.items():
                    if isinstance(v, torch.Tensor):
                        input_dict[k] = v.to(device)
                labels = labels.to(device)

                inputs = input_dict["input"]
                extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
                extra_kwargs: dict[str, Any] = {}

                if attn_type in ("flex", "varlen"):
                    assert tokenizer is not None
                    model_ref = cast(ModelProtocol, model)
                    extra_kwargs["attention_masks"] = model_ref.get_attention_masks(
                        input_batch=inputs,
                        tokenizer=tokenizer,
                        extra_inputs=extra_inputs,
                    )

                pred = model(inputs, **extra_inputs, **extra_kwargs)

                # Flatten batch dims, cast to float32, then free bf16 logits
                # immediately to halve peak memory (~13 GiB saved at 131k)
                pred_flat = pred.flatten(0, 1).float()
                del pred

                labels_flat = labels.flatten(0, 1)
                loss = torch.nn.functional.cross_entropy(
                    pred_flat,
                    labels_flat,
                    reduction="sum",
                    ignore_index=IGNORE_INDEX,
                )
                del pred_flat

                num_tokens = (labels_flat != IGNORE_INDEX).sum().item()
                total_loss += loss.item()
                total_tokens += num_tokens

                # Move back to CPU so same batches can be reused
                for k, v in input_dict.items():
                    if isinstance(v, torch.Tensor):
                        input_dict[k] = v.cpu()

                logger.info(
                    f"  Eval batch {batch_idx + 1}/{len(eval_batches)}: "
                    f"{num_tokens} tokens, "
                    f"running avg loss={total_loss / max(total_tokens, 1):.4f}"
                )

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


# ---------------------------------------------------------------------------
# Phase 3: Export — collect amax, gather weights, pack NVFP4, save
# ---------------------------------------------------------------------------


def collect_amax_values(
    model_parts: list[torch.nn.Module],
) -> dict[str, dict[str, float]]:
    """Extract calibrated amax values from quantizer modules.

    Returns a map: {module_path: {quantizer_name: amax_value}}
    Module paths are normalized to strip activation checkpoint wrappers
    (e.g. "_checkpoint_wrapped_module.") so they match state dict keys.
    """
    amax_map = {}
    for model in model_parts:
        for name, module in model.named_modules():
            if hasattr(module, "mlp1_weight_quantizer"):
                amaxes = {}
                for qname in [
                    "mlp1_weight_quantizer",
                    "mlp2_weight_quantizer",
                    "mlp1_input_quantizer",
                    "mlp2_input_quantizer",
                ]:
                    quantizer = getattr(module, qname, None)
                    if quantizer is not None and hasattr(quantizer, "_amax"):
                        amaxes[qname] = quantizer._amax.detach().cpu().float().item()
                if amaxes:
                    # Strip AC wrapper prefix so keys match state dict paths
                    clean_name = name.replace("_checkpoint_wrapped_module.", "")
                    amax_map[clean_name] = amaxes
                    logger.info(f"  {clean_name}: {amaxes}")
    return amax_map


def pack_nvfp4_weight(
    weight: torch.Tensor,
    weight_amax: float,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a single weight tensor to NVFP4 format.

    Args:
        weight: BF16/FP32 weight tensor, any shape. Last dim must be divisible by block_size.
        weight_amax: Calibrated max absolute value for per-tensor scale.
        block_size: Quantization block size (default 16).

    Returns:
        Tuple of (packed_weight, per_block_scale, per_tensor_scale):
        - packed_weight: uint8, shape (..., last_dim // 2) — 2 FP4 values per byte
        - per_block_scale: float8_e4m3fn, shape (..., last_dim // block_size)
        - per_tensor_scale: float32 scalar
    """
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    # Per-tensor scale (sf2): amax / (max_fp4_value * max_fp8_value)
    sf2 = torch.tensor(weight_amax, dtype=torch.float32) / (FP4_E2M1_MAX * FP8_E4M3_MAX)

    quantized, per_block_scale, per_tensor_scale = NVFP4QTensor.quantize(
        weight.float(),  # quantize in FP32 for precision
        block_size=block_size,
        weights_scaling_factor_2=sf2,
    )

    return quantized._quantized_data, per_block_scale, per_tensor_scale


def export_hf_checkpoint(
    model_parts: list[torch.nn.Module],
    sd_adapter,
    output_dir: str,
    amax_map: dict[str, dict[str, float]],
    export_dtype: str = "bfloat16",
    pack_nvfp4: bool = True,
    hf_assets_path: str | None = None,
):
    """Export model to HF safetensors format with optional NVFP4 packing.

    1. Gathers full (unsharded) state dict from all ranks
    2. Converts keys to HF format via sd_adapter
    3. On rank 0: packs expert weights as NVFP4, saves sharded safetensors
    4. Writes standard HF config files (config.json, hf_quant_config.json)
    """
    rank = dist.get_rank()

    # Gather full state dict on all ranks (collective operation)
    logger.info("Gathering full state dict from distributed model...")
    begin = time.monotonic()

    full_state_dict = {}
    for model in model_parts:
        sd = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        full_state_dict.update(sd)

    logger.info(
        f"State dict gathered in {time.monotonic() - begin:.2f}s "
        f"({len(full_state_dict)} keys)"
    )

    if rank != 0:
        # Only rank 0 does the export; free gathered tensors.
        # No barrier — rank 0's export takes minutes of CPU work and would
        # trigger NCCL timeout on waiting ranks.
        del full_state_dict
        torch.cuda.empty_cache()
        logger.info("Rank != 0, export is rank-0 only. Done.")
        return

    # Filter out quantizer keys (only keep original model params/buffers)
    model_state_dict = {}
    quantizer_keys_dropped = 0
    for key, value in full_state_dict.items():
        if any(q in key for q in ["_quantizer", "_amax", "_pre_quant_scale"]):
            quantizer_keys_dropped += 1
            continue
        model_state_dict[key] = value
    del full_state_dict

    logger.info(
        f"Filtered state dict: {len(model_state_dict)} model keys, "
        f"{quantizer_keys_dropped} quantizer keys dropped"
    )

    # Convert to HF format (key mapping + expert weight transpose)
    hf_state_dict = sd_adapter.to_hf(model_state_dict)
    del model_state_dict
    logger.info(f"Converted to HF format: {len(hf_state_dict)} keys")

    # Apply export dtype to non-expert weights
    target_dtype = TORCH_DTYPE_MAP.get(export_dtype, torch.bfloat16)

    if pack_nvfp4:
        # Pack expert weights as NVFP4
        logger.info("Packing expert weights as NVFP4...")
        _pack_expert_weights(hf_state_dict, amax_map, target_dtype)
    else:
        # Just convert all weights to target dtype
        hf_state_dict = {
            k: v.to(target_dtype) if v.is_floating_point() else v
            for k, v in hf_state_dict.items()
        }

    # Save sharded safetensors
    os.makedirs(output_dir, exist_ok=True)
    _save_sharded_safetensors(hf_state_dict, output_dir)

    # Save amax values for reference
    amax_path = os.path.join(output_dir, "quantizer_amax.json")
    with open(amax_path, "w") as f:
        json.dump(amax_map, f, indent=2)

    # Write standard HF config files
    if pack_nvfp4:
        exclude_modules = _compute_exclude_modules(hf_state_dict, amax_map)
        _write_hf_quant_config(output_dir, exclude_modules)
        _write_model_config(output_dir, hf_assets_path, exclude_modules)
    else:
        _write_model_config(
            output_dir, hf_assets_path, exclude_modules=[], is_quantized=False
        )

    # Copy tokenizer and template files from HF assets for a self-contained repo
    if hf_assets_path:
        for fname in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "chat_template.jinja",
        ]:
            src = os.path.join(hf_assets_path, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, fname))
                logger.info(f"Copied {fname} from HF assets")

    logger.info(f"Export complete: {output_dir}")


def _pack_expert_weights(
    hf_state_dict: dict[str, torch.Tensor],
    amax_map: dict[str, dict[str, float]],
    target_dtype: torch.dtype,
):
    """Pack expert MLP weights as NVFP4 in-place in the HF state dict.

    Expert weight keys in HF format:
        model.layers.N.mlp.experts.gate_up_proj  (mlp1)
        model.layers.N.mlp.experts.down_proj     (mlp2)

    For each, we:
    1. Get the weight tensor (already transposed to HF format: E, in, out)
    2. Transpose to (E, out, in) for block quantization (last dim = in_dim)
    3. Pack with NVFP4QTensor.quantize()
    4. Transpose packed weight + scale BACK to HF layout (E, in_packed, out)
    5. Store scalar weight_scale_2 and input_scale (matching official ModelOpt format)

    This matches the official ModelOpt unified_export_hf.py flow:
        transpose → quantize → transpose back
    """
    packed_count = 0
    keys_to_process = []

    for key in list(hf_state_dict.keys()):
        if "mlp.experts" in key and (
            key.endswith("gate_up_proj") or key.endswith("down_proj")
        ):
            keys_to_process.append(key)

    for hf_key in keys_to_process:
        weight = hf_state_dict[hf_key]
        if weight.ndim != 3:
            continue

        # Extract layer number and determine quantizer names
        layer_match = re.search(r"layers\.(\d+)", hf_key)
        if not layer_match:
            continue
        layer_num = layer_match.group(1)

        # Map HF key back to TorchTitan module path for amax lookup
        tt_module_path = f"layers.{layer_num}.moe.experts"
        if tt_module_path not in amax_map:
            logger.warning(
                f"No amax found for {tt_module_path}, skipping NVFP4 packing"
            )
            hf_state_dict[hf_key] = weight.to(target_dtype)
            continue

        amaxes = amax_map[tt_module_path]

        if hf_key.endswith("gate_up_proj"):
            weight_amax = amaxes.get("mlp1_weight_quantizer", None)
            input_amax = amaxes.get("mlp1_input_quantizer", None)
        else:  # down_proj
            weight_amax = amaxes.get("mlp2_weight_quantizer", None)
            input_amax = amaxes.get("mlp2_input_quantizer", None)

        if weight_amax is None:
            logger.warning(f"No weight amax for {hf_key}, skipping packing")
            hf_state_dict[hf_key] = weight.to(target_dtype)
            continue

        # HF format is (E, in_dim, out_dim). Transpose to (E, out_dim, in_dim)
        # so that in_dim is at dim -1 for NVFP4 block quantization
        weight_for_quant = weight.transpose(1, 2).contiguous()

        # Pack NVFP4
        packed_data, per_block_scale, per_tensor_scale = pack_nvfp4_weight(
            weight_for_quant, weight_amax
        )

        # Verify packed shape before transpose: (E, out_dim, in_dim//2)
        # HF weight was (E, in_dim, out_dim), transposed to (E, out_dim, in_dim) for quant
        E, out_dim, in_dim = weight.shape[0], weight.shape[2], weight.shape[1]
        assert packed_data.shape == (E, out_dim, in_dim // 2), (
            f"Unexpected packed shape {packed_data.shape}, "
            f"expected ({E}, {out_dim}, {in_dim // 2})"
        )

        # Transpose BACK to HF layout: (E, out, in_packed) → (E, in_packed, out)
        # This matches official ModelOpt's maybe_transpose_expert_weight_dimensions()
        # which is called after quantization to restore the BMM-format layout.
        packed_data = packed_data.transpose(1, 2).contiguous()
        per_block_scale = per_block_scale.transpose(1, 2).contiguous()

        hf_state_dict[hf_key] = packed_data
        hf_state_dict[f"{hf_key}_weight_scale"] = per_block_scale

        # Per-tensor scale: scalar () matching official ModelOpt format.
        # ModelOpt's unified_export_hf.py stores weight_scale_2 as a squeezed
        # scalar via get_weight_scaling_factor_2().squeeze().
        per_tensor_scale_f32 = per_tensor_scale.float().squeeze()
        hf_state_dict[f"{hf_key}_weight_scale_2"] = per_tensor_scale_f32

        # Store activation scale if available — scalar () matching ModelOpt format.
        if input_amax is not None:
            input_scale_val = torch.tensor(input_amax, dtype=torch.float32) / (
                FP4_E2M1_MAX * FP8_E4M3_MAX
            )
            hf_state_dict[f"{hf_key}_input_scale"] = input_scale_val.squeeze()

        packed_count += 1
        logger.info(
            f"  Packed {hf_key}: {tuple(weight.shape)} → {tuple(packed_data.shape)} uint8 "
            f"(amax={weight_amax:.4f})"
        )

    # Convert remaining (non-expert) weights to target dtype.
    # Preserve: uint8 (packed NVFP4), float8 (per-block scales), and
    # float32 scale tensors (weight_scale_2, input_scale).
    _scale_suffixes = ("_weight_scale_2", "_input_scale")
    for key in list(hf_state_dict.keys()):
        value = hf_state_dict[key]
        if not value.is_floating_point() or value.dtype == torch.uint8:
            continue
        if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            continue
        if any(key.endswith(s) for s in _scale_suffixes):
            continue  # keep float32 scales intact
        hf_state_dict[key] = value.to(target_dtype)

    logger.info(f"Packed {packed_count} expert weight tensors as NVFP4")


def _compute_exclude_modules(
    hf_state_dict: dict[str, torch.Tensor],
    amax_map: dict[str, dict[str, float]],
) -> list[str]:
    """Compute HF module paths that are NOT quantized.

    Derives the list by comparing all modules in the HF state dict against
    the calibrated modules in amax_map. Used for hf_quant_config.json's
    exclude_modules field.
    """
    # Build set of quantized HF module prefixes from amax_map.
    # amax_map keys are TorchTitan paths: "layers.0.moe.experts"
    # HF equivalents:                     "model.layers.0.mlp.experts"
    quantized_prefixes = set()
    for tt_path in amax_map:
        hf_prefix = "model." + tt_path.replace(".moe.", ".mlp.")
        quantized_prefixes.add(hf_prefix)

    # Collect leaf module paths from state dict keys
    all_modules = set()
    for key in hf_state_dict:
        # Skip derived scale keys added during packing
        if any(
            key.endswith(s)
            for s in ("_weight_scale", "_weight_scale_2", "_input_scale")
        ):
            continue
        # Strip .weight/.bias suffix to get module path
        if key.endswith((".weight", ".bias")):
            module_path = key.rsplit(".", 1)[0]
        else:
            module_path = key
        all_modules.add(module_path)

    # Only include Linear-like modules in exclude list. Embeddings, norms,
    # and buffers are implicitly excluded by targets=["Linear"] in the
    # compressed-tensors config, so listing them would just add noise.
    linear_suffixes = ("_proj", "lm_head", "router")
    exclude = []
    for mod in sorted(all_modules):
        if not any(mod.startswith(qp) for qp in quantized_prefixes):
            if any(mod.endswith(s) for s in linear_suffixes):
                exclude.append(mod)

    return exclude


def _save_sharded_safetensors(
    state_dict: dict[str, torch.Tensor],
    output_dir: str,
    max_shard_bytes: int = 5 * 1024**3,
) -> None:
    """Save state dict as sharded safetensors with index file.

    If total size <= max_shard_bytes, saves a single model.safetensors.
    Otherwise, splits into model-00001-of-NNNNN.safetensors shards
    with a model.safetensors.index.json weight map.
    """
    from safetensors.torch import save_file

    sorted_keys = sorted(state_dict.keys())
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())

    if total_bytes <= max_shard_bytes:
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
        logger.info(f"Saved single safetensors file ({total_bytes / 1024**3:.2f} GiB)")
        return

    # Build shards by accumulating tensors until size limit
    shards: list[dict[str, torch.Tensor]] = []
    current_shard: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for key in sorted_keys:
        tensor = state_dict[key]
        tensor_bytes = tensor.numel() * tensor.element_size()

        if current_bytes + tensor_bytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_bytes = 0

        current_shard[key] = tensor
        current_bytes += tensor_bytes

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    weight_map = {}

    for i, shard in enumerate(shards):
        shard_name = f"model-{i + 1:05d}-of-{num_shards:05d}.safetensors"
        save_file(shard, os.path.join(output_dir, shard_name))
        for key in shard:
            weight_map[key] = shard_name
        shard_bytes = sum(v.numel() * v.element_size() for v in shard.values())
        logger.info(
            f"  Saved {shard_name} "
            f"({shard_bytes / 1024**3:.2f} GiB, {len(shard)} tensors)"
        )

    # Write index file
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"Saved {num_shards} shards ({total_bytes / 1024**3:.2f} GiB total)")


def _write_hf_quant_config(output_dir: str, exclude_modules: list[str]) -> None:
    """Write hf_quant_config.json in raw ModelOpt format.

    This is the backward-compatibility artifact that ModelOpt's export pipeline
    writes alongside the compressed-tensors format in config.json.
    """
    try:
        import modelopt

        version = getattr(modelopt, "__version__", "unknown")
    except ImportError:
        version = "unknown"

    config = {
        "producer": {
            "name": "modelopt",
            "version": version,
        },
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "group_size": 16,
            "exclude_modules": exclude_modules,
        },
    }

    path = os.path.join(output_dir, "hf_quant_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Wrote {path}")


def _write_model_config(
    output_dir: str,
    hf_assets_path: str | None,
    exclude_modules: list[str],
    is_quantized: bool = True,
) -> None:
    """Write config.json: copy base from HF assets, inject quantization_config.

    Follows ModelOpt's convention: base config from model.save_pretrained(),
    then patched with quantization_config in compressed-tensors format.
    """
    config_data = {}

    # Load base config from HF assets
    if hf_assets_path:
        src = os.path.join(hf_assets_path, "config.json")
        if os.path.exists(src):
            with open(src) as f:
                config_data = json.load(f)
            logger.info(f"Loaded base config.json from {src}")

    if not config_data:
        logger.warning("No base config.json found. Writing minimal config.")
        config_data = {"model_type": "gpt_oss"}

    # Replace quantization_config with NVFP4 compressed-tensors format
    if is_quantized:
        try:
            import modelopt

            version = getattr(modelopt, "__version__", "unknown")
        except ImportError:
            version = "unknown"

        config_data["quantization_config"] = {
            "config_groups": {
                "group_0": {
                    "input_activations": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                    "weights": {
                        "dynamic": False,
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                    },
                    "targets": ["Linear"],
                },
            },
            "ignore": exclude_modules,
            "quant_algo": "NVFP4",
            "producer": {"name": "modelopt", "version": version},
            "quant_method": "modelopt",
        }
    elif "quantization_config" in config_data:
        # Remove original quantization config for non-quantized export
        del config_data["quantization_config"]

    path = os.path.join(output_dir, "config.json")
    with open(path, "w") as f:
        json.dump(config_data, f, indent=4)
    logger.info(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@record
def main():
    init_logger()

    # Parse config
    config_manager = ConfigManager()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for HF checkpoint",
    )
    parser.add_argument(
        "--no_pack_nvfp4",
        action="store_true",
        help="Save BF16 weights without NVFP4 packing",
    )

    known_args, remaining_args = parser.parse_known_args()

    # Pass remaining args to ConfigManager
    job_config = config_manager.parse_args(remaining_args)

    output_dir = known_args.output_dir
    do_pack = not known_args.no_pack_nvfp4

    # Override calibration steps: 32 for NVFP4 pack, 64 for BF16 export.
    # NVFP4's per-tensor amax converges fast (32 steps); without packing
    # we collect more samples (64) for better distribution estimates.
    calib_steps = 32 if do_pack else 64
    job_config.qat.calib_steps = calib_steps

    logger.info(
        f"PTQ Export: output_dir={output_dir}, pack_nvfp4={do_pack}, "
        f"calib_steps={calib_steps}"
    )

    # Phase 1: Setup
    (
        model_parts,
        parallel_dims,
        device,
        train_spec,
        model_args,
        tokenizer,
        dataloader,
        sd_adapter,
    ) = setup_model(job_config)

    # Collect eval batches on CPU before calibration consumes the iterator.
    # Same batches are replayed for both pre- and post-quantization eval.
    eval_batches = collect_eval_batches(dataloader, num_batches=4)

    # Phase 1.5: Pre-quantization eval (baseline loss)
    logger.info("Computing pre-quantization eval loss...")
    pre_loss = eval_model_loss(
        model_parts, eval_batches, parallel_dims, model_args, tokenizer, device
    )
    logger.info(f"Pre-quantization eval loss: {pre_loss:.4f}")

    # Phase 2: Calibrate
    calibrate(
        model_parts,
        parallel_dims,
        device,
        job_config,
        model_args,
        tokenizer,
        dataloader,
    )

    # Phase 2.5: Post-quantization eval (measures fake-quant degradation)
    logger.info("Computing post-quantization eval loss...")
    post_loss = eval_model_loss(
        model_parts, eval_batches, parallel_dims, model_args, tokenizer, device
    )
    logger.info(f"Post-quantization eval loss: {post_loss:.4f}")
    if pre_loss > 0:
        logger.info(
            f"Quantization impact: {pre_loss:.4f} -> {post_loss:.4f} "
            f"(delta = {post_loss - pre_loss:+.4f}, "
            f"{(post_loss - pre_loss) / pre_loss * 100:+.1f}%)"
        )
    del eval_batches

    # Phase 3: Export
    amax_map = collect_amax_values(model_parts)

    assert sd_adapter is not None, (
        "state_dict_adapter required for HF export. Set model.hf_assets_path in config."
    )

    # Use bfloat16 for NVFP4 export (matches NVIDIA convention),
    # fall back to config value for non-quantized export
    export_dtype = "bfloat16" if do_pack else job_config.checkpoint.export_dtype

    export_hf_checkpoint(
        model_parts,
        sd_adapter,
        output_dir,
        amax_map,
        export_dtype=export_dtype,
        pack_nvfp4=do_pack,
        hf_assets_path=job_config.model.hf_assets_path,
    )

    # Cleanup
    dist.destroy_process_group()
    logger.info("PTQ export finished successfully")


if __name__ == "__main__":
    main()
