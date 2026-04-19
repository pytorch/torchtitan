# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Flattener: traces a model's forward+backward training step and writes it
as a straight-line sequence of PyTorch ops in a standalone Python file.

Usage (via shell script):
    NGPU=8 HARDWARE=h100-sm90 ./torchtitan/experiments/llm_trainer/run_flattener.sh \
        --module graph_trainer.llama3 \
        --config graph_trainer_llama3_debugmodel

Usage (direct torchrun):
    torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
        -m torchtitan.experiments.llm_trainer.flattener \
        --hardware h100-sm90 \
        --module graph_trainer.llama3 \
        --config graph_trainer_llama3_debugmodel

This will:
  1. Build and parallelize the model across GPUs via torchrun
  2. Trace each rank's fwd+loss+bwd step via make_fx
  3. Write the traced graph to <hw>/flattened_models/<name>_rank{i}.py
  4. Verify bitwise equivalence between in-memory graph and generated file
  5. Copy the verified file to <hw>/optimized_models/ as the initial baseline
"""

import argparse
import contextlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.graph_trainer.common_utils import (
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import trace_train_step
from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step
from torchtitan.models.common.decoder import Decoder
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger


def _parse_flattener_args():
    """Parse flattener-specific args, forwarding the rest to ConfigManager."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--hardware",
        type=str,
        required=True,
        help="Hardware/software specialization namespace (e.g. h100-sm90).",
    )
    known, remaining = parser.parse_known_args()
    return known, remaining


def _build_fingerprint(hardware, parallel_dims):
    """Build a directory fingerprint encoding hardware + parallelism.

    Only includes parallelism dimensions > 1 to keep the name concise.
    E.g. "h100-sm90" with tp=2, dp_shard=4 -> "h100-sm90_tp2_fsdp4"
    """
    parts = [hardware]
    dims = [
        ("tp", parallel_dims.tp),
        ("fsdp", parallel_dims.dp_shard),
        ("dp", parallel_dims.dp_replicate),
        ("pp", parallel_dims.pp),
        ("cp", parallel_dims.cp),
        ("ep", parallel_dims.ep),
    ]
    for name, val in dims:
        if val > 1:
            parts.append(f"{name}{val}")
    return "_".join(parts)


def _derive_output_name(model_spec):
    """Derive output filename stem from model spec.

    E.g. model_spec.name="graph_trainer/llama3", flavor="debugmodel"
    -> "llama3_debugmodel"
    """
    model_name = model_spec.name.split("/")[-1]
    return f"{model_name}_{model_spec.flavor}"


def _bitwise_equal(a, b):
    """Compare two tensors at the byte level (handles NaN correctly)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a_flat = a.contiguous().reshape(-1)
    b_flat = b.contiguous().reshape(-1)
    if a_flat.numel() == 0:
        return True
    return torch.equal(
        a_flat.view(torch.uint8),
        b_flat.view(torch.uint8),
    )


def _create_real_inputs(example_inputs, device):
    """Create real tensors matching FakeTensor shapes from tracing."""
    torch.manual_seed(42)
    real = []
    for x in example_inputs:
        if isinstance(x, torch.Tensor):
            shape = tuple(int(s) for s in x.shape)
            if x.is_floating_point():
                real.append(torch.randn(shape, dtype=x.dtype, device=device) * 0.01)
            elif x.dtype in (torch.int32, torch.int64):
                real.append(torch.randint(0, 1000, shape, dtype=x.dtype, device=device))
            else:
                real.append(torch.zeros(shape, dtype=x.dtype, device=device))
        else:
            real.append(x)
    return real


def _load_graph_module(filepath):
    """Import a generated model file and return an instantiated GraphModule."""
    spec = importlib.util.spec_from_file_location("_gen_model", str(filepath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GraphModule()


def _setup_model(config):
    """Build and parallelize the model.

    Expects torch.distributed to already be initialized (via torchrun).
    """
    device_type = utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)

    world_size = dist_utils.init_distributed(
        config.comm,
        enable_cpu_backend=config.training.enable_cpu_offload,
        base_folder=config.dump_folder,
    )

    parallel_dims = ParallelDims.from_config(config.parallelism, world_size)

    dist_utils.set_determinism(
        parallel_dims,
        device,
        config.debug,
        distinct_seed_mesh_dims=["pp"],
    )

    model_spec = config.model_spec
    if model_spec is None:
        raise ValueError(
            "model_spec must be set. Pass --module to specify the model "
            "(e.g. --module graph_trainer.llama3)."
        )

    model_config = model_spec.model
    model_config.update_from_config(trainer_config=config)

    logger.info(f"Building {model_spec.name} {model_spec.flavor} on meta device")
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
    ):
        model = model_config.build()

    model.verify_module_protocol()

    compile_config = config.compile

    model = model_spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=config.training,
        model_converters=config.model_converters,
        parallelism=config.parallelism,
        compile_config=compile_config,
        ac_config=config.activation_checkpoint,
        dump_folder=config.dump_folder,
    )

    model.to_empty(device=device_type)
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    model.train()

    logger.info("Model parallelized and materialized")

    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)

    return (
        model,
        model_config,
        model_spec,
        compile_config,
        parallel_dims,
        device,
        tokenizer,
    )


def _generate_model_file(gm, state_fqns, output_name, rank, output_dir):
    """Write the traced GraphModule as a standalone Python file."""
    code = gm.print_readable(
        print_output=False,
        include_stride=True,
        include_device=True,
    )

    state_lines = []
    for i, fqn in enumerate(state_fqns):
        state_lines.append(f"#   [{i}] {fqn}")
    state_desc = "\n".join(state_lines)

    header = f'''"""
Flattened model: {output_name} (rank {rank})

Auto-generated by llm_trainer flattener. DO NOT EDIT this file directly.
To optimize: copy this to candidate_models/ and modify the copy.

This file contains the complete forward+backward training step as a
straight-line sequence of PyTorch operations. All model parameters,
inputs, and intermediate activations flow through the forward() method
as explicit tensor arguments.

State inputs (model parameters/buffers, in order):
{state_desc}

Remaining inputs are user data (tokens, labels, positional embeddings, etc.)

Outputs (in order):
  [0]     loss (scalar)
  [1..N]  gradients for each trainable parameter

IMPORTANT: Any optimized version MUST produce a validation loss that is
<= the baseline's after 100 training steps. The benchmarker verifies this.
"""
from math import inf, nan
import torch
import torch.nn as nn
from torch import device, tensor


'''

    import re

    code = re.sub(
        r"class\s+\S+\(torch\.nn\.Module\)",
        "class GraphModule(torch.nn.Module)",
        code,
        count=1,
    )

    full_code = header + code + "\n"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{output_name}_rank{rank}.py"
    filepath.write_text(full_code)
    return filepath


def _generate_metadata(
    state_fqns,
    example_inputs,
    output_name,
    rank,
    num_flops_per_token,
    model_param_count,
    config,
    parallel_dims,
    output_dir,
):
    """Save input specs and model info as JSON for the benchmarker."""
    specs = []
    for i, inp in enumerate(example_inputs):
        if isinstance(inp, torch.Tensor):
            specs.append(
                {
                    "index": i,
                    "dtype": str(inp.dtype),
                    "shape": [int(s) for s in inp.shape],
                }
            )
        else:
            try:
                json.dumps(inp)
                value = inp
            except (TypeError, ValueError):
                value = str(inp)
            specs.append(
                {
                    "index": i,
                    "type": type(inp).__name__,
                    "value": value,
                }
            )

    metadata = {
        "output_name": output_name,
        "rank": rank,
        "num_state_inputs": len(state_fqns),
        "state_fqns": state_fqns,
        "input_specs": specs,
        "num_inputs": len(example_inputs),
        "num_flops_per_token": num_flops_per_token,
        "model_param_count": model_param_count,
        "seq_len": config.training.seq_len,
        "local_batch_size": config.training.local_batch_size,
        "world_size": parallel_dims.world_size,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{output_name}_rank{rank}_meta.json"
    filepath.write_text(json.dumps(metadata, indent=2))
    return filepath


def _verify_equivalence(gm, example_inputs, model_filepath, device):
    """Verify generated file produces bitwise identical results to in-memory graph.

    Creates random tensors matching the traced input shapes, runs both the
    in-memory GraphModule and the file-loaded GraphModule, and compares
    every output tensor at the byte level.
    """
    real_inputs = _create_real_inputs(example_inputs, device)

    with torch.no_grad():
        ref_outputs = gm(*real_inputs)

    gen_gm = _load_graph_module(model_filepath)
    with torch.no_grad():
        gen_outputs = gen_gm(*real_inputs)

    if isinstance(ref_outputs, torch.Tensor):
        ref_outputs = (ref_outputs,)
    if isinstance(gen_outputs, torch.Tensor):
        gen_outputs = (gen_outputs,)
    if isinstance(ref_outputs, list):
        ref_outputs = tuple(ref_outputs)
    if isinstance(gen_outputs, list):
        gen_outputs = tuple(gen_outputs)

    if len(ref_outputs) != len(gen_outputs):
        logger.error(
            f"Output count mismatch: in-memory={len(ref_outputs)}, "
            f"generated={len(gen_outputs)}"
        )
        return False

    all_match = True
    for i, (ref, gen) in enumerate(zip(ref_outputs, gen_outputs)):
        if isinstance(ref, torch.Tensor) and isinstance(gen, torch.Tensor):
            if _bitwise_equal(ref, gen):
                if i == 0:
                    logger.info(
                        f"  Output {i} (loss): MATCH "
                        f"(shape={list(ref.shape)}, dtype={ref.dtype})"
                    )
            else:
                max_diff = (ref.float() - gen.float()).abs().max().item()
                logger.error(
                    f"  Output {i}: MISMATCH! "
                    f"(shape={list(ref.shape)}, dtype={ref.dtype}, "
                    f"max_diff={max_diff:.6e})"
                )
                all_match = False
    if all_match:
        n = len(ref_outputs)
        logger.info(f"  All {n} outputs match (loss + {n - 1} gradients)")

    return all_match


def main():
    init_logger()

    flattener_args, remaining_args = _parse_flattener_args()

    config_manager = ConfigManager()
    config = config_manager.parse_args(remaining_args)

    config.compile.mode = "aot_fx_trace"
    config.debug.deterministic = True
    if config.debug.seed is None:
        config.debug.seed = 42

    (
        model,
        model_config,
        model_spec,
        compile_config,
        parallel_dims,
        device,
        tokenizer,
    ) = _setup_model(config)

    model_param_count, num_flops_per_token = model_config.get_nparams_and_flops(
        model, config.training.seq_len
    )
    logger.info(
        f"Model params: {model_param_count:,}, " f"FLOPs/token: {num_flops_per_token:,}"
    )

    loss_fn = model_spec.build_loss_fn(compile_config, parallel_dims=parallel_dims)
    fwd_bwd_fn = make_fwd_bwd_step(loss_fn)

    seq_len = config.training.seq_len
    local_batch_size = config.training.local_batch_size
    vocab_size = model_config.vocab_size

    dummy_inputs = torch.randint(
        0, vocab_size, (local_batch_size, seq_len), device=device
    )
    dummy_labels = torch.randint(
        0, vocab_size, (local_batch_size, seq_len), device=device
    )

    global_batch_size = (
        local_batch_size
        * parallel_dims.dp_shard
        * parallel_dims.dp_replicate
        * parallel_dims.cp
    )
    dummy_global_valid_tokens = float(global_batch_size * seq_len)
    extra_inputs: dict[str, torch.Tensor] = {}
    extra_kwargs: dict[str, Any] = {}

    if isinstance(model_config, Decoder.Config):
        layer = model_config.layers[0]
        attn_config = layer.attention
    else:
        attn_config = None
    mask_type = getattr(attn_config, "mask_type", "causal")

    if mask_type == "block_causal":
        extra_kwargs["positions"] = (
            torch.arange(0, seq_len, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(local_batch_size, -1)
        )
    elif parallel_dims.cp_enabled:
        extra_kwargs["positions"] = torch.arange(
            0, seq_len, dtype=torch.int32, device=device
        ).expand(local_batch_size, seq_len)

    inner_attention = getattr(attn_config, "inner_attention", None)
    if inner_attention is not None:
        from torchtitan.models.common.attention import FlexAttention, VarlenAttention

        if isinstance(inner_attention, (FlexAttention.Config, VarlenAttention.Config)):
            extra_kwargs["attention_masks"] = model.get_attention_masks(
                input_batch=dummy_inputs,
                tokenizer=tokenizer,
                extra_inputs=extra_inputs,
            )

    loss_parallel_enabled = (
        parallel_dims.tp_enabled and not config.parallelism.disable_loss_parallel
    )
    loss_parallel_ctx = (
        torch.distributed.tensor.parallel.loss_parallel()
        if loss_parallel_enabled
        else contextlib.nullcontext()
    )

    maybe_register_blockmask_pytree_node()

    logger.info("Tracing fwd+loss+bwd via make_fx...")
    with loss_parallel_ctx:
        traced_result = trace_train_step(fwd_bwd_fn)(
            model,
            dummy_inputs,
            dummy_labels,
            dummy_global_valid_tokens,
            extra_inputs,
            extra_kwargs,
        )

    gm = traced_result.gm
    num_nodes = len(list(gm.graph.nodes))
    logger.info(
        f"Traced graph: {num_nodes} nodes, "
        f"{len(traced_result.state_fqns)} state entries"
    )

    output_name = _derive_output_name(model_spec)
    rank = dist.get_rank()
    fingerprint = _build_fingerprint(flattener_args.hardware, parallel_dims)
    llm_trainer_dir = Path(__file__).parent
    fingerprint_dir = llm_trainer_dir / "targets" / fingerprint
    logger.info(f"Fingerprint: {fingerprint}")

    flattened_dir = fingerprint_dir / "flattened_models"
    model_file = _generate_model_file(
        gm, traced_result.state_fqns, output_name, rank, flattened_dir
    )
    logger.info(f"Wrote flattened model: {model_file}")

    meta_file = _generate_metadata(
        traced_result.state_fqns,
        traced_result.example_inputs,
        output_name,
        rank,
        num_flops_per_token,
        model_param_count,
        config,
        parallel_dims,
        flattened_dir,
    )
    logger.info(f"Wrote metadata: {meta_file}")

    logger.info("Verifying bitwise equivalence (in-memory graph vs generated file)...")
    verified = _verify_equivalence(gm, traced_result.example_inputs, model_file, device)

    if not verified:
        logger.error(
            "Verification FAILED! The generated file does not match "
            "the in-memory graph. This is a bug in the flattener."
        )
        dist.destroy_process_group()
        raise RuntimeError("Flattener verification failed")

    optimized_dir = fingerprint_dir / "optimized_models"
    optimized_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_file, optimized_dir / model_file.name)
    shutil.copy2(meta_file, optimized_dir / meta_file.name)
    logger.info(f"Copied baseline to {optimized_dir}/")

    dist.destroy_process_group()
    logger.info("Flattening complete!")


if __name__ == "__main__":
    main()
