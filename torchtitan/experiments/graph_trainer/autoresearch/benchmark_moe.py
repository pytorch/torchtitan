# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate and benchmark a production DeepSeek V3 MoE invocation after unsharding.

Run baseline and fused modes in separate processes on the same GPU. The layer
uses the production MXFP8 recipe, fake EP communication, and a synthetic scalar
loss. Pipeline scheduling, optimizer state, and parameter unsharding are outside
this single-layer measurement.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeProcessGroup

from torchtitan.config.override import apply_overrides
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.utils import set_determinism
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    canonicalize_graph_pass,
    eliminate_dead_code_pass,
)
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor
from torchtitan_recipes.graph_trainer.deepseek_v3 import (
    graph_trainer_deepseek_v3_671b_dist_moe_mxfp8,
)

from .moe_fusion import fuse_production_moe

# Shape suffixes: T = local tokens, D = model width.


def _fingerprint(tensor: torch.Tensor) -> dict:
    value = tensor.detach().contiguous().cpu()
    data = value.reshape(-1).view(torch.uint8).numpy()
    return {
        "shape": list(value.shape),
        "stride": list(tensor.stride()),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(memoryview(data)).hexdigest(),
    }


def _same_bits(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and actual.stride() == expected.stride()
        and torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8),
        )
    )


def _time_replays(graph: torch.cuda.CUDAGraph, *, steps: int) -> list[float]:
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / steps)
    return samples


def _copy_roofline(num_bytes: int) -> dict:
    source = torch.empty(num_bytes // 4, device="cuda", dtype=torch.float32)
    destination = torch.empty_like(source)
    source.fill_(1)
    for _ in range(10):
        torch.add(source, 0.0, out=destination)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            torch.add(source, 0.0, out=destination)
    samples = [sample / 100 for sample in _time_replays(graph, steps=10)]
    duration_us = statistics.median(samples)
    return {
        "operation": "torch.add(source, 0.0, out=destination)",
        "payload_bytes": num_bytes,
        "read_write_bytes": 2 * num_bytes,
        "median_us": duration_us,
        "measured_tb_per_s": 2 * num_bytes / duration_us / 1e6,
    }


def _build_layer():
    config = graph_trainer_deepseek_v3_671b_dist_moe_mxfp8()
    config.debug.seed = 42
    config.debug.deterministic = True
    apply_overrides(config.override, config)
    parallelism = config.parallelism
    world_size = (
        parallelism.data_parallel_replicate_degree
        * parallelism.data_parallel_shard_degree
        * parallelism.context_parallel_degree
        * parallelism.tensor_parallel_degree
    )
    options = FakeProcessGroup.Options()
    options.simulate_uniform_ranks = True
    dist.init_process_group(
        "fake",
        store=dist.HashStore(),
        rank=0,
        world_size=world_size,
        pg_options=options,
    )
    context = ParallelismContext(
        dp_replicate=parallelism.data_parallel_replicate_degree,
        dp_shard=parallelism.data_parallel_shard_degree,
        cp=parallelism.context_parallel_degree,
        tp=parallelism.tensor_parallel_degree,
        pp=1,
        ep=parallelism.expert_parallel_degree,
        world_size=world_size,
        enable_sequence_parallel=parallelism.enable_sequence_parallel,
    )
    context.build_mesh()
    set_determinism(
        context,
        device=torch.device("cuda"),
        debug_config=config.debug,
        distinct_seed_mesh_axes=[],
    )
    layer_index, layer_config = next(
        (index, layer.moe)
        for index, layer in enumerate(config.model.layers)
        if getattr(layer, "moe", None) is not None
    )
    with context.activate_spmd(), torch.device("meta"):
        model = layer_config.build()
        model._parallelize(context)
    model.to_empty(device="cuda")
    with context.activate_spmd(), torch.no_grad():
        model.init_states(buffer_device=torch.device("cuda"))
    # The invocation starts with the local BF16 operands delivered by unsharding.
    # Plain parameters also retain the gradient edges through make_fx flattening.
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            storage = (
                parameter._tensor
                if isinstance(parameter, _ShardedFSDPTensor)
                else parameter
            )
            module.register_parameter(
                name,
                torch.nn.Parameter(
                    storage.to(torch.bfloat16), requires_grad=parameter.requires_grad
                ),
            )
    model.train()
    runtime = config.dist_moe_runtime.build(
        model_parts=[model],
        parallelism_context=context,
        device=torch.device("cuda"),
        num_tokens_per_microbatch_per_dp_rank=(
            config.training.num_tokens_per_microbatch_per_dp_rank
        ),
        pp_schedule=None,
        wgrad_dtype=torch.bfloat16,
    )
    return config, layer_index, layer_config, model, context, runtime


def benchmark(args: argparse.Namespace) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config, layer_index, layer_config, model, context, runtime = _build_layer()
    print("Production layer and fake EP runtime initialized", flush=True)
    num_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    model_dim = layer_config.router.gate.in_features
    x_TD = torch.empty(num_tokens, model_dim, device="cuda", dtype=torch.bfloat16)
    upstream_TD = torch.empty_like(x_TD)
    denominator = torch.tensor(
        config.training.num_tokens_per_train_step, device="cuda", dtype=torch.float32
    )
    parameter_names = [name for name, _ in model.named_parameters()]
    output_names = ["output", "loss", "input_gradient", *parameter_names]
    buffers = dict(model.named_buffers())
    initial_buffers = {name: tensor.clone() for name, tensor in buffers.items()}

    def reset_buffers():
        for name, tensor in buffers.items():
            tensor.copy_(initial_buffers[name])

    def set_inputs(seed):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x_TD.normal_(generator=generator)
        upstream_TD.normal_(mean=0, std=0.01, generator=generator)

    def step(x_TD, upstream_TD, denominator):
        x_TD = x_TD.detach().requires_grad_()
        output_TD = model(x_TD, aux_loss_denominator=denominator)
        loss = (output_TD.float() * upstream_TD.float()).sum()
        gradients = torch.autograd.grad(loss, (x_TD, *tuple(model.parameters())))
        return (output_TD.detach(), loss.detach(), *gradients)

    set_inputs(42)
    with context.activate_spmd():
        traced = minimal_fx_tracer(step, module=model)(x_TD, upstream_TD, denominator)
    canonicalize_graph_pass(traced.gm)
    eliminate_dead_code_pass(traced.gm)
    baseline_targets = Counter(
        str(node.target) for node in traced.gm.graph.nodes if node.op == "call_function"
    )
    (args.output_dir / "original_graph.py").write_text(traced.gm.code)
    counts = {}
    if args.mode == "fused":
        counts = fuse_production_moe(traced.gm, regions=tuple(args.regions.split(",")))
    (args.output_dir / "graph.py").write_text(traced.gm.code)
    if args.mode == "fused":
        if not counts or not all(counts.values()):
            raise RuntimeError(f"Requested production fusion did not match: {counts}")
    runner = run_traced(traced, module=model)
    print(
        f"Graph ready: {len(list(traced.gm.graph.nodes))} nodes; fusions={counts}",
        flush=True,
    )

    with context.activate_spmd():
        reset_buffers()
        eager = step(x_TD, upstream_TD, denominator)
        eager_buffers = {name: tensor.clone() for name, tensor in buffers.items()}
        reset_buffers()
        actual = runner(x_TD, upstream_TD, denominator)
        eager_parity = {
            name: _same_bits(value, expected)
            for name, value, expected in zip(output_names, actual, eager, strict=True)
        }
        buffer_parity = {
            name: _same_bits(tensor, eager_buffers[name])
            for name, tensor in buffers.items()
        }
        if not all(eager_parity.values()) or not all(buffer_parity.values()):
            raise RuntimeError(f"Eager parity failed: {eager_parity}; {buffer_parity}")
        if not all(bool(torch.isfinite(value).all()) for value in actual):
            raise RuntimeError("The original layer produced a non-finite value")
        del eager, eager_buffers, actual
        for _ in range(args.warmup_steps):
            outputs = runner(x_TD, upstream_TD, denominator)
            del outputs
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_before_capture = torch.cuda.memory_allocated()
    reserved_before_capture = torch.cuda.memory_reserved()
    graph = torch.cuda.CUDAGraph()
    with context.activate_spmd(), torch.cuda.graph(graph):
        outputs = runner(x_TD, upstream_TD, denominator)
    graph.replay()
    torch.cuda.synchronize()
    memory = {
        "allocated_before_capture": allocated_before_capture,
        "reserved_before_capture": reserved_before_capture,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "allocated_after_capture": torch.cuda.memory_allocated(),
        "reserved_after_capture": torch.cuda.memory_reserved(),
    }
    (args.output_dir / "memory_summary.txt").write_text(torch.cuda.memory_summary())
    torch.save(torch.cuda.memory_snapshot(), args.output_dir / "memory_snapshot.pt")
    reference = json.loads(args.reference.read_text()) if args.reference else None
    memory_parity = (
        {name: value <= reference["memory"][name] for name, value in memory.items()}
        if reference is not None
        else {}
    )
    if reference is not None and not all(memory_parity.values()):
        raise RuntimeError(
            f"Memory increased relative to the baseline: {memory_parity}"
        )
    signatures = []
    reset_buffers()
    for index in range(args.validation_steps):
        set_inputs(42 + index)
        graph.replay()
        torch.cuda.synchronize()
        tensors = dict(zip(output_names, outputs, strict=True))
        tensors.update({f"buffer:{name}": tensor for name, tensor in buffers.items()})
        norm = torch.linalg.vector_norm(
            torch.stack(
                [torch.linalg.vector_norm(value.float()) for value in outputs[3:]]
            )
        )
        signature = {
            "seed": 42 + index,
            "loss": outputs[1].item(),
            "grad_norm": norm.item(),
            "input_grad_norm": torch.linalg.vector_norm(outputs[2].float()).item(),
            "tensors": {name: _fingerprint(value) for name, value in tensors.items()},
        }
        if reference is not None and signature != reference["signatures"][index]:
            (args.output_dir / "mismatched_signature.json").write_text(
                json.dumps(signature, indent=2)
            )
            raise RuntimeError(f"Bitwise parity failed at validation step {index}")
        signatures.append(signature)
    print(
        f"Bitwise validation passed for {args.validation_steps} distinct inputs",
        flush=True,
    )
    set_inputs(42)
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    samples = _time_replays(graph, steps=args.steps)
    print(f"Forward/backward median: {statistics.median(samples):.3f} us", flush=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(args.output_dir / "trace.json"))
    kernels = [
        {
            "name": event.key,
            "count": event.count,
            "device_time_us": event.device_time_total,
        }
        for event in profile.key_averages()
        if event.device_time_total
    ]
    roofline = {
        "l2_copy": _copy_roofline(num_tokens * layer_config.num_experts * 4),
        "hbm_copy": _copy_roofline(512 * 1024 * 1024),
    }
    result = {
        "mode": args.mode,
        "recipe": "graph_trainer_deepseek_v3_671b_dist_moe_mxfp8",
        "layer_index": layer_index,
        "shape": {
            "num_tokens": num_tokens,
            "model_dim": model_dim,
            "intermediate_dim": layer_config.routed_experts.w2.in_features,
            "num_experts": layer_config.num_experts,
            "num_local_experts": model.routed_experts.w13.weight.shape[0],
            "top_k": layer_config.router.top_k,
            "num_expert_groups": layer_config.router.num_expert_groups,
            "num_limited_groups": layer_config.router.num_limited_groups,
        },
        "production_parallelism": asdict(config.parallelism),
        "scope": "One MoE after unsharding; fake EP; synthetic loss; no optimizer or PP schedule",
        "seed": 42,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
        "hardware": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torchao_version": version("torchao"),
        "dist_moe_version": version("dist-moe"),
        "fusion_counts": counts,
        "original_call_targets": dict(baseline_targets),
        "call_targets": dict(
            Counter(
                str(node.target)
                for node in traced.gm.graph.nodes
                if node.op == "call_function"
            )
        ),
        "eager_parity": eager_parity,
        "buffer_parity": buffer_parity,
        "signatures": signatures,
        "memory": memory,
        "memory_parity": memory_parity,
        "warmup_steps": args.warmup_steps,
        "steps_per_sample": args.steps,
        "latency_us": samples,
        "median_us": statistics.median(samples),
        "copy_roofline": roofline,
        "profile": kernels,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    runtime.close()
    dist.destroy_process_group()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "fused"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--regions", default="selection,weights,gate_backward,counts,aux,swiglu"
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()
    if min(args.steps, args.validation_steps, args.warmup_steps) < 10:
        parser.error("Use at least 10 warmup, validation, and benchmark steps")
    if args.mode == "fused" and args.reference is None:
        parser.error("Fused mode requires the baseline result.json as --reference")
    benchmark(args)


if __name__ == "__main__":
    main()
