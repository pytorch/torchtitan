# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replay a production DeepSeek-V3 debug recipe with native packed ranks."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch_remat
from torch.distributed.tensor import DTensor
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel

from torchtitan.config import PrecisionConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.observability.logging import init_logger


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if isinstance(value, DTensor) else value


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _local_tensor(value).detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _tensor_hash(value: torch.Tensor) -> str:
    local = _local_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(local.view(torch.uint8).numpy().tobytes()).hexdigest()


def _parameters(engine: Any, *, gradients: bool = False) -> dict[str, Any]:
    values = {}
    for part_id, model in enumerate(engine.model_parts):
        for name, parameter in model.named_parameters():
            value = parameter.grad if gradients else parameter
            values[f"{part_id}:{name}"] = _cpu_tree(value)
    return values


def _buffers(engine: Any) -> dict[str, Any]:
    return {
        f"{part_id}:{name}": _cpu_tree(buffer)
        for part_id, model in enumerate(engine.model_parts)
        for name, buffer in model.named_buffers()
    }


def _layout(engine: Any) -> dict[str, Any]:
    result = {}
    for part_id, model in enumerate(engine.model_parts):
        for name, parameter in model.named_parameters():
            local = _local_tensor(parameter)
            entry = {
                "global_shape": list(parameter.shape),
                "local_shape": list(local.shape),
                "local_stride": list(local.stride()),
                "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
            }
            if isinstance(parameter, DTensor):
                entry["placements"] = [str(item) for item in parameter.placements]
                entry["mesh"] = parameter.device_mesh.mesh.cpu().tolist()
                entry["mesh_axis_names"] = list(parameter.device_mesh.mesh_dim_names)
            result[f"{part_id}:{name}"] = entry
    return result


def _finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def main() -> None:
    init_logger()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    output = Path(os.environ["PACKED_OUTPUT"])
    steps = int(os.environ["PACKED_STEPS"])
    seq_len = int(os.environ["PACKED_SEQ_LEN"])
    if world_size != 8:
        raise ValueError("This recipe requires exactly eight logical ranks.")
    config = deepseek_v3_debugmodel(
        seq_len=seq_len,
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2,
            data_parallel_shard_degree=4,
            expert_parallel_degree=8,
            save_parallelism_folder="mesh",
        ),
    )
    config.training.steps = steps
    config.training.num_tokens_per_microbatch_per_dp_rank = 2 * seq_len
    config.debug.seed = 42
    config.debug.deterministic = True
    config.debug.deterministic_warn_only = False
    config.debug.enable_structured_logging = False
    config.precision = PrecisionConfig(
        dtype="float32", param_dtype="bfloat16", reduce_dtype="float32"
    )
    config.dump_folder = str(output)
    config.comm.init_timeout_seconds = 180
    config.comm.train_timeout_seconds = 90
    config.metrics.loggers = []
    if os.environ.get("PACKED_DIST_MOE") == "1":
        from torchtitan.config.transform import apply_transforms
        from torchtitan.config.transform.dist_moe import DistMoeTransform
        from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime

        config.dist_moe_runtime = DistMoeRuntime.Config(
            scratch_capacity_factor=4.0,
            activation_slot_capacity_factor=1.0,
        )
        config = apply_transforms(config, [DistMoeTransform()])
    print(f"RANK {rank}: build production Trainer", flush=True)
    trainer = config.build()
    engine = trainer.engine
    metadata = {
        "rank": rank,
        "local_rank": int(os.environ["LOCAL_RANK"]),
        "world_size": dist.get_world_size(),
        "device": str(engine.device),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu": torch.cuda.get_device_name(engine.device),
        "torch": torch.__version__,
        "nccl": torch.cuda.nccl.version(),
        "parallelism": {
            "tp": 2,
            "dp_shard": 4,
            "ep": 8,
            "pp": 1,
            "sequence_parallel": True,
        },
        "model": "DeepSeekV3 debugmodel",
        "model_parameters": engine.model_param_count,
        "seq_len": seq_len,
        "tokens_per_dp_rank": config.training.num_tokens_per_microbatch_per_dp_rank,
        "compute_regions": config.model.local_compile.regions,
        "parameter_layout": _layout(engine),
        "requested_steps": steps,
        "dist_moe": os.environ.get("PACKED_DIST_MOE") == "1",
        "attention_impl": torch.nn.attention.current_flash_attention_impl(),
        "modules": {
            name: type(module).__module__ + "." + type(module).__qualname__
            for name, module in engine.model_parts[0].named_modules()
            if "router" in name
            or "token_dispatcher" in name
            or "inner_attention" in name
            or "routed_experts" in name
        },
    }
    (output / f"metadata_rank_{rank}.json").write_text(json.dumps(metadata, indent=2))
    torch.save(
        {
            "parameters": _parameters(engine),
            "buffers": _buffers(engine),
            "engine": _cpu_tree(engine.state_dict()),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(),
        },
        output / f"initial_rank_{rank}.pt",
    )
    routes = []
    hooks = []
    for name, module in engine.model_parts[0].named_modules():
        if isinstance(module, TokenChoiceTopKRouter):

            def record_route(module, inputs, result, *, name=name):
                expert_ids = _local_tensor(result[1]).detach()
                counts = torch.bincount(expert_ids.flatten().long(), minlength=8)
                routes.append(
                    {
                        "module": name,
                        "counts": counts.cpu().tolist(),
                        "recomputing": torch_remat.is_recomputing(),
                        "expert_ids_sha256": _tensor_hash(expert_ids),
                        "scores_sha256": _tensor_hash(result[0]),
                    }
                )

            hooks.append(module.register_forward_hook(record_route))
    if len(hooks) != 5:
        raise ValueError(f"Expected five learned MoE routers, found {len(hooks)}.")
    data_iterator = trainer.microbatch_generator(trainer.dataloader)
    summaries = []
    profile = None
    if os.environ.get("PACKED_PROFILE") == "1":
        profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )
        profile.__enter__()
    try:
        for step in range(1, steps + 1):
            print(f"RANK {rank}: training step {step}", flush=True)
            before_routes = len(routes)
            started = time.monotonic()
            original_step = engine.optim_step
            gradients = {}
            grad_norm_value = None

            def captured_optim_step(*, original_step=original_step):
                nonlocal gradients, grad_norm_value
                gradients = _parameters(engine, gradients=True)
                grad_norm = original_step()
                grad_norm_value = _local_tensor(grad_norm).item()
                return grad_norm

            engine.optim_step = captured_optim_step
            try:
                trainer.train_step(data_iterator)
            finally:
                engine.optim_step = original_step
            torch.cuda.synchronize()
            parameters = _parameters(engine)
            optimizer = _cpu_tree(engine.optim.optimizers.state_dict())
            named_buffers = _buffers(engine)
            required = {
                name
                for name, entry in metadata["parameter_layout"].items()
                if entry["requires_grad"]
            }
            if any(gradients[name] is None for name in required):
                raise ValueError("A trainable parameter is missing its gradient.")
            if engine.num_completed_steps != step:
                raise ValueError("The production optimizer update did not complete.")
            if not (
                _finite(parameters)
                and _finite(gradients)
                and _finite(optimizer)
                and _finite(named_buffers)
            ):
                raise ValueError("Nonfinite training state.")
            for name, value in optimizer.items():
                if name.endswith(".step") and float(value) != step:
                    raise ValueError(f"Optimizer counter did not advance: {name}")
            observed_routers = {
                item["module"]
                for item in routes[before_routes:]
                if not item["recomputing"]
            }
            if len(observed_routers) != 5:
                raise ValueError("Not all five production routers executed.")
            torch.save(
                {
                    "parameters": parameters,
                    "gradients_pre_clip": gradients,
                    "named_buffers": named_buffers,
                    "engine": _cpu_tree(engine.state_dict()),
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state(),
                    "optimizer": optimizer,
                    "scheduler": _cpu_tree(engine.optim.lr_schedulers.state_dict()),
                    "routes": routes[before_routes:],
                    "loss": _local_tensor(engine.loss).item(),
                    "grad_norm": grad_norm_value,
                },
                output / f"step_{step}_rank_{rank}.pt",
            )
            summary = {
                "step": step,
                "loss": _local_tensor(engine.loss).item(),
                "grad_norm": grad_norm_value,
                "seconds": time.monotonic() - started,
                "max_memory_allocated": torch.cuda.max_memory_allocated(),
                "routes": routes[before_routes:],
                "finite_parameters": _finite(parameters),
                "finite_gradients": _finite(gradients),
            }
            summaries.append(summary)
            (output / f"steps_rank_{rank}.json").write_text(
                json.dumps(summaries, indent=2)
            )
            print(f"RANK {rank}: " + json.dumps(summary), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        if profile is not None:
            profile.__exit__(None, None, None)
            profile.export_chrome_trace(str(output / f"trace_rank_{rank}.json"))
        trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
    print(f"RANK {rank}: completed {steps} optimizer updates", flush=True)


if __name__ == "__main__":
    main()
