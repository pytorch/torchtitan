# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import inspect
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecision

class _SimpleMPPolicy:
    def __init__(self, param_dtype, reduce_dtype, buffer_dtype, output_dtype, cast_forward_inputs=True):
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.buffer_dtype = buffer_dtype
        self.output_dtype = output_dtype
        self.cast_forward_inputs = cast_forward_inputs
try:
    from torch.distributed.elastic.multiprocessing.errors import record
except Exception:
    record = None
try:
    from torch.distributed.fsdp._fully_shard._fsdp_api import CPUOffloadPolicy
except Exception:
    CPUOffloadPolicy = None

try:
    from torch.distributed.fsdp import fully_shard as fully_shard_fn
    _FULLY_SHARD_SRC = 'torch.distributed.fsdp.fully_shard'
except Exception:
    try:
        from torch.distributed._composable.fsdp import fully_shard as fully_shard_fn
        _FULLY_SHARD_SRC = 'torch.distributed._composable.fsdp.fully_shard'
    except Exception:
        fully_shard_fn = None
        _FULLY_SHARD_SRC = None

from torchtitan.config import ConfigManager
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torchtitan.tools.logging import init_logger, logger
from torchtitan.models.flux2.flux2_datasets import Flux2DataLoader
from torchtitan.models.flux2.src.flux2.util import load_text_encoder
from torchtitan.models.flux2.src.flux2.text_encoder import (
    Mistral3SmallEmbedder,
    load_mistral_small_state_dict,
)


def _get_rank_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank


def _init_device_mesh(world_size: int):
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except Exception:
        return None
    try:
        return init_device_mesh('cuda', (world_size,))
    except Exception:
        return None


def _build_fully_shard_kwargs(world_size: int, device: torch.device, offload_policy):
    if fully_shard_fn is None:
        raise RuntimeError('torch.distributed.fsdp.fully_shard is not available in this PyTorch build')
    sig = inspect.signature(fully_shard_fn)
    kwargs = {}
    if 'mesh' in sig.parameters:
        mesh = _init_device_mesh(world_size)
        if mesh is not None:
            kwargs['mesh'] = mesh
    if 'mp_policy' in sig.parameters:
        kwargs['mp_policy'] = _SimpleMPPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
        )
    elif 'mixed_precision' in sig.parameters:
        kwargs['mixed_precision'] = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    if 'offload_policy' in sig.parameters:
        kwargs['offload_policy'] = offload_policy
    if 'device_id' in sig.parameters:
        kwargs['device_id'] = device
    if 'reshard_after_forward' in sig.parameters:
        kwargs['reshard_after_forward'] = True
    if 'limit_all_gathers' in sig.parameters:
        kwargs['limit_all_gathers'] = True
    return kwargs


def _wrap_module(module, fsdp_kwargs, seen: set[int]) -> None:
    if module is None:
        return
    mid = id(module)
    if mid in seen:
        return
    fully_shard_fn(module, **fsdp_kwargs)
    seen.add(mid)


def _wrap_layers(layers, fsdp_kwargs, seen: set[int]) -> bool:
    if layers is None:
        return False
    if isinstance(layers, (nn.ModuleList, list, tuple)):
        for layer in layers:
            _wrap_module(layer, fsdp_kwargs, seen)
        return True
    return False


def _apply_fsdp_sharding(model, fsdp_kwargs) -> None:
    seen: set[int] = set()
    wrapped_any = False

    # Wrap transformer blocks wherever they live. This keeps all-gathers small.
    for mod in model.modules():
        layers = getattr(mod, "layers", None)
        if isinstance(layers, (nn.ModuleList, list, tuple)):
            for layer in layers:
                _wrap_module(layer, fsdp_kwargs, seen)
                wrapped_any = True

        transformer = getattr(mod, "transformer", None)
        layers = getattr(transformer, "layers", None)
        if isinstance(layers, (nn.ModuleList, list, tuple)):
            for layer in layers:
                _wrap_module(layer, fsdp_kwargs, seen)
                wrapped_any = True

    # Optionally wrap other sizable submodules if present.
    for name, mod in model.named_modules():
        if name.endswith("multi_modal_projector"):
            _wrap_module(mod, fsdp_kwargs, seen)
            wrapped_any = True

    # Fallback: if we didn't find any layers, shard the root.
    if not wrapped_any:
        _wrap_module(model, fsdp_kwargs, seen)


def _safe_save(path: Path, payload: dict, rank: int) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp.{rank}"
    torch.save(payload, tmp)
    try:
        os.replace(tmp, path)
    except FileExistsError:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        if tmp.exists():
            tmp.unlink()
        raise


def _cache_path(cache_dir: Path, namespace: str, prompt: str) -> Path:
    h = hashlib.sha1(f"{namespace}\n{prompt}".encode("utf-8")).hexdigest()
    return cache_dir / h[:2] / f"{h}.pt"


def main() -> None:
    init_logger()
    config = ConfigManager().parse_args()

    world_size, rank, local_rank = _get_rank_info()
    use_distributed = world_size > 1

    if use_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    encoder_cfg = config.encoder
    cache_dir = encoder_cfg.text_encoder_cache_dir
    if not cache_dir:
        raise ValueError("encoder.text_encoder_cache_dir must be set")

    cache_dir = Path(cache_dir)
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)

    namespace = encoder_cfg.model_name.lower()
    if rank == 0:
        logger.info("Precomputing text cache for %s into %s", namespace, cache_dir)

    use_cuda = str(encoder_cfg.text_encoder_device).startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    use_fsdp = use_cuda and use_distributed

    use_sharded_load = bool(getattr(encoder_cfg, "text_encoder_fsdp_sharded_load", True))
    if use_fsdp and use_sharded_load and namespace == "flux.2-dev":
        # Build a meta-init Mistral and load weights after fully_shard to avoid GPU OOM.
        text_encoder = Mistral3SmallEmbedder.init_empty()
        cpu_state = load_mistral_small_state_dict() if rank == 0 else {}
    elif use_fsdp:
        # Load on CPU first to avoid single-GPU OOM, then shard with fully_shard.
        text_encoder = load_text_encoder(namespace, device="cpu")
        cpu_state = None
    else:
        text_encoder = load_text_encoder(namespace, device=encoder_cfg.text_encoder_device)
        cpu_state = None

    text_encoder = text_encoder.eval().requires_grad_(False)

    fsdp_offload = bool(getattr(encoder_cfg, "text_encoder_fsdp_offload", False))
    if use_fsdp and use_sharded_load and fsdp_offload:
        logger.warning("text_encoder_fsdp_offload is incompatible with sharded_load; disabling offload.")
        fsdp_offload = False
    offload_policy = CPUOffloadPolicy() if (fsdp_offload and CPUOffloadPolicy is not None) else None

    if use_fsdp:
        fsdp_kwargs = _build_fully_shard_kwargs(world_size, device, offload_policy)
        logger.info('Using %s with args: %s', _FULLY_SHARD_SRC, sorted(fsdp_kwargs.keys()))
        _apply_fsdp_sharding(text_encoder.model, fsdp_kwargs)
        if cpu_state is not None:
            options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=True)
            missing, unexpected = set_model_state_dict(
                text_encoder.model,
                model_state_dict=cpu_state,
                options=options,
            )
            if missing or unexpected:
                logger.warning("State dict load issues. Missing=%s Unexpected=%s", missing, unexpected)

    dataloader = Flux2DataLoader(
        config.dataloader,
        dp_world_size=world_size,
        dp_rank=rank,
        local_batch_size=1,
    )

    max_steps = config.training.steps
    step = 0

    for input_dict, _labels in dataloader:
        prompts = input_dict.get("prompt")
        if prompts is None:
            continue
        if isinstance(prompts, str):
            prompts = [prompts]

        missing = []
        for prompt in prompts:
            if not _cache_path(cache_dir, namespace, prompt).exists():
                missing.append(prompt)

        if missing:
            with torch.no_grad():
                enc = text_encoder(missing).detach().cpu()
                tokens = None
                if hasattr(text_encoder, "tokenize"):
                    tokens = text_encoder.tokenize(missing)

            for j, prompt in enumerate(missing):
                path = _cache_path(cache_dir, namespace, prompt)
                payload = {
                    "prompt": prompt,
                    "namespace": namespace,
                    "encoding": enc[j],
                }
                if tokens is not None:
                    payload["tokens"] = {k: v[j : j + 1].cpu() for k, v in tokens.items()}
                _safe_save(path, payload, rank)

        step += 1
        if max_steps is not None and step >= max_steps:
            break

    if rank == 0:
        logger.info("Precompute finished. Steps processed: %s", step)

    if use_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    if record is not None:
        record(main)()
    else:
        main()
