# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The TorchTitan model an inference engine runs: how the generator builds it, loads its
initial weights, and loads model state dicts into it. Shared by every engine's model
wrapper; the wrappers replace the attention layers before the build."""

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._state_dict_utils import _create_cpu_state_dict

# Registers the "loss" local-compile region. The generator shares the trainer's
# model config, whose local_compile.regions may name it, but never builds a loss.
from torchtitan.components import loss as _loss  # noqa: F401
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.decoder import Decoder
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor
from torchtitan.rl.generator.engine import ModelStateDictLoadMode
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.tools.utils import set_default_dtype


def get_attention_dimensions(
    attention_config, model_dim: int
) -> tuple[int, int, int, int]:
    """Return query heads, KV heads, Q/K head dim, and value head dim."""
    num_heads = attention_config.n_heads
    num_kv_heads = getattr(attention_config, "n_kv_heads", None) or num_heads
    if hasattr(attention_config, "qk_nope_head_dim"):
        head_dim = attention_config.qk_nope_head_dim + attention_config.qk_rope_head_dim
        value_head_dim = attention_config.v_head_dim
    else:
        head_dim = getattr(attention_config, "head_dim", None)
        head_dim = head_dim if head_dim is not None else model_dim // num_heads
        value_head_dim = head_dim
    return num_heads, num_kv_heads, head_dim, value_head_dim


def build_inference_model(
    model_config: Decoder.Config,
    *,
    parallelism: InferenceParallelismConfig,
    precision: PrecisionConfig,
    device: torch.device,
) -> tuple[Decoder, ParallelismContext]:
    """Build, parallelize and materialize the generator's model, without its weights.

    The model computes in ``precision.param_dtype``, holds ``preserve_dtype_params`` in
    the trainer's master dtype, skips its lm_head in forward (the engine computes logits
    separately), and keeps every FSDP group's unsharded storage at a fixed address
    across model state dict loads, for CUDA graphs.

    Args:
        model_config: The generator's model config, with the engine's attention layers.
        parallelism: The generator's parallelism.
        precision: The trainer's dtypes, which the model mirrors.
        device: The device to materialize the model on.

    Returns:
        ``(model, parallelism_context)``, the model's buffers initialized.
    """
    # Translate the inference parallelism into torchtitan's full
    # ParallelismConfig that ParallelismContext and model.parallelize consume.
    training_parallelism = parallelism.to_training()

    # Build ParallelismContext from the translated ParallelismConfig so TP/EP
    # sharding sees the same mesh shape as the engine. The DP axis is represented
    # as generator FSDP.
    parallelism_context = ParallelismContext(
        dp_replicate=training_parallelism.data_parallel_replicate_degree,
        dp_shard=training_parallelism.data_parallel_shard_degree,
        cp=training_parallelism.context_parallel_degree,
        tp=training_parallelism.tensor_parallel_degree,
        pp=training_parallelism.pipeline_parallel_degree,
        ep=training_parallelism.expert_parallel_degree,
        world_size=dist.get_world_size(),
        enable_sequence_parallel=training_parallelism.enable_sequence_parallel,
    )

    param_dtype = TORCH_DTYPE_MAP[precision.param_dtype]
    # Build model on meta device to avoid allocating full model on every GPU.
    # Build in the trainer's master dtype (`precision.dtype`), so buffers and
    # `preserve_dtype_params` get the dtypes the trainer has and sends.
    with (
        parallelism_context.activate_spmd(),
        torch.device("meta"),
        set_default_dtype(TORCH_DTYPE_MAP[precision.dtype]),
    ):
        model = model_config.build()
    # The generator never computes gradients. Frozen parameters also let FSDP
    # hold mixed dtypes, and Dist-MoE's inference context rejects any operand
    # with requires_grad.
    model.requires_grad_(False)
    # Hold the other parameters in `precision.param_dtype`, the dtype the trainer
    # computes with and sends, so the FSDP below casts none of them. Swap each
    # parameter's tensor rather than assigning `param.data`, which keeps a
    # quantized tensor subclass's inner tensor in the old dtype; keep the
    # parameter's attributes (e.g. its SPMD type).
    preserved = {param for _, param in model.named_preserve_dtype_params()}
    for param in model.parameters():
        if param in preserved or param.dtype == param_dtype:
            continue
        new_param = nn.Parameter(param.detach().to(param_dtype), requires_grad=False)
        for name, value in param.__dict__.items():
            new_param.__dict__.setdefault(name, value)
        torch.utils.swap_tensors(param, new_param)
    model._skip_lm_head = True
    if getattr(model, "vision_encoder", None) is not None:
        model.vision_encoder = None

    model = model.parallelize(
        parallelism_context=parallelism_context,
        precision=precision,
        parallelism=training_parallelism,
        local_compile=model_config.local_compile,
        activation_remat=None,
        dump_folder="",
    )

    # Preserve compute storage addresses across model state dict loads for CUDA graphs
    assert isinstance(model, FSDPModule)
    model.set_keep_unsharded_storage(True)

    # Materialize model on GPU — only allocates local shards (not full
    # model) thanks to EP/TP DTensor sharding applied above.
    model.to_empty(device=device)
    # HF checkpoints do not necessarily contain every TorchTitan buffer
    # (for example MoE expert_bias_E).
    # TODO: When checkpoint doesn't contains expert_bias_E, check the config
    # should use loss based load balancing strategy.
    with torch.no_grad():
        # spmd_types parameter init needs the current mesh to materialize
        # local shards for fused parameters, including the fused QKV linear
        # used by model variants such as Qwen3.
        # TODO: Consider an init_non_persistent_buffers contract on the
        # Decoder / Model class so buffer-only init does not need this
        # spmd context.
        with parallelism_context.activate_spmd():
            model.init_weights(buffer_device=None)
    return model, parallelism_context


def load_initial_weights(
    model: Decoder,
    model_config: Decoder.Config,
    checkpointer_config: CheckpointManager.Config | None,
) -> None:
    """Load initial HF weights via CheckpointManager.

    A configured manager loads initial weights. ``None`` skips loading for
    the RL loop, where weights arrive via TorchStore.
    """
    if checkpointer_config is None:
        return

    adapter_cls = type(model).state_dict_adapter_cls
    sd_adapter = None
    if adapter_cls is not None:
        sd_adapter = adapter_cls(
            model_config=model_config,
            hf_assets_path=checkpointer_config.initial_load_path,
        )

    # Model-only CheckpointManager: initial_load_model_only=True (default)
    # ensures only MODEL state is loaded, so None optimizer/lr_scheduler
    # are never accessed.
    checkpointer = checkpointer_config.build(
        model_parts=[model],
        optimizers=None,
        lr_schedulers=None,
        ema=None,
        extra_states={},
        sd_adapter=sd_adapter,
    )
    checkpointer.load()
    # Free the large transient allocations the HF load/from_hf conversion left in the
    # caching allocator, so the later CUDA-graph capture (which needs its own private
    # pool) has room. Without this, large models (e.g. 235B) OOM capture even though
    # the live weights fit.
    torch.cuda.empty_cache()


class ModelStateDictLoader:
    """Loads model state dicts into a generator model built by ``build_inference_model``.

    Build it once the model holds its initial weights and is still sharded: it groups
    the state dict by FSDP group, allocates the pinned CPU buffers of a
    ``"cpu_prefetch"`` load, and then unshards every group, so the engine profiles GPU
    memory with the representation the forward uses. Bind the process to its GPU's NUMA
    node first, so first-touch places the pinned buffers on that node.

    Args:
        model: The generator's model, every FSDP group sharded.
        model_state_dict_load_mode: ``"cpu_prefetch"`` allocates the pinned CPU buffers
            of ``prefetched_model_state_dict``; ``"gpu_fetch"`` does not.
    """

    def __init__(
        self, model: nn.Module, *, model_state_dict_load_mode: ModelStateDictLoadMode
    ) -> None:
        self._model = model
        self._fsdp_load_groups, self._buffer_keys = _fsdp_load_groups(model)
        if model_state_dict_load_mode == "gpu_fetch":
            # gpu_fetch writes into the sharded parameter itself, not a quantized
            # parameter's high-precision _tensor.
            for _, keyed_params in self._fsdp_load_groups:
                for key, fsdp_param in keyed_params:
                    if isinstance(fsdp_param._sharded_param_data, _ShardedFSDPTensor):
                        raise ValueError(
                            "model_state_dict_load_mode='gpu_fetch' does not support "
                            f"quantized parameters ({key})"
                        )

        self.prefetched_model_state_dict: dict[str, torch.Tensor] | None = None
        """The pinned CPU buffers a model state dict is prefetched into, and loaded from
        by ``load``. None with ``model_state_dict_load_mode="gpu_fetch"``."""
        if model_state_dict_load_mode == "cpu_prefetch":
            # Built while the model is still sharded. The sharded state dict has the
            # dtypes the trainer sends (see `Trainer._put_model_state_dict`);
            # torchstore's RDMA transport copies bytes and asserts matching dtypes.
            with torch.device("cpu"):
                self.prefetched_model_state_dict = _create_cpu_state_dict(
                    model.state_dict(), pin_memory=True
                )

        for module in model.modules():
            if isinstance(module, FSDPModule):
                _unshard_and_free_sharded_params(module)

    def load(self, fetch: Callable[[dict[str, torch.Tensor]], Iterable[str]]) -> None:
        """Fetch new weights into the model, allocating one FSDP group's sharded buffers at a time.

        Each group is allocated, filled by `fetch`, then unsharded and freed:

            FSDP group:     [emb] | [l0] | ... | [lN] | [norm, lm_head] | buffers
            allocate:       emb   | l0   |     | lN   | norm, lm_head   |
            fetch:          emb   | l0   |     | lN   | norm, lm_head   | buffers
            unshard, free:  emb   | l0   |     | lN   | norm, lm_head   |

        With tied embeddings the group is [tok_embeddings, norm, lm_head].

        Why this shape:
          - Memory: allocating every sharded buffer first adds a second copy of the weights at
            FSDP degree 1: 64.6 GiB extra for bf16 Qwen3.5-35B-A3B at TP1, vs 1.6 GiB here.
          - Per group: modules from one `fully_shard` call share one FSDP state, so unsharding
            after `norm` would all-gather `lm_head` before it loads.

        Args:
            fetch: Fills each tensor of the given subset of the model's state dict
                (each in its own dtype, as the trainer sends it) in place with the
                new weights and returns the keys it filled, e.g. a
                copy from the generator's pinned CPU buffers or a TorchStore read. Each
                group's sharded storage is reallocated just before its fetch, so a
                fetch writing outside the current stream (e.g. RDMA) must first wait
                for queued device work, and must finish writing before it returns.
        """
        error = None
        for module, keyed_params in self._fsdp_load_groups:
            # unshard() is a no-op while the group is unsharded. Resharding registers
            # the sharded parameters and keeps the unsharded storage CUDA graphs read.
            _reshard(module)
            _restore_sharded_params(module)
            # After a failed fetch, keep unsharding the remaining groups: unshard()
            # all-gathers across the FSDP group and the other ranks are still loading.
            if error is None:
                try:
                    # The trainer sends each parameter in the dtype the generator
                    # holds it in, so fetch fills the sharded parameters in place.
                    params = {
                        key: fsdp_param.sharded_param
                        for key, fsdp_param in keyed_params
                    }
                    # An unfilled parameter would load its just-reallocated,
                    # uninitialized storage.
                    missing = params.keys() - fetch(params)
                    if missing:
                        raise RuntimeError(
                            f"fetch did not fill {len(missing)} weights, e.g. "
                            f"{sorted(missing)[:5]}"
                        )
                except Exception as e:
                    error = e
            _unshard_and_free_sharded_params(module)
        if error is not None:
            raise error
        # Buffers load after every FSDP collective, so a failure here cannot hang
        # peers. A fetch that skips unpublished keys (gpu_fetch) leaves them as is.
        if self._buffer_keys:
            fetch({key: self._model.get_buffer(key) for key in self._buffer_keys})


def _sharded_param_storages(module: FSDPModule) -> Iterator[torch.Tensor]:
    """Yield the sharded parameter tensors of ``module``'s FSDP group."""
    for param_group in module._get_fsdp_state()._fsdp_param_groups:
        for param in param_group.fsdp_params:
            sharded_data = param._sharded_param_data
            if isinstance(sharded_data, _ShardedFSDPTensor):
                sharded_data = sharded_data._tensor
            yield sharded_data


def _reshard(module: FSDPModule) -> None:
    """Reshard ``module``'s FSDP group unconditionally, like FSDP's load_state_dict pre-hook.

    ``module.reshard()`` is a no-op for a group left in the FORWARD state with
    reshard_after_forward=False, e.g. the tied [tok_embeddings, norm, lm_head]
    group after the engine computes logits with lm_head alone.
    """
    # TODO: replace with a public FSDP API that reshards regardless of training state.
    for param_group in module._get_fsdp_state()._fsdp_param_groups:
        param_group._to_sharded()


def _restore_sharded_params(module: FSDPModule) -> None:
    """Re-allocate the sharded buffers of `module`'s FSDP group so new weights can be copied in.

    before:  sharded: freed                     unsharded: old weights (what CUDA graphs read)
    after:   sharded: allocated, uninitialized  unsharded: unchanged

    FSDP's own load pre-hook may have resharded the group already (e.g.
    tok_embeddings), so ``is_sharded`` can be True while the storage is still
    freed; allocate unconditionally.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_param import alloc_storage

    # TODO: replace this with FSDPModule._restore_sharded_params()
    for sharded_data in _sharded_param_storages(module):
        alloc_storage(sharded_data)


def _unshard_and_free_sharded_params(module: FSDPModule) -> None:
    """Unshard `module`'s FSDP group into its unsharded buffers, then free its sharded ones.

        before:  sharded: new weights   unsharded: old weights
        after:   sharded: freed         unsharded: new weights (what CUDA graphs read)

    The unshard is explicit because CUDA graph replays skip FSDP's forward pre-hook.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_param import free_storage

    with torch.inference_mode():
        module.unshard()
        # TODO: replace this with FSDPModule._free_sharded_params()
        for sharded_data in _sharded_param_storages(module):
            free_storage(sharded_data)


def _fsdp_load_groups(
    model: nn.Module,
) -> tuple[list[tuple[FSDPModule, list[tuple[str, Any]]]], list[str]]:
    """Group ``model.state_dict()`` keys by the FSDP group whose sharded parameters hold them.

    Must run while every group is sharded, so the state dict holds the sharded
    parameters. Tied parameters list each of their keys under the same group.

    Returns:
        ``(groups, buffer_keys)``: one ``(module, [(key, fsdp_param), ...])`` per FSDP
        group in module order, where ``module`` is the group's first module; and the
        keys of the buffers.
    """
    sharded_param_to_group: dict[int, tuple[FSDPModule, Any]] = {}
    group_modules: dict[Any, FSDPModule] = {}
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        state = module._get_fsdp_state()
        if state in group_modules:
            continue
        group_modules[state] = module
        for param_group in state._fsdp_param_groups:
            for fsdp_param in param_group.fsdp_params:
                sharded_param_to_group[id(fsdp_param.sharded_param)] = (
                    module,
                    fsdp_param,
                )

    keys_by_module: dict[FSDPModule, list[tuple[str, Any]]] = {
        module: [] for module in group_modules.values()
    }
    buffers = {id(buffer) for buffer in model.buffers()}
    buffer_keys = []
    for key, value in model.state_dict(keep_vars=True).items():
        group = sharded_param_to_group.get(id(value))
        if group is None:
            # A parameter here would be loaded outside the reshard/unshard sequence.
            assert id(value) in buffers, f"{key} is neither FSDP-sharded nor a buffer"
            buffer_keys.append(key)
            continue
        module, fsdp_param = group
        keys_by_module[module].append((key, fsdp_param))
    groups = [(module, keyed) for module, keyed in keys_by_module.items() if keyed]
    return groups, buffer_keys
