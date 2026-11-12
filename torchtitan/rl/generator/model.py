# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The TorchTitan model an inference engine runs: how the generator builds it, loads its
initial weights, and loads model state dicts into it. Shared by every engine's model
wrapper; the wrappers replace the attention layers before the build."""

import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import spmd_types as spmd
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
from torchtitan.distributed.fsdp import maybe_store_frozen_params_in_param_dtype_
from torchtitan.distributed.fsdp_lifecycle import complete_fsdp_grouped_forward
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts, DistMoeRuntime
from torchtitan.models.common.moe.token_dispatcher import (
    BaseEPTokenDispatcher,
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
)
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor
from torchtitan.rl.generator.engine import ModelStateDictLoadMode
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.tools.utils import set_default_dtype

logger = logging.getLogger(__name__)


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
    keep_vision_encoder: bool = False,
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
        keep_vision_encoder: Keep the model's vision encoder, for an engine that serves
            image prompts; otherwise it is dropped.

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
    # Store the parameters as the trainer stores its frozen ones. The trainer
    # sends weights in `precision.param_dtype`, so the FSDP below casts none of them.
    maybe_store_frozen_params_in_param_dtype_(model, precision)
    model._skip_lm_head = True
    if not keep_vision_encoder and getattr(model, "vision_encoder", None) is not None:
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


def compute_logits(
    model: Decoder,
    hidden_states: torch.Tensor,
    parallelism_context: ParallelismContext,
) -> torch.Tensor:
    """Full-vocabulary logits of ``hidden_states`` on every rank: the model's lm_head,
    gathered over TP (lm_head returns vocab-sharded logits under TP)."""
    with parallelism_context.activate_spmd():
        logits = model.lm_head(hidden_states)
        if parallelism_context.tp_enabled:
            mesh = current_spmd_mesh()
            assert mesh is not None
            logits = spmd.redistribute(
                logits,
                mesh.get_group("tp"),
                src=spmd.S(-1),
                dst=spmd.R,
                backward_options={"op_dtype": logits.dtype},
            )
    return logits


# The engine TP all-reduce that route_tp_all_reduce's op swaps call. Process-global:
# the swaps are installed once, and a later call (e.g. a rebuilt engine) replaces it.
_tp_all_reduce: Callable[[torch.Tensor], torch.Tensor] | None = None


def route_tp_all_reduce(all_reduce: Callable[[torch.Tensor], torch.Tensor]) -> None:
    """Route the model's tensor-parallel sum all-reduces through an engine's TP
    all-reduce (e.g. its custom one-shot all-reduce) instead of NCCL's ring, for
    the steps outside batch-invariant mode.

    ``all_reduce`` sums a tensor over the engine's TP group: the same ranks as the
    model's TP mesh axis, in the group that owns the engine's communicator buffers.
    Every routed sum is assumed to be over those ranks; TP is the only axis the
    generator's forward reduces over. ``all_reduce`` must be capturable in CUDA
    graphs over the expandable_segments (VMM) memory the RL stack enables. It may
    reduce in place and return its input (SGLang's NCCL fallback does): a Partial
    input is dead once reduced.

    Two entry points are swapped:

    1. ``torch.ops._c10d_functional.all_reduce``, which every DTensor Partial ->
       Replicate redistribute calls. DTensor wraps the synchronous result in an
       AsyncCollectiveTensor whose wait_tensor is a no-op. Only sum reductions are
       routed; others fall through to the original op.
    2. ``spmd.redistribute`` from Partial to Replicate or Invariant, without grad.
       Its all-reduce reaches the first swap too (through funcol), but calling
       ``all_reduce`` directly skips spmd's autograd Function and funcol's
       wrapping: about 2-3% of a vLLM decode step at TP 2 (Qwen3-0.6B, GB300).
       Under grad, spmd.redistribute keeps its dst-dependent backward.

    TODO: this is a stopgap to close the generator's TP all-reduce perf gap.
    Improve our native (DTensor) all-reduce path and remove this patch.
    """
    global _tp_all_reduce
    installed = _tp_all_reduce is not None
    _tp_all_reduce = all_reduce
    if installed:
        return

    c10d = torch.ops._c10d_functional
    # Op schema: all_reduce(Tensor input, str reduce_op, Any group_name) -> Tensor.
    original_all_reduce = c10d.all_reduce

    def routed_all_reduce(input, reduce_op, group_name):
        if reduce_op == "sum":
            return _tp_all_reduce(input)
        return original_all_reduce(input, reduce_op, group_name)

    c10d.all_reduce = routed_all_reduce

    original_redistribute = spmd.redistribute

    def redistribute(x, group, *, src, dst, **kwargs):
        if src == spmd.P and dst in (spmd.R, spmd.I) and not torch.is_grad_enabled():
            return _tp_all_reduce(x)
        return original_redistribute(x, group, src=src, dst=dst, **kwargs)

    spmd.redistribute = redistribute
    logger.info("Routed the model's TP sum all-reduces through the engine's all-reduce")


def build_dist_moe_runtime(
    model: Decoder,
    runtime_config: DistMoeRuntime.Config | None,
    *,
    parallelism_context: ParallelismContext,
    device: torch.device,
    max_num_tokens_per_step: int,
) -> DistMoeRuntime | None:
    """Build the runtime the model's Dist-MoE experts run with, or None without them.

    Dist-MoE experts refuse to run without a runtime. The trainer gets one from
    ``TrainingEngine``; the generator builds its own, once, before any CUDA-graph
    capture, so no per-forward Python work runs under capture or replay.

    Args:
        max_num_tokens_per_step: The most tokens a forward passes to the model on this
            rank, which sizes the runtime's buffers.
    """
    if not any(isinstance(module, DistMoeRoutedExperts) for module in model.modules()):
        if runtime_config is not None:
            logger.warning(
                "Ignoring dist_moe_runtime: the generator model has no "
                "Dist-MoE routed experts."
            )
        return None
    if runtime_config is None:
        raise ValueError(
            "The generator model uses Dist-MoE routed experts but no "
            "dist_moe_runtime was set. Set Generator.Config.dist_moe_runtime to a "
            "DistMoeRuntime.Config(inference=True)."
        )
    if not runtime_config.inference:
        raise ValueError(
            "dist_moe_runtime must set inference=True; got "
            f"{runtime_config!r}. The generator runs no "
            "backward, so a training context's activation slots would reserve "
            "memory it can never use."
        )
    return runtime_config.build(
        model_parts=[model],
        parallelism_context=parallelism_context,
        device=device,
        num_tokens_per_microbatch_per_dp_rank=max_num_tokens_per_step,
        pp_schedule=None,
    )


def validate_ep_token_dispatchers(
    model: Decoder.Config,
    *,
    parallelism: InferenceParallelismConfig,
    cuda_graph: bool,
    deterministic: bool,
) -> None:
    """Raise for EP token dispatchers a generator engine cannot run.

    Args:
        cuda_graph: Whether the engine captures CUDA graphs.
        deterministic: ``DebugConfig.deterministic``, under which DeepEP builds its
            deterministic communication buffer.
    """
    if parallelism.expert_parallel_degree <= 1:
        return
    # Dist-MoE experts have no token dispatcher, so this sees only the stock ones.
    for fqn, dispatcher, _, _ in model.traverse(BaseEPTokenDispatcher.Config):
        if isinstance(dispatcher, DeepEPTokenDispatcher.Config):
            syncs_host = not dispatcher.cuda_graph_compatible
            # TODO: DeepEP's deterministic dispatch sorts the received tokens with
            # data-dependent shapes (ElasticBuffer's deterministic_sort), which CUDA
            # graphs cannot capture. https://github.com/deepseek-ai/DeepEP/issues/820
            if deterministic and cuda_graph:
                raise ValueError(
                    f"{fqn}: DeepEP's deterministic dispatch "
                    "(DebugConfig.deterministic) cannot run in CUDA graphs: turn off "
                    "the engine's CUDA graphs"
                )
        elif isinstance(dispatcher, HybridEPTokenDispatcher.Config):
            syncs_host = dispatcher.non_blocking_capacity_factor is None
            # Below 1.0, HybridEP drops the tokens past its capacity without an error,
            # which silently changes the sampled tokens.
            if (
                dispatcher.non_blocking_capacity_factor is not None
                and dispatcher.non_blocking_capacity_factor < 1.0
            ):
                raise ValueError(
                    f"{fqn}: a HybridEP non_blocking_capacity_factor below 1.0 drops "
                    "tokens in the generator; set it to 1.0, or to None with the "
                    "engine's CUDA graphs off"
                )
            # A generator step can have any token count, while HybridEP's TMA loads
            # need a rank's tokens times the ranks in its node, in 4-byte words, to
            # fill 16-byte rows: any count does when the ranks are a multiple of 4.
            if parallelism.expert_parallel_degree % 4 != 0:
                raise ValueError(
                    f"{fqn}: HybridEP needs an expert_parallel_degree divisible by 4 "
                    "in the generator, got "
                    f"{parallelism.expert_parallel_degree}"
                )
        else:
            # The all-to-all dispatchers.
            syncs_host = True
        if syncs_host and cuda_graph:
            raise ValueError(
                f"{fqn}: {type(dispatcher).__qualname__} reads its split sizes on the "
                "host, which CUDA graphs cannot capture: turn off the engine's CUDA "
                "graphs, or use Dist-MoE, DeepEP with cuda_graph_compatible, or "
                "HybridEP with non_blocking_capacity_factor"
            )


def validate_ep_token_dispatcher_capacity(
    model: Decoder.Config,
    *,
    parallelism: InferenceParallelismConfig,
    max_num_tokens_per_step: int,
) -> None:
    """Raise if a DeepEP or HybridEP buffer cannot hold an EP rank's tokens in a step.

    Args:
        max_num_tokens_per_step: The most tokens a step passes to the model on this
            rank, padding included.
    """
    if parallelism.expert_parallel_degree <= 1:
        return
    # The MoE splits each step's tokens evenly over the TP ranks.
    max_num_tokens_per_rank = (
        max_num_tokens_per_step // parallelism.expert_sequence_parallel_size
    )
    for fqn, dispatcher, _, _ in model.traverse(BaseEPTokenDispatcher.Config):
        if (
            isinstance(
                dispatcher,
                (DeepEPTokenDispatcher.Config, HybridEPTokenDispatcher.Config),
            )
            and dispatcher.num_max_tokens_per_rank is not None
            and dispatcher.num_max_tokens_per_rank < max_num_tokens_per_rank
        ):
            raise ValueError(
                f"{fqn}: num_max_tokens_per_rank={dispatcher.num_max_tokens_per_rank} "
                f"is below the {max_num_tokens_per_rank} tokens an EP rank can get in "
                "one engine step"
            )


def _load_initial_weights(
    model: Decoder, checkpointer_config: CheckpointManager.Config
) -> None:
    """Load the initial weights of ``model`` with a model-only ``CheckpointManager``."""
    adapter_cls = type(model).state_dict_adapter_cls
    sd_adapter = None
    if adapter_cls is not None:
        sd_adapter = adapter_cls(
            model_config=model.config,
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

    Build it once the model is built, every FSDP group still sharded: it loads the
    initial weights, groups the state dict by FSDP group, allocates the pinned CPU
    buffers of a ``"cpu_prefetch"`` load, and then unshards every group, so the engine
    profiles GPU memory with the representation the forward uses. Bind the process to
    its GPU's NUMA node first, so first-touch places the pinned buffers on that node.

    Args:
        model: The generator's model, every FSDP group sharded.
        checkpointer_config: Loads the initial weights; None skips it, as in the RL
            loop, where the weights arrive by model state dict loads.
        model_state_dict_load_mode: ``"cpu_prefetch"`` allocates the pinned CPU buffers
            of ``prefetched_model_state_dict``; ``"gpu_fetch"`` does not.
    """

    def __init__(
        self,
        model: Decoder,
        *,
        checkpointer_config: CheckpointManager.Config | None,
        model_state_dict_load_mode: ModelStateDictLoadMode,
    ) -> None:
        if checkpointer_config is not None:
            _load_initial_weights(model, checkpointer_config)
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
    """Reshard ``module``'s FSDP group, first finishing a forward it was left in.

    The engine computes logits with lm_head alone, so the [norm, lm_head] group (or the
    tied [tok_embeddings, norm, lm_head] one) stays in the FORWARD state: norm runs in
    CUDA graph replays, without FSDP hooks. In that state:
    - ``module.reshard()`` is a no-op (reshard_after_forward=False);
    - ``module.unshard()`` copies the shards in and all-gathers on FSDP's prefetch
      streams, which wait for the current stream only in the root's pre-forward, so
      the load's all-gather would not wait for device work the load queued on the
      current stream to fill the group's sharded buffers.
    """
    complete_fsdp_grouped_forward(module, None)
    module.reshard()


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
