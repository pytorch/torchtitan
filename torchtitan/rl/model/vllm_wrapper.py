# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Base wrapper for TorchTitan models to work with vLLM V1 engine.

This module provides TorchTitanVLLMModel: Core model class that adapts
TorchTitan models for vLLM.
"""

import copy
import dataclasses

import spmd_types as spmd

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._state_dict_utils import _create_cpu_state_dict
from torch.distributed.tensor import DTensor, Replicate
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import apply_overrides, OverrideConfig, TrainingConfig
from torchtitan.distributed import maybe_apply_numa_binding
from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import current_spmd_mesh
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.protocols.module import Module
from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.utils import torch_utils as _torch_utils


logger = init_logger(__name__)


def _replace_vllm_layer_configs(model_config):
    """Replace inner-computation configs with vLLM generation variants."""
    # These modules inspect the breakable-CUDA graph environment at import time.
    # Defer imports until vLLM constructs the model, after the generator has set
    # that environment. Import the GDN adapter only for hybrid models so other
    # models do not acquire its vLLM-specific dependencies.
    from torchtitan.rl.model.attention import (
        get_attention_dimensions,
        VLLMAttentionWrapper,
    )

    new_layers = []
    for layer_idx, layer_cfg in enumerate(model_config.layers):
        new_layer_cfg = layer_cfg

        attention_cfg = getattr(layer_cfg, "attention", None)
        if attention_cfg is not None:
            attention_metadata_key = attention_cfg.inner_attention._owner
            assert attention_metadata_key is not None and issubclass(
                attention_metadata_key, InnerAttention
            )
            (
                num_heads,
                num_kv_heads,
                head_dim,
                value_head_dim,
            ) = get_attention_dimensions(attention_cfg, model_config.dim)
            vllm_attention_cfg = VLLMAttentionWrapper.Config(
                attention_metadata_key=attention_metadata_key,
                hidden_size=model_config.dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                value_head_dim=value_head_dim,
                sliding_window_size=getattr(attention_cfg, "sliding_window_size", None),
                sharding_config=attention_cfg.inner_attention.sharding_config,
            )
            new_layer_cfg = dataclasses.replace(
                new_layer_cfg,
                attention=dataclasses.replace(
                    attention_cfg,
                    inner_attention=vllm_attention_cfg,
                ),
            )

        delta_net_cfg = getattr(layer_cfg, "delta_net", None)
        if delta_net_cfg is not None:
            from torchtitan.rl.model.gdn import VLLMInnerGatedDeltaNet

            vllm_inner_gdn_cfg = VLLMInnerGatedDeltaNet.Config(
                layer_idx=layer_idx,
                num_k_heads=(
                    delta_net_cfg.in_proj_q.out_features // delta_net_cfg.key_head_dim
                ),
                num_v_heads=(
                    delta_net_cfg.in_proj_v.out_features // delta_net_cfg.value_head_dim
                ),
                head_k_dim=delta_net_cfg.key_head_dim,
                head_v_dim=delta_net_cfg.value_head_dim,
                conv_kernel_size=delta_net_cfg.conv_kernel_size,
                sharding_config=delta_net_cfg.inner_gated_delta_net.sharding_config,
            )
            new_layer_cfg = dataclasses.replace(
                new_layer_cfg,
                delta_net=dataclasses.replace(
                    delta_net_cfg,
                    inner_gated_delta_net=vllm_inner_gdn_cfg,
                ),
            )

        kda_cfg = getattr(layer_cfg, "delta_attention", None)
        if kda_cfg is not None:
            from torchtitan.rl.model.kda import VLLMInnerKDA

            vllm_inner_kda_cfg = VLLMInnerKDA.Config(
                num_heads=kda_cfg.num_heads,
                head_dim=kda_cfg.head_dim,
                conv_kernel_size=kda_cfg.conv_kernel_size,
                lower_bound=kda_cfg.inner_kda.kernel.lower_bound,
                layer_index=layer_idx,
                sharding_config=kda_cfg.inner_kda.sharding_config,
            )
            new_layer_cfg = dataclasses.replace(
                new_layer_cfg,
                delta_attention=dataclasses.replace(
                    kda_cfg,
                    inner_kda=vllm_inner_kda_cfg,
                ),
            )

        new_layers.append(new_layer_cfg)

    return dataclasses.replace(model_config, layers=new_layers)


# NOTE: Monkeypatch vLLM's weak_ref_tensor to handle DTensor
# This is because piecewise CUDA-graph capture calls weak_ref_tensor()
# on every subgraphoutput (see vllm/compilation/cuda_graph.py).
# When TP is active some of those outputs are DTensors which fail with
# ("The specified pointer resides on host memory").  to_local
# converts the DTensor to a plain tensor. which succeeds with this
# CUDA graph implementation.
_original_weak_ref_tensor = _torch_utils.weak_ref_tensor


def _dtensor_safe_weak_ref_tensor(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor._local_tensor
    return _original_weak_ref_tensor(tensor)


_torch_utils.weak_ref_tensor = _dtensor_safe_weak_ref_tensor


# Process-global: install the op swap at most once even if the generator is
# re-initialized in the same process (re-wrapping would chain shims).
_tp_all_reduce_patched = False


def _patch_vllm_all_reduce() -> None:
    """Route the generator's tensor-parallel all-reduce through vLLM's custom
    one-shot/multimem AR instead of DTensor's NCCL ring redistribute (applied
    when batch-invariant mode is off). Idempotent. Two changes:

    1. Swap torch.ops._c10d_functional.all_reduce (the op every DTensor
       Partial -> Replicate redistribute calls) for
       tensor_model_parallel_all_reduce, which reuses vLLM's TP GroupCoordinator
       -- the same ranks, but the group that owns the custom-AR shared buffers.
       DTensor wraps the synchronous result in an AsyncCollectiveTensor whose
       wait_tensor is a no-op. Only sum reductions are routed (the custom AR is
       sum-only); others fall through to the original op. No-op at world_size 1.

    2. Force the custom AR onto its registered=False path so CUDA graph capture
       works. registered=True records graph buffers and calls cudaIpcGetMemHandle
       on them, which fails for the expandable_segments (VMM) memory the RL stack
       enables for Monarch RDMA. registered=False reduces via the init-time
       buffer_ptrs (raw cudaMalloc, IPC-able), records no graph buffers, at the
       cost of one staging copy per AR.

    3. (spmd_types) spmd.redistribute issues an in-place dist.all_reduce that the
       point-1 swap can't see, so route its Partial -> {R,I} reduce straight to
       the same custom AR. Guarded on no-grad: the generator is inference-only and
       the custom AR is sum-only/forward-only; under grad we keep spmd.redistribute
       (its dst-dependent backward is correct).

    TODO: this is a stopgap to close the generator's TP all-reduce perf gap.
    Improve our native (DTensor) all-reduce path and remove this patch.
    """
    global _tp_all_reduce_patched
    if _tp_all_reduce_patched:
        return

    from vllm.distributed import get_tp_group

    c10d = torch.ops._c10d_functional
    # Op schema: all_reduce(Tensor input, str reduce_op, Any group_name) -> Tensor.
    original_all_reduce = c10d.all_reduce

    def all_reduce(input, reduce_op, group_name):
        if reduce_op == "sum":
            # Use vLLM's helper rather than the incoming group_name: the custom-AR
            # kernel is bound to vLLM's TP GroupCoordinator and the shared buffers
            # it registered, so the reduction must run on vLLM's TP group, not
            # DTensor's TP mesh PG (a different PG over the same ranks -> rank-for-
            # rank equivalent). The helper resolves that group, guards
            # world_size==1, and on CUDA dispatches to torch.ops.vllm.all_reduce.
            return tensor_model_parallel_all_reduce(input)
        return original_all_reduce(input, reduce_op, group_name)

    c10d.all_reduce = all_reduce

    # Force vLLM's TP custom AR onto its registered=False path (see point 2).
    device_comm = get_tp_group().device_communicator
    ca = getattr(device_comm, "ca_comm", None) if device_comm is not None else None
    if ca is not None and not ca.disabled:

        def custom_all_reduce(input):
            # Mirrors CustomAllreduce.custom_all_reduce but always registered=False.
            if ca.disabled or not ca.should_custom_ar(input):
                return None
            return ca.all_reduce(input, registered=False)

        ca.custom_all_reduce = custom_all_reduce

    original_redistribute = spmd.redistribute

    def redistribute(x, group, *, src, dst, **kwargs):
        if src == spmd.P and dst in (spmd.R, spmd.I) and not torch.is_grad_enabled():
            return tensor_model_parallel_all_reduce(x)
        return original_redistribute(x, group, src=src, dst=dst, **kwargs)

    spmd.redistribute = redistribute

    _tp_all_reduce_patched = True
    logger.info(
        "vllm_allreduce: routed _c10d_functional.all_reduce (TP sum reductions) "
        "through vLLM custom all-reduce (registered=False, CUDA-graph-safe)"
    )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
    }
)
class VLLMModelWrapper(Module):
    """
    Generic vLLM-compatible model wrapper for TorchTitan models. Implemented
    required interface required by vLLM Engine.
    Doc: https://docs.vllm.ai/en/latest/contributing/model/basic/
    Reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py

    The wrapper handles:
    - Direct usage of TorchTitan model args (no HF config mapping needed)
    - Attention replacement with vLLM paged attention
    - Parallelism setup and DTensor conversion between torchtitan and vLLM
    - Weight loading from HF checkpoints
    - vLLM forward/compute_logits interface
    """

    is_text_generation_model = True  # Required for vLLM runner validation
    supports_pp = False  # Pipeline parallelism not supported yet
    supports_multimodal = False

    def __init__(
        self,
        *,
        model_config: Decoder.Config,
        parallelism: InferenceParallelismConfig,
        checkpointer_config: CheckpointManager.Config | None,
        vllm_config: VllmConfig,
        prefix: str = "",
        override: OverrideConfig,
        return_routed_experts: bool = False,
    ):
        super().__init__()

        assert vllm_config is not None, "vllm_config is required"

        # Translate the inference parallelism into torchtitan's full
        # ParallelismConfig that ParallelismContext and model.parallelize consume.
        training_parallelism = parallelism.to_training()
        model_config = copy.deepcopy(model_config)
        model_config.set_sharding_(training_parallelism)
        self.config = _replace_vllm_layer_configs(model_config)

        # Build ParallelismContext from the translated ParallelismConfig so TP/EP
        # sharding sees the same mesh shape as vLLM. The DP axis is represented
        # as generator FSDP.
        self.parallelism_context = ParallelismContext(
            dp_replicate=training_parallelism.data_parallel_replicate_degree,
            dp_shard=training_parallelism.data_parallel_shard_degree,
            cp=training_parallelism.context_parallel_degree,
            tp=training_parallelism.tensor_parallel_degree,
            pp=training_parallelism.pipeline_parallel_degree,
            ep=training_parallelism.expert_parallel_degree,
            world_size=dist.get_world_size(),
            enable_sequence_parallel=training_parallelism.enable_sequence_parallel,
        )

        if override.imports:
            apply_overrides(override, self.config)
        logger.debug(f"Creating model with config: {self.config.to_dict()}")

        # Build model on meta device to avoid allocating full model on every GPU
        with self.parallelism_context.activate_spmd(), torch.device("meta"):
            self.model = self.config.build()
        self.model._skip_lm_head = True
        if getattr(self.model, "vision_encoder", None) is not None:
            self.model.vision_encoder = None

        self.model = self.model.parallelize(
            parallelism_context=self.parallelism_context,
            training=TrainingConfig(),
            parallelism=training_parallelism,
            local_compile_regions=self.config.local_compile_regions,
            ac_config=None,
            dump_folder="",
        )

        # Preserve compute storage addresses across weight syncs for CUDA graphs
        assert isinstance(self.model, FSDPModule)
        self.model.set_keep_unsharded_storage(True)

        # Load initial weights based on checkpoint config.
        self._checkpointer_config = checkpointer_config

        # Materialize model on GPU — only allocates local shards (not full
        # model) thanks to EP/TP DTensor sharding applied above.
        self.model.to_empty(device=vllm_config.device_config.device)
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
            with self.parallelism_context.activate_spmd():
                self.model.init_weights(buffer_device=None)
        self._maybe_initial_load_weights()

        # Give each gpt-oss attention's vLLM backend its sink rescale.
        self._inject_attention_sinks()

        if return_routed_experts:
            self._expose_routed_experts_to_vllm()

        # Route the TP all-reduce through vLLM's custom AR (off under
        # batch-invariant mode, where its size-dependent algorithm breaks).
        if self.parallelism_context.tp_enabled and not is_in_batch_invariant_mode():
            _patch_vllm_all_reduce()

        # Build pinned CPU receive buffers for weight sync while the model
        # is still sharded. Bind first so first-touch places them on the
        # NUMA node local to this rank's GPU.
        maybe_apply_numa_binding(torch.cuda.current_device(), "cuda")
        with torch.device("cpu"):
            self._prefetched_model_state_dict = _create_cpu_state_dict(
                self.model.state_dict(), pin_memory=True
            )

        # Unshard the model here so vLLM performs its GPU memory profiling
        # based on the model's actual representation used during forward
        self.prepare_for_forward()

    def prepare_for_state_dict_load(self) -> None:
        """
        Re-allocate previously freed sharded buffers for receiving weights.

        Weight sync lifecycle:
          1. prepare_for_state_dict_load: re-allocate sharded buffers
          2. model.state_dict() hook: reshard model + keep unsharded buffers
          3. ts.get_state_dict: fetch new weights into sharded buffers
          4. prepare_for_forward: unshard model + free sharded buffers

        At the end of weight sync, only unsharded buffers are resident in
        memory. Sharded buffers are only re-allocated temporarily to receive
        updated weights during weight sync, and are promptly freed afterwards.

        Note: This currently incurs a memory spike every weight sync (step 2),
        since FSDP maintains separate buffers for sharded and unsharded weights,
        and both must be resident during the weight sync. E.g. for bf16 generator
        and fsdp=1, weight sync currently maintains 2x model memory. This may be
        fine since this is not peak memory, but we should revisit in the future.
        """
        from torch.distributed.fsdp._fully_shard._fsdp_param import alloc_storage

        for module in self.model.modules():
            if not isinstance(module, FSDPModule):
                continue
            # TODO: replace this with FSDPModule._restore_sharded_params()
            for param_group in module._get_fsdp_state()._fsdp_param_groups:
                if param_group.is_sharded:
                    continue
                for param in param_group.fsdp_params:
                    sharded_data = param._sharded_param_data
                    if isinstance(sharded_data, _ShardedFSDPTensor):
                        sharded_data = sharded_data._tensor
                    alloc_storage(sharded_data)

    def prepare_for_forward(self) -> None:
        """
        After weight sync, prepare the model for prefill/decode by:
          1. Explicitly unsharding model to refill existing unsharded operands
          2. Freeing sharded buffers since they are not needed during forward

        For (1), unshard must be an explicit call here, since CUDA graph replays
        are not guaranteed to execute the forward pre hook (which normally
        triggers unshard).

        This should be called during initialization and after each weight sync.
        For the full weight sync lifecycle, see `prepare_for_state_dict_load`.
        """
        from torch.distributed.fsdp._fully_shard._fsdp_param import free_storage

        with torch.inference_mode():
            for module in self.model.modules():
                if not isinstance(module, FSDPModule):
                    continue
                module.unshard()
                # TODO: replace this with FSDPModule._free_sharded_params()
                for param_group in module._get_fsdp_state()._fsdp_param_groups:
                    for param in param_group.fsdp_params:
                        sharded_data = param._sharded_param_data
                        if isinstance(sharded_data, _ShardedFSDPTensor):
                            sharded_data = sharded_data._tensor
                        free_storage(sharded_data)

    # TODO: followup with potentially adding extra kwarg ``sinks`` to vLLM attn
    def _inject_attention_sinks(self) -> None:
        """Give each gpt-oss attention's vLLM backend its sink-rescale hook."""
        from torchtitan.models.gpt_oss.model import (
            apply_attention_sink_rescale,
            Attention,
        )

        for module in self.model.modules():
            if not isinstance(module, Attention):
                continue
            # Read sinks per call: FSDP swaps the parameter between its sharded
            # and unsharded tensors and frees the sharded storage after unsharding.
            module.inner_attention.vllm_attn.impl.out_transform = (
                lambda out, lse, attention=module: apply_attention_sink_rescale(
                    out, lse, attention.sinks
                )
            )

    def _expose_routed_experts_to_vllm(self) -> None:
        """Let vLLM record each MoE router's expert ids per token.

        vLLM binds ``capture_fn`` on every module that has ``layer_id`` and ``capture_fn``
        attributes (``RoutedExpertsCaptureSource``) after its profile run, stores the ids
        by KV slot, and returns them as ``CompletionOutput.routed_experts``. A prefix-cache
        hit therefore returns the ids computed with that cached KV.

        Example (2 layers, top-2, a 3-token prompt that generated 2 tokens):

            completion_output.routed_experts.shape
            # -> (4, 2, 2): prompt tokens + all generated tokens but the last,
            #    which never ran forward
        """
        # With EP and TP > 1 each TP rank routes a shard of the tokens; vLLM's
        # capture expects all of this DP rank's tokens.
        gather_across_tp = (
            self.parallelism_context.ep_enabled and self.parallelism_context.tp_enabled
        )

        def capture(router, args, output):
            if router.capture_fn is None:
                return
            topk_expert_ids_TK = output[1]
            if gather_across_tp:
                topk_expert_ids_TK = get_tp_group().all_gather(
                    topk_expert_ids_TK, dim=0
                )
            router.capture_fn(topk_expert_ids_TK)

        for layer_name, layer in self.model.layers.items():
            for module in layer.modules():
                if isinstance(module, TokenChoiceTopKRouter):
                    # DeepSeek-V4 routers already store this same decoder index.
                    module.layer_id = int(layer_name)
                    module.capture_fn = None
                    module.register_forward_hook(capture)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """vLLM required API.
        Convert input token IDs to embeddings."""
        with self.parallelism_context.activate_spmd():
            return self.model.tok_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """vLLM required API.
        Convert input token IDs to embeddings (deprecated vLLM interface)."""
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        vLLM required API.
        Forward pass with vLLM interface.

        Args:
            input_ids: Token IDs [total_tokens] (1D varlen format)
            positions: Position indices [total_tokens] (1D varlen format)
            inputs_embeds: Pre-computed embeddings (optional)
            **kwargs: Additional vLLM kwargs

        Returns:
            hidden_states: Final hidden states [total_tokens, hidden_size]
        """
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds not yet supported")

        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        with self.parallelism_context.activate_spmd():
            h = self.model(input_ids, attention_metadata=None, positions=positions)
        # Inference disables sequence parallelism, so final hidden states should
        # already be replicated before returning to vLLM.
        if isinstance(h, DTensor):
            assert all(isinstance(p, Replicate) for p in h.placements)
            h = h._local_tensor
        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """vLLM required API.
        Compute logits from hidden states."""

        with self.parallelism_context.activate_spmd():
            logits = self.model.lm_head(hidden_states)

            # lm_head returns vocab-sharded logits under TP; gather to the
            # full local logits tensor that vLLM expects.
            if self.parallelism_context.tp_enabled:
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

    def _maybe_initial_load_weights(self) -> None:
        """Load initial HF weights via CheckpointManager.

        A configured manager loads initial weights. ``None`` skips loading for
        the RL loop, where weights arrive via TorchStore.
        """
        cfg = self._checkpointer_config
        if cfg is None:
            return

        adapter_cls = type(self.model).state_dict_adapter_cls
        sd_adapter = None
        if adapter_cls is not None:
            sd_adapter = adapter_cls(
                model_config=self.config,
                hf_assets_path=cfg.initial_load_path,
            )

        # Model-only CheckpointManager: initial_load_model_only=True (default)
        # ensures only MODEL state is loaded, so None optimizer/lr_scheduler
        # are never accessed.
        checkpointer = cfg.build(
            model_parts=[self.model],
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

    def load_weights(self, weights_iter):
        """
        vLLM required API.

        This is a no-op method since model weights are already loaded during initialization.
        Returns the names of all parameters that have been loaded so vLLM's safety check passes.

        Args:
            weights_iter: Iterator of (name, tensor) pairs from HF checkpoint

        Returns:
            Set of loaded parameter names
        """

        loaded_param_names = set()
        for name, _ in self.model.named_parameters():
            loaded_param_names.add("model." + name)

        logger.info(
            f"Weights already loaded during model initialization. \
            Returning {len(loaded_param_names)} loaded parameter names to satisfy vLLM safety check."
        )

        # Return the names of all loaded parameters so vLLM knows they were handled
        return loaded_param_names
