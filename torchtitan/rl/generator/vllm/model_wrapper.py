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
from typing import cast

import torch
from torch.distributed.tensor import DTensor
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import supports_multimodal
from vllm.utils import torch_utils as _torch_utils

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import maybe_apply_numa_binding
from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import (
    Attention,
    FlexInnerAttention,
    LinearAttention,
    MLAAttention,
)
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.rl.generator.engine import ModelStateDictLoadMode
from torchtitan.rl.generator.model import (
    build_dist_moe_runtime,
    build_inference_model,
    compute_logits,
    get_attention_dimensions,
    ModelStateDictLoader,
    route_tp_all_reduce,
    validate_ep_token_dispatcher_capacity,
)
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.vllm.registry import uses_compressed_mla_cache


logger = init_logger(__name__)


def _replace_vllm_layer_configs(model_config):
    """Replace inner-computation configs with vLLM generation variants."""
    # These modules inspect the breakable-CUDA graph environment at import time.
    # Defer imports until vLLM constructs the model, after the generator has set
    # that environment. Import the GDN adapter only for hybrid models so other
    # models do not acquire its vLLM-specific dependencies.
    from torchtitan.rl.generator.vllm.attention.attention import VLLMInnerAttention
    from torchtitan.rl.generator.vllm.attention.mla import (
        VLLMMLAAttention,
        VLLMMLAInnerAttention,
    )

    new_layers = []
    for layer_idx, layer_cfg in enumerate(model_config.layers):
        new_layer_cfg = layer_cfg

        attention_cfg = getattr(layer_cfg, "attention", None)
        if isinstance(attention_cfg, Attention.Config) and not isinstance(
            attention_cfg, LinearAttention.Config
        ):
            (
                num_heads,
                num_kv_heads,
                head_dim,
                value_head_dim,
            ) = get_attention_dimensions(attention_cfg, model_config.dim)
            backend_cfg = attention_cfg.inner_attention
            is_mla = isinstance(backend_cfg, MLAAttention.Config)
            if uses_compressed_mla_cache(backend_cfg):
                vllm_attention_cfg = VLLMMLAAttention.Config(
                    wkv_b=backend_cfg.wkv_b,
                    packed_kv_head_dim=backend_cfg.packed_kv_head_dim,
                    inner_attention=VLLMMLAInnerAttention.Config(
                        num_heads=num_heads,
                        kv_lora_rank=attention_cfg.kv_lora_rank,
                        qk_nope_head_dim=attention_cfg.qk_nope_head_dim,
                        qk_rope_head_dim=attention_cfg.qk_rope_head_dim,
                        v_head_dim=attention_cfg.v_head_dim,
                        scale=attention_cfg.softmax_scale,
                        sharding_config=backend_cfg.inner_attention.sharding_config,
                    ),
                    sharding_config=backend_cfg.sharding_config,
                )
            else:
                kernel_cfg = backend_cfg.inner_attention if is_mla else backend_cfg
                if is_mla and isinstance(kernel_cfg, FlexInnerAttention.Config):
                    raise ValueError(
                        f"layers[{layer_idx}]: MLA with a materialized K/V cache "
                        "does not support the flex attention backend in the "
                        "TorchTitan vLLM wrapper yet; use varlen."
                    )
                vllm_attention_cfg = VLLMInnerAttention.Config(
                    hidden_size=model_config.dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    value_head_dim=value_head_dim,
                    scale=attention_cfg.softmax_scale,
                    sliding_window_size=getattr(
                        attention_cfg, "sliding_window_size", None
                    ),
                    sharding_config=kernel_cfg.sharding_config,
                )
                if is_mla:
                    vllm_attention_cfg = MLAAttention.Config(
                        wkv_b=backend_cfg.wkv_b,
                        packed_kv_head_dim=backend_cfg.packed_kv_head_dim,
                        inner_attention=vllm_attention_cfg,
                        sharding_config=backend_cfg.sharding_config,
                    )
            attention_cfg = dataclasses.replace(
                attention_cfg,
                inner_attention=vllm_attention_cfg,
            )
            new_layer_cfg = dataclasses.replace(
                new_layer_cfg,
                attention=attention_cfg,
            )

        elif isinstance(attention_cfg, GDN.Config):
            delta_net_cfg = attention_cfg
            from torchtitan.rl.generator.vllm.attention.gdn import VLLMInnerGDN

            vllm_inner_gdn_cfg = VLLMInnerGDN.Config(
                layer_idx=layer_idx,
                num_k_heads=delta_net_cfg.n_key_heads,
                num_v_heads=delta_net_cfg.n_heads,
                head_k_dim=delta_net_cfg.key_head_dim,
                head_v_dim=delta_net_cfg.value_head_dim,
                conv_kernel_size=delta_net_cfg.conv_kernel_size,
                sharding_config=delta_net_cfg.inner_attention.sharding_config,
            )
            new_layer_cfg = dataclasses.replace(
                new_layer_cfg,
                attention=dataclasses.replace(
                    delta_net_cfg,
                    inner_attention=vllm_inner_gdn_cfg,
                ),
            )

        elif isinstance(attention_cfg, KDA.Config):
            kda_cfg = attention_cfg
            from torchtitan.rl.generator.vllm.attention.kda import VLLMInnerKDA

            vllm_inner_kda_cfg = VLLMInnerKDA.Config(
                num_heads=kda_cfg.n_heads,
                head_dim=kda_cfg.head_dim,
                conv_kernel_size=kda_cfg.conv_kernel_size,
                lower_bound=kda_cfg.inner_attention.kernel.lower_bound,
                layer_index=layer_idx,
                sharding_config=kda_cfg.inner_attention.sharding_config,
            )
            new_layer_cfg = dataclasses.replace(
                new_layer_cfg,
                attention=dataclasses.replace(
                    kda_cfg,
                    inner_attention=vllm_inner_kda_cfg,
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


def _patch_vllm_all_reduce() -> None:
    """Route the generator's tensor-parallel all-reduce through vLLM's custom
    one-shot/multimem AR instead of NCCL's ring (``route_tp_all_reduce``; applied
    when batch-invariant mode is off).

    ``tensor_model_parallel_all_reduce`` reuses vLLM's TP GroupCoordinator: the
    same ranks as DTensor's TP mesh group, but the group that owns the custom-AR
    shared buffers. It guards world_size == 1 and on CUDA dispatches to
    torch.ops.vllm.all_reduce.

    The custom AR is also forced onto its registered=False path so CUDA graph
    capture works. registered=True records graph buffers and calls
    cudaIpcGetMemHandle on them, which fails for the expandable_segments (VMM)
    memory the RL stack enables for Monarch RDMA. registered=False reduces via the
    init-time buffer_ptrs (raw cudaMalloc, IPC-able), records no graph buffers, at
    the cost of one staging copy per AR.
    """
    from vllm.distributed import get_tp_group

    route_tp_all_reduce(tensor_model_parallel_all_reduce)

    device_comm = get_tp_group().device_communicator
    ca = getattr(device_comm, "ca_comm", None) if device_comm is not None else None
    if ca is not None and not ca.disabled:

        def custom_all_reduce(input):
            # Mirrors CustomAllreduce.custom_all_reduce but always registered=False.
            if ca.disabled or not ca.should_custom_ar(input):
                return None
            return ca.all_reduce(input, registered=False)

        ca.custom_all_reduce = custom_all_reduce


@support_torch_compile(
    # The forward arguments whose size follows the step's number of tokens, and which of
    # their dims that is. vLLM marks that dim dynamic for torch.compile, so one compiled
    # graph serves every step size.
    dynamic_arg_dims={
        # (num_tokens,)
        "input_ids": 0,
        # (num_tokens,) for 1D RoPE; (3, num_tokens) for M-RoPE models serving image prompts
        # (temporal, height and width positions). The token dim is the last one in both,
        # hence -1, as in vLLM's Qwen2-VL.
        "positions": -1,
        # (num_tokens, dim): what vLLM passes multimodal models instead of input_ids.
        "inputs_embeds": 0,
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

    def __init__(
        self,
        *,
        model_config: Decoder.Config,
        parallelism: InferenceParallelismConfig,
        precision: PrecisionConfig,
        checkpointer_config: CheckpointManager.Config | None,
        vllm_config: VllmConfig,
        prefix: str = "",
        dist_moe_runtime: DistMoeRuntime.Config | None = None,
        model_state_dict_load_mode: ModelStateDictLoadMode,
    ):
        super().__init__()

        assert vllm_config is not None, "vllm_config is required"

        model_config = copy.deepcopy(model_config)
        self.config = _replace_vllm_layer_configs(model_config)
        logger.debug(f"Creating model with config: {self.config.to_dict()}")

        if vllm_config.model_config.dtype != TORCH_DTYPE_MAP[precision.param_dtype]:
            raise ValueError(
                f"vLLM computes in {vllm_config.model_config.dtype}, but the trainer computes "
                f"in precision.param_dtype={precision.param_dtype!r}. Pass "
                "precision.param_dtype as the engine dtype."
            )
        device = cast(torch.device, vllm_config.device_config.device)
        self.model, self.parallelism_context, frozen_param_keys = build_inference_model(
            self.config,
            parallelism=parallelism,
            precision=precision,
            device=device,
            # Text-only generators drop the vision encoder; VLLMMultimodalModelWrapper keeps it.
            keep_vision_encoder=supports_multimodal(self),
        )
        # TorchTitanGPUModelRunner pads a step to a multiple of TP when TP shards the
        # routed tokens.
        token_multiple = parallelism.expert_sequence_parallel_size
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        validate_ep_token_dispatcher_capacity(
            self.config,
            parallelism=parallelism,
            max_num_tokens_per_step=-(-max_num_batched_tokens // token_multiple)
            * token_multiple,
        )
        self._initialize_dist_moe_runtime(
            dist_moe_runtime, max_num_batched_tokens, device
        )
        maybe_apply_numa_binding(torch.cuda.current_device(), "cuda")
        # Loads the initial weights, then unshards the model, so vLLM profiles GPU
        # memory with the model's representation during forward.
        self.model_state_dict_loader = ModelStateDictLoader(
            self.model,
            checkpointer_config=checkpointer_config,
            model_state_dict_load_mode=model_state_dict_load_mode,
            frozen_param_keys=frozen_param_keys,
            hf_assets_path=vllm_config.model_config.model,
        )

        # Give each gpt-oss attention's vLLM backend its sink rescale.
        self._inject_attention_sinks()

        # Route the TP all-reduce through vLLM's custom AR (off under
        # batch-invariant mode, where its size-dependent algorithm breaks). It sends
        # every sum all-reduce to the TP group, so it comes after the initial load.
        if self.parallelism_context.tp_enabled and not is_in_batch_invariant_mode():
            _patch_vllm_all_reduce()

    def set_num_valid_tokens(self, num_valid_tokens: int) -> None:
        """Publish how many of this step's tokens are real, for the padding mask.

        vLLM pads a step (tensor-parallel rounding, CUDA-graph capture size,
        data-parallel equalization) before the model runs. The runner calls this
        with the unpadded count. It is a device write with no host sync, so it
        can run before every graph replay. A no-op for models without Dist-MoE.
        """
        if self._num_valid_tokens is not None:
            self._num_valid_tokens.fill_(num_valid_tokens)

    def _dist_moe_num_local_input_tokens(self, num_tokens: int) -> int:
        """Return the Dist-MoE row count every EP rank passes in this step.

        Every EP rank must pass the same count, and the routed tokens are split
        evenly over TP, so this is the step's token count over TP.
        """
        tp = self.parallelism_context.tp
        # TorchTitanGPUModelRunner pads every step to a multiple of TP when TP
        # shards the routed tokens, and every DP replica does the same.
        assert num_tokens % tp == 0, f"TP={tp} does not divide {num_tokens} tokens"
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is not None:
            # vLLM DP > 1: vLLM builds DP metadata for every MoE forward (real
            # steps and capture runs alike), and the EP group spans the DP
            # replicas, so every replica uses the largest count:
            # - without CUDA graphs, DP replicas may run different token counts;
            # - with CUDA graphs, vLLM already padded every replica to the graph
            #   size outside the model forward, so the max equals this count.
            num_tokens = int(dp_metadata.num_tokens_across_dp_cpu.max())
            assert num_tokens % tp == 0, f"TP={tp} does not divide {num_tokens} tokens"
        # Otherwise DP = 1: the EP group is the TP group, which shares this
        # step's batch, so this rank's own count is every EP rank's count.
        return num_tokens // tp

    def _initialize_dist_moe_runtime(
        self,
        runtime_config: DistMoeRuntime.Config | None,
        max_num_batched_tokens: int,
        device: torch.device,
    ) -> None:
        """Build the Dist-MoE runtime (see ``build_dist_moe_runtime``) and the
        device scalar of each step's real token count."""
        self._dist_moe_runtime = build_dist_moe_runtime(
            self.model,
            runtime_config,
            parallelism_context=self.parallelism_context,
            device=device,
            max_num_tokens_per_step=max_num_batched_tokens,
        )
        self._num_valid_tokens: torch.Tensor | None = None
        if self._dist_moe_runtime is not None:
            # The runner writes each step's real token count here before the
            # forward runs, and forward builds the padding mask from it on the
            # device, so CUDA-graph replay sees the current count. The address must
            # stay fixed from before capture. Until a step publishes, every token
            # counts as real.
            self._num_valid_tokens = torch.full(
                (), torch.iinfo(torch.int32).max, dtype=torch.int32, device=device
            )

    # TODO: followup with potentially adding extra kwarg ``sinks`` to vLLM attn
    def _inject_attention_sinks(self) -> None:
        """Give each gpt-oss attention's vLLM backend its sink-rescale hook."""
        from torchtitan.models.gpt_oss.model import (
            apply_attention_sink_rescale,
            GptOssAttention,
        )

        for module in self.model.modules():
            if not isinstance(module, GptOssAttention):
                continue
            # Read sinks per call: FSDP swaps the parameter between its sharded
            # and unsharded tensors and frees the sharded storage after unsharding.
            module.inner_attention.vllm_attn.impl.out_transform = (
                lambda out, lse, attention=module: apply_attention_sink_rescale(
                    out, lse, attention.sinks
                )
            )

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
            h = self.model(
                input_ids,
                attention_metadata=None,
                positions=positions,
                **self._prepare_step_kwargs(input_ids.shape[0]),
            )
        return h

    def _prepare_step_kwargs(self, num_tokens: int) -> dict[str, torch.Tensor]:
        """Set this step's Dist-MoE token count and return the layers' padding-mask kwarg.

        Call once per forward.
        """
        # Rows past the real token count are padding that vLLM added (true
        # for padding). Only the MoE reads this mask: Dist-MoE routes those
        # rows to expert -1. Attention takes its metadata from vLLM, which
        # already excludes padding.
        if self._dist_moe_runtime is None:
            return {}
        assert self._num_valid_tokens is not None
        self._dist_moe_runtime.set_num_local_input_tokens_per_call(
            self._dist_moe_num_local_input_tokens(num_tokens)
        )
        return {
            "padding_mask": torch.arange(
                num_tokens, device=self._num_valid_tokens.device
            )
            >= self._num_valid_tokens
        }

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """vLLM required API.
        Compute logits from hidden states."""

        return compute_logits(self.model, hidden_states, self.parallelism_context)

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
