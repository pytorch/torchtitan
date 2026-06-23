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

import dataclasses
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import (
    apply_overrides,
    CompileConfig,
    OverrideConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.rl.models.attention import VLLMAttentionWrapper
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.module import Module
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import torch_utils as _torch_utils


logger = init_logger(__name__)

# NOTE: Monkeypatch vLLM's weak_ref_tensor to handle DTensor
# This is because piecewise CUDA-graph capture calls weak_ref_tensor()
# on every subgraphoutput (see vllm/compilation/cuda_graph.py).
# When TP is active some of those outputs are DTensors which fail with
# ("The specified pointer resides on host memory").  to_local
# converts the DTensor to a plain tensor. which succeeds with this
# cudagraph implementation.
_original_weak_ref_tensor = _torch_utils.weak_ref_tensor


def _dtensor_safe_weak_ref_tensor(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor._local_tensor
    return _original_weak_ref_tensor(tensor)


_torch_utils.weak_ref_tensor = _dtensor_safe_weak_ref_tensor


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
        model_spec: ModelSpec,
        parallelism: InferenceParallelismConfig,
        compile_config: CompileConfig,
        checkpoint_config: CheckpointManager.Config,
        vllm_config: VllmConfig,
        prefix: str = "",
        override: OverrideConfig,
    ):
        super().__init__()

        assert vllm_config is not None, "vllm_config is required"

        # Store components from model_spec
        self.state_dict_adapter = model_spec.state_dict_adapter
        self.parallelize_fn = model_spec.parallelize_fn

        # Replace inner_attention with VLLMAttentionWrapper in config
        model_config = model_spec.model
        attn_config = model_config.layers[0].attention
        n_heads = attn_config.n_heads
        n_kv_heads = attn_config.n_kv_heads or n_heads
        head_dim = (
            attn_config.head_dim
            if attn_config.head_dim is not None
            else model_config.dim // n_heads
        )
        new_layers = []
        for layer_cfg in model_config.layers:
            inner = layer_cfg.attention.inner_attention
            vllm_backend = VLLMAttentionWrapper.Config(
                hidden_size=model_config.dim,
                num_heads=n_heads,
                num_kv_heads=n_kv_heads,
                head_dim=head_dim,
                sliding_window_size=getattr(
                    layer_cfg.attention, "sliding_window_size", None
                ),
            )
            new_layers.append(
                dataclasses.replace(
                    layer_cfg,
                    attention=dataclasses.replace(
                        layer_cfg.attention, inner_attention=vllm_backend
                    ),
                )
            )
        self.config = dataclasses.replace(model_config, layers=new_layers)
        logger.debug(f"Creating model with config: {self.config.to_dict()}")

        # Translate the inference parallelism into torchtitan's full
        # ParallelismConfig that ParallelDims / parallelize_fn consume.
        training_parallelism = parallelism.to_training()

        # Build ParallelDims from the translated ParallelismConfig so TP/EP
        # sharding sees the same mesh shape as vLLM. data_parallel_shard_degree
        # carries vLLM's pure DP here (skip_dp=True below), not TorchTitan FSDP.
        self.parallel_dims = ParallelDims(
            dp_replicate=training_parallelism.data_parallel_replicate_degree,
            dp_shard=training_parallelism.data_parallel_shard_degree,
            cp=training_parallelism.context_parallel_degree,
            tp=training_parallelism.tensor_parallel_degree,
            pp=training_parallelism.pipeline_parallel_degree,
            ep=training_parallelism.expert_parallel_degree,
            world_size=dist.get_world_size(),
        )

        # Fill sharding configs on the config BEFORE build so every sub-module
        # is constructed with its ShardingConfig attached (required by the
        # declarative model.parallelize() API). Need to be called after Attention
        # module replacement.
        # Provides the generic config shape (has .parallelism) so
        # update_from_config can extract parallelism uniformly.
        @dataclass(kw_only=True, slots=True)
        class _InferenceConfig:
            parallelism: ParallelismConfig

        self.config.update_from_config(
            config=_InferenceConfig(parallelism=training_parallelism)
        )

        # Apply config overrides (e.g. the fused gate+up SwiGLU) after
        # update_from_config (which fills the sharding the override factories
        # read) and before build
        if override.imports:
            apply_overrides(override, self.config)

        # Build model on meta device to avoid allocating full model on every GPU
        with torch.device("meta"):
            self.model = self.config.build()

        self.model = self.parallelize_fn(
            model=self.model,
            parallel_dims=self.parallel_dims,
            training=TrainingConfig(),
            parallelism=training_parallelism,
            compile_config=compile_config,
            ac_config=None,
            dump_folder="",
            # Generator inference replicates parameters across vLLM DP groups.
            # Keep TP/EP sharding above, but do not translate dp_shard into
            # TorchTitan FSDP/DDP here.
            skip_dp=True,
        )

        # Load initial weights based on checkpoint config.
        self._checkpoint_config = checkpoint_config

        # Materialize model on GPU — only allocates local shards (not full
        # model) thanks to EP/TP DTensor sharding applied above.
        self.model.to_empty(device=vllm_config.device_config.device)
        # HF checkpoints do not necessarily contain every TorchTitan buffer
        # (for example MoE expert_bias_E).
        # TODO: When checkpoint doesn't contains expert_bias_E, check the config
        # should use loss based load balancing strategy.
        with torch.no_grad():
            self.model.init_weights(buffer_device=None)
        self._maybe_initial_load_weights()

        # Give each gpt-oss attention's vLLM backend its sink rescale.
        # Need to do it here after parallelize + weight load so sinks are
        # TP-sharded.
        self._inject_attention_sinks()

        # Optionally route the row-parallel wo/w2 TP all-reduce through vLLM's
        # custom all-reduce instead of DTensor's NCCL ring redistribute. Only
        # meaningful with TP; bound per-instance so shared classes are untouched.
        if parallelism.allreduce_backend == "vllm" and self.parallel_dims.tp_enabled:
            from torchtitan.experiments.rl.models.vllm_allreduce import (
                apply_vllm_allreduce,
            )

            apply_vllm_allreduce(self.model)
        elif parallelism.allreduce_backend not in ("nccl", "vllm"):
            raise ValueError(
                f"Unknown allreduce_backend {parallelism.allreduce_backend!r}; "
                "expected 'nccl' or 'vllm'."
            )

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
            sinks = module.sinks
            local_sinks = sinks._local_tensor if isinstance(sinks, DTensor) else sinks
            module.inner_attention.vllm_attn.impl.out_transform = partial(
                apply_attention_sink_rescale, sinks=local_sinks
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """vLLM required API.
        Convert input token IDs to embeddings."""
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

        # Convert vLLM interface to TorchTitan interface
        # vLLM: [total_tokens] → TorchTitan: [batch_size, seq_len]
        tokens_2d = input_ids.unsqueeze(0)

        # Get embeddings
        h = self.model.tok_embeddings(tokens_2d)

        positions = positions.unsqueeze(0)

        # Pass through transformer layers
        for layer in self.model.layers.values():
            h = layer(h, attention_masks=None, positions=positions)

        h = self.model.norm(h)
        # When parallelism is applied, get full tensor before return to vLLM Engine
        if isinstance(h, DTensor):
            h = h.full_tensor()

        # Convert to vLLM format: [total_tokens, hidden_size]
        if h.dim() == 3:
            hidden_size = h.size(-1)
            h = h.view(-1, hidden_size)
        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """vLLM required API.
        Compute logits from hidden states."""

        # When TP is applied, we return the full tensor (plain tensor) to vLLM engine
        # at the end of VLLMModelWrapper.forward().
        # We need to wrap the input from vLLM engine back to DTensor with Replicate() placement.
        if self.parallel_dims.tp_enabled:
            hidden_states = DTensor.from_local(
                hidden_states,
                device_mesh=self.parallel_dims.get_mesh("tp"),
                placements=[
                    Replicate(),
                ],
            )

        logits = self.model.lm_head(hidden_states)

        # Full DTensor path returns vocab-sharded logits as DTensor; vLLM
        # expects full plain tensors.
        if isinstance(logits, DTensor):
            placements = tuple(
                Replicate()
                if isinstance(p, Shard) and p.dim in (-1, logits.ndim - 1)
                else p
                for p in logits.placements
            )
            logits = logits.redistribute(placements=placements).to_local()

        return logits

    def _maybe_initial_load_weights(self) -> None:
        """Load initial HF weights via CheckpointManager.

        Controlled by ``self._checkpoint_config``:
        - ``enable=True`` and ``initial_load_in_hf=True``: load from HF
          via CheckpointManager (standalone inference path).
        - ``enable=False``: skip (RL loop — weights arrive via TorchStore).
        """
        cfg = self._checkpoint_config
        if not cfg.enable:
            return

        sd_adapter = None
        if self.state_dict_adapter is not None:
            sd_adapter = self.state_dict_adapter(
                model_config=self.config,
                hf_assets_path=cfg.initial_load_path,
            )

        # Model-only CheckpointManager: initial_load_model_only=True (default)
        # ensures only MODEL state is loaded, so None optimizer/lr_scheduler
        # are never accessed.
        checkpointer = cfg.build(
            dataloader=None,
            model_parts=[self.model],
            optimizers=None,
            lr_schedulers=None,
            states={},
            sd_adapter=sd_adapter,
        )
        checkpointer.load()

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
