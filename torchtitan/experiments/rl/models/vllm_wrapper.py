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

import torch
import torch._dynamo
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.rl.models.attention import VLLMAttentionWrapper
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.module import Module
from vllm.compilation import codegen as _codegen
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


# NOTE: Monkeypatch vLLM's _node_ref to handle DTensor placement types
# whose repr() uses unqualified class names not available in the generated
# code's exec namespace (which only has `import torch`).
_original_node_ref = _codegen._node_ref


# TODO: Followup with core vLLM fix
# https://github.com/pytorch/torchtitan/issues/3067
def _patched_node_ref(arg):
    try:
        from torch.distributed.tensor.placement_types import Partial, Placement

        if isinstance(arg, Placement):
            cls = type(arg)
            # Partial.__repr__ leaves reduce_op unquoted (e.g. "Partial(sum)")
            # which would resolve to the builtin sum, not the string "sum".
            if isinstance(arg, Partial):
                return f"{cls.__module__}.{cls.__name__}({arg.reduce_op!r})"
            return f"{cls.__module__}.{repr(arg)}"
    except ImportError:
        pass
    return _original_node_ref(arg)


_codegen._node_ref = _patched_node_ref


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
        parallelism: ParallelismConfig,
        compile_config: CompileConfig,
        checkpoint_config: CheckpointManager.Config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        assert vllm_config is not None, "vllm_config is required"

        # PP and CP are not supported on this inference path. User-facing
        # validation lives in Generator.Config.__post_init__; these are
        # internal invariants — by the time we get here, parallelism has
        # already been validated.
        assert parallelism.data_parallel_shard_degree == 1, (
            "vLLM wrapper requires data_parallel_shard_degree=1, "
            f"got {parallelism.data_parallel_shard_degree}"
        )
        assert parallelism.pipeline_parallel_degree == 1, (
            "vLLM wrapper requires pipeline_parallel_degree=1, "
            f"got {parallelism.pipeline_parallel_degree}"
        )
        assert parallelism.context_parallel_degree == 1, (
            "vLLM wrapper requires context_parallel_degree=1, "
            f"got {parallelism.context_parallel_degree}"
        )

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
        vllm_backend = VLLMAttentionWrapper.Config(
            hidden_size=model_config.dim,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        new_layers = [
            dataclasses.replace(
                layer_cfg,
                attention=dataclasses.replace(
                    layer_cfg.attention, inner_attention=vllm_backend
                ),
            )
            for layer_cfg in model_config.layers
        ]
        self.config = dataclasses.replace(model_config, layers=new_layers)
        logger.debug(f"Creating model with config: {self.config.to_dict()}")

        # Build ParallelDims from the torchtitan ParallelismConfig (the
        # controller's source of truth) rather than vLLM's parallel_config.
        self.parallel_dims = ParallelDims(
            dp_replicate=parallelism.data_parallel_replicate_degree,
            dp_shard=parallelism.data_parallel_shard_degree,
            cp=parallelism.context_parallel_degree,
            tp=parallelism.tensor_parallel_degree,
            pp=parallelism.pipeline_parallel_degree,
            ep=parallelism.expert_parallel_degree,
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

        self.config.update_from_config(config=_InferenceConfig(parallelism=parallelism))

        # Build model on meta device to avoid allocating full model on every GPU
        with torch.device("meta"):
            self.model = self.config.build()

        # With TP, collectives may return AsyncCollectiveTensor (overlap
        # path) or plain Tensor (sync path) depending on timing.  Dynamo
        # specializes on tensor type, so each switch triggers a
        # recompile.  Because of this, the default recompile_limit (8) is
        # too low; exceeding it fails under
        # fullgraph=True so set to 10 for now
        if compile_config.enable:
            torch._dynamo.config.recompile_limit = 10

        self.model = self.parallelize_fn(
            model=self.model,
            parallel_dims=self.parallel_dims,
            training=TrainingConfig(),
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ActivationCheckpointConfig(mode="none"),
            dump_folder="",
            skip_dp=True,
        )

        # Load initial weights based on checkpoint config.
        self._checkpoint_config = checkpoint_config

        # Materialize model on GPU — only allocates local shards (not full
        # model) thanks to EP/TP DTensor sharding applied above.
        self.model.to_empty(device=vllm_config.device_config.device)
        self._maybe_initial_load_weights()


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

        # Full DTensor path returns logits as DTensor; vLLM expects plain tensors.
        # disable_loss_parallel=True already makes lm_head output Replicate
        if isinstance(logits, DTensor):
            logits = logits.to_local()

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
