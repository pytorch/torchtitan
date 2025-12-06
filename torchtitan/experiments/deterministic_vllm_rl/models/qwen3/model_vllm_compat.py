# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from vllm.config import VllmConfig
from vllm.logger import init_logger

from torchtitan.experiments.deterministic_vllm_rl.models.attention import (
    VLLMPagedFlashAttention,
)
from torchtitan.models.qwen3.infra.parallelize import apply_non_moe_tp
from torchtitan.tools.utils import device_type


logger = init_logger(__name__)


class TorchTitanQwen3ForCausalLM(nn.Module):
    """
    vLLM-compatible wrapper for TorchTitan's Qwen3 model.

    This class integrates TorchTitan's Qwen3Model with vLLM by:
    1. Importing TorchTitan's model architecture
    2. Replacing attention with vLLM's Attention with PagedAttention and kv cache capability.
    3. Implementing the vLLM model interface
    """

    is_text_generation_model = True  # Required for vLLM runner validation
    supports_pp = False  # Pipeline parallelism not supported yet
    supports_multimodal = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",  # This is required for vLLM interface
    ):
        super().__init__()

        # vLLM config is required
        assert vllm_config is not None, "vllm_config is required"

        # Import TorchTitan's Qwen3 model (deferred import to avoid CUDA init issues)
        from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
        from torchtitan.models.qwen3.model.model import Qwen3Model

        # Map HuggingFace config to TorchTitan ModelArgs
        logger.info("vllm config: " + str(vllm_config.__class__))
        hf_config = vllm_config.model_config.hf_config
        logger.info("hf_config: " + str(hf_config))
        model_args = Qwen3ModelArgs(
            vocab_size=getattr(hf_config, "vocab_size", 151936),
            dim=getattr(hf_config, "hidden_size", 2048),
            n_layers=getattr(hf_config, "num_hidden_layers", 4),
            n_heads=getattr(hf_config, "num_attention_heads", 16),
            n_kv_heads=getattr(hf_config, "num_key_value_heads", 2),
            head_dim=getattr(hf_config, "head_dim", 128),
            hidden_dim=getattr(hf_config, "intermediate_size", 11008),
            norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            max_seq_len=getattr(hf_config, "max_position_embeddings", 8192),
            rope_theta=getattr(hf_config, "rope_theta", 1000000.0),
            qk_norm=getattr(hf_config, "qk_norm", True),
        )

        print(f"In attention initialization, model args are : {model_args}")

        # Create TorchTitan model
        self.model = Qwen3Model(model_args)
        self.config = model_args

        self._replice_with_vllm_paged_attention(model_args)

        (
            dp_size,
            mp_size,
            cp_size,
            pp_size,
            ep_size,
            etp_size,
        ) = self._process_parallelism_settings(vllm_config)

        # Build device mesh and apply parallelization
        if mp_size > 1 or ep_size > 1:
            self._build_device_mesh_and_parallelize(
                dp_size, mp_size, cp_size, pp_size, ep_size, etp_size
            )

    def _replice_with_vllm_paged_attention(self, model_args):
        # The `vllm.Attention` module handles QKV projection, RoPE, etc., and calls `inner_attention`
        if not hasattr(self.model, "layers"):
            raise AttributeError(
                f"Model {type(self.model).__name__} must have .layers attribute"
            )

        for layer_name, layer in self.model.layers.items():
            if not hasattr(layer, "attention"):
                raise ValueError(f"Layer {layer_name} must have .attention attribute")

            vllm_attn = VLLMPagedFlashAttention(
                hidden_size=model_args.dim,
                num_heads=model_args.n_heads,  # 16 (8 when TP =2)
                # NOTE(jianiw): Before feeding into inner_attention, the n_kv_heads has been replicated -> num_heads
                num_kv_heads=model_args.n_heads,  # 16 (8 When TP=2)
                head_dim=model_args.head_dim,
                causal=True,
            )

            layer.attention.inner_attention = vllm_attn
        logger.info(
            "Successfully replaced TorchTitan attention with vLLM PagedFlashAttention"
        )

    def _process_parallelism_settings(
        self, vllm_config: VllmConfig, use_token_shuffling_moe: bool = False
    ):
        """
        Parse parallel config from vllm config
        """
        world_size = (
            vllm_config.parallel_config.data_parallel_size
            * vllm_config.parallel_config.tensor_parallel_size
        )
        ep_size = (
            world_size if vllm_config.parallel_config.enable_expert_parallel else 1
        )
        etp_size = (
            1 if vllm_config.parallel_config.enable_expert_parallel else world_size
        )
        dp_size = vllm_config.parallel_config.data_parallel_size

        mp_size = vllm_config.parallel_config.tensor_parallel_size
        cp_size = vllm_config.parallel_config.decode_context_parallel_size
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_size = pp_size

        return dp_size, mp_size, cp_size, pp_size, ep_size, etp_size

    def _build_device_mesh_and_parallelize(
        self,
        dp_size: int,
        mp_size: int,
        cp_size: int,
        pp_size: int,
        ep_size: int,
        etp_size: int,
    ):
        """
        Build device mesh in TorchTitan style and apply parallelization to the model.

        This follows the same approach as TorchTitan's ParallelDims.build_mesh()
        and parallelize_qwen3() functions.
        """
        import torch.distributed as dist

        # Get world size and validate
        world_size = dist.get_world_size()

        # For now, assume dp_shard=1 (no data parallel sharding)
        # In full implementation, you may need to calculate dp_replicate and dp_shard
        dp_replicate = dp_size
        dp_shard = 1

        # Validate parallelism settings
        assert dp_replicate * dp_shard * cp_size * mp_size * pp_size == world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp_size}) * tp({mp_size}) * pp({pp_size}) != WORLD_SIZE({world_size})"
        )

        # Build device mesh following TorchTitan's _build_mesh_without_ep pattern
        # (assuming no EP for now)
        dims = []
        names = []
        for d, name in zip(
            [pp_size, dp_replicate, dp_shard, cp_size, mp_size],
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
        ):
            if d > 1:
                dims.append(d)
                names.append(name)

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        world_mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        logger.info(f"Build torchtitan device mesh: {world_mesh}")

        # Apply tensor parallelism if enabled
        if mp_size > 1:
            tp_mesh = world_mesh["tp"]
            apply_non_moe_tp(
                model=self.model,
                tp_mesh=tp_mesh,
                loss_parallel=False,  # vLLM handles loss computation separately
                enable_float8_tensorwise_tp=False,  # Can be enabled if needed
                enable_async_tp=False,  # Can be enabled if needed
            )
            logger.info(f"Applied Tensor Parallelism with TP={mp_size}")

        # Store the mesh for future use
        self.world_mesh = world_mesh

    def _extend_rope_cache_if_needed(
        self, rope_cache: torch.Tensor, max_position: int
    ) -> torch.Tensor:
        """
        Extend rope_cache if needed (e.g., during vLLM profiling with 2x max_seq_len).

        Args:
            rope_cache: Current RoPE cache tensor
            max_position: Maximum position index needed

        Returns:
            Extended rope_cache if needed, otherwise original rope_cache
        """
        required_len = max_position + 1

        # No extension needed
        if required_len <= rope_cache.shape[0]:
            return rope_cache

        # Handle DTensor case - convert to local tensor first
        from torch.distributed._tensor import DTensor, Replicate

        is_dtensor = isinstance(rope_cache, DTensor)
        if is_dtensor:
            # Get the local tensor and device mesh
            device_mesh = rope_cache.device_mesh
            local_rope_cache = rope_cache.to_local()
            device = local_rope_cache.device
            dtype = local_rope_cache.dtype
        else:
            device = rope_cache.device
            dtype = rope_cache.dtype

        # Import precompute_rope_cache from the model module
        from torchtitan.models.qwen3.model.model import precompute_rope_cache

        # Precompute additional RoPE frequencies on-the-fly
        # TODO: This also has strong assumptions on config names
        rope_theta = self.config.rope_theta
        head_dim = self.config.head_dim
        extended_cache = precompute_rope_cache(
            dim=head_dim,
            max_seq_len=required_len,
            base=rope_theta,
        )
        extended_cache = extended_cache.to(device=device, dtype=dtype)

        # If original was DTensor, convert extended cache to DTensor too
        if is_dtensor:
            rope_cache = DTensor.from_local(
                extended_cache,
                device_mesh=device_mesh,
                placements=[Replicate()],
            )
        else:
            rope_cache = extended_cache

        return rope_cache

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings.

        This is the vLLM-standard method name for embedding tokens.
        """
        return self.model.tok_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings (deprecated vLLM interface)."""
        return self.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with vLLM interface.

        Args:
            input_ids: Token IDs from vLLM [total_tokens] (1D varlen format)
            positions: Position indices from vLLM [total_tokens] (1D varlen format)
            inputs_embeds: Pre-computed embeddings (optional, used by vLLM)
            **kwargs: Additional vLLM kwargs

        Returns:
            hidden_states: Final hidden states [total_tokens, hidden_size]
        """
        # Handle inputs_embeds vs input_ids properly
        if inputs_embeds is not None:
            raise NotImplementedError(
                "inputs_embeds is not yet supported by TorchTitan Qwen3. "
                "The model expects token IDs and computes embeddings internally. "
                "Please provide input_ids instead."
            )

        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Convert vLLM interface to TorchTitan interface
        # vLLM passes input_ids as [total_tokens] but TorchTitan expects [batch_size, seq_len]
        # For now, reshape to [1, total_tokens] as a simple batch of 1
        # TODO: In future, use attn_metadata.seq_lens to properly reconstruct batch structure
        tokens_2d = input_ids.unsqueeze(0)  # [total_tokens] -> [1, total_tokens]

        # Store positions in forward context for attention layers
        # Also convert positions to 2D format
        # TODO: The position id information is not properly used yet
        # TODO: The batch_size might not be 1, need double check
        if positions is not None:
            positions_2d = positions.unsqueeze(0)  # [total_tokens] -> [1, total_tokens]

        # Get embeddings from 2D tokens
        h = self.model.tok_embeddings(tokens_2d)  # [1, total_tokens, hidden_size]

        # Extend RoPE cache if needed (vLLM profiling may use 2x max_seq_len)
        max_position = positions.max().item() if positions is not None else 0
        rope_cache = self._extend_rope_cache_if_needed(
            self.model.rope_cache, max_position
        )

        # Get RoPE cache indexed by positions
        rope_cache = rope_cache[positions]

        # Pass through transformer layers
        for layer in self.model.layers.values():
            h = layer(h, rope_cache, attention_masks=None)

        # Convert output format back to vLLM expectations
        # vLLM expects hidden_states in [total_tokens, hidden_size] format
        # TorchTitan returns [batch_size, seq_len, hidden_size], so we need to flatten
        if h.dim() == 3:  # [batch_size, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = h.shape
            h = h.view(batch_size * seq_len, hidden_size)  # [total_tokens, hidden_size]

        # TODO(jianiw): explicitly insert communication and return full tensor to vLLM Engine. To be checked.
        if isinstance(h, DTensor):
            h = h.full_tensor()
        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states.

        Returns:
            Logits tensor, or None if TP rank > 0
        """
        # Apply final layer norm
        h = self.model.norm(hidden_states)

        # Apply output projection to get logits
        logits = self.model.output(h)

        # When using TP, only rank 0 returns logits
        # vLLM expects None from other ranks
        if isinstance(logits, DTensor):
            # Convert DTensor to local tensor for vLLM
            logits = logits.full_tensor()

        return logits

    def load_weights(self, weights_iter):
        """
        Uses TorchTitan's Qwen3StateDictAdapter to map HF → TorchTitan naming,
        then uses set_model_state_dict for proper distributed tensor handling.

        Args:
            weights_iter: Iterator of (name, tensor) pairs from HF checkpoint

        Returns:
            Set of loaded parameter names (for vLLM compatibility)
        """
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.checkpoint.state_dict import (
            set_model_state_dict,
            StateDictOptions,
        )

        from torchtitan.models.qwen3.model.state_dict_adapter import (
            Qwen3StateDictAdapter,
        )

        # Collect weights from iterator into a dict
        hf_state_dict = {}
        for name, tensor in weights_iter:
            hf_state_dict[name] = tensor

        # Use TorchTitan's adapter to convert HF → TorchTitan format
        adapter = Qwen3StateDictAdapter(
            model_args=self.config,
            hf_assets_path=None,  # Not needed for from_hf conversion
        )

        torchtitan_state_dict = adapter.from_hf(hf_state_dict)
        model_state_dict = {k: v for k, v in self.model.state_dict().items()}

        # Convert HF tensors to replicate DTensor if target is DTensor
        for name, tensor in torchtitan_state_dict.items():
            if name in model_state_dict and isinstance(model_state_dict[name], DTensor):
                # Get the device mesh from the target DTensor
                target_dtensor = model_state_dict[name]
                device_mesh = target_dtensor.device_mesh
                # Convert to replicate DTensor
                torchtitan_state_dict[name] = DTensor.from_local(
                    tensor.to(device_mesh.device_type),
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )

        # Use TorchTitan's distributed state dict loading
        # This handles TP/PP sharding automatically
        set_model_state_dict(
            model=self.model,
            model_state_dict=torchtitan_state_dict,
            options=StateDictOptions(
                strict=False,  # Allow missing keys
            ),
        )

        # manually patch the loaded
        loaded_params = {f"model.{name}" for name in torchtitan_state_dict.keys()}
        logger.info(
            f"Loaded {len(loaded_params)} parameters from checkpoint using distributed-aware loading"
        )

        return loaded_params
