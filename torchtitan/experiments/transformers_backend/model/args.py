# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torch import nn
from torchtitan.config import JobConfig
from torchtitan.models.utils import (
    get_dense_model_nparams_and_flops,
    get_moe_model_nparams_and_flops,
)
from torchtitan.protocols import BaseModelArgs
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface


@dataclass
class HFTransformerModelArgs(PretrainedConfig, BaseModelArgs):
    """
    Configuration class that bridges TorchTitan and HuggingFace Transformers naming conventions.

    Uses properties to provide TorchTitan-style access while maintaining HuggingFace compatibility.
    Properties are created dynamically based on which arguments are provided.
    """

    # Define all possible mappings organized by argument type
    _TT_TO_HF_MAPPINGS = {
        "dense": {
            # TorchTitan dense model mappings (always available)
            "dim": "hidden_size",
            "n_layers": "num_hidden_layers",
            "n_heads": "num_attention_heads",
            "n_kv_heads": "num_key_value_heads",
            "norm_eps": "rms_norm_eps",
            "max_seq_len": "max_position_embeddings",
            "eos_id": "eos_token_id",
        },
        "moe": {
            # TorchTitan moe model specific mappings (only when titan_moe_args provided)
            "inter_dim": "intermediate_size",
            "n_dense_layers": "first_k_dense_replace",
        },
    }

    def __init__(
        self,
        titan_dense_args,
        titan_moe_args=None,
        # HuggingFace specific args
        attn_implementation: str = "sdpa_torchtitan",
        **kwargs,
    ):
        super().__init__(attn_implementation=attn_implementation, **kwargs)
        assert titan_dense_args is not None, "titan_dense_args is required"

        active_mappings = {}

        active_mappings.update(self._TT_TO_HF_MAPPINGS["dense"])

        if titan_moe_args is not None:
            active_mappings.update(self._TT_TO_HF_MAPPINGS["moe"])

        self._active_mappings = active_mappings

        self._create_dynamic_properties()

        # Set HF attributes from titan_args based on mappings
        for titan_name, hf_name in self._active_mappings.items():
            if hasattr(titan_dense_args, titan_name):
                setattr(self, hf_name, getattr(titan_dense_args, titan_name))

        # Fill all TorchTitan-specific args (no HF equivalent)
        self.multiple_of = titan_dense_args.multiple_of
        self.ffn_dim_multiplier = titan_dense_args.ffn_dim_multiplier
        self.depth_init = titan_dense_args.depth_init
        self.use_flex_attn = titan_dense_args.use_flex_attn
        self.attn_mask_type = titan_dense_args.attn_mask_type

        # HuggingFace specific args
        self.attn_implementation = attn_implementation
        # NOTE:(3outeille):This will force create_causal_mask to return None
        AttentionInterface._global_mapping[attn_implementation] = sdpa_attention_forward

        # Start with passed_args as just titan_args
        self._passed_args = {
            **titan_dense_args.__dict__,
            "attn_implementation": attn_implementation,
        }
        self._passed_args.update(kwargs)

        # NOTE(3outeille): Wait for transformers uniformization of MoE args
        if titan_moe_args is not None:
            # For DeepSeekV3, setting q_lora_rank to 0 in TorchTitan is equivalent to
            # setting it to None in HuggingFace.
            q_lora_rank = titan_moe_args.q_lora_rank
            if q_lora_rank == 0:
                q_lora_rank = None
            titan_moe_args.q_lora_rank = q_lora_rank

            self._passed_args.update(**titan_moe_args.__dict__)
            
            if titan_moe_args.moe_args is not None:
                moe_args = titan_moe_args.moe_args
                
                # Store moe_args for nparams/flops calculation
                self.moe_args = moe_args
                self.num_experts_per_tok = moe_args.top_k
                self.n_routed_experts = moe_args.num_experts
                self.n_shared_experts = moe_args.num_shared_experts
                self.moe_intermediate_size = titan_moe_args.moe_inter_dim
                
                # Set MoE-specific attributes directly on config for model access
                if hasattr(titan_moe_args, 'rope_interleave'):
                    self.rope_interleave = titan_moe_args.rope_interleave
                if hasattr(titan_moe_args, 'partial_rotary_factor'):
                    self.partial_rotary_factor = titan_moe_args.partial_rotary_factor
                if hasattr(titan_moe_args, 'n_group'):
                    self.n_group = titan_moe_args.n_group
                if hasattr(titan_moe_args, 'topk_group'):
                    self.topk_group = titan_moe_args.topk_group
                if hasattr(titan_moe_args, 'kv_lora_rank'):
                    self.kv_lora_rank = titan_moe_args.kv_lora_rank
                if hasattr(titan_moe_args, 'q_lora_rank'):
                    self.q_lora_rank = q_lora_rank  # Use the modified version (0 -> None)
                if hasattr(titan_moe_args, 'qk_nope_head_dim'):
                    self.qk_nope_head_dim = titan_moe_args.qk_nope_head_dim
                if hasattr(titan_moe_args, 'qk_rope_head_dim'):
                    self.qk_rope_head_dim = titan_moe_args.qk_rope_head_dim
                if hasattr(titan_moe_args, 'v_head_dim'):
                    self.v_head_dim = titan_moe_args.v_head_dim
        
                self._passed_args.update(
                    dict(
                        num_experts_per_tok=moe_args.top_k,
                        n_routed_experts=moe_args.num_experts,
                        n_shared_experts=moe_args.num_shared_experts,
                        moe_intermediate_size=titan_moe_args.moe_inter_dim,
                        rope_interleave=titan_moe_args.rope_interleave,
                        partial_rotary_factor=titan_moe_args.partial_rotary_factor,
                        n_group=titan_moe_args.n_group,
                        topk_group=titan_moe_args.topk_group,
                        kv_lora_rank=titan_moe_args.kv_lora_rank,
                        qk_nope_head_dim=titan_moe_args.qk_nope_head_dim,
                        qk_rope_head_dim=titan_moe_args.qk_rope_head_dim,
                        v_head_dim=titan_moe_args.v_head_dim,
                    )
                )

    def _create_dynamic_properties(self):
        """Create properties dynamically based on active mappings."""

        def _create_property(hf_name: str) -> property:
            def getter(self):
                return getattr(self, hf_name)

            def setter(self, value):
                setattr(self, hf_name, value)

            return property(getter, setter)

        for titan_name, hf_name in self._active_mappings.items():
            # Create getter/setter for attribute that don't already exist
            if not hasattr(self.__class__, titan_name):
                setattr(self.__class__, titan_name, _create_property(hf_name))

    def __repr__(self) -> str:
        # HFTransformerModelArgs is a dataclass that also inherits from PretrainedConfig.
        # PretrainedConfig has a __repr__ that serializes the object to JSON, but it
        # doesn't work well with how HFTransformerModelArgs is initialized.
        # This custom __repr__ provides a dataclass-like representation that correctly
        # displays the arguments passed during initialization.
        args_lines = [
            f"{k}={getattr(self, k)!r}"
            for k in sorted(self._passed_args.keys())
            if hasattr(self, k)
        ]
        args_str = "\n".join(args_lines)
        return f"{self.__class__.__name__}(\n{args_str}\n)"

    def update_from_config(self, job_config: JobConfig):
        # Load HF config (overwrites our HF attributes)
        hf_model_config = AutoConfig.from_pretrained(
            job_config.hf_transformers.model,
            attn_implementation=self.attn_implementation,
            trust_remote_code=True,
        )

        # Explicitly update attributes based on mappings
        for titan_name, hf_name in self._active_mappings.items():
            if hasattr(hf_model_config, hf_name):
                setattr(self, titan_name, getattr(hf_model_config, hf_name))

        # Copy any other attributes that might not be in the mapping
        for key, value in hf_model_config.to_dict().items():
            setattr(self, key, value)

        # Update our attributes with the passed args from flavors
        for key, value in self._passed_args.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

        # MoE
        if hasattr(self, "qk_nope_head_dim") and hasattr(self, "qk_rope_head_dim"):
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # Configure HF-specific settings to match TorchTitan settings
        self.attention_bias = False
        self.mlp_bias = False
        self.use_cache = False
        self.initializer_range = 1.0  # use as std for normal init in embedding

        if not hasattr(self, "inter_dim"):  # Only for llama model
            ffn_hidden_size = 4 * self.dim
            ffn_hidden_size = int(2 * ffn_hidden_size / 3)
            if self.ffn_dim_multiplier is not None:
                ffn_hidden_size = int(self.ffn_dim_multiplier * ffn_hidden_size)
            self.intermediate_size = self.multiple_of * (
                (ffn_hidden_size + self.multiple_of - 1) // self.multiple_of
            )

        self.head_dim = self.dim // self.num_attention_heads

        return self

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        is_moe = hasattr(self, "n_routed_experts")

        if is_moe:
            return get_moe_model_nparams_and_flops(self, model, seq_len)
        else:
            return get_dense_model_nparams_and_flops(self, model, seq_len)
