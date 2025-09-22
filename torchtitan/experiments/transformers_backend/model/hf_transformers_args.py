# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from dataclasses import dataclass
from typing import Optional

from torch import nn
from torchtitan.config import JobConfig
from torchtitan.protocols import BaseModelArgs
from torchtitan.tools.logging import logger
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

@dataclass
class HFTransformerModelArgs(PretrainedConfig, BaseModelArgs):
    """
    Configuration class that bridges TorchTitan and HuggingFace Transformers naming conventions.
    
    Uses properties to provide TorchTitan-style access while maintaining HuggingFace compatibility.
    Properties are created dynamically based on which arguments are provided.
    """
    
    # Define all possible mappings organized by argument type
    _TT_TO_HF_MAPPINGS = {
        "base": {
            # Core TorchTitan mappings (always available)
            "dim": "hidden_size",
            "n_layers": "num_hidden_layers",
            "n_heads": "num_attention_heads",
            "n_kv_heads": "num_key_value_heads",
            "norm_eps": "rms_norm_eps",
            "max_seq_len": "max_position_embeddings",
            "eos_id": "eos_token_id",
        },
        "deepseek_v3": {
            # DeepSeekV3 specific mappings (only when deepseek_v3_args provided)
            "inter_dim": "intermediate_size",
            "n_dense_layers": "first_k_dense_replace",
        },
    }

    def __init__(
        self,
        titan_args,
        deepseek_v3_args=None,
        # HuggingFace specific args
        attn_implementation: str = "sdpa",
        **kwargs,
    ):
        assert titan_args is not None, "titan_args is required"

        active_mappings = {}
        
        active_mappings.update(self._TT_TO_HF_MAPPINGS["base"])
        
        if deepseek_v3_args is not None:
            active_mappings.update(self._TT_TO_HF_MAPPINGS["deepseek_v3"])
        
        self._active_mappings = active_mappings
        
        self._create_dynamic_properties()

        # Fill all TorchTitan-specific args (no HF equivalent)
        self.multiple_of = titan_args.multiple_of
        self.ffn_dim_multiplier = titan_args.ffn_dim_multiplier
        self.depth_init = titan_args.depth_init
        self.use_flex_attn = titan_args.use_flex_attn
        self.attn_mask_type = titan_args.attn_mask_type

        # HuggingFace specific args
        self.attn_implementation = attn_implementation

        # Start with passed_args as just titan_args
        self._passed_args = {**titan_args.__dict__, "attn_implementation": attn_implementation}
        self._passed_args.update(kwargs)

        # If DeepSeekV3 args are provided, fill the rest
        if deepseek_v3_args is not None:
            # For DeepSeekV3, setting q_lora_rank to 0 in TorchTitan is equivalent to
            # setting it to None in HuggingFace.
            q_lora_rank = deepseek_v3_args.q_lora_rank
            if q_lora_rank == 0:
                q_lora_rank = None
            deepseek_v3_args.q_lora_rank = q_lora_rank

            self._passed_args.update(**deepseek_v3_args.__dict__)

            self.partial_rotary_factor = deepseek_v3_args.partial_rotary_factor

            if deepseek_v3_args.moe_args is not None:
                moe_args = deepseek_v3_args.moe_args
                self.num_experts_per_tok = moe_args.top_k
                self.n_routed_experts = moe_args.num_experts
                self.n_shared_experts = moe_args.num_shared_experts
                self.moe_intermediate_size = deepseek_v3_args.moe_inter_dim
                self._passed_args.update(
                    dict(
                        num_experts_per_tok=moe_args.top_k,
                        n_routed_experts=moe_args.num_experts,
                        n_shared_experts=moe_args.num_shared_experts,
                        moe_intermediate_size=deepseek_v3_args.moe_inter_dim,
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
        args_lines = [f"{k}={v!r}" for k, v in sorted(self._passed_args.items())]
        args_str = "\n".join(args_lines)
        return f"{self.__class__.__name__}(\n{args_str}\n)"

    def update_from_config(self, job_config: JobConfig):
        # Load HF config (overwrites our HF attributes)
        hf_model_config = AutoConfig.from_pretrained(
            job_config.model.name,
            attn_implementation=self.attn_implementation,
        )

        self.__dict__.update(hf_model_config.__dict__)
        
        # Update our attributes with the passed args from flavors
        for key, value in self._passed_args.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # MoE
        if hasattr(self, "qk_nope_head_dim") and hasattr(self, "qk_rope_head_dim"):
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        
        # Configure HF-specific settings to match TorchTitan settings
        self.tie_word_embeddings = False
        self.attention_bias = False
        self.mlp_bias = False
        self.use_cache = False
        self.initializer_range = 1.0  # use as std for normal init in embedding
        
        if not hasattr(self, "inter_dim"): # Only for llama model
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
        #TODO(3outeille): adapt to handle MoE
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

        return nparams, num_flops_per_token

    def debug_structure_param(self, model: nn.Module):
        logger.info("Model Structure Parameter Breakdown:")

        def _format_module(module: nn.Module, prefix: str = ""):
            for name, sub_module in module.named_children():
                sub_module_params = sum(p.numel() for p in sub_module.parameters())
                if sub_module_params == 0:
                    continue

                # For HF models, we want to "unwrap" the ".model" attribute
                # to get a view comparable to the native TorchTitan models.
                if name == "model":
                    _format_module(sub_module, prefix)
                else:
                    logger.info(
                        f"{prefix}({name}): {sub_module.__class__.__name__} - {sub_module_params:,} params"
                    )
                    _format_module(sub_module, prefix + "  ")

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"{model.__class__.__name__} - {total_params:,} params")
        _format_module(model, "  ")


class HFTransformerModel(nn.Module):
    def __init__(self, model_args: HFTransformerModelArgs):
        super().__init__()
        
        # Try to import the model class dynamically from the transformers library if not found in globals
        model_class_name = model_args.architectures[0]
        model_cls = globals().get(model_class_name, None)
        if model_cls is None:
            try:
                transformers_mod = importlib.import_module("transformers")
                model_cls = getattr(transformers_mod, model_class_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Could not find model class '{model_class_name}' in globals or transformers. "
                    f"Make sure the class is available. Original error: {e}"
                )
        self.model = model_cls(config=model_args)

    @property
    def layers(self):
        """Returns the model's layers, handling different Hugging Face model structures."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):  # Llama-like
            return self.model.model.layers
        else:
            # Add more cases here if needed for other model architectures
            raise AttributeError("Could not find layers in the model. Please check the model structure.")

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, CausalLMOutputWithPast):
            return output.logits
        return output

    def init_weights(self, *args, **kwargs):
        self.model.post_init()