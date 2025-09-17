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

from .hf_llama_patch import patch_hf_llama
patch_hf_llama()

@dataclass
class HFTransformerModelArgs(PretrainedConfig, BaseModelArgs):
    """
    Configuration class that bridges TorchTitan and HuggingFace Transformers naming conventions.
    
    Uses properties to provide TorchTitan-style access while maintaining HuggingFace compatibility.
    """
    
    def __init__(
        self,
        # TorchTitan args
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        vocab_size: int = 128256,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000,
        max_seq_len: int = 2048,
        depth_init: bool = True,
        use_flex_attn: bool = False,
        attn_mask_type: str = "causal",
        eos_id: int = 0,
        # HuggingFace specific args
        attn_implementation: str = "sdpa",
        **kwargs
    ):  
        # Store TorchTitan-specific args (no HF equivalent)
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.depth_init = depth_init
        self.use_flex_attn = use_flex_attn
        self.attn_mask_type = attn_mask_type
        
        # HuggingFace specific args
        self.attn_implementation = attn_implementation

        self._passed_args = dict(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            depth_init=depth_init,
            use_flex_attn=use_flex_attn,
            attn_mask_type=attn_mask_type,
            eos_id=eos_id,
            attn_implementation=attn_implementation,
            **kwargs
        )

    def __repr__(self) -> str:
        # HFTransformerModelArgs is a dataclass that also inherits from PretrainedConfig.
        # PretrainedConfig has a __repr__ that serializes the object to JSON, but it
        # doesn't work well with how HFTransformerModelArgs is initialized.
        # This custom __repr__ provides a dataclass-like representation that correctly
        # displays the arguments passed during initialization.
        args_str = ", ".join(f"{k}={v!r}" for k, v in self._passed_args.items())
        return f"{self.__class__.__name__}({args_str})"

    @property
    def dim(self) -> int:
        """TorchTitan: Model dimension (alias for HF hidden_size)"""
        return self.hidden_size
    
    @dim.setter
    def dim(self, value: int):
        self.hidden_size = value
    
    @property
    def n_layers(self) -> int:
        """TorchTitan: Number of layers (alias for HF num_hidden_layers)"""
        return self.num_hidden_layers
    
    @n_layers.setter
    def n_layers(self, value: int):
        self.num_hidden_layers = value
    
    @property
    def n_heads(self) -> int:
        """TorchTitan: Number of attention heads (alias for HF num_attention_heads)"""
        return self.num_attention_heads
    
    @n_heads.setter
    def n_heads(self, value: int):
        self.num_attention_heads = value
    
    @property
    def n_kv_heads(self) -> Optional[int]:
        """TorchTitan: Number of key-value heads (alias for HF num_key_value_heads)"""
        return self.num_key_value_heads
    
    @n_kv_heads.setter
    def n_kv_heads(self, value: Optional[int]):
        self.num_key_value_heads = value
    
    @property
    def norm_eps(self) -> float:
        """TorchTitan: Layer norm epsilon (alias for HF rms_norm_eps)"""
        return self.rms_norm_eps
    
    @norm_eps.setter
    def norm_eps(self, value: float):
        self.rms_norm_eps = value
    
    @property
    def max_seq_len(self) -> int:
        """TorchTitan: Maximum sequence length (alias for HF max_position_embeddings)"""
        return self.max_position_embeddings
    
    @max_seq_len.setter
    def max_seq_len(self, value: int):
        self.max_position_embeddings = value
    
    @property
    def eos_id(self) -> int:
        """TorchTitan: End of sequence token ID (alias for HF eos_token_id)"""
        return self.eos_token_id
    
    @eos_id.setter
    def eos_id(self, value: int):
        self.eos_token_id = value

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
        
        # Configure HF-specific settings to match TorchTitan settings
        self.tie_word_embeddings = False
        self.attention_bias = False
        self.mlp_bias = False
        self.use_cache = False
        self.initializer_range = 1.0  # use as std for normal init in embedding
        
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