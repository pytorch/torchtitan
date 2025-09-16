# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional, Union
import os

from torch import nn
from torchtitan.config import JobConfig
from torchtitan.protocols import BaseModelArgs
from torchtitan.tools.logging import logger
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# NOTE(3outeille): monkey-patch PreTrainedModel to handle meta device initialization correctly
# The default _initialize_weights sets _is_hf_initialized = True even on a meta device,
# which prevents subsequent proper initialization.
def _initialize_weights_patched(self, module):
    """
    Patched version of _initialize_weights that skips initialization and setting
    the _is_hf_initialized flag if the module is on a meta device.
    """
    if getattr(module, "_is_hf_initialized", False):
        return

    for param in module.parameters(recurse=True):
        if param.device.type == "meta":
            return
    
    # If not on a meta device, call the original weight initialization
    self._init_weights(module)
    module._is_hf_initialized = True


#TODO(3outeille): find a better way to do this
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

_original_llama_decoder_layer_init = LlamaDecoderLayer.__init__

def _llama_decoder_layer_init_patched(self, config: LlamaConfig, layer_idx: int):
    _original_llama_decoder_layer_init(self, config, layer_idx)
    self.mlp.layer_idx = layer_idx

LlamaDecoderLayer.__init__ = _llama_decoder_layer_init_patched


def _init_weights_patched(self, module):
    """
    Patched version of _init_weights to match TorchTitan's initialization for Llama.
    `self` is a LlamaPreTrainedModel instance.
    """
    config = self.config

    if isinstance(module, (LlamaAttention, LlamaMLP)):
        layer_idx = module.layer_idx

        if config.depth_init:
            init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5
        else:
            init_std = 0.02 / (2 * config.num_hidden_layers) ** 0.5

    if isinstance(module, LlamaAttention):
        nn.init.trunc_normal_(module.q_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.k_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.v_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.o_proj.weight, mean=0.0, std=init_std)
    
    elif isinstance(module, LlamaMLP):
        nn.init.trunc_normal_(module.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(module.up_proj.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(module.down_proj.weight, mean=0.0, std=init_std)

    elif module is getattr(self, "lm_head", None): #TODO(3outeille): find a better way to detect lm_head
        final_out_std = config.hidden_size**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )
        if module.bias is not None:
            module.bias.data.zero_()

    elif isinstance(module, nn.Embedding):
        std = config.initializer_range
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    
    elif (
        isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        or "LayerNorm" in module.__class__.__name__
        or "RMSNorm" in module.__class__.__name__
    ):
        # Norms can exist without weights (in which case they are None from torch primitives)
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(1.0)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()


PreTrainedModel._init_weights = _init_weights_patched
PreTrainedModel._initialize_weights = _initialize_weights_patched

@dataclass
class HFTransformerModelArgs(LlamaConfig, BaseModelArgs):
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
        # Map TorchTitan arguments to HuggingFace arguments for parent class initialization
        hf_config_dict = dict(
            hidden_size=dim,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            vocab_size=vocab_size,
            rms_norm_eps=norm_eps,
            rope_theta=rope_theta,
            max_position_embeddings=max_seq_len,
            eos_token_id=eos_id,
            **kwargs
        )
        
        super().__init__(**hf_config_dict)
        
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
        hf_model_config = LlamaConfig.from_pretrained(
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

        layer_params = {}  # int -> int
        embedding_params = 0
        norm_params = 0
        lm_head_params = 0
        misc_params = {}

        for name, p in model.named_parameters():
            if "model.embed_tokens" in name:
                embedding_params += p.numel()
            elif "model.layers." in name:
                try:
                    layer_num = int(name.split("layers.")[1].split(".")[0])
                    if layer_num not in layer_params:
                        layer_params[layer_num] = 0
                    layer_params[layer_num] += p.numel()
                except (ValueError, IndexError):
                    # Should not happen with standard HF llama names
                    component = "misc_layer_parts"
                    if component not in misc_params:
                        misc_params[component] = 0
                    misc_params[component] += p.numel()
            elif "model.norm" in name:
                norm_params += p.numel()
            elif "lm_head" in name:
                lm_head_params += p.numel()
            else:
                # Catch anything else
                component = name.split(".")[0]
                if component not in misc_params:
                    misc_params[component] = 0
                misc_params[component] += p.numel()

        logger.info("Parameter breakdown:")
        logger.info(f"  - embedding: {embedding_params:,} parameters")
        for layer_num in sorted(layer_params.keys()):
            params = layer_params[layer_num]
            logger.info(f"  - layer_{layer_num}: {params:,} parameters")
        logger.info(f"  - final_norm: {norm_params:,} parameters")
        logger.info(f"  - lm_head: {lm_head_params:,} parameters")
        if misc_params:
            for name, params in misc_params.items():
                logger.info(f"  - {name} (misc): {params:,} parameters")

        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = self.n_layers, self.n_heads, self.dim // self.n_heads, seq_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return nparams, num_flops_per_token


class HFTransformerModel(LlamaForCausalLM):
    def __init__(self, model_args: HFTransformerModelArgs):
        super().__init__(model_args)

    def init_weights(self, *args, **kwargs):
        # Taken from transformers.modeling_utils.PreTrainedModel.init_weights
        super().init_weights()
        self._backward_compatibility_gradient_checkpointing()

        # Make sure the modules correctly exist if the flag is active
        if self._keep_in_fp32_modules is not None or self._keep_in_fp32_modules_strict is not None:
            all_parameters = {name for name, _ in self.named_parameters() if len(name) > 0}
            unique_module_names = set()
            # Get all unique module names in the module graph, without the prefixes
            for param in all_parameters:
                unique_module_names.update(
                    [name for name in param.split(".") if not name.isnumeric() and name not in ["weight", "bias"]]
                )
            # Check that every module in the keep_in_fp32 list is part of the module graph
            if self._keep_in_fp32_modules is not None:
                for module in self._keep_in_fp32_modules:
                    if module not in unique_module_names:
                        raise ValueError(
                            f"{module} was specified in the `_keep_in_fp32_modules` list, but is not part of the modules in"
                            f" {self.__class__.__name__}"
                        )

            if self._keep_in_fp32_modules_strict is not None:
                for module in self._keep_in_fp32_modules_strict:
                    if module not in unique_module_names:
                        raise ValueError(
                            f"{module} was specified in the `_keep_in_fp32_modules_strict` list, but is not part of the modules in"
                            f" {self.__class__.__name__}"
                        )