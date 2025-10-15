# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from dataclasses import dataclass
import torch
from torch import nn
from torchtitan.config import JobConfig
from torchtitan.protocols import BaseModelArgs
from torchtitan.tools.logging import logger
from transformers import AutoConfig
from transformers.utils import is_torch_deterministic
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from torchtitan.experiments.transformers_backend.model.hf_llama_like_patch import patch_hf_llama_like
from torchtitan.experiments.transformers_backend.model.hf_moe_like_patch import patch_hf_moe_like

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
        attn_implementation: str = "sdpa_torchtitan",
        **kwargs,
    ):
        super().__init__(attn_implementation=attn_implementation, **kwargs)
        assert titan_args is not None, "titan_args is required"

        active_mappings = {}
        
        active_mappings.update(self._TT_TO_HF_MAPPINGS["base"])
        
        if deepseek_v3_args is not None:
            active_mappings.update(self._TT_TO_HF_MAPPINGS["deepseek_v3"])
        
        self._active_mappings = active_mappings
        
        self._create_dynamic_properties()

        # Set HF attributes from titan_args based on mappings
        for titan_name, hf_name in self._active_mappings.items():
            if hasattr(titan_args, titan_name):
                setattr(self, hf_name, getattr(titan_args, titan_name))

        # Fill all TorchTitan-specific args (no HF equivalent)
        self.multiple_of = titan_args.multiple_of
        self.ffn_dim_multiplier = titan_args.ffn_dim_multiplier
        self.depth_init = titan_args.depth_init
        self.use_flex_attn = titan_args.use_flex_attn
        self.attn_mask_type = titan_args.attn_mask_type

        # HuggingFace specific args
        self.attn_implementation = attn_implementation
        #NOTE:(3outeille):This will force create_causal_mask to return None
        AttentionInterface._global_mapping[attn_implementation] = sdpa_attention_forward

        # Start with passed_args as just titan_args
        self._passed_args = {**titan_args.__dict__, "attn_implementation": attn_implementation}
        self._passed_args.update(kwargs)

        #NOTE(3outeille): Wait for transformers uniformization of MoE args
        if deepseek_v3_args is not None:
            # For DeepSeekV3, setting q_lora_rank to 0 in TorchTitan is equivalent to
            # setting it to None in HuggingFace.
            q_lora_rank = deepseek_v3_args.q_lora_rank
            if q_lora_rank == 0:
                q_lora_rank = None
            deepseek_v3_args.q_lora_rank = q_lora_rank

            self._passed_args.update(**deepseek_v3_args.__dict__)

            self.rope_interleave = deepseek_v3_args.rope_interleave
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
            job_config.model.name,
            attn_implementation=self.attn_implementation,
            trust_remote_code=True
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
        # Check if this is a MoE model by looking for MoE attributes
        is_moe = hasattr(self, 'n_routed_experts')
        
        if is_moe:
            # MoE parameter counting (adapted from DeepSeek V3 implementation)
            nparams_embedding = 0
            nparams_moe_router = 0
            nparams_shared_experts = 0
            nparams_experts = 0
            nparams_dense = 0

            for name, p in model.named_parameters():
                if "embedding" in name:
                    nparams_embedding += p.numel()
                    nparams_dense += p.numel()
                elif "moe.shared_experts" in name:
                    nparams_shared_experts += p.numel()
                elif "moe.router" in name:
                    nparams_moe_router += p.numel()
                elif "moe.experts" in name:
                    nparams_experts += p.numel()
                else:
                    nparams_dense += p.numel()

            nparams_sparse = nparams_moe_router + nparams_shared_experts + nparams_experts
            nparams = nparams_dense + nparams_sparse
            nparams_sparse_active = (
                nparams_moe_router
                + nparams_shared_experts
                + nparams_experts * self.num_experts_per_tok // self.n_routed_experts
            )

            logger.info(
                f"Total parameter count: dense {nparams_dense:,}, "
                f"sparse {nparams_sparse:,}, active {nparams_dense + nparams_sparse_active:,}"
            )

            l, h, q, t = (
                self.n_layers,
                self.n_heads,
                self.dim // self.n_heads,
                seq_len,
            )
            # Use active parameters for FLOPS calculation in MoE
            num_flops_per_token = (
                6 * (nparams_dense - nparams_embedding + nparams_sparse_active)
                + 12 * l * h * q * t
            )
        else:
            # Dense model parameter counting (original implementation)
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
        
        # Attempt to patch model weight initialization based on architecture type
        try:
            model_name_prefix = model_class_name.replace("ForCausalLM", "")
            model_module = importlib.import_module(model_cls.__module__)

            attention_cls = getattr(model_module, f"{model_name_prefix}Attention", None)
            mlp_cls = getattr(model_module, f"{model_name_prefix}MLP", None)
            decoder_layer_cls = getattr(model_module, f"{model_name_prefix}DecoderLayer", None)

            is_moe = hasattr(model_args, "n_routed_experts") #TODO(3outeille): check if this is the most reliable to detect a moe model
            if is_moe:
                moe_cls = getattr(model_module, f"{model_name_prefix}MoE", None)
                required_classes = {
                    "Attention": attention_cls,
                    "MLP": mlp_cls, 
                    "DecoderLayer": decoder_layer_cls,
                    "MoE": moe_cls
                }
                
                if all(required_classes.values()):
                    logger.info(f"Applying MoE-like patch for {model_name_prefix}")
                    patch_hf_moe_like(
                        decoder_layer_cls=decoder_layer_cls,
                        attention_cls=attention_cls,
                        mlp_cls=mlp_cls,
                        moe_cls=moe_cls
                    )
                else:
                    missing = [name for name, cls in required_classes.items() if not cls]
                    logger.warning(
                        f"Could not find required classes ({', '.join(missing)}) for MoE patching of {model_name_prefix}. "
                        "Skipping MoE-like patch."
                    )
            else:
                required_classes = {
                    "Attention": attention_cls,
                    "DecoderLayer": decoder_layer_cls
                }
                
                if all(required_classes.values()):
                    logger.info(f"Applying Llama-like patch for {model_name_prefix}")
                    patch_hf_llama_like(
                        decoder_layer_cls=decoder_layer_cls,
                        attention_cls=attention_cls,
                        mlp_cls=mlp_cls  # mlp_cls can be None
                    )
                else:
                    missing = [name for name, cls in required_classes.items() if not cls]
                    logger.warning(
                        f"Could not find required classes ({', '.join(missing)}) for {model_name_prefix}. "
                        "Skipping Llama-like patch."
                    )

        except Exception as e:
            logger.warning(
                f"Failed to apply agnostic patch for {model_class_name} due to: {e}. "
                "Weight initialization might not match TorchTitan."
            )

        self.model = model_cls(config=model_args)
        self.max_seq_len = model_args.max_seq_len
        
        for layer in self.model.model.layers:
            if hasattr(model_args, "first_k_dense_replace") and layer.layer_idx >= model_args.first_k_dense_replace:
                layer.moe_enabled = True
            else:
                layer.moe_enabled = False

        self.cp_mesh = None
        self.tp_mesh = None
        self.pp_mesh = None

    def set_cp_mesh(self, mesh):
        self.cp_mesh = mesh
    
    def set_tp_mesh(self, mesh):
        self.tp_mesh = mesh
    
    def set_pp_mesh(self, mesh):
        self.pp_mesh = mesh

    @property
    def tok_embeddings(self):
        """Returns the model's embed_tokens, handling different Hugging Face model structures."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):  # Llama-like
            return self.model.model.embed_tokens
        else:
            raise AttributeError("Could not find embed_tokens in the model. Please check the model structure.")

    @tok_embeddings.setter
    def tok_embeddings(self, value):
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):  # Llama-like
            setattr(self.model.model, "embed_tokens", value)
        else:
            raise AttributeError("Could not find embed_tokens in the model. Please check the model structure.")

    @property
    def layers(self):
        """Returns the model's layers, handling different Hugging Face model structures."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):  # Llama-like
            return self.model.model.layers
        else:
            # Add more cases here if needed for other model architectures
            raise AttributeError("Could not find layers in the model. Please check the model structure.")

    @layers.setter
    def layers(self, value):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):  # Llama-like
            setattr(self.model.model, "layers", value)
        else:
            raise AttributeError("Could not find layers in the model. Please check the model structure.")

    @property
    def norm(self):
        """Returns the model's norm, handling different Hugging Face model structures."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):  # Llama-like
            return self.model.model.norm
        elif hasattr(self.model, "model") and hasattr(self.model.model, "final_layernorm"):  # Phi-like
            return self.model.model.final_layernorm
        else:
            raise AttributeError("Could not find norm in the model. Please check the model structure.")

    @norm.setter
    def norm(self, value):
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):  # Llama-like
            setattr(self.model.model, "norm", value)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "final_layernorm"):  # Phi-like
            setattr(self.model.model, "final_layernorm", value)
        else:
            raise AttributeError("Could not find norm in the model. Please check the model structure.")

    @property
    def output(self):
        """Returns the model's output layer, handling different Hugging Face model structures."""
        if hasattr(self.model, "lm_head"):  # For models like LlamaForCausalLM
            return self.model.lm_head
        else:
            # Add more cases here if needed for other model architectures
            raise AttributeError("Could not find output (lm_head) in the model. Please check the model structure.")

    @output.setter
    def output(self, value):
        if hasattr(self.model, "lm_head"):  # For models like LlamaForCausalLM
            setattr(self.model, "lm_head", value)
        else:
            raise AttributeError("Could not find output (lm_head) in the model. Please check the model structure.")

    @property
    def rotary_emb(self):
        """Returns the model's rotary_emb, handling different Hugging Face model structures."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "rotary_emb"):  # Llama-like
            return self.model.model.rotary_emb
        else:
            raise AttributeError("Could not find rotary_emb in the model. Please check the model structure.")

    @rotary_emb.setter
    def rotary_emb(self, value):
        if hasattr(self.model, "model") and hasattr(self.model.model, "rotary_emb"):  # Llama-like
            setattr(self.model.model, "rotary_emb", value)
        else:
            raise AttributeError("Could not find rotary_emb in the model. Please check the model structure.")

    def forward(self, *args, **kwargs):
        local_seq_len = self.max_seq_len
        local_seq_len //= self.cp_mesh.size() if self.cp_mesh is not None and self.cp_mesh.size() > 1 else 1
        kwargs["position_ids"] = torch.arange(local_seq_len, device=args[0].device).unsqueeze(0)
        output = self.model.model(*args, **kwargs)
        output = self.model.lm_head(output.last_hidden_state)
        return output

    def init_weights(self, *args, **kwargs):
        # This method replicates the behavior of the original PreTrainedModel.init_weights,
        # but with a custom weight initialization function that skips nn.Identity modules (when PP is enabled)

        if self.model.config.pruned_heads:
            logger.info("Pruning heads as per model configuration.")
            self.model.prune_heads(self.model.config.pruned_heads)

        original_init_weights_fn = self.model._init_weights

        def selective_init(module):
            # For pipeline parallel, we need to skip nn.Identity modules
            if not isinstance(module, nn.Identity):
                original_init_weights_fn(module)
            else:
                logger.info("Skipping nn.Identity module during weight initialization.")

        self.model.apply(selective_init)

        self.model.tie_weights()
    
    def named_children(self):
        """
        Provides a flattened view of the model's main components,
        making it compatible with TorchTitan's expectations.
        """
        yield "tok_embeddings", self.tok_embeddings
        yield "layers", self.layers
        yield "norm", self.norm
        yield "output", self.output
        yield "rotary_emb", self.rotary_emb

    def __setattr__(self, name, value):
        # If a property with a setter exists for this name, use it.
        # This is to bypass the nn.Module.__setattr__ logic that
        # directly registers modules and skips property setters.
        cls = self.__class__
        if hasattr(cls, name):
            prop = getattr(cls, name)
            if isinstance(prop, property) and prop.fset is not None:
                prop.fset(self, value)
                return

        # Otherwise, fall back to the default nn.Module behavior.
        super().__setattr__(name, value)