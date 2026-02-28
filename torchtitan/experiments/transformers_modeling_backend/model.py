# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import init
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface, PreTrainedModel

from torchtitan.models.common import trunc_normal_
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.model import BaseModel
from torchtitan.tools.logging import logger


class SliceableModuleDict(nn.ModuleDict):
    """
    A ModuleDict that supports slicing like ModuleList.
    Keys are expected to be string representations of integers (e.g., "0", "1", "2").
    """

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Handle slicing: convert slice to list of keys
            keys = sorted(
                self.keys(), key=lambda x: int(x) if x.isdigit() else float("inf")
            )
            sliced_keys = keys[key]
            # Return a new SliceableModuleDict with the sliced items
            return SliceableModuleDict({k: self[k] for k in sliced_keys})
        return super().__getitem__(key)

    def __iter__(self):
        # Iterate over values in sorted order by key (as integers)
        keys = sorted(
            self.keys(), key=lambda x: int(x) if x.isdigit() else float("inf")
        )
        for key in keys:
            yield self[key]

    def __len__(self):
        return len(self._modules)


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
    }
}

# Declarative list of TorchTitan-only attributes (no HF equivalent)
_TT_SPECIFIC_ATTRIBUTES = [
    "multiple_of",
    "ffn_dim_multiplier",
    "depth_init",
    "use_flex_attn",
    "attn_mask_type",
]


class HFTransformerModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config, PretrainedConfig):
        """Configuration that bridges TorchTitan and HuggingFace Transformers.

        Uses properties to provide TorchTitan-style access while maintaining
        HuggingFace compatibility.
        """

        def __init__(
            self,
            titan_dense_config,
            # HuggingFace specific args
            attn_implementation: str = "sdpa_torchtitan",
            **kwargs,
        ):
            # Explicitly call PretrainedConfig.__init__ (not via MRO, since
            # Configurable.Config's generated __init__ doesn't chain to it)
            PretrainedConfig.__init__(
                self, attn_implementation=attn_implementation, **kwargs
            )
            assert titan_dense_config is not None, "titan_dense_config is required"

            # Create getter/setter dynamically for TT <-> HF attribute mappings
            self._create_getter_setter_dynamically(has_moe=False)

            self._titan_injected_model_args = {}
            self._configure_hf_attention(attn_implementation)

            self._initialize_dense_attributes(titan_dense_config)

        def _initialize_dense_attributes(self, titan_dense_config):
            """Initialize all dense model attributes."""
            # Set mapped attributes (TorchTitan <-> HuggingFace)
            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                if hasattr(titan_dense_config, titan_name):
                    value = getattr(titan_dense_config, titan_name)
                    setattr(self, hf_name, value)

            # Set TorchTitan-only attributes
            for attr_name in _TT_SPECIFIC_ATTRIBUTES:
                if hasattr(titan_dense_config, attr_name):
                    setattr(self, attr_name, getattr(titan_dense_config, attr_name))

            # Update passed_args
            self._titan_injected_model_args.update(titan_dense_config.__dict__)

        def _configure_hf_attention(self, attn_implementation: str):
            """Configure HuggingFace attention settings."""
            self._titan_injected_model_args["attn_implementation"] = attn_implementation
            self.attn_implementation = attn_implementation
            # NOTE:(3outeille):This will force create_causal_mask to return None
            AttentionInterface._global_mapping[
                attn_implementation
            ] = sdpa_attention_forward

        def _create_getter_setter_dynamically(self, has_moe: bool):
            """
            Create properties dynamically based on tt and hf attribute mappings.
            For example, creates a property 'dim' that reads/writes to 'hidden_size'.
            """

            def _create_property(hf_name: str) -> property:
                def getter(self):
                    return getattr(self, hf_name)

                def setter(self, value):
                    setattr(self, hf_name, value)

                return property(getter, setter)

            # Setup attribute mappings
            self._tt_to_hf_attribute_map = dict(_TT_TO_HF_MAPPINGS["dense"])
            if has_moe:
                self._tt_to_hf_attribute_map.update(_TT_TO_HF_MAPPINGS["moe"])

            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                # Create getter/setter for attribute that don't already exist
                if not hasattr(self.__class__, titan_name):
                    setattr(self.__class__, titan_name, _create_property(hf_name))

        def __repr__(self) -> str:
            args_lines = [
                f"{k}={getattr(self, k)!r}"
                for k in sorted(self._titan_injected_model_args.keys())
                if hasattr(self, k)
            ]
            args_str = "\n".join(args_lines)
            return f"{self.__class__.__name__}(\n{args_str}\n)"

        def update_from_config(
            self,
            *,
            trainer_config=None,
            **kwargs,
        ):
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug
            # Extract HF model ID from the extended trainer_config
            hf_model_id = getattr(trainer_config, "hf_model", "")
            # Load HF config (overwrites our HF attributes)
            hf_model_config = AutoConfig.from_pretrained(
                hf_model_id,
                attn_implementation=self.attn_implementation,
                trust_remote_code=True,
            )

            # Explicitly update attributes based on mappings
            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                if hasattr(hf_model_config, hf_name):
                    setattr(self, titan_name, getattr(hf_model_config, hf_name))

            # Copy any other attributes that might not be in the mapping
            for key, value in hf_model_config.to_dict().items():
                setattr(self, key, value)

            # Update our attributes with the passed args from flavors
            for key, value in self._titan_injected_model_args.items():
                if hasattr(self, key) and value is not None:
                    setattr(self, key, value)

            self.max_seq_len = training.seq_len

            self.deterministic = debug.deterministic

            # Configure HF-specific settings to match TorchTitan settings
            # TODO: false ?
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

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                self,
                model,
                n_heads=self.n_heads,
                head_dims=self.head_dim,
                seq_len=seq_len,
            )

    def __init__(self, config: Config):
        super().__init__()

        # NOTE(3outeille): This prevents Hugging Face modeling from initializing ROPE (inv_freq) buffers to NaN.
        # Needed when loading from seed checkpoint.
        if hasattr(config, "deterministic") and config.deterministic:
            torch.utils.deterministic.fill_uninitialized_memory = False

        # Try to import the model class dynamically from the transformers library if not found in globals
        model_class_name = config.architectures[0]
        model_cls = globals().get(model_class_name, None)
        if model_cls is None:
            try:
                transformers_mod = importlib.import_module("transformers")
                model_cls = getattr(transformers_mod, model_class_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Could not find model class '{model_class_name}' in globals or transformers. "
                    f"Make sure the class is available. Original error: {e}"
                ) from e

        # Attempt to patch model weight initialization based on architecture type
        try:
            model_name_prefix = model_class_name.replace("ForCausalLM", "")
            model_module = importlib.import_module(model_cls.__module__)

            attention_cls = getattr(model_module, f"{model_name_prefix}Attention", None)
            mlp_cls = getattr(model_module, f"{model_name_prefix}MLP", None)
            decoder_layer_cls = getattr(
                model_module, f"{model_name_prefix}DecoderLayer", None
            )

            required_classes = {
                "Attention": attention_cls,
                "DecoderLayer": decoder_layer_cls,
            }

            if all(required_classes.values()):
                logger.info(f"Applying Llama-like patch for {model_name_prefix}")
                self._patch_hf_llama_like(
                    decoder_layer_cls=decoder_layer_cls,
                    attention_cls=attention_cls,
                    mlp_cls=mlp_cls,  # mlp_cls can be None
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

        self.model = model_cls(config=config)
        self.max_seq_len = config.max_seq_len
        self.cp_mesh = None

        # Convert ModuleList to ModuleDict to preserve original indices
        # This ensures state dict keys match checkpoint keys
        if isinstance(self.model.model.layers, nn.ModuleList):
            self.model.model.layers = SliceableModuleDict(
                {str(i): layer for i, layer in enumerate(self.model.model.layers)}
            )

        for layer in self.model.model.layers.values():
            layer.moe_enabled = False

    def set_cp_mesh(self, mesh):
        self.cp_mesh = mesh

    def _patch_hf_llama_like(self, decoder_layer_cls, attention_cls, mlp_cls=None):
        """
        This patch modifies a Hugging Face Llama-like model's weight initialization to match
        the initialization scheme used in TorchTitan. This is crucial for ensuring
        bit-for-bit reproducibility when converting checkpoints between the native
        TorchTitan format and the Hugging Face format.

        The patch targets the following aspects of the model:
        - `PreTrainedModel._initialize_weights`: Handles meta device initialization correctly.
        - `PreTrainedModel._init_weights`: Implements TorchTitan's specific initialization
          for attention, MLP, embedding, and layer norm layers. This includes depth-dependent
          initialization for attention and MLP layers.
        - `DecoderLayer.__init__`: Adds `layer_idx` to attention and MLP modules within
          each decoder layer, which is required for the depth-dependent initialization.
        """

        _original_decoder_layer_init = decoder_layer_cls.__init__

        def _decoder_layer_init_patched(self, config: PretrainedConfig, layer_idx: int):
            _original_decoder_layer_init(self, config, layer_idx)
            self.layer_idx = layer_idx
            # Ensure both attention and mlp modules have layer_idx for depth-based init
            if hasattr(self, "self_attn"):
                self.self_attn.layer_idx = layer_idx
            # some models might not have mlp in each layer
            if hasattr(self, "mlp") and self.mlp is not None:
                self.mlp.layer_idx = layer_idx

        def _initialize_weights_patched(self, module):
            # NOTE(3outeille): monkey-patch PreTrainedModel to handle meta device initialization correctly
            # The default _initialize_weights sets _is_hf_initialized = True even on a meta device,
            # which prevents subsequent proper initialization.
            if getattr(module, "_is_hf_initialized", False):
                return

            for param in module.parameters(recurse=True):
                if param.device.type == "meta":
                    return

            # If not on a meta device, call the original weight initialization
            self._init_weights(module)
            module._is_hf_initialized = True

        def _init_weights_patched(self, module):
            """
            Patched version of _init_weights to match TorchTitan's initialization for Llama-like models.
            `self` is a PreTrainedModel instance.
            """
            config = self.config
            # Build tuple of classes to check for layer_idx-based init_std calculation
            layer_idx_classes = [attention_cls]
            if mlp_cls:
                layer_idx_classes.append(mlp_cls)
            layer_idx_classes = tuple(layer_idx_classes)

            if isinstance(module, layer_idx_classes):
                if not hasattr(module, "layer_idx"):
                    raise ValueError(
                        f"Module {module} does not have a layer_idx attribute"
                    )

                layer_idx = module.layer_idx

                if hasattr(config, "depth_init") and config.depth_init:
                    init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5
                else:
                    init_std = 0.02 / (2 * config.num_hidden_layers) ** 0.5

            if isinstance(module, attention_cls):
                # Initialize weights and biases for q, k, v projections
                for proj_name in ["q_proj", "k_proj", "v_proj"]:
                    proj = getattr(module, proj_name)
                    trunc_normal_(proj.weight, mean=0.0, std=0.02)
                    if proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(proj.bias, -bound, bound)

                # Handle different names for the output projection layer
                o_proj = getattr(module, "o_proj", getattr(module, "dense", None))
                if o_proj is not None:
                    trunc_normal_(o_proj.weight, mean=0.0, std=init_std)
                    if o_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(o_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(o_proj.bias, -bound, bound)

            elif mlp_cls and isinstance(module, mlp_cls):
                # Handle different names for MLP layers
                gate_proj = getattr(module, "gate_proj", getattr(module, "fc1", None))
                up_proj = getattr(module, "up_proj", None)
                down_proj = getattr(module, "down_proj", getattr(module, "fc2", None))

                # gate_proj (or fc1) should always use std=0.02 for numerical stability.
                if gate_proj is not None:
                    trunc_normal_(gate_proj.weight, mean=0.0, std=0.02)
                    if gate_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(gate_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(gate_proj.bias, -bound, bound)
                # up_proj and down_proj (or fc2) use the depth-dependent init_std.
                if up_proj is not None:
                    trunc_normal_(up_proj.weight, mean=0.0, std=init_std)
                    if up_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(up_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(up_proj.bias, -bound, bound)
                if down_proj is not None:
                    trunc_normal_(down_proj.weight, mean=0.0, std=init_std)
                    if down_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(down_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(down_proj.bias, -bound, bound)

            elif module is getattr(
                self, "lm_head", None
            ):  # TODO(3outeille): find a better way to detect lm_head
                final_out_std = config.hidden_size**-0.5
                cutoff_factor = 3
                trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )
                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.Embedding):
                # When tie_word_embeddings is True, use lm_head initialization
                if (
                    hasattr(config, "tie_word_embeddings")
                    and config.tie_word_embeddings
                ):
                    final_out_std = config.hidden_size**-0.5
                    cutoff_factor = 3
                    trunc_normal_(
                        module.weight,
                        mean=0.0,
                        std=final_out_std,
                        a=-cutoff_factor * final_out_std,
                        b=cutoff_factor * final_out_std,
                    )
                else:
                    std = config.initializer_range
                    module.weight.data.normal_(mean=0.0, std=std)

                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

            elif (
                isinstance(
                    module,
                    (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
                )
                or "LayerNorm" in module.__class__.__name__
                or "RMSNorm" in module.__class__.__name__
            ):
                # Norms can exist without weights (in which case they are None from torch primitives)
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.data.fill_(1.0)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.zero_()

        decoder_layer_cls.__init__ = _decoder_layer_init_patched
        PreTrainedModel._init_weights = _init_weights_patched
        PreTrainedModel._initialize_weights = _initialize_weights_patched

    @property
    def tok_embeddings(self):
        """Returns the model's embed_tokens, handling different Hugging Face model structures."""
        if hasattr(self.model, "model") and hasattr(
            self.model.model, "embed_tokens"
        ):  # Llama-like
            return self.model.model.embed_tokens
        else:
            raise AttributeError(
                "Could not find embed_tokens in the model. Please check the model structure."
            )

    @tok_embeddings.setter
    def tok_embeddings(self, value):
        if hasattr(self.model, "model") and hasattr(
            self.model.model, "embed_tokens"
        ):  # Llama-like
            self.model.model.embed_tokens = value
        else:
            raise AttributeError(
                "Could not find embed_tokens in the model. Please check the model structure."
            )

    @property
    def layers(self):
        """Returns the model's layers, handling different Hugging Face model structures."""
        if hasattr(self.model, "model") and hasattr(
            self.model.model, "layers"
        ):  # Llama-like
            return self.model.model.layers
        else:
            # Add more cases here if needed for other model architectures
            raise AttributeError(
                "Could not find layers in the model. Please check the model structure."
            )

    @layers.setter
    def layers(self, value):
        if hasattr(self.model, "model") and hasattr(
            self.model.model, "layers"
        ):  # Llama-like
            self.model.model.layers = value
        else:
            raise AttributeError(
                "Could not find layers in the model. Please check the model structure."
            )

    @property
    def norm(self):
        """Returns the model's norm, handling different Hugging Face model structures."""
        if hasattr(self.model, "model") and hasattr(
            self.model.model, "norm"
        ):  # Llama-like
            return self.model.model.norm
        elif hasattr(self.model, "model") and hasattr(
            self.model.model, "final_layernorm"
        ):  # Phi-like
            return self.model.model.final_layernorm
        else:
            raise AttributeError(
                "Could not find norm in the model. Please check the model structure."
            )

    @norm.setter
    def norm(self, value):
        if hasattr(self.model, "model") and hasattr(
            self.model.model, "norm"
        ):  # Llama-like
            self.model.model.norm = value
        elif hasattr(self.model, "model") and hasattr(
            self.model.model, "final_layernorm"
        ):  # Phi-like
            self.model.model.final_layernorm = value
        else:
            raise AttributeError(
                "Could not find norm in the model. Please check the model structure."
            )

    @property
    def output(self):
        """Returns the model's output layer, handling different Hugging Face model structures."""
        if hasattr(self.model, "lm_head"):  # For models like LlamaForCausalLM
            return self.model.lm_head
        else:
            # Add more cases here if needed for other model architectures
            raise AttributeError(
                "Could not find output (lm_head) in the model. Please check the model structure."
            )

    @output.setter
    def output(self, value):
        if hasattr(self.model, "lm_head"):  # For models like LlamaForCausalLM
            self.model.lm_head = value
        else:
            raise AttributeError(
                "Could not find output (lm_head) in the model. Please check the model structure."
            )

    @property
    def rotary_emb(self):
        """Returns the model's rotary_emb, handling different Hugging Face model structures."""
        if hasattr(self.model, "model") and hasattr(
            self.model.model, "rotary_emb"
        ):  # Llama-like
            return self.model.model.rotary_emb
        else:
            raise AttributeError(
                "Could not find rotary_emb in the model. Please check the model structure."
            )

    @rotary_emb.setter
    def rotary_emb(self, value):
        if hasattr(self.model, "model") and hasattr(
            self.model.model, "rotary_emb"
        ):  # Llama-like
            self.model.model.rotary_emb = value
        else:
            raise AttributeError(
                "Could not find rotary_emb in the model. Please check the model structure."
            )

    def forward(self, *args, **kwargs):
        local_seq_len = self.max_seq_len
        local_seq_len //= (
            self.cp_mesh.size()
            if self.cp_mesh is not None and self.cp_mesh.size() > 1
            else 1
        )
        kwargs["position_ids"] = torch.arange(
            local_seq_len, device=args[0].device
        ).unsqueeze(0)
        output = self.model.model(*args, **kwargs)
        output = self.model.lm_head(output.last_hidden_state)
        return output

    def init_weights(self, *args, **kwargs):
        # This method replicates the behavior of the original PreTrainedModel.init_weights,
        # but with a custom weight initialization function that skips nn.Identity modules (when PP is enabled)

        if getattr(self.model.config, "pruned_heads", None):
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

        # TODO(3outeille): For pipeline parallel, only tie weights if both input and output embeddings are on the same device
        # Maybe better way of handling this?
        if not isinstance(self.tok_embeddings, nn.Identity) and not isinstance(
            self.output, nn.Identity
        ):
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
