# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torch import nn
from torchtitan.config.job_config import JobConfig
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols import BaseModelArgs
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface


@dataclass
class TitanDenseModelArgs:
    """Arguments for the base TorchTitan model."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int | None = None
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 2048
    depth_init: bool = True
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"


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

    def __init__(
        self,
        titan_dense_args,
        # HuggingFace specific args
        attn_implementation: str = "sdpa_torchtitan",
        **kwargs,
    ):
        super().__init__(attn_implementation=attn_implementation, **kwargs)
        assert titan_dense_args is not None, "titan_dense_args is required"

        # Create getter/setter dynamically for TT <-> HF attribute mappings
        self._create_getter_setter_dynamically(has_moe=False)

        self._titan_injected_model_args = {}
        self._configure_hf_attention(attn_implementation)

        self._initialize_dense_attributes(titan_dense_args)

    def _initialize_dense_attributes(self, titan_dense_args):
        """Initialize all dense model attributes."""
        # Set mapped attributes (TorchTitan <-> HuggingFace)
        for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
            if hasattr(titan_dense_args, titan_name):
                value = getattr(titan_dense_args, titan_name)
                setattr(self, hf_name, value)

        # Set TorchTitan-only attributes
        for attr_name in self._TT_SPECIFIC_ATTRIBUTES:
            if hasattr(titan_dense_args, attr_name):
                setattr(self, attr_name, getattr(titan_dense_args, attr_name))

        # Update passed_args
        self._titan_injected_model_args.update(titan_dense_args.__dict__)

    def _configure_hf_attention(self, attn_implementation: str):
        """Configure HuggingFace attention settings."""
        self._titan_injected_model_args["attn_implementation"] = attn_implementation
        self.attn_implementation = attn_implementation
        # NOTE:(3outeille):This will force create_causal_mask to return None
        AttentionInterface._global_mapping[attn_implementation] = sdpa_attention_forward

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
        self._tt_to_hf_attribute_map = dict(self._TT_TO_HF_MAPPINGS["dense"])
        if has_moe:
            self._tt_to_hf_attribute_map.update(self._TT_TO_HF_MAPPINGS["moe"])

        for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
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
            for k in sorted(self._titan_injected_model_args.keys())
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

        self.max_seq_len = job_config.training.seq_len

        self.deterministic = job_config.debug.deterministic

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

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        return get_dense_model_nparams_and_flops(
            self, model, head_dims=self.head_dim, seq_len=seq_len
        )
