# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
import math
from dataclasses import dataclass, field, fields, MISSING

import torch
from torch import nn
from torch.nn import init
from torch.nn.attention.flex_attention import and_masks
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations.flex_attention import flex_attention_forward
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import AttentionInterface, PreTrainedModel

from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.attention import (
    create_attention_mask,
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
)
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict
from torchtitan.tools.logging import logger


class HFFlexKernel(Module):
    """Flex-attention kernel wrapped as a titan Module for declarative TP.

    Runs the flex HOP over q/k/v. Under TP the Module protocol wraps this
    forward with ``local_map`` (driven by the ``ShardingConfig`` set in
    hf_sharding.py): q/k/v arrive head-sharded as DTensors, are converted to
    local tensors so the document ``mask_mod`` -- which closes over a plain
    ``positions`` tensor -- sees plain tensors, and the output is wrapped back
    head-sharded. Expressing the sharding declaratively
    (``ShardingConfig``/``LocalMapConfig``) keeps it consistent with Titan's own
    attention and lets it ride the ``spmd_types`` backend switch, instead of a
    hand-rolled ``local_map`` call.

    The HF attention module and the BlockMask ride as passthrough keyword args
    (non-tensors, so ``local_map`` leaves them untouched). CP is not handled
    here (guarded in ``parallelize_hf_transformers``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: "HFFlexKernel.Config") -> None:
        super().__init__()

    def forward(self, query, key, value, *, module, block_mask=None, **kwargs):
        # flex_attention_forward returns (output, lse); output is already
        # transposed to (b, seq, heads, dim). Return the single tensor so the
        # local_map out_placements is a 1-tuple.
        out, _ = flex_attention_forward(module, query, key, value, block_mask, **kwargs)
        return out


def _flex_attention_torchtitan(module, query, key, value, attention_mask, **kwargs):
    """HF ``AttentionInterface`` shim for flex attention.

    Delegates to the per-attention-module ``HFFlexKernel`` when present (attached
    under TP/EP in hf_sharding.py) so the Module protocol applies the declarative
    ``local_map``. When no kernel is attached (e.g. FSDP-only, where the sharding
    pass does not run), q/k/v are plain tensors and flex runs directly -- no
    mapping needed. CP is not handled here (see the guard in
    ``parallelize_hf_transformers``).
    """
    kernel = getattr(module, "_titan_flex_kernel", None)
    if kernel is None:
        return flex_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )
    out = kernel(query, key, value, module=module, block_mask=attention_mask, **kwargs)
    return out, None


class SliceableModuleDict(ModuleDict):
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

    def init_states(self, **_kwargs) -> None:
        """No-op: HFTransformerModel handles initialization via HF mechanisms."""
        pass


# Define all possible mappings organized by argument type
_TT_TO_HF_MAPPINGS = {
    "dense": {
        # TorchTitan dense model mappings (always available)
        "dim": "hidden_size",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "n_kv_heads": "num_key_value_heads",
        "vocab_size": "vocab_size",
        "norm_eps": "rms_norm_eps",
        "max_seq_len": "max_position_embeddings",
        "eos_id": "eos_token_id",
    },
    # MoE attrs use the same names in TorchTitan and HuggingFace, no remapping needed
    "moe": {},
}

# Declarative list of TorchTitan-only attributes (no HF equivalent)
_TT_SPECIFIC_ATTRIBUTES = [
    "multiple_of",
    "ffn_dim_multiplier",
    "depth_init",
    "use_flex_attn",
    "attn_mask_type",
]

# NOTE: This backend instantiates model classes from the installed
# ``transformers`` package. When a model repo ships an older remote config
# implementation, mixing that remote config with the newer local model code can
# drop compatibility attrs the local model expects. Prefer the local
# ``transformers`` config/model pair for denylisted model types.
_REMOTE_CONFIG_DENYLIST = frozenset({"deepseek_v2", "deepseek_v3"})


def _get_moe_attr_name(layer: nn.Module) -> str | None:
    """Return the attribute name holding the MoE block on a decoder layer.

    Models use ``mlp``.  Returns ``None`` if not present.
    """
    return "mlp" if hasattr(layer, "mlp") else None


class HFTransformerModel(BaseModel):
    # TODO(#ISSUE): Remove after fixing PP backward to skip non-tensor inputs.
    _skip_lm_head: bool = False

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config, PretrainedConfig):
        """Configuration that bridges TorchTitan and HuggingFace Transformers.

        Uses properties to provide TorchTitan-style access while maintaining
        HuggingFace compatibility.
        """

        # Redeclare PretrainedConfig dataclass fields as init=False so
        # ``Configurable.__init_subclass__`` kw_only check passes.  These
        # fields are set dynamically in ``__init__`` via
        # ``PretrainedConfig.__init__``, not through the generated __init__.
        transformers_version: str = field(init=False, default="")
        architectures: list | None = field(init=False, default=None)
        output_hidden_states: bool = field(init=False, default=False)
        return_dict: bool = field(init=False, default=True)
        dtype: str | None = field(init=False, default=None)
        chunk_size_feed_forward: int = field(init=False, default=0)
        is_encoder_decoder: bool = field(init=False, default=False)
        id2label: dict | None = field(init=False, default=None)
        label2id: dict | None = field(init=False, default=None)
        problem_type: str | None = field(init=False, default=None)

        def __init__(
            self,
            model_config,
            # HuggingFace specific args
            attn_implementation: str = "sdpa_torchtitan",
            **kwargs,
        ):
            # Explicitly call PretrainedConfig.__init__ (not via MRO, since
            # Configurable.Config's generated __init__ doesn't chain to it)
            PretrainedConfig.__init__(
                self, attn_implementation=attn_implementation, **kwargs
            )
            # Set param_init and sharding_config before Module.Config.build()
            # accesses them. PretrainedConfig.__getattribute__ doesn't
            # recognize slots inherited from Module.Config.
            self.param_init = (
                None  # noqa: this sets Config.param_init, not Module._param_init
            )
            self.sharding_config = None

            assert model_config is not None, "model_config is required"

            from torchtitan.experiments.transformers_modeling_backend import (
                TitanMoeModelConfig,
            )

            self.is_moe = isinstance(model_config, TitanMoeModelConfig)

            # Create getter/setter dynamically for TT <-> HF attribute mappings
            self._create_getter_setter_dynamically(is_moe=self.is_moe)

            self._titan_injected_model_args = {}
            use_flex = getattr(model_config, "use_flex_attn", False)
            self._configure_hf_attention(attn_implementation, use_flex=use_flex)

            self._initialize_attributes(model_config)

        def build(self, **kwargs):
            """Override build() to use _replace() instead of dataclasses.replace().

            dataclasses.replace() re-invokes __init__, which is incompatible
            with the custom __init__ here (expects titan_dense_config).
            """
            clone = self._replace()
            instance = self._owner(config=clone, **kwargs)
            if self.param_init is not None:
                instance._param_init = self.param_init
            return instance

        def _replace(self, **overrides):
            """Override to use ``copy.copy()`` instead of ``dataclasses.replace()``.

            ``dataclasses.replace()`` re-invokes ``__init__``, which is
            incompatible with the custom ``__init__`` here (it expects
            ``model_config`` and calls ``PretrainedConfig.__init__``).
            A shallow copy preserves all dynamically-set HF attributes.
            """
            clone = copy.copy(self)
            for f in fields(self):
                if f.init:
                    continue
                if f.name in overrides:
                    setattr(clone, f.name, overrides[f.name])
                elif hasattr(self, f.name):
                    setattr(clone, f.name, getattr(self, f.name))
                else:
                    raise TypeError(
                        f"{type(self).__name__} field '{f.name}' "
                        f"(init=False) was not provided via build()"
                    )
            return clone

        def _initialize_attributes(self, model_config):
            """Initialize all model attributes from the config.

            Only stores explicitly-set (non-default) fields in
            ``_titan_injected_model_args`` so that ``update_from_config``
            only overrides HF config values the user intentionally set
            in the flavor, preserving model-specific HF attrs like
            ``qk_head_dim`` or ``n_routed_experts``.
            """
            # Determine which fields were explicitly set (not defaults)
            explicit_overrides = {}
            for f in fields(model_config):
                value = getattr(model_config, f.name)
                default = f.default
                if default is MISSING:
                    # No default — always explicit
                    explicit_overrides[f.name] = value
                elif value != default:
                    explicit_overrides[f.name] = value

            # Set mapped attributes (TorchTitan -> HuggingFace)
            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                if hasattr(model_config, titan_name):
                    setattr(self, hf_name, getattr(model_config, titan_name))

            # Set all remaining attributes directly (TorchTitan-only + MoE)
            for attr_name, value in vars(model_config).items():
                if (
                    not attr_name.startswith("_")
                    and attr_name not in self._tt_to_hf_attribute_map
                ):
                    setattr(self, attr_name, value)

            # Store only EXPLICIT overrides for re-application after HF
            # config load. Mapped attrs use HF names; others use titan names.
            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                if titan_name in explicit_overrides:
                    self._titan_injected_model_args[hf_name] = explicit_overrides[
                        titan_name
                    ]
            for attr_name, value in explicit_overrides.items():
                if attr_name not in self._tt_to_hf_attribute_map:
                    self._titan_injected_model_args[attr_name] = value

        def _configure_hf_attention(
            self, attn_implementation: str, use_flex: bool = False
        ):
            """Configure HuggingFace attention settings.

            Default is SDPA with is_causal and no explicit mask. When ``use_flex``
            is set, route attention through flex instead so a document/packing
            BlockMask can be applied — ``is_causal`` cannot express cross-sample
            (packed) masking. We pass the titan-built BlockMask through HF's
            normal ``attention_mask`` argument (HF returns an already-4D/BlockMask
            mask as-is), so no custom mask plumbing is needed. The custom impl
            name only exists to bypass HF's per-model ``_supports_flex_attn`` gate.
            """
            if use_flex:
                attn_implementation = "flex_torchtitan"
                AttentionInterface._global_mapping[
                    attn_implementation
                ] = _flex_attention_torchtitan
            else:
                # NOTE:(3outeille):This will force create_causal_mask to return None
                AttentionInterface._global_mapping[
                    attn_implementation
                ] = sdpa_attention_forward
            self._titan_injected_model_args["attn_implementation"] = attn_implementation
            self.attn_implementation = attn_implementation
            # HF selects the attention function from ``config._attn_implementation``.
            # PretrainedConfig has no ``attn_implementation`` property in this
            # version, so the line above only sets a dead plain attribute — set the
            # underscore field directly (it is preserved through update_from_config,
            # which skips underscore keys when copying the loaded HF config).
            self._attn_implementation = attn_implementation

        def _create_getter_setter_dynamically(self, is_moe: bool):
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
            if is_moe:
                self._tt_to_hf_attribute_map.update(_TT_TO_HF_MAPPINGS["moe"])

            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                # Identity mappings (e.g. vocab_size -> vocab_size) need no
                # property: the HF attribute is already reachable under the same
                # name. Creating one would make the setter ``setattr(self,
                # hf_name)`` recurse into itself. The map entry is still used by
                # _initialize_attributes to copy the value from the titan config.
                if titan_name == hf_name:
                    continue
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
            config=None,
            **kwargs,
        ):
            training = config.training
            parallelism = config.parallelism
            debug = config.debug
            # Extract HF model ID from the extended config
            hf_model_id = getattr(config, "hf_model", "")
            config_dict, _ = PretrainedConfig.get_config_dict(hf_model_id)
            trust_remote_code = (
                config_dict.get("model_type", "") not in _REMOTE_CONFIG_DENYLIST
            )
            # Load HF config (overwrites our HF attributes)
            hf_model_config = AutoConfig.from_pretrained(
                hf_model_id,
                attn_implementation=self.attn_implementation,
                trust_remote_code=trust_remote_code,
            )

            # For composite (VL) models, use the nested text config so we
            # build the text-only CausalLM instead of the conditional model.
            if hasattr(hf_model_config, "text_config"):
                hf_model_config = hf_model_config.text_config
                hf_model_config.attn_implementation = self.attn_implementation
                # Ensure the text config has architectures for model class lookup
                if not getattr(hf_model_config, "architectures", None):
                    from transformers.models.auto.modeling_auto import (
                        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
                    )

                    model_type = getattr(hf_model_config, "model_type", "")
                    cls_name = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(model_type)
                    if cls_name:
                        hf_model_config.architectures = [cls_name]

            # Explicitly update attributes based on mappings
            for titan_name, hf_name in self._tt_to_hf_attribute_map.items():
                if hasattr(hf_model_config, hf_name):
                    setattr(self, titan_name, getattr(hf_model_config, hf_name))

            # Copy all HF config attributes including computed ones.
            # to_dict() misses attrs computed in __init__ (e.g., DeepSeek V3's
            # qk_head_dim), so we copy from vars() which has them.
            for key, value in vars(hf_model_config).items():
                if not key.startswith("_"):
                    setattr(self, key, value)

            # Copy attribute_map for models that alias config names
            # (e.g., DeepSeek V3 maps num_local_experts → n_routed_experts)
            if hf_model_config.attribute_map:
                self.attribute_map.update(hf_model_config.attribute_map)

            # Re-apply explicitly-set flavor overrides (not defaults)
            for key, value in self._titan_injected_model_args.items():
                setattr(self, key, value)
                # Sync expert count aliases for models that use different naming
                # (e.g., DeepSeek V3 uses n_routed_experts, GLM uses num_local_experts)
                if key == "num_experts" and hasattr(self, "n_routed_experts"):
                    self.n_routed_experts = value

            self.max_seq_len = training.seq_len

            self.deterministic = debug.deterministic

            # Configure HF-specific settings to match TorchTitan settings
            # TODO: false ?
            self.attention_bias = False
            self.mlp_bias = False
            self.use_cache = False
            self.initializer_range = 1.0  # use as std for normal init in embedding

            # When dim is explicitly overridden (e.g. debugmodel), derive the
            # dependent sizes from it. Otherwise keep what AutoConfig loaded from
            # the HF config -- models like Qwen3 decouple head_dim and
            # intermediate_size from hidden_size/num_heads, so deriving them here
            # would silently build the wrong architecture.
            dim_overridden = self._titan_injected_model_args.get("dim") is not None
            if (
                dim_overridden
                and not getattr(self, "is_moe", False)
                and not hasattr(self, "inter_dim")
            ):
                ffn_hidden_size = 4 * self.dim
                ffn_hidden_size = int(2 * ffn_hidden_size / 3)
                if self.ffn_dim_multiplier is not None:
                    ffn_hidden_size = int(self.ffn_dim_multiplier * ffn_hidden_size)
                self.intermediate_size = self.multiple_of * (
                    (ffn_hidden_size + self.multiple_of - 1) // self.multiple_of
                )

            # MLA models (DeepSeek V3, GLM-5) set head_dim = qk_rope_head_dim
            # in the HF config for RoPE; don't clobber it with the standard
            # computation. Also force num_key_value_heads = num_attention_heads
            # because MLA has no GQA -- the KV LoRA path always produces
            # num_attention_heads heads.
            if hasattr(self, "qk_rope_head_dim"):
                self.num_key_value_heads = self.num_attention_heads
                # Ensure head_dim is set for MLA models; remote configs
                # may not compute it in __post_init__.
                if not getattr(self, "head_dim", None):
                    self.head_dim = self.qk_rope_head_dim
            elif dim_overridden:
                # dim explicitly overridden: derive head_dim from it.
                self.head_dim = self.dim // self.num_attention_heads
            elif not getattr(self, "head_dim", None):
                # HF config did not provide head_dim; use the standard derivation.
                self.head_dim = self.dim // self.num_attention_heads

            # Ensure expert groups are consistent with (possibly overridden)
            # num_experts for models with group-level routing (DeepSeek V3).
            # Each group needs >= 2 experts for the in-group topk(2).
            if hasattr(self, "n_group") and hasattr(self, "n_routed_experts"):
                while self.n_group > 1 and self.n_routed_experts // self.n_group < 2:
                    self.n_group //= 2
                if hasattr(self, "topk_group"):
                    self.topk_group = min(self.topk_group, self.n_group)

            return self

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                model,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                head_dims=self.head_dim,
                seq_len=seq_len,
            )

    def __init__(self, config: Config):
        super().__init__()

        # Try to import the model class dynamically from the transformers library if not found in globals
        model_class_name = config.architectures[0]
        model_cls = globals().get(model_class_name, None)
        if model_cls is None:
            try:
                transformers_mod = importlib.import_module("transformers")
                model_cls = getattr(transformers_mod, model_class_name)
            except (ImportError, AttributeError):
                model_cls = None

        if model_cls is None:
            # Fallback: resolve via model_type → Auto mapping.
            # Handles cases where the config's architecture name doesn't match
            # the actual class name.
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            )

            model_type = getattr(config, "model_type", "")
            resolved_name = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(model_type)
            if resolved_name:
                transformers_mod = importlib.import_module("transformers")
                model_cls = getattr(transformers_mod, resolved_name, None)
                if model_cls is not None:
                    model_class_name = resolved_name

            if model_cls is None:
                raise ImportError(
                    f"Could not find model class '{model_class_name}' in globals or transformers. "
                    f"Make sure the class is available."
                )

        # Attempt to patch model weight initialization based on architecture type
        try:
            model_name_prefix = model_class_name.replace("ForCausalLM", "")
            model_module = importlib.import_module(model_cls.__module__)

            attention_cls = getattr(model_module, f"{model_name_prefix}Attention", None)
            mlp_cls = getattr(model_module, f"{model_name_prefix}MLP", None)
            decoder_layer_cls = getattr(
                model_module, f"{model_name_prefix}DecoderLayer", None
            )

            # Discover MoE-specific classes
            experts_cls = getattr(model_module, f"{model_name_prefix}Experts", None)
            router_cls = getattr(model_module, f"{model_name_prefix}TopKRouter", None)

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
                    experts_cls=experts_cls,
                    router_cls=router_cls,
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

        # Select the HF experts forward kernel from the explicit request in
        # TitanMoeModelConfig.experts_implementation. Honor it or fail — never
        # silently substitute a different kernel than the user asked for.
        #
        # - "native": use the model's own built-in experts unchanged (valid for
        #   every model; the only valid choice for models that can't take a
        #   settable implementation).
        # - "grouped_mm"/"batched_mm"/"eager": require a model that supports a
        #   settable experts implementation (the @use_experts_implementation
        #   decorator, probed up front as a classmethod). If the model can't, the
        #   request cannot be honored -> raise rather than ignore it.
        #
        # All paths work transparently with hook-based EP/TP because the hooks
        # preserve the (hidden_states, top_k_index, top_k_weights) interface.
        if config.is_moe:
            impl = config.experts_implementation
            if impl == "native":
                config._experts_implementation = None
            elif model_cls._can_set_experts_implementation():
                config._experts_implementation = impl
            else:
                raise ValueError(
                    f"{model_class_name} does not support a settable experts "
                    f"implementation, so experts_implementation='{impl}' cannot "
                    "be honored. Set experts_implementation='native' to use the "
                    "model's built-in experts kernel."
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
            # Detect MoE layers by checking for gate/router and experts sub-modules.
            # Gemma4 has router/experts as siblings of the dense MLP at the
            # layer level (not inside ``mlp``).
            moe_attr = _get_moe_attr_name(layer)
            moe_module = getattr(layer, moe_attr, None) if moe_attr else None
            if moe_module is not None:
                has_gate = hasattr(moe_module, "gate") or hasattr(moe_module, "router")
                layer.moe_enabled = has_gate and hasattr(moe_module, "experts")
            else:
                layer.moe_enabled = False

            # Layer-level MoE: router and experts are direct children of the
            # decoder layer, not nested inside the MLP block (e.g. Gemma4).
            if not layer.moe_enabled:
                has_layer_router = hasattr(layer, "router") or hasattr(layer, "gate")
                has_layer_experts = hasattr(layer, "experts")
                if has_layer_router and has_layer_experts:
                    layer.moe_enabled = True
                    layer._layer_level_moe = True

        if config.is_moe:
            from .moe_replacement import prepare_native_moe_configs

            prepare_native_moe_configs(self, config)

    def set_cp_mesh(self, mesh):
        self.cp_mesh = mesh

    def _patch_hf_llama_like(
        self,
        decoder_layer_cls,
        attention_cls,
        mlp_cls=None,
        experts_cls=None,
        router_cls=None,
    ):
        """
        This patch modifies a Hugging Face Llama-like model's weight initialization to match
        the initialization scheme used in TorchTitan. This is crucial for ensuring
        bit-for-bit reproducibility when converting checkpoints between the
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
                # Propagate to shared experts (e.g., Qwen2 MoE, DeepSeek V3)
                for shared_name in ("shared_expert", "shared_experts"):
                    shared = getattr(self.mlp, shared_name, None)
                    if shared is not None:
                        shared.layer_idx = layer_idx

        def _initialize_weights_patched(self, module, is_remote_code: bool = False):
            # NOTE(3outeille): monkey-patch PreTrainedModel to handle meta device initialization correctly
            # The default _initialize_weights sets _is_hf_initialized = True even on a meta device,
            # which prevents subsequent proper initialization.
            #
            # This mirrors HF's PreTrainedModel._initialize_weights and only adds
            # the meta-device early-return below. The `is_remote_code` arg and the
            # branch that follows are copied verbatim from HF: transformers 5.x
            # calls this via smart_apply as `fn(module, self.is_remote_code())`,
            # so the replacement must accept the arg and keep the remote-code
            # guard (remote _init_weights may write params in place, so already
            # initialized modules must be skipped). Source (transformers v5.9.0):
            # https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/modeling_utils.py#L2464
            if getattr(module, "_is_hf_initialized", False):
                return

            if (
                is_remote_code
                and all(
                    getattr(param, "_is_hf_initialized", False)
                    for param in module.parameters(recurse=False)
                )
                and all(
                    getattr(buffer, "_is_hf_initialized", False)
                    for buffer in module.buffers(recurse=False)
                    if buffer is not None
                )
            ):
                module._is_hf_initialized = True
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
                # (some models use q_a_proj/q_b_proj instead of q_proj)
                for proj_name in ["q_proj", "k_proj", "v_proj"]:
                    proj = getattr(module, proj_name, None)
                    if proj is None:
                        continue
                    nn.init.trunc_normal_(proj.weight, mean=0.0, std=0.02)
                    if proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(proj.bias, -bound, bound)

                # Handle different names for the output projection layer
                o_proj = getattr(module, "o_proj", getattr(module, "dense", None))
                if o_proj is not None:
                    nn.init.trunc_normal_(o_proj.weight, mean=0.0, std=init_std)
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
                    nn.init.trunc_normal_(gate_proj.weight, mean=0.0, std=0.02)
                    if gate_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(gate_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(gate_proj.bias, -bound, bound)
                # up_proj and down_proj (or fc2) use the depth-dependent init_std.
                if up_proj is not None:
                    nn.init.trunc_normal_(up_proj.weight, mean=0.0, std=init_std)
                    if up_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(up_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(up_proj.bias, -bound, bound)
                if down_proj is not None:
                    nn.init.trunc_normal_(down_proj.weight, mean=0.0, std=init_std)
                    if down_proj.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(down_proj.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        init.uniform_(down_proj.bias, -bound, bound)

            elif experts_cls and isinstance(module, experts_cls):
                # MoE expert weights are 3D parameter tensors (not nn.Linear)
                if hasattr(module, "gate_up_proj"):
                    nn.init.trunc_normal_(module.gate_up_proj, mean=0.0, std=0.02)
                if hasattr(module, "down_proj"):
                    nn.init.trunc_normal_(module.down_proj, mean=0.0, std=0.02)

            elif router_cls and isinstance(module, router_cls):
                if hasattr(module, "weight"):
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)

            elif module is getattr(
                self, "lm_head", None
            ):  # TODO(3outeille): find a better way to detect lm_head
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
                # When tie_word_embeddings is True, use lm_head initialization
                if (
                    hasattr(config, "tie_word_embeddings")
                    and config.tie_word_embeddings
                ):
                    final_out_std = config.hidden_size**-0.5
                    cutoff_factor = 3
                    nn.init.trunc_normal_(
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
    def lm_head(self):
        """Returns the model's output layer, handling different Hugging Face model structures."""
        if hasattr(self.model, "lm_head"):  # For models like LlamaForCausalLM
            return self.model.lm_head
        else:
            raise AttributeError(
                "Could not find lm_head in the model. Please check the model structure."
            )

    @lm_head.setter
    def lm_head(self, value):
        if hasattr(self.model, "lm_head"):  # For models like LlamaForCausalLM
            self.model.lm_head = value
        else:
            raise AttributeError(
                "Could not find lm_head in the model. Please check the model structure."
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

    def get_attention_masks(self, positions: torch.Tensor):
        """Build a document-causal flex BlockMask for packed sequences.

        Returns None unless flex attention is enabled (``use_flex_attn``).
        ``forward`` calls this and passes the result through as the HF
        ``attention_mask``. The mask is causal AND same-document, so packed
        samples don't attend across boundaries -- which ``is_causal`` alone
        (the SDPA path) cannot express.
        """
        if not getattr(self.model.config, "use_flex_attn", False):
            return None
        mask_mod = and_masks(
            get_causal_mask_mod(),
            get_efficient_causal_mask_mod_for_packed_document(positions),
        )
        batch_size, seq_len = positions.shape
        return create_attention_mask(
            mask_mod,
            batch_size,
            None,
            seq_len,
            seq_len,
            device=positions.device,
            BLOCK_SIZE=128,
            separate_full_blocks=not is_in_batch_invariant_mode(),
        )

    def forward(self, *args, **kwargs):
        positions = kwargs.pop("positions", None)
        attention_masks = kwargs.pop("attention_masks", None)
        use_flex = getattr(self.model.config, "use_flex_attn", False)

        if use_flex and positions is not None:
            # Per-document positions (reset at packed-sample boundaries) drive
            # RoPE under flex; the BlockMask handles cross-sample masking.
            kwargs["position_ids"] = positions
            # Build the document-causal BlockMask here rather than in the core
            # trainer: the trainer only builds masks for Decoder.Config models,
            # so this backend opts in by building its own (keeps trainer.py free
            # of HF-backend special-casing). This outer forward runs eager (only
            # the inner transformer blocks are compiled), so BlockMask creation
            # here is safe.
            if attention_masks is None:
                attention_masks = self.get_attention_masks(positions)
        else:
            local_seq_len = self.max_seq_len
            local_seq_len //= (
                self.cp_mesh.size()
                if self.cp_mesh is not None and self.cp_mesh.size() > 1
                else 1
            )
            kwargs["position_ids"] = torch.arange(
                local_seq_len, device=args[0].device
            ).unsqueeze(0)

        if attention_masks is not None:
            # HF returns an already-4D mask / BlockMask as-is (see
            # masking_utils._preprocess_mask_arguments), so the titan-built
            # BlockMask flows straight through to the flex attention function.
            kwargs["attention_mask"] = attention_masks

        output = self.model.model(*args, **kwargs)

        if self._skip_lm_head:
            return output.last_hidden_state
        output = self.model.lm_head(output.last_hidden_state)
        return output

    def verify_module_protocol(self) -> None:
        """Skip recursive verification for HuggingFace model internals.

        HF PreTrainedModel submodules are plain nn.Module and cannot
        conform to the Module protocol. Initialization is handled
        entirely by HF's own _init_weights mechanism.
        """
        pass

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
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

        # HF rotary embeddings compute their `inv_freq` buffer in __init__, not in
        # `_init_weights`. With meta-device init + `to_empty()`, that buffer is
        # left uninitialized (zeros), which silently disables RoPE (no positional
        # information -> near-random outputs). Recompute it from each rotary
        # module's `rope_init_fn` so positions work after materialization.
        for module in self.model.modules():
            rope_init_fn = getattr(module, "rope_init_fn", None)
            if rope_init_fn is not None and hasattr(module, "inv_freq"):
                device = module.inv_freq.device
                inv_freq, attention_scaling = rope_init_fn(module.config, device)
                module.inv_freq.copy_(
                    inv_freq.to(device=device, dtype=module.inv_freq.dtype)
                )
                module.attention_scaling = attention_scaling

        # TODO(3outeille): For pipeline parallel, only tie weights if both input and output embeddings are on the same device
        # Maybe better way of handling this?
        if not isinstance(self.tok_embeddings, nn.Identity) and not isinstance(
            self.lm_head, nn.Identity
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
        yield "lm_head", self.lm_head
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
