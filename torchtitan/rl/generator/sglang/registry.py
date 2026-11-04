# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Register the TorchTitan model class, config parser and attention backend with SGLang.

Per-engine TorchTitan config (``model_config``, ``parallelism``, ...) is captured in the
closure of a dynamic model class; SGLang's HF-shaped config carries only the fields
SGLang reads before it builds the model.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import PrecisionConfig
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.rl.generator.engine import ModelStateDictLoadMode
from torchtitan.rl.generator.inference_model import get_attention_dimensions
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig

logger = logging.getLogger(__name__)

# The HF ``architectures`` entry SGLang's ModelRegistry resolves to the model class.
TORCHTITAN_ARCHITECTURE = "TorchTitanForCausalLM"

# ``ServerArgs.model_config_parser`` value selecting the TorchTitan config parser.
TORCHTITAN_CONFIG_PARSER = "torchtitan"


def _model_config_to_hf_config_dict(cfg: Decoder.Config) -> dict[str, Any]:
    """Build the HF-shaped config dict SGLang reads before it builds the model.

    SGLang sizes the KV pool from the attention head counts, head dim and layer count,
    and the logits buffers from the vocabulary size.
    """
    attn = cfg.first_full_attention
    if attn is None:
        raise ValueError(
            f"Model config {type(cfg).__qualname__} has no full attention layer."
        )
    n_heads, n_kv_heads, head_dim, value_head_dim = get_attention_dimensions(
        attn, cfg.dim
    )
    return {
        "architectures": [TORCHTITAN_ARCHITECTURE],
        "model_type": "torchtitan",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.dim,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "v_head_dim": value_head_dim,
        "num_hidden_layers": len(cfg.layers),
        "max_position_embeddings": cfg.max_context_length,
        # The wrapper builds and ties the TorchTitan model itself.
        "tie_word_embeddings": False,
        # Requests pass their stop token ids; the model config does not know the
        # tokenizer's EOS.
        "bos_token_id": None,
        "eos_token_id": None,
    }


def _linear_attention_config_kwargs(
    cfg: Decoder.Config, attention_cls: type, dimension_fields: dict[str, str]
) -> dict[str, Any] | None:
    """The linear attention config fields of a model's ``attention_cls`` layers, or
    None without any; SGLang sizes one state slot shape for all of them.

    Args:
        dimension_fields: The HF-shaped config field of each layer config field.
    """
    layers = {
        layer_id: layer.attention
        for layer_id, layer in enumerate(cfg.layers)
        if isinstance(layer.attention, attention_cls.Config)
    }
    if not layers:
        return None
    dimensions = {
        tuple(getattr(attention, field) for field in dimension_fields.values())
        for attention in layers.values()
    }
    if len(dimensions) != 1:
        raise ValueError(
            f"All {attention_cls.__name__} layers must have the same dimensions, "
            f"got {dimensions}"
        )
    (values,) = dimensions
    return dict(linear_layer_ids=sorted(layers), **dict(zip(dimension_fields, values)))


def _patch_hybrid_full_attention_v_head_dim() -> None:
    """Build the full attention KV pool of TorchTitan's hybrid linear attention models
    with the model's V head dim, which MLA's per-head K/V (Kimi K3) make differ from
    K's.

    TODO(https://github.com/sgl-project/sglang/issues/48461): SGLang's
    HybridLinearKVPool builds its MHATokenToKVPool without ``v_head_dim`` (V rows of
    K's head dim), though its pool sizing counts it.
    """
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    original = KVCacheConfigurator._hybrid_full_attention_pool_class

    def _hybrid_full_attention_pool_class(self, **kwargs):
        pool_class = original(self, **kwargs)
        model_config = self.model_config
        if (
            model_config.hf_config.architectures != [TORCHTITAN_ARCHITECTURE]
            or self.use_mla_backend
            or model_config.v_head_dim == model_config.head_dim
        ):
            return pool_class
        return functools.partial(pool_class, v_head_dim=model_config.v_head_dim)

    KVCacheConfigurator._hybrid_full_attention_pool_class = (
        _hybrid_full_attention_pool_class
    )


def register_to_sglang(
    model_config: Decoder.Config,
    *,
    parallelism: InferenceParallelismConfig,
    precision: PrecisionConfig,
    checkpointer_config: CheckpointManager.Config | None,
    dist_moe_runtime: DistMoeRuntime.Config | None,
    model_state_dict_load_mode: ModelStateDictLoadMode,
) -> None:
    """Register the TorchTitan model class, config parser and attention backend with SGLang.

    Call before building SGLang's scheduler. Registers:

      1. A dynamic ``SGLangModelWrapper`` subclass under ``TORCHTITAN_ARCHITECTURE`` in
         SGLang's ``ModelRegistry``; it closes over the arguments below.
      2. A config parser under ``TORCHTITAN_CONFIG_PARSER`` that builds the HF-shaped
         config from ``model_config`` instead of reading ``config.json``: a
         ``TorchTitanGDNConfig`` for models with GDN layers, and a
         ``TorchTitanKDAConfig`` for models with KDA layers.
      3. ``TorchTitanVarlenAttnBackend`` under ``TORCHTITAN_VARLEN_ATTENTION_BACKEND``,
         also as a backend SGLang's deterministic mode accepts with its radix cache.
      4. ``TorchTitanGDNAttnBackend`` and ``TorchTitanKDAAttnBackend`` as the linear
         attention backends of ``TorchTitanGDNConfig`` and ``TorchTitanKDAConfig``
         models, with SGLang's radix cache of their states.

    Args:
        model_config: The generator's model config, with ``SGLangInnerAttention``
            layers (see ``replace_sglang_layer_configs``).
        parallelism: The generator's parallelism.
        precision: The trainer's dtypes, which the model mirrors.
        checkpointer_config: Initial-weight loader; None when weights arrive through
            model state dict loads.
        dist_moe_runtime: The runtime of the model's Dist-MoE experts, if any (see
            ``build_dist_moe_runtime``).
        model_state_dict_load_mode: See ``ModelStateDictLoader``.
    """
    from sglang.srt.configs.model_config_parser_registry import (
        ModelConfigParserBase,
        register_model_config_parser,
    )
    from sglang.srt.layers.attention.attention_registry import (
        ATTENTION_BACKENDS,
        register_attention_backend,
    )
    from sglang.srt.models.registry import ModelRegistry
    from transformers import PretrainedConfig

    from torchtitan.rl.generator.sglang.attention import (
        TORCHTITAN_VARLEN_ATTENTION_BACKEND,
        TorchTitanVarlenAttnBackend,
    )
    from torchtitan.rl.generator.sglang.gdn import (
        TorchTitanGDNAttnBackend,
        TorchTitanGDNConfig,
    )
    from torchtitan.rl.generator.sglang.kda import (
        TorchTitanKDAAttnBackend,
        TorchTitanKDAConfig,
    )
    from torchtitan.rl.generator.sglang.model_wrapper import SGLangModelWrapper

    class TorchTitanForCausalLM(SGLangModelWrapper):
        def __init__(self, *, config, quant_config=None, **kwargs):
            if quant_config is not None:
                raise ValueError("SGLang quantization is not supported")
            super().__init__(
                hf_config=config,
                model_config=model_config,
                parallelism=parallelism,
                precision=precision,
                checkpointer_config=checkpointer_config,
                dist_moe_runtime=dist_moe_runtime,
                model_state_dict_load_mode=model_state_dict_load_mode,
            )

    # TODO(https://github.com/sgl-project/sglang/issues/48475): SGLang registers models
    # by importing packages (ModelRegistry.register(package_name)); write the dict
    # directly to register a class built at runtime.
    ModelRegistry.models[TORCHTITAN_ARCHITECTURE] = TorchTitanForCausalLM

    gdn_kwargs = _linear_attention_config_kwargs(
        model_config,
        GDN,
        {
            "num_k_heads": "n_key_heads",
            "num_v_heads": "n_heads",
            "head_k_dim": "key_head_dim",
            "head_v_dim": "value_head_dim",
            "conv_kernel_size": "conv_kernel_size",
        },
    )
    kda_kwargs = _linear_attention_config_kwargs(
        model_config,
        KDA,
        {
            "kda_num_heads": "n_heads",
            "kda_head_dim": "head_dim",
            "conv_kernel_size": "conv_kernel_size",
        },
    )
    if gdn_kwargs is not None and kda_kwargs is not None:
        raise ValueError(
            "The SGLang engine does not support models with both GDN and KDA layers"
        )

    @register_model_config_parser(TORCHTITAN_CONFIG_PARSER)
    class TorchTitanConfigParser(ModelConfigParserBase):
        def parse(self, model, trust_remote_code, revision=None, **kwargs):
            config_dict = _model_config_to_hf_config_dict(model_config)
            if gdn_kwargs is not None:
                return TorchTitanGDNConfig(
                    **config_dict, **gdn_kwargs, conv_state_dtype=precision.param_dtype
                )
            if kda_kwargs is not None:
                return TorchTitanKDAConfig(
                    **config_dict, **kda_kwargs, conv_state_dtype=precision.param_dtype
                )
            return PretrainedConfig.from_dict(config_dict)

    if TORCHTITAN_VARLEN_ATTENTION_BACKEND not in ATTENTION_BACKENDS:
        from sglang.srt.arg_groups.choices import (
            add_attention_backend_choices,
            add_deterministic_attention_backend_choices,
            add_radix_supported_deterministic_attention_backend_choices,
        )

        add_attention_backend_choices([TORCHTITAN_VARLEN_ATTENTION_BACKEND])
        # The engine turns on SGLang's deterministic mode for seeded sampling only;
        # batch invariance, prefix caching included, is DebugConfig.batch_invariant's,
        # with the trainer's kernel configuration. Keep the radix cache in that mode.
        add_deterministic_attention_backend_choices(
            [TORCHTITAN_VARLEN_ATTENTION_BACKEND]
        )
        add_radix_supported_deterministic_attention_backend_choices(
            [TORCHTITAN_VARLEN_ATTENTION_BACKEND]
        )
        register_attention_backend(TORCHTITAN_VARLEN_ATTENTION_BACKEND)(
            TorchTitanVarlenAttnBackend
        )
        from sglang.srt.configs.linear_attn_model_registry import (
            LinearAttnModelSpec,
            register_linear_attn_model,
        )

        _patch_hybrid_full_attention_v_head_dim()
        # TODO(https://github.com/sgl-project/sglang/issues/48463): support SGLang's
        # extra_buffer mamba cache strategy (support_mamba_cache_extra_buffer), which
        # keeps overlap scheduling; it needs the kernels to write state snapshots at
        # intermediate positions.
        for config_class, backend_class in (
            (TorchTitanGDNConfig, TorchTitanGDNAttnBackend),
            (TorchTitanKDAConfig, TorchTitanKDAAttnBackend),
        ):
            register_linear_attn_model(
                LinearAttnModelSpec(
                    config_class=config_class,
                    backend_class_name=(
                        f"{backend_class.__module__}.{backend_class.__qualname__}"
                    ),
                    uses_mamba_radix_cache=True,
                    support_mamba_cache=True,
                )
            )

    logger.info(
        f"Registered {TORCHTITAN_ARCHITECTURE} + config parser "
        f"{TORCHTITAN_CONFIG_PARSER!r} + attention backend "
        f"{TORCHTITAN_VARLEN_ATTENTION_BACKEND!r} with SGLang "
        f"(model={type(model_config).__qualname__})"
    )
