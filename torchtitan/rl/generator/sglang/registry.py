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

import logging
from typing import Any

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import PrecisionConfig
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.rl.generator.engine import ModelStateDictLoadMode
from torchtitan.rl.generator.model import get_attention_dimensions
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig

logger = logging.getLogger(__name__)

# The HF ``architectures`` entry SGLang's ModelRegistry resolves to the model class.
TORCHTITAN_ARCHITECTURE = "TorchTitanForCausalLM"

# ``ServerArgs.model_config_parser`` value selecting the TorchTitan config parser.
TORCHTITAN_CONFIG_PARSER = "torchtitan"

# SGLang caps a request at context_len - 1 tokens and rejects prompts of
# context_len - 6 tokens or more (max_req_len and max_req_input_len in its
# TpModelWorker). SGLangEngine sets SGLang's context length this many tokens above
# the model's max_context_length, so SGLang accepts the prompts vLLM accepts (up to
# max_context_length - 1 tokens), and caps each request's output itself.
CONTEXT_LENGTH_MARGIN = 6


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
    n_heads, n_kv_heads, head_dim, _ = get_attention_dimensions(attn, cfg.dim)
    return {
        "architectures": [TORCHTITAN_ARCHITECTURE],
        "model_type": "torchtitan",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.dim,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "num_hidden_layers": len(cfg.layers),
        # SGLang rejects a context length above this.
        "max_position_embeddings": cfg.max_context_length + CONTEXT_LENGTH_MARGIN,
        # The wrapper builds and ties the TorchTitan model itself.
        "tie_word_embeddings": False,
        # Requests pass their stop token ids; the model config does not know the
        # tokenizer's EOS.
        "bos_token_id": None,
        "eos_token_id": None,
    }


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
         config from ``model_config`` instead of reading ``config.json``.
      3. ``TorchTitanVarlenAttnBackend`` under ``TORCHTITAN_VARLEN_ATTENTION_BACKEND``,
         also as a backend SGLang's deterministic mode accepts with its radix cache.

    Args:
        model_config: The generator's model config.
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

    from torchtitan.rl.generator.sglang.attention.attention import (
        TORCHTITAN_VARLEN_ATTENTION_BACKEND,
        TorchTitanVarlenAttnBackend,
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

    @register_model_config_parser(TORCHTITAN_CONFIG_PARSER)
    class TorchTitanConfigParser(ModelConfigParserBase):
        def parse(self, model, trust_remote_code, revision=None, **kwargs):
            return PretrainedConfig.from_dict(
                _model_config_to_hf_config_dict(model_config)
            )

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

    logger.info(
        f"Registered {TORCHTITAN_ARCHITECTURE} + config parser "
        f"{TORCHTITAN_CONFIG_PARSER!r} + attention backend "
        f"{TORCHTITAN_VARLEN_ATTENTION_BACKEND!r} with SGLang "
        f"(model={type(model_config).__qualname__})"
    )
