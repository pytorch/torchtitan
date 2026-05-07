# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Single entry point that registers the TorchTitan model class and the
TorchTitan custom ConfigParser with vLLM, plus the HF-shaped config-dict
helper they share. All per-engine torchtitan config (``model_spec``,
``parallelism``, ``compile_config``) is captured via closure on dynamic
subclasses — vLLM's ``hf_config`` only carries HF-shaped fields.

Usage:
    from torchtitan.experiments.rl.models.vllm_registry import (
        registry_to_vllm,
        TORCHTITAN_CONFIG_FORMAT,
    )

    registry_to_vllm(
        model_spec,
        parallelism=parallelism_config,
        compile_config=compile_config,
    )
    # then construct EngineArgs(config_format=TORCHTITAN_CONFIG_FORMAT, ...)
"""

from __future__ import annotations

from typing import Any

from torchtitan.config import CompileConfig, ParallelismConfig
from torchtitan.protocols.model_spec import ModelSpec


# Model-agnostic name used for vLLM model registration.
VLLM_MODEL_NAME = "TorchTitanCausalLM"

# Identifier passed to ``EngineArgs(config_format=...)`` to select the
# torchtitan ConfigParser registered below.
TORCHTITAN_CONFIG_FORMAT = "torchtitan"


def model_spec_to_hf_config_dict(spec: ModelSpec) -> dict[str, Any]:
    """Build the HF-shaped config dict that vLLM's engine init reads.

    Field names match HF conventions because vLLM's engine reads them by
    hardcoded name (``vocab_size``, ``hidden_size``, ``num_attention_heads``,
    …) before any model class is constructed.
    """
    cfg = spec.model
    if not cfg.layers:
        raise ValueError(f"ModelSpec {spec.name!r} has no layers")
    layer0 = cfg.layers[0]
    attn = layer0.attention

    n_heads = attn.n_heads
    n_kv_heads = attn.n_kv_heads or n_heads
    head_dim = attn.head_dim if attn.head_dim is not None else cfg.dim // n_heads

    hf: dict[str, Any] = {
        # All torchtitan-backed models register under VLLM_MODEL_NAME.
        "architectures": [VLLM_MODEL_NAME],
        "model_type": "torchtitan",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.dim,
        "num_hidden_layers": len(cfg.layers),
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": cfg.rope.max_seq_len,
        "rope_theta": cfg.rope.theta,
        "rms_norm_eps": cfg.norm.eps,
        "tie_word_embeddings": getattr(cfg, "enable_weight_tying", False),
        # Best-effort special tokens: tokenizer_config.json overrides at
        # request-processing time.
        "bos_token_id": 0,
        "eos_token_id": 1,
    }

    ffn = getattr(layer0, "feed_forward", None)
    if ffn is not None:
        # FeedForward.Config holds w1/w2/w3 Linear.Config objects; the SwiGLU
        # hidden dim is w1.out_features (== w3.out_features == w2.in_features).
        hf["intermediate_size"] = ffn.w1.out_features

    moe = getattr(layer0, "moe", None)
    if moe is not None:
        hf["num_experts"] = moe.experts.num_experts
        # top_k lives on the router config, not the experts config.
        hf["num_experts_per_tok"] = moe.router.top_k
        hf["moe_intermediate_size"] = moe.experts.hidden_dim
        # vLLM's qwen3_moe model loader checks this for sparse layer placement.
        hf["decoder_sparse_step"] = 1
        hf.setdefault("norm_topk_prob", True)

    return hf


def registry_to_vllm(
    model_spec: ModelSpec,
    *,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
) -> None:
    """Register the TorchTitan model class and the TorchTitan config parser with vLLM.

    Single entry point for vLLM integration. Must be called before creating
    a vLLM engine that uses a TorchTitan model. Registers two things:

      1. ``TorchTitanVLLMModelFromSpec`` (subclass of
         ``TorchTitanVLLMModelWrapper``) with vLLM's ``ModelRegistry`` under
         the name ``VLLM_MODEL_NAME``. The dynamic subclass closes over
         ``model_spec``/``parallelism``/``compile_config`` and forwards them
         when vLLM constructs the model.
      2. ``TorchTitanConfigParserForSpec`` (subclass of ``ConfigParserBase``)
         with vLLM's parser registry under ``TORCHTITAN_CONFIG_FORMAT``. This
         produces the HF-shaped ``PretrainedConfig`` from ``model_spec``.

    Per-engine torchtitan config (parallelism, compile) is delivered to the
    wrapper via closure rather than via vLLM's ``hf_overrides`` channel. This
    keeps the parser scope strictly HF-shaped and isolates vLLM-specific
    plumbing from torchtitan-specific config.

    Args:
        model_spec: TorchTitan ModelSpec containing model config and components.
        parallelism: Authoritative parallelism configuration. The wrapper
            uses this directly to build ``ParallelDims``; the caller is
            responsible for translating the relevant fields (TP, EP) to
            ``EngineArgs`` so vLLM's own world layout matches.
        compile_config: torch.compile config applied per-layer by the
            wrapper's parallelize step.
    """
    from torchtitan.experiments.rl.models.vllm_wrapper import TorchTitanVLLMModelWrapper
    from transformers import PretrainedConfig
    from vllm.logger import init_logger
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import register_config_parser
    from vllm.transformers_utils.config_parser_base import ConfigParserBase

    logger = init_logger(__name__)

    # Dynamic model class capturing torchtitan config in the closure.
    class TorchTitanVLLMModelFromSpec(TorchTitanVLLMModelWrapper):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(
                model_spec=model_spec,
                parallelism=parallelism,
                compile_config=compile_config,
                vllm_config=vllm_config,
                prefix=prefix,
            )

    TorchTitanVLLMModelFromSpec.__name__ = VLLM_MODEL_NAME
    TorchTitanVLLMModelFromSpec.__qualname__ = VLLM_MODEL_NAME
    ModelRegistry.register_model(VLLM_MODEL_NAME, TorchTitanVLLMModelFromSpec)

    # Dynamic config parser class capturing ModelSpec in the closure. This
    # parser only produces HF-shaped fields; torchtitan-specific config is
    # delivered through the model-class closure above.
    @register_config_parser(TORCHTITAN_CONFIG_FORMAT)
    class TorchTitanConfigParserForSpec(ConfigParserBase):
        def parse(
            self,
            model,
            trust_remote_code,
            revision=None,
            code_revision=None,
            **kwargs,
        ):
            config_dict = model_spec_to_hf_config_dict(model_spec)
            return config_dict, PretrainedConfig.from_dict(config_dict)

    logger.info(
        f"Registered {VLLM_MODEL_NAME} + ConfigParser({TORCHTITAN_CONFIG_FORMAT!r}) "
        f"with vLLM (model={model_spec.name}, flavor={model_spec.flavor})"
    )
