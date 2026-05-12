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

    Fields are grouped into three categories:
      1. Value used — vLLM reads the actual value and its magnitude
         affects behavior.
      2. Presence required — only existence / non-empty / positive
         matters; the specific value is not consumed.
      3. Unused — present so ``PretrainedConfig`` has the keys other
         vLLM helpers may ``getattr`` against, but the values are not
         consumed in our flow (V1 engine, ``TorchTitanCausalLM`` model
         class, no KV transfer, no MFU metrics, no multimodal).
    """
    cfg = spec.model
    if not cfg.layers:
        raise ValueError(f"ModelSpec {spec.name!r} has no layers")
    # Some models mix dense and MoE layers (e.g. deepseek_v3 has dense
    # first layers, MoE later); scan the layer list for a representative
    # of each component rather than relying on layer 0.
    attn = cfg.layers[0].attention
    ffn = next(
        (
            ff
            for layer in cfg.layers
            if (ff := getattr(layer, "feed_forward", None)) is not None
        ),
        None,
    )
    moe = next(
        (m for layer in cfg.layers if (m := getattr(layer, "moe", None)) is not None),
        None,
    )

    n_heads = attn.n_heads
    n_kv_heads = attn.n_kv_heads or n_heads
    head_dim = attn.head_dim if attn.head_dim is not None else cfg.dim // n_heads

    hf: dict[str, Any] = {
        # Value used
        "architectures": [VLLM_MODEL_NAME],  # ModelRegistry lookup key
        "vocab_size": cfg.vocab_size,  # V1 logits buffer + out of vocabulary check
        "hidden_size": cfg.dim,  # vLLM compile-pass thresholds (SP, flashinfer)
        "num_attention_heads": n_heads,  # TP divisibility + FA3 num_heads_q
        "num_key_value_heads": n_kv_heads,  # DCP divisibility + FA3 num_heads_kv
        "head_dim": head_dim,  # FA3 scheduler headdim
        "max_position_embeddings": cfg.rope.max_seq_len,  # caps max_model_len
        # Presence required
        "model_type": "torchtitan",  # any non-empty string
        "num_hidden_layers": len(
            cfg.layers
        ),  # positive int; only PP/KV-transfer read magnitude
        # Unused
        "rope_theta": cfg.rope.theta,  # only used for non-default rope_type; wrapper builds RoPE
        "rms_norm_eps": cfg.norm.eps,  # only minimax-qk-norm fusion reads it; wrapper builds RMSNorm
        "tie_word_embeddings": getattr(
            cfg, "enable_weight_tying", False
        ),  # multimodal/GGUF only; wrapper ties weights
        "bos_token_id": 0,  # Fuyu-only; engine reads tokenizer/sampling tokens
        "eos_token_id": 1,  # per-model files only; engine reads tokenizer/sampling tokens
    }

    if ffn is not None:
        # Unused: only v1/metrics/perf.py reads it (off by default). SwiGLU hidden == w1.out_features.
        hf["intermediate_size"] = ffn.w1.out_features

    if moe is not None:
        # Presence required: >0 toggles MoE/EP branches.
        hf["num_experts"] = moe.experts.num_experts
        # Unused: only per-model loaders (qwen3_moe, deepseek_v2, ...) and v1/metrics/perf.py (off) read these.
        hf[
            "num_experts_per_tok"
        ] = moe.router.top_k  # top_k is on the router, not experts
        hf["moe_intermediate_size"] = moe.experts.hidden_dim
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

      1. ``VLLMModelFromSpec`` (subclass of ``VLLMModelWrapper``)
         with vLLM's ``ModelRegistry`` under the name ``VLLM_MODEL_NAME``.
         The dynamic subclass closes over
         ``model_spec``/``parallelism``/``compile_config`` and forwards them
         when vLLM constructs the model.
      2. ``TorchTitanConfigParser`` (subclass of ``ConfigParserBase``)
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
    from torchtitan.experiments.rl.models.vllm_wrapper import VLLMModelWrapper
    from vllm.logger import init_logger
    from vllm.model_executor.models.registry import ModelRegistry

    # Pull ``PretrainedConfig`` through vLLM's transformers re-export rather
    # than from ``transformers`` directly. vLLM already depends on
    # transformers internally, so this keeps torchtitan free of a direct
    # ``transformers`` import — when vLLM eventually drops it, this path
    # disappears with it.
    from vllm.transformers_utils.config import PretrainedConfig, register_config_parser
    from vllm.transformers_utils.config_parser_base import ConfigParserBase

    logger = init_logger(__name__)

    # Dynamic model class capturing torchtitan config in the closure.
    class VLLMModelFromSpec(VLLMModelWrapper):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(
                model_spec=model_spec,
                parallelism=parallelism,
                compile_config=compile_config,
                vllm_config=vllm_config,
                prefix=prefix,
            )

    VLLMModelFromSpec.__name__ = VLLM_MODEL_NAME
    VLLMModelFromSpec.__qualname__ = VLLM_MODEL_NAME
    ModelRegistry.register_model(VLLM_MODEL_NAME, VLLMModelFromSpec)

    # Dynamic config parser class capturing ModelSpec in the closure. This
    # parser only produces HF-shaped fields; torchtitan-specific config is
    # delivered through the model-class closure above.
    @register_config_parser(TORCHTITAN_CONFIG_FORMAT)
    class TorchTitanConfigParser(ConfigParserBase):
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
