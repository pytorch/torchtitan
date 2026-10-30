# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Single entry point that registers the TorchTitan model class and the
TorchTitan custom ConfigParser with vLLM, plus the HF-shaped config-dict
helper they share. All per-engine torchtitan config (``model_config``,
``parallelism``) is captured via closure on dynamic subclasses — vLLM's
``hf_config`` only carries HF-shaped fields.

Usage:
    from torchtitan.rl.generator.vllm.registry import (
        register_to_vllm,
        TORCHTITAN_CONFIG_FORMAT,
    )

    register_to_vllm(
        model_config,
        parallelism=parallelism_config,
    )
    # then construct EngineArgs(config_format=TORCHTITAN_CONFIG_FORMAT, ...)
"""

from __future__ import annotations

import logging
import os
from typing import Any, TYPE_CHECKING

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    InnerAttention,
    MLAAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig

if TYPE_CHECKING:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)

# Model-agnostic name used for vLLM model registration.
VLLM_MODEL_NAME = "TorchTitanCausalLM"

# Identifier passed to ``EngineArgs(config_format=...)`` to select the
# torchtitan ConfigParser registered below.
TORCHTITAN_CONFIG_FORMAT = "torchtitan"

# ``model_type`` of the HF-shaped config; selects the arch config converter
# registered below.
TORCHTITAN_MODEL_TYPE = "torchtitan"

# Selects the experiment-owned runner that pads tokens for dense and expert SP.
TORCHTITAN_WORKER_CLS = "torchtitan.rl.generator.vllm.worker.TorchTitanGPUWorker"


def uses_compressed_mla_cache(inner_attention: InnerAttention.Config) -> bool:
    """Return whether an attention layer caches MLA's compressed latent in vLLM.

    The cache layout and the generator's attention layers must agree, so every
    compressed-vs-materialized decision goes through this predicate.
    Batch-invariant mode keeps materialized K/V for bitwise parity with the
    trainer.
    """
    return (
        isinstance(inner_attention, MLAAttention.Config)
        and not is_in_batch_invariant_mode()
    )


def vllm_attention_backend(cfg: Decoder.Config) -> AttentionBackendEnum | None:
    """Return the vLLM attention backend for ``cfg``'s full attention layers.

    Returns ``None`` for MLA with a compressed cache: vLLM auto-selects one of
    its MLA backends for the hardware, since vLLM's MLA kernel runs on that path
    rather than a TorchTitan inner attention.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    attention = cfg.first_full_attention
    if attention is None:
        raise ValueError("No full attention layer found in the model spec.")
    kernel_config = attention.inner_attention
    if uses_compressed_mla_cache(kernel_config):
        return None
    if isinstance(kernel_config, MLAAttention.Config):
        kernel_config = kernel_config.inner_attention
    if isinstance(kernel_config, FlexInnerAttention.Config):
        return AttentionBackendEnum.FLEX_ATTENTION
    if isinstance(kernel_config, VarlenInnerAttention.Config):
        return AttentionBackendEnum.CUSTOM
    raise ValueError(
        "Only varlen and flex attention backends, or MLA with a compressed "
        "cache, are supported."
    )


def model_config_to_hf_config_dict(cfg: Decoder.Config) -> dict[str, Any]:
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
    if not cfg.layers:
        raise ValueError(f"Model config {type(cfg).__qualname__} has no layers")
    attn = cfg.first_full_attention
    if attn is None:
        raise ValueError(
            f"Model config {type(cfg).__qualname__} has no full attention layer. "
            "vLLM needs attention head counts and dimensions before building the model."
        )
    ffn = cfg.first_feed_forward
    moe = cfg.first_moe

    from torchtitan.rl.generator.vllm.attention import get_attention_dimensions

    n_heads, n_kv_heads, head_dim, _ = get_attention_dimensions(attn, cfg.dim)
    rope = getattr(attn, "rope", None)
    rope_theta = None if rope is None else rope.theta

    hf: dict[str, Any] = {
        # Value used
        "architectures": [VLLM_MODEL_NAME],  # ModelRegistry lookup key
        "vocab_size": cfg.vocab_size,  # V1 logits buffer + out of vocabulary check
        "hidden_size": cfg.dim,  # vLLM compile-pass thresholds (SP, flashinfer)
        "num_attention_heads": n_heads,  # TP divisibility + FA3 num_heads_q
        "num_key_value_heads": n_kv_heads,  # DCP divisibility + FA3 num_heads_kv
        "head_dim": head_dim,  # FA3 scheduler headdim
        "max_position_embeddings": cfg.max_context_length,  # caps max_model_len
        # Presence required
        "model_type": TORCHTITAN_MODEL_TYPE,  # selects the arch config converter
        "num_hidden_layers": len(
            cfg.layers
        ),  # positive int; only PP/KV-transfer read magnitude
        # Unused
        "rope_theta": rope_theta,  # only used for non-default rope_type; wrapper builds RoPE
        "rms_norm_eps": cfg.norm.eps,  # only minimax-qk-norm fusion reads it; wrapper builds RMSNorm
        "tie_word_embeddings": getattr(
            cfg, "enable_weight_tying", False
        ),  # multimodal/GGUF only; wrapper ties weights
        # Value used: without a generation_config.json in the checkpoint, vLLM
        # derives the generation config from this dict and adds its
        # eos_token_id to every request's stop set. The model config does not
        # know the tokenizer's ids, so leave them unset; vLLM takes EOS from
        # the tokenizer and callers pass any other stop tokens.
        "bos_token_id": None,
        "eos_token_id": None,
    }

    if uses_compressed_mla_cache(attn.inner_attention):
        hf["kv_lora_rank"] = attn.kv_lora_rank
        hf["qk_nope_head_dim"] = attn.qk_nope_head_dim
        hf["qk_rope_head_dim"] = attn.qk_rope_head_dim
        hf["v_head_dim"] = attn.v_head_dim

    if ffn is not None:
        # Unused: only v1/metrics/perf.py reads it (off by default).
        hf["intermediate_size"] = ffn.w13.out_features

    if moe is not None:
        # Presence required: >0 toggles MoE/EP branches.
        hf["num_experts"] = moe.router.num_experts
        # Unused: only per-model loaders (qwen3_moe, deepseek_v2, ...) and v1/metrics/perf.py (off) read these.
        hf["num_experts_per_tok"] = moe.router.top_k
        hf["moe_intermediate_size"] = moe.routed_experts.w2.in_features
        hf["decoder_sparse_step"] = 1
        hf.setdefault("norm_topk_prob", True)

    return hf


def register_to_vllm(
    model_config: Decoder.Config,
    *,
    parallelism: InferenceParallelismConfig,
    checkpointer_config: CheckpointManager.Config | None,
    dist_moe_runtime: DistMoeRuntime.Config | None = None,
) -> None:
    """Register the TorchTitan model class and the TorchTitan config parser with vLLM.

    Single entry point for vLLM integration. Must be called before creating
    a vLLM engine that uses a TorchTitan model. Registers three things:

      1. ``VLLMModelFromSpec`` (subclass of ``VLLMModelWrapper``)
         with vLLM's ``ModelRegistry`` under the name ``VLLM_MODEL_NAME``.
         The dynamic subclass closes over
         ``model_config``/``parallelism``/``checkpointer_config``
         and forwards them when vLLM constructs the model.
      2. ``TorchTitanConfigParser`` (subclass of ``ConfigParserBase``)
         with vLLM's parser registry under ``TORCHTITAN_CONFIG_FORMAT``. This
         produces the HF-shaped ``PretrainedConfig`` from ``model_config``.
      3. ``TorchTitanModelArchConfigConvertor`` with vLLM's arch config
         converters under ``TORCHTITAN_MODEL_TYPE``. It reports MLA when the
         HF config has ``kv_lora_rank``, which ``model_config_to_hf_config_dict``
         sets only for compressed MLA caches, so vLLM sizes the cache from the
         latent and sets ``use_mla``.

    Per-engine torchtitan config (parallelism and checkpoint) is
    delivered to the wrapper via closure rather than via vLLM's
    ``hf_overrides`` channel. This keeps the parser scope strictly HF-shaped
    and isolates vLLM-specific plumbing from torchtitan-specific config.

    Args:
        model_config: TorchTitan decoder model config.
        parallelism: Inference parallelism configuration. The wrapper
            translates it to a full ``ParallelismConfig`` to build
            ``ParallelismContext``; the caller is responsible for translating the
            relevant fields (TP, EP) to ``EngineArgs`` so vLLM's own world
            layout matches.
        checkpointer_config: Optional CheckpointManager configuration for
            initial weight loading. Pass ``None`` for the RL loop, where
            weights arrive from TorchStore.
        dist_moe_runtime: ``DistMoeRuntime.Config(inference=True)`` for models
            with Dist-MoE routed experts; ``None`` otherwise.
    """
    has_gdn = any(
        isinstance(layer.attention, GDN.Config) for layer in model_config.layers
    )
    has_kda = any(
        isinstance(layer.attention, KDA.Config) for layer in model_config.layers
    )
    if has_gdn or has_kda:
        # Attention Gym's paged linear-attention kernels require channels to
        # be contiguous in vLLM's convolution state cache.
        os.environ["VLLM_SSM_CONV_STATE_LAYOUT"] = "SD"

    from vllm.model_executor.models.registry import ModelRegistry

    # Pull ``PretrainedConfig`` through vLLM's transformers re-export rather
    # than from ``transformers`` directly. vLLM already depends on
    # transformers internally, so this keeps torchtitan free of a direct
    # ``transformers`` import — when vLLM eventually drops it, this path
    # disappears with it.
    from vllm.transformers_utils.config import PretrainedConfig, register_config_parser
    from vllm.transformers_utils.config_parser_base import ConfigParserBase
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        ModelArchConfigConvertorBase,
    )

    from torchtitan.rl.generator.vllm.model_wrapper import VLLMModelWrapper

    # Dynamic model class capturing torchtitan config in the closure.
    class VLLMModelFromSpec(VLLMModelWrapper):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(
                model_config=model_config,
                parallelism=parallelism,
                checkpointer_config=checkpointer_config,
                vllm_config=vllm_config,
                prefix=prefix,
                dist_moe_runtime=dist_moe_runtime,
            )

    VLLMModelFromSpec.__name__ = VLLM_MODEL_NAME
    VLLMModelFromSpec.__qualname__ = VLLM_MODEL_NAME
    # vLLM needs a model-level state contract to allocate shared attention/GDN
    # cache pages before individual layers are constructed.
    if has_gdn:
        from torchtitan.rl.generator.vllm.gdn import maybe_configure_gdn_hybrid_model

        maybe_configure_gdn_hybrid_model(VLLMModelFromSpec, model_config)
    if has_kda:
        from torchtitan.rl.generator.vllm.kda import maybe_configure_kda_hybrid_model

        maybe_configure_kda_hybrid_model(VLLMModelFromSpec, model_config)
    ModelRegistry.register_model(VLLM_MODEL_NAME, VLLMModelFromSpec)

    # Dynamic config parser class capturing the model config in the closure. This
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
            config_dict = model_config_to_hf_config_dict(model_config)
            return config_dict, PretrainedConfig.from_dict(config_dict)

    # vLLM's default is_deepseek_mla only recognizes a hardcoded list of model
    # types, which excludes model_type="torchtitan", so is_mla would be False.
    # Despite the name, vLLM uses it for any DeepSeek-style MLA model (e.g. Kimi,
    # GLM). This override reports MLA whenever the HF config has kv_lora_rank,
    # which model_config_to_hf_config_dict sets only for the compressed MLA path.
    class TorchTitanModelArchConfigConvertor(ModelArchConfigConvertorBase):
        def is_deepseek_mla(self) -> bool:
            return getattr(self.hf_text_config, "kv_lora_rank", None) is not None

    # TODO: Wait until the vLLM PR adding register_model_arch_config_convertor
    # lands, then remove this direct registry write and use that API.
    MODEL_ARCH_CONFIG_CONVERTORS[
        TORCHTITAN_MODEL_TYPE
    ] = TorchTitanModelArchConfigConvertor

    logger.info(
        f"Registered {VLLM_MODEL_NAME} + ConfigParser({TORCHTITAN_CONFIG_FORMAT!r}) "
        f"with vLLM (model={type(model_config).__qualname__})"
    )
