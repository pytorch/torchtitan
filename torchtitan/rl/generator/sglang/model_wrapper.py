# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The TorchTitan model as an SGLang model class."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner_utils.forward_batch import set_forward_batch
from sglang.srt.runtime_context import get_schedule

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import PrecisionConfig
from torchtitan.distributed import maybe_apply_numa_binding
from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import (
    Attention,
    FlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.attention.linear_attention import LinearAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.rl.generator.engine import ModelStateDictLoadMode
from torchtitan.rl.generator.inference_model import (
    build_dist_moe_runtime,
    build_inference_model,
    compute_logits,
    get_attention_dimensions,
    load_initial_weights,
    ModelStateDictLoader,
    route_tp_all_reduce,
)
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.sglang.attention import SGLangInnerAttention
from torchtitan.rl.generator.sglang.gdn import SGLangInnerGDN

logger = logging.getLogger(__name__)


def replace_sglang_layer_configs(model_config: Decoder.Config) -> Decoder.Config:
    """Replace each layer's inner attention with ``SGLangInnerAttention``, or, for GDN
    layers, ``SGLangInnerGDN``.

    Only GDN and softmax attention with the varlen backend, full or sliding-window, are
    supported; other layers raise ``ValueError``. The inner attention keeps the trainer
    attention's sharding config.
    """
    from torchtitan.models.gpt_oss.model import GptOssAttention

    new_layers = []
    for layer_id, layer_cfg in enumerate(model_config.layers):
        attention_cfg = layer_cfg.attention
        if isinstance(attention_cfg, GDN.Config):
            sglang_inner_gdn_cfg = SGLangInnerGDN.Config(
                layer_id=layer_id,
                num_k_heads=attention_cfg.n_key_heads,
                num_v_heads=attention_cfg.n_heads,
                head_k_dim=attention_cfg.key_head_dim,
                head_v_dim=attention_cfg.value_head_dim,
                conv_kernel_size=attention_cfg.conv_kernel_size,
                sharding_config=attention_cfg.inner_attention.sharding_config,
            )
            new_layers.append(
                dataclasses.replace(
                    layer_cfg,
                    attention=dataclasses.replace(
                        attention_cfg, inner_attention=sglang_inner_gdn_cfg
                    ),
                )
            )
            continue
        if isinstance(attention_cfg, LinearAttention.Config) or not isinstance(
            attention_cfg, Attention.Config
        ):
            raise ValueError(
                f"layers[{layer_id}]: the SGLang engine supports softmax attention "
                f"and GDN only, got {type(attention_cfg).__qualname__}"
            )
        if isinstance(attention_cfg, GptOssAttention.Config):
            # TODO: rescale the attention output by the sinks, from the log-sum-exp
            # SGLang's FlashAttention backend returns.
            raise ValueError(
                f"layers[{layer_id}]: the SGLang engine does not support attention "
                "sinks (GPT-OSS)"
            )
        inner_cfg = attention_cfg.inner_attention
        if isinstance(inner_cfg, FlexInnerAttention.Config):
            # TODO: run flex attention through an SGLang backend with the trainer's
            # block sizes, as the vLLM engine does with vLLM's FLEX_ATTENTION.
            raise ValueError(
                f"layers[{layer_id}]: the SGLang engine does not support the flex "
                "attention backend yet; use varlen."
            )
        if type(inner_cfg) not in (
            VarlenInnerAttention.Config,
            SlidingWindowVarlenInnerAttention.Config,
        ):
            raise ValueError(
                f"layers[{layer_id}]: the SGLang engine supports varlen attention "
                f"only, got {type(inner_cfg).__qualname__}"
            )
        num_heads, num_kv_heads, head_dim, value_head_dim = get_attention_dimensions(
            attention_cfg, model_config.dim
        )
        if value_head_dim != head_dim:
            raise ValueError(
                f"layers[{layer_id}]: the SGLang engine needs equal query/key and "
                f"value head dims, got {head_dim} and {value_head_dim}"
            )
        sglang_inner_cfg = SGLangInnerAttention.Config(
            layer_id=layer_id,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=attention_cfg.softmax_scale,
            sliding_window_size=(
                inner_cfg.sliding_window_size
                if isinstance(inner_cfg, SlidingWindowVarlenInnerAttention.Config)
                else None
            ),
            sharding_config=inner_cfg.sharding_config,
        )
        new_layers.append(
            dataclasses.replace(
                layer_cfg,
                attention=dataclasses.replace(
                    attention_cfg, inner_attention=sglang_inner_cfg
                ),
            )
        )
    return dataclasses.replace(model_config, layers=new_layers)


class _TorchTitanLogitsProcessor(LogitsProcessor):
    """SGLang's ``LogitsProcessor`` with the model wrapper's full-vocabulary logits.

    SGLang keeps the hidden states of the positions it samples and handles logprob
    metadata; ``compute_logits`` maps them to logits, gathered over TP.
    """

    def __init__(self, hf_config, compute_logits) -> None:
        # compute_logits gathers over TP itself.
        super().__init__(hf_config, skip_all_gather=True)
        self._compute_logits = compute_logits

    def _compute_lm_head(
        self, hidden_states, lm_head, embedding_bias=None, *, out=None
    ) -> torch.Tensor:
        if embedding_bias is not None:
            raise NotImplementedError("embedding_bias is not supported")
        return self._compute_logits(hidden_states)


class _DecoderBody(Module):
    """The decoder up to its final norm, as the module SGLang's prefill CUDA graphs
    capture.

    SGLang captures ``forward(input_ids, positions, forward_batch)`` of the module that
    owns ``layers`` behind ``SGLangModelWrapper.model``, and replays it inside the
    wrapper's eager forward, which keeps the LM head and logits processing eager.

    With expert parallelism, the MoE layers shard the step's tokens over TP, so the
    body pads them to a multiple of the TP degree; ``SGLangInnerAttention`` runs on the
    rows SGLang scheduled only.
    """

    def __init__(
        self,
        decoder: Decoder,
        parallelism_context,
        *,
        token_multiple: int,
        dist_moe_runtime: DistMoeRuntime | None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self._parallelism_context = parallelism_context
        self._token_multiple = token_multiple
        self._dist_moe_runtime = dist_moe_runtime
        # TODO(https://github.com/sgl-project/sglang/issues/48475): SGLang finds each
        # layer's RadixAttention or RadixLinearAttention by attribute path
        # (``layers[i].attention.attn`` among others); alias it there, unregistered so
        # the state dict is unchanged.
        for layer in decoder.layers.values():
            inner = layer.attention.inner_attention
            object.__setattr__(
                layer.attention,
                "attn",
                (
                    inner.linear_attention
                    if isinstance(inner, SGLangInnerGDN)
                    else inner.radix_attention
                ),
            )

    @property
    def layers(self):
        return self.decoder.layers

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # The attention layers read forward_batch from SGLang's context variable too.
        num_tokens = input_ids.shape[0]
        num_padding = -num_tokens % self._token_multiple
        if num_padding:
            input_ids = F.pad(input_ids, (0, num_padding))
            positions = F.pad(positions, (0, num_padding))
        padding_mask = None
        if self._dist_moe_runtime is not None:
            # Dist-MoE routes padding rows to no expert (the stock token dispatchers
            # dispatch them like real tokens): the rows padding the step to the token
            # multiple, and those SGLang pads a batch with to a CUDA graph's size.
            # TODO: this relies on two SGLang internals: its allocators never hand
            # out KV slot 0, and its graph buffers zero-fill the padded part of
            # out_cache_loc. SGLang's own real-token count (num_token_non_padded)
            # is filled only when SGLang itself runs EP.
            padding_mask = F.pad(
                forward_batch.out_cache_loc == 0, (0, num_padding), value=True
            )
            # Every EP rank passes the same count: the EP group is the TP group, which
            # shares the step, and the MoE splits its tokens evenly over TP. The count
            # is fixed per captured graph, so it is set at capture.
            tp = self._parallelism_context.tp
            assert (
                input_ids.shape[0] % tp == 0
            ), f"TP={tp} does not divide {input_ids.shape[0]} tokens"
            self._dist_moe_runtime.set_num_local_input_tokens_per_call(
                input_ids.shape[0] // tp
            )
        with self._parallelism_context.activate_spmd():
            # Keywords: some models' forward (e.g. Qwen3.5's) takes them only so.
            hidden_states = self.decoder(
                input_ids, positions=positions, padding_mask=padding_mask
            )
        return hidden_states[:num_tokens]


class SGLangModelWrapper(Module):
    """A TorchTitan model as an SGLang model class.

    SGLang constructs it inside its model runner, after its process groups exist. The
    wrapper builds the TorchTitan model with ``SGLangInnerAttention`` layers (see
    ``build_inference_model``), loads the initial weights, and serves SGLang's
    ``forward(input_ids, positions, forward_batch) -> LogitsProcessorOutput``.

    Args:
        hf_config: The HF-shaped config SGLang parsed (see ``register_to_sglang``).
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

    def __init__(
        self,
        *,
        hf_config,
        model_config: Decoder.Config,
        parallelism: InferenceParallelismConfig,
        precision: PrecisionConfig,
        checkpointer_config: CheckpointManager.Config | None,
        dist_moe_runtime: DistMoeRuntime.Config | None,
        model_state_dict_load_mode: ModelStateDictLoadMode,
    ) -> None:
        super().__init__()
        self.config = model_config
        device_index = torch.accelerator.current_device_index()
        device_type = torch.accelerator.current_accelerator().type
        device = torch.device(device_type, device_index)
        decoder, self.parallelism_context = build_inference_model(
            self.config, parallelism=parallelism, precision=precision, device=device
        )
        schedule = get_schedule()
        token_multiple = parallelism.expert_sequence_parallel_size
        # A step runs at most chunked_prefill_size prefill tokens plus a decode token
        # per running request, padded to token_multiple.
        max_num_tokens_per_step = (
            schedule.chunked_prefill_size + schedule.max_running_requests
        )
        dist_moe = build_dist_moe_runtime(
            decoder,
            dist_moe_runtime,
            parallelism_context=self.parallelism_context,
            device=device,
            max_num_tokens_per_step=-(-max_num_tokens_per_step // token_multiple)
            * token_multiple,
        )
        load_initial_weights(decoder, self.config, checkpointer_config)
        # SGLang's TP all-reduce picks its algorithm by message size, which breaks
        # batch invariance.
        if self.parallelism_context.tp_enabled and not is_in_batch_invariant_mode():
            route_tp_all_reduce(tensor_model_parallel_all_reduce)
        maybe_apply_numa_binding(device_index, device_type)
        # Unshards the model, so SGLang profiles GPU memory with the model's
        # representation during forward.
        self.model_state_dict_loader = ModelStateDictLoader(
            decoder, model_state_dict_load_mode=model_state_dict_load_mode
        )
        self.model = _DecoderBody(
            decoder,
            self.parallelism_context,
            token_multiple=token_multiple,
            dist_moe_runtime=dist_moe,
        )
        self.logits_processor = _TorchTitanLogitsProcessor(
            hf_config,
            lambda hidden_states: compute_logits(
                decoder, hidden_states, self.parallelism_context
            ),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        """SGLang's model API: logits of the positions SGLang samples.

        Args:
            input_ids: ``[num_tokens]`` token ids of the step (1D varlen).
            positions: ``[num_tokens]`` position of each token.
            forward_batch: The step's batch metadata, published for the attention
                layers to read.
        """
        if input_embeds is not None or get_embedding or pp_proxy_tensors is not None:
            raise NotImplementedError(
                "input_embeds, embeddings and pipeline parallelism are not supported"
            )
        with set_forward_batch(forward_batch):
            hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.model.decoder.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """SGLang's model API. A no-op: the wrapper loaded its weights when built, and
        never reads SGLang's checkpoint iterator."""
