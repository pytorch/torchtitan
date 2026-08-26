# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Live-weight bridge from TorchTitan Qwen3.5 to native vLLM Qwen3.5.

The trainer publishes its normal TorchTitan state dict through TorchStore.
This module exposes views into the native vLLM model under those same names,
with the destination TP layouts attached as DTensors. TorchStore can therefore
reshard trainer FSDP=4/TP=2 weights directly into generator DP=2/TP=4 storage.

The views account for vLLM's packed parameters (QKV, gate/up, QKVZ, BA, and
depthwise convolution) while preserving the native model's parameter storage.
Updating the views updates the parameters used by already-captured CUDA graphs.
"""

from __future__ import annotations

from typing import Any

import spmd_types as spmd
import torch
import torch.distributed as dist

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.parallel_dims import MeshAxisName, SpmdLayout
from torchtitan.distributed.spmd_types import plain_tensor_to_dtensor_state_dict
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.protocols.model_spec import ModelSpec

DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def is_qwen35_text_weight(name: str) -> bool:
    """Return whether a TorchTitan state-dict entry belongs to the text model."""
    return name.startswith(("tok_embeddings.", "layers.", "norm.", "lm_head."))


def qwen35_text_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop the unused vision tower from a Qwen3.5/3.6 RL weight snapshot."""
    return {
        name: value for name, value in state_dict.items() if is_qwen35_text_weight(name)
    }


def _param_weight(module: Any) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"Expected {type(module).__name__}.weight to be a tensor")
    return weight


def _split_rows(
    tensor: torch.Tensor,
    sizes: list[int],
    *,
    description: str,
) -> list[torch.Tensor]:
    if tensor.ndim < 1 or tensor.shape[0] != sum(sizes):
        raise ValueError(
            f"Unexpected {description} shape {tuple(tensor.shape)}; "
            f"expected first dimension {sum(sizes)} from local splits {sizes}"
        )
    return list(tensor.split(sizes, dim=0))


def build_qwen35_native_weight_views(
    native_model: torch.nn.Module,
    model_spec: ModelSpec,
    *,
    tensor_parallel_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, SpmdLayout]]:
    """Map TorchTitan text-weight names to native vLLM parameter views.

    The returned tensor dict has exactly the keys published by
    :func:`qwen35_text_state_dict`. Each value is the rank-local destination
    storage for that logical TorchTitan tensor. The accompanying layouts tell
    TorchStore how to reshard the global tensor into that local view.
    """
    if model_spec.name != "qwen3_5":
        raise ValueError(
            "The native vLLM weight bridge currently supports only the "
            f"qwen3_5 model family, got {model_spec.name!r}"
        )
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")

    language_model = getattr(native_model, "language_model", native_model)
    core = getattr(language_model, "model", None)
    if core is None or not hasattr(core, "layers"):
        raise TypeError(
            "Expected a native vLLM Qwen3.5 conditional or causal LM with "
            "language_model.model.layers"
        )

    shard0 = SpmdLayout({DP: spmd.R, CP: spmd.R, TP: spmd.S(0)})
    shard1 = SpmdLayout({DP: spmd.R, CP: spmd.R, TP: spmd.S(1)})
    replicated = SpmdLayout({DP: spmd.R, CP: spmd.R, TP: spmd.R})
    views: dict[str, torch.Tensor] = {}
    layouts: dict[str, SpmdLayout] = {}

    def add(name: str, tensor: torch.Tensor, layout: SpmdLayout) -> None:
        if name in views:
            raise ValueError(f"Duplicate native vLLM weight view for {name}")
        # vLLM initializes model parameters under no_grad. Packed-parameter
        # slices inherit the resulting view metadata, which rejects a later
        # TorchStore copy_ in grad-enabled async code. detach() keeps the same
        # storage and pointer while making the synchronization target explicitly
        # inference-only.
        tensor = tensor.detach()
        if not tensor.is_contiguous():
            raise ValueError(
                f"Native vLLM destination view for {name} is not contiguous: "
                f"shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}"
            )
        views[name] = tensor
        layouts[name] = layout

    add("tok_embeddings.weight", _param_weight(core.embed_tokens), shard0)
    add("norm.weight", _param_weight(core.norm), replicated)
    add("lm_head.weight", _param_weight(language_model.lm_head), shard0)

    model_config = model_spec.model
    if len(core.layers) != len(model_config.layers):
        raise ValueError(
            "Native vLLM/TorchTitan layer-count mismatch: "
            f"{len(core.layers)} != {len(model_config.layers)}"
        )

    for layer_idx, (native_layer, layer_config) in enumerate(
        zip(core.layers, model_config.layers, strict=True)
    ):
        prefix = f"layers.{layer_idx}"
        add(
            f"{prefix}.attention_norm.weight",
            _param_weight(native_layer.input_layernorm),
            replicated,
        )
        add(
            f"{prefix}.ffn_norm.weight",
            _param_weight(native_layer.post_attention_layernorm),
            replicated,
        )

        feed_forward = layer_config.feed_forward
        if feed_forward is None or layer_config.moe is not None:
            raise NotImplementedError(
                "The native Qwen3.5 bridge currently supports dense FFN layers only"
            )
        gate_rows = feed_forward.w1.out_features // tensor_parallel_size
        up_rows = feed_forward.w3.out_features // tensor_parallel_size
        gate, up = _split_rows(
            _param_weight(native_layer.mlp.gate_up_proj),
            [gate_rows, up_rows],
            description=f"{prefix}.mlp.gate_up_proj.weight",
        )
        add(f"{prefix}.feed_forward.w1.weight", gate, shard0)
        add(f"{prefix}.feed_forward.w3.weight", up, shard0)
        add(
            f"{prefix}.feed_forward.w2.weight",
            _param_weight(native_layer.mlp.down_proj),
            shard1,
        )

        if layer_config.attention is not None:
            if not hasattr(native_layer, "self_attn"):
                raise TypeError(f"Native layer {layer_idx} is missing self_attn")
            tt_attention = layer_config.attention
            native_attention = native_layer.self_attn
            q_rows = tt_attention.wq.out_features // tensor_parallel_size
            k_rows = tt_attention.wk.out_features // tensor_parallel_size
            v_rows = tt_attention.wv.out_features // tensor_parallel_size
            q, k, v = _split_rows(
                _param_weight(native_attention.qkv_proj),
                [q_rows, k_rows, v_rows],
                description=f"{prefix}.self_attn.qkv_proj.weight",
            )
            add(f"{prefix}.attn.wq.weight", q, shard0)
            add(f"{prefix}.attn.wk.weight", k, shard0)
            add(f"{prefix}.attn.wv.weight", v, shard0)
            add(
                f"{prefix}.attn.wo.weight",
                _param_weight(native_attention.o_proj),
                shard1,
            )
            add(
                f"{prefix}.attn.q_norm.weight",
                _param_weight(native_attention.q_norm),
                replicated,
            )
            add(
                f"{prefix}.attn.k_norm.weight",
                _param_weight(native_attention.k_norm),
                replicated,
            )
            continue

        delta_net = layer_config.delta_net
        if delta_net is None or not hasattr(native_layer, "linear_attn"):
            raise TypeError(f"Native layer {layer_idx} is missing linear_attn")
        native_gdn = native_layer.linear_attn

        q_rows = delta_net.in_proj_q.out_features // tensor_parallel_size
        k_rows = delta_net.in_proj_k.out_features // tensor_parallel_size
        v_rows = delta_net.in_proj_v.out_features // tensor_parallel_size
        z_rows = delta_net.in_proj_z.out_features // tensor_parallel_size
        q, k, v, z = _split_rows(
            _param_weight(native_gdn.in_proj_qkvz),
            [q_rows, k_rows, v_rows, z_rows],
            description=f"{prefix}.linear_attn.in_proj_qkvz.weight",
        )
        add(f"{prefix}.attn.in_proj_q.weight", q, shard0)
        add(f"{prefix}.attn.in_proj_k.weight", k, shard0)
        add(f"{prefix}.attn.in_proj_v.weight", v, shard0)
        add(f"{prefix}.attn.in_proj_z.weight", z, shard0)

        b_rows = delta_net.in_proj_b.out_features // tensor_parallel_size
        a_rows = delta_net.in_proj_a.out_features // tensor_parallel_size
        b, a = _split_rows(
            _param_weight(native_gdn.in_proj_ba),
            [b_rows, a_rows],
            description=f"{prefix}.linear_attn.in_proj_ba.weight",
        )
        add(f"{prefix}.attn.in_proj_b.weight", b, shard0)
        add(f"{prefix}.attn.in_proj_a.weight", a, shard0)

        conv_q, conv_k, conv_v = _split_rows(
            _param_weight(native_gdn.conv1d),
            [q_rows, k_rows, v_rows],
            description=f"{prefix}.linear_attn.conv1d.weight",
        )
        add(f"{prefix}.attn.conv_q.weight", conv_q, shard0)
        add(f"{prefix}.attn.conv_k.weight", conv_k, shard0)
        add(f"{prefix}.attn.conv_v.weight", conv_v, shard0)
        add(f"{prefix}.attn.A_log", native_gdn.A_log, shard0)
        add(f"{prefix}.attn.dt_bias", native_gdn.dt_bias, shard0)
        add(f"{prefix}.attn.norm.weight", _param_weight(native_gdn.norm), replicated)
        add(
            f"{prefix}.attn.out_proj.weight", _param_weight(native_gdn.out_proj), shard1
        )

    return views, layouts


class NativeQwen35WeightBridge:
    """TorchStore destination backed directly by native vLLM parameters."""

    def __init__(
        self,
        native_model: torch.nn.Module,
        model_spec: ModelSpec,
        parallelism: InferenceParallelismConfig,
    ) -> None:
        training_parallelism = parallelism.to_training()
        self.parallel_dims = ParallelDims.from_config(
            training_parallelism,
            dist.get_world_size(),
        )
        self.parallel_dims.build_mesh()
        views, layouts = build_qwen35_native_weight_views(
            native_model,
            model_spec,
            tensor_parallel_size=parallelism.tensor_parallel_degree,
        )
        self.state_dict = plain_tensor_to_dtensor_state_dict(
            views,
            state_dict_layouts=layouts,
            parallel_dims=self.parallel_dims,
        )
