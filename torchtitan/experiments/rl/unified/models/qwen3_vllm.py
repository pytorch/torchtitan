# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 model variant optimised for vLLM inference.

Differences from the base ``Qwen3Model`` / ``Qwen3TransformerBlock``:

* **Fused QKV projection** — one DTensor ``F.linear`` + ``split`` instead of
  three separate wq/wk/wv linears.
* **Fused gate-up projection** — one DTensor ``F.linear`` +
  ``silu_and_mul`` instead of two linears + silu + mul.
* **DTensor-native custom ops** — ``torchtitan::rotary_embedding``,
  ``torchtitan::vllm_attention``, and ``torchtitan::silu_and_mul`` have
  registered DTensor sharding strategies so the forward stays in DTensor.
* **Local only at all-reduce** — converts to local tensors only for the
  ``wo`` / ``w2`` matmul + ``vllm::all_reduce`` (to use vLLM's fast
  custom all-reduce instead of NCCL), then re-wraps as DTensor.
* **2-D tensor flow** — operates on ``[T, D]`` tensors (flattened after
  embedding) to avoid 3-D ``aten::linear`` decomposition under
  ``torch.compile``.
* **F.rms_norm for QK norms** — calls ``F.rms_norm`` directly on DTensor
  instead of going through the ``NoParallel(Shard(2))`` module hook
  (which assumes 4-D layout).

Usage::

    from torchtitan.experiments.rl.unified.models.qwen3_vllm import (
        vllm_qwen3_model_registry,
    )

    model_spec = vllm_qwen3_model_registry("1.7B")
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.qwen3 import qwen3_configs
from torchtitan.models.qwen3.model import Qwen3Model, Qwen3TransformerBlock
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan.experiments.rl.unified.models.attention import (
    _convert_rope_cache_to_vllm,
    _local,
    _VLLM_ATTN_MODULES,
)
from torchtitan.experiments.rl.unified.models.parallelize import parallelize_qwen3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Register DTensor sharding strategies for vLLM custom ops
# ---------------------------------------------------------------------------
# These registrations tell DTensor how to propagate placements through the
# opaque vLLM CUDA kernels.  Each op operates independently per head, so
# sharding on the head dimension is always valid.


@register_sharding(torch.ops.torchtitan.rotary_embedding.default)
def _rotary_embedding_sharding(positions, query, key, head_size, cos_sin_cache):
    """rotary_embedding: per-head RoPE, no cross-head interaction.

    Inputs:  positions [T] Replicate, query/key [T, H*D] Shard(-1),
             head_size (int, non-tensor), cos_sin_cache [max_pos, D] Replicate.
    Outputs: query/key [T, H*D] same placement as input.
    """
    strategies = []
    # All replicated
    strategies.append(
        ([Replicate(), Replicate()], [Replicate(), Replicate(), Replicate(), None, Replicate()])
    )
    # Query/key sharded on head dim (dim -1 = dim 1 for 2-D)
    strategies.append(
        ([Shard(1), Shard(1)], [Replicate(), Shard(1), Shard(1), None, Replicate()])
    )
    return strategies


@register_sharding(torch.ops.torchtitan.vllm_attention.default)
def _vllm_attention_sharding(query, key, value, layer_name):
    """vllm_attention: GQA, heads are independent.

    Inputs:  query/key/value [T, H, D] Shard(1).
    Output:  [T, H*D] Shard(1) (2-D flattened).
    """
    strategies = []
    # All replicated
    strategies.append(
        ([Replicate()], [Replicate(), Replicate(), Replicate(), None])
    )
    # Shard on head dim: input dim 1, output dim 1 (H*D flattened)
    strategies.append(
        ([Shard(1)], [Shard(1), Shard(1), Shard(1), None])
    )
    return strategies


@register_sharding(torch.ops.torchtitan.silu_and_mul.default)
def _silu_and_mul_sharding(x):
    """silu_and_mul: element-wise on each shard.

    Input:  [T, 2*ffn_dim] Shard(-1).
    Output: [T, ffn_dim] Shard(-1).
    """
    strategies = []
    strategies.append(([Replicate()], [Replicate()]))
    strategies.append(([Shard(1)], [Shard(1)]))
    return strategies


# ---------------------------------------------------------------------------
# Transformer block with fused attention + FFN forwards
# ---------------------------------------------------------------------------


class Qwen3VLLMBlock(Qwen3TransformerBlock):
    """Qwen3 transformer block with fused vLLM-optimised forwards.

    Inherits the standard ``Qwen3TransformerBlock`` structure (same weights,
    same ``init_weights``, same state-dict keys) but overrides ``forward``
    to operate primarily in DTensor, dropping to local tensors only for the
    ``wo`` / ``w2`` matmul + ``vllm::all_reduce`` boundary.

    Attributes set by :meth:`Qwen3VLLMModel.prepare_for_vllm` after weight
    loading:
        _fused_qkv_weight (local), _nq_local, _nkv_local,
        _local_wo_weight, _fused_gate_up_weight (DTensor), _local_w2_weight,
        _tp_group_name, _vllm_cos_sin_cache, _vllm_layer_name
    """

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention sub-block
        x = x + self._attention_forward(self.attention_norm(x), positions)
        # FFN sub-block
        x = x + self._ffn_forward(self.ffn_norm(x))
        return x

    # -- Attention ----------------------------------------------------------

    def _attention_forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fused attention: QKV → QK norms → RoPE → attention → wo + all-reduce.

        DTensor for norms + custom ops (via register_sharding).
        Local for fused QKV (split requires local) and wo + all_reduce.
        """
        attn = self.attention
        device_mesh = x.device_mesh
        num_tokens = x.shape[0]

        # --- Fused QKV locally (DTensor split along sharded dim → Replicate) ---
        xqkv = F.linear(x.to_local(), attn._fused_qkv_weight)
        # View [T, (nq+2*nkv)*D] → [T, nq+2*nkv, D], then split on heads dim
        xqkv = xqkv.view(num_tokens, -1, attn.head_dim)
        xq, xk, xv = xqkv.split(
            [attn._nq_local, attn._nkv_local, attn._nkv_local], dim=1
        )
        # Enter DTensor: [T, H, D] Shard(1)
        xq = DTensor.from_local(xq, device_mesh=device_mesh, placements=[Shard(1)])
        xk = DTensor.from_local(xk, device_mesh=device_mesh, placements=[Shard(1)])
        xv = DTensor.from_local(xv, device_mesh=device_mesh, placements=[Shard(1)])

        # --- QK norms in DTensor (rms_norm over last dim D, not sharded dim H) ---
        if attn.q_norm is not None:
            xq = F.rms_norm(
                xq, attn.q_norm.normalized_shape,
                attn.q_norm.weight, attn.q_norm.eps,
            )
        if attn.k_norm is not None:
            xk = F.rms_norm(
                xk, attn.k_norm.normalized_shape,
                attn.k_norm.weight, attn.k_norm.eps,
            )

        # --- RoPE in DTensor (via register_sharding) ---
        # Wrap positions as DTensor Replicate for DTensor dispatch.
        # torch._check tells the compiler the dim != 1 (needed for
        # mark_unbacked path; no-op when shapes are concrete).
        if positions is not None and not isinstance(positions, DTensor):
            torch._check(positions.shape[0] != 1)
            positions = DTensor.from_local(
                positions, device_mesh=device_mesh, placements=[Replicate()]
            )

        # Flatten [T, H, D] → [T, H*D] for fused RoPE kernel
        xq_2d = xq.view(num_tokens, -1)
        xk_2d = xk.view(num_tokens, -1)
        xq_2d, xk_2d = torch.ops.torchtitan.rotary_embedding(
            positions, xq_2d, xk_2d, attn.head_dim, attn._vllm_cos_sin_cache,
        )

        # --- Attention in DTensor (via register_sharding) ---
        q_3d = xq_2d.view(num_tokens, -1, attn.head_dim)
        k_3d = xk_2d.view(num_tokens, -1, attn.head_dim)
        attn_out = torch.ops.torchtitan.vllm_attention(
            q_3d, k_3d, xv, attn._vllm_layer_name,
        )

        # --- Drop to local for wo + vllm::all_reduce ---
        h = F.linear(attn_out.to_local(), attn._local_wo_weight)
        if attn._tp_group_name is not None:
            h = torch.ops.vllm.all_reduce(h, group_name=attn._tp_group_name)

        return DTensor.from_local(
            h, device_mesh=device_mesh, placements=[Replicate()]
        )

    # -- FFN ----------------------------------------------------------------

    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused FFN: gate_up → silu_and_mul → w2 + all-reduce.

        Stays in DTensor through gate_up + silu_and_mul, drops to local for
        w2 + all_reduce.
        """
        # gate_up — DTensor linear: Replicate × Shard(0) → Shard(-1)
        gate_up = F.linear(x, self.feed_forward._fused_gate_up_weight)

        # silu_and_mul — registered DTensor sharding: Shard(1) → Shard(1)
        h = torch.ops.torchtitan.silu_and_mul(gate_up)

        # === Drop to local for w2 + vllm::all_reduce ===
        h = F.linear(h.to_local(), self.feed_forward._local_w2_weight)
        if self.feed_forward._tp_group_name is not None:
            h = torch.ops.vllm.all_reduce(
                h, group_name=self.feed_forward._tp_group_name
            )

        # Re-wrap as DTensor Replicate
        return DTensor.from_local(
            h, device_mesh=x.device_mesh, placements=[Replicate()]
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Qwen3VLLMModel(Qwen3Model):
    """Qwen3 model with vLLM-optimised transformer blocks.

    Uses the same config, the same state-dict keys, and the same
    parallelise/weight-loading infra as ``Qwen3Model``.  The only
    difference is that each layer is a :class:`Qwen3VLLMBlock` with
    baked-in fused forwards.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3Model.Config):
        """Identical to Qwen3Model.Config but builds Qwen3VLLMBlock layers."""

        def build(self) -> "Qwen3VLLMModel":
            return Qwen3VLLMModel(self)

    def __init__(self, config: Config):
        # Build via normal Qwen3Model, then swap block classes.
        super().__init__(config)
        for layer in self.layers.values():
            layer.__class__ = Qwen3VLLMBlock

    def prepare_for_vllm(
        self,
        tp_group_name: str | None = None,
        tp_degree: int = 1,
    ) -> None:
        """Set up fused weights, vLLM attention modules, and cos-sin cache.

        Must be called **after** parallelization and weight loading.

        Creates DTensor fused weights (QKV, gate_up) so the forward stays
        in DTensor through the ColwiseParallel matmuls and custom ops,
        only dropping to local at the RowwiseParallel boundary (wo, w2).

        Args:
            tp_group_name: vLLM TP group name for ``vllm::all_reduce``.
            tp_degree: Tensor parallel degree.
        """
        # Convert RoPE cache once
        cos_sin_cache_local = _convert_rope_cache_to_vllm(
            self.freqs_cis, dtype=torch.bfloat16
        )

        # Get device_mesh from any DTensor weight for wrapping plain tensors
        first_layer = next(iter(self.layers.values()))
        device_mesh = first_layer.attention.wq.weight.device_mesh

        # Wrap cos_sin_cache as DTensor Replicate so custom ops get uniform
        # DTensor inputs (DTensor dispatch rejects mixed DTensor/plain tensor).
        cos_sin_cache = DTensor.from_local(
            cos_sin_cache_local, device_mesh=device_mesh, placements=[Replicate()]
        )

        for layer_name, layer in self.layers.items():
            attn = layer.attention

            # Register vLLM attention in global dict for custom op dispatch
            vllm_layer_name = (
                f"model.layers.{layer_name}.attention.inner_attention"
            )
            attn._vllm_layer_name = vllm_layer_name
            _VLLM_ATTN_MODULES[vllm_layer_name] = attn.inner_attention.vllm_attn

            # -- RoPE cache (DTensor Replicate) --
            attn._vllm_cos_sin_cache = cos_sin_cache

            # -- Fused QKV weight (local) --
            # Stored as local tensor: forward does F.linear(x.to_local(), weight)
            # then view+split in head-count space before re-entering DTensor.
            local_wq = _local(attn.wq.weight)
            local_wk = _local(attn.wk.weight)
            local_wv = _local(attn.wv.weight)
            attn._fused_qkv_weight = torch.cat([local_wq, local_wk, local_wv], dim=0)
            attn._nq_local = local_wq.shape[0] // attn.head_dim
            attn._nkv_local = local_wk.shape[0] // attn.head_dim

            # -- wo: local weight for the RowwiseParallel + all_reduce path --
            attn._local_wo_weight = _local(attn.wo.weight)
            attn._tp_group_name = tp_group_name

            # -- Fused gate_up weight as DTensor Shard(0) --
            ff = layer.feed_forward
            local_gate_up = torch.cat(
                [_local(ff.w1.weight), _local(ff.w3.weight)], dim=0
            )
            ff._fused_gate_up_weight = DTensor.from_local(
                local_gate_up, device_mesh=device_mesh, placements=[Shard(0)]
            )

            # -- w2: local weight for the RowwiseParallel + all_reduce path --
            ff._local_w2_weight = _local(ff.w2.weight)
            ff._tp_group_name = tp_group_name

        logger.info(
            f"Prepared Qwen3VLLMModel for vLLM inference "
            f"({len(self.layers)} layers, tp_group={tp_group_name})"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _make_vllm_config(flavor: str) -> Qwen3VLLMModel.Config:
    """Create a Qwen3VLLMModel.Config from an existing Qwen3 flavor."""
    base = qwen3_configs[flavor]
    import dataclasses

    field_vals = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    return Qwen3VLLMModel.Config(**field_vals)


def vllm_qwen3_model_registry(flavor: str) -> ModelSpec:
    """Create a ModelSpec for the vLLM-optimised Qwen3 model.

    Drop-in replacement for ``torchtitan.models.qwen3.model_registry``
    when used with the vLLM wrapper.
    """
    return ModelSpec(
        name="qwen3_vllm",
        flavor=flavor,
        model=_make_vllm_config(flavor),
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_loss_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
