# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
from fla.modules.conv.triton.ops import CausalConv1dFunction
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule,
)
from fla.ops.gated_delta_rule.chunk import ChunkGatedDeltaRuleFunction
from fla.ops.gated_delta_rule.fused_recurrent import FusedRecurrentFunction
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_varlen_metadata_for_document,
    local_head_split,
    VarlenAttention,
    VarlenMetadata,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.multimodal import (
    get_vision_positions,
    multimodal_context,
    scatter_vision_embeds,
)
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.observability import tensor_logging
from torchtitan.protocols.module import Module

from .rope import MRoPE
from .sharding import annotate_qwen35_input_spmd_types, set_qwen35_sharding_config
from .vision_encoder import Qwen35VisionEncoder

GatedDeltaBackend = Literal["fla_chunked", "fla_fused_recurrent"]
Qwen35AttentionMaskDict = dict[str, BlockMask | VarlenMetadata | None]

spmd.register_local_autograd_function(ChunkGatedDeltaRuleFunction)
spmd.register_local_autograd_function(FusedRecurrentFunction)
spmd.register_local_autograd_function(CausalConv1dFunction)


@spmd.local_map(
    in_types=(
        {"dp": spmd.S(1), "tp": spmd.S(2)},
        {"dp": spmd.R, "tp": spmd.S(0)},
        {"dp": spmd.V, "tp": spmd.R},
        {"dp": spmd.V, "tp": spmd.R},
    ),
    out_types={"dp": spmd.S(1), "tp": spmd.S(2)},
)
def _causal_conv1d_varlen(
    x_BTD: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None,
) -> torch.Tensor:
    """FLA depthwise causal conv with per-document resets (CUDA-only).

    A pure-torch per-document reference lives in
    ``tests/unit_tests/test_qwen3_5_deltanet.py``.
    """
    if cu_seqlens_cpu is None:
        raise ValueError(
            "Qwen3.5 FLA varlen conv requires a CPU cu_seqlens tensor. "
            "Build VarlenMetadata with include_host_offsets=True."
        )

    from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d

    out_BTD, _ = _fla_causal_conv1d(
        x=x_BTD,
        weight=weight.squeeze(1),
        bias=None,
        activation="silu",
        backend="triton",
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    return out_BTD


@spmd.local_map(
    in_types=({"dp": spmd.S(1), "tp": spmd.S(2)}, None, None),
    out_types={"dp": spmd.S(0), "tp": spmd.S(2)},
)
def unflatten_to_bld(
    tensor: torch.Tensor, batch_size: int, seq_len: int
) -> torch.Tensor:
    return tensor.reshape(batch_size, seq_len, -1)


class OffsetRMSNorm(Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``.

    Weight is zero-initialized so the norm starts as identity-scaled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability in pow/rsqrt
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)


class RMSNormGated(Module):
    """Gated RMSNorm: ``silu(gate) * weight * norm(x)``.

    Takes ``(x, gate)`` separately. Weight is ones-initialized.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability in pow/rsqrt
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = (self.weight.float() * x).to(input_dtype)
        x = x * F.silu(gate.float())
        return x.to(input_dtype)


class GatedDeltaKernel(Module):
    """Stateless dispatch to the configured FLA gated delta kernel.

    Provides a module boundary for the sharding code to wrap forward with
    DTensor→local conversion — same pattern as FlexAttention. Handles Q/K
    head expansion for grouped linear attention internally so that
    repeat_interleave runs on local tensors under TP. A pure-torch reference
    implementation lives in ``tests/unit_tests/test_qwen3_5_deltanet.py``;
    it is far too slow for training use.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # "fla_chunked": parallel within chunks for training (default)
        # "fla_fused_recurrent": for inference only in rl, no backward
        backend: GatedDeltaBackend = "fla_chunked"

    def __init__(self, config: Config):
        super().__init__()
        self.backend = config.backend

    def forward(
        self,
        xq_BLNK: torch.Tensor,
        xk_BLNK: torch.Tensor,
        xv_BLNV: torch.Tensor,
        g_BLN: torch.Tensor,
        beta_BLN: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Expand Q/K heads to match V when n_value_heads > n_key_heads
        if xq_BLNK.shape[2] != xv_BLNV.shape[2]:
            assert xv_BLNV.shape[2] % xq_BLNK.shape[2] == 0
            repeat = xv_BLNV.shape[2] // xq_BLNK.shape[2]
            xq_BLNK = xq_BLNK.repeat_interleave(repeat, dim=2)
            xk_BLNK = xk_BLNK.repeat_interleave(repeat, dim=2)

        if cu_seqlens is not None and xq_BLNK.shape[0] != 1:
            raise ValueError(
                f"Gated DeltaNet varlen kernels require flattened inputs with "
                f"batch size 1, got batch size {xq_BLNK.shape[0]}."
            )

        if self.backend == "fla_chunked":
            if cu_seqlens is not None and cu_seqlens_cpu is None:
                raise ValueError(
                    "Qwen3.5 FLA varlen DeltaNet requires a CPU cu_seqlens tensor."
                )
            result = _fla_chunk_gated_delta_rule(
                xq_BLNK,
                xk_BLNK,
                xv_BLNV,
                g_BLN,
                beta_BLN,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
        elif self.backend == "fla_fused_recurrent":
            result = _fla_fused_recurrent_gated_delta_rule(
                xq_BLNK,
                xk_BLNK,
                xv_BLNV,
                g_BLN,
                beta=beta_BLN,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise ValueError(
                f"Unknown fla_backend '{self.backend}'. "
                "Valid: 'fla_chunked', 'fla_fused_recurrent'."
            )

        # FLA kernels return (output, final_state); we only need output
        return result[0]


class GatedDeltaNet(Module):
    """Gated DeltaNet linear attention.

    Uses recurrent state + gated delta rule instead of softmax attention.
    No RoPE, different head structure from standard attention. Conv and
    recurrent state are reset at document boundaries whenever document
    offsets (``VarlenMetadata``) are provided -- the transformer block picks
    them out of the model's attention-mask dict under the ``"deltanet"`` key
    (both attention backends). With no offsets (``None``) the packed sequence
    is processed as a single continuous stream.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        key_head_dim: int
        value_head_dim: int
        conv_kernel_size: int = 4

        # Sub-module configs
        in_proj_q: Linear.Config
        in_proj_k: Linear.Config
        in_proj_v: Linear.Config
        in_proj_z: Linear.Config
        in_proj_a: Linear.Config
        in_proj_b: Linear.Config
        conv_q: Conv1d.Config
        conv_k: Conv1d.Config
        conv_v: Conv1d.Config
        kernel: GatedDeltaKernel.Config
        norm: RMSNormGated.Config
        out_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.key_head_dim = config.key_head_dim
        self.value_head_dim = config.value_head_dim
        self.conv_kernel_size = config.conv_kernel_size

        value_dim = config.in_proj_v.out_features

        self.in_proj_q = config.in_proj_q.build()
        self.in_proj_k = config.in_proj_k.build()
        self.in_proj_v = config.in_proj_v.build()
        self.in_proj_z = config.in_proj_z.build()
        self.in_proj_a = config.in_proj_a.build()
        self.in_proj_b = config.in_proj_b.build()

        self.conv_q = config.conv_q.build()
        self.conv_k = config.conv_k.build()
        self.conv_v = config.conv_v.build()

        n_value_heads = value_dim // config.value_head_dim
        self.A_log = nn.Parameter(torch.empty(n_value_heads))
        self.dt_bias = nn.Parameter(torch.empty(n_value_heads))

        self.kernel = config.kernel.build()
        self.norm = config.norm.build()
        self.out_proj = config.out_proj.build()
        tensor_logging.register_fwd_bwd(self, ["head_out"])

    @staticmethod
    def _local_map_conv(
        x: DTensor,
        conv: Conv1d,
        conv_fn: Callable[..., torch.Tensor],
        *extra_args: torch.Tensor,
    ) -> torch.Tensor:
        """Run a depthwise, channel-sharded conv on local shards via local_map.

        ``conv_fn`` receives the local (x, weight, *extra_args) tensors. Trailing
        ``extra_args`` (e.g. ``cu_seqlens``) are plain replicated tensors passed
        through with a ``None`` placement (unmapped). Input is channel-sharded
        and the weight is ``Shard(0)``; DTensor-ness and gradient placements are
        restored explicitly.
        """
        x_plc = x.placements
        w = conv.weight
        w_plc = w.placements  # pyrefly: ignore [missing-attribute]
        extra_plc = (None,) * len(extra_args)
        conv_dt = local_map(
            conv_fn,
            out_placements=(x_plc,),
            in_placements=(x_plc, w_plc, *extra_plc),
            in_grad_placements=(x_plc, w_plc, *extra_plc),
            device_mesh=x.device_mesh,
        )
        return conv_dt(x, w, *extra_args)  # pyrefly: ignore

    def _causal_conv(
        self,
        x_BLD: torch.Tensor,
        conv: Conv1d,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            if isinstance(x_BLD, DTensor):

                def _conv_varlen(
                    x_local_BLD: torch.Tensor,
                    w_local: torch.Tensor,
                    cu_seqlens_local: torch.Tensor,
                ) -> torch.Tensor:
                    return _causal_conv1d_varlen(
                        x_local_BLD,
                        w_local,
                        cu_seqlens_local,
                        cu_seqlens_cpu,
                    )

                return self._local_map_conv(x_BLD, conv, _conv_varlen, cu_seqlens)
            else:
                return _causal_conv1d_varlen(
                    x_BLD,
                    conv.weight,
                    cu_seqlens,
                    cu_seqlens_cpu,
                )

        # standard fixed-length convolution path
        x_BDL = F.pad(x_BLD.transpose(1, 2), [self.conv_kernel_size - 1, 0])

        @spmd.local_map(
            in_types=(
                {"dp": spmd.S(2), "tp": spmd.S(1)},
                {"dp": spmd.R, "tp": spmd.S(0)},
            ),
            out_types={"dp": spmd.S(2), "tp": spmd.S(1)},
        )
        def _local_depthwise_conv1d(
            x_local_BDL: torch.Tensor, w_local: torch.Tensor
        ) -> torch.Tensor:
            return F.conv1d(
                x_local_BDL,
                w_local,
                None,
                conv.stride,
                conv.padding,
                conv.dilation,
                w_local.size(0),
            )

        if isinstance(x_BDL, DTensor):
            # TODO: Remove once the DTensor Conv1d dispatch fix for sharded
            # groups lands in a released torch.
            x_BDL = self._local_map_conv(x_BDL, conv, _local_depthwise_conv1d)
        else:
            x_BDL = _local_depthwise_conv1d(x_BDL, conv.weight)
        return F.silu(x_BDL).transpose(1, 2)

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: VarlenMetadata | None = None,
    ) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        cu_seqlens = None
        cu_seqlens_cpu = None
        if attention_masks is not None:
            # FLA caches varlen index helpers by tensor identity. A fresh
            # tensor ensures forward and activation-checkpoint recompute both
            # execute the helpers instead of taking different cache paths.
            with spmd.local():
                cu_seqlens = attention_masks.cu_seq_q.clone()
            cu_seqlens_host = attention_masks.cu_seq_q_host
            if cu_seqlens_host is None:
                raise ValueError(
                    "Qwen3.5 GatedDeltaNet varlen requires CPU cu_seqlens "
                    "metadata. Build VarlenMetadata with include_host_offsets=True."
                )
            # Keep host metadata as Python values in VarlenMetadata so
            # SelectiveAC does not treat integer CPU tensors as checkpoint
            # inputs; build the FLA API tensor at the DeltaNet boundary.
            cu_seqlens_cpu = torch.tensor(
                cu_seqlens_host,
                dtype=cu_seqlens.dtype,
                device="cpu",
            )
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                # Python host metadata loses the DP-varying provenance of the
                # device offsets when it is materialized as a new tensor.
                spmd.mutate_type(cu_seqlens_cpu, "dp", src=spmd.R, dst=spmd.V)

        def fold_bl_dim(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(1, B * L, *tensor.shape[2:])

        # Folded recurrence shapes:
        #   xq_BLNK, xk_BLNK: (1, B * L, n_key_heads, key_head_dim)
        #   xv_BLNV, xz_BLNV: (1, B * L, n_value_heads, value_head_dim)
        #   xa_BLN, xb_BLN: (1, B * L, n_value_heads)
        xq_BLNK = self._causal_conv(
            fold_bl_dim(self.in_proj_q(x_BLD)),
            self.conv_q,
            cu_seqlens,
            cu_seqlens_cpu,
        )
        xq_BLNK = local_head_split(xq_BLNK, self.key_head_dim, dp_shard_dim=1)
        xk_BLNK = self._causal_conv(
            fold_bl_dim(self.in_proj_k(x_BLD)),
            self.conv_k,
            cu_seqlens,
            cu_seqlens_cpu,
        )
        xk_BLNK = local_head_split(xk_BLNK, self.key_head_dim, dp_shard_dim=1)
        xv_BLNV = self._causal_conv(
            fold_bl_dim(self.in_proj_v(x_BLD)),
            self.conv_v,
            cu_seqlens,
            cu_seqlens_cpu,
        )
        xv_BLNV = local_head_split(xv_BLNV, self.value_head_dim, dp_shard_dim=1)
        xz_BLNV = local_head_split(
            fold_bl_dim(self.in_proj_z(x_BLD)),
            self.value_head_dim,
            dp_shard_dim=1,
        )
        xa_BLN = fold_bl_dim(self.in_proj_a(x_BLD))
        xb_BLN = fold_bl_dim(self.in_proj_b(x_BLD))

        # Gating signals have shape (1, B * L, n_value_heads):
        #   g_BLN:    decay rate per head, always negative
        #   beta_BLN: update gate in (0, 1)
        g_BLN = -torch.exp(self.A_log.float()) * F.softplus(
            xa_BLN.float() + self.dt_bias
        )
        beta_BLN = torch.sigmoid(xb_BLN)

        out_BLNV = self.kernel(
            xq_BLNK,
            xk_BLNK,
            xv_BLNV,
            g_BLN,
            beta_BLN,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

        out_BLNV = self.norm(out_BLNV, xz_BLNV)
        tensor_logging.log_fwd_bwd_stats(self, head_out=out_BLNV)

        # Merge value heads and restore (B, L) from the folded (1, B * L) layout.
        out_BLD = unflatten_to_bld(out_BLNV, B, L)
        return self.out_proj(out_BLD)


class Qwen35Attention(BaseAttention):
    """Full attention with output gating and partial RoPE for Qwen3.5.

    Differences from GQAttention:
    - wq is 2x wider: produces both query and sigmoid gate
    - Partial RoPE: only first ``rotary_dim`` elements get RoPE
    - Output gating: ``attn_output * sigmoid(gate)`` before ``wo``
    - QK norm uses OffsetRMSNorm

    Uses separate ``wq``/``wk``/``wv`` instead of the common fused ``qkv_linear``
    (so this subclasses ``BaseAttention``, not ``GQAttention``): the 2x-wide,
    gated ``wq`` doesn't fit a fused QKV projection that TP-shards by head.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        n_kv_heads: int
        head_dim: int
        rotary_dim: int
        rope: MRoPE.Config
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        wo: Linear.Config
        q_norm: OffsetRMSNorm.Config
        k_norm: OffsetRMSNorm.Config
        inner_attention: Module.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.wo = config.wo.build()

        self.rope = config.rope.build()

        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()

        self.scaling = self.head_dim**-0.5

        self.inner_attention = config.inner_attention.build()
        tensor_logging.register_fwd_bwd(
            self,
            [
                "xq",
                "xk",
                "xv",
                "xq_normed",
                "xk_normed",
                "output_gate",
                "head_out_pre_gate",
                "head_out",
            ],
        )

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x_BLD.shape

        # wq is 2x wider: produces query + gate
        xq_gate_BLN2H = local_head_split(self.wq(x_BLD), self.head_dim * 2)
        xq_BLNH, gate_BLNH = xq_gate_BLN2H.chunk(2, dim=-1)
        xk_BLNH = local_head_split(self.wk(x_BLD), self.head_dim)
        xv_BLNH = local_head_split(self.wv(x_BLD), self.head_dim)
        tensor_logging.log_fwd_bwd_stats(
            self,
            xq=xq_BLNH,
            xk=xk_BLNH,
            xv=xv_BLNH,
        )

        # QK norm (before RoPE)
        xq_BLNH = self.q_norm(xq_BLNH)
        xk_BLNH = self.k_norm(xk_BLNH)
        tensor_logging.log_fwd_bwd_stats(
            self,
            xq_normed=xq_BLNH,
            xk_normed=xk_BLNH,
        )

        # Partial RoPE: only first rotary_dim elements get positional encoding
        assert self.rotary_dim <= self.head_dim
        xq_BLNR, xq_BLNP = (
            xq_BLNH[..., : self.rotary_dim],
            xq_BLNH[..., self.rotary_dim :],
        )
        xk_BLNR, xk_BLNP = (
            xk_BLNH[..., : self.rotary_dim],
            xk_BLNH[..., self.rotary_dim :],
        )
        xq_BLNR, xk_BLNR = self.rope(xq_BLNR, xk_BLNR, positions)
        xq_BLNH = torch.cat([xq_BLNR, xq_BLNP], dim=-1)
        xk_BLNH = torch.cat([xk_BLNR, xk_BLNP], dim=-1)

        head_out_pre_gate = self.inner_attention(
            xq_BLNH,
            xk_BLNH,
            xv_BLNH,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()

        # Output gating
        output_gate = torch.sigmoid(gate_BLNH)
        head_out = head_out_pre_gate * output_gate
        tensor_logging.log_fwd_bwd_stats(
            self,
            output_gate=output_gate,
            head_out_pre_gate=head_out_pre_gate,
            head_out=head_out,
        )
        out_BLD = head_out.view(B, L, -1)
        return self.wo(out_BLD)


class Qwen35TransformerBlock(Module):
    """Hybrid transformer block for Qwen3.5.

    Each layer uses either full attention (Qwen35Attention) or linear
    attention (GatedDeltaNet), determined by which config is provided.
    Both types share the same FFN/MoE structure.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: Qwen35Attention.Config | None = None
        delta_net: GatedDeltaNet.Config | None = None
        feed_forward: Module.Config | None = None
        moe: Module.Config | None = None
        attention_norm: OffsetRMSNorm.Config
        ffn_norm: OffsetRMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.full_attn = config.attention is not None
        self.attn_mask_key = "quadratic_attention" if self.full_attn else "deltanet"

        if self.full_attn:
            self.attn = config.attention.build()  # pyrefly: ignore [missing-attribute]
        else:
            assert config.delta_net is not None
            self.attn = config.delta_net.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            # pyrefly: ignore [missing-attribute]
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: Qwen35AttentionMaskDict | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer_mask = (
            attention_masks[self.attn_mask_key] if attention_masks is not None else None
        )

        h_BLD = self.attention_norm(x_BLD)
        if self.full_attn:
            h_BLD = self.attn(h_BLD, layer_mask, positions)
        else:
            h_BLD = self.attn(h_BLD, layer_mask)
        x_BLD = x_BLD + h_BLD

        h_BLD = self.ffn_norm(x_BLD)
        if self.moe_enabled:
            x_BLD = x_BLD + self.moe(h_BLD)
        else:
            x_BLD = x_BLD + self.feed_forward(h_BLD)
        return x_BLD


class Qwen35Model(Decoder):
    """Qwen3.5: Multimodal model with hybrid attention.

    Combines a hybrid decoder (GatedDeltaNet linear attention + full
    attention with output gating and partial RoPE) with a Vision
    Transformer encoder for multimodal understanding.

    Key architectural features:
    - Hybrid attention: 75% GatedDeltaNet (linear) + 25% full attention
    - Output gating on full attention: ``attn_out * sigmoid(gate)``
    - Partial RoPE: only first ``rotary_dim`` elements get positional encoding
    - OffsetRMSNorm: ``(1 + weight) * norm(x)`` with zero-init weight
    - MRoPE: 3D (temporal/height/width) position IDs for multimodal batches;
      text batches use the plain 1D positions
    - MoE variant: routed experts + shared expert with sigmoid gate

    MRoPE positions (``mrope_positions``, shape ``(batch, seq, 3)``) are built by
    the dataloader and forwarded to every pipeline stage, so RoPE stays consistent
    across stages even though the raw vision inputs (``pixel_values``/``grid_thw``)
    only reach the first stage. Text batches carry no ``mrope_positions`` and use
    the 2D ``positions`` instead.

    Forward pass flow::

        forward(tokens, pixel_values, grid_thw, mrope_positions, ...)
          │
          ├─ _prepare_multimodal_embeds
          │    ├─ tok_embeddings(tokens)              → text embeddings
          │    ├─ _get_vision_embeds(pixel_values)     → vision embeddings
          │    │    └─ vision_encoder(pixel_values)     → merge patches
          │    ├─ _get_vision_positions             → locate vision regions
          │    └─ _scatter_vision_embeds                → scatter into text sequence
          │
          └─ transformer layers (hybrid), each given (mrope_positions or positions)
               └─ for each layer:
                    ├─ full attention (every Nth):  QK-norm → partial RoPE → SDPA → gate
                    │    (the layer's MRoPE builds the cos/sin cache from positions)
                    └─ GatedDeltaNet (others):      Conv1d → gated delta rule → gated norm
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        vision_encoder: Qwen35VisionEncoder.Config

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                dn_cfg = next(
                    (
                        layer_cfg.delta_net
                        for layer_cfg in self.layers
                        if layer_cfg.delta_net is not None
                    ),
                    None,
                )
                if dn_cfg is not None:
                    n_key_heads = dn_cfg.in_proj_q.out_features // dn_cfg.key_head_dim
                    n_value_heads = (
                        dn_cfg.in_proj_v.out_features // dn_cfg.value_head_dim
                    )
                    if n_key_heads % tp != 0 or n_value_heads % tp != 0:
                        raise ValueError(
                            f"tensor_parallel_degree ({tp}) must divide "
                            f"n_key_heads ({n_key_heads}) and "
                            f"n_value_heads ({n_value_heads})."
                        )

            set_qwen35_sharding_config(
                self,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            # The shared helper excludes the vision encoder from the per-token
            # FLOP term (ViT cost scales with patches, not seq_len), so this MFU
            # is decoder-only. TODO: add a per-batch vision FLOP term for VLMs.
            attn_cfg = self.first_attention
            # pyrefly: ignore [missing-attribute]
            n_heads = attn_cfg.n_heads
            # pyrefly: ignore [missing-attribute]
            head_dim = attn_cfg.head_dim
            return get_moe_model_nparams_and_flops(
                self,
                model,
                n_heads,
                2 * head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)

        self.vision_encoder = config.vision_encoder.build()
        self.spatial_merge_size = config.vision_encoder.spatial_merge_size
        tensor_logging.register_fwd_bwd(
            self,
            ["vision_embeddings_after_projection"],
        )

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> Qwen35AttentionMaskDict:
        """Build the per-consumer mask dict for the hybrid stack.

        A ``BlockMask`` isolates documents in the quadratic layers. The value
        is ``None`` if the config has no quadratic layer. GatedDeltaNet uses
        document offsets under the ``"deltanet"`` key. Each block selects its
        value by ``attn_mask_key``. The trainer builds this dictionary for
        each pipeline microbatch.
        """
        attn_config = self.config.first_attention

        # Host offsets are a GatedDeltaNet-only need: the FLA varlen kernels
        # take cu_seqlens as a CPU tensor to size their launches, whereas
        # quadratic attention (torch.nn.attention.varlen) consumes the device
        # tensor directly. They are stored as Python ints so SelectiveAC
        # checkpoint metadata stays tensor-free.
        deltanet_metadata = create_varlen_metadata_for_document(
            positions,
            include_host_offsets=True,
        )
        if attn_config is None:
            quadratic_attention = None
        elif isinstance(attn_config.inner_attention, VarlenAttention.Config):
            # Under varlen both consumers read the same document offsets.
            quadratic_attention = deltanet_metadata
        else:
            quadratic_masks = super().get_attention_masks(positions)
            assert isinstance(quadratic_masks, BlockMask)
            quadratic_attention = quadratic_masks
        return {
            "quadratic_attention": quadratic_attention,
            "deltanet": deltanet_metadata,
        }

    def _get_vision_positions(
        self,
        tokens: torch.Tensor,
        num_tokens_per_item: torch.Tensor,
        vision_token_id: int,
    ) -> list[tuple[int, int, int, int]]:
        """Compute (item_idx, sample_idx, vision_start, n_tokens) for each vision item.

        Finds where each contiguous run of vision placeholder tokens starts
        in the text sequence.

        Args:
            tokens: Token IDs (batch, seq_len)
            num_tokens_per_item: (num_items,) actual tokens per vision item
            vision_token_id: Placeholder token ID

        Returns:
            List of (item_idx, sample_idx, vision_start, n_tokens) tuples
        """
        vision_mask = tokens == vision_token_id
        flat_mask = vision_mask.view(-1)
        prev_mask = torch.cat(
            [torch.zeros(1, dtype=torch.bool, device=flat_mask.device), flat_mask[:-1]]
        )
        region_starts = torch.where(flat_mask & ~prev_mask)[0]
        seq_len = tokens.shape[1]

        positions = []
        for i in range(num_tokens_per_item.shape[0]):
            start = int(region_starts[i].item())
            n_tokens = int(num_tokens_per_item[i].item())
            positions.append((i, start // seq_len, start % seq_len, n_tokens))
        return positions

    def _get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run vision encoder and return padded embeddings with token counts.

        Args:
            pixel_values: Padded patches (num_items, max_num_patch, patch_dim)
            grid_thw: Grid dimensions (num_items, 3) for [t, h, w]

        Returns:
            vision_embeds: (num_items, max_tokens, dim) padded vision embeddings
            num_tokens_per_item: (num_items,) actual token count per item
        """
        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)
        tensor_logging.log_fwd_bwd_stats(
            self,
            vision_embeddings_after_projection=vision_embeds,
        )

        merge_unit = self.vision_encoder.spatial_merge_unit
        num_tokens_per_item = grid_thw.prod(-1) // merge_unit

        return vision_embeds, num_tokens_per_item

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int],
    ) -> torch.Tensor:
        """Embed tokens, run vision encoder, scatter vision into text.

        Args:
            tokens: Input token IDs (batch_size, seq_len)
            pixel_values: Image patches or None
            pixel_values_videos: Video patches or None
            grid_thw: Grid dimensions for images or None
            grid_thw_videos: Grid dimensions for videos or None
            special_tokens: Special token definitions

        Returns:
            (batch, seq_len, dim) embeddings with vision tokens scattered in
        """
        image_token_id = special_tokens["image_id"]
        video_token_id = special_tokens["video_id"]

        inputs_embeds = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )
        if self.tok_embeddings is not None:
            tensor_logging.log_fwd_bwd_stats(self, input=inputs_embeds)

        if pixel_values is not None and grid_thw is not None:
            vision_embeds, num_tokens = self._get_vision_embeds(
                pixel_values, grid_thw=grid_thw
            )
            image_positions = get_vision_positions(tokens, num_tokens, image_token_id)
            if image_positions:
                inputs_embeds = scatter_vision_embeds(
                    inputs_embeds,
                    vision_embeds=vision_embeds,
                    vision_positions=image_positions,
                )

        if pixel_values_videos is not None and grid_thw_videos is not None:
            vision_embeds, num_tokens = self._get_vision_embeds(
                pixel_values_videos, grid_thw=grid_thw_videos
            )
            video_positions = get_vision_positions(tokens, num_tokens, video_token_id)
            if video_positions:
                inputs_embeds = scatter_vision_embeds(
                    inputs_embeds,
                    vision_embeds=vision_embeds,
                    vision_positions=video_positions,
                )

        return inputs_embeds

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_masks: Qwen35AttentionMaskDict | None = None,
        positions: torch.Tensor | None = None,
        mrope_positions: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
    ):
        with multimodal_context():
            if get_spmd_backend() == "spmd_types":
                annotate_qwen35_input_spmd_types(
                    attention_masks=attention_masks,
                    mrope_positions=mrope_positions,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    grid_thw=grid_thw,
                    grid_thw_videos=grid_thw_videos,
                )

            if self.tok_embeddings is not None:
                x = self._prepare_multimodal_embeds(
                    tokens,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    grid_thw=grid_thw,
                    grid_thw_videos=grid_thw_videos,
                    special_tokens=special_tokens,  # pyrefly: ignore [bad-argument-type]
                )
            else:
                x = tokens

        if get_spmd_backend() == "spmd_types":
            # The scatter restores a token-aligned tensor, so text-model DP
            # resumes as global batch sharding after the multimodal region.
            spmd.assert_type(x, {"dp": spmd.S(0), "tp": spmd.R})

        # 3D MRoPE positions for multimodal batches, else 2D text positions.
        rope_positions = mrope_positions if mrope_positions is not None else positions
        assert rope_positions is not None
        for layer in self.layers.values():
            x = layer(x, attention_masks, rope_positions)

        x = self.norm(x) if self.norm is not None else x
        if self._skip_lm_head:
            return x
        if self.lm_head is None:
            return x
        output = self.lm_head(x)
        tensor_logging.log_fwd_bwd_stats(self.lm_head, output=output)
        return output
