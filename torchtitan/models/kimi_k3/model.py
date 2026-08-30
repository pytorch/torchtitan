# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.distributed.fsdp import add_zero_valued_dependency

from torchtitan.hf_datasets.multimodal.mm_datasets import MMSamplePackingConfig
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    FlexAttention,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.multimodal import (
    get_vision_positions,
    scatter_vision_embeds,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger

from .kda import KimiDeltaAttention
from .moe import KimiFeedForward, KimiLatentMoE
from .sharding import mla_ulysses_attention
from .vision_encoder import KimiK3VisionEncoder

# Shape suffixes:
# T = packed tokens, D = model dimension, H = heads,
# K = key head dimension, V = value head dimension,
# N = attention-residual entries.


class KimiMLAAttention(BaseAttention):
    """Kimi K3 multi-head latent attention.

    Unlike DeepSeek-V3 MLA, the released K3 configuration sets
    ``mla_use_nope=True``: the RoPE-sized query/key slices remain part of the
    projected head, but no rotary transform is applied, so this has no rope
    config at all. Attention delegates to the configured inner backend.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv_a: Linear.Config
        kv_norm: RMSNorm.Config
        wkv_b: Linear.Config
        gate: Linear.Config
        wo: Linear.Config
        inner_attention: Module.Config = field(default_factory=FlexAttention.Config)

    # Set by apply_cp_kimi_k3; None means the layer runs without CP. MLA is
    # Ulysses under either KDA CP mode -- KCP describes a recurrence that MLA
    # does not have.
    _cp_group = None

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.gate = config.gate.build()
        self.wo = config.wo.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        num_tokens = x_TD.shape[0]
        # The head count is DERIVED from the projection width, not read off
        # self.n_heads. Ulysses splits whatever head count this rank actually
        # holds, and the two differ once another parallelism has already split
        # the head axis; deriving works either way and needs no branch.
        q_proj_TE = self.wq_b(self.q_norm(self.wq_a(x_TD)))
        h_local = q_proj_TE.shape[-1] // self.q_head_dim
        q_THK = q_proj_TE.view(num_tokens, h_local, self.q_head_dim)

        compressed_kv_TC = self.wkv_a(x_TD)
        kv_latent_TC, k_rope_TK = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_THC = self.wkv_b(self.kv_norm(kv_latent_TC)).view(
            num_tokens,
            h_local,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope_THK, v_THV = torch.split(
            kv_THC,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        k_rope_THK = k_rope_TK.view(num_tokens, 1, self.qk_rope_head_dim).expand(
            -1, h_local, -1
        )
        k_THK = torch.cat((k_nope_THK, k_rope_THK), dim=-1)

        cp_group = self._cp_group
        if cp_group is not None and dist.get_world_size(cp_group) > 1:
            out_THV = mla_ulysses_attention(
                self, q_THK, kv_THC, k_rope_TK, cp_group, positions
            )
        else:
            out_THV = self.inner_attention(
                q_THK,
                k_THK,
                v_THV,
                attention_masks=attention_masks,
                scale=self.scale,
            )
        out_TD = out_THV.reshape(num_tokens, h_local * self.v_head_dim)
        out_TD = out_TD * torch.sigmoid(self.gate(x_TD))
        return self.wo(out_TD)


def _apply_attention_residual(
    prefix_sum_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection: Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    """Apply Kimi's block-level attention residual in FP32.

    TODO: Add TP Support. The current implementation assumes that the input tensors are on a single device.
    """
    assert norm.eps is not None

    values_TND = torch.cat((block_residual_TND, prefix_sum_TD.unsqueeze(1)), dim=1)
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + norm.eps)
    score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    output_TD = torch.matmul(probs_T1N, values_float).squeeze(1)
    return output_TD.to(values_TND.dtype)


class KimiK3TransformerBlock(Module):
    """Hybrid KDA/MLA decoder block with Kimi attention residuals."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        attn_res_block_size: int
        attention: KimiMLAAttention.Config | None
        delta_attention: KimiDeltaAttention.Config | None
        feed_forward: KimiFeedForward.Config | None
        moe: KimiLatentMoE.Config | None
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config
        attention_res_norm: RMSNorm.Config | None
        attention_res_proj: Linear.Config | None
        ffn_res_norm: RMSNorm.Config
        ffn_res_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        if (config.attention is None) == (config.delta_attention is None):
            raise ValueError(
                "Exactly one of attention or delta_attention must be configured."
            )
        if (config.feed_forward is None) == (config.moe is None):
            raise ValueError("Exactly one of feed_forward or moe must be configured.")
        self.layer_id = config.layer_id
        self.attn_res_block_size = config.attn_res_block_size
        self.attention = (
            config.attention.build() if config.attention is not None else None
        )
        self.delta_attention = (
            config.delta_attention.build()
            if config.delta_attention is not None
            else None
        )
        self.feed_forward = (
            config.feed_forward.build() if config.feed_forward is not None else None
        )
        self.moe = config.moe.build() if config.moe is not None else None
        self.moe_enabled = self.moe is not None
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.attention_res_norm = (
            config.attention_res_norm.build()
            if config.attention_res_norm is not None
            else None
        )
        self.attention_res_proj = (
            config.attention_res_proj.build()
            if config.attention_res_proj is not None
            else None
        )
        self.ffn_res_norm = config.ffn_res_norm.build()
        self.ffn_res_proj = config.ffn_res_proj.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        block_residual_TND: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_sum_TD = x_TD

        if block_residual_TND.shape[1] > 0:
            assert self.attention_res_proj is not None
            assert self.attention_res_norm is not None
            x_TD = _apply_attention_residual(
                prefix_sum_TD,
                block_residual_TND,
                self.attention_res_proj,
                self.attention_res_norm,
            )

        opens_block = self.layer_id % self.attn_res_block_size == 0
        if opens_block:
            block_residual_TND = torch.cat(
                (
                    block_residual_TND,
                    prefix_sum_TD.unsqueeze(1),
                ),
                dim=1,
            )

        h_TD = self.attention_norm(x_TD)
        if self.attention is not None:
            h_TD = self.attention(h_TD, attention_masks, positions)
        else:
            assert self.delta_attention is not None
            h_TD = self.delta_attention(h_TD, None, positions)
        prefix_sum_TD = h_TD if opens_block else prefix_sum_TD + h_TD

        h_TD = _apply_attention_residual(
            prefix_sum_TD,
            block_residual_TND,
            self.ffn_res_proj,
            self.ffn_res_norm,
        )
        h_TD = self.ffn_norm(h_TD)
        if self.moe is not None:
            h_TD = self.moe(h_TD)
        else:
            assert self.feed_forward is not None
            h_TD = self.feed_forward(h_TD)
        return prefix_sum_TD + h_TD, block_residual_TND


class _PlainGradBoundary(torch.autograd.Function):
    """Identity forward; forces the incoming gradient to be a plain tensor.

    The vision tower must stay plain in BOTH directions. Its dynamic CP is a
    separate mechanism from the decoder's, and the CP path runs hand-written
    collectives whose transpose is a reduce_scatter --
    _c10d_functional.reduce_scatter_tensor has no DTensor sharding strategy.

    to_local() alone is not enough and grad_placements is the wrong knob: the
    first re-wraps the gradient with the forward placements, the second states
    which placements to re-wrap WITH. Neither can say "do not re-wrap". That is
    what this states, and only an autograd.Function can.
    """

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        return grad.to_local() if isinstance(grad, DTensor) else grad


class KimiK3Model(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        layers: list[KimiK3TransformerBlock.Config]
        output_res_norm: RMSNorm.Config
        output_res_proj: Linear.Config
        vision_encoder: KimiK3VisionEncoder.Config | None = None
        # Smallest image worth partitioning across CP ranks. Below it the
        # replicated encode is cheaper: splitting buys one gather per layer.
        dynamic_cp_min_patches: int = 256

        def _validate_cp_backend(self, parallelism) -> None:
            """This model's CP is not ShardingConfig-driven -- the KDA kernels
            are fla triton and never see a DTensor -- so the spmd_types
            requirement does not apply; apply_cp_kimi_k3 checks its own
            preconditions at wiring time."""

        def update_from_config(self, *, config, **kwargs) -> None:
            dataset = config.dataloader.dataset
            # TODO: Support sample packing by resetting the Q/K/V causal-convolution
            # and KDA recurrent states at document boundaries.
            if isinstance(dataset, MMSamplePackingConfig):
                raise ValueError("Kimi K3 does not yet support sample packing.")
            parallelism = config.parallelism
            if (
                parallelism.context_parallel_degree > 1
                and parallelism.context_parallel_load_balancer is not None
            ):
                # Both CP algorithms here read the sequence as rank-ordered
                # contiguous chunks: the Ulysses all-to-all reassembles it in
                # rank order, and KDA's recurrence passes state from rank r to
                # rank r+1. A load balancer permutes tokens across ranks, which
                # silently breaks both -- the shapes still line up.
                raise ValueError(
                    "Kimi K3 context parallel requires "
                    "parallelism.context_parallel_load_balancer=None; "
                    f"got {parallelism.context_parallel_load_balancer!r}."
                )
            Decoder.Config.update_from_config(self, config=config, **kwargs)

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attention_config = self.first_attention
            if not isinstance(attention_config, KimiMLAAttention.Config):
                raise ValueError(
                    "Kimi K3 requires at least one MLA layer for FLOP accounting."
                )
            # KDA and the vision encoder have no dedicated term here, so their
            # parameters only contribute the dense 6*N estimate; reported MFU is
            # approximate.
            return get_moe_model_nparams_and_flops(
                self,
                model,
                attention_config.n_heads,
                attention_config.qk_nope_head_dim
                + attention_config.qk_rope_head_dim
                + attention_config.v_head_dim,
                seq_len,
            )

    # Set by apply_cp_kimi_k3 to this model's context-parallel process group.
    _cp_group = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.output_res_norm = config.output_res_norm.build()
        self.output_res_proj = config.output_res_proj.build()
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )
        self.dynamic_cp_min_patches = config.dynamic_cp_min_patches
        self._dyncp_logged = False

    def _tower_needs_collectives(self) -> bool:
        """Is the tower wrapped in something that issues per-forward collectives?

        True once FSDP has sharded it, which is when skipping it desynchronizes
        the process group. A replicated DTensor -- what a tp-invariant module
        holds -- issues no all-gather to match, so the test is on the placement
        and not merely on the type.
        """
        return any(
            isinstance(p, DTensor) and any(pl.is_shard() for pl in p.placements)
            for p in self.vision_encoder.parameters()
        )

    def _tower_placeholder(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The smallest input the tower accepts, for a rank with no images."""
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        grid = torch.tensor(
            [[1, kernel_h, kernel_w]], dtype=torch.long, device=self._device()
        )
        weight = self.vision_encoder.patch_embed.weight
        # A plain tensor, not weight.new_zeros: once FSDP has sharded the
        # tower the weight is a DTensor, and a placeholder inheriting that
        # meets the tower's own plain tensors as "aten.mm got mixed".
        patches = torch.zeros(
            kernel_h * kernel_w,
            weight.shape[-1],
            dtype=weight.dtype,
            device=self._device(),
        )
        return patches, grid

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
    ) -> torch.Tensor:
        embeddings_TD = self.tok_embeddings(tokens)
        if (pixel_values is None) != (grid_thw is None):
            raise ValueError(
                "pixel_values and grid_thw must either both be provided or "
                "both be omitted."
            )
        if pixel_values is None:
            # An image-free batch is normal, but FSDP2 issues the tower's
            # all-gather from its pre-forward hook, so every rank must run it.
            # A zero-valued placeholder keeps collectives and the DP average right.
            if self.vision_encoder is not None and self._tower_needs_collectives():
                placeholder, placeholder_grid = self._tower_placeholder()
                unused = self.vision_encoder(placeholder, grid_thw=placeholder_grid)
                return add_zero_valued_dependency(embeddings_TD, unused)
            return embeddings_TD
        assert grid_thw is not None
        if self.vision_encoder is None:
            raise ValueError("pixel_values were provided without a vision encoder.")
        if special_tokens is None:
            raise ValueError("special_tokens are required for multimodal inputs.")

        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self._encode_images(pixel_values, grid_thw)
        # MoonViT collapses time and merges spatially, so the text-side token
        # count per item is (h/kh)*(w/kw), independent of t.
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        num_tokens_per_item = (grid_thw[:, 1] // kernel_h) * (
            grid_thw[:, 2] // kernel_w
        )
        if self._cp_group is not None and dist.get_world_size(self._cp_group) > 1:
            # This rank holds a sequence shard but encoded every image: take the
            # feature slice its placeholders correspond to and scatter that.
            # get_vision_positions needs whole visual items, which a shard does
            # not have -- it raises "found N contiguous run(s) ... but received M
            # visual item(s)" as soon as a shard splits or omits an item.
            local_mask = tokens == special_tokens["image_id"]
            counts = self._exchange_sentinel_counts(int(local_mask.sum().item()))
            mine = self._select_cp_shard(vision_embeds, counts)
            embeddings_TD = embeddings_TD.masked_scatter(
                local_mask.unsqueeze(-1), mine.to(embeddings_TD.dtype)
            )
            # Rows this rank did not consume still have to reach the graph, or
            # the tower's reduce-scatter is issued by a subset of the group.
            return add_zero_valued_dependency(embeddings_TD, vision_embeds)

        vision_positions = get_vision_positions(
            tokens,
            num_tokens_per_item,
            special_tokens["image_id"],
        )
        return scatter_vision_embeds(
            embeddings_TD,
            vision_embeds=vision_embeds,
            vision_positions=vision_positions,
        )

    def _exchange_sentinel_counts(self, local: int) -> torch.Tensor:
        """Per-rank vision-placeholder counts across the CP group.

        Called whenever CP is on, including on ranks holding no placeholders:
        the collective's participants are decided by the mesh, never by the data.
        """
        group = self._cp_group
        counts = torch.zeros(
            dist.get_world_size(group),
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        counts[dist.get_rank(group)] = local
        dist.all_reduce(counts, group=group)
        return counts

    def _select_cp_shard(
        self, vision_embeds: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        """Keep only the visual features belonging to this CP rank's shard.

        ``prepare_context_parallel_input`` shards inputs, labels and positions
        along the sequence but leaves ``pixel_values`` whole, so every rank
        encodes every image while holding only a slice of the placeholders. The
        features are ordered by sequence position and the shards are contiguous
        and equal -- the config rejects a load balancer under CP precisely
        because a permuting one would break that -- so this rank's slice starts
        after however many placeholders the lower ranks hold.

        This is correctness, not an optimization: the encoder still runs
        redundantly on every CP rank.
        """
        num_rows = vision_embeds.shape[0]
        if int(counts.sum().item()) != num_rows:
            raise ValueError(
                f"CP ranks hold {int(counts.sum().item())} vision "
                f"placeholder(s) in total but {num_rows} visual token(s) were "
                "encoded; the sequence shard and the image batch disagree"
            )
        rank = dist.get_rank(self._cp_group)
        start = int(counts[:rank].sum().item())
        local = int(counts[rank].item())
        return vision_embeds[start : start + local]

    def _encode_images(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Encode every image, partitioning the large ones (report sec 5.2.3).

        Report 5.2.3 has two halves and both are needed: a single large image
        is split along the patch dimension with attention gathering key-value
        pairs across ranks, AND each CP group is divided into sub-CP groups
        with the large images distributed across them, which is what keeps the
        communication fraction from growing with scale.

        Every large image is encoded by one sub-CP group, with its patches split
        across that sub-group's ranks. Images below the threshold, or whose grid
        height does not divide the merge kernel, stay whole and are encoded
        replicated -- splitting one buys a gather per layer and saves nothing.
        """
        import torch.distributed._functional_collectives as funcol

        from torchtitan.models.kimi_k3.vision_encoder import CPPatchPlan
        from torchtitan.models.kimi_k3.vit_cp_plan import (
            balance_images,
            classify,
            merged_tokens,
            row_partition,
            subgroup_layout,
        )

        grids = grid_thw.tolist()
        counts = [t * h * w for t, h, w in grids]
        kh, kw = self.vision_encoder.merge_kernel_size
        offsets = [0]
        for c in counts:
            offsets.append(offsets[-1] + c)

        def _replicated(which: list[int]) -> dict[int, torch.Tensor]:
            """Encode a subset redundantly on every rank."""
            out = {}
            for i in which:
                item = pixel_values[offsets[i] : offsets[i + 1]]
                item_grid = torch.tensor(
                    [grids[i]], dtype=grid_thw.dtype, device=grid_thw.device
                )
                out[i] = self.vision_encoder(item, grid_thw=item_grid)
            return out

        subgroups = getattr(self, "_cp_subgroups", None)
        group_all = self._cp_group
        cp_size = dist.get_world_size(group_all) if group_all is not None else 1
        if not subgroups or cp_size <= 1:
            return torch.cat(
                [_replicated(list(range(len(counts))))[i] for i in range(len(counts))],
                dim=0,
            )

        large = classify(counts, cp_size, min_patches=self.dynamic_cp_min_patches)
        # Grid heights must divide the merge kernel for a partition to be legal.
        # An image that fails it is left replicated instead of being cut unsafely.
        large = [i for i in large if grids[i][1] % kh == 0]
        if not large:
            return torch.cat(
                [_replicated(list(range(len(counts))))[i] for i in range(len(counts))],
                dim=0,
            )

        n_sub, g = subgroup_layout(len(large), cp_size)
        group = subgroups.get(n_sub)
        if group is None or g <= 1:
            # No usable sub-group of size > 1 means there is nothing to partition
            # across.
            return torch.cat(
                [_replicated(list(range(len(counts))))[i] for i in range(len(counts))],
                dim=0,
            )

        cp_rank = dist.get_rank(group_all)
        my_sub = cp_rank // g
        rank_in_sub = cp_rank % g
        group_of = balance_images([counts[i] for i in large], n_sub)
        my_large = [img for img, sub in zip(large, group_of) if sub == my_sub]

        if not self._dyncp_logged:
            self._dyncp_logged = True
            logger.info(
                "Dynamic CP: %d large image(s) of %d over %d sub-CP group(s) of "
                "%d rank(s); min_patches=%d.",
                len(large),
                len(counts),
                n_sub,
                g,
                self.dynamic_cp_min_patches,
            )

        out: dict[int, torch.Tensor] = {}
        # Every sub-group must run the same NUMBER of passes or the collectives
        # inside them desynchronise. The count is the max over sub-groups, and a
        # sub-group with fewer images pads with an empty pass.
        per_sub = [sum(1 for s in group_of if s == k) for k in range(n_sub)]
        n_passes = max(per_sub) if per_sub else 0

        for p in range(n_passes):
            img = my_large[p] if p < len(my_large) else None
            if img is None:
                # An empty pass still joins this sub-group's collectives. One
                # merge block keeps every shape valid; the output is discarded.
                local = pixel_values.new_zeros(kh * kw, *pixel_values.shape[1:])
                local_grid = torch.tensor(
                    [[1, kh, kw]], dtype=grid_thw.dtype, device=grid_thw.device
                )
                plan = CPPatchPlan(
                    group=group,
                    valid_total=kh * kw * g,
                    full_grid=(1, kh * g, kw),
                    row_start=0,
                    band=kh,
                    real_rows=kh,
                )
            else:
                t, h, w = grids[img]
                shards = row_partition(t, h, w, kh=kh, group_size=g)
                sh = shards[rank_in_sub]
                bands = [s.row_end - s.row_start for s in shards]
                band = max(bands)
                # The ceiling split keeps any deficit on the TRAILING ranks, so
                # every rank's padding lands at the end of the gathered stream
                # rather than inside it. Taking a prefix below depends on that.
                if bands != sorted(bands, reverse=True):
                    raise AssertionError(
                        f"bands {bands} are not non-increasing; padding would land "
                        "inside the gathered token stream and corrupt the order"
                    )
                flat = pixel_values[offsets[img] : offsets[img + 1]]
                # This rank's rows of EVERY frame: the projector's temporal mean
                # spans all frames, so splitting by frame would give each rank the
                # mean of its own frames instead.
                pad_rows = band - (sh.row_end - sh.row_start)
                pieces = []
                for a, b in sh.ranges:
                    pieces.append(flat[a:b])
                    if pad_rows:
                        pieces.append(flat.new_zeros(pad_rows * w, *flat.shape[1:]))
                local = torch.cat(pieces, dim=0)
                local_grid = torch.tensor(
                    [[t, band, w]], dtype=grid_thw.dtype, device=grid_thw.device
                )
                plan = CPPatchPlan(
                    group=group,
                    valid_total=counts[img],
                    full_grid=(t, h, w),
                    row_start=sh.row_start,
                    band=band,
                    real_rows=sh.row_end - sh.row_start,
                )

            local = local.to(self.vision_encoder.patch_embed.weight.dtype)
            feats = self.vision_encoder(local, grid_thw=local_grid, cp_plan=plan)
            # to_local unwraps the value but its backward re-wraps the gradient,
            # and the all_gather below has a reduce_scatter transpose with no
            # DTensor rule.
            if isinstance(feats, DTensor):
                feats = feats.to_local()
            local_feat = _PlainGradBoundary.apply(feats)
            # The boundary belongs on the OUTPUT too: the gradient arrives from
            # downstream, so sealing only the input leaves the transpose
            # receiving a DTensor.
            gathered = _PlainGradBoundary.apply(
                funcol.all_gather_tensor(
                    local_feat.contiguous(), gather_dim=0, group=group
                )
            )
            if img is not None:
                t, h, w = grids[img]
                # NOT counts // merge: the projector collapses time, so a video's
                # token count carries no t.
                out[img] = gathered[: merged_tokens(h, w, kh, kw)]

        rest = [i for i in range(len(counts)) if i not in out]
        if rest:
            out.update(_replicated(rest))
        return torch.cat([out[i] for i in range(len(counts))], dim=0)

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        if pixel_values_videos is not None or grid_thw_videos is not None:
            raise NotImplementedError("Kimi K3 v1 supports images but not videos.")
        if self.tok_embeddings is not None:
            h_TD = self._prepare_multimodal_embeds(
                tokens,
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                special_tokens=special_tokens,
            )
        else:
            h_TD = tokens

        num_tokens, D = h_TD.shape
        block_residual_TND = h_TD.new_zeros(num_tokens, 0, D)
        for layer in self.layers.values():
            h_TD, block_residual_TND = layer(
                h_TD,
                block_residual_TND,
                attention_masks,
                positions,
            )

        h_TD = _apply_attention_residual(
            h_TD,
            block_residual_TND,
            self.output_res_proj,
            self.output_res_norm,
        )
        h_TD = self.norm(h_TD) if self.norm is not None else h_TD
        if self._skip_lm_head:
            return h_TD
        return self.lm_head(h_TD) if self.lm_head is not None else h_TD
