# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The vision pipeline stage under DEP (report sec 5.2.3).

Ported from the reference tree's ``KimiK3ViTStage``. The tower cannot be split
across stages by FQN: ``_split_module`` only descends into a ModuleDict or
ModuleList that is a DIRECT child of the model, so ``vision_encoder`` is kept or
nulled whole. A stage subclass is what the reference uses, and this is the same
class adapted to the Config-tree model.
"""

import torch

from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.tools.logging import logger


class KimiK3ViTStage(KimiK3Model):
    """Tower + embedding + splice, as one pipeline stage.

    Owns ``tok_embeddings`` as well as the tower, because the splice needs both
    the features and the token ids and pipelining passes positional args to the
    first stage only.
    """

    @classmethod
    def promote(cls, part: KimiK3Model) -> "KimiK3ViTStage":
        """Re-class a split chunk as a vision stage, in place.

        The reference tree has ``from_parts`` here instead, which assembles the
        multimodal WRAPPER from a tower and a language model. That exists because
        the wrapper hides its children from ``_split_module``. This layout has no
        wrapper -- the chunk core hands back is already a KimiK3Model carrying
        ``vision_encoder`` and ``tok_embeddings`` -- so there is nothing to
        assemble and the chunk only needs this class's forward. No counterpart in
        the reference; it is the layout difference, not a behaviour change.
        """
        if not isinstance(part, KimiK3Model):
            raise TypeError(f"expected a KimiK3Model chunk, got {type(part)!r}")
        part.__class__ = cls
        return part  # pyrefly: ignore [bad-return]

    _dep_role: str = "both"
    _dep_bounds: tuple[int, int] | None = None
    _dep_num_shares: int = 1
    _dep_step_inputs = None

    def set_dep_role(
        self,
        role: str,
        *,
        bounds: tuple[int, int] | None = None,
        num_shares: int = 1,
        step_inputs=None,
    ) -> None:
        """Declare which share of a split tower this stage carries.

        ``step_inputs`` supplies ``grid_thw`` per micro-batch to the body and
        tail stages, which never see the batch: pipelining hands positional args
        and kwargs to the first stage only. They recompute their block inputs
        from it rather than receiving them, because RoPE indices and segment
        bounds cannot survive pipelining's dummy metadata values.
        """
        if role not in ("both", "head", "body", "tail"):
            raise ValueError(f"unknown DEP vision stage role {role!r}")
        if role != "both" and bounds is None:
            raise ValueError(f"role {role!r} needs its block bounds")
        self._dep_role = role
        self._dep_bounds = bounds
        self._dep_num_shares = num_shares
        self._dep_step_inputs = step_inputs

    def _dep_grid_for_current_mb(self) -> torch.Tensor | None:
        """This micro-batch's ``grid_thw``, for a stage that never sees the batch."""
        si = self._dep_step_inputs
        mb = getattr(self, "_dep_current_mb", None)
        if si is None or mb is None:
            return None
        return si.grid_for(mb)

    def _dep_patch_capacity(self) -> int:
        from torchtitan.models.kimi_k3.vit_cp_plan import stage_patch_capacity

        cfg = self.config
        return stage_patch_capacity(
            cfg.dep_max_grid_h, cfg.dep_max_grid_w, cfg.dep_max_images
        )

    def _dep_reject_cp(self) -> None:
        group = getattr(self, "_cp_group", None)
        import torch.distributed as dist

        if group is not None and dist.get_world_size(group) > 1:
            raise NotImplementedError(
                "a tower split across PP stages does not yet support CP: the "
                "shard decision and the dynamic-CP patch plan are made inside "
                "_encode_images, and each share would have to recompute them "
                "identically. Use one vision stage with CP for now."
            )

    def _dep_placeholder_grid(self) -> torch.Tensor:
        """Grid for the smallest image a share can process: one merged token."""
        merge = self.vision_encoder.merge_kernel_size[0]
        return torch.tensor(
            [[1, merge, merge]],
            dtype=torch.int32,
            device=self.vision_encoder.patch_embed.weight.device,
        )

    def _dep_placeholder_patches(self) -> torch.Tensor:
        """Zero PATCHES matching :meth:`_dep_placeholder_grid`.

        Distinct from the single-stage tower placeholder, which returns FEATURES
        because it runs the whole tower -- correct there, wrong here: a share
        must exercise only its own parameters' collectives, and feeding features
        into the head reaches the patch conv with the wrong rank.
        """
        weight = self.vision_encoder.patch_embed.weight
        merge = self.vision_encoder.merge_kernel_size[0]
        return torch.zeros(
            merge * merge,
            weight.shape[1],
            device=weight.device,
            dtype=weight.dtype,
        )

    def _dep_forward_head(self, tokens, pixel_values, grid_thw, special_tokens):
        """First share: patch_embed + early blocks, and the text embedding.

        Emits ``(patches_padded, text_embeds, sentinel_mask)``. All three are
        float activations, so pipelining's dummy metadata values are harmless --
        nothing downstream indexes with them, which is the property that lets the
        tower span stages at all.
        """
        from torchtitan.models.kimi_k3.vit_cp_plan import pack_stage_patches

        if self.tok_embeddings is None:
            raise ValueError(
                "the DEP vision head stage must own tok_embeddings: it produces "
                "the text embedding stream, and the ids cannot be forwarded on"
            )
        self._dep_reject_cp()

        sentinel = special_tokens["image_id"]
        is_sentinel = tokens == sentinel
        # Embed with the sentinel replaced: the sentinel id may be out of range.
        safe_ids = torch.where(is_sentinel, torch.zeros_like(tokens), tokens)
        text_embeds = self.tok_embeddings(safe_ids)
        sentinel_mask = is_sentinel.to(text_embeds.dtype)

        _, hi = self._dep_bounds
        if pixel_values is None or grid_thw is None:
            # No images is a normal batch, but the tower still has to run or
            # FSDP2's all-gather is issued by some ranks only -- and through
            # __call__, where FSDP2 hooks; a direct method call stays sharded.
            x = self.vision_encoder(
                self._dep_placeholder_patches(),
                grid_thw=self._dep_placeholder_grid(),
                part="head",
                upto_block=hi,
            )
            x = x * 0.0
        else:
            x = self.vision_encoder(
                pixel_values.to(self.vision_encoder.patch_embed.weight.dtype),
                grid_thw=grid_thw,
                part="head",
                upto_block=hi,
            )
        return (
            pack_stage_patches(x, self._dep_patch_capacity()),
            text_embeds,
            sentinel_mask,
        )

    def _dep_forward_later(
        self, patches_padded, text_embeds, sentinel_mask, grid_thw=None
    ):
        """A body or tail share.

        Pipelining passes the upstream stage's output tuple POSITIONALLY, so the
        three parameters carry the patch stream, the text embeddings and the
        sentinel mask here -- not ids, pixels and grid.
        """
        from torchtitan.models.kimi_k3.vit_cp_plan import (
            pack_stage_patches,
            unpack_stage_patches,
        )

        self._dep_reject_cp()
        lo, hi = self._dep_bounds

        if getattr(self, "_dep_current_mb", None) is None:
            # Pipelining's metadata inference runs forward with no micro-batch in
            # flight. Shapes are what it measures, and every payload keeps its
            # shape through this stage, so passing them through is enough.
            return (
                (patches_padded, text_embeds, sentinel_mask)
                if self._dep_role == "body"
                else text_embeds
            )

        grid = grid_thw if grid_thw is not None else self._dep_grid_for_current_mb()
        if grid is None and float(sentinel_mask.sum()) > 0:
            raise ValueError(
                "a later DEP vision share has sentinels to fill but received no "
                "grid_thw: pipelining normally forwards the batch kwargs to every "
                "stage, and the step-inputs cache is the fallback. Neither "
                "provided it, so the patch stream cannot be unpacked"
            )
        if grid is None:
            # A micro-batch IS in flight and it has no images. Do NOT skip the
            # tower: gate on the mesh, never on the data.
            grid = self._dep_placeholder_grid()

        real_rows = int(grid.prod(dim=-1).sum())
        x = unpack_stage_patches(patches_padded, real_rows)

        if self._dep_role == "body":
            x = self.vision_encoder(x, grid_thw=grid, part="body", lo=lo, hi=hi)
            return (
                pack_stage_patches(x, self._dep_patch_capacity()),
                text_embeds,
                sentinel_mask,
            )

        feats = self.vision_encoder(x, grid_thw=grid, part="tail", from_block=lo)
        num_sentinels = int(sentinel_mask.sum().item())
        if num_sentinels == 0:
            # Keep the tower in the loss graph even with nothing to splice, or
            # this rank skips a gradient reduction its peers issue.
            from torchtitan.distributed.fsdp import add_zero_valued_dependency

            return add_zero_valued_dependency(text_embeds, feats)
        if num_sentinels != feats.size(0):
            raise ValueError(
                f"{num_sentinels} sentinel(s) but {feats.size(0)} visual token(s): "
                "a tower split across stages supports only the per-token collator "
                "convention, where the sequence length is already correct"
            )
        mask = (sentinel_mask > 0.5).unsqueeze(-1).expand_as(text_embeds)
        return text_embeds.masked_scatter(mask, feats.to(text_embeds.dtype))

    def forward(self, *args, **kwargs):
        """Dispatch on the role. Untyped ``*args`` for one specific reason.

        Pipelining forwards the batch's kwargs to EVERY stage, and a later share
        also receives its upstream's three-tensor output POSITIONALLY. With a
        named signature those collide -- "got multiple values for argument
        'pixel_values'" -- because the patch stream binds to ``tokens`` and the
        batch's own ``pixel_values`` then arrives by keyword as well.
        """
        if self._dep_role in ("body", "tail"):
            patches, text_embeds, sentinel_mask = args[:3]
            return self._dep_forward_later(
                patches, text_embeds, sentinel_mask, grid_thw=kwargs.get("grid_thw")
            )

        tokens = args[0] if args else kwargs["tokens"]
        pixel_values = args[1] if len(args) > 1 else kwargs.get("pixel_values")
        grid_thw = args[2] if len(args) > 2 else kwargs.get("grid_thw")
        if self._dep_role == "head":
            return self._dep_forward_head(
                tokens, pixel_values, grid_thw, kwargs.get("special_tokens")
            )
        return super().forward(*args, **kwargs)
