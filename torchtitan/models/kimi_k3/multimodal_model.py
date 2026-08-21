# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT-V2 + Kimi Linear, wired as the K3 release has it.

``KimiK3MultimodalModel`` owns the native vision path: the tower produces
variable-length per-image features (native resolution), the projector belongs to the
tower (``mm_projector`` is a MoonViT child in the checkpoint), and the features are
spliced into pre-reserved sentinel positions in the LLM's embedding stream.
``KimiK3ViTStage`` is the same model wearing a pipeline stage's interface, used when
DEP (report 5.2.3) gives the tower its own stage.

The LLaVA-style scaffold that used to live here -- a frozen tower plus a separate
2-layer projector, reached only by its own test -- is gone. It described the opposite
recipe to the release, which trains MoonViT-V2 jointly rather than freezing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.distributed.fsdp import add_zero_valued_dependency
from torchtitan.models.kimi_k3.attn_res_model import KimiK3AttnResModel
from torchtitan.models.kimi_k3.model import KimiK3Config, KimiK3Model, KimiK3Spec
from torchtitan.models.kimi_k3.moonvit import MoonViTConfig  # noqa: F401

from torchtitan.tools.logging import logger


def _knob(config, field: str, env: str):
    """Kept as the local name; the implementation is shared with the topology knobs."""
    from torchtitan.models.kimi_k3.knobs import resolve_knob

    return resolve_knob(config, field, env)


# ----- K3's own vision path ---------------------------------------------- #


@dataclass(kw_only=True, slots=True)
class KimiK3MultimodalConfig:
    """Config for K3's native vision path.

    Three properties follow from the release rather than from the LLaVA recipe
    the deleted scaffold implemented:

    * the projector belongs to the tower (``mm_projector`` is a MoonViT child in
      the checkpoint), so there is no separate projector here;
    * the tower is NOT frozen -- report sec 2.4 trains MoonViT-V2 from scratch
      with next-token prediction, and the whole point of that choice was joint
      stability, so freezing it reproduces the opposite recipe;
    * vision features are variable length per sample (native resolution), so
      they arrive as a list rather than a padded ``[B, num_images, N, D]``.
    """

    kimi_config: KimiK3Config
    vision_config: "MoonViTConfig"
    num_blocks: int | None = None
    # Block size for Block AttnRes, when the flavor derives its block count
    # from one. num_blocks alone cannot express K3's "full blocks plus a short
    # tail" partition (see KimiK3AttnResModel.__init__), so carry the size and
    # let the model use it verbatim. None keeps the equal-split reading.
    attn_res_block_size: int | None = None
    vision_token_id: int = -200

    # --- DEP (report 5.2.3): the ViT/text stage boundary ------------------- #
    # The exchange buffer between the ViT stage and the first text stage must be
    # a FIXED shape, because PP sizes its point-to-point buffers once rather than
    # per step. These are therefore configured maxima, not batch-derived: a
    # batch-derived shape works until a later batch carries more image tokens,
    # and then it fails inside the P2P far from the cause. A batch that exceeds
    # them raises at the sender -- a truncated vision feature is a silently wrong
    # model the receiving stage cannot detect.
    dep_max_images: int = 8
    dep_max_grid_h: int = 32
    dep_max_grid_w: int = 32

    # --- vision CP / scheduling knobs (finding 32) ------------------------- #
    # These were environment variables. Config fields are the primary source now, with
    # the old names still honoured as an override so the repro commands recorded across
    # a dozen documents keep working; see `_knob`.
    dynamic_cp: bool = True
    cp_image_shard: bool = True
    vision_side_stream: bool = False
    # Smallest image worth partitioning across a sub-CP group, in PRE-merge patches
    # (``grid_thw.prod(-1)``, i.e. t*h*w). 256 of them is a 16x16 patch grid, which at
    # patch_size 14 is a 224x224 image and 64 tokens after the 2x2 merge -- so the
    # default reads as "at least one full standard-resolution image". Below it the
    # image-level round robin balances better, since splitting buys one gather per
    # layer.
    #
    # Measured on this flavor's tower and NOT changed as a result, which is worth stating
    # rather than leaving as an unexplained default. matrix_scripts/dynamic_cp_threshold.py
    # times the tower on a whole image against one rank's share of a row-partitioned one,
    # at cp=2: 64 patches 0.99x, 144 1.00x, 256 1.01x, 400 1.01x, 1024 1.06x. Fixed cost
    # dominates at this scale -- every size sits near a 2.5 ms floor -- so partitioning
    # buys at most 6% even at 16x the threshold, before charging the per-layer gather the
    # probe does not charge for. The honest reading is that a debug-scale tower cannot
    # locate this crossing, not that 256 is validated; K3's 447M tower on
    # high-resolution input is where the number would come from. Left at 256 because a
    # value tuned on a floor of launch overhead would be worse than a stated guess.
    dynamic_cp_min_patches: int = 256

    # --- vision PP / TP topology (finding 32) ------------------------------ #
    # These decide the STAGE COUNT and the attention plan, so a launcher that
    # exported them non-uniformly gave different ranks different topologies and hung
    # in a collective with nothing naming the cause. Resolved through
    # ``knobs.register_topology``; the old env names still override, with a warning.
    vit_dep: bool = False
    vit_dep_stages: int = 1
    vit_prefetch: int = 0
    vit_tp_heads: bool = True


class _PlainGradBoundary(torch.autograd.Function):
    """Identity forward; forces the incoming gradient to be a plain tensor.

    The vision tower must stay plain in BOTH directions. Its TP and its dynamic
    CP are separate mechanisms from the decoder's, and the CP path runs
    hand-written collectives whose transpose is a reduce_scatter --
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


class KimiK3MultimodalModel(nn.Module):
    """MoonViT-V2 + Kimi Linear backbone, wired as the release has it.

    Submodule names mirror the checkpoint (``vision_tower``, ``language_model``)
    so ``hf_key_map`` is a prefix rename.
    """

    @classmethod
    def from_parts(
        cls,
        config: KimiK3MultimodalConfig,
        vision_tower: nn.Module,
        language_model: nn.Module,
    ) -> "KimiK3MultimodalModel":
        """Assemble from already-built parts, skipping ``__init__``'s construction.

        The PP split cannot see through this wrapper: core's ``_split_module``
        walks only top-level ``named_children()``, so neither the flat FQNs
        (``embed_tokens``, ``layers.N``) nor dotted ones
        (``language_model.layers.N``) match anything here, and every child is
        replaced by None -- the stage ends up with zero parameters. The adapter
        therefore splits the TEXT model and rebuilds this wrapper around the
        chunk that owns ``embed_tokens``, which is where vision features are
        consumed.

        Under DEP (``KIMI_VIT_DEP=1``, report 5.2.3) the tower gets its own stage
        instead, and what crosses the hop is the SPLICED EMBEDDING stream -- so the
        older claim that "nothing vision-side ever crosses a stage boundary" holds
        only for the non-DEP path. It is the embeddings rather than the ids that
        cross, because PP's metadata inference pushes dummy values through the pipe
        and indexing an embedding table with those asserts out of bounds.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = vision_tower
        self.language_model = language_model
        return self

    def __init__(self, config: KimiK3MultimodalConfig) -> None:
        super().__init__()
        from torchtitan.models.kimi_k3.moonvit import MoonViT

        self.config = config
        self.vision_tower = MoonViT(config.vision_config)
        if config.num_blocks is None:
            self.language_model = KimiK3Model.make_config(config.kimi_config).build()
        else:
            self.language_model = KimiK3AttnResModel(
                config.kimi_config,
                num_blocks=config.num_blocks,
                layers_per_block=config.attn_res_block_size,
            )
        if config.vision_config.text_hidden_size != config.kimi_config.hidden_size:
            raise ValueError(
                "the projector's output width must equal the LLM's hidden size: "
                f"{config.vision_config.text_hidden_size} != "
                f"{config.kimi_config.hidden_size}"
            )

    @property
    def enable_weight_tying(self) -> bool:
        lm = getattr(self, "language_model", None)
        return bool(getattr(lm, "enable_weight_tying", False))

    @property
    def tok_embeddings(self):
        """The text model's embedding, surfaced for the shared FSDP helper.

        The helper must be called on THIS wrapper rather than on language_model. Handing
        it the inner model instead makes language_model its own FSDP unit, an extra level
        between the layers and the root, and that deadlocks every CP cell on an
        _ALLGATHER_BASE -- measured, multimodal only, text unaffected.
        """
        lm = getattr(self, "language_model", None)
        return getattr(lm, "embed_tokens", None)

    @property
    def norm(self):
        lm = getattr(self, "language_model", None)
        return getattr(lm, "norm", None)

    @property
    def lm_head(self):
        lm = getattr(self, "language_model", None)
        return getattr(lm, "lm_head", None)

    def encode_images(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> list[torch.Tensor]:
        """Collator patches -> one ``[N_i, D_llm]`` feature block per sample.

        The two sides disagree on layout and the shapes do not collide loudly:
        ``MMCollator`` emits ``[num_images, max_patches, C*P*P]``, zero-PADDED
        to the largest image in the batch, while MoonViT's patch_embed is a
        ``Conv2d`` over ``[L, C, P, P]`` with the images CONCATENATED and no
        padding. Feeding the collator's tensor straight through reaches the
        conv as a 3-D input and fails there.

        ``grid_thw`` carries each image's ``(t, h, w)``, whose product is that
        image's real patch count, so the padding is dropped exactly rather than
        by scanning for zero rows -- a black patch is legitimately all zeros.
        """
        cfg = self.config.vision_config
        counts = grid_thw.prod(dim=-1).tolist()

        # Context parallel over IMAGES. Without this every CP rank encodes the
        # whole batch's images and discards the part its sequence shard does not
        # need. The tower is a per-image function, so splitting the images
        # changes no arithmetic. The group is the static _cp_group -- only the
        # work assignment is per-batch, so no dynamic mesh is needed.
        # KIMI_VIT_CP_IMAGE_SHARD=0 forces the replicated path for A/B.

        cp_size = self._cp_world_size()

        # Dynamic CP (report 5.2.3) comes FIRST, because it covers the case
        # image-level round-robin structurally cannot: fewer images than ranks, or
        # one image so much larger than the rest that whole-image assignment
        # leaves ranks idle. Round-robin then handles the many-small-images case.
        if cp_size > 1 and _knob(self.config, "dynamic_cp", "KIMI_VIT_DYNAMIC_CP"):
            planned = self._encode_images_dynamic_cp(
                pixel_values, grid_thw, counts, cp_size
            )
            if planned is not None:
                return planned

        if (
            cp_size > 1
            and len(counts) >= cp_size
            and _knob(self.config, "cp_image_shard", "KIMI_VIT_CP_IMAGE_SHARD")
        ):
            return self._encode_images_cp(pixel_values, grid_thw, counts, cp_size)

        packed = torch.cat([pixel_values[i, :n] for i, n in enumerate(counts)], dim=0)
        packed = packed.reshape(-1, cfg.in_channels, cfg.patch_size, cfg.patch_size)
        # The collator emits float32; under FSDP's mixed precision the tower's
        # weights are bf16, and Conv2d refuses the mix rather than promoting.
        weight = self.vision_tower.patch_embed.proj.weight
        packed = packed.to(weight.dtype)

        # Under TP the tower's params are replicated DTensors (parallelize.py
        # distributes them so grad-norm clipping sees one mesh). Lift the input
        # in and drop the outputs back out here: every placement is Replicate,
        # so both conversions are local metadata changes, not collectives.
        # Keyed on the mesh parallelize recorded, NOT on whether the weight is
        # a DTensor -- under FSDP it is one either way, and lifting onto the
        # FSDP mesh meets the plain all-gathered weight inside the conv.
        tp_mesh = getattr(self, "_vision_tp_mesh", None)
        if tp_mesh is not None:
            packed = DTensor.from_local(
                packed, tp_mesh, (Replicate(),), run_check=False
            )
        if _knob(self.config, "vision_side_stream", "KIMI_VIT_SIDE_STREAM"):
            features = self._run_on_vision_stream(
                lambda: self.vision_tower(packed, grid_thw),
                packed if isinstance(packed, torch.Tensor) else None,
            )
        else:
            features = self.vision_tower(packed, grid_thw)
        # --debug.detect-anomaly named this line after six attempts spent on the
        # dynamic-CP path; the failing forward was its sibling, the replicated
        # path, which every batch also goes through.

        def _seal(f):
            if isinstance(f, DTensor):
                f = f.to_local()
            return _PlainGradBoundary.apply(f)

        if isinstance(features, torch.Tensor):
            return _seal(features)
        return [_seal(f) for f in features]

    def _vision_stream(self):
        """A dedicated CUDA stream for the tower, created once per module.

        Groundwork for DEP's concurrent design (report 5.2.3), and it is only
        groundwork: running on a side stream and immediately waiting for it cannot
        overlap anything. The overlap needs the encode for micro-batch m+k issued
        during micro-batch m's text compute, which is a scheduling change. What this
        establishes is the part that has to be right first -- cross-stream tensor
        lifetime and the interaction with FSDP2's tower all-gather, neither of which
        the AttnRes PP adapter has ever had to deal with (it touches no streams at
        all).

        Same THREAD, separate stream. Not a worker thread: the adapter keys its
        per-microbatch cache in a ``threading.local``, and its forward reads a
        missing key as "this call is PP's shape inference" and diverts WITHOUT
        raising. A worker thread would therefore take the shape-inference path and
        return wrong shapes with no error.
        """
        if not torch.cuda.is_available():
            return None
        # Only when no autograd graph is being recorded. A graph recorded here has its
        # backward run here too, and with prefetch several micro-batches then accumulate
        # into the same tower parameters from two streams with nothing ordering them --
        # which cost mm_full/tp2_pp2_cp2 its reproducibility: seven runs, seven distinct
        # traces. Forcing the encode onto the current stream gives one trace over three
        # runs, bit-identical to the DEP-without-prefetch numbers, so the stream was only
        # ever changing reduction order, never the result.
        #
        # Nothing is lost today because both callers join immediately, so the stream
        # overlaps nothing while grad is on. The machinery stays for the deferred design
        # (report 5.2.3), which needs cross-stream collective ordering this does not yet
        # establish -- and will need ordered accumulation before it can carry gradients.
        if torch.is_grad_enabled():
            return None
        s = getattr(self, "_vision_side_stream", None)
        if s is None:
            s = torch.cuda.Stream()
            self._vision_side_stream = s
        return s

    def _run_on_vision_stream(self, fn, *tensors):
        """Run ``fn`` on the vision stream with the synchronisation it needs.

        Three edges, and all three are required rather than defensive:

        * the side stream waits for the current one, because ``fn``'s inputs were
          produced there;
        * every input is marked ``record_stream`` on the side stream, or the caching
          allocator may hand its memory to another allocation while the side stream
          is still reading it -- a correctness bug, not a slowdown;
        * the current stream waits for the side stream and every output is marked
          against the current stream, for the same reason in the other direction.
        """
        out, done = self._issue_on_vision_stream(fn, *tensors)
        self._join_vision_stream(out, done)
        return out

    def _issue_on_vision_stream(self, fn, *tensors):
        """Issue ``fn`` on the vision stream and return ``(out, event)`` WITHOUT waiting.

        This is the half that makes overlap possible. :meth:`_run_on_vision_stream` joins
        immediately, which is correct for a synchronous encode but means the side stream
        buys nothing -- the caller blocks on it before running anything else. The
        run-ahead needs the encode for micro-batch m+k in flight WHILE m's text compute
        runs, so it issues here and joins later, in :meth:`_join_vision_stream`.

        The input-side edges are the same as the synchronous path and equally required:
        the side stream waits for the current one because ``fn``'s inputs were produced
        there, and each input is ``record_stream``'d so the caching allocator cannot hand
        its memory to another allocation while the side stream still reads it.
        """
        side = self._vision_stream()
        if side is None:
            return fn(), None
        cur = torch.cuda.current_stream()
        side.wait_stream(cur)
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t.record_stream(side)
        # Bracket the encode ON THE SIDE STREAM so its own GPU time is measurable.
        # Without this the only observable is the span between issue and join, which is
        # dominated by text compute and PP communication and therefore reads the same
        # whether or not the encode ran concurrently -- a metric that cannot be falsified.
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(side):
            started.record(side)
            out = fn()
            finished.record(side)
        done = finished
        self._last_encode_span = (started, finished)
        return out, done

    def _join_vision_stream(self, out, done) -> None:
        """Make the current stream wait for an issued encode, and hand the outputs over.

        Both halves are needed: without the wait the consumer reads memory the side
        stream is still writing, and without ``record_stream`` on the outputs the
        allocator may reuse buffers the side stream produced while the current stream
        still holds them.
        """
        if done is None:
            return
        cur = torch.cuda.current_stream()
        cur.wait_event(done)
        outs = out if isinstance(out, (list, tuple)) else [out]
        for t in outs:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t.record_stream(cur)

    def _encode_images_dynamic_cp(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        counts: list[int],
        cp_size: int,
    ) -> list[torch.Tensor] | None:
        """Partition large images along the patch dimension (report 5.2.3).

        Returns ``None`` when the batch has no image worth partitioning, so the
        caller falls through to image-level round-robin. Returning None rather than
        silently doing nothing matters: a path that "handled" a batch by leaving it
        replicated is how the first CP attempt looked perfect while never engaging.

        Every large image is encoded by one sub-CP group, with its patches split
        across that sub-group's ranks and attention gathering keys and values
        inside the sub-group. Small images stay whole and are round-robined over
        the sub-groups' first ranks, so no rank sits idle.

        One large image per sub-group per pass, and the pass runs once per image
        slot: the gather-KV attention path assumes the local stream IS one shard of
        one image, and a mixed stream would let attention run across image
        boundaries. That is enforced here rather than hoped for.
        """

        import torch.distributed._functional_collectives as funcol

        from torchtitan.models.kimi_k3.moonvit import CPPatchPlan
        from torchtitan.models.kimi_k3.vit_cp_plan import (
            balance_images,
            classify,
            merged_tokens,
            row_partition,
            subgroup_layout,
        )

        subgroups = getattr(self, "_cp_subgroups", None)
        if not subgroups:
            return None

        cfg = self.config.vision_config
        kh, kw = cfg.merge_kernel_size
        merge = kh * kw
        min_patches = _knob(
            self.config, "dynamic_cp_min_patches", "KIMI_VIT_DYNAMIC_CP_MIN_PATCHES"
        )
        large = classify(counts, cp_size, min_patches=min_patches)
        if not large:
            return None

        n_sub, g = subgroup_layout(len(large), cp_size)
        group = subgroups.get(n_sub)
        if group is None or g <= 1:
            # No usable sub-group of size > 1 means there is nothing to partition
            # across; round-robin is the better tool for that batch.
            return None

        # Grid heights must divide the merge kernel for a partition to be legal.
        # An image that fails it is left to round-robin instead of being cut
        # unsafely.
        grids = grid_thw.tolist()
        large = [i for i in large if grids[i][1] % kh == 0]
        if not large:
            return None

        cp_rank = torch.distributed.get_rank(self._cp_group)
        my_sub = cp_rank // g
        rank_in_sub = cp_rank % g
        group_of = balance_images([counts[i] for i in large], n_sub)
        my_large = [img for img, sub in zip(large, group_of) if sub == my_sub]

        if not getattr(self, "_dynamic_cp_logged", False):
            self._dynamic_cp_logged = True
            logger.info(
                "MoonViT dynamic CP: %d large image(s) of %d over %d sub-CP "
                "group(s) of %d rank(s); min_patches=%d",
                len(large),
                len(counts),
                n_sub,
                g,
                min_patches,
            )

        weight = self.vision_tower.patch_embed.proj.weight
        tp_mesh = getattr(self, "_vision_tp_mesh", None)
        out: dict[int, torch.Tensor] = {}

        # Every sub-group must run the same NUMBER of passes or the collectives
        # inside them desynchronise. The count is the max over sub-groups, and a
        # sub-group with fewer images pads with an empty pass.
        per_sub = [sum(1 for s in group_of if s == k) for k in range(n_sub)]
        n_passes = max(per_sub) if per_sub else 0

        for p in range(n_passes):
            img = my_large[p] if p < len(my_large) else None
            if img is None:
                # An empty pass still joins this sub-group's collectives, or the
                # sub-groups desynchronise. One merge block keeps every shape valid
                # and the output is discarded.
                local = torch.zeros(
                    kh * kw, cfg.in_channels, cfg.patch_size, cfg.patch_size
                )
                local_grid = torch.tensor([[1, kh, kw]], device=grid_thw.device)
                plan_grid, row_start, band, real_rows = (1, kh * g, kw), 0, kh, kh
                valid_total = kh * kw * g
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
                flat = pixel_values[img, : counts[img]].reshape(
                    -1, cfg.in_channels, cfg.patch_size, cfg.patch_size
                )
                # This rank's rows of EVERY frame: the projector's temporal mean
                # spans all frames, so splitting by frame would give each rank the
                # mean of its own frames instead.
                pad_rows = band - (sh.row_end - sh.row_start)
                pieces = []
                for a, b in sh.ranges:
                    pieces.append(flat[a:b])
                    if pad_rows:
                        pieces.append(
                            flat.new_zeros(
                                pad_rows * w,
                                cfg.in_channels,
                                cfg.patch_size,
                                cfg.patch_size,
                            )
                        )
                local = torch.cat(pieces, dim=0)
                local_grid = torch.tensor([[t, band, w]], device=grid_thw.device)
                plan_grid = (t, h, w)
                row_start = sh.row_start
                real_rows = sh.row_end - sh.row_start
                valid_total = counts[img]

            local = local.to(weight.dtype).to(pixel_values.device)
            if tp_mesh is not None:
                local = DTensor.from_local(
                    local, tp_mesh, (Replicate(),), run_check=False
                )
            plan = CPPatchPlan(
                group=group,
                valid_total=valid_total,
                full_grid=plan_grid,
                row_start=row_start,
                band=band,
                real_rows=real_rows,
            )
            feats = self.vision_tower(local, local_grid, plan)
            if isinstance(feats, torch.Tensor):
                feats = [feats]
            # Same boundary as the replicated path: to_local unwraps the value
            # but its backward re-wraps the gradient, and the all_gather below
            # has a reduce_scatter transpose with no DTensor rule.
            feats = [
                _PlainGradBoundary.apply(f.to_local() if isinstance(f, DTensor) else f)
                for f in feats
            ]
            local_feat = torch.cat(feats, dim=0)

            # The boundary belongs on the OUTPUT: the DTensor gradient arrives
            # from downstream, so sealing the all_gather's input leaves its
            # transpose (a reduce_scatter, no DTensor rule) still receiving one.
            # Located by --debug.detect-anomaly, which moved the reported
            # forward line each time a site was fixed -- that movement is what
            # distinguishes "fixed, next one" from "not fixed".
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

        # Images below the threshold, or with an illegal grid, still need
        # encoding. Round-robin them over sub-group leaders and share the result.
        rest = [i for i in range(len(counts)) if i not in out]
        if rest:
            small = self._encode_images_replicated(pixel_values, grid_thw, rest)
            for i, f in zip(rest, small):
                out[i] = f
        return [out[i] for i in range(len(counts))]

    def _encode_images_replicated(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        which: list[int],
    ) -> list[torch.Tensor]:
        """Encode a subset of the batch's images redundantly on every rank.

        Used for the images dynamic CP leaves alone. Redundant rather than sharded
        because these are the small ones by construction, so the encode is cheap
        and a second collective would cost more than it saves.
        """
        cfg = self.config.vision_config
        counts = grid_thw.prod(dim=-1).tolist()
        packed = torch.cat([pixel_values[i, : counts[i]] for i in which], dim=0)
        packed = packed.reshape(-1, cfg.in_channels, cfg.patch_size, cfg.patch_size)
        weight = self.vision_tower.patch_embed.proj.weight
        packed = packed.to(weight.dtype)
        tp_mesh = getattr(self, "_vision_tp_mesh", None)
        if tp_mesh is not None:
            packed = DTensor.from_local(
                packed, tp_mesh, (Replicate(),), run_check=False
            )
        feats = self.vision_tower(packed, grid_thw[which])
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        return [f.to_local() if isinstance(f, DTensor) else f for f in feats]

    def _encode_images_cp(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        counts: list[int],
        cp_size: int,
    ) -> list[torch.Tensor]:
        """Encode a disjoint slice of the batch's images, then share the features.

        Sizing the collective needs no extra communication: ``grid_thw`` is
        replicated and the projector's merge kernel divides each image's patch
        count by a fixed factor, so every rank knows every image's output length
        up front and the exchange is a fixed-shape all-gather.

        ``funcol.all_gather_tensor`` because it is differentiable: its transpose
        is the reduce-scatter that hands each rank the gradient for exactly the
        images it encoded, summed over every rank's token shard.
        """
        import torch.distributed._functional_collectives as funcol

        group = self._cp_group
        rank = torch.distributed.get_rank(group)
        cfg = self.config.vision_config
        kh, kw = cfg.merge_kernel_size
        merge = kh * kw

        if any(c % merge for c in counts):
            raise ValueError(
                f"image patch counts {counts} must divide the projector merge "
                f"kernel {kh}x{kw}; CP image sharding sizes its all-gather from "
                "these counts and cannot do so otherwise"
            )
        out_lens = [c // merge for c in counts]

        owner = [i % cp_size for i in range(len(counts))]
        mine = [i for i, o in enumerate(owner) if o == rank]
        slot = max(
            sum(out_lens[i] for i, o in enumerate(owner) if o == r)
            for r in range(cp_size)
        )

        if not getattr(self, "_cp_image_shard_logged", False):
            self._cp_image_shard_logged = True
            logger.info(
                "MoonViT CP: sharding %d images over %d CP ranks", len(counts), cp_size
            )

        local_packed = torch.cat([pixel_values[i, : counts[i]] for i in mine], dim=0)
        local_packed = local_packed.reshape(
            -1, cfg.in_channels, cfg.patch_size, cfg.patch_size
        )
        weight = self.vision_tower.patch_embed.proj.weight
        local_packed = local_packed.to(weight.dtype)
        tp_mesh = getattr(self, "_vision_tp_mesh", None)
        if tp_mesh is not None:
            local_packed = DTensor.from_local(
                local_packed, tp_mesh, (Replicate(),), run_check=False
            )
        local_features = self.vision_tower(local_packed, grid_thw[mine])
        if isinstance(local_features, torch.Tensor):
            local_features = [local_features]
        local_features = [
            f.to_local() if isinstance(f, DTensor) else f for f in local_features
        ]

        flat = torch.cat(local_features, dim=0)
        pad = slot - flat.size(0)
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad, flat.size(1))], dim=0)
        gathered = funcol.all_gather_tensor(flat, gather_dim=0, group=group)

        out: list[torch.Tensor | None] = [None] * len(counts)
        for r in range(cp_size):
            base = r * slot
            for i in [j for j, o in enumerate(owner) if o == r]:
                out[i] = gathered[base : base + out_lens[i]]
                base += out_lens[i]
        return [f for f in out if f is not None]

    def _cp_world_size(self) -> int:
        group = getattr(self, "_cp_group", None)
        return 1 if group is None else torch.distributed.get_world_size(group)

    @staticmethod
    def _keep_tower_alive(output, unused_output: torch.Tensor):
        """add_zero_valued_dependency, but tolerant of a PP stage's tuple.

        A non-last PP stage returns ``(hidden_state, block_residuals)`` -- the
        AttnRes adapter ships the block payload alongside. The graph edge only
        has to land on one of them, so put it on the hidden state and rebuild
        the tuple, exactly as the adapter's own _keepalive_touch does.

        Kept here rather than in add_zero_valued_dependency so that helper
        stays byte-identical to #4025's and the rebase is a clean delete.
        """
        if isinstance(output, tuple):
            head, *tail = output
            return (add_zero_valued_dependency(head, unused_output), *tail)
        return add_zero_valued_dependency(output, unused_output)

    def _tower_needs_collectives(self) -> bool:
        """Is the tower wrapped in something that issues per-forward collectives?

        True once FSDP has sharded it, which is when skipping it desynchronizes
        the process group.
        """
        return any(
            isinstance(p, DTensor) and any(pl.is_shard() for pl in p.placements)
            for p in self.vision_tower.parameters()
        )

    def _tower_placeholder(self) -> torch.Tensor:
        """Smallest input that still exercises every tower collective."""
        cfg = self.config.vision_config
        weight = self.vision_tower.patch_embed.proj.weight
        dev = weight.device
        dtype = weight.dtype if not isinstance(weight, DTensor) else weight.dtype
        merge = cfg.merge_kernel_size[0]
        side = merge  # one post-merge token
        patches = torch.zeros(
            side * side,
            cfg.in_channels,
            cfg.patch_size,
            cfg.patch_size,
            device=dev,
            dtype=dtype,
        )
        grid = torch.tensor([[1, side, side]], device=dev, dtype=torch.long)
        tp_mesh = getattr(self, "_vision_tp_mesh", None)
        if tp_mesh is not None:
            patches = DTensor.from_local(
                patches, tp_mesh, (Replicate(),), run_check=False
            )
        feats = self.vision_tower(patches, grid)
        if isinstance(feats, torch.Tensor):
            return feats.to_local() if isinstance(feats, DTensor) else feats
        f0 = feats[0]
        return f0.to_local() if isinstance(f0, DTensor) else f0

    def _exchange_sentinel_counts(self, local: int) -> torch.Tensor:
        """Per-rank vision-sentinel counts across the CP group.

        Called unconditionally whenever CP is on, including on ranks with no
        images: the collective's participants are decided by the mesh, never by
        the batch.
        """
        group = self._cp_group
        counts = torch.zeros(
            torch.distributed.get_world_size(group),
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        counts[torch.distributed.get_rank(group)] = local
        torch.distributed.all_reduce(counts, group=group)
        return counts

    def _select_cp_shard(
        self,
        features: list[torch.Tensor] | torch.Tensor,
        num_rows: int,
        counts: torch.Tensor | None,
    ) -> list[torch.Tensor] | torch.Tensor:
        """Keep only the visual features belonging to this CP rank's shard.

        ``prepare_context_parallel_input`` shards inputs, labels and positions
        along the sequence but leaves ``pixel_values`` whole, so every CP rank
        encodes every image while holding only a slice of the sentinels. The
        features are ordered by sequence position and the shards are contiguous
        and equal -- the flavor pins ``context_parallel_load_balancer`` to None
        precisely because a permuting balancer would break that -- so this
        rank's slice starts after however many sentinels the lower ranks hold.

        This is correctness, not the report's sec 5.2.3 optimization: the
        encoder still runs redundantly on every CP rank. Dynamic CP would shard
        the encoder itself along the patch dimension and gather KV instead.
        """
        if counts is None:
            return features

        if int(counts.sum().item()) != num_rows:
            raise ValueError(
                f"CP ranks hold {int(counts.sum().item())} vision sentinel(s) "
                f"in total but {num_rows} visual token(s) were encoded; the "
                "sequence shard and the image batch disagree"
            )
        rank = torch.distributed.get_rank(self._cp_group)
        start = int(counts[:rank].sum().item())
        local = int(counts[rank].item())
        flat = (
            features
            if isinstance(features, torch.Tensor)
            else torch.cat(list(features), dim=0)
        )
        return flat[start : start + local]

    def _splice_per_token(
        self,
        input_ids: torch.Tensor,
        features: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        """Scatter visual features into pre-reserved sentinel positions.

        ``MMCollator`` reserves ONE sentinel per post-merge visual token, so the
        sequence length is already correct and the features drop straight in;
        this is the convention the release uses. :meth:`_splice` implements the
        other one -- a single sentinel per image, expanded in place -- which
        changes the sequence length and cannot be used with this collator.
        Which one applies is decided by counting, in :meth:`forward`.
        """
        sentinel = self.config.vision_token_id
        embed = self.language_model.embed_tokens
        safe_ids = torch.where(
            input_ids == sentinel, torch.zeros_like(input_ids), input_ids
        )
        text = embed(safe_ids)

        flat = (
            features
            if isinstance(features, torch.Tensor)
            else torch.cat(list(features), dim=0)
        )
        mask = (input_ids == sentinel).unsqueeze(-1).expand_as(text)
        if isinstance(text, DTensor):
            # The text stream is a DTensor now; the vision tower still hands out
            # plain tensors because its TP is a separate mechanism
            # (_apply_tp_moonvit_mlp), so LIFT the vision side to the text
            # stream's layout rather than unwrapping the stream. Both are
            # Replicate on the tp axis here, so this is metadata only.
            flat = DTensor.from_local(
                flat.to(text.to_local().dtype),
                text.device_mesh,
                text.placements,
                run_check=False,
            )
            mask = DTensor.from_local(
                mask, text.device_mesh, text.placements, run_check=False
            )
            # aten.masked_scatter has no DTensor rule. Build the scattered
            # result positionally instead: the sentinel positions are exactly
            # the vision slots, in order, so a scatter along the flattened token
            # axis is the same operation and does have a DTensor rule.
            local_text = text.to_local()
            idx = mask[..., 0].to_local().reshape(-1).nonzero(as_tuple=True)[0]
            out = local_text.reshape(-1, local_text.shape[-1]).clone()
            out[idx] = flat.to_local().to(out.dtype)
            return DTensor.from_local(
                out.view_as(local_text),
                text.device_mesh,
                text.placements,
                run_check=False,
            )
        return text.masked_scatter(mask, flat.to(text.dtype))

    def _splice(
        self,
        input_ids: torch.Tensor,
        features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Replace each vision sentinel with that sample's feature block.

        One sentinel per image, expanded in place to its own token count, so the
        sequence grows by a different amount per sample. Rows are right-padded
        with the embedding of token 0 to a common length; the caller masks the
        padding in the loss, exactly as it must for the sentinel positions.
        """
        B, T = input_ids.shape
        sentinel = self.config.vision_token_id
        embed = self.language_model.embed_tokens
        safe_ids = torch.where(
            input_ids == sentinel, torch.zeros_like(input_ids), input_ids
        )
        text = embed(safe_ids)

        rows, feat_iter = [], iter(features)
        for b in range(B):
            positions = (input_ids[b] == sentinel).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                rows.append(text[b])
                continue
            pieces, cursor = [], 0
            for pos in positions.tolist():
                pieces.append(text[b, cursor:pos])
                pieces.append(next(feat_iter).to(text.dtype))
                cursor = pos + 1
            pieces.append(text[b, cursor:])
            rows.append(torch.cat(pieces, dim=0))

        width = max(r.size(0) for r in rows)
        pad = embed(torch.zeros(1, dtype=input_ids.dtype, device=input_ids.device))
        return torch.stack(
            [
                r
                if r.size(0) == width
                else torch.cat([r, pad.expand(width - r.size(0), -1)], dim=0)
                for r in rows
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """``[B, T]`` ids (+ packed patches) -> logits.

        ``**kwargs`` is ignored, mirroring KimiK3Model: torchtitan's Trainer
        and Validator inject ``attention_masks=None`` and ``positions=...`` for
        the FlexAttention / CP paths, and K3 uses plain SDPA plus KDA Triton
        kernels which take neither.

        Text-only when no images are supplied or no sentinel is present.

        The image parameters are named for the COLLATOR's output keys, not this
        model's internal vocabulary. torchtitan's trainer forwards a batch as
        ``model(inputs, **extra_kwargs)``, so a parameter spelled any other way
        is absorbed by ``**kwargs`` and the tower silently never runs -- which
        is exactly how a whole multimodal parallelism matrix once passed while
        validating nothing vision-side.
        """
        num_sentinels = int((input_ids == self.config.vision_token_id).sum().item())
        cp_active = self._cp_world_size() > 1

        # Under CP the per-rank sentinel counts have to be exchanged, and the
        # decision to exchange them must not depend on data. A batch carrying
        # no images at all is a normal occurrence, and letting that rank return
        # early leaves its CP peers waiting in the collective forever -- a
        # 100-second NCCL watchdog timeout, not an error. Gate on the mesh,
        # which every rank agrees on before looking at anything.
        cp_counts = self._exchange_sentinel_counts(num_sentinels) if cp_active else None

        if pixel_values is None:
            out = self.language_model(input_ids)
            if self._tower_needs_collectives():
                # FSDP2 issues the tower's all-gather from its pre-forward hook.
                # A rank that skips the tower on an image-free batch does not
                # issue it, and its peers wait in that collective until the
                # NCCL watchdog fires. Run the tower on a placeholder and keep
                # the graph edge with a zero-valued dependency, so every rank
                # issues the same collectives and the tower's contribution to
                # the data-parallel average is a correct zero.
                out = self._keep_tower_alive(out, self._tower_placeholder())
            return out
        if num_sentinels == 0 and not cp_active:
            raise ValueError(
                "pixel_values supplied but input_ids contains no "
                f"vision_token_id ({self.config.vision_token_id}); the "
                "images would be silently dropped"
            )
        if grid_thw is None:
            raise ValueError("grid_thw is required alongside pixel_values")

        # Under CP a rank's sequence shard legitimately holds no sentinel at all
        # -- every image's tokens landed in another rank's half. It still has to
        # reach _select_cp_shard, whose all_reduce every CP rank participates in;
        # returning early here would hang the others. So encode and select
        # first, and only then decide whether there is anything to splice.
        features = self.encode_images(pixel_values, grid_thw)

        def _rows(f):
            if isinstance(f, torch.Tensor):
                return f.shape[0]
            return sum(int(x.shape[0]) for x in f)

        features = self._select_cp_shard(features, _rows(features), cp_counts)
        num_rows = _rows(features)
        # Counted AFTER the shard selection, because the counts below are
        # compared against THIS rank's sentinels. Under CP the shard is a flat
        # tensor -- the per-image grouping is gone -- so an image count is not a
        # meaningful thing to match against, and it degenerates to the row
        # count, which the per-token branch handles. That branch is also the one
        # that necessarily fires under CP: the shard is sized by this rank's
        # sentinel count, so num_rows == num_sentinels by construction, and
        # _select_cp_shard already rejects the one convention where they could
        # differ.
        num_images = num_rows if isinstance(features, torch.Tensor) else len(features)
        if num_sentinels == 0:
            # This rank's sequence shard holds no sentinel: every image's
            # tokens landed on a CP peer. The tower ran, so its forward
            # all-gathers matched, but returning here would drop its output
            # out of the loss graph -- and FSDP2 takes its reduce-scatter from
            # the autograd hooks on that output, so this rank would skip a
            # gradient reduction its peers issue. Same hazard as the
            # image-free path above, reached by a different route.
            out = self.language_model(input_ids)
            return self._keep_tower_alive(out, features)
        if num_sentinels == num_rows:
            # Collator convention: one sentinel per post-merge visual token.
            embeds = self._splice_per_token(input_ids, features)
        elif num_sentinels == num_images:
            # LLaVA convention: one sentinel per image, expanded in place.
            embeds = self._splice(input_ids, features)
        else:
            raise ValueError(
                f"{num_sentinels} vision sentinel(s) in input_ids match neither "
                f"the image count ({num_images}, one sentinel per image) nor the "
                f"visual-token count ({num_rows}, one sentinel per token)"
            )
        # The backbone's forward embeds int ids; we already embedded, so detach
        # embed_tokens to take its pre-embedded branch. Same mechanism as
        saved = self.language_model.embed_tokens
        try:
            self.language_model.embed_tokens = None
            return self.language_model(embeds)
        finally:
            self.language_model.embed_tokens = saved

    def init_weights(self, init_range: float | None = None, **kwargs) -> None:
        # Under PP the module is split into stages and the pieces a stage does
        # not own are set to None -- only the first stage keeps the tower, only
        # the last keeps lm_head. Guard both rather than assume a whole model.
        if self.vision_tower is not None:
            self.vision_tower.init_weights(init_range)
        if self.language_model is not None:
            self.language_model.init_weights(init_range, **kwargs)

    def get_attention_masks(self, *args, **kwargs):
        """No mask passthrough, same as the text model -- see KimiK3Model.

        Spelled out here rather than copied onto the class at import time. The
        loop that used to do that also listed init_weights, which this class
        defines itself, so that half of it could never fire; a reader of the
        class saw neither method.
        """
        return None


class KimiK3ViTStage(KimiK3MultimodalModel):
    """The vision PP stage under DEP (report 5.2.3): tower + embed + splice.

    Owns ``embed_tokens`` and the splice as well as the tower, because the splice needs
    both the features and ``input_ids`` and torchtitan passes positional args only to the
    first stage.

    See ``phase13_k3like_48b_posttrain/VIT_STAGE_OWNERSHIP.md``.
    """

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

        ``step_inputs`` supplies ``grid_thw`` per micro-batch to the body and tail
        stages, which never see the batch: PP hands positional args and kwargs to the
        first stage only. They recompute their block inputs from it rather than
        receiving them, because RoPE indices and segment bounds cannot survive PP's
        dummy metadata values.
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

    def _dep_packed_patches(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Collator patches -> the tower's ``[L, C, P, P]`` layout, padding dropped.

        Same reshape as :meth:`_encode_images_replicated`: the collator emits
        ``[num_images, max_patches, C*P*P]`` zero-padded to the batch's largest image,
        while patch_embed is a conv over concatenated images with no padding. The real
        count comes from ``grid_thw``, not from scanning for zero rows -- a black patch
        is legitimately all zeros.
        """
        cfg = self.config.vision_config
        counts = grid_thw.prod(dim=-1).tolist()
        packed = torch.cat(
            [pixel_values[i, : counts[i]] for i in range(len(counts))], dim=0
        )
        packed = packed.reshape(-1, cfg.in_channels, cfg.patch_size, cfg.patch_size)
        return packed.to(self.vision_tower.patch_embed.proj.weight.dtype)

    def _dep_reject_cp(self) -> None:
        if self._cp_world_size() > 1:
            raise NotImplementedError(
                "a tower split across PP stages does not yet support CP: the shard "
                "decision and the dynamic-CP patch plan are made inside "
                "encode_images, and each share would have to recompute them "
                "identically. Use KIMI_VIT_DEP_STAGES=1 with CP for now."
            )

    def _dep_forward_head(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
    ):
        """First share: patch_embed + early blocks, and the text embedding.

        Emits ``(patches_padded, text_embeds, sentinel_mask)``. All three are float
        activations, so PP's dummy metadata values are harmless -- nothing downstream
        indexes with them, which is the property that lets the tower span stages at
        all.
        """
        from torchtitan.models.kimi_k3.vit_cp_plan import pack_stage_patches

        embed = self.language_model.embed_tokens
        if embed is None:
            raise ValueError(
                "the DEP vision head stage must own embed_tokens: it produces the "
                "text embedding stream, and the ids cannot be forwarded onward"
            )
        self._dep_reject_cp()

        sentinel = self.config.vision_token_id
        is_sentinel = input_ids == sentinel
        # Embed with the sentinel replaced, exactly as _splice_per_token does: the
        # sentinel id is negative, so embedding it directly indexes out of bounds.
        safe_ids = torch.where(is_sentinel, torch.zeros_like(input_ids), input_ids)
        text_embeds = embed(safe_ids)
        sentinel_mask = is_sentinel.to(text_embeds.dtype)

        _, hi = self._dep_bounds
        if pixel_values is None or grid_thw is None:
            # No images is a normal batch. The tower still has to run, or FSDP2's
            # all-gather for these parameters is issued by some ranks and not
            # others -- the hazard _keep_tower_alive exists for.
            # Through the tower's __call__, not forward_head directly: FSDP2
            # registers its all-gather there, and a direct method call leaves
            # patch_embed.proj.weight a sharded DTensor.
            x = self.vision_tower(
                self._dep_placeholder_patches(),
                self._dep_placeholder_grid(),
                part="head",
                upto_block=hi,
            )
            x = x * 0.0
        else:
            x = self.vision_tower(
                self._dep_packed_patches(pixel_values, grid_thw),
                grid_thw,
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

        PP passes the upstream stage's output tuple POSITIONALLY, so ``forward``'s
        three parameters carry the patch stream, the text embeddings and the sentinel
        mask here -- not ids, pixels and grid. Renamed at this boundary rather than
        threaded onward under misleading names.
        """
        from torchtitan.models.kimi_k3.vit_cp_plan import (
            pack_stage_patches,
            unpack_stage_patches,
        )

        self._dep_reject_cp()
        lo, hi = self._dep_bounds

        if getattr(self, "_dep_current_mb", None) is None:
            # PP's metadata inference runs forward with no micro-batch in flight.
            # Shapes are what it measures, and every payload keeps its shape through
            # this stage (the tail's output matches text_embeds because the per-token
            # splice preserves length), so passing them through is safe and enough.
            return (
                (patches_padded, text_embeds, sentinel_mask)
                if self._dep_role == "body"
                else text_embeds
            )

        # PP forwards the batch kwargs to EVERY stage, not just the first, so a later
        # share usually has grid_thw handed to it directly -- no pipe payload and no
        # cache needed. The step-inputs cache stays as a fallback for a schedule that
        # does not forward them.
        grid = grid_thw if grid_thw is not None else self._dep_grid_for_current_mb()
        if grid is None and float(sentinel_mask.sum()) > 0:
            # The placeholder path below exists for a batch with NO images. Reaching it
            # while sentinels are present means grid_thw did not arrive at all -- a wiring
            # or launcher problem -- and continuing would slice the REAL patch payload to
            # placeholder length and splice the result. Silent wrong output; raise instead.
            raise ValueError(
                "a later DEP vision share has sentinels to fill but received no "
                "grid_thw: PP normally forwards the batch kwargs to every stage, and "
                "the step-inputs cache is the fallback. Neither provided it, so the "
                "patch stream cannot be unpacked to its real length"
            )
        if grid is None:
            # A micro-batch IS in flight and it has no images. Do NOT skip the tower:
            # gate on the mesh, never on the data. Skipping means this rank does not
            # issue FSDP2's all-gather for these blocks while its peers do, and they
            # wait until the NCCL watchdog fires. The head sent a placeholder-sized
            # payload for exactly this case, so the shapes line up.
            grid = self._dep_placeholder_grid()

        real_rows = int(grid.prod(dim=-1).sum())
        x = unpack_stage_patches(patches_padded, real_rows)

        if self._dep_role == "body":
            x = self.vision_tower(x, grid, part="body", lo=lo, hi=hi)
            return (
                pack_stage_patches(x, self._dep_patch_capacity()),
                text_embeds,
                sentinel_mask,
            )

        feats = self.vision_tower(x, grid, part="tail", from_block=lo)
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        # Cut the graph so the tower's backward can be replayed in a bubble, the same seam
        # _forward_single_stage uses. Cut on the TAIL share: that is where the encode
        # finishes, and cutting on head or body would replay only a prefix.
        gq = getattr(self, "_vision_grad_queue", None)
        mb = getattr(self, "_dep_current_mb", None)
        if gq is not None and mb is not None:
            from torchtitan.models.kimi_k3.dep_bubble_backward import (
                cut_for_deferred_backward,
            )

            feats = [cut_for_deferred_backward(f, gq, mb) for f in feats]
        flat = torch.cat(list(feats), dim=0)
        num_sentinels = int(sentinel_mask.sum().item())
        if num_sentinels == 0:
            # Keep the tower in the loss graph even with nothing to splice, or this
            # rank skips a gradient reduction its peers issue.
            return self._keep_tower_alive(text_embeds, flat)
        if num_sentinels != flat.size(0):
            raise ValueError(
                f"{num_sentinels} sentinel(s) but {flat.size(0)} visual token(s): a "
                "tower split across stages supports only the per-token collator "
                "convention, where the sequence length is already correct. The "
                "one-sentinel-per-image convention changes the sequence length per "
                "sample, which PP cannot size a buffer for"
            )
        mask = (sentinel_mask > 0.5).unsqueeze(-1).expand_as(text_embeds)
        return text_embeds.masked_scatter(mask, flat.to(text_embeds.dtype))

    def _dep_placeholder_grid(self) -> torch.Tensor:
        """Grid for the smallest image a share can process: one merged token."""
        merge = self.config.vision_config.merge_kernel_size[0]
        return torch.tensor(
            [[1, merge, merge]],
            dtype=torch.int32,
            device=self.vision_tower.patch_embed.proj.weight.device,
        )

    def _dep_placeholder_patches(self) -> torch.Tensor:
        """Zero PATCHES matching :meth:`_dep_placeholder_grid`.

        Distinct from :meth:`_tower_placeholder`, which returns FEATURES because it
        runs the whole tower -- correct for the single-stage keep-alive, wrong here:
        a share must exercise only its own parameters' collectives, and feeding
        features into ``forward_head`` reaches the patch conv with a 2-D input.
        """
        cfg = self.config.vision_config
        weight = self.vision_tower.patch_embed.proj.weight
        merge = cfg.merge_kernel_size[0]
        return torch.zeros(
            merge * merge,
            cfg.in_channels,
            cfg.patch_size,
            cfg.patch_size,
            device=weight.device,
            dtype=weight.dtype,
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Dispatch on the role. Untyped ``*args`` for one specific reason.

        PP forwards the batch's kwargs to EVERY stage, and a later share also receives
        its upstream's three-tensor output POSITIONALLY. With a named signature those
        collide -- "got multiple values for argument 'pixel_values'" -- because the
        patch stream binds to ``input_ids`` and the batch's own ``pixel_values`` then
        arrives by keyword as well. Taking both positionally and pulling the batch
        metadata out by name keeps the two channels apart.
        """
        if self._dep_role in ("body", "tail"):
            patches, text_embeds, sentinel_mask = args[:3]
            return self._dep_forward_later(
                patches, text_embeds, sentinel_mask, grid_thw=kwargs.get("grid_thw")
            )

        input_ids = args[0] if args else kwargs["input_ids"]
        pixel_values = args[1] if len(args) > 1 else kwargs.get("pixel_values")
        grid_thw = args[2] if len(args) > 2 else kwargs.get("grid_thw")
        if self._dep_role == "head":
            return self._dep_forward_head(input_ids, pixel_values, grid_thw)
        return self._forward_single_stage(input_ids, pixel_values, grid_thw)

    def _forward_single_stage(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The unsplit vision stage: tower + embed + splice, in one place.

        Left exactly as it was when its numerics were pinned; the role dispatch above
        is the only thing in front of it.
        """
        embed = self.language_model.embed_tokens
        if embed is None:
            raise ValueError(
                "the DEP vision stage must own embed_tokens: it produces the "
                "spliced embedding stream, and the ids cannot be forwarded to a "
                "later stage"
            )

        cp_active = self._cp_world_size() > 1
        num_sentinels = int((input_ids == self.config.vision_token_id).sum().item())
        # Gate the exchange on the MESH, never on the data: a batch with no images
        # is normal, and a rank that returns early leaves its CP peers waiting in
        # the collective until the watchdog fires.
        #
        # KEEP the counts. Discarding them and never calling _select_cp_shard was a
        # defect: prepare_context_parallel_input shards the sequence but leaves
        # pixel_values whole, so every CP rank encodes every image while holding only a
        # slice of the sentinels. Without the selection this rank splices ALL the
        # features into its own shard's sentinels.
        #
        # Whether that is visible depends entirely on how the sentinels fall across the
        # CP shards. The debug flavor at seq 256 puts all of them on one rank
        # (counts=[64, 0]), and there the omission is a no-op -- the rank holding none
        # has nowhere to splice -- which is why removing the call again moves no number.
        # At seq 96 the split is counts=[47, 1] and the pre-fix path fails outright.
        # See DEP_60_VERIFIED_2026-08-10.md in the logbook for both arms.
        cp_counts = None
        if cp_active:
            cp_counts = self._exchange_sentinel_counts(num_sentinels)

        if pixel_values is None or grid_thw is None:
            out = embed(input_ids)
            if self._tower_needs_collectives():
                # FSDP2 issues the tower's all-gather from its pre-forward hook, so
                # a rank that skips the tower does not issue it and its peers wait.
                out = self._keep_tower_alive(out, self._tower_placeholder())
            return out

        # DEP run-ahead: take this micro-batch's features if a previous action
        # already encoded them on the vision stream, and start the next ones. The
        # depth is a mesh property (micro-batch count), never a data one, so every
        # rank issues the same encodes in the same order -- otherwise two
        # communicators can deadlock on a cyclic wait with neither one's ordering
        # violated.
        pf = getattr(self, "_vision_prefetcher", None)
        mb = getattr(self, "_dep_current_mb", None)
        feats = None
        if pf is not None and mb is not None:
            feats = pf.take(mb)
        if feats is None:
            feats = self.encode_images(pixel_values, grid_thw)
        if pf is not None and mb is not None:
            from torchtitan.models.kimi_k3.vit_prefetch import prefetch_depth

            pf.advance(mb, prefetch_depth())
        # Cut the graph here when the bubble runtime is driving: what gets spliced is a
        # detached stand-in, and the tower's backward is replayed at a planned slot
        # instead of running inside this stage's backward. Placed before the CP-shard
        # selection so the cut sees the tower's own output, not a sliced view of it --
        # replaying a gradient into a slice would train the tower on part of its batch.
        gq = getattr(self, "_vision_grad_queue", None)
        if gq is not None and mb is not None:
            from torchtitan.models.kimi_k3.dep_bubble_backward import (
                cut_for_deferred_backward,
            )

            if isinstance(feats, torch.Tensor):
                feats = cut_for_deferred_backward(feats, gq, mb)
            else:
                feats = [cut_for_deferred_backward(f, gq, mb) for f in feats]
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        num_rows = sum(f.size(0) for f in feats)
        # Same selection the non-DEP path performs, and for the same reason. Every CP
        # rank reaches it because the counts exchange above is mesh-gated, so its
        # internal all_reduce cannot be left half-issued.
        feats = self._select_cp_shard(feats, num_rows, cp_counts)
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        num_rows = sum(f.size(0) for f in feats)
        num_images = len(feats)

        if num_sentinels == num_rows:
            return self._splice_per_token(input_ids, feats)
        if num_sentinels == num_images:
            return self._splice(input_ids, feats)
        if num_sentinels == 0:
            # Under CP a rank's sequence shard can legitimately hold no sentinel.
            # The features still have to stay in the graph, or FSDP2 skips a
            # gradient reduction this rank's peers issue.
            return self._keep_tower_alive(embed(input_ids), torch.cat(feats, dim=0))
        raise ValueError(
            f"{num_sentinels} vision sentinel(s) in input_ids match neither the "
            f"image count ({num_images}, one sentinel per image) nor the "
            f"visual-token count ({num_rows}, one sentinel per token)"
        )


def _mm_layers(self):
    """Expose the text stack where parallelize.py and the PP splitter look.

    Both walk ``model.layers`` (a ModuleDict keyed by layer id -- the PP
    adapter's layer_to_stage discovery depends on those string keys). The
    multimodal wrapper keeps the text model at ``self.language_model``, so
    without this the FSDP wrap fails with "no attribute 'layers'" before any
    step runs.
    """
    return self.language_model.layers


KimiK3MultimodalModel.layers = property(_mm_layers)


def _mm_verify_module_protocol(self) -> None:
    """No-op, delegating to the text model's reasoning.

    KimiK3Model overrides this as a no-op because its internals are plain
    nn.Modules rather than Config-built ``Module`` instances -- it ports the HF
    reference layer by layer. The multimodal wrapper adds a MoonViT tower built
    the same way, so the same holds. The trainer calls this post-build; without
    it the multimodal flavor cannot be constructed at all.
    """
    return None


KimiK3MultimodalModel.verify_module_protocol = _mm_verify_module_protocol


@dataclass(kw_only=True, slots=True)
class KimiK3MultimodalSpec(KimiK3Spec):
    """``BaseModel.Config``-compatible spec for the multimodal model.

    KimiK3Spec exists because torchtitan's trainer calls
    ``update_from_config`` and the property accessors on whatever sits at
    ``model_spec.model``; a bare dataclass config fails there. This subclasses it
    so the multimodal flavor gets the same integration surface, and overrides
    only ``build`` to construct the vision-bearing model.
    """

    vision_config: "MoonViTConfig" = None  # type: ignore[assignment]

    vision_token_id: int = -200
    """Sentinel id the splice scans for; must equal the tokenizer's image id.

    Defaulted to the LLaVA convention for the standalone/test path. A flavor
    driving a real collator has to override it: at a value the tokenizer never
    emits, the sentinel scan finds nothing and forward takes its text-only
    branch without complaint.
    """

    def build(self, **kwargs):
        return self.apply_build_time_features(
            KimiK3MultimodalModel(
                KimiK3MultimodalConfig(
                    kimi_config=self.kimi_config,
                    vision_config=self.vision_config,
                    num_blocks=self.num_blocks,
                    attn_res_block_size=self.attn_res_block_size,
                    vision_token_id=self.vision_token_id,
                )
            )
        )
