# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plans training steps from rollout group summaries (`Batcher`), and packs the planned
training samples into microbatches (`MicrobatchPacker`, on each trainer DP replica's leader).

    Batcher:            summaries -> decide -> plan -> `StepPlan`
    replica leader:     plan column + rollout arrays -> materialize -> `MicrobatchPacker.pack`
"""

import heapq
import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from torchtitan.components.data.multimodal.mrope import get_mrope_positions

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.data.padding import append_padded_segment
from torchtitan.components.data.types import TokenizedTrainingMicrobatch

from torchtitan.config import Configurable
from torchtitan.models.deepseek_v3.mtp import get_mtp_token_counts
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import PromptImage, RolloutGroupSummary
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleBuilder,
    TrainingSampleSpec,
    TrainingSampleSpecGroup,
)
from torchtitan.tools.utils import round_up

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class TrainingMicrobatch(TokenizedTrainingMicrobatch):
    """Packed training microbatch for the RL trainer.

    Each training_sample's raw tokens (length N) are split into
    ``token_ids = raw[:-1]`` and ``labels = raw[1:]`` (both length
    N-1), matching the pre-training dataloader convention.

    With images, ``model_kwargs`` holds each distinct image of the microbatch once
    (`MicrobatchPacker._pack_image_inputs`, whose example draws the dedup):

        from the packer          image_files, image_file_sizes   the files, concatenated
                                 grid_thw                        one row per distinct image
                                 image_vision_bank_indices_T     each token's row in the image
                                                                 embeddings, -1 for text
                                 mrope_positions                 with build_mrope_positions
        after preprocess_images  pixel_values replaces image_files and image_file_sizes
    """

    generator_logprobs: torch.Tensor  # [T]
    temperature: torch.Tensor  # [T]
    loss_mask: torch.Tensor  # [T]
    advantages: torch.Tensor  # [T]
    vision_processor: VisionProcessor.Config | None = None
    """How `preprocess_images` turns ``model_kwargs["image_files"]`` into
    ``pixel_values``. None without images."""

    def preprocess_images(self) -> "TrainingMicrobatch":
        """This microbatch with its image files turned into pixel values, as the generator's
        `_preprocess_images` does for a request; unchanged without images.

        The packer ships each distinct image file once, concatenated into one uint8 tensor
        (``image_files``, split by ``image_file_sizes``): a Geometry3K file is ~15 KB, its
        float32 pixel values ~2.4 MB. Each trainer rank calls this on each microbatch it
        receives, on its reader thread (`MicrobatchGroupReader`), and each file becomes pixel
        values on the CPU, as in the SFT dataloader (`VisionProcessor.resize_and_normalize`,
        then `patchify_all`).
        """
        if "image_files" not in self.model_kwargs:
            return self
        assert self.vision_processor is not None
        processor = self.vision_processor.build()
        model_kwargs = dict(self.model_kwargs)
        image_files = model_kwargs.pop("image_files")
        file_sizes = model_kwargs.pop("image_file_sizes")
        pixel_values, grid_thw = processor.patchify_all(
            [
                processor.resize_and_normalize(file.numpy().tobytes())[0]
                for file in image_files.split(file_sizes.tolist())
            ]
        )
        # Placeholders and M-RoPE positions follow the prompts' grids (from the file
        # headers), the pixel values the decoded images.
        prompt_grid_thw = model_kwargs["grid_thw"]
        if not torch.equal(grid_thw, prompt_grid_thw):
            index = int((grid_thw != prompt_grid_thw).any(-1).nonzero()[0])
            raise ValueError(
                f"Image file {index} resizes to grid {grid_thw[index].tolist()}, but its "
                f"prompt has {prompt_grid_thw[index].tolist()}; the renderer and the "
                "trainer must use the same vision processor."
            )
        model_kwargs["pixel_values"] = pixel_values
        return replace(self, model_kwargs=model_kwargs)

    def as_input_dict(self) -> dict[str, Any]:
        """Return the model inputs plus the per-token RL loss arguments.

        ``preprocess_inputs`` shards ``loss_kwargs`` like the labels and
        returns them separately for the loss.
        """
        # Explicit base call: zero-argument super() fails in slots=True dataclasses.
        return {
            **TokenizedTrainingMicrobatch.as_input_dict(self),
            "loss_kwargs": {
                "generator_logprobs": self.generator_logprobs,
                "temperature": self.temperature,
                "loss_mask": self.loss_mask,
                "advantages": self.advantages,
            },
        }


@dataclass(frozen=True, slots=True)
class StepPlan:
    """One optimizer step, decided from specs alone: which training samples each microbatch
    packs, the step's global token counts, and the step's groups and metrics.

    Example:
        # 2 DP ranks, one microbatch per group, 3 samples from groups 7 and 9
        StepPlan(
            microbatch_groups=[[[[spec_a, spec_c], [spec_b]]]],
            global_loss_token_counts=tensor([41]),
            global_routing_token_counts=tensor([60]),
            metrics=[...],
            group_ids=[7, 8, 9],          # group 8 had no training sample
            trainable_group_ids=[7, 9],
            min_policy_versions=[3, 3, 4],
        )
    """

    microbatch_groups: list[list[list[list[TrainingSampleSpec]]]]
    """[num_groups][num_microbatches_per_group][dp_degree] -> the microbatch's specs, in packing
    order. Under PP, each group runs as one pipeline step; DP rank d trains column d."""
    global_loss_token_counts: torch.Tensor
    """int64 [1 + num_mtp_layers]: the specs' loss-token counts summed. Packing checks each
    microbatch's counts against its specs, so this equals the packed counts summed."""
    global_routing_token_counts: torch.Tensor
    """int64 [1 + num_mtp_layers]: the specs' routing-token counts summed."""
    metrics: list[m.Metric]
    """The taken groups' metrics plus the step's packing metrics."""
    group_ids: list[int]
    """Every taken rollout group, including metric-only groups."""
    trainable_group_ids: list[int]
    """The groups in `group_ids` with training samples."""
    min_policy_versions: list[int]
    """One per planned sample: the oldest policy version it was sampled under."""


_MAX_CONSECUTIVE_UNTRAINABLE_BATCHES = 10

# Per-field pad values + dtypes for a packed row.
_PAD_VALUES: dict[str, int | float | bool] = {
    "input_ids": 0,  # overwritten with pad_id in __init__-bound builds
    "labels": 0,
    "generator_logprobs": 0.0,
    "loss_mask": False,
    "advantages": 0.0,
}
_DTYPES: dict[str, type[np.generic]] = {
    "input_ids": np.int64,
    "labels": np.int64,
    "generator_logprobs": np.float32,
    "loss_mask": np.bool_,
    "advantages": np.float32,
}


def assign_sorted_bins_to_dp_ranks(
    workloads: list[int], *, dp_degree: int, num_pp_microbatches: int
) -> list[list[int]]:
    """Assign D-wide rows of sorted bins to DP ranks within each G step.

    ``workloads`` is ordered from highest to lowest attention cost. For each
    row of D bins, give the heaviest bin to the rank with the lowest cost
    accumulated in this G step, the next to the next-lowest rank, and so on.
    Rank totals reset for the next G step; every rank receives exactly one
    bin from each of its M rows. This is a greedy heuristic: it does not
    guarantee a smaller maximum rank cost than alternating zig-zag. For M=1,
    keep the original global zig-zag rank order.

    Example:
        # D=2, M=3; the bins are already sorted by attention cost.
        assign_sorted_bins_to_dp_ranks(
            [20, 10, 9, 8, 7, 1], dp_degree=2, num_pp_microbatches=3
        )
        # -> [[0, 1], [3, 2], [5, 4]] (bin indices per rank, per row)
        # Rank totals are [29, 26], versus [35, 20] for zig-zag.
    """
    assert dp_degree > 0 and num_pp_microbatches > 0
    num_bins_per_step = dp_degree * num_pp_microbatches
    assert len(workloads) % num_bins_per_step == 0

    assignments: list[list[int]] = []
    for step_start in range(0, len(workloads), num_bins_per_step):
        rank_workloads = [0] * dp_degree
        for microbatch in range(num_pp_microbatches):
            row_start = step_start + microbatch * dp_degree
            rank_assignments = [0] * dp_degree
            # Preserve the original global zig-zag when M=1, and break ties
            # with the original within-step zig-zag when M>1.
            if num_pp_microbatches == 1:
                tie_direction = -1 if step_start // num_bins_per_step % 2 else 1
            else:
                tie_direction = -1 if microbatch % 2 else 1
            ranks_by_workload = sorted(
                range(dp_degree),
                key=lambda rank: (rank_workloads[rank], tie_direction * rank),
            )
            for offset, rank in enumerate(ranks_by_workload):
                bin_index = row_start + offset
                rank_assignments[rank] = bin_index
                rank_workloads[rank] += workloads[bin_index]
            assignments.append(rank_assignments)

    return assignments


class Batcher(Configurable):
    """Accumulate `num_prompts_per_train_step` trainable groups and plan one step of
    `[num_microbatches][dp_degree]` microbatches from their summaries.

    It reads only group summaries (`RolloutGroupSummary`: lengths, rewards, no bytes). It
    decides each group's samples (`TrainingSampleBuilder.decide`) and plans the step from their
    specs (`StepPlan`); each trainer DP replica's leader materializes and packs its column
    (`MicrobatchPacker`):

        add_group:  decide (summary) -> accumulate specs
        when num_prompts_per_train_step trainable groups:
                    plan (specs) -> StepPlan

    Example:
        # num_prompts_per_train_step=2, dp_degree=2, 256 tokens/rank
        # The trigger is 2 trainable GROUPS, regardless of how many samples/tokens each contains.
        batcher = Batcher.Config().build(
            num_tokens_per_microbatch_per_dp_rank=256,
            max_context_length=128,
            num_prompts_per_train_step=2,
            dp_degree=2,
        )
        pending, _ = batcher.add_group(summary=summary0)
        batch, _ = batcher.add_group(summary=summary1)
        # pending is None; batch.microbatch_groups: [num_groups][1][2 ranks] -> specs

    With `vision_processor` (`Controller.Config.vision_processor`), it logs how many images
    the planned microbatches encode.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        max_num_documents: int | None = None
        """Maximum non-padding document segments per local trainer microbatch.

        When unset, the trainer uses the microbatch token capacity as the
        CUDA-graph-safe metadata bound without imposing a tighter packing cap.
        """
        training_sample_builder: TrainingSampleBuilder.Config = field(
            default_factory=TrainingSampleBuilder.Config
        )
        # Defined below this class.
        microbatch_packer: "MicrobatchPacker.Config" = field(
            default_factory=lambda: MicrobatchPacker.Config()
        )
        """How each planned microbatch is packed. The batcher plans with the same padding and
        MTP depth; the trainer's DP replica leaders build and run the packer."""

        def __post_init__(self) -> None:
            if self.max_num_documents is not None and self.max_num_documents <= 0:
                raise ValueError("max_num_documents must be positive")

    def __init__(
        self,
        config: Config,
        *,
        num_tokens_per_microbatch_per_dp_rank: int,
        max_context_length: int,
        num_prompts_per_train_step: int,
        dp_degree: int,
        num_pp_microbatches: int = 1,
        vision_processor: VisionProcessor.Config | None = None,
    ) -> None:
        self._training_sample_builder = config.training_sample_builder.build()
        self.seq_len = max_context_length
        if num_tokens_per_microbatch_per_dp_rank < max_context_length:
            raise ValueError(
                "num_tokens_per_microbatch_per_dp_rank "
                f"({num_tokens_per_microbatch_per_dp_rank}) must be at least "
                f"max_context_length ({max_context_length})."
            )
        self._num_tokens_per_microbatch = num_tokens_per_microbatch_per_dp_rank
        self._per_sample_pad_multiple = config.microbatch_packer.per_sample_pad_multiple
        self._max_num_documents = config.max_num_documents
        self._num_mtp_layers = config.microbatch_packer.num_mtp_layers
        self._num_prompts_per_train_step = num_prompts_per_train_step
        self._dp_degree = dp_degree
        # Trainer PP runs this many consecutive microbatches per pipeline step.
        self._num_pp_microbatches = num_pp_microbatches
        self._groups_for_next_batch: list[TrainingSampleSpecGroup] = []
        self._num_consecutive_zero_output_groups = 0
        self._logs_image_metrics = vision_processor is not None

    def _record_untrainable_groups(self, *, group_is_trainable: bool) -> None:
        """Fail when consecutive groups cannot contribute one training sample.

        Each ``num_prompts_per_train_step`` consecutive untrainable groups
        represents one complete untrainable batch. For a target of 8 groups,
        warn after each block of 8 and fail after 10 such batches (80 groups).
        Any trainable group resets the count.
        """
        # A useful group proves the pipeline is making progress, even before
        # enough useful groups have accumulated to form a complete batch.
        if group_is_trainable:
            self._num_consecutive_zero_output_groups = 0
            return

        self._num_consecutive_zero_output_groups += 1
        # Report once per group count that would normally produce a training step.
        if self._num_consecutive_zero_output_groups % self._num_prompts_per_train_step:
            return

        num_untrainable_batches = (
            self._num_consecutive_zero_output_groups // self._num_prompts_per_train_step
        )
        logger.warning(
            "Consecutive untrainable batches: %d/%d (%d rollout groups "
            "produced no trainable samples).",
            num_untrainable_batches,
            _MAX_CONSECUTIVE_UNTRAINABLE_BATCHES,
            self._num_consecutive_zero_output_groups,
        )
        if num_untrainable_batches < _MAX_CONSECUTIVE_UNTRAINABLE_BATCHES:
            return

        raise RuntimeError(
            f"{num_untrainable_batches} consecutive untrainable batches "
            f"({self._num_consecutive_zero_output_groups} rollout groups); "
            "check reward diversity and training-sample filters."
        )

    def add_group(
        self, *, summary: RolloutGroupSummary
    ) -> tuple[StepPlan | None, bool]:
        """Add one group's summary and report whether any samples survive batcher filtering.

        Args:
            summary: One scored rollout group without its bytes, plus its rollout metrics.

        Example:
            batcher = Batcher.Config().build(
                num_tokens_per_microbatch_per_dp_rank=16384,
                max_context_length=2048,
                num_prompts_per_train_step=2,
                dp_degree=1,
            )
            batcher.add_group(summary=summary0)  # -> (None, True)
            batcher.add_group(summary=summary1)  # -> (StepPlan, True)
        """
        with sl.log_trace_span("batcher_decide"):
            training_sample_spec_group = self._training_sample_builder.decide(
                rollout_group=summary.group,
                turn_lengths=summary.turn_lengths,
                num_mtp_layers=self._num_mtp_layers,
            )

        # Drop samples longer than seq_len, the model's context length
        specs = training_sample_spec_group.training_sample_specs
        kept = [spec for spec in specs if self.num_tokens_to_pack(spec) <= self.seq_len]
        num_dropped = len(specs) - len(kept)
        if num_dropped:
            logger.warning(
                "Batcher dropped %d/%d sample(s) exceeding seq_len=%d.",
                num_dropped,
                len(specs),
                self.seq_len,
            )
            training_sample_spec_group = replace(
                training_sample_spec_group,
                training_sample_specs=kept,
                metrics=[
                    *training_sample_spec_group.metrics,
                    m.Metric(
                        "batcher/num_samples_dropped_oversized",
                        m.Sum(float(num_dropped)),
                    ),
                ],
            )

        group_is_trainable = bool(training_sample_spec_group.training_sample_specs)
        # Keep every group, including groups made empty by rollout, builder, or
        # batcher filtering. Empty groups do not count toward the trainable-group
        # target, but they ride with the next training batch so its `group_ids`
        # covers every terminal input. The controller commits those IDs only
        # at the optimizer/checkpoint boundary; dropping an empty group here would
        # leave its dataloader entry pending forever.
        self._groups_for_next_batch.append(training_sample_spec_group)
        self._record_untrainable_groups(group_is_trainable=group_is_trainable)
        num_trainable_groups = sum(
            bool(group.training_sample_specs) for group in self._groups_for_next_batch
        )
        if num_trainable_groups < self._num_prompts_per_train_step:
            return None, group_is_trainable  # accumulate until one full batch is ready
        with sl.log_trace_span("batcher_plan"):
            return self._plan_one_training_batch(), group_is_trainable

    def _plan_one_training_batch(self) -> StepPlan:
        """Plan the oldest accumulated groups (up to `num_prompts_per_train_step` trainable groups)
        into one step from their specs alone."""
        (
            training_sample_specs,
            metrics,
            num_metric_only_groups,
            group_ids,
            trainable_group_ids,
        ) = self._take_groups()
        num_rollout_groups = len(trainable_group_ids)
        assignments = self._assign_training_samples_to_microbatches(
            training_sample_specs, num_pp_microbatches=self._num_pp_microbatches
        )
        # Integer sums, so equal to the packed microbatches' counts summed in any order.
        num_depths = self._num_mtp_layers + 1
        global_loss_token_counts = torch.tensor(
            [
                sum(spec.loss_token_counts[depth] for spec in training_sample_specs)
                for depth in range(num_depths)
            ],
            dtype=torch.int64,
        )
        global_routing_token_counts = torch.tensor(
            [
                sum(spec.routing_token_counts[depth] for spec in training_sample_specs)
                for depth in range(num_depths)
            ],
            dtype=torch.int64,
        )
        num_global_loss_tokens = int(global_loss_token_counts[0])
        num_response_tokens = sum(
            spec.num_response_tokens for spec in training_sample_specs
        )
        num_pp_microbatches = self._num_pp_microbatches
        metrics = [
            *metrics,
            # Keep this response-level metric exact without adding a second
            # token-count field to StepPlan.
            m.Metric(
                "loss/generator_logprob_nan_frac",
                m.NoReduce(
                    (num_response_tokens - num_global_loss_tokens)
                    / max(num_response_tokens, 1)
                ),
            ),
            *self._packing_metrics(
                assignments,
                training_sample_specs,
                num_rollout_groups,
                num_metric_only_groups,
            ),
        ]
        return StepPlan(
            microbatch_groups=[
                assignments[start : start + num_pp_microbatches]
                for start in range(0, len(assignments), num_pp_microbatches)
            ],
            global_loss_token_counts=global_loss_token_counts,
            global_routing_token_counts=global_routing_token_counts,
            metrics=metrics,
            group_ids=group_ids,
            trainable_group_ids=trainable_group_ids,
            # Trainer computes policy_age from these at consume time (faithful to what it trains on).
            # min_policy_version is the oldest version this sample was sampled under.
            min_policy_versions=[
                spec.min_policy_version for spec in training_sample_specs
            ],
        )

    def _take_groups(
        self,
    ) -> tuple[list[TrainingSampleSpec], list[m.Metric], int, list[int], list[int]]:
        """Pop accumulated groups oldest-first until `num_prompts_per_train_step` are taken."""
        taken_training_sample_specs: list[TrainingSampleSpec] = []
        taken_metrics: list[m.Metric] = []
        num_trainable_groups = 0
        cut = 0
        for group in self._groups_for_next_batch:
            if num_trainable_groups >= self._num_prompts_per_train_step:
                break
            cut += 1
            taken_metrics.extend(group.metrics)
            if group.training_sample_specs:
                num_trainable_groups += 1

        # Pack in group-id order so on-policy runs stay reproducible whatever the finish order.
        taken_groups = sorted(
            self._groups_for_next_batch[:cut], key=lambda taken: taken.group_id
        )
        for group in taken_groups:
            taken_training_sample_specs.extend(group.training_sample_specs)

        # surplus carried over
        self._groups_for_next_batch = self._groups_for_next_batch[cut:]
        num_metric_only_groups: int = cut - num_trainable_groups

        return (
            taken_training_sample_specs,
            taken_metrics,
            num_metric_only_groups,
            [group.group_id for group in taken_groups],
            [group.group_id for group in taken_groups if group.training_sample_specs],
        )

    def _assign_training_samples_to_microbatches(
        self,
        specs: list[TrainingSampleSpec],
        *,
        num_pp_microbatches: int = 1,
    ) -> list[list[list[TrainingSampleSpec]]]:
        """Pack samples (their specs) into an FFD-derived grid and balance its attention work.

        T is the token capacity per DP rank, D the DP degree, and M the number
        of PP microbatches per scheduling step. First-fit decreasing (FFD)
        determines B bins subject to T and max_num_documents. Each step has
        M * D slots, so G = ceil(B / (M * D)) steps form a ``[G][M][D]`` grid.
        Try longest-processing-time (LPT) packing: place longest samples first,
        preferring a feasible bin that already holds one of the sample's images,
        then the one with the highest padding-aware attention work, breaking
        ties by fewer packed tokens. If any
        sample cannot fit, discard the partial LPT assignment and reuse FFD,
        splitting its bins to fill the extra slots where possible (the FFD
        fallback ignores image affinity). Both paths
        have the same grid size and therefore the same total padding fraction.

        Sort the bins by padding-aware attention cost (see
        ``_attention_workload``), group consecutive bins into D-wide rows, and
        greedily assign each row's heaviest bin to the rank with the least
        cumulative cost in its G step. This heuristic does not guarantee an
        optimal or zig-zag-dominating maximum rank cost. With M=1 it preserves
        the original alternating rank order between G steps.

        This method returns the first two grid axes flattened as
        ``[G * M][D]``; the trainer runs each M consecutive microbatches as
        one pipeline step.
        """
        num_tokens_per_rank = self._num_tokens_per_microbatch

        # Step 1: find the bin count with FFD, retaining its fallback packing.
        ordered_specs = sorted(
            specs,
            key=self.num_tokens_to_pack,
            reverse=True,
        )
        ffd_bins: list[list[TrainingSampleSpec]] = []
        ffd_bin_num_tokens: list[int] = []

        for spec in ordered_specs:
            num_tokens = self.num_tokens_to_pack(spec)
            destination = next(
                (
                    index
                    for index, bin_ in enumerate(ffd_bins)
                    if ffd_bin_num_tokens[index] + num_tokens <= num_tokens_per_rank
                    and (
                        self._max_num_documents is None
                        or len(bin_) < self._max_num_documents
                    )
                ),
                None,
            )
            if destination is None:
                ffd_bins.append([])
                ffd_bin_num_tokens.append(0)
                destination = len(ffd_bins) - 1

            ffd_bins[destination].append(spec)
            ffd_bin_num_tokens[destination] += num_tokens

        # Step 2: make the [G][M][D] grid rectangular.
        num_bins_per_step = num_pp_microbatches * self._dp_degree
        assert num_bins_per_step > 0
        target_num_bins = (
            math.ceil(len(ffd_bins) / num_bins_per_step) * num_bins_per_step
        )

        # Step 3: repack across the entire grid, preserving the FFD bin count.
        bins = self._pack_with_lpt(ordered_specs, target_num_bins=target_num_bins)
        if bins is None:
            bins = ffd_bins
            self._expand_bins_by_splitting(bins, target_num_bins=target_num_bins)

        # Step 4: estimate attention work, including tail padding.
        workloads = [self._attention_workload(bin_) for bin_ in bins]
        sorted_bin_indices = sorted(
            range(len(bins)), key=workloads.__getitem__, reverse=True
        )

        # Steps 5-6: keep adjacent bins together and balance their rank totals.
        bin_indices_by_rank = assign_sorted_bins_to_dp_ranks(
            [workloads[index] for index in sorted_bin_indices],
            dp_degree=self._dp_degree,
            num_pp_microbatches=num_pp_microbatches,
        )
        return [
            [bins[sorted_bin_indices[index]] for index in rank_indices]
            for rank_indices in bin_indices_by_rank
        ]

    def _attention_workload(self, specs: list[TrainingSampleSpec]) -> int:
        """Estimate document and tail-padding attention as squared lengths.

        For packed sample lengths L and token capacity T, let
        ``q, r = divmod(T - sum(L), seq_len)``. The estimate is
        ``sum(L**2) + q * seq_len**2 + r**2``. Padding positions reset at
        seq_len, so even an empty bin costs ``(T // seq_len) * seq_len**2 + (T % seq_len)**2``.
        This models the existing padding segments without increasing their
        number or changing fixed-size varlen metadata.
        """
        # TODO(rl): Account for hybrid sliding-window or linear-attention layers, and for
        #   vision encoder work (distinct images per bin), which image affinity concentrates.
        sample_lengths = [self.num_tokens_to_pack(spec) for spec in specs]
        num_tokens_per_bin = self._num_tokens_per_microbatch
        num_full_padding_segments, remaining_padding = divmod(
            num_tokens_per_bin - sum(sample_lengths), self.seq_len
        )
        return (
            sum(length**2 for length in sample_lengths)
            + num_full_padding_segments * self.seq_len**2
            + remaining_padding**2
        )

    def _pack_with_lpt(
        self,
        ordered_specs: list[TrainingSampleSpec],
        *,
        target_num_bins: int,
    ) -> list[list[TrainingSampleSpec]] | None:
        """Put longest samples first into a feasible bin that already holds one of their
        images, else into the highest-workload feasible bin.

        Image affinity lets the trainer encode the images of a GRPO group in fewer
        microbatches (step 1 in `_pack_image_inputs`'s example).
        Adding a sample cannot increase padding-aware attention work. Break
        workload ties by fewer packed tokens to spread full-length documents
        whose work exactly replaces one padding segment.
        Return None if token or document capacity blocks a sample; the caller
        then discards this partial assignment and uses the FFD fallback.
        """
        num_tokens_per_bin = self._num_tokens_per_microbatch
        bins: list[list[TrainingSampleSpec]] = [[] for _ in range(target_num_bins)]
        bin_num_tokens = [0] * target_num_bins
        bin_workloads = [self._attention_workload([])] * target_num_bins
        bin_image_hashes: list[set[str]] = [set() for _ in range(target_num_bins)]

        for spec in ordered_specs:
            num_tokens = self.num_tokens_to_pack(spec)
            image_hashes = {image.hash_value for image in spec.images}
            destination = max(
                (
                    index
                    for index, bin_ in enumerate(bins)
                    if bin_num_tokens[index] + num_tokens <= num_tokens_per_bin
                    and (
                        self._max_num_documents is None
                        or len(bin_) < self._max_num_documents
                    )
                ),
                # False for every bin when a sample has no images, so text samples pick
                # the same bin as before.
                key=lambda index: (
                    not image_hashes.isdisjoint(bin_image_hashes[index]),
                    bin_workloads[index],
                    -bin_num_tokens[index],
                ),
                default=None,
            )
            if destination is None:
                return None

            bins[destination].append(spec)
            bin_num_tokens[destination] += num_tokens
            bin_workloads[destination] = self._attention_workload(bins[destination])
            bin_image_hashes[destination] |= image_hashes

        return bins

    def _expand_bins_by_splitting(
        self,
        bins: list[list[TrainingSampleSpec]],
        *,
        target_num_bins: int,
    ) -> None:
        """Fill extra FFD grid slots from multi-sample bins when this lowers cost.

        A max-heap tracks each bin with more than one sample by attention
        workload. For every required bin, move a sample from the current
        heaviest donor only if it reduces the higher of the two bins' estimated
        costs, including padding. Filling stops at the new bin's
        token or document limit. Keeping one sample in every donor avoids
        replacing one empty bin with another.

        Every successful inner-loop iteration moves one sample. The outer loop
        either appends a non-empty bin or stops and pads the remaining slots, so
        neither loop can stall when no further redistribution is possible.
        """
        num_tokens_per_bin = self._num_tokens_per_microbatch
        donor_heap = [
            (-self._attention_workload(bin_), -len(bin_), index)
            for index, bin_ in enumerate(bins)
            if len(bin_) > 1
        ]
        heapq.heapify(donor_heap)

        while len(bins) < target_num_bins:
            new_bin: list[TrainingSampleSpec] = []
            new_bin_num_tokens = 0
            skipped_donors: list[tuple[int, int, int]] = []

            while donor_heap and (
                self._max_num_documents is None
                or len(new_bin) < self._max_num_documents
            ):
                negative_workload, _, donor_index = heapq.heappop(donor_heap)
                donor = bins[donor_index]
                donor_workload = -negative_workload
                new_bin_workload = self._attention_workload(new_bin)
                current_max_workload = max(donor_workload, new_bin_workload)
                best_move: tuple[int, int, int, int] | None = None
                for spec_index, spec in enumerate(donor):
                    num_tokens = self.num_tokens_to_pack(spec)
                    if new_bin_num_tokens + num_tokens > num_tokens_per_bin:
                        continue
                    donor_after = self._attention_workload(
                        donor[:spec_index] + donor[spec_index + 1 :]
                    )
                    new_bin_after = self._attention_workload([*new_bin, spec])
                    max_after = max(donor_after, new_bin_after)
                    if max_after >= current_max_workload:
                        continue
                    move = (max_after, -num_tokens, spec_index, donor_after)
                    if best_move is None or move < best_move:
                        best_move = move

                if best_move is None:
                    skipped_donors.append((negative_workload, -len(donor), donor_index))
                    continue

                _, negative_num_tokens, spec_index, donor_after = best_move
                num_tokens = -negative_num_tokens
                spec = donor[spec_index]
                donor.pop(spec_index)
                new_bin.append(spec)
                new_bin_num_tokens += num_tokens

                if len(donor) > 1:
                    heapq.heappush(
                        donor_heap,
                        (-donor_after, -len(donor), donor_index),
                    )

                # Filling the new bin can make a previously skipped move useful.
                for donor_entry in skipped_donors:
                    heapq.heappush(donor_heap, donor_entry)
                skipped_donors.clear()

            for donor_entry in skipped_donors:
                heapq.heappush(donor_heap, donor_entry)

            if not new_bin:
                break

            bins.append(new_bin)
            if len(new_bin) > 1:
                heapq.heappush(
                    donor_heap,
                    (-self._attention_workload(new_bin), -len(new_bin), len(bins) - 1),
                )

        bins.extend([] for _ in range(target_num_bins - len(bins)))

    def num_tokens_to_pack(self, spec: TrainingSampleSpec) -> int:
        """Tokens the sample `spec` describes contributes to a packed input.

        The loss-target split drops the last token (``input_ids = raw[:-1]``), and batch-invariant
        mode rounds the length up to ``per_sample_pad_multiple``.

        Example:

            # token_ids of length 6, per_sample_pad_multiple=1     -> 5
            # token_ids of length 6, per_sample_pad_multiple=8     -> 8
        """
        num_tokens = spec.num_tokens - 1
        return round_up(num_tokens, self._per_sample_pad_multiple)

    def _padding_fraction(
        self,
        *,
        num_microbatches: int,
        specs: list[TrainingSampleSpec],
    ) -> float:
        num_tokens_per_rank = self._num_tokens_per_microbatch
        total_slots = num_microbatches * self._dp_degree * num_tokens_per_rank
        num_real_tokens = sum(spec.num_tokens - 1 for spec in specs)
        return (total_slots - num_real_tokens) / total_slots

    def _packing_metrics(
        self,
        assignments: list[list[list[TrainingSampleSpec]]],
        specs: list[TrainingSampleSpec],
        num_rollout_groups: int,
        num_metric_only_groups: int,
    ) -> list[m.Metric]:
        """Per-training-batch packing + count metrics. (policy age is logged at trainer consume time.)"""
        padding_frac = self._padding_fraction(
            num_microbatches=len(assignments),
            specs=specs,
        )
        image_metrics = []
        if self._logs_image_metrics:
            # Images the trainer encodes (once per microbatch) vs images the samples show.
            num_encoded_images = sum(
                len({image.hash_value for spec in bin_ for image in spec.images})
                for row in assignments
                for bin_ in row
            )
            image_metrics = [
                m.Metric(
                    "train_batch/num_sample_images",
                    m.NoReduce(float(sum(len(spec.images) for spec in specs))),
                ),
                m.Metric(
                    "train_batch/num_encoded_images",
                    m.NoReduce(float(num_encoded_images)),
                ),
            ]
        return [
            *image_metrics,
            m.Metric(
                "train_batch/padding_frac",
                m.NoReduce(padding_frac),
            ),
            m.Metric(
                "train_batch/num_microbatches",
                m.NoReduce(float(len(assignments))),
            ),
            m.Metric(
                "train_batch/num_rollout_groups", m.NoReduce(float(num_rollout_groups))
            ),
            m.Metric(
                "train_batch/num_metric_only_groups",
                m.NoReduce(float(num_metric_only_groups)),
            ),
            m.Metric(
                "train_batch/num_training_samples",
                m.NoReduce(float(len(specs))),
            ),
        ]


# TODO(async-rl): make packing pluggable -- a `Packer` protocol on `Batcher.Config` (e.g. `TextPacker`)
#   so callers swap logic per modality, and move the image path (`_pack_image_inputs`, the
#   `images` branches) into an image packer.
class MicrobatchPacker(Configurable):
    """Pack training samples into one DP rank's fixed-size `TrainingMicrobatch`.

    The bytes half of batching: the batcher plans which samples share a microbatch from their
    specs (`StepPlan`), and each trainer DP replica's leader packs each microbatch of its column
    here from the materialized samples (`MicrobatchGroupReader`).

    Example:
        packer = MicrobatchPacker.Config().build(
            num_tokens_per_microbatch_per_dp_rank=8,
            max_context_length=8,
            pad_id=0,
            temperature=1.0,
            vision_processor=None,
        )
        packer.pack([sample_a, sample_b]).input  # [8]: both samples, then padding
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        per_sample_pad_multiple: int = 1
        """Pad each sample to this multiple after shifting labels, before packing.
        1 leaves lengths unchanged. Used by flex attention in batch-invariant mode
        so that block boundaries align regardless of batch composition."""
        num_mtp_layers: int = 0
        """Model-derived MTP depth used while preparing token counts."""
        build_mrope_positions: bool = False
        """Give microbatches with images 3D M-RoPE positions (``mrope_positions``), as SFT's
        ``MultiModalCollator.Config.build_mrope_positions`` does: True for models whose image
        tokens take M-RoPE positions (``MRoPE.Config`` attention, e.g. Qwen3.5); other models'
        image tokens take the 1D ``positions``. `Controller.Config` checks it against the
        trainer model."""

        def __post_init__(self) -> None:
            if self.num_mtp_layers < 0:
                raise ValueError("num_mtp_layers must be non-negative")
            if self.per_sample_pad_multiple < 1:
                raise ValueError("per_sample_pad_multiple must be positive")

    def __init__(
        self,
        config: Config,
        *,
        num_tokens_per_microbatch_per_dp_rank: int,
        max_context_length: int,
        pad_id: int,
        temperature: float,
        vision_processor: VisionProcessor.Config | None,
    ) -> None:
        self._num_tokens_per_microbatch = num_tokens_per_microbatch_per_dp_rank
        self._max_context_length = max_context_length
        self._pad_id = pad_id
        self._temperature = temperature
        self._per_sample_pad_multiple = config.per_sample_pad_multiple
        self._num_mtp_layers = config.num_mtp_layers
        self._build_mrope_positions = config.build_mrope_positions
        self._vision_processor = (
            None if vision_processor is None else vision_processor.build()
        )

    def pack_planned(
        self,
        specs: list[TrainingSampleSpec],
        training_samples: list[TrainingSample],
    ) -> TrainingMicrobatch:
        """`pack(training_samples)` for a microbatch the batcher planned from `specs` (the
        samples' specs, in the same order), checking the samples against them.

        The step's global token counts are the specs' counts summed, and the loss is divided by
        them, so a sample that disagrees with its spec would silently rescale the loss.

        Raises:
            ValueError: A sample's length is not its spec's `num_tokens`, or the packed loss-
                or routing-token counts are not the specs' counts summed.
        """
        disagreement = (
            "the samples TrainingSampleBuilder.materialize built disagree with the specs "
            "TrainingSampleBuilder.rollout_to_training_sample_specs planned them from."
        )
        for spec, training_sample in zip(specs, training_samples, strict=True):
            if len(training_sample.token_ids) != spec.num_tokens:
                raise ValueError(
                    f"Sample {spec.rollout_id} has {len(training_sample.token_ids)} tokens, but "
                    f"its spec has {spec.num_tokens}: {disagreement}"
                )
        microbatch = self.pack(training_samples)
        num_depths = self._num_mtp_layers + 1
        expected_loss_token_counts = [
            sum(spec.loss_token_counts[depth] for spec in specs)
            for depth in range(num_depths)
        ]
        expected_routing_token_counts = [
            sum(spec.routing_token_counts[depth] for spec in specs)
            for depth in range(num_depths)
        ]
        loss_token_counts = microbatch.loss_token_counts.tolist()
        routing_token_counts = microbatch.routing_token_counts.tolist()
        if (
            loss_token_counts != expected_loss_token_counts
            or routing_token_counts != expected_routing_token_counts
        ):
            raise ValueError(
                f"A packed microbatch has loss-token counts {loss_token_counts} and routing-token "
                f"counts {routing_token_counts}, but its specs sum to "
                f"{expected_loss_token_counts} and {expected_routing_token_counts}: {disagreement}"
            )
        return microbatch

    def pack(self, training_samples: list[TrainingSample]) -> TrainingMicrobatch:
        """Concatenate samples into one fixed-size local microbatch.

        - Labels and logits are shifted
        -`positions` restart at 0 per sample

        Example:

            # Two three-token samples in an eight-token local microbatch.
            input_ids = [10, 11, 20, 21, 0, 0, 0, 0]
            labels    = [11, 12, 21, 22, 0, 0, 0, 0]
            positions = [ 0,  1,  0,  1, 0, 1, 2, 3]
        """
        pad_values = {**_PAD_VALUES, "input_ids": self._pad_id, "labels": self._pad_id}
        keys = list(pad_values)
        packed_fields: dict[str, list[np.ndarray]] = {
            key: [] for key in (*keys, "positions", "padding_mask")
        }
        sample_lens: list[int] = []

        def pad(key: str, num_tokens: int) -> np.ndarray:
            return np.full(num_tokens, pad_values[key], dtype=_DTYPES[key])

        # Shift labels/logits and pad to per_sample_pad_multiple.
        for training_sample in training_samples:
            sample = {
                "input_ids": training_sample.token_ids[:-1],
                "labels": training_sample.token_ids[1:],
                "generator_logprobs": training_sample.logprobs[1:],
                "loss_mask": training_sample.loss_mask[1:],
                "advantages": training_sample.advantage[1:],
            }
            sample_len = append_padded_segment(
                packed_fields,
                sample,
                per_sample_pad_multiple=self._per_sample_pad_multiple,
                pad_values=pad_values,
                padding_mask=None,
            )
            sample_lens.append(sample_len)

        num_tokens_per_rank = self._num_tokens_per_microbatch
        pad_len = num_tokens_per_rank - sum(sample_lens)
        assert pad_len >= 0
        if pad_len > 0:
            for key in keys:
                packed_fields[key].append(pad(key, pad_len))
            packed_fields["positions"].append(
                np.arange(pad_len, dtype=np.int64) % self._max_context_length
            )
            packed_fields["padding_mask"].append(np.ones(pad_len, dtype=bool))

        # Concatenate into the packed dtype (e.g. float64 logprobs -> float32); torch.from_numpy doesn't copy
        packed = {
            key: torch.from_numpy(
                np.concatenate(packed_fields[key], dtype=_DTYPES[key])
            )
            for key in keys
        }
        generator_logprobs = packed["generator_logprobs"]
        loss_mask = packed["loss_mask"]
        target_mask = loss_mask & torch.isfinite(generator_logprobs)
        positions_tensor = torch.from_numpy(np.concatenate(packed_fields["positions"]))
        padding_mask_tensor = torch.from_numpy(
            np.concatenate(packed_fields["padding_mask"])
        )
        loss_token_counts, routing_token_counts = get_mtp_token_counts(
            target_mask=target_mask,
            positions=positions_tensor,
            padding_mask=padding_mask_tensor,
            num_mtp_layers=self._num_mtp_layers,
        )
        model_kwargs = {}
        vision_processor = None
        if any(sample.images for sample in training_samples):
            # The controller gives the packer the renderer's vision processor.
            assert self._vision_processor is not None
            model_kwargs = self._pack_image_inputs(
                training_samples,
                processor=self._vision_processor,
                sample_lens=sample_lens,
                positions=positions_tensor,
            )
            vision_processor = self._vision_processor.config
        return TrainingMicrobatch(
            input=packed["input_ids"],
            labels=packed["labels"],
            positions=positions_tensor,
            generator_logprobs=generator_logprobs,
            # TODO: support per-turn temperature: record it on each RolloutTurn and carry it
            # per token like generator_logprobs, instead of the run's sampling temperature.
            temperature=torch.full_like(generator_logprobs, self._temperature),
            loss_mask=loss_mask,
            advantages=packed["advantages"],
            padding_mask=padding_mask_tensor,
            loss_token_counts=loss_token_counts,
            routing_token_counts=routing_token_counts,
            model_kwargs=model_kwargs,
            vision_processor=vision_processor,
        )

    def _pack_image_inputs(
        self,
        training_samples: list[TrainingSample],
        *,
        processor: VisionProcessor,
        sample_lens: list[int],
        positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build a packed microbatch's image model inputs: each distinct image file once, its
        grid, each placeholder token's row in the image embeddings (-1 for text), and with
        `build_mrope_positions` 3D M-RoPE positions.

        The samples of a GRPO group show the same images. They are deduped in two
        steps: the batcher plans a group's samples into the same microbatch (step 1), then the
        packer ships and the trainer encodes each distinct image of a microbatch once (step 2,
        here). Example: one step of
        2 groups x 2 samples on 2 DP ranks, image k in group g0 and image m in group g1, each
        placeholder token drawn as its image's letter:

            samples       a0: t k k k t t     a1: t k k k t t t      (group g0)
                          b0: t m m t t       b1: t m m t t t        (group g1)

            1. `_pack_with_lpt` picks each sample's microbatch, preferring one that already
               holds the sample's image:
                          rank 0: [a0 a1]  rank 1: [b0 b1]     k and m each encoded once
               instead of rank 0: [a0 b0]  rank 1: [a1 b1]     k and m each encoded on both ranks

            2. This method builds rank 0's image inputs. The image embeddings (the model's
               "vision bank") are the vision encoder's output for the microbatch's distinct
               images, stacked in order, one row per image token; each token gets its row:
                          tokens       t  k k k  t  t | t  k k k  t  t  t | pad
                          rows        -1  0 1 2 -1 -1 |-1  0 1 2 -1 -1 -1 |  -1   image_vision_bank_indices_T
                          image_files  [file k]    each distinct image (by hash) once, ~15 KB
                          grid_thw     [grid of k]

            3. `TrainingMicrobatch.preprocess_images`, on the trainer: file k ->
               resize_and_normalize -> patchify_all -> pixel_values (~2.4 MB of float32),
               once per distinct image.

            4. The model encodes k once into embedding rows [e0 e1 e2], and each placeholder
               token takes its row: tokens 1 and 7 both take e0 (`gather_vision_embeds`).

        Placement follows the renderer's placeholder ranges, not the token ids: the policy can
        sample the placeholder id in a completion, where it is text. ``mrope_positions`` (with
        `build_mrope_positions`) restart per sample, like ``positions``; text and padding repeat
        the 1D position on all three axes. Without images the model uses ``positions`` (the
        same values) and runs its dummy vision forward, so every DP rank still runs the encoder.

        Args:
            training_samples: The packed samples, in packing order.
            processor: The vision processor the renderer computed the grids with.
            sample_lens: Packed length of each sample (input tokens + alignment padding).
            positions: The microbatch's packed 1D positions, padding included.
        """
        merge = processor.config.spatial_merge_size
        bank_rows = np.full(len(positions), -1, dtype=np.int64)
        bank_start_by_hash: dict[str, int] = {}
        # Each distinct image of the microbatch once, in the order of their embedding rows:
        # the trainer encodes these, and every placeholder token takes its row.
        distinct_images: list[PromptImage] = []
        num_bank_rows = 0
        mrope_positions = (
            positions.unsqueeze(1).repeat(1, 3) if self._build_mrope_positions else None
        )
        start = 0
        for training_sample, sample_len in zip(
            training_samples, sample_lens, strict=True
        ):
            for image, placeholder in zip(
                training_sample.images, training_sample.image_placeholders, strict=True
            ):
                # A run's embedding rows follow its placeholder count, the vision encoder's
                # embeddings the grid, and the gather does not check that they agree.
                num_tokens = processor.num_placeholder_tokens(image.grid_thw)
                if placeholder.length != num_tokens:
                    raise ValueError(
                        f"Image {image.hash_value} ({image.source_path}): grid {image.grid_thw} "
                        f"gives {num_tokens} tokens, but the prompt has "
                        f"{placeholder.length} placeholders; the renderer and the trainer "
                        "must use the same vision processor."
                    )
                if image.hash_value not in bank_start_by_hash:
                    bank_start_by_hash[image.hash_value] = num_bank_rows
                    distinct_images.append(image)
                    num_bank_rows += num_tokens
                bank_start = bank_start_by_hash[image.hash_value]
                token = start + placeholder.offset
                bank_rows[token : token + num_tokens] = np.arange(
                    bank_start, bank_start + num_tokens
                )
            if training_sample.images and mrope_positions is not None:
                mrope_positions[start : start + sample_len] = get_mrope_positions(
                    sample_len,
                    [
                        (placeholder.offset, (t, h // merge, w // merge))
                        for (t, h, w), placeholder in zip(
                            (image.grid_thw for image in training_sample.images),
                            training_sample.image_placeholders,
                            strict=True,
                        )
                    ],
                )
            start += sample_len
        image_inputs = {
            # The image files, concatenated; `TrainingMicrobatch.preprocess_images` turns
            # them into pixel values on the trainer.
            "image_files": torch.frombuffer(
                bytearray(b"".join(image.data for image in distinct_images)),
                dtype=torch.uint8,
            ),
            "image_file_sizes": torch.tensor(
                [len(image.data) for image in distinct_images], dtype=torch.int64
            ),
            "grid_thw": torch.tensor(
                [image.grid_thw for image in distinct_images], dtype=torch.int64
            ),
            "image_vision_bank_indices_T": torch.from_numpy(bank_rows),
        }
        if mrope_positions is not None:
            image_inputs["mrope_positions"] = mrope_positions
        return image_inputs
