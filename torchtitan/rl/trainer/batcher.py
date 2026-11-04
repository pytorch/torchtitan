# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collects trainable `TrainingSample`s until a group-count training batch is ready, then packs it.
`Batcher` packs a `TrainerStepBatch` of `[num_microbatches][dp_degree]`
`TrainingMicrobatch`es;
"""

import asyncio
import heapq
import itertools
import logging
import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from torchtitan.components.data.multimodal.mrope import get_mrope_positions

from torchtitan.components.data.multimodal.utils.image import (
    decode_image_files_flat,
    ImageSpec,
    patchify_flat,
)
from torchtitan.components.data.types import TokenizedTrainingMicrobatch

from torchtitan.config import Configurable
from torchtitan.models.deepseek_v3.mtp import get_mtp_token_counts
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import PromptImage, RolloutGroup
from torchtitan.rl.trainer.rdma_microbatch import RDMAMicrobatch, RegisterRDMAMicrobatch
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleBuilder,
    TrainingSampleGroup,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class TrainingMicrobatch(TokenizedTrainingMicrobatch):
    """Packed training microbatch for the RL trainer.

    Each training_sample's raw tokens (length N) are split into
    ``token_ids = raw[:-1]`` and ``labels = raw[1:]`` (both length
    N-1), matching the pre-training dataloader convention.
    """

    generator_logprobs: torch.Tensor  # [T]
    temperature: torch.Tensor  # [T]
    loss_mask: torch.Tensor  # [T]
    advantages: torch.Tensor  # [T]
    image_spec: ImageSpec | None = None
    """Image decode and patchify settings for ``model_kwargs["image_files"]`` (image files,
    concatenated); `to_input_dict` turns them into ``pixel_values``. None without images."""

    def to_input_dict(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> dict[str, Any]:
        """Move the inputs to ``device``, decoding the image files on the CPU and
        patchifying them there.

        The batcher ships the image files, 10x+ fewer bytes than their pixels; uint8 pixels
        go to the device, 8x fewer bytes than float32 patches, and the patches computed on
        the device equal the CPU's bitwise. ``as_input_dict`` still holds the files; only
        this method gives model inputs.
        """
        model_kwargs = dict(self.model_kwargs)
        image_files = model_kwargs.pop("image_files", None)
        file_sizes = model_kwargs.pop("image_file_sizes", None)
        # Explicit base call: zero-argument super() fails in slots=True dataclasses.
        input_dict = TokenizedTrainingMicrobatch.to_input_dict(
            replace(self, model_kwargs=model_kwargs),
            device,
            non_blocking=non_blocking,
        )
        if image_files is not None:
            assert self.image_spec is not None and file_sizes is not None
            # The CPU copy of the grid: no device sync for the shapes.
            grid_thw = self.model_kwargs["grid_thw"].tolist()
            decoded_images = decode_image_files_flat(
                image_files, file_sizes.tolist(), grid_thw, self.image_spec
            )
            input_dict["pixel_values"] = patchify_flat(
                decoded_images.to(device, non_blocking=non_blocking),
                grid_thw,
                self.image_spec,
            )
        return input_dict

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
class TrainerStepBatch:
    """Packed microbatches for one optimizer step.

    Example:
        # 5 training samples, effective length 5 each; 2 rows/rank, dp_degree=1
        # next-fit rows -> [[s5, s5], [s5, s5], [s5]] = 3 rows; rows_per_microbatch = 2 * 1 = 2
        # -> 2 microbatches (3 rows padded to 4 with one pad-only row), one per
        #    group without pipeline parallelism:
        #    microbatch_groups = [[[TrainingMicrobatch(input=[20])]],
        #                         [[TrainingMicrobatch(input=[20])]]]
        # The second microbatch contains one real row and one pad row.
        # global_loss_token_counts = per-objective loss-token counts
        # global_routing_token_counts = per-depth non-padding token counts
    """

    # [num_groups][num_microbatches_per_group][dp_degree]; under PP, each group runs as one
    # pipeline step. RDMAMicrobatches when `rdma_min_tensor_bytes` is set.
    microbatch_groups: (
        list[list[list[TrainingMicrobatch]]] | list[list[list[RDMAMicrobatch]]]
    )
    global_loss_token_counts: torch.Tensor
    global_routing_token_counts: torch.Tensor
    metrics: list[m.Metric]
    group_ids: list[int]
    """Every consumed rollout group, including metric-only groups."""
    # one per packed training_sample; trainer computes policy_age at consume time
    min_policy_versions: list[int]


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
    """Accumulate `num_prompts_per_train_step` groups and packs
    `[num_microbatches][dp_degree]` flat `TrainingMicrobatch`es.

    Example:
        # num_prompts_per_train_step=2, dp_degree=2, 256 tokens/rank
        # The trigger is 2 trainable GROUPS, regardless of how many samples/tokens each contains.
        batcher = Batcher.Config().build(
            num_tokens_per_microbatch_per_dp_rank=256,
            max_context_length=128,
            num_prompts_per_train_step=2,
            dp_degree=2,
            pad_id=0,
            temperature=1.0,
        )
        pending, _ = batcher.add_training_samples(training_sample_group=group0)
        batch, _ = batcher.add_training_samples(training_sample_group=group1)
        # pending is None; batch.microbatch_groups: [num_groups][1][2 ranks];
        # each TrainingMicrobatch.input: [256 tokens]

    With `rdma_min_tensor_bytes` set, it returns `RDMAMicrobatch`es registered with
    `register_rdma_microbatch`, the actor system's RDMA registration (e.g.
    `MonarchRDMAMicrobatch.register`), which is then required.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        num_mtp_layers: int = 0
        """Model-derived MTP depth used while preparing token counts."""
        per_sample_pad_multiple: int | None = None
        """When non-zero, pad each sample to a multiple of this value
        before packing. Used by flex attention in batch-invariant mode
        so that block boundaries align regardless of batch composition."""
        max_num_documents: int | None = None
        """Maximum non-padding document segments per local trainer microbatch.

        When unset, the trainer uses the microbatch token capacity as the
        CUDA-graph-safe metadata bound without imposing a tighter packing cap.
        """
        rdma_min_tensor_bytes: int | None = None
        """Send microbatch tensors of at least this many bytes as RDMA buffers (the actor system's
        `RDMAMicrobatch`), not in the RPC message:
            None (default): every tensor in the message.
            e.g. 1024: tensors >= 1 KiB stay in the batcher; token counts (8 B) and empty tensors go in the message.
        The Monarch actor system uses ibverbs where the host has it, else TCP; a host that lists unusable
        ibverbs devices fails instead. MONARCH_RDMA_DISABLE_IBVERBS=1 forces TCP in every process, weight sync
        included."""
        training_sample_builder: TrainingSampleBuilder.Config = field(
            default_factory=TrainingSampleBuilder.Config
        )
        image_spec: ImageSpec | None = None
        """How the model turns an image into vision tokens (resize, patch and merge sizes);
        the trainer decodes the samples' image files with it. Must equal the renderer's and
        `VLLMEngine.Config.image_spec`, so the trainer sees the generator's pixels. Set it to
        train on image prompts (Qwen3.5-family models: M-RoPE positions and bank-gather
        vision inputs). None: text only; samples with images raise."""

        def __post_init__(self) -> None:
            if self.num_mtp_layers < 0:
                raise ValueError("num_mtp_layers must be non-negative")
            if self.max_num_documents is not None and self.max_num_documents <= 0:
                raise ValueError("max_num_documents must be positive")
            if (
                self.rdma_min_tensor_bytes is not None
                and self.rdma_min_tensor_bytes < 1
            ):
                raise ValueError(
                    "rdma_min_tensor_bytes must be None or >= 1 (RDMABuffer rejects empty tensors), "
                    f"got {self.rdma_min_tensor_bytes}"
                )

    def __init__(
        self,
        config: Config,
        *,
        num_tokens_per_microbatch_per_dp_rank: int,
        max_context_length: int,
        num_prompts_per_train_step: int,
        dp_degree: int,
        num_pp_microbatches: int = 1,
        pad_id: int,
        temperature: float,
        register_rdma_microbatch: RegisterRDMAMicrobatch | None = None,
    ) -> None:
        if (
            config.rdma_min_tensor_bytes is not None
            and register_rdma_microbatch is None
        ):
            raise ValueError(
                "rdma_min_tensor_bytes is set, but no register_rdma_microbatch was given: "
                "the batcher must run in an actor system that provides RDMA."
            )
        self._training_sample_builder = config.training_sample_builder.build()
        self.seq_len = max_context_length
        self._num_rows_per_microbatch, remainder = divmod(
            num_tokens_per_microbatch_per_dp_rank,
            max_context_length,
        )
        if remainder:
            raise ValueError(
                "num_tokens_per_microbatch_per_dp_rank "
                f"({num_tokens_per_microbatch_per_dp_rank}) must be divisible by "
                f"max_context_length ({max_context_length})."
            )
        self.pad_id = pad_id
        self._temperature = temperature
        self._per_sample_pad_multiple = config.per_sample_pad_multiple
        self._max_num_documents = config.max_num_documents
        self._num_mtp_layers = config.num_mtp_layers
        self._num_prompts_per_train_step = num_prompts_per_train_step
        self._dp_degree = dp_degree
        # Trainer PP runs this many consecutive microbatches per pipeline step.
        self._num_pp_microbatches = num_pp_microbatches
        self._groups_for_next_batch: list[TrainingSampleGroup] = []
        self._num_consecutive_zero_output_groups = 0
        self._rdma_min_tensor_bytes = config.rdma_min_tensor_bytes
        self._register_rdma_microbatch = register_rdma_microbatch
        # Batches that trainer ranks may still read, oldest first; each buffer keeps its tensor alive until drop().
        self._batches_being_read: deque[list[list[list[RDMAMicrobatch]]]] = deque()
        self._num_batches_registered = 0
        self._image_spec = config.image_spec

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
        self, *, rollout_group: RolloutGroup
    ) -> tuple[TrainerStepBatch | None, bool]:
        """Build one rollout group's training samples and add them; see `add_training_samples`."""
        with sl.log_trace_span("training_sample_builder"):
            training_sample_group = self._training_sample_builder.build_from_group(
                rollout_group=rollout_group
            )
        with sl.log_trace_span("batcher_pack"):
            return self.add_training_samples(
                training_sample_group=training_sample_group
            )

    def add_training_samples(
        self, *, training_sample_group: TrainingSampleGroup
    ) -> tuple[TrainerStepBatch | None, bool]:
        """Add one group and report whether any samples survive batcher filtering.

        Args:
            training_sample_group: One rollout group's trainable samples plus rollout metrics.

        Example:
            batcher = Batcher.Config().build(
                num_tokens_per_microbatch_per_dp_rank=16384,
                max_context_length=2048,
                num_prompts_per_train_step=2,
                dp_degree=1,
                pad_id=0,
                temperature=1.0,
            )
            batcher.add_training_samples(training_sample_group=group0)  # -> (None, True)
            batcher.add_training_samples(training_sample_group=group1)  # -> (TrainerStepBatch, True)
        """
        # Drop samples longer than seq_len: can't fill a row
        samples = training_sample_group.training_samples
        kept = [s for s in samples if self.num_tokens_to_pack(s) <= self.seq_len]
        num_dropped = len(samples) - len(kept)
        if num_dropped:
            logger.warning(
                "Batcher dropped %d/%d sample(s) exceeding seq_len=%d.",
                num_dropped,
                len(samples),
                self.seq_len,
            )
            training_sample_group = replace(
                training_sample_group,
                training_samples=kept,
                metrics=[
                    *training_sample_group.metrics,
                    m.Metric(
                        "batcher/num_samples_dropped_oversized",
                        m.Sum(float(num_dropped)),
                    ),
                ],
            )

        group_is_trainable = bool(training_sample_group.training_samples)
        # Keep every group, including groups made empty by rollout, builder, or
        # batcher filtering. Empty groups do not count toward the trainable-group
        # target, but they ride with the next training batch so its `group_ids`
        # covers every terminal input. The controller commits those IDs only
        # at the optimizer/checkpoint boundary; dropping an empty group here would
        # leave its dataloader entry pending forever.
        self._groups_for_next_batch.append(training_sample_group)
        self._record_untrainable_groups(group_is_trainable=group_is_trainable)
        num_trainable_groups = sum(
            bool(group.training_samples) for group in self._groups_for_next_batch
        )
        if num_trainable_groups < self._num_prompts_per_train_step:
            return None, group_is_trainable  # accumulate until one full batch is ready
        batch = self._pack_one_training_batch()
        if self._rdma_min_tensor_bytes is None:
            return batch, group_is_trainable
        assert self._register_rdma_microbatch is not None

        # Trainer ranks read these tensors from this process. Microbatch indices
        # count rows across groups, in training order.
        microbatch_indices = itertools.count()
        rdma_microbatch_groups = [
            [
                [
                    self._register_rdma_microbatch(
                        microbatch,
                        min_bytes=self._rdma_min_tensor_bytes,
                        batch_index=self._num_batches_registered,
                        dp_rank=dp_rank,
                        microbatch_index=microbatch_index,
                    )
                    for dp_rank, microbatch in enumerate(row)
                ]
                for row, microbatch_index in zip(group, microbatch_indices)
            ]
            for group in batch.microbatch_groups
        ]
        self._num_batches_registered += 1
        self._batches_being_read.append(rdma_microbatch_groups)
        return (
            replace(batch, microbatch_groups=rdma_microbatch_groups),
            group_is_trainable,
        )

    async def maybe_release_oldest_batch(self) -> None:
        """Free the oldest batch's RDMA buffers; a no-op when `rdma_min_tensor_bytes` is None.

        Call once per trained batch, in training order, after `forward_backward` returns or raises.
        """
        if self._rdma_min_tensor_bytes is None:
            return
        rdma_microbatch_groups = self._batches_being_read.popleft()
        await asyncio.gather(
            *(
                rdma_microbatch.drop()
                for group in rdma_microbatch_groups
                for row in group
                for rdma_microbatch in row
            )
        )

    def _pack_one_training_batch(self) -> TrainerStepBatch:
        """Pack the oldest accumulated groups (up to `num_prompts_per_train_step` trainable groups) into one batch."""
        (
            training_samples,
            metrics,
            num_rollout_groups,
            num_metric_only_groups,
            group_ids,
        ) = self._take_groups()
        assignments = self._assign_training_samples_to_microbatches(
            training_samples, num_pp_microbatches=self._num_pp_microbatches
        )
        microbatches = [
            [self._pack_training_samples(samples) for samples in rank_assignments]
            for rank_assignments in assignments
        ]
        global_loss_token_counts: torch.Tensor | None = None
        global_routing_token_counts: torch.Tensor | None = None
        for rank_microbatches in microbatches:
            for microbatch in rank_microbatches:
                if global_loss_token_counts is None:
                    global_loss_token_counts = torch.zeros_like(
                        microbatch.loss_token_counts
                    )
                    global_routing_token_counts = torch.zeros_like(
                        microbatch.routing_token_counts
                    )
                global_loss_token_counts.add_(microbatch.loss_token_counts)
                assert global_routing_token_counts is not None
                global_routing_token_counts.add_(microbatch.routing_token_counts)
        assert global_loss_token_counts is not None
        assert global_routing_token_counts is not None
        num_global_loss_tokens = int(global_loss_token_counts.reshape(-1)[0].item())
        num_response_tokens = sum(
            int(microbatch.loss_mask.sum().item())
            for rank_microbatches in microbatches
            for microbatch in rank_microbatches
        )
        num_pp_microbatches = self._num_pp_microbatches
        return TrainerStepBatch(
            microbatch_groups=[
                microbatches[start : start + num_pp_microbatches]
                for start in range(0, len(microbatches), num_pp_microbatches)
            ],
            global_loss_token_counts=global_loss_token_counts,
            global_routing_token_counts=global_routing_token_counts,
            metrics=[
                *metrics,
                # Keep this response-level metric exact without adding a second
                # token-count field to TrainerStepBatch.
                m.Metric(
                    "loss/generator_logprob_nan_frac",
                    m.NoReduce(
                        (num_response_tokens - num_global_loss_tokens)
                        / max(num_response_tokens, 1)
                    ),
                ),
                *self._packing_metrics(
                    assignments,
                    training_samples,
                    num_rollout_groups,
                    num_metric_only_groups,
                ),
            ],
            group_ids=group_ids,
            # Trainer computes policy_age from these at consume time (faithful to what it trains on).
            # min_policy_version is the oldest version this training_sample was sampled under.
            min_policy_versions=[
                training_sample.min_policy_version
                for training_sample in training_samples
            ],
        )

    def _take_groups(
        self,
    ) -> tuple[list[TrainingSample], list[m.Metric], int, int, list[int]]:
        """Pop accumulated groups oldest-first until `num_prompts_per_train_step` are taken."""
        taken_training_samples: list[TrainingSample] = []
        taken_metrics: list[m.Metric] = []
        num_trainable_groups = 0
        cut = 0
        for group in self._groups_for_next_batch:
            if num_trainable_groups >= self._num_prompts_per_train_step:
                break
            cut += 1
            taken_metrics.extend(group.metrics)
            if group.training_samples:
                num_trainable_groups += 1

        # Pack in group-id order so on-policy runs stay reproducible whatever the finish order.
        taken_groups = sorted(
            self._groups_for_next_batch[:cut], key=lambda taken: taken.group_id
        )
        for group in taken_groups:
            taken_training_samples.extend(group.training_samples)

        # surplus carried over
        self._groups_for_next_batch = self._groups_for_next_batch[cut:]
        num_metric_only_groups: int = cut - num_trainable_groups

        return (
            taken_training_samples,
            taken_metrics,
            num_trainable_groups,
            num_metric_only_groups,
            [group.group_id for group in taken_groups],
        )

    def _assign_training_samples_to_microbatches(
        self,
        training_samples: list[TrainingSample],
        *,
        num_pp_microbatches: int = 1,
    ) -> list[list[list[TrainingSample]]]:
        """Pack samples into an FFD-derived grid and balance its attention work.

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
        num_tokens_per_rank = self._num_rows_per_microbatch * self.seq_len

        # Step 1: find the bin count with FFD, retaining its fallback packing.
        ordered_samples = sorted(
            training_samples,
            key=self.num_tokens_to_pack,
            reverse=True,
        )
        ffd_bins: list[list[TrainingSample]] = []
        ffd_bin_num_tokens: list[int] = []

        for training_sample in ordered_samples:
            num_tokens = self.num_tokens_to_pack(training_sample)
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

            ffd_bins[destination].append(training_sample)
            ffd_bin_num_tokens[destination] += num_tokens

        # Step 2: make the [G][M][D] grid rectangular.
        num_bins_per_step = num_pp_microbatches * self._dp_degree
        assert num_bins_per_step > 0
        target_num_bins = (
            math.ceil(len(ffd_bins) / num_bins_per_step) * num_bins_per_step
        )

        # Step 3: repack across the entire grid, preserving the FFD bin count.
        bins = self._pack_with_lpt(ordered_samples, target_num_bins=target_num_bins)
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

    def _attention_workload(self, training_samples: list[TrainingSample]) -> int:
        """Estimate document and tail-padding attention as squared lengths.

        For packed sample lengths L and token capacity T, let
        ``q, r = divmod(T - sum(L), seq_len)``. The estimate is
        ``sum(L**2) + q * seq_len**2 + r**2``. Padding positions reset at
        seq_len, so even an empty bin costs ``(T // seq_len) * seq_len**2``.
        This models the existing padding segments without increasing their
        number or changing fixed-size varlen metadata.
        """
        # TODO(rl): Account for hybrid sliding-window or linear-attention layers, and for
        #   vision encoder work (distinct images per bin), which image affinity concentrates.
        sample_lengths = [
            self.num_tokens_to_pack(sample) for sample in training_samples
        ]
        num_tokens_per_bin = self._num_rows_per_microbatch * self.seq_len
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
        ordered_samples: list[TrainingSample],
        *,
        target_num_bins: int,
    ) -> list[list[TrainingSample]] | None:
        """Put longest samples first into a feasible bin that already holds one of their
        images, else into the highest-workload feasible bin.

        Image affinity lets the trainer encode the images of a GRPO group in fewer
        microbatches.
        Adding a sample cannot increase padding-aware attention work. Break
        workload ties by fewer packed tokens to spread full-length documents
        whose work exactly replaces one padding segment.
        Return None if token or document capacity blocks a sample; the caller
        then discards this partial assignment and uses the FFD fallback.
        """
        num_tokens_per_bin = self._num_rows_per_microbatch * self.seq_len
        bins: list[list[TrainingSample]] = [[] for _ in range(target_num_bins)]
        bin_num_tokens = [0] * target_num_bins
        bin_workloads = [self._attention_workload([])] * target_num_bins
        bin_image_keys: list[set[str]] = [set() for _ in range(target_num_bins)]

        for sample in ordered_samples:
            num_tokens = self.num_tokens_to_pack(sample)
            image_keys = {image.key for image in sample.images}
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
                    not image_keys.isdisjoint(bin_image_keys[index]),
                    bin_workloads[index],
                    -bin_num_tokens[index],
                ),
                default=None,
            )
            if destination is None:
                return None

            bins[destination].append(sample)
            bin_num_tokens[destination] += num_tokens
            bin_workloads[destination] = self._attention_workload(bins[destination])
            bin_image_keys[destination] |= image_keys

        return bins

    def _expand_bins_by_splitting(
        self,
        bins: list[list[TrainingSample]],
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
        num_tokens_per_bin = self._num_rows_per_microbatch * self.seq_len
        donor_heap = [
            (-self._attention_workload(bin_), -len(bin_), index)
            for index, bin_ in enumerate(bins)
            if len(bin_) > 1
        ]
        heapq.heapify(donor_heap)

        while len(bins) < target_num_bins:
            new_bin: list[TrainingSample] = []
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
                for sample_index, sample in enumerate(donor):
                    num_tokens = self.num_tokens_to_pack(sample)
                    if new_bin_num_tokens + num_tokens > num_tokens_per_bin:
                        continue
                    donor_after = self._attention_workload(
                        donor[:sample_index] + donor[sample_index + 1 :]
                    )
                    new_bin_after = self._attention_workload([*new_bin, sample])
                    max_after = max(donor_after, new_bin_after)
                    if max_after >= current_max_workload:
                        continue
                    move = (max_after, -num_tokens, sample_index, donor_after)
                    if best_move is None or move < best_move:
                        best_move = move

                if best_move is None:
                    skipped_donors.append((negative_workload, -len(donor), donor_index))
                    continue

                _, negative_num_tokens, sample_index, donor_after = best_move
                num_tokens = -negative_num_tokens
                sample = donor[sample_index]
                donor.pop(sample_index)
                new_bin.append(sample)
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

    def num_tokens_to_pack(self, training_sample: TrainingSample) -> int:
        """Tokens this training_sample contributes to a packed input.

        The loss-target split drops the last token (``input_ids = raw[:-1]``), and batch-invariant
        mode rounds the length up to ``per_sample_pad_multiple``.

        Example:

            # token_ids of length 6, per_sample_pad_multiple=None  -> 5
            # token_ids of length 6, per_sample_pad_multiple=8     -> 8
        """
        num_tokens = len(training_sample.token_ids) - 1
        if self._per_sample_pad_multiple:
            multiple = self._per_sample_pad_multiple
            num_tokens = ((num_tokens + multiple - 1) // multiple) * multiple
        return num_tokens

    # TODO(async-rl): make packing pluggable -- a `Packer` protocol on `Batcher.Config` (e.g. `TextPacker`)
    #   so callers swap logic per modality, and move the image path (`_pack_image_inputs`, the
    #   `images` branches) into an image packer.
    def _pack_training_samples(
        self, training_samples: list[TrainingSample]
    ) -> TrainingMicrobatch:
        """Concatenate samples into one fixed-size local microbatch.

        - Labels and logits are shifted
        -`positions` restart at 0 per sample

        Example:

            # Two three-token samples in an eight-token local microbatch.
            input_ids = [10, 11, 20, 21, 0, 0, 0, 0]
            labels    = [11, 12, 21, 22, 0, 0, 0, 0]
            positions = [ 0,  1,  0,  1, 0, 1, 2, 3]
        """
        pad_values = {**_PAD_VALUES, "input_ids": self.pad_id, "labels": self.pad_id}
        keys = list(pad_values)
        packed_fields: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        positions: list[np.ndarray] = []
        padding_mask: list[np.ndarray] = []

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
            sample_len = len(sample["input_ids"])
            unpadded_len = sample_len

            # pad to multiple
            if self._per_sample_pad_multiple:
                align = self._per_sample_pad_multiple
                sample_len = ((sample_len + align - 1) // align) * align

            # extend row
            for key in keys:
                packed_fields[key] += [sample[key], pad(key, sample_len - unpadded_len)]
            positions.append(np.arange(sample_len, dtype=np.int64))
            padding_mask.append(np.arange(sample_len) >= unpadded_len)

        num_tokens_per_rank = self._num_rows_per_microbatch * self.seq_len
        sample_lens = [len(sample_positions) for sample_positions in positions]
        pad_len = num_tokens_per_rank - sum(sample_lens)
        assert pad_len >= 0
        if pad_len > 0:
            for key in keys:
                packed_fields[key].append(pad(key, pad_len))
            positions.append(np.arange(pad_len, dtype=np.int64) % self.seq_len)
            padding_mask.append(np.ones(pad_len, dtype=bool))

        # Concatenate into the packed dtype (e.g. float64 logprobs -> float32); torch.from_numpy doesn't copy
        packed = {
            key: torch.from_numpy(np.concatenate(arrays, dtype=_DTYPES[key]))
            for key, arrays in packed_fields.items()
        }
        generator_logprobs = packed["generator_logprobs"]
        loss_mask = packed["loss_mask"]
        target_mask = loss_mask & torch.isfinite(generator_logprobs)
        positions_tensor = torch.from_numpy(np.concatenate(positions))
        padding_mask_tensor = torch.from_numpy(np.concatenate(padding_mask))
        loss_token_counts, routing_token_counts = get_mtp_token_counts(
            target_mask=target_mask,
            positions=positions_tensor,
            padding_mask=padding_mask_tensor,
            num_mtp_layers=self._num_mtp_layers,
        )
        model_kwargs = {}
        if self._image_spec is not None:
            model_kwargs = self._pack_image_inputs(
                training_samples, sample_lens=sample_lens, positions=positions_tensor
            )
        elif any(sample.images for sample in training_samples):
            raise ValueError(
                "Training samples have images, but Batcher.Config.image_spec is None; set it "
                "to the image spec the renderer and the generator use."
            )
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
            image_spec=(self._image_spec if model_kwargs else None),
        )

    def _pack_image_inputs(
        self,
        training_samples: list[TrainingSample],
        *,
        sample_lens: list[int],
        positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build a packed microbatch's image model inputs: each distinct image file once, its
        grid, the bank row of every placeholder token, and 3D M-RoPE positions. Empty when no
        sample has images; text samples in the microbatch get no bank rows.

        The samples of a GRPO group show the same images, so the microbatch's vision bank holds
        each distinct image once and every placeholder token names its bank row
        (``image_vision_bank_indices_T``, -1 for text); the model gathers from the bank instead
        of encoding the image once per sample. ``mrope_positions`` restart per sample, like
        ``positions``; text and padding repeat the 1D position on all three axes. Without
        images the model uses ``positions`` (the same values) and runs its dummy vision
        forward, so every DP rank still runs the encoder.

        Example: two samples of one group, 6 input tokens each, image k (3 tokens) at offset 1:

            input tokens                 t  k  k  k  t  t | t  k  k  k  t  t
            image_vision_bank_indices_T -1  0  1  2 -1 -1 |-1  0  1  2 -1 -1
            image_files                  the file of k, once (decoded on the trainer)

        Args:
            training_samples: The packed samples, in packing order.
            sample_lens: Packed length of each sample (input tokens + alignment padding).
            positions: The microbatch's packed 1D positions, padding included.
        """
        assert self._image_spec is not None
        if not any(sample.images for sample in training_samples):
            return {}
        bank_rows = np.full(len(positions), -1, dtype=np.int64)
        bank_start_by_key: dict[str, int] = {}
        # Each distinct image of the microbatch once, in bank-row order: the trainer encodes
        # these, and every placeholder token gathers its bank row.
        distinct_images: list[PromptImage] = []
        num_bank_rows = 0
        mrope_positions = positions.unsqueeze(1).repeat(1, 3)
        start = 0
        for training_sample, sample_len in zip(
            training_samples, sample_lens, strict=True
        ):
            for image in training_sample.images:
                if image.key not in bank_start_by_key:
                    bank_start_by_key[image.key] = num_bank_rows
                    distinct_images.append(image)
                    num_bank_rows += image.length
                bank_start = bank_start_by_key[image.key]
                token = start + image.offset
                bank_rows[token : token + image.length] = np.arange(
                    bank_start, bank_start + image.length
                )
            if training_sample.images:
                mrope_positions[start : start + sample_len] = get_mrope_positions(
                    sample_len,
                    [
                        (image.offset, image.grid_thw)
                        for image in training_sample.images
                    ],
                    spatial_merge_size=self._image_spec.merge_size,
                )
            start += sample_len
        self._check_placeholder_counts(distinct_images)
        return {
            "mrope_positions": mrope_positions,
            # The image files, concatenated; `TrainingMicrobatch.to_input_dict` decodes the
            # images from them.
            # and patchifies them on the trainer's device.
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

    def _check_placeholder_counts(self, images: list[PromptImage]) -> None:
        """Raise unless each image's placeholder count matches its grid.

        Bank rows follow ``image.length``, the vision encoder's outputs the grid, and the
        bank gather does not check them. The trainer checks the grid against the decoded file.
        """
        assert self._image_spec is not None
        merge = self._image_spec.merge_size
        for image in images:
            t, h, w = image.grid_thw
            if t * h * w // merge**2 != image.length:
                raise ValueError(
                    f"Image {image.key} ({image.path}): grid {image.grid_thw} gives "
                    f"{t * h * w // merge**2} tokens with merge size {merge}, but the prompt "
                    f"has {image.length} placeholders; the renderer and the batcher must use "
                    "the same ImageSpec."
                )

    def _padding_fraction(
        self,
        *,
        num_microbatches: int,
        training_samples: list[TrainingSample],
    ) -> float:
        num_tokens_per_rank = self._num_rows_per_microbatch * self.seq_len
        total_slots = num_microbatches * self._dp_degree * num_tokens_per_rank
        num_real_tokens = sum(len(sample.token_ids) - 1 for sample in training_samples)
        return (total_slots - num_real_tokens) / total_slots

    def _packing_metrics(
        self,
        assignments: list[list[list[TrainingSample]]],
        training_samples: list[TrainingSample],
        num_rollout_groups: int,
        num_metric_only_groups: int,
    ) -> list[m.Metric]:
        """Per-training-batch packing + count metrics. (policy age is logged at trainer consume time.)"""
        padding_frac = self._padding_fraction(
            num_microbatches=len(assignments),
            training_samples=training_samples,
        )
        image_metrics = []
        if self._image_spec is not None:
            # Images the trainer encodes (once per microbatch) vs images the samples show.
            num_encoded_images = sum(
                len({image.key for sample in bin_ for image in sample.images})
                for row in assignments
                for bin_ in row
            )
            image_metrics = [
                m.Metric(
                    "train_batch/num_sample_images",
                    m.NoReduce(
                        float(sum(len(sample.images) for sample in training_samples))
                    ),
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
                m.NoReduce(float(len(training_samples))),
            ),
        ]
