# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Settings for MoE paged stashing, shared by the eager trainer and
GraphTrainer. Importing it does not import Triton."""

from dataclasses import dataclass
from typing import Literal


OverflowCheckMode = Literal["deferred", "blocking"]


@dataclass(kw_only=True, slots=True)
class PagedStashConfig:
    """Settings for MoE paged stashing, shared by the eager trainer and
    GraphTrainer."""

    page_size: int = 64
    """Tokens per paged-stash page. Smaller pages waste less on stashes that do
    not fill a page, at the cost of a longer page record per activation."""

    buffer_size_factor_cuda: float = 1.10
    """Headroom multiplier on the CUDA paged-stash buffers, over the pages the
    measured step and the pipeline schedule say are needed. Matches Megatron's
    moe_paged_stash_buffer_size_factor_cuda."""

    buffer_size_factor_cpu: float = 0.0
    """Headroom multiplier for an optional pinned-host spill buffer, using the
    same page basis as the CUDA factor. 0 disables host spilling, so a full
    CUDA stash goes straight to overflow. Matches Megatron's
    moe_paged_stash_buffer_size_factor_cpu."""

    prefetch_n_layers: int = 1
    """Issue a backward's paged-stash reloads this many paged-layer schedule
    entries early -- at the end of the forward or backward that precedes it on
    the rank, even in another pipeline action -- so the page reads overlap with
    compute. 1 matches Megatron; 0 reloads an activation when backward first
    reads it."""

    skip_immediate_backward: bool = True
    """Keep an activation resident instead of stashing it when the pipeline
    schedule runs that microbatch's backward next, so the stash would be written
    and read straight back with no bubble to hide it in. Applies to the last
    paged layer, the only one whose backward can be the next scheduled compute.
    Matches Megatron's ``remove_paged_tensor_from_stash``."""

    overflow_check: OverflowCheckMode = "deferred"
    """How a paged-stash overflow is detected and acted on. Overflow means
    backward read activations that were never written back, so the step's
    gradients are invalid.
        deferred: copy each rank's flag into pinned host memory
            asynchronously each step and read an earlier step's copy once it
            has landed, so no step syncs. Overflow is fatal and rank-local: a
            rank raises when it reads its own overflow, typically two steps
            later. Until then every rank skips its optimizer steps on device,
            as AMP skips a step with an inf, so the parameters, optimizer
            state and MoE expert biases -- and any checkpoint saved meanwhile
            -- stay as they were before the overflow; the step count, learning
            rate schedule and data position still advance. The flag is sticky,
            so an overflow cannot be lost however far the CPU runs ahead, and a
            blocking read at shutdown covers the final steps. Needs an
            optimizer that can skip a step on device: fused Adam or AdamW
            (``fused=True``, their default). Other optimizers are only
            supported in blocking mode for now.
        blocking: Megatron's behavior. Read the all-reduced flag with .item()
            at every step boundary, rerun an overflowing step with paging
            disabled, and page again from the next step. Unlike Megatron, the
            rerun measures the step's live pages, so the buffers reallocated
            after it fit the step that overflowed them. Under CUDA graphs the
            rerun drops the captured graphs and runs eagerly, and the next step
            captures them again. Never applies an invalid step, recovers in
            process and works with any optimizer, but costs one device sync per
            step."""

    module_fqn: str = "layers.*.moe.routed_experts"
    """Module FQN pattern of the paged-stash region. Paged stashing owns the
    region: every activation saved inside it that the stash can hold is paged,
    and nothing in it is recomputed. The default covers the whole routed-expert
    subtree, including the token dispatch."""

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}.")
        if self.buffer_size_factor_cuda <= 0:
            raise ValueError(
                "buffer_size_factor_cuda must be positive, got "
                f"{self.buffer_size_factor_cuda}."
            )
        if self.buffer_size_factor_cpu < 0:
            raise ValueError(
                "buffer_size_factor_cpu must be non-negative, got "
                f"{self.buffer_size_factor_cpu}."
            )
        if self.prefetch_n_layers < 0:
            raise ValueError(
                "prefetch_n_layers must be non-negative, got "
                f"{self.prefetch_n_layers}."
            )
        if self.overflow_check not in ("deferred", "blocking"):
            raise ValueError(
                "overflow_check must be 'deferred' or 'blocking', got "
                f"{self.overflow_check!r}."
            )


__all__ = ["OverflowCheckMode", "PagedStashConfig"]
