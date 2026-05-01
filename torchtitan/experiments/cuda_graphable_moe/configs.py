# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, fields

from torchtitan.components.loss import ChunkedCELoss, CrossEntropyLoss
from torchtitan.trainer import Trainer


@dataclass(frozen=True, slots=True)
class PagedStashConfig:
    """Buffer and runtime parameters for a paged stash memory policy.

    Each registered paged stash policy maps to one instance of this class.
    The trainer reads these values for buffer creation and overflow handling.
    """

    enabled: bool = True
    """Create paged stash buffers and apply graph surgery.

    When False, SAC saves fc1 _grouped_mm outputs as regular tensors
    (baseline for measuring fragmentation cost).
    """

    buffer_size_factor: float = 1.1
    """Factor to scale estimated_tokens for CUDA buffer over-provisioning."""

    host_buffer_size_factor: float = 0.0
    """Factor for host (pinned CPU) spillover buffer sizing.

    0 means no host buffer.  Positive value sizes the host buffer relative
    to estimated_tokens (same base as CUDA buffer).
    """

    page_size: int = 64
    """Number of tokens per page in the paged stash buffer."""

    buffer_device: str = "cuda"
    """Device for the paged stash buffer."""

    overflow_detection: bool = True
    """Check overflow flags via all_reduce after each step."""

    max_retries: int = 1
    """Maximum retries on overflow (total attempts = 1 + max_retries)."""

    grow_on_overflow: bool = True
    """Grow CUDA buffers 2x on overflow before retrying."""


PAGED_STASH_POLICY_CONFIGS: dict[str, PagedStashConfig] = {
    "paged_stash": PagedStashConfig(),
    "paged_stash_save_only": PagedStashConfig(enabled=False),
    "paged_stash_spillover": PagedStashConfig(
        buffer_size_factor=0.30,
        host_buffer_size_factor=1.0,
    ),
    "paged_stash_overflow_test": PagedStashConfig(
        buffer_size_factor=0.20,
    ),
}


def get_paged_stash_config(memory_policy: str) -> PagedStashConfig | None:
    """Look up paged stash config for a memory policy, or None if not paged stash."""
    return PAGED_STASH_POLICY_CONFIGS.get(memory_policy)


def to_paged_stash_config(
    base_config: Trainer.Config,
):
    """Convert a base Trainer.Config to a PagedStashTrainer.Config.

    Copies all fields from the base config. The compile and
    activation_checkpoint fields are removed (they have different types
    in PagedStashTrainer.Config) and left as defaults; callers should
    explicitly set them via ``_apply_aot_fx_trace_defaults``.
    """
    from torchtitan.experiments.cuda_graphable_moe.train import PagedStashTrainer

    d = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
    d.pop("compile")
    d.pop("activation_checkpoint")

    # aot_fx_trace traces fwd+loss+bwd with torch.autograd.grad which requires
    # all parameters to participate in forward. ChunkedCELoss sets
    # _skip_lm_head=True making lm_head.weight unused in the traced graph.
    if isinstance(d.get("loss"), ChunkedCELoss.Config):
        d["loss"] = CrossEntropyLoss.Config()

    return PagedStashTrainer.Config(**d)
