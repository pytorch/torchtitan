# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import cast

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor

from torchtitan.components.checkpoint import (
    AsyncMode,
    CheckpointManager,
    MODEL,
    ModelWrapper,
)
from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl
from torchtitan.tools import filesystem
from torchtitan.tools.logging import logger


@dataclass(frozen=True, slots=True)
class AveragedCheckpoint:
    checkpoint_dir: str
    output_name: str
    source_steps: tuple[int, ...]


class EMA(Configurable):
    """Build model-only checkpoints from checkpoint averages or an on-disk EMA."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        """Whether to build averaged checkpoints."""

        freq: int = 100
        """Averaging frequency in training steps."""

        checkpoint_count: int = 1
        """Number of equally spaced checkpoints in a windowed average."""

        checkpoint_interval: int = 1000
        """Step spacing between regular checkpoints used as sources."""

        start_step: int = -1
        """First averaging step, or -1 to infer it from the window size."""

        decay: float = 1.0
        """Decay inside a fixed averaging window; 1.0 is a uniform average."""

        stateful_decay: float = 0.0
        """Decay for a stateful on-disk EMA; 0.0 uses fixed-window averaging."""

        def __post_init__(self) -> None:
            if self.freq < 1:
                raise ValueError("EMA frequency needs to be at least 1 step.")
            if self.checkpoint_count < 1:
                raise ValueError("EMA checkpoint_count must be at least 1.")
            if self.checkpoint_interval < 1:
                raise ValueError("EMA checkpoint_interval must be at least 1.")
            if self.start_step == 0 or self.start_step < -1:
                raise ValueError("EMA start_step must be positive or -1.")
            if not 0.0 < self.decay <= 1.0:
                raise ValueError(f"EMA decay must be in (0, 1], got {self.decay}.")
            if not 0.0 <= self.stateful_decay < 1.0:
                raise ValueError(
                    "EMA stateful_decay must be in [0, 1), got "
                    f"{self.stateful_decay}."
                )
            if self.checkpoint_count > 1:
                first_step = self._first_step()
                first_source_step = (
                    first_step
                    - (self.checkpoint_count - 1) * self.checkpoint_interval
                )
                if first_source_step < 1:
                    raise ValueError(
                        "EMA start_step does not leave enough positive source steps."
                    )

        def _first_step(self) -> int:
            return (
                self.checkpoint_count * self.checkpoint_interval
                if self.start_step == -1
                else self.start_step
            )

    def __init__(self, config: Config, checkpointer: CheckpointManager):
        self.config = config
        self.checkpointer = checkpointer
        self._warned_window_warmup = False

    def should_update(self, step: int) -> bool:
        return self.config.enable and step % self.config.freq == 0

    def _first_step(self) -> int:
        return self.config._first_step()

    def window_source_steps(self, step: int) -> list[int]:
        if not self.should_update(step) or self.config.checkpoint_count == 1:
            return []
        if step < self._first_step():
            return []

        first_source_step = (
            step
            - (self.config.checkpoint_count - 1)
            * self.config.checkpoint_interval
        )
        return [
            first_source_step + i * self.config.checkpoint_interval
            for i in range(self.config.checkpoint_count)
        ]

    def stateful_source_steps(self, step: int) -> list[int]:
        """Return regular checkpoints not yet folded into the running EMA."""
        if not self.should_update(step) or self.config.stateful_decay <= 0.0:
            return []
        if step < self._first_step():
            return []

        interval = self.config.checkpoint_interval
        previous_update_step = step - self.config.freq
        first_step = max(
            ((previous_update_step // interval) + 1) * interval,
            interval,
        )
        return list(range(first_step, step + 1, interval))

    def previous_stateful_step(self, step: int) -> int | None:
        candidate = step - self.config.freq
        return candidate if candidate >= self._first_step() else None

    def maybe_save(
        self,
        step: int,
        *,
        warmup_steps: int = 0,
    ) -> AveragedCheckpoint | None:
        """Create the averaged checkpoint scheduled for ``step``, if any."""
        if not self.should_update(step):
            return None
        if not self.checkpointer.enable:
            raise ValueError("EMA requires checkpoint.enable=True.")

        stateful_steps = self.stateful_source_steps(step)
        if stateful_steps:
            self._warn_if_window_spans_warmup(stateful_steps, warmup_steps)
            checkpoint_dir = self.save_stateful_checkpoint(
                new_steps=stateful_steps,
                curr_step=step,
                decay=self.config.stateful_decay,
                prev_ema_step=self.previous_stateful_step(step),
            )
            return AveragedCheckpoint(
                checkpoint_dir=checkpoint_dir,
                output_name=f"step-{step}-ema{self.config.stateful_decay:g}",
                source_steps=tuple(stateful_steps),
            )

        window_steps = self.window_source_steps(step)
        if not window_steps:
            return None

        self._warn_if_window_spans_warmup(window_steps, warmup_steps)
        checkpoint_dir = self.save_window_checkpoint(
            source_steps=window_steps,
            curr_step=step,
            decay=self.config.decay,
        )
        output_name = f"step-{step}-averaged-{len(window_steps)}"
        if self.config.decay != 1.0:
            output_name += f"-decay{self.config.decay:g}"
        return AveragedCheckpoint(
            checkpoint_dir=checkpoint_dir,
            output_name=output_name,
            source_steps=tuple(window_steps),
        )

    def _warn_if_window_spans_warmup(
        self,
        source_steps: list[int],
        warmup_steps: int,
    ) -> None:
        if (
            self._warned_window_warmup
            or not warmup_steps
            or source_steps[0] >= warmup_steps
        ):
            return

        self._warned_window_warmup = True
        suggested_start = (
            warmup_steps
            + (self.config.checkpoint_count - 1)
            * self.config.checkpoint_interval
        )
        logger.warning(
            "EMA source window %s reaches back before LR warmup ends at step %s. "
            "Set ema.start_step >= %s to keep the full window after warmup.",
            source_steps,
            warmup_steps,
            suggested_start,
        )

    @staticmethod
    def _local_checkpoint_tensor(value: torch.Tensor) -> torch.Tensor:
        if isinstance(value, DTensor):
            return value.to_local()
        return value

    def _validate_source_checkpoints(
        self,
        source_steps: list[int],
    ) -> list[str]:
        checkpoint_ids = [
            self.checkpointer.get_checkpoint_id(step) for step in source_steps
        ]
        for step, checkpoint_id in zip(source_steps, checkpoint_ids, strict=True):
            if not filesystem.isfile(filesystem.join(checkpoint_id, ".metadata")):
                raise FileNotFoundError(
                    f"Checkpoint step {step} required for EMA was not found at "
                    f"{checkpoint_id}. Increase checkpoint.keep_latest_k or align "
                    "checkpoint.interval with ema.checkpoint_interval."
                )
        return checkpoint_ids

    @sl.log_trace_span("ema_window_checkpoint_save")
    @torch.no_grad()
    def save_window_checkpoint(
        self,
        *,
        source_steps: list[int],
        curr_step: int,
        decay: float = 1.0,
    ) -> str:
        """Save a normalized weighted average of regular model checkpoints."""
        if not self.checkpointer.enable:
            raise ValueError(
                "Cannot average checkpoints when checkpointing is disabled."
            )
        if self.checkpointer.load_only:
            raise ValueError("Cannot average checkpoints in load_only mode.")
        if len(source_steps) < 2:
            raise ValueError("Checkpoint averaging requires at least two source steps.")
        if source_steps != sorted(set(source_steps)):
            raise ValueError(
                "source_steps must be unique and sorted in increasing order, got "
                f"{source_steps}"
            )
        if source_steps[-1] != curr_step:
            raise ValueError(
                f"The newest source step must equal curr_step={curr_step}, got "
                f"{source_steps[-1]}"
            )
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")

        self.checkpointer.maybe_wait_for_saving()
        source_checkpoint_ids = self._validate_source_checkpoints(source_steps)

        num_sources = len(source_steps)
        weights = [
            decay ** (num_sources - 1 - index) for index in range(num_sources)
        ]
        weight_normalizer = sum(weights)
        normalized_weights = [weight / weight_normalizer for weight in weights]
        average_tag = (
            f"avg-{num_sources}"
            if decay == 1.0
            else f"window-{num_sources}-d{decay:g}"
        )
        output_folder = filesystem.join(
            filesystem.join(self.checkpointer.folder, "ema"),
            average_tag,
        )
        checkpoint_id = self.checkpointer.get_checkpoint_id(
            curr_step,
            folder=output_folder,
        )
        if filesystem.isfile(filesystem.join(checkpoint_id, ".metadata")):
            logger.info("Reusing averaged checkpoint: %s", checkpoint_id)
            return checkpoint_id

        logger.info(
            "Averaging model checkpoints at steps %s (decay=%s, weights=%s).",
            source_steps,
            decay,
            [round(weight, 4) for weight in normalized_weights],
        )
        self._write_weighted_model_merge(
            weighted_sources=list(
                zip(source_checkpoint_ids, normalized_weights, strict=True)
            ),
            output_checkpoint_id=checkpoint_id,
            restore_checkpoint_id=source_checkpoint_ids[-1],
        )
        logger.info("Saved averaged model checkpoint to %s.", checkpoint_id)
        return checkpoint_id

    @sl.log_trace_span("ema_stateful_checkpoint_save")
    @torch.no_grad()
    def save_stateful_checkpoint(
        self,
        *,
        new_steps: list[int],
        curr_step: int,
        decay: float,
        prev_ema_step: int | None = None,
    ) -> str:
        """Fold new regular checkpoints into a stateful on-disk EMA."""
        if not self.checkpointer.enable:
            raise ValueError("Cannot save EMA when checkpointing is disabled.")
        if self.checkpointer.load_only:
            raise ValueError("Cannot save EMA in load_only mode.")
        if not new_steps:
            raise ValueError("EMA requires at least one new source step.")
        if new_steps != sorted(set(new_steps)):
            raise ValueError(
                f"new_steps must be unique and sorted increasing, got {new_steps}"
            )
        if new_steps[-1] != curr_step:
            raise ValueError(
                f"The newest new step must equal curr_step={curr_step}, got "
                f"{new_steps[-1]}"
            )
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")

        self.checkpointer.maybe_wait_for_saving()
        new_checkpoint_ids = self._validate_source_checkpoints(new_steps)
        output_folder = filesystem.join(
            filesystem.join(self.checkpointer.folder, "ema"),
            f"stateful-d{decay:g}",
        )
        checkpoint_id = self.checkpointer.get_checkpoint_id(
            curr_step,
            folder=output_folder,
        )
        if filesystem.isfile(filesystem.join(checkpoint_id, ".metadata")):
            logger.info("Reusing EMA checkpoint: %s", checkpoint_id)
            return checkpoint_id

        previous_ema_checkpoint_id: str | None = None
        if prev_ema_step is not None:
            candidate = self.checkpointer.get_checkpoint_id(
                prev_ema_step,
                folder=output_folder,
            )
            if filesystem.isfile(filesystem.join(candidate, ".metadata")):
                previous_ema_checkpoint_id = candidate
            else:
                logger.warning(
                    "Previous EMA checkpoint for step %s not found at %s. "
                    "Restarting the EMA chain from the plain average of %s.",
                    prev_ema_step,
                    candidate,
                    new_steps,
                )

        block_weight = 1.0 / len(new_steps)
        if previous_ema_checkpoint_id is None:
            weighted_sources = [
                (source_checkpoint_id, block_weight)
                for source_checkpoint_id in new_checkpoint_ids
            ]
        else:
            weighted_sources = [(previous_ema_checkpoint_id, decay)] + [
                (source_checkpoint_id, (1.0 - decay) * block_weight)
                for source_checkpoint_id in new_checkpoint_ids
            ]

        self._write_weighted_model_merge(
            weighted_sources=weighted_sources,
            output_checkpoint_id=checkpoint_id,
            restore_checkpoint_id=new_checkpoint_ids[-1],
        )
        logger.info("Saved EMA model checkpoint to %s.", checkpoint_id)
        return checkpoint_id

    @torch.no_grad()
    def _write_weighted_model_merge(
        self,
        *,
        weighted_sources: list[tuple[str, float]],
        output_checkpoint_id: str,
        restore_checkpoint_id: str,
    ) -> None:
        model = cast(ModelWrapper, self.checkpointer.states[MODEL])
        accumulator: dict[str, torch.Tensor] = {}

        try:
            for source_index, (source_checkpoint_id, weight) in enumerate(
                weighted_sources
            ):
                state_dict = model.state_dict()
                dcp.load(state_dict, checkpoint_id=source_checkpoint_id)
                model.load_state_dict(state_dict)

                for key, value in model.state_dict().items():
                    if not torch.is_tensor(value):
                        continue
                    local_value = self._local_checkpoint_tensor(value)
                    if not torch.is_floating_point(local_value):
                        continue
                    cpu_value = local_value.detach().to(
                        device="cpu",
                        dtype=torch.float32,
                    )
                    if key not in accumulator:
                        accumulator[key] = torch.zeros_like(cpu_value)
                    accumulator[key].add_(cpu_value, alpha=weight)

                logger.info(
                    "Accumulated %s with weight %.6f (%s/%s).",
                    source_checkpoint_id,
                    weight,
                    source_index + 1,
                    len(weighted_sources),
                )

            merged_state = model.state_dict()
            for key, accumulated_value in accumulator.items():
                target = self._local_checkpoint_tensor(merged_state[key])
                target.copy_(
                    accumulated_value.to(device=target.device, dtype=target.dtype),
                    non_blocking=False,
                )
            model.load_state_dict(merged_state)
            self.checkpointer.dcp_save(
                model.state_dict(),
                checkpoint_id=output_checkpoint_id,
                async_mode=AsyncMode.DISABLED,
                enable_garbage_collection=True,
            )
        finally:
            logger.info(
                "Restoring live model from %s after averaging.",
                restore_checkpoint_id,
            )
            current_state = model.state_dict()
            dcp.load(current_state, checkpoint_id=restore_checkpoint_id)
            model.load_state_dict(current_state)
