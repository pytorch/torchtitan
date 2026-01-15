# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-stage data training support.

This module provides DataStageManager and StageAwareDataloader for training
with different data mixtures at different stages of training, similar to
approaches used in Qwen3, DeepSeek-V3, and Llama 3.

Example usage:
    [[training.data_stages]]
    name = "general"
    start_step = 0
    end_step = 100000
    dataset_weights = [0.7, 0.2, 0.1]

    [[training.data_stages]]
    name = "reasoning"
    start_step = 100000
    dataset_weights = [0.3, 0.35, 0.35]
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterator

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.config.job_config import DataStage, JobConfig
from torchtitan.tools.logging import logger


@dataclass
class EffectiveStageConfig:
    """Resolved stage config with inherited values from Training."""

    dataset: str
    dataset_path: str | None
    dataset_type: str
    dataset_folders: list[str]
    dataset_weights: list[float] | None
    dataset_random_seed: int
    seq_len: int


class DataStageManager:
    """Manages data stage transitions during training.

    Tracks current stage based on training step, handles stage transitions,
    and builds dataloaders with stage-specific configurations.
    """

    def __init__(
        self,
        job_config: JobConfig,
        build_dataloader_fn: Callable,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: Any,
    ):
        self.job_config = job_config
        self.build_dataloader_fn = build_dataloader_fn
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.tokenizer = tokenizer

        # Convert dicts to DataStage objects if needed (TOML parser returns dicts)
        raw_stages = job_config.training.data_stages
        self.stages: list[DataStage] = []
        for stage in raw_stages:
            if isinstance(stage, dict):
                self.stages.append(DataStage(**stage))
            else:
                self.stages.append(stage)

        self._current_stage_idx = 0

        # Sort stages by start_step for consistent ordering
        if self.stages:
            self.stages.sort(key=lambda s: s.start_step)
            self._validate_stages()
            self._log_stage_plan()

    @property
    def is_enabled(self) -> bool:
        """Check if multi-stage training is enabled."""
        return len(self.stages) > 0

    @property
    def current_stage(self) -> DataStage | None:
        """Get current stage config."""
        if not self.is_enabled:
            return None
        return self.stages[self._current_stage_idx]

    @property
    def current_stage_idx(self) -> int:
        """Get current stage index."""
        return self._current_stage_idx

    def _validate_stages(self) -> None:
        """Validate stage configurations."""
        for i, stage in enumerate(self.stages):
            if stage.start_step < 0:
                raise ValueError(
                    f"Stage '{stage.name}' has invalid start_step: {stage.start_step}"
                )
            if stage.end_step is not None and stage.end_step <= stage.start_step:
                raise ValueError(
                    f"Stage '{stage.name}' has end_step ({stage.end_step}) <= start_step ({stage.start_step})"
                )

        # Check for gaps or overlaps
        for i in range(len(self.stages) - 1):
            current = self.stages[i]
            next_stage = self.stages[i + 1]
            current_end = (
                current.end_step
                if current.end_step is not None
                else next_stage.start_step
            )
            if current_end != next_stage.start_step:
                logger.warning(
                    f"Data stages have gap or overlap between '{current.name}' "
                    f"(end={current_end}) and '{next_stage.name}' (start={next_stage.start_step})"
                )

    def _log_stage_plan(self) -> None:
        """Log the data stage training plan."""
        logger.info("=" * 60)
        logger.info("DATA STAGE TRAINING PLAN")
        logger.info("=" * 60)
        logger.info(f"Total stages: {len(self.stages)}")
        logger.info("")

        training = self.job_config.training
        total_steps = training.steps

        for i, stage in enumerate(self.stages):
            effective = self.get_effective_config(stage)
            end_step = stage.end_step if stage.end_step is not None else total_steps
            stage_steps = end_step - stage.start_step

            # Calculate tokens for this stage
            # tokens = steps * global_batch_size * seq_len
            # Note: global_batch_size may be -1 (auto), so we show what we can
            global_bs = training.global_batch_size
            if global_bs > 0:
                tokens = stage_steps * global_bs * effective.seq_len
                token_str = f"{tokens / 1e9:.2f}B tokens"
            else:
                token_str = (
                    f"{stage_steps} steps × batch_size × {effective.seq_len} seq_len"
                )

            logger.info(f"Stage {i + 1}: {stage.name}")
            logger.info(
                f"  Steps: {stage.start_step:,} -> {end_step:,} ({stage_steps:,} steps)"
            )
            logger.info(f"  Estimated tokens: {token_str}")
            logger.info(f"  Dataset type: {effective.dataset_type}")

            if effective.dataset_folders:
                logger.info(
                    f"  Dataset folders: {len(effective.dataset_folders)} folders"
                )
                for folder in effective.dataset_folders[:3]:  # Show first 3
                    logger.info(f"    - {folder}")
                if len(effective.dataset_folders) > 3:
                    logger.info(
                        f"    ... and {len(effective.dataset_folders) - 3} more"
                    )
            else:
                logger.info(f"  Dataset: {effective.dataset}")

            if effective.dataset_weights:
                weights_str = ", ".join(
                    f"{w:.3f}" for w in effective.dataset_weights[:5]
                )
                if len(effective.dataset_weights) > 5:
                    weights_str += f", ... ({len(effective.dataset_weights)} total)"
                logger.info(f"  Weights: [{weights_str}]")

            logger.info(f"  Sequence length: {effective.seq_len}")
            logger.info("")

        logger.info("=" * 60)

    def get_effective_config(self, stage: DataStage) -> EffectiveStageConfig:
        """Get effective config for a stage, inheriting from Training where not overridden."""
        training = self.job_config.training
        return EffectiveStageConfig(
            dataset=stage.dataset if stage.dataset is not None else training.dataset,
            dataset_path=stage.dataset_path
            if stage.dataset_path is not None
            else training.dataset_path,
            dataset_type=stage.dataset_type
            if stage.dataset_type is not None
            else training.dataset_type,
            dataset_folders=stage.dataset_folders
            if stage.dataset_folders
            else training.dataset_folders,
            dataset_weights=stage.dataset_weights
            if stage.dataset_weights is not None
            else training.dataset_weights,
            dataset_random_seed=stage.dataset_random_seed
            if stage.dataset_random_seed is not None
            else training.dataset_random_seed,
            seq_len=stage.seq_len if stage.seq_len is not None else training.seq_len,
        )

    def find_stage_for_step(self, step: int) -> int:
        """Find the stage index for the given training step."""
        for i, stage in enumerate(self.stages):
            in_range = step >= stage.start_step
            if stage.end_step is not None:
                in_range = in_range and step < stage.end_step
            elif i + 1 < len(self.stages):
                # If no end_step, use next stage's start_step
                in_range = in_range and step < self.stages[i + 1].start_step
            if in_range:
                return i
        # Default to last stage if step exceeds all ranges
        return len(self.stages) - 1

    def set_stage_for_step(self, step: int) -> bool:
        """Set current stage based on step. Returns True if stage changed."""
        new_idx = self.find_stage_for_step(step)
        if new_idx != self._current_stage_idx:
            old_stage = self.stages[self._current_stage_idx]
            self._current_stage_idx = new_idx
            new_stage = self.stages[new_idx]
            return True
        return False

    def maybe_transition_stage(self, step: int) -> bool:
        """Check if stage transition needed at this step. Returns True if transitioned."""
        if not self.is_enabled:
            return False

        new_idx = self.find_stage_for_step(step)
        if new_idx != self._current_stage_idx:
            old_stage = self.stages[self._current_stage_idx]
            new_stage = self.stages[new_idx]

            logger.info("=" * 60)
            logger.info("DATA STAGE TRANSITION")
            logger.info("=" * 60)
            logger.info(f"Step {step}: '{old_stage.name}' -> '{new_stage.name}'")

            old_effective = self.get_effective_config(old_stage)
            new_effective = self.get_effective_config(new_stage)

            # Log what changed
            changes = []
            if old_effective.dataset_weights != new_effective.dataset_weights:
                changes.append("dataset_weights")
            if old_effective.dataset_folders != new_effective.dataset_folders:
                changes.append("dataset_folders")
            if old_effective.seq_len != new_effective.seq_len:
                changes.append(
                    f"seq_len: {old_effective.seq_len} -> {new_effective.seq_len}"
                )

            if changes:
                logger.info(f"Changes: {', '.join(changes)}")
            else:
                logger.info("No config changes (stage name only)")

            if new_effective.dataset_weights:
                weights_str = ", ".join(
                    f"{w:.3f}" for w in new_effective.dataset_weights[:5]
                )
                if len(new_effective.dataset_weights) > 5:
                    weights_str += f", ... ({len(new_effective.dataset_weights)} total)"
                logger.info(f"New weights: [{weights_str}]")

            logger.info("=" * 60)

            self._current_stage_idx = new_idx
            return True
        return False

    def build_dataloader_for_stage(
        self, stage_idx: int | None = None
    ) -> BaseDataLoader:
        """Build dataloader for the specified stage (or current stage if None)."""
        if stage_idx is None:
            stage_idx = self._current_stage_idx

        if not self.is_enabled:
            # No stages configured, use default behavior
            return self.build_dataloader_fn(
                dp_world_size=self.dp_world_size,
                dp_rank=self.dp_rank,
                tokenizer=self.tokenizer,
                job_config=self.job_config,
            )

        stage = self.stages[stage_idx]
        effective = self.get_effective_config(stage)

        logger.info(f"Building dataloader for stage '{stage.name}' (idx={stage_idx})")

        # Temporarily override training config with stage-specific values
        training = self.job_config.training
        original_values = {}
        override_fields = [
            ("dataset", effective.dataset),
            ("dataset_path", effective.dataset_path),
            ("dataset_type", effective.dataset_type),
            ("dataset_folders", effective.dataset_folders),
            ("dataset_weights", effective.dataset_weights),
            ("dataset_random_seed", effective.dataset_random_seed),
            ("seq_len", effective.seq_len),
        ]

        for field_name, new_value in override_fields:
            original_values[field_name] = getattr(training, field_name)
            setattr(training, field_name, new_value)

        try:
            dataloader = self.build_dataloader_fn(
                dp_world_size=self.dp_world_size,
                dp_rank=self.dp_rank,
                tokenizer=self.tokenizer,
                job_config=self.job_config,
            )
        finally:
            # Restore original config values
            for field_name, original_value in original_values.items():
                setattr(training, field_name, original_value)

        return dataloader

    def state_dict(self) -> dict[str, Any]:
        """Return state for checkpointing."""
        return {"current_stage_idx": self._current_stage_idx}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        if "current_stage_idx" in state_dict:
            old_idx = self._current_stage_idx
            self._current_stage_idx = state_dict["current_stage_idx"]
            if self.is_enabled and old_idx != self._current_stage_idx:
                logger.info(
                    f"Restored data stage index: {old_idx} -> {self._current_stage_idx} "
                    f"(stage: '{self.current_stage.name}')"
                )


class StageAwareDataloader(BaseDataLoader):
    """Dataloader wrapper that handles multi-stage training with proper checkpoint support.

    This wrapper:
    1. Manages the underlying dataloader for the current stage
    2. Saves/restores both stage index AND dataloader state for exact checkpoint resume
    3. Rebuilds dataloader on stage transitions

    The key insight for checkpoint correctness:
    - When saving: we save {stage_idx, dataloader_state_for_current_stage}
    - When loading: we restore stage_idx, rebuild dataloader for that stage,
      then restore the dataloader's internal state

    This ensures exact resume: same stage, same position within the dataset.
    """

    def __init__(
        self,
        stage_manager: DataStageManager,
        initial_dataloader: BaseDataLoader,
    ):
        self._stage_manager = stage_manager
        self._dataloader = initial_dataloader
        self._dp_rank = stage_manager.dp_rank
        self._dp_world_size = stage_manager.dp_world_size

    @property
    def dataloader(self) -> BaseDataLoader:
        """Get the underlying dataloader."""
        return self._dataloader

    def rebuild_for_current_stage(self) -> None:
        """Rebuild the underlying dataloader for the current stage."""
        self._dataloader = self._stage_manager.build_dataloader_for_stage()

    def maybe_transition(self, step: int) -> bool:
        """Check for stage transition and rebuild if needed. Returns True if transitioned."""
        if self._stage_manager.maybe_transition_stage(step):
            self.rebuild_for_current_stage()
            return True
        return False

    def __iter__(self) -> Iterator:
        """Iterate over the underlying dataloader."""
        return iter(self._dataloader)

    def __len__(self) -> int:
        """Return length of underlying dataloader if available."""
        return len(self._dataloader)

    def state_dict(self) -> dict[str, Any]:
        """Save state for checkpointing.

        Saves:
        - stage_idx: Which stage we're in
        - dataloader_state: Position within current stage's dataset
        - dp_world_size: For validation on resume
        """
        return {
            "stage_idx": self._stage_manager.current_stage_idx,
            "stage_name": (
                self._stage_manager.current_stage.name
                if self._stage_manager.current_stage
                else None
            ),
            "dataloader_state": self._dataloader.state_dict(),
            "world_size": self._dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state from checkpoint.

        This is the critical method for exact checkpoint resume:
        1. Restore stage_idx to stage manager
        2. Rebuild dataloader for the correct stage
        3. Restore dataloader's internal state (position in dataset)
        """
        if not state_dict:
            return

        # Validate world size consistency
        if "world_size" in state_dict:
            saved_world_size = state_dict["world_size"]
            if saved_world_size != self._dp_world_size:
                raise ValueError(
                    f"Data parallel world size changed from {saved_world_size} to "
                    f"{self._dp_world_size}. Dataloader state is incompatible."
                )

        # Restore stage index
        if "stage_idx" in state_dict and self._stage_manager.is_enabled:
            saved_stage_idx = state_dict["stage_idx"]
            saved_stage_name = state_dict.get("stage_name", "unknown")
            current_stage_idx = self._stage_manager.current_stage_idx

            if saved_stage_idx != current_stage_idx:
                logger.info(
                    f"Checkpoint was at stage '{saved_stage_name}' (idx={saved_stage_idx}), "
                    f"rebuilding dataloader..."
                )
                # Update stage manager's index
                self._stage_manager._current_stage_idx = saved_stage_idx
                # Rebuild dataloader for the restored stage
                self.rebuild_for_current_stage()

        # Restore dataloader state (position in dataset)
        if "dataloader_state" in state_dict:
            try:
                self._dataloader.load_state_dict(state_dict["dataloader_state"])
                logger.info("Restored dataloader position from checkpoint")
            except Exception as e:
                logger.warning(
                    f"Failed to restore dataloader state: {e}. "
                    "Training will resume from beginning of current stage's dataset."
                )


def build_stage_aware_dataloader(
    job_config: JobConfig,
    build_dataloader_fn: Callable,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Any,
) -> tuple[BaseDataLoader, DataStageManager]:
    """Build a stage-aware dataloader if data stages are configured.

    Returns:
        tuple of (dataloader, stage_manager)

        If data_stages is empty, returns (regular_dataloader, disabled_stage_manager)
        If data_stages is configured, returns (StageAwareDataloader, active_stage_manager)
    """
    stage_manager = DataStageManager(
        job_config=job_config,
        build_dataloader_fn=build_dataloader_fn,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
    )

    if stage_manager.is_enabled:
        # Build initial dataloader for stage 0 (or whichever stage step 0 falls into)
        initial_dataloader = stage_manager.build_dataloader_for_stage()
        dataloader = StageAwareDataloader(stage_manager, initial_dataloader)
        logger.info(
            f"Created StageAwareDataloader with {len(stage_manager.stages)} stages"
        )
    else:
        # No stages configured, use standard dataloader
        dataloader = build_dataloader_fn(
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            job_config=job_config,
        )
        logger.info("No data stages configured, using single-stage training")

    return dataloader, stage_manager
