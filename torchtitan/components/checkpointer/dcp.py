# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import enum
import logging
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, cast, Literal, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.observability import structured_logger as sl
from torchtitan.tools import filesystem
from torchtitan.tools.utils import GarbageCollection

from .base import (
    BaseCheckpointManager,
    DATALOADER,
    LR_SCHEDULER,
    MODEL,
    ModelWrapper,
    OPTIMIZER,
    purge_thread,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import torch.nn as nn

    from torchtitan.components.data.loader import BaseDataLoader
    from torchtitan.components.optimizer import (
        LRSchedulersContainer,
        OptimizersContainer,
    )
    from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class SaveDone:
    pass


class _FilesystemCheckpointStorage:
    """``CheckpointStorage`` backed by ``torchtitan.tools.filesystem``.

    Local paths go through ``os``/``shutil`` and remote fsspec URIs through
    fsspec, so DCP keeps reading and writing remote checkpoint folders exactly
    as it did before.
    """

    def isdir(self, path: str) -> bool:
        return filesystem.isdir(path)

    def isfile(self, path: str) -> bool:
        return filesystem.isfile(path)

    def listdir(self, path: str) -> list[str]:
        return filesystem.listdir(path)

    def remove(self, path: str) -> None:
        filesystem.rmtree(path)


class CheckpointManager(BaseCheckpointManager):
    """This class manages the checkpointing logic for the TorchTitan trainer.


    Note: Pipeline Parallelism and Virtual Stages

    1. even for simple PP schedules, there is a separate optimizer each PP rank.
    rank0's optimizer would have a param_group[0] which refers to layers.0 in the
    original model. rank1's would _also_ have a param_group[0], since it's index based,
    but referring to layers.1. When saving, these collide and one of them is lost.
    Then when reloading, only one stage can restore its optimizer states, others will
    error.

        The solution to this problem is optimizer flattening.
        TorchTitan's OptimizersContainer flattens optimizer state dicts to FQN-keyed
        flat dicts using the utilities in torchtitan/components/optimizer/utils.py.

    2. With complex PP schedules, we have multiple model chunks per pp rank. This
    compounds challenge (1) by also requiring us to reason about multiple 'optim'
    objects locally.

        We solve this in the Model and Optimizer wrapper classes by flattening the state
        dicts from each object into one state dict before saving/loading. We rely on the
        individual state_dicts to not collide, which is guaranteed for the model by
        correct pipeline splitting and for the optimizer by the flattening support
        described in (1).

    3. LR schedulers also index model states like optimizers. Here we flatten the
    lr_schedulers with the assumption that all lr_schedulers have the same state_dict.

    Args:
        config (Checkpoint): The config used to configure the checkpointing.
        dataloader (BaseDataLoader): The dataloader used to load the data.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizers (OptimizersContainer): The optimizers used to optimize the model.
        lr_schedulers (LRSchedulersContainer): The lr schedulers used to optimize
            the model.
        states (Dict[str, Any]): The states that need to be saved, other than the
            previous 4 components.
        sd_adapter (Optional[type[BaseStateDictAdapter]]): The adapter used to convert
            model state dicts between native format and other formats.
        base_folder (str): The base folder to save the checkpoint. Will be concatenated
            with config.folder

    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseCheckpointManager.Config):
        async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
        """DCP save mode: synchronous, threaded async, or pinned-memory async."""

        def __post_init__(self) -> None:
            BaseCheckpointManager.Config.__post_init__(self)
            async_lowered = self.async_mode.lower()
            if async_lowered not in (
                "disabled",
                "async",
                "async_with_pinned_mem",
            ):
                raise ValueError(f"Invalid async_mode: {async_lowered}")
            self.async_mode = async_lowered

    def __init__(
        self,
        config: Config,
        *,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        sd_adapter: BaseStateDictAdapter | None,
        base_folder: str = "",
    ) -> None:

        self.enable = config.enable
        if not self.enable:
            return

        self.folder = filesystem.join(base_folder, config.folder)
        self.interval = config.interval
        self._storage = _FilesystemCheckpointStorage()

        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )

        # Loading & Saving Policy
        self.load_only = config.load_only
        self.exclude_from_loading = config.exclude_from_loading
        self.initial_load_path = config.initial_load_path
        self.initial_load_model_only = config.initial_load_model_only
        self.initial_load_in_hf = config.initial_load_in_hf
        self.initial_load_in_hf_quantized = config.initial_load_in_hf_quantized

        self.enable_first_step_checkpoint = config.enable_first_step_checkpoint
        self.last_save_model_only = config.last_save_model_only
        self.last_save_in_hf = config.last_save_in_hf
        self.export_dtype = TORCH_DTYPE_MAP[config.export_dtype]

        self.sd_adapter = sd_adapter
        if self.last_save_in_hf and self.sd_adapter is None:
            raise ValueError(
                "checkpoint.last_save_in_hf is True, but sd_adapter is not provided."
            )

        # Async & Distributed Infrastructure
        try:
            self.async_mode = AsyncMode(config.async_mode)
        except ValueError as e:
            raise ValueError(
                f"Unknown checkpoint async_mode {config.async_mode}"
            ) from e

        self.pg: dist.ProcessGroup | None = None
        if self.async_mode in (AsyncMode.ASYNC, AsyncMode.ASYNC_WITH_PINNED_MEM):
            self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))

        self.stager: DefaultStager | None = None
        self.staging_future: Future | None = None
        self.save_future: Future | None = None

        # Retention Policy (Purge)
        self.keep_latest_k = config.keep_latest_k
        self.purge_exempt = (
            config.purge_exempt.build() if config.purge_exempt is not None else None
        )
        self.purge_thread: threading.Thread | None = None
        if self.keep_latest_k > 0:
            self.purge_queue: queue.Queue[str | None] = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread,
                args=(self.purge_queue, self._storage.remove),
                daemon=True,
            )
            self.purge_thread.start()

        logger.info(
            "Checkpointing active. Checkpoints will be loaded from and saved "
            f"to {self.folder}"
        )

    def __del__(self):
        self.close()

    def _close(self):
        if (
            hasattr(self, "purge_thread")
            and self.purge_thread
            and self.purge_thread.is_alive()
        ):
            self.purge_queue.put(None)
            self.purge_thread.join()

        if self.stager is not None:
            self.stager.close()

    def dcp_save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ) -> Future | AsyncSaveResponse | None:
        """Execute the DCP saving process.

        This method orchestrates the state_dict transformation (e.g., to HuggingFace
        format), selects the appropriate storage writer, and dispatches the save
        operation based on the requested synchronicity mode.

        Args:
            state_dict (dict): The state dict to save.
            checkpoint_id (str): Unique identifier (usually a path) for the checkpoint.
            async_mode (AsyncMode): The saving/staging strategy.
            enable_garbage_collection (bool): To trigger a manual GC collect after save.
            to_hf (bool): If True, uses a HuggingFaceStorageWriter and adapts the
                state_dict to be compatible with safetensors and HF model definitions.

        Returns:
            - None: If saved synchronously (AsyncMode.DISABLED).
            - Future: If AsyncMode.ASYNC is used (tracks disk I/O).
            - AsyncSaveResponse: If AsyncMode.ASYNC_WITH_PINNED_MEM is used
              (tracks both staging and disk I/O).
        """

        ret: Future | AsyncSaveResponse | None = None

        storage_writer: HuggingFaceStorageWriter | None = None
        fqn_to_index_mapping: dict[Any, int] | None = None

        # HF Format Conversion
        if to_hf:
            assert self.sd_adapter is not None, "sd_adapter is required for to_hf=True"
            state_dict = self.sd_adapter.to_hf(state_dict)
            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping

            # If sharded, we save to a subdir then consolidate
            save_path = (
                os.path.join(checkpoint_id, "sharded")
                if fqn_to_index_mapping
                else checkpoint_id
            )
            storage_writer = HuggingFaceStorageWriter(
                path=save_path,
                save_distributed=True,
                fqn_to_index_mapping=fqn_to_index_mapping,
                enable_consolidation=not fqn_to_index_mapping,
            )
            # NOTE: If `fqn_to_index_mapping` is absent, all FQNs are saved into a
            # single unified file. In this case, the StorageWriter can handle
            # consolidation internally on a single rank. However, when a mapping
            # exists, the weights are distributed across multiple files (sharded).
            # The internal consolidation is disabled here and instead
            # `consolidate_safetensors_files_on_every_rank` is used later to manage
            # the multi-file merging process.

        # Execution Dispatch
        checkpoint_save_id = (
            None if to_hf else checkpoint_id
        )  # for HF the storage_writer handles the path

        if async_mode == AsyncMode.ASYNC:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
            )
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ret = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
                process_group=self.pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self.stager,
            )
        else:
            ret = dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_save_id,
            )

        # Post-Processing
        if to_hf and fqn_to_index_mapping:
            consolidate_safetensors_files_on_every_rank(
                input_dir=os.path.join(checkpoint_id, "sharded"),
                output_dir=checkpoint_id,
                fqn_to_index_mapping=fqn_to_index_mapping,
                num_threads=5,
            )

        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

        return ret

    def _load_checkpoint(
        self,
        states: dict[str, Any],
        checkpoint_id: str,
        *,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        """Restore selected states through DCP or its Hugging Face reader.

        This method handles both standard DCP sharded checkpoints and HuggingFace
        safetensors. If loading from HF, it utilizes an adapter to map FQNs and
        handle format-specific sharding logic.

        Args:
            states: Live state objects selected for restoration.
            checkpoint_id: Path or identifier for the source checkpoint.
            from_hf: If True, adapts the load process for HuggingFace model
                definitions and safetensors format.
            from_quantized: Indicates if the source is in a quantized format
                (e.g., 4-bit/8-bit), requiring the storage reader to handle
                specialized data types and sharding structures.

        Raises:
            AssertionError: If `from_hf` is True but no `sd_adapter` is available.
        """
        state_dict = self._flattened_model_states_sd(states)

        if from_hf:
            assert self.sd_adapter is not None, (
                "trying to load checkpoint in HF safetensors format, "
                "but sd_adapter is not provided."
            )

            hf_state_dict = self.sd_adapter.to_hf(state_dict)
            hf_storage_reader = self.sd_adapter.get_hf_storage_reader(
                checkpoint_id, from_quantized
            )

            dcp.load(hf_state_dict, storage_reader=hf_storage_reader)

            state_dict = self.sd_adapter.from_hf(hf_state_dict)
            states[MODEL].load_state_dict(state_dict)
        else:
            dcp.load(state_dict, checkpoint_id=checkpoint_id)

            # TODO: Since we flatten the model states in state_dict, we need to
            # manually call load_state_dict() for the model. Need to fix this.
            if MODEL in states:
                states[MODEL].load_state_dict(state_dict)

    def _save(self, curr_step: int, last_step: bool = False) -> bool:
        """Save the checkpoint for the current step.

        This function manages the checkpointing lifecycle for the current step.
        A save is performed if any of the following conditions are met:
        1. It is the initial seed checkpoint (step 0).
        2. The current step matches the configured saving interval.
        3. `last_step` is True, which forces a save regardless of the interval.
           This typically happens when the training reaches its final step.

        Args:
            curr_step (int): The current training step.
            last_step (bool, optional): Whether this is the final step of training.

        Returns:
            bool: True if a checkpoint was written (or staged, for async modes) on
            this step.
        """

        if not self._should_save(curr_step, last_step):
            return False

        sl.add_step_tag("checkpoint_save")

        self.maybe_wait_for_saving()
        self._purge_stale_checkpoints(saving_step=curr_step)

        begin = time.monotonic()
        checkpoint_phase = (
            "saving" if self.async_mode == AsyncMode.DISABLED else "staging"
        )
        logger.info(f"{checkpoint_phase.capitalize()} the checkpoint.")

        if last_step:
            self._save_last_step(curr_step)
            logger.info(
                f"Last step checkpoint completed in {time.monotonic() - begin:.2f}s"
            )
            return True

        checkpoint_id = self._create_checkpoint_id(curr_step)
        states = self._flattened_model_states_sd()
        async_save_started_at: float | None = None

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            if self.stager is None:
                self.stager = DefaultStager(
                    StagingOptions(
                        use_pinned_memory=True,
                        use_shared_memory=True,
                        use_async_staging=True,
                        use_non_blocking_copy=True,
                    )
                )

            async_save_started_at = time.monotonic()
            result = self.dcp_save(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=self.async_mode,
            )
            # Calling GC here is not required for this path.

            assert isinstance(result, AsyncSaveResponse)
            self.staging_future = result.staging_completion
            self.save_future = result.upload_completion

        elif self.async_mode == AsyncMode.ASYNC:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            async_save_started_at = time.monotonic()
            result = self.dcp_save(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=self.async_mode,
            )
            GarbageCollection.collect("GC collection invoked by checkpointer.")

            assert isinstance(result, Future)
            self.save_future = result

        else:
            self.dcp_save(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=AsyncMode.DISABLED,
                enable_garbage_collection=True,
            )

        if async_save_started_at is not None:
            assert self.save_future is not None
            self.save_future.add_done_callback(
                lambda _: sl.log_trace_scalar(
                    {
                        "train.checkpoint_write.native_dcp.execute.async_total.latency_ms": (
                            time.monotonic() - async_save_started_at
                        )
                        * 1000
                    }
                )
            )
        logger.info(
            f"Finished {checkpoint_phase} the checkpoint in "
            f"{time.monotonic() - begin:.2f} seconds."
        )
        return True

    def _maybe_wait_for_staging(self) -> None:
        """Wait for the staging process to complete if it is active.

        In `ASYNC_WITH_PINNED_MEM` mode, the checkpoint data is first staged from
        device (GPU) memory to pinned host (CPU) memory. This staging process is
        asynchronous and designed to overlap with the subsequent training
        computation (forward/backward passes).

        This method ensures that the staging process has finished before the next
        checkpoint cycle begins or before training completes, preventing memory
        contention or race conditions in the pinned memory buffers.

        Raises:
            RuntimeError: If a staging future is detected while asynchronous mode
                isn't ASYNC_WITH_PINNED_MEM.
        """

        if self.staging_future is None:
            return

        if self.async_mode != AsyncMode.ASYNC_WITH_PINNED_MEM:
            raise RuntimeError(
                "self.staging_future is not None, "
                "but self.async_mode isn't ASYNC_WITH_PINNED_MEM."
            )

        self.staging_future.result()
        self.staging_future = None

    def _wait_for_saving(self) -> None:
        """Wait for any async background checkpoint saving operation to complete.

        This is a blocking call that ensures all checkpoint data has been fully
        saved to storage. Upon completion, the tracking future is cleared
        to signify that no background save operations are currently active.

        Raises:
            RuntimeError: If a save future is detected while asynchronous mode
                is DISABLED.
        """

        if self.async_mode == AsyncMode.DISABLED:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is DISABLED."
            )

        # Clear before awaiting so a failed save is not retried by a later
        # close() or __del__ call.
        save_future = self.save_future
        assert save_future is not None
        self.save_future = None
        save_future.result()

    def _is_resumable_checkpoint(self, checkpoint_dir: str) -> bool:
        return self._storage.isfile(filesystem.join(checkpoint_dir, ".metadata"))

    def _is_valid_checkpoint(self, checkpoint_dir: str) -> bool:
        return self._is_resumable_checkpoint(checkpoint_dir) or (
            self._storage.isfile(
                filesystem.join(checkpoint_dir, "model.safetensors.index.json")
            )
        )

    def _flattened_model_states_sd(
        self, state_dict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Extract and flatten model parameters into a single state dictionary.

        This method merges the internal state of the model object into the top-level
        dictionary while keeping auxiliary states (such as optimizers or lr_schedulers)
        unflattened. This ensures a consistent format for the DCP writer.

        Args:
            state_dict (dict[str, Any], optional): A custom dictionary to flatten.
                Defaults to None (uses the instance's internal states).

        Returns:
            dict[str, Any]: A unified dictionary containing both flattened model
                parameters and top-level auxiliary states.
        """
        states = state_dict if state_dict is not None else self.states
        sd = {k: v for k, v in states.items() if k != MODEL}
        if MODEL in states:
            sd.update(states[MODEL].state_dict())
        return sd

    def _save_last_step(self, curr_step: int) -> None:
        """Execute the final checkpoint save at the completion of training.

        This method handles the specific requirements for the final training
        artifact. It allows for saving model weights exclusively (stripping
        optimizer states), performing data type conversion for export, and
        optionally formatting the output for HuggingFace compatibility.

        Args:
            curr_step (int): The final training step index.
        """

        # If `last_save_model_only` is False, we save the full training state
        # without dtype conversion to ensure training can be resumed safely.
        # Otherwise, we assume training is fully complete and save only the model
        # with dtype conversion if the current dtype isn't equal to the export dtype.

        if self.last_save_in_hf:
            assert (
                self.last_save_model_only
            ), "Only model can be saved when saving in HF safetensors format."

        if self.last_save_model_only:
            states = self.states[MODEL].state_dict()

            states = {
                k: v.to(self.export_dtype)
                if isinstance(v, torch.Tensor)
                and v.is_floating_point()
                and v.dtype != self.export_dtype
                else v
                for k, v in states.items()
            }
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            states = self._flattened_model_states_sd()

        self.dcp_save(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf,
        )
