# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import enum
import queue
import re
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, cast, Literal
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.storage import StorageWriter
from torch.distributed.tensor import DTensor
from torchtitan.components import fs
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection

MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``a`` and ``b`` are backed by the same storage.

    For ``DTensor`` the local shard's storage is compared via ``_local_tensor``
    rather than ``to_local()``, which is autograd-aware; this is a read-only
    identity check on the local storage.
    """
    if isinstance(a, DTensor):
        a = a._local_tensor
    if isinstance(b, DTensor):
        b = b._local_tensor
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


class ModelWrapper(Stateful):
    """
    A wrapper for `nn.Module` (or a list of modules) that provides a unified `Stateful`
    interface for distributed checkpointing.

    This class serves two purposes:
        1. Flattening/Aggregation: It combines the state dicts of multiple
           different modules (like individual chunks in Pipeline Parallelism)
           into a single flat view so checkpointing code can interact
           with them through a unified interface.
        2. Stable-storage caching: It caches the flattened state dict and, on
           every `state_dict()` call, returns tensors backed by the same
           storage. Async DCP staging may cache pinned host buffers keyed by the
           source storage, so keeping the storage stable lets it reuse those
           buffers across saves (the fast checkpoint path). Parameter tensors
           already satisfy this because the cached view shares the parameter
           storage; tensors produced by module `state_dict` hooks (e.g. one that
           splits a fused parameter) may be freshly allocated each call, so they
           are refreshed in place to keep their storage stable while their values
           track the current parameters.

    Notes:
        - Calling `load_state_dict` updates the underlying modules and
        refreshes the cached state_dict.
        - The model architecture should not be structurally modified (e.g.,
        changing keys or replacing tensor references) after wrapping, or the
        cache will become stale.
    """

    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cached_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        # TorchTitan already makes model state_dict keys canonical.
        return {k: v for model in self.model for k, v in model.state_dict().items()}

    def state_dict(self) -> dict[str, Any]:
        # Recompute the state dict so hook-produced tensors reflect the current
        # parameters, then merge into the cache without changing storage objects.
        for key, value in self._get_state_dict().items():
            cached = self.cached_state_dict.get(key)
            if (
                cached is None
                or cached.shape != value.shape
                or cached.dtype != value.dtype
            ):
                self.cached_state_dict[key] = value
            elif not _shares_storage(cached, value):
                cached.copy_(value)
        return self.cached_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # strict=False because state_dict is the flattened checkpoint dict, which
        # mixes model FQN keys with non-model keys (optimizer, lr_scheduler, ...).
        for model in self.model:
            model.load_state_dict(state_dict, strict=False)
        # Refresh the cache so state_dict() reflects the freshly loaded values.
        self.cached_state_dict = self._get_state_dict()


class Terminate:
    pass


class SaveDone:
    pass


def purge_thread(purge_queue: queue.Queue):
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and
        Terminate signal.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            assert isinstance(path, str)
            logger.info("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            fs.rm(path, recursive=True)
            logger.info(
                "Checkpointer deleted %s in %.2f seconds.",
                path,
                time.monotonic() - begin,
            )
    finally:
        logger.info("Destroying the purge thread.")


class CheckpointManager(Configurable):
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
        flat dicts using the utilities in torchtitan/components/checkpoint_utils.py.

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
    class Config(Configurable.Config):
        enable: bool = False
        """Whether to enable checkpoint"""

        folder: str = "checkpoint"
        """
        The folder to store the checkpoints.
        When enable is set to true, checkpoints will be in
        {--dump_folder}/{--checkpoint.folder}.
        """

        interval: int = 500
        """Checkpointing interval in steps."""

        initial_load_path: str | None = None
        """
        This option specifies the path to the initial checkpoint to load, which is
        particularly useful for resuming training from a previous run with a
        different output path or when loading a checkpoint from a pre-trained model.
        If the checkpoint folder for the current run is not empty,
        located at {--dump_folder}/{--checkpoint.folder}, this option will be ignored.
        This feature allows users to load an initial checkpoint from a different folder
        and continue training, saving new checkpoints to the specified folder without
        affecting the existing ones.

        Note that the path should contain the absolute path to the checkpoint folder,
        including the step number, if any; for example,
        "//pre_train/checkpoints/llama3/llama3_8b/step_10000".
        """

        initial_load_model_only: bool = True
        """
        This option specifies if only the model should be loaded during the initial
        checkpoint load. The option is only used when `initial_load_path` is specified.
        If False, the checkpoint at `initial_load_path` is treated as a standard
        training checkpoint, including optimizer, lr scheduler, training states, etc.
        The default setting for this option is True. Note that you will have to use
        `--checkpoint.no_initial_load_model_only` to override the default setting.
        """

        initial_load_in_hf: bool = False
        """
        Enable the use of HuggingFace's safetensors format for checkpointing. This will
        load checkpoints in HF's model definition and safetensors format instead of the
        default torchtitan model definition and DCP format, after necessary model state
        dict transformation.
        If `initial_load_path` is not provided, this option will look for weights
        in `sd_adapter.hf_assets_path`. `initial_load_model_only` must be True
        because safetensors doesn't support saving non-tensors.
        The default value is False.
        """

        initial_load_in_hf_quantized: bool = False
        """
        Enable loading of HuggingFace's safetensors format with quantized state dict
        keys. The option is only used when `initial_load_path` and
        `initial_load_path_in_hf` is specified. This will load checkpoints in HF's model
        definition and dequantize on model weights if necessary. To support this
        parameter, the model need to define proper HuggingFaceStorageReader to perform
        dequantize.
        """

        last_save_model_only: bool = True
        """
        When last_save_model_only=True, only the model will be saved at the end of
        training, the last save. With this, checkpoints can be loaded using
        `torch.load(..., weights_only=True)` after conversion. When
        last_save_model_only=False, the full checkpoint will be saved. A full
        checkpoint includes model, optimizer and train_state, which can be used to
        resume training. The default value is True.
        """

        last_save_in_hf: bool = False
        """
        Enable the use of Hugging Face's safetensors format for checkpointing. This will
        save the final checkpoints in safetensors format instead of the default DCP
        format, after necessary model state dict transformation. There will be a
        performance cost in using this as we need to consolidate the sharded tensors to
        full tensors as a separate step. last_save_model_only must be true because
        safetensors doesn't support saving non-tensors. On load, this argument isn't
        needed as we will detect whether the loaded checkpoint is in safetensors format
        or not. The default value is False.
        """

        export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
        """
        Converts to the specified precision when training completes and
        last_save_model_only=true.
        """

        async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
        """
        Which async checkpoint mode to use. Currently there are 3 different modes.

        - "disabled": Synchronized checkpointing. The training loop is blocked until all
        data is successfully saved to the persistence storage device (disk).

        - "async": Uses threading and `torch.distributed.checkpoint.async_save`.
        The training loop is blocked only during the GPU-to-CPU memory transfer. Once
        data reaches host RAM, training resumes while a background thread manages the
        final write to disk. This reduces idle time but remains subject to GIL
        contention.

        - "async_with_pinned_mem": Uses a separate process and pre-allocated pinned
        shared memory.
        The training loop resumes almost immediately by overlapping the GPU-to-CPU DMA
        transfer with the next iteration's computation. The process then persists the
        data from pinned shared memory to disk.
        This eliminates GIL contention and minimizes the blocking window to near-zero
        (< 1s), at the cost of significantly higher fixed CPU RAM usage (pinned memory).
        If case of insufficient CPU memory, performance may degrade due to memory
        paging.

        "disabled" is the default mode.
        """

        keep_latest_k: int = 10
        """
        Keeps only the latest k checkpoints, and purging older ones. If 0, keep all
        checkpoints. K cannot be 1 as the last one may be in the process of being
        saved. As a result, the metadata of the last one may not be ready yet. The
        default value is 10 to avoid filling up the disk.
        """

        load_step: int = -1
        """Load the checkpoint at the specified step. If -1, load the latest
        checkpoint."""

        exclude_from_loading: list[str] = field(default_factory=list)
        """
        Exclude specific keys from being loaded from the checkpoint.
        Provide a comma-separated list of keys to exclude,
        e.g. 'optimizer,lr_scheduler,dataloader'.
        Keys shouldn't include 'model' key.
        """

        enable_first_step_checkpoint: bool = False
        """
        Enable the checkpoint save at first step. This will save a checkpoint
        immediately after the first step to ensure checkpointing functions correctly.
        This is useful when running on a new cluster or storage to verify checkpointing
        without waiting for many steps or checkpointing too frequently. The default
        value is False.
        """

        create_seed_checkpoint: bool = False
        """
        Initializes the full model without applying parallelisms, and then saves it as a
        seed checkpoint. Note: requires user to call train.py without specifying any
        parallelisms, e.g. NGPU=1. Could be implemented as a separate script, but this
        way shares more code.
        """

        load_only: bool = False
        """
        In certain scenarios, you may only need to load checkpoints for verification or
        debugging purposes, without saving any new checkpoints. For example, you might
        use seed checkpoints to validate model correctness. Enabling this option allows
        checkpoints to be loaded without saving any during the training.
        """

        def __post_init__(self):
            if not self.folder.strip():
                raise ValueError("The 'folder' field cannot be empty.")
            if self.interval < 1:
                raise ValueError("Checkpoint interval needs to be at least 1 step.")
            if self.keep_latest_k < 0:
                raise ValueError("keep_latest_k cannot be negative.")
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            if MODEL in self.exclude_from_loading:
                raise ValueError(f"{MODEL} key shouldn't be in exclude_from_loading.")

            if self.initial_load_path:
                self.initial_load_path = self.initial_load_path.strip()
                parsed_initial_load_path = urlparse(self.initial_load_path)
                if not (
                    self.initial_load_path.startswith("/")
                    or parsed_initial_load_path.scheme
                ):
                    raise ValueError(
                        "initial_load_path must be absolute or a valid fsspec URL: "
                        f"{self.initial_load_path}"
                    )
            if self.initial_load_in_hf and not self.initial_load_model_only:
                raise ValueError("initial_load_in_hf requires initial_load_model_only.")
            if self.initial_load_in_hf_quantized and not (
                self.initial_load_in_hf and self.initial_load_path
            ):
                raise ValueError(
                    "initial_load_in_hf_quantized requires initial_load_in_hf "
                    "and initial_load_path."
                )
            if self.last_save_in_hf and not self.last_save_model_only:
                raise ValueError("last_save_in_hf requires last_save_model_only=True.")

            async_lowered = self.async_mode.lower()
            if async_lowered in ("disabled", "async", "async_with_pinned_mem"):
                self.async_mode = async_lowered
            else:
                raise ValueError(f"Invalid async_mode: {async_lowered}")

            if self.load_only and self.enable_first_step_checkpoint:
                logger.warning(
                    "checkpoint.load_only is True; enable_first_step_checkpoint "
                    "will be ignored."
                )
            if self.initial_load_model_only and not self.initial_load_path:
                logger.warning(
                    "initial_load_model_only=True has no effect without "
                    "an initial_load_path."
                )

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

        self.folder = fs.join_path(base_folder, config.folder)
        self.interval = config.interval

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
        self.purge_thread: threading.Thread | None = None
        if self.keep_latest_k > 0:
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()

        logger.info(
            "Checkpointing active. Checkpoints will be loaded from and saved "
            f"to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "enable") and self.enable:
            if (
                hasattr(self, "purge_thread")
                and self.purge_thread
                and self.purge_thread.is_alive()
            ):
                self.purge_queue.put(Terminate())
                self.purge_thread.join()

            if self.stager is not None:
                self.stager.close()

    @torch.no_grad()
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

        storage_writer: StorageWriter | None = None
        fqn_to_index_mapping: dict[Any, int] | None = None

        # HF Format Conversion
        if to_hf:
            assert self.sd_adapter is not None, "sd_adapter is required for to_hf=True"
            state_dict = self.sd_adapter.to_hf(state_dict)
            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping

            # If sharded, we save to a subdir then consolidate
            save_path = (
                fs.join_path(checkpoint_id, "sharded")
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
        else:
            storage_writer = FsspecWriter(checkpoint_id)

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
                input_dir=fs.join_path(checkpoint_id, "sharded"),
                output_dir=checkpoint_id,
                fqn_to_index_mapping=fqn_to_index_mapping,
                num_threads=5,
            )

        if enable_garbage_collection:
            GarbageCollection.collect("GC collection invoked by checkpointer.")

        return ret

    def dcp_load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        """Load a DCP into the provided state dictionary.

        This method handles both standard DCP sharded checkpoints and HuggingFace
        safetensors. If loading from HF, it utilizes an adapter to map FQNs and
        handle format-specific sharding logic.

        Args:
            state_dict (dict): The target dictionary to populate with checkpoint data.
            checkpoint_id (str): Path or identifier for the source checkpoint.
            from_hf (bool): If True, adapts the load process for HuggingFace model
                definitions and safetensors format.
            from_quantized (bool): Indicates if the source is in a quantized format
                (e.g., 4-bit/8-bit), requiring the storage reader to handle
                specialized data types and sharding structures.

        Raises:
            AssertionError: If `from_hf` is True but no `sd_adapter` is available.
        """

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
            self.states[MODEL].load_state_dict(state_dict)
        else:
            dcp.load(
                state_dict,
                storage_reader=FsspecReader(checkpoint_id),
                checkpoint_id=checkpoint_id,
            )

            # TODO: Since we flatten the model states in state_dict, we need to
            # manually call load_state_dict() for the model. Need to fix this.
            if MODEL in self.states:
                self.states[MODEL].load_state_dict(state_dict)

    @sl.log_trace_span("checkpoint_save")
    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> bool:
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

        self._purge_stale_checkpoints()

        logger.info(
            f"Finished {checkpoint_phase} the checkpoint in "
            f"{time.monotonic() - begin:.2f} seconds."
        )
        return True

    @sl.log_trace_span("checkpoint_load")
    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load the checkpoint for the given step.

        This function orchestrates the states loading process.
        If the local checkpoint folder does not yet exist, it attempts an initial load
        from a specified path (in either native or HF format) or performs loading using
        provided HF assets path from the state dict adapter. Otherwise, it retrieves
        the checkpoint corresponding to the specified step, defaulting to the latest
        available if the `step` is -1.

        Args:
            step (int, optional): The training step to restore.
                Defaults to -1 (latest available).

        Returns:
            bool: Whether the checkpoint was successfully located and loaded.
        """

        if not self.enable:
            return False

        model_only = False
        from_hf = False
        from_quantized = False

        if not self._checkpoint_folder_exists():
            model_only = self.initial_load_model_only
            from_hf = self.initial_load_in_hf
            from_quantized = self.initial_load_in_hf_quantized

            if from_hf:
                assert model_only, (
                    "Only model can be loaded when loading from "
                    "HF's safetensors checkpoint."
                )
            if from_quantized:
                assert from_hf, "Quantized checkpoint can only be loaded from HF format"

            if self.initial_load_path:
                checkpoint_id = self.initial_load_path
                if not self._checkpoint_exists(checkpoint_id, from_hf=from_hf):
                    raise ValueError(
                        f"Checkpoint.initial_load_path is invalid: {checkpoint_id}"
                    )
                if from_hf:
                    logger.info(
                        "Loading from HF safetensors from "
                        f"--checkpoint.initial_load_path: {checkpoint_id}"
                    )

            elif from_hf:
                assert (
                    self.sd_adapter and self.sd_adapter.hf_assets_path
                ), "from_hf=True requires sd_adapter and hf_assets_path."
                checkpoint_id = self.sd_adapter.hf_assets_path
                if not fs.isdir(checkpoint_id):
                    raise ValueError(
                        "model.hf_assets_path is being used to load HF weights "
                        "but the path is not valid. Either make sure hf_assets_path is "
                        "correct or provide a valid checkpoint.initial_load_path"
                    )
                logger.info(
                    "Loading HF safetensors from "
                    f"--model.hf_assets_path: {checkpoint_id}"
                )

            else:
                return False

        else:
            if self.initial_load_path:
                logger.warning(
                    "checkpoint.initial_load_path is provided but the "
                    "checkpoint.folder exists. Checkpointer will use the checkpoints "
                    f"from the checkpoint.folder {self.folder}."
                )
            if self.initial_load_in_hf:
                logger.warning(
                    "checkpoint.initial_load_in_hf is True but the checkpoint.folder "
                    "exists. Checkpointer will not load from HF safetensors"
                )

            step = self._find_load_step() if step == -1 else step
            if step == -1:
                return False

            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)

            if not self._checkpoint_exists(checkpoint_id, from_hf=False):
                raise FileNotFoundError(
                    f"--checkpoint.load_step={step} not found at {checkpoint_id}"
                )

        logger.info(f"Loading the checkpoint from {checkpoint_id}.")
        begin = time.monotonic()

        states = self._states_to_load(model_only)
        self.dcp_load(
            states,
            checkpoint_id=checkpoint_id,
            from_hf=from_hf,
            from_quantized=from_quantized,
        )

        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            "Finished loading the checkpoint in "
            f"{time.monotonic() - begin:.2f} seconds."
        )

        return True

    def maybe_wait_for_staging(self) -> None:
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

        if not self.enable or self.staging_future is None:
            return

        if self.async_mode != AsyncMode.ASYNC_WITH_PINNED_MEM:
            raise RuntimeError(
                "self.staging_future is not None, "
                "but self.async_mode isn't ASYNC_WITH_PINNED_MEM."
            )

        self.staging_future.result()
        self.staging_future = None

    def maybe_wait_for_saving(self) -> None:
        """Wait for any async background checkpoint saving operation to complete.

        This is a blocking call that ensures all checkpoint data has been fully
        saved to storage. Upon completion, the tracking future is cleared
        to signify that no background save operations are currently active.

        Raises:
            RuntimeError: If a save future is detected while asynchronous mode
                is DISABLED.
        """

        if not self.enable or self.save_future is None:
            return

        if self.async_mode == AsyncMode.DISABLED:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is DISABLED."
            )

        self.save_future.result()
        self.save_future = None

    def _find_load_step(self, folder: str = "") -> int:
        """Identify the highest available checkpoint step in the specified directory.

        This method scans the target folder for subdirectories matching the
        'step-N' pattern. A folder is only considered a valid checkpoint if
        it contains either a DCP metadata file or a HuggingFace safetensors
        index.

        Args:
            folder (str, optional): The directory to scan. Defaults to `self.folder`.

        Returns:
            int: The maximum step number found among valid checkpoints,
                or -1 if no valid checkpoints are detected.
        """

        folder = folder or self.folder
        valid_steps = self._list_checkpoint_steps(folder, require_metadata=True)
        return max(valid_steps) if valid_steps else -1

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        """Generate the standardized filesystem path for a checkpoint
        (e.g., 'checkpoints/step-100')."""
        folder = folder or self.folder
        return fs.join_path(folder, f"step-{step}")

    def _checkpoint_folder_exists(self, folder: str = "") -> bool:
        folder = folder or self.folder
        return fs.isdir(folder)

    def _checkpoint_exists(self, checkpoint_id: str, from_hf: bool) -> bool:
        metadata_names = ["model.safetensors.index.json"] if from_hf else [".metadata"]
        return fs.exists(checkpoint_id) or any(
            fs.exists(fs.join_path(checkpoint_id, metadata_name))
            for metadata_name in metadata_names
        )

    def _list_checkpoint_steps(
        self, folder: str, *, require_metadata: bool
    ) -> list[int]:
        if not fs.isdir(folder):
            return []

        valid_steps = []
        for path in fs.ls(folder):
            name = fs.basename(path)
            match = re.fullmatch(r"step-(\d+)", name)
            if not match:
                continue

            if not require_metadata or any(
                fs.exists(fs.join_path(folder, name, metadata_name))
                for metadata_name in (".metadata", "model.safetensors.index.json")
            ):
                valid_steps.append(int(match.group(1)))

        return valid_steps

    def _discover_checkpoints(self, folder: str = "") -> list[tuple[int, str]]:
        folder = folder or self.folder
        return [
            (step, fs.join_path(folder, f"step-{step}"))
            for step in self._list_checkpoint_steps(folder, require_metadata=False)
        ]

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

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        """Determine which state objects should be restored during loading.

        This method filters the checkpointer's state dictionary based on the
        loading context. It supports partial restoration for specific steps
        (e.g., loading only model weights for step 0) and respects explicit
        exclusion rules for auxiliary states.

        Args:
            model_only (bool): If True, returns only the model's parameters,
                bypassing optimizers and other training metadata.

        Returns:
            dict[str, Any]: A prepared dictionary of states to be passed to
                the loader.
        """
        # For the first step, we will only load the model.
        if model_only:
            return self.states[MODEL].state_dict()

        for exclude_key in self.exclude_from_loading:
            if exclude_key not in self.states:
                raise ValueError(f"{exclude_key} not found in state_dict.")

        states_to_load = {
            k: v for k, v in self.states.items() if k not in self.exclude_from_loading
        }

        return self._flattened_model_states_sd(states_to_load)

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

            if self.export_dtype != torch.float32:
                states = {k: v.to(self.export_dtype) for k, v in states.items()}
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

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        """Determine whether a checkpoint should be saved based on
        the current step, interval, and training status."""

        if not self.enable or self.load_only:
            return False

        if curr_step == 1 and self.enable_first_step_checkpoint:
            return True

        if last_step:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    def _should_purge(self) -> bool:
        """Whether this rank should purge stale checkpoints.

        Extracted so subclasses (e.g. TorchFTCheckpointManager) can add
        additional guards (like participating_rank) without duplicating
        the purge loop in _purge_stale_checkpoints.
        """
        return (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and self._checkpoint_folder_exists()
        )

    def _purge_stale_checkpoints(self):
        """Remove older checkpoint directories from storage to maintain
        only the most recent 'k' copies."""
        if self._should_purge():
            discovered_checkpoints = self._discover_checkpoints()
            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                assert self.purge_thread is not None
                self.purge_queue.put(path)
