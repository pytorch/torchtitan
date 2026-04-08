# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import enum
import functools
import os
import queue
import re
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, cast, Literal, NamedTuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torch.distributed.checkpoint.stateful import Stateful
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.state_dict_transforms import StateDictTransforms
from torchtitan.config import Configurable
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection


MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"

_STEP_DIR_RE = re.compile(r"step-(\d+)")


class _CheckpointLoadSpec(NamedTuple):
    """Describes which checkpoint to load and how."""

    checkpoint_id: str
    model_only: bool
    from_hf: bool
    from_quantized: bool


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class ModelWrapper(Stateful):
    """Wraps model parts into a :class:`Stateful` for checkpoint integration.

    Args:
        model: A single model or list of model parts (e.g. pipeline stages).
        key_filter: Optional callable that returns True for converter-owned
            keys. Used by ``state_dict(mode="base")`` to exclude these
            keys when creating HF containers.
        state_dict_transform: An optional pure function that transforms the
            model state dict for export saves (last-step / model-only).
            For example, a converter may filter or reshape the state dict.
            Applied only for ``state_dict(mode="export")``.
    """

    def __init__(
        self,
        model: nn.Module | list[nn.Module],
        *,
        key_filter: Callable[[str], bool] | None = None,
        state_dict_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self._key_filter = key_filter
        self._state_dict_transform = state_dict_transform

    def _get_state_dict(self) -> dict[str, Any]:
        state_dict = {
            k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()
        }
        return state_dict

    _VALID_MODES = frozenset({"full", "base", "export"})

    def state_dict(self, mode: str = "full") -> dict[str, Any]:
        """Return the model state dict in the requested mode.

        Note: We intentionally do not cache the state dict.
        ``set_model_state_dict()`` mutates keys of the input state dict,
        so a cached copy would go stale after ``load_state_dict()``.

        Args:
            mode: One of:
                - ``"full"``: Complete state dict for interval saves and resume.
                - ``"base"``: Base model keys only (excludes converter-owned
                  keys identified by ``key_filter``). Used for HF container
                  creation during primary HF load.
                - ``"export"``: Apply ``state_dict_transform`` for last-step
                  export saves (e.g. QAT dequant, converter filtering).
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Invalid state_dict mode {mode!r}, expected one of {sorted(self._VALID_MODES)}"
            )
        sd = self._get_state_dict()
        if mode == "base" and self._key_filter is not None:
            return {k: v for k, v in sd.items() if not self._key_filter(k)}
        elif mode == "export" and self._state_dict_transform is not None:
            return self._state_dict_transform(sd)
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class Terminate:
    pass


class SaveDone:
    pass


def purge_thread(purge_queue: queue.Queue):
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive the path to purge and Terminate signal.
    """
    try:
        while True:
            path = purge_queue.get()
            if isinstance(path, Terminate):
                return
            assert isinstance(path, str)
            logger.info("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            shutil.rmtree(path, ignore_errors=True)
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
    rank0's optimizer would have a param_group[0] which refers to layers.0 in the original
    model.  rank1's would _also_ have a param_group[0], since it's index based, but
    referring to layers.1.  When saving, these collide and one of them is lost.  Then when
    reloading, only one stage can restore its optimizer states, others will error.

        The solution to this problem is optimizer flattening: it landed in #127071 and is
        enabled in TorchTitan by passing the 'flatten_optimizer_state_dict' kwarg to DCP
        functions called in the OptimizerContainer.
        See PR #127071 (https://github.com/pytorch/pytorch/pull/127071) for the example of
        a flattening state_dict.

    2. With complex PP schedules, we have multiple model chunks per pp rank. This compounds
    challenge (1) by also requiring us to reason about multiple 'optim' objects locally.

        We solve this in the Model and Optimizer wrapper classes by flattening the state dicts
        from each object into one state dict before saving/loading. We rely on the individual
        state_dicts to not collide, which is guaranteed for the model by correct pipeline
        splitting and for the optimizer by the flattening support described in (1).

    3. LR schedulers also index model states like optimizers. Here we flatten the lr_schedulers
    with the assumption that all lr_schedulers have the same state_dict.

    Args:
        dataloader (DataLoader): The dataloader used to load the data.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizers (OptimizersContainer): The optimizers used to optimize the model.
        lr_schedulers (LRSchedulersContainer): The lr schedulers used to optimize the model.
        states (Dict[str, Any]): The states that need to be saved, other than the
            previous 4 components.
        config (Checkpoint): The config used to configure the checkpointing.
        base_folder (str): The base folder to save the checkpoint. Will be concatenated
            with config.folder
        sd_transforms (StateDictTransforms): Owns all state dict content
            transforms (dtype conversion, HF format, etc.).

    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable: bool = False
        """Whether to enable checkpoint"""

        folder: str = "checkpoint"
        """
        The folder to store the checkpoints.
        When enable is set to true, checkpoints will be in {--dump_folder}/{--checkpoint.folder}.
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
        This feature allows users to load an initial checkpoint from a different folder and
        continue training, saving new checkpoints to the specified folder without affecting
        the existing ones.

        Note that the path should contain the full path to the checkpoint folder,
        including the step number, if any; for example,
        "//pre_train/checkpoints/llama3/llama3_8b/step_10000".
        """

        initial_load_model_only: bool = True
        """
        This option specifies if only the model should be loaded during the initial
        checkpoint load. The option is only used when `initial_load_path` is specified.
        If False, the checkpoint at `initial_load_path` is treated as a standard training
        checkpoint, including optimizer, lr scheduler, training states, etc.
        The default setting for this option is True. Note that you will have to use
        `--checkpoint.no_initial_load_model_only` to override the default setting.
        """

        initial_load_in_hf: bool = False
        """
        Enable the use of HuggingFace's safetensors format for checkpointing. The option
        is only used when `initial_load_path` is specified. This will load checkpoints
        in HF's model definition and safetensors format instead of the default torchtitan
        model definition and DCP format, after necessary model state dict transformation.
        `initial_load_model_only` must be true because safetensors doesn't support saving
        non-tensors. The default value is False.
        """

        initial_load_in_hf_quantized: bool = False
        """
        Enable loading of HuggingFace's safetensors format with quantized state dict keys. The option
        is only used when `initial_load_path` and `initial_load_path_in_hf` is specified. This will load
        checkpoints in HF's model definition and dequantize on model weights if necessary. To support
        this parameter, the model need to define proper HuggingFaceStorageReader to perform dequantize.
        """

        last_save_model_only: bool = True
        """
        When last_save_model_only=True, only the model will be saved at the end of training,
        the last save.  With this, checkpoints can be loaded using `torch.load(..., weights_only=True)`
        after conversion.  When last_save_model_only=False, the full checkpoint will be saved.
        A full checkpoint includes model, optimizer and train_state, which can be used to resume training.
        The default value is True.
        """

        last_save_in_hf: bool = False
        """
        Enable the use of Hugging Face's safetensors format for checkpointing. This will save the
        final checkpoints in safetensors format instead of the default DCP format, after necessary
        model state dict transformation. There will be a performance cost in using this as we need
        to consolidate the sharded tensors to full tensors as a separate step.
        last_save_model_only must be true because safetensors doesn't support saving
        non-tensors. On load, this argument isn't needed as we will detect whether the loaded
        checkpoint is in safetensors format or not. The default value is False.
        """

        export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
        """
        Converts to the specified precision when training completes and last_save_model_only=true.
        """

        async_mode: Literal["disabled", "async", "async_with_pinned_mem"] = "disabled"
        """
        Which async checkpoint mode to use. Currently there are 3 different modes.

        - "disabled": synchronized checkpointing will be used.
        - "async": torch.distributed.checkpoint.async_save will be used.
        - "async_with_pinned_mem": this option utilizes a dedicated pinned memory space and creates a
          separate process for faster GPU->CPU transfer performance and eliminating GIL contention.
          The cost is increased CPU memory usage. If insufficient CPU memory is available, performance
          may degrade due to memory paging. For most users, "async" should suffice as the performance
          overhead is typically small (on the order of tens of seconds) compared to checkpointing
          frequency. This mode can be employed to pursue near-zero checkpointing times
          (e.g., < 1 second) given appropriate hardware support such as ample CPU memory and fast PCIe.

        "disabled" is the default mode.
        """

        keep_latest_k: int = 10
        """
        Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints.
        K cannot be 1 as the last one may be in the process of being saved. As a result,
        the metadata of the last one may not be ready yet. The default value is 10 to avoid
        filling up the disk.
        """

        load_step: int = -1
        """Load the checkpoint at the specified step. If -1, load the latest checkpoint."""

        exclude_from_loading: list[str] = field(default_factory=list)
        """
        Exclude specific keys from being loaded from the checkpoint.
        Provide a comma-separated list of keys to exclude, e.g. 'optimizer,lr_scheduler,dataloader'.
        This will load the model only, excluding the specified keys.
        """

        additional_load_path: str = ""
        """
        Additional checkpoint path to load from after the primary checkpoint.
        Useful for loading converter-specific weights (e.g. LoRA adapter)
        from a separate source. Must be a valid DCP or HF safetensors checkpoint.
        When a ``converter_sd_adapter`` is provided, this path is loaded
        using the converter's format adapter (e.g. PEFT safetensors).
        """

        enable_first_step_checkpoint: bool = False
        """
        Enable the checkpoint save at first step. This will save a checkpoint immediately
        after the first step to ensure checkpointing functions correctly. This is useful
        when running on a new cluster or storage to verify checkpointing without waiting
        for many steps or checkpointing too frequently. The default value is False.
        """

        create_seed_checkpoint: bool = False
        """
        Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint.
        Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.
        Could be implemented as a separate script, but this way shares more code.
        """

        load_only: bool = False
        """
        In certain scenarios, you may only need to load checkpoints for verification or debugging
        purposes, without saving any new checkpoints. For example, you might use seed checkpoints
        to validate model correctness. Enabling this option allows checkpoints to be loaded
        without saving any during the training.
        """

    def __init__(
        self,
        config: Config,
        *,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        sd_transforms: StateDictTransforms,
        base_folder: str = "",
        key_filter: Callable[[str], bool] | None = None,
        state_dict_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        converter_sd_adapters: (
            list[tuple[BaseStateDictAdapter, Callable[[str], bool]]] | None
        ) = None,
    ) -> None:
        self.enable = config.enable
        self.load_only = config.load_only

        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(
                    model_parts,
                    key_filter=key_filter,
                    state_dict_transform=state_dict_transform,
                ),
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )
        self._converter_sd_adapters = converter_sd_adapters or []

        # Config fields — always initialized so the object has a consistent
        # shape regardless of whether checkpointing is enabled.
        self.folder = os.path.join(base_folder, config.folder)
        self.sd_transforms = sd_transforms
        self.initial_load_model_only = config.initial_load_model_only
        self.initial_load_in_hf = config.initial_load_in_hf
        self.initial_load_path = config.initial_load_path
        self.initial_load_in_hf_quantized = config.initial_load_in_hf_quantized
        self.last_save_model_only = config.last_save_model_only
        self.last_save_in_hf = config.last_save_in_hf
        self.exclude_from_loading = config.exclude_from_loading
        self.additional_load_path = config.additional_load_path
        self.interval = config.interval
        self.enable_first_step_checkpoint = config.enable_first_step_checkpoint
        self.keep_latest_k = config.keep_latest_k

        # Runtime state defaults.
        self.async_mode: AsyncMode = AsyncMode.DISABLED
        self.pg: dist.ProcessGroup | None = None
        self.enable_staging: bool = False
        self.staging: bool = False
        self.staging_future: Future | None = None
        self.save_future: Future | None = None
        self.stager: DefaultStager | None = None
        self.purge_thread: threading.Thread | None = None

        if not self.enable:
            return

        # Validation that only matters when checkpointing is active.
        if self.last_save_in_hf and sd_transforms.sd_adapter is None:
            raise ValueError(
                "checkpoint.last_save_in_hf is True, but sd_adapter is not provided."
            )

        # Async checkpoint related fields.
        self.async_mode = AsyncMode(config.async_mode.lower())
        if self.async_mode in (AsyncMode.ASYNC, AsyncMode.ASYNC_WITH_PINNED_MEM):
            self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))
        self.enable_staging = self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM

        if self.keep_latest_k > 0:
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            self.purge_queue = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread, args=(self.purge_queue,), daemon=True
            )
            self.purge_thread.start()

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if not self.enable:
            return
        if self.purge_thread is not None and self.purge_thread.is_alive():
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
        fqn_to_index_mapping: dict[Any, int] | None = None,
    ) -> Future | AsyncSaveResponse | None:
        """Save the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to save.
            checkpoint_id (str): The checkpoint id to save.
            async_mode (AsyncMode): Whether the checkpoint is async.
            enable_garbage_collection (bool): Whether to enable garbage collection after save.
            to_hf (bool): Whether to save in HF model definition and safetensors format.
            fqn_to_index_mapping (dict): Optional mapping for HF safetensors sharding.
                When provided, saves to multiple sharded files. When None,
                saves all keys to a single file.

        Returns:
            Future: The future object if the checkpoint is async, otherwise None.
        """

        ret: Future | AsyncSaveResponse | None = None

        storage_writer: HuggingFaceStorageWriter | None = None
        checkpoint_save_id: str | None = None
        if to_hf and self.sd_transforms.sd_adapter is None:
            raise ValueError(
                "Trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            )
            # The state_dict is already in HF format (content transform applied
            # by the caller).  Here we only set up the HF storage writer
            # (I/O concern).  The caller must pass the correct
            # fqn_to_index_mapping for the adapter being used.
            if fqn_to_index_mapping:
                storage_writer = HuggingFaceStorageWriter(
                    path=os.path.join(checkpoint_id, "sharded"),
                    save_distributed=True,
                    fqn_to_index_mapping=fqn_to_index_mapping,
                    enable_consolidation=False,
                )
            else:
                # the reason for only enabling consolidation if there is
                # no mapping is because no mapping implies that we save all fqns
                # to one file. This means we only need one rank to consolidate.
                # Otherwise we should use consolidate_safetensors_files_on_every_rank
                storage_writer = HuggingFaceStorageWriter(
                    path=checkpoint_id,
                    save_distributed=True,
                    enable_consolidation=True,
                )

        else:
            checkpoint_save_id = checkpoint_id

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

    def dcp_load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        """Load the checkpoint(s) with dcp.

        Args:
            state_dict (dict): The state dict to load.
            checkpoint_id (str): The primary checkpoint id to load.
            from_hf (bool): Whether to load from HuggingFace safetensors format.
            from_quantized (bool): Whether the HuggingFace checkpoint is quantized.
        """
        has_converter_keys = self.states[MODEL]._key_filter is not None
        # Primary: partial when converter keys won't be in the checkpoint
        # (e.g. LoRA keys absent from a base model checkpoint).
        primary_planner = DefaultLoadPlanner(
            allow_partial_load=has_converter_keys or bool(self.additional_load_path)
        )

        # Load primary checkpoint
        if from_hf:
            self._load_with_adapter(checkpoint_id, from_quantized, primary_planner)
        else:
            self._load_from_dcp(state_dict, checkpoint_id, primary_planner)

        # Load additional checkpoint source (e.g. LoRA adapter weights).
        # Only one additional source is supported; uses the first converter
        # adapter if available.  Always partial (subset of keys), never quantized.
        if self.additional_load_path:
            additional_planner = DefaultLoadPlanner(allow_partial_load=True)
            if self._converter_sd_adapters:
                conv_adapter, conv_kf = self._converter_sd_adapters[0]
                self._load_with_adapter(
                    self.additional_load_path,
                    False,
                    additional_planner,
                    adapter=conv_adapter,
                    key_filter=conv_kf,
                )
            elif from_hf:
                self._load_with_adapter(
                    self.additional_load_path, False, additional_planner
                )
            else:
                self._load_from_dcp(
                    state_dict, self.additional_load_path, additional_planner
                )

    def _load_from_dcp(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        planner: DefaultLoadPlanner,
    ) -> None:
        """Load a DCP checkpoint and set model state.

        Note: ``load_state_dict`` is called unconditionally on the full
        *state_dict* even when loading from a secondary source (e.g.
        ``additional_load_path``).  This is safe because the planner's
        ``allow_partial_load=True`` ensures ``dcp.load`` only overwrites
        keys present in the checkpoint — other keys retain their values
        from the primary load.
        """
        dcp.load(state_dict, checkpoint_id=checkpoint_id, planner=planner)
        # TODO: Since we flatten the model states in state_dict, we need to
        # manually call load_state_dict() for the model. Need to fix this.
        if MODEL in self.states:
            self.states[MODEL].load_state_dict(state_dict)

    def _load_with_adapter(
        self,
        checkpoint_id: str,
        from_quantized: bool,
        planner: DefaultLoadPlanner,
        adapter: BaseStateDictAdapter | None = None,
        key_filter: Callable[[str], bool] | None = None,
    ) -> None:
        """Load a safetensors checkpoint using a state dict adapter.

        Creates appropriately-shaped tensor containers via ``adapter.to_hf``,
        loads via HuggingFaceStorageReader, then reverse-transforms back to
        torchtitan FQNs via ``adapter.from_hf``.

        Args:
            adapter: The state dict adapter for FQN mapping. Defaults to the
                model's HF adapter (``sd_transforms.sd_adapter``).
            key_filter: Per-converter key filter. When provided, the load
                container is built from only the matching keys.  When
                ``None``, uses ``mode="base"`` (excludes all converter keys).
        """
        if adapter is None:
            adapter = self.sd_transforms.sd_adapter
        assert adapter is not None

        if key_filter is not None:
            sd = {
                k: v
                for k, v in self.states[MODEL].state_dict().items()
                if key_filter(k)
            }
        else:
            sd = self.states[MODEL].state_dict(mode="base")
        hf_state_dict = adapter.to_hf(sd)
        hf_storage_reader = adapter.get_hf_storage_reader(checkpoint_id, from_quantized)
        dcp.load(
            hf_state_dict,
            storage_reader=hf_storage_reader,
            planner=planner,
        )
        converted_sd = adapter.from_hf(hf_state_dict)
        if MODEL in self.states:
            self.states[MODEL].load_state_dict(converted_sd)

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> None:
        """Save the checkpoint for the current step.

        This function will save the checkpoint for the current step. If ``last_step`` is
        true, it will save the checkpoint even if the interval has not been reached.
        This only happens when train_state.step == trainer_config.training.steps, or
        for initial seed checkpoint.

        Args:
            curr_step (int): The current step.
            last_step (bool, optional): Whether this is the last step of training.

        Returns:
            None
        """
        if not self._should_save(curr_step, last_step):
            return
        self._do_save(curr_step, last_step)

    def _do_save(self, curr_step: int, last_step: bool) -> None:
        """Execute the actual checkpoint save (no guard checks)."""
        begin = time.monotonic()
        logger.info("Saving the checkpoint (or staging if async is enabled).")
        checkpoint_id = self._create_checkpoint_id(curr_step)
        self._async_wait()
        # This GC is called for async checkpoint as it is useless to do
        # GC right after async_save -- the CPU memory is not able to be
        # freed until _async_wait()
        if last_step:
            self._save_last_step(curr_step)
            return

        states = self._flattened_model_states_sd()
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            if self.stager is None:
                self.stager = DefaultStager(StagingOptions(True, True, True, True))
            result = self.dcp_save(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=self.async_mode,
            )
            assert isinstance(result, AsyncSaveResponse)
            self.save_future = result.upload_completion
            self.staging_future = result.staging_completion
            self.staging = True
        elif self.async_mode == AsyncMode.ASYNC:
            GarbageCollection.collect("GC collection invoked by checkpointer.")
            self.save_future = cast(
                Future,
                self.dcp_save(
                    states, checkpoint_id=checkpoint_id, async_mode=self.async_mode
                ),
            )
            GarbageCollection.collect("GC collection invoked by checkpointer.")
        else:
            self.dcp_save(
                states,
                checkpoint_id=checkpoint_id,
                async_mode=AsyncMode.DISABLED,
                enable_garbage_collection=True,
            )
        self._purge_stale_checkpoints()

        logger.info(
            "Finished saving the checkpoint (or staging if async is enabled) "
            f"in {time.monotonic() - begin:.2f} seconds."
        )

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load the checkpoint for the given step.

        This function will load the checkpoint for the given step. If ``step`` is -1, it
        will load the latest checkpoint. If the checkpoint does not exist, it will return
        False and load nothing.

        Args:
            step (int, optional): The step to load the checkpoint for. Defaults to -1.

        Returns:
            bool: Whether the checkpoint was loaded successfully.
        """
        if not self.enable:
            return False

        if self.additional_load_path and not os.path.isdir(self.additional_load_path):
            raise ValueError(
                f"checkpoint.additional_load_path is invalid: {self.additional_load_path}"
            )

        spec = self._resolve_checkpoint_path(step)
        if spec is None:
            return False

        logger.info(f"Loading the checkpoint from {spec.checkpoint_id}.")
        begin = time.monotonic()
        states = self._states_to_load(spec.model_only)
        self.dcp_load(
            states,
            checkpoint_id=spec.checkpoint_id,
            from_hf=spec.from_hf,
            from_quantized=spec.from_quantized,
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        return True

    def _resolve_checkpoint_path(self, step: int) -> _CheckpointLoadSpec | None:
        """Resolve which checkpoint to load.

        Returns:
            A ``_CheckpointLoadSpec`` describing the checkpoint, or ``None``
            if there is no checkpoint to load.
        """
        if not os.path.exists(self.folder):
            return self._resolve_initial_load()

        # Checkpoint folder exists — resume from it.
        if self.initial_load_path:
            logger.warning(
                "checkpoint.initial_load_path is provided but the checkpoint.folder exists. "
                f"Checkpointer will use the checkpoints from the checkpoint.folder {self.folder}."
            )
        if self.initial_load_in_hf:
            logger.warning(
                "checkpoint.initial_load_in_hf is True but the checkpoint.folder exists. "
                "Checkpointer will not load from HF safetensors"
            )

        step = self._find_load_step() if step == -1 else step
        if step == -1:
            return None
        checkpoint_id = self._create_checkpoint_id(step)

        if not os.path.isdir(checkpoint_id):
            raise FileNotFoundError(
                f"--checkpoint.load_step={step} but checkpoint {checkpoint_id} is not found."
            )
        return _CheckpointLoadSpec(
            checkpoint_id, model_only=step == 0, from_hf=False, from_quantized=False
        )

    def _resolve_initial_load(self) -> _CheckpointLoadSpec | None:
        """Resolve the checkpoint path for initial loading (no checkpoint folder yet)."""
        model_only = self.initial_load_model_only
        from_hf = self.initial_load_in_hf
        from_quantized = self.initial_load_in_hf_quantized

        if from_hf and not model_only:
            raise ValueError(
                "Only model can be loaded when loading from HF's safetensors checkpoint."
            )
        if from_quantized and not from_hf:
            raise ValueError(
                "Quantized checkpoint can only be loaded from HuggingFace format."
            )

        if self.initial_load_path:
            checkpoint_id = self.initial_load_path
            if not os.path.isdir(checkpoint_id):
                raise ValueError(
                    "checkpoint.initial_load_path is specified but the path is not valid."
                )
            if from_hf:
                logger.info(
                    f"loading from HF safetensors from --checkpoint.initial_load_path: {self.initial_load_path}"
                )
            return _CheckpointLoadSpec(
                checkpoint_id, model_only, from_hf, from_quantized
            )

        if from_hf and (
            self.sd_transforms.sd_adapter is None
            or self.sd_transforms.hf_assets_path is None
        ):
            raise ValueError(
                "from_hf is True but sd_adapter or hf_assets_path is not provided."
            )

        if from_hf:
            checkpoint_id = self.sd_transforms.hf_assets_path
            assert checkpoint_id is not None  # guarded above
            if not os.path.isdir(checkpoint_id):
                raise ValueError(
                    "model.hf_assets_path is being used to load HF weights but the path is not valid. "
                    "Either make sure hf_assets_path is correct or provide a valid checkpoint.initial_load_path"
                )
            logger.info(
                f"loading HF safetensors from --model.hf_assets_path: {checkpoint_id}"
            )
            return _CheckpointLoadSpec(
                checkpoint_id, model_only, from_hf, from_quantized
            )

        return None

    def maybe_wait_for_staging(self) -> None:
        """Wait for the staging to finish if it is enabled.

        This function will wait for staging to finish. The staging is only enabled
        with ``async_checkpoint_with_pinned_memory``.
        """
        if self.enable_staging and self.staging:
            assert self.staging_future is not None
            self.staging_future.result()
            self.staging = False

    def _find_load_step(self, folder: str = "") -> int:
        """Find the step to load the checkpoint for.

        Args:
            folder (str, optional): The folder to find the checkpoint for. If ``folder``
            is "", then ``self.folder`` will be used.

        Returns:
            int: The step to load the checkpoint for.
        """
        folder = folder if folder else self.folder
        step_counts = []

        if not os.path.isdir(folder):
            return -1

        for filename in os.listdir(folder):
            match = _STEP_DIR_RE.search(filename)
            dcp_metadata_probe = os.path.join(folder, filename, ".metadata")
            safetensors_metadata_probe = os.path.join(
                folder, filename, "model.safetensors.index.json"
            )
            if match and os.path.isfile(dcp_metadata_probe):
                step_counts.append(int(match.group(1)))
            elif match and os.path.isfile(safetensors_metadata_probe):
                step_counts.append(int(match.group(1)))
        if not step_counts:
            return -1
        return max(step_counts)

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        folder = folder if folder else self.folder
        return os.path.join(folder, f"step-{step}")

    def _flattened_model_states_sd(
        self, state_dict: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Flatten the model states into a single dictionary.

        Note that other states, such as optimizer states, are not flattened.
        """
        states = state_dict if state_dict is not None else self.states
        sd = {k: v for k, v in states.items() if k != MODEL}
        if MODEL in states:
            sd.update(states[MODEL].state_dict())
        return sd

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        """Determines which states to load for the given step.

        This API is used to determine which states to load based on the
        configurations.

        Args:
            model_only (bool): Whether to load the model only.

        Returns:
            Dict[str, Any]: The states to load for the given step.
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

        states_to_load = self._flattened_model_states_sd(states_to_load)

        return states_to_load

    def _save_last_step(self, curr_step: int) -> None:
        # We only consider saving model only at the end of the training. So
        # this won't affect preemption and training resume. We also only allow
        # dtype conversion when we are checkpointing model only and the current
        # dtype is not the same as the export dtype at the end of the training.
        #
        # Last-step save pipeline:
        #   1. converter transform  (via state_dict(mode="export") — e.g. filter/reshape)
        #   2. dtype transform      (sd_transforms.apply_dtype_convert)
        #   3. FQN mapping          (split by key_filter, per-converter mapping)
        #   4. dcp_save
        #
        # FQN mapping splits the export state dict by each converter's
        # key_filter.  Matched keys go through that converter's
        # state_dict_adapter; remaining (base) keys go through the model's
        # HF adapter.

        model_wrapper = self.states[MODEL]

        if self.last_save_model_only:
            states = model_wrapper.state_dict(mode="export")
            states = self.sd_transforms.apply_dtype_convert(states)
            logger.info(
                f"Saving a model only checkpoint in "
                f"{self.sd_transforms.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            states = self._flattened_model_states_sd()

        fqn_to_index_mapping = None
        if self.last_save_in_hf and not self.last_save_model_only:
            raise ValueError(
                "Only model can be saved when saving in HF safetensors format."
            )
            # Split by converter key_filters; each adapter maps its own keys.
            # Unmatched keys fall through to the model's HF adapter.
            # Reverse order so the last-applied converter claims its keys
            # first, consistent with state_dict_transform undo order.
            remaining = dict(states)
            mapped: dict[str, Any] = {}
            for conv_adapter, conv_kf in reversed(self._converter_sd_adapters):
                matched = {k: v for k, v in remaining.items() if conv_kf(k)}
                remaining = {k: v for k, v in remaining.items() if not conv_kf(k)}
                if matched:
                    mapped.update(conv_adapter.to_hf(matched))
            if remaining:
                model_adapter = self.sd_transforms.sd_adapter
                assert model_adapter is not None
                mapped.update(model_adapter.to_hf(remaining))
            states = mapped

            # Merge fqn_to_index_mapping from all adapters.
            all_mappings: dict[Any, int] = {}
            for conv_adapter, _ in self._converter_sd_adapters:
                if conv_adapter.fqn_to_index_mapping:
                    all_mappings.update(conv_adapter.fqn_to_index_mapping)
            if self.sd_transforms.fqn_to_index_mapping:
                all_mappings.update(self.sd_transforms.fqn_to_index_mapping)
            fqn_to_index_mapping = all_mappings or None

        self.dcp_save(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf,
            fqn_to_index_mapping=fqn_to_index_mapping,
        )

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        if not self.enable or self.load_only:
            return False

        if curr_step == 1 and self.enable_first_step_checkpoint:
            return True

        if last_step:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    def _async_wait(self) -> None:
        if self.save_future is None:
            return
        if self.async_mode == AsyncMode.DISABLED:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is not enabled."
            )
        self.save_future.result()
        # ASYNC_WITH_PINNED_MEM: the stager manages the future's lifecycle,
        # so we do not clear it here.  For ASYNC we clear it ourselves.
        if self.async_mode == AsyncMode.ASYNC:
            self.save_future = None

    def _should_purge(self) -> bool:
        """Whether this rank should purge stale checkpoints.

        Extracted so subclasses (e.g. FTCheckpointManager) can add
        additional guards (like participating_rank) without duplicating
        the purge loop in ``_purge_stale_checkpoints``.
        """
        return (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and os.path.isdir(self.folder)
        )

    def _purge_stale_checkpoints(self):
        if not self._should_purge():
            return
        discovered_checkpoints = []
        for filename in os.listdir(self.folder):
            match = _STEP_DIR_RE.search(filename)
            if match:
                path = os.path.join(self.folder, filename)
                discovered_checkpoints.append((int(match.group(1)), path))

        discovered_checkpoints.sort()
        to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

        for _, path in to_delete:
            assert self.purge_thread is not None
            self.purge_queue.put(path)
