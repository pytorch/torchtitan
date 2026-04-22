# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import enum
import os
import queue
import re
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, cast, Literal

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
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.model_wrapper import ModelWrapper, StateDictMode
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection


MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"


@dataclass(frozen=True)
class HFStorageConfig:
    """I/O config for HuggingFace checkpoint format.

    Extracted from BaseStateDictAdapter by the trainer. Contains only
    the storage-level config that CheckpointManager needs — no content
    transforms (to_hf/from_hf).
    """

    fqn_to_index_mapping: dict[Any, int] | None
    hf_assets_path: str | None
    get_storage_reader: Callable[..., Any]


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


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
    """I/O orchestration for checkpoint save/load.

    Handles when and where to save/load checkpoints via DCP, async scheduling,
    and stale checkpoint purging. Never touches state dict content directly —
    all content transforms (converter key filtering, HF format conversion) are
    owned by ``ModelWrapper``.

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
        model_wrapper (ModelWrapper): Owns model content transforms (converter
            key filtering, converter state dict transforms). No HF awareness.
        hf_storage_config (HFStorageConfig | None): I/O config for HF format
            (fqn_to_index_mapping, storage reader, hf_assets_path). No content
            transforms — those live in ModelWrapper as callables.
        optimizers (OptimizersContainer): The optimizers used to optimize the model.
        lr_schedulers (LRSchedulersContainer): The lr schedulers used to optimize the model.
        states (Dict[str, Any]): The states that need to be saved, other than the
            previous 4 components.
        config (Checkpoint): The config used to configure the checkpointing.
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
        When enable is set to true, checkpoints will be in {--dump_folder}/{--checkpoint.folder}.
        """

        interval: int = 500
        """Checkpointing interval in steps."""

        initial_load_path: str | None = None
        """
        Path to the initial checkpoint to load. Always loaded when set,
        even if the checkpoint folder for the current run already exists.
        This is useful for:
        - Loading a pre-trained model for the first training run.
        - Providing base/frozen weights on resume when interval checkpoints
          are partial (e.g. LoRA saves only adapter keys).

        When the checkpoint folder exists, the initial checkpoint is loaded
        first (phase 1), then the resume checkpoint is loaded on top (phase 2).

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
        is only used when `initial_load_path` and `initial_load_in_hf` is specified. This will load
        checkpoints in HF's model definition and dequantize on model weights if necessary. To support
        this parameter, the model need to define proper HuggingFaceStorageReader to perform dequantize.
        """

        additional_load_path: str | None = None
        """
        Path to an additional checkpoint to load on top of the initial checkpoint.
        This is useful for loading pre-trained adapter weights (e.g. LoRA adapters)
        alongside the base model from `initial_load_path`.

        On initial load (no checkpoint folder): loads after `initial_load_path`.
        On resume (checkpoint folder exists): ignored — the checkpoint folder
        serves as the overlay.

        The loaded checkpoint is always model-only in DCP format and uses
        `strict=False` so missing keys are silently skipped.
        HF/safetensors format is not yet supported for additional loads.

        Note that the path should contain the full path to the checkpoint folder,
        including the step number, if any.
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
        model_wrapper: ModelWrapper,
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        hf_storage_config: HFStorageConfig | None = None,
        base_folder: str = "",
    ) -> None:
        self.enable = config.enable
        self.load_only = config.load_only

        self.states = states
        self.states.update(
            {
                MODEL: model_wrapper,
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )

        async_mode = config.async_mode.lower()
        self.enable_staging = (
            self.enable and async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        )

        if not self.enable:
            self.folder = ""
            return

        self.staging = False
        self.sending_to_checkpoint_mp = False
        self.staging_id = None
        self.cpu_offload_state_dict = None
        self.stager = None
        self.pg: dist.ProcessGroup | None = None

        self.folder = os.path.join(base_folder, config.folder)

        # Checkpoint policy related fields.
        self.initial_load_model_only = config.initial_load_model_only
        self.initial_load_in_hf = config.initial_load_in_hf
        self.initial_load_path = config.initial_load_path
        self.initial_load_in_hf_quantized = config.initial_load_in_hf_quantized
        self.additional_load_path = config.additional_load_path
        self.last_save_model_only = config.last_save_model_only
        self.last_save_in_hf = config.last_save_in_hf
        if self.last_save_in_hf:
            assert (
                hf_storage_config is not None
            ), "checkpoint.last_save_in_hf is True, but hf_storage_config is not provided."
        self.hf_storage_config = hf_storage_config
        self.export_dtype = TORCH_DTYPE_MAP[config.export_dtype]
        self.exclude_from_loading = config.exclude_from_loading
        self.interval = config.interval
        self.enable_first_step_checkpoint = config.enable_first_step_checkpoint

        # Async checkpoint related fields.
        async_mode = config.async_mode.lower()
        if (
            async_mode == AsyncMode.ASYNC
            or async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
        ):
            self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))

        self.keep_latest_k = config.keep_latest_k
        self.purge_thread: threading.Thread | None = None
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

        self.mp = None
        self.staging_future = None
        self.save_future = None
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            raise ValueError(f"Unknown checkpoint async_mode {config.async_mode}")

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from and saved to {self.folder}"
        )

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "enable") and self.enable:
            if hasattr(self, "mp") and self.mp and self.mp.is_alive():
                # pyrefly: ignore [missing-attribute]
                self.mp_queue_send.put(Terminate())
                self.mp.join()
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
        """Save the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to save.
            checkpoint_id (str): The checkpoint id to save.
            async_mode (AsyncMode): Whether the checkpoint is async.
            enable_garbage_collection (bool): Whether to enable garbage collection after save.
            to_hf (bool): Whether to save in HF model definition and safetensors format.

        Returns:
            Future: The future object if the checkpoint is async, otherwise None.
        """

        ret: Future | AsyncSaveResponse | None = None

        storage_writer: HuggingFaceStorageWriter | None = None
        checkpoint_save_id: str | None = None
        fqn_to_index_mapping: dict[Any, int] | None = None
        if to_hf:
            assert (
                self.hf_storage_config is not None
            ), "trying to save checkpoint in HF safetensors format, but hf_storage_config is not provided."
            # state_dict is already in HF format (converted by ModelWrapper)
            fqn_to_index_mapping = self.hf_storage_config.fqn_to_index_mapping
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
        """Load the checkpoint with dcp.
        Args:
            state_dict (dict): The state dict to load.
            checkpoint_id (str): The checkpoint id to load.
            from_hf (bool): Whether to load from HuggingFace checkpoint with
                its own model definition and safetensors format.
        """

        if from_hf:
            assert (
                self.hf_storage_config is not None
            ), "trying to load checkpoint in HF safetensors format, but hf_storage_config is not provided."
            model_wrapper: ModelWrapper = self.states[MODEL]
            hf_state_dict = model_wrapper.to_hf(
                model_wrapper.filter_base_keys(model_wrapper.state_dict())
            )
            hf_storage_reader = self.hf_storage_config.get_storage_reader(
                checkpoint_id, from_quantized
            )

            dcp.load(
                hf_state_dict,
                storage_reader=hf_storage_reader,
            )

            model_wrapper.load_state_dict(model_wrapper.from_hf(hf_state_dict))
        else:
            dcp.load(state_dict, checkpoint_id=checkpoint_id)

            # TODO: Since we flatten the model states in state_dict, we need to
            # manually call load_state_dict() for the model. Need to fix this.
            if MODEL in self.states:
                self.states[MODEL].load_state_dict(state_dict)

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
            self.save_future = self.dcp_save(
                states, checkpoint_id=checkpoint_id, async_mode=self.async_mode
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
        """Load the checkpoint, potentially from multiple sources.

        Two-phase loading:

        1. **Initial load**: If ``initial_load_path`` is set, always load it
           first (model weights only). This provides base/pretrained weights.
        2. **Overlay load**: Either the checkpoint folder (resume) or
           ``additional_load_path`` (pretrained adapter). The checkpoint
           folder takes priority.

        When both phases run, Phase 2 builds a FULL-mode load container via
        ``get_model_state_dict()``, which reflects Phase 1's loaded weights.
        DCP only overwrites keys present in the overlay checkpoint (e.g.
        adapter keys), leaving Phase 1's base weights intact. This relies
        on ``get_model_state_dict()`` returning current model state.

        Args:
            step (int, optional): The step to load the checkpoint for. Defaults to -1.

        Returns:
            bool: Whether any checkpoint was loaded successfully.
        """
        if not self.enable:
            return False

        loaded = False

        # Phase 1: Load initial checkpoint (base/pretrained weights).
        # Always loaded when set, even if checkpoint folder exists.
        loaded |= self._load_initial()

        # Phase 2: Overlay — checkpoint folder (resume) or additional_load_path.
        loaded |= self._load_overlay(step)

        if loaded:
            GarbageCollection.collect("GC collection for checkpoint loading.")
        return loaded

    def _resolve_initial_load_path(self) -> str | None:
        """Resolve the initial load path for base/pretrained weights.

        When ``initial_load_path`` is explicitly set, it is always used.
        The ``hf_assets_path`` fallback is only used on fresh start (no
        checkpoint folder) to avoid wasteful HF reloads on every resume.
        """
        if self.initial_load_path:
            if not os.path.isdir(self.initial_load_path):
                raise ValueError(
                    "checkpoint.initial_load_path is specified but the path is not valid."
                )
            return self.initial_load_path

        # hf_assets_path fallback: only on fresh start (no checkpoint folder)
        if self.initial_load_in_hf and not os.path.isdir(self.folder):
            assert (
                self.hf_storage_config is not None
                and self.hf_storage_config.hf_assets_path is not None
            ), "from_hf is True but hf_storage_config or hf_assets_path is not provided."
            hf_assets_path = self.hf_storage_config.hf_assets_path
            if not os.path.isdir(hf_assets_path):
                raise ValueError(
                    "model.hf_assets_path is being used to load HF weights but the path is not valid. "
                    "Either make sure hf_assets_path is correct or provide a valid checkpoint.initial_load_path"
                )
            return hf_assets_path

        return None

    def _load_initial(self) -> bool:
        """Phase 1: Load base/pretrained weights from initial_load_path.

        Always loads when initial_load_path is explicitly set, regardless
        of whether the checkpoint folder exists. The hf_assets_path fallback
        is only used on fresh start (no checkpoint folder).
        """
        checkpoint_id = self._resolve_initial_load_path()
        if checkpoint_id is None:
            return False

        from_hf = self.initial_load_in_hf
        from_quantized = self.initial_load_in_hf_quantized

        if from_hf:
            assert (
                self.initial_load_model_only
            ), "Only model can be loaded when loading from HF's safetensors checkpoint."
        if from_quantized:
            assert (
                from_hf
            ), "Quantized checkpoint can only be loaded from HuggingFace format."

        logger.info(
            f"Loading initial checkpoint from {checkpoint_id}"
            f"{' (HF format)' if from_hf else ''}."
        )
        begin = time.monotonic()
        states = self._states_to_load(model_only=self.initial_load_model_only)
        self.dcp_load(
            states,
            checkpoint_id=checkpoint_id,
            from_hf=from_hf,
            from_quantized=from_quantized,
        )
        logger.info(
            f"Finished loading initial checkpoint in "
            f"{time.monotonic() - begin:.2f} seconds."
        )
        return True

    def _load_overlay(self, step: int = -1) -> bool:
        """Phase 2: Load overlay checkpoint (resume or additional weights).

        Checkpoint folder takes priority over additional_load_path.
        On resume, loads the full training state. additional_load_path
        loads model weights only.
        """
        if os.path.isdir(self.folder):
            if self.additional_load_path:
                logger.warning(
                    "checkpoint.additional_load_path is set but the checkpoint "
                    "folder exists. The additional_load_path will be ignored — "
                    "the checkpoint folder serves as the overlay."
                )
            step = self._find_load_step() if step == -1 else step
            if step == -1:
                return False
            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)

            if not os.path.isdir(checkpoint_id):
                raise FileNotFoundError(
                    f"--checkpoint.load_step={step} but checkpoint "
                    f"{checkpoint_id} is not found."
                )

            logger.info(f"Loading resume checkpoint from {checkpoint_id}.")
            begin = time.monotonic()
            states = self._states_to_load(model_only)
            self.dcp_load(
                states,
                checkpoint_id=checkpoint_id,
                from_hf=False,
                from_quantized=False,
            )
            logger.info(
                f"Finished loading resume checkpoint in "
                f"{time.monotonic() - begin:.2f} seconds."
            )
            return True

        if self.additional_load_path:
            if not os.path.isdir(self.additional_load_path):
                raise ValueError(
                    "checkpoint.additional_load_path is specified but the path "
                    "is not valid."
                )

            logger.info(
                f"Loading additional checkpoint from {self.additional_load_path}."
            )
            begin = time.monotonic()
            states = self._states_to_load(model_only=True)
            self.dcp_load(
                states,
                checkpoint_id=self.additional_load_path,
                from_hf=False,
                from_quantized=False,
            )
            logger.info(
                f"Finished loading additional checkpoint in "
                f"{time.monotonic() - begin:.2f} seconds."
            )
            return True

        return False

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
        pattern = r"step-(\d+)"
        step_counts = []

        if not os.path.isdir(folder):
            return -1

        for filename in os.listdir(folder):
            match = re.search(pattern, filename)
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
        self,
        last_step: bool = False,
    ) -> dict[str, Any]:
        """Flatten the model states into a single dictionary.

        Uses EXPORT mode so that converter transforms are applied
        (e.g. LoRA filters to adapter keys on interval saves).
        Other states (optimizer, lr_scheduler, etc.) are not flattened.
        """
        sd = {k: v for k, v in self.states.items() if k != MODEL}
        if MODEL in self.states:
            sd.update(
                self.states[MODEL].state_dict(
                    mode=StateDictMode.EXPORT, last_step=last_step
                )
            )
        return sd

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        """Determines which states to load for the given step.

        Uses RAW mode for model keys so that the load container has all
        keys for DCP to populate.

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
            k: v
            for k, v in self.states.items()
            if k not in self.exclude_from_loading and k != MODEL
        }
        if MODEL not in self.exclude_from_loading and MODEL in self.states:
            states_to_load.update(self.states[MODEL].state_dict())

        return states_to_load

    def _save_last_step(self, curr_step: int) -> None:
        # We only consider saving model only at the end of the training. So this
        # won't affect preemption and training resume. We also only allow dtype
        # conversion when we are checkpointing model only and the current dtype
        # is not the same as the export dtype at the end of the training.
        #
        # Seed checkpoints (step 0) use RAW mode to preserve the exact model
        # state without converter transforms (e.g. dequantization).
        is_seed = curr_step == 0

        if self.last_save_model_only:
            if is_seed:
                states = self.states[MODEL].state_dict()
            else:
                states = self.states[MODEL].state_dict(
                    mode=StateDictMode.EXPORT,
                    last_step=True,
                )
                if self.last_save_in_hf:
                    states = self.states[MODEL].to_hf(states)

            if not is_seed and self.export_dtype != torch.float32:
                states = {k: v.to(self.export_dtype) for k, v in states.items()}
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")
            if is_seed:
                # Seed: RAW mode, no converter transforms
                sd = {k: v for k, v in self.states.items() if k != MODEL}
                sd.update(self.states[MODEL].state_dict())
                states = sd
            else:
                states = self._flattened_model_states_sd(last_step=True)

        if self.last_save_in_hf:
            assert (
                self.last_save_model_only
            ), "Only model can be saved when saving in HF safetensors format."

        self.dcp_save(
            states,
            checkpoint_id=self._create_checkpoint_id(curr_step),
            async_mode=AsyncMode.DISABLED,
            enable_garbage_collection=True,
            to_hf=self.last_save_in_hf and not is_seed,
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
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.save_future is not None:
                self.save_future.result()
        elif self.async_mode == AsyncMode.ASYNC:
            if self.save_future is not None:
                self.save_future.result()
                self.save_future = None
        elif self.save_future is not None:
            raise RuntimeError(
                "self.save_future is not None, but self.async_mode is not enabled."
            )

    def _should_purge(self) -> bool:
        """Whether this rank should purge stale checkpoints.

        Extracted so subclasses (e.g. FTCheckpointManager) can add
        additional guards (like participating_rank) without duplicating
        the purge loop in _purge_stale_checkpoints.
        """
        return (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and os.path.isdir(self.folder)
        )

    def _purge_stale_checkpoints(self):
        if self._should_purge():
            discovered_checkpoints = []
            for filename in os.listdir(self.folder):
                match = re.search(r"step-(\d+)", filename)
                if match:
                    path = os.path.join(self.folder, filename)
                    discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                assert self.purge_thread is not None
                self.purge_queue.put(path)
